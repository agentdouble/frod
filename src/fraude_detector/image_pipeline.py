"""Standalone image analysis without converting the source to PDF."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from statistics import fmean

from PIL import Image

from fraude_detector.ai_images import (
    AiImageEvaluation,
    AiImageModelAdapter,
    consensus_evaluations,
    evaluate_ai_image_adapter,
    one_evaluation_per_family,
)
from fraude_detector.config import AnalysisConfig
from fraude_detector.detectors.ocr import OcrDetector
from fraude_detector.errors import AnalysisError
from fraude_detector.gapl import GaplError
from fraude_detector.gapl_windows import (
    analyze_centered_windows,
    build_gapl_global_finding,
    write_gapl_window_artifacts,
)
from fraude_detector.image_provenance import (
    AiDeclarationTrust,
    C2paSummary,
    EvidenceState,
    ImageProvenanceReport,
    ai_declaration_weight,
    analyze_image_provenance,
    find_ai_metadata_markers,
)
from fraude_detector.llm_classifier import classify_document
from fraude_detector.models import (
    DetectorResult,
    Finding,
    ImageAnalysisReport,
    ImageInfo,
    OcrReport,
)
from fraude_detector.scoring import assess_risk


class ImageAnalysisPipeline:
    """Analyze the original bytes of one standalone raster image."""

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        ai_image_adapters: Iterable[AiImageModelAdapter] = (),
    ) -> None:
        self.config = config or AnalysisConfig()
        self.ai_image_adapters = tuple(ai_image_adapters)

    def analyze(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ImageAnalysisReport:
        def report_progress(value: float, label: str) -> None:
            if progress_callback is not None:
                progress_callback(min(1.0, max(0.0, value)), label)

        report_progress(0.02, "Lecture de l'image")
        source = Path(input_path).expanduser().resolve()
        destination = Path(output_dir).expanduser().resolve()
        raw_image = self._read_input(source)
        report_progress(0.08, "Analyse de la provenance")
        provenance = analyze_image_provenance(
            raw_image,
            source.suffix,
            max_pixels=self.config.max_embedded_image_pixels,
        )
        pillow = provenance.pillow
        if pillow.width is None or pillow.height is None or pillow.decoded_format is None:
            if pillow.error == "image_pixel_limit_exceeded":
                raise AnalysisError("image_too_large", "L'image depasse la limite de pixels.")
            raise AnalysisError("invalid_image", "Le fichier n'est pas une image lisible.")

        destination.mkdir(parents=True, exist_ok=True)
        provenance_artifact = self._write_provenance_artifact(
            destination,
            provenance,
            raw_image,
        )
        provenance_findings = _build_provenance_findings(
            provenance,
            image_sha256=hashlib.sha256(raw_image).hexdigest(),
            artifact=provenance_artifact,
        )
        provenance_status = (
            "partial"
            if provenance.c2pa.state in {EvidenceState.UNAVAILABLE, EvidenceState.INDETERMINATE}
            else "completed"
        )
        provenance_detector = DetectorResult(
            name="image_provenance",
            status=provenance_status,
            findings=provenance_findings,
            notes=(
                "Provenance analysee sur les octets originaux de l'image.",
                "Les manifestes distants et OCSP ne sont pas charges.",
                "L'absence de C2PA ou de metadonnees ne prouve pas une origine humaine.",
            ),
            artifacts=(provenance_artifact,),
        )
        ocr_report: OcrReport | None = None
        ocr_detector: OcrDetector | None = None
        classification_result = None
        if self.config.ocr_enabled:
            report_progress(0.18, "Reconnaissance du contenu")
            ocr_detector = OcrDetector(self.config)
            ocr_report = ocr_detector.detect(source, destination)
            if self.config.classification_enabled and ocr_report.success:
                report_progress(0.20, "Classification du document")
                try:
                    classification_result = classify_document(
                        ocr_report.markdown,
                        self.config,
                    )
                except Exception as error:
                    warnings.warn(f"Classification failed: {error}", stacklevel=2)

        report_progress(0.22, "Analyse des pixels")
        ai_detector = self._analyze_pixels(
            raw_image,
            destination,
            progress_callback=lambda value, label: report_progress(
                0.22 + 0.68 * value,
                label,
            ),
        )
        report_progress(0.92, "Calcul du score")
        forensics = tuple(dict.fromkeys((*provenance_detector.artifacts, *ai_detector.artifacts)))
        detectors = [provenance_detector, ai_detector]
        if ocr_detector is not None and ocr_report is not None:
            detectors.append(ocr_detector.result(ocr_report))
        findings = tuple(
            finding for detector_result in detectors for finding in detector_result.findings
        )
        report = ImageAnalysisReport(
            schema_version="1.0",
            analyzed_at=datetime.now(UTC).isoformat(),
            image=ImageInfo(
                filename=source.name,
                sha256=hashlib.sha256(raw_image).hexdigest(),
                size_bytes=len(raw_image),
                format=pillow.decoded_format,
                width=pillow.width,
                height=pillow.height,
                mode=pillow.mode or "unknown",
            ),
            assessment=assess_risk(findings),
            detectors=tuple(detectors),
            findings=findings,
            artifacts={
                "forensics": forensics,
                "ocr_json": _ocr_artifacts(ocr_report, ".json"),
                "ocr_markdown": _ocr_artifacts(ocr_report, ".md"),
                "ocr_layout": ocr_report.layout_images if ocr_report else (),
            },
            limitations=(
                "Le score priorise une revue; il ne prouve ni fraude ni authenticite.",
                "Une provenance C2PA decrit une origine technique, pas une intention frauduleuse.",
                "Les metadonnees non signees peuvent etre supprimees ou falsifiees.",
                "Les modeles passifs sont sensibles au domaine et aux recompressions.",
                "GLM-OCR ne fournit pas actuellement de confiance par caractere. "
                "Les controles de contenu utilisent une fiabilite de representation "
                "plafonnee et peuvent etre neutralises si l'extraction est insuffisante.",
            ),
            classification=classification_result,
        )
        (destination / "report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_progress(1.0, "Analyse terminee")
        return report

    def _read_input(self, source: Path) -> bytes:
        if not source.is_file():
            raise AnalysisError("input_not_found", f"Fichier introuvable: {source}")
        maximum_size = self.config.max_file_size_mb * 1024 * 1024
        if source.stat().st_size > maximum_size:
            raise AnalysisError(
                "input_too_large",
                f"Le fichier depasse la limite de {self.config.max_file_size_mb} Mo.",
            )
        return source.read_bytes()

    def _analyze_pixels(
        self,
        raw_image: bytes,
        destination: Path,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> DetectorResult:
        def report_progress(value: float, label: str) -> None:
            if progress_callback is not None:
                progress_callback(min(1.0, max(0.0, value)), label)

        if not self.ai_image_adapters:
            return DetectorResult(
                name="ai_generated_image",
                status="partial",
                notes=(
                    "Analyse pixel IA non executee: aucun modele passif charge.",
                    "L'absence de modele n'est pas un indice d'authenticite.",
                ),
            )

        evaluations: list[AiImageEvaluation] = []
        gapl_analyses = []
        gapl_errors: list[str] = []
        adapter_count = len(self.ai_image_adapters)
        with Image.open(BytesIO(raw_image)) as source_image:
            source_image.load()
            for index, adapter in enumerate(self.ai_image_adapters):
                adapter_start = index / adapter_count
                adapter_width = 1 / adapter_count
                report_progress(adapter_start, "Analyse du modele IA")
                evaluation = evaluate_ai_image_adapter(
                    adapter,
                    source_image,
                    max_dimension=self.config.ai_max_image_dimension,
                    stability_max_delta=self.config.ai_stability_max_delta,
                )
                evaluations.append(evaluation)
                if adapter.adapter_id != "gapl_cvpr2026":
                    continue
                try:

                    def report_window_progress(
                        completed: int,
                        total: int,
                        start: float = adapter_start,
                        width: float = adapter_width,
                    ) -> None:
                        report_progress(
                            start + width * (0.2 + 0.8 * completed / total),
                            f"Analyse GAPL des zones - {completed}/{total}",
                        )

                    analysis, overlay = analyze_centered_windows(
                        adapter,
                        source_image,
                        max_dimension=self.config.ai_max_image_dimension,
                        progress_callback=report_window_progress,
                    )
                    gapl_analyses.append((analysis, overlay))
                except GaplError as error:
                    gapl_errors.append(str(error))

        evaluation_tuple = tuple(evaluations)
        artifacts = list(self._write_ai_artifacts(destination, evaluation_tuple))
        findings = list(_build_ai_findings(evaluation_tuple, self.config, tuple(artifacts)))
        for analysis, overlay in gapl_analyses:
            window_artifacts = write_gapl_window_artifacts(
                destination,
                analysis,
                overlay,
                stem="image-gapl_cvpr2026",
            )
            artifacts.extend(window_artifacts)
            findings.append(
                build_gapl_global_finding(
                    analysis,
                    artifacts=window_artifacts,
                )
            )
        partial = (
            any(item.status != "completed" for item in evaluation_tuple)
            or (
                not gapl_analyses
                and len({item.method_family for item in evaluation_tuple})
                < self.config.ai_min_consensus_families
            )
            or bool(gapl_errors)
        )
        report_progress(1.0, "Analyse des pixels terminee")
        notes = [
            f"Modeles passifs executes: {len(evaluation_tuple)}.",
            "Un score de modele indique une trace statistique, pas une fraude.",
        ]
        if gapl_errors:
            notes.append("Analyse GAPL multi-zone incomplete: " + "; ".join(gapl_errors))
        return DetectorResult(
            name="ai_generated_image",
            status="partial" if partial else "completed",
            findings=tuple(findings),
            notes=tuple(notes),
            artifacts=tuple(dict.fromkeys(artifacts)),
        )

    def _write_provenance_artifact(
        self,
        destination: Path,
        report: ImageProvenanceReport,
        raw_image: bytes,
    ) -> str:
        relative = Path("forensics/ai/image-provenance.json")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        markers = find_ai_metadata_markers(report.pillow)
        payload = {
            "asset": {
                "sha256": hashlib.sha256(raw_image).hexdigest(),
                "format": report.pillow.decoded_format,
                "width": report.pillow.width,
                "height": report.pillow.height,
                "mode": report.pillow.mode,
            },
            "metadata": {
                "state": report.pillow.state.value,
                "exif_keys": [entry.key for entry in report.pillow.exif.entries],
                "xmp_keys": [entry.key for entry in report.pillow.xmp.entries],
                "info_keys": [entry.key for entry in report.pillow.info.entries],
                "ai_markers": [marker.marker for marker in markers],
            },
            "c2pa": _c2pa_payload(report.c2pa),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return relative.as_posix()

    def _write_ai_artifacts(
        self,
        destination: Path,
        evaluations: tuple[AiImageEvaluation, ...],
    ) -> tuple[str, ...]:
        directory = destination / "forensics/ai"
        directory.mkdir(parents=True, exist_ok=True)
        heatmaps: dict[str, str] = {}
        for evaluation in evaluations:
            if evaluation.heatmap is None:
                continue
            identifier = re.sub(r"[^a-zA-Z0-9_.-]+", "-", evaluation.adapter_id).strip("-")
            relative = Path("forensics/ai") / f"image-{identifier or 'adapter'}-heatmap.png"
            evaluation.heatmap.save(destination / relative, format="PNG")
            heatmaps[evaluation.adapter_id] = relative.as_posix()

        relative_analysis = Path("forensics/ai/image-analysis.json")
        payload = {
            "policy": {
                "score_threshold": self.config.ai_model_score_threshold,
                "stability_max_delta": self.config.ai_stability_max_delta,
                "minimum_consensus_families": self.config.ai_min_consensus_families,
            },
            "evaluations": [
                {
                    "adapter_id": item.adapter_id,
                    "model_version": item.model_version,
                    "method_family": item.method_family,
                    "status": item.status,
                    "variant_scores": {
                        score.name: score.synthetic_score for score in item.variant_scores
                    },
                    "stable": item.stable,
                    "max_score_delta": item.max_score_delta,
                    "out_of_domain_reason": item.out_of_domain_reason,
                    "error": item.error,
                    "details": item.details,
                    "heatmap": heatmaps.get(item.adapter_id),
                }
                for item in evaluations
            ],
        }
        (destination / relative_analysis).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return (relative_analysis.as_posix(), *heatmaps.values())


def _build_provenance_findings(
    report: ImageProvenanceReport,
    *,
    image_sha256: str,
    artifact: str,
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    summary = report.c2pa
    evidence = {
        "scope": "image",
        "image_sha256": image_sha256,
        "validation": summary.validation.value,
        "raw_validation_state": summary.raw_validation_state,
        "claim_generator": summary.claim_generator,
        "active_manifest_label": summary.active_manifest_label,
    }
    if summary.provenance_invalidated:
        findings.append(
            Finding(
                detector="image_provenance",
                code="IMAGE_C2PA_INVALID",
                category="provenance_integrity",
                title="Provenance C2PA invalidee",
                description=(
                    "Le manifeste est present mais son integrite ou sa signature ne se "
                    "valide pas. Cela ne prouve pas une generation par IA."
                ),
                risk_points=18.0,
                confidence=0.9,
                evidence=evidence
                | {
                    "failure_codes": [
                        status.code
                        for status in summary.statuses
                        if status.outcome.casefold() == "failure"
                    ]
                },
                artifacts=(artifact,),
            )
        )

    declaration = summary.ai_declaration
    if (
        declaration.state is EvidenceState.DETECTED
        and declaration.trust is not AiDeclarationTrust.INVALIDATED
    ):
        points, confidence = ai_declaration_weight(declaration.trust)
        findings.append(
            Finding(
                detector="image_provenance",
                code="AI_IMAGE_C2PA_DECLARATION",
                category="synthetic_media",
                title="Origine IA declaree dans une provenance C2PA",
                description=(
                    "La provenance declare une creation ou une composition algorithmique. "
                    "La confiance depend du certificat signataire."
                ),
                risk_points=points,
                confidence=confidence,
                evidence=evidence
                | {
                    "declaration_trust": declaration.trust.value,
                    "digital_source_types": list(declaration.digital_source_types),
                },
                artifacts=(artifact,),
            )
        )

    markers = find_ai_metadata_markers(report.pillow)
    if markers:
        findings.append(
            Finding(
                detector="image_provenance",
                code="AI_GENERATOR_METADATA_MENTIONED",
                category="synthetic_media",
                title="Generateur IA mentionne dans les metadonnees image",
                description=(
                    "Les metadonnees non signees mentionnent un outil ou des parametres "
                    "de generation; elles restent falsifiables."
                ),
                risk_points=15.0,
                confidence=0.65,
                evidence={
                    "image_sha256": image_sha256,
                    "markers": [marker.marker for marker in markers],
                    "metadata_keys": sorted({marker.metadata_key for marker in markers}),
                },
                artifacts=(artifact,),
            )
        )
    return tuple(findings)


def _build_ai_findings(
    evaluations: tuple[AiImageEvaluation, ...],
    config: AnalysisConfig,
    artifacts: tuple[str, ...],
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for evaluation in evaluations:
        if evaluation.status == "out_of_domain" or evaluation.stable is False:
            findings.append(
                Finding(
                    detector="ai_generated_image",
                    code=(
                        "AI_ANALYSIS_OUT_OF_DOMAIN"
                        if evaluation.status == "out_of_domain"
                        else "AI_ANALYSIS_UNSTABLE"
                    ),
                    category="analysis_quality",
                    title=(
                        "Modele IA hors de son domaine d'analyse"
                        if evaluation.status == "out_of_domain"
                        else "Score IA instable apres transformations mineures"
                    ),
                    description="Le resultat est exclu du consensus et ne produit aucun risque.",
                    risk_points=0.0,
                    confidence=1.0,
                    evidence={
                        "adapter_id": evaluation.adapter_id,
                        "model_version": evaluation.model_version,
                        "method_family": evaluation.method_family,
                        "original_score": evaluation.original_score,
                        "max_score_delta": evaluation.max_score_delta,
                        "out_of_domain_reason": evaluation.out_of_domain_reason,
                    },
                    artifacts=artifacts,
                )
            )

    consensus = consensus_evaluations(evaluations, threshold=config.ai_model_score_threshold)
    if len({item.method_family for item in consensus}) >= config.ai_min_consensus_families:
        selected = one_evaluation_per_family(consensus)
        scores = [item.original_score for item in selected if item.original_score is not None]
        findings.append(
            Finding(
                detector="ai_generated_image",
                code="AI_PIXEL_TRACE_CONSENSUS",
                category="synthetic_media",
                title="Consensus de traces statistiques de generation IA",
                description=(
                    "Plusieurs familles de modeles independantes produisent des scores "
                    "eleves et stables; ce signal doit etre corrobore."
                ),
                risk_points=35.0,
                confidence=min(0.85, fmean(scores)),
                evidence={
                    "threshold": config.ai_model_score_threshold,
                    "adapters": [
                        {
                            "adapter_id": item.adapter_id,
                            "model_version": item.model_version,
                            "method_family": item.method_family,
                            "score": item.original_score,
                            "max_score_delta": item.max_score_delta,
                        }
                        for item in selected
                    ],
                },
                artifacts=artifacts,
            )
        )
    elif consensus:
        for evaluation in consensus:
            findings.append(
                Finding(
                    detector="ai_generated_image",
                    code="AI_PIXEL_TRACE_SINGLE_MODEL",
                    category="analysis_quality",
                    title="Score IA eleve mais non corrobore",
                    description=(
                        "Un modele produit un score eleve et stable sans famille "
                        "independante pour le corroborer."
                    ),
                    risk_points=0.0,
                    confidence=0.5,
                    evidence={
                        "adapter_id": evaluation.adapter_id,
                        "model_version": evaluation.model_version,
                        "method_family": evaluation.method_family,
                        "score": evaluation.original_score,
                        "max_score_delta": evaluation.max_score_delta,
                    },
                    artifacts=artifacts,
                )
            )
    return tuple(findings)


def _c2pa_payload(summary: C2paSummary) -> dict[str, object]:
    return {
        "state": summary.state.value,
        "validation": summary.validation.value,
        "raw_validation_state": summary.raw_validation_state,
        "active_manifest_label": summary.active_manifest_label,
        "claim_generator": summary.claim_generator,
        "embedded": summary.embedded,
        "remote_url": summary.remote_url,
        "statuses": [
            {
                "outcome": status.outcome,
                "code": status.code,
                "explanation": status.explanation,
            }
            for status in summary.statuses
        ],
        "ai_declaration": {
            "state": summary.ai_declaration.state.value,
            "trust": summary.ai_declaration.trust.value,
            "digital_source_types": list(summary.ai_declaration.digital_source_types),
        },
        "error": summary.error,
    }


def _ocr_artifacts(report: OcrReport | None, suffix: str) -> tuple[str, ...]:
    if report is None or not report.success:
        return ()
    return tuple(path for path in report.artifacts if Path(path).suffix.casefold() == suffix)
