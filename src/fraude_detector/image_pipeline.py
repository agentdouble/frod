"""Standalone image analysis without converting the source to PDF."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
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
from fraude_detector.errors import AnalysisError
from fraude_detector.image_provenance import (
    AiDeclarationTrust,
    C2paSummary,
    EvidenceState,
    ImageProvenanceReport,
    ai_declaration_weight,
    analyze_image_provenance,
    find_ai_metadata_markers,
)
from fraude_detector.models import (
    DetectorResult,
    Finding,
    ImageAnalysisReport,
    ImageInfo,
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
    ) -> ImageAnalysisReport:
        source = Path(input_path).expanduser().resolve()
        destination = Path(output_dir).expanduser().resolve()
        raw_image = self._read_input(source)
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
        ai_detector = self._analyze_pixels(raw_image, destination)
        findings = (*provenance_findings, *ai_detector.findings)
        forensics = tuple(dict.fromkeys((*provenance_detector.artifacts, *ai_detector.artifacts)))
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
            detectors=(provenance_detector, ai_detector),
            findings=findings,
            artifacts={"forensics": forensics},
            limitations=(
                "Le score priorise une revue; il ne prouve ni fraude ni authenticite.",
                "Une provenance C2PA decrit une origine technique, pas une intention frauduleuse.",
                "Les metadonnees non signees peuvent etre supprimees ou falsifiees.",
                "Les modeles passifs sont sensibles au domaine et aux recompressions.",
            ),
        )
        (destination / "report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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

    def _analyze_pixels(self, raw_image: bytes, destination: Path) -> DetectorResult:
        if not self.ai_image_adapters:
            return DetectorResult(
                name="ai_generated_image",
                status="partial",
                notes=(
                    "Analyse pixel IA non executee: aucun modele passif charge.",
                    "L'absence de modele n'est pas un indice d'authenticite.",
                ),
            )

        with Image.open(BytesIO(raw_image)) as source_image:
            evaluations = tuple(
                evaluate_ai_image_adapter(
                    adapter,
                    source_image,
                    max_dimension=self.config.ai_max_image_dimension,
                    stability_max_delta=self.config.ai_stability_max_delta,
                )
                for adapter in self.ai_image_adapters
            )
        artifacts = self._write_ai_artifacts(destination, evaluations)
        findings = _build_ai_findings(evaluations, self.config, artifacts)
        partial = (
            any(item.status != "completed" for item in evaluations)
            or len({item.method_family for item in evaluations})
            < self.config.ai_min_consensus_families
        )
        return DetectorResult(
            name="ai_generated_image",
            status="partial" if partial else "completed",
            findings=findings,
            notes=(
                f"Modeles passifs executes: {len(evaluations)}.",
                "Un score de modele indique une trace statistique, pas une fraude.",
            ),
            artifacts=artifacts,
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
