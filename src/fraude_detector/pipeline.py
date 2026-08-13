"""Orchestration of PDF validation, detectors, scoring and artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pypdfium2
from pypdf import PdfReader

from fraude_detector.ai_images import AiImageModelAdapter
from fraude_detector.config import AnalysisConfig
from fraude_detector.content_analysis import ContentAnalysisResult, analyze_document_content
from fraude_detector.detectors import (
    AiGeneratedImageDetector,
    ImageProvenanceDetector,
    OcrDetector,
    PageCompositionDetector,
    PdfStructureDetector,
    RasterAnomalyDetector,
    RevisionDiffDetector,
)
from fraude_detector.detectors.base import AnalysisContext, Detector
from fraude_detector.errors import AnalysisError
from fraude_detector.models import (
    AnalysisReport,
    DetectorResult,
    DocumentInfo,
    OcrReport,
)
from fraude_detector.parallel_progress import ParallelProgress
from fraude_detector.pdf_revisions import find_valid_revision_end_offsets
from fraude_detector.rendering import create_review_overlays, render_input_pages
from fraude_detector.scoring import assess_risk


class AnalysisPipeline:
    """Run independent detectors over an immutable source PDF."""

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        detectors: Iterable[Detector] | None = None,
        ai_image_adapters: Iterable[AiImageModelAdapter] = (),
        ai_image_adapter_loaders: Iterable[Callable[[], AiImageModelAdapter]] = (),
    ) -> None:
        self.config = config or AnalysisConfig()
        self._custom_detectors = tuple(detectors) if detectors else None
        self._ai_image_adapters = tuple(ai_image_adapters)
        self._ai_image_adapter_loaders = tuple(ai_image_adapter_loaders)

    def _detectors(self, progress: ParallelProgress) -> tuple[Detector, ...]:
        if self._custom_detectors is not None:
            return self._custom_detectors
        adapters = list(self._ai_image_adapters)
        if self._ai_image_adapter_loaders:
            progress.update("technical", 0.0, "Chargement des modèles d'analyse")
            adapters.extend(loader() for loader in self._ai_image_adapter_loaders)
        return (
            PdfStructureDetector(),
            RevisionDiffDetector(),
            PageCompositionDetector(),
            RasterAnomalyDetector(),
            ImageProvenanceDetector(),
            AiGeneratedImageDetector(adapters),
        )

    def analyze(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        password: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> AnalysisReport:
        def report_progress(value: float, label: str) -> None:
            if progress_callback is not None:
                progress_callback(min(1.0, max(0.0, value)), label)

        report_progress(0.02, "Lecture du document")
        source = Path(input_path).expanduser().resolve()
        destination = Path(output_dir).expanduser().resolve()
        raw_pdf = self._read_input(source)

        report_progress(0.06, "Validation du PDF")
        reader = self._open_reader(raw_pdf, password)
        pdfium_document = self._open_pdfium(raw_pdf, password)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pdfium_document.init_forms()
            page_count = len(pdfium_document)
            if page_count < 1:
                raise AnalysisError("empty_pdf", "Le PDF ne contient aucune page.")

            destination.mkdir(parents=True, exist_ok=True)
            report_progress(0.10, "Preparation des pages")
            revision_offsets = find_valid_revision_end_offsets(raw_pdf, password)
            context = AnalysisContext(
                input_path=source,
                output_dir=destination,
                raw_pdf=raw_pdf,
                pdfium_document=pdfium_document,
                reader=reader,
                config=self.config,
                revision_end_offsets=revision_offsets,
                password=password,
            )

            rendered_pages = render_input_pages(context)
            report_progress(0.20, "Pages preparees")
            content_result: ContentAnalysisResult | None = None
            branch_weights = (
                {"technical": 0.45, "content": 0.55}
                if self.config.ocr_enabled
                else {"technical": 1.0}
            )
            progress = ParallelProgress(
                progress_callback,
                start=0.20,
                end=0.85,
                weights=branch_weights,
            )
            if self.config.ocr_enabled:
                with ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="frod-content",
                ) as executor:
                    content_future = executor.submit(
                        analyze_document_content,
                        source,
                        destination,
                        self.config,
                        rendered_pages=rendered_pages,
                        progress_callback=progress.worker_callback("content"),
                    )
                    detector_results_list = self._run_detectors(context, progress)
                    content_result = progress.wait(content_future)
            else:
                detector_results_list = self._run_detectors(context, progress)

            ocr_report = content_result.ocr_report if content_result is not None else None
            if content_result is not None:
                detector_results_list.append(content_result.detector_result)
            detector_results = tuple(detector_results_list)
            report_progress(0.86, "Calcul du score")
            findings = tuple(
                finding
                for detector_result in detector_results
                for finding in detector_result.findings
            )
            scored_findings = tuple(finding for finding in findings if finding.risk_points > 0)
            review_overlays = create_review_overlays(
                context,
                rendered_pages,
                scored_findings,
            )
            report_progress(0.94, "Preparation des resultats")
            forensics = tuple(
                dict.fromkeys(
                    artifact
                    for detector_result in detector_results
                    if detector_result.name != OcrDetector.name
                    for artifact in detector_result.artifacts
                )
            )

            metadata = {
                str(key): str(value)
                for key, value in pdfium_document.get_metadata_dict(skip_empty=True).items()
            }
            report = AnalysisReport(
                schema_version="1.0",
                analyzed_at=datetime.now(UTC).isoformat(),
                document=DocumentInfo(
                    filename=source.name,
                    sha256=hashlib.sha256(raw_pdf).hexdigest(),
                    size_bytes=len(raw_pdf),
                    page_count=page_count,
                    analyzed_pages=context.analyzed_page_count,
                    metadata=metadata,
                ),
                assessment=assess_risk(findings),
                detectors=detector_results,
                findings=findings,
                artifacts={
                    "page_renders": rendered_pages,
                    "review_overlays": review_overlays,
                    "forensics": forensics,
                    "ocr_json": _ocr_artifacts(ocr_report, ".json"),
                    "ocr_markdown": _ocr_artifacts(ocr_report, ".md"),
                    "ocr_layout": ocr_report.layout_images if ocr_report else (),
                },
                limitations=(
                    "Le score est une priorite de revue, pas une probabilite "
                    "ni une preuve de fraude.",
                    "Une reecriture complete du PDF peut supprimer tout historique de revisions.",
                    "Une signature, un formulaire, un tampon ou une annotation "
                    "peuvent expliquer legitimement un changement.",
                    "L'analyse ELA est un signal faible limite aux JPEG originaux "
                    "embarques et doit etre corroboree.",
                    "L'absence de C2PA ou de metadonnees ne prouve pas une origine humaine; "
                    "une declaration C2PA decrit l'origine, pas une intention frauduleuse.",
                    "Les modeles passifs d'images IA sont sensibles au domaine, aux nouveaux "
                    "generateurs et aux recompressions; seul un consensus stable est score.",
                    "GLM-OCR ne fournit pas actuellement de confiance par caractere. "
                    "Les controles de contenu utilisent une fiabilite de representation "
                    "plafonnee et peuvent etre neutralises si l'extraction est insuffisante.",
                    "Le routage image du MVP est geometrique: une photo pleine page peut etre "
                    "exclue et un document partiel peut rester candidat au modele passif.",
                    "Les masques alpha PDF separes ne sont pas recomposes lors du decodage "
                    "natif; certains pixels analyses peuvent differer du rendu visible.",
                    "L'absence de signal ne prouve pas l'authenticite du document.",
                ),
                classification=content_result.classification if content_result else None,
                extraction=content_result.extraction if content_result else None,
                extraction_verification=content_result.verification if content_result else None,
            )
            self._write_report(report, destination / "report.json")
            report_progress(1.0, "Analyse terminee")
            return report
        finally:
            pdfium_document.close()

    def _read_input(self, source: Path) -> bytes:
        if not source.is_file():
            raise AnalysisError("input_not_found", f"PDF introuvable: {source}")
        size = source.stat().st_size
        maximum_size = self.config.max_file_size_mb * 1024 * 1024
        if size > maximum_size:
            raise AnalysisError(
                "input_too_large",
                f"Le PDF depasse la limite de {self.config.max_file_size_mb} Mo.",
            )
        raw_pdf = source.read_bytes()
        if b"%PDF-" not in raw_pdf[:1024]:
            raise AnalysisError("invalid_pdf", "Le fichier n'a pas d'en-tete PDF valide.")
        return raw_pdf

    @staticmethod
    def _open_reader(raw_pdf: bytes, password: str | None) -> PdfReader:
        try:
            reader = PdfReader(io.BytesIO(raw_pdf), strict=False)
            if reader.is_encrypted and (not password or reader.decrypt(password) == 0):
                raise AnalysisError(
                    "encrypted_pdf",
                    "Le PDF est chiffre; fournissez un mot de passe valide.",
                )
            _ = len(reader.pages)
            return reader
        except AnalysisError:
            raise
        except Exception as error:
            raise AnalysisError("invalid_pdf", "Le PDF est illisible ou tronque.") from error

    @staticmethod
    def _open_pdfium(raw_pdf: bytes, password: str | None) -> pypdfium2.PdfDocument:
        try:
            return pypdfium2.PdfDocument(raw_pdf, password=password)
        except Exception as error:
            raise AnalysisError("invalid_pdf", "PDFium ne peut pas ouvrir ce document.") from error

    @staticmethod
    def _run_detector(detector: Detector, context: AnalysisContext) -> DetectorResult:
        try:
            return detector.analyze(context)
        except Exception as error:
            return DetectorResult(
                name=detector.name,
                status="partial",
                notes=(f"Detecteur interrompu: {type(error).__name__}: {str(error)[:240]}",),
            )

    def _run_detectors(
        self,
        context: AnalysisContext,
        progress: ParallelProgress,
    ) -> list[DetectorResult]:
        results: list[DetectorResult] = []
        detectors = self._detectors(progress)
        detector_count = max(1, len(detectors))
        for index, detector in enumerate(detectors):
            start = index / detector_count
            width = 1 / detector_count
            detector_context = replace(
                context,
                progress_callback=lambda value, label, start=start, width=width: progress.update(
                    "technical",
                    start + width * value,
                    label,
                ),
            )
            progress.update("technical", start, f"Analyse technique : {detector.name}")
            results.append(self._run_detector(detector, detector_context))
            progress.update(
                "technical",
                start + width,
                f"Analyse technique terminée : {detector.name}",
            )
        return results

    @staticmethod
    def _write_report(report: AnalysisReport, path: Path) -> None:
        path.write_text(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _ocr_artifacts(report: OcrReport | None, suffix: str) -> tuple[str, ...]:
    if report is None or not report.success:
        return ()
    return tuple(path for path in report.artifacts if Path(path).suffix.casefold() == suffix)
