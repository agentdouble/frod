"""Sequential OCR and semantic analysis branch shared by PDF and image pipelines."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fraude_detector.config import AnalysisConfig
from fraude_detector.detectors.ocr import OcrDetector
from fraude_detector.extraction_reconciliation import reconcile_extraction
from fraude_detector.llm_classifier import classify_document
from fraude_detector.llm_extractor import extract_document
from fraude_detector.llm_verifier import verify_extraction
from fraude_detector.models import (
    DetectorResult,
    DocumentClassification,
    DocumentExtraction,
    ExtractionVerification,
    OcrReport,
)


@dataclass(frozen=True, slots=True)
class ContentAnalysisResult:
    """Immutable result of the OCR, classification, extraction and verification chain."""

    ocr_report: OcrReport
    detector_result: DetectorResult
    classification: DocumentClassification | None = None
    extraction: DocumentExtraction | None = None
    verification: ExtractionVerification | None = None


def analyze_document_content(
    source: Path,
    destination: Path,
    config: AnalysisConfig,
    *,
    rendered_pages: tuple[str, ...] = (),
    progress_callback: Callable[[float, str], None] | None = None,
) -> ContentAnalysisResult:
    """Run the dependent content operations without touching the final report."""

    def report_progress(value: float, label: str) -> None:
        if progress_callback is not None:
            progress_callback(min(1.0, max(0.0, value)), label)

    detector = OcrDetector(config)
    report_progress(0.02, "Reconnaissance du contenu")
    ocr_report = detector.detect(
        source,
        destination,
        rendered_pages=rendered_pages,
    )
    if not ocr_report.success:
        report_progress(1.0, "Analyse du contenu indisponible")
        return ContentAnalysisResult(
            ocr_report=ocr_report,
            detector_result=detector.result(ocr_report),
        )

    classification = None
    extraction = None
    verification = None
    report_progress(0.25, "Contenu reconnu")

    if config.classification_enabled:
        report_progress(0.28, "Classification du document")
        try:
            classification = classify_document(ocr_report.markdown, config)
        except Exception as error:
            warnings.warn(f"Classification failed: {error}", stacklevel=2)
    report_progress(0.48, "Classification terminée")

    if config.extraction_enabled:
        report_progress(0.50, "Extraction des informations")
        try:
            extraction = extract_document(
                ocr_report.json_result,
                classification,
                config,
            )
        except Exception as error:
            warnings.warn(f"Extraction failed: {error}", stacklevel=2)
    report_progress(0.78, "Extraction terminée")

    if config.verification_enabled and extraction is not None:
        report_progress(0.80, "Vérification indépendante de l'extraction")
        try:
            verification = verify_extraction(
                ocr_report.json_result,
                extraction,
                classification,
                config,
            )
            extraction, verification = reconcile_extraction(extraction, verification)
        except Exception as error:
            warnings.warn(f"Extraction verification failed: {error}", stacklevel=2)

    report_progress(1.0, "Analyse du contenu terminée")
    return ContentAnalysisResult(
        ocr_report=ocr_report,
        detector_result=detector.result(ocr_report),
        classification=classification,
        extraction=extraction,
        verification=verification,
    )
