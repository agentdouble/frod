"""Sequential OCR and semantic analysis branch shared by PDF and image pipelines."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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

    report_progress(0.25, "Contenu reconnu")
    return analyze_recognized_content(
        ocr_report,
        config,
        progress_callback=lambda value, label: report_progress(0.25 + 0.75 * value, label),
    )


def analyze_recognized_content(
    ocr_report: OcrReport,
    config: AnalysisConfig,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> ContentAnalysisResult:
    """Run semantic operations over an existing OCR result without repeating OCR."""

    def report_progress(value: float, label: str) -> None:
        if progress_callback is not None:
            progress_callback(min(1.0, max(0.0, value)), label)

    detector = OcrDetector(config)
    if not ocr_report.success:
        report_progress(1.0, "Analyse du contenu indisponible")
        return ContentAnalysisResult(
            ocr_report=ocr_report,
            detector_result=detector.result(ocr_report),
        )

    classification = None
    extraction = None
    verification = None

    def cancelled() -> bool:
        return cancel_callback is not None and cancel_callback()

    if config.classification_enabled and not cancelled():
        report_progress(0.04, "Classification du document")
        try:
            classification = classify_document(ocr_report.markdown, config)
        except Exception as error:
            logger.warning("Classification failed: %s", error)
    report_progress(0.30, "Classification terminée")

    if config.extraction_enabled and not cancelled():
        report_progress(0.34, "Extraction des informations")
        try:
            extraction = extract_document(
                ocr_report.json_result,
                classification,
                config,
            )
        except Exception as error:
            logger.warning("Extraction failed: %s", error)
    report_progress(0.70, "Extraction terminée")

    if config.verification_enabled and extraction is not None and not cancelled():
        report_progress(0.74, "Vérification indépendante de l'extraction")
        try:
            verification = verify_extraction(
                ocr_report.json_result,
                extraction,
                classification,
                config,
            )
            extraction, verification = reconcile_extraction(extraction, verification)
        except Exception as error:
            logger.warning("Extraction verification failed: %s", error)

    report_progress(1.0, "Analyse sémantique terminée")
    return ContentAnalysisResult(
        ocr_report=ocr_report,
        detector_result=detector.result(ocr_report),
        classification=classification,
        extraction=extraction,
        verification=verification,
    )
