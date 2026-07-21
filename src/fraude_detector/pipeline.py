"""Orchestration of PDF validation, detectors, scoring and artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pypdfium2
from pypdf import PdfReader

from fraude_detector.config import AnalysisConfig
from fraude_detector.detectors import (
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
)
from fraude_detector.pdf_revisions import find_valid_revision_end_offsets
from fraude_detector.rendering import create_review_overlays, render_input_pages
from fraude_detector.scoring import assess_risk


class AnalysisPipeline:
    """Run independent detectors over an immutable source PDF."""

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        detectors: Iterable[Detector] | None = None,
    ) -> None:
        self.config = config or AnalysisConfig()
        self.detectors = tuple(
            detectors
            or (
                PdfStructureDetector(),
                RevisionDiffDetector(),
                PageCompositionDetector(),
                RasterAnomalyDetector(),
            )
        )

    def analyze(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        password: str | None = None,
    ) -> AnalysisReport:
        source = Path(input_path).expanduser().resolve()
        destination = Path(output_dir).expanduser().resolve()
        raw_pdf = self._read_input(source)

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
            detector_results = tuple(
                self._run_detector(detector, context) for detector in self.detectors
            )
            findings = tuple(
                finding
                for detector_result in detector_results
                for finding in detector_result.findings
            )
            review_overlays = create_review_overlays(context, rendered_pages, findings)
            forensics = tuple(
                dict.fromkeys(
                    artifact
                    for detector_result in detector_results
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
                },
                limitations=(
                    "Le score est une priorite de revue, pas une probabilite "
                    "ni une preuve de fraude.",
                    "Une reecriture complete du PDF peut supprimer tout historique de revisions.",
                    "Une signature, un formulaire, un tampon ou une annotation "
                    "peuvent expliquer legitimement un changement.",
                    "L'analyse ELA est un signal faible limite aux JPEG originaux "
                    "embarques et doit etre corroboree.",
                    "L'absence de signal ne prouve pas l'authenticite du document.",
                ),
            )
            self._write_report(report, destination / "report.json")
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
