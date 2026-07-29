"""Experimental controls kept outside the production scoring pipeline."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pypdfium2
from pypdf import PdfReader

from fraude_detector.config import AnalysisConfig
from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.laboratory.facturx import analyze_facturx
from fraude_detector.laboratory.images import analyze_image_laboratory
from fraude_detector.laboratory.objects import analyze_fonts_and_hidden_objects
from fraude_detector.laboratory.revisions import analyze_all_revisions
from fraude_detector.laboratory.signatures import analyze_pdf_signatures
from fraude_detector.laboratory.two_d_doc import analyze_two_d_doc
from fraude_detector.models import LaboratoryCheck, LaboratoryReport
from fraude_detector.pdf_revisions import find_valid_revision_end_offsets


def analyze_pdf_laboratory(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    config: AnalysisConfig,
    password: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> LaboratoryReport:
    """Run non-scoring experiments over a PDF already accepted by Frod."""

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() / "laboratory"
    destination.mkdir(parents=True, exist_ok=True)
    raw_pdf = source.read_bytes()
    reader = PdfReader(io.BytesIO(raw_pdf), strict=False)
    if reader.is_encrypted and (not password or reader.decrypt(password) == 0):
        raise ValueError("Le laboratoire ne peut pas ouvrir ce PDF chiffre.")
    pdfium_document = pypdfium2.PdfDocument(raw_pdf, password=password)

    def progress(value: float, label: str) -> None:
        if progress_callback is not None:
            progress_callback(min(1.0, max(0.0, value)), label)

    try:
        context = AnalysisContext(
            input_path=source,
            output_dir=destination,
            raw_pdf=raw_pdf,
            pdfium_document=pdfium_document,
            reader=reader,
            config=config,
            revision_end_offsets=find_valid_revision_end_offsets(raw_pdf, password),
            password=password,
        )
        checks: list[LaboratoryCheck] = []

        progress(0.03, "Laboratoire : signatures PDF")
        signature_checks = _guard_many(
            ("pades", "post_signature"),
            lambda: analyze_pdf_signatures(context),
        )
        checks.extend(signature_checks)

        progress(0.20, "Laboratoire : Factur-X")
        checks.append(_guard_one("facturx", lambda: analyze_facturx(context)))

        progress(0.34, "Laboratoire : 2D-Doc")
        checks.append(_guard_one("two_d_doc", lambda: analyze_two_d_doc(context)))

        progress(0.56, "Laboratoire : polices et objets")
        checks.append(
            _guard_one(
                "fonts_hidden_objects",
                lambda: analyze_fonts_and_hidden_objects(context),
            )
        )

        progress(0.72, "Laboratoire : historique complet")
        checks.append(
            _guard_one(
                "all_revisions",
                lambda: analyze_all_revisions(context),
            )
        )
        progress(1.0, "Laboratoire termine")
        return LaboratoryReport(schema_version="0.1-experimental", checks=tuple(checks))
    finally:
        pdfium_document.close()


def _guard_one(code: str, operation: Callable[[], LaboratoryCheck]) -> LaboratoryCheck:
    try:
        return operation()
    except Exception as error:
        return _error_check(code, error)


def _guard_many(
    codes: tuple[str, ...],
    operation: Callable[[], tuple[LaboratoryCheck, ...]],
) -> tuple[LaboratoryCheck, ...]:
    try:
        return operation()
    except Exception as error:
        return tuple(_error_check(code, error) for code in codes)


def _error_check(code: str, error: Exception) -> LaboratoryCheck:
    labels = {
        "pades": ("Signature PDF / PAdES", "Verifier les signatures numeriques du PDF."),
        "post_signature": (
            "Modifications apres signature",
            "Identifier les changements ajoutes apres chaque signature.",
        ),
        "facturx": (
            "Factur-X",
            "Detecter et valider le XML de facture embarque.",
        ),
        "two_d_doc": (
            "2D-Doc",
            "Lire et verifier cryptographiquement les codes 2D-Doc.",
        ),
        "fonts_hidden_objects": (
            "Polices et objets masques",
            "Rechercher des objets invisibles et des ruptures typographiques ciblees.",
        ),
        "all_revisions": (
            "Historique complet",
            "Comparer toutes les revisions incrementales conservees.",
        ),
    }
    title, purpose = labels[code]
    return LaboratoryCheck(
        code=code,
        title=title,
        purpose=purpose,
        state="error",
        summary="Controle interrompu.",
        limitations=(f"{type(error).__name__}: {str(error)[:240]}",),
    )


__all__ = ["analyze_image_laboratory", "analyze_pdf_laboratory"]
