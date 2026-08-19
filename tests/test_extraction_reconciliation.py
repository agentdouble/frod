from __future__ import annotations

from fraude_detector.extraction_reconciliation import reconcile_extraction
from fraude_detector.models import (
    DocumentExtraction,
    ExtractedFact,
    ExtractionCoverage,
    ExtractionReview,
    ExtractionVerification,
)


def test_payment_date_role_correction_is_applied_with_an_audit_trail() -> None:
    extraction = _extraction()
    verification = ExtractionVerification(
        schema_version="0.1-experimental",
        status="attention",
        expected_targets=1,
        reviewed_targets=1,
        reviews=(
            ExtractionReview(
                target_id="fact-0001",
                target_type="fact",
                verdict="contradicted",
                confidence=1.0,
                explanation="The source identifies a payment date, not a due date.",
                source_region_ids=("p001-r007",),
                suggested_field_code="date",
                suggested_role="payment",
            ),
        ),
        omissions=(),
    )

    resolved, reviewed = reconcile_extraction(extraction, verification)

    assert extraction.facts[0].role == "due"
    assert resolved.facts[0].role == "payment"
    assert resolved.facts[0].raw_value == "18/07/2026"
    assert reviewed.status == "clean"
    assert reviewed.reviews[0].correction_applied is True
    assert reviewed.reviews[0].original_field_code == "date"
    assert reviewed.reviews[0].original_role == "due"
    assert reviewed.reviews[0].original_value == "18/07/2026"


def test_corrected_value_is_normalized_again() -> None:
    extraction = _extraction()
    verification = ExtractionVerification(
        schema_version="0.1-experimental",
        status="attention",
        expected_targets=1,
        reviewed_targets=1,
        reviews=(
            ExtractionReview(
                target_id="fact-0001",
                target_type="fact",
                verdict="contradicted",
                confidence=0.99,
                explanation="The OCR date was misread.",
                source_region_ids=("p001-r007",),
                suggested_value="19/07/2026",
            ),
        ),
        omissions=(),
    )

    resolved, reviewed = reconcile_extraction(extraction, verification)

    assert resolved.facts[0].raw_value == "18/07/2026"
    assert resolved.facts[0].corrected_value == "19/07/2026"
    assert resolved.facts[0].normalized_value == "2026-07-19"
    assert resolved.facts[0].normalization_status == "normalized"
    assert reviewed.reviews[0].correction_applied is True


def _extraction() -> DocumentExtraction:
    return DocumentExtraction(
        schema_version="0.1-experimental",
        family="facture_recu",
        language="fr",
        country="LU",
        facts=(
            ExtractedFact(
                field_code="date",
                role="due",
                raw_label="Payé le",
                raw_value="18/07/2026",
                normalized_value="2026-07-18",
                normalization_status="normalized",
                confidence=0.95,
                page=1,
                region_ids=("p001-r007",),
            ),
        ),
        additional_fields=(),
        tables=(),
        coverage=ExtractionCoverage(
            total_regions=1,
            accounted_regions=1,
            mapped_regions=1,
            table_regions=0,
            boilerplate_regions=0,
            unstructured_regions=0,
            unreadable_regions=0,
        ),
        passes=1,
    )
