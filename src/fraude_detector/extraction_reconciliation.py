"""Deterministic application of actionable extraction verification corrections."""

from __future__ import annotations

from dataclasses import replace

from fraude_detector.llm_extractor import normalize_extracted_value
from fraude_detector.models import (
    DocumentExtraction,
    ExtractionReview,
    ExtractionVerification,
)


def reconcile_extraction(
    extraction: DocumentExtraction,
    verification: ExtractionVerification,
) -> tuple[DocumentExtraction, ExtractionVerification]:
    """Apply explicit fact and additional-field corrections with an audit trail."""

    facts = list(extraction.facts)
    additional_fields = list(extraction.additional_fields)
    reviews: list[ExtractionReview] = []

    for review in verification.reviews:
        applied_review = review
        if review.verdict == "contradicted":
            if review.target_type == "fact":
                index = _target_index(review.target_id, "fact", len(facts))
                if index is not None:
                    current = facts[index]
                    field_code = review.suggested_field_code or current.field_code
                    role = review.suggested_role or current.role
                    current_value = current.corrected_value or current.raw_value
                    corrected_value = current.corrected_value
                    if review.suggested_value is not None:
                        corrected_value = (
                            review.suggested_value
                            if review.suggested_value != current.raw_value
                            else None
                        )
                    effective_value = corrected_value or current.raw_value
                    if (field_code, role, effective_value) != (
                        current.field_code,
                        current.role,
                        current_value,
                    ):
                        normalized_value, normalization_status = normalize_extracted_value(
                            field_code,
                            effective_value,
                            extraction.language,
                            extraction.country,
                        )
                        facts[index] = replace(
                            current,
                            field_code=field_code,
                            role=role,
                            corrected_value=corrected_value,
                            normalized_value=normalized_value,
                            normalization_status=normalization_status,
                        )
                        applied_review = replace(
                            review,
                            correction_applied=True,
                            original_value=current_value,
                            original_field_code=current.field_code,
                            original_role=current.role,
                        )
            elif review.target_type == "additional_field":
                index = _target_index(review.target_id, "additional", len(additional_fields))
                if index is not None and review.suggested_value is not None:
                    current = additional_fields[index]
                    current_value = current.corrected_value or current.raw_value
                    if review.suggested_value != current_value:
                        additional_fields[index] = replace(
                            current,
                            corrected_value=(
                                review.suggested_value
                                if review.suggested_value != current.raw_value
                                else None
                            ),
                        )
                        applied_review = replace(
                            review,
                            correction_applied=True,
                            original_value=current_value,
                        )
        reviews.append(applied_review)

    corrected_count = sum(review.correction_applied for review in reviews)
    unresolved = any(
        review.verdict in {"ambiguous", "contradicted"} and not review.correction_applied
        for review in reviews
    ) or bool(verification.omissions)
    if verification.status == "incomplete":
        status = "incomplete"
    else:
        status = "attention" if unresolved else "clean"

    resolved_extraction = replace(
        extraction,
        facts=tuple(facts),
        additional_fields=tuple(additional_fields),
    )
    resolved_verification = replace(
        verification,
        status=status,
        reviews=tuple(reviews),
        limitations=(
            (
                *verification.limitations,
                f"Corrections structurées appliquées automatiquement: {corrected_count}.",
            )
            if corrected_count
            else verification.limitations
        ),
    )
    return resolved_extraction, resolved_verification


def _target_index(target_id: str, prefix: str, length: int) -> int | None:
    expected_prefix = f"{prefix}-"
    if not target_id.startswith(expected_prefix):
        return None
    try:
        index = int(target_id.removeprefix(expected_prefix)) - 1
    except ValueError:
        return None
    return index if 0 <= index < length else None
