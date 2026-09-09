from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraude_detector.laboratory.ocr import analyze_ocr_laboratory
from fraude_detector.models import OcrReport
from fraude_detector.ocr_consistency import (
    build_ocr_content_result,
    measure_ocr_reliability,
)

FIXTURE = Path("tests/fixtures/ocr/releve-bancaire-anomalies")


def test_precomputed_bank_statement_reaches_review_with_independent_groups() -> None:
    report = _fixture_report()

    result = build_ocr_content_result(report)
    finding = result.findings[0]

    assert finding.code == "OCR_CONTENT_CONSISTENCY"
    assert finding.risk_points == 30
    assert finding.confidence == 0.8
    assert set(finding.evidence["groups"]) == {
        "identity_consistency",
        "geographic_consistency",
        "financial_consistency",
    }


def test_low_representation_agreement_keeps_anomalies_unscored() -> None:
    report = _fixture_report(markdown="unrelated OCR fragment")

    result = build_ocr_content_result(report)
    finding = result.findings[0]

    assert finding.code == "OCR_CONTENT_UNSCORED"
    assert finding.risk_points == 0
    assert finding.evidence["point_factor"] == 0


def test_native_region_confidence_is_used_when_available() -> None:
    payload = [
        [
            {
                "label": "text",
                "content": "A sufficiently long recognized document region",
                "confidence": 0.91,
            }
        ]
    ]

    reliability = measure_ocr_reliability(payload, "")

    assert reliability.source == "native_confidence"
    assert reliability.native_confidence == pytest.approx(0.91)
    assert reliability.score == pytest.approx(0.91)


def test_native_confidence_uses_a_conservative_lower_quartile() -> None:
    payload = [
        [
            {"label": "text", "content": "High confidence region", "confidence": 0.99},
            {"label": "text", "content": "Low confidence region", "confidence": 0.40},
        ]
    ]

    reliability = measure_ocr_reliability(payload, "")

    assert reliability.native_confidence == pytest.approx(0.40)
    assert reliability.score == pytest.approx(0.40)


def test_insufficient_ocr_quality_disables_points_even_with_matching_views() -> None:
    payload = [[{"label": "text", "content": "SWIFT BIL2000488"}]]
    markdown = "SWIFT BIL2000488"
    checks = analyze_ocr_laboratory(payload)
    assert checks[0].state == "indeterminate"

    result = build_ocr_content_result(
        OcrReport(
            success=True,
            error_message=None,
            markdown=markdown,
            json_result=payload,
        )
    )

    finding = result.findings[0]
    assert finding.code == "OCR_CONTENT_UNSCORED"
    assert finding.risk_points == 0
    assert finding.evidence["quality_gate"] == "indeterminate"


def test_one_foreign_banking_field_remains_an_unscored_diagnostic() -> None:
    payload = [
        [
            {
                "label": "text",
                "content": "BRANCH ADDRESS : 38 Place de la Gare, Luxembourg",
            },
            {
                "label": "text",
                "content": "CKYC ID : 12345678901234",
            },
        ]
    ]
    markdown = "\n".join(item["content"] for item in payload[0])

    result = build_ocr_content_result(
        OcrReport(
            success=True,
            error_message=None,
            markdown=markdown,
            json_result=payload,
        )
    )
    finding = result.findings[0]

    assert finding.code == "OCR_CONTENT_DIAGNOSTIC"
    assert finding.risk_points == 0
    assert finding.evidence["groups"] == {}


def test_card_conflict_and_luhn_failure_share_one_point_group() -> None:
    payload = [
        [
            {
                "label": "text",
                "content": "POST GOLD VISA CREDIT CARD ACCOUNT 4111-1111-1111-1111",
            },
            {
                "label": "text",
                "content": "CREDIT CARD NO : 4111-1111-1111-1121",
            },
        ]
    ]
    markdown = "\n".join(item["content"] for item in payload[0])

    result = build_ocr_content_result(
        OcrReport(
            success=True,
            error_message=None,
            markdown=markdown,
            json_result=payload,
        )
    )
    finding = result.findings[0]

    assert finding.risk_points == 18
    identity = finding.evidence["groups"]["identity_consistency"]
    assert identity["base_points"] == 18
    assert set(identity["signals"]) == {
        "OCR_CARD_VALUES_CONFLICT",
        "OCR_CARD_LUHN_INVALID",
    }


def test_two_distinct_valid_cards_do_not_create_a_finding() -> None:
    payload = [
        [
            {
                "label": "text",
                "content": "Primary credit card number: 4111 1111 1111 1111",
            },
            {
                "label": "text",
                "content": "Secondary credit card number: 5555 5555 5555 4444",
            },
        ]
    ]
    markdown = "\n".join(item["content"] for item in payload[0])

    result = build_ocr_content_result(
        OcrReport(
            success=True,
            error_message=None,
            markdown=markdown,
            json_result=payload,
        )
    )
    assert result.findings == ()


def test_masked_card_does_not_create_a_content_finding() -> None:
    payload = [
        [
            {
                "label": "text",
                "content": "Numéro de carte : 9401 XXXX XXXX 0100 00",
            }
        ]
    ]
    markdown = payload[0][0]["content"]

    result = build_ocr_content_result(
        OcrReport(
            success=True,
            error_message=None,
            markdown=markdown,
            json_result=payload,
        )
    )

    assert result.findings == ()


def test_invalid_labeled_siret_uses_the_existing_ocr_reliability_gate() -> None:
    payload = [
        [
            {
                "label": "text",
                "content": (
                    "Facture avec un contenu suffisamment long pour contrôler "
                    "le SIRET : 732 829 320 00075"
                ),
            },
            {"label": "text", "content": "Adresse et détails complémentaires du fournisseur"},
        ]
    ]
    markdown = "\n".join(item["content"] for item in payload[0])

    result = build_ocr_content_result(
        OcrReport(
            success=True,
            error_message=None,
            markdown=markdown,
            json_result=payload,
        )
    )

    finding = result.findings[0]
    assert finding.risk_points == 6
    assert finding.evidence["groups"]["identity_consistency"]["signals"] == ("OCR_SIRET_INVALID",)


def test_invalid_belgian_rrn_uses_the_existing_ocr_reliability_gate() -> None:
    payload = [
        [
            {
                "label": "text",
                "content": (
                    "Attestation belge contenant suffisamment de texte pour contrôler "
                    "le RRN : 85.07.30-033.29"
                ),
            },
            {"label": "text", "content": "Coordonnées complémentaires du titulaire belge"},
        ]
    ]
    markdown = "\n".join(item["content"] for item in payload[0])

    result = build_ocr_content_result(
        OcrReport(success=True, error_message=None, markdown=markdown, json_result=payload)
    )

    finding = result.findings[0]
    assert finding.risk_points == 6
    assert finding.evidence["groups"]["identity_consistency"]["signals"] == (
        "OCR_BELGIAN_RRN_INVALID",
    )


def _fixture_report(*, markdown: str | None = None) -> OcrReport:
    payload = json.loads((FIXTURE / "document.json").read_text(encoding="utf-8"))
    return OcrReport(
        success=True,
        error_message=None,
        markdown=(
            (FIXTURE / "document.md").read_text(encoding="utf-8") if markdown is None else markdown
        ),
        json_result=payload,
    )
