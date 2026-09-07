from __future__ import annotations

from lxml import etree

from fraude_detector.config import AnalysisConfig
from fraude_detector.document_authenticity import build_document_authenticity_result
from fraude_detector.laboratory.facturx import _facturx_fields
from fraude_detector.laboratory.structured_consistency import (
    add_extraction_consistency_checks,
    compare_visible_fields,
)
from fraude_detector.models import (
    DocumentExtraction,
    ExtractedFact,
    ExtractionCoverage,
    LaboratoryCheck,
    LaboratoryObservation,
    LaboratoryReport,
)


def test_structured_fields_match_punctuation_and_date_variants() -> None:
    comparison = compare_visible_fields(
        {
            "invoice_number": "FAC-2026-0042",
            "issue_date": "20260804",
            "grand_total": "2500.00",
            "seller_name": "Prestataire absent",
        },
        "Facture FAC 2026/0042 émise le 04/08/2026. Total : 2.500,00 EUR.",
    )

    assert comparison.matched_fields == (
        "invoice_number",
        "issue_date",
        "grand_total",
    )
    assert comparison.unmatched_fields == ("seller_name",)


def test_facturx_extracts_only_comparable_header_fields() -> None:
    root = etree.fromstring(
        b"""<rsm:CrossIndustryInvoice
          xmlns:rsm="urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100"
          xmlns:ram="urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100">
          <rsm:ExchangedDocument>
            <ram:ID>FAC-42</ram:ID>
            <ram:IssueDateTime><ram:DateTimeString>20260907</ram:DateTimeString></ram:IssueDateTime>
          </rsm:ExchangedDocument>
          <rsm:SupplyChainTradeTransaction>
            <ram:ApplicableHeaderTradeAgreement>
              <ram:SellerTradeParty><ram:Name>Cabinet Exemple</ram:Name></ram:SellerTradeParty>
            </ram:ApplicableHeaderTradeAgreement>
            <ram:ApplicableHeaderTradeSettlement>
              <ram:SpecifiedTradeSettlementHeaderMonetarySummation>
                <ram:GrandTotalAmount>125.00</ram:GrandTotalAmount>
              </ram:SpecifiedTradeSettlementHeaderMonetarySummation>
            </ram:ApplicableHeaderTradeSettlement>
          </rsm:SupplyChainTradeTransaction>
        </rsm:CrossIndustryInvoice>"""
    )

    assert _facturx_fields(root) == {
        "invoice_number": "FAC-42",
        "issue_date": "20260907",
        "seller_name": "Cabinet Exemple",
        "grand_total": "125.00",
    }


def test_only_actionable_authenticity_observations_are_scored() -> None:
    report = LaboratoryReport(
        schema_version="1.0",
        checks=(
            LaboratoryCheck(
                code="pades",
                title="Signature",
                purpose="test",
                state="attention",
                summary="test",
                observations=(
                    LaboratoryObservation(
                        code="PDF_SIGNATURE_INVALID",
                        title="Signature invalide",
                        summary="test",
                        state="attention",
                        strength="strong",
                        explanation="test",
                    ),
                    LaboratoryObservation(
                        code="PDF_SIGNATURE_INTACT_UNTRUSTED",
                        title="Certificat inconnu",
                        summary="test",
                        state="indeterminate",
                        strength="moderate",
                        explanation="test",
                    ),
                ),
            ),
            LaboratoryCheck(
                code="facturx",
                title="Factur-X",
                purpose="test",
                state="attention",
                summary="test",
                observations=(
                    LaboratoryObservation(
                        code="FACTURX_VISIBLE_MISMATCH",
                        title="Divergence",
                        summary="test",
                        state="attention",
                        strength="strong",
                        explanation="test",
                    ),
                ),
            ),
        ),
    )

    result = build_document_authenticity_result(report, AnalysisConfig())

    assert result.status == "completed"
    assert [finding.code for finding in result.findings] == [
        "PDF_SIGNATURE_INVALID",
        "FACTURX_VISIBLE_MISMATCH",
    ]
    assert [finding.risk_points for finding in result.findings] == [20.0, 20.0]
    assert all("inconnu" not in finding.title.casefold() for finding in result.findings)


def test_structured_mismatch_requires_matching_anchors_and_a_conflicting_fact() -> None:
    report = LaboratoryReport(
        schema_version="1.0",
        checks=(
            LaboratoryCheck(
                code="facturx",
                title="Factur-X",
                purpose="test",
                state="clear",
                summary="XML valide",
                observations=(
                    LaboratoryObservation(
                        code="FACTURX_XSD_VALID",
                        title="XML valide",
                        summary="test",
                        state="clear",
                        strength="moderate",
                        explanation="test",
                        evidence={
                            "structured_fields": {
                                "invoice_number": "FAC-42",
                                "seller_name": "Cabinet Exemple",
                                "grand_total": "125.00",
                            }
                        },
                    ),
                ),
            ),
        ),
    )
    extraction = DocumentExtraction(
        schema_version="test",
        family="facture_recu",
        language="fr",
        country="LU",
        facts=(
            _fact("invoice_number", "invoice", "FAC 42"),
            _fact("organization_name", "issuer", "Cabinet Exemple"),
            _fact("monetary_amount", "total", "130,00 EUR"),
        ),
        additional_fields=(),
        tables=(),
        coverage=ExtractionCoverage(3, 3, 3, 0, 0, 0, 0),
        passes=1,
    )

    compared = add_extraction_consistency_checks(report, extraction, minimum_matches=2)

    check = compared.checks[0]
    assert check.state == "attention"
    mismatch = next(
        observation
        for observation in check.observations
        if observation.code == "FACTURX_VISIBLE_MISMATCH"
    )
    assert mismatch.evidence["matched_field_count"] == 2
    assert mismatch.evidence["mismatched_fields"] == ("grand_total",)
    assert "structured_fields" not in check.observations[0].evidence


def _fact(field_code: str, role: str, value: str) -> ExtractedFact:
    return ExtractedFact(
        field_code=field_code,
        role=role,
        raw_label=None,
        raw_value=value,
        normalized_value=None,
        normalization_status="raw_only",
        page=1,
    )
