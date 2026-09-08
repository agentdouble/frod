from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import requests

from fraude_detector.config import AnalysisConfig
from fraude_detector.llm_verifier import LLMExtractionVerifier
from fraude_detector.models import (
    DocumentClassification,
    DocumentExtraction,
    ExtractedFact,
    ExtractedTable,
    ExtractionCoverage,
)


class _Response:
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self) -> dict[str, Any]:
        return self.payload


def _chat_response(payload: dict[str, Any]) -> _Response:
    return _Response({"choices": [{"message": {"content": json.dumps(payload)}}]})


def _extraction() -> DocumentExtraction:
    return DocumentExtraction(
        schema_version="0.1-experimental",
        family="facture_recu",
        language="fr",
        country="LU",
        facts=(
            ExtractedFact(
                field_code="invoice_number",
                role="invoice",
                raw_label="N° facture",
                raw_value="FAC 2026/42",
                normalized_value="fac 2026 42",
                normalization_status="normalized",
                page=1,
                region_ids=("p001-r000",),
            ),
            ExtractedFact(
                field_code="date",
                role="issue",
                raw_label="Date",
                raw_value="31/O8/2O26",
                normalized_value=None,
                normalization_status="ambiguous",
                page=1,
                region_ids=("p001-r001",),
            ),
        ),
        additional_fields=(),
        tables=(),
        coverage=ExtractionCoverage(
            total_regions=2,
            accounted_regions=2,
            mapped_regions=2,
            table_regions=0,
            boilerplate_regions=0,
            unstructured_regions=0,
            unreadable_regions=0,
        ),
        passes=1,
    )


def _ocr() -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "index": 4,
                "label": "text",
                "native_label": "paragraph",
                "bbox_2d": [10, 20, 400, 90],
                "content": "N° facture : FAC 2026/42",
            },
            {
                "index": 5,
                "label": "text",
                "native_label": "paragraph",
                "bbox_2d": [10, 100, 400, 170],
                "content": "Date : 31/O8/2O26",
            },
        ]
    ]


def _classification() -> DocumentClassification:
    return DocumentClassification(
        family="facture_recu",
        reliability=0.94,
        language="fr",
        country="LU",
    )


def _review(
    target_id: str,
    verdict: str,
    region_id: str,
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "target_type": "fact",
        "verdict": verdict,
        "explanation": "La valeur et son rôle sont compatibles avec la source.",
        "source_region_ids": [region_id],
        "suggested_value": None,
        "suggested_field_code": None,
        "suggested_role": None,
        "problematic_row_indexes": [],
    }


def test_fresh_conservative_verification_accepts_a_clean_extraction(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    result = {
        "issues": [],
        "possible_omissions": [],
    }

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return _chat_response(result)

    monkeypatch.setattr(requests, "post", fake_post)
    verification = LLMExtractionVerifier(
        AnalysisConfig(
            verification_enabled=True,
            verification_url="http://minimax.internal:8030",
        )
    ).verify(_ocr(), _extraction(), _classification())

    assert verification.status == "clean"
    assert verification.reviews == ()
    assert calls[0]["url"] == "http://minimax.internal:8030/v1/chat/completions"
    assert "reasoning_effort" not in calls[0]["json"]
    assert calls[0]["json"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }
    messages = calls[0]["json"]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Il est normal et attendu" in messages[0]["content"]
    assert "Ne cherche jamais à produire un quota" in messages[0]["content"]
    assert "strictement\ndifférentielle" in messages[1]["content"]
    assert "ne retourne aucun avis" in messages[1]["content"]
    assert "doivent être rédigées en français" in messages[1]["content"]
    assert 'order="4"' in messages[1]["content"]
    assert 'native_label="paragraph"' in messages[1]["content"]
    assert 'bbox_2d="10,20,400,90"' in messages[1]["content"]
    assert "payment = paiement effectué" in messages[1]["content"]
    assert "declaration = déclaration ou dépôt" in messages[1]["content"]
    assert "event = événement/sinistre" in messages[1]["content"]
    assert verification.schema_version == "0.4-experimental"
    assert verification.prompt_version.startswith("verification-")
    suggested_role_schema = calls[0]["json"]["response_format"]["json_schema"]["schema"][
        "properties"
    ]["issues"]["items"]["properties"]["suggested_role"]
    assert "payment" in suggested_role_schema["enum"]
    issue_properties = calls[0]["json"]["response_format"]["json_schema"]["schema"][
        "properties"
    ]["issues"]["items"]["properties"]
    assert "confidence" not in issue_properties
    assert "problematic_row_indexes" in issue_properties


def test_omission_without_an_explicit_value_cannot_create_a_visible_issue(
    monkeypatch: Any,
) -> None:
    result = {
        "issues": [],
        "possible_omissions": [
            {
                "description": "Une référence pourrait manquer.",
                "proposed_field_code": "document_number",
                "proposed_role": "document",
                "proposed_value": None,
                "source_region_ids": ["p001-r000"],
            }
        ],
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    verification = LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), _extraction(), _classification()
    )

    assert verification.status == "clean"
    assert verification.reviews == ()
    assert verification.omissions == ()


def test_unlocated_contradiction_cannot_create_a_visible_issue(monkeypatch: Any) -> None:
    unlocated = _review("fact-0001", "contradicted", "unknown-region")
    unlocated["suggested_value"] = "FAC-2026-43"
    result = {
        "issues": [unlocated],
        "possible_omissions": [],
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    verification = LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), _extraction(), _classification()
    )

    assert verification.status == "clean"
    assert verification.reviews == ()


def test_concrete_grounded_contradiction_is_reported_without_mutation(
    monkeypatch: Any,
) -> None:
    contradiction = _review("fact-0001", "contradicted", "p001-r000")
    contradiction["explanation"] = "La source indique 42 et non 43."
    contradiction["suggested_value"] = "FAC 2026/42"
    result = {
        "issues": [contradiction],
        "possible_omissions": [],
    }
    initial = _extraction()
    extraction = replace(
        initial,
        facts=(
            replace(
                initial.facts[0],
                raw_value="FAC 2026/43",
                normalized_value="fac 2026 43",
            ),
            initial.facts[1],
        ),
    )
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    verification = LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), extraction, _classification()
    )

    assert verification.status == "attention"
    assert verification.reviews[0].verdict == "contradicted"
    assert verification.reviews[0].suggested_value == "FAC 2026/42"
    assert extraction.facts[0].raw_value == "FAC 2026/43"


def test_contradiction_without_an_applicable_change_is_not_reported(
    monkeypatch: Any,
) -> None:
    contradiction = _review("fact-0001", "contradicted", "p001-r000")
    contradiction["suggested_field_code"] = "invoice_number"
    result = {
        "issues": [contradiction],
        "possible_omissions": [],
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    verification = LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), _extraction(), _classification()
    )

    assert verification.status == "clean"
    assert verification.reviews == ()


def test_empty_differential_output_means_all_targets_were_controlled(monkeypatch: Any) -> None:
    result = {
        "issues": [],
        "possible_omissions": [],
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    verification = LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), _extraction(), _classification()
    )

    assert verification.status == "clean"
    assert verification.expected_targets == 2


def test_verifier_receives_table_cells_for_semantic_validation(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    extraction = replace(
        _extraction(),
        tables=(
            ExtractedTable(
                title="Opérations",
                semantic_type="transactions",
                headers=("Date", "Libellé", "Montant"),
                column_roles=("transaction_date", "description", "amount"),
                rows=(("01/01/2026", "SECRET_ROW_VALUE", "10 EUR"),),
                row_roles=("transaction",),
                pages=(1,),
                region_ids=("p001-r000",),
            ),
        ),
    )

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        calls.append(kwargs["json"])
        return _chat_response({"issues": [], "possible_omissions": []})

    monkeypatch.setattr(requests, "post", fake_post)
    LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), extraction, _classification()
    )

    prompt = calls[0]["messages"][1]["content"]
    assert '"row_count":1' in prompt
    assert "SECRET_ROW_VALUE" in prompt
    assert "problematic_row_indexes" in prompt


def test_concrete_table_misalignment_is_reported_without_mutating_rows(
    monkeypatch: Any,
) -> None:
    extraction = replace(
        _extraction(),
        tables=(
            ExtractedTable(
                title="Opérations",
                semantic_type="transactions",
                headers=("Date", "Lieu", "Montant"),
                column_roles=("transaction_date", "description", "amount"),
                rows=(("Luxembourg", "21/01/2025", "125,-"),),
                row_roles=("transaction",),
                pages=(1,),
                region_ids=("p001-r000",),
            ),
        ),
    )
    issue = {
        "target_id": "table-0001",
        "target_type": "table",
        "verdict": "ambiguous",
        "explanation": "La date et le lieu semblent placés dans les colonnes opposées.",
        "source_region_ids": ["p001-r000"],
        "suggested_value": None,
        "suggested_field_code": None,
        "suggested_role": None,
        "problematic_row_indexes": [0],
    }
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _chat_response({"issues": [issue], "possible_omissions": []}),
    )

    verification = LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), extraction, _classification()
    )

    assert verification.status == "attention"
    assert verification.reviews[0].target_type == "table"
    assert verification.reviews[0].problematic_row_indexes == (0,)
    assert extraction.tables[0].rows == (("Luxembourg", "21/01/2025", "125,-"),)
