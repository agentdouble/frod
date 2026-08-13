from __future__ import annotations

import json
from typing import Any

import requests

from fraude_detector.config import AnalysisConfig
from fraude_detector.llm_verifier import LLMExtractionVerifier
from fraude_detector.models import (
    DocumentClassification,
    DocumentExtraction,
    ExtractedFact,
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
                confidence=0.95,
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
                confidence=0.78,
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
            {"label": "text", "content": "N° facture : FAC 2026/42"},
            {"label": "text", "content": "Date : 31/O8/2O26"},
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
    confidence: float,
    region_id: str,
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "target_type": "fact",
        "verdict": verdict,
        "confidence": confidence,
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
        "reviews": [
            _review("fact-0001", "supported", 0.99, "p001-r000"),
            _review("fact-0002", "plausible", 0.90, "p001-r001"),
        ],
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
    assert verification.reviewed_targets == 2
    assert [review.verdict for review in verification.reviews] == ["supported", "plausible"]
    assert calls[0]["url"] == "http://minimax.internal:8030/v1/chat/completions"
    messages = calls[0]["json"]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Il est normal et attendu" in messages[0]["content"]
    assert "Ne cherche jamais à produire un quota" in messages[0]["content"]
    assert "pas une anomalie" in messages[1]["content"]
    assert "tous les verdicts" in messages[1]["content"]
    assert "doivent être\n   en anglais" in messages[1]["content"]


def test_low_confidence_suspicion_cannot_create_a_visible_issue(monkeypatch: Any) -> None:
    weak_review = _review("fact-0001", "contradicted", 0.61, "p001-r000")
    weak_review["suggested_value"] = "FAC-2026-43"
    result = {
        "reviews": [
            weak_review,
            _review("fact-0002", "supported", 0.93, "p001-r001"),
        ],
        "possible_omissions": [
            {
                "description": "Une référence pourrait manquer.",
                "proposed_field_code": "document_number",
                "proposed_role": "document",
                "proposed_value": "INCERTAIN",
                "confidence": 0.62,
                "source_region_ids": ["p001-r000"],
            }
        ],
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    verification = LLMExtractionVerifier(
        AnalysisConfig(verification_enabled=True, verification_issue_min_confidence=0.80)
    ).verify(_ocr(), _extraction(), _classification())

    assert verification.status == "clean"
    assert verification.reviews[0].verdict == "plausible"
    assert verification.reviews[0].suggested_value is None
    assert verification.omissions == ()


def test_unlocated_contradiction_cannot_create_a_visible_issue(monkeypatch: Any) -> None:
    unlocated = _review("fact-0001", "contradicted", 0.99, "unknown-region")
    unlocated["suggested_value"] = "FAC-2026-43"
    result = {
        "reviews": [
            unlocated,
            _review("fact-0002", "supported", 0.95, "p001-r001"),
        ],
        "possible_omissions": [],
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    verification = LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), _extraction(), _classification()
    )

    assert verification.status == "clean"
    assert verification.reviews[0].verdict == "plausible"
    assert verification.reviews[0].source_region_ids == ()
    assert verification.reviews[0].suggested_value is None


def test_concrete_high_confidence_contradiction_is_reported_without_mutation(
    monkeypatch: Any,
) -> None:
    contradiction = _review("fact-0001", "contradicted", 0.96, "p001-r000")
    contradiction["explanation"] = "La source indique 42 et non 43."
    contradiction["suggested_value"] = "FAC 2026/42"
    result = {
        "reviews": [
            contradiction,
            _review("fact-0002", "supported", 0.94, "p001-r001"),
        ],
        "possible_omissions": [],
    }
    extraction = _extraction()
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    verification = LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), extraction, _classification()
    )

    assert verification.status == "attention"
    assert verification.reviews[0].verdict == "contradicted"
    assert verification.reviews[0].suggested_value == "FAC 2026/42"
    assert extraction.facts[0].raw_value == "FAC 2026/42"


def test_missing_target_review_marks_verification_incomplete(monkeypatch: Any) -> None:
    result = {
        "reviews": [_review("fact-0001", "supported", 0.99, "p001-r000")],
        "possible_omissions": [],
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    verification = LLMExtractionVerifier(AnalysisConfig(verification_enabled=True)).verify(
        _ocr(), _extraction(), _classification()
    )

    assert verification.status == "incomplete"
    assert verification.expected_targets == 2
    assert verification.reviewed_targets == 1
