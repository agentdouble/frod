from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from fraude_detector.config import AnalysisConfig
from fraude_detector.llm_synthesizer import SynthesisError, summarize_analysis
from fraude_detector.models import (
    DetectorResult,
    DocumentClassification,
    Finding,
    LaboratoryCheck,
    LaboratoryReport,
)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def _finding() -> Finding:
    return Finding(
        detector="metadata",
        code="EDITING_SOFTWARE",
        category="metadata",
        title="Logiciel d'édition détecté",
        description="Le fichier mentionne un logiciel d'édition.",
        risk_points=8,
        confidence=0.9,
        page=1,
    )


def _laboratory() -> LaboratoryReport:
    return LaboratoryReport(
        schema_version="1.0",
        checks=(
            LaboratoryCheck(
                code="SIGNATURE",
                title="Signature électronique",
                purpose="Contrôler une signature existante.",
                state="not_applicable",
                summary="Aucune signature électronique présente.",
            ),
        ),
    )


def _response(content: dict[str, Any]) -> _Response:
    return _Response({"choices": [{"message": {"content": json.dumps(content)}}]})


def test_synthesis_is_short_grounded_and_non_decisional(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return _response(
            {
                "document_summary": {
                    "text": "Ce document est classé comme une facture.",
                    "evidence_ids": ["E002"],
                },
                "review_summary": {
                    "text": "Un logiciel d'édition est mentionné et mérite une revue ciblée.",
                    "evidence_ids": ["E003"],
                },
                "highlights": [
                    {
                        "text": "Aucune signature électronique n'est présente.",
                        "evidence_ids": ["E004"],
                    }
                ],
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    finding = _finding()
    result = summarize_analysis(
        classification=DocumentClassification(
            family="facture_recu",
            reliability=0.92,
            language="fr",
            country="LU",
        ),
        extraction=None,
        verification=None,
        findings=(finding,),
        detectors=(DetectorResult(name="metadata", status="completed", findings=(finding,)),),
        laboratory=_laboratory(),
        config=AnalysisConfig(
            synthesis_enabled=True,
            synthesis_url="http://minimax.internal:8030",
        ),
        assessment_score=8,
        assessment_label="Faible",
    )

    assert result.document_summary.evidence_ids == ("E002",)
    assert result.review_summary.evidence_ids == ("E003",)
    assert result.highlights[0].evidence_ids == ("E004",)
    assert calls[0]["url"] == "http://minimax.internal:8030/v1/chat/completions"
    payload = calls[0]["json"]
    assert payload["max_tokens"] == 450
    assert payload["response_format"]["type"] == "json_schema"
    assert "n'emploie jamais les mots" in payload["messages"][1]["content"]
    assert "EDITING_SOFTWARE" not in payload["messages"][1]["content"]


@pytest.mark.parametrize(
    "review_summary",
    [
        "Le document est frauduleux.",
        "Le document paraît authentique.",
        "Aucune fraude n'est détectée.",
    ],
)
def test_synthesis_rejects_a_verdict(monkeypatch: Any, review_summary: str) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _response(
            {
                "document_summary": {
                    "text": "Le document contient un relevé d'opérations.",
                    "evidence_ids": ["E001"],
                },
                "review_summary": {"text": review_summary, "evidence_ids": ["E001"]},
                "highlights": [],
            }
        ),
    )

    with pytest.raises(SynthesisError, match="reliée aux preuves"):
        summarize_analysis(
            classification=None,
            extraction=None,
            verification=None,
            findings=(_finding(),),
            detectors=(),
            laboratory=None,
            config=AnalysisConfig(synthesis_enabled=True),
        )


def test_synthesis_rejects_unknown_evidence_references(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _response(
            {
                "document_summary": {"text": "Une facture est présentée.", "evidence_ids": ["X"]},
                "review_summary": {
                    "text": "Un indice matériel est affiché.",
                    "evidence_ids": ["X"],
                },
                "highlights": [],
            }
        ),
    )

    with pytest.raises(SynthesisError, match="reliée aux preuves"):
        summarize_analysis(
            classification=None,
            extraction=None,
            verification=None,
            findings=(_finding(),),
            detectors=(),
            laboratory=None,
            config=AnalysisConfig(synthesis_enabled=True),
        )
