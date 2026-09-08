from __future__ import annotations

from typing import Any

import pytest
import requests

from fraude_detector.config import AnalysisConfig
from fraude_detector.llm_synthesizer import SynthesisError, summarize_analysis
from fraude_detector.models import (
    DetectorResult,
    DocumentClassification,
    ExtractionReview,
    ExtractionVerification,
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
        risk_points=12,
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


def _response(content: str, *, finish_reason: str = "stop") -> _Response:
    return _Response(
        {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    )


def test_synthesis_is_short_grounded_and_non_decisional(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return _response(
            "Une revue manuelle est nécessaire car le fichier mentionne un logiciel d'édition. "
            "L'analyste doit confronter cet historique technique au document visible et à sa "
            "source afin de déterminer si cette intervention était attendue."
        )

    monkeypatch.setattr(requests, "post", fake_post)
    finding = _finding()
    weak_finding = Finding(
        detector="ai_generated",
        code="AI_WEAK",
        category="synthetic_media",
        title="Faible ressemblance avec une image générée",
        description="Le signal statistique reste faible.",
        risk_points=0,
        confidence=0.55,
    )
    result = summarize_analysis(
        classification=DocumentClassification(
            family="facture_recu",
            reliability=0.92,
            language="fr",
            country="LU",
        ),
        extraction=None,
        verification=None,
        findings=(finding, weak_finding),
        detectors=(DetectorResult(name="metadata", status="completed", findings=(finding,)),),
        laboratory=_laboratory(),
        config=AnalysisConfig(
            synthesis_enabled=True,
            synthesis_url="http://minimax.internal:8030",
        ),
        assessment_score=30,
        assessment_label="Revue manuelle nécessaire",
    )

    assert result.text.startswith("Une revue manuelle est nécessaire")
    assert calls[0]["url"] == "http://minimax.internal:8030/v1/chat/completions"
    payload = calls[0]["json"]
    assert payload["max_tokens"] == 32_768
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["stream"] is True
    assert "response_format" not in payload
    assert "ne décides jamais" in payload["messages"][0]["content"]
    assert "sans JSON" in payload["messages"][0]["content"]
    assert "ne cite jamais le score numérique" in payload["messages"][1]["content"]
    assert "indices retenus dans le score" in payload["messages"][1]["content"]
    assert "sans dépasser 180 mots" in payload["messages"][1]["content"]
    assert "Ne commence pas systématiquement" in payload["messages"][1]["content"]
    assert "Score global calculé" not in payload["messages"][1]["content"]
    assert "pays=LU" not in payload["messages"][1]["content"]
    assert "Aucune signature électronique" not in payload["messages"][1]["content"]
    assert "Faible ressemblance avec une image générée" not in payload["messages"][1]["content"]
    assert "EDITING_SOFTWARE" not in payload["messages"][1]["content"]


def test_synthesis_excludes_extraction_quality_and_masks_exact_signal_values(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        calls.append(kwargs["json"])
        return _response(
            "Revue manuelle nécessaire : un identifiant financier présente une incohérence "
            "de format. Il faut le comparer au document d'origine."
        )

    monkeypatch.setattr(requests, "post", fake_post)
    finding = Finding(
        detector="financial_identifiers",
        code="INVALID_IBAN",
        category="content_consistency",
        title="Identifiant financier incohérent",
        description="L'IBAN LU280019400644750000 associé au montant 2.500,00 EUR est invalide.",
        risk_points=12,
        confidence=0.9,
    )
    verification = ExtractionVerification(
        schema_version="0.4-experimental",
        status="attention",
        expected_targets=1,
        reviews=(
            ExtractionReview(
                target_id="fact-0001",
                target_type="fact",
                verdict="ambiguous",
                explanation="Le montant OCR a peut-être été rattaché à la mauvaise colonne.",
                source_region_ids=("p001-r001",),
            ),
        ),
        omissions=(),
    )

    summarize_analysis(
        classification=None,
        extraction=None,
        verification=verification,
        findings=(finding,),
        detectors=(),
        laboratory=None,
        config=AnalysisConfig(synthesis_enabled=True),
        assessment_score=30,
    )

    prompt = calls[0]["messages"][1]["content"]
    assert "LU280019400644750000" not in prompt
    assert "2.500,00" not in prompt
    assert "mauvaise colonne" not in prompt
    assert "Certaines informations n'ont pas pu être extraites" in prompt


def test_synthesis_keeps_the_model_note_without_imposing_sections(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _response(
            "Ce document contient un relevé d'opérations. "
            "Un logiciel d'édition est mentionné et demande une revue ciblée."
        ),
    )

    result = summarize_analysis(
        classification=None,
        extraction=None,
        verification=None,
        findings=(_finding(),),
        detectors=(),
        laboratory=None,
        config=AnalysisConfig(synthesis_enabled=True),
    )

    assert result.text == (
        "Ce document contient un relevé d'opérations. "
        "Un logiciel d'édition est mentionné et demande une revue ciblée."
    )


@pytest.mark.parametrize(
    "review_summary",
    [
        "Le document est frauduleux.",
        "Le document paraît authentique.",
        "Aucune fraude n'est détectée.",
        "Le document ne nécessite pas de revue.",
    ],
)
def test_synthesis_rejects_a_verdict_or_review_dismissal(
    monkeypatch: Any,
    review_summary: str,
) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _response(
            "DOCUMENT: Le document contient un relevé d'opérations.\n"
            f"REVUE: {review_summary}\n"
            "POINTS: Aucun"
        ),
    )

    with pytest.raises(SynthesisError, match="verdict|revue"):
        summarize_analysis(
            classification=None,
            extraction=None,
            verification=None,
            findings=(_finding(),),
            detectors=(),
            laboratory=None,
            config=AnalysisConfig(synthesis_enabled=True),
        )


def test_high_risk_context_explicitly_requests_reinforced_vigilance(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        calls.append(kwargs["json"])
        return _response(
            "Vigilance renforcée : une revue manuelle prioritaire doit contrôler le logiciel "
            "d'édition mentionné dans le fichier."
        )

    monkeypatch.setattr(requests, "post", fake_post)
    summarize_analysis(
        classification=None,
        extraction=None,
        verification=None,
        findings=(_finding(),),
        detectors=(),
        laboratory=None,
        config=AnalysisConfig(synthesis_enabled=True),
        assessment_score=75,
    )

    assert "Vigilance renforcée" in calls[0]["messages"][1]["content"]


def test_low_risk_context_never_says_that_review_is_unnecessary(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        calls.append(kwargs["json"])
        return _response(
            "Aucun indice prioritaire n'a été relevé par les contrôles disponibles. "
            "Cette absence ne valide pas le document."
        )

    monkeypatch.setattr(requests, "post", fake_post)
    result = summarize_analysis(
        classification=None,
        extraction=None,
        verification=None,
        findings=(),
        detectors=(),
        laboratory=None,
        config=AnalysisConfig(synthesis_enabled=True),
        assessment_score=0,
    )

    prompt = calls[0]["messages"][1]["content"]
    assert "Pas de revue automatique" not in prompt
    assert "Aucun indice particulier" in prompt
    assert "ne valide pas le document" in result.text
