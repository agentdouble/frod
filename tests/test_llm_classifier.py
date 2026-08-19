from __future__ import annotations

from typing import Any

import requests

from fraude_detector.config import AnalysisConfig
from fraude_detector.llm_classifier import LLMClassifier


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self.payload


def _chat_response(content: str, status_code: int = 200) -> _Response:
    return _Response(
        {"choices": [{"message": {"content": content}}]},
        status_code=status_code,
    )


def test_classification_accepts_paraphrased_reasons(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return _chat_response(
            """{
              "category": "facture_recu",
              "model_confidence": 0.97,
              "ambiguous": false,
              "language": "fr",
              "country": "LU",
              "category_evidence": [
                "Le document porte un numéro de facture",
                "Un total TTC est indiqué"
              ],
              "country_evidence": "L'adresse se situe au Luxembourg"
            }"""
        )

    monkeypatch.setattr(requests, "post", fake_post)
    result = LLMClassifier().classify(
        "FACTURE N° 2026-42\nCabinet Exemple\nL-1234 Luxembourg\nTotal TTC 125,00 EUR"
    )

    assert result.family == "facture_recu"
    assert result.reliability == 0.97
    assert result.language == "fr"
    assert result.country == "LU"
    assert result.evidence == (
        "Le document porte un numéro de facture",
        "Un total TTC est indiqué",
    )
    assert calls[0]["url"] == "http://127.0.0.1:8030/v1/chat/completions"
    assert calls[0]["json"]["messages"][0]["role"] == "system"
    assert calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert '"model_confidence": "confidence"' in calls[0]["json"]["messages"][1]["content"]
    assert (
        "Rédige category_evidence et country_evidence en français"
        in calls[0]["json"]["messages"][1]["content"]
    )
    schema_properties = calls[0]["json"]["response_format"]["json_schema"]["schema"]["properties"]
    assert set(schema_properties) == {
        "category",
        "model_confidence",
        "ambiguous",
        "language",
        "country",
        "category_evidence",
        "country_evidence",
    }


def test_classification_does_not_require_verbatim_reasons(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _chat_response(
            """{
              "category": "releve_bancaire",
              "model_confidence": 0.99,
              "ambiguous": false,
              "language": "fr",
              "country": "FR",
              "category_evidence": ["Relevé bancaire complet"],
              "country_evidence": "République française"
            }"""
        ),
    )

    result = LLMClassifier().classify(
        "Synthèse mensuelle\nOpérations du compte\nSolde au 31 janvier"
    )

    assert result.family == "releve_bancaire"
    assert result.reliability == 0.99
    assert result.country == "FR"
    assert result.language == "fr"
    assert result.evidence == ("Relevé bancaire complet",)


def test_ambiguous_classification_never_looks_reliable(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _chat_response(
            """{
              "category": "devis",
              "model_confidence": 0.95,
              "ambiguous": true,
              "language": "fr",
              "country": null,
              "category_evidence": ["Montant estimé : 800 EUR"],
              "country_evidence": null
            }"""
        ),
    )

    result = LLMClassifier().classify("Proposition de réparation\nMontant estimé : 800 EUR")

    assert result.family == "devis"
    assert result.reliability == 0.49


def test_classifier_retains_json_mode_for_older_vllm(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    responses = iter(
        [
            _chat_response("", status_code=400),
            _chat_response(
                """{
                  "category": "autre",
                  "model_confidence": 0.4,
                  "ambiguous": true,
                  "language": null,
                  "country": null,
                  "category_evidence": [],
                  "country_evidence": null
                }"""
            ),
        ]
    )

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(requests, "post", fake_post)
    result = LLMClassifier().classify("Contenu documentaire suffisamment long pour le test")

    assert result.family == "autre"
    assert len(calls) == 2
    assert calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert calls[1]["json"]["response_format"] == {"type": "json_object"}


def test_short_ocr_text_abstains_without_calling_vllm(monkeypatch: Any) -> None:
    def unexpected_call(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("vLLM must not be called")

    monkeypatch.setattr(requests, "post", unexpected_call)
    result = LLMClassifier(AnalysisConfig()).classify("illisible")

    assert result.family == "autre"
    assert result.reliability == 0.1
