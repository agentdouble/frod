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


def test_classification_is_grounded_and_reliability_is_capped(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return _chat_response(
            """{
              "categorie": "facture_recu",
              "confiance_modele": 0.97,
              "ambigu": false,
              "langue": "fr",
              "pays": "LU",
              "indices_categorie": ["FACTURE N° 2026-42", "Total TTC 125,00 EUR"],
              "indice_pays": "L-1234 Luxembourg"
            }"""
        )

    monkeypatch.setattr(requests, "post", fake_post)
    result = LLMClassifier().classify(
        "FACTURE N° 2026-42\nCabinet Exemple\nL-1234 Luxembourg\nTotal TTC 125,00 EUR"
    )

    assert result.family == "facture_recu"
    assert result.reliability == 0.82
    assert result.language == "fr"
    assert result.country == "LU"
    assert result.evidence == ("FACTURE N° 2026-42", "Total TTC 125,00 EUR")
    assert calls[0]["url"] == "http://127.0.0.1:8030/v1/chat/completions"
    assert calls[0]["json"]["messages"][0]["role"] == "system"
    assert calls[0]["json"]["response_format"]["type"] == "json_schema"


def test_ungrounded_family_and_country_are_discarded(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _chat_response(
            """{
              "categorie": "releve_bancaire",
              "confiance_modele": 0.99,
              "ambigu": false,
              "langue": "fr",
              "pays": "FR",
              "indices_categorie": ["Relevé bancaire complet"],
              "indice_pays": "République française"
            }"""
        ),
    )

    result = LLMClassifier().classify(
        "Synthèse mensuelle\nOpérations du compte\nSolde au 31 janvier"
    )

    assert result.family == "autre"
    assert result.reliability == 0.0
    assert result.country is None
    assert result.language == "fr"
    assert result.evidence == ()


def test_ambiguous_classification_never_looks_reliable(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _chat_response(
            """{
              "categorie": "devis",
              "confiance_modele": 0.95,
              "ambigu": true,
              "langue": "fr",
              "pays": null,
              "indices_categorie": ["Montant estimé : 800 EUR"],
              "indice_pays": null
            }"""
        ),
    )

    result = LLMClassifier().classify("Proposition de réparation\nMontant estimé : 800 EUR")

    assert result.family == "devis"
    assert result.reliability == 0.49


def test_classifier_retries_without_schema_for_older_vllm(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    responses = iter(
        [
            _chat_response("", status_code=400),
            _chat_response(
                """{
                  "categorie": "autre",
                  "confiance_modele": 0.4,
                  "ambigu": true,
                  "langue": null,
                  "pays": null,
                  "indices_categorie": [],
                  "indice_pays": null
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
    assert "response_format" in calls[0]["json"]
    assert "response_format" not in calls[1]["json"]


def test_short_ocr_text_abstains_without_calling_vllm(monkeypatch: Any) -> None:
    def unexpected_call(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("vLLM must not be called")

    monkeypatch.setattr(requests, "post", unexpected_call)
    result = LLMClassifier(AnalysisConfig()).classify("illisible")

    assert result.family == "autre"
    assert result.reliability == 0.1
