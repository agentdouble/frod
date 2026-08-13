from __future__ import annotations

import json
from typing import Any

import requests

from fraude_detector.config import AnalysisConfig
from fraude_detector.llm_classifier import DOCUMENT_FAMILIES
from fraude_detector.llm_extractor import FAMILY_GUIDANCE, LLMDocumentExtractor
from fraude_detector.models import DocumentClassification


class _Response:
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self) -> dict[str, Any]:
        return self.payload


def _chat_response(content: dict[str, Any]) -> _Response:
    return _Response({"choices": [{"message": {"content": json.dumps(content)}}]})


def _region(content: str, label: str = "text") -> dict[str, Any]:
    return {"label": label, "content": content, "bbox_2d": [0, 0, 100, 100]}


def _empty_result() -> dict[str, list[Any]]:
    return {
        "facts": [],
        "additional_fields": [],
        "tables": [],
        "region_dispositions": [],
    }


def test_every_classified_family_has_dedicated_extraction_guidance() -> None:
    assert set(FAMILY_GUIDANCE) == set(DOCUMENT_FAMILIES)


def test_extractor_preserves_raw_values_tables_and_complete_region_coverage(
    monkeypatch: Any,
) -> None:
    payload = [
        [
            _region("Facture N° FAC-2026-0042"),
            _region("Total TTC : 1 234,50 EUR"),
            _region("Date | Désignation | Montant", "table"),
            _region("Merci pour votre confiance", "footer"),
        ]
    ]
    result = {
        "facts": [
            {
                "field_code": "invoice_number",
                "role": "invoice",
                "raw_label": "Facture N°",
                "raw_value": "FAC-2026-0042",
                "confidence": 0.98,
                "region_ids": ["p001-r000"],
            },
            {
                "field_code": "monetary_amount",
                "role": "total",
                "raw_label": "Total TTC",
                "raw_value": "1 234,50 EUR",
                "confidence": 0.95,
                "region_ids": ["p001-r001"],
            },
        ],
        "additional_fields": [],
        "tables": [
            {
                "title": "Prestations",
                "semantic_type": "invoice_lines",
                "headers": ["Date", "Désignation", "Montant"],
                "column_roles": ["transaction_date", "description", "line_total"],
                "rows": [["12/08/2026", "Consultation", "80,00 EUR"]],
                "row_roles": ["line_item"],
                "confidence": 0.91,
                "region_ids": ["p001-r002"],
            }
        ],
        "region_dispositions": [
            {
                "region_id": "p001-r003",
                "disposition": "boilerplate",
                "reason": "Formule de politesse",
            }
        ],
    }
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return _chat_response(result)

    monkeypatch.setattr(requests, "post", fake_post)
    classification = DocumentClassification(
        family="facture_recu",
        reliability=0.9,
        language="fr",
        country="LU",
    )
    extraction = LLMDocumentExtractor(
        AnalysisConfig(extraction_enabled=True, extraction_url="http://llm.internal:8030")
    ).extract(payload, classification)

    assert len(calls) == 1
    assert calls[0]["url"] == "http://llm.internal:8030/v1/chat/completions"
    assert calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert "famille: facture_recu" in calls[0]["json"]["messages"][1]["content"]
    assert extraction.family == "facture_recu"
    assert extraction.facts[1].raw_value == "1 234,50 EUR"
    assert extraction.facts[1].normalized_value == "1234.50 EUR"
    assert extraction.tables[0].rows == (("12/08/2026", "Consultation", "80,00 EUR"),)
    assert extraction.tables[0].column_roles == (
        "transaction_date",
        "description",
        "line_total",
    )
    assert extraction.tables[0].row_roles == ("line_item",)
    assert extraction.coverage.ratio == 1
    assert extraction.coverage.mapped_regions == 2
    assert extraction.coverage.table_regions == 1
    assert extraction.coverage.boilerplate_regions == 1
    assert extraction.passes == 1


def test_extractor_retries_only_regions_not_accounted_for(monkeypatch: Any) -> None:
    payload = [[_region("REFERENCE_ALPHA"), _region("CHAMP_BETA")]]
    first = _empty_result()
    first["facts"] = [
        {
            "field_code": "document_number",
            "role": "document",
            "raw_label": "Référence",
            "raw_value": "REFERENCE_ALPHA",
            "confidence": 0.9,
            "region_ids": ["p001-r000"],
        }
    ]
    second = _empty_result()
    second["additional_fields"] = [
        {
            "raw_label": "Champ libre",
            "raw_value": "CHAMP_BETA",
            "semantic_hint": "information spécifique",
            "confidence": 0.8,
            "region_ids": ["p001-r001"],
        }
    ]
    responses = iter((_chat_response(first), _chat_response(second)))
    prompts: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return next(responses)

    monkeypatch.setattr(requests, "post", fake_post)
    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        payload,
        None,
    )

    assert len(prompts) == 2
    assert "REFERENCE_ALPHA" in prompts[0]
    assert "CHAMP_BETA" in prompts[0]
    assert "REFERENCE_ALPHA" not in prompts[1]
    assert "CHAMP_BETA" in prompts[1]
    assert "passe de couverture" in prompts[1]
    assert extraction.passes == 2
    assert extraction.coverage.ratio == 1
    assert extraction.additional_fields[0].raw_value == "CHAMP_BETA"


def test_uncertain_classification_keeps_generic_extraction_as_priority(monkeypatch: Any) -> None:
    result = _empty_result()
    result["region_dispositions"] = [
        {
            "region_id": "p001-r000",
            "disposition": "unstructured",
            "reason": "Contenu non structuré",
        }
    ]
    prompts: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return _chat_response(result)

    monkeypatch.setattr(requests, "post", fake_post)
    classification = DocumentClassification(
        family="document_medical",
        reliability=0.42,
        language="fr",
        country=None,
    )
    LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("Document difficile à classer")]],
        classification,
    )

    assert "fiabilité du classement: 42%" in prompts[0]
    assert "La famille proposée est incertaine" in prompts[0]
    assert "N'impose aucune structure métier" in prompts[0]


def test_extractor_falls_back_when_vllm_rejects_json_schema(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        calls.append(kwargs["json"])
        if len(calls) == 1:
            return _Response({}, status_code=400)
        result = _empty_result()
        result["region_dispositions"] = [
            {
                "region_id": "p001-r000",
                "disposition": "unstructured",
                "reason": "Texte libre",
            }
        ]
        return _chat_response(result)

    monkeypatch.setattr(requests, "post", fake_post)
    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("Texte libre")]],
        None,
    )

    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
    assert extraction.coverage.unstructured_regions == 1
    assert extraction.coverage.ratio == 1


def test_unreferenced_model_value_is_retained_with_low_confidence(monkeypatch: Any) -> None:
    result = _empty_result()
    result["facts"] = [
        {
            "field_code": "iban",
            "role": "beneficiary",
            "raw_label": "IBAN",
            "raw_value": "LU28 0019 4006 4475 0000",
            "confidence": 0.99,
            "region_ids": ["unknown-region"],
        }
    ]
    result["region_dispositions"] = [
        {
            "region_id": "p001-r000",
            "disposition": "unreadable",
            "reason": "Aucune association",
        }
    ]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("IBAN partiellement lisible")]],
        None,
    )

    assert extraction.facts[0].raw_value == "LU28 0019 4006 4475 0000"
    assert extraction.facts[0].normalized_value == "LU280019400644750000"
    assert extraction.facts[0].region_ids == ()
    assert extraction.facts[0].confidence == 0.3
    assert extraction.coverage.unreadable_regions == 1


def test_bank_prompt_requires_summary_rows_to_be_separated_from_transactions(
    monkeypatch: Any,
) -> None:
    table = (
        "<table><tr><td>Total ancien solde</td><td></td></tr>"
        "<tr><td>OPERATIONS CARTE VISA NUMERO 4697</td><td>-2500</td></tr>"
        "<tr><td>Voyages Arosa</td><td>-2500</td></tr>"
        "<tr><td>TOTAL CARTE</td><td>-2500</td></tr>"
        "<tr><td>TOTAL NOUVEAU SOLDE</td><td>-2500</td></tr></table>"
    )
    result = _empty_result()
    result["tables"] = [
        {
            "title": "Opérations carte VISA",
            "semantic_type": "transactions",
            "headers": ["Libellé", "Montant"],
            "column_roles": ["description", "amount"],
            "rows": [
                ["Total ancien solde", ""],
                ["OPERATIONS CARTE VISA NUMERO 4697", "-2500"],
                ["Voyages Arosa", "-2500"],
                ["TOTAL CARTE", "-2500"],
                ["TOTAL NOUVEAU SOLDE", "-2500"],
            ],
            "row_roles": [
                "opening_balance",
                "section_header",
                "transaction",
                "total",
                "closing_balance",
            ],
            "confidence": 0.92,
            "region_ids": ["p001-r000"],
        }
    ]
    prompts: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return _chat_response(result)

    monkeypatch.setattr(requests, "post", fake_post)
    classification = DocumentClassification(
        family="releve_bancaire",
        reliability=0.9,
        language="fr",
        country="CH",
    )
    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region(table, "table")]],
        classification,
    )

    assert extraction.tables[0].row_roles == (
        "opening_balance",
        "section_header",
        "transaction",
        "total",
        "closing_balance",
    )
    assert extraction.tables[0].row_roles.count("transaction") == 1
    assert "Un ancien solde" in prompts[0]
    assert "ne sont jamais\n   des transactions" in prompts[0]
