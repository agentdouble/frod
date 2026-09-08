from __future__ import annotations

import json
from typing import Any

import requests

from fraude_detector.config import AnalysisConfig
from fraude_detector.llm_classifier import DOCUMENT_FAMILIES
from fraude_detector.llm_extractor import (
    FAMILY_GUIDANCE,
    LLMDocumentExtractor,
    normalize_extracted_value,
)
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
    return {
        "index": 7,
        "label": label,
        "native_label": "paragraph",
        "content": content,
        "bbox_2d": [10, 20, 300, 120],
    }


def _empty_result() -> dict[str, Any]:
    return {
        "facts": [],
        "additional_fields": [],
        "tables": [],
        "region_dispositions": {
            "boilerplate": [],
            "unstructured": [],
            "unreadable": [],
        },
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
            _region(
                "<table><thead><tr><th>Date</th><th>Désignation</th><th>Montant</th></tr>"
                "</thead><tbody><tr><td>12/08/2026</td><td>Consultation</td>"
                "<td>80,00 EUR</td></tr></tbody></table>",
                "table",
            ),
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
                "region_ids": ["p001-r000"],
            },
            {
                "field_code": "monetary_amount",
                "role": "total",
                "raw_label": "Total TTC",
                "raw_value": "1 234,50 EUR",
                "region_ids": ["p001-r001"],
            },
        ],
        "additional_fields": [],
        "tables": [
            {
                "title": "Prestations",
                "semantic_type": "invoice_lines",
                "column_roles": ["transaction_date", "description", "line_total"],
                "default_row_role": "line_item",
                "row_role_overrides": [],
                "region_ids": ["p001-r002"],
            }
        ],
        "region_dispositions": {
            "boilerplate": ["p001-r003"],
            "unstructured": [],
            "unreadable": [],
        },
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
    schema_properties = calls[0]["json"]["response_format"]["json_schema"]["schema"]["properties"]
    table_properties = schema_properties["tables"]["items"]["properties"]
    assert "rows" not in table_properties
    assert "headers" not in table_properties
    assert "default_row_role" in table_properties
    assert "confidence" not in schema_properties["facts"]["items"]["properties"]
    assert "confidence" not in schema_properties["additional_fields"]["items"]["properties"]
    assert "confidence" not in table_properties
    assert schema_properties["region_dispositions"]["type"] == "object"
    assert "famille: facture_recu" in calls[0]["json"]["messages"][1]["content"]
    assert "Toutes les clés JSON" in calls[0]["json"]["messages"][1]["content"]
    assert "valeurs d'énumération doivent être en" in calls[0]["json"]["messages"][1]["content"]
    assert 'order="7"' in calls[0]["json"]["messages"][1]["content"]
    assert 'native_label="paragraph"' in calls[0]["json"]["messages"][1]["content"]
    assert 'bbox_2d="10,20,300,120"' in calls[0]["json"]["messages"][1]["content"]
    assert extraction.family == "facture_recu"
    assert extraction.facts[1].raw_value == "1 234,50 EUR"
    assert extraction.facts[1].normalized_value == 1234.5
    assert extraction.facts[1].normalized_currency == "EUR"
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
    assert extraction.schema_version == "0.4-experimental"
    assert extraction.prompt_version.startswith("extraction-")
    assert extraction.vocabulary_version.startswith("document-fields-")


def test_extractor_retries_only_regions_not_accounted_for(monkeypatch: Any) -> None:
    payload = [[_region("REFERENCE_ALPHA"), _region("CHAMP_BETA")]]
    first = _empty_result()
    first["facts"] = [
        {
            "field_code": "document_number",
            "role": "document",
            "raw_label": "Référence",
            "raw_value": "REFERENCE_ALPHA",
            "region_ids": ["p001-r000"],
        }
    ]
    second = _empty_result()
    second["additional_fields"] = [
        {
            "raw_label": "Champ libre",
            "raw_value": "CHAMP_BETA",
            "semantic_hint": "other_material",
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
    extraction = LLMDocumentExtractor(
        AnalysisConfig(extraction_enabled=True, extraction_coverage_retry=True)
    ).extract(
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


def test_obvious_decorative_message_is_not_kept_as_additional_information(
    monkeypatch: Any,
) -> None:
    result = _empty_result()
    result["additional_fields"] = [
        {
            "raw_label": "Message",
            "raw_value": "Please think about the environment before printing this email.",
            "semantic_hint": "other_material",
            "region_ids": ["p001-r000"],
        }
    ]
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> _Response:
        nonlocal calls
        del url, kwargs
        calls += 1
        return _chat_response(result)

    monkeypatch.setattr(requests, "post", fake_post)
    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("Please think about the environment before printing this email.")]],
        None,
    )

    assert calls == 0
    assert extraction.additional_fields == ()
    assert extraction.coverage.boilerplate_regions == 1
    assert extraction.coverage.ratio == 1


def test_uncertain_classification_keeps_generic_extraction_as_priority(monkeypatch: Any) -> None:
    result = _empty_result()
    result["region_dispositions"]["unstructured"] = ["p001-r000"]
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


def test_extractor_retains_json_mode_when_vllm_rejects_json_schema(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        calls.append(kwargs["json"])
        if len(calls) == 1:
            return _Response({}, status_code=400)
        result = _empty_result()
        result["region_dispositions"]["unstructured"] = ["p001-r000"]
        return _chat_response(result)

    monkeypatch.setattr(requests, "post", fake_post)
    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("Texte libre")]],
        None,
    )

    assert len(calls) == 2
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert extraction.coverage.unstructured_regions == 1
    assert extraction.coverage.ratio == 1


def test_extractor_retries_a_non_json_generation_with_constrained_output(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []
    valid = _empty_result()
    valid["region_dispositions"]["unstructured"] = ["p001-r000"]
    responses = iter(
        (
            _Response({"choices": [{"message": {"content": "analysis..."}}]}),
            _chat_response(valid),
        )
    )

    def fake_post(url: str, **kwargs: Any) -> _Response:
        del url
        calls.append(kwargs["json"])
        return next(responses)

    monkeypatch.setattr(requests, "post", fake_post)
    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("Free text")]],
        None,
    )

    assert extraction.coverage.ratio == 1
    assert len(calls) == 2
    assert all(call["response_format"]["type"] == "json_schema" for call in calls)
    assert calls[1]["max_tokens"] == 32_768


def test_unreferenced_model_value_is_retained_without_inventing_source_quality(
    monkeypatch: Any,
) -> None:
    result = _empty_result()
    result["facts"] = [
        {
            "field_code": "iban",
            "role": "beneficiary",
            "raw_label": "IBAN",
            "raw_value": "LU28 0019 4006 4475 0000",
            "region_ids": ["unknown-region"],
        }
    ]
    result["region_dispositions"]["unreadable"] = ["p001-r000"]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("IBAN partiellement lisible")]],
        None,
    )

    assert extraction.facts[0].raw_value == "LU28 0019 4006 4475 0000"
    assert extraction.facts[0].normalized_value == "LU280019400644750000"
    assert extraction.facts[0].region_ids == ()
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
            "column_roles": ["description", "amount"],
            "default_row_role": "transaction",
            "row_role_overrides": [
                {"row_index": 0, "role": "opening_balance"},
                {"row_index": 1, "role": "section_header"},
                {"row_index": 3, "role": "total"},
                {"row_index": 4, "role": "closing_balance"},
            ],
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
    assert extraction.tables[0].headers == ("", "")
    assert extraction.tables[0].column_roles == ("description", "amount")
    assert "Un ancien solde" in prompts[0]
    assert "ne sont jamais\n   des transactions" in prompts[0]


def test_explicit_headers_replace_only_other_column_roles(monkeypatch: Any) -> None:
    table = (
        "<table><thead><tr><th>Transaction Date</th><th>Particulars</th>"
        "<th>Debit</th><th>Credit</th><th>Balance</th></tr></thead>"
        "<tbody><tr><td>12/08/2026</td><td>Consultation</td><td>80</td>"
        "<td></td><td>120</td></tr></tbody></table>"
    )
    result = _empty_result()
    result["tables"] = [
        {
            "title": "Operations",
            "semantic_type": "transactions",
            "column_roles": ["other", "other", "other", "other", "other"],
            "default_row_role": "transaction",
            "row_role_overrides": [],
            "region_ids": ["p001-r000"],
        }
    ]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region(table, "table")]],
        None,
    )

    assert extraction.tables[0].column_roles == (
        "transaction_date",
        "description",
        "debit_amount",
        "credit_amount",
        "balance",
    )


def test_international_amount_and_date_formats_are_normalized_deterministically() -> None:
    assert normalize_extracted_value("monetary_amount", "5.01", None, None) == (
        5.01,
        "normalized",
    )
    assert normalize_extracted_value("monetary_amount", "1,234.56 USD", None, None) == (
        1234.56,
        "normalized",
    )
    assert normalize_extracted_value("monetary_amount", "1.234,56 EUR", None, None) == (
        1234.56,
        "normalized",
    )
    assert normalize_extracted_value("monetary_amount", "125,-", None, None) == (
        125,
        "normalized",
    )
    assert normalize_extracted_value("monetary_amount", "125,–", None, None) == (
        125,
        "normalized",
    )
    assert normalize_extracted_value("monetary_amount", "2,356,45", "fr", "LU") == (
        2356.45,
        "normalized",
    )
    assert normalize_extracted_value("date", "21,01,2025", None, None) == (
        "2025-01-21",
        "normalized",
    )


def test_approximate_amount_ocr_noise_does_not_select_the_first_digit() -> None:
    assert normalize_extracted_value("monetary_amount", "+/-500", None, None) == (
        500,
        "normalized",
    )
    assert normalize_extracted_value("monetary_amount", "+1-500", None, None) == (
        500,
        "normalized",
    )
    assert normalize_extracted_value("monetary_amount", "400-500", None, None) == (
        None,
        "ambiguous",
    )


def test_short_year_dates_use_locale_only_when_it_resolves_the_order() -> None:
    assert normalize_extracted_value("date", "21/01/25", "fr", "LU") == (
        "2025-01-21",
        "normalized",
    )
    assert normalize_extracted_value("date", "1/21/25", "en", "US") == (
        "2025-01-21",
        "normalized",
    )
    assert normalize_extracted_value("date", "31/12/99", "fr", "LU") == (
        "1999-12-31",
        "normalized",
    )
    assert normalize_extracted_value("date", "01/02/25", None, None) == (
        None,
        "ambiguous",
    )


def test_iban_label_and_common_ocr_misread_are_removed_from_normalized_value() -> None:
    expected = ("LU280019400644750000", "normalized")

    assert normalize_extracted_value(
        "iban", "IBANLU28 0019 4006 4475 0000", None, None
    ) == expected
    assert normalize_extracted_value(
        "iban", "IBAU : LU28 0019 4006 4475 0000", None, None
    ) == expected
    assert normalize_extracted_value(
        "iban", "IBAN LU28 0019 4006 4475 0000 with KBC EUR", None, None
    ) == expected


def test_currency_is_inherited_from_linked_table_context(monkeypatch: Any) -> None:
    table = (
        "<table><thead><tr><th>Opérations en EUR</th></tr></thead>"
        "<tbody><tr><td>5.01</td></tr></tbody></table>"
    )
    result = _empty_result()
    result["facts"] = [
        {
            "field_code": "monetary_amount",
            "role": "transaction",
            "raw_label": "Montant",
            "raw_value": "5.01",
            "region_ids": ["p001-r000"],
        }
    ]
    result["tables"] = [
        {
            "title": "Opérations",
            "semantic_type": "transactions",
            "column_roles": ["amount"],
            "default_row_role": "transaction",
            "row_role_overrides": [],
            "region_ids": ["p001-r000"],
        }
    ]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region(table, "table")]],
        None,
    )

    assert extraction.facts[0].normalized_value == 5.01
    assert extraction.facts[0].normalized_currency == "EUR"


def test_identical_additional_label_and_value_are_discarded(monkeypatch: Any) -> None:
    result = _empty_result()
    result["additional_fields"] = [
        {
            "raw_label": "Information générale",
            "raw_value": "Information générale",
            "semantic_hint": "other_material",
            "region_ids": ["p001-r000"],
        }
    ]
    result["region_dispositions"]["unstructured"] = ["p001-r000"]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("Information générale")]],
        None,
    )

    assert extraction.additional_fields == ()


def test_obvious_ocr_replacement_garbage_is_not_extracted(monkeypatch: Any) -> None:
    result = _empty_result()
    result["additional_fields"] = [
        {
            "raw_label": "Référence",
            "raw_value": "锟斤拷",
            "semantic_hint": "other_material",
            "region_ids": ["p001-r000"],
        }
    ]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("Référence : 锟斤拷")]],
        None,
    )

    assert extraction.additional_fields == ()
    assert extraction.coverage.uncovered_region_ids == ("p001-r000",)


def test_repeated_person_name_uses_the_clearer_near_identical_spelling(monkeypatch: Any) -> None:
    result = _empty_result()
    result["facts"] = [
        {
            "field_code": "person_name",
            "role": "patient",
            "raw_label": "Patient",
            "raw_value": "Vonne Muller",
            "region_ids": ["p001-r000"],
        },
        {
            "field_code": "person_name",
            "role": "patient",
            "raw_label": "Patient",
            "raw_value": "Yvonne Muller",
            "region_ids": ["p001-r001"],
        },
    ]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("Patient : Vonne Muller"), _region("Patient : Yvonne Muller")]],
        None,
    )

    assert [fact.raw_value for fact in extraction.facts] == ["Vonne Muller", "Yvonne Muller"]
    assert [fact.corrected_value for fact in extraction.facts] == ["Yvonne Muller", None]
    assert len({fact.normalized_value for fact in extraction.facts}) == 1


def test_repeated_person_name_does_not_cross_roles_or_labels(monkeypatch: Any) -> None:
    result = _empty_result()
    result["facts"] = [
        {
            "field_code": "person_name",
            "role": "patient",
            "raw_label": "Patient",
            "raw_value": "Vonne Muller",
            "region_ids": ["p001-r000"],
        },
        {
            "field_code": "person_name",
            "role": "practitioner",
            "raw_label": "Médecin",
            "raw_value": "Yvonne Muller",
            "region_ids": ["p001-r001"],
        },
    ]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _chat_response(result))

    extraction = LLMDocumentExtractor(AnalysisConfig(extraction_enabled=True)).extract(
        [[_region("Patient : Vonne Muller"), _region("Médecin : Yvonne Muller")]],
        None,
    )

    assert all(fact.corrected_value is None for fact in extraction.facts)
