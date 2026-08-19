from fraude_detector.ocr_tables import parse_ocr_table


def test_html_table_is_reconstructed_without_language_model_output() -> None:
    table = parse_ocr_table(
        "<table><thead><tr><th>Date</th><th>Libellé</th><th>Montant</th></tr></thead>"
        "<tbody><tr><td>12/08/2026</td><td>Consultation</td><td>80,00 EUR</td></tr>"
        "<tr><td>13/08/2026</td><td>Pharmacie</td><td>12,50 EUR</td></tr></tbody></table>"
    )

    assert table is not None
    assert table.headers == ("Date", "Libellé", "Montant")
    assert table.rows == (
        ("12/08/2026", "Consultation", "80,00 EUR"),
        ("13/08/2026", "Pharmacie", "12,50 EUR"),
    )


def test_markdown_table_is_reconstructed() -> None:
    table = parse_ocr_table(
        "| Service | Total |\n|:---|---:|\n| Consultation | 80 EUR |\n| Radio | 40 EUR |"
    )

    assert table is not None
    assert table.headers == ("Service", "Total")
    assert table.rows == (("Consultation", "80 EUR"), ("Radio", "40 EUR"))


def test_plain_text_is_not_mistaken_for_a_table() -> None:
    assert parse_ocr_table("Date | Libellé | Montant") is None
