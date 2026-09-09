from fraude_detector.ocr_tables import (
    html_tables_to_markdown,
    infer_table_column_role,
    parse_ocr_table,
)


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


def test_html_tables_are_rendered_as_markdown_without_altering_surrounding_text() -> None:
    source = (
        "## Garanties\n\n"
        '<table border="1"><tr><td>De genaamde</td><td>Yvonne Muller</td></tr>'
        "<tr><td>Adres</td><td>Rue du Parc | 12</td></tr></table>\n\nFin"
    )

    rendered = html_tables_to_markdown(source)

    assert rendered.startswith("## Garanties")
    assert "<table" not in rendered
    assert "| Colonne 1 | Colonne 2 |" in rendered
    assert "| De genaamde | Yvonne Muller |" in rendered
    assert r"Rue du Parc \| 12" in rendered
    assert rendered.rstrip().endswith("Fin")


def test_html_colspan_keeps_following_cells_in_their_original_columns() -> None:
    table = parse_ocr_table(
        "<table><thead><tr><th>Date</th><th>Libellé</th><th>Montant</th></tr></thead>"
        "<tbody><tr><td colspan='2'>Report de solde</td><td>125,-</td></tr></tbody></table>"
    )

    assert table is not None
    assert table.rows == (("Report de solde", "", "125,-"),)


def test_html_rowspan_repeats_the_shared_value_without_shifting_columns() -> None:
    table = parse_ocr_table(
        "<table><thead><tr><th>Date</th><th>Libellé</th><th>Montant</th></tr></thead>"
        "<tbody><tr><td rowspan='2'>21/08/2026</td><td>Consultation</td><td>80</td></tr>"
        "<tr><td>Pharmacie</td><td>12,50</td></tr></tbody></table>"
    )

    assert table is not None
    assert table.rows == (
        ("21/08/2026", "Consultation", "80"),
        ("21/08/2026", "Pharmacie", "12,50"),
    )


def test_multilevel_headers_keep_rowspan_and_combine_header_labels() -> None:
    table = parse_ocr_table(
        "<table><thead><tr><th rowspan='2'>Date</th>"
        "<th colspan='2'>Opérations en EUR</th></tr>"
        "<tr><th>Libellé</th><th>Montant</th></tr></thead>"
        "<tbody><tr><td>21/08/2026</td><td>Consultation</td><td>80</td></tr>"
        "</tbody></table>"
    )

    assert table is not None
    assert table.headers == ("Date", "Opérations en EUR / Libellé", "Montant")
    assert table.rows == (("21/08/2026", "Consultation", "80"),)


def test_block_elements_inside_a_header_do_not_concatenate_labels() -> None:
    table = parse_ocr_table(
        "<table><tr><th><div>Date</div><div>opération</div></th></tr>"
        "<tr><td>21/08/2026</td></tr></table>"
    )

    assert table is not None
    assert table.headers == ("Date opération",)


def test_row_header_inside_table_body_is_kept_as_data() -> None:
    table = parse_ocr_table(
        "<table><thead><tr><th>Type</th><th>Montant</th></tr></thead>"
        "<tbody><tr><th scope='row'>Total</th><td>125,-</td></tr></tbody></table>"
    )

    assert table is not None
    assert table.headers == ("Type", "Montant")
    assert table.rows == (("Total", "125,-"),)


def test_explicit_multilingual_headers_have_deterministic_roles() -> None:
    assert infer_table_column_role("Transaction Date", "transactions") == "transaction_date"
    assert infer_table_column_role("Date de valeur", "transactions") == "value_date"
    assert infer_table_column_role("Libellé", "transactions") == "description"
    assert infer_table_column_role("Débit", "transactions") == "debit_amount"
    assert infer_table_column_role("Crédit", "transactions") == "credit_amount"
    assert infer_table_column_role("Solde", "transactions") == "balance"
    assert infer_table_column_role("Prix unitaire", "invoice_lines") == "unit_price"
    assert infer_table_column_role("Montant", "invoice_lines") == "line_total"


def test_ambiguous_header_stays_other() -> None:
    assert infer_table_column_role("Informations", "generic") == "other"
