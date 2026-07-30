from __future__ import annotations

from datetime import date

from fraude_detector.laboratory.ocr import analyze_ocr_laboratory


def test_bank_statement_checks_identifiers_summary_and_ledger() -> None:
    payload = [
        [
            _region(
                "text",
                "POST GOLD VISA CREDIT CARD ACCOUNT 4111-1111-1111-1111",
            ),
            _region(
                "text",
                "CREDIT CARD NO : 4111-1111-1111-1121",
            ),
            _region("text", "SWIFT BIL2000488"),
            _region("text", "CURRENCY : EUR"),
            _region(
                "table",
                """
                <table>
                  <tr>
                    <th>Opening Balance</th><th>Total Debit</th>
                    <th>Total Credit</th><th>Pending Balance</th>
                  </tr>
                  <tr><td>5.01</td><td>100.00</td><td>201.00</td><td>106.01</td></tr>
                </table>
                """,
            ),
            _region(
                "table",
                """
                <table>
                  <tr>
                    <th>Transaction Date</th><th>Debit</th>
                    <th>Credit</th><th>Balance</th>
                  </tr>
                  <tr><td></td><td></td><td></td><td>100.00</td></tr>
                  <tr><td>15-Feb-2026</td><td></td><td></td><td>100.00</td></tr>
                  <tr><td>18-Feb-2026</td><td>20.00 (USD)</td><td></td><td>81.00</td></tr>
                  <tr><td>16-Feb-2026</td><td>1.00</td><td></td><td>81.00</td></tr>
                </table>
                """,
            ),
        ]
    ]

    checks = {
        check.code: check
        for check in analyze_ocr_laboratory(
            payload,
            reference_date=date(2026, 7, 30),
        )
    }

    identifiers = checks["ocr_identifiers"]
    assert identifiers.state == "attention"
    assert {item.code for item in identifiers.observations} == {
        "OCR_CARD_LUHN_VALID",
        "OCR_CARD_LUHN_INVALID",
        "OCR_CARD_VALUES_CONFLICT",
        "OCR_BIC_INVALID",
    }
    conflict = identifiers.observations[0]
    assert conflict.code == "OCR_CARD_VALUES_CONFLICT"
    assert conflict.evidence["differing_characters"] == 1
    bic = next(item for item in identifiers.observations if item.code == "OCR_BIC_INVALID")
    assert set(bic.evidence["invalid_reasons"]) == {
        "invalid_length",
        "invalid_institution_code",
        "invalid_country_code",
    }

    financial = checks["ocr_financial_consistency"]
    assert financial.state == "attention"
    assert [item.code for item in financial.observations] == [
        "OCR_STATEMENT_SUMMARY_VALID",
        "OCR_LEDGER_MISMATCH",
    ]
    ledger = financial.observations[1]
    assert ledger.evidence["balance_mismatch_rows"] == [4]
    assert ledger.evidence["chronology_inversion_rows"] == [4]
    assert ledger.evidence["foreign_currency_rows_skipped"] == 1


def test_sparse_ocr_is_indeterminate_instead_of_suspicious() -> None:
    checks = analyze_ocr_laboratory(
        [[_region("text", "Short unreadable fragment")]],
        reference_date=date(2026, 7, 30),
    )

    assert checks[0].code == "ocr_quality"
    assert checks[0].state == "indeterminate"
    assert all(check.state != "attention" for check in checks)


def test_invoice_total_mismatch_is_explainable() -> None:
    payload = [
        [
            _region("text", "Invoice 2026-001"),
            _region(
                "table",
                """
                <table>
                  <tr><td>Subtotal</td><td>100.00 EUR</td></tr>
                  <tr><td>VAT</td><td>17.00 EUR</td></tr>
                  <tr><td>Grand total</td><td>120.00 EUR</td></tr>
                </table>
                """,
            ),
        ]
    ]

    checks = analyze_ocr_laboratory(payload, reference_date=date(2026, 7, 30))
    financial = next(check for check in checks if check.code == "ocr_financial_consistency")

    assert financial.state == "attention"
    assert financial.observations[0].code == "OCR_INVOICE_TOTAL_MISMATCH"
    assert financial.observations[0].evidence["expected"] == "117.00"
    assert financial.observations[0].evidence["observed"] == "120.00"


def test_future_and_invalid_dates_are_reported_without_semantic_claim() -> None:
    payload = [
        [
            _region("text", "Document dates"),
            _region("text", "31/02/2026"),
            _region("text", "2030-01-05"),
        ]
    ]

    checks = analyze_ocr_laboratory(payload, reference_date=date(2026, 7, 30))
    dates = next(check for check in checks if check.code == "ocr_dates")

    assert dates.state == "attention"
    assert {item.code for item in dates.observations} >= {
        "OCR_DATE_INVALID",
        "OCR_DATE_IN_FUTURE",
    }


def test_international_iban_and_bic_formats_are_supported() -> None:
    payload = [
        [
            _region("text", "IBAN LU28 0019 4006 4475 0000"),
            _region("text", "IBAN GB82 WEST 1234 5698 7654 32"),
            _region("text", "IBAN SA03 8000 0000 6080 1016 7519"),
            _region("text", "BIC CCPLLULL"),
            _region("text", "SWIFT PBNKDEFFXXX"),
        ]
    ]

    checks = analyze_ocr_laboratory(payload, reference_date=date(2026, 7, 30))
    identifiers = next(check for check in checks if check.code == "ocr_identifiers")

    assert identifiers.state == "clear"
    assert len(identifiers.observations) == 5
    assert {item.code for item in identifiers.observations} == {
        "OCR_IBAN_VALID",
        "OCR_BIC_VALID",
    }


def test_iban_checks_country_specific_structure_not_only_mod_97() -> None:
    payload = [
        [
            _region("text", "IBAN LU28 0019 4006 4475 0000"),
            # Mod-97 is valid, but Luxembourg IBANs must contain 20 characters.
            _region("text", "IBAN LU58 0019 4006 4475 0000 00"),
        ]
    ]

    checks = analyze_ocr_laboratory(payload, reference_date=date(2026, 7, 30))
    identifiers = next(check for check in checks if check.code == "ocr_identifiers")

    assert identifiers.state == "attention"
    invalid = next(item for item in identifiers.observations if item.code == "OCR_IBAN_INVALID")
    assert "invalid_country_structure" in invalid.evidence["invalid_reasons"]


def test_identifier_labels_are_general_and_do_not_consume_the_next_field() -> None:
    payload = [
        [
            _region("text", "Numéro de carte : 4111 1111 1111 1111"),
            _region(
                "text",
                "IBAN number: GB82 WEST 1234 5698 7654 32 BIC: HDFCINBB",
            ),
            _region("text", "CKYC Number: 12345678901234"),
            _region("text", "CKYC ID: O12345678901234"),
            _region("text", "MICR Code: 123456789"),
        ]
    ]

    checks = analyze_ocr_laboratory(payload, reference_date=date(2026, 7, 30))
    identifiers = next(check for check in checks if check.code == "ocr_identifiers")

    assert identifiers.state == "clear"
    assert {item.code for item in identifiers.observations} == {
        "OCR_CARD_LUHN_VALID",
        "OCR_IBAN_VALID",
        "OCR_BIC_VALID",
        "OCR_CKYC_VALID",
        "OCR_MICR_VALID",
    }
    iban = next(item for item in identifiers.observations if item.code == "OCR_IBAN_VALID")
    assert iban.evidence["length"] == 22
    assert sum(item.code == "OCR_CKYC_VALID" for item in identifiers.observations) == 2


def test_micr_without_country_context_is_observed_but_not_declared_invalid() -> None:
    payload = [
        [
            _region("text", "US check information"),
            _region("text", "MICR number: 123456789012"),
        ]
    ]

    checks = analyze_ocr_laboratory(payload, reference_date=date(2026, 7, 30))
    identifiers = next(check for check in checks if check.code == "ocr_identifiers")

    assert identifiers.state == "clear"
    assert identifiers.observations[0].code == "OCR_MICR_OBSERVED"
    assert identifiers.observations[0].evidence["jurisdiction"] is None


def test_unlabeled_long_numbers_are_not_guessed_as_bank_identifiers() -> None:
    payload = [
        [
            _region("text", "Document reference 4111111111111111"),
            _region("text", "Internal processing number 12345678901234"),
        ]
    ]

    checks = analyze_ocr_laboratory(payload, reference_date=date(2026, 7, 30))
    identifiers = next(check for check in checks if check.code == "ocr_identifiers")

    assert identifiers.state == "not_applicable"
    assert identifiers.observations == ()


def test_descending_statement_dates_are_not_a_chronology_anomaly() -> None:
    payload = [
        [
            _region("text", "Currency: EUR"),
            _region(
                "table",
                """
                <table>
                  <tr>
                    <th>Booking date</th><th>Withdrawal</th>
                    <th>Deposit</th><th>Current balance</th>
                  </tr>
                  <tr><td></td><td></td><td></td><td>100.00</td></tr>
                  <tr><td>2026-03-03</td><td>10.00</td><td></td><td>90.00</td></tr>
                  <tr><td>2026-03-02</td><td>5.00</td><td></td><td>85.00</td></tr>
                  <tr><td>2026-03-01</td><td></td><td>2.00</td><td>87.00</td></tr>
                </table>
                """,
            ),
        ]
    ]

    checks = analyze_ocr_laboratory(payload, reference_date=date(2026, 7, 30))
    financial = next(check for check in checks if check.code == "ocr_financial_consistency")

    assert financial.state == "clear"
    assert financial.observations[0].code == "OCR_LEDGER_VALID"
    assert financial.observations[0].evidence["chronology_inversion_rows"] == []


def test_french_statement_summary_aliases_are_supported() -> None:
    payload = [
        [
            _region(
                "table",
                """
                <table>
                  <tr>
                    <th>Solde d'ouverture</th><th>Débit total</th>
                    <th>Crédit total</th><th>Solde de clôture</th>
                  </tr>
                  <tr><td>100,00</td><td>30,00</td><td>5,00</td><td>75,00</td></tr>
                </table>
                """,
            ),
        ]
    ]

    checks = analyze_ocr_laboratory(payload, reference_date=date(2026, 7, 30))
    financial = next(check for check in checks if check.code == "ocr_financial_consistency")

    assert financial.state == "clear"
    assert financial.observations[0].code == "OCR_STATEMENT_SUMMARY_VALID"


def _region(label: str, content: str) -> dict[str, object]:
    return {
        "index": 0,
        "label": label,
        "native_label": label,
        "content": content,
        "bbox_2d": [0, 0, 1000, 1000],
        "polygon": [],
    }
