from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from fraude_detector.laboratory.ocr import analyze_ocr_laboratory

FIXTURE_ROOT = Path("tests/fixtures/ocr")


def test_precomputed_ocr_fixtures_cover_expected_states() -> None:
    expectations = {
        "facture-soins": {
            "ocr_quality": "clear",
            "ocr_dates": "clear",
            "ocr_financial_consistency": "not_applicable",
        },
        "declaration-coherente": {
            "ocr_quality": "clear",
            "ocr_dates": "clear",
        },
        "contrat-bruite": {
            "ocr_quality": "clear",
            "ocr_financial_consistency": "indeterminate",
        },
        "releve-bancaire-anomalies": {
            "ocr_identifiers": "attention",
            "ocr_financial_consistency": "attention",
        },
        "texte-insuffisant": {
            "ocr_quality": "indeterminate",
            "ocr_identifiers": "not_applicable",
        },
    }

    for fixture_name, expected_states in expectations.items():
        payload = json.loads(
            (FIXTURE_ROOT / fixture_name / "document.json").read_text(encoding="utf-8")
        )
        checks = {
            check.code: check.state
            for check in analyze_ocr_laboratory(
                payload,
                reference_date=date(2026, 7, 30),
            )
        }
        assert checks | expected_states == checks
        assert all(checks[code] == state for code, state in expected_states.items())


def test_precomputed_ocr_fixtures_use_only_demo_identities() -> None:
    fixture_text = "\n".join(
        path.read_text(encoding="utf-8") for path in FIXTURE_ROOT.glob("*/*") if path.is_file()
    ).casefold()

    emails = re.findall(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", fixture_text)
    card_numbers = {
        re.sub(r"\D", "", candidate)
        for candidate in re.findall(r"(?<!\d)(?:\d[\s-]?){15}\d(?!\d)", fixture_text)
    }

    assert emails
    assert all(email.endswith("@example.test") for email in emails)
    assert card_numbers == {"4111111111111111"}
    assert "client demo" in fixture_text
    assert "camille exemple" in fixture_text
