from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from streamlit.testing.v1 import AppTest


def test_demo_waits_for_explicit_analysis(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("Montant modifie").run()

    markdown = "\n".join(element.value for element in app.markdown)
    assert not app.exception
    assert [button.label for button in app.button] == ["Analyser le fichier"]
    assert "Fichier pret pour analyse" in markdown
    assert not app.tabs


def test_modified_demo_renders_integrated_report(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("Montant modifie").run()
    app.button[0].click().run(timeout=30)

    markdown = "\n".join(element.value for element in app.markdown)
    assert not app.exception
    assert "Indicateurs" in markdown
    assert "Zones a reviser" in markdown
    assert "Laboratoire" in markdown
    assert [tab.label for tab in app.tabs] == ["Analyse", "OCR", "Laboratoire"]
    assert "Generation par IA" in markdown

    assert "Signature PDF / PAdES" in markdown
    assert "Factur-X" in markdown
    assert "2D-Doc" in markdown
    assert "Historique complet" in markdown


def test_ocr_results_and_solo_checks_have_dedicated_tabs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))
    monkeypatch.setenv("FROD_OCR_URL", "http://ocr.test:8007")
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "json_result": [
                    [
                        _region("text", "Document reconnu"),
                        _region("text", "Date du sinistre : 18/07/2026"),
                    ]
                ],
                "markdown_result": "## Document reconnu\n\nDate du sinistre : 18/07/2026",
            }
        ),
    )

    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("Montant modifie").run()
    app.button[0].click().run(timeout=30)

    markdown = "\n".join(element.value for element in app.markdown)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Analyse", "OCR", "Laboratoire"]
    assert "Texte reconnu" in markdown
    assert "Analyse OCR solo" in markdown
    assert "Exploitabilite OCR" in markdown
    assert "Coherence des dates" in markdown


def test_precomputed_ocr_demo_runs_without_source_document(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))

    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("OCR - Releve bancaire a anomalies").run()

    assert [button.label for button in app.button] == ["Analyser les donnees OCR"]
    assert not app.slider
    assert not app.select_slider

    app.button[0].click().run(timeout=30)
    markdown = "\n".join(element.value for element in app.markdown)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Analyse", "OCR", "Laboratoire"]
    assert "30/30 points contenu" in markdown
    assert "Coherence du contenu" in markdown
    assert "Analyse OCR solo" in markdown
    assert "Identifiants structures" in markdown
    assert "Coherence des montants" in markdown
    assert "anomalie(s)" in markdown
    assert "BIC / SWIFT" in markdown
    assert "Referentiels bancaires de pays differents" in markdown


def test_local_original_ocr_demo_is_discovered_when_present(
    monkeypatch,
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "frod"
    fixture_dir = work_dir / "ocr-fixtures/original/releve-bancaire-anomalies"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "document.json").write_text("[]\n", encoding="utf-8")
    (fixture_dir / "document.md").write_text("Original local\n", encoding="utf-8")
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(work_dir))

    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()

    assert "OCR original local - Releve bancaire a anomalies" in app.selectbox[0].options


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def _region(label: str, content: str) -> dict[str, object]:
    return {
        "index": 0,
        "label": label,
        "native_label": label,
        "content": content,
        "bbox_2d": [0, 0, 1000, 1000],
        "polygon": [],
    }
