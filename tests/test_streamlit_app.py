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
    app.selectbox[0].select("Montant modifié").run()

    markdown = "\n".join(element.value for element in app.markdown)
    assert not app.exception
    assert [button.label for button in app.button] == ["Analyser le fichier"]
    assert "Fichier prêt pour analyse" in markdown
    assert not app.tabs
    assert not app.slider
    assert not app.select_slider


def test_modified_demo_renders_single_review_workspace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("Montant modifié").run()
    app.button[0].click().run(timeout=30)

    markdown = "\n".join(element.value for element in app.markdown)
    assert not app.exception
    assert "Synthèse de revue" in markdown
    assert "category-counters" in markdown
    assert "Modifications visuelles" in markdown
    assert "Différences visibles entre les versions du document." in markdown
    assert "Structure du fichier" in markdown
    assert "Signatures, structure interne et altérations du fichier." in markdown
    assert '<div class="score-meta' not in markdown
    assert ">Frod<" not in markdown
    assert "category-counter danger" in markdown
    assert "Comment lire cette vue" in markdown
    assert not app.tabs
    assert not app.expander
    assert "Zones à revoir - page 1" in app.selectbox[1].options

    assert "Signature électronique du PDF" in markdown
    assert "Facture électronique embarquée" in markdown
    assert "Code de vérification 2D-Doc" in markdown
    assert "Historique complet des versions" in markdown


def test_ocr_results_are_integrated_into_the_review_workspace(
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
    app.selectbox[0].select("Montant modifié").run()
    app.button[0].click().run(timeout=30)

    markdown = "\n".join(element.value for element in app.markdown)
    assert not app.exception
    assert not app.tabs
    assert not app.expander
    assert "Zones de texte reconnues - page 1" in app.selectbox[1].options
    assert "Contrôles effectués" in markdown
    assert "Qualité du texte reconnu" in markdown
    assert "Cohérence des dates" in markdown


def test_precomputed_ocr_demo_runs_without_source_document(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))

    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("OCR - Relevé bancaire à anomalies").run()

    assert [button.label for button in app.button] == ["Analyser les données OCR"]
    assert not app.slider
    assert not app.select_slider

    app.button[0].click().run(timeout=30)
    markdown = "\n".join(element.value for element in app.markdown)

    assert not app.exception
    assert not app.tabs
    assert not app.expander
    assert "<span>30</span><small>/30</small>" in markdown
    assert "Cohérence du contenu" in markdown
    assert "Identifiants reconnus" in markdown
    assert "Validité des identifiants" in markdown
    assert "Cohérence des montants" in markdown
    assert "anomalie(s)" in markdown
    assert "BIC / SWIFT" in markdown
    assert "Referentiels bancaires de pays differents" in markdown
    assert "4111 1111 1111 1111" in markdown
    assert "********" not in markdown


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


def test_business_ui_contains_no_json_renderer() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    document_view = source[
        source.index("def _render_document_view") : source.index("def _render_review_summary")
    ]

    assert "st.json" not in source
    assert "st.tabs" not in source
    assert "st.expander" not in source
    assert (
        document_view.index('st.selectbox("Vue affichée"')
        < document_view.index("view-guidance")
        < document_view.index("st.image")
    )


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
