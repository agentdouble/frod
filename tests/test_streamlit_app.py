from __future__ import annotations

from pathlib import Path

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
    assert not app.tabs
    assert "Generation par IA" in markdown

    app.toggle[0].set_value(True).run(timeout=30)
    markdown = "\n".join(element.value for element in app.markdown)
    assert "Signature PDF / PAdES" in markdown
    assert "Factur-X" in markdown
    assert "2D-Doc" in markdown
    assert "Historique complet" in markdown
