from __future__ import annotations

from pathlib import Path

import pytest

from fraude_detector.run_config import RunConfigError, load_run_config


def test_load_run_config_resolves_input_relative_to_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text('input_path: "documents/test.png"\n', encoding="utf-8")

    config = load_run_config(config_path)

    assert config.input_path == (tmp_path / "documents/test.png").resolve()


def test_load_run_config_accepts_legacy_pdf_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text('pdf_path: "documents/test.pdf"\n', encoding="utf-8")

    config = load_run_config(config_path)

    assert config.input_path == (tmp_path / "documents/test.pdf").resolve()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{}\n", "input_path"),
        ("input_path: ''\n", "input_path"),
        ("input_path: a.png\npdf_path: a.pdf\n", "exactement une"),
        ("pdf: document.pdf\n", "inconnue"),
        ("- document.pdf\n", "objet YAML"),
    ],
)
def test_load_run_config_rejects_invalid_contract(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(RunConfigError, match=message):
        load_run_config(config_path)


def test_load_run_config_reports_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(RunConfigError, match="introuvable"):
        load_run_config(config_path)
