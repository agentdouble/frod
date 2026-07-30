from __future__ import annotations

from pathlib import Path

import fraude_detector.config_cli as config_cli


def test_config_cli_requires_an_input_path(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    result = config_cli.main(["--config", str(config_path)])

    assert result == 2
    assert "input.path" in capsys.readouterr().err


def test_config_cli_forwards_the_same_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text('input:\n  path: "document.pdf"\n', encoding="utf-8")
    received: list[str] = []
    monkeypatch.setattr(
        config_cli,
        "analysis_main",
        lambda arguments: received.extend(arguments) or 0,
    )

    result = config_cli.main(["--config", str(config_path)])

    assert result == 0
    assert received == [
        str((tmp_path / "document.pdf").resolve()),
        "--config",
        str(config_path.resolve()),
    ]
