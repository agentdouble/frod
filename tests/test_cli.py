from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import fraude_detector.cli as cli


def test_cli_routes_a_standalone_image_and_prints_its_type(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "white").save(source)

    result = cli.main([str(source), "--output", str(tmp_path / "output")])

    assert result == 0
    assert "Type d'entree: image" in capsys.readouterr().out


def test_default_cli_does_not_load_ai_model() -> None:
    args = cli.build_parser().parse_args(["document.pdf"])

    assert cli._build_ai_adapters(args) == ()


def test_ai_model_requires_explicit_weights_or_download() -> None:
    args = cli.build_parser().parse_args(["document.pdf", "--ai-model", "community-forensics"])

    with pytest.raises(cli.CommunityForensicsError, match="allow_hf_download=True"):
        cli._build_ai_adapters(args)


def test_local_ai_model_is_built_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"test")
    sentinel = object()
    received: dict[str, object] = {}

    def fake_factory(**kwargs: object) -> object:
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(cli, "create_community_forensics_adapter", fake_factory)
    args = cli.build_parser().parse_args(
        [
            "document.pdf",
            "--ai-model",
            "community-forensics",
            "--ai-model-path",
            str(weights),
            "--ai-model-variant",
            "224",
            "--ai-model-device",
            "mps",
        ]
    )

    assert cli._build_ai_adapters(args) == (sentinel,)
    assert received == {
        "weights_path": weights,
        "variant": "224",
        "device": "mps",
        "allow_hf_download": False,
    }
