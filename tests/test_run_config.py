from __future__ import annotations

from pathlib import Path

import pytest

from fraude_detector.run_config import RunConfigError, load_run_config


def test_load_run_config_resolves_input_relative_to_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text('input:\n  path: "documents/test.png"\n', encoding="utf-8")

    config = load_run_config(config_path, environ={})

    assert config.input_path == (tmp_path / "documents/test.png").resolve()
    assert config.application.work_dir == (tmp_path / ".frod").resolve()
    assert config.analysis.render_dpi == 144


def test_load_run_config_accepts_legacy_pdf_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text('pdf_path: "documents/test.pdf"\n', encoding="utf-8")

    config = load_run_config(config_path, environ={})

    assert config.input_path == (tmp_path / "documents/test.pdf").resolve()


def test_load_run_config_reads_all_runtime_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
application:
  host: "0.0.0.0"
  port: 8600
  work_dir: "runtime"
  max_upload_size_mb: 250
analysis:
  rendering:
    dpi: 180
  limits:
    max_pages: 12
    max_file_size_mb: 120
    max_render_pixels: 10000000
    max_embedded_image_pixels: 20000000
  composition:
    full_page_image_coverage: 0.70
    minimum_overlay_coverage: 0.002
    maximum_overlay_coverage: 0.40
  ela:
    jpeg_quality: 88
    block_size: 32
    robust_z_threshold: 4.0
    max_image_dimension: 1800
    max_region_area_fraction: 0.20
  ai_images:
    max_images: 10
    max_inventory_images: 50
    analyze_pdf_images: true
    min_photo_page_coverage: 0.03
    min_photo_side: 300
    min_photo_pixels: 300000
    max_image_dimension: 1200
    model_score_threshold: 0.75
    min_consensus_families: 3
    stability_max_delta: 0.10
ocr:
  enabled: true
  url: "http://ocr.internal:9000"
  timeout_seconds: 420
models:
  gapl:
    enabled: false
    weights_path: "weights/gapl.pt"
    device: "cpu"
  trufor:
    enabled: true
    weights_path: "weights/trufor.pth.tar"
    max_pixels: 500000
    timeout_seconds: 600
laboratory:
  pdf_enabled: false
  image_enabled: true
""",
        encoding="utf-8",
    )

    config = load_run_config(config_path, environ={})

    assert config.application.host == "0.0.0.0"
    assert config.application.port == 8600
    assert config.application.work_dir == (tmp_path / "runtime").resolve()
    assert config.analysis.render_dpi == 180
    assert config.analysis.max_pages == 12
    assert config.analysis.ela_jpeg_quality == 88
    assert config.analysis.ai_max_images == 10
    assert config.analysis.ai_analyze_pdf_images is True
    assert config.analysis.ocr_enabled is True
    assert config.analysis.ocr_url == "http://ocr.internal:9000"
    assert config.analysis.ocr_timeout_seconds == 420
    assert config.gapl.enabled is False
    assert config.gapl.device == "cpu"
    assert config.gapl.weights_path == (tmp_path / "weights/gapl.pt").resolve()
    assert config.trufor.max_pixels == 500000
    assert config.trufor.timeout_seconds == 600
    assert config.laboratory.pdf_enabled is False


def test_environment_explicitly_overrides_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ocr:
  enabled: false
  url: "http://yaml:8007"
models:
  gapl:
    enabled: true
""",
        encoding="utf-8",
    )

    config = load_run_config(
        config_path,
        environ={
            "FROD_PORT": "8700",
            "FROD_OCR_URL": "http://environment:8100",
            "FROD_GAPL_ENABLED": "false",
            "FROD_AI_ANALYZE_PDF_IMAGES": "true",
            "FROD_TRUFOR_MAX_PIXELS": "600000",
        },
    )

    assert config.application.port == 8700
    assert config.analysis.ocr_enabled is True
    assert config.analysis.ocr_url == "http://environment:8100"
    assert config.gapl.enabled is False
    assert config.analysis.ai_analyze_pdf_images is True
    assert config.trufor.max_pixels == 600000


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("input: []\n", "input"),
        ("input_path: a.png\npdf_path: a.pdf\n", "seul chemin"),
        ("pdf: document.pdf\n", "inconnue"),
        ("- document.pdf\n", "objet YAML"),
        ("ocr:\n  enabled: maybe\n", "true ou false"),
        ("analysis:\n  rendering:\n    dpp: 144\n", "inconnue"),
        (
            "application:\n  max_upload_size_mb: 50\n"
            "analysis:\n  limits:\n    max_file_size_mb: 100\n",
            "superieur ou egal",
        ),
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
        load_run_config(config_path, environ={})


def test_load_run_config_reports_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(RunConfigError, match="introuvable"):
        load_run_config(config_path, environ={})
