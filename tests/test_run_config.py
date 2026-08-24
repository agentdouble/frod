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
    assert config.analysis.classification_model == "minimax_m2_1"
    assert config.analysis.extraction_model == "minimax_m2_1"
    assert config.analysis.verification_model == "minimax_m2_1"
    assert config.analysis.synthesis_model == "minimax_m2_1"


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
classification:
  enabled: true
  url: "http://llm.internal:8030"
  model: "minimax-local"
  timeout_seconds: 90
  max_input_chars: 16000
  max_tokens: 500
  temperature: 0.1
extraction:
  enabled: true
  url: "http://extract.internal:8030"
  model: "minimax-extract"
  timeout_seconds: 240
  max_input_chars: 18000
  max_tokens: 7000
  temperature: 0.05
  coverage_retry: false
verification:
  enabled: true
  url: "http://verify.internal:8030"
  model: "minimax-verify"
  timeout_seconds: 210
  max_input_chars: 70000
  max_tokens: 5500
  temperature: 0.02
  issue_min_confidence: 0.88
synthesis:
  enabled: true
  url: "http://summary.internal:8030"
  model: "minimax-summary"
  timeout_seconds: 80
  max_input_chars: 22000
  max_tokens: 350
  temperature: 0.01
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
  visual_repetition_enabled: true
  visual_repetition_min_pages: 4
  visual_repetition_similarity: 0.93
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
    assert config.analysis.classification_enabled is True
    assert config.analysis.classification_url == "http://llm.internal:8030"
    assert config.analysis.classification_model == "minimax-local"
    assert config.analysis.classification_timeout_seconds == 90
    assert config.analysis.classification_max_input_chars == 16000
    assert config.analysis.classification_max_tokens == 500
    assert config.analysis.classification_temperature == 0.1
    assert config.analysis.extraction_enabled is True
    assert config.analysis.extraction_url == "http://extract.internal:8030"
    assert config.analysis.extraction_model == "minimax-extract"
    assert config.analysis.extraction_timeout_seconds == 240
    assert config.analysis.extraction_max_input_chars == 18000
    assert config.analysis.extraction_max_tokens == 7000
    assert config.analysis.extraction_temperature == 0.05
    assert config.analysis.extraction_coverage_retry is False
    assert config.analysis.verification_enabled is True
    assert config.analysis.verification_url == "http://verify.internal:8030"
    assert config.analysis.verification_model == "minimax-verify"
    assert config.analysis.verification_timeout_seconds == 210
    assert config.analysis.verification_max_input_chars == 70000
    assert config.analysis.verification_max_tokens == 5500
    assert config.analysis.verification_temperature == 0.02
    assert config.analysis.verification_issue_min_confidence == 0.88
    assert config.analysis.synthesis_enabled is True
    assert config.analysis.synthesis_url == "http://summary.internal:8030"
    assert config.analysis.synthesis_model == "minimax-summary"
    assert config.analysis.synthesis_timeout_seconds == 80
    assert config.analysis.synthesis_max_input_chars == 22000
    assert config.analysis.synthesis_max_tokens == 350
    assert config.analysis.synthesis_temperature == 0.01
    assert config.gapl.enabled is False
    assert config.gapl.device == "cpu"
    assert config.gapl.weights_path == (tmp_path / "weights/gapl.pt").resolve()
    assert config.trufor.max_pixels == 500000
    assert config.trufor.timeout_seconds == 600
    assert config.laboratory.pdf_enabled is False
    assert config.laboratory.visual_repetition_enabled is True
    assert config.laboratory.visual_repetition_min_pages == 4
    assert config.laboratory.visual_repetition_similarity == 0.93


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
            "FROD_CLASSIFICATION_URL": "http://llm:8030",
            "FROD_CLASSIFICATION_MODEL": "minimax-test",
            "FROD_EXTRACTION_URL": "http://extract:8030",
            "FROD_EXTRACTION_MODEL": "minimax-extract-test",
            "FROD_EXTRACTION_COVERAGE_RETRY": "false",
            "FROD_VERIFICATION_URL": "http://verify:8030",
            "FROD_VERIFICATION_MODEL": "minimax-verify-test",
            "FROD_SYNTHESIS_URL": "http://summary:8030",
            "FROD_SYNTHESIS_MODEL": "minimax-summary-test",
            "FROD_GAPL_ENABLED": "false",
            "FROD_AI_ANALYZE_PDF_IMAGES": "true",
            "FROD_TRUFOR_MAX_PIXELS": "600000",
        },
    )

    assert config.application.port == 8700
    assert config.analysis.ocr_enabled is True
    assert config.analysis.ocr_url == "http://environment:8100"
    assert config.analysis.classification_enabled is True
    assert config.analysis.classification_url == "http://llm:8030"
    assert config.analysis.classification_model == "minimax-test"
    assert config.analysis.extraction_enabled is True
    assert config.analysis.extraction_url == "http://extract:8030"
    assert config.analysis.extraction_model == "minimax-extract-test"
    assert config.analysis.extraction_coverage_retry is False
    assert config.analysis.verification_enabled is True
    assert config.analysis.verification_url == "http://verify:8030"
    assert config.analysis.verification_model == "minimax-verify-test"
    assert config.analysis.synthesis_enabled is True
    assert config.analysis.synthesis_url == "http://summary:8030"
    assert config.analysis.synthesis_model == "minimax-summary-test"
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
