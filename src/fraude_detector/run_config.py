"""Load and validate the project-wide YAML configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fraude_detector.config import AnalysisConfig
from fraude_detector.gapl import GAPL_DEFAULT_CHECKPOINT
from fraude_detector.trufor import (
    TRUFOR_DEFAULT_CHECKPOINT,
    TRUFOR_MAX_PIXELS,
    TRUFOR_TIMEOUT_SECONDS,
)


class RunConfigError(ValueError):
    """A user-facing error in the project configuration."""


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    """Local Streamlit runtime settings."""

    host: str
    port: int
    work_dir: Path
    max_upload_size_mb: int


@dataclass(frozen=True, slots=True)
class GaplConfig:
    """Settings for the optional local AI-image detector."""

    enabled: bool
    weights_path: Path
    device: str


@dataclass(frozen=True, slots=True)
class TruForConfig:
    """Settings for the optional local manipulation detector."""

    enabled: bool
    weights_path: Path
    max_pixels: int
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class LaboratoryConfig:
    """Experimental controls that run outside the main score."""

    pdf_enabled: bool
    image_enabled: bool
    visual_repetition_enabled: bool
    visual_repetition_min_pages: int
    visual_repetition_similarity: float


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Complete project configuration loaded from ``config.yaml``."""

    source: Path
    input_path: Path | None
    application: ApplicationConfig
    analysis: AnalysisConfig
    gapl: GaplConfig
    trufor: TruForConfig
    laboratory: LaboratoryConfig


def load_run_config(
    config_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> RunConfig:
    """Read the YAML configuration and apply explicit environment overrides."""

    source = Path(config_path).expanduser().resolve()
    try:
        raw_config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RunConfigError(f"Configuration introuvable: {source}") from error
    except OSError as error:
        raise RunConfigError(f"Configuration illisible: {source}: {error}") from error
    except yaml.YAMLError as error:
        raise RunConfigError(f"YAML invalide dans {source}: {error}") from error

    config = _require_mapping(raw_config, source, "racine")
    _reject_unknown(
        config,
        {
            "input",
            "input_path",
            "pdf_path",
            "application",
            "analysis",
            "ocr",
            "classification",
            "extraction",
            "verification",
            "synthesis",
            "models",
            "laboratory",
        },
        source,
        "racine",
    )
    environment = os.environ if environ is None else environ
    base = source.parent

    analysis = _load_analysis(config, source, environment)
    application = _load_application(config, source, base, environment)
    if application.max_upload_size_mb < analysis.max_file_size_mb:
        raise RunConfigError(
            "application.max_upload_size_mb doit etre superieur ou egal "
            "a analysis.limits.max_file_size_mb"
        )

    gapl, trufor = _load_models(config, source, base, environment)
    laboratory = _load_laboratory(config, source, environment)
    return RunConfig(
        source=source,
        input_path=_load_input_path(config, source, base),
        application=application,
        analysis=analysis,
        gapl=gapl,
        trufor=trufor,
        laboratory=laboratory,
    )


def _load_input_path(
    config: dict[str, Any],
    source: Path,
    base: Path,
) -> Path | None:
    input_section = _section(config, "input", source)
    _reject_unknown(input_section, {"path"}, source, "input")
    configured = [
        ("input.path", input_section.get("path")),
        ("input_path", config.get("input_path")),
        ("pdf_path", config.get("pdf_path")),
    ]
    present = [(name, value) for name, value in configured if value is not None and value != ""]
    if len(present) > 1:
        raise RunConfigError(
            f"Definissez un seul chemin d'entree dans {source}: "
            + ", ".join(name for name, _ in present)
        )
    if not present:
        return None
    name, value = present[0]
    return _path(value, base, source, name)


def _load_application(
    config: dict[str, Any],
    source: Path,
    base: Path,
    environ: Mapping[str, str],
) -> ApplicationConfig:
    section = _section(config, "application", source)
    _reject_unknown(
        section,
        {"host", "port", "work_dir", "max_upload_size_mb"},
        source,
        "application",
    )
    host = _env_text(environ, "FROD_HOST") or _text(
        section.get("host", "127.0.0.1"),
        source,
        "application.host",
    )
    port_override = _env_int(environ, "FROD_PORT")
    port = (
        port_override
        if port_override is not None
        else _integer(section.get("port", 8501), source, "application.port")
    )
    upload_override = _env_int(environ, "FROD_MAX_UPLOAD_SIZE_MB")
    max_upload = (
        upload_override
        if upload_override is not None
        else _integer(
            section.get("max_upload_size_mb", 200),
            source,
            "application.max_upload_size_mb",
        )
    )
    raw_work_dir: object = environ.get("FROD_WORK_DIR") or section.get("work_dir", ".frod")
    if not 1 <= port <= 65_535:
        raise RunConfigError("application.port doit etre compris entre 1 et 65535")
    if max_upload < 1:
        raise RunConfigError("application.max_upload_size_mb doit etre au moins 1")
    return ApplicationConfig(
        host=host,
        port=port,
        work_dir=_path(raw_work_dir, base, source, "application.work_dir"),
        max_upload_size_mb=max_upload,
    )


def _load_analysis(
    config: dict[str, Any],
    source: Path,
    environ: Mapping[str, str],
) -> AnalysisConfig:
    section = _section(config, "analysis", source)
    _reject_unknown(
        section,
        {"rendering", "limits", "composition", "ela", "ai_images"},
        source,
        "analysis",
    )
    rendering = _section(section, "rendering", source, prefix="analysis")
    limits = _section(section, "limits", source, prefix="analysis")
    composition = _section(section, "composition", source, prefix="analysis")
    ela = _section(section, "ela", source, prefix="analysis")
    ai_images = _section(section, "ai_images", source, prefix="analysis")
    ocr = _section(config, "ocr", source)
    classification = _section(config, "classification", source)
    extraction = _section(config, "extraction", source)
    verification = _section(config, "verification", source)
    synthesis = _section(config, "synthesis", source)

    _reject_unknown(rendering, {"dpi"}, source, "analysis.rendering")
    _reject_unknown(
        limits,
        {
            "max_pages",
            "max_file_size_mb",
            "max_render_pixels",
            "max_embedded_image_pixels",
        },
        source,
        "analysis.limits",
    )
    _reject_unknown(
        composition,
        {
            "full_page_image_coverage",
            "minimum_overlay_coverage",
            "maximum_overlay_coverage",
        },
        source,
        "analysis.composition",
    )
    _reject_unknown(
        ela,
        {
            "jpeg_quality",
            "block_size",
            "robust_z_threshold",
            "max_image_dimension",
            "max_region_area_fraction",
        },
        source,
        "analysis.ela",
    )
    _reject_unknown(
        ai_images,
        {
            "max_images",
            "max_inventory_images",
            "analyze_pdf_images",
            "min_photo_page_coverage",
            "min_photo_side",
            "min_photo_pixels",
            "max_image_dimension",
            "model_score_threshold",
            "min_consensus_families",
            "stability_max_delta",
        },
        source,
        "analysis.ai_images",
    )
    _reject_unknown(ocr, {"enabled", "url", "timeout_seconds"}, source, "ocr")
    _reject_unknown(
        classification,
        {
            "enabled",
            "url",
            "model",
            "timeout_seconds",
            "max_input_chars",
            "max_tokens",
            "temperature",
        },
        source,
        "classification",
    )
    _reject_unknown(
        extraction,
        {
            "enabled",
            "url",
            "model",
            "timeout_seconds",
            "max_input_chars",
            "max_tokens",
            "temperature",
            "coverage_retry",
        },
        source,
        "extraction",
    )
    _reject_unknown(
        verification,
        {
            "enabled",
            "url",
            "model",
            "timeout_seconds",
            "max_input_chars",
            "max_tokens",
            "temperature",
            "issue_min_confidence",
        },
        source,
        "verification",
    )
    _reject_unknown(
        synthesis,
        {
            "enabled",
            "url",
            "model",
            "timeout_seconds",
            "max_input_chars",
            "max_tokens",
            "temperature",
        },
        source,
        "synthesis",
    )

    ocr_url_override = _env_text(environ, "FROD_OCR_URL")
    ocr_enabled_override = _env_bool(environ, "FROD_OCR_ENABLED")
    ocr_enabled = (
        ocr_enabled_override
        if ocr_enabled_override is not None
        else bool(ocr_url_override) or _boolean(ocr.get("enabled", False), source, "ocr.enabled")
    )
    ocr_url = ocr_url_override or _text(
        ocr.get("url", "http://127.0.0.1:8007"),
        source,
        "ocr.url",
    )
    ocr_timeout_override = _env_int(environ, "FROD_OCR_TIMEOUT_SECONDS")
    ocr_timeout = (
        ocr_timeout_override
        if ocr_timeout_override is not None
        else _integer(
            ocr.get("timeout_seconds", 300),
            source,
            "ocr.timeout_seconds",
        )
    )
    classification_url_override = _env_text(environ, "FROD_CLASSIFICATION_URL")
    classification_enabled_override = _env_bool(environ, "FROD_CLASSIFICATION_ENABLED")
    classification_enabled = (
        classification_enabled_override
        if classification_enabled_override is not None
        else bool(classification_url_override)
        or _boolean(
            classification.get("enabled", False),
            source,
            "classification.enabled",
        )
    )
    classification_url = classification_url_override or _text(
        classification.get("url", "http://127.0.0.1:8030"),
        source,
        "classification.url",
    )
    classification_model = _env_text(environ, "FROD_CLASSIFICATION_MODEL") or _text(
        classification.get("model", "minimax_m2_1"),
        source,
        "classification.model",
    )
    classification_timeout_override = _env_int(environ, "FROD_CLASSIFICATION_TIMEOUT_SECONDS")
    classification_timeout = (
        classification_timeout_override
        if classification_timeout_override is not None
        else _integer(
            classification.get("timeout_seconds", 900),
            source,
            "classification.timeout_seconds",
        )
    )
    classification_max_input_chars = _integer(
        classification.get("max_input_chars", 20_000),
        source,
        "classification.max_input_chars",
    )
    classification_max_tokens = _integer(
        classification.get("max_tokens", 32_768),
        source,
        "classification.max_tokens",
    )
    classification_temperature = _number(
        classification.get("temperature", 0.0),
        source,
        "classification.temperature",
    )
    extraction_url_override = _env_text(environ, "FROD_EXTRACTION_URL")
    extraction_enabled_override = _env_bool(environ, "FROD_EXTRACTION_ENABLED")
    extraction_enabled = (
        extraction_enabled_override
        if extraction_enabled_override is not None
        else bool(extraction_url_override)
        or _boolean(extraction.get("enabled", False), source, "extraction.enabled")
    )
    extraction_url = extraction_url_override or _text(
        extraction.get("url", "http://127.0.0.1:8030"),
        source,
        "extraction.url",
    )
    extraction_model = _env_text(environ, "FROD_EXTRACTION_MODEL") or _text(
        extraction.get("model", "minimax_m2_1"),
        source,
        "extraction.model",
    )
    extraction_timeout_override = _env_int(environ, "FROD_EXTRACTION_TIMEOUT_SECONDS")
    extraction_timeout = (
        extraction_timeout_override
        if extraction_timeout_override is not None
        else _integer(
            extraction.get("timeout_seconds", 1_800),
            source,
            "extraction.timeout_seconds",
        )
    )
    extraction_max_input_chars = _integer(
        extraction.get("max_input_chars", 48_000),
        source,
        "extraction.max_input_chars",
    )
    extraction_max_tokens = _integer(
        extraction.get("max_tokens", 32_768),
        source,
        "extraction.max_tokens",
    )
    extraction_temperature = _number(
        extraction.get("temperature", 0.0),
        source,
        "extraction.temperature",
    )
    extraction_coverage_retry_override = _env_bool(
        environ,
        "FROD_EXTRACTION_COVERAGE_RETRY",
    )
    extraction_coverage_retry = (
        extraction_coverage_retry_override
        if extraction_coverage_retry_override is not None
        else _boolean(
            extraction.get("coverage_retry", False),
            source,
            "extraction.coverage_retry",
        )
    )
    verification_url_override = _env_text(environ, "FROD_VERIFICATION_URL")
    verification_enabled_override = _env_bool(environ, "FROD_VERIFICATION_ENABLED")
    verification_enabled = (
        verification_enabled_override
        if verification_enabled_override is not None
        else bool(verification_url_override)
        or _boolean(verification.get("enabled", False), source, "verification.enabled")
    )
    verification_url = verification_url_override or _text(
        verification.get("url", "http://127.0.0.1:8030"),
        source,
        "verification.url",
    )
    verification_model = _env_text(environ, "FROD_VERIFICATION_MODEL") or _text(
        verification.get("model", "minimax_m2_1"),
        source,
        "verification.model",
    )
    verification_timeout_override = _env_int(environ, "FROD_VERIFICATION_TIMEOUT_SECONDS")
    verification_timeout = (
        verification_timeout_override
        if verification_timeout_override is not None
        else _integer(
            verification.get("timeout_seconds", 1_800),
            source,
            "verification.timeout_seconds",
        )
    )
    verification_max_input_chars = _integer(
        verification.get("max_input_chars", 80_000),
        source,
        "verification.max_input_chars",
    )
    verification_max_tokens = _integer(
        verification.get("max_tokens", 32_768),
        source,
        "verification.max_tokens",
    )
    verification_temperature = _number(
        verification.get("temperature", 0.0),
        source,
        "verification.temperature",
    )
    verification_issue_min_confidence = _number(
        verification.get("issue_min_confidence", 0.80),
        source,
        "verification.issue_min_confidence",
    )
    synthesis_url_override = _env_text(environ, "FROD_SYNTHESIS_URL")
    synthesis_enabled_override = _env_bool(environ, "FROD_SYNTHESIS_ENABLED")
    synthesis_enabled = (
        synthesis_enabled_override
        if synthesis_enabled_override is not None
        else bool(synthesis_url_override)
        or _boolean(synthesis.get("enabled", False), source, "synthesis.enabled")
    )
    synthesis_url = synthesis_url_override or _text(
        synthesis.get("url", "http://127.0.0.1:8030"),
        source,
        "synthesis.url",
    )
    synthesis_model = _env_text(environ, "FROD_SYNTHESIS_MODEL") or _text(
        synthesis.get("model", "minimax_m2_1"),
        source,
        "synthesis.model",
    )
    synthesis_timeout_override = _env_int(environ, "FROD_SYNTHESIS_TIMEOUT_SECONDS")
    synthesis_timeout = (
        synthesis_timeout_override
        if synthesis_timeout_override is not None
        else _integer(
            synthesis.get("timeout_seconds", 900),
            source,
            "synthesis.timeout_seconds",
        )
    )
    synthesis_max_input_chars = _integer(
        synthesis.get("max_input_chars", 24_000),
        source,
        "synthesis.max_input_chars",
    )
    synthesis_max_tokens = _integer(
        synthesis.get("max_tokens", 32_768),
        source,
        "synthesis.max_tokens",
    )
    synthesis_temperature = _number(
        synthesis.get("temperature", 0.0),
        source,
        "synthesis.temperature",
    )
    ai_pdf_override = _env_bool(environ, "FROD_AI_ANALYZE_PDF_IMAGES")

    try:
        return AnalysisConfig(
            render_dpi=_integer(rendering.get("dpi", 144), source, "analysis.rendering.dpi"),
            max_pages=_integer(limits.get("max_pages", 25), source, "analysis.limits.max_pages"),
            max_file_size_mb=_integer(
                limits.get("max_file_size_mb", 100),
                source,
                "analysis.limits.max_file_size_mb",
            ),
            max_render_pixels=_integer(
                limits.get("max_render_pixels", 25_000_000),
                source,
                "analysis.limits.max_render_pixels",
            ),
            max_embedded_image_pixels=_integer(
                limits.get("max_embedded_image_pixels", 50_000_000),
                source,
                "analysis.limits.max_embedded_image_pixels",
            ),
            full_page_image_coverage=_number(
                composition.get("full_page_image_coverage", 0.65),
                source,
                "analysis.composition.full_page_image_coverage",
            ),
            minimum_overlay_coverage=_number(
                composition.get("minimum_overlay_coverage", 0.001),
                source,
                "analysis.composition.minimum_overlay_coverage",
            ),
            maximum_overlay_coverage=_number(
                composition.get("maximum_overlay_coverage", 0.50),
                source,
                "analysis.composition.maximum_overlay_coverage",
            ),
            ela_jpeg_quality=_integer(
                ela.get("jpeg_quality", 90),
                source,
                "analysis.ela.jpeg_quality",
            ),
            ela_block_size=_integer(
                ela.get("block_size", 64),
                source,
                "analysis.ela.block_size",
            ),
            ela_robust_z_threshold=_number(
                ela.get("robust_z_threshold", 3.5),
                source,
                "analysis.ela.robust_z_threshold",
            ),
            ela_max_image_dimension=_integer(
                ela.get("max_image_dimension", 2400),
                source,
                "analysis.ela.max_image_dimension",
            ),
            ela_max_region_area_fraction=_number(
                ela.get("max_region_area_fraction", 0.25),
                source,
                "analysis.ela.max_region_area_fraction",
            ),
            ai_max_images=_integer(
                ai_images.get("max_images", 20),
                source,
                "analysis.ai_images.max_images",
            ),
            ai_max_inventory_images=_integer(
                ai_images.get("max_inventory_images", 100),
                source,
                "analysis.ai_images.max_inventory_images",
            ),
            ai_analyze_pdf_images=(
                ai_pdf_override
                if ai_pdf_override is not None
                else _boolean(
                    ai_images.get("analyze_pdf_images", False),
                    source,
                    "analysis.ai_images.analyze_pdf_images",
                )
            ),
            ai_min_photo_page_coverage=_number(
                ai_images.get("min_photo_page_coverage", 0.02),
                source,
                "analysis.ai_images.min_photo_page_coverage",
            ),
            ai_min_photo_side=_integer(
                ai_images.get("min_photo_side", 256),
                source,
                "analysis.ai_images.min_photo_side",
            ),
            ai_min_photo_pixels=_integer(
                ai_images.get("min_photo_pixels", 262_144),
                source,
                "analysis.ai_images.min_photo_pixels",
            ),
            ai_max_image_dimension=_integer(
                ai_images.get("max_image_dimension", 1536),
                source,
                "analysis.ai_images.max_image_dimension",
            ),
            ai_model_score_threshold=_number(
                ai_images.get("model_score_threshold", 0.80),
                source,
                "analysis.ai_images.model_score_threshold",
            ),
            ai_min_consensus_families=_integer(
                ai_images.get("min_consensus_families", 2),
                source,
                "analysis.ai_images.min_consensus_families",
            ),
            ai_stability_max_delta=_number(
                ai_images.get("stability_max_delta", 0.15),
                source,
                "analysis.ai_images.stability_max_delta",
            ),
            ocr_enabled=ocr_enabled,
            ocr_url=ocr_url,
            ocr_timeout_seconds=ocr_timeout,
            classification_enabled=classification_enabled,
            classification_url=classification_url,
            classification_model=classification_model,
            classification_timeout_seconds=classification_timeout,
            classification_max_input_chars=classification_max_input_chars,
            classification_max_tokens=classification_max_tokens,
            classification_temperature=classification_temperature,
            extraction_enabled=extraction_enabled,
            extraction_url=extraction_url,
            extraction_model=extraction_model,
            extraction_timeout_seconds=extraction_timeout,
            extraction_max_input_chars=extraction_max_input_chars,
            extraction_max_tokens=extraction_max_tokens,
            extraction_temperature=extraction_temperature,
            extraction_coverage_retry=extraction_coverage_retry,
            verification_enabled=verification_enabled,
            verification_url=verification_url,
            verification_model=verification_model,
            verification_timeout_seconds=verification_timeout,
            verification_max_input_chars=verification_max_input_chars,
            verification_max_tokens=verification_max_tokens,
            verification_temperature=verification_temperature,
            verification_issue_min_confidence=verification_issue_min_confidence,
            synthesis_enabled=synthesis_enabled,
            synthesis_url=synthesis_url,
            synthesis_model=synthesis_model,
            synthesis_timeout_seconds=synthesis_timeout,
            synthesis_max_input_chars=synthesis_max_input_chars,
            synthesis_max_tokens=synthesis_max_tokens,
            synthesis_temperature=synthesis_temperature,
        )
    except ValueError as error:
        raise RunConfigError(f"Configuration d'analyse invalide: {error}") from error


def _load_models(
    config: dict[str, Any],
    source: Path,
    base: Path,
    environ: Mapping[str, str],
) -> tuple[GaplConfig, TruForConfig]:
    models = _section(config, "models", source)
    _reject_unknown(models, {"gapl", "trufor"}, source, "models")
    gapl = _section(models, "gapl", source, prefix="models")
    trufor = _section(models, "trufor", source, prefix="models")
    _reject_unknown(gapl, {"enabled", "weights_path", "device"}, source, "models.gapl")
    _reject_unknown(
        trufor,
        {"enabled", "weights_path", "max_pixels", "timeout_seconds"},
        source,
        "models.trufor",
    )

    gapl_enabled = _environment_boolean(
        environ,
        "FROD_GAPL_ENABLED",
        gapl.get("enabled", True),
        source,
        "models.gapl.enabled",
    )
    gapl_device = _env_text(environ, "FROD_GAPL_DEVICE") or _text(
        gapl.get("device", "auto"),
        source,
        "models.gapl.device",
    )
    if gapl_device not in {"auto", "cpu", "cuda", "mps"}:
        raise RunConfigError("models.gapl.device doit valoir auto, cpu, cuda ou mps")
    gapl_path = _path(
        environ.get("FROD_GAPL_WEIGHTS") or gapl.get("weights_path", str(GAPL_DEFAULT_CHECKPOINT)),
        base,
        source,
        "models.gapl.weights_path",
    )

    trufor_enabled = _environment_boolean(
        environ,
        "FROD_TRUFOR_ENABLED",
        trufor.get("enabled", True),
        source,
        "models.trufor.enabled",
    )
    trufor_path = _path(
        environ.get("FROD_TRUFOR_WEIGHTS")
        or trufor.get("weights_path", str(TRUFOR_DEFAULT_CHECKPOINT)),
        base,
        source,
        "models.trufor.weights_path",
    )
    trufor_pixels_override = _env_int(environ, "FROD_TRUFOR_MAX_PIXELS")
    trufor_max_pixels = (
        trufor_pixels_override
        if trufor_pixels_override is not None
        else _integer(
            trufor.get("max_pixels", TRUFOR_MAX_PIXELS),
            source,
            "models.trufor.max_pixels",
        )
    )
    trufor_timeout_override = _env_int(environ, "FROD_TRUFOR_TIMEOUT_SECONDS")
    trufor_timeout = (
        trufor_timeout_override
        if trufor_timeout_override is not None
        else _integer(
            trufor.get("timeout_seconds", TRUFOR_TIMEOUT_SECONDS),
            source,
            "models.trufor.timeout_seconds",
        )
    )
    if trufor_max_pixels < 65_536:
        raise RunConfigError("models.trufor.max_pixels doit etre au moins 65536")
    if trufor_timeout < 1:
        raise RunConfigError("models.trufor.timeout_seconds doit etre au moins 1")
    return (
        GaplConfig(enabled=gapl_enabled, weights_path=gapl_path, device=gapl_device),
        TruForConfig(
            enabled=trufor_enabled,
            weights_path=trufor_path,
            max_pixels=trufor_max_pixels,
            timeout_seconds=trufor_timeout,
        ),
    )


def _load_laboratory(
    config: dict[str, Any],
    source: Path,
    environ: Mapping[str, str],
) -> LaboratoryConfig:
    section = _section(config, "laboratory", source)
    _reject_unknown(
        section,
        {
            "pdf_enabled",
            "image_enabled",
            "visual_repetition_enabled",
            "visual_repetition_min_pages",
            "visual_repetition_similarity",
        },
        source,
        "laboratory",
    )
    minimum_pages = _integer(
        section.get("visual_repetition_min_pages", 3),
        source,
        "laboratory.visual_repetition_min_pages",
    )
    similarity = _number(
        section.get("visual_repetition_similarity", 0.90),
        source,
        "laboratory.visual_repetition_similarity",
    )
    if minimum_pages < 3:
        raise RunConfigError("laboratory.visual_repetition_min_pages doit etre au moins 3")
    if not 0 < similarity <= 1:
        raise RunConfigError(
            "laboratory.visual_repetition_similarity doit etre compris entre 0 et 1"
        )
    return LaboratoryConfig(
        pdf_enabled=_environment_boolean(
            environ,
            "FROD_PDF_LAB_ENABLED",
            section.get("pdf_enabled", True),
            source,
            "laboratory.pdf_enabled",
        ),
        image_enabled=_environment_boolean(
            environ,
            "FROD_IMAGE_LAB_ENABLED",
            section.get("image_enabled", True),
            source,
            "laboratory.image_enabled",
        ),
        visual_repetition_enabled=_boolean(
            section.get("visual_repetition_enabled", True),
            source,
            "laboratory.visual_repetition_enabled",
        ),
        visual_repetition_min_pages=minimum_pages,
        visual_repetition_similarity=similarity,
    )


def _section(
    mapping: dict[str, Any],
    key: str,
    source: Path,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    value = mapping.get(key, {})
    path = f"{prefix}.{key}" if prefix else key
    return _require_mapping(value, source, path)


def _require_mapping(raw_config: Any, source: Path, path: str) -> dict[str, Any]:
    if not isinstance(raw_config, dict):
        raise RunConfigError(f"{path} doit etre un objet YAML dans {source}")
    if not all(isinstance(key, str) for key in raw_config):
        raise RunConfigError(f"Toutes les cles de {path} doivent etre du texte dans {source}")
    return raw_config


def _reject_unknown(
    mapping: dict[str, Any],
    allowed: set[str],
    source: Path,
    path: str,
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise RunConfigError(f"Cle(s) inconnue(s) dans {path} ({source}): " + ", ".join(unknown))


def _path(value: object, base: Path, source: Path, name: str) -> Path:
    text = _text(value, source, name)
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _text(value: object, source: Path, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunConfigError(f"{name} doit contenir du texte dans {source}")
    return value.strip()


def _integer(value: object, source: Path, name: str) -> int:
    if type(value) is not int:
        raise RunConfigError(f"{name} doit etre un entier dans {source}")
    return value


def _number(value: object, source: Path, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunConfigError(f"{name} doit etre un nombre dans {source}")
    return float(value)


def _boolean(value: object, source: Path, name: str) -> bool:
    if type(value) is not bool:
        raise RunConfigError(f"{name} doit etre true ou false dans {source}")
    return value


def _env_text(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return value.strip() if value is not None and value.strip() else None


def _env_int(environ: Mapping[str, str], name: str) -> int | None:
    value = _env_text(environ, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise RunConfigError(f"{name} doit etre un entier") from error


def _env_bool(environ: Mapping[str, str], name: str) -> bool | None:
    value = _env_text(environ, name)
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RunConfigError(f"{name} doit valoir true ou false")


def _environment_boolean(
    environ: Mapping[str, str],
    env_name: str,
    configured: object,
    source: Path,
    path: str,
) -> bool:
    override = _env_bool(environ, env_name)
    return override if override is not None else _boolean(configured, source, path)
