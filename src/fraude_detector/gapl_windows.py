"""Centered multi-window experiment for the local GAPL adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fraude_detector.ai_images import AiImageModelAdapter
from fraude_detector.gapl import GAPL_INPUT_SIZE, GaplError
from fraude_detector.models import BoundingBox, Finding

GAPL_FROD_SCORE_START = 0.50
GAPL_FROD_SCORE_MAXIMUM = 0.90
GAPL_FROD_MAX_POINTS = 30.0


@dataclass(frozen=True, slots=True)
class GaplWindow:
    index: int
    ring: int
    box: tuple[int, int, int, int]
    is_center: bool


@dataclass(frozen=True, slots=True)
class GaplWindowScore:
    window: GaplWindow
    score: float


@dataclass(frozen=True, slots=True)
class GaplWindowAnalysis:
    image_size: tuple[int, int]
    windows: tuple[GaplWindowScore, ...]
    center_score: float
    median_score: float
    percentile_10_score: float
    percentile_25_score: float
    percentile_90_score: float
    maximum_score: float
    high_window_ratio: float
    coverage_ratio: float

    @property
    def global_score(self) -> float:
        score = (
            0.5 * self.median_score + 0.3 * self.percentile_25_score + 0.2 * self.high_window_ratio
        )
        return min(1.0, max(0.0, score))

    @property
    def dispersion(self) -> float:
        return self.percentile_90_score - self.percentile_10_score


def gapl_frod_points(global_score: float) -> float:
    """Map an uncalibrated GAPL index to a conservative Frod contribution."""

    if not 0 <= global_score <= 1:
        raise ValueError("global_score must be between 0 and 1")
    span = GAPL_FROD_SCORE_MAXIMUM - GAPL_FROD_SCORE_START
    normalized = min(
        1.0,
        max(0.0, (global_score - GAPL_FROD_SCORE_START) / span),
    )
    return round(GAPL_FROD_MAX_POINTS * normalized**2, 2)


def build_gapl_global_finding(
    analysis: GaplWindowAnalysis,
    *,
    artifacts: tuple[str, ...],
    page: int | None = None,
    bbox: BoundingBox | None = None,
) -> Finding:
    points = gapl_frod_points(analysis.global_score)
    return Finding(
        detector="ai_generated_image",
        code="AI_GAPL_GLOBAL_TRACE",
        category="synthetic_media" if points > 0 else "analysis_quality",
        title=(
            "Traces globales compatibles avec une generation IA"
            if points > 0
            else "Indice global GAPL faible"
        ),
        description=(
            "GAPL retrouve des caracteristiques apprises sur des images generees "
            "dans plusieurs zones. Ce signal statistique ne prouve pas une fraude."
        ),
        risk_points=points,
        confidence=min(
            0.8,
            0.4 + 0.3 * analysis.high_window_ratio + 0.1 * analysis.coverage_ratio,
        ),
        page=page,
        bbox=bbox,
        evidence={
            "global_index": analysis.global_score,
            "median_score": analysis.median_score,
            "percentile_10_score": analysis.percentile_10_score,
            "percentile_25_score": analysis.percentile_25_score,
            "high_window_ratio": analysis.high_window_ratio,
            "coverage_ratio": analysis.coverage_ratio,
            "window_count": len(analysis.windows),
            "point_policy": {
                "start": GAPL_FROD_SCORE_START,
                "maximum": GAPL_FROD_SCORE_MAXIMUM,
                "maximum_points": GAPL_FROD_MAX_POINTS,
                "curve": "quadratic",
            },
        },
        artifacts=artifacts,
    )


def write_gapl_window_artifacts(
    output_dir: Path,
    analysis: GaplWindowAnalysis,
    overlay: Image.Image,
    *,
    stem: str,
) -> tuple[str, str]:
    """Persist the score map and its machine-readable evidence."""

    relative_image = Path("forensics/ai") / f"{stem}-windows.png"
    relative_json = Path("forensics/ai") / f"{stem}-windows.json"
    image_path = output_dir / relative_image
    json_path = output_dir / relative_json
    image_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(image_path, format="PNG")
    json_path.write_text(
        json.dumps(gapl_window_payload(analysis), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return relative_json.as_posix(), relative_image.as_posix()


def gapl_window_payload(analysis: GaplWindowAnalysis) -> dict[str, Any]:
    return {
        "image_size": list(analysis.image_size),
        "window_size": GAPL_INPUT_SIZE,
        "stride": GAPL_INPUT_SIZE,
        "aggregation": {
            "formula": "0.50 * median + 0.30 * percentile_25 + 0.20 * high_window_ratio",
            "high_window_threshold": 0.5,
        },
        "global_score": analysis.global_score,
        "frod_points": gapl_frod_points(analysis.global_score),
        "center_score": analysis.center_score,
        "median_score": analysis.median_score,
        "percentile_10_score": analysis.percentile_10_score,
        "percentile_25_score": analysis.percentile_25_score,
        "percentile_90_score": analysis.percentile_90_score,
        "maximum_score": analysis.maximum_score,
        "high_window_ratio": analysis.high_window_ratio,
        "coverage_ratio": analysis.coverage_ratio,
        "dispersion": analysis.dispersion,
        "windows": [
            {
                "index": item.window.index,
                "ring": item.window.ring,
                "box": list(item.window.box),
                "center": item.window.is_center,
                "score": item.score,
            }
            for item in analysis.windows
        ],
    }


def analyze_centered_windows(
    adapter: AiImageModelAdapter,
    image: Image.Image,
    *,
    max_dimension: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[GaplWindowAnalysis, Image.Image]:
    """Score complete 224px windows, starting at the official center crop."""

    prepared = _bounded_rgb(image, max_dimension)
    windows = centered_windows(prepared.size)
    if not windows:
        raise GaplError("L'image est trop petite pour une fenetre complete de 224 x 224.")

    scores: list[GaplWindowScore] = []
    if progress_callback is not None:
        progress_callback(0, len(windows))
    for completed, window in enumerate(windows, start=1):
        prediction = adapter.predict(prepared.crop(window.box))
        if not prediction.in_domain:
            raise GaplError(
                prediction.out_of_domain_reason or "Une fenetre est hors du domaine du modele GAPL."
            )
        scores.append(GaplWindowScore(window=window, score=prediction.synthetic_score))
        if progress_callback is not None:
            progress_callback(completed, len(windows))

    values = np.asarray([item.score for item in scores], dtype=np.float64)
    center = next(item.score for item in scores if item.window.is_center)
    window_area = GAPL_INPUT_SIZE * GAPL_INPUT_SIZE
    image_area = prepared.width * prepared.height
    analysis = GaplWindowAnalysis(
        image_size=prepared.size,
        windows=tuple(scores),
        center_score=center,
        median_score=float(median(values)),
        percentile_10_score=float(np.quantile(values, 0.1)),
        percentile_25_score=float(np.quantile(values, 0.25)),
        percentile_90_score=float(np.quantile(values, 0.9)),
        maximum_score=float(values.max()),
        high_window_ratio=float(np.mean(values >= 0.5)),
        coverage_ratio=min(1.0, len(windows) * window_area / image_area),
    )
    return analysis, render_window_scores(prepared, analysis)


def centered_windows(
    image_size: tuple[int, int],
    *,
    window_size: int = GAPL_INPUT_SIZE,
) -> tuple[GaplWindow, ...]:
    """Build a centered, non-overlapping lattice without padding or partial crops."""

    width, height = image_size
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if width < window_size or height < window_size:
        return ()

    center_left = int(round((width - window_size) / 2.0))
    center_top = int(round((height - window_size) / 2.0))
    horizontal = _axis_offsets(center_left, width, window_size)
    vertical = _axis_offsets(center_top, height, window_size)
    candidates = [
        (
            max(abs(row_offset), abs(column_offset)),
            abs(row_offset) + abs(column_offset),
            row_offset,
            column_offset,
            left,
            top,
        )
        for column_offset, left in horizontal
        for row_offset, top in vertical
    ]
    candidates.sort()

    return tuple(
        GaplWindow(
            index=index,
            ring=ring,
            box=(left, top, left + window_size, top + window_size),
            is_center=row_offset == 0 and column_offset == 0,
        )
        for index, (
            ring,
            _distance,
            row_offset,
            column_offset,
            left,
            top,
        ) in enumerate(candidates)
    )


def render_window_scores(
    image: Image.Image,
    analysis: GaplWindowAnalysis,
) -> Image.Image:
    """Overlay absolute GAPL scores without interpolating unobserved image areas."""

    base = image.convert("RGBA")
    fill_layer = Image.new("RGBA", base.size)
    fills = ImageDraw.Draw(fill_layer)
    for item in analysis.windows:
        color = _score_color(item.score)
        fills.rectangle(item.window.box, fill=(*color, 50))
    base = Image.alpha_composite(base, fill_layer)

    draw = ImageDraw.Draw(base)
    font = _overlay_font()
    for item in analysis.windows:
        left, top, right, bottom = item.window.box
        color = _score_color(item.score)
        draw.rectangle(
            (left, top, right - 1, bottom - 1),
            outline=color,
            width=4 if item.window.is_center else 2,
        )
        if item.window.is_center:
            draw.rectangle(
                (left + 4, top + 4, right - 5, bottom - 5),
                outline=(34, 211, 238),
                width=3,
            )
        label = f"{'C ' if item.window.is_center else ''}{item.score:.0%}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_box = (
            left + 8,
            top + 8,
            left + 18 + text_width,
            top + 16 + text_height,
        )
        draw.rounded_rectangle(label_box, radius=4, fill=(6, 13, 23, 218))
        draw.text(
            (left + 13, top + 10),
            label,
            fill=(255, 255, 255),
            font=font,
        )
    return base.convert("RGB")


def _axis_offsets(
    center: int,
    length: int,
    window_size: int,
) -> tuple[tuple[int, int], ...]:
    positions: list[tuple[int, int]] = [(0, center)]
    distance = 1
    while center - distance * window_size >= 0:
        positions.append((-distance, center - distance * window_size))
        distance += 1
    distance = 1
    while center + distance * window_size + window_size <= length:
        positions.append((distance, center + distance * window_size))
        distance += 1
    return tuple(positions)


def _bounded_rgb(image: Image.Image, max_dimension: int) -> Image.Image:
    if max_dimension < 1:
        raise ValueError("max_dimension must be positive")
    rgb = image.convert("RGB")
    if max(rgb.size) <= max_dimension:
        return rgb.copy()
    scale = max_dimension / max(rgb.size)
    size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    return rgb.resize(size, Image.Resampling.LANCZOS)


def _score_color(score: float) -> tuple[int, int, int]:
    if score >= 0.5:
        return (251, 113, 133)
    if score >= 0.25:
        return (251, 191, 36)
    return (52, 211, 153)


def _overlay_font() -> Any:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        return ImageFont.load_default()
