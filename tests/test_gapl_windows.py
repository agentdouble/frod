from __future__ import annotations

import pytest
from PIL import Image, ImageStat

from fraude_detector.ai_images import AiImagePrediction
from fraude_detector.gapl_windows import (
    analyze_centered_windows,
    build_gapl_global_finding,
    centered_windows,
    gapl_frod_points,
)


def test_centered_windows_start_at_center_and_stop_before_borders() -> None:
    windows = centered_windows((1152, 1536))

    assert len(windows) == 25
    assert windows[0].is_center is True
    assert windows[0].box == (464, 656, 688, 880)
    assert {window.ring for window in windows} == {0, 1, 2}
    assert all(
        0 <= left < right <= 1152 and 0 <= top < bottom <= 1536
        for left, top, right, bottom in (window.box for window in windows)
    )
    assert len({window.box for window in windows}) == len(windows)


def test_centered_windows_do_not_pad_small_images() -> None:
    assert centered_windows((223, 500)) == ()
    assert centered_windows((500, 223)) == ()


def test_window_analysis_scores_center_first_and_renders_overlay() -> None:
    image = Image.new("RGB", (672, 672))
    image.paste((255, 255, 255), (224, 224, 448, 448))

    analysis, overlay = analyze_centered_windows(
        _BrightnessAdapter(),
        image,
        max_dimension=1536,
    )

    assert len(analysis.windows) == 9
    assert analysis.windows[0].window.is_center is True
    assert analysis.center_score == 1.0
    assert analysis.median_score == 0.0
    assert analysis.maximum_score == 1.0
    assert analysis.percentile_10_score == 0.0
    assert analysis.high_window_ratio == 1 / 9
    assert analysis.coverage_ratio == 1.0
    assert analysis.global_score == pytest.approx(0.2 / 9)
    assert overlay.size == image.size
    assert overlay.getpixel((224, 224)) != image.getpixel((224, 224))


def test_window_analysis_reports_real_progress() -> None:
    calls: list[tuple[int, int]] = []

    analysis, _ = analyze_centered_windows(
        _BrightnessAdapter(),
        Image.new("RGB", (672, 672)),
        max_dimension=1536,
        progress_callback=lambda completed, total: calls.append((completed, total)),
    )

    assert len(analysis.windows) == 9
    assert calls == [(completed, 9) for completed in range(10)]


@pytest.mark.parametrize(
    ("global_score", "expected_points"),
    (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.6, 1.87),
        (0.7, 7.5),
        (0.8, 16.88),
        (0.9, 30.0),
        (0.95, 30.0),
        (1.0, 30.0),
    ),
)
def test_gapl_points_follow_the_conservative_curve(
    global_score: float,
    expected_points: float,
) -> None:
    assert gapl_frod_points(global_score) == expected_points


def test_gapl_global_finding_is_scored_from_all_windows() -> None:
    analysis, _ = analyze_centered_windows(
        _ConstantAdapter(0.95),
        Image.new("RGB", (672, 672)),
        max_dimension=1536,
    )

    finding = build_gapl_global_finding(
        analysis,
        artifacts=("forensics/ai/windows.json",),
    )

    assert finding.code == "AI_GAPL_GLOBAL_TRACE"
    assert finding.category == "synthetic_media"
    assert finding.risk_points == 30
    assert finding.evidence["global_index"] == pytest.approx(0.96)


def test_percentile_25_tolerates_a_small_number_of_weak_windows() -> None:
    scores = (0.13, 0.47, 0.65, 0.73, 0.86, 0.86, *([0.95] * 19))
    analysis, _ = analyze_centered_windows(
        _SequenceAdapter(scores),
        Image.new("RGB", (1120, 1120)),
        max_dimension=1536,
    )

    assert analysis.percentile_10_score == pytest.approx(0.682)
    assert analysis.percentile_25_score == pytest.approx(0.95)
    assert analysis.global_score == pytest.approx(0.944)
    assert gapl_frod_points(analysis.global_score) == 30


class _BrightnessAdapter:
    adapter_id = "brightness"
    method_family = "test"
    model_version = "test"

    def predict(self, image: Image.Image) -> AiImagePrediction:
        mean = ImageStat.Stat(image.convert("L")).mean[0]
        return AiImagePrediction(synthetic_score=mean / 255.0)


class _ConstantAdapter:
    adapter_id = "gapl_cvpr2026"
    method_family = "clip_prototype"
    model_version = "test"

    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, image: Image.Image) -> AiImagePrediction:
        return AiImagePrediction(synthetic_score=self.score)


class _SequenceAdapter:
    adapter_id = "gapl_cvpr2026"
    method_family = "clip_prototype"
    model_version = "test"

    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = iter(scores)

    def predict(self, image: Image.Image) -> AiImagePrediction:
        return AiImagePrediction(synthetic_score=next(self.scores))
