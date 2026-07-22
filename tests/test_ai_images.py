from __future__ import annotations

from collections.abc import Callable

import pytest
from PIL import Image

from fraude_detector.ai_images import (
    AiImagePrediction,
    evaluate_ai_image_adapter,
)


class FakeAdapter:
    adapter_id = "fake"
    model_version = "test-1"
    method_family = "frequency"

    def __init__(self, predict: Callable[[Image.Image], AiImagePrediction]) -> None:
        self._predict = predict

    def predict(self, image: Image.Image) -> AiImagePrediction:
        return self._predict(image)


def test_stable_scores_complete_evaluation() -> None:
    adapter = FakeAdapter(lambda image: AiImagePrediction(0.91))

    evaluation = evaluate_ai_image_adapter(
        adapter,
        Image.new("RGB", (2000, 1000), "navy"),
        max_dimension=1000,
        stability_max_delta=0.15,
    )

    assert evaluation.status == "completed"
    assert evaluation.stable is True
    assert evaluation.max_score_delta == pytest.approx(0)
    assert evaluation.original_score == pytest.approx(0.91)
    assert [score.name for score in evaluation.variant_scores] == [
        "original",
        "jpeg_quality_95",
        "jpeg_quality_75",
        "resize_75_percent",
    ]


def test_score_sensitive_to_resize_is_marked_unstable() -> None:
    adapter = FakeAdapter(lambda image: AiImagePrediction(0.9 if image.width >= 1000 else 0.2))

    evaluation = evaluate_ai_image_adapter(
        adapter,
        Image.new("RGB", (1000, 800), "white"),
        max_dimension=1200,
        stability_max_delta=0.15,
    )

    assert evaluation.status == "completed"
    assert evaluation.stable is False
    assert evaluation.max_score_delta == pytest.approx(0.7)


def test_out_of_domain_prediction_abstains() -> None:
    adapter = FakeAdapter(
        lambda image: AiImagePrediction(
            0.0,
            in_domain=False,
            out_of_domain_reason="text_dense_document",
        )
    )

    evaluation = evaluate_ai_image_adapter(
        adapter,
        Image.new("RGB", (800, 600), "white"),
        max_dimension=1000,
        stability_max_delta=0.15,
    )

    assert evaluation.status == "out_of_domain"
    assert evaluation.out_of_domain_reason == "text_dense_document"


def test_adapter_failure_is_captured() -> None:
    def fail(image: Image.Image) -> AiImagePrediction:
        raise RuntimeError("weights unavailable")

    evaluation = evaluate_ai_image_adapter(
        FakeAdapter(fail),
        Image.new("RGB", (800, 600), "white"),
        max_dimension=1000,
        stability_max_delta=0.15,
    )

    assert evaluation.status == "error"
    assert evaluation.error == "RuntimeError: weights unavailable"


def test_prediction_validates_score_range() -> None:
    with pytest.raises(ValueError, match="synthetic_score"):
        AiImagePrediction(1.1)
