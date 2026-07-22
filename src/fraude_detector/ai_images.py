"""Stable contracts and perturbation checks for passive AI-image models."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Literal, Protocol

from PIL import Image

EvaluationStatus = Literal["completed", "out_of_domain", "error"]


@dataclass(frozen=True, slots=True)
class AiImagePrediction:
    """One adapter output; the score is not a probability of document fraud."""

    synthetic_score: float
    in_domain: bool = True
    out_of_domain_reason: str | None = None
    heatmap: Image.Image | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.synthetic_score <= 1:
            raise ValueError("synthetic_score must be between 0 and 1")
        if self.in_domain and self.out_of_domain_reason is not None:
            raise ValueError("in-domain predictions cannot have an out-of-domain reason")
        if not self.in_domain and not self.out_of_domain_reason:
            raise ValueError("out-of-domain predictions require a reason")


class AiImageModelAdapter(Protocol):
    """Replaceable passive model; loading weights is the adapter's concern."""

    adapter_id: str
    model_version: str
    method_family: str

    def predict(self, image: Image.Image) -> AiImagePrediction:
        """Return a synthetic-media score for a detached RGB image."""
        ...


@dataclass(frozen=True, slots=True)
class VariantScore:
    name: str
    synthetic_score: float


@dataclass(frozen=True, slots=True)
class AiImageEvaluation:
    adapter_id: str
    model_version: str
    method_family: str
    status: EvaluationStatus
    variant_scores: tuple[VariantScore, ...] = ()
    stable: bool | None = None
    max_score_delta: float | None = None
    out_of_domain_reason: str | None = None
    error: str | None = None
    heatmap: Image.Image | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def original_score(self) -> float | None:
        return self.variant_scores[0].synthetic_score if self.variant_scores else None


def evaluate_ai_image_adapter(
    adapter: AiImageModelAdapter,
    image: Image.Image,
    *,
    max_dimension: int,
    stability_max_delta: float,
) -> AiImageEvaluation:
    """Run an adapter on controlled perturbations and report score stability."""

    identity = _adapter_identity(adapter)
    if max_dimension < 1:
        raise ValueError("max_dimension must be at least 1")
    if not 0 <= stability_max_delta <= 1:
        raise ValueError("stability_max_delta must be between 0 and 1")

    base = _bounded_rgb(image, max_dimension)
    variants = (
        ("original", base),
        ("jpeg_quality_95", _jpeg_roundtrip(base, quality=95)),
        ("jpeg_quality_75", _jpeg_roundtrip(base, quality=75)),
        ("resize_75_percent", _resize(base, 0.75)),
    )
    scores: list[VariantScore] = []
    original_prediction: AiImagePrediction | None = None
    try:
        for name, variant in variants:
            prediction = adapter.predict(variant.copy())
            if not isinstance(prediction, AiImagePrediction):
                raise TypeError("adapter.predict must return AiImagePrediction")
            if name == "original":
                original_prediction = prediction
            if not prediction.in_domain:
                return AiImageEvaluation(
                    **identity,
                    status="out_of_domain",
                    variant_scores=tuple(scores),
                    out_of_domain_reason=prediction.out_of_domain_reason,
                    details=_json_safe_details(prediction.details),
                )
            scores.append(VariantScore(name=name, synthetic_score=prediction.synthetic_score))
    except Exception as error:
        return AiImageEvaluation(
            **identity,
            status="error",
            variant_scores=tuple(scores),
            error=_safe_error(error),
        )

    values = [item.synthetic_score for item in scores]
    score_delta = max(values) - min(values)
    assert original_prediction is not None
    return AiImageEvaluation(
        **identity,
        status="completed",
        variant_scores=tuple(scores),
        stable=score_delta <= stability_max_delta,
        max_score_delta=score_delta,
        heatmap=original_prediction.heatmap,
        details=_json_safe_details(original_prediction.details),
    )


def consensus_evaluations(
    evaluations: tuple[AiImageEvaluation, ...],
    *,
    threshold: float,
) -> tuple[AiImageEvaluation, ...]:
    """Keep only high, stable and completed passive-model evaluations."""

    return tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.status == "completed"
        and evaluation.stable is True
        and evaluation.original_score is not None
        and evaluation.original_score >= threshold
    )


def one_evaluation_per_family(
    evaluations: tuple[AiImageEvaluation, ...],
) -> tuple[AiImageEvaluation, ...]:
    """Keep the strongest evaluation for each independent method family."""

    by_family: dict[str, AiImageEvaluation] = {}
    for evaluation in evaluations:
        current = by_family.get(evaluation.method_family)
        if current is None or (evaluation.original_score or 0) > (current.original_score or 0):
            by_family[evaluation.method_family] = evaluation
    return tuple(by_family[family] for family in sorted(by_family))


def _adapter_identity(adapter: AiImageModelAdapter) -> dict[str, str]:
    identity = {
        "adapter_id": str(getattr(adapter, "adapter_id", "")).strip(),
        "model_version": str(getattr(adapter, "model_version", "")).strip(),
        "method_family": str(getattr(adapter, "method_family", "")).strip(),
    }
    if not all(identity.values()):
        raise ValueError("adapter identity fields must not be empty")
    return identity


def _bounded_rgb(image: Image.Image, max_dimension: int) -> Image.Image:
    rgb = image.convert("RGB")
    if max(rgb.size) <= max_dimension:
        return rgb.copy()
    scale = max_dimension / max(rgb.size)
    size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    return rgb.resize(size, Image.Resampling.LANCZOS)


def _jpeg_roundtrip(image: Image.Image, *, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        recompressed.load()
        return recompressed.convert("RGB").copy()


def _resize(image: Image.Image, scale: float) -> Image.Image:
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _json_safe_details(details: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
    return safe


def _safe_error(error: Exception) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message[:240]}" if message else type(error).__name__
