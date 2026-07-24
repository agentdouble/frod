from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import fraude_detector.gapl as gapl
from fraude_detector.ai_images import AiImagePrediction
from fraude_detector.gapl import (
    GaplAdapter,
    GaplWeightsError,
    create_gapl_adapter,
)


def test_module_has_no_eager_ml_imports() -> None:
    source = Path(gapl.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert {"torch", "transformers", "peft"}.isdisjoint(imported_roots)


def test_factory_requires_existing_local_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(GaplWeightsError, match="introuvable"):
        create_gapl_adapter(weights_path=tmp_path / "checkpoint.pt")


def test_factory_rejects_unexpected_checksum(tmp_path: Path) -> None:
    weights = tmp_path / "checkpoint.pt"
    weights.write_bytes(b"not the official checkpoint")

    with pytest.raises(GaplWeightsError, match="checksum"):
        create_gapl_adapter(weights_path=weights)


def test_factory_loads_only_local_weight_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "checkpoint.pt"
    weights.write_bytes(b"official")
    torch = _FakeTorch()
    runtime = SimpleNamespace(torch=torch)
    model = _FakeModel()

    monkeypatch.setattr(gapl, "_sha256", lambda path: gapl.GAPL_CHECKPOINT_SHA256)
    monkeypatch.setattr(gapl, "_load_runtime", lambda: runtime)
    monkeypatch.setattr(
        gapl,
        "_load_checkpoint",
        lambda path, torch_module: {
            "model": {"weight": "tensor"},
            "prototype": "prototype",
        },
    )
    monkeypatch.setattr(gapl, "_build_model", lambda loaded_runtime, device: model)

    adapter = create_gapl_adapter(weights_path=weights)

    assert adapter.adapter_id == "gapl_cvpr2026"
    assert adapter.method_family == "prototype_guided_clip_classifier"
    assert adapter.model_version.startswith("cvpr2026:sha256-")
    assert model.loaded_state == {"weight": "tensor"}
    assert model.prototype == "prototype"
    assert model.device == "cpu"
    assert model.training is False


def test_checkpoint_loading_enforces_weights_only(tmp_path: Path) -> None:
    weights = tmp_path / "checkpoint.pt"
    weights.write_bytes(b"checkpoint")
    torch = _RecordingTorch()

    result = gapl._load_checkpoint(weights, torch)

    assert result == {"model": {}, "prototype": "prototype"}
    assert torch.call == {
        "path": weights,
        "map_location": "cpu",
        "weights_only": True,
    }


def test_center_crop_matches_official_input_size() -> None:
    image = Image.new("RGB", (640, 480), color=(12, 34, 56))

    result = gapl._center_crop_with_padding(image, 224)

    assert result.size == (224, 224)
    assert result.getpixel((0, 0)) == (12, 34, 56)


def test_small_image_is_zero_padded_before_crop() -> None:
    image = Image.new("RGB", (20, 10), color=(255, 255, 255))

    result = gapl._center_crop_with_padding(image, 224)

    assert result.size == (224, 224)
    assert result.getpixel((0, 0)) == (0, 0, 0)
    assert result.getpixel((112, 112)) == (255, 255, 255)


def test_predict_returns_passive_score(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _FakeTorch()
    adapter = GaplAdapter(
        model=_FakePredictionModel(),
        torch_module=torch,
        device="cpu",
        model_version="test",
    )
    batch = _FakeBatch()
    monkeypatch.setattr(gapl, "_preprocess_image", lambda image, torch_module: batch)

    prediction = adapter.predict(Image.new("RGB", (224, 224)))

    assert isinstance(prediction, AiImagePrediction)
    assert prediction.synthetic_score == pytest.approx(0.82)
    assert prediction.details["official_threshold"] == 0.5
    assert batch.device == "cpu"


class _FakeBatch:
    device: str | None = None

    def to(self, device: str) -> _FakeBatch:
        self.device = device
        return self


class _FakeScalar:
    def item(self) -> float:
        return 0.82


class _FakeOutput:
    def reshape(self, *shape: int) -> _FakeOutput:
        return self

    def __getitem__(self, index: int) -> _FakeOutput:
        return self


class _FakeTorch:
    def inference_mode(self) -> nullcontext[None]:
        return nullcontext()

    def sigmoid(self, value: object) -> _FakeScalar:
        return _FakeScalar()


class _RecordingTorch:
    call: dict[str, object] | None = None

    def load(self, path: Path, **kwargs: object) -> dict[str, object]:
        self.call = {"path": path, **kwargs}
        return {"model": {}, "prototype": "prototype"}


class _FakePredictionModel:
    def __call__(self, batch: _FakeBatch) -> _FakeOutput:
        return _FakeOutput()


class _FakeModel:
    loaded_state: dict[str, object] | None = None
    prototype: object | None = None
    device: str | None = None
    training = True

    def load_state_dict(
        self,
        state_dict: dict[str, object],
        *,
        strict: bool,
    ) -> SimpleNamespace:
        assert strict is False
        self.loaded_state = state_dict
        return SimpleNamespace(missing_keys=(), unexpected_keys=())

    def load_prototype(self, prototype: object) -> None:
        self.prototype = prototype

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.training = False
