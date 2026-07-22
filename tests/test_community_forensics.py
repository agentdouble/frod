from __future__ import annotations

import ast
import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import fraude_detector.community_forensics as community_forensics
from fraude_detector.ai_images import AiImagePrediction
from fraude_detector.community_forensics import (
    CommunityForensicsAdapter,
    CommunityForensicsDependencyError,
    CommunityForensicsWeightsError,
    create_community_forensics_adapter,
    get_community_forensics_spec,
)


def test_module_has_no_eager_optional_imports() -> None:
    source = Path(community_forensics.__file__).read_text(encoding="utf-8")
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

    assert {"torch", "timm", "safetensors", "huggingface_hub"}.isdisjoint(imported_roots)


def test_factory_requires_explicit_local_path_or_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_import(name: str) -> object:
        raise AssertionError(f"optional import attempted: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)

    with pytest.raises(CommunityForensicsWeightsError, match="allow_hf_download=True"):
        create_community_forensics_adapter()


def test_local_weights_never_call_hugging_face(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"local weights")
    runtime, model = _fake_runtime()

    monkeypatch.setattr(community_forensics, "_load_runtime", lambda: runtime)
    monkeypatch.setattr(
        community_forensics,
        "_download_hf_weights",
        lambda **kwargs: pytest.fail(f"unexpected download: {kwargs}"),
    )

    adapter = create_community_forensics_adapter(weights_path=weights)

    assert adapter.adapter_id == "community_forensics"
    assert adapter.method_family == "supervised_vit_classifier"
    assert adapter.model_version.startswith("384:sha256-")
    assert callable(adapter.predict)
    assert model.pretrained is False
    assert model.num_classes == 1
    assert model.strict is True
    assert model.device == "cpu"
    assert model.training is False


def test_hugging_face_download_occurs_only_after_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"downloaded weights")
    runtime, _ = _fake_runtime()
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> Path:
        calls.append(kwargs)
        return weights

    monkeypatch.setattr(community_forensics, "_download_hf_weights", fake_download)
    monkeypatch.setattr(community_forensics, "_load_runtime", lambda: runtime)

    adapter = create_community_forensics_adapter(allow_hf_download=True)

    spec = get_community_forensics_spec("384")
    assert calls == [
        {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "cache_dir": None,
        }
    ]
    assert adapter.model_version == f"384:{spec.repo_id}@{spec.revision}"


def test_hugging_face_failure_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: (_ for _ in ()).throw(OSError("offline"))
    )
    monkeypatch.setattr(
        community_forensics,
        "_import_optional",
        lambda module_name, guidance: hub,
    )

    with pytest.raises(CommunityForensicsWeightsError, match="telechargement"):
        community_forensics._download_hf_weights(
            repo_id="OwensLab/commfor-model-384",
            revision="a" * 40,
            cache_dir=None,
        )


def test_missing_optional_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def fake_import(name: str) -> object:
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(CommunityForensicsDependencyError, match="torch"):
        community_forensics._load_runtime()


@pytest.mark.parametrize(
    ("variant", "resize_size", "input_size", "architecture"),
    [
        ("224", 256, 224, "vit_small_patch16_224.augreg_in21k_ft_in1k"),
        ("384", 440, 384, "vit_small_patch16_384.augreg_in21k_ft_in1k"),
    ],
)
def test_official_model_specs(
    variant: community_forensics.CommunityForensicsVariant,
    resize_size: int,
    input_size: int,
    architecture: str,
) -> None:
    spec = get_community_forensics_spec(variant)

    assert spec.resize_size == resize_size
    assert spec.input_size == input_size
    assert spec.architecture == architecture
    assert len(spec.revision) == 40


def test_official_eval_resize_and_center_crop_is_deterministic() -> None:
    image = Image.new("L", (800, 400), color=127)

    result = community_forensics._resize_and_center_crop(
        image,
        resize_size=440,
        crop_size=384,
    )

    assert result.mode == "RGB"
    assert result.size == (384, 384)


def test_state_dict_keys_from_official_wrapper_are_normalized() -> None:
    state = {
        "vit.head.weight": "head",
        "module._orig_mod.vit.patch_embed.proj.weight": "patch",
    }

    assert community_forensics._normalize_state_dict_keys(state) == {
        "head.weight": "head",
        "patch_embed.proj.weight": "patch",
    }


def test_suffixless_hugging_face_cache_blob_loads_as_safetensors(tmp_path: Path) -> None:
    blob = tmp_path / "d34db33f"
    blob.write_bytes(b"content-addressed cache blob")
    runtime, _ = _fake_runtime()

    state = community_forensics._load_state_dict(blob, runtime)

    assert set(state) == {"vit.head.weight"}


def test_corrupt_safetensors_error_is_normalized(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"corrupt")
    runtime, _ = _fake_runtime()
    runtime.safetensors_torch.load_file = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("invalid header")
    )

    with pytest.raises(CommunityForensicsWeightsError, match="corrompu"):
        community_forensics._load_state_dict(weights, runtime)


def test_predict_returns_contract_without_claiming_fraud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _FakeTorch()
    model = _FakeModel()
    spec = get_community_forensics_spec("384")
    adapter = CommunityForensicsAdapter(
        model=model,
        torch_module=torch,
        spec=spec,
        device="cpu",
        model_version="384:test",
    )
    batch = _FakeBatch()
    monkeypatch.setattr(community_forensics, "_preprocess_image", lambda *args: batch)

    prediction = adapter.predict(Image.new("RGB", (640, 480)))

    assert isinstance(prediction, AiImagePrediction)
    assert prediction.synthetic_score == pytest.approx(0.8)
    assert prediction.details["input_size"] == 384
    assert "fraud" not in prediction.details
    assert batch.device == "cpu"


class _FakeBatch:
    device: str | None = None

    def to(self, device: str) -> _FakeBatch:
        self.device = device
        return self


class _FakeScalar:
    def item(self) -> float:
        return 0.8


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


class _FakeModel:
    pretrained: bool | None = None
    num_classes: int | None = None
    strict: bool | None = None
    device: str | None = None
    training = True

    def load_state_dict(self, state: dict[str, object], *, strict: bool) -> None:
        self.strict = strict

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.training = False

    def __call__(self, batch: object) -> _FakeOutput:
        return _FakeOutput()


class _FakeTimm:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model

    def create_model(
        self,
        architecture: str,
        *,
        pretrained: bool,
        num_classes: int,
    ) -> _FakeModel:
        self.model.pretrained = pretrained
        self.model.num_classes = num_classes
        return self.model


class _FakeSafeTensors:
    def load_file(self, path: str, *, device: str) -> dict[str, object]:
        return {"vit.head.weight": object()}


def _fake_runtime() -> tuple[SimpleNamespace, _FakeModel]:
    model = _FakeModel()
    runtime = SimpleNamespace(
        torch=_FakeTorch(),
        timm=_FakeTimm(model),
        safetensors_torch=_FakeSafeTensors(),
    )
    return runtime, model
