"""Optional local adapter for the official GAPL CVPR 2026 checkpoint."""

from __future__ import annotations

import hashlib
import importlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fraude_detector.ai_images import AiImagePrediction

GAPL_CHECKPOINT_SHA256 = "ffbcb5eb526f0df0fd197d7266bdd0325b66813e95010f1285685acf2d267235"
GAPL_DEFAULT_CHECKPOINT = Path("models/gapl/checkpoint.pt")
GAPL_INPUT_SIZE = 224
GAPL_OFFICIAL_THRESHOLD = 0.5

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GaplError(RuntimeError):
    """Base error for optional GAPL support."""


class GaplDependencyError(GaplError):
    """Raised when the optional local runtime is incomplete."""


class GaplWeightsError(GaplError):
    """Raised when the official checkpoint is missing or incompatible."""


@dataclass(frozen=True, slots=True)
class _Runtime:
    torch: Any
    transformers: Any
    peft: Any


class GaplAdapter:
    """GAPL implementation of the passive image-model adapter contract."""

    adapter_id = "gapl_cvpr2026"
    method_family = "prototype_guided_clip_classifier"

    def __init__(
        self,
        *,
        model: Any,
        torch_module: Any,
        device: str,
        model_version: str,
    ) -> None:
        self._model = model
        self._torch = torch_module
        self._device = device
        self.model_version = model_version

    def predict(self, image: Image.Image) -> AiImagePrediction:
        """Return GAPL's sigmoid score, not a probability of document fraud."""

        batch = _preprocess_image(image, self._torch).to(self._device)
        with self._torch.inference_mode():
            output = self._model(batch)
            score = float(self._torch.sigmoid(output.reshape(-1)[0]).item())
        return AiImagePrediction(
            synthetic_score=min(1.0, max(0.0, score)),
            details={
                "architecture": "CLIP-ViT-L/14 + GAPL prototypes",
                "input_size": GAPL_INPUT_SIZE,
                "official_threshold": GAPL_OFFICIAL_THRESHOLD,
                "preprocessing": "official_center_crop_imagenet",
                "source": "GAPL (CVPR 2026)",
            },
        )


def create_gapl_adapter(
    *,
    weights_path: str | Path,
    device: str = "cpu",
    expected_sha256: str = GAPL_CHECKPOINT_SHA256,
) -> GaplAdapter:
    """Load the official checkpoint from disk without network access."""

    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise GaplWeightsError(f"Checkpoint GAPL introuvable: {path}")
    digest = _sha256(path)
    if expected_sha256 and digest.casefold() != expected_sha256.casefold():
        raise GaplWeightsError(
            "Le checksum du checkpoint GAPL ne correspond pas au fichier officiel."
        )

    runtime = _load_runtime()
    checkpoint = _load_checkpoint(path, runtime.torch)
    model = _build_model(runtime, device)
    state_dict = checkpoint.get("model")
    prototype = checkpoint.get("prototype")
    if not isinstance(state_dict, dict) or prototype is None:
        raise GaplWeightsError("Le checkpoint GAPL ne contient pas les poids attendus.")

    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = tuple(incompatible.missing_keys)
        unexpected = tuple(incompatible.unexpected_keys)
        if missing or unexpected:
            raise GaplWeightsError(
                "Le checkpoint GAPL est incompatible "
                f"(manquants={len(missing)}, inattendus={len(unexpected)})."
            )
        model.load_prototype(prototype)
        model = model.to(device)
        model.eval()
    except GaplWeightsError:
        raise
    except Exception as error:
        raise GaplWeightsError("Impossible d'initialiser le checkpoint GAPL.") from error

    return GaplAdapter(
        model=model,
        torch_module=runtime.torch,
        device=device,
        model_version=f"cvpr2026:sha256-{digest[:16]}",
    )


def best_available_device() -> str:
    """Select CUDA only when the optional runtime exposes it."""

    with suppress(ImportError):
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def _load_runtime() -> _Runtime:
    guidance = "Installez l'extra ai pour utiliser GAPL."
    return _Runtime(
        torch=_import_optional("torch", guidance),
        transformers=_import_optional("transformers", guidance),
        peft=_import_optional("peft", guidance),
    )


def _import_optional(module_name: str, guidance: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise GaplDependencyError(
            f"Dependance optionnelle absente: {module_name}. {guidance}"
        ) from error


def _load_checkpoint(path: Path, torch: Any) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise GaplDependencyError(
            "La version de torch doit supporter weights_only=True."
        ) from error
    except Exception as error:
        raise GaplWeightsError("Checkpoint GAPL illisible ou corrompu.") from error
    if not isinstance(checkpoint, dict):
        raise GaplWeightsError("Le checkpoint GAPL n'est pas un dictionnaire de poids.")
    return checkpoint


def _build_model(runtime: _Runtime, device: str) -> Any:
    torch = runtime.torch
    transformers = runtime.transformers
    peft = runtime.peft

    config = transformers.CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=GAPL_INPUT_SIZE,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
    )
    vision = transformers.CLIPVisionModel(config)
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=("q_proj", "k_proj", "v_proj"),
    )
    peft_vision = peft.get_peft_model(vision, lora_config)
    feature_extractor = peft_vision.vision_model

    class _GaplNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extractor = feature_extractor
            self.cross_attention = torch.nn.MultiheadAttention(
                embed_dim=128,
                num_heads=4,
                batch_first=True,
            ).to(device)
            self.foren_proj = torch.nn.Linear(1024, 128, bias=False)
            self.fc = torch.nn.Linear(128, 1, bias=False).to(device)
            self.proVec = None

        def forward(self, pixels: Any) -> Any:
            features = self.feature_extractor(pixels)["pooler_output"]
            features = self.foren_proj(features)
            features = torch.nn.functional.normalize(features, dim=1)
            prototypes = self.proVec.unsqueeze(0).expand(features.shape[0], -1, -1)
            attended, _ = self.cross_attention(
                query=features.unsqueeze(1),
                key=prototypes,
                value=prototypes,
            )
            return self.fc(attended.squeeze(1))

        def load_prototype(self, prototype: Any) -> None:
            self.proVec = prototype.detach().clone().to(device)

    return _GaplNetwork()


def _preprocess_image(image: Image.Image, torch: Any) -> Any:
    prepared = _center_crop_with_padding(image, GAPL_INPUT_SIZE)
    pixels = np.array(prepared, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def _center_crop_with_padding(image: Image.Image, size: int) -> Image.Image:
    """Match torchvision CenterCrop, including zero padding for small images."""

    if size < 1:
        raise ValueError("size must be positive")
    rgb = image.convert("RGB")
    canvas_width = max(size, rgb.width)
    canvas_height = max(size, rgb.height)
    if (canvas_width, canvas_height) != rgb.size:
        canvas = Image.new("RGB", (canvas_width, canvas_height))
        canvas.paste(
            rgb,
            ((canvas_width - rgb.width) // 2, (canvas_height - rgb.height) // 2),
        )
        rgb = canvas
    left = int(round((rgb.width - size) / 2.0))
    top = int(round((rgb.height - size) / 2.0))
    return rgb.crop((left, top, left + size, top + size))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
