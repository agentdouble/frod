"""Optional Community Forensics adapter with explicit weight loading.

The official Community Forensics model is a supervised ViT classifier.  This
module deliberately keeps its machine-learning dependencies lazy: importing it
does not import PyTorch, timm, safetensors or huggingface_hub, and never performs
network access.  Call :func:`create_community_forensics_adapter` explicitly to
load a local checkpoint or to opt into a Hugging Face download.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from fraude_detector.ai_images import AiImagePrediction

CommunityForensicsVariant = Literal["224", "384"]


class CommunityForensicsError(RuntimeError):
    """Base error for optional Community Forensics support."""


class CommunityForensicsDependencyError(CommunityForensicsError):
    """Raised when explicitly loading the adapter without its optional runtime."""


class CommunityForensicsWeightsError(CommunityForensicsError):
    """Raised when weights are absent, unsupported or incompatible."""


@dataclass(frozen=True, slots=True)
class CommunityForensicsSpec:
    """Pinned architecture and evaluation preprocessing for one official model."""

    variant: CommunityForensicsVariant
    input_size: int
    resize_size: int
    architecture: str
    repo_id: str
    revision: str


SPECS: dict[CommunityForensicsVariant, CommunityForensicsSpec] = {
    "224": CommunityForensicsSpec(
        variant="224",
        input_size=224,
        resize_size=256,
        architecture="vit_small_patch16_224.augreg_in21k_ft_in1k",
        repo_id="OwensLab/commfor-model-224",
        revision="26afc31e6b40c312c3fd42c05a758be62446215b",
    ),
    "384": CommunityForensicsSpec(
        variant="384",
        input_size=384,
        resize_size=440,
        architecture="vit_small_patch16_384.augreg_in21k_ft_in1k",
        repo_id="OwensLab/commfor-model-384",
        revision="6076002bf0d9dd37537f965ee2f06f826c333b61",
    ),
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_HF_WEIGHTS_FILENAME = "model.safetensors"


@dataclass(frozen=True, slots=True)
class _Runtime:
    torch: Any
    timm: Any
    safetensors_torch: Any


class CommunityForensicsAdapter:
    """Loaded Community Forensics model implementing ``AiImageModelAdapter``."""

    adapter_id = "community_forensics"
    method_family = "supervised_vit_classifier"

    def __init__(
        self,
        *,
        model: Any,
        torch_module: Any,
        spec: CommunityForensicsSpec,
        device: str,
        model_version: str,
    ) -> None:
        self._model = model
        self._torch = torch_module
        self._spec = spec
        self._device = device
        self.model_version = model_version

    def predict(self, image: Image.Image) -> AiImagePrediction:
        """Return the sigmoid output; it is not a probability of document fraud."""

        batch = _preprocess_image(image, self._spec, self._torch).to(self._device)
        with self._torch.inference_mode():
            output = self._model(batch)
            logit = output.reshape(-1)[0]
            score = float(self._torch.sigmoid(logit).item())
        return AiImagePrediction(
            synthetic_score=min(1.0, max(0.0, score)),
            details={
                "architecture": self._spec.architecture,
                "input_size": self._spec.input_size,
                "preprocessing": "official_eval_resize_crop_imagenet_v1",
                "source": "Community Forensics (CVPR 2025)",
                "domain_gate": "external_geometry_route_only",
            },
        )


def create_community_forensics_adapter(
    *,
    weights_path: str | Path | None = None,
    variant: CommunityForensicsVariant = "384",
    device: str = "cpu",
    allow_hf_download: bool = False,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> CommunityForensicsAdapter:
    """Explicitly load Community Forensics from local or official HF weights.

    Passing ``weights_path`` never contacts Hugging Face.  Without a local path,
    callers must set ``allow_hf_download=True``; this explicit opt-in is the only
    branch that imports ``huggingface_hub`` or performs network-capable work.
    """

    spec = get_community_forensics_spec(variant)
    resolved_path, source_version = _resolve_weights(
        weights_path=weights_path,
        spec=spec,
        allow_hf_download=allow_hf_download,
        cache_dir=cache_dir,
        revision=revision,
    )
    runtime = _load_runtime()
    state_dict = _load_state_dict(resolved_path, runtime)
    model = runtime.timm.create_model(
        spec.architecture,
        pretrained=False,
        num_classes=1,
    )
    normalized_state = _normalize_state_dict_keys(state_dict)
    try:
        model.load_state_dict(normalized_state, strict=True)
    except Exception as error:
        raise CommunityForensicsWeightsError(
            "Le checkpoint Community Forensics ne correspond pas a "
            f"l'architecture {spec.architecture}."
        ) from error
    try:
        model = model.to(device)
        model.eval()
    except Exception as error:
        raise CommunityForensicsError(
            f"Impossible de charger Community Forensics sur le peripherique {device}."
        ) from error
    return CommunityForensicsAdapter(
        model=model,
        torch_module=runtime.torch,
        spec=spec,
        device=device,
        model_version=f"{variant}:{source_version}",
    )


def get_community_forensics_spec(
    variant: CommunityForensicsVariant,
) -> CommunityForensicsSpec:
    """Return a supported, immutable official model specification."""

    try:
        return SPECS[variant]
    except KeyError as error:
        raise ValueError("variant must be '224' or '384'") from error


def _resolve_weights(
    *,
    weights_path: str | Path | None,
    spec: CommunityForensicsSpec,
    allow_hf_download: bool,
    cache_dir: str | Path | None,
    revision: str | None,
) -> tuple[Path, str]:
    if weights_path is not None:
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise CommunityForensicsWeightsError(
                f"Checkpoint Community Forensics introuvable: {path}"
            )
        try:
            digest = _sha256(path)
        except OSError as error:
            raise CommunityForensicsWeightsError(
                f"Checkpoint Community Forensics illisible: {path}"
            ) from error
        return path, f"sha256-{digest[:16]}"

    if not allow_hf_download:
        raise CommunityForensicsWeightsError(
            "Aucun checkpoint local fourni. Passez weights_path, ou activez "
            "explicitement allow_hf_download=True."
        )

    selected_revision = revision or spec.revision
    path = _download_hf_weights(
        repo_id=spec.repo_id,
        revision=selected_revision,
        cache_dir=cache_dir,
    )
    return path, f"{spec.repo_id}@{selected_revision}"


def _download_hf_weights(
    *,
    repo_id: str,
    revision: str,
    cache_dir: str | Path | None,
) -> Path:
    hub = _import_optional(
        "huggingface_hub",
        "Installez huggingface-hub pour telecharger le checkpoint officiel.",
    )
    try:
        downloaded = hub.hf_hub_download(
            repo_id=repo_id,
            filename=_HF_WEIGHTS_FILENAME,
            revision=revision,
            cache_dir=str(Path(cache_dir).expanduser()) if cache_dir is not None else None,
        )
    except Exception as error:
        raise CommunityForensicsWeightsError(
            "Echec du telechargement du checkpoint Community Forensics epingle."
        ) from error
    return Path(downloaded).resolve()


def _load_runtime() -> _Runtime:
    torch = _import_optional(
        "torch",
        "Installez torch, timm et safetensors pour utiliser Community Forensics.",
    )
    timm = _import_optional(
        "timm",
        "Installez torch, timm et safetensors pour utiliser Community Forensics.",
    )
    safetensors_torch = _import_optional(
        "safetensors.torch",
        "Installez safetensors pour lire les poids officiels.",
    )
    return _Runtime(torch=torch, timm=timm, safetensors_torch=safetensors_torch)


def _import_optional(module_name: str, guidance: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise CommunityForensicsDependencyError(
            f"Dependance optionnelle absente: {module_name}. {guidance}"
        ) from error


def _load_state_dict(path: Path, runtime: _Runtime) -> dict[str, Any]:
    try:
        # Hugging Face's cache resolves model.safetensors to a content-addressed
        # blob whose filesystem path has no suffix.
        if path.suffix in {"", ".safetensors"}:
            state_dict = runtime.safetensors_torch.load_file(str(path), device="cpu")
        elif path.suffix in {".pt", ".pth"}:
            checkpoint = runtime.torch.load(path, map_location="cpu", weights_only=True)
            state_dict = _unwrap_checkpoint(checkpoint)
        else:
            raise CommunityForensicsWeightsError(
                "Format de checkpoint non supporte; utilisez .safetensors, .pt ou .pth."
            )
    except CommunityForensicsWeightsError:
        raise
    except TypeError as error:
        if path.suffix in {".pt", ".pth"}:
            raise CommunityForensicsDependencyError(
                "Cette version de torch ne supporte pas le chargement sur "
                "weights_only=True; mettez torch a jour."
            ) from error
        raise CommunityForensicsWeightsError(
            f"Checkpoint Community Forensics illisible: {path}"
        ) from error
    except Exception as error:
        raise CommunityForensicsWeightsError(
            f"Checkpoint Community Forensics illisible ou corrompu: {path}"
        ) from error
    if not isinstance(state_dict, dict) or not state_dict:
        raise CommunityForensicsWeightsError("Le checkpoint ne contient aucun poids.")
    return state_dict


def _unwrap_checkpoint(checkpoint: Any) -> Any:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            return nested
    return checkpoint


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove wrappers used by DDP, torch.compile and the official hub class."""

    normalized: dict[str, Any] = {}
    prefixes = ("module.", "_orig_mod.", "vit.")
    for original_key, value in state_dict.items():
        key = str(original_key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key.removeprefix(prefix)
                    changed = True
        normalized[key] = value
    return normalized


def _preprocess_image(image: Image.Image, spec: CommunityForensicsSpec, torch: Any) -> Any:
    prepared = _resize_and_center_crop(image, spec.resize_size, spec.input_size)
    pixels = np.array(prepared, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def _resize_and_center_crop(
    image: Image.Image,
    resize_size: int,
    crop_size: int,
) -> Image.Image:
    """Match torchvision Resize(short edge) followed by CenterCrop."""

    if resize_size < crop_size or crop_size < 1:
        raise ValueError("resize_size must be greater than or equal to crop_size")
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if width <= height:
        resized_width = resize_size
        resized_height = int(resize_size * height / width)
    else:
        resized_height = resize_size
        resized_width = int(resize_size * width / height)
    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = int(round((resized_width - crop_size) / 2.0))
    top = int(round((resized_height - crop_size) / 2.0))
    return resized.crop((left, top, left + crop_size, top + crop_size))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
