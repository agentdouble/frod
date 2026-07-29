"""Short-lived inference worker for the vendored TruFor runtime."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


class _Config(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pixels", required=True, type=int)
    args = parser.parse_args()
    _run(
        Path(args.input),
        Path(args.weights),
        Path(args.output),
        max_pixels=args.max_pixels,
    )


def _run(source: Path, checkpoint_path: Path, output: Path, *, max_pixels: int) -> None:
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from torch.nn import functional as F

    runtime = Path(__file__).parent / "_vendor" / "trufor_runtime"
    sys.path.insert(0, str(runtime))
    from trufor_models.cmx.builder_np_conf import myEncoderDecoder

    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    config = _trufor_config()
    model = myEncoderDecoder(cfg=config)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint TruFor sans state_dict.")
    model.load_state_dict(state_dict)
    model.eval()

    with Image.open(source) as image_context:
        image = ImageOps.exif_transpose(image_context).convert("RGB")
        original_size = image.size
        image = _resize_to_budget(image, max_pixels=max_pixels)
        analyzed_size = image.size
        pixels = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0) / 256.0

    with torch.inference_mode():
        prediction, confidence, detection, _ = model(tensor)
        anomaly_map = F.softmax(prediction.squeeze(0), dim=0)[1].cpu().numpy()
        confidence_map = torch.sigmoid(confidence.squeeze(0))[0].cpu().numpy()
        score = torch.sigmoid(detection).item()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        map=anomaly_map.astype(np.float32),
        conf=confidence_map.astype(np.float32),
        score=np.asarray([score], dtype=np.float32),
        original_size=np.asarray(original_size, dtype=np.int32),
        analyzed_size=np.asarray(analyzed_size, dtype=np.int32),
    )


def _resize_to_budget(image: Any, *, max_pixels: int) -> Any:
    from PIL import Image

    pixel_count = image.width * image.height
    if pixel_count <= max_pixels:
        return image.copy()
    scale = (max_pixels / pixel_count) ** 0.5
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _trufor_config() -> _Config:
    return _Config(
        MODEL=_Config(
            PRETRAINED="",
            MODS=("RGB", "NP++"),
            EXTRA=_Config(
                BACKBONE="mit_b2",
                DECODER="MLPDecoder",
                DECODER_EMBED_DIM=512,
                PREPRC="imagenet",
                BN_EPS=0.001,
                BN_MOMENTUM=0.1,
                DETECTION="confpool",
                CONF=True,
            ),
        ),
        DATASET=_Config(NUM_CLASSES=2),
    )


if __name__ == "__main__":
    main()
