"""Render GLM-OCR layout regions on the analyzed document."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "text": (82, 197, 34),
    "table": (246, 130, 59),
    "formula": (22, 115, 249),
    "image": (245, 85, 168),
    "seal": (68, 68, 239),
    "title": (82, 197, 34),
    "paragraph": (82, 197, 34),
    "header": (82, 197, 34),
    "footer": (82, 197, 34),
    "figure": (245, 85, 168),
    "chart": (245, 85, 168),
    "caption": (82, 197, 34),
    "reference": (150, 150, 150),
    "default": (128, 128, 128),
}


def draw_layout_bboxes(
    page_image: Image.Image,
    ocr_json: Any,
    page_index: int,
) -> Image.Image:
    """Draw normalized GLM-OCR bounding boxes on one page."""

    image = np.asarray(page_image.convert("RGB"), dtype=np.uint8)
    rendered = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = rendered.shape[:2]
    page_data = _page_regions(ocr_json, page_index)

    for region in page_data:
        bbox = region.get("bbox_2d")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            continue
        try:
            normalized = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = (
            round(normalized[0] * width / 1000),
            round(normalized[1] * height / 1000),
            round(normalized[2] * width / 1000),
            round(normalized[3] * height / 1000),
        )
        x1, x2 = sorted((min(width - 1, max(0, x1)), min(width - 1, max(0, x2))))
        y1, y2 = sorted((min(height - 1, max(0, y1)), min(height - 1, max(0, y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(region.get("label", "default")).casefold()
        color = LABEL_COLORS.get(label, LABEL_COLORS["default"])
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)

        label_text = label[:32]
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        label_top = max(0, y1 - text_height - baseline - 4)
        label_right = min(width - 1, x1 + text_width + 6)
        cv2.rectangle(rendered, (x1, label_top), (label_right, y1), color, -1)
        cv2.putText(
            rendered,
            label_text,
            (x1 + 3, max(text_height, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))


def save_layout_visualizations(
    page_images: tuple[Path, ...],
    ocr_json: Any,
    output_dir: Path,
    artifact_root: Path,
) -> tuple[str, ...]:
    """Persist one explainable layout image for each available OCR page."""

    artifacts: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_index, page_path in enumerate(page_images):
        if not _page_regions(ocr_json, page_index):
            continue
        try:
            with Image.open(page_path) as source:
                visualization = draw_layout_bboxes(source, ocr_json, page_index)
            output_path = output_dir / f"page-{page_index + 1:03d}-layout.png"
            visualization.save(output_path, format="PNG")
            artifacts.append(output_path.relative_to(artifact_root).as_posix())
        except (OSError, ValueError):
            continue
    return tuple(artifacts)


def _page_regions(ocr_json: Any, page_index: int) -> list[dict[str, Any]]:
    if not isinstance(ocr_json, list) or not 0 <= page_index < len(ocr_json):
        return []
    page = ocr_json[page_index]
    if not isinstance(page, list):
        return []
    return [region for region in page if isinstance(region, dict)]
