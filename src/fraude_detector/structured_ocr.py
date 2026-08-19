"""Shared structured representation of GLM-OCR regions for local LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StructuredOcrRegion:
    """One OCR region with the layout context needed for semantic interpretation."""

    region_id: str
    page: int
    reading_order: int
    label: str
    native_label: str
    bbox_2d: tuple[float, float, float, float] | None
    content: str


def extract_structured_ocr_regions(payload: Any) -> tuple[StructuredOcrRegion, ...]:
    """Normalize supported GLM-OCR envelopes without discarding layout metadata."""

    if isinstance(payload, list):
        pages = payload
    elif isinstance(payload, dict) and isinstance(payload.get("pages"), list):
        pages = payload["pages"]
    else:
        return ()

    regions: list[StructuredOcrRegion] = []
    for page_index, page in enumerate(pages, start=1):
        if isinstance(page, list):
            page_regions = page
        elif isinstance(page, dict):
            page_regions = page.get("regions", [])
        else:
            page_regions = []
        if not isinstance(page_regions, list):
            continue
        for region_index, region in enumerate(page_regions):
            if not isinstance(region, dict):
                continue
            content = str(region.get("content", "")).strip()
            label = str(region.get("label", "unknown")).strip() or "unknown"
            native_label = str(region.get("native_label", label)).strip() or label
            reading_order = _reading_order(region.get("index"), region_index)
            regions.append(
                StructuredOcrRegion(
                    region_id=f"p{page_index:03d}-r{region_index:03d}",
                    page=page_index,
                    reading_order=reading_order,
                    label=label[:40],
                    native_label=native_label[:40],
                    bbox_2d=_bbox_2d(region.get("bbox_2d")),
                    content=content or "[aucun contenu textuel reconnu]",
                )
            )
    return tuple(regions)


def format_structured_ocr_region(region: StructuredOcrRegion) -> str:
    """Serialize one region as compact, prompt-safe contextual markup."""

    bbox = (
        ",".join(_format_coordinate(value) for value in region.bbox_2d)
        if region.bbox_2d is not None
        else "unknown"
    )
    return (
        f'<region id="{region.region_id}" page="{region.page}" '
        f'order="{region.reading_order}" label="{region.label}" '
        f'native_label="{region.native_label}" bbox_2d="{bbox}">\n'
        f"{region.content}\n</region>"
    )


def _reading_order(value: object, fallback: int) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    return fallback


def _bbox_2d(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        coordinates = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if any(not 0 <= item <= 1000 for item in coordinates):
        return None
    x0, y0, x1, y1 = coordinates
    if x1 <= x0 or y1 <= y0:
        return None
    return coordinates


def _format_coordinate(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")
