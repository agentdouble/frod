"""Coordinate conversions shared by detectors and report rendering."""

from __future__ import annotations

import math

from fraude_detector.models import BoundingBox


def pdfium_bounds_to_top_left(
    bounds: tuple[float, float, float, float],
    page_height: float,
) -> BoundingBox | None:
    """Convert PDFium's bottom-left coordinates to report coordinates."""

    left, bottom, right, top = (float(value) for value in bounds)
    values = (left, bottom, right, top, page_height)
    if not all(math.isfinite(value) for value in values):
        return None
    x0, x1 = sorted((left, right))
    y0, y1 = sorted((page_height - top, page_height - bottom))
    if x1 <= x0 or y1 <= y0:
        return None
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def bbox_area(box: BoundingBox) -> float:
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)


def bbox_coverage(box: BoundingBox, page_width: float, page_height: float) -> float:
    page_area = max(1.0, page_width * page_height)
    return min(1.0, bbox_area(box) / page_area)


def merge_nearby_boxes(
    boxes: list[BoundingBox],
    horizontal_gap: float = 12.0,
    vertical_gap: float = 6.0,
) -> list[BoundingBox]:
    """Merge overlapping or nearby boxes without depending on text contents."""

    merged: list[BoundingBox] = []
    for candidate in sorted(boxes, key=lambda box: (box.y0, box.x0)):
        for index, current in enumerate(merged):
            if _boxes_are_near(current, candidate, horizontal_gap, vertical_gap):
                merged[index] = BoundingBox(
                    x0=min(current.x0, candidate.x0),
                    y0=min(current.y0, candidate.y0),
                    x1=max(current.x1, candidate.x1),
                    y1=max(current.y1, candidate.y1),
                )
                break
        else:
            merged.append(candidate)
    return merged


def _boxes_are_near(
    first: BoundingBox,
    second: BoundingBox,
    horizontal_gap: float,
    vertical_gap: float,
) -> bool:
    separated_horizontally = (
        first.x1 + horizontal_gap < second.x0 or second.x1 + horizontal_gap < first.x0
    )
    separated_vertically = (
        first.y1 + vertical_gap < second.y0 or second.y1 + vertical_gap < first.y0
    )
    return not separated_horizontally and not separated_vertically
