"""Conservative cross-page comparison of OCR-localized visual marks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from fraude_detector.models import LaboratoryCheck, LaboratoryObservation

_VISUAL_LABELS = {"figure", "image", "seal", "signature", "stamp"}


@dataclass(frozen=True, slots=True)
class _VisualCandidate:
    page: int
    label: str
    bbox: tuple[float, float, float, float]
    crop: Image.Image
    mask: np.ndarray
    aspect_ratio: float


def analyze_repeated_visual_regions(
    ocr_json: Any,
    page_images: tuple[Path, ...],
    output_dir: Path,
    *,
    minimum_pages: int = 3,
    minimum_similarity: float = 0.90,
) -> LaboratoryCheck:
    """Find highly similar lower-page marks without assigning a semantic identity."""

    if minimum_pages < 3:
        raise ValueError("minimum_pages must be at least 3")
    if not 0 < minimum_similarity <= 1:
        raise ValueError("minimum_similarity must be between 0 and 1")
    if len(page_images) < minimum_pages:
        return _not_applicable("Moins de trois pages sont disponibles pour la comparaison.")

    candidates = _extract_candidates(ocr_json, page_images)
    if len({candidate.page for candidate in candidates}) < minimum_pages:
        return _not_applicable(
            "Moins de trois pages contiennent une zone visuelle comparable dans le périmètre."
        )

    groups = _similar_groups(
        candidates,
        minimum_pages=minimum_pages,
        minimum_similarity=minimum_similarity,
    )
    if not groups:
        return LaboratoryCheck(
            code="ocr_visual_repetition",
            title="Éléments visuels répétés",
            purpose="Comparer les cachets, signatures ou images localisés sur plusieurs pages.",
            state="clear",
            summary="Aucune répétition visuelle forte n'a été trouvée sur au moins trois pages.",
            limitations=_limitations(),
        )

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    observations: list[LaboratoryObservation] = []
    for index, (group, similarities) in enumerate(groups, start=1):
        artifact_path = destination / f"repeated-visual-region-{index:02d}.png"
        _write_montage(group, similarities, artifact_path)
        pages = tuple(candidate.page for candidate in group)
        observations.append(
            LaboratoryObservation(
                code="OCR_REPEATED_VISUAL_REGION",
                title="Élément visuel très similaire sur plusieurs pages",
                summary=(
                    f"Une même forme visuelle apparaît sur les pages {', '.join(map(str, pages))}."
                ),
                state="detected",
                strength="weak",
                explanation=(
                    "Une signature ou un cachet copié peut produire ce motif, mais aussi un "
                    "pied de page ou un élément de formulaire répété. Le montage doit être "
                    "comparé au document avant toute conclusion."
                ),
                page=pages[0],
                evidence={
                    "pages": pages,
                    "labels": tuple(candidate.label for candidate in group),
                    "normalized_boxes": tuple(candidate.bbox for candidate in group),
                    "minimum_similarity": round(min(similarities), 4),
                    "scored_by_frod": False,
                },
                artifacts=(artifact_path.name,),
            )
        )

    return LaboratoryCheck(
        code="ocr_visual_repetition",
        title="Éléments visuels répétés",
        purpose="Comparer les cachets, signatures ou images localisés sur plusieurs pages.",
        state="detected",
        summary=f"{len(observations)} répétition(s) visuelle(s) à comparer manuellement.",
        observations=tuple(observations),
        limitations=_limitations(),
    )


def _extract_candidates(
    ocr_json: Any,
    page_images: tuple[Path, ...],
) -> tuple[_VisualCandidate, ...]:
    if not isinstance(ocr_json, list):
        return ()
    candidates: list[_VisualCandidate] = []
    for page_index, page_regions in enumerate(ocr_json[: len(page_images)]):
        if not isinstance(page_regions, list) or not page_images[page_index].is_file():
            continue
        try:
            with Image.open(page_images[page_index]) as source:
                page_image = source.convert("RGB")
        except OSError:
            continue
        for region in page_regions:
            candidate = _candidate_from_region(region, page_index + 1, page_image)
            if candidate is not None:
                candidates.append(candidate)
    return tuple(candidates)


def _candidate_from_region(
    region: Any,
    page: int,
    page_image: Image.Image,
) -> _VisualCandidate | None:
    if not isinstance(region, dict):
        return None
    label = str(region.get("label") or region.get("native_label") or "").casefold()
    if label not in _VISUAL_LABELS:
        return None
    raw_bbox = region.get("bbox_2d")
    if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0.0, x0), min(1000.0, x1)))
    y0, y1 = sorted((max(0.0, y0), min(1000.0, y1)))
    width = x1 - x0
    height = y1 - y0
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    if not (30 <= width <= 500 and 20 <= height <= 350):
        return None
    if center_x < 450 or center_y < 550:
        return None

    pixel_box = (
        round(x0 * page_image.width / 1000),
        round(y0 * page_image.height / 1000),
        round(x1 * page_image.width / 1000),
        round(y1 * page_image.height / 1000),
    )
    crop = page_image.crop(pixel_box)
    prepared = _prepare_mask(crop)
    if prepared is None:
        return None
    normalized_crop, mask = prepared
    return _VisualCandidate(
        page=page,
        label=label,
        bbox=(x0, y0, x1, y1),
        crop=normalized_crop,
        mask=mask,
        aspect_ratio=width / max(1.0, height),
    )


def _prepare_mask(crop: Image.Image) -> tuple[Image.Image, np.ndarray] | None:
    grayscale = np.asarray(ImageOps.grayscale(crop), dtype=np.uint8)
    contrast = float(np.percentile(grayscale, 95)) - float(np.percentile(grayscale, 5))
    if grayscale.size < 256 or contrast < 18:
        return None

    threshold = min(240.0, float(np.percentile(grayscale, 90)) - 12.0)
    foreground = grayscale < threshold
    coordinates = np.argwhere(foreground)
    if coordinates.size == 0:
        return None
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    if x1 - x0 < 8 or y1 - y0 < 6:
        return None

    trimmed = grayscale[y0:y1, x0:x1]
    resized = cv2.resize(trimmed, (96, 48), interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(resized, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coverage = float(mask.mean())
    if not 0.01 <= coverage <= 0.60:
        return None
    normalized_crop = Image.fromarray(trimmed).convert("RGB")
    return normalized_crop, mask.astype(bool)


def _similar_groups(
    candidates: tuple[_VisualCandidate, ...],
    *,
    minimum_pages: int,
    minimum_similarity: float,
) -> tuple[tuple[tuple[_VisualCandidate, ...], tuple[float, ...]], ...]:
    groups: list[tuple[tuple[_VisualCandidate, ...], tuple[float, ...]]] = []
    fingerprints: set[tuple[int, ...]] = set()
    for anchor in candidates:
        matches = [anchor]
        similarities = [1.0]
        pages = sorted(
            {candidate.page for candidate in candidates if candidate.page != anchor.page}
        )
        for page in pages:
            options = [candidate for candidate in candidates if candidate.page == page]
            scored = [
                (_visual_similarity(anchor, candidate), candidate)
                for candidate in options
                if _compatible_shape(anchor, candidate)
            ]
            if not scored:
                continue
            similarity, candidate = max(scored, key=lambda item: item[0])
            if similarity >= minimum_similarity:
                matches.append(candidate)
                similarities.append(similarity)
        if len(matches) < minimum_pages:
            continue
        fingerprint = tuple(candidate.page for candidate in matches)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        groups.append((tuple(matches), tuple(similarities)))
        if len(groups) >= 5:
            break
    return tuple(groups)


def _compatible_shape(left: _VisualCandidate, right: _VisualCandidate) -> bool:
    ratio = min(left.aspect_ratio, right.aspect_ratio) / max(
        left.aspect_ratio,
        right.aspect_ratio,
    )
    return ratio >= 0.78


def _visual_similarity(left: _VisualCandidate, right: _VisualCandidate) -> float:
    best = 0.0
    for shift_y in range(-2, 3):
        for shift_x in range(-2, 3):
            shifted = _shift_mask(right.mask, shift_x, shift_y)
            overlap = np.logical_and(left.mask, shifted).sum()
            denominator = left.mask.sum() + shifted.sum()
            if denominator:
                best = max(best, float(2 * overlap / denominator))
    return best


def _shift_mask(mask: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    source_x0 = max(0, -shift_x)
    source_x1 = min(mask.shape[1], mask.shape[1] - shift_x)
    source_y0 = max(0, -shift_y)
    source_y1 = min(mask.shape[0], mask.shape[0] - shift_y)
    target_x0 = source_x0 + shift_x
    target_x1 = source_x1 + shift_x
    target_y0 = source_y0 + shift_y
    target_y1 = source_y1 + shift_y
    shifted[target_y0:target_y1, target_x0:target_x1] = mask[
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return shifted


def _write_montage(
    candidates: tuple[_VisualCandidate, ...],
    similarities: tuple[float, ...],
    path: Path,
) -> None:
    tile_width = 280
    tile_height = 170
    canvas = Image.new("RGB", (tile_width * len(candidates), tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (candidate, similarity) in enumerate(zip(candidates, similarities, strict=True)):
        preview = ImageOps.contain(candidate.crop, (tile_width - 24, 118))
        x = index * tile_width + (tile_width - preview.width) // 2
        y = 32 + (118 - preview.height) // 2
        canvas.paste(preview, (x, y))
        draw.text(
            (index * tile_width + 12, 8),
            f"Page {candidate.page} - similarite {similarity:.0%}",
            fill=(20, 28, 45),
        )
    canvas.save(path, format="PNG")


def _not_applicable(reason: str) -> LaboratoryCheck:
    return LaboratoryCheck(
        code="ocr_visual_repetition",
        title="Éléments visuels répétés",
        purpose="Comparer les cachets, signatures ou images localisés sur plusieurs pages.",
        state="not_applicable",
        summary=reason,
        limitations=_limitations(),
    )


def _limitations() -> tuple[str, ...]:
    return (
        "Le contrôle dépend des zones image, cachet ou signature proposées par l'OCR.",
        "Un logo, un pied de page ou un formulaire répété peut produire la même similarité.",
        "Une signature aplatie non localisée par l'OCR ne peut pas être comparée ici.",
        "Cette observation n'ajoute aucun point au score.",
    )
