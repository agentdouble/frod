"""Detect potentially meaningful overlays on scanned PDF pages."""

from __future__ import annotations

from typing import Any

import pypdfium2.raw as pdfium_c

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.geometry import (
    bbox_coverage,
    merge_nearby_boxes,
    pdfium_bounds_to_top_left,
)
from fraude_detector.models import BoundingBox, DetectorResult, Finding


class PageCompositionDetector:
    name = "page_composition"

    def analyze(self, context: AnalysisContext) -> DetectorResult:
        findings: list[Finding] = []
        scanned_pages = 0
        image_overlays = 0
        visible_text_overlays = 0

        for page_index in range(context.analyzed_page_count):
            page = context.pdfium_document[page_index]
            try:
                page_width, page_height = page.get_size()
                text_page = page.get_textpage()
                try:
                    objects = list(page.get_objects(textpage=text_page))
                finally:
                    text_page.close()

                image_objects = [obj for obj in objects if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE]
                images = _describe_images(image_objects, page_width, page_height)
                if not images:
                    continue

                background = max(images, key=lambda item: item["coverage"])
                if background["coverage"] < context.config.full_page_image_coverage:
                    continue
                scanned_pages += 1
                page_is_segmented_scan = any(
                    _looks_like_scanner_segment(image, background)
                    for image in images
                    if image is not background
                )
                has_retained_update = len(context.revision_end_offsets) > 1

                for overlay in images:
                    if overlay is background:
                        continue
                    coverage = overlay["coverage"]
                    if not (
                        context.config.minimum_overlay_coverage
                        <= coverage
                        <= context.config.maximum_overlay_coverage
                    ):
                        continue
                    image_overlays += 1
                    scanner_segment = _looks_like_scanner_segment(overlay, background)
                    scored = has_retained_update and not scanner_segment
                    findings.append(
                        Finding(
                            detector=self.name,
                            code="SCAN_IMAGE_OVERLAY",
                            category="page_composition",
                            title=(
                                "Image ajoutee a une page numerisee"
                                if scored
                                else "Couche image integree a une page numerisee"
                            ),
                            description=(
                                "Une image distincte est placee au-dessus du scan et le PDF "
                                "conserve une mise a jour posterieure. Elle doit etre rapprochee "
                                "de l'historique visuel."
                                if scored
                                else "Le PDF contient plusieurs couches image. Cette structure "
                                "est courante avec les scanners segmentant logos, texte et fond; "
                                "elle reste visible sans modifier le score."
                            ),
                            risk_points=35.0 if scored else 0.0,
                            confidence=0.72 if scored else 0.5,
                            page=page_index + 1,
                            bbox=overlay["bbox"],
                            evidence={
                                "background_coverage": round(background["coverage"], 5),
                                "overlay_coverage": round(coverage, 5),
                                "overlay_pixels": overlay["pixel_size"],
                                "filters": overlay["filters"],
                                "image_metadata": overlay["metadata"],
                                "scanner_segment": scanner_segment,
                                "retained_incremental_update": has_retained_update,
                                "score_reason": (
                                    "overlay_with_retained_update"
                                    if scored
                                    else (
                                        "scanner_segmentation_pattern"
                                        if scanner_segment
                                        else "single_retained_revision"
                                    )
                                ),
                            },
                        )
                    )

                visible_boxes = _visible_text_boxes(objects, page_height)
                for box in merge_nearby_boxes(visible_boxes)[:8]:
                    coverage = bbox_coverage(box, page_width, page_height)
                    if coverage > 0.35:
                        continue
                    visible_text_overlays += 1
                    scored = has_retained_update and not page_is_segmented_scan
                    findings.append(
                        Finding(
                            detector=self.name,
                            code="SCAN_VISIBLE_TEXT_OVERLAY",
                            category="page_composition",
                            title=(
                                "Texte ajoute a une page numerisee"
                                if scored
                                else "Couche texte visible sur une page numerisee"
                            ),
                            description=(
                                "Le texte visible est distinct du scan et le PDF conserve une "
                                "mise a jour posterieure. Un formulaire rempli peut toutefois "
                                "expliquer cette structure."
                                if scored
                                else "Le texte visible appartient a la construction du PDF. "
                                "Un formulaire numerise puis rempli ou un scanner segmente peut "
                                "produire cette structure sans modification frauduleuse."
                            ),
                            risk_points=30.0 if scored else 0.0,
                            confidence=0.75 if scored else 0.5,
                            page=page_index + 1,
                            bbox=box,
                            evidence={
                                "overlay_coverage": round(coverage, 5),
                                "segmented_scan": page_is_segmented_scan,
                                "retained_incremental_update": has_retained_update,
                                "score_reason": (
                                    "visible_text_with_retained_update"
                                    if scored
                                    else (
                                        "segmented_scan_structure"
                                        if page_is_segmented_scan
                                        else "single_retained_revision"
                                    )
                                ),
                            },
                        )
                    )
            finally:
                page.close()

        annotation_findings = _annotation_findings(context)
        findings.extend(annotation_findings)

        status = "completed" if scanned_pages else "not_applicable"
        notes = (
            f"Pages de type scan detectees: {scanned_pages}.",
            f"Overlays image localises: {image_overlays}.",
            f"Overlays texte visible localises: {visible_text_overlays}.",
            f"Annotations de modification localisees: {len(annotation_findings)}.",
        )
        return DetectorResult(
            name=self.name,
            status=status,
            findings=tuple(findings),
            notes=notes,
        )


def _describe_images(
    image_objects: list[Any],
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    descriptions: list[dict[str, Any]] = []
    for obj in image_objects:
        try:
            bbox = pdfium_bounds_to_top_left(obj.get_bounds(), page_height)
            if bbox is None:
                continue
            descriptions.append(
                {
                    "object": obj,
                    "bbox": bbox,
                    "coverage": bbox_coverage(bbox, page_width, page_height),
                    "pixel_size": list(obj.get_px_size()),
                    "filters": list(obj.get_filters()),
                    "metadata": _image_metadata(obj),
                }
            )
        except Exception:
            continue
    return descriptions


def _image_metadata(image_object: Any) -> dict[str, int | float]:
    try:
        metadata = image_object.get_metadata()
    except Exception:
        return {}
    return {
        "width": int(metadata.width),
        "height": int(metadata.height),
        "bits_per_pixel": int(metadata.bits_per_pixel),
        "colorspace": int(metadata.colorspace),
        "horizontal_dpi": round(float(metadata.horizontal_dpi), 2),
        "vertical_dpi": round(float(metadata.vertical_dpi), 2),
        "marked_content_id": int(metadata.marked_content_id),
    }


def _looks_like_scanner_segment(
    overlay: dict[str, Any],
    background: dict[str, Any],
) -> bool:
    """Recognise common MRC/bitonal layers without treating DCT as scanner provenance."""

    filters = set(overlay["filters"])
    if filters & {"JBIG2Decode", "CCITTFaxDecode"}:
        return True

    metadata = overlay["metadata"]
    bits_per_pixel = int(metadata.get("bits_per_pixel", 0))
    if not bits_per_pixel or bits_per_pixel > 8:
        return False
    if not filters & {"FlateDecode", "RunLengthDecode"}:
        return False

    overlay_dpi = float(metadata.get("horizontal_dpi", 0))
    background_dpi = float(background["metadata"].get("horizontal_dpi", 0))
    if overlay_dpi <= 0 or background_dpi <= 0:
        return True
    return abs(overlay_dpi - background_dpi) / max(overlay_dpi, background_dpi) <= 0.2


def _visible_text_boxes(objects: list[Any], page_height: float) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    for obj in objects:
        if obj.type != pdfium_c.FPDF_PAGEOBJ_TEXT:
            continue
        try:
            render_mode = pdfium_c.FPDFTextObj_GetTextRenderMode(obj)
            if render_mode == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE:
                continue
            bbox = pdfium_bounds_to_top_left(obj.get_bounds(), page_height)
            if bbox is not None:
                boxes.append(bbox)
        except Exception:
            continue
    return boxes


def _annotation_findings(context: AnalysisContext) -> list[Finding]:
    findings: list[Finding] = []
    reviewable_subtypes = {
        "/Circle",
        "/FreeText",
        "/Highlight",
        "/Ink",
        "/Square",
        "/Stamp",
        "/StrikeOut",
        "/Text",
        "/Underline",
    }

    for page_index, page in enumerate(context.reader.pages[: context.analyzed_page_count]):
        page_height = float(page.mediabox.height)
        for annotation_ref in page.get("/Annots", []):
            try:
                annotation = annotation_ref.get_object()
                subtype = str(annotation.get("/Subtype", ""))
                if subtype not in reviewable_subtypes:
                    continue
                rect = [float(value) for value in annotation.get("/Rect", [])]
                if len(rect) != 4:
                    continue
                bbox = pdfium_bounds_to_top_left(tuple(rect), page_height)
                if bbox is None:
                    continue
                findings.append(
                    Finding(
                        detector="page_composition",
                        code="PDF_REVIEW_ANNOTATION",
                        category="annotations",
                        title="Annotation PDF modifiable",
                        description=(
                            "Une annotation visible a ete ajoutee au document. Elle peut "
                            "etre legitime et doit etre rapprochee du processus metier."
                        ),
                        risk_points=6.0,
                        confidence=0.9,
                        page=page_index + 1,
                        bbox=bbox,
                        evidence={"annotation_subtype": subtype},
                    )
                )
            except Exception:
                continue
    return findings
