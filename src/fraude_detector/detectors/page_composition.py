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
                    findings.append(
                        Finding(
                            detector=self.name,
                            code="SCAN_IMAGE_OVERLAY",
                            category="page_composition",
                            title="Image superposee a un scan",
                            description=(
                                "Une petite image est posee au-dessus d'une image couvrant "
                                "la page. Il peut s'agir d'un collage, mais aussi d'un logo, "
                                "d'un tampon ou d'une signature legitime."
                            ),
                            risk_points=35.0,
                            confidence=0.72,
                            page=page_index + 1,
                            bbox=overlay["bbox"],
                            evidence={
                                "background_coverage": round(background["coverage"], 5),
                                "overlay_coverage": round(coverage, 5),
                                "overlay_pixels": overlay["pixel_size"],
                                "filters": overlay["filters"],
                            },
                        )
                    )

                visible_boxes = _visible_text_boxes(objects, page_height)
                for box in merge_nearby_boxes(visible_boxes)[:8]:
                    coverage = bbox_coverage(box, page_width, page_height)
                    if coverage > 0.35:
                        continue
                    visible_text_overlays += 1
                    findings.append(
                        Finding(
                            detector=self.name,
                            code="SCAN_VISIBLE_TEXT_OVERLAY",
                            category="page_composition",
                            title="Texte visible superpose a un scan",
                            description=(
                                "Un objet texte visible est ajoute au-dessus d'une image de "
                                "scan. Un calque OCR invisible est ignore; un champ rempli "
                                "ou une annotation legitime restent possibles."
                            ),
                            risk_points=30.0,
                            confidence=0.75,
                            page=page_index + 1,
                            bbox=box,
                            evidence={"overlay_coverage": round(coverage, 5)},
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
                }
            )
        except Exception:
            continue
    return descriptions


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
