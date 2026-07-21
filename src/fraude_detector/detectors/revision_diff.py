"""Visual comparison of the two latest retained PDF revisions."""

from __future__ import annotations

import cv2
import numpy as np
import pypdfium2
from PIL import Image

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.models import BoundingBox, DetectorResult, Finding
from fraude_detector.rendering import render_page


class RevisionDiffDetector:
    name = "revision_diff"

    def analyze(self, context: AnalysisContext) -> DetectorResult:
        if len(context.revision_end_offsets) < 2:
            return DetectorResult(
                name=self.name,
                status="not_applicable",
                notes=("Moins de deux revisions PDF valides a comparer.",),
            )

        previous_data = context.raw_pdf[: context.revision_end_offsets[-2]]
        current_data = context.raw_pdf[: context.revision_end_offsets[-1]]
        previous = pypdfium2.PdfDocument(previous_data, password=context.password)
        current = pypdfium2.PdfDocument(current_data, password=context.password)
        findings: list[Finding] = []
        artifacts: list[str] = []
        changed_pages = 0

        try:
            comparable_pages = min(len(previous), len(current), context.config.max_pages)
            for page_index in range(comparable_pages):
                previous_page = previous[page_index]
                current_page = current[page_index]
                try:
                    previous_image = render_page(
                        previous_page,
                        context.config.render_dpi,
                        context.config.max_render_pixels,
                    )
                    current_image = render_page(
                        current_page,
                        context.config.render_dpi,
                        context.config.max_render_pixels,
                    )
                    page_width, page_height = current_page.get_size()
                finally:
                    previous_page.close()
                    current_page.close()

                page_findings, heatmap = compare_revision_images(
                    previous_image,
                    current_image,
                    page_number=page_index + 1,
                    page_width=page_width,
                    page_height=page_height,
                )
                if not page_findings or heatmap is None:
                    continue
                changed_pages += 1
                artifact_path = context.artifact_path(
                    "forensics", f"page-{page_index + 1:03d}-revision-diff.png"
                )
                heatmap.save(artifact_path)
                relative_artifact = context.relative_artifact(artifact_path)
                artifacts.append(relative_artifact)
                findings.extend(
                    Finding(
                        detector=finding.detector,
                        code=finding.code,
                        category=finding.category,
                        title=finding.title,
                        description=finding.description,
                        risk_points=finding.risk_points,
                        confidence=finding.confidence,
                        page=finding.page,
                        bbox=finding.bbox,
                        evidence=finding.evidence,
                        artifacts=(relative_artifact,),
                    )
                    for finding in page_findings
                )

            if len(previous) != len(current):
                findings.append(
                    Finding(
                        detector=self.name,
                        code="PDF_REVISION_PAGE_COUNT_CHANGED",
                        category="revision_visual",
                        title="Nombre de pages modifie entre deux revisions",
                        description=(
                            "La derniere mise a jour incrementale a ajoute ou retire des "
                            "pages. Ce changement peut etre legitime et doit etre verifie."
                        ),
                        risk_points=35.0,
                        confidence=0.98,
                        evidence={
                            "previous_page_count": len(previous),
                            "current_page_count": len(current),
                        },
                    )
                )
        finally:
            previous.close()
            current.close()

        return DetectorResult(
            name=self.name,
            status="completed",
            findings=tuple(findings),
            notes=(
                "Comparaison de la derniere revision avec la revision precedente.",
                f"Pages comportant une difference visuelle: {changed_pages}.",
            ),
            artifacts=tuple(artifacts),
        )


def compare_revision_images(
    previous: Image.Image,
    current: Image.Image,
    page_number: int,
    page_width: float,
    page_height: float,
) -> tuple[list[Finding], Image.Image | None]:
    """Return localized, renderer-stable changes between two page images."""

    previous_rgb = np.asarray(previous.convert("RGB"), dtype=np.uint8)
    current_rgb = np.asarray(current.convert("RGB"), dtype=np.uint8)
    if previous_rgb.shape != current_rgb.shape:
        previous_rgb = cv2.resize(
            previous_rgb,
            (current_rgb.shape[1], current_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    absolute = cv2.absdiff(previous_rgb, current_rgb)
    delta = np.max(absolute, axis=2)
    raw_mask = (delta >= 18).astype(np.uint8)
    changed_pixels = int(np.count_nonzero(raw_mask))
    if changed_pixels == 0:
        return [], None

    grouped_mask = cv2.dilate(
        raw_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        grouped_mask, connectivity=8
    )
    height, width = raw_mask.shape
    page_pixel_area = max(1, width * height)
    changed_fraction = changed_pixels / page_pixel_area

    components: list[tuple[int, int, int, int, int]] = []
    for label in range(1, component_count):
        x, y, component_width, component_height, _ = stats[label]
        original_pixels = int(np.count_nonzero(raw_mask[labels == label]))
        if original_pixels < 20:
            continue
        components.append((x, y, component_width, component_height, original_pixels))
    components.sort(key=lambda item: item[4], reverse=True)
    if not components:
        return [], None

    findings: list[Finding] = []
    for x, y, component_width, component_height, pixels in components[:8]:
        bbox = BoundingBox(
            x0=(x / width) * page_width,
            y0=(y / height) * page_height,
            x1=((x + component_width) / width) * page_width,
            y1=((y + component_height) / height) * page_height,
        )
        local_fraction = pixels / page_pixel_area
        findings.append(
            Finding(
                detector="revision_diff",
                code="PDF_REVISION_VISUAL_CHANGE",
                category="revision_visual",
                title="Changement visuel entre deux revisions",
                description=(
                    "Cette zone differe entre les deux dernieres revisions conservees "
                    "du PDF. La difference est technique et localisee, mais son motif "
                    "metier doit encore etre controle."
                ),
                risk_points=55.0 if changed_fraction <= 0.25 else 45.0,
                confidence=0.96,
                page=page_number,
                bbox=bbox,
                evidence={
                    "algorithm": "retained_revision_pixel_diff_v1",
                    "document_changed_fraction": round(changed_fraction, 6),
                    "region_changed_fraction": round(local_fraction, 6),
                    "pixel_threshold": 18,
                },
            )
        )

    heatmap = _revision_heatmap(current_rgb, delta, grouped_mask)
    return findings, heatmap


def _revision_heatmap(
    current_rgb: np.ndarray,
    delta: np.ndarray,
    grouped_mask: np.ndarray,
) -> Image.Image:
    normalized = np.uint8(np.clip(delta.astype(np.float32) / 80.0, 0.0, 1.0) * 255)
    colored = cv2.cvtColor(
        cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )
    alpha = (grouped_mask.astype(np.float32) * 0.7)[..., None]
    overlay = current_rgb * (1.0 - alpha) + colored * alpha
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))
