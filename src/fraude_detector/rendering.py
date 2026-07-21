"""PDF rendering and human-review overlays."""

from __future__ import annotations

import math
from dataclasses import replace

import pypdfium2
from PIL import Image, ImageDraw, ImageFont

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.geometry import merge_nearby_boxes
from fraude_detector.models import Finding


def render_page(
    page: pypdfium2.PdfPage,
    dpi: int,
    max_pixels: int | None = None,
) -> Image.Image:
    """Render a PDFium page to a detached RGB Pillow image."""

    scale = dpi / 72.0
    if max_pixels is not None:
        page_width, page_height = page.get_size()
        estimated_pixels = max(1.0, page_width * page_height * scale * scale)
        if estimated_pixels > max_pixels:
            scale *= math.sqrt(max_pixels / estimated_pixels)
    bitmap = page.render(
        scale=scale,
        rev_byteorder=True,
        fill_color=(255, 255, 255, 255),
    )
    try:
        return bitmap.to_pil().convert("RGB").copy()
    finally:
        bitmap.close()


def render_input_pages(context: AnalysisContext) -> tuple[str, ...]:
    artifacts: list[str] = []
    for page_index in range(context.analyzed_page_count):
        page = context.pdfium_document[page_index]
        try:
            image = render_page(
                page,
                context.config.render_dpi,
                context.config.max_render_pixels,
            )
        finally:
            page.close()
        output_path = context.artifact_path("pages", f"page-{page_index + 1:03d}.png")
        image.save(output_path)
        artifacts.append(context.relative_artifact(output_path))
    return tuple(artifacts)


def create_review_overlays(
    context: AnalysisContext,
    rendered_pages: tuple[str, ...],
    findings: tuple[Finding, ...],
) -> tuple[str, ...]:
    """Draw every localized signal on a copy of its rendered page."""

    by_page: dict[int, list[Finding]] = {}
    for finding in findings:
        if finding.page is not None and finding.bbox is not None:
            by_page.setdefault(finding.page, []).append(finding)

    artifacts: list[str] = []
    for page_number, page_findings in sorted(by_page.items()):
        if page_number < 1 or page_number > len(rendered_pages):
            continue
        page_findings = _merge_display_findings(page_findings)
        render_path = context.output_dir / rendered_pages[page_number - 1]
        page_image = Image.open(render_path).convert("RGBA")
        page = context.pdfium_document[page_number - 1]
        try:
            page_width, page_height = page.get_size()
        finally:
            page.close()
        scale_x = page_image.width / max(1.0, page_width)
        scale_y = page_image.height / max(1.0, page_height)

        legend_width = max(420, page_image.width // 3)
        canvas_height = max(page_image.height, 120 + len(page_findings) * 76)
        canvas = Image.new(
            "RGBA",
            (page_image.width + legend_width, canvas_height),
            (248, 250, 252, 255),
        )
        canvas.paste(page_image, (0, 0))
        draw = ImageDraw.Draw(canvas, "RGBA")
        title_font = ImageFont.load_default(size=24)
        label_font = ImageFont.load_default(size=18)
        detail_font = ImageFont.load_default(size=15)
        legend_x = page_image.width + 24
        draw.text(
            (legend_x, 28),
            "Zones a controler",
            font=title_font,
            fill=(15, 23, 42, 255),
        )
        palette = (
            (220, 38, 38, 255),
            (234, 88, 12, 255),
            (202, 138, 4, 255),
            (124, 58, 237, 255),
            (2, 132, 199, 255),
            (5, 150, 105, 255),
        )
        for index, finding in enumerate(page_findings, start=1):
            assert finding.bbox is not None
            rectangle = (
                round(finding.bbox.x0 * scale_x),
                round(finding.bbox.y0 * scale_y),
                round(finding.bbox.x1 * scale_x),
                round(finding.bbox.y1 * scale_y),
            )
            color = palette[(index - 1) % len(palette)]
            draw.rectangle(rectangle, outline=color, width=4)
            badge = str(index)
            badge_box = draw.textbbox((0, 0), badge, font=label_font)
            badge_width = badge_box[2] - badge_box[0] + 12
            badge_height = badge_box[3] - badge_box[1] + 8
            badge_x = max(0, rectangle[0] + ((index - 1) % 4) * (badge_width + 3))
            badge_y = max(0, rectangle[1] - badge_height)
            draw.rectangle(
                (badge_x, badge_y, badge_x + badge_width, badge_y + badge_height),
                fill=color,
            )
            draw.text(
                (badge_x + 6, badge_y + 3),
                badge,
                font=label_font,
                fill=(255, 255, 255, 255),
            )

            legend_y = 88 + (index - 1) * 76
            draw.rectangle(
                (legend_x, legend_y + 2, legend_x + 18, legend_y + 20),
                fill=color,
            )
            draw.text(
                (legend_x + 28, legend_y),
                f"{index}. {finding.code}",
                font=label_font,
                fill=(15, 23, 42, 255),
            )
            draw.text(
                (legend_x + 28, legend_y + 28),
                f"Score: {finding.risk_points:g} | Confiance: {finding.confidence:.0%}",
                font=detail_font,
                fill=(71, 85, 105, 255),
            )

        output_path = context.artifact_path("review", f"page-{page_number:03d}-review.png")
        canvas.convert("RGB").save(output_path)
        artifacts.append(context.relative_artifact(output_path))
    return tuple(artifacts)


def _merge_display_findings(findings: list[Finding]) -> list[Finding]:
    """Reduce repeated boxes in the visual overlay while preserving report detail."""

    by_code: dict[str, list[Finding]] = {}
    for finding in findings:
        by_code.setdefault(finding.code, []).append(finding)

    display_findings: list[Finding] = []
    for same_code in by_code.values():
        boxes = [finding.bbox for finding in same_code if finding.bbox is not None]
        representative = max(same_code, key=lambda finding: finding.risk_points)
        for box in merge_nearby_boxes(boxes, horizontal_gap=12.0, vertical_gap=8.0):
            display_findings.append(replace(representative, bbox=box))
    return display_findings
