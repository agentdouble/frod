"""Local compression anomaly analysis for embedded JPEG scans."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageOps

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.geometry import bbox_coverage, pdfium_bounds_to_top_left
from fraude_detector.models import BoundingBox, DetectorResult, Finding


@dataclass(frozen=True, slots=True)
class RasterRegion:
    pixel_box: tuple[int, int, int, int]
    max_robust_z: float
    area_fraction: float


@dataclass(frozen=True, slots=True)
class ElaAnalysis:
    heatmap: np.ndarray
    regions: tuple[RasterRegion, ...]


class RasterAnomalyDetector:
    name = "raster_anomaly"

    def analyze(self, context: AnalysisContext) -> DetectorResult:
        findings: list[Finding] = []
        artifacts: list[str] = []
        scan_images = 0
        jpeg_images = 0
        oversized_images = 0
        non_local_regions = 0

        for page_index in range(context.analyzed_page_count):
            page = context.pdfium_document[page_index]
            try:
                page_width, page_height = page.get_size()
                image_objects = list(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]))
                for image_index, image_object in enumerate(image_objects, start=1):
                    bbox = pdfium_bounds_to_top_left(image_object.get_bounds(), page_height)
                    if bbox is None:
                        continue
                    coverage = bbox_coverage(bbox, page_width, page_height)
                    if coverage < context.config.full_page_image_coverage:
                        continue
                    scan_images += 1

                    filters = list(image_object.get_filters())
                    if "DCTDecode" not in filters:
                        continue
                    jpeg_data = bytes(image_object.get_data(decode_simple=True))
                    if not jpeg_data.startswith(b"\xff\xd8"):
                        continue
                    jpeg_images += 1

                    try:
                        image = Image.open(io.BytesIO(jpeg_data))
                        if image.width * image.height > context.config.max_embedded_image_pixels:
                            oversized_images += 1
                            continue
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        image.load()
                    except Exception:
                        continue

                    image = _bounded_image(image, context.config.ela_max_image_dimension)
                    analysis = analyze_jpeg_ela(
                        image=image,
                        jpeg_quality=context.config.ela_jpeg_quality,
                        block_size=context.config.ela_block_size,
                        robust_z_threshold=context.config.ela_robust_z_threshold,
                    )
                    localized_regions = tuple(
                        region
                        for region in analysis.regions
                        if region.area_fraction <= context.config.ela_max_region_area_fraction
                    )
                    non_local_regions += len(analysis.regions) - len(localized_regions)
                    if not localized_regions:
                        continue

                    artifact_path = context.artifact_path(
                        "forensics",
                        f"page-{page_index + 1:03d}-image-{image_index:02d}-ela.png",
                    )
                    _write_heatmap_overlay(image, analysis.heatmap, artifact_path)
                    relative_artifact = context.relative_artifact(artifact_path)
                    artifacts.append(relative_artifact)

                    for region in localized_regions[:5]:
                        pdf_box = _map_image_region_to_page(
                            region.pixel_box,
                            image.size,
                            bbox,
                        )
                        strength = min(1.0, region.max_robust_z / 8.0)
                        findings.append(
                            Finding(
                                detector=self.name,
                                code="RASTER_LOCAL_COMPRESSION_ANOMALY",
                                category="raster_forensics",
                                title="Anomalie locale de compression JPEG",
                                description=(
                                    "Une zone du JPEG embarque reagit differemment a la "
                                    "recompression. C'est un indice faible a corroborer: "
                                    "texte, bords et scans successifs peuvent produire le "
                                    "meme effet."
                                ),
                                risk_points=round(18.0 + 12.0 * strength, 2),
                                confidence=round(0.45 + 0.35 * strength, 3),
                                page=page_index + 1,
                                bbox=pdf_box,
                                evidence={
                                    "algorithm": "ela_block_robust_z_v1",
                                    "max_robust_z": round(region.max_robust_z, 3),
                                    "region_area_fraction": round(region.area_fraction, 6),
                                    "jpeg_quality": context.config.ela_jpeg_quality,
                                },
                                artifacts=(relative_artifact,),
                            )
                        )
            finally:
                page.close()

        status = "completed" if jpeg_images else "not_applicable"
        notes = (
            f"Images de scan couvrant la page: {scan_images}.",
            f"JPEG originaux analyses par ELA: {jpeg_images}.",
            f"Images ignorees car trop grandes: {oversized_images}.",
            f"Regions ELA trop etendues pour etre locales: {non_local_regions}.",
            "ELA n'est pas appliquee aux rendus de page ni aux images sans compression JPEG.",
        )
        return DetectorResult(
            name=self.name,
            status=status,
            findings=tuple(findings),
            notes=notes,
            artifacts=tuple(artifacts),
        )


def analyze_jpeg_ela(
    image: Image.Image,
    jpeg_quality: int,
    block_size: int,
    robust_z_threshold: float,
) -> ElaAnalysis:
    """Locate blocks whose ELA energy differs robustly from the page."""

    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=False)
    buffer.seek(0)
    recompressed = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)

    difference = cv2.absdiff(source, recompressed)
    energy = np.max(difference, axis=2).astype(np.float32)
    energy = cv2.GaussianBlur(energy, (0, 0), sigmaX=1.2)

    height, width = energy.shape
    rows = max(1, math.ceil(height / block_size))
    columns = max(1, math.ceil(width / block_size))
    block_energy = np.zeros((rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            y0, y1 = row * block_size, min(height, (row + 1) * block_size)
            x0, x1 = column * block_size, min(width, (column + 1) * block_size)
            block_energy[row, column] = float(np.percentile(energy[y0:y1, x0:x1], 75))

    median = float(np.median(block_energy))
    mad = float(np.median(np.abs(block_energy - median)))
    robust_scale = max(1e-3, 1.4826 * mad)
    robust_z = (block_energy - median) / robust_scale

    if rows * columns < 9 or mad < 1e-3:
        robust_z.fill(0.0)

    suspicious = (robust_z >= robust_z_threshold).astype(np.uint8)
    suspicious = cv2.morphologyEx(
        suspicious,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), dtype=np.uint8),
    )
    region_count, labels, stats, _ = cv2.connectedComponentsWithStats(suspicious, connectivity=8)

    regions: list[RasterRegion] = []
    image_area = max(1, width * height)
    for label in range(1, region_count):
        x, y, block_width, block_height, block_area = stats[label]
        if block_area < 1:
            continue
        x0, y0 = int(x * block_size), int(y * block_size)
        x1 = min(width, int((x + block_width) * block_size))
        y1 = min(height, int((y + block_height) * block_size))
        component_values = robust_z[labels == label]
        regions.append(
            RasterRegion(
                pixel_box=(x0, y0, x1, y1),
                max_robust_z=float(np.max(component_values)),
                area_fraction=((x1 - x0) * (y1 - y0)) / image_area,
            )
        )

    regions.sort(key=lambda region: region.max_robust_z, reverse=True)
    pixel_heatmap = cv2.resize(
        np.clip(robust_z, 0.0, 8.0),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return ElaAnalysis(heatmap=pixel_heatmap, regions=tuple(regions))


def _bounded_image(image: Image.Image, max_dimension: int) -> Image.Image:
    if max(image.size) <= max_dimension:
        return image
    bounded = image.copy()
    bounded.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    return bounded


def _map_image_region_to_page(
    pixel_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    page_box: BoundingBox,
) -> BoundingBox:
    x0, y0, x1, y1 = pixel_box
    width, height = image_size
    return BoundingBox(
        x0=page_box.x0 + (x0 / width) * (page_box.x1 - page_box.x0),
        y0=page_box.y0 + (y0 / height) * (page_box.y1 - page_box.y0),
        x1=page_box.x0 + (x1 / width) * (page_box.x1 - page_box.x0),
        y1=page_box.y0 + (y1 / height) * (page_box.y1 - page_box.y0),
    )


def _write_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: Path,
) -> None:
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    normalized = np.uint8(np.clip(heatmap / 8.0, 0.0, 1.0) * 255)
    colored_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    alpha = np.clip(heatmap / 5.0, 0.0, 0.75)[..., None]
    overlay = source * (1.0 - alpha) + colored_rgb * alpha
    Image.fromarray(np.uint8(np.clip(overlay, 0, 255))).save(output_path)
