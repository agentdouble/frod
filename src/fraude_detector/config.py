"""Configuration shared by the analysis pipeline and its detectors."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Tunable MVP thresholds.

    Defaults deliberately favor manual review over a definitive fraud claim.
    """

    render_dpi: int = 144
    max_pages: int = 25
    max_file_size_mb: int = 100
    max_render_pixels: int = 25_000_000
    max_embedded_image_pixels: int = 50_000_000
    full_page_image_coverage: float = 0.65
    minimum_overlay_coverage: float = 0.001
    maximum_overlay_coverage: float = 0.50
    ela_jpeg_quality: int = 90
    ela_block_size: int = 64
    ela_robust_z_threshold: float = 3.5
    ela_max_image_dimension: int = 2400
    ela_max_region_area_fraction: float = 0.25
    ai_max_images: int = 20
    ai_max_inventory_images: int = 100
    ai_min_photo_page_coverage: float = 0.02
    ai_min_photo_side: int = 256
    ai_min_photo_pixels: int = 262_144
    ai_max_image_dimension: int = 1536
    ai_model_score_threshold: float = 0.80
    ai_min_consensus_families: int = 2
    ai_stability_max_delta: float = 0.15
    ocr_enabled: bool = False
    ocr_url: str = "http://127.0.0.1:8007"
    ocr_timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.render_dpi < 72:
            raise ValueError("render_dpi must be at least 72")
        if self.max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        if self.max_file_size_mb < 1:
            raise ValueError("max_file_size_mb must be at least 1")
        if self.max_render_pixels < 1:
            raise ValueError("max_render_pixels must be at least 1")
        if self.max_embedded_image_pixels < 1:
            raise ValueError("max_embedded_image_pixels must be at least 1")
        if not 1 <= self.ela_jpeg_quality <= 100:
            raise ValueError("ela_jpeg_quality must be between 1 and 100")
        if not 0 < self.ela_max_region_area_fraction <= 1:
            raise ValueError("ela_max_region_area_fraction must be between 0 and 1")
        if self.ai_max_images < 1:
            raise ValueError("ai_max_images must be at least 1")
        if self.ai_max_inventory_images < self.ai_max_images:
            raise ValueError("ai_max_inventory_images must be at least ai_max_images")
        if not 0 < self.ai_min_photo_page_coverage <= 1:
            raise ValueError("ai_min_photo_page_coverage must be between 0 and 1")
        if self.ai_min_photo_side < 1:
            raise ValueError("ai_min_photo_side must be at least 1")
        if self.ai_min_photo_pixels < 1:
            raise ValueError("ai_min_photo_pixels must be at least 1")
        if self.ai_max_image_dimension < 256:
            raise ValueError("ai_max_image_dimension must be at least 256")
        if not 0 <= self.ai_model_score_threshold <= 1:
            raise ValueError("ai_model_score_threshold must be between 0 and 1")
        if self.ai_min_consensus_families < 2:
            raise ValueError("ai_min_consensus_families must be at least 2")
        if not 0 <= self.ai_stability_max_delta <= 1:
            raise ValueError("ai_stability_max_delta must be between 0 and 1")
        if self.ocr_enabled and not self.ocr_url.strip():
            raise ValueError("ocr_url must not be empty when OCR is enabled")
        if self.ocr_timeout_seconds < 1:
            raise ValueError("ocr_timeout_seconds must be at least 1")
