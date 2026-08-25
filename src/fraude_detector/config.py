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
    ai_analyze_pdf_images: bool = False
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
    classification_enabled: bool = False
    classification_url: str = "http://127.0.0.1:8030"
    classification_model: str = "minimax_m2_1"
    classification_timeout_seconds: int = 900
    classification_max_input_chars: int = 20_000
    classification_max_tokens: int = 32_768
    classification_temperature: float = 0.0
    extraction_enabled: bool = False
    extraction_url: str = "http://127.0.0.1:8030"
    extraction_model: str = "minimax_m2_1"
    extraction_timeout_seconds: int = 1_800
    extraction_max_input_chars: int = 48_000
    extraction_max_tokens: int = 32_768
    extraction_temperature: float = 0.0
    extraction_coverage_retry: bool = False
    verification_enabled: bool = False
    verification_url: str = "http://127.0.0.1:8030"
    verification_model: str = "minimax_m2_1"
    verification_timeout_seconds: int = 1_800
    verification_max_input_chars: int = 80_000
    verification_max_tokens: int = 32_768
    verification_temperature: float = 0.0
    synthesis_enabled: bool = False
    synthesis_url: str = "http://127.0.0.1:8030"
    synthesis_model: str = "minimax_m2_1"
    synthesis_timeout_seconds: int = 900
    synthesis_max_input_chars: int = 24_000
    synthesis_max_tokens: int = 32_768
    synthesis_temperature: float = 0.0

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
        if not 0 < self.full_page_image_coverage <= 1:
            raise ValueError("full_page_image_coverage must be between 0 and 1")
        if not 0 <= self.minimum_overlay_coverage <= 1:
            raise ValueError("minimum_overlay_coverage must be between 0 and 1")
        if not 0 < self.maximum_overlay_coverage <= 1:
            raise ValueError("maximum_overlay_coverage must be between 0 and 1")
        if self.minimum_overlay_coverage > self.maximum_overlay_coverage:
            raise ValueError("minimum_overlay_coverage must not exceed maximum_overlay_coverage")
        if not 1 <= self.ela_jpeg_quality <= 100:
            raise ValueError("ela_jpeg_quality must be between 1 and 100")
        if self.ela_block_size < 1:
            raise ValueError("ela_block_size must be at least 1")
        if self.ela_robust_z_threshold <= 0:
            raise ValueError("ela_robust_z_threshold must be positive")
        if self.ela_max_image_dimension < 64:
            raise ValueError("ela_max_image_dimension must be at least 64")
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
        if self.classification_enabled and not self.classification_url.strip():
            raise ValueError("classification_url must not be empty when classification is enabled")
        if self.classification_enabled and not self.classification_model.strip():
            raise ValueError(
                "classification_model must not be empty when classification is enabled"
            )
        if self.classification_timeout_seconds < 1:
            raise ValueError("classification_timeout_seconds must be at least 1")
        if self.classification_max_input_chars < 100:
            raise ValueError("classification_max_input_chars must be at least 100")
        if self.classification_max_tokens < 100:
            raise ValueError("classification_max_tokens must be at least 100")
        if not 0 <= self.classification_temperature <= 2:
            raise ValueError("classification_temperature must be between 0 and 2")
        if self.extraction_enabled and not self.extraction_url.strip():
            raise ValueError("extraction_url must not be empty when extraction is enabled")
        if self.extraction_enabled and not self.extraction_model.strip():
            raise ValueError("extraction_model must not be empty when extraction is enabled")
        if self.extraction_timeout_seconds < 1:
            raise ValueError("extraction_timeout_seconds must be at least 1")
        if self.extraction_max_input_chars < 500:
            raise ValueError("extraction_max_input_chars must be at least 500")
        if self.extraction_max_tokens < 500:
            raise ValueError("extraction_max_tokens must be at least 500")
        if not 0 <= self.extraction_temperature <= 2:
            raise ValueError("extraction_temperature must be between 0 and 2")
        if self.verification_enabled and not self.verification_url.strip():
            raise ValueError("verification_url must not be empty when verification is enabled")
        if self.verification_enabled and not self.verification_model.strip():
            raise ValueError("verification_model must not be empty when verification is enabled")
        if self.verification_timeout_seconds < 1:
            raise ValueError("verification_timeout_seconds must be at least 1")
        if self.verification_max_input_chars < 1_000:
            raise ValueError("verification_max_input_chars must be at least 1000")
        if self.verification_max_tokens < 500:
            raise ValueError("verification_max_tokens must be at least 500")
        if not 0 <= self.verification_temperature <= 2:
            raise ValueError("verification_temperature must be between 0 and 2")
        if self.synthesis_enabled and not self.synthesis_url.strip():
            raise ValueError("synthesis_url must not be empty when synthesis is enabled")
        if self.synthesis_enabled and not self.synthesis_model.strip():
            raise ValueError("synthesis_model must not be empty when synthesis is enabled")
        if self.synthesis_timeout_seconds < 1:
            raise ValueError("synthesis_timeout_seconds must be at least 1")
        if self.synthesis_max_input_chars < 1_000:
            raise ValueError("synthesis_max_input_chars must be at least 1000")
        if self.synthesis_max_tokens < 100:
            raise ValueError("synthesis_max_tokens must be at least 100")
        if not 0 <= self.synthesis_temperature <= 2:
            raise ValueError("synthesis_temperature must be between 0 and 2")
