from __future__ import annotations

from fraude_detector.config import AnalysisConfig
from fraude_detector.image_assets import PdfImageAsset
from fraude_detector.image_roles import route_image_asset, route_image_assets
from fraude_detector.models import BoundingBox


def test_full_page_image_is_routed_as_document() -> None:
    routed = route_image_asset(_asset(coverage=1.0, pixel_size=(1200, 1697)), AnalysisConfig())

    assert routed.role == "document"
    assert routed.reason == "image_covers_most_of_page"


def test_small_overlay_is_routed_as_decorative() -> None:
    routed = route_image_asset(
        _asset(coverage=0.008, pixel_size=(360, 180)),
        AnalysisConfig(),
    )

    assert routed.role == "decorative"
    assert routed.reason == "page_coverage_below_photo_threshold"


def test_large_embedded_image_is_routed_as_photo() -> None:
    routed = route_image_asset(
        _asset(coverage=0.30, pixel_size=(1200, 900)),
        AnalysisConfig(),
    )

    assert routed.role == "photo"
    assert routed.reason == "eligible_embedded_photo"


def test_missing_dimensions_abstains_from_role_classification() -> None:
    routed = route_image_asset(_asset(coverage=0.20, pixel_size=None), AnalysisConfig())

    assert routed.role == "unknown"
    assert routed.reason == "pixel_dimensions_unavailable"


def test_routing_does_not_apply_analysis_quota_before_role_selection() -> None:
    assets = (
        _asset(coverage=0.001, pixel_size=(64, 64)),
        _asset(coverage=0.30, pixel_size=(1200, 900)),
    )

    routed = route_image_assets(assets, AnalysisConfig(ai_max_images=1))

    assert [item.role for item in routed] == ["decorative", "photo"]


def _asset(
    *,
    coverage: float,
    pixel_size: tuple[int, int] | None,
) -> PdfImageAsset:
    return PdfImageAsset(
        page=1,
        index=1,
        bbox=BoundingBox(0, 0, 100, 100),
        coverage=coverage,
        pixel_size=pixel_size,
        filters=("DCTDecode",),
        sha256="a" * 64,
        format="jpeg",
        mime_type="image/jpeg",
        native_bytes=b"jpeg",
        native_status="available",
        native_reason=None,
    )
