"""Explainable routing of embedded PDF images by their document role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fraude_detector.config import AnalysisConfig
from fraude_detector.image_assets import PdfImageAsset

ImageRole = Literal["photo", "document", "decorative", "unknown"]


@dataclass(frozen=True, slots=True)
class RoutedImageAsset:
    """An image asset paired with the reason for its analysis route."""

    asset: PdfImageAsset
    role: ImageRole
    reason: str


def route_image_asset(
    asset: PdfImageAsset,
    config: AnalysisConfig,
) -> RoutedImageAsset:
    """Classify structural role without making claims about authenticity."""

    if asset.coverage >= config.full_page_image_coverage:
        return RoutedImageAsset(
            asset=asset,
            role="document",
            reason="image_covers_most_of_page",
        )

    if asset.pixel_size is None:
        return RoutedImageAsset(
            asset=asset,
            role="unknown",
            reason="pixel_dimensions_unavailable",
        )

    width, height = asset.pixel_size
    if asset.coverage < config.ai_min_photo_page_coverage:
        return RoutedImageAsset(
            asset=asset,
            role="decorative",
            reason="page_coverage_below_photo_threshold",
        )
    if min(width, height) < config.ai_min_photo_side:
        return RoutedImageAsset(
            asset=asset,
            role="decorative",
            reason="short_side_below_photo_threshold",
        )
    if width * height < config.ai_min_photo_pixels:
        return RoutedImageAsset(
            asset=asset,
            role="decorative",
            reason="pixel_count_below_photo_threshold",
        )

    return RoutedImageAsset(
        asset=asset,
        role="photo",
        reason="eligible_embedded_photo",
    )


def route_image_assets(
    assets: tuple[PdfImageAsset, ...],
    config: AnalysisConfig,
) -> tuple[RoutedImageAsset, ...]:
    """Route an image inventory in stable document order.

    Analysis quotas are applied by detectors after role selection, so repeated
    logos cannot consume the budget reserved for eligible photos.
    """

    return tuple(route_image_asset(asset, config) for asset in assets)
