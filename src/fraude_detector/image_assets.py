"""Inventory native image assets embedded in PDF pages."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, Protocol

import pypdfium2
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageOps, UnidentifiedImageError

from fraude_detector.geometry import bbox_coverage, pdfium_bounds_to_top_left
from fraude_detector.models import BoundingBox

NativeDataStatus = Literal["available", "unavailable"]
DecodedImageSource = Literal["native", "pdfium"]


class ImageAssetDecodeError(RuntimeError):
    """Raised when an inventoried PDF image cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class DecodedImageAsset:
    """Detached RGB pixels ready for a passive image analyzer."""

    image: Image.Image
    source: DecodedImageSource


class ImageAssetContext(Protocol):
    """Minimum analysis context required to inventory images."""

    pdfium_document: pypdfium2.PdfDocument

    @property
    def analyzed_page_count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class PdfImageAsset:
    """One image occurrence, detached from the lifetime of its PDF page.

    ``page`` and ``index`` are one-based. Coordinates use PDF points with an
    origin at the top-left, consistently with findings emitted by detectors.
    The hash covers ``native_bytes`` and is therefore absent when PDFium cannot
    expose a standalone encoded image without recompression.
    """

    page: int
    index: int
    bbox: BoundingBox
    coverage: float
    pixel_size: tuple[int, int] | None
    filters: tuple[str, ...]
    sha256: str | None
    format: str | None
    mime_type: str | None
    native_bytes: bytes | None
    native_status: NativeDataStatus
    native_reason: str | None

    def __post_init__(self) -> None:
        if self.page < 1 or self.index < 1:
            raise ValueError("page and index must be one-based positive integers")
        if not 0 <= self.coverage <= 1:
            raise ValueError("coverage must be between 0 and 1")
        if self.native_status == "available":
            if self.native_bytes is None or self.sha256 is None:
                raise ValueError("available native data requires bytes and sha256")
            if self.native_reason is not None:
                raise ValueError("available native data cannot have an error reason")
        else:
            if self.native_reason is None:
                raise ValueError("unavailable native data requires an error reason")
            if self.native_bytes is not None or self.sha256 is not None:
                raise ValueError("unavailable native data cannot expose bytes or sha256")


def extract_image_assets(
    source: pypdfium2.PdfDocument | ImageAssetContext,
    *,
    max_pages: int | None = None,
    max_images: int | None = None,
) -> tuple[PdfImageAsset, ...]:
    """Extract image occurrences from a PDF document or analysis context.

    DCT (JPEG) and JPX (JPEG 2000) payloads are returned without recompression.
    Other encodings remain inventoried but are marked unavailable because their
    PDF streams are not independently usable image files.
    """

    document, context_page_limit = _resolve_source(source)
    if max_pages is not None and max_pages < 0:
        raise ValueError("max_pages must be greater than or equal to zero")
    if max_images is not None and max_images < 0:
        raise ValueError("max_images must be greater than or equal to zero")
    if max_images == 0:
        return ()

    page_limit = len(document)
    if context_page_limit is not None:
        page_limit = min(page_limit, context_page_limit)
    if max_pages is not None:
        page_limit = min(page_limit, max_pages)

    assets: list[PdfImageAsset] = []
    native_cache: dict[tuple[str, int], _NativeData] = {}
    for page_index in range(page_limit):
        page = document[page_index]
        try:
            page_width, page_height = page.get_size()
            image_objects = page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])
            for image_index, image_object in enumerate(image_objects, start=1):
                asset = _describe_image(
                    image_object,
                    page=page_index + 1,
                    index=image_index,
                    page_width=page_width,
                    page_height=page_height,
                    native_cache=native_cache,
                )
                if asset is not None:
                    assets.append(asset)
                    if max_images is not None and len(assets) >= max_images:
                        return tuple(assets)
        finally:
            page.close()
    return tuple(assets)


def decode_image_asset(
    source: pypdfium2.PdfDocument | ImageAssetContext,
    asset: PdfImageAsset,
    *,
    max_pixels: int,
) -> DecodedImageAsset:
    """Decode an inventoried image without changing its native evidence bytes.

    Native JPEG/JPX bytes are preferred. For PDF stream encodings that are not
    standalone files, PDFium decodes the image stream without applying its page
    transform. Separate alpha masks are not recomposed. The returned Pillow image
    owns its pixels and outlives the PDF page handle.
    """

    if max_pixels < 1:
        raise ValueError("max_pixels must be at least 1")
    if asset.pixel_size is not None and asset.pixel_size[0] * asset.pixel_size[1] > max_pixels:
        raise ImageAssetDecodeError("image_pixel_limit_exceeded")

    if asset.native_bytes is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                native_image_context = Image.open(BytesIO(asset.native_bytes))
            with native_image_context as native_image:
                if native_image.width * native_image.height > max_pixels:
                    raise ImageAssetDecodeError("image_pixel_limit_exceeded")
                native_image.load()
                image = ImageOps.exif_transpose(native_image).convert("RGB")
                image.load()
        except ImageAssetDecodeError:
            raise
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as error:
            raise ImageAssetDecodeError(_error_reason("native_decode_failed", error)) from error
        return DecodedImageAsset(image=image, source="native")

    document, _ = _resolve_source(source)
    if asset.page > len(document):
        raise ImageAssetDecodeError("page_index_out_of_range")

    page = document[asset.page - 1]
    try:
        image_objects = tuple(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]))
        if asset.index > len(image_objects):
            raise ImageAssetDecodeError("image_index_out_of_range")
        # Decode at the stream's native dimensions. Applying the page transform
        # can rotate or resample the object according to its placement matrix,
        # which would manufacture traces before passive analysis.
        bitmap = image_objects[asset.index - 1].get_bitmap(render=False)
        try:
            image = bitmap.to_pil().convert("RGB").copy()
        finally:
            bitmap.close()
    except ImageAssetDecodeError:
        raise
    except Exception as error:
        raise ImageAssetDecodeError(_error_reason("pdfium_decode_failed", error)) from error
    finally:
        page.close()

    if image.width * image.height > max_pixels:
        raise ImageAssetDecodeError("image_pixel_limit_exceeded")
    return DecodedImageAsset(image=image, source="pdfium")


def _resolve_source(
    source: pypdfium2.PdfDocument | ImageAssetContext,
) -> tuple[pypdfium2.PdfDocument, int | None]:
    if isinstance(source, pypdfium2.PdfDocument):
        return source, None

    try:
        document = source.pdfium_document
        page_limit = source.analyzed_page_count
    except AttributeError as error:
        raise TypeError("source must be a PdfDocument or an image asset context") from error
    if not isinstance(document, pypdfium2.PdfDocument):
        raise TypeError("source.pdfium_document must be a PdfDocument")
    return document, page_limit


def _describe_image(
    image_object: pypdfium2.PdfImage,
    *,
    page: int,
    index: int,
    page_width: float,
    page_height: float,
    native_cache: dict[tuple[str, int], _NativeData],
) -> PdfImageAsset | None:
    try:
        bbox = pdfium_bounds_to_top_left(image_object.get_bounds(), page_height)
    except Exception:
        return None
    if bbox is None:
        return None

    try:
        pixel_size = tuple(image_object.get_px_size())
    except Exception:
        pixel_size = None

    try:
        filters = tuple(image_object.get_filters())
        filters_error = None
    except Exception as error:
        filters = ()
        filters_error = _error_reason("filter_metadata_unavailable", error)

    native = _extract_native_data(image_object, filters, filters_error)
    if native.sha256 is not None and native.data is not None:
        cache_key = (native.sha256, len(native.data))
        native = native_cache.setdefault(cache_key, native)
    return PdfImageAsset(
        page=page,
        index=index,
        bbox=bbox,
        coverage=bbox_coverage(bbox, page_width, page_height),
        pixel_size=pixel_size,
        filters=filters,
        sha256=native.sha256,
        format=native.format,
        mime_type=native.mime_type,
        native_bytes=native.data,
        native_status=native.status,
        native_reason=native.reason,
    )


@dataclass(frozen=True, slots=True)
class _NativeData:
    data: bytes | None
    sha256: str | None
    format: str | None
    mime_type: str | None
    status: NativeDataStatus
    reason: str | None


def _extract_native_data(
    image_object: pypdfium2.PdfImage,
    filters: tuple[str, ...],
    filters_error: str | None,
) -> _NativeData:
    if filters_error is not None:
        return _unavailable_native_data(reason=filters_error)

    complex_filters = tuple(
        filter_name
        for filter_name in filters
        if filter_name not in pypdfium2.PdfImage.SIMPLE_FILTERS
    )
    encoding = _native_encoding(complex_filters)
    if encoding is None:
        if not complex_filters:
            reason = "no_standalone_native_encoding"
        else:
            reason = f"unsupported_filter_chain:{','.join(complex_filters)}"
        return _unavailable_native_data(reason=reason)

    image_format, mime_type = encoding
    try:
        # PDFium removes simple transport filters (ASCII85, Flate, etc.) but
        # leaves the DCT/JPX payload untouched, so no pixel re-encoding occurs.
        data = bytes(image_object.get_data(decode_simple=True))
    except Exception as error:
        return _unavailable_native_data(
            reason=_error_reason("native_stream_unavailable", error),
            image_format=image_format,
            mime_type=mime_type,
        )
    if not data:
        return _unavailable_native_data(
            reason="native_stream_empty",
            image_format=image_format,
            mime_type=mime_type,
        )
    return _NativeData(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        format=image_format,
        mime_type=mime_type,
        status="available",
        reason=None,
    )


def _native_encoding(complex_filters: tuple[str, ...]) -> tuple[str, str] | None:
    if complex_filters == ("DCTDecode",):
        return "jpeg", "image/jpeg"
    if complex_filters == ("JPXDecode",):
        return "jpeg2000", "image/jp2"
    return None


def _unavailable_native_data(
    *,
    reason: str,
    image_format: str | None = None,
    mime_type: str | None = None,
) -> _NativeData:
    return _NativeData(
        data=None,
        sha256=None,
        format=image_format,
        mime_type=mime_type,
        status="unavailable",
        reason=reason,
    )


def _error_reason(prefix: str, error: Exception) -> str:
    detail = str(error).strip()
    error_name = type(error).__name__
    return f"{prefix}:{error_name}:{detail}" if detail else f"{prefix}:{error_name}"
