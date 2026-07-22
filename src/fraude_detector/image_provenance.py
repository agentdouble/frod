"""Explainable provenance analysis for standalone image assets.

The module deliberately does not turn missing metadata into a fraud signal.  It
only reports what can be observed in Pillow metadata and in a C2PA manifest.
The native C2PA dependency is loaded lazily so an import problem degrades one
detector instead of breaking the complete PDF pipeline.
"""

from __future__ import annotations

import importlib
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from typing import Any, Protocol

from PIL import ExifTags, Image, UnidentifiedImageError


class EvidenceState(StrEnum):
    """Whether an evidence source yielded an observable result."""

    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    INDETERMINATE = "indeterminate"
    UNAVAILABLE = "unavailable"


class C2paValidation(StrEnum):
    """Normalized integrity/trust state derived from the C2PA SDK."""

    TRUSTED = "trusted"
    VALID = "valid"
    UNTRUSTED = "untrusted"
    INVALIDATED = "invalidated"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


class AiDeclarationTrust(StrEnum):
    """How strongly a C2PA AI declaration can be relied upon."""

    TRUSTED = "trusted"
    VALID = "valid"
    UNTRUSTED = "untrusted"
    INVALIDATED = "invalidated"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class MetadataEntry:
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class MetadataSection:
    state: EvidenceState
    entries: tuple[MetadataEntry, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PillowMetadataSummary:
    state: EvidenceState
    decoded_format: str | None
    width: int | None
    height: int | None
    mode: str | None
    exif: MetadataSection
    xmp: MetadataSection
    info: MetadataSection
    error: str | None = None


@dataclass(frozen=True, slots=True)
class C2paValidationStatus:
    outcome: str
    code: str
    explanation: str | None = None


@dataclass(frozen=True, slots=True)
class C2paManifestData:
    """Normalized data returned by a C2PA reader adapter."""

    active_manifest: Mapping[str, Any] | None
    active_manifest_label: str | None
    validation_state: str | None
    validation_results: Mapping[str, Any] | None = None
    embedded: bool | None = None
    remote_url: str | None = None


class C2paAdapter(Protocol):
    """Replaceable boundary around a C2PA implementation."""

    def read(self, image_bytes: bytes, media_type: str) -> C2paManifestData | None:
        """Return normalized manifest data, or ``None`` when none is present."""
        ...


@dataclass(frozen=True, slots=True)
class AiDeclaration:
    state: EvidenceState
    trust: AiDeclarationTrust
    digital_source_types: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class C2paSummary:
    state: EvidenceState
    validation: C2paValidation
    raw_validation_state: str | None
    active_manifest_label: str | None
    claim_generator: str | None
    embedded: bool | None
    remote_url: str | None
    statuses: tuple[C2paValidationStatus, ...]
    ai_declaration: AiDeclaration
    error: str | None = None

    @property
    def provenance_invalidated(self) -> bool:
        return self.validation is C2paValidation.INVALIDATED


@dataclass(frozen=True, slots=True)
class ImageProvenanceReport:
    requested_format: str
    media_type: str
    pillow: PillowMetadataSummary
    c2pa: C2paSummary


@dataclass(frozen=True, slots=True)
class AiMetadataMarker:
    """One explicit generator reference found in unsigned image metadata."""

    marker: str
    metadata_key: str


def ai_declaration_weight(trust: AiDeclarationTrust) -> tuple[float, float]:
    """Return review severity and confidence for an explicit C2PA AI declaration.

    A valid declaration is a strong business-review signal even when its signer is
    not in the local trust store. Trust changes confidence, not whether the declared
    algorithmic origin should trigger an immediate review.
    """

    return {
        AiDeclarationTrust.TRUSTED: (45.0, 0.99),
        AiDeclarationTrust.VALID: (45.0, 0.85),
        AiDeclarationTrust.UNTRUSTED: (40.0, 0.65),
        AiDeclarationTrust.UNKNOWN: (30.0, 0.55),
        AiDeclarationTrust.NOT_APPLICABLE: (0.0, 0.0),
        AiDeclarationTrust.INVALIDATED: (0.0, 0.0),
    }[trust]


class C2paPythonAdapter:
    """Adapter for the current official ``c2pa-python`` Reader API."""

    _OFFLINE_SETTINGS = {
        "verify": {
            "remote_manifest_fetch": False,
            "ocsp_fetch": False,
        }
    }

    def __init__(self, c2pa_module: Any) -> None:
        self._c2pa = c2pa_module

    @classmethod
    def try_load(cls) -> C2paPythonAdapter | None:
        """Load the optional native SDK without leaking import failures."""

        try:
            module = importlib.import_module("c2pa")
        except Exception:
            # Native-library loading errors can be OSError or another SDK-specific
            # exception, not only ImportError.
            return None
        return cls(module)

    def read(self, image_bytes: bytes, media_type: str) -> C2paManifestData | None:
        """Read embedded provenance without contacting manifest or OCSP servers."""

        with self._c2pa.Context.from_dict(self._OFFLINE_SETTINGS) as context:
            reader = self._c2pa.Reader.try_create(
                media_type,
                BytesIO(image_bytes),
                context=context,
            )
            if reader is None:
                return None

            with reader:
                manifest_store = json.loads(reader.json())
                active_label = _optional_string(manifest_store.get("active_manifest"))
                return C2paManifestData(
                    active_manifest=reader.get_active_manifest(),
                    active_manifest_label=active_label,
                    validation_state=reader.get_validation_state(),
                    validation_results=reader.get_validation_results(),
                    embedded=reader.is_embedded(),
                    remote_url=reader.get_remote_url(),
                )


_AI_DIGITAL_SOURCE_TYPES = frozenset(
    {
        "compositewithtrainedalgorithmicmedia",
        "trainedalgorithmicmedia",
    }
)

_FORMAT_TO_MEDIA_TYPE = {
    "avif": "image/avif",
    "bmp": "image/bmp",
    "gif": "image/gif",
    "heic": "image/heic",
    "heif": "image/heif",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "webp": "image/webp",
}

_AI_METADATA_MARKERS = {
    "automatic1111": "automatic1111",
    "comfyui": "comfyui",
    "dall-e": "dall-e",
    "firefly": "adobe_firefly",
    "fooocus": "fooocus",
    "gpt image": "gpt_image",
    "invokeai": "invokeai",
    "midjourney": "midjourney",
    "novelai": "novelai",
    "stable diffusion": "stable_diffusion",
    "stability ai": "stability_ai",
}


def analyze_image_provenance(
    image_bytes: bytes,
    image_format: str,
    *,
    c2pa_adapter: C2paAdapter | None = None,
    max_pixels: int | None = None,
) -> ImageProvenanceReport:
    """Analyze metadata and optional C2PA provenance for one encoded image.

    ``image_format`` accepts a MIME type, a file extension, or a Pillow format
    name.  Supplying an adapter makes C2PA behavior deterministic in tests or
    allows another implementation to replace ``c2pa-python``.
    """

    if not isinstance(image_bytes, bytes):
        raise TypeError("image_bytes must be bytes")
    if not image_format.strip():
        raise ValueError("image_format must not be empty")
    if max_pixels is not None and max_pixels < 1:
        raise ValueError("max_pixels must be at least 1")

    media_type = normalize_image_media_type(image_format)
    pillow = _analyze_pillow_metadata(image_bytes, max_pixels=max_pixels)

    c2pa = analyze_c2pa_provenance(
        image_bytes,
        media_type,
        c2pa_adapter=c2pa_adapter,
    )
    return ImageProvenanceReport(
        requested_format=image_format,
        media_type=media_type,
        pillow=pillow,
        c2pa=c2pa,
    )


def analyze_c2pa_provenance(
    asset_bytes: bytes,
    media_type: str,
    *,
    c2pa_adapter: C2paAdapter | None = None,
) -> C2paSummary:
    """Analyze C2PA on any supported asset, including a complete PDF."""

    if not isinstance(asset_bytes, bytes):
        raise TypeError("asset_bytes must be bytes")
    if not media_type.strip():
        raise ValueError("media_type must not be empty")

    adapter = c2pa_adapter
    if adapter is None:
        adapter = C2paPythonAdapter.try_load()
    return _analyze_c2pa(asset_bytes, media_type.strip().lower(), adapter)


def find_ai_metadata_markers(
    summary: PillowMetadataSummary,
) -> tuple[AiMetadataMarker, ...]:
    """Find explicit AI-tool names without treating missing metadata as evidence."""

    markers: set[tuple[str, str]] = set()
    sections = (summary.exif, summary.xmp, summary.info)
    for section in sections:
        for entry in section.entries:
            text = f"{entry.key} {entry.value}".casefold()
            for needle, marker in _AI_METADATA_MARKERS.items():
                if needle in text:
                    markers.add((marker, entry.key))
            if "parameters" in entry.key.casefold() and (
                "negative prompt" in text or ("sampler" in text and "steps" in text)
            ):
                markers.add(("diffusion_generation_parameters", entry.key))

    return tuple(
        AiMetadataMarker(marker=marker, metadata_key=key) for marker, key in sorted(markers)
    )


def normalize_image_media_type(image_format: str) -> str:
    """Normalize an extension or Pillow format name to a MIME type."""

    normalized = image_format.strip().lower()
    if "/" in normalized:
        return normalized
    extension = normalized.removeprefix(".")
    return _FORMAT_TO_MEDIA_TYPE.get(extension, f"image/{extension}")


def _analyze_pillow_metadata(
    image_bytes: bytes,
    *,
    max_pixels: int | None,
) -> PillowMetadataSummary:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            image_context = Image.open(BytesIO(image_bytes))
        with image_context as image:
            if max_pixels is not None and image.width * image.height > max_pixels:
                error = "image_pixel_limit_exceeded"
                section = MetadataSection(
                    state=EvidenceState.INDETERMINATE,
                    error=error,
                )
                return PillowMetadataSummary(
                    state=EvidenceState.INDETERMINATE,
                    decoded_format=image.format,
                    width=image.width,
                    height=image.height,
                    mode=image.mode,
                    exif=section,
                    xmp=section,
                    info=section,
                    error=error,
                )
            image.load()
            exif = _extract_exif(image)
            xmp = _extract_xmp(image)
            info = _extract_pillow_info(image)
            sections = (exif, xmp, info)
            if any(section.state is EvidenceState.DETECTED for section in sections):
                state = EvidenceState.DETECTED
            elif any(section.state is EvidenceState.INDETERMINATE for section in sections):
                state = EvidenceState.INDETERMINATE
            else:
                state = EvidenceState.NOT_DETECTED
            return PillowMetadataSummary(
                state=state,
                decoded_format=image.format,
                width=image.width,
                height=image.height,
                mode=image.mode,
                exif=exif,
                xmp=xmp,
                info=info,
            )
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as error:
        message = _safe_error(error)
        section = MetadataSection(state=EvidenceState.INDETERMINATE, error=message)
        return PillowMetadataSummary(
            state=EvidenceState.INDETERMINATE,
            decoded_format=None,
            width=None,
            height=None,
            mode=None,
            exif=section,
            xmp=section,
            info=section,
            error=message,
        )


def _extract_exif(image: Image.Image) -> MetadataSection:
    try:
        exif = image.getexif()
        entries = tuple(
            sorted(
                (
                    MetadataEntry(
                        key=str(ExifTags.TAGS.get(tag_id, tag_id)),
                        value=_safe_metadata_value(value),
                    )
                    for tag_id, value in exif.items()
                ),
                key=lambda entry: entry.key,
            )
        )
    except (OSError, TypeError, ValueError) as error:
        return MetadataSection(
            state=EvidenceState.INDETERMINATE,
            error=_safe_error(error),
        )
    return _metadata_section(entries)


def _extract_xmp(image: Image.Image) -> MetadataSection:
    raw_xmp = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp")
    if raw_xmp:
        return _metadata_section((MetadataEntry("raw", _safe_metadata_value(raw_xmp)),))

    get_xmp = getattr(image, "getxmp", None)
    if get_xmp is None:
        return MetadataSection(state=EvidenceState.NOT_DETECTED)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = get_xmp()
    except (OSError, TypeError, ValueError) as error:
        return MetadataSection(
            state=EvidenceState.INDETERMINATE,
            error=_safe_error(error),
        )

    entries = tuple(
        MetadataEntry(key=key, value=_safe_metadata_value(value))
        for key, value in _flatten_mapping(parsed)
    )
    return _metadata_section(entries)


def _extract_pillow_info(image: Image.Image) -> MetadataSection:
    entries = tuple(
        MetadataEntry(key=str(key), value=_safe_metadata_value(value))
        for key, value in sorted(image.info.items(), key=lambda item: str(item[0]))
        if key not in {"exif", "xmp", "XML:com.adobe.xmp"}
    )
    return _metadata_section(entries)


def _analyze_c2pa(
    image_bytes: bytes,
    media_type: str,
    adapter: C2paAdapter | None,
) -> C2paSummary:
    if adapter is None:
        return C2paSummary(
            state=EvidenceState.UNAVAILABLE,
            validation=C2paValidation.UNAVAILABLE,
            raw_validation_state=None,
            active_manifest_label=None,
            claim_generator=None,
            embedded=None,
            remote_url=None,
            statuses=(),
            ai_declaration=AiDeclaration(
                state=EvidenceState.UNAVAILABLE,
                trust=AiDeclarationTrust.UNKNOWN,
            ),
            error="c2pa-python is not installed or could not be loaded",
        )

    try:
        manifest = adapter.read(image_bytes, media_type)
    except Exception as error:
        return C2paSummary(
            state=EvidenceState.INDETERMINATE,
            validation=C2paValidation.UNKNOWN,
            raw_validation_state=None,
            active_manifest_label=None,
            claim_generator=None,
            embedded=None,
            remote_url=None,
            statuses=(),
            ai_declaration=AiDeclaration(
                state=EvidenceState.INDETERMINATE,
                trust=AiDeclarationTrust.UNKNOWN,
            ),
            error=_safe_error(error),
        )

    if manifest is None:
        return C2paSummary(
            state=EvidenceState.NOT_DETECTED,
            validation=C2paValidation.NOT_APPLICABLE,
            raw_validation_state=None,
            active_manifest_label=None,
            claim_generator=None,
            embedded=None,
            remote_url=None,
            statuses=(),
            ai_declaration=AiDeclaration(
                state=EvidenceState.NOT_DETECTED,
                trust=AiDeclarationTrust.NOT_APPLICABLE,
            ),
        )

    statuses = _extract_validation_statuses(manifest.validation_results)
    validation = _classify_validation(manifest.validation_state, statuses)
    ai_declaration = _classify_ai_declaration(manifest.active_manifest, validation)
    claim_generator = None
    if manifest.active_manifest is not None:
        claim_generator = _optional_string(manifest.active_manifest.get("claim_generator"))

    return C2paSummary(
        state=EvidenceState.DETECTED,
        validation=validation,
        raw_validation_state=manifest.validation_state,
        active_manifest_label=manifest.active_manifest_label,
        claim_generator=claim_generator,
        embedded=manifest.embedded,
        remote_url=manifest.remote_url,
        statuses=statuses,
        ai_declaration=ai_declaration,
    )


def _classify_validation(
    raw_state: str | None,
    statuses: tuple[C2paValidationStatus, ...],
) -> C2paValidation:
    normalized = (raw_state or "").strip().casefold()
    failure_codes = {
        status.code.casefold() for status in statuses if status.outcome.casefold() == "failure"
    }
    untrusted_only = bool(failure_codes) and all("untrusted" in code for code in failure_codes)

    if normalized in {"invalid", "invalidated"} or (failure_codes and not untrusted_only):
        return C2paValidation.INVALIDATED
    if normalized == "trusted":
        return C2paValidation.TRUSTED
    if normalized == "untrusted" or untrusted_only:
        return C2paValidation.UNTRUSTED
    if normalized == "valid":
        return C2paValidation.VALID
    return C2paValidation.UNKNOWN


def _classify_ai_declaration(
    active_manifest: Mapping[str, Any] | None,
    validation: C2paValidation,
) -> AiDeclaration:
    if active_manifest is None:
        return AiDeclaration(
            state=EvidenceState.INDETERMINATE,
            trust=AiDeclarationTrust.UNKNOWN,
        )

    source_types = tuple(sorted(set(_find_digital_source_types(active_manifest))))
    ai_source_types = tuple(
        source_type for source_type in source_types if _is_ai_source_type(source_type)
    )
    if not ai_source_types:
        return AiDeclaration(
            state=EvidenceState.NOT_DETECTED,
            trust=AiDeclarationTrust.NOT_APPLICABLE,
        )

    trust_by_validation = {
        C2paValidation.TRUSTED: AiDeclarationTrust.TRUSTED,
        C2paValidation.VALID: AiDeclarationTrust.VALID,
        C2paValidation.UNTRUSTED: AiDeclarationTrust.UNTRUSTED,
        C2paValidation.INVALIDATED: AiDeclarationTrust.INVALIDATED,
    }
    return AiDeclaration(
        state=EvidenceState.DETECTED,
        trust=trust_by_validation.get(validation, AiDeclarationTrust.UNKNOWN),
        digital_source_types=ai_source_types,
    )


def _extract_validation_statuses(
    validation_results: Mapping[str, Any] | None,
) -> tuple[C2paValidationStatus, ...]:
    if not validation_results:
        return ()

    statuses: list[C2paValidationStatus] = []

    def visit(value: Any, outcome: str | None = None) -> None:
        if isinstance(value, Mapping):
            code = value.get("code")
            if outcome and isinstance(code, str):
                statuses.append(
                    C2paValidationStatus(
                        outcome=outcome,
                        code=code,
                        explanation=_optional_string(value.get("explanation")),
                    )
                )
            for key, child in value.items():
                next_outcome = (
                    str(key) if key in {"success", "informational", "failure"} else outcome
                )
                visit(child, next_outcome)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, outcome)

    visit(validation_results)
    return tuple(statuses)


def _find_digital_source_types(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).replace("_", "").casefold()
            if normalized_key == "digitalsourcetype" and isinstance(child, str):
                found.append(child)
            else:
                found.extend(_find_digital_source_types(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_find_digital_source_types(child))
    return found


def _is_ai_source_type(source_type: str) -> bool:
    suffix = source_type.rstrip("/").rsplit("/", 1)[-1].casefold()
    return suffix in _AI_DIGITAL_SOURCE_TYPES


def _metadata_section(entries: tuple[MetadataEntry, ...]) -> MetadataSection:
    state = EvidenceState.DETECTED if entries else EvidenceState.NOT_DETECTED
    return MetadataSection(state=state, entries=entries)


def _flatten_mapping(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if not isinstance(value, Mapping):
        return [(prefix or "value", value)]

    entries: list[tuple[str, Any]] = []
    for key, child in sorted(value.items(), key=lambda item: str(item[0])):
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            entries.extend(_flatten_mapping(child, path))
        else:
            entries.append((path, child))
    return entries


def _safe_metadata_value(value: Any, *, limit: int = 500) -> str:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary:{len(value)} bytes>"
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_error(error: Exception) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
