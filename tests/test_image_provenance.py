from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from fraude_detector.image_provenance import (
    AiDeclarationTrust,
    C2paManifestData,
    C2paPythonAdapter,
    C2paValidation,
    EvidenceState,
    MetadataEntry,
    MetadataSection,
    PillowMetadataSummary,
    analyze_c2pa_provenance,
    analyze_image_provenance,
    find_ai_metadata_markers,
    normalize_image_media_type,
)


class StubC2paAdapter:
    def __init__(self, result: C2paManifestData | None) -> None:
        self.result = result
        self.calls: list[tuple[bytes, str]] = []

    def read(self, image_bytes: bytes, media_type: str) -> C2paManifestData | None:
        self.calls.append((image_bytes, media_type))
        return self.result


class FailingC2paAdapter:
    def read(self, image_bytes: bytes, media_type: str) -> C2paManifestData | None:
        raise RuntimeError("native validator failed")


class FakeOfficialReader:
    def __init__(self) -> None:
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.closed = True

    def json(self) -> str:
        return json.dumps({"active_manifest": "urn:c2pa:official-api"})

    def get_active_manifest(self) -> dict[str, str]:
        return {"claim_generator": "official-test"}

    def get_validation_state(self) -> str:
        return "Trusted"

    def get_validation_results(self) -> dict[str, object]:
        return {"activeManifest": {"failure": [], "success": []}}

    def is_embedded(self) -> bool:
        return True

    def get_remote_url(self) -> None:
        return None


def _image_bytes(
    image_format: str = "PNG",
    *,
    with_metadata: bool = False,
) -> bytes:
    image = Image.new("RGB", (16, 12), "white")
    output = BytesIO()
    save_options: dict[str, object] = {}
    if with_metadata:
        exif = Image.Exif()
        exif[271] = "Synthetic Camera"
        exif[305] = "Frod tests"
        save_options["exif"] = exif
        save_options["xmp"] = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF/></x:xmpmeta>'
    image.save(output, format=image_format, **save_options)
    return output.getvalue()


def _manifest(
    validation_state: str,
    *,
    digital_source_type: str = (
        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    ),
    validation_results: dict[str, object] | None = None,
) -> C2paManifestData:
    return C2paManifestData(
        active_manifest={
            "claim_generator": "Synthetic Studio/1.0",
            "assertions": [
                {
                    "label": "c2pa.actions.v2",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "digitalSourceType": digital_source_type,
                            }
                        ]
                    },
                }
            ],
        },
        active_manifest_label="urn:c2pa:test",
        validation_state=validation_state,
        validation_results=validation_results,
        embedded=True,
    )


def test_absent_metadata_and_manifest_are_not_detections() -> None:
    image_bytes = _image_bytes()
    adapter = StubC2paAdapter(None)

    report = analyze_image_provenance(image_bytes, ".png", c2pa_adapter=adapter)

    assert report.pillow.state is EvidenceState.NOT_DETECTED
    assert report.pillow.exif.state is EvidenceState.NOT_DETECTED
    assert report.pillow.xmp.state is EvidenceState.NOT_DETECTED
    assert report.c2pa.state is EvidenceState.NOT_DETECTED
    assert report.c2pa.validation is C2paValidation.NOT_APPLICABLE
    assert report.c2pa.ai_declaration.state is EvidenceState.NOT_DETECTED
    assert adapter.calls == [(image_bytes, "image/png")]


def test_pixel_limit_abstains_before_decode_but_still_checks_c2pa() -> None:
    image_bytes = _image_bytes()
    adapter = StubC2paAdapter(None)

    report = analyze_image_provenance(
        image_bytes,
        ".png",
        c2pa_adapter=adapter,
        max_pixels=100,
    )

    assert report.pillow.state is EvidenceState.INDETERMINATE
    assert report.pillow.error == "image_pixel_limit_exceeded"
    assert report.pillow.width == 16
    assert report.pillow.height == 12
    assert report.c2pa.state is EvidenceState.NOT_DETECTED
    assert adapter.calls == [(image_bytes, "image/png")]


def test_extracts_exif_xmp_and_pillow_info() -> None:
    report = analyze_image_provenance(
        _image_bytes("JPEG", with_metadata=True),
        "JPEG",
        c2pa_adapter=StubC2paAdapter(None),
    )

    exif = {entry.key: entry.value for entry in report.pillow.exif.entries}
    xmp = {entry.key: entry.value for entry in report.pillow.xmp.entries}
    info = {entry.key: entry.value for entry in report.pillow.info.entries}

    assert report.pillow.state is EvidenceState.DETECTED
    assert exif["Make"] == "Synthetic Camera"
    assert exif["Software"] == "Frod tests"
    assert "x:xmpmeta" in xmp["raw"]
    assert "jfif" in info
    assert "exif" not in info
    assert "xmp" not in info


def test_official_adapter_uses_reader_try_create_and_current_summary_api() -> None:
    reader = FakeOfficialReader()
    calls: list[tuple[str, bytes]] = []
    contexts: list[object] = []
    settings: list[dict[str, object]] = []

    class FakeContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

    class ContextFactory:
        @staticmethod
        def from_dict(config: dict[str, object]) -> FakeContext:
            settings.append(config)
            return FakeContext()

    class ReaderFactory:
        @staticmethod
        def try_create(
            media_type: str,
            stream: BytesIO,
            *,
            context: object,
        ) -> FakeOfficialReader:
            calls.append((media_type, stream.read()))
            contexts.append(context)
            return reader

    adapter = C2paPythonAdapter(
        SimpleNamespace(
            Context=ContextFactory,
            Reader=ReaderFactory,
        )
    )
    result = adapter.read(b"encoded image", "image/jpeg")

    assert calls == [("image/jpeg", b"encoded image")]
    assert len(contexts) == 1
    assert settings == [
        {
            "verify": {
                "remote_manifest_fetch": False,
                "ocsp_fetch": False,
            }
        }
    ]
    assert reader.closed is True
    assert result is not None
    assert result.active_manifest_label == "urn:c2pa:official-api"
    assert result.validation_state == "Trusted"
    assert result.embedded is True


@pytest.mark.parametrize(
    ("validation_state", "expected_validation", "expected_trust"),
    [
        ("Trusted", C2paValidation.TRUSTED, AiDeclarationTrust.TRUSTED),
        ("Valid", C2paValidation.VALID, AiDeclarationTrust.VALID),
    ],
)
def test_ai_declaration_preserves_c2pa_validation_strength(
    validation_state: str,
    expected_validation: C2paValidation,
    expected_trust: AiDeclarationTrust,
) -> None:
    report = analyze_image_provenance(
        _image_bytes(),
        "image/png",
        c2pa_adapter=StubC2paAdapter(_manifest(validation_state)),
    )

    assert report.c2pa.state is EvidenceState.DETECTED
    assert report.c2pa.validation is expected_validation
    assert report.c2pa.ai_declaration.state is EvidenceState.DETECTED
    assert report.c2pa.ai_declaration.trust is expected_trust
    assert report.c2pa.claim_generator == "Synthetic Studio/1.0"


@pytest.mark.parametrize(
    "digital_source_type",
    [
        "algorithmicallyEnhanced",
        "algorithmicMedia",
        "compositeSynthetic",
    ],
)
def test_non_trained_algorithmic_c2pa_types_are_not_called_ai(
    digital_source_type: str,
) -> None:
    report = analyze_image_provenance(
        _image_bytes(),
        "image/png",
        c2pa_adapter=StubC2paAdapter(
            _manifest(
                "Trusted",
                digital_source_type=(
                    f"http://cv.iptc.org/newscodes/digitalsourcetype/{digital_source_type}"
                ),
            )
        ),
    )

    assert report.c2pa.state is EvidenceState.DETECTED
    assert report.c2pa.ai_declaration.state is EvidenceState.NOT_DETECTED
    assert report.c2pa.ai_declaration.trust is AiDeclarationTrust.NOT_APPLICABLE


def test_valid_manifest_with_untrusted_signer_is_reported_as_untrusted() -> None:
    results = {
        "activeManifest": {
            "failure": [
                {
                    "code": "signingCredential.untrusted",
                    "explanation": "signing certificate untrusted",
                }
            ],
            "success": [{"code": "claimSignature.validated"}],
        }
    }
    report = analyze_image_provenance(
        _image_bytes(),
        "PNG",
        c2pa_adapter=StubC2paAdapter(_manifest("Valid", validation_results=results)),
    )

    assert report.c2pa.raw_validation_state == "Valid"
    assert report.c2pa.validation is C2paValidation.UNTRUSTED
    assert report.c2pa.ai_declaration.trust is AiDeclarationTrust.UNTRUSTED
    assert report.c2pa.statuses[0].code == "signingCredential.untrusted"


def test_invalid_provenance_invalidates_ai_declaration() -> None:
    results = {
        "activeManifest": {
            "failure": [
                {
                    "code": "assertion.dataHash.mismatch",
                    "explanation": "asset bytes changed",
                }
            ]
        }
    }
    report = analyze_image_provenance(
        _image_bytes(),
        "PNG",
        c2pa_adapter=StubC2paAdapter(_manifest("Invalid", validation_results=results)),
    )

    assert report.c2pa.validation is C2paValidation.INVALIDATED
    assert report.c2pa.provenance_invalidated is True
    assert report.c2pa.ai_declaration.trust is AiDeclarationTrust.INVALIDATED


def test_non_ai_digital_source_type_is_not_an_ai_declaration() -> None:
    report = analyze_image_provenance(
        _image_bytes(),
        "PNG",
        c2pa_adapter=StubC2paAdapter(
            _manifest(
                "Trusted",
                digital_source_type=(
                    "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
                ),
            )
        ),
    )

    assert report.c2pa.state is EvidenceState.DETECTED
    assert report.c2pa.ai_declaration.state is EvidenceState.NOT_DETECTED
    assert report.c2pa.ai_declaration.trust is AiDeclarationTrust.NOT_APPLICABLE


def test_optional_c2pa_sdk_unavailable_does_not_fail_analysis(monkeypatch) -> None:
    monkeypatch.setattr(C2paPythonAdapter, "try_load", classmethod(lambda cls: None))

    report = analyze_image_provenance(_image_bytes(), "PNG")

    assert report.c2pa.state is EvidenceState.UNAVAILABLE
    assert report.c2pa.validation is C2paValidation.UNAVAILABLE
    assert report.c2pa.ai_declaration.state is EvidenceState.UNAVAILABLE


def test_c2pa_adapter_error_is_indeterminate_not_an_exception() -> None:
    report = analyze_image_provenance(
        _image_bytes(),
        "PNG",
        c2pa_adapter=FailingC2paAdapter(),
    )

    assert report.c2pa.state is EvidenceState.INDETERMINATE
    assert report.c2pa.validation is C2paValidation.UNKNOWN
    assert report.c2pa.error == "RuntimeError: native validator failed"


def test_unreadable_image_metadata_is_indeterminate_without_blocking_c2pa() -> None:
    report = analyze_image_provenance(
        b"not an image",
        "PNG",
        c2pa_adapter=StubC2paAdapter(None),
    )

    assert report.pillow.state is EvidenceState.INDETERMINATE
    assert report.c2pa.state is EvidenceState.NOT_DETECTED


@pytest.mark.parametrize(
    ("image_format", "expected"),
    [
        ("JPG", "image/jpeg"),
        (".tiff", "image/tiff"),
        ("image/avif", "image/avif"),
    ],
)
def test_normalize_image_media_type(image_format: str, expected: str) -> None:
    assert normalize_image_media_type(image_format) == expected


def test_c2pa_can_be_analyzed_on_a_complete_pdf_asset() -> None:
    adapter = StubC2paAdapter(_manifest("Trusted"))

    summary = analyze_c2pa_provenance(
        b"%PDF-1.7 synthetic test",
        "application/pdf",
        c2pa_adapter=adapter,
    )

    assert summary.ai_declaration.state is EvidenceState.DETECTED
    assert adapter.calls == [(b"%PDF-1.7 synthetic test", "application/pdf")]


def test_explicit_ai_tool_names_are_extracted_from_unsigned_metadata() -> None:
    empty = MetadataSection(state=EvidenceState.NOT_DETECTED)
    summary = PillowMetadataSummary(
        state=EvidenceState.DETECTED,
        decoded_format="PNG",
        width=512,
        height=512,
        mode="RGB",
        exif=empty,
        xmp=empty,
        info=MetadataSection(
            state=EvidenceState.DETECTED,
            entries=(
                MetadataEntry(
                    key="parameters",
                    value="Stable Diffusion; Steps: 20, Sampler: Euler",
                ),
            ),
        ),
    )

    markers = find_ai_metadata_markers(summary)

    assert {marker.marker for marker in markers} == {
        "diffusion_generation_parameters",
        "stable_diffusion",
    }


def test_generic_metadata_does_not_create_ai_markers() -> None:
    report = analyze_image_provenance(
        _image_bytes("JPEG", with_metadata=True),
        "JPEG",
        c2pa_adapter=StubC2paAdapter(None),
    )

    assert find_ai_metadata_markers(report.pillow) == ()
