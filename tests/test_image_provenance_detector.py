from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pypdfium2
from PIL import Image
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from fraude_detector.config import AnalysisConfig
from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.detectors.image_provenance import ImageProvenanceDetector
from fraude_detector.image_provenance import C2paManifestData


class ConditionalC2paAdapter:
    def __init__(
        self,
        image_result: C2paManifestData | None = None,
        pdf_result: C2paManifestData | None = None,
    ) -> None:
        self.image_result = image_result
        self.pdf_result = pdf_result

    def read(self, image_bytes: bytes, media_type: str) -> C2paManifestData | None:
        return self.pdf_result if media_type == "application/pdf" else self.image_result


def test_absent_provenance_and_metadata_do_not_create_findings(tmp_path: Path) -> None:
    context = _photo_context(tmp_path)
    try:
        result = ImageProvenanceDetector(ConditionalC2paAdapter()).analyze(context)
    finally:
        context.pdfium_document.close()

    assert result.status == "completed"
    assert result.findings == ()
    assert result.artifacts == ()


def test_trusted_ai_declaration_is_localized_to_photo(tmp_path: Path) -> None:
    context = _photo_context(tmp_path)
    try:
        result = ImageProvenanceDetector(
            ConditionalC2paAdapter(image_result=_ai_manifest("Trusted"))
        ).analyze(context)
    finally:
        context.pdfium_document.close()

    finding = next(item for item in result.findings if item.code == "AI_IMAGE_C2PA_DECLARATION")
    assert finding.category == "synthetic_media"
    assert finding.risk_points == 45
    assert finding.page == 1
    assert finding.bbox is not None
    assert finding.evidence["declaration_trust"] == "trusted"
    assert len(finding.artifacts) == 1
    assert (context.output_dir / finding.artifacts[0]).is_file()


def test_explicit_generator_metadata_is_a_weak_signal(tmp_path: Path) -> None:
    context = _photo_context(tmp_path, software="Stable Diffusion WebUI")
    try:
        result = ImageProvenanceDetector(ConditionalC2paAdapter()).analyze(context)
    finally:
        context.pdfium_document.close()

    finding = next(
        item for item in result.findings if item.code == "AI_GENERATOR_METADATA_MENTIONED"
    )
    assert finding.risk_points == 15
    assert finding.evidence["markers"] == ["stable_diffusion"]


def test_invalid_pdf_provenance_is_not_reported_as_ai_origin(tmp_path: Path) -> None:
    context = _photo_context(tmp_path)
    invalid = _ai_manifest(
        "Invalid",
        validation_results={
            "activeManifest": {
                "failure": [{"code": "assertion.dataHash.mismatch"}],
                "success": [{"code": "claimSignature.validated"}],
            }
        },
    )
    try:
        result = ImageProvenanceDetector(ConditionalC2paAdapter(pdf_result=invalid)).analyze(
            context
        )
    finally:
        context.pdfium_document.close()

    codes = {finding.code for finding in result.findings}
    assert "PDF_C2PA_INVALID" in codes
    assert "AI_PDF_C2PA_DECLARATION" not in codes
    invalid_finding = next(item for item in result.findings if item.code == "PDF_C2PA_INVALID")
    assert invalid_finding.evidence["failure_codes"] == ["assertion.dataHash.mismatch"]


def test_full_page_scan_is_still_checked_for_explicit_provenance(tmp_path: Path) -> None:
    context = _photo_context(tmp_path, coverage=1.0)
    try:
        result = ImageProvenanceDetector(
            ConditionalC2paAdapter(image_result=_ai_manifest("Valid"))
        ).analyze(context)
    finally:
        context.pdfium_document.close()

    finding = next(item for item in result.findings if item.code == "AI_IMAGE_C2PA_DECLARATION")
    assert finding.evidence["image_role"] == "document"
    assert finding.risk_points == 45


def _photo_context(
    tmp_path: Path,
    *,
    software: str | None = None,
    coverage: float = 0.25,
) -> AnalysisContext:
    image_path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (800, 600), (80, 120, 160))
    save_options: dict[str, object] = {"quality": 90}
    if software is not None:
        exif = Image.Exif()
        exif[305] = software
        save_options["exif"] = exif
    image.save(image_path, format="JPEG", **save_options)

    page_width, page_height = 600.0, 800.0
    if coverage == 1.0:
        x, y, width, height = 0.0, 0.0, page_width, page_height
    else:
        width, height = 400.0, 300.0
        x, y = 100.0, 250.0
    pdf_path = tmp_path / "photo.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_width, page_height))
    pdf.drawImage(str(image_path), x, y, width=width, height=height)
    pdf.showPage()
    pdf.save()

    raw_pdf = pdf_path.read_bytes()
    return AnalysisContext(
        input_path=pdf_path,
        output_dir=tmp_path / "analysis",
        raw_pdf=raw_pdf,
        pdfium_document=pypdfium2.PdfDocument(raw_pdf),
        reader=PdfReader(BytesIO(raw_pdf)),
        config=AnalysisConfig(render_dpi=72),
    )


def _ai_manifest(
    validation_state: str,
    *,
    validation_results: dict[str, object] | None = None,
) -> C2paManifestData:
    return C2paManifestData(
        active_manifest={
            "claim_generator": "Frod synthetic fixture",
            "assertions": [
                {
                    "label": "c2pa.actions.v2",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "digitalSourceType": (
                                    "http://cv.iptc.org/newscodes/digitalsourcetype/"
                                    "trainedAlgorithmicMedia"
                                ),
                            }
                        ]
                    },
                }
            ],
        },
        active_manifest_label="urn:c2pa:frod-test",
        validation_state=validation_state,
        validation_results=validation_results,
        embedded=True,
    )
