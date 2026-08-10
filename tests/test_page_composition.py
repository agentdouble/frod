from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pypdfium2
from PIL import Image
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from fraude_detector.config import AnalysisConfig
from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.detectors.page_composition import PageCompositionDetector


def test_single_revision_scan_layers_are_visible_but_do_not_score(tmp_path: Path) -> None:
    background_path = tmp_path / "scan.jpg"
    logo_path = tmp_path / "logo.png"
    Image.new("RGB", (1200, 1600), (235, 235, 230)).save(
        background_path,
        format="JPEG",
        quality=80,
    )
    Image.new("RGB", (180, 80), (30, 80, 140)).save(logo_path, format="PNG")

    pdf_path = tmp_path / "segmented-scan.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(600, 800))
    pdf.drawImage(str(background_path), 0, 0, width=600, height=800)
    pdf.drawImage(str(logo_path), 40, 700, width=90, height=40)
    pdf.drawString(60, 650, "Formulaire prérempli")
    pdf.save()

    raw_pdf = pdf_path.read_bytes()
    context = AnalysisContext(
        input_path=pdf_path,
        output_dir=tmp_path / "analysis",
        raw_pdf=raw_pdf,
        pdfium_document=pypdfium2.PdfDocument(raw_pdf),
        reader=PdfReader(BytesIO(raw_pdf)),
        config=AnalysisConfig(render_dpi=72),
        revision_end_offsets=(len(raw_pdf),),
    )
    try:
        result = PageCompositionDetector().analyze(context)
    finally:
        context.pdfium_document.close()

    overlays = [
        finding
        for finding in result.findings
        if finding.code in {"SCAN_IMAGE_OVERLAY", "SCAN_VISIBLE_TEXT_OVERLAY"}
    ]
    assert overlays
    assert all(finding.risk_points == 0 for finding in overlays)
    assert all(
        finding.evidence["score_reason"] == "single_retained_revision" for finding in overlays
    )
