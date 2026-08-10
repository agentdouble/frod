from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pypdfium2
from PIL import Image
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from fraude_detector.ai_images import AiImagePrediction
from fraude_detector.config import AnalysisConfig
from fraude_detector.detectors.ai_generated_image import AiGeneratedImageDetector
from fraude_detector.detectors.base import AnalysisContext


class ConstantAdapter:
    model_version = "test-1"

    def __init__(
        self,
        adapter_id: str,
        method_family: str,
        score: float,
    ) -> None:
        self.adapter_id = adapter_id
        self.method_family = method_family
        self.score = score
        self.calls = 0

    def predict(self, image: Image.Image) -> AiImagePrediction:
        self.calls += 1
        return AiImagePrediction(self.score)


class SizeSensitiveAdapter(ConstantAdapter):
    def predict(self, image: Image.Image) -> AiImagePrediction:
        self.calls += 1
        return AiImagePrediction(0.95 if image.width >= 800 else 0.10)


def test_photo_without_adapter_abstains(tmp_path: Path) -> None:
    context = _context(tmp_path, coverage=0.25)
    try:
        result = AiGeneratedImageDetector().analyze(context)
    finally:
        context.pdfium_document.close()

    assert result.status == "partial"
    assert result.findings == ()
    assert result.artifacts == ()


def test_pdf_image_analysis_is_disabled_by_default(tmp_path: Path) -> None:
    adapter = ConstantAdapter("model-a", "frequency", 0.99)
    context = _context(tmp_path, coverage=0.25, analyze_pdf_images=False)
    try:
        result = AiGeneratedImageDetector((adapter,)).analyze(context)
    finally:
        context.pdfium_document.close()

    assert result.status == "not_applicable"
    assert adapter.calls == 0
    assert result.findings == ()


def test_full_page_document_is_not_sent_to_model(tmp_path: Path) -> None:
    adapter = ConstantAdapter("model-a", "frequency", 0.99)
    context = _context(tmp_path, coverage=1.0)
    try:
        result = AiGeneratedImageDetector((adapter,)).analyze(context)
    finally:
        context.pdfium_document.close()

    assert result.status == "not_applicable"
    assert adapter.calls == 0
    assert result.findings == ()


def test_two_stable_method_families_create_consensus(tmp_path: Path) -> None:
    context = _context(tmp_path, coverage=0.25)
    adapters = (
        ConstantAdapter("model-a", "frequency", 0.92),
        ConstantAdapter("model-b", "semantic", 0.88),
    )
    try:
        result = AiGeneratedImageDetector(adapters).analyze(context)
    finally:
        context.pdfium_document.close()

    assert result.status == "completed"
    finding = next(item for item in result.findings if item.code == "AI_PIXEL_TRACE_CONSENSUS")
    assert finding.risk_points == 35
    assert {item["method_family"] for item in finding.evidence["adapters"]} == {
        "frequency",
        "semantic",
    }
    assert all(adapter.calls == 4 for adapter in adapters)
    assert len(result.artifacts) == 1
    assert (context.output_dir / result.artifacts[0]).is_file()


def test_same_family_cannot_form_consensus(tmp_path: Path) -> None:
    context = _context(tmp_path, coverage=0.25)
    adapters = (
        ConstantAdapter("model-a", "frequency", 0.95),
        ConstantAdapter("model-b", "frequency", 0.95),
    )
    try:
        result = AiGeneratedImageDetector(adapters).analyze(context)
    finally:
        context.pdfium_document.close()

    assert result.status == "partial"
    assert "AI_PIXEL_TRACE_CONSENSUS" not in {item.code for item in result.findings}
    assert {item.code for item in result.findings} == {"AI_PIXEL_TRACE_SINGLE_MODEL"}
    assert all(item.risk_points == 0 for item in result.findings)


def test_one_high_model_is_only_a_zero_risk_diagnostic(tmp_path: Path) -> None:
    context = _context(tmp_path, coverage=0.25)
    adapter = ConstantAdapter("model-a", "frequency", 0.95)
    try:
        result = AiGeneratedImageDetector((adapter,)).analyze(context)
    finally:
        context.pdfium_document.close()

    finding = next(item for item in result.findings if item.code == "AI_PIXEL_TRACE_SINGLE_MODEL")
    assert result.status == "partial"
    assert finding.risk_points == 0
    assert finding.evidence["score"] == 0.95


def test_gapl_global_index_scores_an_eligible_pdf_photo(tmp_path: Path) -> None:
    context = _context(tmp_path, coverage=0.25)
    adapter = ConstantAdapter("gapl_cvpr2026", "clip_prototype", 0.95)
    try:
        result = AiGeneratedImageDetector((adapter,)).analyze(context)
    finally:
        context.pdfium_document.close()

    finding = next(item for item in result.findings if item.code == "AI_GAPL_GLOBAL_TRACE")
    assert finding.risk_points == 30
    assert round(float(finding.evidence["global_index"]), 2) == 0.96
    assert len(finding.artifacts) == 2
    assert all((context.output_dir / path).is_file() for path in finding.artifacts)
    assert result.status == "completed"


def test_unstable_model_adds_zero_risk_quality_finding(tmp_path: Path) -> None:
    context = _context(tmp_path, coverage=0.25)
    adapters = (
        SizeSensitiveAdapter("unstable", "frequency", 0.95),
        ConstantAdapter("stable", "semantic", 0.92),
    )
    try:
        result = AiGeneratedImageDetector(adapters).analyze(context)
    finally:
        context.pdfium_document.close()

    finding = next(item for item in result.findings if item.code == "AI_ANALYSIS_UNSTABLE")
    assert finding.risk_points == 0
    assert "AI_PIXEL_TRACE_CONSENSUS" not in {item.code for item in result.findings}


def _context(
    tmp_path: Path,
    *,
    coverage: float,
    analyze_pdf_images: bool = True,
) -> AnalysisContext:
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (1000, 800), (80, 120, 160)).save(
        image_path,
        format="JPEG",
        quality=90,
    )
    page_width, page_height = 600.0, 800.0
    if coverage == 1.0:
        x, y, width, height = 0.0, 0.0, page_width, page_height
    else:
        x, y, width, height = 100.0, 250.0, 400.0, 300.0
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
        config=AnalysisConfig(
            render_dpi=72,
            ai_analyze_pdf_images=analyze_pdf_images,
        ),
    )
