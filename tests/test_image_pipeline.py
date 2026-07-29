from __future__ import annotations

from pathlib import Path

from PIL import Image

from fraude_detector.ai_images import AiImagePrediction
from fraude_detector.image_pipeline import ImageAnalysisPipeline


def test_standalone_png_is_analyzed_without_pdf_conversion(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 24), "blue").save(source)
    output = tmp_path / "analysis"

    report = ImageAnalysisPipeline().analyze(source, output)

    assert report.image.filename == "source.png"
    assert report.image.format == "PNG"
    assert (report.image.width, report.image.height) == (32, 24)
    assert report.assessment.score == 0
    assert (output / "report.json").is_file()
    assert (output / "forensics/ai/image-provenance.json").is_file()


def test_committed_c2pa_png_reports_declared_ai_origin(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures/0ca1ba88-2a9c-49bc-abe2-604986b6b06a.png"

    report = ImageAnalysisPipeline().analyze(source, tmp_path / "analysis")

    finding = next(item for item in report.findings if item.code == "AI_IMAGE_C2PA_DECLARATION")
    assert finding.evidence["declaration_trust"] == "untrusted"
    assert finding.risk_points == 40
    assert report.assessment.score == 40
    assert report.assessment.level == "review"


def test_gapl_global_index_contributes_to_standalone_image_score(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (672, 672), "blue").save(source)

    report = ImageAnalysisPipeline(
        ai_image_adapters=(_ConstantGaplAdapter(0.95),),
    ).analyze(source, tmp_path / "analysis")

    finding = next(item for item in report.findings if item.code == "AI_GAPL_GLOBAL_TRACE")
    assert finding.risk_points == 30
    assert round(float(finding.evidence["global_index"]), 2) == 0.96
    assert report.assessment.score == 30
    assert report.assessment.level == "review"
    assert report.detectors[-1].status == "completed"


class _ConstantGaplAdapter:
    adapter_id = "gapl_cvpr2026"
    method_family = "clip_prototype"
    model_version = "test"

    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, image: Image.Image) -> AiImagePrediction:
        return AiImagePrediction(synthetic_score=self.score)
