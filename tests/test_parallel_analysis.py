from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from fraude_detector.ai_images import AiImagePrediction
from fraude_detector.config import AnalysisConfig
from fraude_detector.content_analysis import ContentAnalysisResult
from fraude_detector.image_pipeline import ImageAnalysisPipeline
from fraude_detector.models import DetectorResult, OcrReport
from fraude_detector.pipeline import AnalysisPipeline


class _BarrierDetector:
    name = "barrier_detector"

    def __init__(self, barrier: threading.Barrier) -> None:
        self.barrier = barrier

    def analyze(self, context: Any) -> DetectorResult:
        context.report_progress(0.2, "Analyse technique en cours")
        self.barrier.wait(timeout=2)
        context.report_progress(0.8, "Analyse technique presque terminée")
        return DetectorResult(name=self.name, status="completed")


def test_pdf_content_and_technical_branches_run_concurrently(
    monkeypatch: Any,
    vector_pdf: Path,
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)
    main_thread = threading.get_ident()
    content_threads: list[int] = []
    callback_threads: list[int] = []
    progress_values: list[float] = []

    def fake_content(
        source: Path,
        destination: Path,
        config: AnalysisConfig,
        **kwargs: Any,
    ) -> ContentAnalysisResult:
        del source, destination, config
        content_threads.append(threading.get_ident())
        callback = kwargs["progress_callback"]
        callback(0.2, "OCR en cours")
        barrier.wait(timeout=2)
        callback(1.0, "Contenu terminé")
        return _content_result()

    def progress(value: float, label: str) -> None:
        del label
        callback_threads.append(threading.get_ident())
        progress_values.append(value)

    monkeypatch.setattr("fraude_detector.pipeline.analyze_document_content", fake_content)
    output = tmp_path / "parallel-pdf"
    report = AnalysisPipeline(
        config=AnalysisConfig(render_dpi=72, max_pages=1, ocr_enabled=True),
        detectors=(_BarrierDetector(barrier),),
    ).analyze(vector_pdf, output, progress_callback=progress)

    assert content_threads and content_threads[0] != main_thread
    assert set(callback_threads) == {main_thread}
    assert progress_values == sorted(progress_values)
    assert [detector.name for detector in report.detectors] == [
        "barrier_detector",
        "ocr_content",
    ]
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["schema_version"]


def test_image_content_and_model_branches_run_concurrently(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 48), "white").save(source)
    barrier = threading.Barrier(2)
    main_thread = threading.get_ident()
    content_threads: list[int] = []
    callback_threads: list[int] = []
    progress_values: list[float] = []

    def fake_content(
        source_path: Path,
        destination: Path,
        config: AnalysisConfig,
        **kwargs: Any,
    ) -> ContentAnalysisResult:
        del source_path, destination, config
        content_threads.append(threading.get_ident())
        callback = kwargs["progress_callback"]
        callback(0.25, "OCR en cours")
        barrier.wait(timeout=2)
        callback(1.0, "Contenu terminé")
        return _content_result()

    def load_adapter() -> _ConstantAdapter:
        barrier.wait(timeout=2)
        return _ConstantAdapter()

    def progress(value: float, label: str) -> None:
        del label
        callback_threads.append(threading.get_ident())
        progress_values.append(value)

    monkeypatch.setattr("fraude_detector.image_pipeline.analyze_document_content", fake_content)
    output = tmp_path / "parallel-image"
    report = ImageAnalysisPipeline(
        config=AnalysisConfig(ocr_enabled=True),
        ai_image_adapter_loaders=(load_adapter,),
    ).analyze(source, output, progress_callback=progress)

    assert content_threads and content_threads[0] != main_thread
    assert set(callback_threads) == {main_thread}
    assert progress_values == sorted(progress_values)
    assert [detector.name for detector in report.detectors] == [
        "image_provenance",
        "ai_generated_image",
        "ocr_content",
    ]
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["schema_version"]


def _content_result() -> ContentAnalysisResult:
    return ContentAnalysisResult(
        ocr_report=OcrReport(
            success=True,
            error_message=None,
            markdown="Document",
            json_result=[],
        ),
        detector_result=DetectorResult(name="ocr_content", status="completed"),
    )


class _ConstantAdapter:
    adapter_id = "parallel_test"
    method_family = "parallel_test"
    model_version = "test"

    def predict(self, image: Image.Image) -> AiImagePrediction:
        del image
        return AiImagePrediction(synthetic_score=0.1)
