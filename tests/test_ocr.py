from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from fraude_detector.config import AnalysisConfig
from fraude_detector.detectors.ocr import OcrDetector
from fraude_detector.image_pipeline import ImageAnalysisPipeline
from fraude_detector.models import (
    DocumentClassification,
    DocumentExtraction,
    ExtractionCoverage,
    ExtractionVerification,
)
from fraude_detector.pipeline import AnalysisPipeline


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def test_ocr_detector_persists_scoped_structured_artifacts(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    source = tmp_path / "document.png"
    Image.new("RGB", (200, 100), "white").save(source)
    output = tmp_path / "analysis"
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return _Response(
            {
                "json_result": json.dumps(
                    [
                        [
                            {
                                "index": 0,
                                "label": "text",
                                "content": "MICR : 123456789",
                                "bbox_2d": [100, 100, 900, 400],
                            }
                        ]
                    ]
                ),
                "markdown_result": "MICR : 123456789",
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    detector = OcrDetector(AnalysisConfig(ocr_enabled=True, ocr_url="http://ocr.internal:8007"))
    report = detector.detect(source, output)

    assert report.success
    assert calls[0]["url"] == "http://ocr.internal:8007/glmocr/parse"
    assert calls[0]["json"]["images"] == [source.resolve().as_uri()]
    assert report.artifacts == (
        "ocr/document.json",
        "ocr/document.md",
        "ocr/layout/page-001-layout.png",
    )
    assert (output / "ocr/document.json").is_file()
    assert (output / "ocr/document.md").read_text(encoding="utf-8") == "MICR : 123456789"
    assert (output / "ocr/layout/page-001-layout.png").is_file()
    assert detector.result(report).status == "completed"


def test_ocr_unavailability_is_non_fatal(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "document.png"
    Image.new("RGB", (32, 24), "white").save(source)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "post", fail)
    detector = OcrDetector(AnalysisConfig(ocr_enabled=True))
    report = detector.detect(source, tmp_path / "analysis")

    assert not report.success
    assert report.artifacts == ()
    assert detector.result(report).status == "partial"


def test_pdf_pipeline_exposes_clean_ocr_content_without_scoring_it(
    monkeypatch: Any,
    vector_pdf: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "json_result": [
                    [
                        {
                            "index": 0,
                            "label": "text",
                            "content": "Document",
                            "bbox_2d": [100, 100, 400, 200],
                        }
                    ]
                ],
                "markdown_result": "Document",
            }
        ),
    )
    output = tmp_path / "analysis"
    report = AnalysisPipeline(
        config=AnalysisConfig(
            render_dpi=72,
            max_pages=1,
            ocr_enabled=True,
        )
    ).analyze(vector_pdf, output)

    ocr = next(detector for detector in report.detectors if detector.name == "ocr_content")
    assert ocr.status == "completed"
    assert ocr.findings == ()
    assert report.artifacts["ocr_json"] == ("ocr/document.json",)
    assert report.artifacts["ocr_markdown"] == ("ocr/document.md",)
    assert report.artifacts["ocr_layout"] == ("ocr/layout/page-001-layout.png",)
    assert report.assessment.score >= 0


def test_pdf_pipeline_exposes_classification_outside_artifacts(
    monkeypatch: Any,
    vector_pdf: Path,
    tmp_path: Path,
) -> None:
    markdown = "Relevé de compte\nSolde précédent\nOpérations du mois"
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "json_result": [[{"index": 0, "label": "text", "content": markdown}]],
                "markdown_result": markdown,
            }
        ),
    )
    classifier_calls: list[tuple[str, AnalysisConfig]] = []

    def fake_classify(text: str, config: AnalysisConfig) -> DocumentClassification:
        classifier_calls.append((text, config))
        return DocumentClassification(
            family="releve_bancaire",
            reliability=0.82,
            language="fr",
            country=None,
            evidence=("Relevé de compte", "Opérations du mois"),
        )

    monkeypatch.setattr("fraude_detector.content_analysis.classify_document", fake_classify)
    extraction = DocumentExtraction(
        schema_version="0.1-experimental",
        family="releve_bancaire",
        language="fr",
        country=None,
        facts=(),
        additional_fields=(),
        tables=(),
        coverage=ExtractionCoverage(
            total_regions=1,
            accounted_regions=1,
            mapped_regions=0,
            table_regions=0,
            boilerplate_regions=1,
            unstructured_regions=0,
            unreadable_regions=0,
        ),
        passes=1,
    )
    extractor_calls: list[tuple[Any, DocumentClassification | None, AnalysisConfig]] = []

    def fake_extract(
        payload: Any,
        classification: DocumentClassification | None,
        extraction_config: AnalysisConfig,
    ) -> DocumentExtraction:
        extractor_calls.append((payload, classification, extraction_config))
        return extraction

    monkeypatch.setattr("fraude_detector.content_analysis.extract_document", fake_extract)
    verification = ExtractionVerification(
        schema_version="0.1-experimental",
        status="clean",
        expected_targets=0,
        reviewed_targets=0,
        reviews=(),
        omissions=(),
    )
    verifier_calls: list[
        tuple[Any, DocumentExtraction, DocumentClassification | None, AnalysisConfig]
    ] = []

    def fake_verify(
        payload: Any,
        extracted: DocumentExtraction,
        classification: DocumentClassification | None,
        verification_config: AnalysisConfig,
    ) -> ExtractionVerification:
        verifier_calls.append((payload, extracted, classification, verification_config))
        return verification

    monkeypatch.setattr("fraude_detector.content_analysis.verify_extraction", fake_verify)
    config = AnalysisConfig(
        render_dpi=72,
        max_pages=1,
        ocr_enabled=True,
        classification_enabled=True,
        extraction_enabled=True,
        verification_enabled=True,
    )
    output = tmp_path / "classification-analysis"
    report = AnalysisPipeline(config=config).analyze(vector_pdf, output)

    assert classifier_calls == [(markdown, config)]
    assert report.classification is not None
    assert report.classification.family == "releve_bancaire"
    assert len(extractor_calls) == 1
    assert extractor_calls[0][1] == report.classification
    assert extractor_calls[0][2] == config
    assert report.extraction == extraction
    assert len(verifier_calls) == 1
    assert verifier_calls[0][1] == extraction
    assert verifier_calls[0][2] == report.classification
    assert verifier_calls[0][3] == config
    assert report.extraction_verification == verification
    assert "classification" not in report.artifacts
    assert "extraction" not in report.artifacts
    assert all(isinstance(paths, tuple) for paths in report.artifacts.values())
    serialized = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert serialized["classification"]["reliability"] == 0.82
    assert serialized["extraction"]["coverage"]["accounted_regions"] == 1
    assert serialized["extraction_verification"]["status"] == "clean"


def test_image_pipeline_exposes_clean_ocr_content_without_scoring_it(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    source = tmp_path / "document.png"
    Image.new("RGB", (200, 100), "white").save(source)
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "json_result": [
                    [
                        {
                            "index": 0,
                            "label": "text",
                            "content": "Document",
                            "bbox_2d": [100, 100, 400, 200],
                        }
                    ]
                ],
                "markdown_result": "Document",
            }
        ),
    )

    report = ImageAnalysisPipeline(
        config=AnalysisConfig(ocr_enabled=True),
    ).analyze(source, tmp_path / "analysis")

    ocr = next(detector for detector in report.detectors if detector.name == "ocr_content")
    assert ocr.status == "completed"
    assert ocr.findings == ()
    assert report.artifacts["ocr_layout"] == ("ocr/layout/page-001-layout.png",)
    assert report.assessment.score == 0


def test_image_pipeline_scores_corroborated_ocr_content(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    source = tmp_path / "document.png"
    Image.new("RGB", (200, 100), "white").save(source)
    fixture = Path("tests/fixtures/ocr/releve-bancaire-anomalies")
    payload = json.loads((fixture / "document.json").read_text(encoding="utf-8"))
    markdown = (fixture / "document.md").read_text(encoding="utf-8")
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "json_result": payload,
                "markdown_result": markdown,
            }
        ),
    )

    report = ImageAnalysisPipeline(
        config=AnalysisConfig(ocr_enabled=True),
    ).analyze(source, tmp_path / "analysis")

    ocr = next(detector for detector in report.detectors if detector.name == "ocr_content")
    finding = next(item for item in ocr.findings if item.code == "OCR_CONTENT_CONSISTENCY")
    assert finding.risk_points == 30
    assert finding.confidence == 0.8
    assert report.assessment.score == 30
