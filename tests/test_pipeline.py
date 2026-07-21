from __future__ import annotations

import json
from pathlib import Path

from fraude_detector.config import AnalysisConfig
from fraude_detector.pipeline import AnalysisPipeline


def test_pipeline_writes_report_and_page_render(vector_pdf: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "analysis"
    report = AnalysisPipeline(config=AnalysisConfig(render_dpi=72, max_pages=5)).analyze(
        vector_pdf, output_dir
    )

    assert report.document.page_count == 1
    assert report.document.sha256
    assert report.assessment.level in {"low", "review", "high"}
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "pages/page-001.png").is_file()

    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "fraud_confirmed" not in serialized
    assert "authentic" not in payload["assessment"]["level"]
    for paths in payload["artifacts"].values():
        for relative_path in paths:
            assert ".." not in Path(relative_path).parts
            assert (output_dir / relative_path).is_file()


def test_scan_pdf_runs_image_detectors(scan_pdf: Path, tmp_path: Path) -> None:
    report = AnalysisPipeline(config=AnalysisConfig(render_dpi=72, max_pages=5)).analyze(
        scan_pdf, tmp_path / "scan-analysis"
    )

    detector_by_name = {result.name: result for result in report.detectors}
    assert detector_by_name["page_composition"].status == "completed"
    assert detector_by_name["raster_anomaly"].status == "completed"
