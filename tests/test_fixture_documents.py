from __future__ import annotations

from pathlib import Path

import pytest

from fraude_detector.config import AnalysisConfig
from fraude_detector.models import AnalysisReport, BoundingBox
from fraude_detector.pipeline import AnalysisPipeline

FIXTURES = Path(__file__).parent / "fixtures"


def test_fraud_fixture_preserves_the_clean_pdf_as_its_first_revision() -> None:
    clean_pdf = (FIXTURES / "assurance-sans-fraude.pdf").read_bytes()
    fraud_pdf = (FIXTURES / "assurance-fraude.pdf").read_bytes()

    assert fraud_pdf.startswith(clean_pdf)


def test_legitimate_update_fixture_preserves_the_clean_first_revision() -> None:
    clean_pdf = (FIXTURES / "assurance-sans-fraude.pdf").read_bytes()
    legitimate_pdf = (FIXTURES / "assurance-ajout-legitime.pdf").read_bytes()

    assert legitimate_pdf.startswith(clean_pdf)


@pytest.mark.parametrize("render_dpi", [72, 144])
def test_committed_clean_fixture_has_no_detected_signal(
    tmp_path: Path,
    render_dpi: int,
) -> None:
    output_dir = tmp_path / f"clean-analysis-{render_dpi}"
    report = _analyze_fixture(
        "assurance-sans-fraude.pdf",
        output_dir,
        render_dpi,
    )

    detector_statuses = {result.name: result.status for result in report.detectors}
    assert report.document.page_count == 1
    assert report.assessment.level == "low"
    assert report.assessment.score == 0
    assert not any(finding.risk_points for finding in report.findings)
    assert {finding.code for finding in report.findings} == {"PDF_SOFTWARE_UNKNOWN"}
    assert detector_statuses["pdf_structure"] == "completed"
    assert detector_statuses["revision_diff"] == "not_applicable"
    assert (output_dir / "pages/page-001.png").is_file()
    assert report.artifacts["review_overlays"] == ()
    assert report.artifacts["forensics"] == ()


@pytest.mark.parametrize("render_dpi", [72, 144])
def test_committed_fraud_fixture_is_localized_and_high_risk(
    tmp_path: Path,
    render_dpi: int,
) -> None:
    output_dir = tmp_path / f"fraud-analysis-{render_dpi}"
    report = _analyze_fixture("assurance-fraude.pdf", output_dir, render_dpi)
    finding_codes = {finding.code for finding in report.findings}

    assert report.assessment.level == "high"
    assert report.assessment.score >= 70
    assert {
        "PDF_INCREMENTAL_UPDATES",
        "PDF_REVISION_VISUAL_CHANGE",
        "SCAN_IMAGE_OVERLAY",
        "SCAN_VISIBLE_TEXT_OVERLAY",
    } <= finding_codes

    revision_history = next(
        finding for finding in report.findings if finding.code == "PDF_INCREMENTAL_UPDATES"
    )
    assert revision_history.evidence["incremental_updates_detected"] == 1
    assert revision_history.evidence["valid_revision_count"] == 2

    expected_amount_region = BoundingBox(
        x0=47.0,
        y0=297.0,
        x1=447.0,
        y1=348.0,
    )
    visual_changes = [
        finding for finding in report.findings if finding.code == "PDF_REVISION_VISUAL_CHANGE"
    ]
    assert any(
        finding.page == 1
        and finding.bbox is not None
        and _boxes_intersect(finding.bbox, expected_amount_region)
        for finding in visual_changes
    )
    assert (output_dir / "forensics/page-001-revision-diff.png").is_file()
    assert (output_dir / "review/page-001-review.png").is_file()


def _analyze_fixture(
    filename: str,
    output_dir: Path,
    render_dpi: int,
) -> AnalysisReport:
    return AnalysisPipeline(config=AnalysisConfig(render_dpi=render_dpi, max_pages=5)).analyze(
        FIXTURES / filename, output_dir
    )


def _boxes_intersect(first: BoundingBox, second: BoundingBox) -> bool:
    return not (
        first.x1 < second.x0 or second.x1 < first.x0 or first.y1 < second.y0 or second.y1 < first.y0
    )
