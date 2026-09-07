from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fraude_detector.config import AnalysisConfig
from fraude_detector.detectors.pdf_structure import PdfStructureDetector
from fraude_detector.pdf_software import classify_pdf_software
from fraude_detector.scoring import assess_risk


@pytest.mark.parametrize(
    ("value", "category", "points"),
    [
        ("ILovePDF", "online_pdf_service", 4.0),
        ("PDF24 Creator 11.19", "online_pdf_service", 4.0),
        ("Adobe Photoshop 25.0", "visual_editor", 8.0),
        ("Canva", "design_tool", 5.0),
        ("Foxit PDF Editor", "pdf_editor", 3.0),
        ("ComfyUI", "generative_tool", 8.0),
        ("Microsoft Word for Microsoft 365", "document_generator", 0.0),
        ("GPL Ghostscript 10.04", "document_generator", 0.0),
        ("Canon iR-ADV Scan", "scanner", 0.0),
        ("DocuSign", "signature_service", 0.0),
        ("Belgian Government Document Engine", "unknown", 0.0),
    ],
)
def test_pdf_software_categories_are_conservative(
    value: str,
    category: str,
    points: float,
) -> None:
    result = classify_pdf_software(value, AnalysisConfig())

    assert result.category == category
    assert result.points == points


def test_pdf_software_points_are_configurable() -> None:
    config = AnalysisConfig(pdf_metadata_online_service_points=2.5)

    result = classify_pdf_software("iLovePDF - Online PDF tools", config)

    assert result.points == 2.5


class _PdfiumDocument:
    def __init__(self, metadata: dict[str, str]) -> None:
        self.metadata = metadata

    def get_metadata_dict(self, *, skip_empty: bool) -> dict[str, str]:
        assert skip_empty is True
        return self.metadata


def _context(metadata: dict[str, str], config: AnalysisConfig | None = None) -> Any:
    return SimpleNamespace(
        pdfium_document=_PdfiumDocument(metadata),
        revision_end_offsets=(),
        raw_pdf=b"%PDF-1.7\n%%EOF\n",
        config=config or AnalysisConfig(),
    )


def test_pdf_detector_keeps_creator_and_producer_but_scores_only_the_strongest() -> None:
    result = PdfStructureDetector().analyze(
        _context(
            {
                "Creator": "Microsoft Word",
                "Producer": "ILovePDF",
            }
        )
    )

    finding = next(item for item in result.findings if item.code.startswith("PDF_SOFTWARE_"))
    assert finding.code == "PDF_SOFTWARE_ONLINE_PDF_SERVICE"
    assert finding.risk_points == 4.0
    assert finding.category == "metadata"
    assert finding.evidence["creator"] == "Microsoft Word"
    assert finding.evidence["producer"] == "ILovePDF"
    assert finding.evidence["scoring_policy"] == "highest_declared_software_category_only"
    assert [item["risk_points"] for item in finding.evidence["software"]] == [0.0, 4.0]
    assert "Creator : « Microsoft Word »" in finding.description
    assert "Producer : « ILovePDF »" in finding.description
    assert assess_risk((finding,)).score == 4
    assert assess_risk((finding,)).level == "low"


def test_known_document_generator_is_visible_without_adding_points() -> None:
    result = PdfStructureDetector().analyze(_context({"Producer": "ReportLab PDF Library"}))

    finding = next(item for item in result.findings if item.code.startswith("PDF_SOFTWARE_"))
    assert finding.code == "PDF_SOFTWARE_DOCUMENT_GENERATOR"
    assert finding.risk_points == 0
    assert finding.title == "Provenance logicielle déclarée"
    assert "usages documentaires courants" in finding.description


def test_unknown_software_is_visible_but_never_scored() -> None:
    result = PdfStructureDetector().analyze(_context({"creator": "Issuer Internal Engine"}))

    finding = next(item for item in result.findings if item.code.startswith("PDF_SOFTWARE_"))
    assert finding.code == "PDF_SOFTWARE_UNKNOWN"
    assert finding.risk_points == 0
    assert "ne constitue pas un signal" in finding.description
