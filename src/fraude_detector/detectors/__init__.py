"""Built-in, independently testable PDF detectors."""

from fraude_detector.detectors.page_composition import PageCompositionDetector
from fraude_detector.detectors.pdf_structure import PdfStructureDetector
from fraude_detector.detectors.raster_anomaly import RasterAnomalyDetector
from fraude_detector.detectors.revision_diff import RevisionDiffDetector

__all__ = [
    "PageCompositionDetector",
    "PdfStructureDetector",
    "RasterAnomalyDetector",
    "RevisionDiffDetector",
]
