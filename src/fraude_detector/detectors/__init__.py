"""Built-in, independently testable PDF detectors."""

from fraude_detector.detectors.ai_generated_image import AiGeneratedImageDetector
from fraude_detector.detectors.image_provenance import ImageProvenanceDetector
from fraude_detector.detectors.ocr import OcrDetector
from fraude_detector.detectors.page_composition import PageCompositionDetector
from fraude_detector.detectors.pdf_structure import PdfStructureDetector
from fraude_detector.detectors.raster_anomaly import RasterAnomalyDetector
from fraude_detector.detectors.revision_diff import RevisionDiffDetector

__all__ = [
    "AiGeneratedImageDetector",
    "ImageProvenanceDetector",
    "OcrDetector",
    "PageCompositionDetector",
    "PdfStructureDetector",
    "RasterAnomalyDetector",
    "RevisionDiffDetector",
]
