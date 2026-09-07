"""Serializable domain models for detectors and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DetectorStatus = Literal["completed", "not_applicable", "partial"]
RiskLevel = Literal["low", "review", "high"]
LaboratoryState = Literal[
    "clear",
    "attention",
    "detected",
    "indeterminate",
    "not_applicable",
    "error",
]
EvidenceStrength = Literal["strong", "moderate", "weak", "informational"]
NormalizationStatus = Literal["normalized", "ambiguous", "raw_only"]
VerificationVerdict = Literal["supported", "plausible", "ambiguous", "contradicted"]
VerificationStatus = Literal["clean", "attention", "incomplete"]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A location in PDF page coordinates (points, origin at top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float
    unit: str = "pdf_points"


@dataclass(frozen=True, slots=True)
class Finding:
    """One explainable signal emitted by a detector."""

    detector: str
    code: str
    category: str
    title: str
    description: str
    risk_points: float
    confidence: float
    page: int | None = None
    bbox: BoundingBox | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.risk_points <= 100:
            raise ValueError("risk_points must be between 0 and 100")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Execution summary for one detector."""

    name: str
    status: DetectorStatus
    findings: tuple[Finding, ...] = ()
    notes: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    filename: str
    sha256: str
    size_bytes: int
    page_count: int
    analyzed_pages: int
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class ImageInfo:
    filename: str
    sha256: str
    size_bytes: int
    format: str
    width: int
    height: int
    mode: str


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    score: int
    level: RiskLevel
    label: str
    explanation: str


@dataclass(frozen=True, slots=True)
class DocumentClassification:
    """Grounded semantic family inferred from OCR text."""

    family: str
    reliability: float
    language: str | None
    country: str | None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.reliability <= 1:
            raise ValueError("reliability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    """One comparable value proposed from the OCR document."""

    field_code: str
    role: str
    raw_label: str | None
    raw_value: str
    normalized_value: str | int | float | None
    normalization_status: NormalizationStatus
    page: int | None
    region_ids: tuple[str, ...] = ()
    corrected_value: str | None = None
    normalized_currency: str | None = None


@dataclass(frozen=True, slots=True)
class AdditionalExtractionField:
    """Useful document value not yet represented by the canonical vocabulary."""

    raw_label: str
    raw_value: str
    page: int | None
    region_ids: tuple[str, ...] = ()
    corrected_value: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedTable:
    """One table preserved with its original headers and cells."""

    title: str | None
    semantic_type: str
    headers: tuple[str, ...]
    column_roles: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    row_roles: tuple[str, ...]
    pages: tuple[int, ...] = ()
    region_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionCoverage:
    """How much of the OCR region inventory was accounted for."""

    total_regions: int
    accounted_regions: int
    mapped_regions: int
    table_regions: int
    boilerplate_regions: int
    unstructured_regions: int
    unreadable_regions: int
    uncovered_region_ids: tuple[str, ...] = ()

    @property
    def ratio(self) -> float:
        if self.total_regions == 0:
            return 0.0
        return self.accounted_regions / self.total_regions


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    """Experimental structured inventory kept outside fraud scoring."""

    schema_version: str
    family: str
    language: str | None
    country: str | None
    facts: tuple[ExtractedFact, ...]
    additional_fields: tuple[AdditionalExtractionField, ...]
    tables: tuple[ExtractedTable, ...]
    coverage: ExtractionCoverage
    passes: int
    limitations: tuple[str, ...] = ()
    prompt_version: str = "unknown"
    vocabulary_version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExtractionReview:
    """Independent assessment of one extracted fact, field or table."""

    target_id: str
    target_type: str
    verdict: VerificationVerdict
    explanation: str
    source_region_ids: tuple[str, ...] = ()
    suggested_value: str | None = None
    suggested_field_code: str | None = None
    suggested_role: str | None = None
    problematic_row_indexes: tuple[int, ...] = ()
    correction_applied: bool = False
    original_value: str | None = None
    original_field_code: str | None = None
    original_role: str | None = None

@dataclass(frozen=True, slots=True)
class ExtractionOmission:
    """Material OCR information that may be absent from the extraction."""

    description: str
    proposed_field_code: str | None
    proposed_role: str | None
    proposed_value: str | None
    source_region_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionVerification:
    """Non-destructive, independent review of one structured extraction."""

    schema_version: str
    status: VerificationStatus
    expected_targets: int
    reviews: tuple[ExtractionReview, ...]
    omissions: tuple[ExtractionOmission, ...]
    limitations: tuple[str, ...] = ()
    prompt_version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalysisSynthesis:
    """Concise, non-decisional summary generated after all analysis passes."""

    schema_version: str
    text: str
    prompt_version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    schema_version: str
    analyzed_at: str
    document: DocumentInfo
    assessment: RiskAssessment
    detectors: tuple[DetectorResult, ...]
    findings: tuple[Finding, ...]
    artifacts: dict[str, tuple[str, ...]]
    limitations: tuple[str, ...]
    classification: DocumentClassification | None = None
    extraction: DocumentExtraction | None = None
    extraction_verification: ExtractionVerification | None = None
    _authenticity_checks: LaboratoryReport | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("_authenticity_checks", None)
        return _without_repeated_detector_findings(result)


@dataclass(frozen=True, slots=True)
class ImageAnalysisReport:
    schema_version: str
    analyzed_at: str
    image: ImageInfo
    assessment: RiskAssessment
    detectors: tuple[DetectorResult, ...]
    findings: tuple[Finding, ...]
    artifacts: dict[str, tuple[str, ...]]
    limitations: tuple[str, ...]
    classification: DocumentClassification | None = None
    extraction: DocumentExtraction | None = None
    extraction_verification: ExtractionVerification | None = None

    def to_dict(self) -> dict[str, Any]:
        return _without_repeated_detector_findings(asdict(self))


@dataclass(frozen=True, slots=True)
class LaboratoryObservation:
    """One non-scoring observation emitted by an experimental PDF control."""

    code: str
    title: str
    summary: str
    state: LaboratoryState
    strength: EvidenceStrength
    explanation: str
    page: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LaboratoryCheck:
    """User-facing summary for one experimental control."""

    code: str
    title: str
    purpose: str
    state: LaboratoryState
    summary: str
    observations: tuple[LaboratoryObservation, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LaboratoryReport:
    """Experimental results kept separate from the production risk report."""

    schema_version: str
    checks: tuple[LaboratoryCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OcrReport:
    """Structured, non-scoring output returned by the optional OCR service."""

    success: bool
    error_message: str | None
    markdown: str
    json_result: Any
    artifacts: tuple[str, ...] = ()
    layout_images: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _without_repeated_detector_findings(result: dict[str, Any]) -> dict[str, Any]:
    """Keep findings once at report level instead of duplicating them per detector."""

    for detector in result.get("detectors", ()):
        if isinstance(detector, dict):
            detector.pop("findings", None)
    return result
