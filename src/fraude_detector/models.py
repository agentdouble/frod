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
class AnalysisReport:
    schema_version: str
    analyzed_at: str
    document: DocumentInfo
    assessment: RiskAssessment
    detectors: tuple[DetectorResult, ...]
    findings: tuple[Finding, ...]
    artifacts: dict[str, tuple[str, ...]]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
