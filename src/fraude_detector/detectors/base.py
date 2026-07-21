"""Detector contract and immutable analysis context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pypdfium2
from pypdf import PdfReader

from fraude_detector.config import AnalysisConfig
from fraude_detector.models import DetectorResult


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    input_path: Path
    output_dir: Path
    raw_pdf: bytes
    pdfium_document: pypdfium2.PdfDocument
    reader: PdfReader
    config: AnalysisConfig
    revision_end_offsets: tuple[int, ...] = ()
    password: str | None = None

    @property
    def analyzed_page_count(self) -> int:
        return min(len(self.pdfium_document), self.config.max_pages)

    def artifact_path(self, *parts: str) -> Path:
        path = self.output_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def relative_artifact(self, path: Path) -> str:
        return path.relative_to(self.output_dir).as_posix()


class Detector(Protocol):
    name: str

    def analyze(self, context: AnalysisContext) -> DetectorResult:
        """Analyze a document without deciding whether it is fraudulent."""
        ...
