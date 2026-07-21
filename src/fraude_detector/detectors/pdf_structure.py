"""Document-level PDF history and metadata indicators."""

from __future__ import annotations

import re
from datetime import datetime

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.models import DetectorResult, Finding


class PdfStructureDetector:
    name = "pdf_structure"

    _editing_tools = {
        "adobe photoshop",
        "affinity photo",
        "canva",
        "gimp",
        "inkscape",
        "photopea",
        "pixelmator",
    }

    def analyze(self, context: AnalysisContext) -> DetectorResult:
        metadata = context.pdfium_document.get_metadata_dict(skip_empty=True)
        findings: list[Finding] = []
        notes: list[str] = []

        revision_count = len(context.revision_end_offsets)
        incremental_updates = max(0, revision_count - 1)
        signature_count = _count_signature_fields(context.raw_pdf)

        if incremental_updates:
            points = min(15.0, 8.0 + 3.0 * incremental_updates)
            if signature_count:
                points = min(points, 4.0)
            findings.append(
                Finding(
                    detector=self.name,
                    code="PDF_INCREMENTAL_UPDATES",
                    category="revision_history",
                    title="Mises a jour incrementales detectees",
                    description=(
                        "Le PDF conserve plusieurs revisions lisibles. Cela indique "
                        "des modifications successives, mais une signature ou un "
                        "formulaire rempli peuvent l'expliquer legitimement."
                    ),
                    risk_points=points,
                    confidence=0.95,
                    evidence={
                        "incremental_updates_detected": incremental_updates,
                        "valid_revision_count": revision_count,
                        "digital_signature_markers": signature_count,
                    },
                )
            )
        else:
            notes.append(
                "Aucune mise a jour incrementale conservee. Une reecriture complete "
                "peut toutefois avoir efface un historique anterieur."
            )

        declared_tools = " ".join(
            str(metadata.get(key, "")) for key in ("Creator", "Producer")
        ).casefold()
        matched_tools = sorted(tool for tool in self._editing_tools if tool in declared_tools)
        if matched_tools:
            findings.append(
                Finding(
                    detector=self.name,
                    code="PDF_IMAGE_EDITOR_METADATA",
                    category="metadata",
                    title="Logiciel de retouche declare dans les metadonnees",
                    description=(
                        "Les metadonnees citent un logiciel de creation ou de retouche "
                        "d'image. Ce signal doit etre confirme par le contexte du document."
                    ),
                    risk_points=12.0,
                    confidence=0.8,
                    evidence={"matched_tools": matched_tools},
                )
            )

        creation_date = _parse_pdf_date(metadata.get("CreationDate", ""))
        modification_date = _parse_pdf_date(metadata.get("ModDate", ""))
        if creation_date and modification_date and modification_date < creation_date:
            findings.append(
                Finding(
                    detector=self.name,
                    code="PDF_DATE_CONTRADICTION",
                    category="metadata",
                    title="Dates de metadonnees contradictoires",
                    description=(
                        "La date de modification declaree est anterieure a la date de "
                        "creation. Les metadonnees restent falsifiables et ne suffisent "
                        "pas a prouver une fraude."
                    ),
                    risk_points=15.0,
                    confidence=0.85,
                    evidence={
                        "creation_date": creation_date.isoformat(),
                        "modification_date": modification_date.isoformat(),
                    },
                )
            )

        trailing_bytes = _non_whitespace_after_last_eof(
            context.raw_pdf, context.revision_end_offsets
        )
        if trailing_bytes:
            findings.append(
                Finding(
                    detector=self.name,
                    code="PDF_TRAILING_DATA",
                    category="document_integrity",
                    title="Donnees apres la derniere fin de PDF",
                    description=(
                        "Des octets non blancs suivent la derniere revision PDF valide. "
                        "Ils peuvent provenir d'un fichier concatene ou altere."
                    ),
                    risk_points=12.0,
                    confidence=0.8,
                    evidence={"trailing_bytes": trailing_bytes},
                )
            )

        notes.append(f"Marqueurs de signature numerique detectes: {signature_count}.")
        return DetectorResult(
            name=self.name,
            status="completed",
            findings=tuple(findings),
            notes=tuple(notes),
        )


def _count_signature_fields(raw_pdf: bytes) -> int:
    patterns = (
        rb"/FT\s*/Sig\b",
        rb"/Type\s*/Sig\b",
    )
    return sum(len(re.findall(pattern, raw_pdf)) for pattern in patterns)


def _parse_pdf_date(value: str) -> datetime | None:
    match = re.search(r"(?:D:)?(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", value)
    if match is None:
        return None
    parts = [
        int(part) if part else default
        for part, default in zip(
            match.groups(),
            (0, 1, 1, 0, 0, 0),
            strict=True,
        )
    ]
    try:
        return datetime(*parts)
    except ValueError:
        return None


def _non_whitespace_after_last_eof(raw_pdf: bytes, offsets: tuple[int, ...]) -> int:
    if not offsets:
        return 0
    trailing = raw_pdf[offsets[-1] :].strip(b"\x00\x09\x0a\x0c\x0d\x20")
    return len(trailing)
