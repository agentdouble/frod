"""Document-level PDF history and metadata indicators."""

from __future__ import annotations

import re
from datetime import datetime

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.models import DetectorResult, Finding
from fraude_detector.pdf_software import PdfSoftwareClassification, classify_pdf_software


class PdfStructureDetector:
    name = "pdf_structure"

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

        software_finding = _software_provenance_finding(metadata, context)
        if software_finding is not None:
            findings.append(software_finding)
        else:
            notes.append("Aucun champ Creator ou Producer renseigné dans le PDF.")

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


def _software_provenance_finding(
    metadata: dict[str, object],
    context: AnalysisContext,
) -> Finding | None:
    metadata_by_name = {str(key).casefold(): str(value).strip() for key, value in metadata.items()}
    entries: list[tuple[str, str, PdfSoftwareClassification]] = []
    for field in ("Creator", "Producer"):
        value = metadata_by_name.get(field.casefold(), "")
        if value:
            entries.append((field, value, classify_pdf_software(value, context.config)))
    if not entries:
        return None

    strongest = max(entries, key=lambda entry: entry[2].points)[2]
    declared = "; ".join(f"{field} : « {value} »" for field, value, _profile in entries)
    if strongest.points > 0:
        title, interpretation = _risk_copy(strongest.category)
        description = (
            f"{declared}. {interpretation} Ce passage peut être légitime, par exemple pour "
            "convertir ou compresser le fichier. La métadonnée est déclarative et doit être "
            "rapprochée des autres indices."
        )
    else:
        title = "Provenance logicielle déclarée"
        if all(profile.recognized for _field, _value, profile in entries):
            interpretation = (
                "Les logiciels mentionnés correspondent à des usages documentaires courants."
            )
        else:
            interpretation = (
                "Au moins un logiciel n'est pas encore répertorié; cela ne constitue pas un signal."
            )
        description = f"{declared}. {interpretation} Ces champs peuvent être modifiés ou supprimés."

    return Finding(
        detector=PdfStructureDetector.name,
        code=f"PDF_SOFTWARE_{strongest.category.upper()}",
        category="metadata",
        title=title,
        description=description,
        risk_points=strongest.points,
        confidence=0.95,
        evidence={
            "creator": metadata_by_name.get("creator"),
            "producer": metadata_by_name.get("producer"),
            "software": [
                {
                    "field": field,
                    "value": value,
                    "category": profile.category,
                    "category_label": profile.label,
                    "risk_points": profile.points,
                    "recognized": profile.recognized,
                }
                for field, value, profile in entries
            ],
            "scoring_policy": "highest_declared_software_category_only",
        },
    )


def _risk_copy(category: str) -> tuple[str, str]:
    return {
        "online_pdf_service": (
            "Transformation par un service PDF en ligne",
            "Le fichier déclare un service capable de transformer son contenu ou sa structure.",
        ),
        "pdf_editor": (
            "Éditeur de PDF déclaré",
            "Le fichier déclare un logiciel permettant de modifier directement un PDF.",
        ),
        "design_tool": (
            "Outil de création graphique déclaré",
            "Le fichier déclare un outil permettant de recomposer visuellement un document.",
        ),
        "visual_editor": (
            "Éditeur d'images déclaré",
            "Le fichier déclare un logiciel permettant de retoucher des éléments visuels.",
        ),
        "generative_tool": (
            "Outil de génération par IA déclaré",
            "Le fichier déclare un outil pouvant produire ou modifier du contenu par IA.",
        ),
    }[category]


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
