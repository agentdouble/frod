"""Explainable scoring of OCR-derived content consistency signals."""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from fraude_detector.laboratory.ocr import analyze_ocr_laboratory
from fraude_detector.models import (
    DetectorResult,
    Finding,
    LaboratoryCheck,
    LaboratoryObservation,
    OcrReport,
)

OCR_CONTENT_FAMILY_CAP = 30.0

_GROUP_POINTS: dict[str, dict[str, float]] = {
    "identity_consistency": {
        "OCR_CARD_VALUES_CONFLICT": 18.0,
        "OCR_CARD_LUHN_INVALID": 6.0,
        "OCR_IBAN_INVALID": 6.0,
        "OCR_BIC_INVALID": 6.0,
        "OCR_SIREN_INVALID": 6.0,
        "OCR_SIRET_INVALID": 6.0,
        "OCR_EU_VAT_INVALID": 6.0,
        "OCR_RPPS_INVALID": 6.0,
        "OCR_FINESS_INVALID": 6.0,
        "OCR_CKYC_INVALID": 4.0,
        "OCR_MICR_INVALID": 4.0,
    },
    "geographic_consistency": {
        "OCR_BANKING_GEOGRAPHY_MISMATCH": 12.0,
    },
    "financial_consistency": {
        "OCR_STATEMENT_SUMMARY_MISMATCH": 15.0,
        "OCR_INVOICE_TOTAL_MISMATCH": 15.0,
        "OCR_LEDGER_MISMATCH": 12.0,
    },
    "date_consistency": {
        "OCR_DATE_INVALID": 5.0,
    },
}

_GROUP_LABELS = {
    "identity_consistency": "Identifiants",
    "geographic_consistency": "Coherence geographique",
    "financial_consistency": "Calculs",
    "date_consistency": "Dates",
}


@dataclass(frozen=True, slots=True)
class OcrReliability:
    """Measured extraction reliability, distinct from fraud confidence."""

    score: float
    source: str
    representation_agreement: float | None
    native_confidence: float | None

    @property
    def point_factor(self) -> float:
        if self.score >= 0.78:
            return 1.0
        if self.score >= 0.65:
            return 0.75
        return 0.0

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "score": round(self.score, 4),
            "source": self.source,
            "representation_agreement": (
                round(self.representation_agreement, 4)
                if self.representation_agreement is not None
                else None
            ),
            "native_confidence": (
                round(self.native_confidence, 4) if self.native_confidence is not None else None
            ),
        }


def build_ocr_content_result(report: OcrReport) -> DetectorResult:
    """Turn one OCR report into a conservative production detector result."""

    if not report.success:
        return DetectorResult(
            name="ocr_content",
            status="partial",
            notes=(
                "OCR non execute ou indisponible; aucun controle de contenu n'est score.",
                report.error_message or "Erreur OCR inconnue.",
            ),
        )

    checks = analyze_ocr_laboratory(report.json_result)
    reliability = measure_ocr_reliability(report.json_result, report.markdown)
    finding = score_ocr_checks(checks, reliability)
    return DetectorResult(
        name="ocr_content",
        status="completed",
        findings=(finding,) if finding is not None else (),
        notes=(
            "La fiabilite OCR mesure l'extraction, pas la probabilite de fraude.",
            "Sans confiance native, l'accord JSON/Markdown est utilise et plafonne a 80%.",
            "Les anomalies correlees sont regroupees avant le calcul des points.",
        ),
        artifacts=report.artifacts,
    )


def measure_ocr_reliability(ocr_json: Any, markdown: str) -> OcrReliability:
    """Use native confidence when available, otherwise compare both OCR views."""

    native_values = _native_confidences(ocr_json)
    native_confidence = _lower_quartile(native_values) if native_values else None
    json_text = "\n".join(_content_values(ocr_json))
    agreement = _token_agreement(json_text, markdown) if markdown.strip() else None

    if native_confidence is not None:
        score = (
            0.8 * native_confidence + 0.2 * agreement
            if agreement is not None
            else native_confidence
        )
        source = "native_confidence"
    elif agreement is not None:
        score = min(0.8, 0.55 + 0.25 * agreement)
        source = "json_markdown_agreement"
    else:
        score = 0.5
        source = "no_confidence_available"

    return OcrReliability(
        score=min(1.0, max(0.0, score)),
        source=source,
        representation_agreement=agreement,
        native_confidence=native_confidence,
    )


def score_ocr_checks(
    checks: tuple[LaboratoryCheck, ...],
    reliability: OcrReliability,
) -> Finding | None:
    """Aggregate independent OCR issue groups into one capped finding."""

    attention = tuple(
        observation
        for check in checks
        for observation in check.observations
        if observation.state == "attention"
    )
    if not attention:
        return None

    groups: dict[str, dict[str, Any]] = {}
    for group, code_points in _GROUP_POINTS.items():
        retained = tuple(item for item in attention if item.code in code_points)
        if not retained:
            continue
        base_points = max(code_points[item.code] for item in retained)
        groups[group] = {
            "label": _GROUP_LABELS[group],
            "base_points": base_points,
            "signals": tuple(item.code for item in retained),
        }

    point_factor = reliability.point_factor
    quality_state = next(
        (check.state for check in checks if check.code == "ocr_quality"),
        "indeterminate",
    )
    if quality_state != "clear":
        point_factor = 0.0
    total_points = min(
        OCR_CONTENT_FAMILY_CAP,
        sum(float(group["base_points"]) for group in groups.values()) * point_factor,
    )
    total_points = round(total_points, 1)
    scored = total_points > 0
    retained_observations = tuple(
        item
        for item in attention
        if any(item.code in group["signals"] for group in groups.values())
    )
    displayed = retained_observations or attention
    group_labels = ", ".join(str(group["label"]).lower() for group in groups.values())

    if scored:
        title = "Coherence du contenu a verifier"
        description = f"{len(groups)} famille(s) d'incoherences independantes : {group_labels}."
        code = "OCR_CONTENT_CONSISTENCY"
    elif groups:
        title = "Anomalies OCR a confirmer"
        description = (
            "Des anomalies ont ete reconnues, mais la fiabilite de l'extraction "
            "est insuffisante pour modifier le score."
        )
        code = "OCR_CONTENT_UNSCORED"
    else:
        title = "Observation OCR contextuelle"
        description = (
            "Le controle demande une confirmation humaine mais n'entre pas dans le score Frod."
        )
        code = "OCR_CONTENT_DIAGNOSTIC"

    pages = [item.page for item in displayed if item.page is not None]
    return Finding(
        detector="ocr_content",
        code=code,
        category="content_consistency",
        title=title,
        description=description,
        risk_points=total_points,
        confidence=reliability.score,
        page=min(pages) if pages else None,
        evidence={
            "ocr_reliability": reliability.to_dict(),
            "point_factor": point_factor,
            "quality_gate": quality_state,
            "family_cap": OCR_CONTENT_FAMILY_CAP,
            "groups": groups,
            "observations": tuple(_observation_evidence(item) for item in displayed),
        },
    )


def _observation_evidence(observation: LaboratoryObservation) -> dict[str, Any]:
    return {
        "code": observation.code,
        "title": observation.title,
        "summary": observation.summary,
        "strength": observation.strength,
        "page": observation.page,
    }


def _content_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, list):
        for item in value:
            values.extend(_content_values(item))
    elif isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            values.append(content)
        for key, item in value.items():
            if key != "content":
                values.extend(_content_values(item))
    return values


def _native_confidences(value: Any) -> list[float]:
    values: list[float] = []
    if isinstance(value, list):
        for item in value:
            values.extend(_native_confidences(item))
    elif isinstance(value, dict):
        if isinstance(value.get("content"), str):
            for key in ("confidence", "score", "probability", "prob"):
                candidate = value.get(key)
                if isinstance(candidate, int | float) and not isinstance(candidate, bool):
                    normalized = float(candidate)
                    if 1 < normalized <= 100:
                        normalized /= 100
                    if 0 <= normalized <= 1:
                        values.append(normalized)
                        break
        for item in value.values():
            values.extend(_native_confidences(item))
    return values


def _token_agreement(left: str, right: str) -> float:
    left_tokens = Counter(_tokens(left))
    right_tokens = Counter(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    common = sum((left_tokens & right_tokens).values())
    return 2 * common / (sum(left_tokens.values()) + sum(right_tokens.values()))


def _lower_quartile(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * 0.25)]


def _tokens(value: str) -> list[str]:
    visible = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return re.findall(r"[a-z0-9]+", visible.casefold())
