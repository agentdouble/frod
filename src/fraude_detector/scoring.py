"""Conservative and explainable risk aggregation."""

from __future__ import annotations

from collections import defaultdict

from fraude_detector.models import Finding, RiskAssessment

FAMILY_CAPS: dict[str, float] = {
    "annotations": 10.0,
    "content_consistency": 30.0,
    "document_integrity": 20.0,
    "metadata": 15.0,
    "page_composition": 45.0,
    "raster_forensics": 30.0,
    "provenance_integrity": 20.0,
    "revision_history": 15.0,
    "revision_visual": 60.0,
    "synthetic_media": 45.0,
}


def assess_risk(findings: tuple[Finding, ...]) -> RiskAssessment:
    """Aggregate independent signal families without claiming probability."""

    by_family: dict[str, dict[str, float]] = defaultdict(dict)
    for finding in findings:
        if finding.risk_points > 0:
            code_scores = by_family[finding.category]
            code_scores[finding.code] = max(
                code_scores.get(finding.code, 0.0),
                finding.risk_points,
            )

    family_scores: list[float] = []
    for family, code_scores in by_family.items():
        ordered = sorted(code_scores.values(), reverse=True)
        raw_score = ordered[0] + 0.2 * sum(ordered[1:])
        family_scores.append(min(FAMILY_CAPS.get(family, 20.0), raw_score))

    family_scores.sort(reverse=True)
    if not family_scores:
        score = 0
    else:
        weighted = family_scores[0]
        if len(family_scores) >= 2:
            weighted += 0.35 * family_scores[1]
        if len(family_scores) >= 3:
            weighted += 0.15 * sum(family_scores[2:])
        score = round(min(100.0, weighted))

    if len(family_scores) < 2:
        score = min(score, 69)

    if score >= 70:
        level = "high"
        label = "Indices forts de modification"
        explanation = (
            "Plusieurs familles de signaux indépendantes se corroborent. "
            "Une revue humaine reste obligatoire avant toute conclusion de fraude."
        )
    elif score >= 30:
        level = "review"
        label = "Revue manuelle nécessaire"
        explanation = (
            "Au moins un signal technique mérite un contrôle humain et un "
            "rapprochement avec les données métier."
        )
    else:
        level = "low"
        label = "Aucun signal fort détecté"
        explanation = (
            "L'analyse n'a pas trouvé de signal fort dans son périmètre. "
            "Cela ne prouve ni l'authenticité ni l'absence de modification."
        )

    return RiskAssessment(
        score=score,
        level=level,
        label=label,
        explanation=explanation,
    )
