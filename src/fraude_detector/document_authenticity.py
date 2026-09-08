"""Production scoring for cryptographic and structured-document evidence."""

from __future__ import annotations

from dataclasses import dataclass

from fraude_detector.config import AnalysisConfig
from fraude_detector.models import DetectorResult, Finding, LaboratoryReport


@dataclass(frozen=True, slots=True)
class _FindingRule:
    category: str
    title: str
    description: str
    points_attribute: str


_RULES = {
    "PDF_SIGNATURE_INVALID": _FindingRule(
        "provenance_integrity",
        "Signature électronique invalide",
        "Le contrôle cryptographique de la signature échoue. Le fichier signé peut avoir été "
        "modifié ou la signature peut être corrompue.",
        "invalid_signature_points",
    ),
    "PDF_SUSPICIOUS_POST_SIGNATURE_CHANGE": _FindingRule(
        "document_integrity",
        "Modification suspecte après signature",
        "Du contenu a été ajouté après la signature électronique sans respecter les "
        "modifications autorisées par le document.",
        "post_signature_change_points",
    ),
    "TWO_D_DOC_SIGNATURE_INVALID": _FindingRule(
        "provenance_integrity",
        "Code de vérification non authentifié",
        "La signature du code 2D-Doc ne correspond pas aux données qu'il contient. Le code "
        "peut avoir été modifié ou être endommagé.",
        "invalid_signature_points",
    ),
    "TWO_D_DOC_VISIBLE_MISMATCH": _FindingRule(
        "document_integrity",
        "Données signées différentes du document",
        "Le code 2D-Doc est correctement signé, mais certains champs signés ne correspondent "
        "pas aux informations visibles. Il faut comparer le code avec le document affiché.",
        "structured_content_mismatch_points",
    ),
    "FACTURX_VISIBLE_MISMATCH": _FindingRule(
        "document_integrity",
        "Données de facture contradictoires",
        "Le PDF visible et les données de facture embarquées ne concordent pas sur certains "
        "champs. Ces deux représentations sont à comparer directement.",
        "structured_content_mismatch_points",
    ),
    "TWO_D_DOC_MALFORMED": _FindingRule(
        "document_integrity",
        "Code 2D-Doc illisible",
        "Le document annonce un code 2D-Doc, mais sa structure n'est pas exploitable. Une "
        "dégradation de lecture reste possible.",
        "malformed_structured_content_points",
    ),
    "FACTURX_XML_MALFORMED": _FindingRule(
        "document_integrity",
        "Données de facture embarquées illisibles",
        "Le PDF annonce une facture structurée, mais son fichier de données interne est "
        "incomplet ou illisible.",
        "malformed_structured_content_points",
    ),
    "FACTURX_XSD_INVALID": _FindingRule(
        "document_integrity",
        "Structure de facture électronique non conforme",
        "Les données de facture embarquées ne respectent pas le format annoncé. Il s'agit "
        "d'une anomalie de structure, pas à elle seule d'une preuve de modification.",
        "malformed_structured_content_points",
    ),
}


def build_document_authenticity_result(
    report: LaboratoryReport,
    config: AnalysisConfig,
) -> DetectorResult:
    """Convert only actionable authenticity observations into production findings."""

    findings: list[Finding] = []
    has_error = False
    for check in report.checks:
        if check.code not in {"pades", "post_signature", "facturx", "two_d_doc"}:
            continue
        has_error = has_error or check.state == "error"
        for observation in check.observations:
            rule = _RULES.get(observation.code)
            if rule is None or observation.state != "attention":
                continue
            findings.append(
                Finding(
                    detector="document_authenticity",
                    code=observation.code,
                    category=rule.category,
                    title=rule.title,
                    description=rule.description,
                    risk_points=float(getattr(config, rule.points_attribute)),
                    confidence=0.95 if observation.strength == "strong" else 0.80,
                    page=observation.page,
                    evidence={
                        "control": check.code,
                        "strength": observation.strength,
                        **observation.evidence,
                    },
                    artifacts=observation.artifacts,
                )
            )

    return DetectorResult(
        name="document_authenticity",
        status="partial" if has_error else "completed",
        findings=tuple(findings),
        notes=(
            "L'absence de signature, de 2D-Doc ou de Factur-X n'ajoute aucun point.",
            "Un certificat inconnu reste indéterminé et n'est pas traité comme une anomalie.",
            "Les divergences de contenu exigent plusieurs champs témoins retrouvés par l'OCR.",
        ),
    )
