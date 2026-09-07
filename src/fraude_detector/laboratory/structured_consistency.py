"""Conservative comparison of trusted structured fields with visible OCR text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date

from fraude_detector.models import (
    DocumentExtraction,
    ExtractedFact,
    LaboratoryCheck,
    LaboratoryObservation,
    LaboratoryReport,
)


@dataclass(frozen=True, slots=True)
class VisibleFieldComparison:
    """Names of structured fields found, or not found, in the visible text."""

    eligible_fields: tuple[str, ...]
    matched_fields: tuple[str, ...]
    unmatched_fields: tuple[str, ...]

    @property
    def matched_count(self) -> int:
        return len(self.matched_fields)

    @property
    def unmatched_count(self) -> int:
        return len(self.unmatched_fields)


def compare_visible_fields(
    fields: Mapping[str, str],
    visible_text: str,
) -> VisibleFieldComparison:
    """Find exact normalized values while tolerating punctuation and date formatting."""

    visible = _comparison_key(visible_text)
    eligible: list[str] = []
    matched: list[str] = []
    unmatched: list[str] = []
    for name, raw_value in fields.items():
        variants = _comparison_variants(str(raw_value))
        if not variants:
            continue
        eligible.append(name)
        if visible and any(variant in visible for variant in variants):
            matched.append(name)
        else:
            unmatched.append(name)
    return VisibleFieldComparison(
        eligible_fields=tuple(eligible),
        matched_fields=tuple(matched),
        unmatched_fields=tuple(unmatched),
    )


def field_label(name: str) -> str:
    """Turn one technical XML or 2D-Doc field name into readable French."""

    labels = {
        "invoice_number": "numéro de facture",
        "issue_date": "date d'émission",
        "seller_name": "nom de l'émetteur",
        "buyer_name": "nom du destinataire",
        "grand_total": "montant total",
        "due_amount": "montant à payer",
        "tax_total": "montant de taxe",
        "iban": "IBAN",
    }
    return labels.get(name, name.replace("_", " "))


def add_extraction_consistency_checks(
    report: LaboratoryReport,
    extraction: DocumentExtraction | None,
    *,
    minimum_matches: int,
) -> LaboratoryReport:
    """Add mismatches only when extracted facts provide an explicit conflicting value."""

    checks: list[LaboratoryCheck] = []
    for check in report.checks:
        if check.code not in {"facturx", "two_d_doc"}:
            checks.append(check)
            continue
        observations = list(check.observations)
        if extraction is not None:
            for observation in check.observations:
                structured_fields = observation.evidence.get("structured_fields")
                if not isinstance(structured_fields, Mapping):
                    continue
                comparison = _compare_with_extraction(
                    {str(key): str(value) for key, value in structured_fields.items()},
                    extraction,
                    source=check.code,
                )
                if (
                    comparison.matched_count >= minimum_matches
                    and comparison.unmatched_count > 0
                ):
                    observations.append(
                        _mismatch_observation(
                            check.code,
                            observation.page,
                            comparison,
                        )
                    )
        cleaned_observations = (
            tuple(_without_structured_values(item) for item in observations)
            if extraction is not None
            else tuple(observations)
        )
        state = (
            "attention"
            if any(item.state == "attention" for item in cleaned_observations)
            else check.state
        )
        summary = (
            "Les données structurées et le document visible présentent une divergence."
            if any(
                item.code in {"FACTURX_VISIBLE_MISMATCH", "TWO_D_DOC_VISIBLE_MISMATCH"}
                for item in cleaned_observations
            )
            else check.summary
        )
        checks.append(
            replace(
                check,
                state=state,
                summary=summary,
                observations=cleaned_observations,
            )
        )
    return replace(report, checks=tuple(checks))


def _compare_with_extraction(
    fields: Mapping[str, str],
    extraction: DocumentExtraction,
    *,
    source: str,
) -> VisibleFieldComparison:
    eligible: list[str] = []
    matched: list[str] = []
    unmatched: list[str] = []
    for name, value in fields.items():
        selectors = _fact_selectors(name, source)
        candidates = tuple(
            fact
            for fact in extraction.facts
            if any(_matches_selector(fact, selector) for selector in selectors)
        )
        if not candidates:
            continue
        eligible.append(name)
        if any(_values_match(value, fact.corrected_value or fact.raw_value) for fact in candidates):
            matched.append(name)
        elif len(candidates) == 1:
            unmatched.append(name)
    return VisibleFieldComparison(tuple(eligible), tuple(matched), tuple(unmatched))


def _fact_selectors(name: str, source: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if source == "facturx":
        return {
            "invoice_number": (("invoice_number", ("invoice", "document")),),
            "issue_date": (("date", ("issue",)),),
            "seller_name": (
                ("organization_name", ("issuer", "supplier", "provider")),
            ),
            "buyer_name": (
                ("person_name", ("recipient", "customer")),
                ("organization_name", ("recipient", "customer")),
            ),
            "grand_total": (("monetary_amount", ("total",)),),
            "tax_total": (("monetary_amount", ("tax",)),),
            "iban": (("iban", ()),),
        }.get(name, ())

    lowered = name.casefold()
    if "siret" in lowered:
        return (("registration_identifier", ("employer", "issuer")),)
    if lowered in {"numero_avis", "numero_permis", "numero_cni", "contrat"}:
        return (("document_number", ()), ("contract_number", ()))
    if "adresse" in lowered:
        return (("address", ()),)
    if lowered.startswith("date_") or lowered.startswith("periode_"):
        roles = (
            ("birth",)
            if "naissance" in lowered
            else ("expiry",)
            if any(word in lowered for word in ("expiration", "validite"))
            else ("issue",)
            if any(word in lowered for word in ("emission", "delivrance"))
            else ()
        )
        return (("date", roles),)
    if any(word in lowered for word in ("montant", "salaire", "revenu", "allocation")):
        return (("monetary_amount", ()),)
    if lowered.startswith("nom_employeur") or lowered in {"emetteur", "caisse"}:
        return (("organization_name", ("employer", "issuer")),)
    if lowered.startswith("nom") or lowered.startswith("prenom"):
        return (("person_name", ()),)
    if lowered in {"nir", "identifiant_pe", "vin", "immatriculation"}:
        return (
            ("other_identifier", ()),
            ("registration_identifier", ()),
            ("professional_identifier", ()),
        )
    return ()


def _matches_selector(
    fact: ExtractedFact,
    selector: tuple[str, tuple[str, ...]],
) -> bool:
    field_code, roles = selector
    return fact.field_code == field_code and (not roles or fact.role in roles)


def _values_match(left: str, right: str) -> bool:
    left_variants = _comparison_variants(left)
    right_variants = _comparison_variants(right)
    return any(
        left_value in right_value or right_value in left_value
        for left_value in left_variants
        for right_value in right_variants
    )


def _mismatch_observation(
    source: str,
    page: int | None,
    comparison: VisibleFieldComparison,
) -> LaboratoryObservation:
    labels = tuple(field_label(name) for name in comparison.unmatched_fields)
    prefix = "FACTURX" if source == "facturx" else "TWO_D_DOC"
    subject = "XML de facture" if source == "facturx" else "code 2D-Doc signé"
    return LaboratoryObservation(
        code=f"{prefix}_VISIBLE_MISMATCH",
        title="Données structurées différentes du document visible",
        summary=(
            f"{comparison.unmatched_count} champ(s) comparable(s) contiennent une autre valeur."
        ),
        state="attention",
        strength="strong",
        explanation=(
            f"Plusieurs champs du {subject} correspondent au document, mais pas les éléments "
            f"suivants : {', '.join(labels)}. Cette comparaison porte sur des valeurs "
            "explicitement extraites, pas sur une simple absence de lecture OCR."
        ),
        page=page,
        evidence={
            "matched_field_count": comparison.matched_count,
            "mismatched_field_count": comparison.unmatched_count,
            "matched_fields": comparison.matched_fields,
            "mismatched_fields": comparison.unmatched_fields,
        },
    )


def _without_structured_values(
    observation: LaboratoryObservation,
) -> LaboratoryObservation:
    if "structured_fields" not in observation.evidence:
        return observation
    evidence = dict(observation.evidence)
    evidence["structured_field_count"] = len(evidence.pop("structured_fields"))
    return replace(observation, evidence=evidence)


def _comparison_variants(value: str) -> tuple[str, ...]:
    base = _comparison_key(value)
    if len(base) < 4 or (base.isdigit() and len(base) < 5):
        return ()
    variants = {base}
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:
        for year_first in (True, False):
            try:
                if year_first:
                    parsed = date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))
                else:
                    parsed = date(int(digits[4:]), int(digits[2:4]), int(digits[:2]))
            except ValueError:
                continue
            variants.add(parsed.strftime("%Y%m%d"))
            variants.add(parsed.strftime("%d%m%Y"))
    return tuple(variants)


def _comparison_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if character.isalnum())
