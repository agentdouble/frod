"""Non-scoring, explainable checks over one GLM-OCR result."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Any

from fraude_detector.financial_identifiers import (
    IdentifierValidation,
    expected_iban_length,
    validate_belgian_rrn,
    validate_bic,
    validate_card_number,
    validate_eu_vat,
    validate_finess,
    validate_iban,
    validate_rpps,
    validate_siren,
    validate_siret,
)
from fraude_detector.models import LaboratoryCheck, LaboratoryObservation

_CARD_CONTEXT = re.compile(
    r"\b(?:"
    r"(?:credit|debit|payment)\s+card(?:\s+(?:account|no|number))?"
    r"|card\s+(?:account|no|number)"
    r"|(?:num[eé]ro|n[°o])\s+(?:de\s+)?carte"
    r"|carte\s+(?:bancaire|de\s+credit)"
    r"|kartennummer"
    r")\b",
    re.IGNORECASE,
)
_CARD_NUMBER = re.compile(r"(?<!\d)(?:\d[\s-]?){12,18}\d(?!\d)")
_CARD_GROUPED = re.compile(
    r"(?<![0-9A-Z*•])(?:[0-9X*•]{4}[ \t-]+){2,4}[0-9X*•]{1,4}"
    r"(?![0-9A-Z*•])",
    re.IGNORECASE,
)
_BIC_LABEL = re.compile(
    r"\b(?:"
    r"(?:BIC|SWIFT)(?:\s*[/|-]\s*(?:BIC|SWIFT))?(?:\s+(?:CODE|NO|NUMBER))?"
    r"|CODE\s+(?:BIC|SWIFT)"
    r")\b",
    re.IGNORECASE,
)
_IBAN_LABEL = re.compile(
    r"\bIBAN(?:\s+(?:NO|NUMBER|NUM[EÉ]RO))?\b",
    re.IGNORECASE,
)
_CKYC = re.compile(
    r"\bCKYC(?:\s+(?:ID|NO|NUMBER))?\b\s*[:.]?\s*([SLO]?\d{10,16})",
    re.IGNORECASE,
)
_MICR = re.compile(
    r"\bMICR(?:\s+(?:CODE|NO|NUMBER))?\b\s*[:.]?\s*(\d{6,18})",
    re.IGNORECASE,
)
_SIRET_LABEL = re.compile(r"\b(?:N(?:°|O)\s*)?SIRET\b", re.IGNORECASE)
_SIREN_LABEL = re.compile(r"\b(?:N(?:°|O)\s*)?SIREN(?!T)\b", re.IGNORECASE)
_VAT_LABEL = re.compile(
    r"\b(?:N(?:°|O)\s*)?(?:TVA(?:\s+INTRACOMMUNAUTAIRE)?|VAT(?:\s+(?:ID|NO|NUMBER))?)\b",
    re.IGNORECASE,
)
_RPPS_LABEL = re.compile(r"\b(?:N(?:°|O)\s*)?RPPS\b", re.IGNORECASE)
_FINESS_LABEL = re.compile(
    r"\b(?:N(?:°|O)\s*)?FINESS(?:\s+(?:EJ|ET|JURIDIQUE|G[EÉ]OGRAPHIQUE))?\b",
    re.IGNORECASE,
)
_BELGIAN_RRN_LABEL = re.compile(
    r"\b(?:"
    r"RRN"
    r"|RIJKSREGISTER(?:NUMMER|\s*(?:NR|NUMMER))"
    r"|NATIONALREGISTER(?:NUMMER)?"
    r"|NATIONAL\s+REGISTER\s+(?:NO|NUMBER)"
    r"|NUM[EÉ]RO\s+(?:DE\s+)?REGISTRE\s+NATIONAL"
    r"|NUM[EÉ]RO\s+NATIONAL"
    r")\b",
    re.IGNORECASE,
)
_EU_VAT_LENGTHS: dict[str, tuple[int, ...]] = {
    "AT": (11,),
    "BE": (12,),
    "BG": (11, 12),
    "CY": (11,),
    "CZ": (10, 11, 12),
    "DE": (11,),
    "DK": (10,),
    "EE": (11,),
    "EL": (11,),
    "ES": (11,),
    "FI": (10,),
    "FR": (13,),
    "HR": (13,),
    "HU": (10,),
    "IE": (10, 11),
    "IT": (13,),
    "LT": (11, 14),
    "LU": (10,),
    "LV": (13,),
    "MT": (10,),
    "NL": (14,),
    "PL": (12,),
    "PT": (11,),
    "RO": tuple(range(4, 13)),
    "SE": (14,),
    "SI": (10,),
    "SK": (12,),
    "XI": (11, 14),
}
_CURRENCY = re.compile(r"\b(EUR|USD|GBP|CHF|THB)\b", re.IGNORECASE)
_DATE_PATTERNS = (
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "%Y-%m-%d"),
    (
        re.compile(
            r"\b(\d{1,2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-"
            r"(\d{4})\b",
            re.IGNORECASE,
        ),
        "%d-%b-%Y",
    ),
)
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})([/.-])(\d{1,2})\2(\d{4})\b")


@dataclass(frozen=True, slots=True)
class _Region:
    page: int
    label: str
    content: str


@dataclass(frozen=True, slots=True)
class _ParsedDate:
    page: int
    raw: str
    value: date | None
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class _CardCandidate:
    value: str
    digits: str
    masked: bool
    possible_suffix: str | None


@dataclass(frozen=True, slots=True)
class _Table:
    page: int
    rows: tuple[tuple[str, ...], ...]


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, ...]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(tuple(self._row))
            self._row = None


def analyze_ocr_laboratory(
    ocr_json: Any,
    *,
    reference_date: date | None = None,
) -> tuple[LaboratoryCheck, ...]:
    """Analyze one OCR result without changing the Frod risk score."""

    regions = _regions(ocr_json)
    today = reference_date or date.today()
    return (
        _quality_check(ocr_json, regions),
        _identifier_check(regions),
        _date_check(regions, today),
        _financial_check(regions),
    )


def _quality_check(ocr_json: Any, regions: tuple[_Region, ...]) -> LaboratoryCheck:
    page_count = len(ocr_json) if isinstance(ocr_json, list) else 0
    character_count = sum(len(region.content.strip()) for region in regions)
    table_count = sum(region.label == "table" for region in regions)
    usable = len(regions) >= 2 and character_count >= 40
    state = "clear" if usable else "indeterminate"
    summary = (
        f"{len(regions)} zones exploitables sur {page_count} page(s)."
        if usable
        else "Le contenu reconnu est insuffisant pour des controles fiables."
    )
    return LaboratoryCheck(
        code="ocr_quality",
        title="Exploitabilite OCR",
        purpose="Verifier que le contenu extrait permet des controles documentaires.",
        state=state,
        summary=summary,
        observations=(
            LaboratoryObservation(
                code="OCR_CONTENT_INVENTORY",
                title="Contenu reconnu",
                summary=(
                    f"{character_count} caracteres, {len(regions)} zones, {table_count} tableau(x)."
                ),
                state=state,
                strength="informational",
                explanation=(
                    "Ce controle mesure uniquement la quantite de contenu structure "
                    "disponible, pas sa fidelite au document."
                ),
                evidence={
                    "pages": page_count,
                    "regions": len(regions),
                    "characters": character_count,
                    "tables": table_count,
                },
            ),
        ),
        limitations=(
            "GLM-OCR ne fournit pas ici de confiance par mot ou par zone.",
            "Un contenu abondant peut encore contenir des erreurs de lecture.",
        ),
    )


def _identifier_check(regions: tuple[_Region, ...]) -> LaboratoryCheck:
    observations: list[LaboratoryObservation] = []
    checked = 0
    unverifiable = 0
    anomaly_count = 0

    card_occurrences: dict[str, set[int]] = {}
    masked_card_occurrences: dict[str, tuple[set[int], str | None]] = {}
    suffixed_card_occurrences: dict[str, tuple[set[int], str]] = {}
    for region in regions:
        for candidate in _extract_labeled_cards(region.content):
            if candidate.masked:
                pages, _ = masked_card_occurrences.setdefault(
                    candidate.value,
                    (set(), candidate.possible_suffix),
                )
                pages.add(region.page)
            elif candidate.possible_suffix is not None:
                pages, _ = suffixed_card_occurrences.setdefault(
                    candidate.value,
                    (set(), candidate.possible_suffix),
                )
                pages.add(region.page)
            else:
                card_occurrences.setdefault(candidate.digits, set()).add(region.page)

    for value, (pages, suffix) in masked_card_occurrences.items():
        unverifiable += 1
        observations.append(
            LaboratoryObservation(
                code="OCR_CARD_MASKED",
                title="Numéro de carte masqué",
                summary=(
                    f"Le numéro est masqué et le groupe final « {suffix} » peut être un "
                    "suffixe de compte ; Luhn ne peut pas être calculé."
                    if suffix is not None
                    else "Le numéro est partiellement masqué ; Luhn ne peut pas être calculé."
                ),
                state="indeterminate",
                strength="informational",
                explanation=(
                    "Le masquage est courant sur les relevés bancaires. Il ne constitue "
                    "ni une anomalie ni une confirmation de validité."
                ),
                page=min(pages),
                evidence={
                    "value": value,
                    "visible_digits": sum(character.isdigit() for character in value),
                    "total_positions": len(value),
                    "possible_suffix": suffix,
                    "algorithm": "Luhn non applicable",
                },
            )
        )

    for value, (pages, suffix) in suffixed_card_occurrences.items():
        checked += 1
        full_validation = validate_card_number(value)
        base_value = value[: -len(suffix)]
        base_validation = validate_card_number(base_value)
        if full_validation.valid:
            code = "OCR_CARD_LUHN_VALID"
            state = "clear"
            strength = "informational"
            summary = (
                f"Le numéro complet de {len(value)} chiffres, suffixe « {suffix} » inclus, "
                "respecte Luhn."
            )
        elif base_validation.valid:
            code = "OCR_CARD_SUFFIX_SEPARATED"
            state = "clear"
            strength = "informational"
            summary = (
                f"Les {len(base_value)} premiers chiffres respectent Luhn ; le groupe "
                f"final « {suffix} » est probablement un suffixe distinct."
            )
        else:
            code = "OCR_CARD_LUHN_INVALID"
            state = "attention"
            strength = "moderate"
            anomaly_count += 1
            summary = (
                "Ni le numéro complet ni la variante sans le suffixe séparé ne respectent Luhn."
            )
        observations.append(
            LaboratoryObservation(
                code=code,
                title="Numéro de carte avec suffixe possible",
                summary=summary,
                state=state,
                strength=strength,
                explanation=(
                    "Le contrôle teste le PAN complet puis la variante sans le dernier groupe. "
                    "Les espaces seuls ne suffisent jamais à supprimer des chiffres."
                ),
                page=min(pages),
                evidence={
                    "value": value,
                    "possible_suffix": suffix,
                    "full_length": len(value),
                    "full_pan_valid": full_validation.valid,
                    "base_value": base_value,
                    "base_length": len(base_value),
                    "base_pan_valid": base_validation.valid,
                    "algorithm": "Luhn, double hypothèse",
                },
            )
        )

    if len(card_occurrences) > 1:
        difference = _single_character_difference(tuple(card_occurrences))
        likely_conflict = difference == 1
        anomaly_count += int(likely_conflict)
        observations.append(
            LaboratoryObservation(
                code=(
                    "OCR_CARD_VALUES_CONFLICT" if likely_conflict else "OCR_CARD_MULTIPLE_VALUES"
                ),
                title=(
                    "Numeros de carte presque identiques"
                    if likely_conflict
                    else "Plusieurs numeros de carte"
                ),
                summary=(
                    "Deux numeros de meme longueur ne different que d'un caractere."
                    if likely_conflict
                    else (
                        f"Le document contient {len(card_occurrences)} numeros distincts "
                        f"({', '.join(value[-4:] for value in card_occurrences)})."
                    )
                ),
                state="attention" if likely_conflict else "clear",
                strength="moderate" if likely_conflict else "informational",
                explanation=(
                    (
                        "Une variation d'un seul caractere peut correspondre a une retouche "
                        "ou a une erreur OCR; les deux zones doivent etre confrontees a l'image."
                    )
                    if likely_conflict
                    else (
                        "Plusieurs cartes peuvent legitimement figurer dans un meme document. "
                        "Cette observation reste donc sans effet sur le score."
                    )
                ),
                page=min(min(pages) for pages in card_occurrences.values()),
                evidence={
                    "values": tuple(card_occurrences),
                    "distinct_values": len(card_occurrences),
                    "differing_characters": difference,
                },
            )
        )

    for digits, pages in card_occurrences.items():
        checked += 1
        validation = validate_card_number(digits)
        anomaly_count += int(not validation.valid)
        observations.append(
            LaboratoryObservation(
                code=("OCR_CARD_LUHN_VALID" if validation.valid else "OCR_CARD_LUHN_INVALID"),
                title="Numero de carte",
                summary=(
                    f"Le numero se terminant par {digits[-4:]} respecte Luhn."
                    if validation.valid
                    else f"Le numero se terminant par {digits[-4:]} ne respecte pas Luhn."
                ),
                state="clear" if validation.valid else "attention",
                strength="informational" if validation.valid else "moderate",
                explanation=(
                    "Luhn controle la coherence mathematique du numero. "
                    "Il ne confirme ni son existence ni son emetteur."
                ),
                page=min(pages),
                evidence={
                    "value": digits,
                    "length": len(digits),
                    "occurrences": len(pages),
                    "algorithm": "Luhn",
                    "invalid_reasons": validation.reasons,
                },
            )
        )

    bic_occurrences: dict[str, set[int]] = {}
    iban_occurrences: dict[str, set[int]] = {}
    for region in regions:
        for candidate in _extract_labeled_bics(region.content):
            bic_occurrences.setdefault(candidate, set()).add(region.page)

        for candidate in _extract_labeled_ibans(region.content):
            iban_occurrences.setdefault(candidate, set()).add(region.page)

    for candidate, pages in bic_occurrences.items():
        checked += 1
        validation = validate_bic(candidate)
        anomaly_count += int(not validation.valid)
        observations.append(_bic_observation(validation, min(pages)))

    for candidate, pages in iban_occurrences.items():
        checked += 1
        validation = validate_iban(candidate)
        anomaly_count += int(not validation.valid)
        observations.append(_iban_observation(validation, min(pages)))

    labeled_identifiers = (
        (
            "SIREN",
            _labeled_digit_occurrences(regions, _SIREN_LABEL, 9),
            validate_siren,
            "SIREN français de 9 chiffres et clé de Luhn",
        ),
        (
            "SIRET",
            _labeled_digit_occurrences(regions, _SIRET_LABEL, 14),
            validate_siret,
            "SIRET français de 14 chiffres et clé de contrôle",
        ),
        (
            "RPPS",
            _labeled_digit_occurrences(regions, _RPPS_LABEL, 11),
            validate_rpps,
            "RPPS français de 11 chiffres commençant par 1 et clé de Luhn",
        ),
        (
            "FINESS",
            _labeled_digit_occurrences(regions, _FINESS_LABEL, 9),
            validate_finess,
            "FINESS français de 9 chiffres et clé de Luhn",
        ),
        (
            "BELGIAN_RRN",
            _labeled_digit_occurrences(regions, _BELGIAN_RRN_LABEL, 11),
            validate_belgian_rrn,
            "RRN belge de 11 chiffres, date encodée et clé modulo 97",
        ),
    )
    for identifier_type, occurrences, validator, standard in labeled_identifiers:
        for candidate, pages in occurrences.items():
            checked += 1
            validation = validator(candidate)
            anomaly_count += int(not validation.valid)
            observations.append(
                _national_identifier_observation(
                    identifier_type,
                    validation,
                    min(pages),
                    standard,
                )
            )

    vat_occurrences = _labeled_vat_occurrences(regions)
    for candidate, pages in vat_occurrences.items():
        checked += 1
        validation = validate_eu_vat(candidate)
        anomaly_count += int(not validation.valid)
        observations.append(_vat_observation(validation, min(pages)))

    ckyc_occurrences: list[tuple[str, int]] = []
    micr_occurrences: list[tuple[str, int]] = []
    for region in regions:
        for match in _CKYC.finditer(region.content):
            candidate = match.group(1).upper()
            ckyc_occurrences.append((candidate, region.page))

        for match in _MICR.finditer(region.content):
            candidate = match.group(1)
            micr_occurrences.append((candidate, region.page))

    for candidate, page in ckyc_occurrences:
        checked += 1
        digits = candidate[1:] if candidate[:1] in {"S", "L", "O"} else candidate
        valid = len(digits) == 14
        anomaly_count += int(not valid)
        observations.append(
            LaboratoryObservation(
                code="OCR_CKYC_VALID" if valid else "OCR_CKYC_INVALID",
                title="Identifiant CKYC",
                summary=(
                    "L'identifiant CKYC respecte le format indien a 14 chiffres."
                    if valid
                    else (f"L'identifiant CKYC contient {len(digits)} chiffres au lieu de 14.")
                ),
                state="clear" if valid else "attention",
                strength="informational" if valid else "moderate",
                explanation=(
                    "CKYC est un identifiant du registre KYC central indien. "
                    "Le format ne confirme pas l'existence de l'enregistrement."
                ),
                page=page,
                evidence={
                    "value": candidate,
                    "digit_count": len(digits),
                    "jurisdiction": "IN",
                },
            )
        )

    indian_micr_context = bool(ckyc_occurrences)
    for candidate, page in micr_occurrences:
        checked += 1
        if indian_micr_context:
            valid = len(candidate) == 9
            anomaly_count += int(not valid)
            observations.append(
                LaboratoryObservation(
                    code="OCR_MICR_VALID" if valid else "OCR_MICR_INVALID",
                    title="Code MICR",
                    summary=(
                        "Le code MICR respecte le format indien a 9 chiffres."
                        if valid
                        else f"Le code MICR contient {len(candidate)} chiffres au lieu de 9."
                    ),
                    state="clear" if valid else "attention",
                    strength="informational" if valid else "moderate",
                    explanation=(
                        "La presence de CKYC fournit ici le contexte indien necessaire "
                        "pour interpreter le champ MICR."
                    ),
                    page=page,
                    evidence={
                        "value": candidate,
                        "digit_count": len(candidate),
                        "jurisdiction": "IN",
                    },
                )
            )
        else:
            observations.append(
                LaboratoryObservation(
                    code="OCR_MICR_OBSERVED",
                    title="Champ MICR",
                    summary=f"Un champ MICR de {len(candidate)} chiffres a ete reconnu.",
                    state="clear",
                    strength="informational",
                    explanation=(
                        "MICR est utilise dans plusieurs systemes bancaires. Sans autre "
                        "contexte national, sa longueur n'est pas qualifiee comme valide "
                        "ou invalide."
                    ),
                    page=page,
                    evidence={
                        "value": candidate,
                        "digit_count": len(candidate),
                        "jurisdiction": None,
                    },
                )
            )

    banking_country, country_evidence = _banking_country_context(
        regions,
        bic_occurrences,
    )
    if banking_country and banking_country != "IN" and ckyc_occurrences:
        anomaly_count += 1
        identifier_types = {"CKYC"}
        if micr_occurrences:
            identifier_types.add("MICR")
        corroborated = bool(micr_occurrences)
        country_label = (
            "luxembourgeois" if banking_country == "LU" else f"rattache au pays {banking_country}"
        )
        observations.append(
            LaboratoryObservation(
                code=(
                    "OCR_BANKING_GEOGRAPHY_MISMATCH"
                    if corroborated
                    else "OCR_BANKING_FOREIGN_FIELD"
                ),
                title="Referentiels bancaires de pays differents",
                summary=(
                    f"Le contexte bancaire est {country_label}, mais le document "
                    f"emploie des champs indiens ({', '.join(sorted(identifier_types))})."
                ),
                state="attention",
                strength="moderate" if corroborated else "weak",
                explanation=(
                    (
                        "La presence conjointe de CKYC et MICR peut provenir d'un modele "
                        "indien reutilise. Elle reste a confirmer sur l'image."
                    )
                    if corroborated
                    else (
                        "Un champ etranger isole peut etre legitime et reste un diagnostic "
                        "sans effet sur le score."
                    )
                ),
                evidence={
                    "banking_country": banking_country,
                    "country_evidence": country_evidence,
                    "foreign_identifier_jurisdiction": "IN",
                    "foreign_identifier_types": tuple(sorted(identifier_types)),
                    "corroborated": corroborated,
                },
            )
        )

    if checked == 0 and unverifiable == 0:
        return LaboratoryCheck(
            code="ocr_identifiers",
            title="Identifiants structurés",
            purpose=(
                "Vérifier les identifiants bancaires, d'entreprise et de santé explicitement "
                "reconnus."
            ),
            state="not_applicable",
            summary="Aucun identifiant compatible n'a ete reconnu.",
            limitations=("Les identifiants non etiquetes explicitement ne sont pas devines.",),
        )

    return LaboratoryCheck(
        code="ocr_identifiers",
        title="Identifiants structurés",
        purpose=(
            "Vérifier les identifiants bancaires, d'entreprise et de santé explicitement reconnus."
        ),
        state="attention" if anomaly_count else "clear",
        summary=(
            f"{anomaly_count} anomalie(s) pour {checked} identifiant(s) controles."
            if anomaly_count
            else (
                f"{checked} identifiant(s) contrôlé(s), {unverifiable} non vérifiable(s)."
                if unverifiable
                else f"{checked} identifiant(s) respectent leur controle de structure."
            )
        ),
        observations=tuple(observations),
        limitations=(
            "Une erreur OCR d'un seul caractere suffit a invalider un controle.",
            "Un checksum valide ne confirme pas que l'identifiant existe.",
            "La coherence geographique exige un contexte bancaire explicite.",
            "Aucun registre d'entreprises ou de professionnels n'est interrogé par ce contrôle.",
        ),
    )


def _bic_observation(
    validation: IdentifierValidation,
    page: int,
) -> LaboratoryObservation:
    reason = _validation_reason(validation.reasons)
    return LaboratoryObservation(
        code="OCR_BIC_VALID" if validation.valid else "OCR_BIC_INVALID",
        title="BIC / SWIFT",
        summary=(
            f"Le BIC {validation.normalized} respecte la structure internationale."
            if validation.valid
            else f"Le BIC {validation.normalized} est invalide : {reason}."
        ),
        state="clear" if validation.valid else "attention",
        strength="informational" if validation.valid else "moderate",
        explanation=(
            "Le controle couvre la longueur, les positions et le code pays ISO 9362. "
            "Il ne prouve pas que le BIC est attribue a une banque."
        ),
        page=page,
        evidence={
            "value": validation.normalized,
            "length": len(validation.normalized),
            "country_code": validation.country_code,
            "invalid_reasons": validation.reasons,
            "standard": "ISO 9362",
        },
    )


def _iban_observation(
    validation: IdentifierValidation,
    page: int,
) -> LaboratoryObservation:
    candidate = validation.normalized
    reason = _validation_reason(validation.reasons)
    return LaboratoryObservation(
        code="OCR_IBAN_VALID" if validation.valid else "OCR_IBAN_INVALID",
        title="IBAN",
        summary=(
            f"L'IBAN {candidate[:2]} se terminant par {candidate[-4:]} respecte "
            "le format national et Mod-97."
            if validation.valid
            else (
                f"L'IBAN {candidate[:2]} se terminant par {candidate[-4:]} est invalide : {reason}."
            )
        ),
        state="clear" if validation.valid else "attention",
        strength="informational" if validation.valid else "moderate",
        explanation=(
            "Le controle applique la structure propre au pays et le checksum ISO 13616. "
            "Il ne confirme pas l'existence du compte."
        ),
        page=page,
        evidence={
            "value": candidate,
            "length": len(candidate),
            "country_code": validation.country_code,
            "invalid_reasons": validation.reasons,
            "algorithm": "ISO 13616 Mod-97",
        },
    )


def _national_identifier_observation(
    identifier_type: str,
    validation: IdentifierValidation,
    page: int,
    standard: str,
) -> LaboratoryObservation:
    reason = _validation_reason(validation.reasons)
    display_name = "RRN belge" if identifier_type == "BELGIAN_RRN" else identifier_type
    return LaboratoryObservation(
        code=(
            f"OCR_{identifier_type}_VALID" if validation.valid else f"OCR_{identifier_type}_INVALID"
        ),
        title=display_name,
        summary=(
            f"Le numéro {display_name} respecte sa structure et sa clé de contrôle."
            if validation.valid
            else f"Le numéro {display_name} est invalide : {reason}."
        ),
        state="clear" if validation.valid else "attention",
        strength="informational" if validation.valid else "moderate",
        explanation=(
            "Ce contrôle détecte certaines erreurs de saisie ou de lecture OCR. "
            "Il ne confirme ni l'attribution du numéro ni l'activité de son titulaire."
        ),
        page=page,
        evidence={
            "value": validation.normalized,
            "length": len(validation.normalized),
            "country_code": validation.country_code,
            "invalid_reasons": validation.reasons,
            "standard": standard,
            "registry_checked": False,
        },
    )


def _vat_observation(
    validation: IdentifierValidation,
    page: int,
) -> LaboratoryObservation:
    reason = _validation_reason(validation.reasons)
    return LaboratoryObservation(
        code="OCR_EU_VAT_VALID" if validation.valid else "OCR_EU_VAT_INVALID",
        title="Numéro de TVA intracommunautaire",
        summary=(
            f"Le numéro de TVA {validation.normalized} respecte les règles du pays."
            if validation.valid
            else f"Le numéro de TVA {validation.normalized} est invalide : {reason}."
        ),
        state="clear" if validation.valid else "attention",
        strength="informational" if validation.valid else "moderate",
        explanation=(
            "La structure et la clé nationale sont vérifiées localement. L'inscription "
            "effective de l'entreprise dans VIES n'est pas interrogée."
        ),
        page=page,
        evidence={
            "value": validation.normalized,
            "country_code": validation.country_code,
            "invalid_reasons": validation.reasons,
            "registry_checked": False,
            "registry": "VIES non interrogé",
        },
    )


def _validation_reason(reasons: tuple[str, ...]) -> str:
    labels = {
        "invalid_length": "longueur non conforme",
        "invalid_format": "caracteres ou positions non conformes",
        "invalid_checksum": "checksum invalide",
        "invalid_institution_code": "code institution invalide",
        "invalid_country_code": "code pays invalide",
        "unknown_country_code": "code pays inconnu",
        "invalid_location_code": "code localisation invalide",
        "invalid_branch_code": "code agence invalide",
        "invalid_component": "composant invalide",
        "unknown_country_or_structure": "pays inconnu ou structure nationale absente",
        "invalid_country_structure": "longueur ou structure nationale non conforme",
    }
    return ", ".join(labels.get(reason, reason) for reason in reasons) or "format invalide"


def _extract_labeled_cards(text: str) -> tuple[_CardCandidate, ...]:
    candidates: list[_CardCandidate] = []
    for context in _CARD_CONTEXT.finditer(text):
        nearby_text = text[context.end() : context.end() + 80]
        grouped = _CARD_GROUPED.search(nearby_text)
        numeric = _CARD_NUMBER.search(nearby_text)
        matches = [match for match in (grouped, numeric) if match is not None]
        if not matches:
            continue
        match = min(matches, key=lambda item: (item.start(), -len(item.group())))
        raw = match.group().strip()
        groups = tuple(part for part in re.split(r"[ \t-]+", raw) if part)
        normalized = "".join(groups).upper().replace("*", "X").replace("•", "X")
        masked = "X" in normalized
        possible_suffix = groups[-1] if len(groups) >= 5 and 1 <= len(groups[-1]) <= 2 else None
        candidates.append(
            _CardCandidate(
                value=normalized,
                digits="".join(character for character in normalized if character.isdigit()),
                masked=masked,
                possible_suffix=possible_suffix,
            )
        )
    return tuple(dict.fromkeys(candidates))


def _extract_labeled_ibans(text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for label in _IBAN_LABEL.finditer(text):
        tail = text[label.end() : label.end() + 128].lstrip(" \t:.-")
        prefix = re.match(r"([A-Z]{2})\s*(\d{2})", tail, re.IGNORECASE)
        if prefix is None:
            continue
        expected_length = expected_iban_length(prefix.group(1))
        if expected_length is None:
            continue

        compact: list[str] = []
        source = tail[prefix.start() :]
        for index, character in enumerate(source):
            if character.isascii() and character.isalnum():
                compact.append(character.upper())
            elif character in " \t-\u00a0":
                continue
            else:
                break
            if len(compact) == expected_length:
                extra_digits = re.match(
                    r"[ \t\-\u00a0]+((?:\d[ \t\-\u00a0]*){1,14})"
                    r"(?=$|[^A-Z0-9])",
                    source[index + 1 :],
                    re.IGNORECASE,
                )
                if extra_digits is not None:
                    compact.extend(re.sub(r"\D", "", extra_digits.group(1)))
                candidates.append("".join(compact))
                break
    return tuple(dict.fromkeys(candidates))


def _extract_labeled_bics(text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for label in _BIC_LABEL.finditer(text):
        tail = text[label.end() : label.end() + 40].lstrip(" \t:.-")
        contiguous = re.match(r"[A-Z0-9]{8,}", tail, re.IGNORECASE)
        if contiguous is not None:
            value = contiguous.group(0).upper()
            candidates.append(value)
            continue

        grouped = re.match(
            r"([A-Z0-9]{4})[ \t]+([A-Z0-9]{2})[ \t]+([A-Z0-9]{2})"
            r"(?:[ \t]+([A-Z0-9]{3}))?",
            tail,
            re.IGNORECASE,
        )
        if grouped is not None:
            candidates.append("".join(part or "" for part in grouped.groups()).upper())
    return tuple(dict.fromkeys(candidates))


def _labeled_digit_occurrences(
    regions: tuple[_Region, ...],
    label_pattern: re.Pattern[str],
    length: int,
) -> dict[str, set[int]]:
    occurrences: dict[str, set[int]] = {}
    for region in regions:
        for label in label_pattern.finditer(region.content):
            tail = region.content[label.end() : label.end() + 80].lstrip(" \t:.-")
            pattern = re.compile(rf"(?:\d[ \t.\-]*){{{length - 1}}}\d(?![ \t.\-]*\d)")
            match = pattern.match(tail)
            if match is None:
                continue
            candidate = re.sub(r"\D", "", match.group())
            occurrences.setdefault(candidate, set()).add(region.page)
    return occurrences


def _labeled_vat_occurrences(regions: tuple[_Region, ...]) -> dict[str, set[int]]:
    occurrences: dict[str, set[int]] = {}
    for region in regions:
        for label in _VAT_LABEL.finditer(region.content):
            tail = region.content[label.end() : label.end() + 80].lstrip(" \t:.-")
            prefix = re.match(r"[A-Z]{2}", tail, re.IGNORECASE)
            if prefix is None:
                continue
            country = prefix.group().upper()
            lengths = _EU_VAT_LENGTHS.get(country)
            if not lengths:
                continue
            candidates = [
                item for length in lengths if (item := _take_alphanumeric(tail, length)) is not None
            ]
            if not candidates:
                continue
            valid = [item for item in candidates if validate_eu_vat(item[0]).valid]
            if valid:
                candidate = max(valid, key=lambda item: len(item[0]))[0]
            else:
                candidate = max(
                    candidates,
                    key=lambda item: (_candidate_boundary_score(tail, item[1]), len(item[0])),
                )[0]
            occurrences.setdefault(candidate, set()).add(region.page)
    return occurrences


def _take_alphanumeric(source: str, length: int) -> tuple[str, int] | None:
    compact: list[str] = []
    for index, character in enumerate(source):
        if character.isascii() and character.isalnum():
            compact.append(character.upper())
        elif character in " \t.- ":
            continue
        else:
            return None
        if len(compact) == length:
            return "".join(compact), index + 1
    return None


def _candidate_boundary_score(source: str, consumed: int) -> int:
    remainder = source[consumed:]
    if not remainder or remainder[0] in "\n\r,;|)/]}":
        return 2
    if re.match(r"[ \t]+[A-Z]{3,}\b", remainder, re.IGNORECASE):
        return 1
    return 0


def _single_character_difference(values: tuple[str, ...]) -> int | None:
    if len(values) != 2 or len(values[0]) != len(values[1]):
        return None
    return sum(left != right for left, right in zip(*values, strict=True))


def _banking_country_context(
    regions: tuple[_Region, ...],
    bic_occurrences: dict[str, set[int]],
) -> tuple[str | None, tuple[str, ...]]:
    evidence: list[tuple[str, str]] = []
    for candidate in bic_occurrences:
        validation = validate_bic(candidate)
        if validation.valid and validation.country_code:
            evidence.append((validation.country_code, "BIC"))
    for region in regions:
        normalized = _normalize_label(region.content)
        if "branch address" in normalized and "luxembourg" in normalized:
            evidence.append(("LU", "adresse agence"))
        if "account branch" in normalized and "luxembourg" in normalized:
            evidence.append(("LU", "agence du compte"))

    countries = {country for country, _ in evidence}
    if len(countries) != 1:
        return None, tuple(label for _, label in evidence)
    return next(iter(countries)), tuple(dict.fromkeys(label for _, label in evidence))


def _date_check(regions: tuple[_Region, ...], reference_date: date) -> LaboratoryCheck:
    dates = _extract_dates(regions)
    if not dates:
        return LaboratoryCheck(
            code="ocr_dates",
            title="Coherence des dates",
            purpose="Rechercher des dates invalides ou posterieures a l'analyse.",
            state="not_applicable",
            summary="Aucune date exploitable n'a ete reconnue.",
        )

    observations: list[LaboratoryObservation] = []
    invalid = [item for item in dates if item.value is None]
    future = [
        item
        for item in dates
        if item.value is not None
        and not item.ambiguous
        and item.value > reference_date + timedelta(days=1)
    ]
    for item in invalid:
        observations.append(
            LaboratoryObservation(
                code="OCR_DATE_INVALID",
                title="Date invalide",
                summary=f"La date reconnue « {item.raw} » n'existe pas.",
                state="attention",
                strength="moderate",
                explanation="Le calendrier ne permet pas cette combinaison jour/mois/annee.",
                page=item.page,
            )
        )
    for item in future:
        observations.append(
            LaboratoryObservation(
                code="OCR_DATE_IN_FUTURE",
                title="Date postérieure à l'analyse",
                summary=f"La date reconnue « {item.raw} » est posterieure a l'analyse.",
                state="detected",
                strength="informational",
                explanation=(
                    "Une date d'expiration, d'échéance ou de validité peut légitimement "
                    "être future. Sans rôle sémantique fiable, elle reste informative."
                ),
                page=item.page,
            )
        )
    valid_values = [item.value for item in dates if item.value is not None]
    if valid_values:
        observations.append(
            LaboratoryObservation(
                code="OCR_DATE_INVENTORY",
                title="Dates reconnues",
                summary=(
                    f"{len(valid_values)} occurrence(s), de "
                    f"{min(valid_values).isoformat()} a {max(valid_values).isoformat()}."
                ),
                state="clear",
                strength="informational",
                explanation=("L'inventaire seul ne permet pas d'etablir une chronologie metier."),
                evidence={"occurrences": len(valid_values)},
            )
        )

    anomaly_count = len(invalid)
    return LaboratoryCheck(
        code="ocr_dates",
        title="Coherence des dates",
        purpose="Rechercher des dates invalides ou posterieures a l'analyse.",
        state="attention" if anomaly_count else "clear",
        summary=(
            f"{anomaly_count} date(s) demandent une verification."
            if anomaly_count
            else f"{len(valid_values)} date(s) lisibles, aucune anomalie generique."
        ),
        observations=tuple(observations),
        limitations=(
            "Les relations metier entre dates exigent d'abord de reconnaitre le type de document.",
        ),
    )


def _financial_check(regions: tuple[_Region, ...]) -> LaboratoryCheck:
    tables = _tables(regions)
    if not tables:
        return LaboratoryCheck(
            code="ocr_financial_consistency",
            title="Coherence des montants",
            purpose="Recalculer les totaux et soldes presents dans les tableaux reconnus.",
            state="not_applicable",
            summary="Aucun tableau financier exploitable n'a ete reconnu.",
        )

    full_text = "\n".join(region.content for region in regions)
    account_currency = _account_currency(full_text)
    observations: list[LaboratoryObservation] = []
    supported_checks = 0
    anomaly_count = 0

    for table in tables:
        summary = _statement_summary_observation(table)
        if summary is not None:
            supported_checks += 1
            anomaly_count += int(summary.state == "attention")
            observations.append(summary)

        ledger = _ledger_observation(table, account_currency)
        if ledger is not None:
            supported_checks += 1
            anomaly_count += int(ledger.state == "attention")
            observations.append(ledger)

        invoice = _invoice_total_observation(table)
        if invoice is not None:
            supported_checks += 1
            anomaly_count += int(invoice.state == "attention")
            observations.append(invoice)

    if supported_checks == 0:
        return LaboratoryCheck(
            code="ocr_financial_consistency",
            title="Coherence des montants",
            purpose="Recalculer les totaux et soldes presents dans les tableaux reconnus.",
            state="indeterminate",
            summary="Des tableaux existent, mais aucune formule standard n'est identifiable.",
            limitations=(
                "Les tableaux libres ne sont pas interpretes sans libelles financiers explicites.",
            ),
        )

    return LaboratoryCheck(
        code="ocr_financial_consistency",
        title="Coherence des montants",
        purpose="Recalculer les totaux et soldes presents dans les tableaux reconnus.",
        state="attention" if anomaly_count else "clear",
        summary=(
            f"{anomaly_count} incoherence(s) sur {supported_checks} calcul(s) controle(s)."
            if anomaly_count
            else f"{supported_checks} calcul(s) controles sans incoherence."
        ),
        observations=tuple(observations),
        limitations=(
            "Les lignes en devise etrangere sont ignorees sans montant converti explicite.",
            "Les separateurs numeriques ambigus peuvent provenir de l'OCR.",
        ),
    )


def _statement_summary_observation(table: _Table) -> LaboratoryObservation | None:
    if len(table.rows) < 2:
        return None
    headers = tuple(_normalize_label(cell) for cell in table.rows[0])
    opening = _column(
        headers,
        "opening balance",
        "beginning balance",
        "previous balance",
        "balance brought forward",
        "solde initial",
        "solde d ouverture",
        "ancien solde",
    )
    debit = _column(
        headers,
        "total debit",
        "debit total",
        "total withdrawals",
        "total des debits",
    )
    credit = _column(
        headers,
        "total credit",
        "credit total",
        "total deposits",
        "total des credits",
    )
    closing = _column(
        headers,
        "pending balance",
        "closing balance",
        "ending balance",
        "current balance",
        "new balance",
        "solde final",
        "solde de cloture",
        "nouveau solde",
    )
    if None in {opening, debit, credit, closing}:
        return None
    row = table.rows[1]
    indexes = (opening, debit, credit, closing)
    if any(index is None or index >= len(row) for index in indexes):
        return None
    values = tuple(_parse_amount(row[index]) for index in indexes if index is not None)
    if len(values) != 4 or any(value is None for value in values):
        return None
    opening_value, debit_value, credit_value, closing_value = values
    assert opening_value is not None
    assert debit_value is not None
    assert credit_value is not None
    assert closing_value is not None
    expected = opening_value - debit_value + credit_value
    delta = abs(expected - closing_value)
    valid = delta <= Decimal("0.02")
    return LaboratoryObservation(
        code="OCR_STATEMENT_SUMMARY_VALID" if valid else "OCR_STATEMENT_SUMMARY_MISMATCH",
        title="Synthese du releve",
        summary=(
            "Le solde recapitulatif concorde avec les totaux."
            if valid
            else "Le solde recapitulatif ne concorde pas avec les totaux."
        ),
        state="clear" if valid else "attention",
        strength="informational" if valid else "strong",
        explanation="Calcul applique : solde initial - debits + credits = solde final.",
        page=table.page,
        evidence={
            "expected": str(expected),
            "observed": str(closing_value),
            "absolute_delta": str(delta),
        },
    )


def _ledger_observation(
    table: _Table,
    account_currency: str | None,
) -> LaboratoryObservation | None:
    if len(table.rows) < 2:
        return None
    headers = tuple(_normalize_label(cell) for cell in table.rows[0])
    transaction_date = _column(
        headers,
        "transaction date",
        "posting date",
        "booking date",
        "date transaction",
        "date operation",
        "date comptable",
        "buchungsdatum",
    )
    debit = _column(headers, "debit", "withdrawal", "retrait", "belastung")
    credit = _column(headers, "credit", "deposit", "versement", "gutschrift")
    balance = _column(headers, "balance", "solde", "kontostand")
    if None in {transaction_date, debit, credit, balance}:
        return None

    previous_date: date | None = None
    date_direction: int | None = None
    previous_balance: Decimal | None = None
    inversions: list[int] = []
    mismatches: list[int] = []
    checked_rows = 0
    skipped_currency_rows = 0

    for row_number, row in enumerate(table.rows[1:], start=1):
        indexes = (transaction_date, debit, credit, balance)
        if any(index is None or index >= len(row) for index in indexes):
            continue
        assert transaction_date is not None
        assert debit is not None
        assert credit is not None
        assert balance is not None

        raw_date = row[transaction_date].strip()
        parsed_date = _parse_date_text(raw_date)
        if parsed_date is not None:
            if previous_date is not None and parsed_date != previous_date:
                direction = 1 if parsed_date > previous_date else -1
                if date_direction is None:
                    date_direction = direction
                elif direction != date_direction:
                    inversions.append(row_number)
            previous_date = parsed_date

        current_balance = _parse_amount(row[balance])
        if current_balance is None:
            continue
        debit_value = _parse_amount(row[debit])
        credit_value = _parse_amount(row[credit])
        explicit_currency = _explicit_currency(row[debit]) or _explicit_currency(row[credit])
        comparable = explicit_currency is None or (
            account_currency is not None and explicit_currency == account_currency
        )
        if (
            previous_balance is not None
            and (debit_value is not None or credit_value is not None)
            and comparable
        ):
            expected = previous_balance - (debit_value or Decimal(0)) + (credit_value or Decimal(0))
            checked_rows += 1
            if abs(expected - current_balance) > Decimal("0.02"):
                mismatches.append(row_number)
        elif explicit_currency is not None and not comparable:
            skipped_currency_rows += 1
        previous_balance = current_balance

    if checked_rows == 0 and not inversions:
        return LaboratoryObservation(
            code="OCR_LEDGER_INDETERMINATE",
            title="Evolution du solde",
            summary="Aucune ligne n'est comparable de maniere fiable.",
            state="indeterminate",
            strength="informational",
            explanation="Les devises ou les cellules manquantes empechent le recalcul.",
            page=table.page,
            evidence={"foreign_currency_rows_skipped": skipped_currency_rows},
        )

    valid = not mismatches and not inversions
    return LaboratoryObservation(
        code="OCR_LEDGER_VALID" if valid else "OCR_LEDGER_MISMATCH",
        title="Evolution du solde",
        summary=(
            f"{checked_rows} ligne(s) recalculees sans incoherence."
            if valid
            else (
                f"{len(mismatches)} solde(s) incoherent(s) et "
                f"{len(inversions)} rupture(s) chronologique(s)."
            )
        ),
        state="clear" if valid else "attention",
        strength="informational" if valid else "strong",
        explanation=(
            "Chaque solde comparable est recalcule depuis le solde precedent, "
            "le debit et le credit."
        ),
        page=table.page,
        evidence={
            "rows_checked": checked_rows,
            "balance_mismatch_rows": mismatches,
            "chronology_inversion_rows": inversions,
            "foreign_currency_rows_skipped": skipped_currency_rows,
            "account_currency": account_currency,
        },
    )


def _invoice_total_observation(table: _Table) -> LaboratoryObservation | None:
    amounts: dict[str, Decimal] = {}
    aliases = {
        "subtotal": (
            "subtotal",
            "sub total",
            "total ht",
            "total hors taxe",
            "net total",
            "net amount",
            "nettobetrag",
        ),
        "tax": (
            "tva",
            "vat",
            "tax",
            "tax amount",
            "montant taxe",
            "mwst",
            "mehrwertsteuer",
        ),
        "total": (
            "total ttc",
            "total toutes taxes comprises",
            "grand total",
            "total due",
            "amount due",
            "montant a payer",
            "net a payer",
            "gesamtbetrag",
        ),
    }
    for row in table.rows:
        if not row:
            continue
        label = _normalize_label(row[0])
        for key, candidates in aliases.items():
            if any(candidate in label for candidate in candidates):
                value = next(
                    (
                        parsed
                        for parsed in (_parse_amount(cell) for cell in reversed(row[1:]))
                        if parsed is not None
                    ),
                    None,
                )
                if value is not None:
                    amounts[key] = value
    if set(amounts) != {"subtotal", "tax", "total"}:
        return None
    expected = amounts["subtotal"] + amounts["tax"]
    delta = abs(expected - amounts["total"])
    valid = delta <= Decimal("0.02")
    return LaboratoryObservation(
        code="OCR_INVOICE_TOTAL_VALID" if valid else "OCR_INVOICE_TOTAL_MISMATCH",
        title="Total de facture",
        summary=(
            "Le sous-total et la taxe concordent avec le total."
            if valid
            else "Le sous-total et la taxe ne concordent pas avec le total."
        ),
        state="clear" if valid else "attention",
        strength="informational" if valid else "strong",
        explanation="Calcul applique : sous-total + taxe = total.",
        page=table.page,
        evidence={
            "expected": str(expected),
            "observed": str(amounts["total"]),
            "absolute_delta": str(delta),
        },
    )


def _regions(ocr_json: Any) -> tuple[_Region, ...]:
    if not isinstance(ocr_json, list):
        return ()
    regions: list[_Region] = []
    for page_index, page in enumerate(ocr_json, start=1):
        if not isinstance(page, list):
            continue
        for item in page:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            regions.append(
                _Region(
                    page=page_index,
                    label=str(item.get("label", "text")).casefold(),
                    content=content,
                )
            )
    return tuple(regions)


def _tables(regions: tuple[_Region, ...]) -> tuple[_Table, ...]:
    tables: list[_Table] = []
    for region in regions:
        if region.label != "table" and "<table" not in region.content.casefold():
            continue
        parser = _TableParser()
        try:
            parser.feed(region.content)
            parser.close()
        except ValueError:
            continue
        if parser.rows:
            tables.append(_Table(page=region.page, rows=tuple(parser.rows)))
    return tuple(tables)


def _extract_dates(regions: tuple[_Region, ...]) -> tuple[_ParsedDate, ...]:
    values: list[_ParsedDate] = []
    for region in regions:
        for pattern, date_format in _DATE_PATTERNS:
            for match in pattern.finditer(region.content):
                raw = match.group(0)
                year = int(match.group(1) if date_format == "%Y-%m-%d" else match.group(3))
                if not _supported_document_year(year):
                    continue
                normalized = raw
                if date_format == "%d/%m/%Y":
                    normalized = raw.replace(".", "/")
                try:
                    parsed = datetime.strptime(normalized.title(), date_format).date()
                except ValueError:
                    parsed = None
                values.append(_ParsedDate(page=region.page, raw=raw, value=parsed))
        for match in _NUMERIC_DATE.finditer(region.content):
            raw = match.group(0)
            first = int(match.group(1))
            second = int(match.group(3))
            year = int(match.group(4))
            if not _supported_document_year(year):
                continue
            parsed, ambiguous = _parse_localized_numeric_date(first, second, year)
            values.append(
                _ParsedDate(
                    page=region.page,
                    raw=raw,
                    value=parsed,
                    ambiguous=ambiguous,
                )
            )
    return tuple(values)


def _parse_localized_numeric_date(
    first: int,
    second: int,
    year: int,
) -> tuple[date | None, bool]:
    day_first = _safe_date(year, second, first)
    month_first = _safe_date(year, first, second)
    if day_first is not None and month_first is not None:
        return day_first, day_first != month_first
    if day_first is not None:
        return day_first, False
    if month_first is not None:
        return month_first, False
    return None, False


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _supported_document_year(year: int) -> bool:
    return 1000 <= year <= 2999


def _parse_date_text(value: str) -> date | None:
    region = _Region(page=1, label="text", content=value)
    dates = _extract_dates((region,))
    return dates[0].value if dates and dates[0].value is not None else None


def _column(headers: tuple[str, ...], *aliases: str) -> int | None:
    for index, header in enumerate(headers):
        if any(alias in header for alias in aliases):
            return index
    return None


def _normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", ascii_value).casefold().split())


def _parse_amount(value: str) -> Decimal | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = re.sub(r"[^0-9,.\-]", "", cleaned)
    if not cleaned or cleaned in {"-", ".", ","}:
        return None

    comma = cleaned.rfind(",")
    dot = cleaned.rfind(".")
    if comma >= 0 and dot >= 0:
        decimal_separator = "," if comma > dot else "."
        thousands_separator = "." if decimal_separator == "," else ","
        cleaned = cleaned.replace(thousands_separator, "")
        cleaned = cleaned.replace(decimal_separator, ".")
    elif cleaned.count(",") > 1:
        groups = cleaned.split(",")
        if len(groups[-1]) == 2:
            cleaned = "".join(groups[:-1]) + "." + groups[-1]
        elif all(len(group) == 3 for group in groups[1:]):
            cleaned = "".join(groups)
        else:
            return None
    elif cleaned.count(".") > 1:
        groups = cleaned.split(".")
        if len(groups[-1]) == 2:
            cleaned = "".join(groups[:-1]) + "." + groups[-1]
        elif all(len(group) == 3 for group in groups[1:]):
            cleaned = "".join(groups)
        else:
            return None
    elif "," in cleaned:
        head, tail = cleaned.split(",", maxsplit=1)
        cleaned = head + ("." if len(tail) <= 2 else "") + tail
    elif "." in cleaned:
        head, tail = cleaned.split(".", maxsplit=1)
        cleaned = head + ("." if len(tail) <= 2 else "") + tail

    try:
        parsed = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -parsed if negative and parsed > 0 else parsed


def _account_currency(text: str) -> str | None:
    match = re.search(r"\bCURRENCY\b\s*[:.]?\s*([A-Z]{3})", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _explicit_currency(value: str) -> str | None:
    if "€" in value:
        return "EUR"
    if "£" in value:
        return "GBP"
    if "$" in value:
        return "USD"
    match = _CURRENCY.search(value)
    return match.group(1).upper() if match else None
