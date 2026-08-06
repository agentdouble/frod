"""International, syntax-only validation of financial identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from stdnum import bic, iban, luhn
from stdnum.exceptions import (
    InvalidChecksum,
    InvalidComponent,
    InvalidFormat,
    InvalidLength,
    ValidationError,
)


@dataclass(frozen=True, slots=True)
class IdentifierValidation:
    """Explainable result that never claims an identifier is allocated."""

    normalized: str
    valid: bool
    reasons: tuple[str, ...] = ()
    country_code: str | None = None


def validate_card_number(value: str) -> IdentifierValidation:
    """Validate a payment-card number with the international Luhn checksum."""

    normalized = re.sub(r"[\s-]", "", value)
    reasons: list[str] = []
    if not 13 <= len(normalized) <= 19:
        reasons.append("invalid_length")
    if not normalized.isdigit():
        reasons.append("invalid_format")
    if not reasons and not luhn.is_valid(normalized):
        reasons.append("invalid_checksum")
    return IdentifierValidation(
        normalized=normalized,
        valid=not reasons,
        reasons=tuple(reasons),
    )


def validate_bic(value: str) -> IdentifierValidation:
    """Validate the ISO 9362 structure and country code of a BIC."""

    normalized = bic.compact(value)
    country_code = normalized[4:6] if len(normalized) >= 6 else None
    reasons: list[str] = []

    if len(normalized) not in {8, 11}:
        reasons.append("invalid_length")
    if len(normalized) >= 4 and not re.fullmatch(r"[A-Z]{4}", normalized[:4]):
        reasons.append("invalid_institution_code")
    if country_code is None or not re.fullmatch(r"[A-Z]{2}", country_code):
        reasons.append("invalid_country_code")
    if len(normalized) >= 8 and not re.fullmatch(r"[A-Z0-9]{2}", normalized[6:8]):
        reasons.append("invalid_location_code")
    if len(normalized) == 11 and not re.fullmatch(r"[A-Z0-9]{3}", normalized[8:]):
        reasons.append("invalid_branch_code")

    if not reasons:
        try:
            bic.validate(normalized)
        except InvalidComponent:
            reasons.append("unknown_country_code")
        except InvalidLength:
            reasons.append("invalid_length")
        except InvalidFormat:
            reasons.append("invalid_format")
        except ValidationError:
            reasons.append("invalid_format")

    return IdentifierValidation(
        normalized=normalized,
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        country_code=country_code,
    )


def validate_iban(value: str) -> IdentifierValidation:
    """Validate an IBAN checksum and its country-specific BBAN structure."""

    normalized = iban.compact(value)
    country_code = normalized[:2] if len(normalized) >= 2 else None
    reasons: list[str] = []

    if not 15 <= len(normalized) <= 34:
        reasons.append("invalid_length")
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", normalized):
        reasons.append("invalid_format")

    if not reasons:
        try:
            iban.validate(normalized, check_country=True)
        except InvalidChecksum:
            reasons.append("invalid_checksum")
        except InvalidComponent:
            reasons.append("unknown_country_or_structure")
        except InvalidLength:
            reasons.append("invalid_length")
        except InvalidFormat:
            reasons.append("invalid_country_structure")
        except ValidationError:
            reasons.append("invalid_format")

    return IdentifierValidation(
        normalized=normalized,
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        country_code=country_code,
    )


def expected_iban_length(country_code: str) -> int | None:
    """Return the registry length for one country supported by python-stdnum."""

    normalized = country_code.strip().upper()
    if not re.fullmatch(r"[A-Z]{2}", normalized):
        return None
    try:
        entries = iban._ibandb.info(normalized)  # noqa: SLF001 - stdnum's bundled registry
    except Exception:
        return None
    for prefix, properties in entries:
        if prefix != normalized:
            continue
        structure = properties.get("bban", "")
        component_lengths = [int(value) for value in re.findall(r"(\d+)![anc]", structure)]
        if component_lengths:
            return 4 + sum(component_lengths)
    return None
