"""Cryptographic PDF signature and post-signature update experiments."""

from __future__ import annotations

import io
from typing import Any

from pyhanko.pdf_utils.reader import PdfFileReader
from pyhanko.sign.validation import validate_pdf_signature
from pyhanko_certvalidator import ValidationContext

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.models import (
    LaboratoryCheck,
    LaboratoryObservation,
    LaboratoryState,
)


def analyze_pdf_signatures(
    context: AnalysisContext,
) -> tuple[LaboratoryCheck, LaboratoryCheck]:
    """Validate embedded PDF signatures without network certificate fetching."""

    reader = PdfFileReader(io.BytesIO(context.raw_pdf))
    signatures = tuple(reader.embedded_signatures)
    if not signatures:
        return (_no_signature_check(), _no_post_signature_check())

    pades_observations: list[LaboratoryObservation] = []
    modification_observations: list[LaboratoryObservation] = []
    validation_context = ValidationContext(
        allow_fetching=False,
        revocation_mode="soft-fail",
    )

    for index, embedded_signature in enumerate(signatures, start=1):
        status = validate_pdf_signature(
            embedded_signature,
            signer_validation_context=validation_context,
        )
        field_name = embedded_signature.field_name or f"Signature {index}"
        signer = _signer_name(status)
        common_evidence = {
            "field_name": field_name,
            "signer": signer,
            "digest_algorithm": status.md_algorithm,
            "signature_mechanism": status.pkcs7_signature_mechanism,
            "cryptographically_intact": status.intact,
            "cryptographic_signature_valid": status.valid,
            "certificate_trusted_locally": status.trusted,
            "certificate_revoked": status.revoked,
        }

        if not status.intact or not status.valid:
            pades_observations.append(
                LaboratoryObservation(
                    code="PDF_SIGNATURE_INVALID",
                    title=f"{field_name} : signature invalide",
                    summary="L'integrite cryptographique ne peut pas etre confirmee.",
                    state="attention",
                    strength="strong",
                    explanation=(
                        "Le condensat du document ou la signature CMS ne correspond pas. "
                        "C'est un indice technique fort d'alteration ou de signature corrompue."
                    ),
                    evidence=common_evidence,
                )
            )
        elif status.trusted:
            pades_observations.append(
                LaboratoryObservation(
                    code="PDF_SIGNATURE_TRUSTED",
                    title=f"{field_name} : signature valide et reconnue",
                    summary=f"Signature intacte attribuee a {signer}.",
                    state="clear",
                    strength="strong",
                    explanation=(
                        "La signature est cryptographiquement intacte et sa chaine rejoint "
                        "une autorite reconnue par le magasin de confiance local."
                    ),
                    evidence=common_evidence,
                )
            )
        else:
            pades_observations.append(
                LaboratoryObservation(
                    code="PDF_SIGNATURE_INTACT_UNTRUSTED",
                    title=f"{field_name} : signature intacte, identite non reconnue",
                    summary=(
                        "Les octets signes sont intacts, mais le certificat n'est pas "
                        "approuve localement."
                    ),
                    state="indeterminate",
                    strength="moderate",
                    explanation=(
                        "L'integrite cryptographique est confirmee. En revanche, sans chaine "
                        "de confiance locale ou EUTL chargee, l'identite du signataire reste "
                        "indeterminee; ce n'est pas en soi un indice de fraude."
                    ),
                    evidence=common_evidence,
                )
            )

        modification_observations.append(
            _post_signature_observation(
                index=index,
                field_name=field_name,
                status=status,
            )
        )

    return (
        LaboratoryCheck(
            code="pades",
            title="Signature PDF / PAdES",
            purpose=(
                "Verifier le condensat signe, la signature CMS et la chaine de certificat "
                "sans contacter de service distant."
            ),
            state=_aggregate_state(pades_observations),
            summary=_signature_summary(pades_observations),
            observations=tuple(pades_observations),
            limitations=(
                "Le controle valide les signatures PDF compatibles, mais ne certifie pas "
                "a lui seul la conformite complete a un profil PAdES.",
                "Aucune liste de confiance europeenne n'est telechargee pendant l'analyse.",
            ),
        ),
        LaboratoryCheck(
            code="post_signature",
            title="Modifications apres signature",
            purpose=(
                "Determiner si des revisions ont ete ajoutees apres la signature et si "
                "elles respectent la politique de modification declaree."
            ),
            state=_aggregate_state(modification_observations),
            summary=_modification_summary(modification_observations),
            observations=tuple(modification_observations),
            limitations=(
                "L'analyse de differences de pyHanko est experimentale et applique une "
                "politique prudente aux mises a jour incrementales.",
            ),
        ),
    )


def _post_signature_observation(
    *,
    index: int,
    field_name: str,
    status: Any,
) -> LaboratoryObservation:
    coverage = _enum_label(status.coverage)
    modification = _enum_label(status.modification_level)
    evidence = {
        "field_name": field_name,
        "coverage": coverage,
        "modification_level": modification,
        "document_policy_respected": status.docmdp_ok,
        "difference_result": type(status.diff_result).__name__
        if status.diff_result is not None
        else None,
    }

    suspicious = (
        status.docmdp_ok is False
        or modification in {"OTHER", "Other"}
        or (status.diff_result is not None and "Suspicious" in type(status.diff_result).__name__)
    )
    if suspicious:
        return LaboratoryObservation(
            code="PDF_SUSPICIOUS_POST_SIGNATURE_CHANGE",
            title=f"{field_name} : changement posterieur suspect",
            summary="Une modification non autorisee ou non classee suit la signature.",
            state="attention",
            strength="strong",
            explanation=(
                "La signature peut rester cryptographiquement intacte pour sa revision, "
                "mais des objets ajoutes ensuite ne sont pas couverts ou enfreignent la "
                "politique de modification du document."
            ),
            evidence=evidence,
        )

    no_change = modification in {"NONE", "None", ""} and "ENTIRE_FILE" in coverage
    if no_change:
        return LaboratoryObservation(
            code="PDF_NO_POST_SIGNATURE_CHANGE",
            title=f"{field_name} : aucune modification posterieure",
            summary="La signature couvre le fichier complet.",
            state="clear",
            strength="strong",
            explanation=("Aucune revision supplementaire n'a ete detectee apres cette signature."),
            evidence=evidence,
        )

    return LaboratoryObservation(
        code="PDF_ALLOWED_POST_SIGNATURE_CHANGE",
        title=f"{field_name} : mise a jour posterieure classee",
        summary=f"Modification classee {modification or 'indeterminee'}.",
        state="detected",
        strength="moderate",
        explanation=(
            "Une mise a jour suit la signature, mais le validateur ne la classe pas comme "
            "une alteration interdite. Un remplissage de formulaire ou une autre signature "
            "peuvent etre legitimes."
        ),
        evidence=evidence,
    )


def _no_signature_check() -> LaboratoryCheck:
    return LaboratoryCheck(
        code="pades",
        title="Signature PDF / PAdES",
        purpose="Verifier cryptographiquement les signatures numeriques du PDF.",
        state="not_applicable",
        summary="Aucune signature numerique exploitable.",
        limitations=("L'absence de signature ne constitue pas un indice de fraude.",),
    )


def _no_post_signature_check() -> LaboratoryCheck:
    return LaboratoryCheck(
        code="post_signature",
        title="Modifications apres signature",
        purpose="Rechercher les revisions ajoutees apres une signature numerique.",
        state="not_applicable",
        summary="Aucune signature servant de point de comparaison.",
    )


def _aggregate_state(observations: list[LaboratoryObservation]) -> LaboratoryState:
    priority: tuple[LaboratoryState, ...] = (
        "attention",
        "error",
        "indeterminate",
        "detected",
        "clear",
        "not_applicable",
    )
    return next(state for state in priority if any(item.state == state for item in observations))


def _signature_summary(observations: list[LaboratoryObservation]) -> str:
    invalid = sum(item.code == "PDF_SIGNATURE_INVALID" for item in observations)
    trusted = sum(item.code == "PDF_SIGNATURE_TRUSTED" for item in observations)
    intact = len(observations) - invalid
    if invalid:
        return f"{invalid} signature(s) invalide(s) sur {len(observations)}."
    if trusted == len(observations):
        return f"{trusted} signature(s) valide(s) et reconnue(s)."
    return f"{intact} signature(s) intacte(s), confiance d'identite incomplete."


def _modification_summary(observations: list[LaboratoryObservation]) -> str:
    suspicious = sum(item.code == "PDF_SUSPICIOUS_POST_SIGNATURE_CHANGE" for item in observations)
    if suspicious:
        return f"{suspicious} signature(s) suivie(s) d'un changement suspect."
    unchanged = sum(item.code == "PDF_NO_POST_SIGNATURE_CHANGE" for item in observations)
    if unchanged == len(observations):
        return "Aucune modification posterieure aux signatures."
    return "Des mises a jour posterieures existent, sans violation forte detectee."


def _signer_name(status: Any) -> str:
    try:
        return str(status.signing_cert.subject.human_friendly)
    except Exception:
        return "signataire non renseigne"


def _enum_label(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "name", value))
