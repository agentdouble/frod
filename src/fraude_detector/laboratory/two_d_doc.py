"""Local detection and cryptographic verification of French 2D-Doc barcodes."""

from __future__ import annotations

import zxingcpp
from py2ddoc import (
    CertificateNotFoundError,
    ParseError,
    SignatureError,
    TwoDDoc,
)

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.models import LaboratoryCheck, LaboratoryObservation
from fraude_detector.rendering import render_page


def analyze_two_d_doc(context: AnalysisContext) -> LaboratoryCheck:
    """Decode page barcodes and verify recognised 2D-Doc payloads offline."""

    observations: list[LaboratoryObservation] = []
    seen_payloads: set[bytes] = set()
    decoded_barcodes = 0
    dpi = max(180, min(216, context.config.render_dpi))

    for page_index in range(context.analyzed_page_count):
        page = context.pdfium_document[page_index]
        try:
            image = render_page(
                page,
                dpi,
                context.config.max_render_pixels,
            )
        finally:
            page.close()

        barcodes = zxingcpp.read_barcodes(
            image,
            try_rotate=True,
            try_downscale=True,
            text_mode=zxingcpp.TextMode.Plain,
        )
        decoded_barcodes += len(barcodes)
        for barcode in barcodes:
            payload = bytes(barcode.bytes)
            if payload in seen_payloads or not payload.startswith(b"DC"):
                continue
            seen_payloads.add(payload)
            observations.append(_verify_payload(payload, page_index + 1))

    if not observations:
        return LaboratoryCheck(
            code="two_d_doc",
            title="2D-Doc",
            purpose=(
                "Lire les codes DataMatrix ou QR et verifier la signature des donnees "
                "2D-Doc avec la liste de confiance locale."
            ),
            state="not_applicable",
            summary="Aucun 2D-Doc lisible.",
            limitations=(
                f"Codes-barres decodes sur les pages analysees : {decoded_barcodes}.",
                "Un code trop petit, flou, coupe ou fortement compresse peut ne pas etre lu.",
            ),
        )

    state = (
        "attention"
        if any(item.state == "attention" for item in observations)
        else (
            "indeterminate"
            if any(item.state == "indeterminate" for item in observations)
            else "clear"
        )
    )
    return LaboratoryCheck(
        code="two_d_doc",
        title="2D-Doc",
        purpose=(
            "Lire les codes DataMatrix ou QR et verifier la signature des donnees "
            "2D-Doc avec la liste de confiance locale."
        ),
        state=state,
        summary=_summary(observations),
        observations=tuple(observations),
        limitations=(
            "La verification porte sur les champs signes du code.",
            "Sans OCR, les champs signes ne sont pas compares au texte imprime autour du code.",
            "La liste de confiance est celle embarquee avec py2ddoc et doit etre mise a jour "
            "avec la dependance.",
        ),
    )


def _verify_payload(payload: bytes, page: int) -> LaboratoryObservation:
    try:
        document = TwoDDoc.from_string(payload)
    except CertificateNotFoundError as error:
        try:
            document = TwoDDoc.from_string(payload, verify=False)
            evidence = _document_evidence(document)
        except Exception:
            evidence = {}
        evidence["verification_error"] = str(error)[:300]
        return LaboratoryObservation(
            code="TWO_D_DOC_CERTIFICATE_UNKNOWN",
            title="2D-Doc lisible, certificat absent",
            summary="Le code est structure, mais son certificat n'est pas dans la liste locale.",
            state="indeterminate",
            strength="moderate",
            explanation=(
                "Le document n'est pas declare invalide : la verification est impossible "
                "avec la liste de confiance actuellement embarquee."
            ),
            page=page,
            evidence=evidence,
        )
    except SignatureError as error:
        return LaboratoryObservation(
            code="TWO_D_DOC_SIGNATURE_INVALID",
            title="Signature 2D-Doc invalide",
            summary="Les donnees du code ne correspondent pas a sa signature.",
            state="attention",
            strength="strong",
            explanation=(
                "La signature cryptographique des champs encodes echoue. C'est un indice "
                "fort de modification du code ou de contenu non authentique."
            ),
            page=page,
            evidence={"verification_error": str(error)[:300]},
        )
    except ParseError as error:
        return LaboratoryObservation(
            code="TWO_D_DOC_MALFORMED",
            title="Code annoncant 2D-Doc mais mal forme",
            summary="Le contenu commence par DC mais ne respecte pas le format attendu.",
            state="attention",
            strength="moderate",
            explanation=(
                "Le marqueur 2D-Doc est present, mais les champs ou la signature ne peuvent "
                "pas etre parses. Une lecture imparfaite reste possible."
            ),
            page=page,
            evidence={"parser_error": str(error)[:300]},
        )

    return LaboratoryObservation(
        code="TWO_D_DOC_SIGNATURE_VALID",
        title="Signature 2D-Doc valide",
        summary=f"{document.doc_type_label} : champs signes authentifies.",
        state="clear",
        strength="strong",
        explanation=(
            "La signature du code correspond aux champs encodes et au certificat de la "
            "liste de confiance locale."
        ),
        page=page,
        evidence=_document_evidence(document),
    )


def _document_evidence(document: TwoDDoc) -> dict[str, object]:
    return {
        "version": document.version,
        "certification_authority": document.ca_id,
        "certificate_id": document.cert_id,
        "document_type": document.doc_type,
        "document_type_label": document.doc_type_label,
        "document_date": document.doc_date.isoformat() if document.doc_date else None,
        "signature_date": document.sign_date.isoformat() if document.sign_date else None,
        "signed_field_count": len(document.fields),
        "cryptographically_authentic": document.is_authentic,
    }


def _summary(observations: list[LaboratoryObservation]) -> str:
    invalid = sum(item.state == "attention" for item in observations)
    unknown = sum(item.state == "indeterminate" for item in observations)
    if invalid:
        return f"{invalid} code(s) 2D-Doc en anomalie."
    if unknown:
        return f"{unknown} code(s) lisible(s), verification incomplete."
    return f"{len(observations)} code(s) 2D-Doc valide(s)."
