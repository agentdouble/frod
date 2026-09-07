"""Factur-X embedded XML detection and local XSD validation."""

from __future__ import annotations

from lxml import etree

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.models import LaboratoryCheck, LaboratoryObservation

FACTURX_FILENAMES = {
    "factur-x.xml",
    "factur_x.xml",
    "zugferd-invoice.xml",
    "xrechnung.xml",
}


def analyze_facturx(context: AnalysisContext) -> LaboratoryCheck:
    """Inspect XML attachments and validate recognised invoice payloads."""

    from facturx import get_flavor, get_level, xml_check_xsd

    observations: list[LaboratoryObservation] = []
    xml_attachments = 0
    for attachment in context.reader.attachment_list:
        filename = str(attachment.name or "")
        content = attachment.content
        if not filename.casefold().endswith(".xml") or not content:
            continue
        xml_attachments += 1
        intended_facturx = filename.casefold() in FACTURX_FILENAMES
        try:
            root = etree.fromstring(content)
        except Exception as error:
            if intended_facturx:
                observations.append(
                    LaboratoryObservation(
                        code="FACTURX_XML_MALFORMED",
                        title="XML Factur-X illisible",
                        summary=f"La piece jointe {filename} n'est pas un XML bien forme.",
                        state="attention",
                        strength="moderate",
                        explanation=(
                            "Le conteneur annonce une facture structuree, mais son XML ne "
                            "peut pas etre parse. Cela invalide l'interoperabilite Factur-X "
                            "sans prouver une falsification."
                        ),
                        evidence={
                            "filename": filename,
                            "parser_error": str(error)[:300],
                        },
                    )
                )
            continue

        try:
            flavor = str(get_flavor(root))
            level = str(get_level(root, flavor))
        except Exception as error:
            if intended_facturx:
                observations.append(
                    LaboratoryObservation(
                        code="FACTURX_PROFILE_UNRECOGNISED",
                        title="Profil Factur-X non reconnu",
                        summary=f"{filename} ne correspond pas a un profil pris en charge.",
                        state="attention",
                        strength="weak",
                        explanation=(
                            "Le nom de la piece jointe suggere Factur-X, mais sa structure "
                            "ne permet pas d'identifier un profil officiel."
                        ),
                        evidence={
                            "filename": filename,
                            "validation_error": str(error)[:300],
                        },
                    )
                )
            continue

        try:
            xml_check_xsd(root, flavor=flavor, level=level)
        except Exception as error:
            observations.append(
                LaboratoryObservation(
                    code="FACTURX_XSD_INVALID",
                    title="XML Factur-X non conforme",
                    summary=f"{filename} echoue au schema officiel {flavor} / {level}.",
                    state="attention",
                    strength="moderate",
                    explanation=(
                        "La structure XML ne respecte pas le schema declare. C'est une "
                        "anomalie tangible de format, mais pas une preuve que les donnees "
                        "visibles de la facture sont fausses."
                    ),
                    evidence={
                        "filename": filename,
                        "flavor": flavor,
                        "level": level,
                        "xsd_error": str(error)[:500],
                    },
                )
            )
            continue

        observations.append(
            LaboratoryObservation(
                code="FACTURX_XSD_VALID",
                title="XML Factur-X conforme",
                summary=f"{filename} respecte le schema {flavor} / {level}.",
                state="clear",
                strength="moderate",
                explanation=(
                    "Le XML embarque est bien forme et conforme au schema officiel. "
                    "Cela rassure sur la structure, pas sur l'authenticite de l'emetteur "
                    "ni sur la concordance avec le rendu PDF."
                ),
                evidence={
                    "filename": filename,
                    "flavor": flavor,
                    "level": level,
                    "xml_size_bytes": len(content),
                    "structured_fields": _facturx_fields(root),
                },
            )
        )

    if not observations:
        return LaboratoryCheck(
            code="facturx",
            title="Factur-X",
            purpose="Detecter et valider le XML de facture embarque dans un PDF hybride.",
            state="not_applicable",
            summary="Aucun XML Factur-X reconnu.",
            limitations=(
                f"Pieces jointes XML examinees : {xml_attachments}.",
                "L'absence de Factur-X n'est pas anormale pour une facture PDF classique.",
            ),
        )

    state = "attention" if any(item.state == "attention" for item in observations) else "clear"
    return LaboratoryCheck(
        code="facturx",
        title="Factur-X",
        purpose="Detecter et valider le XML de facture embarque dans un PDF hybride.",
        state=state,
        summary=(
            "Une anomalie de conformite XML a ete detectee."
            if state == "attention"
            else "Le document Factur-X respecte son schema XML."
        ),
        observations=tuple(observations),
        limitations=(
            "La validation XSD est locale; le controle Schematron distant est desactive.",
            "La comparaison avec le rendu exige plusieurs champs témoins retrouvés par l'OCR; "
            "sinon aucune divergence n'est affirmée.",
        ),
    )


def _facturx_fields(root: etree._Element) -> dict[str, str]:
    """Extract a small set of values expected in both XML and visible invoice."""

    paths = {
        "invoice_number": (
            "string((//*[local-name()='ExchangedDocument']/*[local-name()='ID'])[1])"
        ),
        "issue_date": (
            "string((//*[local-name()='ExchangedDocument']//*[local-name()='IssueDateTime']"
            "//*[local-name()='DateTimeString'])[1])"
        ),
        "seller_name": (
            "string((//*[local-name()='SellerTradeParty']/*[local-name()='Name'])[1])"
        ),
        "buyer_name": (
            "string((//*[local-name()='BuyerTradeParty']/*[local-name()='Name'])[1])"
        ),
        "grand_total": (
            "string((//*[local-name()='SpecifiedTradeSettlementHeaderMonetarySummation']"
            "/*[local-name()='GrandTotalAmount'])[1])"
        ),
        "due_amount": (
            "string((//*[local-name()='SpecifiedTradeSettlementHeaderMonetarySummation']"
            "/*[local-name()='DuePayableAmount'])[1])"
        ),
        "tax_total": "string((//*[local-name()='TaxTotalAmount'])[1])",
        "iban": "string((//*[local-name()='IBANID'])[1])",
    }
    fields: dict[str, str] = {}
    for name, xpath in paths.items():
        value = " ".join(str(root.xpath(xpath)).split())
        if value:
            fields[name] = value
    return fields
