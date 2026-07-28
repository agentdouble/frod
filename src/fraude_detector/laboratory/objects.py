"""Conservative inspection of PDF fonts and non-visible page objects."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import pypdfium2.raw as pdfium_c

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.geometry import bbox_coverage, pdfium_bounds_to_top_left
from fraude_detector.models import LaboratoryCheck, LaboratoryObservation


@dataclass(frozen=True, slots=True)
class TextObject:
    page: int
    text: str
    font: str
    font_size: float
    invisible: bool
    outside_page: bool


def analyze_fonts_and_hidden_objects(context: AnalysisContext) -> LaboratoryCheck:
    """Report hidden text and narrowly calibrated rare-font substitutions."""

    text_objects: list[TextObject] = []
    scanned_pages: set[int] = set()
    embedded_fonts: dict[str, bool] = {}

    for page_index in range(context.analyzed_page_count):
        page = context.pdfium_document[page_index]
        try:
            page_width, page_height = page.get_size()
            text_page = page.get_textpage()
            try:
                objects = list(page.get_objects(textpage=text_page))
                if _has_full_page_image(objects, page_width, page_height):
                    scanned_pages.add(page_index + 1)
                for obj in objects:
                    if obj.type != pdfium_c.FPDF_PAGEOBJ_TEXT:
                        continue
                    described = _describe_text_object(
                        obj,
                        page=page_index + 1,
                        page_width=page_width,
                        page_height=page_height,
                    )
                    if described is None:
                        continue
                    text_objects.append(described)
                    try:
                        font = obj.get_font()
                        embedded_fonts.setdefault(
                            described.font,
                            bool(font.is_embedded()),
                        )
                    except Exception:
                        pass
            finally:
                text_page.close()
        finally:
            page.close()

    observations: list[LaboratoryObservation] = []
    observations.extend(_hidden_text_observations(text_objects, scanned_pages))
    observations.extend(_outside_page_observations(text_objects))
    observations.extend(_rare_font_observations(text_objects))

    font_counts = Counter(item.font for item in text_objects if item.text.strip())
    unembedded = sorted(font for font, is_embedded in embedded_fonts.items() if not is_embedded)
    inventory = {
        "text_object_count": len(text_objects),
        "font_count": len(font_counts),
        "fonts_by_object_count": dict(font_counts.most_common()),
        "unembedded_fonts": unembedded,
        "scan_pages_with_possible_ocr_layer": sorted(scanned_pages),
    }
    observations.append(
        LaboratoryObservation(
            code="PDF_FONT_INVENTORY",
            title="Inventaire typographique",
            summary=f"{len(font_counts)} police(s) pour {len(text_objects)} objet(s) texte.",
            state="clear",
            strength="informational",
            explanation=(
                "Cet inventaire sert a contextualiser les ruptures. Une grande diversite "
                "ou une police non incorporee n'est pas anormale a elle seule."
            ),
            evidence=inventory,
        )
    )

    suspicious = [item for item in observations if item.state == "attention"]
    detected = [item for item in observations if item.state == "detected"]
    state = "attention" if suspicious else ("detected" if detected else "clear")
    if suspicious:
        summary = f"{len(suspicious)} anomalie(s) ciblee(s) a examiner."
    elif detected:
        summary = "Objets invisibles expliques par la structure du document."
    else:
        summary = "Aucune rupture ciblee detectee."

    return LaboratoryCheck(
        code="fonts_hidden_objects",
        title="Polices et objets masques",
        purpose=(
            "Rechercher du texte invisible, hors page et des substitutions de police "
            "courtes a dominante numerique."
        ),
        state=state,
        summary=summary,
        observations=tuple(observations),
        limitations=(
            "Un calque OCR invisible sur une page scannee est classe comme information.",
            "La police seule ne permet pas d'affirmer qu'un montant ou un numero a ete remplace.",
            "Les masques complexes, groupes de transparence et objets aplatis dans une image "
            "ne sont pas reconstruits semantiquement.",
        ),
    )


def _describe_text_object(
    obj: Any,
    *,
    page: int,
    page_width: float,
    page_height: float,
) -> TextObject | None:
    try:
        text = obj.extract()
        font = obj.get_font()
        font_name = _normalise_font_name(font.get_base_name())
        font_size = float(obj.get_font_size())
        render_mode = pdfium_c.FPDFTextObj_GetTextRenderMode(obj)
        invisible = render_mode == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE
        bbox = pdfium_bounds_to_top_left(obj.get_bounds(), page_height)
        outside_page = bool(
            bbox and (bbox.x1 < 0 or bbox.y1 < 0 or bbox.x0 > page_width or bbox.y0 > page_height)
        )
    except Exception:
        return None
    return TextObject(
        page=page,
        text=text,
        font=font_name,
        font_size=font_size,
        invisible=invisible,
        outside_page=outside_page,
    )


def _has_full_page_image(
    objects: list[Any],
    page_width: float,
    page_height: float,
) -> bool:
    for obj in objects:
        if obj.type != pdfium_c.FPDF_PAGEOBJ_IMAGE:
            continue
        try:
            bbox = pdfium_bounds_to_top_left(obj.get_bounds(), page_height)
            if bbox and bbox_coverage(bbox, page_width, page_height) >= 0.72:
                return True
        except Exception:
            continue
    return False


def _hidden_text_observations(
    objects: list[TextObject],
    scanned_pages: set[int],
) -> list[LaboratoryObservation]:
    by_page: dict[int, list[TextObject]] = defaultdict(list)
    for item in objects:
        if item.invisible and item.text.strip():
            by_page[item.page].append(item)

    observations: list[LaboratoryObservation] = []
    for page, items in sorted(by_page.items()):
        character_count = sum(len(item.text.strip()) for item in items)
        is_ocr_layer = page in scanned_pages and character_count >= 20
        observations.append(
            LaboratoryObservation(
                code=(
                    "PDF_INVISIBLE_OCR_LAYER" if is_ocr_layer else "PDF_UNEXPLAINED_INVISIBLE_TEXT"
                ),
                title=(
                    "Calque OCR invisible" if is_ocr_layer else "Texte invisible sans fond de scan"
                ),
                summary=(f"Page {page} : {len(items)} objet(s), {character_count} caracteres."),
                state="detected" if is_ocr_layer else "attention",
                strength="informational" if is_ocr_layer else "weak",
                explanation=(
                    "Le texte invisible accompagne une grande image de scan et sert "
                    "probablement a la recherche et a l'accessibilite."
                    if is_ocr_layer
                    else "Du texte non rendu existe sans grande image de scan pour "
                    "l'expliquer. Cela peut rester legitime, notamment pour l'accessibilite."
                ),
                page=page,
                evidence={
                    "object_count": len(items),
                    "character_count": character_count,
                    "full_page_image_present": page in scanned_pages,
                },
            )
        )
    return observations


def _outside_page_observations(
    objects: list[TextObject],
) -> list[LaboratoryObservation]:
    by_page = Counter(item.page for item in objects if item.outside_page and item.text.strip())
    return [
        LaboratoryObservation(
            code="PDF_TEXT_OUTSIDE_PAGE",
            title="Texte place hors de la page",
            summary=f"Page {page} : {count} objet(s) texte hors du cadre visible.",
            state="attention",
            strength="weak",
            explanation=(
                "Ces objets ne sont pas visibles dans le cadrage normal. Ils peuvent venir "
                "d'un gabarit ou d'un export imparfait et ne sont donc qu'un signal faible."
            ),
            page=page,
            evidence={"object_count": count},
        )
        for page, count in sorted(by_page.items())
    ]


def _rare_font_observations(
    objects: list[TextObject],
) -> list[LaboratoryObservation]:
    by_page: dict[int, list[TextObject]] = defaultdict(list)
    for item in objects:
        if item.text.strip() and not item.invisible and not item.outside_page:
            by_page[item.page].append(item)

    observations: list[LaboratoryObservation] = []
    for page, items in sorted(by_page.items()):
        if len(items) < 10:
            continue
        counts = Counter(item.font for item in items)
        dominant_font, dominant_count = counts.most_common(1)[0]
        if dominant_count < 5:
            continue
        candidates = [
            item
            for item in items
            if item.font != dominant_font
            and counts[item.font] <= 2
            and 1 <= len(item.text.strip()) <= 24
            and _numeric_ratio(item.text) >= 0.55
            and item.font_size >= 4
        ]
        if not candidates:
            continue
        samples = [
            {
                "text": _redact_sample(item.text),
                "font": item.font,
                "font_size": round(item.font_size, 2),
            }
            for item in candidates[:5]
        ]
        observations.append(
            LaboratoryObservation(
                code="PDF_RARE_FONT_NUMERIC_FRAGMENT",
                title="Fragment numerique dans une police rare",
                summary=f"Page {page} : {len(candidates)} fragment(s) cible(s).",
                state="attention",
                strength="weak",
                explanation=(
                    "Une police employee une ou deux fois porte un fragment surtout "
                    "numerique, alors qu'une autre police domine la page. Ce motif peut "
                    "correspondre a un remplacement local, mais aussi a une mise en forme "
                    "normale; il doit etre confirme visuellement."
                ),
                page=page,
                evidence={
                    "dominant_font": dominant_font,
                    "dominant_object_count": dominant_count,
                    "rare_fragments": samples,
                },
            )
        )
    return observations


def _normalise_font_name(name: str) -> str:
    if "+" in name and len(name.split("+", 1)[0]) == 6:
        return name.split("+", 1)[1]
    return name or "Police inconnue"


def _numeric_ratio(text: str) -> float:
    compact = "".join(character for character in text if not character.isspace())
    if not compact:
        return 0.0
    numeric = sum(character.isdigit() or character in ".,:/-%€$" for character in compact)
    return numeric / len(compact)


def _redact_sample(text: str) -> str:
    compact = " ".join(text.split())
    return compact[:24]
