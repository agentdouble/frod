"""Conservative classification of software declared by PDF metadata."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from fraude_detector.config import AnalysisConfig

SoftwareCategory = Literal[
    "generative_tool",
    "visual_editor",
    "design_tool",
    "pdf_editor",
    "online_pdf_service",
    "scanner",
    "signature_service",
    "document_generator",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class PdfSoftwareClassification:
    """Interpretation of one Creator or Producer value."""

    category: SoftwareCategory
    label: str
    points: float
    recognized: bool


@dataclass(frozen=True, slots=True)
class _SoftwareRule:
    category: SoftwareCategory
    label: str
    aliases: tuple[str, ...]


# Risk-bearing rules come first. Aliases are deliberately product-oriented: generic words such as
# "editor", "online" or "converter" would create unacceptable false positives.
_SOFTWARE_RULES: tuple[_SoftwareRule, ...] = (
    _SoftwareRule(
        "generative_tool",
        "outil de génération par IA",
        (
            "adobe firefly",
            "automatic1111",
            "chatgpt",
            "comfyui",
            "dall e",
            "deepai",
            "dreamstudio",
            "firefly",
            "flux ai",
            "fooocus",
            "google imagen",
            "ideogram",
            "invokeai",
            "leonardo ai",
            "midjourney",
            "nightcafe",
            "openai image",
            "playground ai",
            "stable diffusion",
            "stability ai",
        ),
    ),
    _SoftwareRule(
        "visual_editor",
        "éditeur d'images",
        (
            "adobe photoshop",
            "affinity photo",
            "capture one",
            "corel paintshop",
            "darktable",
            "digikam",
            "gimp",
            "krita",
            "lightroom",
            "luminar",
            "paint net",
            "photofiltre",
            "photopea",
            "photoscape",
            "pixelmator",
            "rawtherapee",
        ),
    ),
    _SoftwareRule(
        "design_tool",
        "outil de création graphique",
        (
            "adobe illustrator",
            "adobe indesign",
            "affinity designer",
            "affinity publisher",
            "canva",
            "coreldraw",
            "figma",
            "inkscape",
            "microsoft publisher",
            "quarkxpress",
            "scribus",
            "sketch app",
            "vista create",
            "vistacreate",
        ),
    ),
    _SoftwareRule(
        "pdf_editor",
        "éditeur de PDF",
        (
            "abbyy finereader pdf",
            "bluebeam revu",
            "corel pdf fusion",
            "drawboard pdf",
            "foxit phantompdf",
            "foxit pdf editor",
            "kofax power pdf",
            "libreoffice draw",
            "master pdf editor",
            "nitro pdf pro",
            "nitro pro",
            "nuance power pdf",
            "pdf architect",
            "pdf expert",
            "pdfill pdf tools",
            "pdfsam",
            "pdf xchange editor",
            "pdfelement",
            "qoppa pdf studio",
            "updf",
            "wondershare pdfelement",
        ),
    ),
    _SoftwareRule(
        "online_pdf_service",
        "service en ligne de transformation PDF",
        (
            "avepdf",
            "cleverpdf",
            "cloudconvert",
            "compress2go",
            "convertio",
            "deftpdf",
            "dochub",
            "docupub",
            "freepdfconvert",
            "hipdf",
            "i love pdf",
            "ilovepdf",
            "lightpdf",
            "neevia document converter",
            "neevia pdfmerge",
            "neevia technology",
            "online2pdf",
            "online convert",
            "pdf io",
            "pdf24",
            "pdfaid",
            "pdf candy",
            "pdf filler",
            "pdf2go",
            "pdfescape",
            "pdffiller",
            "pdfresizer",
            "pdfsimpli",
            "pdfpro",
            "sejda",
            "smallpdf",
            "soda pdf",
            "tinywow",
            "xodo",
            "zamzar",
        ),
    ),
    _SoftwareRule(
        "scanner",
        "logiciel ou équipement de numérisation",
        (
            "adobe scan",
            "brother",
            "camscanner",
            "canon",
            "controlcenter",
            "document capture pro",
            "doxie",
            "epson scan",
            "fi series",
            "fujitsu",
            "genius scan",
            "hp scan",
            "image capture",
            "iriscanner",
            "konica minolta",
            "kyocera",
            "microsoft lens",
            "naps2",
            "notes scanner",
            "office lens",
            "pfi scan",
            "ricoh",
            "scansnap",
            "xerox",
        ),
    ),
    _SoftwareRule(
        "signature_service",
        "service de signature électronique",
        (
            "adobe sign",
            "certigna",
            "docusign",
            "dropbox sign",
            "eversign",
            "hellosign",
            "signaturit",
            "signnow",
            "universign",
            "yousign",
        ),
    ),
    _SoftwareRule(
        "document_generator",
        "générateur documentaire courant",
        (
            "adobe acrobat",
            "adobe pdf library",
            "acrobat distiller",
            "antenna house",
            "apache fop",
            "apache pdfbox",
            "aspose pdf",
            "apple keynote",
            "apple numbers",
            "apple pages",
            "borb",
            "cairo",
            "chromium",
            "crystal reports",
            "cups",
            "devexpress",
            "dompdf",
            "fpdf",
            "ghostscript",
            "google docs",
            "google sheets",
            "google slides",
            "ironpdf",
            "itext",
            "jasperreports",
            "latex",
            "libreoffice",
            "microsoft excel",
            "microsoft powerpoint",
            "microsoft print to pdf",
            "microsoft report",
            "microsoft word",
            "microsoft xps",
            "macos preview",
            "mpdf",
            "mupdf",
            "openoffice",
            "oracle bi publisher",
            "pagespeed",
            "pdfcreator",
            "pdfkit",
            "pdfium",
            "pdfmake",
            "pdftex",
            "princexml",
            "prawn",
            "pypdf",
            "pypdf2",
            "qpdf",
            "questpdf",
            "quartz pdfcontext",
            "reportlab",
            "sap crystal",
            "skia pdf",
            "spire pdf",
            "syncfusion",
            "tcpdf",
            "telerik",
            "wkhtmltopdf",
            "weasyprint",
            "xetex",
        ),
    ),
)


def classify_pdf_software(
    value: str,
    config: AnalysisConfig,
) -> PdfSoftwareClassification:
    """Classify a declarative metadata value without treating unknown software as risky."""

    normalized = _normalized_software_name(value)
    for rule in _SOFTWARE_RULES:
        if any(_contains_alias(normalized, alias) for alias in rule.aliases):
            return PdfSoftwareClassification(
                category=rule.category,
                label=rule.label,
                points=_points_for_category(rule.category, config),
                recognized=True,
            )
    return PdfSoftwareClassification(
        category="unknown",
        label="logiciel non répertorié",
        points=0.0,
        recognized=False,
    )


def _points_for_category(category: SoftwareCategory, config: AnalysisConfig) -> float:
    return {
        "online_pdf_service": config.pdf_metadata_online_service_points,
        "pdf_editor": config.pdf_metadata_pdf_editor_points,
        "design_tool": config.pdf_metadata_design_tool_points,
        "visual_editor": config.pdf_metadata_visual_editor_points,
        "generative_tool": config.pdf_metadata_generative_tool_points,
    }.get(category, 0.0)


def _normalized_software_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def _contains_alias(normalized: str, alias: str) -> bool:
    normalized_alias = _normalized_software_name(alias)
    return bool(re.search(rf"(?:^|\s){re.escape(normalized_alias)}(?:\s|$)", normalized))
