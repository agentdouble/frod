"""Local Streamlit interface for documentary fraud analysis reports."""

from __future__ import annotations

import gc
import hashlib
import html as html_lib
import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import streamlit as st

from fraude_detector.config import AnalysisConfig
from fraude_detector.errors import AnalysisError
from fraude_detector.extraction_reconciliation import reconcile_extraction
from fraude_detector.gapl import (
    best_available_device,
    create_gapl_adapter,
)
from fraude_detector.image_pipeline import ImageAnalysisPipeline
from fraude_detector.laboratory import (
    analyze_image_laboratory,
    analyze_ocr_laboratory,
    analyze_pdf_laboratory,
)
from fraude_detector.laboratory.visual_repetition import analyze_repeated_visual_regions
from fraude_detector.llm_classifier import classify_document
from fraude_detector.llm_extractor import extract_document
from fraude_detector.llm_verifier import verify_extraction
from fraude_detector.models import (
    AnalysisReport,
    DetectorResult,
    DocumentClassification,
    DocumentExtraction,
    ExtractionVerification,
    Finding,
    ImageAnalysisReport,
    LaboratoryReport,
    OcrReport,
)
from fraude_detector.ocr_consistency import build_ocr_content_result
from fraude_detector.pipeline import AnalysisPipeline
from fraude_detector.run_config import load_run_config
from fraude_detector.scoring import FAMILY_CAPS, assess_risk

CONFIG_PATH = Path(os.environ.get("FROD_CONFIG", "config.yaml"))
PROJECT_CONFIG = load_run_config(CONFIG_PATH)
WORK_DIR = PROJECT_CONFIG.application.work_dir
UPLOAD_DIR = WORK_DIR / "uploads"
RUN_DIR = WORK_DIR / "runs"
GAPL_WEIGHTS = PROJECT_CONFIG.gapl.weights_path
TRUFOR_WEIGHTS = PROJECT_CONFIG.trufor.weights_path
TRUFOR_PIXEL_BUDGET = PROJECT_CONFIG.trufor.max_pixels
CONFIG_FINGERPRINT = hashlib.sha256(repr(PROJECT_CONFIG).encode("utf-8")).hexdigest()[:12]
ANALYSIS_POLICY_VERSION = (
    f"gapl-p25-90-v2-trufor-lab-v1-ocr-content-v3-structured-identifiers-{CONFIG_FINGERPRINT}"
)
INDICATOR_STEP_SECONDS = 0.45
OCR_IDENTIFIER_PREFIXES = (
    "OCR_CARD_",
    "OCR_IBAN_",
    "OCR_BIC_",
    "OCR_CKYC_",
    "OCR_MICR_",
    "OCR_SIREN_",
    "OCR_SIRET_",
    "OCR_EU_VAT_",
    "OCR_RPPS_",
    "OCR_FINESS_",
)

DEMO_DOCUMENTS = {
    "Document intact": Path("tests/fixtures/assurance-sans-fraude.pdf"),
    "Ajout légitime": Path("tests/fixtures/assurance-ajout-legitime.pdf"),
    "Montant modifié": Path("tests/fixtures/assurance-fraude.pdf"),
}
OCR_DEMO_DOCUMENTS = {
    "OCR - Facture de soins": Path("tests/fixtures/ocr/facture-soins"),
    "OCR - Déclaration cohérente": Path("tests/fixtures/ocr/declaration-coherente"),
    "OCR - Contrat bruité": Path("tests/fixtures/ocr/contrat-bruite"),
    "OCR - Relevé bancaire à anomalies": Path("tests/fixtures/ocr/releve-bancaire-anomalies"),
    "OCR - Texte insuffisant": Path("tests/fixtures/ocr/texte-insuffisant"),
}
LOCAL_OCR_DEMO_DOCUMENTS = {
    "OCR original local - Facture de soins": WORK_DIR / "ocr-fixtures/original/facture-soins",
    "OCR original local - Declaration coherente": WORK_DIR
    / "ocr-fixtures/original/declaration-coherente",
    "OCR original local - Contrat bruite": WORK_DIR / "ocr-fixtures/original/contrat-bruite",
    "OCR original local - Releve bancaire a anomalies": WORK_DIR
    / "ocr-fixtures/original/releve-bancaire-anomalies",
    "OCR original local - Texte insuffisant": WORK_DIR / "ocr-fixtures/original/texte-insuffisant",
}

LEVEL_STYLE = {
    "low": {
        "label": "Faible",
        "tone": "low",
        "color": "#34d399",
        "background": "#062d24",
        "border": "#10b981",
    },
    "review": {
        "label": "Revue",
        "tone": "review",
        "color": "#fbbf24",
        "background": "#3b2604",
        "border": "#f59e0b",
    },
    "high": {
        "label": "Élevé",
        "tone": "high",
        "color": "#fb7185",
        "background": "#3c0912",
        "border": "#f43f5e",
    },
}

CATEGORY_LABELS = {
    "analysis_quality": "Qualité de l'analyse",
    "annotations": "Annotations PDF",
    "content_consistency": "Cohérence du contenu",
    "document_integrity": "Structure du fichier",
    "metadata": "Métadonnées du document",
    "page_composition": "Composition de la page",
    "provenance_integrity": "Preuves d'origine",
    "raster_forensics": "Retouches de l'image",
    "revision_history": "Historique des versions",
    "revision_visual": "Modifications visuelles",
    "synthetic_media": "Génération par IA",
}

CATEGORY_DESCRIPTIONS = {
    "analysis_quality": "Qualité et couverture des opérations d'analyse.",
    "annotations": "Commentaires, formulaires et éléments ajoutés au PDF.",
    "content_consistency": "Cohérence des dates, montants et identifiants reconnus.",
    "document_integrity": "Signatures, structure interne et altérations du fichier.",
    "metadata": "Logiciels, dates et propriétés enregistrés dans le document.",
    "page_composition": "Images, textes ou objets superposés à la page.",
    "provenance_integrity": "Origine déclarée, certificats et traces d'authenticité.",
    "raster_forensics": "Compression et variations visuelles pouvant signaler une retouche.",
    "revision_history": "Réenregistrements et versions conservées dans le fichier.",
    "revision_visual": "Différences visibles entre les versions du document.",
    "synthetic_media": "Ressemblance visuelle avec des images générées par IA.",
}

CATEGORY_DETECTORS = {
    "annotations": frozenset({"page_composition"}),
    "content_consistency": frozenset({"ocr_content"}),
    "document_integrity": frozenset({"pdf_structure"}),
    "metadata": frozenset({"pdf_structure"}),
    "page_composition": frozenset({"page_composition"}),
    "provenance_integrity": frozenset({"image_provenance"}),
    "raster_forensics": frozenset({"raster_anomaly"}),
    "revision_history": frozenset({"pdf_structure"}),
    "revision_visual": frozenset({"revision_diff"}),
    "synthetic_media": frozenset({"image_provenance", "ai_generated_image"}),
}


@dataclass(frozen=True, slots=True)
class InputDocument:
    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class OcrDemoDocument:
    name: str
    fixture_dir: Path


@dataclass(frozen=True, slots=True)
class RiskIndicator:
    category: str
    label: str
    points: float
    maximum: float
    bar_percentage: float
    tone: str
    state: str


def main() -> None:
    st.set_page_config(
        page_title="FROD",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()
    workspace_view, header_action = _render_app_header()
    uploaded_file = _render_input_panel(hidden=workspace_view == "Glossaire")

    if uploaded_file is None:
        if workspace_view == "Glossaire":
            _render_indicator_glossary()
        else:
            _render_empty_state()
        return

    _render_header_document_action(header_action)
    if isinstance(uploaded_file, OcrDemoDocument):
        _handle_ocr_demo(uploaded_file, workspace_view=workspace_view)
        _render_pending_scroll_reset()
        return

    file_bytes = uploaded_file.data
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    config = _config_from_state()

    progress = st.empty()

    current_key = (
        ANALYSIS_POLICY_VERSION,
        file_hash,
        uploaded_file.name,
    )
    cached = st.session_state.get("analysis")
    if cached is not None and cached.get("key") == current_key:
        report = cached["report"]
        laboratory = cached.get("laboratory")
        output_dir = cached["output_dir"]
        source_path = cached["source_path"]
    else:
        _render_analysis_progress(progress, 0.0, "Préparation de l'analyse")
        try:
            report, laboratory, output_dir, source_path = _run_analysis(
                file_name=uploaded_file.name,
                file_bytes=file_bytes,
                file_hash=file_hash,
                config=config,
                progress_callback=lambda value, text: _render_analysis_progress(
                    progress,
                    value,
                    text,
                ),
            )
            _render_analysis_progress(progress, 1.0, "Analyse terminée")
        except AnalysisError as error:
            progress.empty()
            st.error(f"Erreur [{error.code}] : {error}")
            return
        except Exception as error:
            progress.empty()
            st.error(f"Erreur inattendue : {type(error).__name__}: {str(error)[:240]}")
            return
        progress.empty()
        st.session_state["analysis"] = {
            "key": current_key,
            "report": report,
            "laboratory": laboratory,
            "output_dir": output_dir,
            "source_path": source_path,
        }

    entering = _render_pdf_loading_if_pending(
        workspace_view,
        label="Chargement du PDF" if _is_pdf_bytes(file_bytes) else "Chargement du document",
    )
    result_key = "analysis_results_entering" if entering else "analysis_results_ready"
    with st.container(key=result_key):
        _render_report(
            report,
            laboratory,
            output_dir,
            source_path,
            document_name=uploaded_file.name,
            workspace_view=workspace_view,
        )
    _render_pending_scroll_reset()


def _render_app_header() -> tuple[str, Any]:
    with st.container(
        key="app_header",
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
        gap="medium",
    ):
        st.markdown(
            '<div class="brandbar"><h1>FROD</h1></div>',
            unsafe_allow_html=True,
        )
        with st.container(
            key="header_actions",
            horizontal=True,
            horizontal_alignment="right",
            vertical_alignment="center",
            gap="small",
            width="content",
        ):
            with st.container(key="header_navigation", width="content"):
                selected = st.segmented_control(
                    "Navigation principale",
                    ("Analyse", "Laboratoire", "Glossaire"),
                    default="Analyse",
                    key="workspace_view",
                    label_visibility="collapsed",
                )
            action_slot = st.empty()
    return selected or "Analyse", action_slot


def _render_input_panel(*, hidden: bool = False) -> InputDocument | OcrDemoDocument | None:
    selected_document = st.session_state.get("selected_document")
    if isinstance(selected_document, dict):
        if selected_document.get("kind") == "input":
            return InputDocument(
                name=str(selected_document["name"]),
                data=bytes(selected_document["data"]),
            )
        if selected_document.get("kind") == "ocr":
            return OcrDemoDocument(
                name=str(selected_document["name"]),
                fixture_dir=Path(selected_document["fixture_dir"]),
            )
    if hidden:
        return None

    revision = int(st.session_state.get("document_input_revision", 0))
    input_slot = st.empty()
    with input_slot.container():
        uploaded_file = st.file_uploader(
            "Déposer un document",
            type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "gif", "bmp"],
            key=f"document_upload_{revision}",
        )
        if uploaded_file is not None:
            document = InputDocument(name=uploaded_file.name, data=uploaded_file.getvalue())
        else:
            available_ocr_demos = {
                **OCR_DEMO_DOCUMENTS,
                **{
                    name: path
                    for name, path in LOCAL_OCR_DEMO_DOCUMENTS.items()
                    if (path / "document.json").is_file() and (path / "document.md").is_file()
                },
            }
            demo_name = st.selectbox(
                "Document de démonstration",
                ("Aucun", *DEMO_DOCUMENTS, *available_ocr_demos),
                key=f"document_demo_{revision}",
            )
            if demo_name == "Aucun":
                document = None
            elif demo_name in DEMO_DOCUMENTS:
                demo_path = DEMO_DOCUMENTS[demo_name]
                if not demo_path.is_file():
                    st.error("Le document de demonstration est indisponible.")
                    document = None
                else:
                    document = InputDocument(name=demo_path.name, data=demo_path.read_bytes())
            else:
                fixture_dir = available_ocr_demos[demo_name]
                if not (fixture_dir / "document.json").is_file():
                    st.error("Les donnees OCR de demonstration sont indisponibles.")
                    document = None
                else:
                    document = OcrDemoDocument(name=demo_name, fixture_dir=fixture_dir)

    if document is not None:
        st.session_state["selected_document"] = (
            {"kind": "input", "name": document.name, "data": document.data}
            if isinstance(document, InputDocument)
            else {
                "kind": "ocr",
                "name": document.name,
                "fixture_dir": str(document.fixture_dir),
            }
        )
        st.session_state["pending_scroll_reset"] = True
        st.session_state["pending_pdf_loading"] = True
        input_slot.empty()
    return document


def _render_header_document_action(target: Any) -> None:
    target.button(
        "Tester un nouveau document",
        key=f"header_reset_document_{st.session_state.get('document_input_revision', 0)}",
        on_click=_reset_document_input,
    )


def _reset_document_input() -> None:
    revision = int(st.session_state.get("document_input_revision", 0))
    st.session_state["document_input_revision"] = revision + 1
    st.session_state.pop("selected_document", None)
    st.session_state.pop("analysis", None)
    st.session_state.pop("pending_pdf_loading", None)


def _render_pending_scroll_reset() -> None:
    if not st.session_state.pop("pending_scroll_reset", False):
        return
    st.html(
        """
        <script>
          (() => {
            const main = document.querySelector('[data-testid="stMain"]');
            const resetScroll = () => main?.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            window.requestAnimationFrame(() => {
              resetScroll();
              window.setTimeout(resetScroll, 120);
              window.setTimeout(resetScroll, 360);
            });
          })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _render_pdf_loading_if_pending(workspace_view: str, *, label: str) -> bool:
    if workspace_view != "Analyse":
        return False
    if not st.session_state.pop("pending_pdf_loading", False):
        return False
    st.markdown(
        f"""
        <div class="pdf-loading-stage" role="status" aria-live="polite" aria-busy="true">
          <i aria-hidden="true"></i><span>{_html(label)}&hellip;</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return True


def _handle_ocr_demo(document: OcrDemoDocument, *, workspace_view: str) -> None:
    json_path = document.fixture_dir / "document.json"
    markdown_path = document.fixture_dir / "document.md"
    json_bytes = json_path.read_bytes()
    markdown = markdown_path.read_text(encoding="utf-8")
    fixture_hash = hashlib.sha256(json_bytes + markdown.encode("utf-8")).hexdigest()
    current_key = ("ocr-demo-v3", document.name, fixture_hash)

    cached = st.session_state.get("analysis")
    if cached is not None and cached.get("key") == current_key:
        payload = cached["ocr_payload"]
        markdown = cached["ocr_markdown"]
        laboratory = cached["laboratory"]
        ocr_detector = cached["ocr_detector"]
        classification = cached.get("classification")
        extraction = cached.get("extraction")
        verification = cached.get("verification")
    else:
        try:
            payload = json.loads(json_bytes)
        except json.JSONDecodeError as error:
            st.error(f"Fixture OCR invalide : {error}")
            return
        laboratory = LaboratoryReport(
            schema_version="0.1-experimental",
            checks=analyze_ocr_laboratory(payload),
        )
        ocr_detector = build_ocr_content_result(
            OcrReport(
                success=True,
                error_message=None,
                markdown=markdown,
                json_result=payload,
            )
        )
        classification = None
        extraction = None
        verification = None
        if PROJECT_CONFIG.analysis.classification_enabled:
            try:
                classification = classify_document(markdown, PROJECT_CONFIG.analysis)
            except Exception:
                classification = None
        if PROJECT_CONFIG.analysis.extraction_enabled:
            try:
                extraction = extract_document(payload, classification, PROJECT_CONFIG.analysis)
            except Exception:
                extraction = None
        if PROJECT_CONFIG.analysis.verification_enabled and extraction is not None:
            try:
                verification = verify_extraction(
                    payload,
                    extraction,
                    classification,
                    PROJECT_CONFIG.analysis,
                )
                extraction, verification = reconcile_extraction(extraction, verification)
            except Exception:
                verification = None
        st.session_state["analysis"] = {
            "key": current_key,
            "ocr_payload": payload,
            "ocr_markdown": markdown,
            "laboratory": laboratory,
            "ocr_detector": ocr_detector,
            "classification": classification,
            "extraction": extraction,
            "verification": verification,
        }

    entering = _render_pdf_loading_if_pending(
        workspace_view,
        label="Chargement du document",
    )
    result_key = "analysis_results_entering" if entering else "analysis_results_ready"
    with st.container(key=result_key):
        _render_ocr_demo_report(
            name=document.name,
            markdown=markdown,
            laboratory=laboratory,
            ocr_detector=ocr_detector,
            classification=classification,
            extraction=extraction,
            verification=verification,
            workspace_view=workspace_view,
        )


def _config_from_state() -> AnalysisConfig:
    return PROJECT_CONFIG.analysis


def _run_analysis(
    *,
    file_name: str,
    file_bytes: bytes,
    file_hash: str,
    config: AnalysisConfig,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[
    AnalysisReport | ImageAnalysisReport,
    LaboratoryReport | None,
    Path,
    Path,
]:
    def report_progress(value: float, text: str) -> None:
        if progress_callback is not None:
            progress_callback(value, text)

    report_progress(0.02, "Preparation du fichier")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(file_name)
    source = UPLOAD_DIR / f"{file_hash[:16]}-{safe_name}"
    source.write_bytes(file_bytes)

    output_dir = RUN_DIR / file_hash[:16]
    if output_dir.exists():
        shutil.rmtree(output_dir)

    adapter_loaders = ()
    if PROJECT_CONFIG.gapl.enabled and GAPL_WEIGHTS.is_file():
        report_progress(0.06, "Préparation de l'analyse des images")
        device = (
            best_available_device()
            if PROJECT_CONFIG.gapl.device == "auto"
            else PROJECT_CONFIG.gapl.device
        )
        weights_path = str(GAPL_WEIGHTS.resolve())
        adapter_loaders = (
            partial(
                _load_gapl_adapter,
                weights_path,
                device,
            ),
        )

    if _is_pdf_bytes(file_bytes):

        def pipeline_progress(value: float, text: str) -> None:
            report_progress(0.12 + 0.62 * value, text)

        report = AnalysisPipeline(
            config=config,
            ai_image_adapter_loaders=adapter_loaders,
        ).analyze(
            source,
            output_dir,
            progress_callback=pipeline_progress,
        )

        laboratory = (
            analyze_pdf_laboratory(
                source,
                output_dir,
                config=config,
                progress_callback=lambda value, text: report_progress(
                    0.74 + 0.26 * value,
                    text,
                ),
            )
            if PROJECT_CONFIG.laboratory.pdf_enabled
            else _empty_laboratory_report()
        )
        laboratory = _with_ocr_laboratory(report, laboratory, output_dir, source)
        return report, laboratory, output_dir, source

    def pipeline_progress(value: float, text: str) -> None:
        report_progress(0.12 + 0.60 * value, text)

    report = ImageAnalysisPipeline(
        config=config,
        ai_image_adapter_loaders=adapter_loaders,
    ).analyze(
        source,
        output_dir,
        progress_callback=pipeline_progress,
    )
    _release_transient_memory()
    laboratory = (
        analyze_image_laboratory(
            source,
            output_dir,
            trufor_weights=TRUFOR_WEIGHTS,
            trufor_max_pixels=TRUFOR_PIXEL_BUDGET,
            trufor_timeout_seconds=PROJECT_CONFIG.trufor.timeout_seconds,
            progress_callback=lambda value, text: report_progress(
                0.72 + 0.28 * value,
                text,
            ),
        )
        if PROJECT_CONFIG.laboratory.image_enabled and PROJECT_CONFIG.trufor.enabled
        else _empty_laboratory_report()
    )
    laboratory = _with_ocr_laboratory(report, laboratory, output_dir, source)
    return report, laboratory, output_dir, source


def _empty_laboratory_report() -> LaboratoryReport:
    return LaboratoryReport(schema_version="1.0", checks=())


def _with_ocr_laboratory(
    report: AnalysisReport | ImageAnalysisReport,
    laboratory: LaboratoryReport,
    output_dir: Path,
    source_path: Path,
) -> LaboratoryReport:
    json_paths = _existing_artifacts(
        output_dir,
        report.artifacts.get("ocr_json", ()),
    )
    if not json_paths:
        return laboratory
    try:
        payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return laboratory
    page_images = _existing_artifacts(
        output_dir,
        report.artifacts.get("page_renders", ()),
    )
    if not page_images and isinstance(report, ImageAnalysisReport) and source_path.is_file():
        page_images = [source_path]
    visual_repetition = (
        analyze_repeated_visual_regions(
            payload,
            tuple(page_images),
            output_dir / "laboratory",
            minimum_pages=PROJECT_CONFIG.laboratory.visual_repetition_min_pages,
            minimum_similarity=PROJECT_CONFIG.laboratory.visual_repetition_similarity,
        )
        if PROJECT_CONFIG.laboratory.visual_repetition_enabled
        else None
    )
    return LaboratoryReport(
        schema_version=laboratory.schema_version,
        checks=(
            *laboratory.checks,
            *analyze_ocr_laboratory(payload),
            *((visual_repetition,) if visual_repetition is not None else ()),
        ),
    )


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
          <h2>Ajouter un document</h2>
          <p>PDF, PNG, JPEG, WebP, TIFF, GIF ou BMP.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_report(
    report: AnalysisReport | ImageAnalysisReport,
    laboratory: LaboratoryReport | None,
    output_dir: Path,
    source_path: Path,
    *,
    document_name: str,
    workspace_view: str,
) -> None:
    findings = sorted(
        report.findings,
        key=lambda finding: (finding.risk_points, finding.confidence),
        reverse=True,
    )
    scored = [finding for finding in findings if finding.risk_points > 0]
    diagnostics = [finding for finding in findings if finding.risk_points == 0]
    layout_images = _read_ocr_layout_images(report, output_dir)

    if workspace_view == "Glossaire":
        _render_indicator_glossary()
        return
    if workspace_view == "Laboratoire":
        _render_extraction_laboratory(
            report.extraction,
            classification=report.classification,
            verification=report.extraction_verification,
            recognized_text=_read_ocr_markdown(report, output_dir),
        )
        return

    document_column, indicators_column = st.columns([0.56, 0.44], gap="large")
    with document_column:
        _render_document_view(
            report=report,
            laboratory=laboratory,
            output_dir=output_dir,
            source_path=source_path,
            layout_images=layout_images,
            document_name=document_name,
        )
    with indicators_column:
        _render_score(report)
        _render_risk_indicators(report.findings, report.detectors)

    _render_review_summary(scored, diagnostics, laboratory)


def _render_ocr_demo_report(
    *,
    name: str,
    markdown: str,
    laboratory: LaboratoryReport,
    ocr_detector: DetectorResult,
    classification: DocumentClassification | None,
    extraction: DocumentExtraction | None,
    verification: ExtractionVerification | None,
    workspace_view: str,
) -> None:
    finding = ocr_detector.findings[0] if ocr_detector.findings else None
    points = finding.risk_points if finding is not None else 0.0
    tone = "review" if points >= 30 else "low"
    color = LEVEL_STYLE[tone]["color"]
    if workspace_view == "Glossaire":
        _render_indicator_glossary(categories=("content_consistency",))
        return
    if workspace_view == "Laboratoire":
        _render_extraction_laboratory(
            extraction,
            classification=classification,
            verification=verification,
            recognized_text=markdown,
        )
        return

    document_column, indicators_column = st.columns([0.56, 0.44], gap="large")
    with document_column:
        st.markdown(
            f'<h2 class="workspace-title document-name">{_html(name)}</h2>',
            unsafe_allow_html=True,
        )
        _render_recognized_text(markdown, compact=True)
    with indicators_column:
        _render_compact_score(
            score=points,
            maximum=30,
            tone=tone,
            color=color,
            label="Score de contrôle",
            score_steps=_build_score_steps(
                ocr_detector.findings,
                categories=("content_consistency",),
            ),
        )
        _render_risk_indicators(
            ocr_detector.findings,
            (ocr_detector,),
            categories=("content_consistency",),
        )

    scored = [finding] if finding is not None and finding.risk_points > 0 else []
    diagnostics = [finding] if finding is not None and finding.risk_points == 0 else []
    _render_review_summary(scored, diagnostics, laboratory)


def _read_ocr_layout_images(
    report: AnalysisReport | ImageAnalysisReport,
    output_dir: Path,
) -> list[Path]:
    return _existing_artifacts(
        output_dir,
        report.artifacts.get("ocr_layout", ()),
    )


def _read_ocr_markdown(
    report: AnalysisReport | ImageAnalysisReport,
    output_dir: Path,
) -> str:
    paths = _existing_artifacts(output_dir, report.artifacts.get("ocr_markdown", ()))
    if not paths:
        return ""
    try:
        return paths[0].read_text(encoding="utf-8")
    except OSError:
        return ""


EXTRACTION_FIELD_LABELS = {
    "person_name": "Personne",
    "organization_name": "Organisation",
    "address": "Adresse",
    "phone_number": "Téléphone",
    "email_address": "Adresse e-mail",
    "document_number": "Numéro de document",
    "invoice_number": "Numéro de facture",
    "contract_number": "Numéro de contrat",
    "claim_number": "Numéro de sinistre",
    "account_number": "Numéro de compte",
    "tax_identifier": "Identifiant fiscal",
    "professional_identifier": "Identifiant professionnel",
    "registration_identifier": "Numéro d'enregistrement",
    "other_identifier": "Autre identifiant",
    "iban": "IBAN",
    "bic": "BIC",
    "payment_card_number": "Numéro de carte",
    "date": "Date",
    "date_period": "Période",
    "monetary_amount": "Montant",
    "quantity": "Quantité",
    "percentage": "Pourcentage",
    "service_description": "Service",
    "service_code": "Code de service",
    "product_description": "Produit",
    "product_code": "Code produit",
    "transaction_description": "Transaction",
}

EXTRACTION_ROLE_LABELS = {
    "issuer": "Émetteur",
    "recipient": "Destinataire",
    "customer": "Client",
    "patient": "Patient",
    "practitioner": "Professionnel",
    "provider": "Prestataire",
    "beneficiary": "Bénéficiaire",
    "payer": "Payeur",
    "account_holder": "Titulaire",
    "bank": "Banque",
    "insurer": "Assureur",
    "employer": "Employeur",
    "supplier": "Fournisseur",
    "document": "Document",
    "invoice": "Facture",
    "contract": "Contrat",
    "claim": "Sinistre",
    "transaction": "Transaction",
    "line_item": "Ligne",
    "subtotal": "Sous-total",
    "tax": "Taxe",
    "total": "Total",
    "opening_balance": "Solde initial",
    "closing_balance": "Solde final",
    "debit": "Débit",
    "credit": "Crédit",
    "unit_price": "Prix unitaire",
    "issue": "Émission",
    "due": "Échéance",
    "payment": "Paiement",
    "service": "Prestation",
    "start": "Début",
    "end": "Fin",
    "birth": "Naissance",
    "expiry": "Expiration",
    "other": "Autre",
}

EXTRACTION_COLUMN_ROLE_LABELS = {
    "transaction_date": "Date de transaction",
    "value_date": "Date de valeur",
    "description": "Libellé",
    "debit_amount": "Débit",
    "credit_amount": "Crédit",
    "amount": "Montant",
    "currency": "Devise",
    "balance": "Solde",
    "quantity": "Quantité",
    "unit_price": "Prix unitaire",
    "tax_rate": "Taux de taxe",
    "tax_amount": "Montant de taxe",
    "line_total": "Total de ligne",
    "service_code": "Code de service",
    "product_code": "Code produit",
    "other": "Autre",
}

EXTRACTION_ROW_ROLE_LABELS = {
    "transaction": "Opération",
    "opening_balance": "Solde initial",
    "closing_balance": "Solde final",
    "subtotal": "Sous-total",
    "total": "Total",
    "section_header": "En-tête",
    "line_item": "Ligne détaillée",
    "informational": "Information",
    "other": "Autre",
}


def _render_extraction_laboratory(
    extraction: DocumentExtraction | None,
    *,
    classification: DocumentClassification | None = None,
    verification: ExtractionVerification | None = None,
    recognized_text: str = "",
) -> None:
    st.markdown(
        '<h2 class="workspace-title">Extraction structurée expérimentale</h2>',
        unsafe_allow_html=True,
    )
    _render_classification(classification)
    if extraction is None:
        st.markdown(
            """
            <div class="extraction-empty">
              <strong>Aucune extraction disponible</strong>
              <span>Activer l'OCR et l'extraction locale, puis relancer l'analyse.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if recognized_text:
            _render_recognized_text(recognized_text)
        return

    coverage = extraction.coverage
    coverage_tone = "clear" if coverage.ratio >= 0.95 else "attention"
    facts_count = len(extraction.facts)
    extra_count = len(extraction.additional_fields)
    table_count = len(extraction.tables)
    st.markdown(
        f"""
        <section class="extraction-overview">
          <article class="extraction-coverage {coverage_tone}">
            <div><strong>{coverage.ratio:.0%}</strong><span>Couverture des zones OCR</span></div>
            <i><b style="width:{coverage.ratio:.1%}"></b></i>
            <small>{coverage.accounted_regions} zone(s) comptabilisée(s)
              sur {coverage.total_regions}</small>
          </article>
          <article><strong>{facts_count}</strong><span>Faits comparables</span></article>
          <article><strong>{extra_count}</strong><span>Informations additionnelles</span></article>
          <article><strong>{table_count}</strong><span>Tableaux conservés</span></article>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if extraction.facts:
        rows = []
        for fact in extraction.facts:
            field_label = EXTRACTION_FIELD_LABELS.get(fact.field_code, fact.field_code)
            role_label = EXTRACTION_ROLE_LABELS.get(fact.role, fact.role)
            normalized = fact.normalized_value or "Non normalisée"
            page = f"Page {fact.page}" if fact.page is not None else "Source non localisée"
            rows.append(
                "<tr>"
                f"<td><strong>{_html(field_label)}</strong><span>{_html(role_label)}</span></td>"
                f"<td>{_html(fact.raw_value)}</td>"
                f"<td>{_html(normalized)}</td>"
                f"<td>{fact.confidence:.0%}</td>"
                f"<td>{_html(page)}</td>"
                "</tr>"
            )
        st.markdown(
            '<h3 class="subsection-title">Informations comparables</h3>'
            '<div class="extraction-table-wrap"><table class="extraction-table">'
            "<thead><tr><th>Champ</th><th>Valeur lue</th><th>Valeur de comparaison</th>"
            "<th>Fiabilité</th><th>Source</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>",
            unsafe_allow_html=True,
        )

    if extraction.additional_fields:
        cards = []
        for field in extraction.additional_fields:
            page = f"Page {field.page}" if field.page is not None else "Source non localisée"
            cards.append(
                '<article class="additional-extraction">'
                f"<span>{_html(field.raw_label)}</span>"
                f"<strong>{_html(field.raw_value)}</strong>"
                f"<small>{_html(page)} · Fiabilité {field.confidence:.0%}</small>"
                "</article>"
            )
        st.markdown(
            '<h3 class="subsection-title">Autres informations présentes</h3>'
            '<section class="additional-extractions">' + "".join(cards) + "</section>",
            unsafe_allow_html=True,
        )

    for index, table in enumerate(extraction.tables, start=1):
        title = table.title or f"Tableau {index}"
        transaction_count = sum(role == "transaction" for role in table.row_roles)
        table_summary = (
            f" · {transaction_count} opération(s) identifiée(s)"
            if table.semantic_type == "transactions"
            else ""
        )
        headers = table.headers or tuple(
            f"Colonne {column + 1}"
            for column in range(max((len(row) for row in table.rows), default=0))
        )
        roles = (*table.column_roles, *("other" for _ in range(len(headers))))[: len(headers)]
        head = "<th>Nature</th>" + "".join(
            "<th>"
            f"{_html(header)}"
            f"<span>{_html(EXTRACTION_COLUMN_ROLE_LABELS.get(role, role))}</span>"
            "</th>"
            for header, role in zip(headers, roles, strict=True)
        )
        body_rows = []
        row_roles = (*table.row_roles, *("other" for _ in range(len(table.rows))))[
            : len(table.rows)
        ]
        for row, row_role in zip(table.rows, row_roles, strict=True):
            cells = (*row, *("" for _ in range(max(0, len(headers) - len(row)))))
            body_rows.append(
                "<tr>"
                '<td><span class="extraction-row-role">'
                f"{_html(EXTRACTION_ROW_ROLE_LABELS.get(row_role, row_role))}"
                "</span></td>"
                + "".join(f"<td>{_html(cell)}</td>" for cell in cells[: len(headers)])
                + "</tr>"
            )
        st.markdown(
            f'<h3 class="subsection-title">{_html(title + table_summary)}</h3>'
            '<div class="extraction-table-wrap"><table class="extraction-table">'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
            "</table></div>",
            unsafe_allow_html=True,
        )

    if coverage.uncovered_region_ids:
        st.markdown(
            f"""
            <div class="extraction-coverage-warning">
              <strong>{len(coverage.uncovered_region_ids)} zone(s)
                restent sans interprétation</strong>
              <span>Elles sont conservées dans la sortie OCR et pourront être retraitées.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _render_extraction_verification(verification)

    if recognized_text:
        st.markdown('<h3 class="subsection-title">Texte reconnu</h3>', unsafe_allow_html=True)
        _render_recognized_text(recognized_text, compact=True)

    with st.expander("JSON final de l'extraction", expanded=False):
        st.json(
            {
                "extraction": extraction.to_dict(),
                "verification": verification.to_dict() if verification is not None else None,
            }
        )


def _render_extraction_verification(
    verification: ExtractionVerification | None,
) -> None:
    if verification is None:
        st.markdown(
            """
            <section class="verification-summary unavailable">
              <strong>Vérification indépendante non exécutée</strong>
              <span>L'extraction affichée reste le résultat initial du modèle.</span>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    attention_reviews = tuple(
        review
        for review in verification.reviews
        if review.verdict in {"ambiguous", "contradicted"} and not review.correction_applied
    )
    plausible_count = sum(review.verdict == "plausible" for review in verification.reviews)
    corrected_count = sum(review.correction_applied for review in verification.reviews)
    if verification.status == "clean":
        if corrected_count:
            title = "Extraction corrigée après vérification"
            detail = (
                f"{corrected_count} correction(s) structurée(s) appliquée(s), "
                f"sans point restant à contrôler."
            )
        else:
            title = "Aucune contradiction concrète relevée"
            detail = (
                f"{verification.reviewed_targets} élément(s) contrôlé(s), dont "
                f"{plausible_count} normalisation(s) ou correction(s) OCR jugée(s) plausible(s)."
            )
    elif verification.status == "incomplete":
        title = "Vérification partielle"
        detail = (
            f"{verification.reviewed_targets} élément(s) contrôlé(s) sur "
            f"{verification.expected_targets}. L'extraction initiale n'a pas été modifiée."
        )
    else:
        title = "Points à contrôler dans l'extraction"
        detail = (
            f"{len(attention_reviews)} interprétation(s) discutée(s) et "
            f"{len(verification.omissions)} omission(s) possible(s)."
        )
    st.markdown(
        f"""
        <section class="verification-summary {verification.status}">
          <strong>{_html(title)}</strong>
          <span>{_html(detail)}</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    issue_cards = []
    for review in attention_reviews:
        label = "Contradiction étayée" if review.verdict == "contradicted" else "Ambiguïté"
        suggestion = (
            f"<small>Proposition : {_html(review.suggested_value)}</small>"
            if review.suggested_value
            else ""
        )
        issue_cards.append(
            '<article class="verification-issue">'
            f"<span>{_html(label)} · {_html(review.target_id)}</span>"
            f"<strong>{_html(review.explanation)}</strong>"
            f"<small>Solidité du contrôle : {review.confidence:.0%}</small>"
            f"{suggestion}</article>"
        )
    for omission in verification.omissions:
        value = f" · {_html(omission.proposed_value)}" if omission.proposed_value else ""
        issue_cards.append(
            '<article class="verification-issue omission">'
            f"<span>Omission possible{value}</span>"
            f"<strong>{_html(omission.description)}</strong>"
            f"<small>Solidité du contrôle : {omission.confidence:.0%}</small>"
            "</article>"
        )
    if issue_cards:
        st.markdown(
            '<section class="verification-issues">' + "".join(issue_cards) + "</section>",
            unsafe_allow_html=True,
        )


LAB_STATE_LABELS = {
    "clear": "Conforme",
    "attention": "À examiner",
    "detected": "Élément détecté",
    "indeterminate": "Indéterminé",
    "not_applicable": "Non applicable",
    "error": "Contrôle interrompu",
}

LAB_STRENGTH_LABELS = {
    "strong": "Indice fort",
    "moderate": "Indice modéré",
    "weak": "Indice faible",
    "informational": "Information",
}


def _render_risk_indicators(
    findings: tuple[Finding, ...],
    detectors: tuple[DetectorResult, ...],
    *,
    categories: tuple[str, ...] | None = None,
) -> None:
    indicators = _build_risk_indicators(findings, detectors, categories=categories)
    rows = []
    accessible_rows = []
    for index, indicator in enumerate(indicators, start=1):
        label = _html(indicator.label)
        points = f"{indicator.points:g}"
        maximum = f"{indicator.maximum:g}"
        icon = _indicator_icon(indicator.tone)
        delay_seconds = index * INDICATOR_STEP_SECONDS
        rows.append(
            f"""
            <li
              class="indicator-step {indicator.tone}"
              data-state="{indicator.tone}"
              style="--step-delay:calc(var(--result-entry-delay, 0s) + {delay_seconds:.2f}s);
                --risk-value:{indicator.bar_percentage:.1f}%"
            >
              <span class="indicator-index">{index:02d}</span>
              <div class="indicator-core">
                <div class="indicator-main">
                  <h3>{label}</h3>
                  <span class="indicator-pending"><b></b>En attente</span>
                  <span class="indicator-final"><b>{icon}</b>{_html(indicator.state)}</span>
                </div>
                <div class="indicator-meter"><i></i></div>
              </div>
              <strong class="indicator-score">{points}<span>/{maximum}</span></strong>
            </li>
            """
        )
        accessible_rows.append(
            f"<li><strong>{label}</strong> — {_html(indicator.state)}, "
            f"{points} points sur {maximum}.</li>"
        )
    st.markdown(
        """
        <section class="indicator-sequence" aria-label="Résultats des indicateurs">
          <ol class="indicator-queue" aria-hidden="true">
        """
        + "".join(row.strip() for row in rows)
        + '</ol><ol class="sr-only">'
        + "".join(accessible_rows)
        + "</ol></section>",
        unsafe_allow_html=True,
    )


def _render_indicator_glossary(*, categories: tuple[str, ...] | None = None) -> None:
    selected_categories = categories or tuple(FAMILY_CAPS)
    cards = []
    for category in selected_categories:
        cards.append(
            f"""
            <article class="glossary-card">
              <span>{_html(CATEGORY_LABELS.get(category, category))}</span>
              <p>{_html(CATEGORY_DESCRIPTIONS.get(category, ""))}</p>
              <small>Score maximal : {FAMILY_CAPS[category]:g} points</small>
            </article>
            """
        )
    st.markdown(
        """
        <section class="indicator-glossary" aria-labelledby="indicator-glossary-title">
          <p class="eyebrow">Référentiel</p>
          <h2 id="indicator-glossary-title">Glossaire des indicateurs</h2>
          <div class="glossary-grid">
        """
        + "".join(card.strip() for card in cards)
        + "</div></section>",
        unsafe_allow_html=True,
    )


def _build_risk_indicators(
    findings: tuple[Finding, ...],
    detectors: tuple[DetectorResult, ...],
    *,
    categories: tuple[str, ...] | None = None,
) -> tuple[RiskIndicator, ...]:
    indicators = []
    selected_categories = categories or tuple(FAMILY_CAPS)
    for category in selected_categories:
        cap = FAMILY_CAPS[category]
        points = _family_score(category, findings)
        percentage = min(100, points / cap * 100) if cap else 0
        tone, state, bar_percentage = _indicator_presentation(
            category=category,
            percentage=percentage,
            detectors=detectors,
        )
        indicators.append(
            RiskIndicator(
                category=category,
                label=CATEGORY_LABELS.get(category, category),
                points=points,
                maximum=cap,
                bar_percentage=bar_percentage,
                tone=tone,
                state=state,
            )
        )
    return tuple(indicators)


def _indicator_icon(tone: str) -> str:
    return {
        "clear": "✓",
        "notice-tone": "!",
        "warning": "!",
        "danger": "!",
        "partial": "≈",
        "unavailable": "—",
    }[tone]


def _indicator_presentation(
    *,
    category: str,
    percentage: float,
    detectors: tuple[DetectorResult, ...],
) -> tuple[str, str, float]:
    if percentage > 0:
        return (
            _tone_for_ratio(percentage),
            _risk_state_for_ratio(percentage),
            percentage,
        )

    expected_detectors = CATEGORY_DETECTORS.get(category, frozenset())
    statuses = {detector.status for detector in detectors if detector.name in expected_detectors}
    if "completed" in statuses:
        return "clear", "Aucun signal détecté", 100.0
    if "partial" in statuses:
        return "partial", "Contrôle partiel", 100.0
    if statuses:
        return "unavailable", "Non applicable", 0.0
    return "unavailable", "Non évalué", 0.0


def _family_score(category: str, findings: tuple[Finding, ...]) -> float:
    by_code: dict[str, float] = {}
    for finding in findings:
        if finding.category != category or finding.risk_points <= 0:
            continue
        by_code[finding.code] = max(by_code.get(finding.code, 0.0), finding.risk_points)
    if not by_code:
        return 0.0
    ordered = sorted(by_code.values(), reverse=True)
    raw_score = ordered[0] + 0.2 * sum(ordered[1:])
    return round(min(FAMILY_CAPS[category], raw_score), 1)


def _build_score_steps(
    findings: tuple[Finding, ...],
    *,
    categories: tuple[str, ...],
) -> tuple[int, ...]:
    """Return the cumulative risk score revealed after each indicator."""

    revealed_categories: set[str] = set()
    steps = []
    for category in categories:
        revealed_categories.add(category)
        revealed_findings = tuple(
            finding for finding in findings if finding.category in revealed_categories
        )
        steps.append(round(assess_risk(revealed_findings).score))
    return tuple(steps)


def _tone_for_ratio(percentage: float) -> str:
    if percentage >= 75:
        return "danger"
    if percentage >= 40:
        return "warning"
    if percentage > 0:
        return "notice-tone"
    return "clear"


def _risk_state_for_ratio(percentage: float) -> str:
    if percentage >= 75:
        return "Risque élevé"
    if percentage >= 40:
        return "Risque modéré"
    if percentage > 0:
        return "Risque faible"
    return "Aucun signal détecté"


def _render_document_view(
    *,
    report: AnalysisReport | ImageAnalysisReport,
    laboratory: LaboratoryReport | None,
    output_dir: Path,
    source_path: Path,
    layout_images: list[Path],
    document_name: str,
) -> None:
    st.markdown(
        f'<h2 class="workspace-title document-name">{_html(document_name)}</h2>',
        unsafe_allow_html=True,
    )
    choices: dict[str, Path] = {}
    if isinstance(report, ImageAnalysisReport) and source_path.is_file():
        choices["Document original"] = source_path

    page_images = _existing_artifacts(output_dir, report.artifacts.get("page_renders", ()))
    for index, path in enumerate(page_images, start=1):
        choices[f"Document - page {index}"] = path

    review_images = _existing_artifacts(output_dir, report.artifacts.get("review_overlays", ()))
    for index, path in enumerate(review_images, start=1):
        choices[f"Zones à revoir - page {index}"] = path

    for index, path in enumerate(layout_images, start=1):
        choices[f"Zones de texte reconnues - page {index}"] = path

    forensic_images = [
        path
        for path in _existing_artifacts(output_dir, report.artifacts.get("forensics", ()))
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    for path in forensic_images:
        choices[_friendly_artifact_caption(path)] = path

    if laboratory is not None:
        for check in laboratory.checks:
            for observation in check.observations:
                for path in _existing_artifacts(
                    output_dir / "laboratory",
                    observation.artifacts,
                ):
                    if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}:
                        if _is_secondary_localization_artifact(path):
                            continue
                        choices[_laboratory_artifact_caption(path)] = path

    if not choices:
        st.info("Aucun apercu visuel disponible.")
        return

    labels = tuple(choices)
    selected = (
        st.selectbox("Vue affichée", labels, label_visibility="collapsed")
        if len(labels) > 1
        else labels[0]
    )
    selected_path = choices[selected]
    st.image(str(selected_path), caption=selected, width="stretch")


def _render_review_summary(
    scored: list[Finding],
    diagnostics: list[Finding],
    laboratory: LaboratoryReport | None,
) -> None:
    st.markdown('<h2 class="workspace-title">Synthèse de revue</h2>', unsafe_allow_html=True)
    grouped_scored = _group_findings(scored)
    if grouped_scored:
        st.markdown(
            f"""
            <div class="review-banner attention">
              <strong>{len(grouped_scored)} type(s) d'indice à contrôler</strong>
              <span>Comparer les zones signalées avec le document.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for finding, occurrences in grouped_scored:
            _finding_card(finding, occurrences=occurrences)
    else:
        st.markdown(
            """
            <div class="review-banner clear">
              <strong>Aucun signal pris en compte dans le score</strong>
              <span>Les contrôles réalisés n'ont pas relevé d'anomalie forte.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if diagnostics:
        st.markdown(
            f'<div class="diagnostic-count">{len(diagnostics)} observation(s) informative(s)</div>',
            unsafe_allow_html=True,
        )

    _render_ocr_field_cards(laboratory)
    _render_attention_observations(laboratory)
    _render_control_matrix(laboratory)


def _render_classification(classification: DocumentClassification | None) -> None:
    """Render the semantic family without exposing model internals."""
    if not classification:
        return

    st.markdown(
        '<h2 class="workspace-title">Type de document reconnu</h2>',
        unsafe_allow_html=True,
    )

    family_labels = {
        "facture_recu": "Facture ou reçu",
        "devis": "Devis",
        "releve_bancaire": "Relevé bancaire",
        "justificatif_bancaire": "Justificatif bancaire",
        "document_medical": "Document médical",
        "declaration_sinistre": "Déclaration de sinistre",
        "constat_accident": "Constat d'accident",
        "contrat_attestation": "Contrat ou attestation",
        "piece_identite": "Pièce d'identité",
        "justificatif_revenus_fiscal": "Justificatif de revenus ou fiscal",
        "justificatif_domicile": "Justificatif de domicile",
        "correspondance": "Correspondance",
        "autre": "Type non déterminé",
    }
    family_label = family_labels.get(classification.family, "Type non déterminé")
    if classification.family == "autre":
        tone = "undetermined"
    elif classification.reliability >= 0.75:
        tone = "reliable"
    elif classification.reliability >= 0.50:
        tone = "attention"
    else:
        tone = "uncertain"

    language = classification.language.upper() if classification.language else "Non déterminée"
    country = classification.country or "Non déterminé"

    st.markdown(
        f"""
        <div class="classification-card {tone}">
            <div class="classification-content">
                <div class="classification-category">{_html(family_label)}</div>
                <div class="classification-metadata">
                    <span>Langue <strong>{_html(language)}</strong></span>
                    <span>Pays <strong>{_html(country)}</strong></span>
                    <span>Fiabilité du classement
                      <strong>{classification.reliability:.0%}</strong>
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if classification.evidence:
        st.markdown(
            '<div class="classification-evidence-title">Éléments ayant guidé le classement</div>',
            unsafe_allow_html=True,
        )
        for excerpt in classification.evidence:
            st.markdown(
                f'<blockquote class="classification-evidence">{_html(excerpt)}</blockquote>',
                unsafe_allow_html=True,
            )


def _group_findings(findings: list[Finding]) -> list[tuple[Finding, int]]:
    groups: dict[tuple[str, str, str], list[Finding]] = {}
    for finding in findings:
        key = (finding.category, finding.code, finding.title)
        groups.setdefault(key, []).append(finding)
    grouped = [
        (
            max(items, key=lambda item: (item.risk_points, item.confidence)),
            len(items),
        )
        for items in groups.values()
    ]
    return sorted(
        grouped,
        key=lambda item: (item[0].risk_points, item[0].confidence),
        reverse=True,
    )


def _render_ocr_field_cards(laboratory: LaboratoryReport | None) -> None:
    if laboratory is None:
        return
    fields = [
        observation
        for check in laboratory.checks
        for observation in check.observations
        if observation.code.startswith(OCR_IDENTIFIER_PREFIXES)
    ]
    if not fields:
        return

    cards = []
    for observation in fields:
        evidence = observation.evidence
        value = evidence.get("value")
        if value is None and evidence.get("values"):
            value = " / ".join(str(item) for item in evidence["values"])
        value = value or "Valeur complète indisponible"
        value = _format_identifier_value(observation.code, str(value))
        state_label = LAB_STATE_LABELS[observation.state]
        cards.append(
            f"""
            <article class="extracted-field {observation.state}">
              <div><span>{_html(_sentence_case(observation.title))}</span><strong>{_html(value)}</strong></div>
              <b>{_html(state_label)}</b>
              <p>{_html(observation.summary)}</p>
            </article>
            """
        )
    st.markdown('<h3 class="subsection-title">Identifiants reconnus</h3>', unsafe_allow_html=True)
    st.markdown(
        '<section class="extracted-fields">'
        + "".join(card.strip() for card in cards)
        + "</section>",
        unsafe_allow_html=True,
    )


def _render_attention_observations(laboratory: LaboratoryReport | None) -> None:
    if laboratory is None:
        return
    observations = [
        observation
        for check in laboratory.checks
        for observation in check.observations
        if (
            observation.state in {"attention", "error"}
            or (observation.state == "detected" and observation.strength != "informational")
        )
        and not observation.code.startswith(OCR_IDENTIFIER_PREFIXES)
    ]
    if not observations:
        return
    st.markdown('<h3 class="subsection-title">Points de contrôle</h3>', unsafe_allow_html=True)
    for observation in observations:
        strength = LAB_STRENGTH_LABELS[observation.strength]
        location = f"Page {observation.page}" if observation.page is not None else "Document"
        st.markdown(
            f"""
            <article class="business-observation {observation.state}">
              <div><strong>{_html(_business_observation_title(observation))}</strong><span>{_html(location)}</span></div>
              <p>{_html(_business_text(observation.summary))}</p>
              <small>{_html(strength)} - {_html(_business_text(observation.explanation))}</small>
            </article>
            """,
            unsafe_allow_html=True,
        )


def _render_control_matrix(laboratory: LaboratoryReport | None) -> None:
    if laboratory is None or not laboratory.checks:
        return
    cards = []
    for check in laboratory.checks:
        if check.state == "not_applicable":
            continue
        cards.append(
            f"""
            <article class="control-status {check.state}">
              <span>{_html(LAB_STATE_LABELS[check.state])}</span>
              <strong>{_html(_business_check_title(check.code, check.title))}</strong>
              <p>{_html(_business_text(check.summary))}</p>
            </article>
            """
        )
    if not cards:
        return
    st.markdown('<h3 class="subsection-title">Contrôles effectués</h3>', unsafe_allow_html=True)
    st.markdown(
        '<section class="control-matrix">' + "".join(card.strip() for card in cards) + "</section>",
        unsafe_allow_html=True,
    )


def _render_recognized_text(markdown: str, *, compact: bool = False) -> None:
    visible = re.sub(r"<[^>]+>", " ", html_lib.unescape(markdown))
    visible = re.sub(r"(?m)^\s*#{1,6}\s*", "", visible)
    visible = re.sub(r"[ \t]+", " ", visible)
    visible = re.sub(r"\n{3,}", "\n\n", visible).strip()
    title = "" if compact else '<h2 class="section-title">Texte reconnu</h2>'
    st.markdown(
        f'{title}<div class="recognized-text">{_html(visible)}</div>',
        unsafe_allow_html=True,
    )


def _render_analysis_progress(target: Any, value: float, label: str) -> None:
    bounded = min(1.0, max(0.0, value))
    percentage = round(bounded * 100)
    target.markdown(
        f"""
        <div class="analysis-progress">
          <div class="analysis-progress-head">
            <span>{_html(label)}</span>
            <strong>{percentage}%</strong>
          </div>
          <div class="analysis-progress-track">
            <i style="width:{percentage}%"></i>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_gapl_adapter(weights_path: str, device: str) -> Any:
    return create_gapl_adapter(weights_path=weights_path, device=device)


def _render_score(
    report: AnalysisReport | ImageAnalysisReport,
) -> None:
    assessment = report.assessment
    style = LEVEL_STYLE[assessment.level]
    _render_compact_score(
        score=assessment.score,
        maximum=100,
        tone=style["tone"],
        color=style["color"],
        label="Score de fraude",
        score_steps=_build_score_steps(
            report.findings,
            categories=tuple(FAMILY_CAPS),
        ),
    )


def _render_compact_score(
    *,
    score: float,
    maximum: int,
    tone: str,
    color: str,
    label: str,
    score_steps: tuple[int, ...],
) -> None:
    displayed_score = round(score)
    normalized_steps = tuple(round(step) for step in score_steps) or (displayed_score,)
    if normalized_steps[-1] != displayed_score:
        normalized_steps = (*normalized_steps[:-1], displayed_score)
    duration_seconds = len(normalized_steps) * INDICATOR_STEP_SECONDS
    animation_name = (
        "fraud-score-count-"
        + hashlib.sha256(repr(normalized_steps).encode("utf-8")).hexdigest()[:10]
    )
    keyframes = ["0% { --animated-score:0; }"]
    for index, step in enumerate(normalized_steps, start=1):
        percentage = index / len(normalized_steps) * 100
        keyframes.append(f"{percentage:.3f}% {{ --animated-score:{step}; }}")
    st.markdown(
        f"""
        <style>@keyframes {animation_name} {{ {" ".join(keyframes)} }}</style>
        <section class="fraud-score {tone}" aria-label="{_html(label)} : {score:g} sur {maximum}">
          <span>{_html(label)}</span>
          <strong style="--score-color:{color};--score-target:{displayed_score};
            --score-duration:{duration_seconds:.2f}s;--score-animation:{animation_name}">
            <span class="fraud-score-number" aria-hidden="true"></span>
            <span class="sr-only">{score:g}</span><small aria-hidden="true">/{maximum}</small>
          </strong>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _finding_card(
    finding: Finding,
    *,
    diagnostic: bool = False,
    occurrences: int = 1,
) -> None:
    tone = "neutral" if diagnostic else _tone_for_points(finding.risk_points, "completed")
    location = "Document"
    if finding.page is not None:
        location = f"Page {finding.page}"
        if finding.bbox is not None:
            location += " - zone localisee"
    if occurrences > 1:
        location = f"{occurrences} zones détectées"
    gapl_index = finding.evidence.get("global_index")
    if gapl_index is not None:
        confidence_label = f"Ressemblance estimée : {float(gapl_index):.0%}"
    elif finding.detector == "ocr_content":
        confidence_label = f"Fiabilité de la lecture : {finding.confidence:.0%}"
    else:
        confidence_label = f"Confiance : {finding.confidence:.0%}"
    title, description = _business_finding_copy(finding)
    st.markdown(
        f"""
        <article class="finding-card {tone}">
          <div class="finding-score">
            <strong>{finding.risk_points:g}</strong>
            <span>points</span>
          </div>
          <div class="finding-content">
            <div class="finding-kicker">
              <span>{_html(CATEGORY_LABELS.get(finding.category, finding.category))}</span>
              <span>{_html(location)}</span>
            </div>
            <h3>{_html(title)}</h3>
            <p>{_html(description)}</p>
            <div class="confidence">{_html(confidence_label)}</div>
          </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _existing_artifacts(output_dir: Path, relatives: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for relative in relatives:
        path = output_dir / relative
        if path.is_file():
            paths.append(path)
    return paths


def _tone_for_points(points: float, status: str) -> str:
    if points >= 45:
        return "danger"
    if points >= 25:
        return "warning"
    if points > 0:
        return "notice-tone"
    if status == "partial":
        return "partial"
    return "neutral"


def _is_pdf_bytes(file_bytes: bytes) -> bool:
    return b"%PDF-" in file_bytes[:1024]


def _safe_filename(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "-", Path(name).name).strip("-")
    return clean or "uploaded-file"


def _release_transient_memory() -> None:
    gc.collect()
    try:
        import ctypes

        malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _friendly_artifact_caption(path: Path) -> str:
    name = path.name
    if "gapl" in name or "windows" in name:
        return "Carte de ressemblance avec une image générée par IA"
    if "revision-diff" in name:
        return "Différences visuelles entre les versions"
    if "ela" in name:
        return "Carte des variations de compression"
    if "provenance" in name:
        return "Origine et métadonnées de l'image"
    if "analysis" in name:
        return "Analyse visuelle détaillée"
    return "Visualisation complémentaire"


def _laboratory_artifact_caption(path: Path) -> str:
    if "trufor-localization" in path.name:
        return "Carte des incohérences locales"
    if "revision" in path.name:
        return "Différences entre les versions"
    if "repeated-visual-region" in path.name:
        return "Comparaison des éléments visuels répétés"
    return "Visualisation complémentaire"


def _is_secondary_localization_artifact(path: Path) -> bool:
    return any(marker in path.name for marker in ("trufor-reliable", "trufor-confidence"))


def _business_finding_copy(finding: Finding) -> tuple[str, str]:
    if finding.code == "AI_GAPL_GLOBAL_TRACE":
        index = float(finding.evidence.get("global_index", 0.0))
        if index >= 0.9:
            title = "Forte ressemblance avec une image générée par IA"
        elif index >= 0.5:
            title = "Ressemblance partielle avec une image générée par IA"
        else:
            title = "Faible ressemblance avec une image générée par IA"
        return (
            title,
            "Plusieurs zones présentent des caractéristiques souvent observées dans des "
            "images générées par IA. La compression, le redimensionnement ou certains motifs "
            "visuels peuvent produire un résultat similaire : ce signal ne prouve pas une fraude.",
        )
    if finding.detector == "ocr_content":
        return (
            "Cohérence du contenu à vérifier",
            _business_text(finding.description),
        )
    return _sentence_case(finding.title), _business_text(finding.description)


def _format_identifier_value(code: str, value: str) -> str:
    if " / " in value:
        return " / ".join(_format_identifier_value(code, item) for item in value.split(" / "))
    compact = re.sub(r"[\s-]", "", value)
    if code.startswith(("OCR_CARD_", "OCR_IBAN_")) and compact.isalnum():
        return " ".join(compact[index : index + 4] for index in range(0, len(compact), 4))
    return value


def _business_observation_title(observation: Any) -> str:
    if str(observation.code).startswith("TRUFOR_"):
        if observation.code == "TRUFOR_UNAVAILABLE":
            return "Analyse des retouches locales indisponible"
        return _sentence_case(observation.title)
    return _sentence_case(observation.title)


def _business_check_title(code: str, title: str) -> str:
    labels = {
        "trufor": "Recherche de retouches locales",
        "pades": "Signature électronique du PDF",
        "facturx": "Facture électronique embarquée",
        "two_d_doc": "Code de vérification 2D-Doc",
        "post_signature": "Modifications après signature",
        "fonts_hidden_objects": "Polices et éléments masqués",
        "all_revisions": "Historique complet des versions",
        "ocr_quality": "Qualité du texte reconnu",
        "ocr_identifiers": "Validité des identifiants",
        "ocr_dates": "Cohérence des dates",
        "ocr_financial_consistency": "Cohérence des montants",
        "ocr_visual_repetition": "Éléments visuels répétés entre les pages",
    }
    return labels.get(code, _sentence_case(title))


def _sentence_case(value: object) -> str:
    text = _business_text(value).strip()
    return text[:1].upper() + text[1:] if text else text


def _business_text(value: object) -> str:
    text = str(value)
    replacements = (
        ("Indice TruFor", "Indice de retouche locale"),
        ("Analyse TruFor", "Analyse des retouches locales"),
        ("TruFor", "le détecteur de retouches locales"),
        ("GAPL", "le détecteur d'images générées par IA"),
        ("score Frod", "score global"),
        ("Score Frod", "Score global"),
        ("Frod", "l'analyse"),
        ("Coherence", "Cohérence"),
        ("coherence", "cohérence"),
        ("Incoherence", "Incohérence"),
        ("incoherence", "incohérence"),
        ("Fiabilite", "Fiabilité"),
        ("fiabilite", "fiabilité"),
        (" a verifier", " à vérifier"),
        (" a ete ", " a été "),
        ("generee", "générée"),
        ("generees", "générées"),
        ("detectee", "détectée"),
        ("detectees", "détectées"),
        ("controle", "contrôle"),
        ("geographique", "géographique"),
        ("Numero", "Numéro"),
        ("numero", "numéro"),
        ("necessaire", "nécessaire"),
        ("donnees", "données"),
        ("metier", "métier"),
        ("perimetre", "périmètre"),
        ("authenticite", "authenticité"),
        ("proprietes", "propriétés"),
        ("elements", "éléments"),
        ("ajoutes", "ajoutés"),
        ("enregistres", "enregistrés"),
        ("revisions", "révisions"),
        ("revision", "révision"),
        ("Difference", "Différence"),
        ("difference", "différence"),
        ("derniere", "dernière"),
        ("localisee", "localisée"),
        ("superposee", "superposée"),
        ("emetteur", "émetteur"),
        ("attribue", "attribué"),
        ("caracteres", "caractères"),
        ("legitimement", "légitimement"),
        ("meme", "même"),
        ("presence", "présence"),
    )
    for technical, business in replacements:
        text = text.replace(technical, business)
    return text


def _html(value: object) -> str:
    return (
        _business_text(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #f5f3ef;
          --muted: #aaa6a0;
          --line: #36343a;
          --panel: #1b1a1e;
          --panel-soft: #242329;
          --bg: #111114;
          --green: #35d07f;
          --amber: #f4b942;
          --coral: #ff6b6b;
          --cyan: #4cc9d8;
          --blue: #70a7ff;
        }
        @property --animated-score {
          syntax: "<integer>";
          initial-value: 0;
          inherits: false;
        }
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
          display: none;
        }
        .stApp {
          background: var(--bg);
          color: var(--ink);
        }
        .block-container {
          max-width: 1520px;
          padding: 1.1rem 1.5rem 2.5rem;
        }
        .st-key-app_header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: nowrap;
          gap: 1rem;
          border-bottom: 1px solid var(--line);
          padding: 0 0 .8rem;
          margin-bottom: -.65rem;
        }
        .st-key-header_actions {
          display: flex;
          align-items: center;
          gap: .5rem;
          width: max-content;
          margin-left: auto;
          flex: 0 0 auto;
        }
        .st-key-header_navigation {
          width: max-content;
          flex: 0 0 auto;
        }
        .brandbar {
          display: flex;
          align-items: center;
          padding: 0;
          margin: 0;
        }
        .brandbar h1 {
          color: var(--ink);
          font-size: 1.5rem;
          font-weight: 950;
          letter-spacing: .16em;
          line-height: 1;
          margin: 0;
        }
        .eyebrow {
          color: var(--cyan);
          font-size: .72rem;
          letter-spacing: 0;
          margin: 0 0 .3rem;
        }
        div[data-testid="stFileUploader"] {
          background: #19191d;
          border: 2px dashed #595760;
          border-radius: 8px;
          padding: .5rem .75rem;
        }
        div[data-testid="stFileUploader"]:hover {
          border-color: var(--cyan);
        }
        div[data-testid="stFileUploader"] label {
          color: var(--ink);
          font-weight: 800;
        }
        .stButton > button {
          min-height: 3rem;
          border-radius: 6px;
          font-weight: 850;
          background: #f5f3ef;
          color: #17171a;
          border: 0;
        }
        .stButton > button:hover {
          background: var(--cyan);
          color: #101316;
        }
        [class*="st-key-header_reset_document_"] {
          flex: 0 0 auto;
          margin-left: 0;
        }
        [class*="st-key-header_reset_document_"] button {
          min-height: 44px;
          padding: .45rem 1rem;
          border: 1px solid #765f35;
          background: #211d16;
          color: #f3cf8a;
          white-space: nowrap;
        }
        [class*="st-key-header_reset_document_"] button:hover {
          border-color: var(--amber);
          background: #302719;
          color: #ffe0a3;
        }
        [class*="st-key-header_reset_document_"] button:focus-visible {
          outline: 2px solid var(--amber);
          outline-offset: 2px;
        }
        .analysis-progress {
          margin: .4rem 0 1rem;
          border: 1px solid var(--line);
          border-radius: 6px;
          background: #19191d;
          padding: .7rem .9rem;
        }
        .analysis-progress-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          color: #d7d4ce;
          font-size: .78rem;
          margin-bottom: .45rem;
        }
        .analysis-progress-head strong {
          color: var(--ink);
        }
        .analysis-progress-track {
          height: 7px;
          background: #323137;
          border-radius: 999px;
          overflow: hidden;
        }
        .analysis-progress-track i {
          display: block;
          height: 100%;
          background: var(--cyan);
          border-radius: inherit;
          transition: width .18s ease;
        }
        .pdf-loading-stage {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: .65rem;
          min-height: 64px;
          max-height: 64px;
          margin: .35rem 0 .55rem;
          overflow: hidden;
          color: #d8d5cf;
          font-size: .78rem;
          font-weight: 750;
          animation: pdf-loading-transition .46s cubic-bezier(.22, .75, .2, 1) forwards;
        }
        .pdf-loading-stage i {
          width: 18px;
          height: 18px;
          border: 2px solid #3a3a42;
          border-top-color: var(--cyan);
          border-radius: 50%;
          animation: pdf-loading-spin .42s linear infinite;
        }
        [class*="st-key-analysis_results_entering"] {
          --result-entry-delay: .24s;
          animation: analysis-results-enter .24s ease-out var(--result-entry-delay) both;
        }
        [class*="st-key-analysis_results_ready"] {
          --result-entry-delay: 0s;
        }
        .empty-state {
          min-height: 110px;
          border: 1px solid var(--line);
          border-left: 5px solid #595760;
          background: #18181c;
          border-radius: 6px;
          padding: 1.2rem;
          align-content: center;
        }
        .empty-state h2 {
          font-size: 1.15rem;
          margin: 0 0 .3rem;
        }
        .empty-state p {
          margin: 0;
          font-size: .86rem;
        }
        .fraud-score {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          min-height: 58px;
          padding: .55rem .8rem;
          margin: .15rem 0 .55rem;
          background: #19191d;
          border: 1px solid var(--line);
          border-left: 4px solid #595760;
          border-radius: 6px;
        }
        .fraud-score.low { border-left-color: var(--green); }
        .fraud-score.review { border-left-color: var(--amber); }
        .fraud-score.high { border-left-color: var(--coral); }
        .fraud-score > span {
          color: #aaa6a0;
          font-size: .72rem;
          font-weight: 800;
          letter-spacing: .02em;
          text-transform: uppercase;
        }
        .fraud-score strong {
          display: inline-flex;
          align-items: baseline;
          color: var(--score-color);
          font-size: 1.85rem;
          font-weight: 900;
          line-height: 1;
        }
        .fraud-score-number {
          --animated-score: 0;
          counter-reset: fraud-score var(--animated-score);
          animation: var(--score-animation) var(--score-duration) steps(1, end)
            var(--result-entry-delay, 0s) forwards;
        }
        .fraud-score-number::after {
          content: counter(fraud-score);
        }
        .fraud-score small {
          color: var(--muted);
          font-size: .65rem;
          font-weight: 700;
          margin-left: .12rem;
        }
        .st-key-app_header [data-testid="stButtonGroup"] {
          flex: 0 0 auto;
        }
        .st-key-app_header [data-testid="stButtonGroup"] button {
          min-height: 44px;
          padding: .45rem 1rem;
          border-radius: 6px;
          color: #96939c;
          font-size: .78rem;
          font-weight: 800;
        }
        .st-key-app_header [data-testid="stButtonGroup"] button[aria-checked="true"] {
          color: var(--ink);
          background: #27272d;
          box-shadow: inset 0 -2px 0 var(--cyan);
        }
        .st-key-app_header [data-testid="stButtonGroup"] button:focus-visible {
          outline: 2px solid var(--cyan);
          outline-offset: 2px;
        }
        .sr-only {
          position: absolute;
          width: 1px;
          height: 1px;
          padding: 0;
          margin: -1px;
          overflow: hidden;
          clip: rect(0, 0, 0, 0);
          white-space: nowrap;
          border: 0;
        }
        .indicator-sequence {
          margin: .7rem 0 1rem;
          padding: .9rem;
          border: 1px solid #313139;
          border-radius: 9px;
          background:
            radial-gradient(circle at 100% 0, rgba(34, 211, 238, .07), transparent 32%),
            #141419;
        }
        .indicator-queue {
          display: grid;
          gap: .34rem;
          padding: 0;
          margin: 0;
          list-style: none;
        }
        .indicator-step {
          --risk-color: #88858e;
          --step-bg: #1b1b20;
          --step-border: #3b3942;
          display: grid;
          grid-template-columns: 32px minmax(0, 1fr) 58px;
          gap: .7rem;
          align-items: center;
          min-width: 0;
          min-height: 54px;
          padding: .55rem .72rem;
          border: 1px solid #3b3942;
          border-radius: 6px;
          background: #1b1b20;
          animation: indicator-step-validate .42s ease-out var(--step-delay) both;
        }
        .indicator-queue > .indicator-step {
          margin: 0;
          padding: .55rem .85rem;
        }
        .indicator-step.notice-tone {
          --risk-color: var(--blue);
          --step-bg: #171d25;
          --step-border: #31526d;
        }
        .indicator-step.warning {
          --risk-color: var(--amber);
          --step-bg: #242015;
          --step-border: #685727;
        }
        .indicator-step.danger {
          --risk-color: var(--coral);
          --step-bg: #27181c;
          --step-border: #733543;
        }
        .indicator-step.clear {
          --risk-color: var(--green);
          --step-bg: #142019;
          --step-border: #24543a;
        }
        .indicator-step.partial {
          --risk-color: var(--amber);
          --step-bg: #211e17;
          --step-border: #625329;
        }
        .indicator-step.unavailable {
          --risk-color: #88848e;
          --step-bg: #1b1b20;
          --step-border: #45434b;
        }
        .indicator-index {
          display: grid;
          width: 28px;
          height: 28px;
          place-content: center;
          border: 1px solid #47454e;
          border-radius: 50%;
          color: #7e7b84;
          font: 800 .62rem/1 ui-monospace, SFMono-Regular, Consolas, monospace;
          animation: indicator-accent-reveal .25s ease-out var(--step-delay) forwards;
        }
        .indicator-core {
          display: grid;
          gap: .38rem;
          min-width: 0;
        }
        .indicator-main {
          display: grid;
          grid-template-columns: minmax(0, 1fr) auto;
          align-items: center;
          gap: .75rem;
          min-width: 0;
        }
        .indicator-main h3 {
          min-width: 0;
          margin: 0;
          color: #ddd9d3;
          font-size: .78rem;
          line-height: 1.2;
        }
        .indicator-pending,
        .indicator-final {
          grid-area: 1 / 2;
          display: inline-flex;
          align-items: center;
          justify-content: flex-end;
          gap: .32rem;
          flex: 0 0 auto;
          font-size: .64rem;
          font-weight: 800;
          white-space: nowrap;
        }
        .indicator-pending {
          color: #88858e;
          animation: indicator-pending-hide .06s linear var(--step-delay) forwards;
        }
        .indicator-pending b {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: currentColor;
          box-shadow: 0 0 0 0 rgba(142, 139, 148, .35);
          animation: indicator-pending-pulse .8s ease-out infinite;
        }
        .indicator-final {
          color: var(--risk-color);
          opacity: 0;
          transform: translateY(4px);
          animation: indicator-final-reveal .24s ease-out var(--step-delay) forwards;
        }
        .indicator-final b {
          display: grid;
          width: 18px;
          height: 18px;
          place-content: center;
          border: 1px solid currentColor;
          border-radius: 50%;
          font-size: .62rem;
        }
        .indicator-meter {
          height: 4px;
          overflow: hidden;
          border-radius: 999px;
          background: #333238;
        }
        .indicator-meter i {
          display: block;
          width: var(--risk-value);
          height: 100%;
          border-radius: inherit;
          background: var(--risk-color);
          transform: scaleX(0);
          transform-origin: left center;
          animation: indicator-meter-fill .34s ease-out var(--step-delay) forwards;
        }
        .indicator-step.partial .indicator-meter i {
          background: repeating-linear-gradient(
            135deg,
            var(--amber) 0 6px,
            #7b601d 6px 12px
          );
        }
        .indicator-score {
          width: 58px;
          min-width: 0;
          box-sizing: border-box;
          justify-self: end;
          padding-right: .2rem;
          color: #d8d4ce;
          font: 850 .75rem/1 ui-monospace, SFMono-Regular, Consolas, monospace;
          font-variant-numeric: tabular-nums;
          text-align: right;
          opacity: .35;
          animation: indicator-score-reveal .2s ease-out var(--step-delay) forwards;
        }
        .indicator-score span {
          color: #77747d;
          font-size: .6rem;
        }
        .indicator-glossary {
          padding: 1rem 0;
        }
        .indicator-glossary h2 {
          margin: 0 0 .8rem;
          font-size: 1.15rem;
        }
        .glossary-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: .5rem;
        }
        .glossary-card {
          min-width: 0;
          padding: .8rem;
          border: 1px solid var(--line);
          border-left: 3px solid #4a6d78;
          border-radius: 6px;
          background: #19191d;
        }
        .glossary-card > span {
          color: var(--ink);
          font-size: .8rem;
          font-weight: 850;
        }
        .glossary-card p {
          margin: .28rem 0 .55rem;
          color: #aaa69f;
          font-size: .72rem;
          line-height: 1.4;
        }
        .glossary-card small {
          color: #77747d;
          font: 700 .62rem/1 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        @keyframes indicator-step-validate {
          0% { background: #1b1b20; border-color: #3b3942; box-shadow: none; }
          45% {
            border-color: var(--cyan);
            box-shadow:
              0 0 0 1px rgba(34, 211, 238, .25),
              0 0 16px rgba(34, 211, 238, .14);
          }
          100% { background: var(--step-bg); border-color: var(--step-border); box-shadow: none; }
        }
        @keyframes indicator-pending-hide {
          from { opacity: 1; visibility: visible; }
          to { opacity: 0; visibility: hidden; }
        }
        @keyframes indicator-pending-pulse {
          70% { box-shadow: 0 0 0 5px rgba(142, 139, 148, 0); }
          100% { box-shadow: 0 0 0 0 rgba(142, 139, 148, 0); }
        }
        @keyframes indicator-final-reveal {
          from { opacity: 0; transform: translateY(4px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes indicator-meter-fill {
          from { transform: scaleX(0); }
          to { transform: scaleX(1); }
        }
        @keyframes indicator-accent-reveal {
          to { color: var(--risk-color); border-color: var(--risk-color); }
        }
        @keyframes indicator-score-reveal {
          to { opacity: 1; }
        }
        @keyframes pdf-loading-spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pdf-loading-transition {
          0% { opacity: 0; transform: translateY(2px); }
          14%, 48% { opacity: 1; transform: translateY(0); }
          100% {
            min-height: 0;
            max-height: 0;
            margin: 0;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-2px);
          }
        }
        @keyframes analysis-results-enter {
          from { opacity: 0; transform: translateY(4px); }
          to { opacity: 1; transform: translateY(0); }
        }
        div[data-testid="stMarkdownContainer"] h2.workspace-title {
          font-size: 1.02rem;
          line-height: 1.3;
          margin: .15rem 0 .55rem;
          padding: 0;
        }
        div[data-testid="stMarkdownContainer"] h2.document-name {
          overflow-wrap: anywhere;
        }
        div[data-testid="stMarkdownContainer"] h3.subsection-title {
          font-size: .88rem;
          line-height: 1.3;
          color: #dedbd5;
          margin: .9rem 0 .45rem;
          padding: 0 0 .32rem;
          border-bottom: 1px solid var(--line);
        }
        [data-testid="stImage"] img {
          width: 100%;
          max-height: 760px;
          object-fit: contain;
          background: #0b0b0d;
          border: 1px solid var(--line);
          border-radius: 4px;
        }
        .review-banner {
          display: flex;
          justify-content: space-between;
          gap: .7rem;
          align-items: center;
          padding: .7rem .8rem;
          border: 1px solid var(--line);
          border-left: 5px solid #595760;
          border-radius: 6px;
          background: #1a191d;
          margin-bottom: .55rem;
        }
        .review-banner.attention { border-left-color: var(--coral); }
        .review-banner.clear { border-left-color: var(--green); }
        .review-banner strong {
          font-size: .9rem;
        }
        .review-banner span {
          color: var(--muted);
          font-size: .76rem;
          text-align: right;
        }
        .finding-card {
          display: grid;
          grid-template-columns: 54px minmax(0, 1fr);
          gap: .7rem;
          padding: .7rem;
          margin-bottom: .5rem;
          border: 1px solid var(--line);
          border-left: 5px solid var(--blue);
          border-radius: 6px;
          background: #1b1a1e;
        }
        .finding-card.danger { border-left-color: var(--coral); background: #25191c; }
        .finding-card.warning { border-left-color: var(--amber); background: #252117; }
        .finding-card.notice-tone { border-left-color: var(--blue); background: #181e28; }
        .finding-score {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          width: 48px;
          height: 48px;
          border-radius: 4px;
          background: #0e0e11;
        }
        .finding-score strong {
          display: block;
          font-size: 1.15rem;
          line-height: 1;
        }
        .finding-score span {
          display: block;
          font-size: .58rem;
          line-height: 1;
          margin-top: .2rem;
        }
        .finding-content h3 {
          font-size: .94rem;
          margin: 0 0 .2rem;
        }
        .finding-content p {
          color: #c7c3bd;
          font-size: .78rem;
          line-height: 1.4;
          margin: 0;
        }
        .finding-kicker {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: .25rem .55rem;
          margin-bottom: .25rem;
        }
        .finding-kicker span {
          border: 0;
          border-radius: 0;
          padding: 0;
          background: transparent;
          color: var(--muted);
          font-size: .64rem;
        }
        .confidence {
          margin-top: .3rem;
          color: var(--cyan);
          font-size: .72rem;
        }
        .diagnostic-count {
          color: var(--muted);
          font-size: .72rem;
          margin: .25rem 0;
        }
        .extracted-fields {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: .45rem;
        }
        .extracted-field {
          padding: .6rem;
          border: 1px solid var(--line);
          border-top: 4px solid #595760;
          border-radius: 6px;
          background: #1a191d;
          min-width: 0;
        }
        .extracted-field.clear { border-top-color: var(--green); }
        .extracted-field.attention { border-top-color: var(--coral); }
        .extracted-field > div {
          min-width: 0;
        }
        .extracted-field span,
        .extracted-field strong {
          display: block;
        }
        .extracted-field span {
          color: var(--muted);
          font-size: .65rem;
        }
        .extracted-field strong {
          margin-top: .15rem;
          font-size: .82rem;
          overflow-wrap: anywhere;
        }
        .extracted-field b {
          display: inline-block;
          margin-top: .4rem;
          color: #e7e3dc;
          font-size: .65rem;
        }
        .extracted-field p {
          margin: .22rem 0 0;
          color: #bdb9b2;
          font-size: .68rem;
          line-height: 1.35;
        }
        .business-observation {
          padding: .62rem .7rem;
          margin-bottom: .4rem;
          border: 1px solid var(--line);
          border-left: 5px solid var(--coral);
          border-radius: 6px;
          background: #24191c;
        }
        .business-observation > div {
          display: flex;
          justify-content: space-between;
          gap: .6rem;
        }
        .business-observation strong {
          font-size: .82rem;
        }
        .business-observation span,
        .business-observation small {
          color: var(--muted);
          font-size: .65rem;
        }
        .business-observation p {
          margin: .25rem 0;
          font-size: .75rem;
        }
        .control-matrix {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: .4rem;
        }
        .control-status {
          min-height: 78px;
          padding: .55rem;
          border: 1px solid var(--line);
          border-left: 4px solid #595760;
          border-radius: 5px;
          background: #19191d;
        }
        .control-status.clear { border-left-color: var(--green); }
        .control-status.attention { border-left-color: var(--coral); }
        .control-status.detected { border-left-color: var(--blue); }
        .control-status.indeterminate { border-left-color: var(--amber); }
        .control-status.error { border-left-color: #b28cff; }
        .control-status span,
        .control-status strong {
          display: block;
        }
        .control-status span {
          color: var(--muted);
          font-size: .62rem;
        }
        .control-status strong {
          margin-top: .15rem;
          font-size: .74rem;
        }
        .control-status p {
          margin: .25rem 0 0;
          color: #bcb8b1;
          font-size: .64rem;
          line-height: 1.3;
        }
        .recognized-text {
          white-space: pre-wrap;
          max-height: 240px;
          overflow: auto;
          padding: .75rem;
          border: 1px solid var(--line);
          border-radius: 6px;
          background: #18181c;
          color: #d3cfc8;
          font: .74rem/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        div[data-testid="stMarkdownContainer"] h2.section-title {
          font-size: 1rem;
          line-height: 1.3;
          margin: 1rem 0 .45rem;
          padding: 0;
        }
        .extraction-empty,
        .extraction-coverage-warning {
          display: flex;
          flex-direction: column;
          gap: .25rem;
          padding: .8rem;
          border: 1px solid var(--line);
          border-left: 5px solid var(--muted);
          border-radius: 6px;
          background: #19191d;
        }
        .extraction-empty span,
        .extraction-coverage-warning span {
          color: var(--muted);
          font-size: .75rem;
        }
        .extraction-coverage-warning {
          margin-top: .8rem;
          border-left-color: var(--amber);
          background: #211d16;
        }
        .verification-summary {
          display: flex;
          flex-direction: column;
          gap: .22rem;
          margin-top: .85rem;
          padding: .75rem .8rem;
          border: 1px solid #28583d;
          border-left: 5px solid var(--green);
          border-radius: 6px;
          background: #142019;
        }
        .verification-summary.attention,
        .verification-summary.incomplete {
          border-color: #665527;
          border-left-color: var(--amber);
          background: #211d16;
        }
        .verification-summary.unavailable {
          border-color: var(--line);
          border-left-color: #77747d;
          background: #19191d;
        }
        .verification-summary span {
          color: var(--muted);
          font-size: .7rem;
        }
        .verification-issues {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: .45rem;
          margin-top: .45rem;
        }
        .verification-issue {
          display: flex;
          flex-direction: column;
          gap: .24rem;
          min-width: 0;
          padding: .65rem;
          border: 1px solid #69424a;
          border-left: 4px solid var(--coral);
          border-radius: 5px;
          background: #26191d;
        }
        .verification-issue.omission {
          border-color: #665527;
          border-left-color: var(--amber);
          background: #211d16;
        }
        .verification-issue span,
        .verification-issue small {
          color: var(--muted);
          font-size: .62rem;
        }
        .verification-issue strong {
          font-size: .75rem;
          overflow-wrap: anywhere;
        }
        .extraction-overview {
          display: grid;
          grid-template-columns: minmax(240px, 1.5fr) repeat(3, minmax(130px, 1fr));
          gap: .55rem;
          margin-bottom: .85rem;
        }
        .extraction-overview > article {
          min-height: 105px;
          padding: .75rem;
          border: 1px solid var(--line);
          border-radius: 6px;
          background: #19191d;
        }
        .extraction-overview > article > strong,
        .extraction-coverage div strong {
          display: block;
          color: var(--ink);
          font-size: 1.45rem;
          line-height: 1;
        }
        .extraction-overview > article > span,
        .extraction-coverage div span,
        .extraction-coverage small {
          display: block;
          margin-top: .35rem;
          color: var(--muted);
          font-size: .68rem;
        }
        .extraction-coverage {
          border-left: 5px solid var(--amber) !important;
        }
        .extraction-coverage.clear {
          border-left-color: var(--green) !important;
        }
        .extraction-coverage i {
          display: block;
          height: 7px;
          margin-top: .55rem;
          overflow: hidden;
          border-radius: 4px;
          background: #2c2c32;
        }
        .extraction-coverage i b {
          display: block;
          height: 100%;
          background: var(--amber);
        }
        .extraction-coverage.clear i b { background: var(--green); }
        .extraction-table-wrap {
          width: 100%;
          max-height: 420px;
          overflow: auto;
          border: 1px solid var(--line);
          border-radius: 6px;
          background: #17171b;
        }
        .extraction-table {
          width: 100%;
          border-collapse: collapse;
          font-size: .72rem;
        }
        .extraction-table th,
        .extraction-table td {
          padding: .55rem .6rem;
          border-bottom: 1px solid var(--line);
          text-align: left;
          vertical-align: top;
        }
        .extraction-table th {
          position: sticky;
          top: 0;
          z-index: 1;
          color: var(--muted);
          background: #202025;
          font-size: .62rem;
          text-transform: uppercase;
        }
        .extraction-table th span {
          display: block;
          margin-top: .16rem;
          color: var(--cyan);
          font-size: .56rem;
          text-transform: none;
        }
        .extraction-table td strong,
        .extraction-table td span {
          display: block;
        }
        .extraction-table td span {
          margin-top: .15rem;
          color: var(--muted);
          font-size: .62rem;
        }
        .extraction-table td .extraction-row-role {
          display: inline-flex;
          margin: 0;
          padding: .2rem .36rem;
          border: 1px solid #3c5965;
          border-radius: 4px;
          color: #bdebf3;
          background: #17252b;
          font-size: .58rem;
          font-weight: 800;
          white-space: nowrap;
        }
        .additional-extractions {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .45rem;
        }
        .additional-extraction {
          min-width: 0;
          padding: .65rem;
          border: 1px solid var(--line);
          border-left: 4px solid var(--blue);
          border-radius: 5px;
          background: #181c24;
        }
        .additional-extraction span,
        .additional-extraction strong,
        .additional-extraction small {
          display: block;
        }
        .additional-extraction span,
        .additional-extraction small {
          color: var(--muted);
          font-size: .64rem;
        }
        .additional-extraction strong {
          margin: .2rem 0;
          overflow-wrap: anywhere;
          font-size: .78rem;
        }
        @media (max-width: 700px) {
          .block-container {
            padding: .8rem .75rem 1.5rem;
          }
          .brandbar {
            align-items: flex-start;
          }
          .st-key-app_header {
            flex-wrap: wrap;
            padding-bottom: .55rem;
            margin-bottom: 0;
          }
          .st-key-header_actions {
            max-width: 100%;
          }
          .st-key-app_header [data-testid="stButtonGroup"] button {
            min-height: 40px;
            padding: .35rem .65rem;
          }
          [class*="st-key-header_reset_document_"] {
            flex: 0 0 auto;
            display: flex;
            justify-content: flex-end;
          }
          [class*="st-key-header_reset_document_"] .stButton {
            width: auto;
          }
          [class*="st-key-header_reset_document_"] button {
            min-height: 44px;
            width: auto;
          }
          .extracted-fields,
          .control-matrix,
          .additional-extractions,
          .extraction-overview {
            grid-template-columns: 1fr;
          }
          .indicator-sequence {
            padding: .75rem;
          }
          .indicator-step {
            gap: .5rem;
            padding: .55rem .6rem;
          }
          .indicator-main h3 {
            font-size: .74rem;
          }
          .glossary-grid {
            grid-template-columns: 1fr;
          }
          .review-banner {
            align-items: flex-start;
            flex-direction: column;
          }
          .review-banner span {
            text-align: left;
          }
        }
        @media (max-width: 520px) {
          .indicator-step {
            grid-template-columns: 28px minmax(0, 1fr) 52px;
            gap: .4rem;
          }
          .indicator-score {
            width: 52px;
            padding-right: .1rem;
          }
          .indicator-pending,
          .indicator-final {
            font-size: .58rem;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          .pdf-loading-stage {
            display: none;
            animation: none;
          }
          [class*="st-key-analysis_results_entering"] {
            --result-entry-delay: 0s;
            animation: none;
          }
          .fraud-score-number {
            --animated-score: var(--score-target);
            animation: none;
          }
          .indicator-step {
            background: var(--step-bg);
            border-color: var(--step-border);
            animation: none;
          }
          .indicator-pending {
            display: none;
            animation: none;
          }
          .indicator-final {
            opacity: 1;
            transform: none;
            animation: none;
          }
          .indicator-index {
            color: var(--risk-color);
            border-color: var(--risk-color);
            animation: none;
          }
          .indicator-meter i {
            transform: scaleX(1);
            animation: none;
          }
          .indicator-score {
            opacity: 1;
            animation: none;
          }
        }
        /* Classification card styles - matching finding-card style */
        .classification-card {
          padding: 0.85rem 0.9rem;
          margin-bottom: 0.5rem;
          border: 1px solid var(--line);
          border-left: 5px solid var(--blue);
          border-radius: 6px;
          background: #181e28;
        }
        .classification-card.reliable { border-left-color: var(--blue); background: #181e28; }
        .classification-card.attention { border-left-color: var(--amber); background: #1d1a17; }
        .classification-card.uncertain,
        .classification-card.undetermined { border-left-color: var(--muted); background: #1b1a1e; }
        .classification-content {
          display: flex;
          flex-direction: column;
          gap: 0.45rem;
        }
        .classification-category {
          color: var(--ink);
          font-size: 1rem;
          font-weight: 800;
        }
        .classification-metadata {
          display: flex;
          align-items: center;
          flex-wrap: wrap;
          gap: 0.45rem 1rem;
          color: var(--muted);
          font-size: 0.72rem;
        }
        .classification-metadata strong {
          color: var(--ink);
          margin-left: 0.2rem;
        }
        .classification-evidence-title {
          color: var(--muted);
          font-size: 0.68rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          margin: 0.5rem 0 0.3rem;
        }
        .classification-evidence {
          color: #d8d5cf;
          font-size: 0.78rem;
          line-height: 1.4;
          margin: 0.25rem 0;
          padding: 0.35rem 0.55rem;
          border-left: 2px solid var(--blue);
          background: #17171b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
