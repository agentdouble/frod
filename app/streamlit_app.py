"""Local Streamlit interface for documentary fraud analysis reports."""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from fraude_detector.analysis_service import AnalysisRunHandle, AnalysisService
from fraude_detector.errors import AnalysisError
from fraude_detector.models import (
    AnalysisReport,
    AnalysisSynthesis,
    DetectorResult,
    DocumentClassification,
    DocumentExtraction,
    ExtractionVerification,
    Finding,
    ImageAnalysisReport,
    LaboratoryReport,
)
from fraude_detector.run_config import RunConfig, load_run_config
from fraude_detector.scoring import FAMILY_CAPS, assess_risk

CONFIG_PATH = Path(os.environ.get("FROD_CONFIG", "config.yaml"))
PROJECT_CONFIG = load_run_config(CONFIG_PATH)
WORK_DIR = PROJECT_CONFIG.application.work_dir
CONFIG_FINGERPRINT = hashlib.sha256(repr(PROJECT_CONFIG).encode("utf-8")).hexdigest()[:12]
ANALYSIS_POLICY_VERSION = (
    f"gapl-p25-90-v2-ocr-content-v3-structured-identifiers-{CONFIG_FINGERPRINT}"
)
INDICATOR_STEP_SECONDS = 0.45
AI_PENDING_VIEW = "Analyse IA · en cours"
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
        "label": "Risque modéré",
        "tone": "low",
        "color": "#35d07f",
        "background": "#062d24",
        "border": "#10b981",
    },
    "review": {
        "label": "Risque important",
        "tone": "review",
        "color": "#f4b942",
        "background": "#3b2604",
        "border": "#f59e0b",
    },
    "high": {
        "label": "Risque critique",
        "tone": "high",
        "color": "#ff6b6b",
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

DOCUMENT_FAMILY_LABELS = {
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

CATEGORY_DETECTORS = {
    "annotations": frozenset({"page_composition"}),
    "content_consistency": frozenset({"ocr_content"}),
    "document_integrity": frozenset({"pdf_structure", "document_authenticity"}),
    "metadata": frozenset({"pdf_structure"}),
    "page_composition": frozenset({"page_composition"}),
    "provenance_integrity": frozenset(
        {"image_provenance", "document_authenticity"}
    ),
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


def _analysis_is_pending(state: object) -> bool:
    return isinstance(state, dict) and state.get("status") in {"core_running", "ai_running"}


def _cancel_analysis(state: object) -> None:
    if not isinstance(state, dict):
        return
    handle = state.get("ai_handle")
    if isinstance(handle, AnalysisRunHandle):
        handle.cancel()


@st.fragment(run_every=0.5)
def _poll_ai_analysis(current_key: tuple[str, str, str]) -> None:
    state = st.session_state.get("analysis")
    if (
        not isinstance(state, dict)
        or state.get("key") != current_key
        or state.get("status") != "ai_running"
    ):
        return

    handle = state.get("ai_handle")
    if isinstance(handle, AnalysisRunHandle):
        for event in handle.drain_progress():
            state["progress"] = max(float(state.get("progress", 0.0)), event.value)
            state["progress_label"] = event.label

    if isinstance(handle, AnalysisRunHandle) and handle.done():
        try:
            result = handle.result()
        except Exception as error:
            state.update(
                status="ready",
                laboratory=_empty_laboratory_report(),
                synthesis=None,
                synthesis_error=(
                    "L'analyse IA n'a pas pu être terminée : "
                    f"{type(error).__name__}: {str(error)[:180]}"
                ),
            )
        else:
            state.update(
                status="ready",
                report=result.report,
                laboratory=result.laboratory,
                synthesis=result.synthesis,
                synthesis_error=result.synthesis_error,
                progress=1.0,
                progress_label="Analyse IA terminée",
            )
        state.pop("ai_handle", None)
        st.session_state["analysis"] = state
        st.rerun()

    _render_analysis_progress(
        st,
        0.5 + 0.5 * float(state.get("progress", 0.0)),
        str(state.get("progress_label", "Analyse IA en cours")),
    )


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
    workspace_view, header_action = _render_app_header(
        ai_pending=_analysis_is_pending(st.session_state.get("analysis"))
    )
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

    current_key = (
        ANALYSIS_POLICY_VERSION,
        file_hash,
        uploaded_file.name,
    )
    cached = st.session_state.get("analysis")
    if not isinstance(cached, dict) or cached.get("key") != current_key:
        _cancel_analysis(cached)
        st.session_state["analysis"] = {
            "key": current_key,
            "status": "core_running",
            "progress": 0.0,
            "progress_label": "Préparation de l'analyse",
        }
        st.rerun()

    cached = st.session_state["analysis"]
    status = cached.get("status")
    if status == "core_running":
        progress = st.empty()
        _render_analysis_progress(progress, 0.0, "Préparation de l'analyse")
        try:
            core = _analysis_service(PROJECT_CONFIG).run_core(
                file_name=uploaded_file.name,
                file_bytes=file_bytes,
                file_hash=file_hash,
                progress_callback=lambda value, text: _render_analysis_progress(
                    progress,
                    0.5 * value,
                    text,
                ),
            )
        except AnalysisError as error:
            progress.empty()
            cached.update(status="core_failed", error=f"Erreur [{error.code}] : {error}")
            st.error(f"Erreur [{error.code}] : {error}")
            return
        except Exception as error:
            progress.empty()
            message = f"Erreur inattendue : {type(error).__name__}: {str(error)[:240]}"
            cached.update(status="core_failed", error=message)
            st.error(message)
            return
        handle = _analysis_service(PROJECT_CONFIG).start_ai(core)
        cached.update(
            status="ai_running",
            report=core.report,
            laboratory=None,
            synthesis=None,
            synthesis_error=None,
            output_dir=core.output_dir,
            source_path=core.source_path,
            progress=0.0,
            progress_label="Préparation de l'analyse IA",
            ai_handle=handle,
        )
        st.rerun()

    if status == "core_failed":
        st.error(str(cached.get("error", "L'analyse principale a échoué.")))
        return

    report = cached["report"]
    laboratory = cached.get("laboratory")
    synthesis = cached.get("synthesis")
    synthesis_error = cached.get("synthesis_error")
    output_dir = cached["output_dir"]
    source_path = cached["source_path"]

    if status == "ai_running":
        _poll_ai_analysis(current_key)

    entering = _render_pdf_loading_if_pending(
        workspace_view,
        label="Chargement du PDF" if _is_pdf_bytes(file_bytes) else "Chargement du document",
    )
    result_key = "analysis_results_entering" if entering else "analysis_results_ready"
    with st.container(key=result_key):
        if workspace_view == AI_PENDING_VIEW:
            _render_ai_pending_state()
        else:
            _render_report(
                report,
                laboratory,
                synthesis,
                output_dir,
                source_path,
                document_name=uploaded_file.name,
                workspace_view=workspace_view,
                synthesis_error=synthesis_error,
            )
    _render_pending_scroll_reset()


def _render_app_header(*, ai_pending: bool = False) -> tuple[str, Any]:
    legacy_view = st.session_state.get("workspace_view")
    if legacy_view == "Analyse":
        st.session_state["workspace_view"] = "Général"
    elif legacy_view == "Laboratoire":
        st.session_state["workspace_view"] = "Analyse IA"
    elif ai_pending and legacy_view == "Analyse IA":
        st.session_state["workspace_view"] = "Général"
    elif not ai_pending and legacy_view == AI_PENDING_VIEW:
        st.session_state["workspace_view"] = "Analyse IA"
    navigation = (
        ("Général", AI_PENDING_VIEW, "Glossaire")
        if ai_pending
        else ("Général", "Analyse IA", "Glossaire")
    )
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
                    navigation,
                    default="Général",
                    key="workspace_view",
                    label_visibility="collapsed",
                )
            action_slot = st.empty()
    return selected or "Général", action_slot


def _render_input_panel(*, hidden: bool = False) -> InputDocument | OcrDemoDocument | None:
    selected_document = st.session_state.get("selected_document")
    if isinstance(selected_document, dict):
        input_slot = st.empty()
        input_slot.empty()
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
        st.rerun()
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
    _cancel_analysis(st.session_state.pop("analysis", None))
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
    if workspace_view != "Général":
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
        synthesis = cached.get("synthesis")
        synthesis_error = cached.get("synthesis_error")
    else:
        try:
            payload = json.loads(json_bytes)
        except json.JSONDecodeError as error:
            st.error(f"Fixture OCR invalide : {error}")
            return
        result = _analysis_service(PROJECT_CONFIG).analyze_ocr_fixture(
            payload=payload,
            markdown=markdown,
        )
        laboratory = result.laboratory
        ocr_detector = result.ocr_detector
        classification = result.classification
        extraction = result.extraction
        verification = result.verification
        synthesis = result.synthesis
        synthesis_error = result.synthesis_error
        st.session_state["analysis"] = {
            "key": current_key,
            "ocr_payload": payload,
            "ocr_markdown": markdown,
            "laboratory": laboratory,
            "ocr_detector": ocr_detector,
            "classification": classification,
            "extraction": extraction,
            "verification": verification,
            "synthesis": synthesis,
            "synthesis_error": synthesis_error,
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
            synthesis=synthesis,
            synthesis_error=synthesis_error,
            workspace_view=workspace_view,
        )


@st.cache_resource(show_spinner=False)
def _analysis_service(config: RunConfig) -> AnalysisService:
    return AnalysisService(config)


def _empty_laboratory_report() -> LaboratoryReport:
    return LaboratoryReport(schema_version="1.0", checks=())


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


def _render_ai_pending_state() -> None:
    st.markdown(
        """
        <section class="ai-pending-state" role="status" aria-live="polite">
          <strong>Analyse IA en cours</strong>
          <span>Les résultats seront affichés dès que tous les traitements seront terminés.</span>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_report(
    report: AnalysisReport | ImageAnalysisReport,
    laboratory: LaboratoryReport | None,
    synthesis: AnalysisSynthesis | None,
    output_dir: Path,
    source_path: Path,
    *,
    document_name: str,
    workspace_view: str,
    synthesis_error: str | None = None,
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
    if workspace_view == "Analyse IA":
        _render_extraction_laboratory(
            report.extraction,
            classification=report.classification,
            verification=report.extraction_verification,
            recognized_text=_read_ocr_markdown(report, output_dir),
            synthesis=synthesis,
            synthesis_error=synthesis_error,
            report=report,
            laboratory=laboratory,
            output_dir=output_dir,
            source_path=source_path,
            layout_images=layout_images,
            document_name=document_name,
        )
        return

    _render_document_type(report.classification)
    document_column, indicators_column = st.columns([0.54, 0.46], gap="large")
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
        _render_laboratory_synthesis(synthesis, synthesis_error=synthesis_error, compact=True)
        _render_risk_indicators(report.findings, report.detectors)

    _render_review_focus(scored)
    _render_review_details(
        diagnostics,
        laboratory,
        represented_findings=tuple((*scored, *diagnostics)),
        extraction=report.extraction,
    )


def _render_ocr_demo_report(
    *,
    name: str,
    markdown: str,
    laboratory: LaboratoryReport,
    ocr_detector: DetectorResult,
    classification: DocumentClassification | None,
    extraction: DocumentExtraction | None,
    verification: ExtractionVerification | None,
    synthesis: AnalysisSynthesis | None,
    synthesis_error: str | None,
    workspace_view: str,
) -> None:
    finding = ocr_detector.findings[0] if ocr_detector.findings else None
    points = finding.risk_points if finding is not None else 0.0
    tone = "review" if points >= 30 else "low"
    color = LEVEL_STYLE[tone]["color"]
    if workspace_view == "Glossaire":
        _render_indicator_glossary(categories=("content_consistency",))
        return
    if workspace_view == "Analyse IA":
        _render_extraction_laboratory(
            extraction,
            classification=classification,
            verification=verification,
            recognized_text=markdown,
            synthesis=synthesis,
            synthesis_error=synthesis_error,
            document_name=name,
        )
        return

    _render_document_type(classification)
    document_column, indicators_column = st.columns([0.54, 0.46], gap="large")
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
    _render_review_summary(scored, diagnostics, laboratory, extraction=extraction)


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
    "declaration": "Déclaration",
    "event": "Événement",
    "signature": "Signature",
    "purchase": "Achat",
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
    synthesis: AnalysisSynthesis | None = None,
    synthesis_error: str | None = None,
    report: AnalysisReport | ImageAnalysisReport | None = None,
    laboratory: LaboratoryReport | None = None,
    output_dir: Path | None = None,
    source_path: Path | None = None,
    layout_images: list[Path] | None = None,
    document_name: str = "Document analysé",
) -> None:
    _render_laboratory_synthesis(synthesis, synthesis_error=synthesis_error)

    has_visual_workspace = (
        report is not None and output_dir is not None and source_path is not None
    )
    if has_visual_workspace:
        document_column, context_column = st.columns([0.56, 0.44], gap="large")
        with document_column:
            _render_document_view(
                report=report,
                laboratory=laboratory,
                output_dir=output_dir,
                source_path=source_path,
                layout_images=layout_images or [],
                document_name=document_name,
            )
        with context_column, st.container(key="laboratory_context"):
            _render_classification(classification)
            if extraction is not None:
                _render_extraction_overview(extraction)
            _render_extraction_verification(verification)
    else:
        st.markdown(
            f'<h2 class="workspace-title document-name">{_html(document_name)}</h2>',
            unsafe_allow_html=True,
        )
        _render_classification(classification)
        if extraction is not None:
            _render_extraction_overview(extraction)
        _render_extraction_verification(verification)

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
            with st.expander("Texte reconnu par OCR", expanded=False):
                _render_recognized_text(recognized_text)
        return

    coverage = extraction.coverage

    if extraction.facts:
        rows = []
        for fact in extraction.facts:
            field_label = EXTRACTION_FIELD_LABELS.get(fact.field_code, fact.field_code)
            role_label = EXTRACTION_ROLE_LABELS.get(fact.role, fact.role)
            normalized = (
                str(fact.normalized_value)
                if fact.normalized_value is not None
                else "Non normalisée"
            )
            if fact.normalized_currency is not None:
                normalized = f"{normalized} {fact.normalized_currency}"
            displayed_value = fact.corrected_value or fact.raw_value
            source_value = (
                f"<span>Lecture OCR : {_html(fact.raw_value)}</span>"
                if fact.corrected_value is not None
                else ""
            )
            page = f"Page {fact.page}" if fact.page is not None else "Source non localisée"
            rows.append(
                "<tr>"
                f"<td><strong>{_html(field_label)}</strong><span>{_html(role_label)}</span></td>"
                f"<td><strong>{_html(displayed_value)}</strong>{source_value}</td>"
                f"<td>{_html(normalized)}</td>"
                f"<td>{_html(page)}</td>"
                "</tr>"
            )
        st.markdown(
            '<h3 class="subsection-title">Informations comparables</h3>'
            '<div class="extraction-table-wrap"><table class="extraction-table">'
            "<thead><tr><th>Champ</th><th>Valeur lue</th><th>Valeur de comparaison</th>"
            "<th>Source</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>",
            unsafe_allow_html=True,
        )

    if extraction.additional_fields:
        cards = []
        for field in extraction.additional_fields:
            page = f"Page {field.page}" if field.page is not None else "Source non localisée"
            displayed_value = field.corrected_value or field.raw_value
            source_value = (
                f"<small>Lecture OCR : {_html(field.raw_value)}</small>"
                if field.corrected_value is not None
                else ""
            )
            cards.append(
                '<article class="additional-extraction">'
                f"<span>{_html(field.raw_label)}</span>"
                f"<strong>{_html(displayed_value)}</strong>"
                f"{source_value}"
                f"<small>{_html(page)}</small>"
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
            f"{_html(header or f'Colonne {index + 1}')}"
            f"<span>{_html(EXTRACTION_COLUMN_ROLE_LABELS.get(role, role))}</span>"
            "</th>"
            for index, (header, role) in enumerate(zip(headers, roles, strict=True))
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

    if recognized_text:
        with st.expander("Texte reconnu par OCR", expanded=False):
            _render_recognized_text(recognized_text, compact=True)

    with st.expander("JSON final de l'extraction", expanded=False):
        st.json(
            {
                "extraction": extraction.to_dict(),
                "verification": verification.to_dict() if verification is not None else None,
            }
        )


def _render_laboratory_synthesis(
    synthesis: AnalysisSynthesis | None,
    *,
    synthesis_error: str | None,
    compact: bool = False,
) -> None:
    if synthesis is None:
        if synthesis_error:
            st.markdown(
                f"""
                <section class="laboratory-synthesis unavailable{' compact' if compact else ''}">
                  <span>Synthèse de l'analyse</span>
                  <strong>Synthèse indisponible</strong>
                  <p>{_html(synthesis_error)}</p>
                </section>
                """,
                unsafe_allow_html=True,
            )
        return

    st.markdown(
        f"""
        <section class="laboratory-synthesis{' compact' if compact else ''}">
          <span>Résumé de l'analyse</span>
          <p>{_html(synthesis.text)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_extraction_overview(extraction: DocumentExtraction) -> None:
    coverage = extraction.coverage
    coverage_tone = "clear" if coverage.ratio >= 0.95 else "attention"
    st.markdown(
        f"""
        <section class="extraction-overview">
          <article class="extraction-coverage {coverage_tone}">
            <div><strong>{coverage.ratio:.0%}</strong><span>Couverture des zones OCR</span></div>
            <i><b style="width:{coverage.ratio:.1%}"></b></i>
            <small>{coverage.accounted_regions} zone(s) comptabilisée(s)
              sur {coverage.total_regions}</small>
          </article>
          <article><strong>{len(extraction.facts)}</strong><span>Faits comparables</span></article>
          <article><strong>{len(extraction.additional_fields)}</strong>
            <span>Informations additionnelles</span></article>
          <article><strong>{len(extraction.tables)}</strong>
            <span>Tableaux conservés</span></article>
        </section>
        """,
        unsafe_allow_html=True,
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
    corrected_count = sum(review.correction_applied for review in verification.reviews)
    if verification.status == "clean":
        if corrected_count:
            title = "Extraction corrigée et cohérente avec l'OCR"
            detail = (
                f"{corrected_count} correction(s) structurée(s) appliquée(s), "
                "sans contradiction restante avec le texte reconnu."
            )
        else:
            title = "Extraction cohérente avec l'OCR"
            detail = (
                f"{verification.expected_targets} élément(s) comparé(s) au texte reconnu, "
                "sans contradiction concrète relevée."
            )
    elif verification.status == "incomplete":
        title = "Vérification partielle"
        detail = (
            "La vérification n'a pas pu couvrir toute l'extraction. "
            "L'extraction initiale n'a pas été modifiée."
        )
    else:
        title = "Qualité de l'extraction à contrôler"
        detail = (
            f"{len(attention_reviews)} interprétation(s) discutée(s) et "
            f"{len(verification.omissions)} omission(s) possible(s). "
            "Ces points n'entrent pas dans le score de risque."
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
        label = (
            "Correction d'extraction proposée"
            if review.verdict == "contradicted"
            else "Interprétation de l'extraction à confirmer"
        )
        affected_rows = (
            "<small>Ligne(s) concernée(s) : "
            + ", ".join(str(index + 1) for index in review.problematic_row_indexes)
            + "</small>"
            if review.problematic_row_indexes
            else ""
        )
        suggestion = (
            f"<small>Proposition : {_html(review.suggested_value)}</small>"
            if review.suggested_value
            else ""
        )
        issue_cards.append(
            '<article class="verification-issue">'
            f"<span>{_html(label)} · {_html(review.target_id)}</span>"
            f"<strong>{_html(review.explanation)}</strong>"
            f"{affected_rows}"
            f"{suggestion}</article>"
        )
    for omission in verification.omissions:
        value = f" · {_html(omission.proposed_value)}" if omission.proposed_value else ""
        issue_cards.append(
            '<article class="verification-issue omission">'
            f"<span>Information possiblement omise par l'extraction{value}</span>"
            f"<strong>{_html(omission.description)}</strong>"
            "</article>"
        )
    if issue_cards:
        st.markdown(
            '<section class="verification-issues">' + "".join(issue_cards) + "</section>",
            unsafe_allow_html=True,
        )


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
        tooltip = _html(CATEGORY_DESCRIPTIONS.get(indicator.category, ""))
        points = str(round(indicator.points))
        maximum = str(round(indicator.maximum))
        icon = _indicator_icon(indicator.tone)
        delay_seconds = index * INDICATOR_STEP_SECONDS
        rows.append(
            f"""
            <li
              class="indicator-step {indicator.tone}"
              data-state="{indicator.tone}"
              title="{tooltip}"
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
    statuses = tuple(
        detector.status for detector in detectors if detector.name in expected_detectors
    )
    if "partial" in statuses:
        return "partial", "Contrôle partiel", 35.0
    if "completed" in statuses:
        return "clear", "Aucun signal détecté", 100.0
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
        return "Critique"
    if percentage >= 40:
        return "Important"
    if percentage > 0:
        return "Modéré"
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


def _render_review_focus(scored: list[Finding]) -> None:
    grouped_scored = _group_findings_by_category(scored)
    explanation = (
        "Ces éléments ont contribué à l'indice de vigilance. Ils sont à comparer "
        "directement avec le document."
        if grouped_scored
        else "Aucun élément prioritaire n'a été relevé par les contrôles exécutés."
    )
    st.markdown(
        f"""
        <header class="review-focus-heading" id="points-a-verifier">
          <h2>Points à vérifier</h2>
          <p>{_html(explanation)}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )
    if grouped_scored:
        for category, findings in grouped_scored:
            _family_finding_card(category, findings)
        return

    st.markdown(
        """
        <div class="review-banner clear compact">
          <strong>Aucun signal pris en compte dans le score</strong>
          <span>Les contrôles exécutés n'ont pas relevé d'élément prioritaire.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_review_details(
    diagnostics: list[Finding],
    laboratory: LaboratoryReport | None,
    *,
    represented_findings: tuple[Finding, ...] = (),
    extraction: DocumentExtraction | None = None,
) -> None:
    del diagnostics
    cards = _recognized_element_cards(
        laboratory,
        extraction=extraction,
        findings=represented_findings,
    )
    if not cards:
        return
    st.markdown(
        """
        <header class="recognized-elements-heading">
          <h2>Éléments reconnus</h2>
          <p>Informations utiles lues dans le document et disponibles pour la vérification.</p>
        </header>
        <section class="recognized-elements">
        """
        + "".join(cards)
        + "</section>",
        unsafe_allow_html=True,
    )


def _render_review_summary(
    scored: list[Finding],
    diagnostics: list[Finding],
    laboratory: LaboratoryReport | None,
    *,
    extraction: DocumentExtraction | None = None,
) -> None:
    _render_review_focus(scored)
    _render_review_details(
        diagnostics,
        laboratory,
        represented_findings=tuple((*scored, *diagnostics)),
        extraction=extraction,
    )


def _render_classification(classification: DocumentClassification | None) -> None:
    """Render the semantic family without exposing model internals."""
    if not classification:
        return

    st.markdown(
        '<h2 class="workspace-title">Type de document reconnu</h2>',
        unsafe_allow_html=True,
    )

    family_label = DOCUMENT_FAMILY_LABELS.get(
        classification.family,
        "Type non déterminé",
    )
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
        with st.expander("Pourquoi ce classement ?", expanded=False):
            st.markdown(
                '<div class="classification-evidence-title">'
                "Éléments ayant guidé le classement</div>"
                + "".join(
                    f'<blockquote class="classification-evidence">{_html(excerpt)}</blockquote>'
                    for excerpt in classification.evidence
                ),
                unsafe_allow_html=True,
            )


def _render_document_type(classification: DocumentClassification | None) -> None:
    if classification is None:
        return
    family_label = DOCUMENT_FAMILY_LABELS.get(
        classification.family,
        "Type non déterminé",
    )
    st.markdown(
        f"""
        <div class="document-type-strip">
          <span>Type de document</span>
          <strong>{_html(family_label)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _group_findings_by_category(
    findings: list[Finding],
) -> list[tuple[str, tuple[Finding, ...]]]:
    groups: dict[str, list[Finding]] = {}
    for finding in findings:
        groups.setdefault(finding.category, []).append(finding)
    grouped = [(category, tuple(items)) for category, items in groups.items()]
    return sorted(
        grouped,
        key=lambda item: _family_score(item[0], item[1]),
        reverse=True,
    )


_RECOGNIZED_FIELD_PRIORITY = {
    code: index
    for index, code in enumerate(
        (
            "person_name",
            "organization_name",
            "invoice_number",
            "contract_number",
            "claim_number",
            "document_number",
            "account_number",
            "tax_identifier",
            "professional_identifier",
            "registration_identifier",
            "iban",
            "bic",
            "payment_card_number",
            "date",
            "date_period",
            "monetary_amount",
            "address",
            "service_description",
            "product_description",
        )
    )
}

_IDENTIFIER_LABELS = {
    "OCR_CARD_": "Numéro de carte",
    "OCR_IBAN_": "IBAN",
    "OCR_BIC_": "BIC / SWIFT",
    "OCR_CKYC_": "Identifiant CKYC",
    "OCR_MICR_": "Code MICR",
    "OCR_SIREN_": "SIREN",
    "OCR_SIRET_": "SIRET",
    "OCR_EU_VAT_": "Numéro de TVA",
    "OCR_RPPS_": "Numéro RPPS",
    "OCR_FINESS_": "Numéro FINESS",
}


def _recognized_element_cards(
    laboratory: LaboratoryReport | None,
    *,
    extraction: DocumentExtraction | None,
    findings: tuple[Finding, ...],
) -> list[str]:
    cards: list[str] = []
    seen: set[tuple[str, str]] = set()
    covered_field_codes: set[str] = set()

    for finding in findings:
        if not finding.code.startswith("PDF_SOFTWARE_"):
            continue
        software_entries = finding.evidence.get("software", ())
        if not isinstance(software_entries, (list, tuple)):
            continue
        for entry in software_entries:
            if not isinstance(entry, dict) or not entry.get("value"):
                continue
            field = str(entry.get("field", "")).casefold()
            label = {
                "creator": "Logiciel créateur",
                "producer": "Logiciel de production PDF",
            }.get(field, "Logiciel déclaré")
            points = float(entry.get("risk_points", 0) or 0)
            cards.append(
                _recognized_element_card(
                    label=label,
                    value=str(entry["value"]),
                    status="À vérifier" if points > 0 else "Reconnu",
                    state="attention" if points > 0 else "clear",
                )
            )

    if laboratory is not None:
        observations = (
            observation
            for check in laboratory.checks
            for observation in check.observations
            if observation.code.startswith(OCR_IDENTIFIER_PREFIXES)
        )
        for observation in observations:
            value = observation.evidence.get("value")
            if value is None and observation.evidence.get("values"):
                value = " / ".join(str(item) for item in observation.evidence["values"])
            if not value:
                continue
            label, field_code = _identifier_label_and_field(observation.code)
            formatted = _format_identifier_value(observation.code, str(value))
            key = (label, re.sub(r"\W", "", formatted).casefold())
            if key in seen:
                continue
            seen.add(key)
            if field_code:
                covered_field_codes.add(field_code)
            status = {
                "clear": "Format valide",
                "attention": "Format à vérifier",
                "indeterminate": "Non vérifiable",
                "error": "Lecture incomplète",
            }.get(observation.state, "Reconnu")
            state = "clear" if observation.state == "clear" else "attention"
            cards.append(
                _recognized_element_card(
                    label=label,
                    value=formatted,
                    status=status,
                    state=state,
                )
            )

    if extraction is None:
        return cards

    ignored_roles = {"transaction", "line_item", "debit", "credit", "unit_price"}
    candidates = sorted(
        (
            fact
            for fact in extraction.facts
            if fact.field_code in _RECOGNIZED_FIELD_PRIORITY
            and fact.field_code not in covered_field_codes
            and fact.role not in ignored_roles
        ),
        key=lambda fact: (_RECOGNIZED_FIELD_PRIORITY[fact.field_code], fact.page or 0),
    )
    for fact in candidates:
        value = fact.corrected_value or fact.raw_value
        if not value:
            continue
        key = (fact.field_code, re.sub(r"\W", "", str(value)).casefold())
        if key in seen:
            continue
        seen.add(key)
        label = EXTRACTION_FIELD_LABELS.get(fact.field_code, fact.field_code)
        role = EXTRACTION_ROLE_LABELS.get(fact.role)
        if role and fact.role not in {"document", "other"}:
            label = f"{label} · {role}"
        normalized = fact.normalization_status == "normalized"
        cards.append(
            _recognized_element_card(
                label=label,
                value=str(value),
                status="Reconnu" if normalized else "Lecture à confirmer",
                state="clear" if normalized else "attention",
            )
        )
        if len(cards) >= 12:
            break
    return cards


def _identifier_label_and_field(code: str) -> tuple[str, str | None]:
    field_codes = {
        "OCR_CARD_": "payment_card_number",
        "OCR_IBAN_": "iban",
        "OCR_BIC_": "bic",
        "OCR_SIREN_": "registration_identifier",
        "OCR_SIRET_": "registration_identifier",
        "OCR_EU_VAT_": "tax_identifier",
        "OCR_RPPS_": "professional_identifier",
        "OCR_FINESS_": "professional_identifier",
        "OCR_CKYC_": "other_identifier",
        "OCR_MICR_": "other_identifier",
    }
    for prefix, label in _IDENTIFIER_LABELS.items():
        if code.startswith(prefix):
            return label, field_codes.get(prefix)
    return "Identifiant", None


def _recognized_element_card(*, label: str, value: str, status: str, state: str) -> str:
    return (
        f'<article class="evidence-card recognized-element {state}">'
        f'<span>{_html(label)}</span><strong>{_html(value)}</strong>'
        f'<small><i></i>{_html(status)}</small></article>'
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
    content_progress = _content_progress_markup(label, bounded)
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
          {content_progress}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _content_progress_markup(label: str, progress: float) -> str:
    config = PROJECT_CONFIG.analysis
    if not config.ocr_enabled:
        return ""
    stages = [
        ("ocr", "OCR", True),
        ("classification", "Classement", config.classification_enabled),
        ("extraction", "Extraction", config.extraction_enabled),
        ("verification", "Vérification", config.verification_enabled),
        ("synthesis", "Synthèse", config.synthesis_enabled),
    ]
    enabled_stages = tuple((key, title) for key, title, enabled in stages if enabled)
    if not enabled_stages:
        return ""

    state_key = "_analysis_content_progress_stage"
    if progress <= 0.02:
        st.session_state[state_key] = -1
    detected = _content_stage_from_label(label)
    if detected is not None:
        detected_index = next(
            (index for index, (key, _) in enumerate(enabled_stages) if key == detected),
            None,
        )
        if detected_index is not None:
            st.session_state[state_key] = max(
                int(st.session_state.get(state_key, -1)),
                detected_index,
            )
    if progress >= 1:
        current_index = len(enabled_stages)
    else:
        current_index = int(st.session_state.get(state_key, -1))

    items = []
    for index, (_, title) in enumerate(enabled_stages):
        state = "completed" if index < current_index else "active" if index == current_index else ""
        items.append(
            f'<span class="content-progress-stage {state}"><b>{_html(title)}</b><i></i></span>'
        )
    return (
        '<div class="content-progress" aria-label="Progression de l’analyse du contenu">'
        + "".join(items)
        + "</div>"
    )


def _content_stage_from_label(label: str) -> str | None:
    lowered = label.casefold()
    if "synthèse" in lowered or "synthese" in lowered:
        return "synthesis"
    if "vérification" in lowered or "verification" in lowered:
        return "verification"
    if "extraction" in lowered:
        return "extraction"
    if "classification" in lowered:
        return "classification"
    if "reconnaissance" in lowered or "contenu reconnu" in lowered:
        return "ocr"
    if "analyse du contenu terminée" in lowered:
        return "verification"
    return None


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
        label="Indice de vigilance",
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
    card_keyframes = [
        "0% { background:linear-gradient(135deg,rgba(53,208,127,.12),#181d1b 58%); "
        "border-left-color:#35d07f; box-shadow:0 0 0 rgba(53,208,127,0); }"
    ]
    for index, step in enumerate(normalized_steps, start=1):
        percentage = index / len(normalized_steps) * 100
        keyframes.append(f"{percentage:.3f}% {{ --animated-score:{step}; }}")
        if step >= 70:
            accent, glow = "#ff6b6b", "rgba(255,107,107,.22)"
            surface = "linear-gradient(135deg,rgba(255,107,107,.16),#24191d 58%)"
        elif step >= 30:
            accent, glow = "#f4b942", "rgba(244,185,66,.19)"
            surface = "linear-gradient(135deg,rgba(244,185,66,.14),#211d17 58%)"
        else:
            accent, glow = "#35d07f", "rgba(53,208,127,.16)"
            surface = "linear-gradient(135deg,rgba(53,208,127,.12),#181d1b 58%)"
        card_keyframes.append(
            f"{percentage:.3f}% {{ background:{surface}; border-left-color:{accent}; "
            f"box-shadow:0 0 18px {glow}; }}"
        )
    card_animation_name = f"{animation_name}-card"
    if tone == "high":
        action_label = "Risque critique · Revue manuelle urgente"
    elif tone == "review":
        action_label = "Risque important · Revue manuelle recommandée"
    else:
        action_label = "Risque modéré"
    findings_link = (
        '<a href="#points-a-verifier">Voir les indices trouvés '
        '<span aria-hidden="true">↓</span></a>'
        if displayed_score > 0
        else ""
    )
    st.markdown(
        f"""
        <style>@keyframes {animation_name} {{ {" ".join(keyframes)} }}</style>
        <style>@keyframes {card_animation_name} {{ {" ".join(card_keyframes)} }}</style>
        <section class="fraud-score {tone}" aria-label="{_html(label)} : {score:g} sur {maximum}"
          style="--score-duration:{duration_seconds:.2f}s;
            --score-card-animation:{card_animation_name}">
          <div class="fraud-score-copy">
            <span>{_html(label)}</span>
            <strong>{_html(action_label)}</strong>
            {findings_link}
          </div>
          <strong style="--score-color:{color};--score-target:{displayed_score};
            --score-duration:{duration_seconds:.2f}s;--score-animation:{animation_name}">
            <span class="fraud-score-number" aria-hidden="true"></span>
            <span class="sr-only">{score:g}</span><small aria-hidden="true">/{maximum}</small>
          </strong>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _family_finding_card(category: str, findings: tuple[Finding, ...]) -> None:
    family_points = _family_score(category, findings)
    cap = FAMILY_CAPS[category]
    percentage = family_points / cap * 100 if cap else 0
    tone = _tone_for_ratio(percentage)
    items = _family_finding_items(findings)
    cards = []
    for title, description, location in items:
        location_html = f"<small>{_html(location)}</small>" if location else ""
        cards.append(
            "<li><div>"
            f"<strong>{_html(title)}</strong>{location_html}"
            f"</div><p>{_html(description)}</p></li>"
        )
    count_label = "indice" if len(items) == 1 else "indices"
    st.markdown(
        f"""
        <article class="evidence-card family-finding-card {tone}">
          <div class="finding-score">
            <strong>{round(family_points)}</strong>
            <span>/{round(cap)}</span>
          </div>
          <div class="family-finding-content">
            <div class="family-finding-heading">
              <h3>{_html(CATEGORY_LABELS.get(category, category))}</h3>
              <span>{len(items)} {count_label}</span>
            </div>
            <ul>{''.join(cards)}</ul>
          </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _family_finding_items(
    findings: tuple[Finding, ...],
) -> list[tuple[str, str, str | None]]:
    items: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for finding in findings:
        observations = finding.evidence.get("observations", ())
        if finding.detector == "ocr_content" and isinstance(observations, (list, tuple)):
            for observation in observations:
                if not isinstance(observation, dict):
                    continue
                code = str(observation.get("code", ""))
                if not code or code in seen:
                    continue
                seen.add(code)
                title, description = _ocr_observation_copy(observation)
                page = observation.get("page")
                items.append((title, description, f"Page {page}" if page else None))
            if observations:
                continue
        if finding.code in seen:
            continue
        seen.add(finding.code)
        title, description = _business_finding_copy(finding)
        location = f"Page {finding.page}" if finding.page is not None else None
        items.append((title, description, location))
    return items


def _ocr_observation_copy(observation: dict[str, Any]) -> tuple[str, str]:
    code = str(observation.get("code", ""))
    copies = {
        "OCR_CARD_LUHN_INVALID": (
            "Numéro de carte à vérifier",
            "Le numéro ne passe pas le contrôle Luhn.",
        ),
        "OCR_IBAN_INVALID": ("IBAN à vérifier", "L'IBAN ne passe pas le contrôle de validité."),
        "OCR_BIC_INVALID": ("BIC / SWIFT à vérifier", "Le code ne respecte pas le format BIC."),
        "OCR_SIREN_INVALID": (
            "SIREN à vérifier",
            "Le numéro ne passe pas son contrôle de validité.",
        ),
        "OCR_SIRET_INVALID": (
            "SIRET à vérifier",
            "Le numéro ne passe pas son contrôle de validité.",
        ),
        "OCR_EU_VAT_INVALID": (
            "Numéro de TVA à vérifier",
            "Le numéro ne passe pas le contrôle prévu pour son pays.",
        ),
        "OCR_RPPS_INVALID": (
            "Numéro RPPS à vérifier",
            "Le numéro ne respecte pas le format attendu.",
        ),
        "OCR_FINESS_INVALID": (
            "Numéro FINESS à vérifier",
            "Le numéro ne passe pas son contrôle de validité.",
        ),
        "OCR_BANKING_GEOGRAPHY_MISMATCH": (
            "Références bancaires de pays différents",
            "Les références bancaires reconnues ne pointent pas vers le même pays.",
        ),
        "OCR_STATEMENT_SUMMARY_MISMATCH": (
            "Totaux du relevé à vérifier",
            "Les totaux reconnus ne correspondent pas au détail des opérations.",
        ),
        "OCR_INVOICE_TOTAL_MISMATCH": (
            "Total de facture à vérifier",
            "Le total reconnu ne correspond pas aux montants détaillés.",
        ),
        "OCR_LEDGER_MISMATCH": (
            "Solde à vérifier",
            "Le solde reconnu ne correspond pas aux opérations du relevé.",
        ),
        "OCR_DATE_INVALID": (
            "Date à vérifier",
            "La date reconnue n'est pas valide dans le calendrier.",
        ),
    }
    if code in copies:
        return copies[code]
    return (
        _sentence_case(observation.get("title", "Information à vérifier")),
        _business_text(observation.get("summary", "Vérifiez cette information dans le document.")),
    )


def _existing_artifacts(output_dir: Path, relatives: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for relative in relatives:
        path = output_dir / relative
        if path.is_file():
            paths.append(path)
    return paths


def _is_pdf_bytes(file_bytes: bytes) -> bool:
    return b"%PDF-" in file_bytes[:1024]


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
    if "revision" in path.name:
        return "Différences entre les versions"
    if "repeated-visual-region" in path.name:
        return "Comparaison des éléments visuels répétés"
    return "Visualisation complémentaire"


def _business_finding_copy(finding: Finding) -> tuple[str, str]:
    if finding.code.startswith("PDF_SOFTWARE_"):
        labels = {"creator": "Créateur", "producer": "Logiciel de production PDF"}
        details = []
        for field in ("creator", "producer"):
            value = finding.evidence.get(field)
            if value:
                details.append(f"{labels[field]} : {value}")
        description = ". ".join(details)
        if description:
            description += ". Ce logiciel peut avoir servi à convertir ou modifier le fichier."
        else:
            description = "Le logiciel déclaré par le PDF est à vérifier avec les autres indices."
        return "Logiciel du fichier à vérifier", description

    concise_copy = {
        "PDF_INCREMENTAL_UPDATES": (
            "Plusieurs versions dans le PDF",
            "Le fichier garde la trace de plusieurs enregistrements. Cela peut être normal, "
            "mais les changements sont à comparer.",
        ),
        "PDF_DATE_CONTRADICTION": (
            "Dates du fichier incohérentes",
            "La date de modification enregistrée précède la date de création. Ces dates peuvent "
            "être modifiées, mais cet ordre est inhabituel.",
        ),
        "PDF_TRAILING_DATA": (
            "Données ajoutées à la fin du fichier",
            "Le PDF contient des données après sa fin normale. Elles peuvent venir d'un assemblage "
            "ou d'une modification du fichier.",
        ),
        "SCAN_IMAGE_OVERLAY": (
            "Image ajoutée sur une page scannée",
            "Une image distincte recouvre une partie du scan. Vérifiez si cet ajout correspond au "
            "processus habituel.",
        ),
        "SCAN_VISIBLE_TEXT_OVERLAY": (
            "Texte ajouté sur une page scannée",
            "Du texte a été placé au-dessus du scan. Un formulaire prérempli peut aussi expliquer "
            "cet ajout.",
        ),
        "PDF_REVIEW_ANNOTATION": (
            "Annotation ajoutée au PDF",
            "Le document contient une annotation encore modifiable. Vérifiez si elle était prévue "
            "dans le traitement du dossier.",
        ),
        "PDF_REVISION_PAGE_COUNT_CHANGED": (
            "Nombre de pages modifié",
            "Des pages ont été ajoutées ou retirées entre deux versions conservées du PDF.",
        ),
        "PDF_REVISION_VISUAL_CHANGE": (
            "Zone modifiée entre deux versions",
            "Une zone visible diffère entre deux versions conservées du PDF. Comparez-la avec "
            "l'aperçu du document.",
        ),
        "RASTER_LOCAL_COMPRESSION_ANOMALY": (
            "Compression différente dans une zone",
            "Une zone ne réagit pas comme le reste de l'image à la compression. Un scan ou des "
            "contours nets peuvent aussi produire cet effet.",
        ),
        "IMAGE_C2PA_INVALID": (
            "Informations d'origine invalides",
            "Les informations signées sur l'origine de l'image ne sont pas valides. Cela ne prouve "
            "pas à lui seul une modification frauduleuse.",
        ),
        "AI_IMAGE_C2PA_DECLARATION": (
            "Création par IA déclarée",
            "Les informations d'origine indiquent une création ou une composition par IA.",
        ),
        "AI_PDF_C2PA_DECLARATION": (
            "Création par IA déclarée",
            "Les informations d'origine indiquent une création ou une composition par IA.",
        ),
        "AI_GENERATOR_METADATA_MENTIONED": (
            "Outil de génération IA mentionné",
            "Les métadonnées mentionnent un outil ou des paramètres de génération. Elles peuvent "
            "toutefois être modifiées.",
        ),
    }
    if finding.code in concise_copy:
        return concise_copy[finding.code]
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
            "Plusieurs zones ressemblent à des images générées par IA. La compression ou le "
            "redimensionnement peuvent aussi produire ce résultat : ce signal ne suffit pas seul.",
        )
    if finding.detector == "ocr_content":
        groups = finding.evidence.get("groups", {})
        labels = (
            [str(group.get("label", "")).lower() for group in groups.values()]
            if isinstance(groups, dict)
            else []
        )
        subject = ", ".join(label for label in labels if label)
        description = (
            f"Le contrôle a relevé des écarts dans les éléments suivants : {subject}. "
            "Vérifiez les valeurs directement dans le document."
            if subject
            else "Certaines informations reconnues ne sont pas cohérentes. Vérifiez-les "
            "directement dans le document."
        )
        return (
            "Cohérence du contenu à vérifier",
            description,
        )
    return _sentence_case(finding.title), _business_text(finding.description)


def _format_identifier_value(code: str, value: str) -> str:
    if " / " in value:
        return " / ".join(_format_identifier_value(code, item) for item in value.split(" / "))
    compact = re.sub(r"[\s-]", "", value)
    if code.startswith(("OCR_CARD_", "OCR_IBAN_")) and compact.isalnum():
        return " ".join(compact[index : index + 4] for index in range(0, len(compact), 4))
    return value


def _sentence_case(value: object) -> str:
    text = _business_text(value).strip()
    return text[:1].upper() + text[1:] if text else text


def _business_text(value: object) -> str:
    text = str(value)
    replacements = (
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
          --ink: #f7f8fa;
          --muted: #aeb4bd;
          --line: #343942;
          --line-strong: #46505d;
          --panel: #191c21;
          --panel-soft: #20252c;
          --panel-raised: #252b33;
          --bg: #101216;
          --green: #35d07f;
          --amber: #f4b942;
          --coral: #ff6b6b;
          --cyan: #45d6e6;
          --blue: #6fa8ff;
          --violet: #a78bfa;
          --accent-gradient: linear-gradient(90deg, var(--cyan), var(--blue));
          --surface-gradient: linear-gradient(145deg, #1c2026 0%, #171a1f 100%);
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
          background: linear-gradient(180deg, #12151a 0, var(--bg) 420px);
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
          font-size: 1.18rem;
          font-weight: 900;
          letter-spacing: 0;
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
          background: var(--surface-gradient);
          border: 2px dashed #596574;
          border-radius: 8px;
          padding: .5rem .75rem;
          transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
        }
        div[data-testid="stFileUploader"]:hover {
          border-color: var(--cyan);
          box-shadow: 0 0 0 1px rgba(69, 214, 230, .14), 0 12px 30px rgba(0, 0, 0, .18);
          transform: translateY(-1px);
        }
        div[data-testid="stFileUploader"] label {
          color: var(--ink);
          font-weight: 800;
        }
        .stButton > button {
          min-height: 3rem;
          border-radius: 6px;
          font-weight: 850;
          background: linear-gradient(135deg, #f8fafc, #dfe8ee);
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
          background: var(--surface-gradient);
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
          position: relative;
          display: block;
          height: 100%;
          overflow: hidden;
          background: var(--accent-gradient);
          border-radius: inherit;
          transition: width .18s ease;
          box-shadow: 0 0 20px rgba(69, 214, 230, .42);
          filter: saturate(1.15) brightness(1.05);
        }
        .analysis-progress-track i::after {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, .68), transparent);
          transform: translateX(-100%);
          animation: progress-sheen 1.4s ease-in-out infinite;
        }
        .content-progress {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(86px, 1fr));
          gap: .5rem;
          margin-top: .65rem;
        }
        .content-progress-stage {
          display: grid;
          gap: .28rem;
          min-width: 0;
          color: #77747d;
          font-size: .72rem;
          font-weight: 750;
        }
        .content-progress-stage b {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .content-progress-stage i {
          position: relative;
          display: block;
          height: 3px;
          overflow: hidden;
          border-radius: 999px;
          background: #323137;
        }
        .content-progress-stage.completed {
          color: #b8b5af;
        }
        .content-progress-stage.completed i {
          background: #4d8fa0;
        }
        .content-progress-stage.active {
          color: var(--ink);
        }
        .content-progress-stage.active i::after {
          content: "";
          position: absolute;
          inset: 0 auto 0 0;
          width: 42%;
          border-radius: inherit;
          background: var(--cyan);
          animation: content-progress-active 1.25s ease-in-out infinite alternate;
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
        .ai-pending-state {
          min-height: 140px;
          display: grid;
          place-content: center;
          gap: .35rem;
          text-align: center;
          border: 1px solid #31566a;
          border-radius: 6px;
          background: #151d22;
          color: #d9e9f0;
        }
        .ai-pending-state strong {
          color: var(--cyan);
          font-size: 1rem;
        }
        .ai-pending-state span {
          color: #aebbc1;
          font-size: .82rem;
        }
        .fraud-score {
          --score-surface: linear-gradient(135deg, #1d2229, #181b20);
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          min-height: 74px;
          padding: .7rem .85rem;
          margin: .15rem 0 .55rem;
          background: var(--score-surface);
          border: 1px solid var(--line);
          border-left: 4px solid #595760;
          border-radius: 6px;
          animation: var(--score-card-animation) var(--score-duration) steps(1, end)
            var(--result-entry-delay, 0s) forwards;
          transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }
        .fraud-score.low {
          --score-surface: linear-gradient(135deg, rgba(53, 208, 127, .12), #181d1b 58%);
          border-left-color: var(--green);
        }
        .fraud-score.review {
          --score-surface: linear-gradient(135deg, rgba(244, 185, 66, .14), #211d17 58%);
          border-left-color: var(--amber);
        }
        .fraud-score.high {
          --score-surface: linear-gradient(135deg, rgba(255, 107, 107, .16), #24191d 58%);
          border-left-color: var(--coral);
        }
        .fraud-score:hover {
          transform: translateY(-1px);
        }
        .fraud-score-copy {
          display: grid;
          justify-items: start;
          gap: .13rem;
        }
        .fraud-score-copy > span {
          color: var(--muted);
          font-size: .68rem;
          font-weight: 800;
          text-transform: uppercase;
        }
        .fraud-score-copy > strong {
          color: var(--ink);
          font-size: .88rem;
          line-height: 1.2;
        }
        .fraud-score.low .fraud-score-copy > strong { color: var(--green); }
        .fraud-score.review .fraud-score-copy > strong { color: var(--amber); }
        .fraud-score.high .fraud-score-copy > strong { color: var(--coral); }
        .fraud-score-copy a {
          margin-top: .1rem;
          color: var(--cyan);
          font-size: .7rem;
          font-weight: 750;
          text-decoration: none;
        }
        .fraud-score-copy a:hover {
          color: #8ceaf3;
          text-decoration: underline;
        }
        .fraud-score > strong {
          display: inline-flex;
          align-items: baseline;
          color: var(--score-color);
          font-size: 2.45rem;
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
        .fraud-score > strong small {
          color: var(--muted);
          font-size: .78rem;
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
          background: linear-gradient(145deg, rgba(69, 214, 230, .045), #14171b 40%);
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
          gap: .6rem;
          align-items: center;
          min-width: 0;
          min-height: 52px;
          padding: .42rem .72rem;
          border: 1px solid #3b3942;
          border-radius: 6px;
          background: linear-gradient(135deg, #1d2127, #1a1d22);
          animation: indicator-step-validate .42s ease-out var(--step-delay) both;
          transition: transform .18s ease, filter .18s ease, box-shadow .18s ease;
        }
        .indicator-step:hover {
          z-index: 1;
          transform: translateX(2px);
          filter: brightness(1.07);
          box-shadow: 0 7px 20px rgba(0, 0, 0, .2);
        }
        .indicator-queue > .indicator-step {
          margin: 0;
          padding: .42rem .85rem;
        }
        .indicator-step.notice-tone {
          --risk-color: var(--blue);
          --step-bg: linear-gradient(135deg, rgba(111, 168, 255, .13), #181d25 62%);
          --step-border: #31526d;
        }
        .indicator-step.warning {
          --risk-color: var(--amber);
          --step-bg: linear-gradient(135deg, rgba(244, 185, 66, .15), #242015 62%);
          --step-border: #685727;
        }
        .indicator-step.danger {
          --risk-color: var(--coral);
          --step-bg: linear-gradient(135deg, rgba(255, 107, 107, .17), #27181c 62%);
          --step-border: #733543;
        }
        .indicator-step.clear {
          --risk-color: var(--green);
          --step-bg: linear-gradient(135deg, rgba(53, 208, 127, .1), #16201b 62%);
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
          gap: .3rem;
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
          font-size: .8rem;
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
          font-size: .7rem;
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
          background: linear-gradient(
            90deg,
            color-mix(in srgb, var(--risk-color) 72%, white),
            var(--risk-color)
          );
          box-shadow: 0 0 10px color-mix(in srgb, var(--risk-color) 45%, transparent);
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
        @keyframes content-progress-active {
          from { transform: translateX(0); opacity: .55; }
          to { transform: translateX(138%); opacity: 1; }
        }
        @keyframes progress-sheen {
          0%, 35% { transform: translateX(-100%); }
          75%, 100% { transform: translateX(100%); }
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
          font-size: 1.08rem;
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
          background: var(--surface-gradient);
          margin-bottom: .55rem;
        }
        .review-banner.attention { border-left-color: var(--coral); }
        .review-banner.clear { border-left-color: var(--green); }
        .review-banner strong {
          font-size: .95rem;
        }
        .review-banner span {
          color: var(--muted);
          font-size: .8rem;
          text-align: right;
        }
        .review-banner.compact {
          align-items: flex-start;
          flex-direction: column;
          margin: .55rem 0 .7rem;
        }
        .review-banner.compact span { text-align: left; }
        .review-focus-heading {
          margin: 1.25rem 0 .65rem;
          padding-top: 1rem;
          border-top: 1px solid var(--line);
        }
        .review-focus-heading h2 {
          margin: 0;
          color: var(--ink);
          font-size: 1.08rem;
          line-height: 1.3;
        }
        .review-focus-heading p {
          max-width: 760px;
          margin: .28rem 0 0;
          color: var(--muted);
          font-size: .8rem;
          line-height: 1.4;
        }
        .document-type-strip {
          display: flex;
          align-items: baseline;
          gap: .65rem;
          margin: .1rem 0 .7rem;
          color: var(--muted);
          font-size: .78rem;
        }
        .document-type-strip strong {
          color: var(--cyan);
          font-size: .88rem;
        }
        .family-finding-card {
          display: grid;
          grid-template-columns: 54px minmax(0, 1fr);
          gap: .7rem;
          padding: .75rem;
          margin-bottom: .5rem;
          border: 1px solid var(--line);
          border-left: 5px solid var(--blue);
          border-radius: 6px;
          background: var(--surface-gradient);
        }
        .evidence-card {
          transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease,
            filter .18s ease;
        }
        .evidence-card:hover {
          transform: translateY(-1px);
          filter: brightness(1.045);
          box-shadow: 0 8px 22px rgba(0, 0, 0, .18);
        }
        .family-finding-card.danger {
          border-left-color: var(--coral);
          background: linear-gradient(135deg, rgba(255, 107, 107, .13), #21191d 62%);
        }
        .family-finding-card.warning {
          border-left-color: var(--amber);
          background: linear-gradient(135deg, rgba(244, 185, 66, .13), #211e18 62%);
        }
        .family-finding-card.notice-tone {
          border-left-color: var(--blue);
          background: linear-gradient(135deg, rgba(111, 168, 255, .12), #181e27 62%);
        }
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
          font-size: .68rem;
          line-height: 1;
          margin-top: .2rem;
        }
        .family-finding-heading {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: .7rem;
        }
        .family-finding-heading h3 {
          margin: 0;
          color: var(--ink);
          font-size: .94rem;
        }
        .family-finding-heading > span {
          color: var(--muted);
          font-size: .7rem;
          white-space: nowrap;
        }
        .family-finding-content ul {
          display: grid;
          gap: .45rem;
          margin: .5rem 0 0;
          padding: 0;
          list-style: none;
        }
        .family-finding-content li {
          padding-top: .45rem;
          border-top: 1px solid var(--line);
          border-top: 1px solid color-mix(in srgb, var(--line) 72%, transparent);
        }
        .family-finding-content li > div {
          display: flex;
          justify-content: space-between;
          gap: .7rem;
        }
        .family-finding-content li strong {
          color: #e8e9ec;
          font-size: .82rem;
        }
        .family-finding-content li small {
          color: var(--muted);
          font-size: .68rem;
          white-space: nowrap;
        }
        .family-finding-content li p {
          margin: .18rem 0 0;
          color: #c3c7cd;
          font-size: .78rem;
          line-height: 1.38;
        }
        .recognized-elements-heading {
          margin: 1.35rem 0 .65rem;
          padding-top: 1rem;
          border-top: 1px solid var(--line);
        }
        .recognized-elements-heading h2 {
          margin: 0;
          color: var(--ink);
          font-size: 1.08rem;
        }
        .recognized-elements-heading p {
          margin: .28rem 0 0;
          color: var(--muted);
          font-size: .8rem;
        }
        .recognized-elements {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .45rem;
        }
        .recognized-element {
          display: grid;
          align-content: start;
          min-width: 0;
          min-height: 92px;
          padding: .65rem .7rem;
          border: 1px solid var(--line);
          border-left: 4px solid var(--blue);
          border-radius: 6px;
          background: var(--surface-gradient);
        }
        .recognized-element.clear { border-left-color: var(--green); }
        .recognized-element.attention {
          border-left-color: var(--amber);
          background: linear-gradient(135deg, rgba(244, 185, 66, .09), #1d1c19 68%);
        }
        .recognized-element > span {
          color: var(--muted);
          font-size: .72rem;
        }
        .recognized-element > strong {
          margin-top: .25rem;
          color: var(--ink);
          font-size: .86rem;
          overflow-wrap: anywhere;
        }
        .recognized-element > small {
          display: flex;
          align-items: center;
          gap: .3rem;
          margin-top: auto;
          padding-top: .45rem;
          color: var(--muted);
          font-size: .7rem;
        }
        .recognized-element > small i {
          width: 7px;
          height: 7px;
          flex: 0 0 7px;
          border-radius: 50%;
          background: var(--blue);
        }
        .recognized-element.clear > small i { background: var(--green); }
        .recognized-element.attention > small i { background: var(--amber); }
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
        .laboratory-synthesis {
          position: relative;
          overflow: hidden;
          margin: .35rem 0 1rem;
          padding: 1rem 1.1rem 1rem 1.2rem;
          border: 1px solid #31526d;
          border-left: 6px solid var(--cyan);
          border-radius: 7px;
          background: linear-gradient(135deg, rgba(69, 214, 230, .1), #161d24 56%);
        }
        .laboratory-synthesis::after {
          content: "";
          position: absolute;
          top: 0;
          right: 0;
          width: 5px;
          height: 100%;
          background: var(--blue);
          opacity: .8;
        }
        .laboratory-synthesis.unavailable {
          border-color: var(--line);
          border-left-color: #77747d;
          background: #19191d;
        }
        .laboratory-synthesis > span,
        .laboratory-synthesis > strong,
        .laboratory-synthesis > small {
          display: block;
        }
        .laboratory-synthesis > span {
          margin-bottom: .35rem;
          color: var(--cyan);
          font-size: .67rem;
          font-weight: 850;
          text-transform: uppercase;
        }
        .laboratory-synthesis > strong {
          max-width: 1100px;
          font-size: 1.02rem;
          line-height: 1.35;
        }
        .laboratory-synthesis p {
          max-width: 1180px;
          margin: .45rem 0 0;
          color: #d0ccc5;
          font-size: .86rem;
          line-height: 1.5;
        }
        .laboratory-synthesis.compact {
          margin: .55rem 0 .7rem;
          padding: .75rem .85rem .8rem 1rem;
        }
        .laboratory-synthesis.compact p {
          margin-top: .3rem;
          line-height: 1.45;
        }
        .laboratory-synthesis ul {
          display: flex;
          flex-wrap: wrap;
          gap: .35rem;
          padding: 0;
          margin: .65rem 0 0;
          list-style: none;
        }
        .laboratory-synthesis li {
          padding: .32rem .46rem;
          border: 1px solid #3c5965;
          border-radius: 4px;
          background: #17252b;
          color: #cdeff4;
          font-size: .68rem;
        }
        .laboratory-synthesis > small {
          margin-top: .65rem;
          color: var(--muted);
          font-size: .64rem;
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
          border: 1px solid #665527;
          border-left: 4px solid var(--amber);
          border-radius: 5px;
          background: #211d16;
        }
        .verification-issue.omission {
          border-color: #665527;
          border-left-color: var(--amber);
          background: #211d16;
        }
        .verification-issue span,
        .verification-issue small {
          color: var(--muted);
          font-size: .7rem;
        }
        .verification-issue strong {
          font-size: .8rem;
          overflow-wrap: anywhere;
        }
        .extraction-overview {
          display: grid;
          grid-template-columns: minmax(240px, 1.5fr) repeat(3, minmax(130px, 1fr));
          gap: .55rem;
          margin-bottom: .85rem;
        }
        .st-key-laboratory_context .extraction-overview {
          grid-template-columns: repeat(2, minmax(0, 1fr));
          margin-top: .75rem;
        }
        .st-key-laboratory_context .extraction-overview > article {
          min-height: 92px;
        }
        .st-key-laboratory_context .verification-summary {
          margin-top: .65rem;
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
          font-size: .74rem;
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
          font-size: .78rem;
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
          font-size: .68rem;
          text-transform: uppercase;
        }
        .extraction-table th span {
          display: block;
          margin-top: .16rem;
          color: var(--cyan);
          font-size: .66rem;
          text-transform: none;
        }
        .extraction-table td strong,
        .extraction-table td span {
          display: block;
        }
        .extraction-table td span {
          margin-top: .15rem;
          color: var(--muted);
          font-size: .7rem;
        }
        .extraction-table td .extraction-row-role {
          display: inline-flex;
          margin: 0;
          padding: .2rem .36rem;
          border: 1px solid #3c5965;
          border-radius: 4px;
          color: #bdebf3;
          background: #17252b;
          font-size: .68rem;
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
          font-size: .7rem;
        }
        .additional-extraction strong {
          margin: .2rem 0;
          overflow-wrap: anywhere;
          font-size: .82rem;
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
          .recognized-elements,
          .additional-extractions,
          .extraction-overview,
          .indicator-queue {
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
          .analysis-progress-track i::after {
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
          .fraud-score,
          .evidence-card,
          .indicator-step {
            transition: none;
          }
          .fraud-score {
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
        /* Classification card styles */
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
