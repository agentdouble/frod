"""Local Streamlit review console for Frod analysis reports."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from fraude_detector.config import AnalysisConfig
from fraude_detector.errors import AnalysisError
from fraude_detector.gapl import (
    GAPL_DEFAULT_CHECKPOINT,
    best_available_device,
    create_gapl_adapter,
)
from fraude_detector.image_pipeline import ImageAnalysisPipeline
from fraude_detector.laboratory import (
    analyze_image_laboratory,
    analyze_ocr_laboratory,
    analyze_pdf_laboratory,
)
from fraude_detector.models import (
    AnalysisReport,
    DetectorResult,
    Finding,
    ImageAnalysisReport,
    LaboratoryCheck,
    LaboratoryObservation,
    LaboratoryReport,
    OcrReport,
)
from fraude_detector.ocr_consistency import build_ocr_content_result
from fraude_detector.pipeline import AnalysisPipeline
from fraude_detector.trufor import TRUFOR_DEFAULT_CHECKPOINT, TRUFOR_MAX_PIXELS

WORK_DIR = Path(os.environ.get("FROD_WORK_DIR", ".frod"))
UPLOAD_DIR = WORK_DIR / "uploads"
RUN_DIR = WORK_DIR / "runs"
GAPL_WEIGHTS = Path(os.environ.get("FROD_GAPL_WEIGHTS", str(GAPL_DEFAULT_CHECKPOINT)))
TRUFOR_WEIGHTS = Path(os.environ.get("FROD_TRUFOR_WEIGHTS", str(TRUFOR_DEFAULT_CHECKPOINT)))
TRUFOR_PIXEL_BUDGET = int(os.environ.get("FROD_TRUFOR_MAX_PIXELS", str(TRUFOR_MAX_PIXELS)))
OCR_URL = os.environ.get("FROD_OCR_URL", "").strip()
ANALYSIS_POLICY_VERSION = f"gapl-p25-90-v2-trufor-lab-v1-ocr-content-v1-{bool(OCR_URL)}"

DEMO_DOCUMENTS = {
    "Document intact": Path("tests/fixtures/assurance-sans-fraude.pdf"),
    "Ajout legitime": Path("tests/fixtures/assurance-ajout-legitime.pdf"),
    "Montant modifie": Path("tests/fixtures/assurance-fraude.pdf"),
}
OCR_DEMO_DOCUMENTS = {
    "OCR - Facture de soins": Path("tests/fixtures/ocr/facture-soins"),
    "OCR - Declaration coherente": Path("tests/fixtures/ocr/declaration-coherente"),
    "OCR - Contrat bruite": Path("tests/fixtures/ocr/contrat-bruite"),
    "OCR - Releve bancaire a anomalies": Path("tests/fixtures/ocr/releve-bancaire-anomalies"),
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
        "label": "Eleve",
        "tone": "high",
        "color": "#fb7185",
        "background": "#3c0912",
        "border": "#f43f5e",
    },
}

CATEGORY_LABELS = {
    "analysis_quality": "Qualite analyse",
    "annotations": "Annotations",
    "content_consistency": "Coherence du contenu",
    "document_integrity": "Integrite document",
    "metadata": "Metadonnees",
    "page_composition": "Composition page",
    "provenance_integrity": "Provenance",
    "raster_forensics": "Forensic image",
    "revision_history": "Historique PDF",
    "revision_visual": "Diff revisions",
    "synthetic_media": "Media synthetique",
}

DETECTOR_LABELS = {
    "ai_generated_image": "Generation par IA",
    "image_provenance": "Provenance image",
    "ocr_content": "Coherence du contenu",
    "page_composition": "Composition",
    "pdf_structure": "Structure PDF",
    "raster_anomaly": "ELA JPEG",
    "revision_diff": "Diff revisions",
}

STATUS_LABELS = {
    "completed": "Analyse terminee",
    "not_applicable": "Non applicable",
    "partial": "Analyse partielle",
}


@dataclass(frozen=True, slots=True)
class InputDocument:
    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class OcrDemoDocument:
    name: str
    fixture_dir: Path


def main() -> None:
    st.set_page_config(
        page_title="Detection de fraude",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()

    st.markdown(
        """
        <section class="topbar">
          <div>
            <p class="eyebrow">Assurance</p>
            <h1>Detection de fraude documentaire</h1>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    controls, results = st.columns([0.34, 0.66], gap="large")
    with controls:
        uploaded_file = _render_input_panel()

    if uploaded_file is None:
        with results:
            _render_empty_state()
        return
    if isinstance(uploaded_file, OcrDemoDocument):
        _handle_ocr_demo(uploaded_file, controls, results)
        return

    file_bytes = uploaded_file.data
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    config = _config_from_state()

    with controls:
        analyze = st.button("Analyser le fichier", type="primary", width="stretch")

    current_key = (
        ANALYSIS_POLICY_VERSION,
        file_hash,
        uploaded_file.name,
        config.render_dpi,
        config.max_pages,
    )
    cached = st.session_state.get("analysis")
    if analyze:
        with results:
            progress = st.empty()
            _render_analysis_progress(progress, 0.0, "Preparation de l'analyse")
            try:
                report, laboratory, output_dir = _run_analysis(
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
                _render_analysis_progress(progress, 1.0, "Analyse terminee")
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
        }
    elif cached is not None and cached.get("key") == current_key:
        report = cached["report"]
        laboratory = cached.get("laboratory")
        output_dir = cached["output_dir"]
    else:
        st.session_state.pop("analysis", None)
        with results:
            _render_ready_state(uploaded_file.name)
        return

    with results:
        _render_report(report, laboratory, output_dir)


def _render_input_panel() -> InputDocument | OcrDemoDocument | None:
    st.markdown('<div class="panel-title">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PDF ou image",
        type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "gif", "bmp"],
        label_visibility="collapsed",
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
        with st.expander("Documents de demonstration", expanded=False):
            demo_name = st.selectbox(
                "Exemple",
                ("Aucun", *DEMO_DOCUMENTS, *available_ocr_demos),
                label_visibility="collapsed",
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

    if not isinstance(document, OcrDemoDocument):
        st.markdown('<div class="panel-title minor">Options</div>', unsafe_allow_html=True)
        st.slider("Pages analysees", min_value=1, max_value=50, value=25, key="max_pages")
        st.select_slider(
            "DPI rendu PDF",
            options=[72, 108, 144, 180, 216],
            value=144,
            key="dpi",
        )
    return document


def _handle_ocr_demo(document: OcrDemoDocument, controls: Any, results: Any) -> None:
    json_path = document.fixture_dir / "document.json"
    markdown_path = document.fixture_dir / "document.md"
    json_bytes = json_path.read_bytes()
    markdown = markdown_path.read_text(encoding="utf-8")
    fixture_hash = hashlib.sha256(json_bytes + markdown.encode("utf-8")).hexdigest()
    current_key = ("ocr-demo-v2", document.name, fixture_hash)

    with controls:
        analyze = st.button("Analyser les donnees OCR", type="primary", width="stretch")

    cached = st.session_state.get("analysis")
    if analyze:
        try:
            payload = json.loads(json_bytes)
        except json.JSONDecodeError as error:
            with results:
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
        st.session_state["analysis"] = {
            "key": current_key,
            "ocr_payload": payload,
            "ocr_markdown": markdown,
            "laboratory": laboratory,
            "ocr_detector": ocr_detector,
        }
    elif cached is not None and cached.get("key") == current_key:
        payload = cached["ocr_payload"]
        markdown = cached["ocr_markdown"]
        laboratory = cached["laboratory"]
        ocr_detector = cached["ocr_detector"]
    else:
        st.session_state.pop("analysis", None)
        with results:
            _render_ready_state(document.name)
        return

    with results:
        _render_ocr_demo_report(
            name=document.name,
            markdown=markdown,
            payload=payload,
            laboratory=laboratory,
            ocr_detector=ocr_detector,
        )


def _config_from_state() -> AnalysisConfig:
    return AnalysisConfig(
        render_dpi=int(st.session_state.get("dpi", 144)),
        max_pages=int(st.session_state.get("max_pages", 25)),
        ocr_enabled=bool(OCR_URL),
        ocr_url=OCR_URL or "http://127.0.0.1:8007",
    )


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

    adapters = ()
    gapl_adapter = None
    if GAPL_WEIGHTS.is_file():
        report_progress(0.06, "Chargement du modele GAPL")
        gapl_adapter = _load_gapl_adapter(
            str(GAPL_WEIGHTS.resolve()),
            best_available_device(),
        )
        adapters = (gapl_adapter,)

    if _is_pdf_bytes(file_bytes):

        def pipeline_progress(value: float, text: str) -> None:
            report_progress(0.12 + 0.62 * value, text)

        report = AnalysisPipeline(
            config=config,
            ai_image_adapters=adapters,
        ).analyze(
            source,
            output_dir,
            progress_callback=pipeline_progress,
        )

        laboratory = analyze_pdf_laboratory(
            source,
            output_dir,
            config=config,
            progress_callback=lambda value, text: report_progress(
                0.74 + 0.26 * value,
                text,
            ),
        )
        laboratory = _with_ocr_laboratory(report, laboratory, output_dir)
        return report, laboratory, output_dir

    def pipeline_progress(value: float, text: str) -> None:
        report_progress(0.12 + 0.60 * value, text)

    report = ImageAnalysisPipeline(
        config=config,
        ai_image_adapters=adapters,
    ).analyze(
        source,
        output_dir,
        progress_callback=pipeline_progress,
    )
    _release_transient_memory()
    laboratory = analyze_image_laboratory(
        source,
        output_dir,
        trufor_weights=TRUFOR_WEIGHTS,
        trufor_max_pixels=TRUFOR_PIXEL_BUDGET,
        progress_callback=lambda value, text: report_progress(
            0.72 + 0.28 * value,
            text,
        ),
    )
    laboratory = _with_ocr_laboratory(report, laboratory, output_dir)
    return report, laboratory, output_dir


def _with_ocr_laboratory(
    report: AnalysisReport | ImageAnalysisReport,
    laboratory: LaboratoryReport,
    output_dir: Path,
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
    return LaboratoryReport(
        schema_version=laboratory.schema_version,
        checks=(*laboratory.checks, *analyze_ocr_laboratory(payload)),
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


def _render_ready_state(filename: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state ready">
          <h2>{_html(filename)}</h2>
          <p>Fichier pret pour analyse.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_report(
    report: AnalysisReport | ImageAnalysisReport,
    laboratory: LaboratoryReport | None,
    output_dir: Path,
) -> None:
    findings = sorted(
        report.findings,
        key=lambda finding: (finding.risk_points, finding.confidence),
        reverse=True,
    )
    scored = [finding for finding in findings if finding.risk_points > 0]
    diagnostics = [finding for finding in findings if finding.risk_points == 0]

    analysis_tab, ocr_tab, laboratory_tab = st.tabs(
        ["Analyse", "OCR", "Laboratoire"],
    )
    with analysis_tab:
        _render_score_header(report)
        _render_detector_grid(report)
        _render_category_strips(scored)
        _render_findings(scored, diagnostics)
        _render_visual_artifacts(report, output_dir)
        _render_json(report, output_dir)

    with ocr_tab:
        _render_ocr_results(report, output_dir)

    with laboratory_tab:
        _render_laboratory(laboratory, output_dir)


def _render_ocr_demo_report(
    *,
    name: str,
    markdown: str,
    payload: Any,
    laboratory: LaboratoryReport,
    ocr_detector: DetectorResult,
) -> None:
    analysis_tab, ocr_tab, laboratory_tab = st.tabs(
        ["Analyse", "OCR", "Laboratoire"],
    )
    with analysis_tab:
        finding = ocr_detector.findings[0] if ocr_detector.findings else None
        points = finding.risk_points if finding is not None else 0.0
        reliability = finding.confidence if finding is not None else None
        level = "Revue manuelle" if points >= 30 else "Controle complementaire"
        reliability_text = f"{reliability:.0%}" if reliability is not None else "-"
        st.markdown(
            f"""
            <section class="ocr-demo-state">
              <strong>{_html(name)}</strong>
              <span>{points:g}/30 points contenu - {level} - OCR {reliability_text}</span>
            </section>
            """,
            unsafe_allow_html=True,
        )
        _render_detector_card(ocr_detector)
        if finding is not None:
            _render_findings(
                [finding] if finding.risk_points > 0 else [],
                [finding] if finding.risk_points == 0 else [],
            )
    with ocr_tab:
        _render_ocr_payload(markdown, payload)
    with laboratory_tab:
        _render_laboratory(laboratory, Path("."))


def _render_ocr_results(
    report: AnalysisReport | ImageAnalysisReport,
    output_dir: Path,
) -> None:
    markdown_paths = _existing_artifacts(
        output_dir,
        report.artifacts.get("ocr_markdown", ()),
    )
    if not markdown_paths:
        st.info("Aucun contenu OCR disponible pour cette analyse.")
        return

    markdown = markdown_paths[0].read_text(encoding="utf-8")
    layout_images = _existing_artifacts(
        output_dir,
        report.artifacts.get("ocr_layout", ()),
    )
    json_paths = _existing_artifacts(
        output_dir,
        report.artifacts.get("ocr_json", ()),
    )
    payload: Any = None
    if json_paths:
        try:
            payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    _render_ocr_payload(markdown, payload, layout_images)


def _render_ocr_payload(
    markdown: str,
    payload: Any,
    layout_images: list[Path] | tuple[Path, ...] = (),
) -> None:
    st.markdown('<h2 class="section-title">Texte reconnu</h2>', unsafe_allow_html=True)
    st.markdown(markdown)

    if layout_images:
        st.markdown('<h2 class="section-title">Zones reconnues</h2>', unsafe_allow_html=True)
        for page_index, path in enumerate(layout_images, start=1):
            st.image(
                str(path),
                caption=f"Page {page_index}",
                width="stretch",
            )

    if payload is not None:
        with st.popover("Donnees structurees"):
            st.json(payload, expanded=False)


LAB_STATE_LABELS = {
    "clear": "Conforme",
    "attention": "A examiner",
    "detected": "Element detecte",
    "indeterminate": "Indetermine",
    "not_applicable": "Non applicable",
    "error": "Controle interrompu",
}

LAB_STRENGTH_LABELS = {
    "strong": "Indice fort",
    "moderate": "Indice modere",
    "weak": "Indice faible",
    "informational": "Information",
}


def _render_laboratory(
    laboratory: LaboratoryReport | None,
    output_dir: Path,
) -> None:
    if laboratory is None:
        st.info("Aucun controle experimental disponible.")
        return

    st.markdown('<h2 class="section-title">Laboratoire</h2>', unsafe_allow_html=True)
    ocr_checks = tuple(check for check in laboratory.checks if check.code.startswith("ocr_"))
    document_checks = tuple(
        check for check in laboratory.checks if not check.code.startswith("ocr_")
    )

    if ocr_checks:
        st.markdown(
            '<h3 class="laboratory-group-title">Analyse OCR solo</h3>',
            unsafe_allow_html=True,
        )
        for check in ocr_checks:
            _render_laboratory_check(check, output_dir)

    if document_checks:
        st.markdown(
            '<h3 class="laboratory-group-title">Controles documentaires</h3>',
            unsafe_allow_html=True,
        )
    for check in document_checks:
        _render_laboratory_check(check, output_dir)


def _render_laboratory_check(check: LaboratoryCheck, output_dir: Path) -> None:
    state_label = LAB_STATE_LABELS[check.state]
    st.markdown(
        f"""
        <section class="lab-check {check.state}">
          <div class="lab-check-head">
            <h3>{_html(check.title)}</h3>
            <span>{_html(state_label)}</span>
          </div>
          <p class="lab-purpose">{_html(check.purpose)}</p>
          <strong class="lab-summary">{_html(check.summary)}</strong>
        </section>
        """,
        unsafe_allow_html=True,
    )

    for observation in check.observations:
        _render_laboratory_observation(observation, output_dir)

    if check.limitations:
        with st.popover(f"Limites - {check.title}"):
            for limitation in check.limitations:
                st.markdown(f"- {limitation}")


def _render_laboratory_observation(
    observation: LaboratoryObservation,
    output_dir: Path,
) -> None:
    strength = LAB_STRENGTH_LABELS[observation.strength]
    location = f"Page {observation.page}" if observation.page is not None else "Document"
    st.markdown(
        f"""
        <article class="lab-observation {observation.state}">
          <div class="lab-observation-head">
            <div>
              <span class="lab-strength {observation.strength}">{_html(strength)}</span>
              <span class="lab-location">{_html(location)}</span>
            </div>
            <strong>{_html(observation.title)}</strong>
          </div>
          <p>{_html(observation.summary)}</p>
          <small>{_html(observation.explanation)}</small>
        </article>
        """,
        unsafe_allow_html=True,
    )

    details = bool(observation.evidence)
    artifacts = _existing_artifacts(
        output_dir / "laboratory",
        observation.artifacts,
    )
    visible_artifacts = []
    detail_artifacts = artifacts
    if observation.code == "TRUFOR_LOCAL_MANIPULATION":
        visible_artifacts = [
            path
            for path in artifacts
            if path.name
            in {
                "trufor-localization-map.png",
                "trufor-reliable-map.png",
            }
        ]
        detail_artifacts = [path for path in artifacts if path not in visible_artifacts]

    if visible_artifacts:
        columns = st.columns(len(visible_artifacts))
        for index, path in enumerate(visible_artifacts):
            with columns[index]:
                st.image(
                    str(path),
                    caption=_laboratory_artifact_caption(path),
                    width="stretch",
                )

    if details or detail_artifacts:
        with st.popover(f"Details - {observation.title}"):
            if details:
                st.json(_json_safe(observation.evidence), expanded=False)
            for path in detail_artifacts:
                if path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                st.image(
                    str(path),
                    caption=_laboratory_artifact_caption(path),
                    width="stretch",
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


def _render_score_header(
    report: AnalysisReport | ImageAnalysisReport,
) -> None:
    assessment = report.assessment
    style = LEVEL_STYLE[assessment.level]
    circumference = max(0, min(100, assessment.score)) * 3.6
    subject = report.document if isinstance(report, AnalysisReport) else report.image
    analyzed_pages = getattr(subject, "analyzed_pages", None)
    page_line = (
        f"{analyzed_pages}/{subject.page_count} pages"
        if analyzed_pages is not None
        else f"{subject.width} x {subject.height}px"
    )
    gapl_findings = [
        finding for finding in report.findings if finding.code == "AI_GAPL_GLOBAL_TRACE"
    ]
    gapl_indices = [
        float(finding.evidence["global_index"])
        for finding in gapl_findings
        if "global_index" in finding.evidence
    ]
    gapl_index = max(gapl_indices, default=None)
    gapl_index_text = f"{gapl_index:.0%}" if gapl_index is not None else "-"
    gapl_points = max(
        (finding.risk_points for finding in gapl_findings),
        default=0.0,
    )
    content_findings = [finding for finding in report.findings if finding.detector == "ocr_content"]
    content_points = max(
        (finding.risk_points for finding in content_findings),
        default=0.0,
    )
    content_reliability = max(
        (finding.confidence for finding in content_findings),
        default=None,
    )
    content_reliability_text = (
        f"{content_reliability:.0%}" if content_reliability is not None else "-"
    )
    st.markdown(
        f"""
        <section class="score-hero {style["tone"]}">
          <div class="score-ring" style="--score-angle: {circumference}deg;
              --score-color: {style["color"]}; --score-bg: {style["background"]};">
            <span>{assessment.score}</span>
            <small>/100</small>
          </div>
          <div class="score-copy">
            <p class="eyebrow">Score</p>
            <h2>{_html(assessment.label)}</h2>
            <p>{_html(assessment.explanation)}</p>
          </div>
          <div class="score-meta">
            <div><strong>{_html(style["label"])}</strong><span>Niveau</span></div>
            <div class="ai-metric"><strong>{gapl_index_text}</strong><span>Indice IA</span></div>
            <div><strong>{gapl_points:g}</strong><span>Points IA</span></div>
            <div><strong>{content_points:g}</strong><span>Points contenu</span></div>
            <div><strong>{content_reliability_text}</strong><span>Fiabilite OCR</span></div>
            <div><strong>{_html(page_line)}</strong><span>Perimetre</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_detector_grid(report: AnalysisReport | ImageAnalysisReport) -> None:
    st.markdown('<h2 class="section-title">Indicateurs</h2>', unsafe_allow_html=True)
    for index in range(0, len(report.detectors), 3):
        columns = st.columns(3, gap="medium")
        for column, detector in zip(columns, report.detectors[index : index + 3], strict=False):
            with column:
                _render_detector_card(detector)


def _render_detector_card(detector: Any) -> None:
    scored = [finding for finding in detector.findings if finding.risk_points > 0]
    diagnostics = [finding for finding in detector.findings if finding.risk_points == 0]
    max_points = max((finding.risk_points for finding in scored), default=0)
    tone = _tone_for_points(max_points, detector.status)
    state = "Detecte" if scored else ("Diagnostic" if diagnostics else "Rien detecte")
    label = DETECTOR_LABELS.get(detector.name, detector.name.replace("_", " ").title())
    gapl_indices = [
        float(finding.evidence["global_index"])
        for finding in detector.findings
        if finding.code == "AI_GAPL_GLOBAL_TRACE" and "global_index" in finding.evidence
    ]
    if gapl_indices:
        detail = f"Indice IA maximal : {max(gapl_indices):.0%}"
    elif detector.name == "ocr_content" and detector.findings:
        reliability = max(item.confidence for item in detector.findings)
        detail = f"Fiabilite OCR : {reliability:.0%}"
    else:
        detail = f"{len(scored)} indice(s), {len(diagnostics)} diagnostic(s)"
    st.markdown(
        f"""
        <div class="detector-card {tone}">
          <div class="detector-head">
            <strong>{_html(label)}</strong>
            <span>{_html(STATUS_LABELS.get(detector.status, detector.status))}</span>
          </div>
          <div class="detector-body">
            <span class="detector-state">{_html(state)}</span>
            <span class="detector-points">{max_points:g} pts max</span>
          </div>
          <p>{_html(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_category_strips(scored: list[Finding]) -> None:
    if not scored:
        st.markdown(
            """
            <div class="clean-band">
              <strong>Aucun signal score detecte.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    by_category: dict[str, list[Finding]] = defaultdict(list)
    for finding in scored:
        by_category[finding.category].append(finding)

    st.markdown('<h2 class="section-title">Familles</h2>', unsafe_allow_html=True)
    strips = []
    for category, items in sorted(
        by_category.items(),
        key=lambda item: max(finding.risk_points for finding in item[1]),
        reverse=True,
    ):
        max_points = max(finding.risk_points for finding in items)
        percent = min(100, round(max_points))
        strips.append(
            f"""
            <div class="category-strip {_tone_for_points(max_points, "completed")}">
              <div>
                <strong>{_html(CATEGORY_LABELS.get(category, category))}</strong>
                <span>{len(items)} signal(aux), max {max_points:g} pts</span>
              </div>
              <div class="bar"><span style="width:{percent}%"></span></div>
            </div>
            """
        )
    st.markdown("\n".join(strips), unsafe_allow_html=True)


def _render_findings(scored: list[Finding], diagnostics: list[Finding]) -> None:
    st.markdown('<h2 class="section-title">Indices</h2>', unsafe_allow_html=True)
    if not scored:
        st.info("Aucun indice score.")
    for finding in scored:
        _finding_card(finding)

    if diagnostics:
        with st.expander(f"Diagnostics non scores ({len(diagnostics)})", expanded=False):
            for finding in diagnostics:
                _finding_card(finding, diagnostic=True)


def _finding_card(finding: Finding, *, diagnostic: bool = False) -> None:
    tone = "neutral" if diagnostic else _tone_for_points(finding.risk_points, "completed")
    location = "Document"
    if finding.page is not None:
        location = f"Page {finding.page}"
        if finding.bbox is not None:
            location += " - zone localisee"
    gapl_index = finding.evidence.get("global_index")
    if gapl_index is not None:
        confidence_label = f"Indice IA : {float(gapl_index):.0%}"
    elif finding.detector == "ocr_content":
        confidence_label = f"Fiabilite OCR : {finding.confidence:.0%}"
    else:
        confidence_label = f"Confiance : {finding.confidence:.0%}"
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
            <h3>{_html(finding.title)}</h3>
            <p>{_html(finding.description)}</p>
            <div class="confidence">{_html(confidence_label)}</div>
          </div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    if finding.evidence:
        with st.expander(f"Preuves techniques - {finding.code}", expanded=False):
            st.json(_json_safe(finding.evidence), expanded=False)


def _render_visual_artifacts(
    report: AnalysisReport | ImageAnalysisReport,
    output_dir: Path,
) -> None:
    st.markdown('<h2 class="section-title">Zones a reviser</h2>', unsafe_allow_html=True)
    artifacts = report.artifacts
    review_images = _existing_artifacts(output_dir, artifacts.get("review_overlays", ()))
    forensic_images = [
        path
        for path in _existing_artifacts(output_dir, artifacts.get("forensics", ()))
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    gapl_images = [path for path in forensic_images if "-windows" in path.name]
    forensic_images = [path for path in forensic_images if path not in gapl_images]
    page_images = _existing_artifacts(output_dir, artifacts.get("page_renders", ()))[:3]

    if review_images:
        st.markdown('<div class="artifact-label">Zones a controler</div>', unsafe_allow_html=True)
        for index, path in enumerate(review_images, start=1):
            st.image(
                str(path),
                caption=f"Page {index} - zones a reviser",
                width="stretch",
            )
    else:
        st.markdown(
            """
            <div class="clean-band compact">
              <strong>Aucune zone localisee.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if forensic_images:
        st.markdown(
            '<div class="artifact-label">Images d\'analyse</div>',
            unsafe_allow_html=True,
        )
        columns = st.columns(2)
        for index, path in enumerate(forensic_images):
            with columns[index % 2]:
                st.image(
                    str(path),
                    caption=_friendly_artifact_caption(path),
                    width="stretch",
                )

    if gapl_images:
        with st.expander("Detail de l'analyse IA", expanded=False):
            for path in gapl_images:
                st.image(
                    str(path),
                    caption="Scores GAPL par zone analysee",
                    width="stretch",
                )

    if page_images:
        with st.expander("Rendus de pages", expanded=False):
            columns = st.columns(min(3, len(page_images)))
            for index, path in enumerate(page_images):
                with columns[index % len(columns)]:
                    st.image(
                        str(path),
                        caption=f"Page {index + 1}",
                        width="stretch",
                    )


def _render_json(report: AnalysisReport | ImageAnalysisReport, output_dir: Path) -> None:
    with st.expander("Dossier technique", expanded=False):
        st.json(report.to_dict(), expanded=False)
        if report.limitations:
            st.markdown("#### Limites")
            for limitation in report.limitations:
                st.markdown(f"- {limitation}")
        json_artifacts = [
            path
            for path in _existing_artifacts(output_dir, report.artifacts.get("forensics", ()))
            if path.suffix.casefold() == ".json"
        ]
        for path in json_artifacts:
            with st.expander(_friendly_artifact_caption(path), expanded=False):
                try:
                    st.json(json.loads(path.read_text(encoding="utf-8")), expanded=False)
                except json.JSONDecodeError:
                    st.code(path.read_text(encoding="utf-8"))


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
    if "revision-diff" in name:
        return "Difference visuelle entre revisions"
    if "ela" in name:
        return "Carte d'anomalie JPEG"
    if "provenance" in name:
        return "Provenance et metadonnees"
    if "analysis" in name:
        return "Analyse image avancee"
    return "Artefact d'analyse"


def _laboratory_artifact_caption(path: Path) -> str:
    if "trufor-localization" in path.name:
        return "Carte de localisation TruFor"
    if "trufor-reliable" in path.name:
        return "Carte ponderee par la fiabilite"
    if "trufor-confidence" in path.name:
        return "Fiabilite locale - noir faible, blanc fort"
    if "revision" in path.name:
        return "Difference entre revisions"
    return "Resultat experimental"


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return asdict(value) if hasattr(value, "__dataclass_fields__") else str(value)


def _html(value: object) -> str:
    return (
        str(value)
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
          --ink: #f7f2ea;
          --muted: #b8afa3;
          --line: #3a343c;
          --panel: #1b1a20;
          --panel-soft: #242029;
          --bg: #111013;
        }
        header[data-testid="stHeader"] {
          display: none;
        }
        .stApp {
          background:
            linear-gradient(135deg, rgba(45, 212, 191, .16), transparent 28%),
            linear-gradient(215deg, rgba(244, 63, 94, .12), transparent 32%),
            linear-gradient(0deg, rgba(251, 191, 36, .07), transparent 44%),
            var(--bg);
          color: var(--ink);
        }
        .block-container {
          padding-top: 2rem;
          max-width: 1480px;
        }
        .topbar {
          display: flex;
          justify-content: space-between;
          gap: 2rem;
          align-items: end;
          border-bottom: 1px solid var(--line);
          padding-bottom: 1.1rem;
          margin-bottom: 1.4rem;
        }
        .topbar h1 {
          font-size: 2.25rem;
          line-height: 1.05;
          margin: 0;
          letter-spacing: 0;
        }
        .eyebrow {
          text-transform: uppercase;
          font-weight: 800;
          font-size: .78rem;
          color: #2dd4bf;
          margin: 0 0 .4rem;
          letter-spacing: .08em;
        }
        .panel-title {
          font-size: 1rem;
          font-weight: 850;
          margin: .2rem 0 .75rem;
        }
        .panel-title.minor {
          margin-top: 1.35rem;
        }
        .empty-state {
          min-height: 430px;
          display: grid;
          align-content: center;
          border: 1px dashed #7dd3fc;
          background: rgba(27, 26, 32, .82);
          border-radius: 8px;
          padding: 3rem;
        }
        .empty-state.ready {
          border-color: #2dd4bf;
          background: rgba(16, 78, 71, .45);
        }
        .empty-state h2 {
          font-size: 2rem;
          margin: 0 0 .75rem;
        }
        .empty-state p {
          color: var(--muted);
          max-width: 680px;
          margin: 0;
        }
        .ocr-demo-state {
          min-height: 180px;
          display: grid;
          align-content: center;
          gap: .35rem;
          border: 1px solid var(--line);
          border-left: 7px solid #38bdf8;
          border-radius: 8px;
          background: #132635;
          padding: 1.25rem;
          margin-top: 1rem;
        }
        .ocr-demo-state strong {
          font-size: 1.15rem;
        }
        .ocr-demo-state span {
          color: var(--muted);
        }
        .score-hero {
          display: grid;
          grid-template-columns: auto minmax(260px, 1fr) minmax(220px, .55fr);
          gap: 1.25rem;
          align-items: center;
          background: var(--panel);
          border-radius: 8px;
          border: 1px solid var(--line);
          padding: 1.25rem;
          box-shadow: 0 18px 50px rgba(0, 0, 0, .24);
        }
        .score-hero.low { border-top: 8px solid #16a34a; }
        .score-hero.review { border-top: 8px solid #d97706; }
        .score-hero.high { border-top: 8px solid #dc2626; }
        .score-ring {
          width: 148px;
          height: 148px;
          border-radius: 50%;
          display: grid;
          place-content: center;
          background:
            radial-gradient(circle at center, #1b1a20 0 58%, transparent 59%),
            conic-gradient(
              var(--score-color) 0 var(--score-angle),
              #3b3741 var(--score-angle) 360deg
            );
          border: 1px solid rgba(255, 255, 255, .10);
        }
        .score-ring span {
          font-size: 3rem;
          font-weight: 900;
          line-height: .9;
          color: var(--score-color);
          text-align: center;
        }
        .score-ring small {
          text-align: center;
          color: var(--muted);
          font-weight: 700;
        }
        .score-copy h2 {
          font-size: 1.8rem;
          line-height: 1.15;
          margin: 0 0 .55rem;
        }
        .score-copy p:not(.eyebrow) {
          color: var(--muted);
          margin: 0;
        }
        .score-meta {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: .7rem;
        }
        .score-meta div {
          background: var(--panel-soft);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: .75rem;
        }
        .score-meta strong, .score-meta span {
          display: block;
        }
        .score-meta strong {
          font-size: 1.05rem;
        }
        .score-meta .ai-metric strong {
          color: #67e8f9;
          font-size: 1.35rem;
        }
        .score-meta span {
          color: var(--muted);
          font-size: .78rem;
          margin-top: .2rem;
        }
        .section-title {
          font-size: 1.22rem;
          margin: 1.55rem 0 .75rem;
          letter-spacing: 0;
        }
        .laboratory-group-title {
          color: #f4ede4;
          font-size: 1rem;
          margin: 1.35rem 0 .25rem;
          padding-bottom: .55rem;
          border-bottom: 1px solid var(--line);
          letter-spacing: 0;
        }
        div[data-baseweb="tab-list"] {
          gap: .35rem;
          border-bottom: 1px solid var(--line);
          margin-bottom: .6rem;
        }
        button[data-baseweb="tab"] {
          min-height: 3rem;
          padding: 0 1rem;
          color: var(--muted);
          font-weight: 800;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
          color: #f4ede4;
          border-bottom-color: #2dd4bf;
        }
        .detector-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .75rem;
        }
        .detector-card,
        .finding-card,
        .category-strip,
        .clean-band {
          border-radius: 8px;
          border: 1px solid var(--line);
          background: var(--panel);
        }
        .detector-card {
          padding: .85rem;
          border-left-width: 7px;
        }
        .detector-card.danger { border-left-color: #fb7185; background: #2d171c; }
        .detector-card.warning { border-left-color: #fbbf24; background: #2b2110; }
        .detector-card.notice-tone { border-left-color: #38bdf8; background: #132635; }
        .detector-card.partial { border-left-color: #a78bfa; background: #211a34; }
        .detector-card.neutral { border-left-color: #64748b; background: #202027; }
        .detector-head {
          display: flex;
          justify-content: space-between;
          gap: .7rem;
        }
        .detector-head strong {
          font-size: .98rem;
        }
        .detector-head span {
          color: var(--muted);
          font-size: .78rem;
          white-space: nowrap;
        }
        .detector-body {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          margin-top: 1rem;
        }
        .detector-state {
          font-size: 1.1rem;
          font-weight: 850;
        }
        .detector-points {
          color: var(--muted);
          font-weight: 750;
        }
        .detector-card p {
          color: var(--muted);
          margin: .45rem 0 0;
          font-size: .85rem;
        }
        .category-strip {
          display: grid;
          grid-template-columns: minmax(210px, .45fr) 1fr;
          gap: 1rem;
          align-items: center;
          padding: .8rem .95rem;
          margin-bottom: .55rem;
          border-left-width: 7px;
        }
        .category-strip.danger { border-left-color: #fb7185; }
        .category-strip.warning { border-left-color: #fbbf24; }
        .category-strip.notice-tone { border-left-color: #38bdf8; }
        .category-strip strong, .category-strip span {
          display: block;
        }
        .category-strip span {
          color: var(--muted);
          font-size: .85rem;
          margin-top: .15rem;
        }
        .bar {
          height: 13px;
          border-radius: 999px;
          background: #3b3741;
          overflow: hidden;
        }
        .bar span {
          display: block;
          height: 100%;
          background: linear-gradient(90deg, #2dd4bf, #fbbf24, #fb7185);
        }
        .finding-card {
          display: grid;
          grid-template-columns: 82px 1fr;
          gap: 1rem;
          padding: 1rem;
          margin-bottom: .75rem;
          border-left-width: 7px;
        }
        .finding-card.danger { border-left-color: #fb7185; background: #2d171c; }
        .finding-card.warning { border-left-color: #fbbf24; background: #2b2110; }
        .finding-card.notice-tone { border-left-color: #38bdf8; background: #132635; }
        .finding-card.neutral { border-left-color: #64748b; background: #202027; }
        .finding-score {
          width: 72px;
          height: 72px;
          border-radius: 8px;
          background: #08070a;
          color: white;
          display: grid;
          place-content: center;
          text-align: center;
        }
        .finding-score strong {
          font-size: 1.55rem;
          line-height: .95;
        }
        .finding-score span {
          font-size: .72rem;
          color: #c7beb3;
        }
        .finding-kicker {
          display: flex;
          flex-wrap: wrap;
          gap: .45rem;
          margin-bottom: .45rem;
        }
        .finding-kicker span {
          border: 1px solid #554d58;
          border-radius: 999px;
          padding: .16rem .5rem;
          font-size: .72rem;
          font-weight: 750;
          color: #f4ede4;
          background: rgba(255, 255, 255, .07);
        }
        .finding-content h3 {
          margin: 0 0 .35rem;
          font-size: 1.15rem;
        }
        .finding-content p {
          margin: 0;
          color: #d0c8bd;
        }
        .confidence {
          margin-top: .6rem;
          font-weight: 800;
          color: #2dd4bf;
          font-size: .88rem;
        }
        .clean-band {
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          align-items: center;
          padding: 1rem;
          background: #0d2f28;
          border-color: #10b981;
          color: #d7fff4;
        }
        .clean-band.compact {
          margin-bottom: 1rem;
        }
        .artifact-label {
          font-size: .95rem;
          font-weight: 850;
          color: #d9d2c7;
          margin: .4rem 0 .55rem;
        }
        div[data-testid="stFileUploader"] {
          background: var(--panel);
          border: 2px dashed #2dd4bf;
          border-radius: 8px;
          padding: .65rem;
        }
        .stButton > button {
          border-radius: 8px;
          font-weight: 850;
          min-height: 3rem;
        }
        .analysis-progress {
          border: 1px solid var(--line);
          border-radius: 8px;
          background: #151419;
          padding: .75rem 1rem;
          margin: .5rem 0 1rem;
        }
        .analysis-progress-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          color: #d9d2c7;
          font-size: .82rem;
          margin-bottom: .5rem;
        }
        .analysis-progress-head strong {
          color: #f4ede4;
        }
        .analysis-progress-track {
          height: 9px;
          border-radius: 999px;
          background: #302e34;
          overflow: hidden;
        }
        .analysis-progress-track i {
          display: block;
          height: 100%;
          border-radius: inherit;
          background: #2dd4bf;
          transition: width .18s ease;
        }
        .lab-intro {
          color: #d0c8bd;
          border-left: 4px solid #38bdf8;
          background: #132635;
          padding: .75rem 1rem;
          margin-bottom: 1rem;
        }
        .lab-check {
          border: 1px solid var(--line);
          border-left: 7px solid #64748b;
          border-radius: 8px;
          background: var(--panel);
          padding: 1rem;
          margin-top: 1.15rem;
        }
        .lab-check.clear { border-left-color: #10b981; }
        .lab-check.attention { border-left-color: #fb7185; background: #2d171c; }
        .lab-check.detected { border-left-color: #38bdf8; background: #132635; }
        .lab-check.indeterminate { border-left-color: #fbbf24; background: #2b2110; }
        .lab-check.error { border-left-color: #a78bfa; background: #211a34; }
        .lab-check-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
        }
        .lab-check-head h3 {
          margin: 0;
          font-size: 1.14rem;
        }
        .lab-check-head span {
          border: 1px solid #5d5662;
          border-radius: 999px;
          padding: .2rem .55rem;
          color: #f4ede4;
          font-size: .74rem;
          font-weight: 800;
          white-space: nowrap;
        }
        .lab-purpose {
          color: var(--muted);
          margin: .55rem 0 .7rem;
          font-size: .87rem;
        }
        .lab-summary {
          display: block;
          color: #f7f2ea;
        }
        .lab-observation {
          border: 1px solid #423d46;
          border-left: 4px solid #64748b;
          background: #1d1c22;
          padding: .85rem 1rem;
          margin: .5rem 0 0 1rem;
        }
        .lab-observation.clear { border-left-color: #10b981; }
        .lab-observation.attention { border-left-color: #fb7185; }
        .lab-observation.detected { border-left-color: #38bdf8; }
        .lab-observation.indeterminate { border-left-color: #fbbf24; }
        .lab-observation-head {
          display: grid;
          grid-template-columns: minmax(170px, .34fr) 1fr;
          gap: .8rem;
          align-items: center;
        }
        .lab-observation-head > div {
          display: flex;
          gap: .45rem;
          flex-wrap: wrap;
        }
        .lab-strength,
        .lab-location {
          border-radius: 999px;
          padding: .18rem .5rem;
          font-size: .7rem;
          font-weight: 800;
        }
        .lab-strength.strong { background: #5b1320; color: #fecdd3; }
        .lab-strength.moderate { background: #4a3108; color: #fde68a; }
        .lab-strength.weak { background: #193247; color: #bae6fd; }
        .lab-strength.informational { background: #303038; color: #d6d3d1; }
        .lab-location {
          border: 1px solid #554d58;
          color: #d6d3d1;
        }
        .lab-observation p {
          margin: .6rem 0 .3rem;
          color: #f0e9df;
        }
        .lab-observation small {
          display: block;
          color: var(--muted);
          line-height: 1.45;
        }
        @media (max-width: 980px) {
          .topbar,
          .score-hero,
          .category-strip {
            grid-template-columns: 1fr;
            display: grid;
          }



          .detector-grid {
            grid-template-columns: 1fr;
          }
          .score-meta {
            grid-template-columns: 1fr 1fr;
          }
          .lab-observation {
            margin-left: 0;
          }
          .lab-observation-head {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
