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
from pathlib import Path
from typing import Any

import streamlit as st

from fraude_detector.config import AnalysisConfig
from fraude_detector.errors import AnalysisError
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
from fraude_detector.models import (
    AnalysisReport,
    DetectorResult,
    Finding,
    ImageAnalysisReport,
    LaboratoryReport,
    OcrReport,
)
from fraude_detector.ocr_consistency import build_ocr_content_result
from fraude_detector.pipeline import AnalysisPipeline
from fraude_detector.run_config import load_run_config
from fraude_detector.scoring import FAMILY_CAPS

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
    f"gapl-p25-90-v2-trufor-lab-v1-ocr-content-v2-full-identifiers-{CONFIG_FINGERPRINT}"
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

CATEGORY_ACCENTS = {
    "annotations": "#4cc9d8",
    "content_consistency": "#35d07f",
    "document_integrity": "#70a7ff",
    "metadata": "#b89cff",
    "page_composition": "#f4b942",
    "provenance_integrity": "#55d6be",
    "raster_forensics": "#ff8a6b",
    "revision_history": "#8d9eff",
    "revision_visual": "#ff6b8a",
    "synthetic_media": "#d58cff",
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
        page_title="Analyse de fraude documentaire",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()

    st.markdown(
        """
        <section class="brandbar">
          <h1>Analyse de fraude documentaire</h1>
        </section>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = _render_input_panel()

    if uploaded_file is None:
        _render_empty_state()
        return
    if isinstance(uploaded_file, OcrDemoDocument):
        _handle_ocr_demo(uploaded_file)
        return

    file_bytes = uploaded_file.data
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    config = _config_from_state()

    action_space, action = st.columns([0.78, 0.22], gap="medium")
    with action_space:
        st.markdown(
            f'<div class="selected-file">{_html(uploaded_file.name)}</div>',
            unsafe_allow_html=True,
        )
    with action:
        analyze = st.button("Analyser le fichier", type="primary", width="stretch")
    progress = st.empty()

    current_key = (
        ANALYSIS_POLICY_VERSION,
        file_hash,
        uploaded_file.name,
    )
    cached = st.session_state.get("analysis")
    if analyze:
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
    elif cached is not None and cached.get("key") == current_key:
        report = cached["report"]
        laboratory = cached.get("laboratory")
        output_dir = cached["output_dir"]
        source_path = cached["source_path"]
    else:
        st.session_state.pop("analysis", None)
        _render_ready_state(uploaded_file.name)
        return

    _render_report(report, laboratory, output_dir, source_path)


def _render_input_panel() -> InputDocument | OcrDemoDocument | None:
    uploaded_file = st.file_uploader(
        "Déposer un document",
        type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "gif", "bmp"],
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

    return document


def _handle_ocr_demo(document: OcrDemoDocument) -> None:
    json_path = document.fixture_dir / "document.json"
    markdown_path = document.fixture_dir / "document.md"
    json_bytes = json_path.read_bytes()
    markdown = markdown_path.read_text(encoding="utf-8")
    fixture_hash = hashlib.sha256(json_bytes + markdown.encode("utf-8")).hexdigest()
    current_key = ("ocr-demo-v2", document.name, fixture_hash)

    action_space, action = st.columns([0.78, 0.22], gap="medium")
    with action_space:
        st.markdown(
            f'<div class="selected-file">{_html(document.name)}</div>',
            unsafe_allow_html=True,
        )
    with action:
        analyze = st.button("Analyser les données OCR", type="primary", width="stretch")

    cached = st.session_state.get("analysis")
    if analyze:
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
        _render_ready_state(document.name)
        return

    _render_ocr_demo_report(
        name=document.name,
        markdown=markdown,
        laboratory=laboratory,
        ocr_detector=ocr_detector,
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

    adapters = ()
    gapl_adapter = None
    if PROJECT_CONFIG.gapl.enabled and GAPL_WEIGHTS.is_file():
        report_progress(0.06, "Chargement de l'analyse des images générées par IA")
        device = (
            best_available_device()
            if PROJECT_CONFIG.gapl.device == "auto"
            else PROJECT_CONFIG.gapl.device
        )
        gapl_adapter = _load_gapl_adapter(
            str(GAPL_WEIGHTS.resolve()),
            device,
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
        laboratory = _with_ocr_laboratory(report, laboratory, output_dir)
        return report, laboratory, output_dir, source

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
    laboratory = _with_ocr_laboratory(report, laboratory, output_dir)
    return report, laboratory, output_dir, source


def _empty_laboratory_report() -> LaboratoryReport:
    return LaboratoryReport(schema_version="1.0", checks=())


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
          <p>Fichier prêt pour analyse.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_report(
    report: AnalysisReport | ImageAnalysisReport,
    laboratory: LaboratoryReport | None,
    output_dir: Path,
    source_path: Path,
) -> None:
    findings = sorted(
        report.findings,
        key=lambda finding: (finding.risk_points, finding.confidence),
        reverse=True,
    )
    scored = [finding for finding in findings if finding.risk_points > 0]
    diagnostics = [finding for finding in findings if finding.risk_points == 0]
    layout_images = _read_ocr_layout_images(report, output_dir)

    _render_score_header(report)
    _render_category_counters(report.findings)

    preview, review = st.columns([0.54, 0.46], gap="large")
    with preview:
        _render_document_view(
            report=report,
            laboratory=laboratory,
            output_dir=output_dir,
            source_path=source_path,
            layout_images=layout_images,
        )
    with review:
        _render_review_summary(scored, diagnostics, laboratory)


def _render_ocr_demo_report(
    *,
    name: str,
    markdown: str,
    laboratory: LaboratoryReport,
    ocr_detector: DetectorResult,
) -> None:
    finding = ocr_detector.findings[0] if ocr_detector.findings else None
    points = finding.risk_points if finding is not None else 0.0
    tone = "review" if points >= 30 else "low"
    color = LEVEL_STYLE[tone]["color"]
    angle = min(100, points / 30 * 100) * 3.6
    st.markdown(
        f"""
        <section class="score-hero {tone}">
          <div class="score-ring" style="--score-angle:{angle}deg;--score-color:{color};">
            <span>{points:g}</span><small>/30</small>
          </div>
          <div class="score-copy">
            <p class="eyebrow">Cohérence du contenu</p>
            <h2>{_html(name)}</h2>
            <p>Contrôle des identifiants, dates et calculs reconnus dans le document.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    preview, review = st.columns([0.54, 0.46], gap="large")
    with preview:
        st.markdown('<h2 class="workspace-title">Document reconnu</h2>', unsafe_allow_html=True)
        _render_recognized_text(markdown, compact=True)
    with review:
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


def _render_category_counters(findings: tuple[Finding, ...]) -> None:
    cards = []
    for category, cap in FAMILY_CAPS.items():
        points = _family_score(category, findings)
        percentage = min(100, points / cap * 100) if cap else 0
        tone = _tone_for_ratio(percentage)
        state = "À examiner" if points > 0 else "Aucun signal"
        accent = CATEGORY_ACCENTS.get(category, "#8b8791")
        cards.append(
            f"""
            <article class="category-counter {tone}" style="--category-accent:{accent}">
              <div class="mini-ring" style="--meter-angle:{percentage * 3.6}deg">
                <strong>{points:g}</strong><span>/{cap:g}</span>
              </div>
              <div class="category-copy">
                <strong>{_html(CATEGORY_LABELS.get(category, category))}</strong>
                <span class="category-state">{state}</span>
                <p>{_html(CATEGORY_DESCRIPTIONS.get(category, ""))}</p>
              </div>
            </article>
            """
        )
    st.markdown(
        '<section class="category-counters">'
        + "".join(card.strip() for card in cards)
        + "</section>",
        unsafe_allow_html=True,
    )


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


def _tone_for_ratio(percentage: float) -> str:
    if percentage >= 75:
        return "danger"
    if percentage >= 40:
        return "warning"
    if percentage > 0:
        return "notice-tone"
    return "clear"


def _render_document_view(
    *,
    report: AnalysisReport | ImageAnalysisReport,
    laboratory: LaboratoryReport | None,
    output_dir: Path,
    source_path: Path,
    layout_images: list[Path],
) -> None:
    st.markdown('<h2 class="workspace-title">Document</h2>', unsafe_allow_html=True)
    choices: dict[str, tuple[Path, str]] = {}
    if isinstance(report, ImageAnalysisReport) and source_path.is_file():
        choices["Document original"] = (
            source_path,
            "Vue de référence du fichier transmis, sans annotation ajoutée par l'analyse.",
        )

    page_images = _existing_artifacts(output_dir, report.artifacts.get("page_renders", ()))
    for index, path in enumerate(page_images, start=1):
        choices[f"Document - page {index}"] = (
            path,
            "Vue de référence de la page telle qu'elle a été analysée.",
        )

    review_images = _existing_artifacts(output_dir, report.artifacts.get("review_overlays", ()))
    for index, path in enumerate(review_images, start=1):
        choices[f"Zones à revoir - page {index}"] = (
            path,
            "Les cadres localisent les éléments associés aux signaux de la synthèse. "
            "Ils indiquent où regarder, mais ne constituent pas une preuve de fraude.",
        )

    for index, path in enumerate(layout_images, start=1):
        choices[f"Zones de texte reconnues - page {index}"] = (
            path,
            "Les cadres montrent les zones utilisées pour reconnaître le texte. "
            "Une mauvaise lecture reste possible, notamment sur les petits caractères.",
        )

    forensic_images = [
        path
        for path in _existing_artifacts(output_dir, report.artifacts.get("forensics", ()))
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    for path in forensic_images:
        choices[_friendly_artifact_caption(path)] = (
            path,
            _artifact_view_explanation(path),
        )

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
                        choices[_laboratory_artifact_caption(path)] = (
                            path,
                            _artifact_view_explanation(path),
                        )

    if not choices:
        st.info("Aucun apercu visuel disponible.")
        return

    labels = tuple(choices)
    selected = (
        st.selectbox("Vue affichée", labels, label_visibility="collapsed")
        if len(labels) > 1
        else labels[0]
    )
    selected_path, explanation = choices[selected]
    st.markdown(
        f"""
        <div class="view-guidance">
          <strong>Comment lire cette vue</strong>
          <p>{_html(explanation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
        if observation.code.startswith(
            ("OCR_CARD_", "OCR_IBAN_", "OCR_BIC_", "OCR_CKYC_", "OCR_MICR_")
        )
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
        if observation.state in {"attention", "detected", "error"}
        and not observation.code.startswith(
            ("OCR_CARD_", "OCR_IBAN_", "OCR_BIC_", "OCR_CKYC_", "OCR_MICR_")
        )
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
        cards.append(
            f"""
            <article class="control-status {check.state}">
              <span>{_html(LAB_STATE_LABELS[check.state])}</span>
              <strong>{_html(_business_check_title(check.code, check.title))}</strong>
              <p>{_html(_business_text(check.summary))}</p>
            </article>
            """
        )
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


def _render_score_header(
    report: AnalysisReport | ImageAnalysisReport,
) -> None:
    assessment = report.assessment
    style = LEVEL_STYLE[assessment.level]
    circumference = max(0, min(100, assessment.score)) * 3.6
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
    return "Visualisation complémentaire"


def _is_secondary_localization_artifact(path: Path) -> bool:
    return any(marker in path.name for marker in ("trufor-reliable", "trufor-confidence"))


def _artifact_view_explanation(path: Path) -> str:
    name = path.name
    if "trufor-localization" in name:
        return (
            "Les couleurs chaudes signalent des incohérences locales plus fortes ; "
            "le bleu correspond à des zones plus régulières. Un logo, une forte compression "
            "ou un traitement d'image légitime peut aussi produire une zone colorée : "
            "cette carte sert à orienter la revue, pas à conclure."
        )
    if "gapl" in name or "windows" in name or "heatmap" in name:
        return (
            "Chaque zone est comparée aux caractéristiques visuelles apprises sur des images "
            "générées par IA. Une valeur élevée indique une ressemblance statistique, sans "
            "prouver que la zone a été générée ou modifiée par IA."
        )
    if "revision-diff" in name or "revision" in name:
        return (
            "Les zones colorées correspondent aux différences conservées entre deux versions "
            "du fichier. Certaines peuvent provenir d'un réenregistrement ou d'une opération "
            "légitime."
        )
    if "ela" in name:
        return (
            "La carte amplifie les différences de compression JPEG. Une zone contrastée peut "
            "indiquer une retouche, mais aussi un assemblage, un logo ou des compressions "
            "successives."
        )
    return (
        "Cette vue complète l'examen visuel du document. Elle doit être interprétée avec les "
        "autres signaux et le contexte du dossier."
    )


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
        .brandbar {
          display: flex;
          align-items: center;
          border-bottom: 1px solid var(--line);
          padding: 0 0 .8rem;
          margin-bottom: .9rem;
        }
        .brandbar h1 {
          color: var(--ink);
          font-size: 1.5rem;
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
        .selected-file {
          min-height: 3rem;
          display: flex;
          align-items: center;
          color: #d7d4ce;
          border-bottom: 1px solid var(--line);
          font-weight: 700;
          overflow-wrap: anywhere;
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
        }
        .analysis-progress-track i {
          background: var(--cyan);
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
        .empty-state.ready {
          border-color: var(--line);
          border-left-color: var(--cyan);
          background: #181d20;
        }
        .empty-state h2 {
          font-size: 1.15rem;
          margin: 0 0 .3rem;
        }
        .empty-state p {
          margin: 0;
          font-size: .86rem;
        }
        .score-hero {
          display: grid;
          grid-template-columns: auto minmax(0, 1fr);
          gap: 1.35rem;
          align-items: center;
          min-height: 150px;
          padding: 1rem 1.3rem;
          margin-top: .9rem;
          background: #19191d;
          border: 1px solid var(--line);
          border-left: 7px solid #595760;
          border-radius: 8px;
          box-shadow: none;
        }
        .score-hero.low { border-top: 1px solid var(--line); border-left-color: var(--green); }
        .score-hero.review { border-top: 1px solid var(--line); border-left-color: var(--amber); }
        .score-hero.high { border-top: 1px solid var(--line); border-left-color: var(--coral); }
        .score-ring {
          width: 118px;
          height: 118px;
          background:
            radial-gradient(circle at center, #19191d 0 58%, transparent 59%),
            conic-gradient(var(--score-color) 0 var(--score-angle), #35343a var(--score-angle));
        }
        .score-ring span {
          font-size: 2.4rem;
        }
        .score-copy h2 {
          font-size: 1.45rem;
          margin: 0 0 .35rem;
        }
        .score-copy {
          max-width: 850px;
        }
        .score-copy p:not(.eyebrow) {
          font-size: .88rem;
          line-height: 1.45;
        }
        .score-meta {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .45rem;
        }
        .score-meta.compact-meta {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .score-meta div {
          min-height: 58px;
          padding: .55rem;
          background: #222127;
          border-radius: 6px;
        }
        .score-meta strong {
          font-size: .98rem;
        }
        .score-meta .ai-metric strong {
          color: var(--cyan);
          font-size: 1.1rem;
        }
        .category-counters {
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: .55rem;
          margin: .7rem 0 1rem;
        }
        .category-counter {
          display: grid;
          grid-template-columns: 48px minmax(0, 1fr);
          align-items: start;
          gap: .55rem;
          min-height: 104px;
          padding: .65rem;
          border: 1px solid var(--line);
          border-top: 2px solid var(--category-accent);
          border-radius: 6px;
          background: #19191d;
          overflow: hidden;
          transition: background-color .16s ease, border-color .16s ease;
        }
        .category-counter:hover {
          background: #1e1e23;
          border-color: var(--category-accent);
        }
        .category-counter.notice-tone,
        .category-counter.warning,
        .category-counter.danger {
          border-left-width: 4px;
          box-shadow: 0 5px 18px rgba(0, 0, 0, .2);
        }
        .category-counter.notice-tone {
          background: #171d25;
          border-left-color: var(--blue);
        }
        .category-counter.warning {
          background: #242015;
          border-left-color: var(--amber);
        }
        .category-counter.danger {
          background: #27181c;
          border-left-color: var(--coral);
        }
        .category-counter > div:last-child strong,
        .category-counter > div:last-child span {
          display: block;
        }
        .category-counter > div:last-child strong {
          font-size: .8rem;
          line-height: 1.25;
        }
        .category-counter > div:last-child span {
          color: var(--muted);
          font-size: .68rem;
          margin-top: .16rem;
        }
        .category-counter .category-state {
          color: var(--category-accent);
          font-weight: 750;
        }
        .category-counter.notice-tone .category-state,
        .category-counter.warning .category-state,
        .category-counter.danger .category-state {
          display: inline-flex;
          width: fit-content;
          border-radius: 999px;
          padding: .12rem .38rem;
          color: #17171a;
          font-weight: 850;
        }
        .category-counter.notice-tone .category-state {
          background: var(--blue);
        }
        .category-counter.warning .category-state {
          background: var(--amber);
        }
        .category-counter.danger .category-state {
          background: var(--coral);
        }
        .category-counter .category-copy p {
          color: #8f8b85;
          font-size: .64rem;
          line-height: 1.35;
          margin: .3rem 0 0;
        }
        .mini-ring {
          width: 44px;
          height: 44px;
          border-radius: 50%;
          display: grid;
          place-content: center;
          text-align: center;
          background:
            radial-gradient(circle at center, #19191d 0 59%, transparent 60%),
            conic-gradient(#66636c 0 var(--meter-angle), #343339 var(--meter-angle));
        }
        .category-counter.notice-tone .mini-ring {
          background:
            radial-gradient(circle at center, #19191d 0 59%, transparent 60%),
            conic-gradient(var(--blue) 0 var(--meter-angle), #343339 var(--meter-angle));
        }
        .category-counter.warning .mini-ring {
          background:
            radial-gradient(circle at center, #19191d 0 59%, transparent 60%),
            conic-gradient(var(--amber) 0 var(--meter-angle), #343339 var(--meter-angle));
        }
        .category-counter.danger .mini-ring {
          background:
            radial-gradient(circle at center, #19191d 0 59%, transparent 60%),
            conic-gradient(var(--coral) 0 var(--meter-angle), #343339 var(--meter-angle));
        }
        .mini-ring strong {
          font-size: .78rem;
          line-height: .8;
        }
        .mini-ring span {
          color: var(--muted);
          font-size: .52rem;
        }
        .workspace-title {
          font-size: 1.02rem;
          margin: .15rem 0 .55rem;
        }
        .subsection-title {
          font-size: .88rem;
          color: #dedbd5;
          margin: .9rem 0 .45rem;
          padding-bottom: .32rem;
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
        .view-guidance {
          border: 1px solid #363942;
          border-left: 4px solid var(--blue);
          border-radius: 6px;
          background: #171a20;
          padding: .7rem .8rem;
          margin-top: .35rem;
        }
        .view-guidance strong {
          display: block;
          color: #dbe7ff;
          font-size: .76rem;
          margin-bottom: .22rem;
        }
        .view-guidance p {
          color: #aaaeb8;
          font-size: .72rem;
          line-height: 1.45;
          margin: 0;
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
          width: 48px;
          height: 48px;
          border-radius: 4px;
          background: #0e0e11;
        }
        .finding-score strong {
          font-size: 1.15rem;
        }
        .finding-score span {
          font-size: .58rem;
        }
        .finding-content h3 {
          font-size: .94rem;
          margin: 0 0 .2rem;
        }
        .finding-content p {
          color: #c7c3bd;
          font-size: .78rem;
          line-height: 1.4;
        }
        .finding-kicker {
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
        .section-title {
          font-size: 1rem;
          margin: 1rem 0 .45rem;
        }
        @media (max-width: 1100px) {
          .category-counters {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .score-hero {
            grid-template-columns: auto 1fr;
          }
        }
        @media (max-width: 700px) {
          .block-container {
            padding: .8rem .75rem 1.5rem;
          }
          .brandbar {
            align-items: flex-start;
          }
          .score-hero {
            grid-template-columns: 1fr;
          }
          .score-ring {
            width: 100px;
            height: 100px;
          }
          .score-meta,
          .score-meta.compact-meta,
          .extracted-fields,
          .control-matrix {
            grid-template-columns: 1fr;
          }
          .category-counters {
            grid-template-columns: 1fr 1fr;
          }
          .review-banner {
            align-items: flex-start;
            flex-direction: column;
          }
          .review-banner span {
            text-align: left;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
