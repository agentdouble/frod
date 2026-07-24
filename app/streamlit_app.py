"""Local Streamlit review console for Frod analysis reports."""

from __future__ import annotations

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
from fraude_detector.models import AnalysisReport, Finding, ImageAnalysisReport
from fraude_detector.pipeline import AnalysisPipeline

WORK_DIR = Path(os.environ.get("FROD_WORK_DIR", ".frod"))
UPLOAD_DIR = WORK_DIR / "uploads"
RUN_DIR = WORK_DIR / "runs"
GAPL_WEIGHTS = Path(os.environ.get("FROD_GAPL_WEIGHTS", str(GAPL_DEFAULT_CHECKPOINT)))
ANALYSIS_POLICY_VERSION = "gapl-p25-90-v2"

DEMO_DOCUMENTS = {
    "Document intact": Path("tests/fixtures/assurance-sans-fraude.pdf"),
    "Ajout legitime": Path("tests/fixtures/assurance-ajout-legitime.pdf"),
    "Montant modifie": Path("tests/fixtures/assurance-fraude.pdf"),
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
                report, output_dir, input_type = _run_analysis(
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
            "output_dir": output_dir,
            "input_type": input_type,
        }
    elif cached is not None and cached.get("key") == current_key:
        report = cached["report"]
        output_dir = cached["output_dir"]
        input_type = cached["input_type"]
    else:
        st.session_state.pop("analysis", None)
        with results:
            _render_ready_state(uploaded_file.name)
        return

    with results:
        _render_report(report, output_dir, input_type)


def _render_input_panel() -> InputDocument | None:
    st.markdown('<div class="panel-title">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PDF ou image",
        type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "gif", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        document = InputDocument(name=uploaded_file.name, data=uploaded_file.getvalue())
    else:
        with st.expander("Documents de demonstration", expanded=False):
            demo_name = st.selectbox(
                "Exemple",
                ("Aucun", *DEMO_DOCUMENTS),
                label_visibility="collapsed",
            )
        if demo_name == "Aucun":
            document = None
        else:
            demo_path = DEMO_DOCUMENTS[demo_name]
            if not demo_path.is_file():
                st.error("Le document de demonstration est indisponible.")
                document = None
            else:
                document = InputDocument(name=demo_path.name, data=demo_path.read_bytes())

    st.markdown('<div class="panel-title minor">Options</div>', unsafe_allow_html=True)
    st.slider("Pages analysees", min_value=1, max_value=50, value=25, key="max_pages")
    st.select_slider("DPI rendu PDF", options=[72, 108, 144, 180, 216], value=144, key="dpi")
    return document


def _config_from_state() -> AnalysisConfig:
    return AnalysisConfig(
        render_dpi=int(st.session_state.get("dpi", 144)),
        max_pages=int(st.session_state.get("max_pages", 25)),
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
    Path,
    str,
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
    if GAPL_WEIGHTS.is_file():
        report_progress(0.06, "Chargement du modele GAPL")
        adapter = _load_gapl_adapter(
            str(GAPL_WEIGHTS.resolve()),
            best_available_device(),
        )
        adapters = (adapter,)

    def pipeline_progress(value: float, text: str) -> None:
        report_progress(0.12 + 0.88 * value, text)

    if _is_pdf_bytes(file_bytes):
        report = AnalysisPipeline(
            config=config,
            ai_image_adapters=adapters,
        ).analyze(
            source,
            output_dir,
            progress_callback=pipeline_progress,
        )
        return report, output_dir, "PDF"

    report = ImageAnalysisPipeline(
        config=config,
        ai_image_adapters=adapters,
    ).analyze(
        source,
        output_dir,
        progress_callback=pipeline_progress,
    )
    return report, output_dir, "Image"


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
    output_dir: Path,
    input_type: str,
) -> None:
    findings = sorted(
        report.findings,
        key=lambda finding: (finding.risk_points, finding.confidence),
        reverse=True,
    )
    scored = [finding for finding in findings if finding.risk_points > 0]
    diagnostics = [finding for finding in findings if finding.risk_points == 0]

    _render_score_header(report, input_type, len(scored))
    _render_detector_grid(report)
    _render_category_strips(scored)
    _render_findings(scored, diagnostics)
    _render_visual_artifacts(report, output_dir)
    _render_json(report, output_dir)


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
    input_type: str,
    scored_count: int,
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
            <div><strong>{scored_count}</strong><span>Indices retenus</span></div>
            <div><strong>{_html(input_type)}</strong><span>Type</span></div>
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
    detail = (
        f"Indice IA maximal : {max(gapl_indices):.0%}"
        if gapl_indices
        else f"{len(scored)} indice(s), {len(diagnostics)} diagnostic(s)"
    )
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
    confidence_label = (
        f"Indice IA : {float(gapl_index):.0%}"
        if gapl_index is not None
        else f"Confiance : {finding.confidence:.0%}"
    )
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
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
