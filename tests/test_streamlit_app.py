from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests
from streamlit.testing.v1 import AppTest


def test_demo_runs_immediately_and_can_reset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    assert app.segmented_control[0].options == ["Analyse", "Laboratoire", "Glossaire"]
    assert app.segmented_control[0].value == "Analyse"
    assert len(app.file_uploader) == 1
    assert [selectbox.label for selectbox in app.selectbox] == ["Document de démonstration"]
    assert not app.button

    app.selectbox[0].select("Montant modifié").run(timeout=30)

    markdown_blocks = [element.value for element in app.markdown]
    markdown = "\n".join(markdown_blocks)
    assert not app.exception
    assert not app.file_uploader
    assert [selectbox.label for selectbox in app.selectbox] == ["Vue affichée"]
    assert [button.label for button in app.button] == ["Tester un nouveau document"]
    assert "Fichier prêt pour analyse" not in markdown
    assert not app.tabs
    assert app.segmented_control[0].value == "Analyse"
    assert not app.slider
    assert not app.select_slider

    app.button[0].click().run(timeout=30)

    assert len(app.file_uploader) == 1
    assert [selectbox.label for selectbox in app.selectbox] == ["Document de démonstration"]
    assert not app.button
    assert not app.tabs
    assert app.segmented_control[0].value == "Analyse"


def test_modified_demo_renders_single_review_workspace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("Montant modifié").run(timeout=30)

    markdown_blocks = [element.value for element in app.markdown]
    markdown = "\n".join(markdown_blocks)
    sequence = _component_markup(app, '<section class="indicator-sequence"')
    assert not app.exception
    assert "<h1>FROD</h1>" in markdown
    assert "Analyse de fraude documentaire" not in markdown
    assert [button.label for button in app.button] == ["Tester un nouveau document"]
    assert '<h2 class="workspace-title document-name">assurance-fraude.pdf</h2>' in markdown
    assert '<h2 class="workspace-title">Document</h2>' not in markdown
    assert "Synthèse de revue" in markdown
    assert not app.tabs
    assert app.segmented_control[0].value == "Analyse"
    assert 'class="analysis-handoff"' not in markdown
    assert "Document reçu" not in markdown
    assert "Préparation des contrôles" not in markdown
    assert 'class="pdf-loading-stage"' in markdown
    assert "Chargement du PDF&hellip;" in markdown
    assert "indicator-sequence" in sequence
    assert "Validation séquentielle" not in sequence
    assert "Contrôles du document" not in sequence
    assert 'aria-label="Résultats des indicateurs"' in sequence
    assert sequence.count('class="indicator-step ') == 10
    assert "01" in sequence
    assert "10" in sequence
    assert "En attente" in sequence
    assert "risk-animation-toggle" not in markdown
    assert 'aria-hidden="true"' in sequence
    assert "Modifications visuelles" in sequence
    assert "Différences visibles entre les versions du document." not in sequence
    assert "Structure du fichier" in sequence
    assert "Signatures, structure interne et altérations du fichier." not in sequence
    assert "fraud-score high" in markdown
    assert "Score de fraude" in markdown
    assert "fraud-score-number" in markdown
    assert "--score-target:71" in markdown
    assert "--score-duration:4.50s" in markdown
    assert "50.000% { --animated-score:41; }" in markdown
    assert "80.000% { --animated-score:45; }" in markdown
    assert "90.000% { --animated-score:71; }" in markdown
    assert "100.000% { --animated-score:71; }" in markdown
    assert "Indices forts de modification" not in markdown
    assert "Plusieurs familles de signaux indépendantes" not in markdown
    assert "indicator-step danger" in sequence
    assert "Risque élevé" in sequence
    assert "Aucun signal détecté" in sequence
    assert "Non évalué" in sequence
    assert "--risk-value:100.0%" in sequence
    assert "--step-delay:calc(var(--result-entry-delay, 0s) + 0.45s)" in sequence
    assert "--step-delay:calc(var(--result-entry-delay, 0s) + 4.50s)" in sequence
    assert "Comment lire cette vue" not in markdown
    assert not app.expander
    assert "Zones à revoir - page 1" in app.selectbox[0].options

    assert "Signature électronique du PDF" not in markdown
    assert "Facture électronique embarquée" not in markdown
    assert "Code de vérification 2D-Doc" not in markdown
    assert "Historique complet des versions" in markdown

    app.segmented_control[0].set_value("Glossaire").run(timeout=30)
    glossary = _component_markup(app, '<section class="indicator-glossary"')

    assert not app.exception
    assert app.segmented_control[0].value == "Glossaire"
    assert [button.label for button in app.button] == ["Tester un nouveau document"]
    assert "Différences visibles entre les versions du document." in glossary
    assert "Signatures, structure interne et altérations du fichier." in glossary
    assert "Glossaire des indicateurs" in glossary
    assert not any(
        '<section class="indicator-sequence"' in element.value for element in app.markdown
    )


def test_ocr_results_are_integrated_into_the_review_workspace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))
    monkeypatch.setenv("FROD_OCR_URL", "http://ocr.test:8007")
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "json_result": [
                    [
                        _region("text", "Document reconnu"),
                        _region("text", "Date du sinistre : 18/07/2026"),
                    ]
                ],
                "markdown_result": "## Document reconnu\n\nDate du sinistre : 18/07/2026",
            }
        ),
    )

    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("Montant modifié").run(timeout=30)

    markdown = "\n".join(element.value for element in app.markdown)
    assert not app.exception
    assert not app.tabs
    assert app.segmented_control[0].value == "Analyse"
    assert not app.expander
    assert "Zones de texte reconnues - page 1" in app.selectbox[0].options
    assert "Contrôles effectués" in markdown
    assert "Qualité du texte reconnu" in markdown
    assert "Cohérence des dates" in markdown


def test_precomputed_ocr_demo_runs_without_source_document(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))

    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("OCR - Relevé bancaire à anomalies").run(timeout=30)
    markdown = "\n".join(element.value for element in app.markdown)

    assert not app.exception
    assert [button.label for button in app.button] == ["Tester un nouveau document"]
    assert not app.slider
    assert not app.select_slider
    assert not app.tabs
    assert app.segmented_control[0].value == "Analyse"
    assert not app.expander
    sequence = _component_markup(app, '<section class="indicator-sequence"')
    assert sequence.count('class="indicator-step ') == 1
    assert "Cohérence des dates, montants et identifiants reconnus." not in sequence
    assert "Score de contrôle" in markdown
    assert (
        '<h2 class="workspace-title document-name">OCR - Relevé bancaire à anomalies</h2>'
        in markdown
    )
    assert "--score-target:30" in markdown
    assert "--score-duration:0.45s" in markdown
    assert "100.000% { --animated-score:30; }" in markdown
    assert "Cohérence du contenu" in markdown
    assert "Identifiants reconnus" in markdown
    assert "Validité des identifiants" in markdown
    assert "Cohérence des montants" in markdown
    assert "anomalie(s)" in markdown
    assert "BIC / SWIFT" in markdown
    assert "Referentiels bancaires de pays differents" in markdown
    assert "4111 1111 1111 1111" in markdown
    assert "********" not in markdown

    app.segmented_control[0].set_value("Glossaire").run(timeout=30)
    glossary = _component_markup(app, '<section class="indicator-glossary"')
    assert "Cohérence des dates, montants et identifiants reconnus." in glossary


def test_extraction_laboratory_has_a_business_readable_empty_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))

    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("OCR - Relevé bancaire à anomalies").run(timeout=30)
    app.segmented_control[0].set_value("Laboratoire").run(timeout=30)
    markdown = "\n".join(element.value for element in app.markdown)

    assert not app.exception
    assert app.segmented_control[0].value == "Laboratoire"
    assert "Extraction structurée expérimentale" in markdown
    assert "Aucune extraction disponible" in markdown
    assert "Texte reconnu" in markdown
    assert "TRANSACTION SUMMARY" in markdown
    assert not app.expander


def test_extraction_laboratory_exposes_final_json_on_demand(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(tmp_path / "frod"))
    monkeypatch.setenv("FROD_CLASSIFICATION_URL", "http://extract.test:8030")
    monkeypatch.setenv("FROD_EXTRACTION_URL", "http://extract.test:8030")
    monkeypatch.setenv("FROD_VERIFICATION_URL", "http://extract.test:8030")
    calls: list[list[str]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        assert url == "http://extract.test:8030/v1/chat/completions"
        roles = [message["role"] for message in kwargs["json"]["messages"]]
        calls.append(roles)
        prompt = kwargs["json"]["messages"][1]["content"]
        if "<document_ocr>" in prompt:
            result = {
                "category": "declaration_sinistre",
                "model_confidence": 0.96,
                "ambiguous": False,
                "language": "fr",
                "country": None,
                "category_evidence": ["Le document décrit un sinistre."],
                "country_evidence": None,
            }
            return _Response({"choices": [{"message": {"content": json.dumps(result)}}]})
        if "<extraction_targets>" in prompt:
            result = {
                "issues": [],
                "possible_omissions": [],
            }
            return _Response({"choices": [{"message": {"content": json.dumps(result)}}]})
        region_ids = sorted(set(re.findall(r'<region id="([^"]+)"', prompt)))
        result = {
            "facts": [
                {
                    "field_code": "document_number",
                    "role": "document",
                    "raw_label": "Document",
                    "raw_value": "DECLARATION DE SINISTRE",
                    "confidence": 0.95,
                    "region_ids": ["p001-r000"],
                }
            ],
            "additional_fields": [],
            "tables": [],
            "region_dispositions": {
                "boilerplate": [],
                "unstructured": region_ids,
                "unreadable": [],
            },
        }
        return _Response(
            {"choices": [{"message": {"content": json.dumps(result)}}]},
        )

    monkeypatch.setattr(requests, "post", fake_post)
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.selectbox[0].select("OCR - Déclaration cohérente").run(timeout=30)
    app.segmented_control[0].set_value("Laboratoire").run(timeout=30)
    markdown_blocks = [element.value for element in app.markdown]
    markdown = "\n".join(markdown_blocks)

    assert not app.exception
    assert [expander.label for expander in app.expander] == ["JSON final de l'extraction"]
    assert "Aucune contradiction concrète relevée" in markdown
    classification_index = next(
        index for index, value in enumerate(markdown_blocks) if "Type de document reconnu" in value
    )
    extraction_index = next(
        index
        for index, value in enumerate(markdown_blocks)
        if '<section class="extraction-overview">' in value
    )
    assert classification_index < extraction_index
    assert calls == [
        ["system", "user"],
        ["system", "user"],
        ["system", "user"],
    ]


def test_local_original_ocr_demo_is_discovered_when_present(
    monkeypatch,
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "frod"
    fixture_dir = work_dir / "ocr-fixtures/original/releve-bancaire-anomalies"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "document.json").write_text("[]\n", encoding="utf-8")
    (fixture_dir / "document.md").write_text("Original local\n", encoding="utf-8")
    monkeypatch.setenv("FROD_GAPL_WEIGHTS", "/tmp/frod-missing-gapl.pt")
    monkeypatch.setenv("FROD_TRUFOR_WEIGHTS", "/tmp/frod-missing-trufor.pth.tar")
    monkeypatch.setenv("FROD_WORK_DIR", str(work_dir))

    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()

    assert "OCR original local - Releve bancaire a anomalies" in app.selectbox[0].options


def test_business_ui_contains_no_json_renderer() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    document_view = source[
        source.index("def _render_document_view") : source.index("def _render_review_summary")
    ]
    laboratory_view = source[
        source.index("def _render_extraction_laboratory") : source.index("LAB_STATE_LABELS")
    ]
    report_view = source[source.index("def _render_report") : source.index("def _read_ocr")]

    assert source.count("st.json(") == 1
    assert '"extraction": extraction.to_dict()' in laboratory_view
    assert '"verification": verification.to_dict()' in laboratory_view
    assert "st.tabs" not in source
    assert source.count("st.segmented_control(") == 1
    assert source.count("st.expander(") == 1
    assert "JSON final de l'extraction" in laboratory_view
    assert "_render_classification(classification)" in laboratory_view
    assert "_render_classification(report.classification)" not in report_view
    assert "raw_response" not in source
    assert 'st.columns([0.56, 0.44], gap="large")' in report_view
    assert (
        report_view.index("with document_column:")
        < report_view.index("with indicators_column:")
        < report_view.index("_render_review_summary")
    )
    assert document_view.index('st.selectbox("Vue affichée"') < document_view.index("st.image")
    assert "view-guidance" not in document_view


def test_business_ui_keeps_structural_component_styles() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    style_start = source.index("<style>", source.index("def _inject_styles"))
    styles = source[style_start : source.index("</style>", style_start)]

    progress_fill = _css_rule(styles, ".analysis-progress-track i")
    risk_fill = _css_rule(styles, ".indicator-meter i")
    fraud_score = _css_rule(styles, ".fraud-score")
    fraud_score_number = _css_rule(styles, ".fraud-score-number")
    result_reveal = _css_rule(styles, '[class*="st-key-analysis_results_entering"]')
    pdf_loading = _css_rule(styles, ".pdf-loading-stage")
    app_header = _css_rule(styles, ".st-key-app_header")
    header_actions = _css_rule(styles, ".st-key-header_actions")
    header_navigation = _css_rule(styles, ".st-key-header_navigation")
    header_action = _css_rule(styles, '[class*="st-key-header_reset_document_"]')
    header_action_button = _css_rule(
        styles,
        '[class*="st-key-header_reset_document_"] button',
    )
    finding_score = _css_rule(styles, ".finding-score")

    assert "display: block;" in progress_fill
    assert "height: 100%;" in progress_fill
    assert "display: block;" in risk_fill
    assert "height: 100%;" in risk_fill
    assert "animation: indicator-meter-fill" in risk_fill
    assert "@keyframes indicator-step-validate" in styles
    assert "@property --animated-score" in styles
    assert ".indicator-pending" in styles
    assert ".indicator-final" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert "display: flex;" in fraud_score
    assert "justify-content: space-between;" in fraud_score
    assert "min-height: 58px;" in fraud_score
    assert "animation: var(--score-animation)" in fraud_score_number
    assert "steps(1, end)" in fraud_score_number
    assert "var(--result-entry-delay, 0s)" in fraud_score_number
    assert "animation: analysis-results-enter" in result_reveal
    assert "animation: pdf-loading-transition" in pdf_loading
    assert "@keyframes pdf-loading-spin" in styles
    assert "@keyframes pdf-loading-transition" in styles
    assert "@keyframes analysis-results-enter" in styles
    assert 'class*="st-key-analysis_results_entering"' in styles
    assert "display: flex;" in app_header
    assert "justify-content: space-between;" in app_header
    assert "display: flex;" in header_actions
    assert "align-items: center;" in header_actions
    assert "gap: .5rem;" in header_actions
    assert "margin-left: auto;" in header_actions
    assert "flex: 0 0 auto;" in header_actions
    assert "width: max-content;" in header_navigation
    assert "margin-left: 0;" in header_action
    assert "border: 1px solid #765f35;" in header_action_button
    assert "color: #f3cf8a;" in header_action_button
    assert '[class*="st-key-header_reset_document_"] .stButton {' in styles
    assert "width: auto;" in styles
    assert "display: flex;" in finding_score
    assert "align-items: center;" in finding_score
    assert "justify-content: center;" in finding_score
    assert "h2.workspace-title {" in styles
    assert "h3.subsection-title {" in styles
    assert "grid-template-columns: 32px minmax(0, 1fr) 58px;" in styles
    indicator_card = _css_rule(styles, ".indicator-queue > .indicator-step")
    assert "margin: 0;" in indicator_card
    assert "padding: .55rem .85rem;" in indicator_card
    assert "font-variant-numeric: tabular-nums;" in styles
    assert ".view-guidance" not in styles


def test_new_document_schedules_a_single_scroll_reset() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    input_panel_start = source.index("def _render_input_panel")
    input_panel_end = source.index("def _render_header_document_action")
    input_panel = source[input_panel_start:input_panel_end]
    scroll_reset = source[
        source.index("def _render_pending_scroll_reset") : source.index("def _handle_ocr_demo")
    ]

    assert 'st.session_state["pending_scroll_reset"] = True' in input_panel
    assert 'st.session_state["pending_pdf_loading"] = True' in input_panel
    assert 'st.session_state.pop("pending_scroll_reset", False)' in scroll_reset
    assert 'st.session_state.pop("pending_pdf_loading", False)' in source
    assert '[data-testid="stMain"]' in scroll_reset
    assert "window.setTimeout(resetScroll, 360)" in scroll_reset


def test_document_action_is_reserved_in_the_app_header() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    main_view = source[source.index("def main") : source.index("def _render_app_header")]
    app_header = source[
        source.index("def _render_app_header") : source.index("def _render_input_panel")
    ]

    assert "workspace_view, header_action = _render_app_header()" in main_view
    assert "_render_header_document_action(header_action)" in main_view
    assert app_header.index("st.segmented_control(") < app_header.index("action_slot = st.empty()")
    assert 'key="header_actions"' in app_header
    assert 'horizontal_alignment="right"' in app_header
    assert 'width="content"' in app_header
    assert 'with st.container(key="header_navigation", width="content"):' in app_header
    assert 'return selected or "Analyse", action_slot' in app_header


def _css_rule(styles: str, selector: str) -> str:
    start = styles.index(f"{selector} {{")
    end = styles.index("}", start)
    return styles[start:end]


def _component_markup(app: AppTest, marker: str) -> str:
    return next(element.value for element in app.markdown if marker in element.value)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def _region(label: str, content: str) -> dict[str, object]:
    return {
        "index": 0,
        "label": label,
        "native_label": label,
        "content": content,
        "bbox_2d": [0, 0, 1000, 1000],
        "polygon": [],
    }
