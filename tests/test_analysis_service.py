from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from queue import SimpleQueue
from threading import Event
from typing import Any

import fraude_detector.analysis_service as service_module
from fraude_detector.analysis_service import (
    AnalysisProgressEvent,
    AnalysisRunHandle,
    AnalysisService,
)
from fraude_detector.run_config import load_run_config


def test_analysis_service_has_no_streamlit_dependency() -> None:
    source = Path("src/fraude_detector/analysis_service.py").read_text(encoding="utf-8")

    assert "import streamlit" not in source
    assert "from streamlit" not in source


def test_run_handle_exposes_results_progress_and_cancellation() -> None:
    future: Future[Any] = Future()
    progress: SimpleQueue[AnalysisProgressEvent] = SimpleQueue()
    cancel_event = Event()
    handle = AnalysisRunHandle(future, progress, cancel_event)

    progress.put(AnalysisProgressEvent(0.25, "Classification"))
    progress.put(AnalysisProgressEvent(0.75, "Vérification"))

    assert handle.drain_progress() == (
        AnalysisProgressEvent(0.25, "Classification"),
        AnalysisProgressEvent(0.75, "Vérification"),
    )
    assert handle.drain_progress() == ()
    assert not handle.done()

    handle.cancel()

    assert cancel_event.is_set()
    assert future.cancelled()
    assert handle.done()


def test_run_core_owns_files_and_pipeline_orchestration(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    config = load_run_config(
        "config.yaml",
        environ={
            "FROD_WORK_DIR": str(tmp_path / "work"),
            "FROD_GAPL_WEIGHTS": str(tmp_path / "missing-gapl.pt"),
        },
    )
    expected_report = object()
    calls: dict[str, Any] = {}

    class FakeImagePipeline:
        def __init__(self, **kwargs: Any) -> None:
            calls["init"] = kwargs

        def analyze(self, source: Path, output_dir: Path, **kwargs: Any) -> object:
            calls["source"] = source
            calls["output_dir"] = output_dir
            kwargs["progress_callback"](0.5, "Contrôles")
            return expected_report

    monkeypatch.setattr(service_module, "ImageAnalysisPipeline", FakeImagePipeline)
    monkeypatch.setattr(service_module, "_release_transient_memory", lambda: None)
    updates: list[tuple[float, str]] = []

    result = AnalysisService(config).run_core(
        file_name="preuve test.png",
        file_bytes=b"not-a-pdf",
        file_hash="a" * 64,
        progress_callback=lambda value, label: updates.append((value, label)),
    )

    assert result.report is expected_report
    assert result.source_path.name == f"{'a' * 16}-preuve-test.png"
    assert result.source_path.read_bytes() == b"not-a-pdf"
    assert result.output_dir.parent == tmp_path / "work" / "runs"
    assert calls["source"] == result.source_path
    assert calls["output_dir"] == result.output_dir
    assert calls["init"]["config"].classification_enabled is False
    assert calls["init"]["config"].extraction_enabled is False
    assert calls["init"]["config"].verification_enabled is False
    assert updates[-1] == (1.0, "Analyse principale terminée")
