"""Application service orchestrating complete document analyses.

This module deliberately has no dependency on Streamlit. Frontends submit inputs,
observe progress, and render the immutable results exposed here.
"""

from __future__ import annotations

import gc
import json
import re
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import cache, partial
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Lock
from uuid import uuid4

from fraude_detector.config import AnalysisConfig
from fraude_detector.content_analysis import ContentAnalysisResult, analyze_recognized_content
from fraude_detector.gapl import best_available_device, create_gapl_adapter
from fraude_detector.image_pipeline import ImageAnalysisPipeline
from fraude_detector.laboratory import (
    analyze_ocr_laboratory,
    analyze_pdf_laboratory,
)
from fraude_detector.laboratory.visual_repetition import analyze_repeated_visual_regions
from fraude_detector.llm_synthesizer import summarize_analysis
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
    OcrReport,
)
from fraude_detector.pipeline import AnalysisPipeline
from fraude_detector.run_config import RunConfig

ProgressCallback = Callable[[float, str], None]
Report = AnalysisReport | ImageAnalysisReport


@dataclass(frozen=True, slots=True)
class AnalysisProgressEvent:
    """One progress update emitted by a background analysis."""

    value: float
    label: str


@dataclass(frozen=True, slots=True)
class CoreAnalysisResult:
    """Result available as soon as the deterministic and model checks finish."""

    report: Report
    output_dir: Path
    source_path: Path


@dataclass(frozen=True, slots=True)
class AnalysisRunResult:
    """Complete backend result returned independently of its presentation layer."""

    report: Report
    laboratory: LaboratoryReport
    synthesis: AnalysisSynthesis | None
    synthesis_error: str | None
    output_dir: Path
    source_path: Path


@dataclass(frozen=True, slots=True)
class OcrFixtureAnalysisResult:
    """Semantic analysis of an already recognized document fixture."""

    laboratory: LaboratoryReport
    ocr_detector: DetectorResult
    classification: DocumentClassification | None
    extraction: DocumentExtraction | None
    verification: ExtractionVerification | None
    synthesis: AnalysisSynthesis | None
    synthesis_error: str | None


class AnalysisRunHandle:
    """Small frontend-facing facade over the asynchronous execution primitives."""

    def __init__(
        self,
        future: Future[AnalysisRunResult],
        progress_queue: SimpleQueue[AnalysisProgressEvent],
        cancel_event: Event,
    ) -> None:
        self._future = future
        self._progress_queue = progress_queue
        self._cancel_event = cancel_event

    def done(self) -> bool:
        return self._future.done()

    def result(self) -> AnalysisRunResult:
        return self._future.result()

    def drain_progress(self) -> tuple[AnalysisProgressEvent, ...]:
        events: list[AnalysisProgressEvent] = []
        while True:
            try:
                events.append(self._progress_queue.get_nowait())
            except Empty:
                return tuple(events)

    def cancel(self) -> None:
        self._cancel_event.set()
        self._future.cancel()


class AnalysisService:
    """Coordinate core, semantic, laboratory and synthesis analysis stages."""

    def __init__(self, config: RunConfig, *, executor: Executor | None = None) -> None:
        self.config = config
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="frod-ai",
        )

    def run_core(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        file_hash: str,
        progress_callback: ProgressCallback | None = None,
    ) -> CoreAnalysisResult:
        """Run the main scoring pipeline and persist the uploaded input locally."""

        def report_progress(value: float, text: str) -> None:
            if progress_callback is not None:
                progress_callback(value, text)

        report_progress(0.02, "Preparation du fichier")
        upload_dir = self.config.application.work_dir / "uploads"
        run_dir = self.config.application.work_dir / "runs"
        upload_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        source = upload_dir / f"{file_hash[:16]}-{_safe_filename(file_name)}"
        source.write_bytes(file_bytes)
        output_dir = run_dir / f"{file_hash[:16]}-{uuid4().hex[:8]}"

        adapter_loaders = ()
        gapl = self.config.gapl
        if gapl.enabled and gapl.weights_path.is_file():
            report_progress(0.06, "Préparation de l'analyse des images")
            device = best_available_device() if gapl.device == "auto" else gapl.device
            adapter_loaders = (
                partial(_load_gapl_adapter, str(gapl.weights_path.resolve()), device),
            )

        core_config = replace(
            self.config.analysis,
            classification_enabled=False,
            extraction_enabled=False,
            verification_enabled=False,
        )

        def pipeline_progress(value: float, text: str) -> None:
            report_progress(0.10 + 0.88 * value, text)

        if _is_pdf_bytes(file_bytes):
            report: Report = AnalysisPipeline(
                config=core_config,
                ai_image_adapter_loaders=adapter_loaders,
            ).analyze(source, output_dir, progress_callback=pipeline_progress)
        else:
            report = ImageAnalysisPipeline(
                config=core_config,
                ai_image_adapter_loaders=adapter_loaders,
            ).analyze(source, output_dir, progress_callback=pipeline_progress)
            _release_transient_memory()

        report_progress(1.0, "Analyse principale terminée")
        return CoreAnalysisResult(report=report, output_dir=output_dir, source_path=source)

    def start_ai(self, core: CoreAnalysisResult) -> AnalysisRunHandle:
        """Start semantic and laboratory work without exposing concurrency internals."""

        progress_queue: SimpleQueue[AnalysisProgressEvent] = SimpleQueue()
        cancel_event = Event()
        future = self._executor.submit(
            self._run_ai_analysis,
            core,
            progress_queue,
            cancel_event,
        )
        return AnalysisRunHandle(future, progress_queue, cancel_event)

    def analyze_ocr_fixture(
        self,
        *,
        payload: object,
        markdown: str,
    ) -> OcrFixtureAnalysisResult:
        """Run the backend semantic chain over precomputed OCR fixture data."""

        laboratory = LaboratoryReport(
            schema_version="0.1-experimental",
            checks=analyze_ocr_laboratory(payload),
        )
        ocr_report = OcrReport(
            success=True,
            error_message=None,
            markdown=markdown,
            json_result=payload,
        )
        semantic = analyze_recognized_content(ocr_report, self.config.analysis)
        score = round(
            max((finding.risk_points for finding in semantic.detector_result.findings), default=0)
        )
        synthesis = None
        synthesis_error = None
        if self.config.analysis.synthesis_enabled:
            synthesis, synthesis_error = self._safe_synthesis(
                classification=semantic.classification,
                extraction=semantic.extraction,
                verification=semantic.verification,
                findings=semantic.detector_result.findings,
                detectors=(semantic.detector_result,),
                laboratory=laboratory,
                config=self.config.analysis,
                assessment_score=score,
                assessment_label=None,
            )
        return OcrFixtureAnalysisResult(
            laboratory=laboratory,
            ocr_detector=semantic.detector_result,
            classification=semantic.classification,
            extraction=semantic.extraction,
            verification=semantic.verification,
            synthesis=synthesis,
            synthesis_error=synthesis_error,
        )

    def _run_ai_analysis(
        self,
        core: CoreAnalysisResult,
        progress_queue: SimpleQueue[AnalysisProgressEvent],
        cancel_event: Event,
    ) -> AnalysisRunResult:
        report = core.report
        branch_values = {"semantic": 0.0, "laboratory": 0.0}
        progress_lock = Lock()

        def branch_progress(branch: str, value: float, label: str) -> None:
            with progress_lock:
                branch_values[branch] = max(
                    branch_values[branch],
                    min(1.0, max(0.0, value)),
                )
                combined = (
                    0.75 * branch_values["semantic"]
                    + 0.15 * branch_values["laboratory"]
                )
            progress_queue.put(AnalysisProgressEvent(combined, label))

        progress_queue.put(AnalysisProgressEvent(0.0, "Préparation de l'analyse IA"))
        ocr_report = _load_existing_ocr_report(report, core.output_dir)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="frod-ai-stage") as executor:
            semantic_future = executor.submit(
                self._run_semantic_analysis,
                ocr_report,
                cancel_event,
                lambda value, label: branch_progress("semantic", value, label),
            )
            laboratory_future = executor.submit(
                self._run_laboratory_analysis,
                report,
                core.output_dir,
                core.source_path,
                cancel_event,
                lambda value, label: branch_progress("laboratory", value, label),
            )
            semantic = semantic_future.result()
            laboratory = laboratory_future.result()

        if cancel_event.is_set():
            raise RuntimeError("Analyse IA annulée")
        completed_report = replace(
            report,
            classification=semantic.classification if semantic is not None else None,
            extraction=semantic.extraction if semantic is not None else None,
            extraction_verification=semantic.verification if semantic is not None else None,
        )
        progress_queue.put(AnalysisProgressEvent(0.92, "Synthèse de l'analyse"))
        synthesis, synthesis_error = self._safe_synthesis(
            classification=completed_report.classification,
            extraction=completed_report.extraction,
            verification=completed_report.extraction_verification,
            findings=completed_report.findings,
            detectors=completed_report.detectors,
            laboratory=laboratory,
            config=self.config.analysis,
            assessment_score=completed_report.assessment.score,
            assessment_label=completed_report.assessment.label,
        )
        _write_completed_report(completed_report, core.output_dir / "report.json")
        progress_queue.put(AnalysisProgressEvent(1.0, "Analyse IA terminée"))
        return AnalysisRunResult(
            report=completed_report,
            laboratory=laboratory,
            synthesis=synthesis,
            synthesis_error=synthesis_error,
            output_dir=core.output_dir,
            source_path=core.source_path,
        )

    def _run_semantic_analysis(
        self,
        ocr_report: OcrReport | None,
        cancel_event: Event,
        progress_callback: ProgressCallback,
    ) -> ContentAnalysisResult | None:
        if cancel_event.is_set() or ocr_report is None:
            progress_callback(1.0, "Analyse sémantique non applicable")
            return None
        return analyze_recognized_content(
            ocr_report,
            self.config.analysis,
            progress_callback=progress_callback,
            cancel_callback=cancel_event.is_set,
        )

    def _run_laboratory_analysis(
        self,
        report: Report,
        output_dir: Path,
        source_path: Path,
        cancel_event: Event,
        progress_callback: ProgressCallback,
    ) -> LaboratoryReport:
        if cancel_event.is_set():
            return _empty_laboratory_report()
        if isinstance(report, AnalysisReport):
            laboratory = (
                analyze_pdf_laboratory(
                    source_path,
                    output_dir,
                    config=self.config.analysis,
                    progress_callback=progress_callback,
                )
                if self.config.laboratory.pdf_enabled
                else _empty_laboratory_report()
            )
        else:
            laboratory = _empty_laboratory_report()
        progress_callback(0.95, "Contrôles expérimentaux terminés")
        laboratory = self._with_ocr_laboratory(
            report,
            laboratory,
            output_dir,
            source_path,
        )
        progress_callback(1.0, "Laboratoire terminé")
        return laboratory

    def _with_ocr_laboratory(
        self,
        report: Report,
        laboratory: LaboratoryReport,
        output_dir: Path,
        source_path: Path,
    ) -> LaboratoryReport:
        json_paths = _existing_artifacts(output_dir, report.artifacts.get("ocr_json", ()))
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
        laboratory_config = self.config.laboratory
        visual_repetition = (
            analyze_repeated_visual_regions(
                payload,
                tuple(page_images),
                output_dir / "laboratory",
                minimum_pages=laboratory_config.visual_repetition_min_pages,
                minimum_similarity=laboratory_config.visual_repetition_similarity,
            )
            if laboratory_config.visual_repetition_enabled
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

    @staticmethod
    def _safe_synthesis(
        *,
        classification: DocumentClassification | None,
        extraction: DocumentExtraction | None,
        verification: ExtractionVerification | None,
        findings: tuple[Finding, ...],
        detectors: tuple[DetectorResult, ...],
        laboratory: LaboratoryReport,
        config: AnalysisConfig,
        assessment_score: int | None,
        assessment_label: str | None,
    ) -> tuple[AnalysisSynthesis | None, str | None]:
        try:
            return (
                summarize_analysis(
                    classification=classification,
                    extraction=extraction,
                    verification=verification,
                    findings=findings,
                    detectors=detectors,
                    laboratory=laboratory,
                    config=config,
                    assessment_score=assessment_score,
                    assessment_label=assessment_label,
                ),
                None,
            )
        except Exception:
            return None, "La synthèse assistée n'a pas pu être produite pour ce document."


@cache
def _load_gapl_adapter(weights_path: str, device: str) -> object:
    return create_gapl_adapter(weights_path=weights_path, device=device)


def _load_existing_ocr_report(report: Report, output_dir: Path) -> OcrReport | None:
    json_paths = _existing_artifacts(output_dir, report.artifacts.get("ocr_json", ()))
    markdown_paths = _existing_artifacts(output_dir, report.artifacts.get("ocr_markdown", ()))
    if not json_paths or not markdown_paths:
        return None
    try:
        payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
        markdown = markdown_paths[0].read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        return None
    return OcrReport(
        success=True,
        error_message=None,
        markdown=markdown,
        json_result=payload,
        artifacts=(
            *report.artifacts.get("ocr_json", ()),
            *report.artifacts.get("ocr_markdown", ()),
        ),
        layout_images=report.artifacts.get("ocr_layout", ()),
    )


def _write_completed_report(report: Report, path: Path) -> None:
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _existing_artifacts(output_dir: Path, relatives: tuple[str, ...]) -> list[Path]:
    return [path for relative in relatives if (path := output_dir / relative).is_file()]


def _empty_laboratory_report() -> LaboratoryReport:
    return LaboratoryReport(schema_version="1.0", checks=())


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
