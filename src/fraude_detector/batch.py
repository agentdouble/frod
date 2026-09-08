"""Sequential batch processing with lightweight, local-only summaries."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from fraude_detector.config import AnalysisConfig
from fraude_detector.image_pipeline import ImageAnalysisPipeline
from fraude_detector.pipeline import AnalysisPipeline


@dataclass
class BatchDocumentSummary:
    """Small in-memory result for one document."""

    filename: str
    success: bool
    score: int = 0
    level: str = ""
    verdict: str = ""
    findings_count: int = 0
    scored_findings: int = 0
    detectors_with_findings: str = ""
    document_family: str = ""
    classification_reliability: float | None = None
    language: str = ""
    country: str = ""
    processing_time_seconds: float = 0.0
    error_message: str | None = None


@dataclass
class BatchResult:
    """Aggregated batch result."""

    summaries: list[BatchDocumentSummary] = field(default_factory=list)
    total_processing_time_seconds: float = 0.0
    started_at: str = ""
    completed_at: str = ""

    @property
    def successful_count(self) -> int:
        return sum(1 for summary in self.summaries if summary.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for summary in self.summaries if not summary.success)

    def get_level_counts(self) -> dict[str, int]:
        counts = {"low": 0, "review": 0, "high": 0}
        for summary in self.summaries:
            if summary.success and summary.level in counts:
                counts[summary.level] += 1
        return counts

    def get_avg_score(self) -> float:
        scores = [summary.score for summary in self.summaries if summary.success]
        return sum(scores) / len(scores) if scores else 0.0


def is_pdf(path: Path) -> bool:
    """Identify a PDF from its header, with an extension fallback on read errors."""

    try:
        with path.open("rb") as input_file:
            return b"%PDF-" in input_file.read(1024)
    except OSError:
        return path.suffix.casefold() == ".pdf"


def batch_document_output_dir(output_dir: Path, path: Path) -> Path:
    """Return a stable path that cannot collide on equal basenames."""

    path_fingerprint = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "-" for character in path.stem
    )
    safe_stem = safe_stem.strip("-") or "document"
    return output_dir / "documents" / f"{safe_stem}-{path_fingerprint}"


def analyze_document(
    path: Path,
    config: AnalysisConfig,
    output_dir: Path,
    ai_adapters: tuple = (),
    password: str | None = None,
) -> BatchDocumentSummary:
    """Analyze one document and retain only its review summary."""

    started = time.monotonic()
    try:
        if is_pdf(path):
            report = AnalysisPipeline(config=config, ai_image_adapters=ai_adapters).analyze(
                input_path=path,
                output_dir=output_dir,
                password=password,
            )
        else:
            report = ImageAnalysisPipeline(
                config=config,
                ai_image_adapters=ai_adapters,
            ).analyze(
                input_path=path,
                output_dir=output_dir,
            )

        findings = report.findings
        classification = report.classification
        return BatchDocumentSummary(
            filename=path.name,
            success=True,
            score=report.assessment.score,
            level=report.assessment.level,
            verdict=report.assessment.label,
            findings_count=len(findings),
            scored_findings=sum(finding.risk_points > 0 for finding in findings),
            detectors_with_findings=",".join(
                detector.name for detector in report.detectors if detector.findings
            ),
            document_family=classification.family if classification else "",
            classification_reliability=(classification.reliability if classification else None),
            language=classification.language or "" if classification else "",
            country=classification.country or "" if classification else "",
            processing_time_seconds=time.monotonic() - started,
        )
    except Exception as error:
        return BatchDocumentSummary(
            filename=path.name,
            success=False,
            error_message=str(error)[:240],
            processing_time_seconds=time.monotonic() - started,
        )


def process_batch(
    input_paths: list[Path],
    config: AnalysisConfig,
    output_dir: Path,
    ai_adapters: tuple = (),
    progress_callback: Callable[[float, str], None] | None = None,
    password: str | None = None,
) -> BatchResult:
    """Process documents sequentially to keep model memory bounded."""

    started_at = datetime.now().isoformat()
    started = time.monotonic()
    summaries: list[BatchDocumentSummary] = []
    total = len(input_paths)

    for index, path in enumerate(input_paths):
        if progress_callback:
            progress_callback(index / total, f"Analyse de {path.name}")
        summary = analyze_document(
            path,
            config,
            batch_document_output_dir(output_dir, path),
            ai_adapters,
            password,
        )
        summaries.append(summary)
        if progress_callback:
            progress_callback((index + 1) / total, f"Terminé : {path.name}")

    return BatchResult(
        summaries=summaries,
        total_processing_time_seconds=time.monotonic() - started,
        started_at=started_at,
        completed_at=datetime.now().isoformat(),
    )


def export_csv(batch_result: BatchResult, output_path: Path) -> None:
    """Export one semicolon-delimited row per input document."""

    fieldnames = list(asdict(BatchDocumentSummary(filename="", success=False)))
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for summary in batch_result.summaries:
            writer.writerow(asdict(summary))


def export_json(batch_result: BatchResult, output_path: Path) -> None:
    """Export aggregate metrics and lightweight document summaries."""

    data = {
        "summary": {
            "total_documents": len(batch_result.summaries),
            "successful": batch_result.successful_count,
            "failed": batch_result.failed_count,
            "level_counts": batch_result.get_level_counts(),
            "average_score": round(batch_result.get_avg_score(), 2),
            "total_processing_time_seconds": round(
                batch_result.total_processing_time_seconds,
                2,
            ),
            "started_at": batch_result.started_at,
            "completed_at": batch_result.completed_at,
        },
        "documents": [asdict(summary) for summary in batch_result.summaries],
    }
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def export_html(batch_result: BatchResult, output_path: Path) -> None:
    """Export a self-contained report without scripts or remote resources."""

    counts = batch_result.get_level_counts()
    rows = "\n".join(_summary_row(summary) for summary in batch_result.summaries)
    total = max(1, batch_result.successful_count)
    distributions = "\n".join(
        _distribution_row(label, counts[level], total, css_class)
        for level, label, css_class in (
            ("low", "Faible", "low"),
            ("review", "À examiner", "review"),
            ("high", "Élevé", "high"),
        )
    )
    content = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Rapport d'analyse documentaire</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0e1117; --panel:#191c22; --line:#343943;
      --text:#f4f5f7; --muted:#a4a9b3; --blue:#55a7ff; --green:#43c78a;
      --amber:#efb84b; --red:#ed6a75; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; padding:28px; background:var(--bg); color:var(--text);
      font:14px/1.45 system-ui,sans-serif; }}
    main {{ max-width:1320px; margin:auto; }}
    h1 {{ margin:0 0 22px; font-size:25px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
      gap:10px; margin-bottom:18px; }}
    .metric,.panel {{ border:1px solid var(--line); background:var(--panel); border-radius:6px; }}
    .metric {{ padding:14px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric strong {{ display:block; margin-top:5px; font-size:24px; }}
    .panel {{ padding:16px; margin-bottom:18px; overflow:auto; }}
    .distribution {{ display:grid; grid-template-columns:90px 1fr 45px; gap:10px;
      align-items:center; margin:8px 0; }}
    .track {{ height:8px; overflow:hidden; border-radius:4px; background:#2a2e36; }}
    .fill {{ height:100%; background:var(--blue); }} .fill.low {{ background:var(--green); }}
    .fill.review {{ background:var(--amber); }} .fill.high {{ background:var(--red); }}
    table {{ width:100%; border-collapse:collapse; min-width:980px; }}
    th,td {{ padding:10px; text-align:left; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .score {{ font-weight:800; }} .error {{ color:var(--red); }}
    footer {{ color:var(--muted); font-size:12px; }}
  </style>
</head>
<body><main>
  <h1>Analyse documentaire par lot</h1>
  <section class="metrics">
    <div class="metric"><span>Documents</span><strong>{len(batch_result.summaries)}</strong></div>
    <div class="metric"><span>Analyses réussies</span>
      <strong>{batch_result.successful_count}</strong></div>
    <div class="metric"><span>Échecs</span><strong>{batch_result.failed_count}</strong></div>
    <div class="metric"><span>Score moyen</span>
      <strong>{batch_result.get_avg_score():.1f}/100</strong></div>
  </section>
  <section class="panel"><h2>Répartition des niveaux</h2>{distributions}</section>
  <section class="panel"><table>
    <thead><tr><th>Document</th><th>Score</th><th>Niveau</th><th>Famille</th>
      <th>Fiabilité</th><th>Langue</th><th>Pays</th><th>Signaux</th><th>État</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></section>
  <footer>Analyse terminée le {html.escape(batch_result.completed_at)}</footer>
</main></body></html>"""
    output_path.write_text(content, encoding="utf-8")


def _summary_row(summary: BatchDocumentSummary) -> str:
    if not summary.success:
        status = f'<span class="error">{html.escape(summary.error_message or "Échec")}</span>'
    else:
        status = "Terminée"
    reliability = (
        f"{summary.classification_reliability:.0%}"
        if summary.classification_reliability is not None
        else "-"
    )
    return (
        "<tr>"
        f"<td>{html.escape(summary.filename)}</td>"
        f'<td class="score">{summary.score if summary.success else "-"}</td>'
        f"<td>{html.escape(summary.level or '-')}</td>"
        f"<td>{html.escape(summary.document_family or '-')}</td>"
        f"<td>{reliability}</td>"
        f"<td>{html.escape(summary.language or '-')}</td>"
        f"<td>{html.escape(summary.country or '-')}</td>"
        f"<td>{summary.scored_findings if summary.success else '-'}</td>"
        f"<td>{status}</td>"
        "</tr>"
    )


def _distribution_row(label: str, count: int, total: int, css_class: str) -> str:
    percentage = 100 * count / total
    return (
        '<div class="distribution">'
        f"<span>{html.escape(label)}</span>"
        '<div class="track">'
        f'<div class="fill {css_class}" style="width:{percentage:.2f}%"></div>'
        "</div>"
        f"<strong>{count}</strong>"
        "</div>"
    )
