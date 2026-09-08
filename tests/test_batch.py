from __future__ import annotations

import json
from pathlib import Path

from fraude_detector.batch import (
    BatchDocumentSummary,
    BatchResult,
    batch_document_output_dir,
    export_html,
    export_json,
)


def test_batch_output_paths_do_not_collide_on_equal_basenames(tmp_path: Path) -> None:
    first = tmp_path / "one" / "document.pdf"
    second = tmp_path / "two" / "document.pdf"

    first_output = batch_document_output_dir(tmp_path / "output", first)
    second_output = batch_document_output_dir(tmp_path / "output", second)

    assert first_output != second_output
    assert first_output.parent == second_output.parent == tmp_path / "output/documents"


def test_batch_exports_classification_without_remote_html_dependency(tmp_path: Path) -> None:
    result = BatchResult(
        summaries=[
            BatchDocumentSummary(
                filename='<document onerror="alert(1)">.pdf',
                success=True,
                score=12,
                level="low",
                verdict="Revue non prioritaire",
                document_family="facture_recu",
                classification_reliability=0.82,
                language="fr",
                country="LU",
            )
        ],
        completed_at="2026-08-12T12:00:00",
    )
    html_path = tmp_path / "report.html"
    json_path = tmp_path / "report.json"

    export_html(result, html_path)
    export_json(result, json_path)

    html = html_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "https://" not in html
    assert "<script" not in html
    assert "&lt;document onerror=&quot;alert(1)&quot;&gt;.pdf" in html
    assert payload["documents"][0]["document_family"] == "facture_recu"
    assert payload["documents"][0]["classification_reliability"] == 0.82
