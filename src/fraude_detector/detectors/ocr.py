"""Optional adapter for a separately hosted GLM-OCR pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

from fraude_detector.config import AnalysisConfig
from fraude_detector.models import DetectorResult, OcrReport
from fraude_detector.ocr_consistency import build_ocr_content_result
from fraude_detector.ocr_rendering import save_layout_visualizations

logger = logging.getLogger(__name__)


class OcrDetector:
    """Call GLM-OCR without turning extracted content into a fraud verdict."""

    name = "ocr_content"

    def __init__(self, config: AnalysisConfig) -> None:
        self.url = config.ocr_url.rstrip("/") + "/glmocr/parse"
        self.timeout_seconds = config.ocr_timeout_seconds

    def detect(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        rendered_pages: tuple[str, ...] = (),
        save_layout: bool = True,
    ) -> OcrReport:
        """Run OCR and write only artifacts scoped to the current analysis."""

        source = input_path.expanduser().resolve()
        destination = output_dir.expanduser().resolve()
        try:
            response = requests.post(
                self.url,
                json={"images": [source.as_uri()]},
                timeout=(5, self.timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
            json_result, markdown = _validated_response(payload)
            ocr_dir = destination / "ocr"
            ocr_dir.mkdir(parents=True, exist_ok=True)
            json_path = ocr_dir / "document.json"
            markdown_path = ocr_dir / "document.md"
            json_path.write_text(
                json.dumps(json_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            markdown_path.write_text(markdown, encoding="utf-8")

            layout_images: tuple[str, ...] = ()
            if save_layout:
                page_images = _source_images(source, destination, rendered_pages)
                layout_images = save_layout_visualizations(
                    page_images,
                    json_result,
                    ocr_dir / "layout",
                    destination,
                )
            artifacts = (
                json_path.relative_to(destination).as_posix(),
                markdown_path.relative_to(destination).as_posix(),
                *layout_images,
            )
        except (OSError, ValueError, requests.RequestException) as error:
            logger.warning("GLM-OCR unavailable for %s: %s", source.name, error)
            return OcrReport(
                success=False,
                error_message=str(error),
                markdown="",
                json_result=[],
            )

        return OcrReport(
            success=True,
            error_message=None,
            markdown=markdown,
            json_result=json_result,
            artifacts=artifacts,
            layout_images=layout_images,
        )

    def result(self, report: OcrReport) -> DetectorResult:
        """Expose OCR content checks as one correlated detector family."""

        return build_ocr_content_result(report)


def _validated_response(payload: Any) -> tuple[list[Any] | dict[str, Any], str]:
    if not isinstance(payload, dict):
        raise ValueError("La reponse GLM-OCR n'est pas un objet JSON.")
    json_result = payload.get("json_result", [])
    if isinstance(json_result, str):
        try:
            json_result = json.loads(json_result)
        except json.JSONDecodeError as error:
            raise ValueError("Le resultat structure GLM-OCR est invalide.") from error
    if not isinstance(json_result, list | dict):
        raise ValueError("Le resultat structure GLM-OCR a un type inattendu.")
    markdown = payload.get("markdown_result", "")
    if not isinstance(markdown, str):
        raise ValueError("Le resultat Markdown GLM-OCR a un type inattendu.")
    return json_result, markdown


def _source_images(
    source: Path,
    output_dir: Path,
    rendered_pages: tuple[str, ...],
) -> tuple[Path, ...]:
    if rendered_pages:
        return tuple(output_dir / relative for relative in rendered_pages)
    return (source,)
