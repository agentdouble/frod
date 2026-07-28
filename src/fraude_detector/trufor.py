"""Isolated local adapter for TruFor image-manipulation localization."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TRUFOR_DEFAULT_CHECKPOINT = Path("models/trufor/trufor.pth.tar")
TRUFOR_CHECKPOINT_SHA256 = "ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"
TRUFOR_UPSTREAM_COMMIT = "ae54475df6f41a491d7615100feb19263dec13f7"
TRUFOR_MAX_PIXELS = 400_000
TRUFOR_TIMEOUT_SECONDS = 900

_RDBU_REVERSED_STOPS = np.asarray(
    (
        (0.00, 5, 48, 97),
        (0.10, 33, 102, 172),
        (0.20, 67, 147, 195),
        (0.30, 146, 197, 222),
        (0.40, 209, 229, 240),
        (0.50, 247, 247, 247),
        (0.60, 253, 219, 199),
        (0.70, 244, 165, 130),
        (0.80, 214, 96, 77),
        (0.90, 178, 24, 43),
        (1.00, 103, 0, 31),
    ),
    dtype=np.float32,
)


class TruForError(RuntimeError):
    """Base error for optional TruFor support."""


class TruForWeightsError(TruForError):
    """Raised when the official TruFor checkpoint is missing or invalid."""


class TruForInferenceError(TruForError):
    """Raised when the isolated inference worker cannot produce a result."""


@dataclass(frozen=True, slots=True)
class TruForAnalysis:
    """Validated outputs emitted by the official TruFor model."""

    score: float
    anomaly_map: np.ndarray
    confidence_map: np.ndarray
    reliable_anomaly_map: np.ndarray
    original_size: tuple[int, int]
    analyzed_size: tuple[int, int]
    resized: bool
    reliable_suspect_ratio: float
    artifacts: tuple[str, ...]


def analyze_trufor_image(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    weights_path: str | Path = TRUFOR_DEFAULT_CHECKPOINT,
    max_pixels: int = TRUFOR_MAX_PIXELS,
    timeout_seconds: int = TRUFOR_TIMEOUT_SECONDS,
    progress_callback: Callable[[float, str], None] | None = None,
) -> TruForAnalysis:
    """Run TruFor in one short-lived process and persist explainable artifacts."""

    source = Path(input_path).expanduser().resolve()
    checkpoint = Path(weights_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() / "laboratory" / "trufor"
    result_path = destination / "result.npz"

    if not source.is_file():
        raise TruForInferenceError(f"Image introuvable: {source}")
    if max_pixels < 65_536:
        raise ValueError("max_pixels must be at least 65536")

    _validate_checkpoint(checkpoint)
    destination.mkdir(parents=True, exist_ok=True)
    _report(progress_callback, 0.08, "Preparation de TruFor")

    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    command = (
        sys.executable,
        "-m",
        "fraude_detector.trufor_worker",
        "--input",
        str(source),
        "--weights",
        str(checkpoint),
        "--output",
        str(result_path),
        "--max-pixels",
        str(max_pixels),
    )
    _report(progress_callback, 0.18, "Analyse des retouches locales")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise TruForInferenceError(
            f"TruFor a depasse la limite de {timeout_seconds} secondes."
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        message = detail[-1][:500] if detail else "erreur inconnue"
        raise TruForInferenceError(f"TruFor a interrompu l'analyse: {message}")
    if not result_path.is_file():
        raise TruForInferenceError("TruFor n'a produit aucun resultat.")

    _report(progress_callback, 0.88, "Preparation de la carte TruFor")
    analysis = _read_result(result_path)
    artifacts = _write_artifacts(destination, analysis)
    _report(progress_callback, 1.0, "Analyse TruFor terminee")
    return TruForAnalysis(
        score=analysis.score,
        anomaly_map=analysis.anomaly_map,
        confidence_map=analysis.confidence_map,
        reliable_anomaly_map=analysis.reliable_anomaly_map,
        original_size=analysis.original_size,
        analyzed_size=analysis.analyzed_size,
        resized=analysis.resized,
        reliable_suspect_ratio=analysis.reliable_suspect_ratio,
        artifacts=artifacts,
    )


def _read_result(path: Path) -> TruForAnalysis:
    try:
        with np.load(path, allow_pickle=False) as payload:
            anomaly = np.asarray(payload["map"], dtype=np.float32)
            confidence = np.asarray(payload["conf"], dtype=np.float32)
            score = float(np.asarray(payload["score"]).reshape(-1)[0])
            original_size = _size_tuple(payload["original_size"])
            analyzed_size = _size_tuple(payload["analyzed_size"])
    except (KeyError, OSError, ValueError, TypeError, IndexError) as error:
        raise TruForInferenceError("La sortie TruFor est illisible ou incomplete.") from error

    if anomaly.ndim != 2 or confidence.shape != anomaly.shape:
        raise TruForInferenceError("Les cartes TruFor ont des dimensions invalides.")
    if not np.isfinite(anomaly).all() or not np.isfinite(confidence).all():
        raise TruForInferenceError("Les cartes TruFor contiennent des valeurs invalides.")
    if not np.isfinite(score):
        raise TruForInferenceError("Le score TruFor est invalide.")

    anomaly = np.clip(anomaly, 0.0, 1.0)
    confidence = np.clip(confidence, 0.0, 1.0)
    reliable = anomaly * confidence
    suspect = (anomaly >= 0.5) & (confidence >= 0.5)
    return TruForAnalysis(
        score=min(1.0, max(0.0, score)),
        anomaly_map=anomaly,
        confidence_map=confidence,
        reliable_anomaly_map=reliable,
        original_size=original_size,
        analyzed_size=analyzed_size,
        resized=original_size != analyzed_size,
        reliable_suspect_ratio=float(np.mean(suspect)),
        artifacts=(),
    )


def _write_artifacts(
    destination: Path,
    analysis: TruForAnalysis,
) -> tuple[str, ...]:
    localization_path = destination / "trufor-localization-map.png"
    reliable_path = destination / "trufor-reliable-map.png"
    confidence_path = destination / "trufor-confidence.png"
    evidence_path = destination / "trufor-analysis.json"
    _render_spectral_map(analysis.anomaly_map).save(localization_path)
    _render_spectral_map(analysis.reliable_anomaly_map).save(reliable_path)
    _render_confidence_map(analysis.confidence_map).save(confidence_path)

    evidence: dict[str, Any] = {
        "model": "TruFor",
        "upstream_commit": TRUFOR_UPSTREAM_COMMIT,
        "score": analysis.score,
        "original_size": list(analysis.original_size),
        "analyzed_size": list(analysis.analyzed_size),
        "resized": analysis.resized,
        "reliable_suspect_ratio": analysis.reliable_suspect_ratio,
        "diagnostic_thresholds": {
            "anomaly": 0.5,
            "confidence": 0.5,
        },
        "interpretation": (
            "Le score et les seuils ne sont pas calibres sur les documents d'assurance."
        ),
    }
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        "trufor/trufor-localization-map.png",
        "trufor/trufor-reliable-map.png",
        "trufor/trufor-confidence.png",
    )


def _render_spectral_map(values: np.ndarray) -> Image.Image:
    """Render the official TruFor-style RdBu_r localization palette."""

    normalized = np.clip(values, 0.0, 1.0)
    positions = _RDBU_REVERSED_STOPS[:, 0]
    channels = [
        np.interp(normalized, positions, _RDBU_REVERSED_STOPS[:, index]) for index in range(1, 4)
    ]
    rendered = np.stack(channels, axis=-1)
    return Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8), mode="RGB")


def _render_confidence_map(values: np.ndarray) -> Image.Image:
    rendered = np.clip(values, 0.0, 1.0) * 255
    return Image.fromarray(rendered.astype(np.uint8), mode="L").convert("RGB")


def _size_tuple(value: np.ndarray) -> tuple[int, int]:
    flattened = np.asarray(value, dtype=np.int64).reshape(-1)
    if flattened.size != 2 or (flattened < 1).any():
        raise ValueError("invalid image size")
    return int(flattened[0]), int(flattened[1])


def _validate_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise TruForWeightsError(
            "Checkpoint TruFor introuvable. Executez ./start.sh pour le telecharger."
        )
    stat = path.stat()
    digest = _checkpoint_sha256(str(path), stat.st_size, stat.st_mtime_ns)
    if digest != TRUFOR_CHECKPOINT_SHA256:
        raise TruForWeightsError("Le checkpoint TruFor ne correspond pas au fichier officiel.")


@lru_cache(maxsize=4)
def _checkpoint_sha256(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _report(
    callback: Callable[[float, str], None] | None,
    value: float,
    label: str,
) -> None:
    if callback is not None:
        callback(min(1.0, max(0.0, value)), label)
