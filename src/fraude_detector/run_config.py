"""Load the small YAML contract used by the local project launcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class RunConfigError(ValueError):
    """A user-facing error in the launcher configuration."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Inputs required to start one analysis from ``config.yaml``."""

    input_path: Path


def load_run_config(config_path: str | Path) -> RunConfig:
    """Read and validate a launcher configuration.

    Relative PDF paths are resolved from the directory containing the YAML file,
    so the launcher behaves consistently regardless of the caller's directory.
    """

    source = Path(config_path).expanduser().resolve()
    try:
        raw_config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RunConfigError(f"Configuration introuvable: {source}") from error
    except OSError as error:
        raise RunConfigError(f"Configuration illisible: {source}: {error}") from error
    except yaml.YAMLError as error:
        raise RunConfigError(f"YAML invalide dans {source}: {error}") from error

    config = _require_mapping(raw_config, source)
    unknown_keys = sorted(set(config) - {"input_path", "pdf_path"})
    if unknown_keys:
        keys = ", ".join(str(key) for key in unknown_keys)
        raise RunConfigError(f"Cle(s) inconnue(s) dans {source}: {keys}")

    configured_keys = [key for key in ("input_path", "pdf_path") if key in config]
    if len(configured_keys) != 1:
        raise RunConfigError(
            f"Definissez exactement une cle input_path dans {source} "
            "(pdf_path reste accepte comme ancien nom)"
        )
    raw_input_path = config[configured_keys[0]]
    if not isinstance(raw_input_path, str) or not raw_input_path.strip():
        raise RunConfigError(f"La cle input_path doit contenir un chemin dans {source}")

    input_path = Path(raw_input_path).expanduser()
    if not input_path.is_absolute():
        input_path = source.parent / input_path
    return RunConfig(input_path=input_path.resolve())


def _require_mapping(raw_config: Any, source: Path) -> dict[str, Any]:
    if not isinstance(raw_config, dict):
        raise RunConfigError(f"La configuration doit etre un objet YAML dans {source}")
    if not all(isinstance(key, str) for key in raw_config):
        raise RunConfigError(f"Toutes les cles doivent etre du texte dans {source}")
    return raw_config
