"""Command-line launcher for analyses configured through YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fraude_detector.cli import main as analysis_main
from fraude_detector.run_config import RunConfigError, load_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="start.sh",
        description="Lance l'analyse du PDF defini dans config.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Fichier YAML a charger (defaut: config.yaml)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_run_config(args.config)
    except RunConfigError as error:
        print(f"Erreur [invalid_config]: {error}", file=sys.stderr)
        return 2

    if config.input_path is None:
        print(
            "Erreur [invalid_config]: Definissez input.path dans la configuration.",
            file=sys.stderr,
        )
        return 2
    return analysis_main(
        [
            str(config.input_path),
            "--config",
            str(config.source),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
