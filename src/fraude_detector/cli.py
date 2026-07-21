"""Command-line interface for the fraud-risk analysis pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fraude_detector import __version__
from fraude_detector.config import AnalysisConfig
from fraude_detector.errors import AnalysisError
from fraude_detector.pipeline import AnalysisPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraude-detect",
        description=(
            "Analyse un PDF d'assurance et produit des indices explicables de "
            "modification avec artefacts visuels."
        ),
    )
    parser.add_argument("input_pdf", type=Path, help="PDF a analyser")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Dossier de sortie (defaut: output/<nom-du-pdf>)",
    )
    parser.add_argument(
        "--password",
        help=(
            "Mot de passe du PDF. FRAUDE_PDF_PASSWORD est prefere pour eviter "
            "l'historique du shell."
        ),
    )
    parser.add_argument("--dpi", type=int, default=144, help="DPI de rendu (>= 72)")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=25,
        help="Nombre maximal de pages analysees visuellement",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output or Path("output") / args.input_pdf.stem
    password = args.password or os.environ.get("FRAUDE_PDF_PASSWORD")

    try:
        config = AnalysisConfig(render_dpi=args.dpi, max_pages=args.max_pages)
        report = AnalysisPipeline(config=config).analyze(
            input_path=args.input_pdf,
            output_dir=output_dir,
            password=password,
        )
    except (AnalysisError, ValueError) as error:
        code = error.code if isinstance(error, AnalysisError) else "invalid_option"
        print(f"Erreur [{code}]: {error}", file=sys.stderr)
        return 1

    print(f"Niveau: {report.assessment.level}")
    print(f"Score de revue: {report.assessment.score}/100")
    print(f"Verdict: {report.assessment.label}")
    print(f"Signaux: {len(report.findings)}")
    print(f"Rapport: {(output_dir / 'report.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
