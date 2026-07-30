"""Command-line interface for the fraud-risk analysis pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from fraude_detector import __version__
from fraude_detector.ai_images import AiImageModelAdapter
from fraude_detector.community_forensics import (
    CommunityForensicsError,
    create_community_forensics_adapter,
)
from fraude_detector.errors import AnalysisError
from fraude_detector.gapl import (
    GaplError,
    best_available_device,
    create_gapl_adapter,
)
from fraude_detector.image_pipeline import ImageAnalysisPipeline
from fraude_detector.pipeline import AnalysisPipeline
from fraude_detector.run_config import GaplConfig, RunConfigError, load_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frod",
        description=(
            "Analyse un PDF ou une image et produit des indices explicables de "
            "modification avec artefacts visuels."
        ),
    )
    parser.add_argument("input_file", type=Path, help="PDF ou image a analyser")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("FROD_CONFIG", "config.yaml")),
        help="Configuration du projet (defaut: config.yaml)",
    )
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
    parser.add_argument("--dpi", type=int, help="Remplacer le DPI defini dans config.yaml")
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Remplacer le nombre maximal de pages defini dans config.yaml",
    )
    parser.add_argument(
        "--without-gapl",
        action="store_true",
        help="Desactiver explicitement l'analyse GAPL",
    )
    parser.add_argument(
        "--gapl-model-path",
        type=Path,
        help="Utiliser un autre checkpoint GAPL local",
    )
    parser.add_argument(
        "--ai-model",
        choices=("community-forensics",),
        help="Modele passif optionnel a charger explicitement",
    )
    parser.add_argument(
        "--ai-model-path",
        type=Path,
        help="Checkpoint local Community Forensics (.safetensors, .pt ou .pth)",
    )
    parser.add_argument(
        "--ai-model-download",
        action="store_true",
        help="Autoriser explicitement le telechargement des poids officiels epingles",
    )
    parser.add_argument(
        "--ai-model-variant",
        choices=("224", "384"),
        default="384",
        help="Variante Community Forensics (defaut: 384)",
    )
    parser.add_argument(
        "--ai-model-device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="Peripherique d'inference du modele passif (defaut: cpu)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output or Path("output") / args.input_file.stem
    password = args.password or os.environ.get("FRAUDE_PDF_PASSWORD")
    input_is_pdf = _is_pdf(args.input_file)

    try:
        project_config = load_run_config(args.config)
        ai_adapters = _build_ai_adapters(args, project_config.gapl)
        config = replace(
            project_config.analysis,
            render_dpi=(args.dpi if args.dpi is not None else project_config.analysis.render_dpi),
            max_pages=(
                args.max_pages if args.max_pages is not None else project_config.analysis.max_pages
            ),
        )
        if input_is_pdf:
            report = AnalysisPipeline(
                config=config,
                ai_image_adapters=ai_adapters,
            ).analyze(
                input_path=args.input_file,
                output_dir=output_dir,
                password=password,
            )
        else:
            report = ImageAnalysisPipeline(
                config=config,
                ai_image_adapters=ai_adapters,
            ).analyze(
                input_path=args.input_file,
                output_dir=output_dir,
            )
    except (
        AnalysisError,
        CommunityForensicsError,
        GaplError,
        RunConfigError,
        RuntimeError,
        ValueError,
    ) as error:
        if isinstance(error, AnalysisError):
            code = error.code
        elif isinstance(error, CommunityForensicsError):
            code = "ai_model_unavailable"
        elif isinstance(error, GaplError):
            code = "gapl_unavailable"
        else:
            code = "invalid_option"
        print(f"Erreur [{code}]: {error}", file=sys.stderr)
        return 1

    print(f"Type d'entree: {'PDF' if input_is_pdf else 'image'}")
    print(f"Niveau: {report.assessment.level}")
    print(f"Score de revue: {report.assessment.score}/100")
    print(f"Verdict: {report.assessment.label}")
    scored_findings = sum(finding.risk_points > 0 for finding in report.findings)
    diagnostics = len(report.findings) - scored_findings
    print(f"Signaux scores: {scored_findings}")
    print(f"Diagnostics non scores: {diagnostics}")
    if scored_findings:
        print("Indices scores:")
        for finding in report.findings:
            if finding.risk_points > 0:
                print(f"- {finding.code}: {finding.title} (+{finding.risk_points:g})")
    print(f"Rapport: {(output_dir / 'report.json').resolve()}")
    return 0


def _is_pdf(input_path: Path) -> bool:
    try:
        with input_path.expanduser().open("rb") as input_file:
            return b"%PDF-" in input_file.read(1024)
    except OSError:
        return input_path.suffix.casefold() == ".pdf"


def _build_ai_adapters(
    args: argparse.Namespace,
    gapl_config: GaplConfig | None = None,
) -> tuple[AiImageModelAdapter, ...]:
    adapters: list[AiImageModelAdapter] = []
    configured_gapl = gapl_config or load_run_config(args.config).gapl
    if configured_gapl.enabled and not args.without_gapl:
        gapl_path = args.gapl_model_path or configured_gapl.weights_path
        if not gapl_path.expanduser().is_file():
            raise GaplError("Le modele GAPL est absent. Executez ./start.sh pour le preparer.")
        device = (
            best_available_device() if configured_gapl.device == "auto" else configured_gapl.device
        )
        adapters.append(
            create_gapl_adapter(
                weights_path=gapl_path,
                device=device,
            )
        )

    model_options_used = (
        args.ai_model_path is not None
        or args.ai_model_download
        or args.ai_model_variant != "384"
        or args.ai_model_device != "cpu"
    )
    if args.ai_model is None:
        if model_options_used:
            raise ValueError("Les options --ai-model-* exigent --ai-model")
        return tuple(adapters)

    if args.ai_model_path is not None and args.ai_model_download:
        raise ValueError("Choisissez un checkpoint local ou --ai-model-download, pas les deux")
    adapters.append(
        create_community_forensics_adapter(
            weights_path=args.ai_model_path,
            variant=args.ai_model_variant,
            device=args.ai_model_device,
            allow_hf_download=args.ai_model_download,
        )
    )
    return tuple(adapters)


if __name__ == "__main__":
    raise SystemExit(main())
