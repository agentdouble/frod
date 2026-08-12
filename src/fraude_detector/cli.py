"""Command-line interface for the fraud-risk analysis pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from fraude_detector import __version__
from fraude_detector.ai_images import AiImageModelAdapter
from fraude_detector.batch import (
    BatchResult,
    analyze_document,
    batch_document_output_dir,
    export_csv,
    export_html,
    export_json,
    process_batch,
)
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

try:
    from tqdm import tqdm as tqdm_lib

    HAS_TQDM = True
except ImportError:
    tqdm_lib = None
    HAS_TQDM = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frod",
        description=(
            "Analyse un PDF ou une image et produit des indices explicables de "
            "modification avec artefacts visuels."
        ),
    )
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        help="Fichier PDF/image ou dossier a analyser",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Mode batch: analyser un dossier de documents",
    )
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
        help="Dossier de sortie (defaut: output/<nom>)",
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
        "--gapl-device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Peripherique GAPL (defaut: auto)",
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Afficher toutes les trouvailles, pas seulement celles avec score",
    )
    # Batch-specific options
    parser.add_argument(
        "--extensions",
        type=str,
        default="pdf,png,jpg,jpeg,webp",
        help="Extensions de fichiers a traiter en mode batch (defaut: pdf,png,jpg,jpeg,webp)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv,html,json",
        help="Formats de sortie batch (csv,html,json, defaut: csv,html,json)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Batch mode
    if args.batch:
        return _handle_batch(args)

    # Single file mode (default)
    if args.input_file is None:
        parser.print_help()
        return 1

    return _handle_analyze(args)


def _handle_analyze(args: argparse.Namespace) -> int:
    """Handle single file analysis."""
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

    # Check if OCR was enabled and ran
    ocr_detector = next(
        (d for d in report.detectors if d.name == "ocr_content"),
        None,
    )
    if ocr_detector:
        ocr_status = "OK" if ocr_detector.status == "success" else ocr_detector.status
        print(f"OCR: {ocr_status}")
        if ocr_detector.findings:
            ocr_scored = sum(f.risk_points > 0 for f in ocr_detector.findings)
            print(f"  - Signaux OCR: {ocr_scored}/{len(ocr_detector.findings)}")

    scored_findings = sum(finding.risk_points > 0 for finding in report.findings)
    diagnostics = len(report.findings) - scored_findings
    print(f"Signaux scores: {scored_findings}")
    print(f"Diagnostics non scores: {diagnostics}")
    if scored_findings:
        print("Indices scores:")
        for finding in report.findings:
            if finding.risk_points > 0:
                print(f"- {finding.code}: {finding.title} (+{finding.risk_points:g})")

    # Show all findings in verbose mode
    if args.verbose and report.findings:
        print("Toutes les trouvailles:")
        for finding in report.findings:
            score_str = (
                f" (+{finding.risk_points:g})" if finding.risk_points > 0 else " (diagnostic)"
            )
            print(f"- {finding.code}: {finding.title}{score_str}")

    # Show detector statuses
    print("Detecteurs:")
    for detector in report.detectors:
        status_icon = "✓" if detector.status in ("success", "completed") else "⚠"
        findings_count = len(detector.findings)
        scored_count = sum(f.risk_points > 0 for f in detector.findings)
        print(
            f"  {status_icon} {detector.name}: "
            f"{findings_count} trouvailles ({scored_count} scorees)"
        )

    print(f"Rapport: {(output_dir / 'report.json').resolve()}")
    return 0


def _handle_batch(args: argparse.Namespace) -> int:
    """Handle batch analysis."""
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / f"batch_{timestamp}"

    # Get input folder
    input_folder = args.input_file
    if input_folder is None:
        print("Erreur: Veuillez specifier un dossier a analyser", file=sys.stderr)
        return 1

    input_folder = input_folder.expanduser().resolve()
    if not input_folder.is_dir():
        print(f"Erreur: Le dossier n'existe pas: {input_folder}", file=sys.stderr)
        return 1

    extensions = [ext.strip().lower().lstrip(".") for ext in args.extensions.split(",")]
    file_patterns = [f"*.{ext}" for ext in extensions]
    input_files = []
    for pattern in file_patterns:
        input_files.extend(input_folder.glob(pattern))
        input_files.extend(input_folder.glob(f"**/{pattern}"))

    # Remove duplicates and sort
    input_files = sorted(set(input_files))

    if not input_files:
        print(f"Aucun fichier trouve avec les extensions: {args.extensions}", file=sys.stderr)
        return 1

    print(f"Batch: {len(input_files)} fichiers a analyser")
    print(f"Dossier de sortie: {output_dir}")
    password = args.password or os.environ.get("FRAUDE_PDF_PASSWORD")
    formats = [output_format.strip().lower() for output_format in args.format.split(",")]
    unknown_formats = sorted(set(formats) - {"csv", "html", "json"})
    if unknown_formats:
        print(
            f"Erreur: formats de sortie inconnus: {', '.join(unknown_formats)}",
            file=sys.stderr,
        )
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        project_config = load_run_config(args.config)
        ai_adapters = _build_ai_adapters(args, project_config.gapl)
        config = project_config.analysis

        # Process batch with progress bar
        print("Demarrage de l'analyse...")

        if HAS_TQDM:
            results = []
            start_time = time.time()
            started_at = datetime.now().isoformat()

            with tqdm_lib(total=len(input_files), desc="Analyse", unit="doc", ncols=80) as pbar:
                for path in input_files:
                    doc_output_dir = batch_document_output_dir(output_dir, path)
                    result = analyze_document(
                        path,
                        config,
                        doc_output_dir,
                        ai_adapters,
                        password,
                    )
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({"current": path.name[:30]})

            total_time = time.time() - start_time

            batch_result = BatchResult(
                summaries=results,
                total_processing_time_seconds=total_time,
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
            )
        else:
            # Without tqdm - simple progress
            batch_result = process_batch(
                input_paths=input_files,
                config=config,
                output_dir=output_dir,
                ai_adapters=ai_adapters,
                password=password,
            )

    except (
        AnalysisError,
        CommunityForensicsError,
        GaplError,
        RunConfigError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Erreur: {error}", file=sys.stderr)
        return 1

    # Export results
    print("\nExport des resultats...")
    if "csv" in formats:
        csv_path = output_dir / "summary.csv"
        export_csv(batch_result, csv_path)
        print(f"  CSV: {csv_path}")

    if "html" in formats:
        html_path = output_dir / "report.html"
        export_html(batch_result, html_path)
        print(f"  HTML: {html_path}")

    if "json" in formats:
        json_path = output_dir / "results.json"
        export_json(batch_result, json_path)
        print(f"  JSON: {json_path}")

    # Summary
    level_counts = batch_result.get_level_counts()
    print("\n=== Resume ===")
    print(f"Total: {len(batch_result.summaries)} documents")
    print(f"  Succes: {batch_result.successful_count}")
    print(f"  Echecs: {batch_result.failed_count}")
    print(f"  Score moyen: {batch_result.get_avg_score():.1f}/100")
    print(
        f"  Faible: {level_counts['low']} | Revue: {level_counts['review']} | "
        f"Eleve: {level_counts['high']}"
    )
    print(f"Temps total: {batch_result.total_processing_time_seconds:.1f}s")

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
        requested_device = args.gapl_device or configured_gapl.device
        device = best_available_device() if requested_device == "auto" else requested_device
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
