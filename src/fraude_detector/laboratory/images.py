"""Experimental controls for standalone raster images."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fraude_detector.models import (
    LaboratoryCheck,
    LaboratoryObservation,
    LaboratoryReport,
)
from fraude_detector.trufor import (
    TRUFOR_DEFAULT_CHECKPOINT,
    TRUFOR_MAX_PIXELS,
    TRUFOR_TIMEOUT_SECONDS,
    TruForError,
    analyze_trufor_image,
)


def analyze_image_laboratory(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    trufor_weights: str | Path = TRUFOR_DEFAULT_CHECKPOINT,
    trufor_max_pixels: int = TRUFOR_MAX_PIXELS,
    trufor_timeout_seconds: int = TRUFOR_TIMEOUT_SECONDS,
    progress_callback: Callable[[float, str], None] | None = None,
) -> LaboratoryReport:
    """Run non-scoring image-only experiments without changing Frod findings."""

    try:
        analysis = analyze_trufor_image(
            input_path,
            output_dir,
            weights_path=trufor_weights,
            max_pixels=trufor_max_pixels,
            timeout_seconds=trufor_timeout_seconds,
            progress_callback=progress_callback,
        )
    except TruForError as error:
        return LaboratoryReport(
            schema_version="1.0",
            checks=(
                LaboratoryCheck(
                    code="trufor",
                    title="Retouches locales - TruFor",
                    purpose="Rechercher des incoherences locales dues a une retouche d'image.",
                    state="error",
                    summary="Analyse TruFor indisponible.",
                    observations=(
                        LaboratoryObservation(
                            code="TRUFOR_UNAVAILABLE",
                            title="Controle non execute",
                            summary=str(error),
                            state="error",
                            strength="informational",
                            explanation=(
                                "Ce controle experimental n'a aucun effet sur le score Frod."
                            ),
                        ),
                    ),
                    limitations=_limitations(),
                ),
            ),
        )

    state, strength, interpretation = _interpret_score(analysis.score)
    ratio = analysis.reliable_suspect_ratio
    resize_note = (
        f" Image reduite de {analysis.original_size[0]} x {analysis.original_size[1]} "
        f"a {analysis.analyzed_size[0]} x {analysis.analyzed_size[1]} pixels."
        if analysis.resized
        else " Image analysee a sa resolution d'origine."
    )
    return LaboratoryReport(
        schema_version="1.0",
        checks=(
            LaboratoryCheck(
                code="trufor",
                title="Retouches locales - TruFor",
                purpose="Rechercher des incoherences locales dues a une retouche d'image.",
                state=state,
                summary=f"Indice TruFor : {analysis.score:.0%}. {interpretation}",
                observations=(
                    LaboratoryObservation(
                        code="TRUFOR_LOCAL_MANIPULATION",
                        title=f"Indice de retouche : {analysis.score:.0%}",
                        summary=(f"Zones a la fois suspectes et fiables : {ratio:.1%} de l'image."),
                        state=state,
                        strength=strength,
                        explanation=(
                            "La premiere carte montre la sortie brute de TruFor. La seconde "
                            "attenue les zones que le modele estime moins fiables." + resize_note
                        ),
                        evidence={
                            "trufor_score": analysis.score,
                            "reliable_suspect_ratio": ratio,
                            "original_size": list(analysis.original_size),
                            "analyzed_size": list(analysis.analyzed_size),
                            "resized": analysis.resized,
                            "scored_by_frod": False,
                        },
                        artifacts=analysis.artifacts,
                    ),
                ),
                limitations=_limitations(),
            ),
        ),
    )


def _interpret_score(score: float) -> tuple[str, str, str]:
    if score >= 0.75:
        return (
            "attention",
            "moderate",
            "Le modele met en evidence une incoherence locale a examiner.",
        )
    if score >= 0.40:
        return (
            "indeterminate",
            "weak",
            "Le resultat est intermediaire et ne permet pas de conclure.",
        )
    return (
        "clear",
        "informational",
        "Aucune incoherence locale nette n'est mise en evidence.",
    )


def _limitations() -> tuple[str, ...]:
    return (
        "L'indice TruFor n'est pas une probabilite de fraude.",
        "Les seuils ne sont pas calibres sur les documents d'assurance.",
        "Une retouche IA recente, une recompression ou une zone tres petite "
        "peut ne pas etre detectee.",
        "Un score faible ne prouve pas que l'image est authentique.",
        "TruFor est limite par sa licence amont aux usages informatifs et non lucratifs.",
    )
