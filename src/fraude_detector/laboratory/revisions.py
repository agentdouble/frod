"""Visual timeline across every retained incremental PDF revision."""

from __future__ import annotations

import pypdfium2

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.detectors.revision_diff import compare_revision_images
from fraude_detector.models import LaboratoryCheck, LaboratoryObservation
from fraude_detector.rendering import render_page


def analyze_all_revisions(context: AnalysisContext) -> LaboratoryCheck:
    """Compare every adjacent pair of valid retained revisions."""

    offsets = context.revision_end_offsets
    if len(offsets) < 2:
        return LaboratoryCheck(
            code="all_revisions",
            title="Historique complet",
            purpose="Comparer visuellement toutes les revisions incrementales conservees.",
            state="not_applicable",
            summary="Une seule revision lisible.",
            limitations=(
                "Une reecriture complete peut avoir efface les revisions plus anciennes.",
            ),
        )

    observations: list[LaboratoryObservation] = []
    total_changed_pages = 0
    total_artifacts = 0
    dpi = min(108, context.config.render_dpi)

    for transition_index, (before_offset, after_offset) in enumerate(
        zip(offsets, offsets[1:], strict=False),
        start=1,
    ):
        before = pypdfium2.PdfDocument(
            context.raw_pdf[:before_offset],
            password=context.password,
        )
        after = pypdfium2.PdfDocument(
            context.raw_pdf[:after_offset],
            password=context.password,
        )
        changed_pages: list[dict[str, object]] = []
        artifacts: list[str] = []
        try:
            comparable = min(len(before), len(after), context.config.max_pages)
            for page_index in range(comparable):
                before_page = before[page_index]
                after_page = after[page_index]
                try:
                    before_image = render_page(
                        before_page,
                        dpi,
                        context.config.max_render_pixels,
                    )
                    after_image = render_page(
                        after_page,
                        dpi,
                        context.config.max_render_pixels,
                    )
                    page_width, page_height = after_page.get_size()
                finally:
                    before_page.close()
                    after_page.close()

                findings, heatmap = compare_revision_images(
                    before_image,
                    after_image,
                    page_number=page_index + 1,
                    page_width=page_width,
                    page_height=page_height,
                )
                if not findings or heatmap is None:
                    continue
                changed_fraction = max(
                    float(item.evidence.get("document_changed_fraction", 0)) for item in findings
                )
                changed_pages.append(
                    {
                        "page": page_index + 1,
                        "changed_fraction": round(changed_fraction, 6),
                        "region_count": len(findings),
                    }
                )
                artifact_path = context.artifact_path(
                    "revision-timeline",
                    (
                        f"revision-{transition_index:03d}-to-"
                        f"{transition_index + 1:03d}-page-{page_index + 1:03d}.png"
                    ),
                )
                heatmap.save(artifact_path)
                artifacts.append(context.relative_artifact(artifact_path))

            page_count_changed = len(before) != len(after)
            total_changed_pages += len(changed_pages)
            total_artifacts += len(artifacts)
            if changed_pages or page_count_changed:
                observations.append(
                    LaboratoryObservation(
                        code="PDF_REVISION_TRANSITION_CHANGED",
                        title=(f"Revision {transition_index} vers {transition_index + 1}"),
                        summary=_transition_summary(
                            changed_pages,
                            len(before),
                            len(after),
                        ),
                        state="detected",
                        strength="moderate",
                        explanation=(
                            "La comparaison porte sur deux revisions consecutives conservees. "
                            "Elle localise un changement tangible, mais son caractere legitime "
                            "depend du processus metier."
                        ),
                        evidence={
                            "from_revision": transition_index,
                            "to_revision": transition_index + 1,
                            "before_page_count": len(before),
                            "after_page_count": len(after),
                            "changed_pages": changed_pages,
                        },
                        artifacts=tuple(artifacts),
                    )
                )
            else:
                observations.append(
                    LaboratoryObservation(
                        code="PDF_REVISION_TRANSITION_VISUALLY_IDENTICAL",
                        title=(f"Revision {transition_index} vers {transition_index + 1}"),
                        summary="Aucun changement visuel au seuil du laboratoire.",
                        state="clear",
                        strength="informational",
                        explanation=(
                            "La revision peut ne contenir que des metadonnees, une signature "
                            "ou une modification non visible au rendu."
                        ),
                        evidence={
                            "from_revision": transition_index,
                            "to_revision": transition_index + 1,
                            "page_count": len(after),
                        },
                    )
                )
        finally:
            before.close()
            after.close()

    changed_transitions = sum(
        item.code == "PDF_REVISION_TRANSITION_CHANGED" for item in observations
    )
    return LaboratoryCheck(
        code="all_revisions",
        title="Historique complet",
        purpose=(
            "Comparer chaque revision incrementale a la precedente, au lieu de limiter "
            "l'analyse aux deux dernieres."
        ),
        state="detected" if changed_transitions else "clear",
        summary=(
            f"{len(offsets)} revisions, {changed_transitions} transition(s) avec "
            f"changement visuel sur {total_changed_pages} page(s)."
        ),
        observations=tuple(observations),
        limitations=(
            "La comparaison est visuelle et n'interprete pas le sens du changement.",
            "Seules les pages comprises dans la limite d'analyse sont rendues.",
            "Une reecriture complete peut avoir efface les revisions anterieures.",
            f"Cartes de differences produites : {total_artifacts}.",
        ),
    )


def _transition_summary(
    changed_pages: list[dict[str, object]],
    before_count: int,
    after_count: int,
) -> str:
    parts = [f"{len(changed_pages)} page(s) visuellement modifiee(s)"]
    if before_count != after_count:
        parts.append(f"nombre de pages {before_count} vers {after_count}")
    return ", ".join(parts) + "."
