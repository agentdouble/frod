"""Conservative passive analysis of eligible photos embedded in PDFs."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from statistics import fmean

from fraude_detector.ai_images import (
    AiImageEvaluation,
    AiImageModelAdapter,
    consensus_evaluations,
    evaluate_ai_image_adapter,
    one_evaluation_per_family,
)
from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.image_assets import (
    ImageAssetDecodeError,
    decode_image_asset,
    extract_image_assets,
)
from fraude_detector.image_roles import RoutedImageAsset, route_image_assets
from fraude_detector.models import DetectorResult, Finding


class AiGeneratedImageDetector:
    """Use optional models only when routing and stability checks are conclusive."""

    name = "ai_generated_image"

    def __init__(self, adapters: Iterable[AiImageModelAdapter] = ()) -> None:
        self._adapters = tuple(adapters)

    def analyze(self, context: AnalysisContext) -> DetectorResult:
        assets = extract_image_assets(
            context,
            max_images=context.config.ai_max_inventory_images + 1,
        )
        inventory_truncated = len(assets) > context.config.ai_max_inventory_images
        inventoried_assets = assets[: context.config.ai_max_inventory_images]
        routed_assets = route_image_assets(inventoried_assets, context.config)
        role_counts = Counter(routed.role for routed in routed_assets)
        eligible_photos = tuple(routed for routed in routed_assets if routed.role == "photo")
        photo_limit_reached = len(eligible_photos) > context.config.ai_max_images
        photos = eligible_photos[: context.config.ai_max_images]
        notes = [
            f"Photos candidates: {len(eligible_photos)}; retenues: {len(photos)}.",
            "Roles image: "
            + ", ".join(
                f"{role}={role_counts.get(role, 0)}"
                for role in ("photo", "document", "decorative", "unknown")
            )
            + ".",
            f"Modeles passifs disponibles: {len(self._adapters)}.",
            "Un score de modele indique une trace statistique, pas une fraude documentaire.",
            "Le routage document/photo est geometrique; il ne remplace pas un filtre de contenu.",
        ]
        if inventory_truncated:
            notes.append(
                f"Inventaire limite aux {context.config.ai_max_inventory_images} premieres images."
            )
        if photo_limit_reached:
            notes.append(f"Analyse limitee aux {context.config.ai_max_images} premieres photos.")

        if not photos:
            notes.append(
                "Aucune photo embarquee eligible; images pleine page et logos exclus par prudence."
            )
            return DetectorResult(
                name=self.name,
                status="partial" if inventory_truncated else "not_applicable",
                notes=tuple(notes),
            )
        if not self._adapters:
            notes.append(
                "Analyse pixel IA non executee; l'absence de modele n'est pas un "
                "indice d'authenticite."
            )
            return DetectorResult(
                name=self.name,
                status="partial",
                notes=tuple(notes),
            )

        findings: list[Finding] = []
        artifacts: list[str] = []
        partial_reasons: list[str] = []
        if inventory_truncated:
            partial_reasons.append("image_inventory_limit_reached")
        if photo_limit_reached:
            partial_reasons.append("image_limit_reached")
        method_families = {
            str(getattr(adapter, "method_family", "")).strip()
            for adapter in self._adapters
            if str(getattr(adapter, "method_family", "")).strip()
        }
        if len(method_families) < context.config.ai_min_consensus_families:
            partial_reasons.append("independent_model_families_insufficient")

        for routed in photos:
            try:
                decoded = decode_image_asset(
                    context,
                    routed.asset,
                    max_pixels=context.config.max_embedded_image_pixels,
                )
            except ImageAssetDecodeError as error:
                partial_reasons.append(
                    f"page_{routed.asset.page}_image_{routed.asset.index}_decode:{error}"
                )
                continue

            try:
                evaluations = tuple(
                    evaluate_ai_image_adapter(
                        adapter,
                        decoded.image,
                        max_dimension=context.config.ai_max_image_dimension,
                        stability_max_delta=context.config.ai_stability_max_delta,
                    )
                    for adapter in self._adapters
                )
            finally:
                decoded.image.close()

            heatmaps = self._write_heatmaps(context, routed, evaluations)
            artifact = self._write_analysis_artifact(
                context,
                routed,
                decoded.source,
                evaluations,
                heatmaps,
            )
            artifacts.append(artifact)
            artifacts.extend(heatmaps.values())
            diagnostic_artifacts = (artifact, *heatmaps.values())

            for evaluation in evaluations:
                if evaluation.status == "error":
                    partial_reasons.append(
                        f"{evaluation.adapter_id}_error:{evaluation.error or 'unknown'}"
                    )
                elif evaluation.status == "out_of_domain":
                    partial_reasons.append(f"{evaluation.adapter_id}_out_of_domain")
                    findings.append(
                        self._quality_finding(
                            routed,
                            evaluation,
                            code="AI_ANALYSIS_OUT_OF_DOMAIN",
                            title="Modele IA hors de son domaine d'analyse",
                            description=(
                                "Le modele passif a refuse cette image; aucun indice "
                                "d'origine ne peut etre tire de ce resultat."
                            ),
                            artifacts=diagnostic_artifacts,
                        )
                    )
                elif evaluation.stable is False:
                    findings.append(
                        self._quality_finding(
                            routed,
                            evaluation,
                            code="AI_ANALYSIS_UNSTABLE",
                            title="Score IA instable apres transformations mineures",
                            description=(
                                "Le score varie trop apres recompression ou "
                                "redimensionnement; il est exclu du consensus."
                            ),
                            artifacts=diagnostic_artifacts,
                        )
                    )

            consensus = consensus_evaluations(
                evaluations,
                threshold=context.config.ai_model_score_threshold,
            )
            if len({item.method_family for item in consensus}) >= (
                context.config.ai_min_consensus_families
            ):
                selected = one_evaluation_per_family(consensus)
                scores = [item.original_score for item in selected]
                numeric_scores = [score for score in scores if score is not None]
                findings.append(
                    Finding(
                        detector=self.name,
                        code="AI_PIXEL_TRACE_CONSENSUS",
                        category="synthetic_media",
                        title="Consensus de traces statistiques de generation IA",
                        description=(
                            "Plusieurs familles de modeles passifs independantes "
                            "produisent des scores eleves et stables sur cette photo. "
                            "Ce signal doit etre corrobore par la provenance et le dossier."
                        ),
                        risk_points=35.0,
                        confidence=min(0.85, fmean(numeric_scores)),
                        page=routed.asset.page,
                        bbox=routed.asset.bbox,
                        evidence={
                            "image_sha256": routed.asset.sha256,
                            "image_role": routed.role,
                            "route_reason": routed.reason,
                            "threshold": context.config.ai_model_score_threshold,
                            "minimum_method_families": (context.config.ai_min_consensus_families),
                            "adapters": [
                                {
                                    "adapter_id": item.adapter_id,
                                    "model_version": item.model_version,
                                    "method_family": item.method_family,
                                    "score": item.original_score,
                                    "max_score_delta": item.max_score_delta,
                                }
                                for item in selected
                            ],
                        },
                        artifacts=diagnostic_artifacts,
                    )
                )
            elif consensus:
                for evaluation in consensus:
                    findings.append(
                        Finding(
                            detector=self.name,
                            code="AI_PIXEL_TRACE_SINGLE_MODEL",
                            category="analysis_quality",
                            title="Score IA eleve mais non corrobore",
                            description=(
                                "Un modele passif produit un score eleve et stable, "
                                "mais aucune famille independante ne le corrobore. "
                                "Le resultat est conserve comme diagnostic sans risque ajoute."
                            ),
                            risk_points=0.0,
                            confidence=0.5,
                            page=routed.asset.page,
                            bbox=routed.asset.bbox,
                            evidence={
                                "adapter_id": evaluation.adapter_id,
                                "model_version": evaluation.model_version,
                                "method_family": evaluation.method_family,
                                "score": evaluation.original_score,
                                "max_score_delta": evaluation.max_score_delta,
                                "threshold": context.config.ai_model_score_threshold,
                            },
                            artifacts=diagnostic_artifacts,
                        )
                    )

        if partial_reasons:
            notes.append("Analyse partielle: " + ", ".join(dict.fromkeys(partial_reasons)) + ".")
        return DetectorResult(
            name=self.name,
            status="partial" if partial_reasons else "completed",
            findings=tuple(findings),
            notes=tuple(notes),
            artifacts=tuple(dict.fromkeys(artifacts)),
        )

    def _write_analysis_artifact(
        self,
        context: AnalysisContext,
        routed: RoutedImageAsset,
        decode_source: str,
        evaluations: tuple[AiImageEvaluation, ...],
        heatmaps: dict[str, str],
    ) -> str:
        asset = routed.asset
        path = context.artifact_path(
            "forensics",
            "ai",
            f"page-{asset.page:03d}-image-{asset.index:02d}-analysis.json",
        )
        payload = {
            "asset": {
                "page": asset.page,
                "index": asset.index,
                "sha256": asset.sha256,
                "pixel_size": asset.pixel_size,
                "role": routed.role,
                "route_reason": routed.reason,
                "decode_source": decode_source,
            },
            "policy": {
                "score_threshold": context.config.ai_model_score_threshold,
                "stability_max_delta": context.config.ai_stability_max_delta,
                "minimum_consensus_families": context.config.ai_min_consensus_families,
            },
            "evaluations": [
                {
                    "adapter_id": item.adapter_id,
                    "model_version": item.model_version,
                    "method_family": item.method_family,
                    "status": item.status,
                    "variant_scores": {
                        score.name: score.synthetic_score for score in item.variant_scores
                    },
                    "stable": item.stable,
                    "max_score_delta": item.max_score_delta,
                    "out_of_domain_reason": item.out_of_domain_reason,
                    "error": item.error,
                    "details": item.details,
                    "heatmap": heatmaps.get(item.adapter_id),
                }
                for item in evaluations
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return context.relative_artifact(path)

    def _write_heatmaps(
        self,
        context: AnalysisContext,
        routed: RoutedImageAsset,
        evaluations: tuple[AiImageEvaluation, ...],
    ) -> dict[str, str]:
        paths: dict[str, str] = {}
        for evaluation in evaluations:
            if evaluation.heatmap is None:
                continue
            identifier = re.sub(r"[^a-zA-Z0-9_.-]+", "-", evaluation.adapter_id).strip("-")
            identifier = identifier or "adapter"
            path = context.artifact_path(
                "forensics",
                "ai",
                (
                    f"page-{routed.asset.page:03d}-image-{routed.asset.index:02d}-"
                    f"{identifier}-heatmap.png"
                ),
            )
            evaluation.heatmap.save(path, format="PNG")
            paths[evaluation.adapter_id] = context.relative_artifact(path)
        return paths

    def _quality_finding(
        self,
        routed: RoutedImageAsset,
        evaluation: AiImageEvaluation,
        *,
        code: str,
        title: str,
        description: str,
        artifacts: tuple[str, ...],
    ) -> Finding:
        return Finding(
            detector=self.name,
            code=code,
            category="analysis_quality",
            title=title,
            description=description,
            risk_points=0.0,
            confidence=1.0,
            page=routed.asset.page,
            bbox=routed.asset.bbox,
            evidence={
                "adapter_id": evaluation.adapter_id,
                "model_version": evaluation.model_version,
                "method_family": evaluation.method_family,
                "original_score": evaluation.original_score,
                "max_score_delta": evaluation.max_score_delta,
                "out_of_domain_reason": evaluation.out_of_domain_reason,
            },
            artifacts=artifacts,
        )
