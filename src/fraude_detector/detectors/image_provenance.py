"""Detect explicit AI provenance declarations in PDFs and embedded images."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from fraude_detector.detectors.base import AnalysisContext
from fraude_detector.image_assets import extract_image_assets
from fraude_detector.image_provenance import (
    AiDeclarationTrust,
    AiMetadataMarker,
    C2paAdapter,
    C2paSummary,
    EvidenceState,
    ImageProvenanceReport,
    ai_declaration_weight,
    analyze_c2pa_provenance,
    analyze_image_provenance,
    find_ai_metadata_markers,
)
from fraude_detector.image_roles import RoutedImageAsset, route_image_assets
from fraude_detector.models import DetectorResult, Finding


class ImageProvenanceDetector:
    """Inspect signed provenance and explicit unsigned generator metadata."""

    name = "image_provenance"

    def __init__(self, c2pa_adapter: C2paAdapter | None = None) -> None:
        self._c2pa_adapter = c2pa_adapter

    def analyze(self, context: AnalysisContext) -> DetectorResult:
        findings: list[Finding] = []
        artifacts: list[str] = []
        notes: list[str] = []
        partial_reasons: list[str] = []

        pdf_summary = analyze_c2pa_provenance(
            context.raw_pdf,
            "application/pdf",
            c2pa_adapter=self._c2pa_adapter,
        )
        pdf_findings, pdf_artifact = self._findings_for_c2pa(
            context=context,
            summary=pdf_summary,
            scope="pdf",
            routed=None,
        )
        findings.extend(pdf_findings)
        if pdf_artifact is not None:
            artifacts.append(pdf_artifact)
        if pdf_summary.state in {EvidenceState.UNAVAILABLE, EvidenceState.INDETERMINATE}:
            partial_reasons.append(f"pdf_c2pa_{pdf_summary.state.value}")

        all_assets = extract_image_assets(
            context,
            max_images=context.config.ai_max_inventory_images + 1,
        )
        inventory_truncated = len(all_assets) > context.config.ai_max_inventory_images
        inventoried_assets = all_assets[: context.config.ai_max_inventory_images]
        routed_inventory = route_image_assets(inventoried_assets, context.config)
        relevant_assets = tuple(
            routed for routed in routed_inventory if routed.role != "decorative"
        )
        relevant_truncated = len(relevant_assets) > context.config.ai_max_images
        routed_assets = relevant_assets[: context.config.ai_max_images]
        if inventory_truncated:
            partial_reasons.append("image_inventory_limit_reached")
        if relevant_truncated:
            partial_reasons.append("image_limit_reached")

        role_counts = Counter(routed.role for routed in routed_inventory)
        native_analyzed = 0
        native_unavailable = 0
        decorative_signals = 0
        c2pa_unavailable = 0

        for routed in routed_assets:
            asset = routed.asset
            if asset.native_bytes is None or asset.format is None:
                native_unavailable += 1
                partial_reasons.append(f"page_{asset.page}_image_{asset.index}_native_unavailable")
                continue

            native_analyzed += 1
            report = analyze_image_provenance(
                asset.native_bytes,
                asset.format,
                c2pa_adapter=self._c2pa_adapter,
                max_pixels=context.config.max_embedded_image_pixels,
            )
            if report.pillow.state is EvidenceState.INDETERMINATE:
                partial_reasons.append(
                    f"page_{asset.page}_image_{asset.index}_metadata_indeterminate"
                )
            if report.c2pa.state in {EvidenceState.UNAVAILABLE, EvidenceState.INDETERMINATE}:
                c2pa_unavailable += 1
                partial_reasons.append(
                    f"page_{asset.page}_image_{asset.index}_c2pa_{report.c2pa.state.value}"
                )

            image_findings, relative_artifact = self._findings_for_image(
                context=context,
                routed=routed,
                report=report,
            )
            findings.extend(image_findings)
            if relative_artifact is not None:
                artifacts.append(relative_artifact)

        # Decorative assets are deliberately not decoded or scored. This count
        # documents the routing decision without treating a logo as claim evidence.
        decorative_signals += role_counts.get("decorative", 0)
        notes.extend(
            (
                "Images inventoriees: "
                f"{len(inventoried_assets)} "
                f"(limite inventaire: {context.config.ai_max_inventory_images}).",
                "Roles image: "
                + ", ".join(
                    f"{role}={role_counts.get(role, 0)}"
                    for role in ("photo", "document", "decorative", "unknown")
                )
                + ".",
                "Images pertinentes retenues: "
                f"{len(routed_assets)} (limite analyse: {context.config.ai_max_images}).",
                f"Images natives analysees pour provenance: {native_analyzed}.",
                f"Images pertinentes sans flux natif autonome: {native_unavailable}.",
                f"Images decoratives exclues: {decorative_signals}.",
                f"Analyses C2PA image indisponibles ou indeterminees: {c2pa_unavailable}.",
                "Validation C2PA locale: manifestes distants et OCSP non charges.",
                "L'absence de C2PA ou de metadonnees ne prouve pas une origine humaine.",
            )
        )
        if partial_reasons:
            notes.append("Analyse partielle: " + ", ".join(dict.fromkeys(partial_reasons)) + ".")

        if partial_reasons:
            status = "partial"
        elif not inventoried_assets and pdf_summary.state is EvidenceState.NOT_DETECTED:
            status = "not_applicable"
        else:
            status = "completed"
        return DetectorResult(
            name=self.name,
            status=status,
            findings=tuple(findings),
            notes=tuple(notes),
            artifacts=tuple(dict.fromkeys(artifacts)),
        )

    def _findings_for_image(
        self,
        *,
        context: AnalysisContext,
        routed: RoutedImageAsset,
        report: ImageProvenanceReport,
    ) -> tuple[list[Finding], str | None]:
        asset = routed.asset
        markers = find_ai_metadata_markers(report.pillow)
        has_c2pa_signal = (
            report.c2pa.provenance_invalidated
            or report.c2pa.ai_declaration.state is EvidenceState.DETECTED
        )
        if not has_c2pa_signal and not markers:
            return [], None

        artifact = self._write_image_artifact(context, routed, report, markers)
        findings, _ = self._findings_for_c2pa(
            context=context,
            summary=report.c2pa,
            scope="image",
            routed=routed,
            artifact=artifact,
        )
        if markers:
            findings.append(
                Finding(
                    detector=self.name,
                    code="AI_GENERATOR_METADATA_MENTIONED",
                    category="synthetic_media",
                    title="Generateur IA mentionne dans les metadonnees image",
                    description=(
                        "Les metadonnees non signees de l'image mentionnent explicitement "
                        "un outil ou des parametres de generation. Elles sont falsifiables "
                        "et constituent seulement un indice a corroborer."
                    ),
                    risk_points=15.0,
                    confidence=0.65,
                    page=asset.page,
                    bbox=asset.bbox,
                    evidence={
                        "image_sha256": asset.sha256,
                        "image_role": routed.role,
                        "markers": [marker.marker for marker in markers],
                        "metadata_keys": sorted({marker.metadata_key for marker in markers}),
                    },
                    artifacts=(artifact,),
                )
            )
        return findings, artifact

    def _findings_for_c2pa(
        self,
        *,
        context: AnalysisContext,
        summary: C2paSummary,
        scope: str,
        routed: RoutedImageAsset | None,
        artifact: str | None = None,
    ) -> tuple[list[Finding], str | None]:
        findings: list[Finding] = []
        asset = routed.asset if routed is not None else None
        if not (
            summary.provenance_invalidated or summary.ai_declaration.state is EvidenceState.DETECTED
        ):
            return findings, artifact

        relative_artifact = artifact
        if relative_artifact is None:
            relative_artifact = self._write_pdf_artifact(context, summary)

        evidence: dict[str, Any] = {
            "scope": scope,
            "validation": summary.validation.value,
            "raw_validation_state": summary.raw_validation_state,
            "claim_generator": summary.claim_generator,
            "active_manifest_label": summary.active_manifest_label,
        }
        if asset is not None:
            evidence.update(
                {
                    "image_sha256": asset.sha256,
                    "image_role": routed.role,
                }
            )

        if summary.provenance_invalidated:
            findings.append(
                Finding(
                    detector=self.name,
                    code="IMAGE_C2PA_INVALID" if asset is not None else "PDF_C2PA_INVALID",
                    category="provenance_integrity",
                    title="Provenance C2PA invalidee",
                    description=(
                        "Le manifeste de provenance est present mais son integrite ou "
                        "sa signature ne se valide pas. Cela signale une provenance "
                        "alteree, pas necessairement une generation par IA."
                    ),
                    risk_points=18.0,
                    confidence=0.9,
                    page=asset.page if asset is not None else None,
                    bbox=asset.bbox if asset is not None else None,
                    evidence=evidence
                    | {
                        "failure_codes": [
                            status.code
                            for status in summary.statuses
                            if status.outcome.casefold() == "failure"
                        ]
                    },
                    artifacts=(relative_artifact,),
                )
            )

        declaration = summary.ai_declaration
        if (
            declaration.state is EvidenceState.DETECTED
            and declaration.trust is not AiDeclarationTrust.INVALIDATED
        ):
            points, confidence = ai_declaration_weight(declaration.trust)
            findings.append(
                Finding(
                    detector=self.name,
                    code=(
                        "AI_IMAGE_C2PA_DECLARATION"
                        if asset is not None
                        else "AI_PDF_C2PA_DECLARATION"
                    ),
                    category="synthetic_media",
                    title="Origine IA declaree dans une provenance C2PA",
                    description=(
                        "Une provenance C2PA declare une creation ou une composition "
                        "algorithmique. Le niveau de confiance depend de la validation "
                        "et de la confiance accordee au certificat signataire."
                    ),
                    risk_points=points,
                    confidence=confidence,
                    page=asset.page if asset is not None else None,
                    bbox=asset.bbox if asset is not None else None,
                    evidence=evidence
                    | {
                        "declaration_trust": declaration.trust.value,
                        "digital_source_types": list(declaration.digital_source_types),
                    },
                    artifacts=(relative_artifact,),
                )
            )
        return findings, relative_artifact

    def _write_image_artifact(
        self,
        context: AnalysisContext,
        routed: RoutedImageAsset,
        report: ImageProvenanceReport,
        markers: tuple[AiMetadataMarker, ...],
    ) -> str:
        asset = routed.asset
        path = context.artifact_path(
            "forensics",
            "ai",
            f"page-{asset.page:03d}-image-{asset.index:02d}-provenance.json",
        )
        payload = {
            "asset": {
                "page": asset.page,
                "index": asset.index,
                "sha256": asset.sha256,
                "role": routed.role,
                "route_reason": routed.reason,
                "pixel_size": asset.pixel_size,
                "format": asset.format,
            },
            "metadata": {
                "state": report.pillow.state.value,
                "exif_keys": [entry.key for entry in report.pillow.exif.entries],
                "xmp_keys": [entry.key for entry in report.pillow.xmp.entries],
                "info_keys": [entry.key for entry in report.pillow.info.entries],
                "ai_markers": [marker.marker for marker in markers],
            },
            "c2pa": _c2pa_artifact_payload(report.c2pa),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return context.relative_artifact(path)

    def _write_pdf_artifact(
        self,
        context: AnalysisContext,
        summary: C2paSummary,
    ) -> str:
        path = context.artifact_path("forensics", "ai", "document-provenance.json")
        path.write_text(
            json.dumps(
                {"scope": "pdf", "c2pa": _c2pa_artifact_payload(summary)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return context.relative_artifact(path)


def _c2pa_artifact_payload(summary: C2paSummary) -> dict[str, Any]:
    return {
        "state": summary.state.value,
        "validation": summary.validation.value,
        "raw_validation_state": summary.raw_validation_state,
        "active_manifest_label": summary.active_manifest_label,
        "claim_generator": summary.claim_generator,
        "embedded": summary.embedded,
        "remote_url": summary.remote_url,
        "statuses": [
            {
                "outcome": status.outcome,
                "code": status.code,
                "explanation": status.explanation,
            }
            for status in summary.statuses
        ],
        "ai_declaration": {
            "state": summary.ai_declaration.state.value,
            "trust": summary.ai_declaration.trust.value,
            "digital_source_types": list(summary.ai_declaration.digital_source_types),
        },
        "error": summary.error,
    }
