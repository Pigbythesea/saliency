"""Validate or run the Paper 1 clean latent geometry lane."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any

from paper1_clean_rerun_common import (
    PROJECT_ROOT,
    CleanRerunValidationError,
    audit_row,
    dry_run_report_path,
    eligible_model_rows,
    load_clean_config,
    model_identity,
    output_paths_from_config,
    pipe_join,
    read_csv,
    reject_forbidden_path,
    run_with_failure_log,
    validate_contract,
    validate_execution_enabled,
    validate_input_path,
    write_csv,
    write_dry_run_report,
    write_lane_audit,
)


LANE = "geometry"
OUTPUT_KEYS = [
    "geometry_scores",
    "geometry_method_agreement",
    "geometry_seed_stability",
    "audit",
    "failure_log",
]
METHOD_AGREEMENT_FIELDNAMES = [
    "model_id",
    "subject_id",
    "roi",
    "stream",
    "roi_class",
    "layer",
    "cka_method",
    "subset_rsa_method",
    "cka_score",
    "subset_rsa_mean_score",
    "subset_rsa_seed_count",
    "subset_rsa_size_count",
    "absolute_score_delta",
    "agreement_status",
    "evidence_status",
]
SEED_STABILITY_FIELDNAMES = [
    "model_id",
    "subject_id",
    "roi",
    "stream",
    "roi_class",
    "layer",
    "geometry_method",
    "subset_size",
    "seed_count",
    "mean_score",
    "std_score",
    "min_score",
    "max_score",
    "score_range",
    "stability_status",
    "evidence_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper1_latent_neural_matrix.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_with_failure_log(LANE, lambda: run(args.config, dry_run=args.dry_run))


def run(config_path: str, *, dry_run: bool) -> int:
    config = load_clean_config(config_path, lane="latent_neural_encoding")
    validate_contract(config, require_authorized=not dry_run)
    validate_execution_enabled(config, dry_run=dry_run)
    outputs = output_paths_from_config(config, OUTPUT_KEYS)
    subject_roi_rows = available_subject_roi_rows(config)
    model_rows = eligible_model_rows("geometry")
    layer_map = model_layer_map(config)
    methods = [str(value) for value in dict(config.get("geometry", {})).get("primary", [])]
    if not methods:
        raise CleanRerunValidationError("Geometry config has no primary methods")
    planned_rows = build_planned_rows(model_rows, subject_roi_rows, layer_map, methods)
    audit_rows = [
        audit_row(
            lane=LANE,
            check_id="publication_paths",
            status="pass",
            artifact="geometry_outputs",
            path=pipe_join(str(path.relative_to(PROJECT_ROOT)) for path in outputs.values()),
            detail="configured geometry outputs are under outputs/paper1_publication_v0",
        ),
        audit_row(
            lane=LANE,
            check_id="roi_stream_scope",
            status="pass",
            artifact="subject_roi_availability",
            path=config["roi_stream"]["subject_roi_availability"],
            detail=f"available_subject_roi_pairs={len(subject_roi_rows)}",
        ),
        audit_row(
            lane=LANE,
            check_id="planned_geometry_rows",
            status="pass",
            artifact="geometry_plan",
            path=dry_run_report_path(LANE).relative_to(PROJECT_ROOT),
            detail=f"planned_model_subject_roi_layer_method_rows={len(planned_rows)}",
        ),
    ]
    audit_path = write_lane_audit(outputs["audit"], audit_rows)
    if dry_run:
        report = write_dry_run_report(
            lane=LANE,
            config_path=config_path,
            expected_outputs=outputs,
            planned_rows=planned_rows,
            audit_path=audit_path,
            extra={
                "models": [row["model_id"] for row in model_rows],
                "geometry_methods": methods,
            },
        )
        print(f"dry_run_passed={report.relative_to(PROJECT_ROOT)}")
        return 0
    cell_files = clean_cell_score_files(outputs["geometry_scores"].parent / "cells")
    rows = materialize_geometry_scores(
        cell_files,
        model_rows=model_rows,
        subject_roi_rows=subject_roi_rows,
    )
    method_agreement_rows = build_geometry_method_agreement_rows(rows)
    seed_stability_rows = build_geometry_seed_stability_rows(rows)
    write_csv(outputs["geometry_scores"], rows)
    write_csv(
        outputs["geometry_method_agreement"],
        method_agreement_rows,
        fieldnames=METHOD_AGREEMENT_FIELDNAMES,
    )
    write_csv(
        outputs["geometry_seed_stability"],
        seed_stability_rows,
        fieldnames=SEED_STABILITY_FIELDNAMES,
    )
    audit_rows.append(
        audit_row(
            lane=LANE,
            check_id="geometry_scores_written",
            status="pass",
            artifact="geometry_scores",
            path=outputs["geometry_scores"].relative_to(PROJECT_ROOT),
            detail=f"source_cell_files={len(cell_files)}; rows={len(rows)}",
        )
    )
    audit_rows.extend(
        [
            audit_row(
                lane=LANE,
                check_id="geometry_method_agreement_written",
                status="pass",
                artifact="geometry_method_agreement",
                path=outputs["geometry_method_agreement"].relative_to(PROJECT_ROOT),
                detail=f"rows={len(method_agreement_rows)}",
            ),
            audit_row(
                lane=LANE,
                check_id="geometry_seed_stability_written",
                status="pass",
                artifact="geometry_seed_stability",
                path=outputs["geometry_seed_stability"].relative_to(PROJECT_ROOT),
                detail=f"rows={len(seed_stability_rows)}",
            ),
        ]
    )
    write_lane_audit(outputs["audit"], audit_rows)
    print(outputs["geometry_scores"].relative_to(PROJECT_ROOT))
    print(outputs["geometry_method_agreement"].relative_to(PROJECT_ROOT))
    print(outputs["geometry_seed_stability"].relative_to(PROJECT_ROOT))
    return 0


def clean_cell_score_files(root: Any) -> list[Any]:
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(root.rglob("geometry_scores.csv")):
        reject_forbidden_path(path, "geometry.cell_score")
        files.append(path)
    if not files:
        raise CleanRerunValidationError(
            "No clean geometry cell scores found under "
            f"{root.relative_to(PROJECT_ROOT)}. Generate clean geometry cell jobs first."
        )
    return files


def materialize_geometry_scores(
    cell_files: list[Any],
    *,
    model_rows: list[dict[str, str]],
    subject_roi_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    identity = {row["model_id"]: model_identity(row) for row in model_rows}
    roi_scope = {
        (row.get("subject_id", ""), row.get("roi", "")): row
        for row in subject_roi_rows
    }
    output_rows: list[dict[str, Any]] = []
    for path in cell_files:
        for row in read_csv(path):
            model_id = str(row.get("model") or row.get("model_id") or "")
            subject_id = str(row.get("subject_id", ""))
            roi = str(row.get("roi", ""))
            scope = roi_scope.get((subject_id, roi), {})
            enriched = {
                **identity.get(model_id, {"model_id": model_id}),
                **row,
                "model_id": model_id,
                "subject_id": subject_id,
                "roi": roi,
                "stream": scope.get("stream", row.get("stream", "")),
                "roi_class": scope.get("roi_class", row.get("roi_class", "")),
                "source_cell_score": str(path.relative_to(PROJECT_ROOT)),
                "evidence_status": "clean_publication_rerun",
            }
            output_rows.append(enriched)
    if not output_rows:
        raise CleanRerunValidationError("Clean geometry cell files contained no rows")
    return output_rows


def build_geometry_method_agreement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("control_type", "observed") or "observed") != "observed":
            continue
        grouped[geometry_group_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        cka_scores = [
            score
            for row in group_rows
            if str(row.get("geometry_method", "")) == "debiased_linear_cka"
            for score in [parse_score(row.get("score"))]
            if score is not None
        ]
        subset_scores = [
            score
            for row in group_rows
            if str(row.get("geometry_method", "")) == "subset_rsa"
            for score in [parse_score(row.get("score"))]
            if score is not None
        ]
        if not cka_scores or not subset_scores:
            continue
        subset_sizes = {
            str(row.get("subset_size", ""))
            for row in group_rows
            if str(row.get("geometry_method", "")) == "subset_rsa" and str(row.get("subset_size", ""))
        }
        subset_seeds = {
            str(row.get("subset_seed", ""))
            for row in group_rows
            if str(row.get("geometry_method", "")) == "subset_rsa" and str(row.get("subset_seed", ""))
        }
        cka_score = mean(cka_scores)
        subset_mean = mean(subset_scores)
        model_id, subject_id, roi, stream, roi_class, layer = key
        output.append(
            {
                "model_id": model_id,
                "subject_id": subject_id,
                "roi": roi,
                "stream": stream,
                "roi_class": roi_class,
                "layer": layer,
                "cka_method": "debiased_linear_cka",
                "subset_rsa_method": "subset_rsa_spearman",
                "cka_score": cka_score,
                "subset_rsa_mean_score": subset_mean,
                "subset_rsa_seed_count": len(subset_seeds),
                "subset_rsa_size_count": len(subset_sizes),
                "absolute_score_delta": abs(cka_score - subset_mean),
                "agreement_status": "paired_primary_methods_present",
                "evidence_status": "clean_publication_rerun",
            }
        )
    return output


def build_geometry_seed_stability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("control_type", "observed") or "observed") != "observed":
            continue
        if str(row.get("geometry_method", "")) != "subset_rsa":
            continue
        score = parse_score(row.get("score"))
        if score is None:
            continue
        key = (
            *geometry_group_key(row),
            str(row.get("subset_size", "")),
        )
        grouped[key].append(score)

    output: list[dict[str, Any]] = []
    for key, scores in sorted(grouped.items()):
        model_id, subject_id, roi, stream, roi_class, layer, subset_size = key
        score_mean = mean(scores)
        score_min = min(scores)
        score_max = max(scores)
        output.append(
            {
                "model_id": model_id,
                "subject_id": subject_id,
                "roi": roi,
                "stream": stream,
                "roi_class": roi_class,
                "layer": layer,
                "geometry_method": "subset_rsa_spearman",
                "subset_size": subset_size,
                "seed_count": len(scores),
                "mean_score": score_mean,
                "std_score": std(scores),
                "min_score": score_min,
                "max_score": score_max,
                "score_range": score_max - score_min,
                "stability_status": "ok" if len(scores) >= 2 else "single_seed_only",
                "evidence_status": "clean_publication_rerun",
            }
        )
    return output


def geometry_group_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("model_id", "") or row.get("model", "")),
        str(row.get("subject_id", "")),
        str(row.get("roi", "")),
        str(row.get("stream", "")),
        str(row.get("roi_class", "")),
        str(row.get("layer", "")),
    )


def parse_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return float(math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)))


def available_subject_roi_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    availability_path = validate_input_path(
        dict(config.get("roi_stream", {})).get(
            "subject_roi_availability",
            "outputs/paper1_publication_v0/roi_stream/subject_roi_availability.csv",
        ),
        "roi_stream.subject_roi_availability",
    )
    rows = [
        row
        for row in read_csv(availability_path)
        if row.get("available") == "yes"
    ]
    if not rows:
        raise CleanRerunValidationError("No available subject/ROI rows for geometry")
    return rows


def model_layer_map(config: dict[str, Any]) -> dict[str, list[str]]:
    model_config = dict(config.get("models", {}))
    layers: dict[str, list[str]] = {}
    for section in ("image_only", "conditioned"):
        entries = dict(model_config.get(section, {}))
        for model_id, value in entries.items():
            layer_values = value.get("layers", []) if isinstance(value, dict) else value
            layers[str(model_id)] = [str(layer) for layer in layer_values]
    if not layers:
        raise CleanRerunValidationError("Geometry config has no model layers")
    return layers


def build_planned_rows(
    models: list[dict[str, str]],
    subject_rois: list[dict[str, str]],
    layers: dict[str, list[str]],
    methods: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        model_id = model["model_id"]
        model_layers = layers.get(model_id) or [
            value for value in str(model.get("latent_tensor_layers", "")).split("|") if value
        ]
        if not model_layers:
            raise CleanRerunValidationError(f"No layers available for model {model_id}")
        for subject_roi in subject_rois:
            for layer in model_layers:
                for method in methods:
                    rows.append(
                        {
                            **model_identity(model),
                            "subject_id": subject_roi.get("subject_id", ""),
                            "roi": subject_roi.get("roi", ""),
                            "stream": subject_roi.get("stream", ""),
                            "roi_class": subject_roi.get("roi_class", ""),
                            "manifest_path": subject_roi.get("manifest_path", ""),
                            "layer": layer,
                            "geometry_method": method,
                            "evidence_type": "latent_feature_geometry",
                            "score_status": "planned_clean_publication_rerun",
                        }
                    )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
