"""Validate or run the Paper 1 clean latent geometry lane."""

from __future__ import annotations

import argparse
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
    run_with_failure_log,
    validate_contract,
    validate_execution_enabled,
    validate_input_path,
    write_dry_run_report,
    write_lane_audit,
)


LANE = "geometry"
OUTPUT_KEYS = ["geometry_scores", "audit", "failure_log"]


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
    raise CleanRerunValidationError(
        "Full geometry computation is cluster-backed and was not launched. "
        "Run after full clean external artifacts are generated for the planned rows."
    )


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
