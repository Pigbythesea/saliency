"""Assemble clean Paper 1 cross-axis coverage without interpretation."""

from __future__ import annotations

import argparse
from pathlib import Path
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
    require_publication_path,
    run_with_failure_log,
    validate_contract,
    validate_execution_enabled,
    write_csv,
    write_dry_run_report,
    write_lane_audit,
)


LANE = "cross_axis"
OUTPUT_KEYS = ["model_axis_scores", "clean_rerun_audit", "failure_log"]
AXES = ["behavioral", "latent_neural_encoding", "geometry", "efficiency_resource"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper1_cross_axis_assembly.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_with_failure_log(LANE, lambda: run(args.config, dry_run=args.dry_run))


def run(config_path: str, *, dry_run: bool) -> int:
    config = load_clean_config(config_path, lane=LANE)
    validate_contract(config, require_authorized=not dry_run)
    validate_execution_enabled(config, dry_run=dry_run)
    outputs = output_paths_from_config(config, OUTPUT_KEYS)
    input_paths = clean_input_paths(config)
    planned_rows = build_planned_rows(input_paths)
    missing_inputs = [
        axis for axis, path in input_paths.items() if not path.is_file()
    ]
    audit_rows = [
        audit_row(
            lane=LANE,
            check_id="publication_paths",
            status="pass",
            artifact="cross_axis_outputs",
            path=pipe_join(str(path.relative_to(PROJECT_ROOT)) for path in outputs.values()),
            detail="configured cross-axis outputs are under outputs/paper1_publication_v0",
        ),
        audit_row(
            lane=LANE,
            check_id="clean_input_paths",
            status="pass",
            artifact="cross_axis_inputs",
            path=pipe_join(str(path.relative_to(PROJECT_ROOT)) for path in input_paths.values()),
            detail="all cross-axis inputs point to clean publication-root outputs, not legacy/admission paths",
        ),
        audit_row(
            lane=LANE,
            check_id="missingness_only",
            status="pass",
            artifact="cross_axis_plan",
            path=dry_run_report_path(LANE).relative_to(PROJECT_ROOT),
            detail="assembly reports coverage, missingness, and validity only; no convergence/dissociation interpretation",
        ),
    ]
    audit_path = write_lane_audit(outputs["clean_rerun_audit"], audit_rows)
    if dry_run:
        report = write_dry_run_report(
            lane=LANE,
            config_path=config_path,
            expected_outputs=outputs,
            planned_rows=planned_rows,
            audit_path=audit_path,
            extra={
                "clean_inputs_missing_before_rerun": missing_inputs,
                "interpretation_allowed": False,
            },
        )
        print(f"dry_run_passed={report.relative_to(PROJECT_ROOT)}")
        return 0
    if missing_inputs:
        raise CleanRerunValidationError(
            "Cross-axis assembly requires clean upstream outputs first: "
            + pipe_join(missing_inputs)
        )
    assembled_rows = assemble_existing_inputs(input_paths)
    write_csv(outputs["model_axis_scores"], assembled_rows)
    print(outputs["model_axis_scores"].relative_to(PROJECT_ROOT))
    return 0


def clean_input_paths(config: dict[str, Any]) -> dict[str, Path]:
    inputs = dict(config.get("inputs", {}))
    paths: dict[str, Path] = {}
    for axis in AXES:
        if axis not in inputs:
            raise CleanRerunValidationError(f"Missing cross-axis input: {axis}")
        paths[axis] = require_publication_path(inputs[axis], f"inputs.{axis}")
    return paths


def build_planned_rows(input_paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    models_by_axis = {
        "behavioral": eligible_model_rows("behavioral"),
        "latent_neural_encoding": eligible_model_rows("latent_neural_encoding"),
        "geometry": eligible_model_rows("geometry"),
        "efficiency_resource": eligible_model_rows("efficiency_resource"),
    }
    all_models = sorted(
        {
            row["model_id"]
            for model_rows in models_by_axis.values()
            for row in model_rows
        }
    )
    identity_by_model = {
        row["model_id"]: model_identity(row)
        for model_rows in models_by_axis.values()
        for row in model_rows
    }
    for model_id in all_models:
        for axis in AXES:
            eligible = any(row["model_id"] == model_id for row in models_by_axis[axis])
            rows.append(
                {
                    **identity_by_model[model_id],
                    "axis": axis,
                    "clean_input_path": str(input_paths[axis].relative_to(PROJECT_ROOT)),
                    "clean_input_exists_now": "yes" if input_paths[axis].is_file() else "no",
                    "eligible_for_axis": "yes" if eligible else "no",
                    "coverage_status": "pending_clean_input" if eligible else "not_axis_eligible",
                    "interpretation": "not_permitted",
                }
            )
    return rows


def assemble_existing_inputs(input_paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for axis, path in input_paths.items():
        source_rows = read_csv(path)
        if not source_rows:
            rows.append(
                {
                    "axis": axis,
                    "clean_input_path": str(path.relative_to(PROJECT_ROOT)),
                    "coverage_status": "empty_clean_input",
                    "interpretation": "not_permitted",
                }
            )
            continue
        models = sorted({row.get("model_id", "") for row in source_rows if row.get("model_id")})
        streams = sorted({row.get("stream", "") for row in source_rows if row.get("stream")})
        for model_id in models:
            rows.append(
                {
                    "model_id": model_id,
                    "axis": axis,
                    "clean_input_path": str(path.relative_to(PROJECT_ROOT)),
                    "coverage_status": "clean_input_present",
                    "row_count_for_axis": sum(1 for row in source_rows if row.get("model_id") == model_id),
                    "streams_observed": pipe_join(streams),
                    "interpretation": "not_permitted",
                }
            )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
