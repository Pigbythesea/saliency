"""Validate or run the Paper 1 clean efficiency/resource lane."""

from __future__ import annotations

import argparse
import json
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
    reject_forbidden_path,
    run_with_failure_log,
    validate_contract,
    validate_execution_enabled,
    write_csv,
    write_dry_run_report,
    write_lane_audit,
)


LANE = "efficiency_resource"
OUTPUT_KEYS = ["efficiency_profiles", "resource_allocation_profiles", "audit", "failure_log"]
REQUIRED_MATCHED_COST_FIELDS = [
    "hardware",
    "precision",
    "batch_size",
    "image_resolution",
    "preprocessing",
]
REQUIRED_TOTAL_COST_FIELDS = [
    "total_cost_per_image",
    "total_cost_per_task",
    "sequential_step_count",
    "stopping_behavior",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper1_efficiency_resource_rerun.yaml")
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
    validate_cost_schema(config)
    model_rows = eligible_model_rows("efficiency_resource")
    planned_rows = build_planned_rows(config, model_rows)
    audit_rows = [
        audit_row(
            lane=LANE,
            check_id="publication_paths",
            status="pass",
            artifact="efficiency_outputs",
            path=pipe_join(str(path.relative_to(PROJECT_ROOT)) for path in outputs.values()),
            detail="configured efficiency/resource outputs are under outputs/paper1_publication_v0",
        ),
        audit_row(
            lane=LANE,
            check_id="matched_cost_schema",
            status="pass",
            artifact="efficiency_schema",
            path=config_path,
            detail="matched-cost and total-cost fields are explicitly required",
        ),
        audit_row(
            lane=LANE,
            check_id="planned_efficiency_rows",
            status="pass",
            artifact="efficiency_plan",
            path=dry_run_report_path(LANE).relative_to(PROJECT_ROOT),
            detail=f"planned_model_rows={len(planned_rows)}",
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
            extra={"models": [row["model_id"] for row in model_rows]},
        )
        print(f"dry_run_passed={report.relative_to(PROJECT_ROOT)}")
        return 0
    profile_files = clean_efficiency_profile_files(outputs["efficiency_profiles"].parent / "cells")
    efficiency_rows, resource_rows = materialize_efficiency_profiles(
        profile_files,
        model_rows=model_rows,
        config=config,
    )
    write_csv(outputs["efficiency_profiles"], efficiency_rows)
    write_csv(outputs["resource_allocation_profiles"], resource_rows)
    audit_rows.extend(
        [
            audit_row(
                lane=LANE,
                check_id="efficiency_profiles_written",
                status="pass",
                artifact="efficiency_profiles",
                path=outputs["efficiency_profiles"].relative_to(PROJECT_ROOT),
                detail=f"source_profile_files={len(profile_files)}; rows={len(efficiency_rows)}",
            ),
            audit_row(
                lane=LANE,
                check_id="resource_allocation_profiles_written",
                status="pass",
                artifact="resource_allocation_profiles",
                path=outputs["resource_allocation_profiles"].relative_to(PROJECT_ROOT),
                detail=f"rows={len(resource_rows)}",
            ),
        ]
    )
    write_lane_audit(outputs["audit"], audit_rows)
    print(outputs["efficiency_profiles"].relative_to(PROJECT_ROOT))
    print(outputs["resource_allocation_profiles"].relative_to(PROJECT_ROOT))
    return 0


def clean_efficiency_profile_files(root: Any) -> list[Any]:
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(root.rglob("efficiency.json")):
        reject_forbidden_path(path, "efficiency.cell_profile")
        files.append(path)
    if not files:
        raise CleanRerunValidationError(
            "No clean efficiency profile JSON files found under "
            f"{root.relative_to(PROJECT_ROOT)}. Generate clean efficiency jobs first."
        )
    return files


def materialize_efficiency_profiles(
    profile_files: list[Any],
    *,
    model_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identity = {row["model_id"]: model_identity(row) for row in model_rows}
    measurement = dict(config.get("measurement", {}))
    efficiency_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for path in profile_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_id = str(payload.get("publication_model_id") or payload.get("model_id") or path.parent.name)
        if model_id not in identity and path.parent.name in identity:
            model_id = path.parent.name
        base = {
            **identity.get(model_id, {"model_id": model_id}),
            "model_id": model_id,
            "matched_hardware": measurement.get("hardware", ""),
            "matched_precision": measurement.get("precision", ""),
            "matched_batch_size": measurement.get("batch_size", ""),
            "matched_image_resolution": measurement.get("image_resolution", ""),
            "matched_preprocessing": measurement.get("preprocessing", ""),
            "source_profile": str(path.relative_to(PROJECT_ROOT)),
            "evidence_status": "clean_publication_rerun",
        }
        resource_summary = dict(payload.get("resource_summary", {}))
        profile_status = payload.get("profile_status") or payload.get("status", "")
        efficiency_rows.append(
            {
                **base,
                "schema_version": payload.get("schema_version", ""),
                "profile_status": profile_status,
                "failure_type": payload.get("failure_type", ""),
                "failure_message": payload.get("failure_message", ""),
                "latency_ms_per_image": payload.get("latency_ms_per_image", ""),
                "peak_memory_bytes": payload.get("peak_memory_bytes", ""),
                "parameter_count": payload.get("parameter_count", ""),
                "theoretical_flops_or_macs": payload.get("theoretical_flops_or_macs", ""),
                "profiler_realized_flops_or_macs": payload.get("profiler_realized_flops_or_macs", ""),
                "total_cost_per_image": payload.get("total_cost_per_image", ""),
                "total_cost_per_task": payload.get("total_cost_per_task", ""),
                "sequential_step_count": payload.get("sequential_step_count", ""),
                "stopping_behavior": payload.get("stopping_behavior", ""),
                "resource_output_count": len(resource_summary),
            }
        )
        if resource_summary:
            for key, summary in sorted(resource_summary.items()):
                summary = dict(summary) if isinstance(summary, dict) else {"value": summary}
                resource_rows.append(
                    {
                        **base,
                        "resource_key": key,
                        **summary,
                    }
                )
        else:
            resource_rows.append(
                {
                    **base,
                    "resource_key": "none",
                    "status": "no_resource_allocation_outputs",
                }
            )
    if not efficiency_rows:
        raise CleanRerunValidationError("Clean efficiency profile files contained no rows")
    return efficiency_rows, resource_rows


def validate_cost_schema(config: dict[str, Any]) -> None:
    schema = dict(config.get("cost_schema", {}))
    matched = set(str(value) for value in schema.get("matched_cost_fields", []))
    total = set(str(value) for value in schema.get("total_cost_fields", []))
    missing_matched = sorted(set(REQUIRED_MATCHED_COST_FIELDS) - matched)
    missing_total = sorted(set(REQUIRED_TOTAL_COST_FIELDS) - total)
    if missing_matched or missing_total:
        raise CleanRerunValidationError(
            "Efficiency config missing cost fields: "
            f"matched={missing_matched}; total={missing_total}"
        )


def build_planned_rows(
    config: dict[str, Any],
    models: list[dict[str, str]],
) -> list[dict[str, Any]]:
    measurement = dict(config.get("measurement", {}))
    rows = []
    for model in models:
        resources = [
            value for value in str(model.get("resource_allocation_outputs", "")).split("|") if value
        ]
        rows.append(
            {
                **model_identity(model),
                "matched_hardware": str(measurement.get("hardware", "")),
                "matched_precision": str(measurement.get("precision", "")),
                "matched_batch_size": str(measurement.get("batch_size", "")),
                "matched_image_resolution": str(measurement.get("image_resolution", "")),
                "matched_preprocessing": str(measurement.get("preprocessing", "")),
                "resource_allocation_outputs": pipe_join(resources),
                "requires_total_cost": "yes" if resources else "no",
                "total_cost_per_image": "required",
                "total_cost_per_task": "required_for_task_or_scanpath_models",
                "evidence_type": "efficiency_resource_allocation",
                "score_status": "planned_clean_publication_rerun",
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
