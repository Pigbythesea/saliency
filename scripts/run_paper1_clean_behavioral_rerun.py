"""Validate or run the Paper 1 clean behavioral rerun lane."""

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
    run_with_failure_log,
    validate_contract,
    validate_execution_enabled,
    validate_input_path,
    write_dry_run_report,
    write_lane_audit,
)


LANE = "behavioral"
OUTPUT_KEYS = ["per_image_metrics_dir", "aggregate", "uncertainty", "audit", "failure_log"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper1_clean_behavioral_rerun.yaml")
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
    if outputs["per_image_metrics_dir"].suffix:
        raise CleanRerunValidationError("per_image_metrics_dir must be a directory path")

    dataset_rows = validate_datasets(config)
    model_rows = eligible_model_rows("behavioral")
    planned_rows = build_planned_rows(dataset_rows, model_rows)
    audit_rows = [
        audit_row(
            lane=LANE,
            check_id="publication_paths",
            status="pass",
            artifact="behavioral_outputs",
            path=pipe_join(str(path.relative_to(PROJECT_ROOT)) for path in outputs.values()),
            detail="all configured behavioral outputs are under outputs/paper1_publication_v0",
        ),
        audit_row(
            lane=LANE,
            check_id="planned_cells",
            status="pass",
            artifact="behavioral_plan",
            path=dry_run_report_path(LANE).relative_to(PROJECT_ROOT),
            detail=f"planned_model_dataset_cells={len(planned_rows)}",
        ),
        audit_row(
            lane=LANE,
            check_id="regime_separation",
            status="pass",
            artifact="behavioral_plan",
            path=config_path,
            detail="free_viewing, task_search, conditional, and scanpath labels are explicit in planned rows",
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
                "datasets": [row["dataset"] for row in dataset_rows],
                "models": [row["model_id"] for row in model_rows],
            },
        )
        print(f"dry_run_passed={report.relative_to(PROJECT_ROOT)}")
        return 0
    raise CleanRerunValidationError(
        "Full behavioral scoring is not launched in validation mode. "
        "Run this script on the cluster only after all model-specific clean scoring backends are available."
    )


def validate_datasets(config: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for dataset, section in dict(config.get("datasets", {})).items():
        manifest = validate_input_path(section["manifest"], f"datasets.{dataset}.manifest")
        observer = section.get("observer_manifest")
        if observer:
            validate_input_path(observer, f"datasets.{dataset}.observer_manifest")
        regime = str(section.get("regime", ""))
        if regime not in {"free_viewing", "task_search"}:
            raise CleanRerunValidationError(
                f"Behavioral dataset {dataset} has unsupported regime {regime}"
            )
        rows.append(
            {
                "dataset": str(dataset),
                "behavioral_regime": regime,
                "manifest": str(manifest.relative_to(PROJECT_ROOT)),
                "split": str(section.get("split", "")),
                "expected_rows": str(section.get("expected_rows", "")),
            }
        )
    if not rows:
        raise CleanRerunValidationError("Behavioral config has no datasets")
    return rows


def build_planned_rows(
    datasets: list[dict[str, str]],
    models: list[dict[str, str]],
) -> list[dict[str, Any]]:
    planned = []
    for dataset in datasets:
        for model in models:
            identity = model_identity(model)
            output_types = str(model.get("behavioral_output_type", ""))
            labels = behavior_labels(output_types, dataset["behavioral_regime"])
            planned.append(
                {
                    **identity,
                    "dataset": dataset["dataset"],
                    "behavioral_regime": labels["behavioral_regime"],
                    "behavioral_object": labels["behavioral_object"],
                    "sequence_label": labels["sequence_label"],
                    "split": dataset["split"],
                    "expected_manifest_rows": dataset["expected_rows"],
                    "evidence_status": "planned_clean_publication_rerun",
                }
            )
    return planned


def behavior_labels(output_types: str, dataset_regime: str) -> dict[str, str]:
    outputs = {value for value in output_types.split("|") if value}
    if "generated_scanpath" in outputs or "text_conditioned_generated_scanpath" in outputs:
        return {
            "behavioral_regime": "scanpath",
            "behavioral_object": pipe_join(sorted(outputs)),
            "sequence_label": "scanpath",
        }
    if "conditional_next_fixation" in outputs:
        return {
            "behavioral_regime": "conditional",
            "behavioral_object": "conditional_next_fixation",
            "sequence_label": "conditional",
        }
    return {
        "behavioral_regime": dataset_regime,
        "behavioral_object": pipe_join(sorted(outputs)) if outputs else "behavioral_output",
        "sequence_label": "map",
    }


if __name__ == "__main__":
    raise SystemExit(main())
