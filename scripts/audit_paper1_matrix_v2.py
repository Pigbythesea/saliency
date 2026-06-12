"""Generate Matrix V2 feasibility, status, scope, and next-run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.external.registry import load_external_registry
from hma.utils.config import load_yaml
from hma.utils.paths import resolve_path

try:
    from scripts.setup_external_model import build_installation_report
except ModuleNotFoundError:
    from setup_external_model import build_installation_report


def audit_matrix_v2(
    *,
    matrix_path: str | Path = "configs/paper1_matrix_v2.yaml",
    output_dir: str | Path = "outputs/paper1_matrix_v2",
) -> dict[str, Path]:
    matrix = load_yaml(resolve_path(matrix_path))
    registry = load_external_registry(matrix["external_registry"])
    root = resolve_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reports = {
        model_id: build_installation_report(
            model_id, registry_path=matrix["external_registry"]
        )
        for model_id in registry.models
    }
    feasibility = _feasibility_payload(matrix, registry, reports)
    feasibility_path = root / "feasibility.json"
    feasibility_path.write_text(
        json.dumps(feasibility, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    environment_path = root / "environment_status.csv"
    _write_csv(environment_path, _environment_rows(registry, reports))
    axis_scope_path = root / "axis_scope.csv"
    _write_csv(axis_scope_path, _axis_scope_rows(registry))
    cells_path = root / "adaptive_matrix_cells.csv"
    _write_csv(cells_path, _adaptive_matrix_rows(matrix))
    next_run_path = root / "next_run.md"
    next_run_path.write_text(_next_run_markdown(matrix), encoding="utf-8")
    return {
        "feasibility": feasibility_path,
        "environment_status": environment_path,
        "axis_scope": axis_scope_path,
        "adaptive_matrix_cells": cells_path,
        "next_run": next_run_path,
    }


def _feasibility_payload(
    matrix: dict[str, Any],
    registry: Any,
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "hma.paper1_matrix_v2.feasibility.v1",
        "matrix_version": matrix["version"],
        "v1_role": matrix["compatibility"]["v1_role"],
        "restore_v1_interpretation": matrix["compatibility"][
            "restore_v1_interpretation"
        ],
        "first_evidence_models": matrix["first_evidence_run"]["models"],
        "first_evidence_rois": [
            cell["roi"] for cell in matrix["first_evidence_run"]["roi_cells"]
        ],
        "models": {
            model_id: {
                "track": registry.model(model_id)["track"],
                "role": registry.model(model_id)["role"],
                "stages": report["stages"],
                "diagnostics": report["diagnostics"],
            }
            for model_id, report in reports.items()
        },
        "policy": {
            "dependency_mismatch": "integration_required",
            "defer_on_dependency_mismatch": False,
            "full_run_requires_scientific64": True,
            "scientific64_status": matrix["first_evidence_run"][
                "execution_status"
            ]["scientific64"],
            "full_matrix_status": matrix["first_evidence_run"][
                "execution_status"
            ]["full_matrix"],
        },
    }


def _environment_rows(
    registry: Any,
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for model_id in registry.models:
        model = registry.model(model_id)
        report = reports[model_id]
        rows.append(
            {
                "model_id": model_id,
                "track": model["track"],
                "role": model["role"],
                "execution": model.get("execution", "linux_gpu"),
                "source_commit": model["source"]["commit"],
                "license": model["source"]["license"],
                **report["stages"],
                "diagnostics": "; ".join(report["diagnostics"]),
            }
        )
    return rows


def _axis_scope_rows(registry: Any) -> list[dict[str, Any]]:
    rows = []
    for model_id in registry.models:
        model = registry.model(model_id)
        behavioral = model["track"] in {
            "adaptive_token_computation",
            "sequential_foveated_behavior",
        }
        neural = bool(model.get("feature_layers"))
        sequential = model["track"] == "sequential_foveated_behavior"
        rows.append(
            {
                "model_id": model_id,
                "track": model["track"],
                "behavioral_alignment": behavioral,
                "neural_encoding": neural,
                "cka_rsa": neural,
                "efficiency": not sequential,
                "scanpath": sequential,
                "operational_outputs": ";".join(model.get("mechanism_outputs", [])),
                "neural_acceptance_gate": (
                    "standard_feature_audit"
                    if neural and not sequential
                    else "behavioral_only_until_feature_audit"
                ),
            }
        )
    return rows


def _adaptive_matrix_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = matrix["first_evidence_run"]
    status = evidence["execution_status"]
    return [
        {
            "model_id": model_id,
            "subject_id": evidence["subject_id"],
            "roi": roi_cell["roi"],
            "roi_class": roi_cell["roi_class"],
            "manifest_path": roi_cell["manifest_path"],
            "adapter16_required": True,
            "scientific64_required": True,
            "adapter16_status": status["adapter16"],
            "scientific64_status": status["scientific64"],
            "full_status": status["full_matrix"],
        }
        for model_id in evidence["models"]
        for roi_cell in evidence["roi_cells"]
    ]


def _next_run_markdown(matrix: dict[str, Any]) -> str:
    models = ", ".join(matrix["first_evidence_run"]["models"])
    return f"""# Matrix V2 Next Run

The adapter16 and scientific64 gates are complete for {models}.

The next task is cluster-readiness implementation before the full `9841`-image
adaptive-computation matrix is submitted:

1. add resource-only external exports for SALICON, CAT2000, and COCO-Search18;
2. cache train-only PCA reductions across the four ROI analyses for each model;
3. add cluster preflight validation and Slurm jobs for three neural exports,
   nine behavioral exports/map conversions/scores, and twelve neural analyses;
4. prove cached and uncached scientific64 outputs are numerically equivalent;
5. pass a bounded cluster smoke before full submission.

Do not launch the full matrix directly from the laptop. The projected external
artifacts are approximately `27.5 GiB`, with approximately `29.5 GiB` of raw
feature memmaps before PCA workspace and analysis outputs.
"""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/paper1_matrix_v2.yaml")
    parser.add_argument("--output-dir", default="outputs/paper1_matrix_v2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = audit_matrix_v2(matrix_path=args.matrix, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
