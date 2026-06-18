"""Run one generated clean Paper 1 cluster cell job."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.utils.config import load_yaml  # noqa: E402
from hma.utils.paths import resolve_path  # noqa: E402


PUBLICATION_ROOT = resolve_path("outputs/paper1_publication_v0")
FORBIDDEN_MARKERS = (
    "admission_panel",
    "_admission",
    "paper1_admission",
    "matrix_v1",
    "matrix_v2",
    "real_matrix_v2",
)


class CleanCellJobError(RuntimeError):
    """Raised when a clean cluster cell job is invalid or fails."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table")
    parser.add_argument("--index", type=int)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    try:
        if args.kind == "geometry_collect":
            require_authorized_if_full(args.mode)
            collect_geometry_cells(args.mode)
            return 0
        if args.kind == "materialize_axes":
            require_authorized_if_full(args.mode)
            materialize_axes()
            return 0
        if args.table is None or args.index is None:
            raise CleanCellJobError("--table and --index are required for indexed jobs")
        row = read_table_row(args.table, args.index)
        validate_row_paths(row)
        require_authorized_if_full(row.get("mode", args.mode))
        if args.kind == "behavior":
            run_behavior(row)
        elif args.kind == "neural_export":
            run_neural_export(row)
        elif args.kind == "neural_analysis":
            run_neural_analysis(row)
        elif args.kind == "efficiency":
            run_efficiency(row)
        else:
            raise CleanCellJobError(f"Unknown clean cell job kind: {args.kind}")
        return 0
    except Exception as exc:
        path = failure_log(args.kind, args.index)
        path.write_text(
            "\n".join(
                [
                    f"kind={args.kind}",
                    f"index={args.index}",
                    f"error_type={type(exc).__name__}",
                    f"error_message={exc}",
                    "traceback:",
                    traceback.format_exc(),
                ]
            ),
            encoding="utf-8",
        )
        print(f"clean_cell_job_failed={path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
        return 1


def read_table_row(path: str | Path, index: int) -> dict[str, str]:
    table = resolve_path(path)
    reject_forbidden(table)
    with table.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if index < 0 or index >= len(rows):
        raise CleanCellJobError(f"Job index {index} outside table range 0..{len(rows)-1}")
    return rows[index]


def validate_row_paths(row: dict[str, str]) -> None:
    for key, value in row.items():
        if not value:
            continue
        if key.endswith(("_dir", "_path", "_root", "config")) or key in {
            "artifact_dir",
            "map_dir",
            "output_dir",
        }:
            reject_forbidden(value)
            if str(value).startswith("outputs/"):
                require_publication_path(value)


def reject_forbidden(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower()
    matched = [marker for marker in FORBIDDEN_MARKERS if marker in text]
    if matched:
        raise CleanCellJobError(f"Forbidden legacy/admission path marker {matched}: {path}")


def require_publication_path(path: str | Path) -> Path:
    resolved = resolve_path(path)
    try:
        resolved.relative_to(PUBLICATION_ROOT)
    except ValueError as exc:
        raise CleanCellJobError(f"Path is outside publication root: {resolved}") from exc
    return resolved


def require_authorized_if_full(mode: str) -> None:
    if mode != "full":
        return
    contract_path = resolve_path("configs/paper1_publication_contract.yaml")
    payload = load_yaml(contract_path)
    if payload.get("execution_authorized") is not True:
        raise CleanCellJobError("Full clean cell jobs require execution_authorized: true")


def run_behavior(row: dict[str, str]) -> None:
    runtime_config = row["runtime_config"]
    prepare = [
        sys.executable,
        "scripts/prepare_paper1_clean_behavior_cell.py",
        "--dataset",
        row["dataset"],
        "--model",
        row["model_id"],
        "--output-dir",
        row["output_dir"],
        "--runtime-config",
        runtime_config,
    ]
    if row.get("max_items"):
        prepare.extend(["--max-items", row["max_items"]])
    if row.get("job_kind") == "behavior_builtin":
        prepare.extend(["--builtin-method", row["model_id"]])
    else:
        run(
            [
                sys.executable,
                "scripts/run_external_model.py",
                "--model",
                row["runtime_model_id"],
                "--manifest",
                row["manifest"],
                "--image-root",
                row["image_root"],
                "--split",
                row["split"],
                "--batch-size",
                row.get("batch_size", "16") or "16",
                "--device",
                row.get("device", "cuda") or "cuda",
                "--artifact-scope",
                "full",
                "--artifact-key",
                "map_key",
                "--skip-efficiency-profile",
                "--output",
                row["artifact_dir"],
            ]
            + optional_max_items(row)
        )
        run(
            [
                sys.executable,
                "scripts/export_external_prediction_maps.py",
                "--artifact",
                row["artifact_dir"],
                "--output-dir",
                row["map_dir"],
            ]
        )
        prepare.extend(["--map-root", row["map_dir"]])
    run(prepare)
    run(
        [
            sys.executable,
            "scripts/run_saliency_benchmark.py",
            "--config",
            runtime_config,
            "--progress-interval",
            "250",
        ]
    )


def run_neural_export(row: dict[str, str]) -> None:
    run(
        [
            sys.executable,
            "scripts/run_external_model.py",
            "--model",
            row["runtime_model_id"],
            "--manifest",
            row["manifest"],
            "--image-root",
            row["image_root"],
            "--split",
            "train",
            "--subject-id",
            row["subject_id"],
            "--roi",
            row["roi"],
            "--batch-size",
            row.get("batch_size", "16") or "16",
            "--device",
            row.get("device", "cuda") or "cuda",
            "--artifact-scope",
            "full",
            "--artifact-key",
            "image_id",
            "--skip-efficiency-profile",
            "--output",
            row["artifact_dir"],
        ]
        + optional_max_items(row)
    )


def run_neural_analysis(row: dict[str, str]) -> None:
    run([sys.executable, "scripts/run_neural_alignment.py", "--config", row["config_path"]])


def run_efficiency(row: dict[str, str]) -> None:
    if not row.get("runtime_model_id"):
        write_builtin_efficiency(row)
        return
    run(
        [
            sys.executable,
            "scripts/profile_external_model_efficiency.py",
            "--model",
            row["runtime_model_id"],
            "--manifest",
            row["manifest"],
            "--image-root",
            row["image_root"],
            "--split",
            "train",
            "--subject-id",
            row["subject_id"],
            "--roi",
            row["roi"],
            "--device",
            row.get("device", "cuda") or "cuda",
            "--output",
            row["output_json"],
        ]
    )


def write_builtin_efficiency(row: dict[str, str]) -> None:
    path = require_publication_path(row["output_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "hma.paper1.clean_builtin_efficiency.v1",
        "model_id": row["model_id"],
        "publication_model_id": row["model_id"],
        "status": "builtin_control_no_external_runtime_profile",
        "latency_ms_per_image": "",
        "peak_memory_bytes": "",
        "parameter_count": "0",
        "theoretical_flops_or_macs": "",
        "profiler_realized_flops_or_macs": "",
        "total_cost_per_image": "",
        "total_cost_per_task": "",
        "sequential_step_count": "",
        "stopping_behavior": "",
        "resource_summary": {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_geometry_cells(mode: str) -> None:
    root = require_publication_path(
        "outputs/paper1_publication_v0/neural_encoding/cells"
        if mode == "full"
        else f"outputs/paper1_publication_v0/preflight/{mode}_cells/neural_encoding/cells"
    )
    target = require_publication_path(
        "outputs/paper1_publication_v0/geometry/cells"
        if mode == "full"
        else f"outputs/paper1_publication_v0/preflight/{mode}_cells/geometry/cells"
    )
    if not root.is_dir():
        raise CleanCellJobError(f"Missing neural cell root for geometry collection: {root}")
    copied = 0
    for source in sorted(root.rglob("geometry_scores.csv")):
        reject_forbidden(source)
        relative = source.relative_to(root)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    if copied == 0:
        raise CleanCellJobError("No neural geometry_scores.csv files found to collect")
    print(f"geometry_cells_collected={copied}")


def materialize_axes() -> None:
    commands = [
        [sys.executable, "scripts/run_paper1_clean_behavioral_rerun.py", "--config", "configs/paper1_clean_behavioral_rerun.yaml"],
        [sys.executable, "scripts/run_paper1_clean_latent_neural_encoding.py", "--config", "configs/paper1_latent_neural_matrix.yaml"],
        [sys.executable, "scripts/run_paper1_clean_geometry.py", "--config", "configs/paper1_latent_neural_matrix.yaml"],
        [sys.executable, "scripts/run_paper1_clean_efficiency_resource.py", "--config", "configs/paper1_efficiency_resource_rerun.yaml"],
        [sys.executable, "scripts/assemble_paper1_clean_cross_axis.py", "--config", "configs/paper1_cross_axis_assembly.yaml"],
    ]
    for command in commands:
        run(command)


def optional_max_items(row: dict[str, str]) -> list[str]:
    return ["--max-items", row["max_items"]] if row.get("max_items") else []


def run(command: list[str]) -> None:
    print("command_cmd=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def failure_log(kind: str, index: int | None) -> Path:
    name = f"{kind}_{'none' if index is None else index}.log"
    path = PUBLICATION_ROOT / "logs" / "clean_cell_jobs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
