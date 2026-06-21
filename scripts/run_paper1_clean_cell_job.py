"""Run one generated clean Paper 1 cluster cell job."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
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
EXTERNAL_ENV_ROOT = resolve_path("external/environments")
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
        elif args.kind == "behavior_latent_export":
            run_behavior_latent_export(row)
        elif args.kind == "behavior_latent_analysis":
            run_behavior_latent_analysis(row, args.table, args.index)
        elif args.kind == "behavior_latent_cleanup":
            cleanup_behavior_latent_intermediates(row)
        elif args.kind == "neural_export":
            run_neural_export(row)
        elif args.kind == "neural_analysis":
            run_neural_analysis(row)
        elif args.kind == "neural_cleanup":
            cleanup_neural_intermediates(row)
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
            external_model_command(
                row["runtime_model_id"],
                [
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
                + optional_max_items(row),
            )
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


def run_behavior_latent_export(row: dict[str, str]) -> None:
    run(
        external_model_command(
            row["runtime_model_id"],
            [
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
                row.get("artifact_key", "map_key") or "map_key",
                "--skip-efficiency-profile",
                "--output",
                row["artifact_dir"],
            ]
            + optional_max_items(row),
        )
    )


def run_behavior_latent_analysis(row: dict[str, str], table: str | Path, index: int) -> None:
    command = [
        sys.executable,
        "scripts/run_paper1_primary_behavioral_latent_fixation.py",
        "--config",
        "configs/paper1_primary_behavioral_latent_fixation.yaml",
        "--cell-table",
        str(table),
        "--cell-index",
        str(index),
    ]
    if row.get("output_dir"):
        command.extend(["--cell-output-dir", row["output_dir"]])
    run(command)


def cleanup_behavior_latent_intermediates(row: dict[str, str]) -> None:
    output_dir = require_publication_path(row["output_dir"])
    artifact_dir = require_publication_path(row["artifact_dir"])
    validate_behavior_latent_outputs(output_dir)
    removed = remove_behavior_latent_intermediate(
        artifact_dir,
        label="artifact_dir",
        output_dir=output_dir,
    )
    cleanup_path = output_dir / "intermediate_cleanup.json"
    cleanup_path.write_text(
        json.dumps(
            {
                "schema_version": "hma.paper1.behavior_latent_intermediate_cleanup.v1",
                "dataset": row.get("dataset", ""),
                "model_id": row.get("model_id", ""),
                "removed": [removed],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"behavior_latent_intermediate_cleanup={cleanup_path.relative_to(PROJECT_ROOT)}")


def validate_behavior_latent_outputs(output_dir: Path) -> None:
    required = [
        "metadata.json",
        "fixation_encoding_scores.csv",
        "fixation_image_scores.csv",
        "feature_reduction_metadata.json",
        "readout_selection_artifact.json",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise CleanCellJobError(
            f"Refusing behavioral latent cleanup because final cell outputs are missing: {missing}"
        )
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "hma.paper1.primary_behavioral_latent_cell.v1":
        raise CleanCellJobError("Refusing behavioral latent cleanup without valid cell metadata")


def remove_behavior_latent_intermediate(
    value: str | Path,
    *,
    label: str,
    output_dir: Path,
) -> dict[str, Any]:
    path = require_publication_path(value)
    assert_behavior_latent_intermediate_path(path, label=label, output_dir=output_dir)
    existed = path.exists()
    bytes_removed = directory_size(path) if existed else 0
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()
    return {
        "label": label,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "existed": existed,
        "bytes_removed": bytes_removed,
        "removed": existed and not path.exists(),
    }


def assert_behavior_latent_intermediate_path(
    path: Path,
    *,
    label: str,
    output_dir: Path,
) -> None:
    normalized = path.as_posix()
    if path in {PUBLICATION_ROOT, output_dir}:
        raise CleanCellJobError(f"Refusing to remove protected path: {path}")
    if label == "artifact_dir":
        if "/behavioral_latent_fixation/external/" not in normalized:
            raise CleanCellJobError(f"Refusing non-behavioral-latent artifact cleanup path: {path}")
        return
    raise CleanCellJobError(f"Unknown behavioral latent cleanup path label: {label}")


def run_neural_export(row: dict[str, str]) -> None:
    run(
        external_model_command(
            row["runtime_model_id"],
            [
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
            + optional_max_items(row),
        )
    )


def run_neural_analysis(row: dict[str, str]) -> None:
    run([sys.executable, "scripts/run_neural_alignment.py", "--config", row["config_path"]])


def cleanup_neural_intermediates(row: dict[str, str]) -> None:
    output_dir = require_publication_path(row["output_dir"])
    config_path = require_publication_path(row["config_path"])
    artifact_dir = require_publication_path(row["artifact_dir"])
    config = load_yaml(config_path)

    validate_neural_outputs(output_dir)
    preserved = preserve_external_artifact_manifest(
        artifact_dir=artifact_dir,
        output_dir=output_dir,
    )

    external_artifact = dict(config.get("external_artifact", {}))
    neural_config = dict(config.get("neural", {}))
    candidates = [
        ("artifact_dir", artifact_dir),
        ("feature_cache_dir", external_artifact.get("feature_cache_dir")),
        ("pca_cache_dir", neural_config.get("pca_cache_dir")),
        ("output_feature_cache", output_dir / "feature_cache"),
    ]
    removed = []
    for label, value in candidates:
        if not value:
            continue
        removed.append(remove_neural_intermediate(value, label=label, output_dir=output_dir))
    cleanup_path = output_dir / "intermediate_cleanup.json"
    cleanup_path.write_text(
        json.dumps(
            {
                "schema_version": "hma.paper1.neural_intermediate_cleanup.v1",
                "model_id": row.get("model_id", ""),
                "subject_id": row.get("subject_id", ""),
                "roi": row.get("roi", ""),
                "removed": removed,
                "preserved": preserved,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"neural_intermediate_cleanup={cleanup_path.relative_to(PROJECT_ROOT)}")


def validate_neural_outputs(output_dir: Path) -> None:
    required = [
        "metadata.json",
        "feature_reduction_metadata.json",
        "encoding_scores.csv",
        "encoding_target_scores.csv",
        "geometry_scores.csv",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise CleanCellJobError(
            f"Refusing cleanup because final neural outputs are missing: {missing}"
        )
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_backend") != "external_artifact":
        raise CleanCellJobError("Refusing cleanup for non-external neural output")
    if metadata.get("external_artifact_schema") != "hma.external.artifact.v1":
        raise CleanCellJobError("Refusing cleanup without external artifact schema metadata")
    if not metadata.get("external_artifact_provenance"):
        raise CleanCellJobError("Refusing cleanup without external artifact provenance")


def preserve_external_artifact_manifest(
    *,
    artifact_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    preserved: dict[str, str] = {}
    manifest_path = artifact_dir / "manifest.json"
    preserved_manifest = output_dir / "external_artifact_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        shutil.copy2(manifest_path, preserved_manifest)
        preserved["external_artifact_manifest"] = str(
            preserved_manifest.relative_to(PROJECT_ROOT)
        )
        image_ids_file = str(payload.get("image_ids_file", "image_ids.json"))
        image_ids_path = artifact_dir / image_ids_file
        if image_ids_path.is_file():
            preserved_ids = output_dir / "external_artifact_image_ids.json"
            shutil.copy2(image_ids_path, preserved_ids)
            preserved["external_artifact_image_ids"] = str(
                preserved_ids.relative_to(PROJECT_ROOT)
            )
        efficiency_file = payload.get("efficiency_file")
        if efficiency_file:
            efficiency_path = artifact_dir / str(efficiency_file)
            if efficiency_path.is_file():
                preserved_efficiency = output_dir / "external_artifact_efficiency.json"
                shutil.copy2(efficiency_path, preserved_efficiency)
                preserved["external_artifact_efficiency"] = str(
                    preserved_efficiency.relative_to(PROJECT_ROOT)
                )
    elif preserved_manifest.is_file():
        preserved["external_artifact_manifest"] = str(
            preserved_manifest.relative_to(PROJECT_ROOT)
        )
    else:
        raise CleanCellJobError(
            f"Refusing cleanup because artifact manifest is missing: {manifest_path}"
        )
    return preserved


def remove_neural_intermediate(
    value: str | Path,
    *,
    label: str,
    output_dir: Path,
) -> dict[str, Any]:
    path = require_publication_path(value)
    assert_neural_intermediate_path(path, label=label, output_dir=output_dir)
    existed = path.exists()
    bytes_removed = directory_size(path) if existed else 0
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()
    if label in {"feature_cache_dir", "pca_cache_dir"}:
        prune_empty_cache_parents(path)
    return {
        "label": label,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "existed": existed,
        "bytes_removed": bytes_removed,
        "removed": existed and not path.exists(),
    }


def assert_neural_intermediate_path(
    path: Path,
    *,
    label: str,
    output_dir: Path,
) -> None:
    normalized = path.as_posix()
    if path in {PUBLICATION_ROOT, output_dir}:
        raise CleanCellJobError(f"Refusing to remove protected path: {path}")
    if label == "artifact_dir":
        if "/external/neural/" not in normalized:
            raise CleanCellJobError(f"Refusing non-neural artifact cleanup path: {path}")
        return
    if label in {"feature_cache_dir", "pca_cache_dir"}:
        if "/neural_encoding/cache/" not in normalized:
            raise CleanCellJobError(f"Refusing non-neural cache cleanup path: {path}")
        return
    if label == "output_feature_cache":
        if path != (output_dir / "feature_cache").resolve():
            raise CleanCellJobError(f"Refusing unexpected output cache path: {path}")
        return
    raise CleanCellJobError(f"Unknown neural cleanup path label: {label}")


def directory_size(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += int(item.stat().st_size)
    return total


def prune_empty_cache_parents(path: Path) -> None:
    cache_root = None
    for parent in path.parents:
        if parent.name == "cache" and parent.parent.name == "neural_encoding":
            cache_root = parent
            break
    if cache_root is None:
        return
    current = path.parent
    while current != cache_root and cache_root in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def run_efficiency(row: dict[str, str]) -> None:
    if not row.get("runtime_model_id"):
        write_builtin_efficiency(row)
        return
    run(
        external_model_command(
            row["runtime_model_id"],
            [
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
                "--allow-failure-profile",
            ],
        )
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
        [
            sys.executable,
            "scripts/run_paper1_primary_behavioral_latent_fixation.py",
            "--config",
            "configs/paper1_primary_behavioral_latent_fixation.yaml",
            "--from-cells",
        ],
        [sys.executable, "scripts/run_paper1_clean_latent_neural_encoding.py", "--config", "configs/paper1_latent_neural_matrix.yaml"],
        [sys.executable, "scripts/run_paper1_clean_geometry.py", "--config", "configs/paper1_latent_neural_matrix.yaml"],
        [sys.executable, "scripts/run_paper1_clean_efficiency_resource.py", "--config", "configs/paper1_efficiency_resource_rerun.yaml"],
        [sys.executable, "scripts/assemble_paper1_clean_cross_axis.py", "--config", "configs/paper1_cross_axis_assembly.yaml"],
    ]
    for command in commands:
        run(command)


def optional_max_items(row: dict[str, str]) -> list[str]:
    return ["--max-items", row["max_items"]] if row.get("max_items") else []


def external_model_command(runtime_model_id: str, args: list[str]) -> list[str]:
    validate_runtime_model_id(runtime_model_id)
    mode = os.environ.get("HMA_EXTERNAL_ENV_MODE", "prefix").strip().lower()
    if mode in {"active", "current", "sys_executable"}:
        print(
            "external_model_runtime=active_env "
            f"model={runtime_model_id} python={sys.executable}",
            flush=True,
        )
        return [sys.executable, *args]
    if mode not in {"prefix", "conda_prefix"}:
        raise CleanCellJobError(
            "HMA_EXTERNAL_ENV_MODE must be one of "
            "prefix, conda_prefix, active, current, sys_executable"
        )
    env_dir = external_environment_dir(runtime_model_id)
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    print(
        "external_model_runtime=conda_prefix "
        f"model={runtime_model_id} prefix={env_dir}",
        flush=True,
    )
    return [
        conda_exe,
        "run",
        "--no-capture-output",
        "-p",
        str(env_dir),
        "python",
        *args,
    ]


def validate_runtime_model_id(runtime_model_id: str) -> None:
    if not runtime_model_id:
        raise CleanCellJobError("External job row is missing runtime_model_id")
    if runtime_model_id != Path(runtime_model_id).name:
        raise CleanCellJobError(f"Invalid runtime_model_id for conda prefix: {runtime_model_id}")


def external_environment_root() -> Path:
    override = os.environ.get("HMA_EXTERNAL_ENV_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return EXTERNAL_ENV_ROOT.resolve()


def external_environment_dir(runtime_model_id: str) -> Path:
    validate_runtime_model_id(runtime_model_id)
    env_root = external_environment_root()
    env_dir = (env_root / runtime_model_id).resolve()
    try:
        env_dir.relative_to(env_root)
    except ValueError as exc:
        raise CleanCellJobError(f"External conda environment escaped root: {env_dir}") from exc
    if not (env_dir / "conda-meta" / "history").is_file():
        raise CleanCellJobError(
            "External conda environment is not ready for "
            f"{runtime_model_id}: {env_dir}"
        )
    return env_dir


def run(command: list[str]) -> None:
    print("command_cmd=" + " ".join(shlex.quote(part) for part in command), flush=True)
    environment = dict(os.environ)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=environment)


def failure_log(kind: str, index: int | None) -> Path:
    name = f"{kind}_{'none' if index is None else index}.log"
    path = PUBLICATION_ROOT / "logs" / "clean_cell_jobs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
