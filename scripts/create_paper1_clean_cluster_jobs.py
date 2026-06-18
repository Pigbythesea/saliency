"""Generate Slurm jobs for Paper 1 clean publication-root cell production."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.utils.config import load_yaml, save_yaml  # noqa: E402
from hma.utils.paths import resolve_path  # noqa: E402


REMOTE_PROJECT = "/scratch/tshu2/zzhan330/saliency"
CONDA_SH = "/scratch/tshu2/zzhan330/miniconda3/etc/profile.d/conda.sh"
PUBLICATION_ROOT = resolve_path("outputs/paper1_publication_v0")
PREFLIGHT_ROOT = PUBLICATION_ROOT / "preflight"
NSD_ROOT = "data/raw/nsd_algonauts"
DATA_ROOTS = {
    "salicon": "data/raw/SALICON",
    "cat2000": "data/raw/CAT2000",
    "coco_search18": "data/raw/COCO-Search18",
}
BUILTIN_BEHAVIOR_MODELS = {
    "center_prior",
    "random_baseline",
    "empirical_spatial_prior",
    "coco_search18_task_prior",
}


def main() -> int:
    args = parse_args()
    generated = create_cluster_jobs(
        mode=args.mode,
        max_items=args.max_items,
        limit_behavior_cells=args.limit_behavior_cells,
        limit_neural_cells=args.limit_neural_cells,
        limit_efficiency_cells=args.limit_efficiency_cells,
    )
    print(f"generated_clean_cluster_jobs={generated['job_root']}")
    print(f"behavior_cells={generated['behavior_cells']}")
    print(f"neural_cells={generated['neural_cells']}")
    print(f"efficiency_cells={generated['efficiency_cells']}")
    print(f"unsupported_behavior_rows={generated['unsupported_behavior_rows']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit image rows per cell. Defaults to 8 in smoke mode and unlimited in full mode.",
    )
    parser.add_argument("--limit-behavior-cells", type=int)
    parser.add_argument("--limit-neural-cells", type=int)
    parser.add_argument("--limit-efficiency-cells", type=int)
    return parser.parse_args()


def create_cluster_jobs(
    *,
    mode: str,
    max_items: int | None,
    limit_behavior_cells: int | None,
    limit_neural_cells: int | None,
    limit_efficiency_cells: int | None,
) -> dict[str, Any]:
    if max_items is None and mode == "smoke":
        max_items = 8
    if mode == "smoke":
        limit_behavior_cells = 2 if limit_behavior_cells is None else limit_behavior_cells
        limit_neural_cells = 1 if limit_neural_cells is None else limit_neural_cells
        limit_efficiency_cells = 1 if limit_efficiency_cells is None else limit_efficiency_cells
    job_root = PREFLIGHT_ROOT / "cluster_jobs" / mode
    runtime_root = PREFLIGHT_ROOT / "runtime_configs" / mode
    cell_root = (
        PUBLICATION_ROOT
        if mode == "full"
        else PREFLIGHT_ROOT / f"{mode}_cells"
    )
    job_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    model_rows = read_csv(PREFLIGHT_ROOT / "model_certification_summary.csv")
    cert_by_model = {row["model_id"]: row for row in model_rows if row.get("model_id")}
    behavior_rows, unsupported = build_behavior_rows(
        mode=mode,
        max_items=max_items,
        cell_root=cell_root,
        runtime_root=runtime_root,
        cert_by_model=cert_by_model,
    )
    neural_rows = build_neural_rows(
        mode=mode,
        max_items=max_items,
        cell_root=cell_root,
        runtime_root=runtime_root,
        cert_by_model=cert_by_model,
    )
    efficiency_rows = build_efficiency_rows(
        mode=mode,
        cell_root=cell_root,
        cert_by_model=cert_by_model,
    )
    behavior_table = write_csv(job_root / "clean_behavior_cells.csv", behavior_rows)
    neural_table = write_csv(job_root / "clean_neural_cells.csv", neural_rows)
    efficiency_table = write_csv(job_root / "clean_efficiency_cells.csv", efficiency_rows)
    write_csv(job_root / "clean_behavior_unsupported_rows.csv", unsupported)
    behavior_rows = limit_rows(behavior_rows, limit_behavior_cells)
    neural_rows = limit_rows(neural_rows, limit_neural_cells)
    efficiency_rows = limit_rows(efficiency_rows, limit_efficiency_cells)
    behavior_table = write_csv(job_root / "clean_behavior_cells.csv", behavior_rows)
    neural_table = write_csv(job_root / "clean_neural_cells.csv", neural_rows)
    efficiency_table = write_csv(job_root / "clean_efficiency_cells.csv", efficiency_rows)
    write_job_files(
        job_root=job_root,
        mode=mode,
        behavior_count=len(behavior_rows),
        neural_count=len(neural_rows),
        efficiency_count=len(efficiency_rows),
        behavior_table=behavior_table,
        neural_table=neural_table,
        efficiency_table=efficiency_table,
    )
    write_manifest(
        job_root=job_root,
        mode=mode,
        max_items=max_items,
        limits={
            "behavior": limit_behavior_cells,
            "neural": limit_neural_cells,
            "efficiency": limit_efficiency_cells,
        },
        cell_root=cell_root,
        behavior_rows=behavior_rows,
        neural_rows=neural_rows,
        efficiency_rows=efficiency_rows,
        unsupported=unsupported,
    )
    return {
        "job_root": str(job_root.relative_to(PROJECT_ROOT)),
        "behavior_cells": len(behavior_rows),
        "neural_cells": len(neural_rows),
        "efficiency_cells": len(efficiency_rows),
        "unsupported_behavior_rows": len(unsupported),
    }


def build_behavior_rows(
    *,
    mode: str,
    max_items: int | None,
    cell_root: Path,
    runtime_root: Path,
    cert_by_model: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    unsupported = []
    for planned in read_csv(PREFLIGHT_ROOT / "dry_runs" / "behavioral_planned_rows.csv"):
        if planned.get("sequence_label") != "map":
            unsupported.append(
                {
                    **planned,
                    "unsupported_reason": "clean_static_map_cell_generator_handles_map_rows_only",
                }
            )
            continue
        dataset = planned["dataset"]
        model_id = planned["model_id"]
        runtime_model_id = planned.get("runtime_model_id") or cert_by_model.get(model_id, {}).get("runtime_model_id", "")
        output_dir = cell_root / "behavioral" / "per_image_metrics" / dataset / safe_name(model_id)
        runtime_config = runtime_root / "behavioral" / f"{dataset}_{safe_name(model_id)}.yaml"
        base = {
            "mode": mode,
            "dataset": dataset,
            "model_id": model_id,
            "runtime_model_id": runtime_model_id,
            "manifest": f"data/manifests/{dataset}_manifest.csv",
            "image_root": DATA_ROOTS[dataset],
            "split": planned["split"],
            "max_items": "" if max_items is None else str(max_items),
            "batch_size": "16",
            "device": "cuda",
            "runtime_config": relative(runtime_config),
            "output_dir": relative(output_dir),
            "behavioral_regime": planned.get("behavioral_regime", ""),
            "behavioral_object": planned.get("behavioral_object", ""),
            "sequence_label": planned.get("sequence_label", ""),
        }
        if model_id in BUILTIN_BEHAVIOR_MODELS:
            rows.append({**base, "job_kind": "behavior_builtin", "artifact_dir": "", "map_dir": ""})
            continue
        if not runtime_model_id:
            unsupported.append({**planned, "unsupported_reason": "missing_runtime_model_id"})
            continue
        artifact_dir = cell_root / "external" / "behavioral" / dataset / safe_name(model_id)
        map_dir = cell_root / "behavioral" / "maps" / dataset / safe_name(model_id)
        rows.append(
            {
                **base,
                "job_kind": "behavior_external",
                "artifact_dir": relative(artifact_dir),
                "map_dir": relative(map_dir),
            }
        )
    return rows, unsupported


def build_neural_rows(
    *,
    mode: str,
    max_items: int | None,
    cell_root: Path,
    runtime_root: Path,
    cert_by_model: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for planned in read_csv(PREFLIGHT_ROOT / "dry_runs" / "latent_neural_encoding_planned_rows.csv"):
        model_id = planned["model_id"]
        key = (model_id, planned["subject_id"], planned["roi"], planned["manifest_path"])
        row = grouped.setdefault(
            key,
            {
                "mode": mode,
                "model_id": model_id,
                "runtime_model_id": planned.get("runtime_model_id") or cert_by_model.get(model_id, {}).get("runtime_model_id", model_id),
                "subject_id": planned["subject_id"],
                "roi": planned["roi"],
                "stream": planned.get("stream", ""),
                "roi_class": planned.get("roi_class", ""),
                "manifest": planned["manifest_path"],
                "image_root": NSD_ROOT,
                "max_items": "" if max_items is None else str(max_items),
                "batch_size": "16",
                "device": "cuda",
                "layers": [],
            },
        )
        row["layers"].append(planned["layer"])
    rows = []
    for row in grouped.values():
        model_id = row["model_id"]
        subject_id = row["subject_id"]
        roi = row["roi"]
        cell_name = f"{safe_name(model_id)}__{safe_name(subject_id)}__{safe_name(roi)}"
        artifact_dir = cell_root / "external" / "neural" / cell_name
        output_dir = cell_root / "neural_encoding" / "cells" / safe_name(model_id) / f"{safe_name(subject_id)}_{safe_name(roi)}"
        cache_dir = cell_root / "neural_encoding" / "cache" / safe_name(model_id) / f"{safe_name(subject_id)}_{safe_name(roi)}"
        config_path = runtime_root / "neural" / f"{cell_name}.yaml"
        layers = unique(row["layers"])
        write_neural_config(
            path=config_path,
            row=row,
            layers=layers,
            artifact_dir=artifact_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            cert=cert_by_model.get(model_id, {}),
            max_items=max_items,
        )
        rows.append(
            {
                **{key: value for key, value in row.items() if key != "layers"},
                "layers": "|".join(layers),
                "artifact_dir": relative(artifact_dir),
                "config_path": relative(config_path),
                "output_dir": relative(output_dir),
            }
        )
    return rows


def build_efficiency_rows(
    *,
    mode: str,
    cell_root: Path,
    cert_by_model: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for planned in read_csv(PREFLIGHT_ROOT / "dry_runs" / "efficiency_resource_planned_rows.csv"):
        model_id = planned["model_id"]
        output_json = cell_root / "efficiency" / "cells" / safe_name(model_id) / "efficiency.json"
        rows.append(
            {
                "mode": mode,
                "model_id": model_id,
                "runtime_model_id": planned.get("runtime_model_id") or cert_by_model.get(model_id, {}).get("runtime_model_id", ""),
                "manifest": "data/manifests/nsd_algonauts_subj01-04_prf_visualrois_full_manifest.csv",
                "image_root": NSD_ROOT,
                "subject_id": "subj01",
                "roi": "V1",
                "device": "cuda",
                "output_json": relative(output_json),
            }
        )
    return rows


def write_neural_config(
    *,
    path: Path,
    row: dict[str, Any],
    layers: list[str],
    artifact_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    cert: dict[str, str],
    max_items: int | None,
) -> None:
    plan = load_yaml(resolve_path("configs/paper1_latent_neural_matrix.yaml"))
    encoding = dict(plan.get("encoding", {}))
    geometry = dict(plan.get("geometry", {}))
    smoke = max_items is not None
    preprocessing = parse_json_field(cert.get("preprocessing_contract", "{}"))
    pca_components = min(int(encoding.get("pca_components", 512)), max(2, (max_items or 512) - 1))
    subset_sizes = [8] if smoke else [int(value) for value in geometry.get("subset_sizes", [512, 1024, 2048])]
    config = {
        "seed": int(plan.get("seed", 123)),
        "device": "cpu",
        "experiment": {
            "name": f"paper1_clean_{row['model_id']}_{row['subject_id']}_{row['roi']}",
            "matrix_version": "paper1_publication_v0",
            "stage": row["mode"],
        },
        "dataset": {
            "name": "nsd_algonauts",
            "label": f"paper1_clean_{row['model_id']}_{row['subject_id']}_{row['roi']}",
            "root": NSD_ROOT,
            "manifest_path": row["manifest"],
            "split": "train",
            "subject_id": row["subject_id"],
            "roi": row["roi"],
            "max_items": max_items,
            "validate_files": True,
        },
        "model": {
            "name": row["model_id"],
            "backend": "external_artifact",
            "pretrained": True,
            "eval_mode": True,
        },
        "external_artifact": {
            "path": relative(artifact_dir),
            "verify_hashes": True,
            "feature_cache_dir": relative(cache_dir / "raw_features"),
        },
        "preprocessing": preprocessing,
        "neural": {
            "layers": layers,
            "response_key": "roi_responses",
            "noise_ceiling_key": "noise_ceiling",
            "feature_reduction": "flatten_pca",
            "pca_components": pca_components,
            "pca_solver": encoding.get("pca_solver", "randomized"),
            "pca_whiten": False,
            "feature_reduction_seed": int(plan.get("seed", 123)),
            "pca_cache_dir": relative(cache_dir / "pca"),
            "train_fraction": float(encoding.get("train_fraction", 0.8)),
            "ridge_alpha": 1.0,
            "ridge_alphas": encoding.get("ridge_alphas", [1.0]),
            "validation_fraction": float(encoding.get("validation_fraction_of_train", 0.2)),
            "metric": "correlation",
            "selection": {
                "enabled": True,
                "validation_fraction": float(encoding.get("validation_fraction_of_train", 0.2)),
                "primary_score": "mean_noise_normalized_score",
            },
            "rsa": {"enabled": False},
            "geometry": {
                "enabled": True,
                "methods": ["debiased_linear_cka", "subset_rsa"],
                "subset_sizes": subset_sizes,
                "subset_seeds": [123] if smoke else geometry.get("subset_seeds", [123, 456, 789]),
                "null_control_seeds": [123] if smoke else geometry.get("subset_seeds", [123, 456, 789]),
                "bootstrap_resamples": 0 if smoke else int(dict(geometry.get("image_resampling", {})).get("resamples", 0)),
                "bootstrap_seed": int(dict(geometry.get("image_resampling", {})).get("seed", 123)),
                "bootstrap_confidence": float(dict(geometry.get("image_resampling", {})).get("confidence", 0.95)),
            },
        },
        "output": {"dir": relative(output_dir)},
    }
    save_yaml(config, path)


def write_job_files(
    *,
    job_root: Path,
    mode: str,
    behavior_count: int,
    neural_count: int,
    efficiency_count: int,
    behavior_table: Path,
    neural_table: Path,
    efficiency_table: Path,
) -> None:
    files = {
        "clean_behavior_cells.sbatch": array_job(
            name=f"p1c_{mode}_beh",
            partition="l40s",
            time="02:00:00" if mode == "smoke" else "12:00:00",
            cpus=12,
            memory="48G",
            gpu=True,
            count=behavior_count,
            command=f'python scripts/run_paper1_clean_cell_job.py --kind behavior --table "{relative(behavior_table)}" --index "$SLURM_ARRAY_TASK_ID" --mode {mode}',
        ),
        "clean_neural_exports.sbatch": array_job(
            name=f"p1c_{mode}_nexp",
            partition="l40s",
            time="02:00:00" if mode == "smoke" else "12:00:00",
            cpus=12,
            memory="64G",
            gpu=True,
            count=neural_count,
            command=f'python scripts/run_paper1_clean_cell_job.py --kind neural_export --table "{relative(neural_table)}" --index "$SLURM_ARRAY_TASK_ID" --mode {mode}',
        ),
        "clean_neural_analysis.sbatch": array_job(
            name=f"p1c_{mode}_nana",
            partition="cpu",
            time="02:00:00" if mode == "smoke" else "2-00:00:00",
            cpus=32,
            memory="120G",
            gpu=False,
            count=neural_count,
            command=f'python scripts/run_paper1_clean_cell_job.py --kind neural_analysis --table "{relative(neural_table)}" --index "$SLURM_ARRAY_TASK_ID" --mode {mode}',
        ),
        "clean_geometry_collect.sbatch": scalar_job(
            name=f"p1c_{mode}_geom",
            partition="cpu",
            time="00:30:00",
            cpus=4,
            memory="16G",
            command=f"python scripts/run_paper1_clean_cell_job.py --kind geometry_collect --mode {mode}",
        ),
        "clean_efficiency_cells.sbatch": array_job(
            name=f"p1c_{mode}_eff",
            partition="l40s",
            time="01:00:00" if mode == "smoke" else "08:00:00",
            cpus=12,
            memory="48G",
            gpu=True,
            count=efficiency_count,
            command=f'python scripts/run_paper1_clean_cell_job.py --kind efficiency --table "{relative(efficiency_table)}" --index "$SLURM_ARRAY_TASK_ID" --mode {mode}',
        ),
    }
    if mode == "full":
        files["clean_materialize_axes.sbatch"] = scalar_job(
            name="p1c_full_axis",
            partition="cpu",
            time="04:00:00",
            cpus=16,
            memory="64G",
            command="python scripts/run_paper1_clean_cell_job.py --kind materialize_axes --mode full",
        )
    for name, content in files.items():
        (job_root / name).write_text(content, encoding="utf-8", newline="\n")
    submit = submit_script(mode=mode, include_materialize=(mode == "full"))
    submit_path = job_root / f"submit_clean_{mode}.sh"
    submit_path.write_text(submit, encoding="utf-8", newline="\n")


def header(
    *,
    name: str,
    partition: str,
    time: str,
    cpus: int,
    memory: str,
    gpu: bool,
    array: str | None = None,
) -> str:
    gres = "#SBATCH --gres=gpu:l40s:1\n" if gpu else ""
    array_line = f"#SBATCH --array={array}\n" if array else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
{gres}{array_line}#SBATCH --output={REMOTE_PROJECT}/slurm_logs/%x_%A_%a.out
#SBATCH --error={REMOTE_PROJECT}/slurm_logs/%x_%A_%a.err

set -euo pipefail
source {CONDA_SH}
conda activate hma
cd {REMOTE_PROJECT}
mkdir -p slurm_logs outputs/paper1_publication_v0/logs/clean_cell_jobs
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"""


def array_job(
    *,
    name: str,
    partition: str,
    time: str,
    cpus: int,
    memory: str,
    gpu: bool,
    count: int,
    command: str,
) -> str:
    if count <= 0:
        command = "echo no_rows_for_this_clean_job"
        array = "0-0"
    else:
        array = f"0-{count - 1}"
    return header(
        name=name,
        partition=partition,
        time=time,
        cpus=cpus,
        memory=memory,
        gpu=gpu,
        array=array,
    ) + f"\n{command}\n"


def scalar_job(
    *,
    name: str,
    partition: str,
    time: str,
    cpus: int,
    memory: str,
    command: str,
) -> str:
    return header(
        name=name,
        partition=partition,
        time=time,
        cpus=cpus,
        memory=memory,
        gpu=False,
    ) + f"\n{command}\n"


def submit_script(*, mode: str, include_materialize: bool) -> str:
    materialize = ""
    if include_materialize:
        materialize = f"""
MATERIALIZE=$(sbatch --parsable --dependency=afterok:$BEHAVIOR:$GEOMETRY:$EFFICIENCY \\
  outputs/paper1_publication_v0/preflight/cluster_jobs/{mode}/clean_materialize_axes.sbatch)
printf 'materialize_axes=%s\\n' "$MATERIALIZE"
"""
    return f"""#!/usr/bin/env bash
set -euo pipefail
source {CONDA_SH}
conda activate hma
cd {REMOTE_PROJECT}
mkdir -p slurm_logs outputs/paper1_publication_v0/preflight/cluster_submissions
python scripts/generate_paper1_publication_preflight.py
if [[ "{mode}" == "full" ]]; then
  python scripts/verify_paper1_clean_rerun_authorization.py --strict
fi
BEHAVIOR=$(sbatch --parsable outputs/paper1_publication_v0/preflight/cluster_jobs/{mode}/clean_behavior_cells.sbatch)
NEURAL_EXPORT=$(sbatch --parsable outputs/paper1_publication_v0/preflight/cluster_jobs/{mode}/clean_neural_exports.sbatch)
NEURAL_ANALYSIS=$(sbatch --parsable --dependency=afterok:$NEURAL_EXPORT \\
  outputs/paper1_publication_v0/preflight/cluster_jobs/{mode}/clean_neural_analysis.sbatch)
GEOMETRY=$(sbatch --parsable --dependency=afterok:$NEURAL_ANALYSIS \\
  outputs/paper1_publication_v0/preflight/cluster_jobs/{mode}/clean_geometry_collect.sbatch)
EFFICIENCY=$(sbatch --parsable outputs/paper1_publication_v0/preflight/cluster_jobs/{mode}/clean_efficiency_cells.sbatch)
printf 'mode={mode}\\n'
printf 'behavior=%s\\n' "$BEHAVIOR"
printf 'neural_export=%s\\n' "$NEURAL_EXPORT"
printf 'neural_analysis=%s\\n' "$NEURAL_ANALYSIS"
printf 'geometry_collect=%s\\n' "$GEOMETRY"
printf 'efficiency=%s\\n' "$EFFICIENCY"
{materialize}
"""


def write_manifest(
    *,
    job_root: Path,
    mode: str,
    max_items: int | None,
    limits: dict[str, int | None],
    cell_root: Path,
    behavior_rows: list[dict[str, Any]],
    neural_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": "hma.paper1.clean_cluster_jobs.v1",
        "mode": mode,
        "max_items": max_items,
        "limits": limits,
        "cell_root": relative(cell_root),
        "behavior_cells": len(behavior_rows),
        "neural_cells": len(neural_rows),
        "efficiency_cells": len(efficiency_rows),
        "unsupported_behavior_rows": len(unsupported),
        "contract_rule": "full mode jobs require configs/paper1_publication_contract.yaml execution_authorized true",
    }
    (job_root / "clean_cluster_job_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: str | Path) -> list[dict[str, str]]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[: max(0, int(limit))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fieldnames})
    return path


def parse_json_field(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return "".join(ch if ord(ch) >= 32 or ch in "\t\n\r" else " " for ch in text)


def relative(path: str | Path) -> str:
    resolved = resolve_path(path)
    return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "item"


def unique(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
