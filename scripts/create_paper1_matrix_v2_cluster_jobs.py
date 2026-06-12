"""Generate JHU Slurm jobs for Matrix V2 smoke and full execution."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT = "/scratch/tshu2/zzhan330/saliency"
JOB_ROOT = Path("cluster/paper1_matrix_v2")


def create_cluster_jobs(output_root: str | Path = JOB_ROOT) -> list[Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "smoke_neural_exports.sbatch": _neural_export_job("scientific64", 64, "02:00:00"),
        "smoke_neural_analysis.sbatch": _neural_analysis_job("scientific64", "04:00:00"),
        "smoke_behavior_exports.sbatch": _behavior_export_job(
            "cluster_smoke16", 16, "02:00:00"
        ),
        "smoke_behavior_scores.sbatch": _behavior_score_job(
            "cluster_smoke16", 16, "02:00:00"
        ),
        "full_neural_exports.sbatch": _neural_export_job("full", None, "12:00:00"),
        "full_neural_analysis.sbatch": _neural_analysis_job("full", "2-00:00:00"),
        "full_behavior_exports.sbatch": _behavior_export_job(
            "full", None, "08:00:00"
        ),
        "full_behavior_scores.sbatch": _behavior_score_job(
            "full", None, "12:00:00"
        ),
        "submit_smoke.sh": _submit_script("smoke"),
        "submit_full.sh": _submit_script("full"),
    }
    paths = []
    for name, content in files.items():
        path = root / name
        path.write_text(content, encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def _header(
    *,
    name: str,
    partition: str,
    time: str,
    cpus: int,
    memory: str,
    array: str,
    gpu: bool,
) -> str:
    gres = "#SBATCH --gres=gpu:l40s:1\n" if gpu else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --array={array}
{gres}#SBATCH --output={PROJECT}/slurm_logs/%x_%A_%a.out
#SBATCH --error={PROJECT}/slurm_logs/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail
PROJECT={PROJECT}
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
CORE="$PROJECT/external/environments/paper1_matrix_v2_core"
cd "$PROJECT"
mkdir -p slurm_logs
"""


def _neural_export_job(stage: str, max_items: int | None, time: str) -> str:
    max_flag = f" --max-items {max_items}" if max_items is not None else ""
    return _header(
        name=f"m2_{stage}_nexp",
        partition="l40s",
        time=time,
        cpus=14,
        memory="48G",
        array="0-2",
        gpu=True,
    ) + f"""
MODELS=(deit_small_static dynamicvit_deit_small_keep_0_7 tome_deit_small_r13)
MODEL="${{MODELS[$SLURM_ARRAY_TASK_ID]}}"
"$MM" run -p "$PROJECT/external/environments/$MODEL" \
  python scripts/run_external_model.py \
  --model "$MODEL" \
  --manifest data/manifests/nsd_algonauts_prf_visualrois_full_manifest.csv \
  --image-root data/raw/nsd_algonauts \
  --split train --subject-id subj01 --roi V1 \
  --batch-size 16{max_flag} \
  --output "outputs/paper1_matrix_v2/external_artifacts/{stage}/$MODEL"
"""


def _neural_analysis_job(stage: str, time: str) -> str:
    return _header(
        name=f"m2_{stage}_nana",
        partition="cpu",
        time=time,
        cpus=32,
        memory="120G",
        array="0-2",
        gpu=False,
    ) + f"""
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
MODELS=(deit_small_static dynamicvit_deit_small_keep_0_7 tome_deit_small_r13)
MODEL="${{MODELS[$SLURM_ARRAY_TASK_ID]}}"
if [[ "{stage}" == "full" ]]; then
  "$MM" run -p "$CORE" python scripts/preflight_paper1_matrix_v2_cluster.py \
    --mode analysis --no-verify-all-images \
    --json-output "outputs/paper1_matrix_v2/preflight/analysis_${{MODEL}}.json"
fi
for ROI in v1 ventral lateral parietal; do
  "$MM" run -p "$CORE" python scripts/run_neural_alignment.py \
    --config "configs/experiments/paper1_matrix_v2/{stage}/${{MODEL}}_${{ROI}}_{stage}.yaml"
done
"""


def _behavior_export_job(stage: str, max_items: int | None, time: str) -> str:
    max_flag = f" --max-items {max_items}" if max_items is not None else ""
    return _header(
        name=f"m2_{stage}_bexp",
        partition="l40s",
        time=time,
        cpus=14,
        memory="48G",
        array="0-8",
        gpu=True,
    ) + f"""
MODELS=(deit_small_static dynamicvit_deit_small_keep_0_7 tome_deit_small_r13)
DATASETS=(salicon cat2000 coco_search18)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MODEL="${{MODELS[$MODEL_INDEX]}}"
DATASET="${{DATASETS[$DATASET_INDEX]}}"
case "$DATASET" in
  salicon)
    MANIFEST=data/manifests/v2/salicon_static2000_manifest.csv
    ROOT=data/raw/SALICON
    SPLIT=val
    ;;
  cat2000)
    MANIFEST=data/manifests/v2/cat2000_static2000_manifest.csv
    ROOT=data/raw/CAT2000
    SPLIT=train
    ;;
  coco_search18)
    MANIFEST=data/manifests/v2/coco_search18_static2000_manifest.csv
    ROOT=data/raw/COCO-Search18
    SPLIT=val
    ;;
esac
"$MM" run -p "$PROJECT/external/environments/$MODEL" \
  python scripts/run_external_model.py \
  --model "$MODEL" --manifest "$MANIFEST" --image-root "$ROOT" \
  --split "$SPLIT" --batch-size 16 --artifact-scope resource_only{max_flag} \
  --output "outputs/paper1_matrix_v2/external_artifacts/behavior_{stage}/$DATASET/$MODEL"
"""


def _behavior_score_job(stage: str, max_items: int | None, time: str) -> str:
    max_flag = f" --max-items {max_items}" if max_items is not None else ""
    return _header(
        name=f"m2_{stage}_bscore",
        partition="cpu",
        time=time,
        cpus=8,
        memory="30G",
        array="0-8",
        gpu=False,
    ) + f"""
MODELS=(deit_small_static dynamicvit_deit_small_keep_0_7 tome_deit_small_r13)
DATASETS=(salicon cat2000 coco_search18)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MODEL="${{MODELS[$MODEL_INDEX]}}"
DATASET="${{DATASETS[$DATASET_INDEX]}}"
ARTIFACT="outputs/paper1_matrix_v2/external_artifacts/behavior_{stage}/$DATASET/$MODEL"
CONFIG=$("$MM" run -p "$CORE" \
  python scripts/prepare_paper1_matrix_v2_behavior_cell.py \
  --dataset "$DATASET" --model "$MODEL" --artifact "$ARTIFACT" \
  --stage "{stage}"{max_flag})
"$MM" run -p "$CORE" python scripts/run_saliency_benchmark.py \
  --config "$CONFIG" --progress-interval 100
"""


def _submit_script(mode: str) -> str:
    stage = "scientific64" if mode == "smoke" else "full"
    preflight = "smoke" if mode == "smoke" else "full"
    return f"""#!/usr/bin/env bash
set -euo pipefail
PROJECT={PROJECT}
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
CORE="$PROJECT/external/environments/paper1_matrix_v2_core"
cd "$PROJECT"
mkdir -p slurm_logs outputs/paper1_matrix_v2/preflight
"$MM" run -p "$CORE" python scripts/preflight_paper1_matrix_v2_cluster.py \
  --mode {preflight} \
  --json-output "outputs/paper1_matrix_v2/preflight/{mode}_submission.json"
NEURAL_EXPORT=$(sbatch --parsable cluster/paper1_matrix_v2/{mode}_neural_exports.sbatch)
BEHAVIOR_EXPORT=$(sbatch --parsable cluster/paper1_matrix_v2/{mode}_behavior_exports.sbatch)
NEURAL_ANALYSIS=$(sbatch --parsable --dependency=afterok:$NEURAL_EXPORT \
  cluster/paper1_matrix_v2/{mode}_neural_analysis.sbatch)
BEHAVIOR_SCORE=$(sbatch --parsable --dependency=afterok:$BEHAVIOR_EXPORT \
  cluster/paper1_matrix_v2/{mode}_behavior_scores.sbatch)
printf 'mode={mode}\\n'
printf 'stage={stage}\\n'
printf 'neural_export=%s\\n' "$NEURAL_EXPORT"
printf 'behavior_export=%s\\n' "$BEHAVIOR_EXPORT"
printf 'neural_analysis=%s\\n' "$NEURAL_ANALYSIS"
printf 'behavior_score=%s\\n' "$BEHAVIOR_SCORE"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(JOB_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = create_cluster_jobs(args.output_root)
    print(f"Generated {len(paths)} cluster files under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
