#!/usr/bin/env bash
#SBATCH --job-name=m2_full_bscore
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --array=0-8
#SBATCH --output=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
CORE="$PROJECT/external/environments/paper1_matrix_v2_core"
cd "$PROJECT"
source "$PROJECT/scripts/cluster_runtime_env.sh"
mkdir -p slurm_logs

MODELS=(deit_small_static dynamicvit_deit_small_keep_0_7 tome_deit_small_r13)
DATASETS=(salicon cat2000 coco_search18)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MODEL="${MODELS[$MODEL_INDEX]}"
DATASET="${DATASETS[$DATASET_INDEX]}"
ARTIFACT="outputs/paper1_matrix_v2/external_artifacts/behavior_full/$DATASET/$MODEL"
CONFIG=$("$MM" run -p "$CORE"   python scripts/prepare_paper1_matrix_v2_behavior_cell.py   --dataset "$DATASET" --model "$MODEL" --artifact "$ARTIFACT"   --stage "full")
"$MM" run -p "$CORE" python scripts/run_saliency_benchmark.py   --config "$CONFIG" --progress-interval 100
