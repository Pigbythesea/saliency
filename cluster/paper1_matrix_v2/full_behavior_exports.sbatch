#!/usr/bin/env bash
#SBATCH --job-name=m2_full_bexp
#SBATCH --partition=l40s
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --array=0-8
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
CORE="$PROJECT/external/environments/paper1_matrix_v2_core"
cd "$PROJECT"
mkdir -p slurm_logs

MODELS=(deit_small_static dynamicvit_deit_small_keep_0_7 tome_deit_small_r13)
DATASETS=(salicon cat2000 coco_search18)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MODEL="${MODELS[$MODEL_INDEX]}"
DATASET="${DATASETS[$DATASET_INDEX]}"
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
"$MM" run -p "$PROJECT/external/environments/$MODEL"   python scripts/run_external_model.py   --model "$MODEL" --manifest "$MANIFEST" --image-root "$ROOT"   --split "$SPLIT" --batch-size 16 --artifact-scope resource_only   --output "outputs/paper1_matrix_v2/external_artifacts/behavior_full/$DATASET/$MODEL"
