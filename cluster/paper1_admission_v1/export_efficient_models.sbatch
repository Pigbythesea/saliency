#!/usr/bin/env bash
#SBATCH --job-name=p1adm_export_eff
#SBATCH --partition=l40s
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --array=0-1
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
MODELS=(dynamicvit_deit_small_keep_0_7 tome_deit_small_r13)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
ENV="$PROJECT/external/environments/$MODEL"
cd "$PROJECT"
source "$PROJECT/scripts/cluster_runtime_env.sh"
mkdir -p slurm_logs

"$MM" run --prefix "$ENV" python scripts/run_external_model.py \
  --model "$MODEL" \
  --manifest data/manifests/nsd_algonauts_prf_visualrois_500_manifest.csv \
  --image-root data/raw/nsd_algonauts \
  --split train --subject-id subj01 --roi V1 \
  --max-items 96 --batch-size 16 \
  --output "outputs/paper1_admission_v1/cluster/external/neural/$MODEL"

"$MM" run --prefix "$ENV" python scripts/run_external_model.py \
  --model "$MODEL" \
  --manifest data/manifests/v2/salicon_static2000_manifest.csv \
  --image-root data/raw/SALICON \
  --split val --max-items 128 --batch-size 16 \
  --artifact-scope resource_only --artifact-key map_key \
  --skip-efficiency-profile \
  --output "outputs/paper1_admission_v1/cluster/external/behavior/$MODEL"
