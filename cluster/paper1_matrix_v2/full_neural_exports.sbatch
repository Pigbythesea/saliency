#!/usr/bin/env bash
#SBATCH --job-name=m2_full_nexp
#SBATCH --partition=l40s
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --array=0-2
#SBATCH --gres=gpu:l40s:1
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
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
"$MM" run -p "$PROJECT/external/environments/$MODEL"   python scripts/run_external_model.py   --model "$MODEL"   --manifest data/manifests/nsd_algonauts_prf_visualrois_full_manifest.csv   --image-root data/raw/nsd_algonauts   --split train --subject-id subj01 --roi V1   --batch-size 16   --output "outputs/paper1_matrix_v2/external_artifacts/full/$MODEL"
