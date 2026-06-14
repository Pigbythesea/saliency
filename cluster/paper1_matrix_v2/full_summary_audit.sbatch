#!/usr/bin/env bash
#SBATCH --job-name=m2_full_audit
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-0
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

"$MM" run -p "$CORE" python scripts/summarize_paper1_matrix_v2_full.py
