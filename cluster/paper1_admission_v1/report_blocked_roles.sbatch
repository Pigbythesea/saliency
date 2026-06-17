#!/usr/bin/env bash
#SBATCH --job-name=p1adm_role_reports
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-2
#SBATCH --output=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.err

set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
MODELS=(hat scandiff adaptivenn_deit_small)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
cd "$PROJECT"
source "$PROJECT/scripts/cluster_runtime_env.sh"
mkdir -p outputs/paper1_admission_v1/cluster/blocked_role_reports
python3 scripts/setup_external_model.py --model "$MODEL" --report-only \
  > "outputs/paper1_admission_v1/cluster/blocked_role_reports/$MODEL.json"
