#!/usr/bin/env bash
#SBATCH --job-name=m2_full_geom
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
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

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$MM" run -p "$CORE" python scripts/recompute_paper1_matrix_v2_geometry.py
