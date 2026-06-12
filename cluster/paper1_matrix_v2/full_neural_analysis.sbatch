#!/usr/bin/env bash
#SBATCH --job-name=m2_full_nana
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --array=0-2
#SBATCH --output=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
CORE="$PROJECT/external/environments/paper1_matrix_v2_core"
cd "$PROJECT"
mkdir -p slurm_logs

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
MODELS=(deit_small_static dynamicvit_deit_small_keep_0_7 tome_deit_small_r13)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
if [[ "full" == "full" ]]; then
  "$MM" run -p "$CORE" python scripts/preflight_paper1_matrix_v2_cluster.py     --mode analysis --no-verify-all-images     --json-output "outputs/paper1_matrix_v2/preflight/analysis_${MODEL}.json"
fi
for ROI in v1 ventral lateral parietal; do
  "$MM" run -p "$CORE" python scripts/run_neural_alignment.py     --config "configs/experiments/paper1_matrix_v2/full/${MODEL}_${ROI}_full.yaml"
done
