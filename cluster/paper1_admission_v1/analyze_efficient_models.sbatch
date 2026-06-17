#!/usr/bin/env bash
#SBATCH --job-name=p1adm_analyze_eff
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --array=0-1
#SBATCH --output=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%A_%a.err

set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
CORE="$PROJECT/external/environments/paper1_matrix_v2_core"
MODELS=(dynamicvit_deit_small_keep_0_7 tome_deit_small_r13)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
cd "$PROJECT"
source "$PROJECT/scripts/cluster_runtime_env.sh"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$MM" run --prefix "$CORE" python scripts/run_neural_alignment.py \
  --config "configs/experiments/paper1_admission_v1/${MODEL}_v1.yaml"

CONFIG=$("$MM" run --prefix "$CORE" \
  python scripts/prepare_paper1_admission_behavior_cell.py \
  --model "$MODEL" \
  --artifact "outputs/paper1_admission_v1/cluster/external/behavior/$MODEL")
"$MM" run --prefix "$CORE" python scripts/run_saliency_benchmark.py \
  --config "$CONFIG" --progress-interval 32
