#!/usr/bin/env bash
#SBATCH --job-name=m2_setup_models
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/tshu2/zzhan330/saliency/slurm_logs/%x_%j.err

set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
CORE="$PROJECT/external/environments/paper1_matrix_v2_core"
cd "$PROJECT"
source "$PROJECT/scripts/cluster_runtime_env.sh"
mkdir -p slurm_logs outputs/paper1_matrix_v2/preflight

MODELS=(
  dynamicvit_deit_small_keep_0_7
  deit_small_static
  tome_deit_small_r13
)
for MODEL in "${MODELS[@]}"; do
  python3 scripts/setup_external_model.py \
    --model "$MODEL" \
    --install \
    --download-checkpoint \
    --smoke
done

if [[ ! -x "$CORE/bin/python" ]]; then
  "$MM" create \
    --yes \
    --channel-priority strict \
    --prefix "$CORE" \
    --file configs/cluster/paper1_matrix_v2_core.yaml
fi
"$MM" run --prefix "$CORE" python -m pip install -e . --no-build-isolation
"$MM" run --prefix "$CORE" \
  python scripts/preflight_paper1_matrix_v2_cluster.py \
  --mode smoke \
  --json-output outputs/paper1_matrix_v2/preflight/setup_models.json
