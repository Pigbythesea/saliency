#!/usr/bin/env bash
set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
CORE="$PROJECT/external/environments/paper1_matrix_v2_core"
cd "$PROJECT"
mkdir -p slurm_logs outputs/paper1_matrix_v2/preflight
"$MM" run -p "$CORE" python scripts/preflight_paper1_matrix_v2_cluster.py   --mode full   --json-output "outputs/paper1_matrix_v2/preflight/full_submission.json"
NEURAL_EXPORT=$(sbatch --parsable cluster/paper1_matrix_v2/full_neural_exports.sbatch)
BEHAVIOR_EXPORT=$(sbatch --parsable cluster/paper1_matrix_v2/full_behavior_exports.sbatch)
NEURAL_ANALYSIS=$(sbatch --parsable --dependency=afterok:$NEURAL_EXPORT   cluster/paper1_matrix_v2/full_neural_analysis.sbatch)
BEHAVIOR_SCORE=$(sbatch --parsable --dependency=afterok:$BEHAVIOR_EXPORT   cluster/paper1_matrix_v2/full_behavior_scores.sbatch)
printf 'mode=full\n'
printf 'stage=full\n'
printf 'neural_export=%s\n' "$NEURAL_EXPORT"
printf 'behavior_export=%s\n' "$BEHAVIOR_EXPORT"
printf 'neural_analysis=%s\n' "$NEURAL_ANALYSIS"
printf 'behavior_score=%s\n' "$BEHAVIOR_SCORE"
