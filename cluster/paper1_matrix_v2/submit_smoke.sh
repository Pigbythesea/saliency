#!/usr/bin/env bash
set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
MM="$PROJECT/external/tools/micromamba/2.8.1-0/micromamba"
CORE="$PROJECT/external/environments/paper1_matrix_v2_core"
cd "$PROJECT"
source "$PROJECT/scripts/cluster_runtime_env.sh"
mkdir -p slurm_logs outputs/paper1_matrix_v2/preflight
"$MM" run -p "$CORE" python scripts/preflight_paper1_matrix_v2_cluster.py   --mode smoke   --json-output "outputs/paper1_matrix_v2/preflight/smoke_submission.json"
NEURAL_EXPORT=$(sbatch --parsable cluster/paper1_matrix_v2/smoke_neural_exports.sbatch)
BEHAVIOR_EXPORT=$(sbatch --parsable cluster/paper1_matrix_v2/smoke_behavior_exports.sbatch)
NEURAL_ANALYSIS=$(sbatch --parsable --dependency=afterok:$NEURAL_EXPORT   cluster/paper1_matrix_v2/smoke_neural_analysis.sbatch)
BEHAVIOR_SCORE=$(sbatch --parsable --dependency=afterok:$BEHAVIOR_EXPORT   cluster/paper1_matrix_v2/smoke_behavior_scores.sbatch)
if [[ "smoke" == "full" ]]; then
  GEOMETRY=$(sbatch --parsable --dependency=afterok:$NEURAL_ANALYSIS     cluster/paper1_matrix_v2/full_geometry_recompute.sbatch)
  EFFICIENCY=$(sbatch --parsable     cluster/paper1_matrix_v2/full_efficiency_profile.sbatch)
  SUMMARY=$(sbatch --parsable     --dependency=afterok:$BEHAVIOR_SCORE:$GEOMETRY:$EFFICIENCY     cluster/paper1_matrix_v2/full_summary_audit.sbatch)
fi
printf 'mode=smoke\n'
printf 'stage=scientific64\n'
printf 'neural_export=%s\n' "$NEURAL_EXPORT"
printf 'behavior_export=%s\n' "$BEHAVIOR_EXPORT"
printf 'neural_analysis=%s\n' "$NEURAL_ANALYSIS"
printf 'behavior_score=%s\n' "$BEHAVIOR_SCORE"
if [[ "smoke" == "full" ]]; then
  printf 'geometry=%s\n' "$GEOMETRY"
  printf 'efficiency=%s\n' "$EFFICIENCY"
  printf 'summary=%s\n' "$SUMMARY"
fi
