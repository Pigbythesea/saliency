#!/usr/bin/env bash
set -euo pipefail
PROJECT=/scratch/tshu2/zzhan330/saliency
cd "$PROJECT"
mkdir -p slurm_logs outputs/paper1_admission_v1/cluster

SETUP=$(sbatch --parsable cluster/paper1_admission_v1/setup_efficient_models.sbatch)
EXPORT=$(sbatch --parsable --dependency=afterok:$SETUP \
  cluster/paper1_admission_v1/export_efficient_models.sbatch)
ANALYZE=$(sbatch --parsable --dependency=afterok:$EXPORT \
  cluster/paper1_admission_v1/analyze_efficient_models.sbatch)
BLOCKERS=$(sbatch --parsable cluster/paper1_admission_v1/report_blocked_roles.sbatch)

printf 'setup=%s\n' "$SETUP"
printf 'export=%s\n' "$EXPORT"
printf 'analyze=%s\n' "$ANALYZE"
printf 'blocked_role_reports=%s\n' "$BLOCKERS"
