#!/usr/bin/env bash
# One array index per GR-1 task. Adjust the SBATCH header for your cluster —
# partition, account and any site-specific required directives.
#
#   GR00T_ROOT=... MODEL_PATH=... sbatch --array=0-23%6 scripts/slurm_gr1_eval.sbatch
#SBATCH --job-name=gr1_tabletop_eval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x_%A_%a.out
set -euo pipefail
mkdir -p slurm_out
exec "$(dirname "${BASH_SOURCE[0]}")/run_gr1_eval.sh" "$SLURM_ARRAY_TASK_ID"
