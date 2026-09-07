#!/usr/bin/env bash
# Submission template for `kitchen_protocol.repair apply --submit`.
#
# Adjust the SBATCH flags for your cluster -- partition, account, and any
# site-specific required directives. The two settings that are *not* free to
# change are N_ENVS=1 (one env per process is what removes the swap) and
# ENV_LAYOUT left at the original lane count (what keeps episode seeds
# identical to the run being repaired).
#
#   SUBMIT_CMD="$(cat scripts/submit_repair_slurm.sh)" \
#   python -m kitchen_protocol.repair apply <run> --ckpt-dir ... --step ... --submit
#
# Placeholders filled in by repair.py: {array} {ckpt} {step} {suffix} {n}
set -euo pipefail

sbatch \
  --job-name="kitchen_repair_{step}" \
  --array="{array}%{n}" \
  --export=ALL,\
CKPT_CONFIGS_OVERRIDE="{ckpt}:{step}",\
MODEL_OUTPUT_DIR="{ckpt}",\
N_ENVS=1,\
ENV_LAYOUT=5,\
EVAL_SUFFIX="{suffix}" \
  "${GR00T_ROOT:?set GR00T_ROOT}/run_scripts/eval/robocasa/sbatch_l40s_wan22_kitchen_24task.sh"
