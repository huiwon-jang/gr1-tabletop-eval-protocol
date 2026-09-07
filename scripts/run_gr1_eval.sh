#!/usr/bin/env bash
# Evaluate a GR00T policy on the 24-task GR-1 tabletop benchmark under the
# reset protocol (5 s settle + frozen seed manifest).
#
#   GR00T_ROOT=/path/to/Isaac-GR00T MODEL_PATH=nvidia/GR00T-N1.5-3B \
#       scripts/run_gr1_eval.sh [TASK_INDEX]
#
# With no TASK_INDEX all 24 tasks run in sequence. Pass an index (0-23), or set
# SLURM_ARRAY_TASK_ID, to run exactly one — that is how the benchmark is
# normally parallelised (see scripts/slurm_gr1_eval.sbatch).
#
# Prereqs: run scripts/apply_protocol.sh against $GR00T_ROOT once, and have the
# GR-1 sim environment installed (see README).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_ROOT="${GR00T_ROOT:?set GR00T_ROOT to your Isaac-GR00T checkout}"
MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.5-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/gr1_eval_output}"

# --- protocol ---------------------------------------------------------------
# Both knobs are what make two runs comparable; unset either and you are back to
# ad-hoc episodes. GR1_PROTOCOL_SEED is the lane base: sub-env i uses seed+i.
export GR1_RESET_SETTLE_S="${GR1_RESET_SETTLE_S:-5.0}"
export GR1_PROTOCOL_SEED="${GR1_PROTOCOL_SEED:-42}"
export GR1_EPISODE_SEED_MANIFEST="${GR1_EPISODE_SEED_MANIFEST:-$HERE/manifest/gr1_episode_seed_manifest.json}"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

# --- rollout shape ----------------------------------------------------------
# 5 concurrent envs x 10 episodes = 50 episodes per task, 1200 per checkpoint.
# The shipped manifest was measured for exactly this shape; changing N_ENVS or
# N_EPISODES changes which (lane, episode) slots exist and silently invalidates
# the substitution table.
N_ENVS="${N_ENVS:-5}"
N_EPISODES="${N_EPISODES:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
PORT="${PORT:-$((5555 + RANDOM % 1000))}"

TASK_NAMES=(
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
)

# The rollout must run in the GR-1 sim environment (robosuite/robocasa/mujoco),
# which is normally a separate venv from the one serving the policy.
ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-$GR00T_ROOT/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python}"
SERVER_PYTHON="${SERVER_PYTHON:-python}"
test -x "$ROLLOUT_PYTHON" || { echo "[e] no GR-1 sim python at $ROLLOUT_PYTHON" >&2; exit 1; }

wait_for_server () {
    local port="$1" timeout="${2:-600}" waited=0
    while ! (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
        sleep 5; waited=$((waited + 5))
        if [ "$waited" -ge "$timeout" ]; then echo "[e] server did not come up in ${timeout}s" >&2; return 1; fi
        if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "[e] server died; see $OUTPUT_ROOT/server.log" >&2; return 1; fi
    done
}

INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [ -n "$INDEX" ]; then
    RUN_INDICES=("$INDEX")
else
    RUN_INDICES=($(seq 0 $((${#TASK_NAMES[@]} - 1))))
fi

mkdir -p "$OUTPUT_ROOT"
echo "[i] policy server: $MODEL_PATH (port $PORT)"
"$SERVER_PYTHON" "$GR00T_ROOT/gr00t/eval/run_gr00t_server.py" \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 --port "$PORT" > "$OUTPUT_ROOT/server.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
wait_for_server "$PORT" 600

for idx in "${RUN_INDICES[@]}"; do
    TASK_NAME="${TASK_NAMES[$idx]}"
    OUT="$OUTPUT_ROOT/$(basename "$TASK_NAME")"
    mkdir -p "$OUT"
    echo "[i] task $idx/${#TASK_NAMES[@]}: $TASK_NAME"
    "$ROLLOUT_PYTHON" "$GR00T_ROOT/gr00t/eval/rollout_policy.py" \
        --n_episodes "$N_EPISODES" \
        --policy_client_host 127.0.0.1 \
        --policy_client_port "$PORT" \
        --max_episode_steps "$MAX_EPISODE_STEPS" \
        --env_name "$TASK_NAME" \
        --n_action_steps "$N_ACTION_STEPS" \
        --n_envs "$N_ENVS" \
        --video_dir "$OUT" \
        --seed "$GR1_PROTOCOL_SEED" \
        2>&1 | tee "$OUT/rollout.log"
done

echo "[i] done. substituted slots:"
grep -h '\[seed-manifest\]' "$OUTPUT_ROOT"/*/rollout.log 2>/dev/null | wc -l
