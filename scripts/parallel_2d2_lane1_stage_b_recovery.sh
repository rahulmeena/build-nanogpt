#!/usr/bin/env bash
set -Eeuo pipefail

MASTER_ROOT=${MASTER_ROOT:-/workspace/parallel_2d2_master}
export MASTER_ROOT
WORKTREE="$MASTER_ROOT/worktrees/2d2g"
OUTPUT="$WORKTREE/results/experiment_2d2g_b2_full_b3_w64"
EPHEMERAL=/tmp/parallel_2d2_ephemeral/2d2g/checkpoints
SMOKE_EPHEMERAL=/tmp/parallel_2d2_ephemeral/2d2g/smoke
PERSISTENT=/workspace/exp2d2g_run/checkpoints

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MASTER_RECOVERY_MODE=1
source "$MASTER_ROOT/worktrees/master/scripts/parallel_2d2_lane_common.sh"
lane_init 1 "2D2G-B recovery from exact Stage-A-191"
mkdir -p "$EPHEMERAL" "$SMOKE_EPHEMERAL"

cd "$WORKTREE"
COMMON=(
  --output-dir "$OUTPUT"
  --pod-id 7i2zyd53ytspwz
  --pod-name empirical_tan_panda
)
DATA_ROOT=/workspace/build-nanogpt/edu_fineweb10B
SOURCE=/workspace/exp2d2b_run/checkpoints/scientific_update_0191.pt
A191="$EPHEMERAL/stage_a_scientific_update_0191.pt"
B96="$EPHEMERAL/stage_b_scientific_update_0096.pt"
B191="$EPHEMERAL/stage_b_scientific_update_0191.pt"

log_command 2D2G_RECOVERY_PREFLIGHT python scripts/experiment_2d2g.py preflight "${COMMON[@]}" \
  --source-checkpoint "$SOURCE" --data-root "$DATA_ROOT"
log_command 2D2G_B_RECOVERY_SMOKE python scripts/experiment_2d2g.py smoke-b "${COMMON[@]}" \
  --stage-a-checkpoint "$A191" --checkpoint-dir "$SMOKE_EPHEMERAL" \
  --data-root "$DATA_ROOT"
log_command 2D2G_B_RECOVERY_TRAIN_TO_96 python scripts/experiment_2d2g.py train-b "${COMMON[@]}" \
  --stage-a-checkpoint "$A191" --checkpoint-dir "$EPHEMERAL" \
  --end-update 96 --data-root "$DATA_ROOT"
log_command 2D2G_B_RECOVERY_RESUME_TO_191 python scripts/experiment_2d2g.py train-b "${COMMON[@]}" \
  --resume "$B96" --checkpoint-dir "$EPHEMERAL" \
  --end-update 191 --data-root "$DATA_ROOT"
log_command 2D2G_RECOVERY_PERSIST_FINAL python scripts/experiment_2d2g.py persist-final \
  --output-dir "$OUTPUT" \
  --local-checkpoint "$B191" \
  --persistent-dir "$PERSISTENT" \
  --lock-path "$MASTER_ROOT/locks/checkpoint_persist.lock"
log_command 2D2G_RECOVERY_FINALIZE python scripts/experiment_2d2g.py finalize "${COMMON[@]}" \
  --stage-b-checkpoint "$PERSISTENT/stage_b_scientific_update_0191.pt" \
  --data-root "$DATA_ROOT"

lane_mark_science_complete
