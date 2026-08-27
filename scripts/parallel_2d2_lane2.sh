#!/usr/bin/env bash
set -Eeuo pipefail

MASTER_ROOT=${MASTER_ROOT:-/workspace/parallel_2d2_master}
export MASTER_ROOT
WORKTREE="$MASTER_ROOT/worktrees/2d2h"
OUTPUT="$WORKTREE/results/experiment_2d2h_no_b1_recurrence_b2_w32"
RUN_ROOT=/workspace/exp2d2h_run
EPHEMERAL=/tmp/parallel_2d2_ephemeral/2d2h

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
source "$MASTER_ROOT/worktrees/master/scripts/parallel_2d2_lane_common.sh"
lane_init 2 "2D2H"
STOP_AUDIT="$MASTER_ROOT/runs/$MASTER_RUN_ID/AUTO_STOP_PREFLIGHT.json"
mkdir -p "$EPHEMERAL"

COMMON=(
  --source-checkpoint /workspace/exp2d2b_run/checkpoints/scientific_update_0191.pt
  --data-root /workspace/build-nanogpt/edu_fineweb10B
  --output-dir "$OUTPUT"
  --run-root "$RUN_ROOT"
  --ephemeral-checkpoint-root "$EPHEMERAL"
  --checkpoint-persist-lock "$MASTER_ROOT/locks/checkpoint_persist.lock"
  --pod-id 7i2zyd53ytspwz
  --pod-name empirical_tan_panda
  --gpu-type "NVIDIA A100-SXM4-80GB"
  --persistent-volume-identity yhzyb27fb5
  --stop-mechanism runpodctl_exact_pod_stop
  --stop-authenticated
  --stop-audit-path "$STOP_AUDIT"
)

cd "$WORKTREE"
log_command 2D2H_PREFLIGHT python scripts/experiment_2d2h.py preflight "${COMMON[@]}"
log_command 2D2H_SMOKE python scripts/experiment_2d2h.py smoke "${COMMON[@]}"
log_command 2D2H_TRAIN_TO_96 python scripts/experiment_2d2h.py train "${COMMON[@]}" --end-update 96
log_command 2D2H_RESUME_TO_191 python scripts/experiment_2d2h.py train "${COMMON[@]}" \
  --end-update 191 --resume "$EPHEMERAL/scientific_update_0096.pt"
log_command 2D2H_FINALIZE python scripts/experiment_2d2h.py finalize "${COMMON[@]}" \
  --final-checkpoint "$RUN_ROOT/checkpoints/scientific_update_0191.pt"

lane_mark_science_complete
