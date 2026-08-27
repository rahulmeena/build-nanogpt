#!/usr/bin/env bash
set -Eeuo pipefail

MASTER_ROOT=${MASTER_ROOT:-/workspace/parallel_2d2_master}
export MASTER_ROOT
WORKTREE="$MASTER_ROOT/worktrees/2d2i"
OUTPUT="$WORKTREE/results/experiment_2d2i_b4_w128_b9_recurrent"
RUN_ROOT=/tmp/parallel_2d2_ephemeral/2d2i
PERSISTENT_FINAL=/workspace/exp2d2i_run/checkpoints/scientific_update_0191.pt

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
source "$MASTER_ROOT/worktrees/master/scripts/parallel_2d2_lane_common.sh"
lane_init 3 "2D2I"
STOP_AUDIT="$MASTER_ROOT/runs/$MASTER_RUN_ID/AUTO_STOP_PREFLIGHT.json"
mkdir -p "$RUN_ROOT"

COMMON=(
  --source-checkpoint /workspace/exp2d2e_run/checkpoints/scientific_update_0191.pt
  --data-root /workspace/build-nanogpt/edu_fineweb10B
  --output-dir "$OUTPUT"
  --run-root "$RUN_ROOT"
  --pod-id 7i2zyd53ytspwz
  --pod-name empirical_tan_panda
  --gpu-type "NVIDIA A100-SXM4-80GB"
  --persistent-volume-identity yhzyb27fb5
  --stop-mechanism runpodctl_exact_pod_stop
  --stop-authenticated
  --stop-audit-path "$STOP_AUDIT"
)

cd "$WORKTREE"
log_command 2D2I_PREFLIGHT python scripts/experiment_2d2i.py preflight "${COMMON[@]}"
log_command 2D2I_SMOKE python scripts/experiment_2d2i.py smoke "${COMMON[@]}"
log_command 2D2I_TRAIN_TO_96 python scripts/experiment_2d2i.py train "${COMMON[@]}" --end-update 96
log_command 2D2I_RESUME_TO_191 python scripts/experiment_2d2i.py train "${COMMON[@]}" \
  --end-update 191 --resume "$RUN_ROOT/checkpoints/scientific_update_0096.pt"
log_command 2D2I_FINALIZE python scripts/experiment_2d2i.py finalize "${COMMON[@]}" \
  --final-checkpoint "$PERSISTENT_FINAL"

lane_mark_science_complete
