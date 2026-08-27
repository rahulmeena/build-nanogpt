#!/usr/bin/env bash
set -Eeuo pipefail

MASTER_ROOT=${MASTER_ROOT:-/workspace/parallel_2d2_master}
export MASTER_ROOT
MASTER_WORKTREE="$MASTER_ROOT/worktrees/master"
F_WORKTREE="$MASTER_ROOT/worktrees/2d2f"
C1_OUTPUT="$MASTER_WORKTREE/results/experiment_2d2e_c1_large_true_self_confirmation"
F_OUTPUT="$F_WORKTREE/results/experiment_2d2f_no_b2_recurrence_b3_w64"
F_RUN_ROOT=/workspace/exp2d2f_run
F_EPHEMERAL=/tmp/parallel_2d2_ephemeral/2d2f

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
source "$MASTER_WORKTREE/scripts/parallel_2d2_lane_common.sh"
lane_init 0 "2D2E-C1 then 2D2F"
STOP_AUDIT="$MASTER_ROOT/runs/$MASTER_RUN_ID/AUTO_STOP_PREFLIGHT.json"
mkdir -p "$F_EPHEMERAL"

log_command C1_TRUE_INCREMENTAL python "$MASTER_WORKTREE/scripts/experiment_2d2e_c1.py" \
  --checkpoint /workspace/exp2d2e_run/checkpoints/scientific_update_0191.pt \
  --validation-shard /workspace/build-nanogpt/edu_fineweb10B/edufineweb_val_000000.npy \
  --prior-incremental /workspace/build-nanogpt-exp2d2e/results/experiment_2d2e_b3_w64_b10_recurrent_960/incremental_validation.json \
  --output-dir "$C1_OUTPUT" \
  --pod-id 7i2zyd53ytspwz \
  --pod-name empirical_tan_panda \
  --stop-audit "$STOP_AUDIT"

F_COMMON=(
  --source-checkpoint /workspace/exp2d2d_run/checkpoints/scientific_update_0191.pt
  --data-root /workspace/build-nanogpt/edu_fineweb10B
  --output-dir "$F_OUTPUT"
  --run-root "$F_RUN_ROOT"
  --ephemeral-checkpoint-dir "$F_EPHEMERAL"
  --pod-id 7i2zyd53ytspwz
  --pod-name empirical_tan_panda
  --gpu-type "NVIDIA A100-SXM4-80GB"
  --persistent-volume-identity yhzyb27fb5
  --stop-mechanism runpodctl_exact_pod_stop
  --stop-authenticated
  --stop-audit-path "$STOP_AUDIT"
)

cd "$F_WORKTREE"
log_command 2D2F_PREFLIGHT python scripts/experiment_2d2f.py preflight "${F_COMMON[@]}"
log_command 2D2F_SMOKE python scripts/experiment_2d2f.py smoke "${F_COMMON[@]}"
log_command 2D2F_TRAIN_TO_96 python scripts/experiment_2d2f.py train "${F_COMMON[@]}" --end-update 96
log_command 2D2F_RESUME_TO_191 python scripts/experiment_2d2f.py train "${F_COMMON[@]}" \
  --end-update 191 --resume "$F_EPHEMERAL/scientific_update_0096.pt"
log_command 2D2F_FINALIZE python scripts/experiment_2d2f.py finalize "${F_COMMON[@]}" \
  --final-checkpoint "$F_RUN_ROOT/checkpoints/scientific_update_0191.pt"

lane_mark_science_complete
