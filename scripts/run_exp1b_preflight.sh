#!/usr/bin/env bash
set -euo pipefail

cd /workspace/build-nanogpt
source /workspace/venvs/exp1b/bin/activate

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

torchrun --standalone --nproc_per_node=4 scripts/verify_exp1b_validation_ddp.py \
  --residual-mode standard \
  --init-checkpoint experiment_artifacts/baseline_init_seed1337.pt \
  --out runs/exp1b_preflight/validation_single_vs_ddp.json
