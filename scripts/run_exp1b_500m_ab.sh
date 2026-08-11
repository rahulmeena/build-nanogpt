#!/usr/bin/env bash
set -euo pipefail

cd /workspace/build-nanogpt
source /workspace/venvs/exp1b/bin/activate

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

base_dir="${1:-runs/exp1b_500m}"
if [[ -e "$base_dir" ]]; then
  echo "refusing to overwrite existing Experiment 1B directory: $base_dir" >&2
  exit 1
fi
mkdir -p "$base_dir"

common_args=(
  --config configs/exp1b_500m.json
  --init-checkpoint experiment_artifacts/baseline_init_seed1337.pt
  --dataset-report experiment_artifacts/exp1b_dataset.json
  --environment-report experiment_artifacts/exp1b_environment.json
  --dataset-manifest experiment_artifacts/edu_fineweb10B_sha256_manifest.txt
)

torchrun --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --residual-mode standard \
  --run-dir "$base_dir/standard" \
  "${common_args[@]}" 2>&1 | tee "$base_dir/standard_console.log"

torchrun --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --residual-mode full_attnres \
  --run-dir "$base_dir/full_attnres" \
  --expected-data-order "$base_dir/standard/data_order.json" \
  "${common_args[@]}" 2>&1 | tee "$base_dir/attnres_console.log"

CUDA_VISIBLE_DEVICES=0 python scripts/attnres_ablate.py \
  --checkpoint "$base_dir/full_attnres/checkpoints/checkpoint_tokens_000500170752.pt" \
  --out "$base_dir/full_attnres/causal_ablation.json" \
  --B 8 \
  --T 1024 \
  --val-steps 20 \
  --sources 0 4 8 12 16 20 24
