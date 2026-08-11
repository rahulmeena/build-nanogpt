#!/usr/bin/env bash
set -euo pipefail

cd /workspace/build-nanogpt
source /workspace/venvs/exp1b/bin/activate

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

base_dir="${1:-runs/exp1b_resume_proof}"
if [[ -e "$base_dir" ]]; then
  echo "refusing to overwrite existing resume-proof directory: $base_dir" >&2
  exit 1
fi
mkdir -p "$base_dir"

common_args=(
  --init-checkpoint experiment_artifacts/baseline_init_seed1337.pt
  --dataset-report experiment_artifacts/exp1b_dataset.json
  --environment-report experiment_artifacts/exp1b_environment.json
  --dataset-manifest experiment_artifacts/edu_fineweb10B_sha256_manifest.txt
)

torchrun --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --config configs/exp1b_resume_path_a.json \
  --residual-mode standard \
  --run-dir "$base_dir/standard_path_a" \
  "${common_args[@]}"

torchrun --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --config configs/exp1b_resume_path_b.json \
  --residual-mode standard \
  --run-dir "$base_dir/standard_path_b" \
  --expected-data-order "$base_dir/standard_path_a/data_order.json" \
  --stop-after-completed-updates 5 \
  "${common_args[@]}"

torchrun --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --config configs/exp1b_resume_path_b.json \
  --residual-mode standard \
  --run-dir "$base_dir/standard_path_b" \
  --resume-checkpoint "$base_dir/standard_path_b/checkpoints/checkpoint_tokens_000002621440.pt" \
  "${common_args[@]}"

python scripts/compare_exp1b_resume.py \
  --path-a-run "$base_dir/standard_path_a" \
  --path-b-run "$base_dir/standard_path_b" \
  --path-a-checkpoint "$base_dir/standard_path_a/checkpoints/checkpoint_tokens_000005242880.pt" \
  --path-b-checkpoint "$base_dir/standard_path_b/checkpoints/checkpoint_tokens_000005242880.pt" \
  --out "$base_dir/standard_exact_resume.json"

torchrun --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --config configs/exp1b_resume_path_a.json \
  --residual-mode full_attnres \
  --run-dir "$base_dir/attnres_path_a" \
  --expected-data-order "$base_dir/standard_path_a/data_order.json" \
  "${common_args[@]}"

torchrun --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --config configs/exp1b_resume_path_b.json \
  --residual-mode full_attnres \
  --run-dir "$base_dir/attnres_path_b" \
  --expected-data-order "$base_dir/standard_path_a/data_order.json" \
  --stop-after-completed-updates 5 \
  "${common_args[@]}"

torchrun --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --config configs/exp1b_resume_path_b.json \
  --residual-mode full_attnres \
  --run-dir "$base_dir/attnres_path_b" \
  --resume-checkpoint "$base_dir/attnres_path_b/checkpoints/checkpoint_tokens_000002621440.pt" \
  "${common_args[@]}"

python scripts/compare_exp1b_resume.py \
  --path-a-run "$base_dir/attnres_path_a" \
  --path-b-run "$base_dir/attnres_path_b" \
  --path-a-checkpoint "$base_dir/attnres_path_a/checkpoints/checkpoint_tokens_000005242880.pt" \
  --path-b-checkpoint "$base_dir/attnres_path_b/checkpoints/checkpoint_tokens_000005242880.pt" \
  --out "$base_dir/attnres_exact_resume.json"
