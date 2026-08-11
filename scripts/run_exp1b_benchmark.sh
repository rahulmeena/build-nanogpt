#!/usr/bin/env bash
set -euo pipefail

cd /workspace/build-nanogpt
source /workspace/venvs/exp1b/bin/activate

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUBLAS_WORKSPACE_CONFIG=:4096:8

base_dir="${1:-runs/exp1b_benchmark}"
if [[ -e "$base_dir" ]]; then
  echo "refusing to overwrite existing benchmark directory: $base_dir" >&2
  exit 1
fi
mkdir -p "$base_dir"

common_args=(
  --config configs/exp1b_benchmark_10.json
  --init-checkpoint experiment_artifacts/baseline_init_seed1337.pt
  --dataset-report experiment_artifacts/exp1b_dataset.json
  --environment-report experiment_artifacts/exp1b_environment.json
  --dataset-manifest experiment_artifacts/edu_fineweb10B_sha256_manifest.txt
)

monitor_pid=""
stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap stop_monitor EXIT

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv -l 1 > "$base_dir/standard_gpu.csv" &
monitor_pid=$!
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --residual-mode standard \
  --run-dir "$base_dir/standard" \
  "${common_args[@]}" 2>&1 | tee "$base_dir/standard_console.log"
stop_monitor

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv -l 1 > "$base_dir/attnres_gpu.csv" &
monitor_pid=$!
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --residual-mode full_attnres \
  --run-dir "$base_dir/full_attnres" \
  --expected-data-order "$base_dir/standard/data_order.json" \
  "${common_args[@]}" 2>&1 | tee "$base_dir/attnres_console.log"
stop_monitor

python scripts/summarize_exp1b_scaling.py \
  --standard-run "$base_dir/standard" \
  --standard-monitor "$base_dir/standard_gpu.csv" \
  --attnres-run "$base_dir/full_attnres" \
  --attnres-monitor "$base_dir/attnres_gpu.csv" \
  --out "$base_dir/scaling_summary.json"
