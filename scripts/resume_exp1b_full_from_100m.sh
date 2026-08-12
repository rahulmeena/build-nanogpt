#!/usr/bin/env bash
set -euo pipefail

cd /workspace/build-nanogpt
source /workspace/venvs/exp1b/bin/activate

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8

base_dir="${1:-runs/exp1b_500m}"
full_dir="$base_dir/full_attnres"
resume_checkpoint="$full_dir/checkpoints/checkpoint_tokens_000100139008.pt"
resume_updates=191

for required in \
  "$resume_checkpoint" \
  "$resume_checkpoint.complete" \
  "$resume_checkpoint.sha256" \
  "$full_dir/checkpoints/checkpoint_tokens_000100139008.verification.json"; do
  if [[ ! -f "$required" ]]; then
    echo "missing verified 100M resume artifact: $required" >&2
    exit 1
  fi
done

# A failed attempt may have appended training/evaluation rows beyond the last
# completed checkpoint. Preserve that evidence, then replay the uncheckpointed
# suffix from the exact 100M state without duplicate metric steps.
recovery_archive="$full_dir/recovery_archive_after_100m"
if [[ ! -d "$recovery_archive" ]]; then
  mkdir -p "$recovery_archive"
  cp "$full_dir/metrics.jsonl" "$recovery_archive/metrics_before_truncation.jsonl"
  if [[ -f "$full_dir/attnres_stats.jsonl" ]]; then
    cp "$full_dir/attnres_stats.jsonl" "$recovery_archive/attnres_stats_before_truncation.jsonl"
  fi
  for rejected in "$full_dir/checkpoints"/*; do
    [[ -e "$rejected" ]] || continue
    case "$(basename "$rejected")" in
      checkpoint_tokens_000100139008.pt|\
      checkpoint_tokens_000100139008.pt.complete|\
      checkpoint_tokens_000100139008.pt.sha256|\
      checkpoint_tokens_000100139008.verification.json)
        ;;
      *) mv "$rejected" "$recovery_archive/" ;;
    esac
  done
  /workspace/venvs/exp1b/bin/python - "$full_dir" "$resume_updates" <<'PY'
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
resume_updates = int(sys.argv[2])

for name in ("metrics.jsonl", "attnres_stats.jsonl"):
    path = run_dir / name
    if not path.is_file():
        continue
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    kept = [row for row in rows if int(row.get("completed_updates", 0)) <= resume_updates]
    temporary = path.with_suffix(path.suffix + ".recovery.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in kept))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
PY
fi

actual_train_rows="$({ /workspace/venvs/exp1b/bin/python - "$full_dir/metrics.jsonl" "$resume_updates" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
steps = [row["step"] for row in rows if row.get("kind") == "train"]
expected = list(range(int(sys.argv[2])))
if steps != expected:
    raise SystemExit(f"pre-resume metrics are not contiguous through step {expected[-1]}: {steps[:3]}...{steps[-3:]}")
print(len(steps))
PY
} 2>&1)" || {
  echo "$actual_train_rows" >&2
  exit 1
}
if [[ "$actual_train_rows" != "$resume_updates" ]]; then
  echo "unexpected retained training-row count: $actual_train_rows" >&2
  exit 1
fi

common_args=(
  --config configs/exp1b_500m.json
  --init-checkpoint experiment_artifacts/baseline_init_seed1337.pt
  --dataset-report experiment_artifacts/exp1b_dataset.json
  --environment-report experiment_artifacts/exp1b_environment.json
  --dataset-manifest experiment_artifacts/edu_fineweb10B_sha256_manifest.txt
)

python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/experiment_train_ddp.py \
  --residual-mode full_attnres \
  --run-dir "$full_dir" \
  --expected-data-order "$base_dir/standard/data_order.json" \
  --resume-checkpoint "$resume_checkpoint" \
  "${common_args[@]}" 2>&1 | tee "$base_dir/attnres_resume100m_console.log"

CUDA_VISIBLE_DEVICES=0 python scripts/attnres_ablate.py \
  --checkpoint "$full_dir/checkpoints/checkpoint_tokens_000500170752.pt" \
  --out "$full_dir/causal_ablation.json" \
  --B 8 \
  --T 1024 \
  --val-steps 20 \
  --sources 0 4 8 12 16 20 24
