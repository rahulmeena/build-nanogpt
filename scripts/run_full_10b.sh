#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

UPSTREAM_BASE_SHA="6104ab1b53920f6e2159749676073ff7d815c1fa"
EXPECTED_BRANCH="gpt2-124m-10b-reproduction"
CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_SHA="$(git rev-parse HEAD)"

if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "expected branch $EXPECTED_BRANCH, found $CURRENT_BRANCH" >&2
  exit 1
fi

if ! git merge-base --is-ancestor "$UPSTREAM_BASE_SHA" HEAD; then
  echo "upstream base SHA $UPSTREAM_BASE_SHA is not an ancestor of HEAD $CURRENT_SHA" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY_GIT:-0}" != "1" ]]; then
  echo "working tree is dirty. Commit the reproduction harness, or rerun with ALLOW_DIRTY_GIT=1 to record an explicitly dirty run." >&2
  git status --short >&2
  exit 1
fi

"$PYTHON_BIN" scripts/assert_training_config.py --world-size 1 --require-world-size-one
"$PYTHON_BIN" scripts/verify_environment.py \
  --require-cuda \
  --require-a100-80gb \
  --require-bf16 \
  --require-sdpa \
  --require-fused-adamw

"$PYTHON_BIN" scripts/verify_dataset.py \
  --strict \
  --min-total-tokens 9900000000 \
  --max-total-tokens 10200000000

RUN_DIR="runs/gpt2_124m_fineweb10b_$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -e "$RUN_DIR" ]]; then
  echo "run directory already exists: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/log" "$RUN_DIR/plots"
echo "full reproduction run directory: $RUN_DIR"

{
  echo "repository_url: https://github.com/karpathy/build-nanogpt.git"
  echo "upstream_base_sha: $UPSTREAM_BASE_SHA"
  echo "current_sha: $CURRENT_SHA"
  echo "current_branch: $CURRENT_BRANCH"
  echo
  git remote -v
  echo
  git status --short --branch
  echo
  git diff "$UPSTREAM_BASE_SHA"
} > "$RUN_DIR/git_info.txt"

"$PYTHON_BIN" scripts/verify_environment.py --out "$RUN_DIR/environment.json"
"$PYTHON_BIN" scripts/verify_dataset.py \
  --strict \
  --min-total-tokens 9900000000 \
  --max-total-tokens 10200000000 \
  --out "$RUN_DIR/dataset.json"

"$PYTHON_BIN" - "$RUN_DIR" "$UPSTREAM_BASE_SHA" "$CURRENT_SHA" "$CURRENT_BRANCH" <<'PY'
import json
import sys
from pathlib import Path

run_dir, upstream_base_sha, current_sha, current_branch = sys.argv[1:]
metadata = {
    "repository_url": "https://github.com/karpathy/build-nanogpt.git",
    "upstream_base_sha": upstream_base_sha,
    "current_sha": current_sha,
    "current_branch": current_branch,
    "hardware_target": "1 x NVIDIA A100 80GB",
    "B": 64,
    "T": 1024,
    "total_batch_size_tokens": 524288,
    "world_size": 1,
    "gradient_accumulation_steps": 8,
    "max_steps": 19073,
    "precision": "BF16 autocast on CUDA",
    "torch_compile": False,
}
Path(run_dir, "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

set +e
PYTHONUNBUFFERED=1 \
NANOGPT_RUN_DIR="$RUN_DIR" \
NANOGPT_LOG_DIR="$RUN_DIR/log" \
NANOGPT_CHECKPOINT_DIR="$RUN_DIR/checkpoints" \
NANOGPT_METRICS_FILE="$RUN_DIR/metrics.jsonl" \
"$PYTHON_BIN" train_gpt2.py 2>&1 | tee "$RUN_DIR/console.log"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
  echo "full reproduction exited unexpectedly with exit code $status" >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/plot_reproduction.py "$RUN_DIR"
echo "full reproduction complete: $RUN_DIR"
