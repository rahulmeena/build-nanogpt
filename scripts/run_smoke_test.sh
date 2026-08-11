#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_DIR="${1:-runs/smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
INIT_CHECKPOINT="${EXP1_INIT_CHECKPOINT:-experiment_artifacts/baseline_init_seed1337.pt}"
if [[ -e "$RUN_DIR" ]]; then
  echo "run directory already exists: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/plots"

if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  "$PYTHON_BIN" scripts/initialization_control.py --checkpoint "$INIT_CHECKPOINT"
fi

echo "smoke run directory: $RUN_DIR"

"$PYTHON_BIN" scripts/assert_training_config.py --world-size 1 --require-world-size-one
"$PYTHON_BIN" scripts/verify_environment.py \
  --require-cuda \
  --require-a100-80gb \
  --require-bf16 \
  --require-sdpa \
  --require-fused-adamw \
  --out "$RUN_DIR/environment.json"

"$PYTHON_BIN" scripts/verify_dataset.py \
  --strict \
  --min-total-tokens 9900000000 \
  --max-total-tokens 10200000000 \
  --out "$RUN_DIR/dataset.json"

set +e
PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/smoke_test.py \
  --run-dir "$RUN_DIR" \
  --residual-mode standard \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --optimizer-steps "${SMOKE_STEPS:-2}" \
  --val-steps "${SMOKE_VAL_STEPS:-2}" \
  --hellaswag-examples "${SMOKE_HELLASWAG_EXAMPLES:-32}" \
  --require-a100-80gb \
  2>&1 | tee "$RUN_DIR/console.log"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 ]]; then
  echo "smoke test failed with exit code $status" >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/plot_reproduction.py "$RUN_DIR"
echo "smoke test complete: $RUN_DIR"
