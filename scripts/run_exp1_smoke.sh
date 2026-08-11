#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:?usage: scripts/run_exp1_smoke.sh standard|full_attnres [run_dir]}"
if [[ "$MODE" != "standard" && "$MODE" != "full_attnres" ]]; then
  echo "mode must be standard or full_attnres" >&2
  exit 1
fi

INIT_CHECKPOINT="${EXP1_INIT_CHECKPOINT:-experiment_artifacts/baseline_init_seed1337.pt}"
RUN_DIR="${2:-runs/exp1_${MODE}_smoke_seed1337_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "missing canonical initialization checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "run directory already exists: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/plots"

"$PYTHON_BIN" scripts/verify_environment.py \
  --require-cuda --require-a100-80gb --require-bf16 --require-sdpa --require-fused-adamw \
  --out "$RUN_DIR/environment.json"
"$PYTHON_BIN" scripts/verify_dataset.py \
  --strict --min-total-tokens 9900000000 --max-total-tokens 10200000000 \
  --out "$RUN_DIR/dataset.json"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/smoke_test.py \
  --run-dir "$RUN_DIR" \
  --residual-mode "$MODE" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --optimizer-steps "${EXP1_SMOKE_STEPS:-12}" \
  --val-steps "${EXP1_SMOKE_VAL_STEPS:-2}" \
  --hellaswag-examples "${EXP1_SMOKE_HELLASWAG_EXAMPLES:-32}" \
  --require-a100-80gb \
  2>&1 | tee "$RUN_DIR/console.log"

"$PYTHON_BIN" scripts/plot_reproduction.py "$RUN_DIR"
echo "Experiment 1 smoke complete: $RUN_DIR"
