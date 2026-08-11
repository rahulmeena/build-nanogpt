#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:?usage: CONFIRM_EXP1_AB=1 scripts/run_exp1_ab.sh configs/exp1_100m.json}"
if [[ "${CONFIRM_EXP1_AB:-0}" != "1" ]]; then
  echo "This launches paid matched A/B training. Set CONFIRM_EXP1_AB=1 only after explicit approval." >&2
  exit 1
fi
if [[ "$(git branch --show-current)" != "experiment-1-full-attnres" ]]; then
  echo "expected experiment-1-full-attnres branch" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree must be clean" >&2
  exit 1
fi

INIT_CHECKPOINT="${EXP1_INIT_CHECKPOINT:-experiment_artifacts/baseline_init_seed1337.pt}"
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "missing canonical initialization: $INIT_CHECKPOINT" >&2
  exit 1
fi
EXPERIMENT="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment"])' "$CONFIG")"

for MODE in standard full_attnres; do
  RUN_DIR="runs/${EXPERIMENT/exp1_/exp1_${MODE}_}"
  "$PYTHON_BIN" scripts/verify_environment.py \
    --require-cuda --require-a100-80gb --require-bf16 --require-sdpa --require-fused-adamw \
    --out /tmp/exp1_environment.json
  "$PYTHON_BIN" scripts/verify_dataset.py \
    --strict --min-total-tokens 9900000000 --max-total-tokens 10200000000 \
    --out /tmp/exp1_dataset.json
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/experiment_train.py \
    --config "$CONFIG" --residual-mode "$MODE" \
    --init-checkpoint "$INIT_CHECKPOINT" --run-dir "$RUN_DIR" \
    --environment-report /tmp/exp1_environment.json \
    --dataset-report /tmp/exp1_dataset.json \
    2>&1 | tee "/tmp/${EXPERIMENT}_${MODE}.console.log"
  cp "/tmp/${EXPERIMENT}_${MODE}.console.log" "$RUN_DIR/console.log"
done

"$PYTHON_BIN" scripts/plot_exp1.py \
  "runs/${EXPERIMENT/exp1_/exp1_standard_}" \
  "runs/${EXPERIMENT/exp1_/exp1_full_attnres_}" \
  --out-dir "runs/${EXPERIMENT}_comparison_plots"
