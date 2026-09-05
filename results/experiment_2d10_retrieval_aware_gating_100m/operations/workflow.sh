#!/usr/bin/env bash
set -Eeuo pipefail
RUN=/workspace/exp2d10_retrieval_gating_100m/run
REPO=/tmp/exp2d10-repo
PY=/tmp/exp2d10-venv/bin/python
exec 9>"$RUN/workflow.lock"
flock -n 9 || exit 88
test ! -e "$RUN/workflow.status"
trap 'code=$?; printf "%s\n" "$code" > "$RUN/workflow.status"' EXIT
cd "$REPO"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$RUN/panel" "$RUN/continuation" "$RUN/training" "$RUN/evaluation"
cp results/experiment_2d10_retrieval_aware_gating_100m/{EVALUATION_PANEL_MANIFEST,DISJOINTNESS_AUDIT}.json "$RUN/panel/"
cp results/experiment_2d9_token_conditioned_dynamic_recurrent_gating/{MATCHED_BATCH_LEDGER.jsonl,CONTINUATION_MANIFEST.json} "$RUN/continuation/"
worker() {
 local arm="$1" control="$2" gpu="$3"
 export CUDA_VISIBLE_DEVICES="$gpu"
 "$PY" scripts/experiment_2d10.py train --arm "$arm" --parent-checkpoint /workspace/exp2d7_boundary_alignment/run/checkpoints/O/scientific_cumulative_001200619520.pt --preflight-audit "$RUN/preflight/PREFLIGHT_AUDIT.json" --continuation-manifest "$RUN/continuation/CONTINUATION_MANIFEST.json" --continuation-ledger "$RUN/continuation/MATCHED_BATCH_LEDGER.jsonl" --checkpoint-dir "$RUN/checkpoints" --output-dir "$RUN/training/$arm" || return $?
 "$PY" scripts/experiment_2d10.py evaluate --condition "${arm}_REAL" --checkpoint "$RUN/checkpoints/$arm/scientific_cumulative_001300758528.pt" --panel-manifest "$RUN/panel/EVALUATION_PANEL_MANIFEST.json" --data-root /workspace/build-nanogpt/edu_fineweb10B --output-path "$RUN/evaluation/${arm}_REAL.json" || return $?
 "$PY" scripts/experiment_2d10.py evaluate --condition "${control}_REAL" --checkpoint "/workspace/exp2d9_dynamic_gating/run/checkpoints/$control/scientific_cumulative_001300758528.pt" --panel-manifest "$RUN/panel/EVALUATION_PANEL_MANIFEST.json" --data-root /workspace/build-nanogpt/edu_fineweb10B --output-path "$RUN/evaluation/${control}_REAL.json"
}
worker T S 0 > "$RUN/worker_T.log" 2>&1 & pid_t=$!
worker H D 1 > "$RUN/worker_H.log" 2>&1 & pid_h=$!
code_t=0; wait "$pid_t" || code_t=$?
code_h=0; wait "$pid_h" || code_h=$?
if ((code_t != 0 || code_h != 0)); then exit 1; fi
