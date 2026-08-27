#!/usr/bin/env bash
# Shared, run-scoped safety and provenance helpers for the four GPU lanes.
# This file is sourced by each lane after `set -Eeuo pipefail`.

MASTER_EXPECTED_POD_ID=7i2zyd53ytspwz
MASTER_EXPECTED_POD_NAME=empirical_tan_panda
MASTER_EXPECTED_VOLUME_ID=yhzyb27fb5
LANE_PYTHON_BIN=${PYTHON_BIN:-python}

lane_write_record() {
  local kind=$1
  local status=$2
  local phase=$3
  local exit_code=${4:-}
  local message=${5:-}
  LANE_RECORD_KIND="$kind" \
  LANE_RECORD_STATUS="$status" \
  LANE_RECORD_PHASE="$phase" \
  LANE_RECORD_EXIT_CODE="$exit_code" \
  LANE_RECORD_MESSAGE="$message" \
  LANE_RECORD_COMMAND="${LANE_ACTIVE_COMMAND:-}" \
  "$LANE_PYTHON_BIN" - <<'PY'
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

kind = os.environ["LANE_RECORD_KIND"]
lane = os.environ["LANE_NAME"]
suffix = {
    "status": "status.json",
    "error": "error.json",
    "science_complete": "science_complete.json",
}[kind]
path = Path(os.environ["LANE_RUN_DIR"]) / f"lane_{lane.lower()}.{suffix}"
exit_code = os.environ.get("LANE_RECORD_EXIT_CODE", "")
payload = {
    "schema_version": 1,
    "run_id": os.environ["MASTER_RUN_ID"],
    "lane": lane,
    "gpu_index": int(os.environ["LANE_GPU_INDEX"]),
    "experiments": os.environ["LANE_EXPERIMENTS"],
    "status": os.environ["LANE_RECORD_STATUS"],
    "phase": os.environ["LANE_RECORD_PHASE"],
    "shell_pid": int(os.environ["LANE_SHELL_PID"]),
    "process_group_id": int(os.environ["LANE_PGID"]),
    "updated_epoch": time.time(),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}
if exit_code:
    payload["exit_code"] = int(exit_code)
if os.environ.get("LANE_RECORD_MESSAGE"):
    payload["message"] = os.environ["LANE_RECORD_MESSAGE"]
if os.environ.get("LANE_RECORD_COMMAND"):
    payload["command"] = os.environ["LANE_RECORD_COMMAND"]
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
descriptor = os.open(path.parent, os.O_RDONLY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

lane_failure_trap() {
  local rc=$?
  # Disable the trap before doing any reporting.  Reporting failures must not
  # recursively enter this handler and conceal the original return code.
  trap - ERR
  set +e
  lane_write_record error HARD_FAILURE "${LANE_PHASE:-UNKNOWN}" "$rc" \
    "lane command failed"
  lane_write_record status HARD_FAILURE "${LANE_PHASE:-UNKNOWN}" "$rc" \
    "lane shell is exiting after an unrecovered error"
  exit "$rc"
}

lane_init() {
  local gpu_index=$1
  local experiments=$2
  if [[ -z ${MASTER_RUN_ID:-} ]]; then
    echo "MASTER_RUN_ID is required; lanes may only be launched by the run-scoped supervisor" >&2
    return 64
  fi
  export LANE_GPU_INDEX="$gpu_index"
  export LANE_NAME="GPU${gpu_index}"
  export LANE_EXPERIMENTS="$experiments"
  export LANE_SHELL_PID=$$
  export LANE_PGID
  LANE_PGID=$(ps -o pgid= -p "$$" | tr -d ' ')
  export LANE_RUN_DIR="$MASTER_ROOT/runs/$MASTER_RUN_ID"
  export LANE_LOG="$LANE_RUN_DIR/lane_gpu${gpu_index}.log"
  export LANE_COMMAND_LOG="$LANE_RUN_DIR/MASTER_COMMANDS.log"

  MASTER_EXPECTED_POD_ID="$MASTER_EXPECTED_POD_ID" \
  MASTER_EXPECTED_POD_NAME="$MASTER_EXPECTED_POD_NAME" \
  MASTER_EXPECTED_VOLUME_ID="$MASTER_EXPECTED_VOLUME_ID" \
  "$LANE_PYTHON_BIN" - <<'PY'
import json
import os
import re
from pathlib import Path

run_id = os.environ["MASTER_RUN_ID"]
if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", run_id):
    raise SystemExit("MASTER_RUN_ID is not a canonical UUID4")
root = Path(os.environ["MASTER_ROOT"]).resolve()
preflight_path = root / "MASTER_PREFLIGHT.json"
try:
    preflight = json.loads(preflight_path.read_text())
except (FileNotFoundError, json.JSONDecodeError) as error:
    raise SystemExit(f"exact MASTER_PREFLIGHT is unavailable: {error}")
expected_pod = {
    "id": os.environ["MASTER_EXPECTED_POD_ID"],
    "name": os.environ["MASTER_EXPECTED_POD_NAME"],
    "gpu_count": 4,
    "volume_id": os.environ["MASTER_EXPECTED_VOLUME_ID"],
}
expected_run_root = str((root / "runs" / run_id).resolve())
required_checks = {"hardware", "storage", "sources", "dataset", "git", "authenticated_stop"}
problems = []
if preflight.get("passed") is not True:
    problems.append("passed is not exactly true")
if preflight.get("run_id") != run_id:
    problems.append("run_id mismatch (stale preflight)")
if preflight.get("pod") != expected_pod:
    problems.append("pod/volume identity mismatch")
if preflight.get("run_root") != expected_run_root:
    problems.append("run_root mismatch")
checks = preflight.get("checks")
if not isinstance(checks, dict) or set(checks) != required_checks or any(checks[key] is not True for key in required_checks):
    problems.append("required preflight checks are not all exactly true")
run_dir = Path(expected_run_root)
if not run_dir.is_dir():
    problems.append("run directory is missing")
else:
    try:
        scoped_preflight = json.loads((run_dir / "MASTER_PREFLIGHT.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        problems.append(f"run-scoped MASTER_PREFLIGHT is unavailable: {error}")
    else:
        if scoped_preflight != preflight:
            problems.append("top-level and run-scoped preflight records differ")
lane = os.environ["LANE_NAME"].lower()
for suffix in ("error.json", "science_complete.json", "terminal.json"):
    if (run_dir / f"lane_{lane}.{suffix}").exists():
        problems.append(f"stale lane marker exists: lane_{lane}.{suffix}")
if problems:
    raise SystemExit("lane master gate failed: " + "; ".join(problems))
PY

  mkdir -p "$LANE_RUN_DIR"
  touch "$LANE_COMMAND_LOG"
  exec > >(tee -a "$LANE_LOG") 2>&1
  LANE_PHASE=MASTER_GATE
  export LANE_PHASE
  trap lane_failure_trap ERR
  lane_write_record status RUNNING "$LANE_PHASE" "" "run-scoped master preflight accepted"
}

log_command() {
  local phase=$1
  shift
  LANE_PHASE=$phase
  export LANE_PHASE
  local quoted=""
  local item escaped
  for item in "$@"; do
    printf -v escaped '%q' "$item"
    quoted+=" $escaped"
  done
  quoted=${quoted# }
  export LANE_ACTIVE_COMMAND="$quoted"
  lane_write_record status RUNNING "$phase" "" "command starting"

  local record
  printf -v record '%s run_id=%s lane=%s shell_pid=%s pgid=%s phase=%q command=%s' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MASTER_RUN_ID" "$LANE_NAME" \
    "$LANE_SHELL_PID" "$LANE_PGID" "$phase" "$quoted"
  # One complete record is appended while holding the shared exclusive lock.
  # No lane emits a sequence of separately interleavable writes.
  LANE_LOG_RECORD="$record" "$LANE_PYTHON_BIN" - <<'PY'
import fcntl
import os
from pathlib import Path

path = Path(os.environ["LANE_COMMAND_LOG"])
with path.open("a", encoding="utf-8") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    handle.write(os.environ["LANE_LOG_RECORD"] + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
PY

  "$@"
  lane_write_record status RUNNING "$phase" "" "command completed"
  unset LANE_ACTIVE_COMMAND
}

lane_mark_science_complete() {
  LANE_PHASE=SCIENCE_COMPLETE
  export LANE_PHASE
  unset LANE_ACTIVE_COMMAND || true
  lane_write_record science_complete SCIENCE_COMPLETE_PENDING_GIT_AND_MASTER \
    "$LANE_PHASE" 0 "all scientific commands for this lane completed"
  lane_write_record status SCIENCE_COMPLETE_PENDING_GIT_AND_MASTER \
    "$LANE_PHASE" 0 "lane shell will exit successfully"
}
