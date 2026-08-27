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
import hashlib
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
import hashlib
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
lane_name = os.environ["LANE_NAME"]
lane = lane_name.lower()
recovery_mode = os.environ.get("MASTER_RECOVERY_MODE") == "1"
error_path = run_dir / f"lane_{lane}.error.json"
success_path = run_dir / f"lane_{lane}.science_complete.json"
terminal_path = run_dir / f"lane_{lane}.terminal.json"
if recovery_mode:
    recovery_path = run_dir / "RECOVERY_PREFLIGHT.json"
    try:
        recovery = json.loads(recovery_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        problems.append(f"recovery preflight unavailable: {error}")
    else:
        lane_evidence = recovery.get("lane_evidence", {}).get(lane_name, {})
        if recovery.get("passed") is not True or recovery.get("run_id") != run_id:
            problems.append("recovery preflight is stale or not passing")
        if lane_name not in recovery.get("authorized_lanes", []):
            problems.append("lane is not authorized for recovery")
        if lane_evidence.get("strict_checkpoint_reopen_passed") is not True:
            problems.append("recovery base checkpoint was not strictly reopened")
        base_sha = lane_evidence.get("base_checkpoint_sha256")
        if not isinstance(base_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", base_sha):
            problems.append("recovery base checkpoint SHA is invalid")
    try:
        prior_error = json.loads(error_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        problems.append(f"exact prior failure marker unavailable: {error}")
    else:
        if prior_error.get("run_id") != run_id or prior_error.get("lane") != lane_name:
            problems.append("prior failure marker identity mismatch")
        if prior_error.get("status") != "HARD_FAILURE" or not prior_error.get("exit_code"):
            problems.append("prior failure marker is not a hard nonzero failure")
    if success_path.exists():
        problems.append("recovery refuses an existing science-complete marker")
    if terminal_path.exists():
        terminal_gate = recovery.get("original_terminal_recovery_gate", {})
        sealed = lane_evidence.get("original_terminal")
        try:
            terminal_bytes = terminal_path.read_bytes()
            terminal = json.loads(terminal_bytes)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"existing original terminal is unreadable: {error}")
        else:
            terminal_sha = hashlib.sha256(terminal_bytes).hexdigest()
            if (
                terminal_gate.get("passed") is not True
                or terminal_gate.get("explicit_cli_authorization") is not True
                or not isinstance(sealed, dict)
                or sealed.get("passed") is not True
                or sealed.get("path") != str(terminal_path)
                or sealed.get("sha256") != terminal_sha
                or terminal.get("run_id") != run_id
                or terminal.get("lane") != lane_name
                or terminal.get("status") != "HARD_FAILURE"
                or not isinstance(terminal.get("returncode"), int)
                or terminal["returncode"] == 0
            ):
                problems.append("existing original terminal is not exactly sealed for recovery")
else:
    for path in (error_path, success_path, terminal_path):
        if path.exists():
            problems.append(f"stale lane marker exists: {path.name}")
if problems:
    raise SystemExit("lane master gate failed: " + "; ".join(problems))
PY

  mkdir -p "$LANE_RUN_DIR"
  touch "$LANE_COMMAND_LOG"
  if [[ ${MASTER_RECOVERY_MODE:-0} == 1 ]]; then
    export LANE_RECOVERY_COMMAND_LOG="$LANE_RUN_DIR/lane_${LANE_NAME,,}.recovery_commands.jsonl"
    : > "$LANE_RECOVERY_COMMAND_LOG"
  fi
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
  if [[ ${MASTER_RECOVERY_MODE:-0} == 1 ]]; then
    LANE_RECOVERY_COMMAND="$quoted" "$LANE_PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["LANE_RECOVERY_COMMAND_LOG"])
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(os.environ["LANE_RECOVERY_COMMAND"]) + "\n")
PY
  fi
  lane_write_record status RUNNING "$phase" "" "command completed"
  unset LANE_ACTIVE_COMMAND
}

lane_mark_science_complete() {
  LANE_PHASE=SCIENCE_COMPLETE
  export LANE_PHASE
  unset LANE_ACTIVE_COMMAND || true
  local terminal_status=SCIENCE_COMPLETE_PENDING_GIT_AND_MASTER
  if [[ ${MASTER_RECOVERY_MODE:-0} == 1 ]]; then
    terminal_status=RECOVERABLE_FAILURE_RESUMED
  fi
  lane_write_record science_complete "$terminal_status" \
    "$LANE_PHASE" 0 "all scientific commands for this lane completed"
  if [[ ${MASTER_RECOVERY_MODE:-0} == 1 ]]; then
    "$LANE_PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

run_dir = Path(os.environ["LANE_RUN_DIR"])
lane = os.environ["LANE_NAME"]
lower = lane.lower()
error_path = run_dir / f"lane_{lower}.error.json"
success_path = run_dir / f"lane_{lower}.science_complete.json"
recovery = json.loads((run_dir / "RECOVERY_PREFLIGHT.json").read_text())
lane_evidence = recovery["lane_evidence"][lane]
commands = [json.loads(line) for line in Path(os.environ["LANE_RECOVERY_COMMAND_LOG"]).read_text().splitlines() if line]
if not commands:
    raise SystemExit("recovery completed without recorded resumed commands")
payload = json.loads(success_path.read_text())
payload["recovery_evidence"] = {
    "recovery_reason": recovery.get("recovery_reasons", {}).get(
        lane, lane_evidence.get("recovery_reason")
    ),
    "prior_failure_marker_sha256": hashlib.sha256(error_path.read_bytes()).hexdigest(),
    # The lane-specific recovery preflight seals the exact restart base.  This
    # may be a frozen update-zero source or an internal scientific staging
    # checkpoint such as 2D2G Stage-A-191.
    "resume_checkpoint_sha256": lane_evidence["base_checkpoint_sha256"],
    "resumed_command_records": commands,
    "strict_checkpoint_reopen_passed": lane_evidence["strict_checkpoint_reopen_passed"],
    "recovery_preflight": str(run_dir / "RECOVERY_PREFLIGHT.json"),
}
temporary = success_path.with_name(success_path.name + f".tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, success_path)
PY
  fi
  lane_write_record status "$terminal_status" \
    "$LANE_PHASE" 0 "lane shell will exit successfully"
}
