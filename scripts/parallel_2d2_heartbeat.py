#!/usr/bin/env python3
"""Durable master heartbeat for the four fixed GPU lanes.

The monitor is observational only: it never launches, restarts, or stops a
scientific process.  A lane supervisor owns those transitions.  This process
merges each experiment heartbeat with live NVIDIA telemetry and exits only
when the coordinator creates the later, run-scoped finalization sentinel.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from parallel_2d2_supervisor import validate_master_preflight


LANE_RESULTS = {
    "GPU0": (
        ("2D2E-C1", "/workspace/parallel_2d2_master/worktrees/master/results/experiment_2d2e_c1_large_true_self_confirmation/HEARTBEAT.json"),
        ("2D2F", "/workspace/parallel_2d2_master/worktrees/2d2f/results/experiment_2d2f_no_b2_recurrence_b3_w64/HEARTBEAT.json"),
    ),
    "GPU1": (
        ("2D2G", "/workspace/parallel_2d2_master/worktrees/2d2g/results/experiment_2d2g_b2_full_b3_w64/HEARTBEAT.json"),
    ),
    "GPU2": (
        ("2D2H", "/workspace/parallel_2d2_master/worktrees/2d2h/results/experiment_2d2h_no_b1_recurrence_b2_w32/HEARTBEAT.json"),
    ),
    "GPU3": (
        ("2D2I", "/workspace/parallel_2d2_master/worktrees/2d2i/results/experiment_2d2i_b4_w128_b9_recurrent/HEARTBEAT.json"),
    ),
}
NETWORK_VOLUME_CAPACITY_BYTES = 100_000_000_000


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def durable_json(path: Path, payload) -> None:
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


def read_json(path: Path):
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        return {"available": False, "read_error": str(error)}
    if not isinstance(value, dict):
        return {"available": False, "read_error": "JSON root is not an object"}
    return value


def gpu_rows() -> list[dict]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,temperature.gpu",
        "--format=csv,noheader,nounits",
    )
    rows = []
    for line in subprocess.check_output(command, text=True).splitlines():
        fields = [field.strip() for field in line.split(",")]
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "utilization_percent": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "temperature_c": int(fields[5]),
            }
        )
    return rows


def pid_status(pid) -> dict:
    if not isinstance(pid, int) or pid < 1:
        return {"pid": pid, "alive": False}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"pid": pid, "alive": False}
    except PermissionError:
        return {"pid": pid, "alive": True, "inspection_permission_denied": True}
    return {"pid": pid, "alive": True}


def latest_lane_heartbeat(rows, run_started_epoch: float) -> dict:
    candidates = []
    for experiment, raw_path in rows:
        path = Path(raw_path)
        payload = read_json(path)
        try:
            mtime = path.stat().st_mtime if path.is_file() else -1.0
        except FileNotFoundError:
            mtime = -1.0
        candidates.append(
            {
                "experiment": experiment,
                "path": str(path),
                "mtime_epoch": mtime,
                "heartbeat": payload,
                "belongs_to_current_run_window": mtime >= run_started_epoch,
            }
        )
    latest = max(candidates, key=lambda row: row["mtime_epoch"])
    pid = latest["heartbeat"].get("pid")
    latest["process"] = (
        pid_status(pid)
        if latest["belongs_to_current_run_window"]
        else {"pid": pid, "alive": False, "ignored_as_stale": True}
    )
    return latest


def lane_observation(
    run_root: Path,
    run_id: str,
    lane: str,
    rows,
    run_started_epoch: float,
    stalled_seconds: int,
) -> dict:
    lower = lane.lower()
    status_path = run_root / f"lane_{lower}.status.json"
    terminal_path = run_root / f"lane_{lower}.terminal.json"
    error_path = run_root / f"lane_{lower}.error.json"
    status = read_json(status_path)
    terminal = read_json(terminal_path) if terminal_path.is_file() else None
    error = read_json(error_path) if error_path.is_file() else None
    valid_status = isinstance(status, dict) and status.get("run_id") == run_id
    valid_terminal = isinstance(terminal, dict) and terminal.get("run_id") == run_id
    valid_error = isinstance(error, dict) and error.get("run_id") == run_id
    experiment = latest_lane_heartbeat(rows, run_started_epoch)
    activity_epochs = [run_started_epoch]
    if valid_status and isinstance(status.get("updated_epoch"), (int, float)):
        activity_epochs.append(float(status["updated_epoch"]))
    if experiment["belongs_to_current_run_window"]:
        activity_epochs.append(float(experiment["mtime_epoch"]))
    lane_log = run_root / f"lane_{lower}.log"
    if lane_log.is_file():
        activity_epochs.append(lane_log.stat().st_mtime)
    latest_activity = max(activity_epochs)
    age = max(0.0, time.time() - latest_activity)
    shell_pid = status.get("shell_pid") if valid_status else None
    process = (
        {"pid": shell_pid, "alive": False, "terminal_record_present": True}
        if valid_terminal
        else pid_status(shell_pid)
    )
    normalized_status = (
        terminal.get("status")
        if valid_terminal
        else status.get("status")
        if valid_status
        else "WAITING_FOR_LANE_GATE"
    )
    active_statuses = {"RUNNING", "WAITING_FOR_LANE_GATE"}
    return {
        "run_id": run_id,
        "phase": status.get("phase") if valid_status else "NOT_STARTED",
        "status": normalized_status,
        "status_record": status if valid_status else {"available": False, "reason": "missing_or_stale"},
        "terminal_record": terminal if valid_terminal else None,
        "error_record": error if valid_error else None,
        "process": process,
        "process_group_id": status.get("process_group_id") if valid_status else None,
        "latest_activity_epoch": latest_activity,
        "activity_age_seconds": age,
        "stalled_threshold_seconds": stalled_seconds,
        "stalled": bool(process.get("alive") and normalized_status in active_statuses and age > stalled_seconds),
        "experiment_heartbeat": experiment,
    }


def logical_workspace_bytes() -> int:
    total = 0
    for base, _, names in os.walk("/workspace", followlinks=False):
        for name in names:
            path = Path(base) / name
            try:
                if not path.is_symlink():
                    total += path.lstat().st_size
            except (FileNotFoundError, PermissionError, OSError):
                # Atomic artifact replacement may race one observation.
                continue
    return total


def snapshot(
    master_root: Path,
    run_root: Path,
    run_id: str,
    run_started_epoch: float,
    stalled_seconds: int,
) -> dict:
    workspace = shutil.disk_usage("/workspace")
    ephemeral = shutil.disk_usage("/tmp")
    logical_used = logical_workspace_bytes()
    errors = {}
    for path in sorted(run_root.glob("lane_*.error.json")):
        errors[path.name] = read_json(path)
    master_status = read_json(master_root / "MASTER_STATUS.json")
    if master_status.get("run_id") != run_id:
        master_status = {"available": False, "reason": "missing_or_stale"}
    return {
        "schema_version": 1,
        "run_id": run_id,
        "heartbeat_utc": now_utc(),
        "heartbeat_local": datetime.now().astimezone().isoformat(),
        "monitor_pid": os.getpid(),
        "master_status": master_status,
        "lanes": {
            lane: lane_observation(
                run_root,
                run_id,
                lane,
                rows,
                run_started_epoch,
                stalled_seconds,
            )
            for lane, rows in LANE_RESULTS.items()
        },
        "gpus": gpu_rows(),
        "storage": {
            "workspace_backend_free_bytes": workspace.free,
            "workspace_logical_used_bytes": logical_used,
            "workspace_logical_quota_free_bytes": (
                NETWORK_VOLUME_CAPACITY_BYTES - logical_used
            ),
            "ephemeral_free_bytes": ephemeral.free,
            "workspace_backend_df_is_not_volume_quota": True,
        },
        "errors": errors,
        "observational_only": True,
        "process_control_actions": [],
        "pod_stop_automated": False,
    }


def run(args) -> None:
    master_root = Path(args.master_root).resolve()
    preflight, run_root = validate_master_preflight(master_root, args.run_id)
    run_started_epoch = datetime.fromisoformat(preflight["created_utc"]).timestamp()
    output = run_root / "MASTER_HEARTBEAT.json"
    current_output = master_root / "MASTER_HEARTBEAT.json"
    sentinel = run_root / "MASTER_FINALIZATION_COMPLETE"
    while True:
        try:
            current = snapshot(
                master_root,
                run_root,
                args.run_id,
                run_started_epoch,
                int(args.stalled_seconds),
            )
        except Exception as error:  # The observational monitor must survive telemetry faults.
            current = {
                "schema_version": 1,
                "run_id": args.run_id,
                "heartbeat_utc": now_utc(),
                "monitor_pid": os.getpid(),
                "monitor_status": "OBSERVATION_ERROR_RETRYING",
                "observation_error": f"{type(error).__name__}: {error}",
                "observational_only": True,
                "process_control_actions": [],
                "pod_stop_automated": False,
            }
        durable_json(output, current)
        durable_json(current_output, current)
        if sentinel.exists():
            finalization = read_json(sentinel)
            if (
                finalization.get("run_id") == args.run_id
                and finalization.get("status") == "MASTER_FINALIZATION_COMPLETE"
            ):
                final = read_json(output)
                final["monitor_status"] = "MASTER_FINALIZATION_COMPLETE_SENTINEL_OBSERVED"
                final["terminal_utc"] = now_utc()
                final["finalization_sentinel"] = finalization
                durable_json(output, final)
                durable_json(current_output, final)
                return
        time.sleep(int(args.interval_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument("--stalled-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 30 <= args.interval_seconds <= 600:
        raise SystemExit("heartbeat interval must be 30..600 seconds")
    if args.stalled_seconds < args.interval_seconds * 2:
        raise SystemExit("stalled threshold must be at least two heartbeat intervals")
    run(args)


if __name__ == "__main__":
    main()
