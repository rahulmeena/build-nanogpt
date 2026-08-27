#!/usr/bin/env python3
"""Durable master heartbeat for the four fixed GPU lanes.

The monitor is observational only: it never launches, restarts, or stops a
scientific process.  A lane supervisor owns those transitions.  This process
merges each experiment heartbeat with live NVIDIA telemetry and exits only
when the coordinator creates the terminal sentinel.
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
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        return {"available": False, "read_error": str(error)}


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
    except (ProcessLookupError, PermissionError):
        return {"pid": pid, "alive": False}
    return {"pid": pid, "alive": True}


def latest_lane_heartbeat(rows) -> dict:
    candidates = []
    for experiment, raw_path in rows:
        path = Path(raw_path)
        payload = read_json(path)
        mtime = path.stat().st_mtime if path.is_file() else -1.0
        candidates.append(
            {
                "experiment": experiment,
                "path": str(path),
                "mtime_epoch": mtime,
                "heartbeat": payload,
            }
        )
    latest = max(candidates, key=lambda row: row["mtime_epoch"])
    pid = latest["heartbeat"].get("pid")
    latest["process"] = pid_status(pid)
    return latest


def snapshot(master_root: Path) -> dict:
    workspace = shutil.disk_usage("/workspace")
    ephemeral = shutil.disk_usage("/tmp")
    errors = {}
    for path in sorted(master_root.glob("lane_*.error.json")):
        errors[path.name] = read_json(path)
    return {
        "heartbeat_utc": now_utc(),
        "heartbeat_local": datetime.now().astimezone().isoformat(),
        "monitor_pid": os.getpid(),
        "lanes": {
            lane: latest_lane_heartbeat(rows) for lane, rows in LANE_RESULTS.items()
        },
        "gpus": gpu_rows(),
        "storage": {
            "workspace_backend_free_bytes": workspace.free,
            "ephemeral_free_bytes": ephemeral.free,
            "workspace_backend_df_is_not_volume_quota": True,
        },
        "errors": errors,
    }


def run(args) -> None:
    master_root = Path(args.master_root).resolve()
    output = master_root / "MASTER_HEARTBEAT.json"
    sentinel = master_root / "MASTER_ALL_LANES_TERMINAL"
    while True:
        durable_json(output, snapshot(master_root))
        if sentinel.exists():
            final = read_json(output)
            final["monitor_status"] = "TERMINAL_SENTINEL_OBSERVED"
            final["terminal_utc"] = now_utc()
            durable_json(output, final)
            return
        time.sleep(int(args.interval_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", required=True)
    parser.add_argument("--interval-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 30 <= args.interval_seconds <= 600:
        raise SystemExit("heartbeat interval must be 30..600 seconds")
    run(args)


if __name__ == "__main__":
    main()
