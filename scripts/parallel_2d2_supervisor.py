#!/usr/bin/env python3
"""Run-scoped supervisor for the fixed four-lane experiment matrix.

This supervisor launches and observes scientific lanes.  It deliberately has
no pod-stop, pod-delete, Git, reporting, or backup capability.  The heartbeat
is left alive after science so those later coordinator phases remain visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


POD_ID = "7i2zyd53ytspwz"
POD_NAME = "empirical_tan_panda"
VOLUME_ID = "yhzyb27fb5"
RUN_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
REQUIRED_CHECKS = {
    "hardware",
    "storage",
    "sources",
    "dataset",
    "git",
    "authenticated_stop",
}
LANES = {
    "GPU0": "parallel_2d2_lane0.sh",
    "GPU1": "parallel_2d2_lane1.sh",
    "GPU2": "parallel_2d2_lane2.sh",
    "GPU3": "parallel_2d2_lane3.sh",
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


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"cannot read exact JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


def validate_master_preflight(master_root: Path, run_id: str) -> tuple[dict, Path]:
    """Reject stale, partial, identity-mismatched, or merely truthy preflight."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RuntimeError("run_id is not a canonical UUID4")
    master_root = master_root.resolve()
    preflight = read_json(master_root / "MASTER_PREFLIGHT.json")
    run_root = (master_root / "runs" / run_id).resolve()
    expected_pod = {
        "id": POD_ID,
        "name": POD_NAME,
        "gpu_count": 4,
        "volume_id": VOLUME_ID,
    }
    problems = []
    if preflight.get("passed") is not True:
        problems.append("passed is not exactly true")
    if preflight.get("run_id") != run_id:
        problems.append("run_id mismatch: the preflight is stale")
    if preflight.get("pod") != expected_pod:
        problems.append("pod/name/GPU-count/volume identity mismatch")
    if preflight.get("run_root") != str(run_root):
        problems.append("run_root mismatch")
    checks = preflight.get("checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CHECKS:
        problems.append("preflight check set is not exact")
    elif any(checks[name] is not True for name in REQUIRED_CHECKS):
        problems.append("one or more preflight checks is not exactly true")
    if not run_root.is_dir():
        problems.append("run directory does not exist")
    else:
        try:
            scoped_preflight = read_json(run_root / "MASTER_PREFLIGHT.json")
        except RuntimeError as error:
            problems.append(str(error))
        else:
            if scoped_preflight != preflight:
                problems.append("top-level and run-scoped preflight records differ")
    if problems:
        raise RuntimeError("master launch gate failed: " + "; ".join(problems))
    return preflight, run_root


def validate_fresh_execution_scope(run_root: Path) -> None:
    stale = []
    for pattern in (
        "lane_gpu*.error.json",
        "lane_gpu*.science_complete.json",
        "lane_gpu*.terminal.json",
        "MASTER_SUPERVISOR.json",
        "SUPERVISOR_LAUNCH.lock",
        "MASTER_TERMINAL_STATUS.json",
        "MASTER_ALL_LANES_TERMINAL",
        "MASTER_FINALIZATION_COMPLETE",
    ):
        stale.extend(sorted(run_root.glob(pattern)))
    if stale:
        raise RuntimeError(
            "refusing stale or repeated execution scope: "
            + ", ".join(path.name for path in stale)
        )


def valid_marker(marker: dict, run_id: str, lane: str) -> bool:
    return marker.get("run_id") == run_id and marker.get("lane") == lane


def recovery_is_evidenced(
    marker: dict, error_marker: dict | None, error_marker_path: Path
) -> bool:
    """Require structured evidence before using the recovery terminal label."""
    evidence = marker.get("recovery_evidence")
    if not isinstance(evidence, dict) or error_marker is None:
        return False
    required = {
        "prior_failure_marker_sha256",
        "resume_checkpoint_sha256",
        "resumed_command_records",
        "strict_checkpoint_reopen_passed",
    }
    observed_failure_sha = hashlib.sha256(error_marker_path.read_bytes()).hexdigest()
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    return (
        set(evidence) >= required
        and error_marker.get("status") == "HARD_FAILURE"
        and isinstance(error_marker.get("exit_code"), int)
        and error_marker["exit_code"] != 0
        and evidence["prior_failure_marker_sha256"] == observed_failure_sha
        and isinstance(evidence["resume_checkpoint_sha256"], str)
        and sha_pattern.fullmatch(evidence["resume_checkpoint_sha256"]) is not None
        and isinstance(evidence["resumed_command_records"], list)
        and len(evidence["resumed_command_records"]) > 0
        and evidence["strict_checkpoint_reopen_passed"] is True
    )


def normalize_terminal(run_root: Path, run_id: str, lane: str, returncode: int) -> dict:
    success_path = run_root / f"lane_{lane.lower()}.science_complete.json"
    error_path = run_root / f"lane_{lane.lower()}.error.json"
    success = None
    error = None
    try:
        success = read_json(success_path)
    except RuntimeError:
        pass
    try:
        error = read_json(error_path)
    except RuntimeError:
        pass

    status = "HARD_FAILURE"
    rationale = []
    if returncode != 0:
        rationale.append(f"lane shell exited {returncode}")
    elif success is None or not valid_marker(success, run_id, lane):
        rationale.append("exact run-scoped science-complete marker is absent or invalid")
    elif (
        success.get("status") == "SCIENCE_COMPLETE_PENDING_GIT_AND_MASTER"
        and not error_path.exists()
    ):
        status = "SUCCESS"
        rationale.append("shell exited zero and exact science-complete marker passed")
    elif (
        success.get("status") == "RECOVERABLE_FAILURE_RESUMED"
        and error is not None
        and valid_marker(error, run_id, lane)
        and recovery_is_evidenced(success, error, error_path)
    ):
        status = "RECOVERABLE_FAILURE_RESUMED"
        rationale.append("zero exit plus structured prior-failure/resume/reopen evidence")
    else:
        rationale.append("success/recovery marker did not meet a normalized terminal contract")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "lane": lane,
        "returncode": returncode,
        "status": status,
        "rationale": rationale,
        "science_complete_marker": str(success_path),
        "error_marker": str(error_path) if error_path.exists() else None,
        "normalized_utc": now_utc(),
    }


def status_payload(run_id: str, run_root: Path, supervisor_pid: int) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_root": str(run_root),
        "pod": {"id": POD_ID, "name": POD_NAME, "gpu_count": 4, "volume_id": VOLUME_ID},
        "supervisor_pid": supervisor_pid,
        "supervisor_process_group_id": os.getpgrp(),
        "created_utc": now_utc(),
        "status": "LAUNCHING",
        "heartbeat": {},
        "lanes": {},
        "pod_stop_automated": False,
    }


def write_status(master_root: Path, run_root: Path, payload: dict) -> None:
    payload["updated_utc"] = now_utc()
    durable_json(run_root / "MASTER_SUPERVISOR.json", payload)
    # This top-level document is a current-run view, never an execution marker.
    durable_json(master_root / "MASTER_STATUS.json", payload)


def launch(args) -> int:
    master_root = Path(args.master_root).resolve()
    _, run_root = validate_master_preflight(master_root, args.run_id)
    validate_fresh_execution_scope(run_root)
    scripts_dir = Path(args.scripts_dir).resolve()
    heartbeat_script = scripts_dir / "parallel_2d2_heartbeat.py"
    lane_common_script = scripts_dir / "parallel_2d2_lane_common.sh"
    lane_scripts = {lane: scripts_dir / filename for lane, filename in LANES.items()}
    missing = [
        path
        for path in (heartbeat_script, lane_common_script, *lane_scripts.values())
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError("required launch script missing: " + ", ".join(map(str, missing)))
    launch_lock = run_root / "SUPERVISOR_LAUNCH.lock"
    try:
        descriptor = os.open(launch_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("this run_id already has a supervisor launch claim") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {"run_id": args.run_id, "supervisor_pid": os.getpid(), "created_utc": now_utc()},
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    state = status_payload(args.run_id, run_root, os.getpid())
    write_status(master_root, run_root, state)
    environment = os.environ.copy()
    environment["MASTER_RUN_ID"] = args.run_id
    environment["MASTER_ROOT"] = str(master_root)

    heartbeat_log = (run_root / "heartbeat.supervisor.log").open("ab", buffering=0)
    try:
        heartbeat = subprocess.Popen(
            [
                sys.executable,
                str(heartbeat_script),
                "--master-root",
                str(master_root),
                "--run-id",
                args.run_id,
                "--interval-seconds",
                str(args.heartbeat_interval_seconds),
            ],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=heartbeat_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as error:
        heartbeat_log.close()
        state["status"] = "HARD_FAILURE_BEFORE_SCIENCE_LAUNCH"
        state["heartbeat"] = {"status": "HARD_FAILURE", "launch_error": str(error)}
        write_status(master_root, run_root, state)
        raise RuntimeError(f"could not launch mandatory heartbeat: {error}") from error
    heartbeat_pgid = os.getpgid(heartbeat.pid)
    state["heartbeat"] = {
        "pid": heartbeat.pid,
        "process_group_id": heartbeat_pgid,
        "status": "RUNNING",
        "log": str(run_root / "heartbeat.supervisor.log"),
    }
    write_status(master_root, run_root, state)
    time.sleep(0.25)
    if heartbeat.poll() is not None:
        state["heartbeat"]["status"] = "HARD_FAILURE"
        state["heartbeat"]["returncode"] = heartbeat.returncode
        state["status"] = "HARD_FAILURE_BEFORE_SCIENCE_LAUNCH"
        write_status(master_root, run_root, state)
        heartbeat_log.close()
        raise RuntimeError("heartbeat failed its launch gate; no science lane was launched")

    processes: dict[str, subprocess.Popen] = {}
    log_handles = {}
    launch_errors = {}
    for lane, script in lane_scripts.items():
        bootstrap_log_path = run_root / f"lane_{lane.lower()}.supervisor.log"
        handle = bootstrap_log_path.open("ab", buffering=0)
        log_handles[lane] = handle
        try:
            process = subprocess.Popen(
                ["bash", str(script)],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            launch_errors[lane] = str(error)
            state["lanes"][lane] = {
                "script": str(script),
                "status": "HARD_FAILURE_TO_LAUNCH",
                "launch_error": str(error),
                "bootstrap_log": str(bootstrap_log_path),
            }
            handle.close()
            write_status(master_root, run_root, state)
            continue
        processes[lane] = process
        state["lanes"][lane] = {
            "script": str(script),
            "pid": process.pid,
            "process_group_id": os.getpgid(process.pid),
            "status": "RUNNING",
            "bootstrap_log": str(bootstrap_log_path),
        }
        write_status(master_root, run_root, state)

    state["status"] = "SCIENCE_RUNNING"
    write_status(master_root, run_root, state)
    remaining = set(processes)
    while remaining:
        for lane in sorted(tuple(remaining)):
            returncode = processes[lane].poll()
            if returncode is None:
                continue
            state["lanes"][lane]["shell_exited"] = True
            state["lanes"][lane]["returncode"] = returncode
            state["lanes"][lane]["status"] = "SHELL_EXITED_AWAITING_ALL_LANES"
            state["lanes"][lane]["shell_exit_utc"] = now_utc()
            remaining.remove(lane)
            write_status(master_root, run_root, state)
        if remaining:
            time.sleep(args.poll_seconds)

    # Only now, after every independent lane shell has exited, normalize and
    # publish terminal lane records and the all-lanes sentinel.
    terminals = {}
    for lane in LANES:
        process = processes.get(lane)
        if process is None:
            terminal = {
                "schema_version": 1,
                "run_id": args.run_id,
                "lane": lane,
                "returncode": None,
                "status": "HARD_FAILURE",
                "rationale": [f"lane shell failed to launch: {launch_errors[lane]}"],
                "normalized_utc": now_utc(),
            }
        else:
            terminal = normalize_terminal(run_root, args.run_id, lane, process.returncode)
        terminals[lane] = terminal
        durable_json(run_root / f"lane_{lane.lower()}.terminal.json", terminal)
        state["lanes"][lane].update(
            {"status": terminal["status"], "terminal_record": str(run_root / f"lane_{lane.lower()}.terminal.json")}
        )
    heartbeat_returncode = heartbeat.poll()
    if heartbeat_returncode is not None:
        state["heartbeat"].update(status="HARD_FAILURE", returncode=heartbeat_returncode)
    else:
        state["heartbeat"]["status"] = "RUNNING_THROUGH_MASTER_FINALIZATION"
    acceptable = {"SUCCESS", "RECOVERABLE_FAILURE_RESUMED"}
    overall = (
        "SUCCESS"
        if all(row["status"] in acceptable for row in terminals.values())
        and heartbeat_returncode is None
        else "HARD_FAILURE"
    )
    all_four_shells_exited = len(processes) == 4 and all(
        process.returncode is not None for process in processes.values()
    )
    terminal = {
        "schema_version": 1,
        "run_id": args.run_id,
        "pod": {"id": POD_ID, "name": POD_NAME, "gpu_count": 4, "volume_id": VOLUME_ID},
        "all_four_lane_shells_exited": all_four_shells_exited,
        "all_lanes_terminal": True,
        "status": overall,
        "lanes": terminals,
        "heartbeat_pid": heartbeat.pid,
        "heartbeat_process_group_id": heartbeat_pgid,
        "heartbeat_left_running_for_finalization": heartbeat_returncode is None,
        "pod_stop_automated": False,
        "created_utc": now_utc(),
    }
    durable_json(run_root / "MASTER_TERMINAL_STATUS.json", terminal)
    if all_four_shells_exited:
        durable_json(run_root / "MASTER_ALL_LANES_TERMINAL", terminal)
        # Current-run compatibility pointers retain the run_id and are
        # intentionally rejected as stale by the next master preflight.
        durable_json(master_root / "MASTER_TERMINAL_STATUS.json", terminal)
        durable_json(master_root / "MASTER_ALL_LANES_TERMINAL", terminal)
    state["status"] = "ALL_LANES_TERMINAL_PENDING_MASTER_FINALIZATION" if overall == "SUCCESS" else "HARD_FAILURE"
    state["terminal_record"] = str(run_root / "MASTER_TERMINAL_STATUS.json")
    state["all_lanes_terminal_sentinel"] = (
        str(run_root / "MASTER_ALL_LANES_TERMINAL") if all_four_shells_exited else None
    )
    write_status(master_root, run_root, state)
    for handle in log_handles.values():
        handle.close()
    heartbeat_log.close()
    return 0 if overall == "SUCCESS" else 1


def mark_finalization_complete(args) -> int:
    master_root = Path(args.master_root).resolve()
    _, run_root = validate_master_preflight(master_root, args.run_id)
    terminal = read_json(run_root / "MASTER_ALL_LANES_TERMINAL")
    if terminal.get("run_id") != args.run_id or terminal.get("all_four_lane_shells_exited") is not True:
        raise RuntimeError("cannot finalize before the exact run has four exited lane shells")
    evidence_values = (args.git_evidence, args.report_evidence, args.backup_evidence)
    if any(not isinstance(value, str) or not value.strip() for value in evidence_values):
        raise RuntimeError("Git, report, and backup finalization evidence are all required")
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "MASTER_FINALIZATION_COMPLETE",
        "created_utc": now_utc(),
        "evidence": {
            "git": args.git_evidence,
            "report": args.report_evidence,
            "backup": args.backup_evidence,
        },
        "pod_stop_automated": False,
    }
    durable_json(run_root / "MASTER_FINALIZATION_COMPLETE", payload)
    durable_json(master_root / "MASTER_FINALIZATION_COMPLETE", payload)
    print(str(run_root / "MASTER_FINALIZATION_COMPLETE"), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    current = subparsers.add_parser("launch")
    current.add_argument("--master-root", required=True)
    current.add_argument("--run-id", required=True)
    current.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parent))
    current.add_argument("--heartbeat-interval-seconds", type=int, default=120)
    current.add_argument("--poll-seconds", type=float, default=2.0)
    current.set_defaults(function=launch)
    current = subparsers.add_parser("mark-finalization-complete")
    current.add_argument("--master-root", required=True)
    current.add_argument("--run-id", required=True)
    current.add_argument("--git-evidence", required=True)
    current.add_argument("--report-evidence", required=True)
    current.add_argument("--backup-evidence", required=True)
    current.set_defaults(function=mark_finalization_complete)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "heartbeat_interval_seconds", 120) < 30:
        raise SystemExit("heartbeat interval must be at least 30 seconds")
    if getattr(args, "poll_seconds", 2.0) <= 0:
        raise SystemExit("poll interval must be positive")
    try:
        return args.function(args)
    except RuntimeError as error:
        print(f"PARALLEL_2D2_SUPERVISOR_HARD_FAILURE: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
