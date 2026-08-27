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
EXPECTED_POD = {
    "id": POD_ID,
    "name": POD_NAME,
    "gpu_count": 4,
    "volume_id": VOLUME_ID,
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MASTER_COMMAND_PATTERN = re.compile(
    r"^(?P<timestamp>\S+) run_id=(?P<run_id>\S+) lane=(?P<lane>GPU[0-3]) "
    r"shell_pid=(?P<shell_pid>[1-9][0-9]*) pgid=(?P<pgid>[1-9][0-9]*) "
    r"phase=(?P<phase>.*?) command=(?P<command>.*)$"
)
FINAL_REPORTS = {
    "2D2E-C1": "/workspace/parallel_2d2_master/worktrees/master/results/experiment_2d2e_c1_large_true_self_confirmation/C1_FINAL_REPORT.md",
    "2D2F": "/workspace/parallel_2d2_master/worktrees/2d2f/results/experiment_2d2f_no_b2_recurrence_b3_w64/EXPERIMENT_2D2F_FINAL_REPORT.md",
    "2D2G": "/workspace/parallel_2d2_master/worktrees/2d2g/results/experiment_2d2g_b2_full_b3_w64/FINAL_REPORT.md",
    "2D2H": "/workspace/parallel_2d2_master/worktrees/2d2h/results/experiment_2d2h_no_b1_recurrence_b2_w32/EXPERIMENT_2D2H_FINAL_REPORT.md",
    "2D2I": "/workspace/parallel_2d2_master/worktrees/2d2i/results/experiment_2d2i_b4_w128_b9_recurrent/EXPERIMENT_2D2I_FINAL_REPORT.md",
}
FINAL_AUDITS = {
    experiment: str(Path(path).with_name("FINAL_AUDIT.json"))
    for experiment, path in FINAL_REPORTS.items()
}
FINAL_CHECKPOINTS = {
    "2D2F": "/workspace/exp2d2f_run/checkpoints/scientific_update_0191.pt",
    "2D2G": "/workspace/exp2d2g_run/checkpoints/stage_b_scientific_update_0191.pt",
    "2D2H": "/workspace/exp2d2h_run/checkpoints/scientific_update_0191.pt",
    "2D2I": "/workspace/exp2d2i_run/checkpoints/scientific_update_0191.pt",
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


def durable_bytes_exclusive(path: Path, content: bytes) -> None:
    """Atomically create an immutable evidence file without replacing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        raise RuntimeError(f"cannot create immutable evidence temporary {temporary}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"refusing to replace immutable evidence file {path}") from error
        except OSError as error:
            raise RuntimeError(f"cannot publish immutable evidence file {path}: {error}") from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def durable_json_exclusive(path: Path, payload) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    durable_bytes_exclusive(path, content)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"cannot read exact JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeError(f"cannot hash required artifact {path}: {error}") from error
    return digest.hexdigest()


def process_is_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_group_is_alive(process_group_id: int) -> bool:
    if not isinstance(process_group_id, int) or process_group_id <= 0:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate_master_preflight(master_root: Path, run_id: str) -> tuple[dict, Path]:
    """Reject stale, partial, identity-mismatched, or merely truthy preflight."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RuntimeError("run_id is not a canonical UUID4")
    master_root = master_root.resolve()
    preflight = read_json(master_root / "MASTER_PREFLIGHT.json")
    run_root = (master_root / "runs" / run_id).resolve()
    problems = []
    if preflight.get("passed") is not True:
        problems.append("passed is not exactly true")
    if preflight.get("run_id") != run_id:
        problems.append("run_id mismatch: the preflight is stale")
    if preflight.get("pod") != EXPECTED_POD:
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
    return (
        set(evidence) >= required
        and error_marker.get("status") == "HARD_FAILURE"
        and isinstance(error_marker.get("exit_code"), int)
        and error_marker["exit_code"] != 0
        and evidence["prior_failure_marker_sha256"] == observed_failure_sha
        and isinstance(evidence["resume_checkpoint_sha256"], str)
        and SHA256_PATTERN.fullmatch(evidence["resume_checkpoint_sha256"]) is not None
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


def original_supervisor_path(path: Path) -> Path:
    if path.suffix == ".json":
        return path.with_name(path.stem + ".original_supervisor.json")
    return path.with_name(path.name + ".original_supervisor")


def supervisor_artifact_paths(run_root: Path, recovered_lanes: list[str]) -> dict[str, Path]:
    return {
        "MASTER_TERMINAL_STATUS": run_root / "MASTER_TERMINAL_STATUS.json",
        "MASTER_ALL_LANES_TERMINAL": run_root / "MASTER_ALL_LANES_TERMINAL",
        "MASTER_SUPERVISOR": run_root / "MASTER_SUPERVISOR.json",
        **{
            f"{lane}_TERMINAL": run_root / f"lane_{lane.lower()}.terminal.json"
            for lane in recovered_lanes
        },
    }


def read_original_supervisor_artifact(path: Path) -> tuple[Path, bytes, dict]:
    preserved = original_supervisor_path(path)
    source = preserved if preserved.exists() else path
    try:
        content = source.read_bytes()
        payload = json.loads(content)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read original supervisor artifact {source}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"original supervisor artifact is not an object: {source}")
    return source, content, payload


def preserve_original_supervisor_artifact(path: Path, content: bytes) -> dict:
    preserved = original_supervisor_path(path)
    if preserved.exists():
        try:
            observed = preserved.read_bytes()
        except OSError as error:
            raise RuntimeError(f"cannot reread preserved supervisor artifact {preserved}: {error}") from error
        if observed != content:
            raise RuntimeError(f"preserved supervisor artifact changed: {preserved}")
    else:
        durable_bytes_exclusive(preserved, content)
    return {"path": str(preserved), "sha256": hashlib.sha256(content).hexdigest()}


def load_recovery_plan(path: Path, run_id: str, recovered_lanes: list[str]) -> tuple[dict, bytes]:
    try:
        content = path.read_bytes()
        plan = json.loads(content)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read exact recovery command plan {path}: {error}") from error
    if not isinstance(plan, dict):
        raise RuntimeError("recovery command plan must be a JSON object")
    rows = plan.get("recovered_lanes")
    problems = []
    if plan.get("schema_version") != 1:
        problems.append("schema_version must be exactly 1")
    if plan.get("run_id") != run_id:
        problems.append("run_id mismatch")
    if not isinstance(rows, dict) or set(rows) != set(recovered_lanes):
        problems.append("recovered_lanes does not exactly match the explicit CLI lane set")
    else:
        for lane in recovered_lanes:
            row = rows[lane]
            commands = row.get("expected_resumed_command_records") if isinstance(row, dict) else None
            if (
                not isinstance(commands, list)
                or not commands
                or any(not isinstance(command, str) or not command for command in commands)
            ):
                problems.append(f"{lane} expected command sequence is absent or invalid")
            reason = row.get("recovery_reason") if isinstance(row, dict) else None
            if not isinstance(reason, str) or not reason:
                problems.append(f"{lane} recovery reason is absent or invalid")
            schema = row.get("recovery_evidence_schema") if isinstance(row, dict) else None
            if schema not in {
                "v2_with_recovery_reason",
                "legacy_v1_without_recovery_reason",
            }:
                problems.append(f"{lane} recovery evidence schema is absent or invalid")
    if problems:
        raise RuntimeError("recovery command plan failed: " + "; ".join(problems))
    return plan, content


def nvidia_compute_processes() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot audit live NVIDIA compute processes: {error}") from error
    return [line.strip() for line in output.splitlines() if line.strip()]


def validate_final_science_artifacts() -> dict:
    problems = []
    reports = {}
    for experiment, value in FINAL_REPORTS.items():
        path = Path(value)
        try:
            size = path.stat().st_size
        except OSError as error:
            problems.append(f"{experiment} final report is unavailable: {error}")
            continue
        if not path.is_file() or size <= 0:
            problems.append(f"{experiment} final report is not a nonempty regular file")
            continue
        reports[experiment] = {
            "path": str(path),
            "size": size,
            "sha256": file_sha256(path),
        }
    checkpoints = {}
    for experiment, value in FINAL_CHECKPOINTS.items():
        path = Path(value)
        observed_sha = file_sha256(path)
        sha_path = path.with_suffix(path.suffix + ".sha256")
        verification_path = path.with_suffix(path.suffix + ".verification.json")
        try:
            sidecar_sha = sha_path.read_text(encoding="utf-8").split()[0]
        except (OSError, IndexError) as error:
            problems.append(f"{experiment} final checkpoint SHA sidecar failed: {error}")
            sidecar_sha = None
        try:
            verification = read_json(verification_path)
        except RuntimeError as error:
            problems.append(str(error))
            verification = {}
        if (
            not isinstance(sidecar_sha, str)
            or SHA256_PATTERN.fullmatch(sidecar_sha) is None
            or observed_sha != sidecar_sha
        ):
            problems.append(f"{experiment} final checkpoint fresh/sidecar SHA mismatch")
        if verification.get("passed") is not True:
            problems.append(f"{experiment} final checkpoint verification is not passing")
        checkpoints[experiment] = {
            "path": str(path),
            "sha256": observed_sha,
            "sha256_sidecar": str(sha_path),
            "verification": str(verification_path),
        }
    audits = {}
    for experiment, value in FINAL_AUDITS.items():
        path = Path(value)
        try:
            size = path.stat().st_size
        except OSError as error:
            raise RuntimeError(f"{experiment} FINAL_AUDIT is unavailable: {error}") from error
        if not path.is_file() or size <= 0:
            problems.append(f"{experiment} FINAL_AUDIT is not a nonempty regular file")
        audit = read_json(path)
        if audit.get("passed") is not True:
            problems.append(f"{experiment} FINAL_AUDIT is not passing")
        audits[experiment] = {
            "path": str(path),
            "size": size,
            "sha256": file_sha256(path),
        }
    active_compute = nvidia_compute_processes()
    if active_compute:
        problems.append("NVIDIA compute processes remain after claimed science completion")
    if problems:
        raise RuntimeError("final science artifact audit failed: " + "; ".join(problems))
    return {
        "final_reports": reports,
        "final_audits": audits,
        "final_checkpoints": checkpoints,
        "nvidia_compute_processes": active_compute,
        "all_gpus_compute_idle": True,
    }


def parse_recovery_command_log(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot read recovery command log {path}: {error}") from error
    commands = []
    for number, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid recovery command JSONL at {path}:{number}: {error}") from error
        if not isinstance(command, str) or not command:
            raise RuntimeError(f"invalid recovery command record at {path}:{number}")
        commands.append(command)
    return commands


def parse_master_command_log(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot read shared master command log {path}: {error}") from error
    records = []
    for number, line in enumerate(lines, 1):
        if not line:
            continue
        match = MASTER_COMMAND_PATTERN.fullmatch(line)
        if match is None:
            raise RuntimeError(f"malformed whole-record master command log at {path}:{number}")
        row = match.groupdict()
        row["shell_pid"] = int(row["shell_pid"])
        row["pgid"] = int(row["pgid"])
        row["line_number"] = number
        records.append(row)
    return records


def validate_original_supervisor_terminal(
    master_root: Path, run_root: Path, run_id: str, recovered_lanes: list[str]
) -> tuple[dict, dict[str, tuple[Path, bytes, dict]]]:
    paths = supervisor_artifact_paths(run_root, recovered_lanes)
    records = {name: read_original_supervisor_artifact(path) for name, path in paths.items()}
    terminal = records["MASTER_TERMINAL_STATUS"][2]
    sentinel = records["MASTER_ALL_LANES_TERMINAL"][2]
    supervisor_state = records["MASTER_SUPERVISOR"][2]
    problems = []
    if terminal != sentinel:
        problems.append("original master terminal and all-lanes sentinel differ")
    if terminal.get("run_id") != run_id or terminal.get("pod") != EXPECTED_POD:
        problems.append("original master terminal run/pod identity mismatch")
    if terminal.get("status") != "HARD_FAILURE":
        problems.append("reconciliation requires the original automatic HARD_FAILURE state")
    if terminal.get("all_lanes_terminal") is not True:
        problems.append("original supervisor did not mark all lanes terminal")
    if terminal.get("all_four_lane_shells_exited") is not True:
        problems.append("the four original lane shells did not all exit")
    if records["MASTER_TERMINAL_STATUS"][0] == paths["MASTER_TERMINAL_STATUS"]:
        try:
            top_terminal_bytes = (master_root / "MASTER_TERMINAL_STATUS.json").read_bytes()
            top_sentinel_bytes = (master_root / "MASTER_ALL_LANES_TERMINAL").read_bytes()
            top_status_bytes = (master_root / "MASTER_STATUS.json").read_bytes()
        except OSError as error:
            problems.append(f"cannot read top-level original supervisor pointers: {error}")
        else:
            if top_terminal_bytes != records["MASTER_TERMINAL_STATUS"][1]:
                problems.append("top-level and run-scoped original terminal bytes differ")
            if top_sentinel_bytes != records["MASTER_ALL_LANES_TERMINAL"][1]:
                problems.append("top-level and run-scoped original sentinel bytes differ")
            if top_status_bytes != records["MASTER_SUPERVISOR"][1]:
                problems.append("top-level and run-scoped original supervisor status differ")
    if supervisor_state.get("run_id") != run_id or supervisor_state.get("status") != "HARD_FAILURE":
        problems.append("original current supervisor status is not the exact hard failure")
    if not isinstance(supervisor_state.get("lanes"), dict) or set(
        supervisor_state["lanes"]
    ) != set(LANES):
        problems.append("original current supervisor lane set is not exact")
    lanes = terminal.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != set(LANES):
        problems.append("original terminal lane set is not exact")
        lanes = {}
    for lane in LANES:
        embedded = lanes.get(lane)
        lane_record = embedded if isinstance(embedded, dict) else {}
        if lane in recovered_lanes:
            standalone = records[f"{lane}_TERMINAL"][2]
            if embedded != standalone:
                problems.append(f"{lane} original embedded and standalone terminals differ")
                continue
            returncode = lane_record.get("returncode")
            if (
                lane_record.get("run_id") != run_id
                or lane_record.get("lane") != lane
                or lane_record.get("status") != "HARD_FAILURE"
                or not isinstance(returncode, int)
                or returncode == 0
                or lane_record.get("rationale") != [f"lane shell exited {returncode}"]
            ):
                problems.append(f"{lane} is not a solely shell-exit original hard failure")
        elif (
            lane_record.get("run_id") != run_id
            or lane_record.get("lane") != lane
            or lane_record.get("status") != "SUCCESS"
            or lane_record.get("returncode") != 0
        ):
            problems.append(f"unrecovered {lane} was not an original successful lane")
    heartbeat_pid = terminal.get("heartbeat_pid")
    heartbeat_pgid = terminal.get("heartbeat_process_group_id")
    if terminal.get("heartbeat_left_running_for_finalization") is not True:
        problems.append("original supervisor did not leave the heartbeat running")
    if not process_is_alive(heartbeat_pid) or not process_group_is_alive(heartbeat_pgid):
        problems.append("the original supervisor heartbeat PID/PGID is not alive")
    if problems:
        raise RuntimeError("original supervisor terminal is not reconcilable: " + "; ".join(problems))
    return terminal, records


def validate_recovery_lane(
    run_root: Path,
    run_id: str,
    lane: str,
    preflight: dict,
    expected_commands: list[str],
    expected_reason: str,
    evidence_schema: str,
    master_commands: list[dict],
) -> dict:
    lower = lane.lower()
    success_path = run_root / f"lane_{lower}.science_complete.json"
    status_path = run_root / f"lane_{lower}.status.json"
    error_path = run_root / f"lane_{lower}.error.json"
    command_path = run_root / f"lane_{lower}.recovery_commands.jsonl"
    success = read_json(success_path)
    status = read_json(status_path)
    error = read_json(error_path)
    lane_preflight_rows = preflight.get("lane_evidence")
    lane_preflight = (
        lane_preflight_rows.get(lane) if isinstance(lane_preflight_rows, dict) else None
    )
    if not isinstance(lane_preflight, dict):
        raise RuntimeError(f"{lane} is absent from recovery preflight lane evidence")
    evidence = success.get("recovery_evidence")
    required_evidence = {
        "prior_failure_marker_sha256",
        "resume_checkpoint_sha256",
        "resumed_command_records",
        "strict_checkpoint_reopen_passed",
        "recovery_preflight",
    }
    if evidence_schema == "v2_with_recovery_reason":
        required_evidence.add("recovery_reason")
    problems = []
    for name, marker in (("science-complete", success), ("status", status)):
        if marker.get("run_id") != run_id or marker.get("lane") != lane:
            problems.append(f"{name} marker identity mismatch")
        if marker.get("status") != "RECOVERABLE_FAILURE_RESUMED":
            problems.append(f"{name} marker is not RECOVERABLE_FAILURE_RESUMED")
        if marker.get("exit_code") != 0 or marker.get("phase") != "SCIENCE_COMPLETE":
            problems.append(f"{name} marker lacks zero-exit SCIENCE_COMPLETE evidence")
    shell_pid = success.get("shell_pid")
    shell_pgid = success.get("process_group_id")
    if status.get("shell_pid") != shell_pid or status.get("process_group_id") != shell_pgid:
        problems.append("science-complete and status shell PID/PGID differ")
    if not isinstance(shell_pid, int) or shell_pid <= 0 or not isinstance(shell_pgid, int) or shell_pgid <= 0:
        problems.append("effective recovery shell PID/PGID is invalid")
    elif process_is_alive(shell_pid) or process_group_is_alive(shell_pgid):
        problems.append("effective recovery shell PID or process group is still alive")
    if (
        error.get("run_id") != run_id
        or error.get("lane") != lane
        or error.get("status") != "HARD_FAILURE"
        or not isinstance(error.get("exit_code"), int)
        or error["exit_code"] == 0
    ):
        problems.append("current prior-error marker is not an exact hard nonzero failure")
    error_sha = file_sha256(error_path)
    if not isinstance(evidence, dict) or set(evidence) != required_evidence:
        problems.append("structured recovery evidence field set is not exact")
        evidence = {}
    if evidence.get("prior_failure_marker_sha256") != error_sha:
        problems.append("science marker prior-failure SHA does not match current error bytes")
    if lane_preflight.get("prior_failure_marker_sha256") != error_sha:
        problems.append("recovery preflight prior-failure SHA does not match current error bytes")
    preflight_reasons = preflight.get("recovery_reasons")
    preflight_reason = (
        preflight_reasons.get(lane) if isinstance(preflight_reasons, dict) else None
    )
    if preflight_reason != expected_reason or lane_preflight.get("recovery_reason") != expected_reason:
        problems.append("recovery reason differs across success, preflight, and command plan")
    if (
        evidence_schema == "v2_with_recovery_reason"
        and evidence.get("recovery_reason") != expected_reason
    ):
        problems.append("fresh recovery success marker lacks its exact audited reason")
    recovery_preflight_path = run_root / "RECOVERY_PREFLIGHT.json"
    if evidence.get("recovery_preflight") != str(recovery_preflight_path):
        problems.append("science marker recovery-preflight path is not exact")
    if evidence.get("strict_checkpoint_reopen_passed") is not True:
        problems.append("science marker strict checkpoint reopen is not exactly true")
    if lane_preflight.get("strict_checkpoint_reopen_passed") is not True:
        problems.append("recovery preflight strict checkpoint reopen is not exactly true")
    checkpoint_value = lane_preflight.get("base_checkpoint")
    checkpoint_sha = lane_preflight.get("base_checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not Path(checkpoint_value).is_absolute():
        problems.append("recovery base checkpoint path is not absolute")
        checkpoint = None
    else:
        checkpoint = Path(checkpoint_value)
    if not isinstance(checkpoint_sha, str) or SHA256_PATTERN.fullmatch(checkpoint_sha) is None:
        problems.append("recovery base checkpoint SHA is invalid")
    if evidence.get("resume_checkpoint_sha256") != checkpoint_sha:
        problems.append("science marker resume SHA differs from recovery preflight base SHA")
    observed_checkpoint_sha = None
    verification_path = None
    sha_path = None
    if checkpoint is not None:
        observed_checkpoint_sha = file_sha256(checkpoint)
        sha_path = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
        verification_path = checkpoint.with_suffix(checkpoint.suffix + ".verification.json")
        try:
            sidecar_sha = sha_path.read_text(encoding="utf-8").split()[0]
        except (OSError, IndexError) as error_value:
            problems.append(f"cannot read base-checkpoint SHA sidecar: {error_value}")
            sidecar_sha = None
        try:
            verification = read_json(verification_path)
        except RuntimeError as error_value:
            problems.append(str(error_value))
            verification = {}
        if observed_checkpoint_sha != checkpoint_sha:
            problems.append("fresh base-checkpoint SHA does not match recovery preflight")
        if sidecar_sha != checkpoint_sha:
            problems.append("base-checkpoint SHA sidecar does not match recovery preflight")
        if verification.get("passed") is not True:
            problems.append("base-checkpoint strict verification sidecar is not passing")
    resumed_commands = parse_recovery_command_log(command_path)
    if evidence.get("resumed_command_records") != resumed_commands:
        problems.append("science marker commands differ from recovery JSONL")
    if resumed_commands != expected_commands:
        problems.append("recovery JSONL differs from the independent expected command sequence")
    relevant_master = [
        row
        for row in master_commands
        if row["run_id"] == run_id
        and row["lane"] == lane
        and row["shell_pid"] == shell_pid
        and row["pgid"] == shell_pgid
    ]
    if [row["command"] for row in relevant_master] != expected_commands:
        problems.append("flocked master command records differ from expected recovery sequence")
    if problems:
        raise RuntimeError(f"{lane} recovery evidence failed: " + "; ".join(problems))
    return {
        "status": "RECOVERABLE_FAILURE_RESUMED",
        "recovery_reason": expected_reason,
        "recovery_evidence_schema": evidence_schema,
        "effective_shell_pid": shell_pid,
        "effective_shell_process_group_id": shell_pgid,
        "effective_shell_absent": True,
        "os_wait_status_available": False,
        "zero_exit_compensating_evidence": [
            "atomic science-complete marker has exit_code 0",
            "atomic final lane-status marker has exit_code 0",
            "recovery shell PID and process group are absent",
            "all planned commands appear in both recovery JSONL and flocked master log",
        ],
        "error_marker": str(error_path),
        "error_marker_sha256": error_sha,
        "science_complete_marker": str(success_path),
        "science_complete_marker_sha256": file_sha256(success_path),
        "status_marker": str(status_path),
        "status_marker_sha256": file_sha256(status_path),
        "recovery_command_log": str(command_path),
        "recovery_command_log_sha256": file_sha256(command_path),
        "expected_resumed_command_records": expected_commands,
        "master_command_line_numbers": [row["line_number"] for row in relevant_master],
        "master_command_phases": [row["phase"] for row in relevant_master],
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": observed_checkpoint_sha,
        "base_checkpoint_sha_sidecar": str(sha_path),
        "base_checkpoint_verification": str(verification_path),
        "strict_checkpoint_reopen_passed": True,
    }


def reconcile_recovery(args) -> int:
    master_root = Path(args.master_root).resolve()
    _, run_root = validate_master_preflight(master_root, args.run_id)
    if (run_root / "MASTER_FINALIZATION_COMPLETE").exists():
        raise RuntimeError("cannot reconcile after master finalization is complete")
    recovered_lanes = list(args.recovered_lane)
    if (
        not recovered_lanes
        or len(recovered_lanes) != len(set(recovered_lanes))
        or any(lane not in LANES for lane in recovered_lanes)
    ):
        raise RuntimeError("recovered lanes must be a nonempty, duplicate-free subset of GPU0..GPU3")
    recovered_lanes.sort()
    plan_path = Path(args.recovery_plan).resolve()
    plan, plan_bytes = load_recovery_plan(plan_path, args.run_id, recovered_lanes)
    original_terminal, original_records = validate_original_supervisor_terminal(
        master_root, run_root, args.run_id, recovered_lanes
    )
    recovery_preflight_path = run_root / "RECOVERY_PREFLIGHT.json"
    recovery_preflight = read_json(recovery_preflight_path)
    authorized = recovery_preflight.get("authorized_lanes")
    checks = recovery_preflight.get("checks")
    plan_metadata = recovery_preflight.get("recovery_command_plan")
    plan_authorized = (
        plan_metadata.get("authorized_lanes") if isinstance(plan_metadata, dict) else None
    )
    plan_schemas = {
        lane: plan["recovered_lanes"][lane]["recovery_evidence_schema"]
        for lane in recovered_lanes
    }
    preflight_schemas = recovery_preflight.get("recovery_evidence_schemas")
    retained_lanes = recovery_preflight.get("retained_active_lanes")
    legacy_lanes = {
        lane
        for lane, schema in plan_schemas.items()
        if schema == "legacy_v1_without_recovery_reason"
    }
    if (
        recovery_preflight.get("passed") is not True
        or recovery_preflight.get("run_id") != args.run_id
        or not isinstance(authorized, list)
        or len(authorized) != len(set(authorized))
        or set(authorized) != set(recovered_lanes)
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(plan_metadata, dict)
        or plan_metadata.get("path") != str(plan_path)
        or plan_metadata.get("sha256") != hashlib.sha256(plan_bytes).hexdigest()
        or not isinstance(plan_authorized, list)
        or set(plan_authorized) != set(recovered_lanes)
        or not isinstance(preflight_schemas, dict)
        or preflight_schemas != plan_schemas
        or not isinstance(retained_lanes, list)
        or len(retained_lanes) != len(set(retained_lanes))
        or set(retained_lanes) != legacy_lanes
        or legacy_lanes not in (set(), {"GPU2"})
    ):
        raise RuntimeError(
            "RECOVERY_PREFLIGHT or its immutable command plan is stale, changed, "
            "nonpassing, or authorizes a different lane set"
        )
    master_command_path = run_root / "MASTER_COMMANDS.log"
    master_commands = parse_master_command_log(master_command_path)
    lane_evidence = {}
    for lane in recovered_lanes:
        expected = plan["recovered_lanes"][lane]["expected_resumed_command_records"]
        expected_reason = plan["recovered_lanes"][lane]["recovery_reason"]
        evidence_schema = plan_schemas[lane]
        lane_evidence[lane] = validate_recovery_lane(
            run_root,
            args.run_id,
            lane,
            recovery_preflight,
            expected,
            expected_reason,
            evidence_schema,
            master_commands,
        )
    final_science_evidence = validate_final_science_artifacts()
    if (
        not process_is_alive(original_terminal["heartbeat_pid"])
        or not process_group_is_alive(original_terminal["heartbeat_process_group_id"])
    ):
        raise RuntimeError("heartbeat died during the final science artifact audit")

    # Preserve the automatic supervisor result only after every reconciliation
    # gate has passed, and before replacing any canonical terminal pointer.
    preserved_records = {}
    for name, path in supervisor_artifact_paths(run_root, recovered_lanes).items():
        preserved_records[name] = preserve_original_supervisor_artifact(
            path, original_records[name][1]
        )
    preserved_plan_path = run_root / "MASTER_RECOVERY_RECONCILIATION_PLAN.json"
    if preserved_plan_path.exists():
        if preserved_plan_path.read_bytes() != plan_bytes:
            raise RuntimeError("preserved recovery plan differs from requested recovery plan")
    else:
        durable_bytes_exclusive(preserved_plan_path, plan_bytes)

    reconciliation_path = run_root / "MASTER_RECOVERY_RECONCILIATION.json"
    reconciliation = {
        "schema_version": 1,
        "run_id": args.run_id,
        "pod": EXPECTED_POD,
        "status": "PASS",
        "passed": True,
        "created_utc": now_utc(),
        "recovered_lanes": recovered_lanes,
        "canonical_lane_statuses": {
            lane: (
                "RECOVERABLE_FAILURE_RESUMED" if lane in recovered_lanes else "SUCCESS"
            )
            for lane in LANES
        },
        "original_supervisor_status": original_terminal["status"],
        "original_supervisor_records": preserved_records,
        "recovery_preflight": {
            "path": str(recovery_preflight_path),
            "sha256": file_sha256(recovery_preflight_path),
            "authorized_lanes": authorized,
        },
        "recovery_plan": {
            "source_path": str(plan_path),
            "preserved_path": str(preserved_plan_path),
            "sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "expected_commands": {
                lane: plan["recovered_lanes"][lane]["expected_resumed_command_records"]
                for lane in recovered_lanes
            },
        },
        "master_command_log": {
            "path": str(master_command_path),
            "sha256": file_sha256(master_command_path),
        },
        "lane_recovery_evidence": lane_evidence,
        "final_science_evidence": final_science_evidence,
        "heartbeat": {
            "pid": original_terminal["heartbeat_pid"],
            "process_group_id": original_terminal["heartbeat_process_group_id"],
            "alive_during_reconciliation": True,
        },
        "checks": {
            "original_automatic_hard_failure_preserved": True,
            "original_four_lane_shells_exited": True,
            "unaffected_lanes_succeeded": True,
            "recovery_preflight_exact_and_passing": True,
            "recovered_shells_completed_and_are_absent": True,
            "structured_recovery_evidence_exact": True,
            "expected_commands_match_independent_logs": True,
            "base_checkpoints_freshly_hashed_and_strictly_verified": True,
            "all_final_reports_audits_and_checkpoints_verified": True,
            "all_gpus_compute_idle": True,
            "heartbeat_alive": True,
        },
        "pod_stop_automated": False,
    }
    if reconciliation_path.exists():
        existing = read_json(reconciliation_path)
        mismatches = [
            key
            for key, value in reconciliation.items()
            if key != "created_utc" and existing.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                "existing recovery reconciliation differs from current exact evidence: "
                + ", ".join(mismatches)
            )
        reconciliation = existing
    else:
        durable_json_exclusive(reconciliation_path, reconciliation)
    reconciliation_sha = file_sha256(reconciliation_path)

    canonical_lanes = {}
    for lane in LANES:
        if lane not in recovered_lanes:
            canonical_lanes[lane] = original_terminal["lanes"][lane]
            continue
        original_lane = original_records[f"{lane}_TERMINAL"][2]
        evidence = lane_evidence[lane]
        canonical_lanes[lane] = {
            "schema_version": 2,
            "run_id": args.run_id,
            "lane": lane,
            "returncode": 0,
            "status": "RECOVERABLE_FAILURE_RESUMED",
            "rationale": [
                "original automatic shell failure is preserved",
                "completed recovery shell and structured recovery evidence passed reconciliation",
            ],
            "original_supervisor_returncode": original_lane["returncode"],
            "effective_shell_pid": evidence["effective_shell_pid"],
            "effective_shell_process_group_id": evidence[
                "effective_shell_process_group_id"
            ],
            "science_complete_marker": evidence["science_complete_marker"],
            "error_marker": evidence["error_marker"],
            "recovery_reconciled": True,
            "reconciliation_record": str(reconciliation_path),
            "reconciliation_record_sha256": reconciliation_sha,
            "original_supervisor_terminal": preserved_records[f"{lane}_TERMINAL"]["path"],
            "original_supervisor_terminal_sha256": preserved_records[f"{lane}_TERMINAL"]["sha256"],
            "normalized_utc": now_utc(),
        }
    for lane in recovered_lanes:
        payload = canonical_lanes[lane]
        durable_json(run_root / f"lane_{lane.lower()}.terminal.json", payload)

    canonical_terminal = {
        "schema_version": 2,
        "run_id": args.run_id,
        "pod": EXPECTED_POD,
        "all_four_lane_shells_exited": True,
        "all_effective_lane_shells_exited": True,
        "all_lanes_terminal": True,
        "status": "SUCCESS",
        "lanes": canonical_lanes,
        "recovery_reconciled": True,
        "recovered_lanes": recovered_lanes,
        "reconciliation_record": str(reconciliation_path),
        "reconciliation_record_sha256": reconciliation_sha,
        "original_supervisor_status": "HARD_FAILURE",
        "original_supervisor_records": preserved_records,
        "heartbeat_pid": original_terminal["heartbeat_pid"],
        "heartbeat_process_group_id": original_terminal["heartbeat_process_group_id"],
        "heartbeat_left_running_for_finalization": True,
        "pod_stop_automated": False,
        "created_utc": now_utc(),
    }
    durable_json(run_root / "MASTER_TERMINAL_STATUS.json", canonical_terminal)
    durable_json(run_root / "MASTER_ALL_LANES_TERMINAL", canonical_terminal)
    durable_json(master_root / "MASTER_TERMINAL_STATUS.json", canonical_terminal)
    durable_json(master_root / "MASTER_ALL_LANES_TERMINAL", canonical_terminal)
    canonical_state = original_records["MASTER_SUPERVISOR"][2]
    canonical_state.update(
        {
            "status": "ALL_LANES_TERMINAL_PENDING_MASTER_FINALIZATION",
            "recovery_reconciled": True,
            "recovered_lanes": recovered_lanes,
            "reconciliation_record": str(reconciliation_path),
            "reconciliation_record_sha256": reconciliation_sha,
            "terminal_record": str(run_root / "MASTER_TERMINAL_STATUS.json"),
            "all_lanes_terminal_sentinel": str(run_root / "MASTER_ALL_LANES_TERMINAL"),
        }
    )
    for lane in LANES:
        canonical_state["lanes"][lane]["status"] = canonical_lanes[lane]["status"]
        canonical_state["lanes"][lane]["terminal_record"] = str(
            run_root / f"lane_{lane.lower()}.terminal.json"
        )
    write_status(master_root, run_root, canonical_state)
    print(str(reconciliation_path), flush=True)
    return 0


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
    terminal_status = read_json(run_root / "MASTER_TERMINAL_STATUS.json")
    top_terminal = read_json(master_root / "MASTER_ALL_LANES_TERMINAL")
    top_status = read_json(master_root / "MASTER_TERMINAL_STATUS.json")
    supervisor_state = read_json(run_root / "MASTER_SUPERVISOR.json")
    current_status = read_json(master_root / "MASTER_STATUS.json")
    problems = []
    if terminal != terminal_status or terminal != top_terminal or terminal != top_status:
        problems.append("run-scoped and top-level canonical terminal records differ")
    if terminal.get("run_id") != args.run_id or terminal.get("pod") != EXPECTED_POD:
        problems.append("terminal run/pod identity mismatch")
    if terminal.get("status") != "SUCCESS":
        problems.append("canonical terminal status is not SUCCESS")
    if terminal.get("all_four_lane_shells_exited") is not True:
        problems.append("the four original lane shells are not recorded exited")
    if terminal.get("all_lanes_terminal") is not True:
        problems.append("all lanes are not terminal")
    if supervisor_state != current_status:
        problems.append("run-scoped and top-level current supervisor status differ")
    if (
        supervisor_state.get("run_id") != args.run_id
        or supervisor_state.get("status")
        != "ALL_LANES_TERMINAL_PENDING_MASTER_FINALIZATION"
        or supervisor_state.get("terminal_record")
        != str(run_root / "MASTER_TERMINAL_STATUS.json")
    ):
        problems.append("current supervisor status is not exact pending-finalization state")
    lanes = terminal.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != set(LANES):
        problems.append("canonical terminal lane set is not exact")
        lanes = {}
    acceptable = {"SUCCESS", "RECOVERABLE_FAILURE_RESUMED"}
    for lane in LANES:
        row = lanes.get(lane, {})
        try:
            standalone = read_json(run_root / f"lane_{lane.lower()}.terminal.json")
        except RuntimeError as error:
            problems.append(str(error))
            standalone = None
        if (
            row.get("run_id") != args.run_id
            or row.get("lane") != lane
            or row.get("status") not in acceptable
            or row.get("returncode") != 0
        ):
            problems.append(f"{lane} does not have an acceptable exact terminal")
        if standalone != row:
            problems.append(f"{lane} embedded and standalone canonical terminals differ")
    heartbeat_pid = terminal.get("heartbeat_pid")
    heartbeat_pgid = terminal.get("heartbeat_process_group_id")
    if terminal.get("heartbeat_left_running_for_finalization") is not True:
        problems.append("heartbeat was not left running for finalization")
    if not process_is_alive(heartbeat_pid) or not process_group_is_alive(heartbeat_pgid):
        problems.append("heartbeat PID/PGID is not alive")

    recovered_by_status = sorted(
        lane for lane, row in lanes.items() if row.get("status") == "RECOVERABLE_FAILURE_RESUMED"
    )
    hard_error_lanes = []
    for lane in LANES:
        error_path = run_root / f"lane_{lane.lower()}.error.json"
        if not error_path.exists():
            continue
        try:
            error_marker = read_json(error_path)
        except RuntimeError as error:
            problems.append(str(error))
            continue
        if (
            error_marker.get("run_id") != args.run_id
            or error_marker.get("lane") != lane
            or error_marker.get("status") != "HARD_FAILURE"
            or not isinstance(error_marker.get("exit_code"), int)
            or error_marker["exit_code"] == 0
        ):
            problems.append(f"{lane} error marker is not an exact current hard failure")
        else:
            hard_error_lanes.append(lane)
    if sorted(hard_error_lanes) != recovered_by_status:
        problems.append(
            "current hard-failure marker set does not exactly match reconciled lane terminals"
        )
    if recovered_by_status:
        reconciliation_path = run_root / "MASTER_RECOVERY_RECONCILIATION.json"
        if terminal.get("recovery_reconciled") is not True:
            problems.append("recovered lanes have not been explicitly reconciled")
        if terminal.get("recovered_lanes") != recovered_by_status:
            problems.append("canonical recovered-lane set differs from lane terminals")
        if terminal.get("all_effective_lane_shells_exited") is not True:
            problems.append("effective recovery shells are not all recorded exited")
        if (
            supervisor_state.get("recovery_reconciled") is not True
            or supervisor_state.get("recovered_lanes") != recovered_by_status
        ):
            problems.append("current supervisor status lacks exact recovery reconciliation")
        if terminal.get("reconciliation_record") != str(reconciliation_path):
            problems.append("canonical reconciliation path is not exact")
        try:
            reconciliation = read_json(reconciliation_path)
            reconciliation_sha = file_sha256(reconciliation_path)
        except RuntimeError as error:
            problems.append(str(error))
            reconciliation = {}
            reconciliation_sha = None
        if terminal.get("reconciliation_record_sha256") != reconciliation_sha:
            problems.append("canonical reconciliation SHA does not match its bytes")
        if (
            reconciliation.get("run_id") != args.run_id
            or reconciliation.get("pod") != EXPECTED_POD
            or reconciliation.get("status") != "PASS"
            or reconciliation.get("passed") is not True
            or reconciliation.get("recovered_lanes") != recovered_by_status
            or reconciliation.get("canonical_lane_statuses")
            != {lane: lanes[lane].get("status") for lane in LANES}
            or not isinstance(reconciliation.get("checks"), dict)
            or not reconciliation.get("checks")
            or any(value is not True for value in reconciliation.get("checks", {}).values())
            or reconciliation.get("original_supervisor_status") != "HARD_FAILURE"
            or reconciliation.get("original_supervisor_records")
            != terminal.get("original_supervisor_records")
        ):
            problems.append("reconciliation artifact is stale, nonpassing, or inconsistent")
        original_records = reconciliation.get("original_supervisor_records")
        if isinstance(original_records, dict):
            expected_original_paths = {
                name: original_supervisor_path(path)
                for name, path in supervisor_artifact_paths(
                    run_root, recovered_by_status
                ).items()
            }
            if set(original_records) != set(expected_original_paths):
                problems.append("preserved original supervisor record set is not exact")
            for name, record in original_records.items():
                if not isinstance(record, dict):
                    problems.append(f"preserved original record metadata is invalid: {name}")
                    continue
                path_value = record.get("path")
                expected_sha = record.get("sha256")
                if (
                    not isinstance(path_value, str)
                    or not isinstance(expected_sha, str)
                    or SHA256_PATTERN.fullmatch(expected_sha) is None
                ):
                    problems.append(f"preserved original record identity is invalid: {name}")
                    continue
                original_path = Path(path_value)
                if (
                    original_path != expected_original_paths.get(name)
                    or file_sha256(original_path) != expected_sha
                ):
                    problems.append(f"preserved original record bytes changed: {name}")
        else:
            problems.append("reconciliation lacks preserved original supervisor records")
        for lane in recovered_by_status:
            row = lanes[lane]
            if (
                row.get("recovery_reconciled") is not True
                or row.get("reconciliation_record") != str(reconciliation_path)
                or row.get("reconciliation_record_sha256") != reconciliation_sha
            ):
                problems.append(f"{lane} lacks exact reconciliation provenance")
    elif terminal.get("recovery_reconciled") is True or terminal.get("recovered_lanes"):
        problems.append("terminal claims recovery reconciliation without recovered lanes")
    if problems:
        raise RuntimeError("cannot mark master finalization complete: " + "; ".join(problems))
    evidence_values = (args.git_evidence, args.report_evidence, args.backup_evidence)
    if any(not isinstance(value, str) or not value.strip() for value in evidence_values):
        raise RuntimeError("Git, report, and backup finalization evidence are all required")
    run_finalization = run_root / "MASTER_FINALIZATION_COMPLETE"
    top_finalization = master_root / "MASTER_FINALIZATION_COMPLETE"
    if run_finalization.exists():
        raise RuntimeError("finalization sentinel already exists; refusing to overwrite it")
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
        "terminal_record": str(run_root / "MASTER_TERMINAL_STATUS.json"),
        "terminal_record_sha256": file_sha256(run_root / "MASTER_TERMINAL_STATUS.json"),
        "recovery_reconciled": bool(recovered_by_status),
        "recovered_lanes": recovered_by_status,
        "pod_stop_automated": False,
    }
    if top_finalization.exists():
        existing = read_json(top_finalization)
        mismatches = [
            key
            for key, value in payload.items()
            if key != "created_utc" and existing.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                "partial top-level finalization differs from current exact evidence: "
                + ", ".join(mismatches)
            )
    else:
        durable_json_exclusive(top_finalization, payload)
    # This run-scoped path is the heartbeat trigger and is deliberately last.
    durable_bytes_exclusive(run_finalization, top_finalization.read_bytes())
    print(str(run_finalization), flush=True)
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
    current = subparsers.add_parser(
        "reconcile-recovery",
        help="replace an automatic hard terminal only after exact recovery evidence passes",
    )
    current.add_argument("--master-root", required=True)
    current.add_argument("--run-id", required=True)
    current.add_argument(
        "--recovered-lane",
        action="append",
        required=True,
        choices=sorted(LANES),
        help="explicit recovered lane; repeat once per recovered lane",
    )
    current.add_argument(
        "--recovery-plan",
        required=True,
        help=(
            "JSON with schema_version, run_id, and an exact recovered_lanes mapping; "
            "each lane must provide expected_resumed_command_records"
        ),
    )
    current.set_defaults(function=reconcile_recovery)
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
