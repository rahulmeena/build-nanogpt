#!/usr/bin/env python3
"""Durable master heartbeat for the four fixed GPU lanes.

The monitor is observational only: it never launches, restarts, or stops a
scientific process.  A lane supervisor owns those transitions.  This process
merges each experiment heartbeat with live NVIDIA telemetry and exits only
when the coordinator creates the later, run-scoped finalization sentinel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from parallel_2d2_supervisor import (
    EXPECTED_POD,
    FINALIZATION_EVIDENCE_FILES,
    FINAL_EVIDENCE_FRESHNESS,
    SHA256_PATTERN,
    UTC_TIMESTAMP_PATTERN,
    validate_master_preflight,
)


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
FINALIZATION_SENTINEL_KEYS = {
    "schema_version",
    "run_id",
    "status",
    "created_utc",
    "evidence",
    "report_git_binding",
    "final_science_evidence",
    "scientific_processes",
    "recorded_lane_processes",
    "authenticated_stop_identity",
    "timestamp_order",
    "terminal_record",
    "terminal_record_sha256",
    "recovery_reconciled",
    "recovered_lanes",
    "pod_stop_automated",
    "pod_stop_separate_required",
    "finalization_lock",
    "final_revalidation",
    "final_process_gate",
    "final_prepublication_fingerprint",
}
FINAL_REVALIDATION_KEYS = {
    "rechecked_immediately_before_publication",
    "captured_fields",
    "snapshot_sha256",
    "passed",
}
FINAL_PROCESS_GATE_KEYS = {
    "nvidia_compute_processes",
    "scientific_cpu_processes",
    "recorded_lane_processes",
    "all_gpus_compute_idle",
    "passed",
}
REPORT_GIT_BINDING_KEYS = {"checks", "passed"}
FINAL_PREPUBLICATION_FINGERPRINT_KEYS = {
    "captured_after_final_process_gate",
    "all_required_sources_match_signed_backup_inventory",
    "stable_file_count",
    "stable_file_inventory_sha256",
    "validated_boundary_serialized_sha256",
    "backup_manifest_sha256",
    "backup_signature_sha256",
    "passed",
}
NORMAL_HEARTBEAT_KEYS = {
    "schema_version",
    "run_id",
    "heartbeat_utc",
    "heartbeat_local",
    "monitor_pid",
    "monitor_process_group_id",
    "master_status",
    "lanes",
    "gpus",
    "storage",
    "errors",
    "observational_only",
    "process_control_actions",
    "pod_stop_automated",
}
ERROR_HEARTBEAT_KEYS = {
    "schema_version",
    "run_id",
    "heartbeat_utc",
    "monitor_pid",
    "monitor_process_group_id",
    "monitor_status",
    "observation_error",
    "observational_only",
    "process_control_actions",
    "pod_stop_automated",
}
MANIFEST_IDENTITY_KEYS = {"path", "bytes", "sha256", "created_utc"}
FINAL_SCIENCE_KEYS = {
    "final_reports",
    "final_audits",
    "final_checkpoints",
    "nvidia_compute_processes",
    "all_gpus_compute_idle",
}
RECORDED_PROCESS_KEYS = {
    "lanes",
    "archived_attempt_processes",
    "unique_recorded_processes",
    "all_recorded_lane_process_groups_absent",
    "passed",
}
STOP_IDENTITY_KEYS = {
    "path",
    "sha256",
    "run_id",
    "pod",
    "live_query",
    "checks",
    "passed",
}
LIVE_POD_QUERY_KEYS = {
    "run_id",
    "command",
    "authenticated",
    "queried_utc",
    "preflight_path",
    "preflight_sha256",
    "response",
}
TIMESTAMP_ORDER_KEYS = {
    "terminal_created_utc",
    "git_created_utc",
    "report_created_utc",
    "backup_verified_utc",
    "pod_queried_utc",
    "backup_created_utc",
    "passed",
}
NORMAL_TERMINAL_KEYS = {
    "schema_version",
    "run_id",
    "pod",
    "all_four_lane_shells_exited",
    "all_lanes_terminal",
    "status",
    "lanes",
    "heartbeat_pid",
    "heartbeat_process_group_id",
    "heartbeat_left_running_for_finalization",
    "pod_stop_automated",
    "created_utc",
}
RECOVERY_TERMINAL_KEYS = {
    "schema_version",
    "run_id",
    "pod",
    "all_four_lane_shells_exited",
    "all_effective_lane_shells_exited",
    "all_lanes_terminal",
    "status",
    "lanes",
    "recovery_reconciled",
    "recovered_lanes",
    "reconciliation_record",
    "reconciliation_record_sha256",
    "original_supervisor_status",
    "original_supervisor_records",
    "heartbeat_pid",
    "heartbeat_process_group_id",
    "heartbeat_left_running_for_finalization",
    "pod_stop_automated",
    "created_utc",
}
FINAL_HEARTBEAT_ADDITIONAL_KEYS = {
    "monitor_status",
    "terminal_utc",
    "finalization_sentinel",
    "finalization_sentinel_validation",
}
FINAL_REVALIDATION_FIELDS = [
    "git",
    "report",
    "report_git_binding",
    "backup",
    "final_science",
    "scientific_processes",
    "recorded_lane_processes",
    "authenticated_stop_identity",
    "timestamp_order",
]
EVIDENCE_KINDS = {
    "git": "parallel_2d2_final_git_evidence_v1",
    "report": "parallel_2d2_final_report_evidence_v1",
    "backup": "parallel_2d2_final_local_backup_evidence_v1",
}
FINALIZATION_CLOCK_SKEW = timedelta(minutes=2)


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


def parse_utc(value, label: str) -> datetime:
    if not isinstance(value, str) or UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as error:
        raise RuntimeError(f"{label} is not a valid timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise RuntimeError(f"{label} is not UTC")
    return parsed


def file_sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def current_process_identity() -> dict:
    pid = os.getpid()
    try:
        process_group_id = os.getpgid(pid)
        session_id = os.getsid(pid)
    except OSError as error:
        raise RuntimeError(f"cannot inspect the current heartbeat process: {error}") from error
    if (
        pid <= 1
        or process_group_id <= 1
        or session_id <= 1
        or process_group_id != os.getpgrp()
        or process_group_id != pid
        or session_id != pid
    ):
        raise RuntimeError(
            "current heartbeat is not the expected isolated session/process-group leader"
        )
    return {
        "pid": pid,
        "process_group_id": process_group_id,
        "session_id": session_id,
    }


def read_stable_regular_file(path: Path, *, require_read_only: bool) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(
            f"required evidence is unavailable or not symlink-free: {path}: {error}"
        ) from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"required evidence is not a regular file: {path}")
            if require_read_only and before.st_mode & 0o222:
                raise RuntimeError(f"required evidence is not read-only: {path}")
            content = handle.read()
            after = os.fstat(handle.fileno())
        path_after = path.lstat()
    except OSError as error:
        raise RuntimeError(f"cannot read required evidence {path}: {error}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    path_identity_after = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_mode,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if (
        identity_before != identity_after
        or identity_before != path_identity_after
        or not stat.S_ISREG(path_after.st_mode)
        or len(content) != before.st_size
    ):
        raise RuntimeError(f"required evidence changed while being read: {path}")
    return content


def parse_json_object(content: bytes, path: Path) -> dict:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"required evidence is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"required evidence is not a JSON object: {path}")
    return value


def validate_evidence_manifests(
    run_root: Path, run_id: str, evidence: dict
) -> dict[str, dict]:
    audited = {}
    for name, expected_kind in EVIDENCE_KINDS.items():
        row = evidence.get(name)
        manifest = row.get("manifest") if isinstance(row, dict) else None
        expected_path = run_root / FINALIZATION_EVIDENCE_FILES[name]
        if (
            not isinstance(row, dict)
            or row.get("passed") is not True
            or not isinstance(manifest, dict)
            or set(manifest) != MANIFEST_IDENTITY_KEYS
            or manifest.get("path") != str(expected_path)
            or not isinstance(manifest.get("bytes"), int)
            or isinstance(manifest.get("bytes"), bool)
            or manifest["bytes"] <= 0
            or not isinstance(manifest.get("sha256"), str)
            or SHA256_PATTERN.fullmatch(manifest["sha256"]) is None
        ):
            raise RuntimeError(f"finalization {name} evidence identity is not exact")
        content = read_stable_regular_file(expected_path, require_read_only=True)
        payload = parse_json_object(content, expected_path)
        if (
            manifest.get("bytes") != len(content)
            or manifest.get("sha256") != file_sha256_bytes(content)
            or payload.get("schema_version") != 1
            or payload.get("kind") != expected_kind
            or payload.get("run_id") != run_id
            or payload.get("pod") != EXPECTED_POD
            or payload.get("created_utc") != manifest.get("created_utc")
        ):
            raise RuntimeError(
                f"finalization {name} evidence no longer binds this run/pod"
            )
        parse_utc(manifest["created_utc"], f"{name} evidence created_utc")
        audited[name] = {
            "path": str(expected_path),
            "sha256": manifest["sha256"],
            "created_utc": manifest["created_utc"],
            "passed": True,
        }
    return audited


def validate_fresh_heartbeat_record(
    output: Path,
    current_output: Path,
    current: dict,
    run_id: str,
    interval_seconds: int,
    process_identity: dict,
    supervisor_state: dict,
) -> dict:
    run_bytes = read_stable_regular_file(output, require_read_only=False)
    top_bytes = read_stable_regular_file(current_output, require_read_only=False)
    if run_bytes != top_bytes:
        raise RuntimeError("run-scoped and top-level heartbeat records differ")
    observed = parse_json_object(run_bytes, output)
    allowed_schema = set(observed) in (NORMAL_HEARTBEAT_KEYS, ERROR_HEARTBEAT_KEYS)
    heartbeat_utc = parse_utc(observed.get("heartbeat_utc"), "heartbeat_utc")
    age = datetime.now(timezone.utc) - heartbeat_utc
    if (
        not allowed_schema
        or observed != current
        or observed.get("schema_version") != 1
        or observed.get("run_id") != run_id
        or observed.get("monitor_pid") != process_identity["pid"]
        or observed.get("monitor_process_group_id")
        != process_identity["process_group_id"]
        or observed.get("observational_only") is not True
        or observed.get("process_control_actions") != []
        or observed.get("pod_stop_automated") is not False
        or heartbeat_utc > datetime.now(timezone.utc) + FINALIZATION_CLOCK_SKEW
        or age > timedelta(seconds=max(60, 2 * interval_seconds + 30))
    ):
        raise RuntimeError("current master heartbeat identity/schema/freshness differs")
    if set(observed) == NORMAL_HEARTBEAT_KEYS:
        lanes = observed.get("lanes")
        if (
            not isinstance(lanes, dict)
            or set(lanes) != set(LANE_RESULTS)
            or observed.get("master_status") != supervisor_state
            or any(
                not isinstance(row, dict) or row.get("run_id") != run_id
                for row in lanes.values()
            )
        ):
            raise RuntimeError("current master heartbeat lane set is not exact")
    elif observed.get("monitor_status") != "OBSERVATION_ERROR_RETRYING":
        raise RuntimeError("current master heartbeat error schema differs")
    return {
        "path": str(output.resolve()),
        "sha256": file_sha256_bytes(run_bytes),
        "heartbeat_utc": observed["heartbeat_utc"],
        "monitor_pid": process_identity["pid"],
        "monitor_process_group_id": process_identity["process_group_id"],
        "schema": (
            "normal" if set(observed) == NORMAL_HEARTBEAT_KEYS else "observation_error"
        ),
        "passed": True,
    }


def validate_finalization_sentinel(
    master_root: Path,
    run_root: Path,
    run_id: str,
    current: dict,
    interval_seconds: int,
) -> dict:
    process_identity = current_process_identity()
    run_sentinel = run_root / "MASTER_FINALIZATION_COMPLETE"
    top_sentinel = master_root / "MASTER_FINALIZATION_COMPLETE"
    run_bytes = read_stable_regular_file(run_sentinel, require_read_only=True)
    top_bytes = read_stable_regular_file(top_sentinel, require_read_only=True)
    if run_bytes != top_bytes:
        raise RuntimeError("run-scoped and top-level finalization sentinels differ")
    sentinel = parse_json_object(run_bytes, run_sentinel)
    if set(sentinel) != FINALIZATION_SENTINEL_KEYS:
        raise RuntimeError("finalization sentinel key set is not exact")
    created = parse_utc(sentinel.get("created_utc"), "finalization created_utc")
    current_utc = datetime.now(timezone.utc)
    evidence = sentinel.get("evidence")
    report_git_binding = sentinel.get("report_git_binding")
    final_science = sentinel.get("final_science_evidence")
    recorded = sentinel.get("recorded_lane_processes")
    stop_identity = sentinel.get("authenticated_stop_identity")
    timestamp_order = sentinel.get("timestamp_order")
    revalidation = sentinel.get("final_revalidation")
    process_gate = sentinel.get("final_process_gate")
    fingerprint = sentinel.get("final_prepublication_fingerprint")
    recovered = sentinel.get("recovered_lanes")
    live_query = stop_identity.get("live_query") if isinstance(stop_identity, dict) else None
    stop_checks = stop_identity.get("checks") if isinstance(stop_identity, dict) else None
    expected_pod_response = {
        "desiredStatus": "RUNNING",
        "gpuCount": EXPECTED_POD["gpu_count"],
        "id": EXPECTED_POD["id"],
        "name": EXPECTED_POD["name"],
        "networkVolumeId": EXPECTED_POD["volume_id"],
        "runtimeStatus": "running",
    }
    if (
        sentinel.get("schema_version") != 1
        or sentinel.get("run_id") != run_id
        or sentinel.get("status") != "MASTER_FINALIZATION_COMPLETE"
        or created > current_utc + FINALIZATION_CLOCK_SKEW
        or current_utc - created > FINAL_EVIDENCE_FRESHNESS + FINALIZATION_CLOCK_SKEW
        or not isinstance(evidence, dict)
        or set(evidence) != {"git", "report", "backup"}
        or any(
            not isinstance(row, dict) or row.get("passed") is not True
            for row in evidence.values()
        )
        or not isinstance(report_git_binding, dict)
        or set(report_git_binding) != REPORT_GIT_BINDING_KEYS
        or report_git_binding.get("passed") is not True
        or not isinstance(report_git_binding.get("checks"), dict)
        or set(report_git_binding["checks"])
        != set(evidence.get("git", {}).get("repositories", {}))
        or any(value is not True for value in report_git_binding["checks"].values())
        or not isinstance(final_science, dict)
        or set(final_science) != FINAL_SCIENCE_KEYS
        or final_science.get("all_gpus_compute_idle") is not True
        or final_science.get("nvidia_compute_processes") != []
        or sentinel.get("scientific_processes") != []
        or not isinstance(recorded, dict)
        or set(recorded) != RECORDED_PROCESS_KEYS
        or recorded.get("passed") is not True
        or recorded.get("all_recorded_lane_process_groups_absent") is not True
        or not isinstance(stop_identity, dict)
        or set(stop_identity) != STOP_IDENTITY_KEYS
        or stop_identity.get("passed") is not True
        or stop_identity.get("run_id") != run_id
        or stop_identity.get("pod") != EXPECTED_POD
        or stop_identity.get("path") != str(run_root / "AUTO_STOP_PREFLIGHT.json")
        or not isinstance(stop_identity.get("sha256"), str)
        or SHA256_PATTERN.fullmatch(stop_identity["sha256"]) is None
        or not isinstance(stop_checks, dict)
        or not stop_checks
        or any(value is not True for value in stop_checks.values())
        or not isinstance(live_query, dict)
        or set(live_query) != LIVE_POD_QUERY_KEYS
        or live_query.get("run_id") != run_id
        or live_query.get("command")
        != f"runpodctl pod get {EXPECTED_POD['id']} -o json"
        or live_query.get("authenticated") is not True
        or live_query.get("preflight_path")
        != str(run_root / "AUTO_STOP_PREFLIGHT.json")
        or live_query.get("preflight_sha256") != stop_identity.get("sha256")
        or live_query.get("response") != expected_pod_response
        or not isinstance(timestamp_order, dict)
        or set(timestamp_order) != TIMESTAMP_ORDER_KEYS
        or timestamp_order.get("passed") is not True
        or not isinstance(recovered, list)
        or recovered != sorted(set(recovered))
        or any(lane not in LANE_RESULTS for lane in recovered)
        or sentinel.get("recovery_reconciled") is not bool(recovered)
        or sentinel.get("pod_stop_automated") is not False
        or sentinel.get("pod_stop_separate_required") is not True
        or sentinel.get("finalization_lock")
        != str(master_root / "locks/finalize.lock")
        or not isinstance(revalidation, dict)
        or set(revalidation) != FINAL_REVALIDATION_KEYS
        or revalidation.get("rechecked_immediately_before_publication") is not True
        or revalidation.get("captured_fields") != FINAL_REVALIDATION_FIELDS
        or revalidation.get("passed") is not True
        or not isinstance(revalidation.get("snapshot_sha256"), str)
        or SHA256_PATTERN.fullmatch(revalidation["snapshot_sha256"]) is None
        or not isinstance(process_gate, dict)
        or set(process_gate) != FINAL_PROCESS_GATE_KEYS
        or process_gate.get("nvidia_compute_processes") != []
        or process_gate.get("scientific_cpu_processes") != []
        or process_gate.get("all_gpus_compute_idle") is not True
        or process_gate.get("passed") is not True
        or not isinstance(process_gate.get("recorded_lane_processes"), dict)
        or process_gate["recorded_lane_processes"] != recorded
        or not isinstance(fingerprint, dict)
        or set(fingerprint) != FINAL_PREPUBLICATION_FINGERPRINT_KEYS
        or fingerprint.get("captured_after_final_process_gate") is not True
        or fingerprint.get("all_required_sources_match_signed_backup_inventory")
        is not True
        or not isinstance(fingerprint.get("stable_file_count"), int)
        or isinstance(fingerprint.get("stable_file_count"), bool)
        or fingerprint["stable_file_count"] <= 0
        or fingerprint.get("passed") is not True
        or any(
            not isinstance(fingerprint.get(key), str)
            or SHA256_PATTERN.fullmatch(fingerprint[key]) is None
            for key in (
                "stable_file_inventory_sha256",
                "validated_boundary_serialized_sha256",
                "backup_manifest_sha256",
                "backup_signature_sha256",
            )
        )
    ):
        raise RuntimeError("finalization sentinel evidence/identity is invalid")
    evidence_manifests = validate_evidence_manifests(run_root, run_id, evidence)
    backup_signature = evidence["backup"].get("detached_signature")
    boundary_snapshot = {
        "git": evidence["git"],
        "report": evidence["report"],
        "report_git_binding": report_git_binding,
        "backup": evidence["backup"],
        "final_science": final_science,
        "scientific_processes": sentinel["scientific_processes"],
        "recorded_lane_processes": recorded,
        "authenticated_stop_identity": stop_identity,
        "timestamp_order": timestamp_order,
        "evidence_manifests_unchanged": True,
        "pod_stop_invoked": False,
        "passed": True,
    }
    if (
        not isinstance(backup_signature, dict)
        or fingerprint["backup_manifest_sha256"]
        != evidence_manifests["backup"]["sha256"]
        or fingerprint["backup_signature_sha256"]
        != backup_signature.get("signature_sha256")
        or fingerprint["validated_boundary_serialized_sha256"]
        != hashlib.sha256(
            json.dumps(
                boundary_snapshot, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    ):
        raise RuntimeError("final prepublication fingerprint cross-binding differs")

    terminal_path = run_root / "MASTER_TERMINAL_STATUS.json"
    if (
        sentinel.get("terminal_record") != str(terminal_path)
        or not isinstance(sentinel.get("terminal_record_sha256"), str)
        or SHA256_PATTERN.fullmatch(sentinel["terminal_record_sha256"]) is None
    ):
        raise RuntimeError("finalization sentinel terminal path differs")
    terminal_bytes = read_stable_regular_file(terminal_path, require_read_only=False)
    if sentinel.get("terminal_record_sha256") != file_sha256_bytes(terminal_bytes):
        raise RuntimeError("finalization sentinel terminal SHA differs")
    terminal_mirrors = (
        run_root / "MASTER_ALL_LANES_TERMINAL",
        master_root / "MASTER_TERMINAL_STATUS.json",
        master_root / "MASTER_ALL_LANES_TERMINAL",
    )
    if any(
        read_stable_regular_file(path, require_read_only=False) != terminal_bytes
        for path in terminal_mirrors
    ):
        raise RuntimeError("canonical terminal/all-lanes records are not byte-identical")
    terminal = parse_json_object(terminal_bytes, terminal_path)
    terminal_keys = RECOVERY_TERMINAL_KEYS if recovered else NORMAL_TERMINAL_KEYS
    lanes = terminal.get("lanes")
    if (
        set(terminal) != terminal_keys
        or terminal.get("schema_version") != (2 if recovered else 1)
        or terminal.get("run_id") != run_id
        or terminal.get("pod") != EXPECTED_POD
        or terminal.get("status") != "SUCCESS"
        or terminal.get("all_four_lane_shells_exited") is not True
        or terminal.get("all_lanes_terminal") is not True
        or terminal.get("pod_stop_automated") is not False
        or not isinstance(lanes, dict)
        or set(lanes) != set(LANE_RESULTS)
        or any(
            not isinstance(row, dict)
            or row.get("run_id") != run_id
            or row.get("lane") != lane
            or row.get("status")
            != ("RECOVERABLE_FAILURE_RESUMED" if lane in recovered else "SUCCESS")
            for lane, row in lanes.items()
        )
        or (
            bool(recovered)
            and (
            terminal.get("all_effective_lane_shells_exited") is not True
            or terminal.get("recovery_reconciled") is not True
            or terminal.get("recovered_lanes") != recovered
            or terminal.get("original_supervisor_status") != "HARD_FAILURE"
            )
        )
        or terminal.get("heartbeat_left_running_for_finalization") is not True
        or terminal.get("heartbeat_pid") != process_identity["pid"]
        or terminal.get("heartbeat_process_group_id")
        != process_identity["process_group_id"]
    ):
        raise RuntimeError("terminal record does not bind the live heartbeat PID/PGID")
    terminal_created = parse_utc(terminal.get("created_utc"), "terminal created_utc")

    run_state_path = run_root / "MASTER_SUPERVISOR.json"
    top_state_path = master_root / "MASTER_STATUS.json"
    run_state_bytes = read_stable_regular_file(run_state_path, require_read_only=False)
    top_state_bytes = read_stable_regular_file(top_state_path, require_read_only=False)
    if run_state_bytes != top_state_bytes:
        raise RuntimeError("run-scoped and top-level supervisor states differ")
    state = parse_json_object(run_state_bytes, run_state_path)
    heartbeat_state = state.get("heartbeat")
    if (
        state.get("run_id") != run_id
        or state.get("pod") != EXPECTED_POD
        or state.get("status") != "ALL_LANES_TERMINAL_PENDING_MASTER_FINALIZATION"
        or state.get("terminal_record") != str(terminal_path)
        or not isinstance(state.get("lanes"), dict)
        or set(state["lanes"]) != set(LANE_RESULTS)
        or not isinstance(heartbeat_state, dict)
        or set(heartbeat_state) != {"pid", "process_group_id", "status", "log"}
        or heartbeat_state.get("pid") != process_identity["pid"]
        or heartbeat_state.get("process_group_id")
        != process_identity["process_group_id"]
        or heartbeat_state.get("status") != "RUNNING_THROUGH_MASTER_FINALIZATION"
        or heartbeat_state.get("log") != str(run_root / "heartbeat.supervisor.log")
    ):
        raise RuntimeError("supervisor state does not bind the live heartbeat PID/PGID")

    parsed_order = {
        name: parse_utc(timestamp_order[name], f"timestamp_order.{name}")
        for name in TIMESTAMP_ORDER_KEYS - {"passed"}
    }
    backup = evidence["backup"]
    if (
        timestamp_order["terminal_created_utc"] != terminal["created_utc"]
        or timestamp_order["git_created_utc"]
        != evidence_manifests["git"]["created_utc"]
        or timestamp_order["report_created_utc"]
        != evidence_manifests["report"]["created_utc"]
        or timestamp_order["backup_created_utc"]
        != evidence_manifests["backup"]["created_utc"]
        or not isinstance(backup.get("verification_host"), dict)
        or timestamp_order["backup_verified_utc"]
        != backup["verification_host"].get("verified_utc")
        or timestamp_order["pod_queried_utc"] != live_query.get("queried_utc")
        or backup.get("authenticated_pod_query") != live_query
        or terminal_created != parsed_order["terminal_created_utc"]
        or min(
            parsed_order["git_created_utc"],
            parsed_order["report_created_utc"],
            parsed_order["backup_created_utc"],
            parsed_order["backup_verified_utc"],
            parsed_order["pod_queried_utc"],
        )
        < terminal_created
        or parsed_order["backup_created_utc"]
        < max(
            parsed_order["git_created_utc"],
            parsed_order["report_created_utc"],
            parsed_order["backup_verified_utc"],
            parsed_order["pod_queried_utc"],
        )
        or max(parsed_order.values()) > created
    ):
        raise RuntimeError("finalization timestamp/evidence cross-binding differs")
    fresh = validate_fresh_heartbeat_record(
        run_root / "MASTER_HEARTBEAT.json",
        master_root / "MASTER_HEARTBEAT.json",
        current,
        run_id,
        interval_seconds,
        process_identity,
        state,
    )
    if (
        current_process_identity() != process_identity
        or read_stable_regular_file(run_sentinel, require_read_only=True) != run_bytes
        or read_stable_regular_file(top_sentinel, require_read_only=True) != run_bytes
        or read_stable_regular_file(terminal_path, require_read_only=False)
        != terminal_bytes
        or read_stable_regular_file(run_state_path, require_read_only=False)
        != run_state_bytes
        or read_stable_regular_file(top_state_path, require_read_only=False)
        != run_state_bytes
    ):
        raise RuntimeError("finalization evidence changed before heartbeat exit")
    return {
        "sentinel": sentinel,
        "sentinel_sha256": file_sha256_bytes(run_bytes),
        "terminal_sha256": file_sha256_bytes(terminal_bytes),
        "supervisor_state_sha256": file_sha256_bytes(run_state_bytes),
        "evidence_manifests": evidence_manifests,
        "heartbeat_process_identity": process_identity,
        "fresh_heartbeat": fresh,
        "passed": True,
    }


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
        "monitor_process_group_id": os.getpgrp(),
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
    current_process_identity()
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
                "monitor_process_group_id": os.getpgrp(),
                "monitor_status": "OBSERVATION_ERROR_RETRYING",
                "observation_error": f"{type(error).__name__}: {error}",
                "observational_only": True,
                "process_control_actions": [],
                "pod_stop_automated": False,
            }
        durable_json(output, current)
        durable_json(current_output, current)
        if sentinel.exists():
            try:
                validation = validate_finalization_sentinel(
                    master_root,
                    run_root,
                    args.run_id,
                    current,
                    int(args.interval_seconds),
                )
            except RuntimeError as error:
                rejection = {
                    "observed_utc": now_utc(),
                    "error": f"{type(error).__name__}: {error}",
                    "sentinel": str(sentinel),
                }
                if set(current) == NORMAL_HEARTBEAT_KEYS:
                    current["errors"] = dict(current["errors"])
                    current["errors"]["finalization_sentinel_rejected"] = rejection
                else:
                    current["observation_error"] = (
                        str(current.get("observation_error", "observation schema differs"))
                        + "; finalization_sentinel_rejected="
                        + rejection["error"]
                    )
                durable_json(output, current)
                durable_json(current_output, current)
                print(
                    "INVALID_MASTER_FINALIZATION_SENTINEL_IGNORED: "
                    + rejection["error"],
                    flush=True,
                )
            else:
                finalization = validation.pop("sentinel")
                final = dict(current)
                final["monitor_status"] = (
                    "MASTER_FINALIZATION_COMPLETE_SENTINEL_OBSERVED"
                )
                final["terminal_utc"] = now_utc()
                final["finalization_sentinel"] = finalization
                final["finalization_sentinel_validation"] = validation
                expected_final_keys = (
                    set(current) | FINAL_HEARTBEAT_ADDITIONAL_KEYS
                )
                if set(final) != expected_final_keys:
                    raise RuntimeError("terminal heartbeat schema construction failed")
                durable_json(output, final)
                durable_json(current_output, final)
                final_bytes = read_stable_regular_file(
                    output, require_read_only=False
                )
                if (
                    final_bytes
                    != read_stable_regular_file(current_output, require_read_only=False)
                    or parse_json_object(final_bytes, output) != final
                    or current_process_identity()
                    != validation["heartbeat_process_identity"]
                    or file_sha256_bytes(
                        read_stable_regular_file(sentinel, require_read_only=True)
                    )
                    != validation["sentinel_sha256"]
                    or file_sha256_bytes(
                        read_stable_regular_file(
                            master_root / "MASTER_FINALIZATION_COMPLETE",
                            require_read_only=True,
                        )
                    )
                    != validation["sentinel_sha256"]
                    or file_sha256_bytes(
                        read_stable_regular_file(
                            run_root / "MASTER_TERMINAL_STATUS.json",
                            require_read_only=False,
                        )
                    )
                    != validation["terminal_sha256"]
                    or file_sha256_bytes(
                        read_stable_regular_file(
                            run_root / "MASTER_SUPERVISOR.json",
                            require_read_only=False,
                        )
                    )
                    != validation["supervisor_state_sha256"]
                    or file_sha256_bytes(
                        read_stable_regular_file(
                            master_root / "MASTER_STATUS.json",
                            require_read_only=False,
                        )
                    )
                    != validation["supervisor_state_sha256"]
                ):
                    raise RuntimeError(
                        "terminal heartbeat publication changed before exit"
                    )
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
