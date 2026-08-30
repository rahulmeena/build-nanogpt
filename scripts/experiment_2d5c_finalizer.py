#!/usr/bin/env python3
"""Stdlib-only post-stop finalizer for Experiment 2D5C.

This module intentionally has no Torch, NumPy, project-training, RunPod CLI,
or network dependency.  It validates already-written scientific, Git, and
exact-pod guard evidence, then writes the terminal audit and final report.
It never starts, stops, deletes, or otherwise mutates infrastructure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPERIMENT = "2D5C"
BRANCH = "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m"
FINAL_TAG = "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m-final"
POD_ID = "h6of430yxncf6h"
POD_NAME = "opposite_azure_ladybug"
VOLUME_ID = "yhzyb27fb5"
VOLUME_NAME = "unlikely_lime_flamingo"
VOLUME_SIZE_GB = 150
VOLUME_DATACENTER = "US-MD-1"
VOLUME_MOUNT_PATH = "/workspace"
GPU_COUNT = 1

GIT_SCHEMA = "experiment_2d5c_git_verification_v1"
GUARD_AUTHORIZATION_SCHEMA = "experiment_2d5c_runpod_guard_authorization_v1"
GUARD_TRIGGER_SCHEMA = "experiment_2d5c_runpod_guard_trigger_v1"
GUARD_REPORT_SCHEMA = "experiment_2d5c_runpod_guard_report_v1"
FINAL_AUDIT_SCHEMA = "experiment_2d5c_final_audit_v1"
ACTION = "stop_exact_pod_after_terminal_2d5c"
EXACT_STOP_COMMAND = f"runpodctl pod stop {POD_ID} -o json"
FINAL_PHRASE = "STOPPED AFTER C AT EXACTLY 191 UPDATES / 100,139,008 TARGETS"
GUARD_AUTHORIZATION_KEYS = frozenset({
    "schema", "experiment", "action", "pod_id", "pod_name", "gpu_count",
    "network_volume_id", "network_volume_name", "network_volume_size_gb",
    "network_volume_datacenter", "volume_mount_path", "pod_created_at",
    "pod_running_last_status_change", "identity_sha256", "exact_stop_command",
    "credential_source", "issued_at_utc", "expires_at_utc",
    "authorization_nonce",
})
GUARD_TRIGGER_KEYS = frozenset({
    "schema", "experiment", "action", "pod_id", "pod_name", "gpu_count",
    "network_volume_id", "authorization_sha256", "authorization_nonce",
    "terminal_outcome", "exit_code", "source", "created_at_utc",
})

REQUIRED_FINAL_AUDIT_CHECKS = (
    "source_checkpoint_sha_exact",
    "c_started_from_2d3a_source",
    "fixed_control_checkpoint_sha_exact",
    "exactly_one_newly_trained_arm",
    "fixed_control_optimizer_steps_zero",
    "c_optimizer_steps_exact",
    "c_new_targets_exact",
    "final_global_update_exact",
    "final_cumulative_targets_exact",
    "replay_191_batches_exact",
    "replay_chain_hash_exact",
    "initial_terminal_loader_cursor_hashes_exact",
    "pass_cadence_exact",
    "optimizer_continuity",
    "scheduler_continuity",
    "rng_continuity",
    "midpoint_fresh_process_restart_success",
    "parameter_count_unchanged",
    "state_dict_keys_exact",
    "fixed_writers_preserved",
    "b3_b5_lag_coverage_exact",
    "local_recurrent_nonoverlap",
    "causality_tests_passed",
    "deployment_cache_tests_passed",
    "control_specificity_tests_passed",
    "ce_only_objective",
    "attached_writer_gradients",
    "analysis_input_identities_exact",
    "secondary_parallel_c0_c96_c191_completed",
    "all_required_core_conditions_completed",
    "all_14_large_conditions_completed",
    "per_sequence_pairing_intact",
    "large_targets_exact_every_condition",
    "historical_panel_disjointness_checked_where_possible",
    "fourteen_condition_evaluation_not_reduced",
    "memory_accounting_completed",
    "final_checkpoint_strict_reopen_passed",
    "remote_local_checkpoint_sha_match",
    "git_branch_commit_tag_pushed_verified",
    "worktree_clean",
    "no_a_b_or_250m_training",
)

REQUIRED_REPORT_CONTRASTS = (
    "architecture_fixed_minus_c",
    "architecture_c_minus_fixed_penalty",
    "c_b3_off_gain",
    "c_b3_sequence_gap",
    "c_b5_off_gain",
    "c_b5_sequence_gap",
    "c_combined_off_gain",
    "c_combined_sequence_gap",
    "f_b3_off_gain",
    "f_b3_sequence_gap",
    "f_b5_off_gain",
    "f_b5_sequence_gap",
    "f_combined_off_gain",
    "f_combined_sequence_gap",
    "b3_off_gain_lift",
    "b3_sequence_gap_lift",
    "b5_off_gain_lift",
    "b5_sequence_gap_lift",
    "combined_off_gain_lift",
    "combined_sequence_gap_lift",
)

CLASSIFICATIONS = (
    "INVALID — NO SCIENTIFIC CONCLUSION",
    "W2/W2 DEEP-RECURRENT SUBSTITUTION SUPPORTED",
    "W2/W2 DEEP-RECURRENT SUBSTITUTION SUPPORTED WITH ABSOLUTE CE IMPROVEMENT",
    "W2/W2 PARTIAL REPLACEMENT; RECENT NATIVE KV STILL HELPFUL",
    "W2/W2 RECOVERS WITHOUT ESTABLISHED B3/B5 RECURRENT DEPENDENCE",
    "RECURRENT PATH UTILITY ESTABLISHED; ALIGNED SEQUENCE MEMORY NOT ESTABLISHED",
    "W2/W2 PERSISTENT DEGRADATION; DEEP-RECURRENT SUBSTITUTION NOT SUPPORTED",
    "W2/W2 REPRESENTATION-PRESSURE RESULT UNRESOLVED",
)


class FinalizerError(RuntimeError):
    """Raised when terminal evidence is incomplete or inconsistent."""


def _fail(message: str) -> None:
    raise FinalizerError(message)


def _read_bytes(path: str | os.PathLike[str]) -> bytes:
    target = Path(path).resolve()
    if not target.is_file():
        _fail(f"missing required artifact: {target}")
    return target.read_bytes()


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    raw = _read_bytes(path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizerError(f"invalid JSON artifact: {Path(path).resolve()}") from error
    if not isinstance(value, dict):
        _fail(f"JSON artifact is not an object: {Path(path).resolve()}")
    return value


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path).resolve()
    raw = _read_bytes(target)
    return {"path": str(target), "sha256": sha256_bytes(raw), "bytes": len(raw)}


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_text(path: str | os.PathLike[str], value: str) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def durable_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    durable_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _all_true(checks: Mapping[str, Any]) -> bool:
    return bool(checks) and all(value is True for value in checks.values())


def validate_git_verification(payload: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, bool]:
    commit = payload.get("scientific_results_commit")
    checks = {
        "schema": payload.get("schema") == GIT_SCHEMA,
        "experiment": payload.get("experiment") == EXPERIMENT,
        "declared_passed": payload.get("passed") is True,
        "branch": payload.get("branch") == BRANCH == summary.get("git_branch"),
        "implementation_commit": is_git_commit(payload.get("implementation_commit"))
        and payload.get("implementation_commit") == summary.get("git_commit"),
        "scientific_results_commit": is_git_commit(commit),
        "origin_branch_exact": payload.get("origin_branch_commit") == commit,
        "tag_name": payload.get("final_tag") == FINAL_TAG == summary.get("git_tag"),
        "local_tag_exact": payload.get("local_tag_commit") == commit,
        "origin_tag_exact": payload.get("origin_tag_commit") == commit,
        "branch_push_verified": payload.get("branch_push_verified") is True,
        "tag_push_verified": payload.get("tag_push_verified") is True,
        "worktree_clean": payload.get("worktree_clean") is True,
    }
    return checks


def validate_stopped_report(payload: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute the non-secret exact stopped-state projection."""
    pod = payload.get("pod", {})
    volume = payload.get("network_volume", {})
    if not isinstance(pod, Mapping):
        pod = {}
    if not isinstance(volume, Mapping):
        volume = {}
    return {
        "schema": payload.get("schema") == GUARD_REPORT_SCHEMA,
        "declared_passed": payload.get("passed") is True,
        "mode": payload.get("mode")
        in ("stop", "watchdog_supervise", "watchdog_trigger"),
        "terminal_success": payload.get("terminal_outcome") == "success",
        "supervised_child_success": payload.get("mode") != "watchdog_supervise"
        or payload.get("child_exit_code") == 0,
        "safe_terminal_status": payload.get("status")
        in ("already_stopped_verified", "stopped_and_volume_retained_verified"),
        "stop_invocation_recorded": isinstance(payload.get("stop_invoked"), bool),
        "pod_exact": pod.get("id") == POD_ID
        and pod.get("name") == POD_NAME
        and pod.get("gpuCount") == GPU_COUNT
        and pod.get("networkVolumeId") == VOLUME_ID
        and pod.get("volumeMountPath") == VOLUME_MOUNT_PATH,
        "pod_exited": pod.get("desiredStatus") == "EXITED",
        "pod_runtime_stopped": pod.get("runtimeStatus") == "stopped",
        "volume_retained": volume.get("id") == VOLUME_ID
        and volume.get("name") == VOLUME_NAME
        and volume.get("size") == VOLUME_SIZE_GB
        and volume.get("dataCenterId") == VOLUME_DATACENTER,
        "authorization_digest": is_sha256(payload.get("authorization_sha256")),
        "trigger_digest": is_sha256(payload.get("trigger_sha256")),
        "secret_not_recorded": payload.get("secret_recorded") is False,
    }


def validate_guard_artifacts(
    authorization: Mapping[str, Any],
    trigger: Mapping[str, Any],
    stop_report: Mapping[str, Any],
    authorization_identity: Mapping[str, Any],
    trigger_identity: Mapping[str, Any],
    authorization_mode: int,
    trigger_mode: int,
) -> dict[str, bool]:
    pod = stop_report.get("pod", {})
    volume = stop_report.get("network_volume", {})
    if not isinstance(pod, Mapping):
        pod = {}
    if not isinstance(volume, Mapping):
        volume = {}
    checks = {
        "authorization_private": authorization_mode == 0o600,
        "trigger_private": trigger_mode == 0o600,
        "authorization_keys_exact": set(authorization) == GUARD_AUTHORIZATION_KEYS,
        "trigger_keys_exact": set(trigger) == GUARD_TRIGGER_KEYS,
        "authorization_schema": authorization.get("schema") == GUARD_AUTHORIZATION_SCHEMA,
        "trigger_schema": trigger.get("schema") == GUARD_TRIGGER_SCHEMA,
        "report_schema": stop_report.get("schema") == GUARD_REPORT_SCHEMA,
        "experiment": authorization.get("experiment") == trigger.get("experiment") == EXPERIMENT,
        "action": authorization.get("action") == trigger.get("action") == ACTION,
        "authorization_exact_identity": authorization.get("pod_id") == trigger.get("pod_id") == POD_ID
        and authorization.get("pod_name") == trigger.get("pod_name") == POD_NAME
        and authorization.get("gpu_count") == trigger.get("gpu_count") == GPU_COUNT
        and authorization.get("network_volume_id")
        == trigger.get("network_volume_id") == VOLUME_ID,
        "authorization_exact_volume": authorization.get("network_volume_name") == VOLUME_NAME
        and authorization.get("network_volume_size_gb") == VOLUME_SIZE_GB
        and authorization.get("network_volume_datacenter") == VOLUME_DATACENTER
        and authorization.get("volume_mount_path") == VOLUME_MOUNT_PATH,
        "authorization_exact_stop_command": authorization.get("exact_stop_command")
        == EXACT_STOP_COMMAND,
        "authorization_identity_digest": is_sha256(authorization.get("identity_sha256")),
        "authorization_credential_source": authorization.get("credential_source") == {
            "kind": "macOS Keychain generic password",
            "service": "runpod-codex-pod-stopper",
            "account": "rahul",
        },
        "authorization_timestamps": all(
            isinstance(authorization.get(name), str) and authorization.get(name)
            for name in ("issued_at_utc", "expires_at_utc")
        ),
        "authorization_hash_chain": trigger.get("authorization_sha256")
        == authorization_identity.get("sha256")
        == stop_report.get("authorization_sha256"),
        "trigger_hash_chain": trigger_identity.get("sha256")
        == stop_report.get("trigger_sha256"),
        "nonce_bound": isinstance(authorization.get("authorization_nonce"), str)
        and len(authorization.get("authorization_nonce", "")) == 64
        and trigger.get("authorization_nonce") == authorization.get("authorization_nonce"),
        "terminal_success": trigger.get("terminal_outcome") == "success"
        and trigger.get("exit_code") == 0
        and stop_report.get("terminal_outcome") == "success",
        "trigger_source": trigger.get("source") in ("explicit_terminal", "supervised_child"),
        "trigger_timestamp": isinstance(trigger.get("created_at_utc"), str)
        and bool(trigger.get("created_at_utc")),
        "supervised_child_success": stop_report.get("mode") != "watchdog_supervise"
        or stop_report.get("child_exit_code") == 0,
        "guard_passed": stop_report.get("passed") is True,
        "guard_mode": stop_report.get("mode")
        in ("stop", "watchdog_supervise", "watchdog_trigger"),
        "safe_terminal_status": stop_report.get("status")
        in ("already_stopped_verified", "stopped_and_volume_retained_verified"),
        "stop_invocation_recorded": isinstance(stop_report.get("stop_invoked"), bool),
        "pod_exact": pod.get("id") == POD_ID
        and pod.get("name") == POD_NAME
        and pod.get("gpuCount") == GPU_COUNT
        and pod.get("networkVolumeId") == VOLUME_ID
        and pod.get("volumeMountPath") == VOLUME_MOUNT_PATH,
        "pod_exited": pod.get("desiredStatus") == "EXITED",
        "pod_runtime_stopped": pod.get("runtimeStatus") == "stopped",
        "volume_retained": volume.get("id") == VOLUME_ID
        and volume.get("name") == VOLUME_NAME
        and volume.get("size") == VOLUME_SIZE_GB
        and volume.get("dataCenterId") == VOLUME_DATACENTER,
        "secret_not_recorded": stop_report.get("secret_recorded") is False,
        "stopped_report_recomputed": _all_true(
            validate_stopped_report(stop_report)
        ),
    }
    return checks


def validate_final_audit_document(payload: Mapping[str, Any]) -> dict[str, bool]:
    checks = payload.get("checks", {})
    git_checks = payload.get("git_checks", {})
    guard_checks = payload.get("runpod_stop_checks", {})
    validation = payload.get("validation", {})
    return {
        "schema": payload.get("schema") == FINAL_AUDIT_SCHEMA,
        "experiment": payload.get("experiment") == EXPERIMENT,
        "phase": payload.get("phase") == "terminal-post-stop",
        "declared_passed": payload.get("passed") is True,
        "required_checks_present": isinstance(checks, Mapping)
        and set(REQUIRED_FINAL_AUDIT_CHECKS).issubset(checks),
        "all_scientific_and_operational_checks": isinstance(checks, Mapping)
        and _all_true(checks),
        "git_checks": isinstance(git_checks, Mapping) and _all_true(git_checks),
        "guard_checks": isinstance(guard_checks, Mapping) and _all_true(guard_checks),
        "embedded_stop_report": isinstance(
            payload.get("runpod_stop_verification"), Mapping
        ) and _all_true(validate_stopped_report(
            payload["runpod_stop_verification"]
        )),
        "validation": isinstance(validation, Mapping)
        and validation.get("passed") is True
        and validation.get("status") == "PASS",
        "pod_stopped": payload.get("pod_stopped") is True,
        "persistent_volume_retained": payload.get("persistent_volume_retained") is True,
        "runpod_status": payload.get("runpod_status") == "STOPPED",
    }


def run_postflight(
    *,
    provisional_audit_path: str | os.PathLike[str],
    summary_path: str | os.PathLike[str],
    representation_path: str | os.PathLike[str],
    git_verification_path: str | os.PathLike[str],
    stop_verification_path: str | os.PathLike[str],
    guard_authorization_path: str | os.PathLike[str],
    guard_trigger_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    if output.name != "FINAL_AUDIT.json":
        _fail("postflight output must be named FINAL_AUDIT.json")
    scientific_parent = Path(summary_path).resolve().parent
    if (
        output.parent != scientific_parent
        or Path(provisional_audit_path).resolve().parent != scientific_parent
        or Path(representation_path).resolve().parent != scientific_parent
    ):
        _fail("postflight scientific inputs and FINAL_AUDIT.json must share one result directory")
    provisional = read_json(provisional_audit_path)
    summary = read_json(summary_path)
    representation = read_json(representation_path)
    git_verification = read_json(git_verification_path)
    stop_verification = read_json(stop_verification_path)
    authorization = read_json(guard_authorization_path)
    trigger = read_json(guard_trigger_path)

    scientific_checks = provisional.get("checks", {})
    if not isinstance(scientific_checks, dict):
        _fail("provisional audit checks are not an object")
    missing = sorted(set(REQUIRED_FINAL_AUDIT_CHECKS) - set(scientific_checks))
    false_before_git = sorted(
        name for name, value in scientific_checks.items() if value is not True
    )
    provisional_checks = {
        "experiment": provisional.get("experiment") == EXPERIMENT,
        "phase": provisional.get("phase") == "scientific-results-pretag",
        "critical_scientific_checks": provisional.get("critical_scientific_checks_passed") is True,
        "required_checks_present": not missing,
        "only_git_pending": false_before_git
        == ["git_branch_commit_tag_pushed_verified"],
        "pending_list_exact": provisional.get("pending_operational_checks")
        == false_before_git,
        "summary_embeds_exact_provisional": summary.get("audit") == provisional,
        "summary_experiment": summary.get("experiment") == EXPERIMENT,
        "classification_valid": summary.get("classification") in CLASSIFICATIONS
        and summary.get("classification") != "INVALID — NO SCIENTIFIC CONCLUSION",
        "summary_checkpoint_sha": is_sha256(
            summary.get("final_checkpoint", {}).get("sha256")
        ),
        "summary_recommendation": isinstance(summary.get("recommendation"), Mapping)
        and bool(summary.get("recommendation", {}).get("recommendation"))
        and bool(summary.get("recommendation", {}).get("reason")),
        "representation_experiment": representation.get("experiment") == EXPERIMENT,
        "representation_passed": representation.get("passed") is True,
    }
    if not _all_true(provisional_checks):
        _fail(f"provisional scientific evidence failed: {sorted(k for k, v in provisional_checks.items() if not v)}")

    git_checks = validate_git_verification(git_verification, summary)
    if not _all_true(git_checks):
        _fail(f"Git verification failed: {sorted(k for k, v in git_checks.items() if not v)}")

    authorization_path = Path(guard_authorization_path).resolve()
    trigger_path = Path(guard_trigger_path).resolve()
    authorization_identity = file_identity(authorization_path)
    trigger_identity = file_identity(trigger_path)
    guard_checks = validate_guard_artifacts(
        authorization,
        trigger,
        stop_verification,
        authorization_identity,
        trigger_identity,
        stat.S_IMODE(authorization_path.stat().st_mode),
        stat.S_IMODE(trigger_path.stat().st_mode),
    )
    if not _all_true(guard_checks):
        _fail(f"RunPod stop verification failed: {sorted(k for k, v in guard_checks.items() if not v)}")

    final_checks = dict(scientific_checks)
    final_checks["git_branch_commit_tag_pushed_verified"] = True
    final_checks["worktree_clean"] = git_checks["worktree_clean"]
    validation = {
        "required_check_count": len(REQUIRED_FINAL_AUDIT_CHECKS),
        "missing_checks": sorted(set(REQUIRED_FINAL_AUDIT_CHECKS) - set(final_checks)),
        "failing_checks": sorted(name for name, value in final_checks.items() if value is not True),
    }
    validation["passed"] = not validation["missing_checks"] and not validation["failing_checks"]
    validation["classification_allowed"] = validation["passed"]
    validation["status"] = "PASS" if validation["passed"] else "AUDIT INCOMPLETE"

    input_identity = {
        "provisional_audit": file_identity(provisional_audit_path),
        "summary": file_identity(summary_path),
        "representation": file_identity(representation_path),
        "git_verification": file_identity(git_verification_path),
        "stop_verification": file_identity(stop_verification_path),
        "guard_authorization": authorization_identity,
        "guard_trigger": trigger_identity,
    }
    result = {
        "schema": FINAL_AUDIT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "terminal-post-stop",
        "checks": final_checks,
        "provisional_checks": provisional_checks,
        "git_verification": git_verification,
        "git_checks": git_checks,
        "runpod_stop_verification": stop_verification,
        "runpod_stop_checks": guard_checks,
        "input_artifact_identity": input_identity,
        "pod_stopped": True,
        "persistent_volume_retained": True,
        "runpod_status": "STOPPED",
        "pod_id": POD_ID,
        "pod_name": POD_NAME,
        "volume_id": VOLUME_ID,
        "volume_name": VOLUME_NAME,
        "validation": validation,
        "passed": validation["passed"] and _all_true(git_checks) and _all_true(guard_checks),
    }
    if not _all_true(validate_final_audit_document(result)):
        _fail("terminal final-audit self-validation failed")
    durable_json(output, result)
    return result


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise FinalizerError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        _fail(f"{label} is not finite")
    return number


def ci_text(row: Mapping[str, Any]) -> str:
    lower = _finite_number(row.get("lower_95"), "CI lower bound")
    upper = _finite_number(row.get("upper_95"), "CI upper bound")
    return f"[{lower:+.12f}, {upper:+.12f}]"


def _contrast_point(rows: Mapping[str, Any], name: str) -> float:
    row = rows.get(name)
    if not isinstance(row, Mapping):
        _fail(f"missing report contrast: {name}")
    return _finite_number(row.get("point_estimate"), f"{name} point estimate")


def _pressure_table(rows: Mapping[str, Any]) -> list[str]:
    definitions = (
        ("B3", "OFF", "f_b3_off_gain", "c_b3_off_gain", "b3_off_gain_lift"),
        ("B3", "SHUFFLED", "f_b3_sequence_gap", "c_b3_sequence_gap", "b3_sequence_gap_lift"),
        ("B5", "OFF", "f_b5_off_gain", "c_b5_off_gain", "b5_off_gain_lift"),
        ("B5", "SHUFFLED", "f_b5_sequence_gap", "c_b5_sequence_gap", "b5_sequence_gap_lift"),
        ("Combined", "OFF", "f_combined_off_gain", "c_combined_off_gain", "combined_off_gain_lift"),
        ("Combined", "SHUFFLED", "f_combined_sequence_gap", "c_combined_sequence_gap", "combined_sequence_gap_lift"),
    )
    lines = [
        "| Link | Intervention | Fixed effect | C effect | Fixed-to-C lift | Paired 95% CI of lift |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for link, intervention, fixed_name, c_name, lift_name in definitions:
        lift = rows[lift_name]
        lines.append(
            f"| {link} | {intervention} | {_contrast_point(rows, fixed_name):+.12f} | "
            f"{_contrast_point(rows, c_name):+.12f} | {_contrast_point(rows, lift_name):+.12f} | "
            f"{ci_text(lift)} |"
        )
    return lines


def render_report(
    *,
    summary_path: str | os.PathLike[str],
    representation_path: str | os.PathLike[str],
    postflight_audit_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> str:
    output = Path(output_path).resolve()
    if output.name != "EXPERIMENT_2D5C_FINAL_REPORT.md":
        _fail("report output must be named EXPERIMENT_2D5C_FINAL_REPORT.md")
    if not (
        output.parent == Path(summary_path).resolve().parent
        == Path(representation_path).resolve().parent
        == Path(postflight_audit_path).resolve().parent
    ):
        _fail("final report inputs and output must share one result directory")
    summary = read_json(summary_path)
    representation = read_json(representation_path)
    postflight = read_json(postflight_audit_path)
    postflight_checks = validate_final_audit_document(postflight)
    if not _all_true(postflight_checks):
        _fail(f"final report requires passed postflight: {sorted(k for k, v in postflight_checks.items() if not v)}")
    identities = postflight.get("input_artifact_identity", {})
    identity_checks = {
        "summary": identities.get("summary") == file_identity(summary_path),
        "representation": identities.get("representation") == file_identity(representation_path),
        "postflight_git": _all_true(postflight.get("git_checks", {}))
        and _all_true(validate_git_verification(
            postflight.get("git_verification", {}), summary
        )),
        "postflight_guard": _all_true(postflight.get("runpod_stop_checks", {}))
        and _all_true(validate_stopped_report(
            postflight.get("runpod_stop_verification", {})
        )),
        "exact_stopped_state": postflight.get("runpod_stop_verification", {})
        .get("pod", {}).get("desiredStatus") == "EXITED",
        "retained_volume": postflight.get("runpod_stop_verification", {})
        .get("network_volume", {}).get("id") == VOLUME_ID,
        "representation_passed": representation.get("passed") is True,
    }
    if not _all_true(identity_checks):
        _fail(f"final report evidence binding failed: {sorted(k for k, v in identity_checks.items() if not v)}")

    rows = summary.get("bootstrap", {}).get("contrasts", {})
    if not isinstance(rows, Mapping):
        _fail("summary has no paired bootstrap contrasts")
    missing = sorted(set(REQUIRED_REPORT_CONTRASTS) - set(rows))
    if missing:
        _fail(f"summary is missing required contrasts: {missing}")
    for name in REQUIRED_REPORT_CONTRASTS:
        _contrast_point(rows, name)
        ci_text(rows[name])
    longitudinal = summary.get("longitudinal", {})
    c_longitudinal = longitudinal.get("c", {}) if isinstance(longitudinal, Mapping) else {}
    if set(c_longitudinal) != {"0", "48", "96", "144", "191"}:
        _fail("summary longitudinal C milestones are not exactly 0/48/96/144/191")
    memory = summary.get("bf16_persistent_state", {})
    checkpoint = summary.get("final_checkpoint", {})
    git = postflight["git_verification"]
    recovery = summary.get("recovery", {})
    fixed_ce = _finite_number(summary.get("fixed_all_real_ce"), "Fixed ALL_REAL CE")
    c_ce = _finite_number(summary.get("c_all_real_ce"), "C ALL_REAL CE")
    if summary.get("classification") not in CLASSIFICATIONS:
        _fail("summary classification is not a registered 2D5C outcome")
    if not is_sha256(checkpoint.get("sha256")) or not checkpoint.get("path"):
        _fail("summary final checkpoint identity is incomplete")
    initial_shock = _finite_number(
        recovery.get("initial_shock", {}).get("point_estimate"), "initial geometry shock"
    )
    recovery_value = recovery.get("recovery_fraction", {}).get("point_estimate")
    recovery_text = "undefined" if recovery_value is None else f"{_finite_number(recovery_value, 'recovery fraction'):+.12f}"

    lines = [
        "# Experiment 2D5C — Fixed-writer B3/B5 W2 representation pressure", "",
        f"**Classification:** {summary.get('classification')}", "",
        f"- Fixed-100M ALL_REAL CE: `{fixed_ce:.12f}`",
        f"- C-W2/W2 ALL_REAL CE: `{c_ce:.12f}`",
        f"- Fixed−C: `{_contrast_point(rows, 'architecture_fixed_minus_c'):+.12f}`; paired 95% CI {ci_text(rows['architecture_fixed_minus_c'])}",
        f"- C B3 OFF gain / SHUFFLED gap: `{_contrast_point(rows, 'c_b3_off_gain'):+.12f}` / `{_contrast_point(rows, 'c_b3_sequence_gap'):+.12f}`",
        f"- C B5 OFF gain / SHUFFLED gap: `{_contrast_point(rows, 'c_b5_off_gain'):+.12f}` / `{_contrast_point(rows, 'c_b5_sequence_gap'):+.12f}`",
        f"- C combined OFF gain / SHUFFLED gap: `{_contrast_point(rows, 'c_combined_off_gain'):+.12f}` / `{_contrast_point(rows, 'c_combined_sequence_gap'):+.12f}`",
        f"- B3 pressure lifts (OFF / SHUFFLED): `{_contrast_point(rows, 'b3_off_gain_lift'):+.12f}` / `{_contrast_point(rows, 'b3_sequence_gap_lift'):+.12f}`",
        f"- B5 pressure lifts (OFF / SHUFFLED): `{_contrast_point(rows, 'b5_off_gain_lift'):+.12f}` / `{_contrast_point(rows, 'b5_sequence_gap_lift'):+.12f}`",
        f"- Combined pressure lifts (OFF / SHUFFLED): `{_contrast_point(rows, 'combined_off_gain_lift'):+.12f}` / `{_contrast_point(rows, 'combined_sequence_gap_lift'):+.12f}`",
        f"- Initial geometry shock: `{initial_shock:+.12f}`",
        f"- Recovery fraction: `{recovery_text}`",
        f"- BF16 logical / measured physical reduction: `{memory['logical']['reduction_bytes']:,}` / `{memory['allocated_unique_storage']['reduction_bytes']:,}` bytes",
        f"- Final checkpoint: `{checkpoint['path']}`",
        f"- Final checkpoint SHA-256: `{checkpoint['sha256']}`",
        "- Audit: `PASS`",
        f"- Git: `{BRANCH}` / `{git['scientific_results_commit']}` / `{FINAL_TAG}`",
        f"- RunPod `{POD_ID}`: `STOPPED`; volume `{VOLUME_ID}` retained",
        "",
        "## Scientific interpretation", "",
        summary["recommendation"]["reason"] + ". No A, B, Fixed, or 250M continuation was executed.",
        "",
        "## True-incremental longitudinal core", "",
        "| C local update | ALL_REAL CE | B3 OFF | B3 SHUFFLED | B5 OFF | B5 SHUFFLED | Combined OFF | Combined SHUFFLED |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for update in ("0", "48", "96", "144", "191"):
        row = c_longitudinal[update]
        lines.append(
            f"| {update} | {row['all_real_ce']:.12f} | {row['b3_recurrent_gain']:+.12f} | {row['b3_sequence_gap']:+.12f} | "
            f"{row['b5_recurrent_gain']:+.12f} | {row['b5_sequence_gap']:+.12f} | {row['combined_recurrent_gain']:+.12f} | {row['combined_sequence_gap']:+.12f} |"
        )
    lines += [
        "", "## Final large-panel paired contrasts", "",
        "| Contrast | Estimate | Paired 95% CI | Positive sequences | Paired SE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {_contrast_point(rows, name):+.12f} | {ci_text(row)} | "
            f"{row['positive_per_sequence_differences']} / 2048 | {row['paired_standard_error']:.12f} |"
        )
    lines += [
        "", "## Fixed-versus-C pressure-lift table", "",
        "Positive lift means the intervention cost is larger for C than for the matched Fixed control. Each lift is a paired-sequence difference-in-differences using the shared bootstrap index stream.",
        "",
        *_pressure_table(rows),
        "", "## Lag, gradient, and contribution diagnostics", "",
        "All seven requested models passed the 32-sequence diagnostic audit. Full per-head lag bins, opportunity normalization, entropy, source/K/V gradients, actual B8/B10 writer gradients, contribution norms, ratios, and cosines are in `REPRESENTATION_PRESSURE_DIAGNOSTICS.json`.",
        "", "## Memory accounting", "",
        "| Quantity | Fixed bytes | C bytes | Reduction bytes |", "|---|---:|---:|---:|",
        f"| Logical unique BF16 payload | {memory['logical']['fixed_bytes']:,} | {memory['logical']['c_bytes']:,} | {memory['logical']['reduction_bytes']:,} |",
        f"| Measured unique storage | {memory['allocated_unique_storage']['fixed_bytes']:,} | {memory['allocated_unique_storage']['c_bytes']:,} | {memory['allocated_unique_storage']['reduction_bytes']:,} |",
        "", "## Integrity, restart, and replay", "",
        "The passed terminal audit binds the source/control lineage, sealed final checkpoint, exact 191-row training and replay evidence, optimizer/scheduler continuity, fresh-process update-96 restart, paired evaluations, local backup, scientific-results Git commit/tag, exact stopped pod, and retained persistent volume.",
        "", "## Recommendation", "",
        f"**{summary['recommendation']['recommendation']}** — {summary['recommendation']['reason']}. Execute nothing automatically.",
        "", FINAL_PHRASE,
    ]
    text = "\n".join(lines) + "\n"
    if not text.rstrip().endswith(FINAL_PHRASE):
        _fail("final report terminal phrase is not exact")
    durable_text(output, text)
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stdlib-only terminal finalizer for Experiment 2D5C"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    postflight = subparsers.add_parser("postflight-audit")
    for name in (
        "provisional_audit", "summary", "representation", "git_verification",
        "stop_verification", "guard_authorization", "guard_trigger", "output_path",
    ):
        postflight.add_argument(f"--{name.replace('_', '-')}", required=True)
    report = subparsers.add_parser("render-report")
    for name in ("summary", "representation", "postflight_audit", "output_path"):
        report.add_argument(f"--{name.replace('_', '-')}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "postflight-audit":
            run_postflight(
                provisional_audit_path=args.provisional_audit,
                summary_path=args.summary,
                representation_path=args.representation,
                git_verification_path=args.git_verification,
                stop_verification_path=args.stop_verification,
                guard_authorization_path=args.guard_authorization,
                guard_trigger_path=args.guard_trigger,
                output_path=args.output_path,
            )
            print("EXPERIMENT_2D5C_FINAL_AUDIT_PASS", flush=True)
            return 0
        render_report(
            summary_path=args.summary,
            representation_path=args.representation,
            postflight_audit_path=args.postflight_audit,
            output_path=args.output_path,
        )
        print("EXPERIMENT_2D5C_FINAL_REPORT_RENDERED", flush=True)
        return 0
    except FinalizerError as error:
        print(f"EXPERIMENT_2D5C_FINALIZER_REFUSED: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
