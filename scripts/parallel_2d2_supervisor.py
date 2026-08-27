#!/usr/bin/env python3
"""Run-scoped supervisor for the fixed four-lane experiment matrix.

This supervisor launches and observes scientific lanes.  Its finalization
boundary performs read-only Git/report/checkpoint/backup verification, but it
deliberately has no pod-stop, pod-delete, Git-mutation, reporting, or backup-
creation capability.  The heartbeat is left alive after science so those
later coordinator phases remain visible.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
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
SOURCE_CHECKPOINTS = {
    "2D2B": "/workspace/exp2d2b_run/checkpoints/scientific_update_0191.pt",
    "2D2D": "/workspace/exp2d2d_run/checkpoints/scientific_update_0191.pt",
    "2D2E": "/workspace/exp2d2e_run/checkpoints/scientific_update_0191.pt",
}
HISTORICAL_RESULT_DIRECTORIES = {
    "2D2B": "results/experiment_2d2b_full_b12_b1_recurrent_bank",
    "2D2D": "results/experiment_2d2d_b2_w32_b11_recurrent_992",
    "2D2E": "results/experiment_2d2e_b3_w64_b10_recurrent_960",
}
TRAINING_ARTIFACT_MINIMUM = {
    "FINAL_REPORT.md",
    "FINAL_AUDIT.json",
    "result_summary.json",
    "source_manifest.json",
    "architecture_manifest.json",
    "parameter_manifest.json",
    "training_metrics.jsonl",
    "milestone_validation.json",
    "paired_controls.json",
    "gate_diagnostics.json",
    "attention_diagnostics.json",
    "temporal_gradient_diagnostics.json",
    "incremental_validation.json",
    "incremental_cache_audit.json",
    "memory_accounting.json",
    "stability_8pass.json",
    "performance.json",
    "checkpoint_manifest.json",
    "commands_and_runtime.json",
    "storage_cleanup_manifest.json",
    "HEARTBEAT.json",
    "UNATTENDED_FINAL_HANDOFF.md",
}
C1_ARTIFACT_MINIMUM = {
    "C1_FINAL_REPORT.md",
    "subset_manifest.json",
    "paired_results.json",
    "bootstrap_results.json",
    "FINAL_AUDIT.json",
}
CANONICAL_ORIGIN_URL = "https://github.com/rahulmeena/build-nanogpt.git"
GIT_EXECUTABLE = "/usr/bin/git"
SSH_KEYGEN_EXECUTABLE = "/usr/bin/ssh-keygen"
PS_EXECUTABLE = "/bin/ps"
NVIDIA_SMI_EXECUTABLE = "/usr/bin/nvidia-smi"
SANITIZED_TOOL_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C",
    "LC_ALL": "C",
}
MASTER_FINALIZATION_IMPLEMENTATION_TAG = (
    "parallel-2d2-master-finalization-implementation-v2"
)
CHECKPOINT_PERSIST_LOCK = (
    "/workspace/parallel_2d2_master/locks/checkpoint_persist.lock"
)
FINAL_CHECKPOINT_CONTRACTS = {
    "2D2F": {
        "updates_key": "completed_2d2f_updates",
        "targets_key": "cumulative_2d2_targets",
        "targets": 350_748_672,
        "strict_keys": {
            "fields_exact", "schema", "updates", "additional_targets",
            "cumulative_targets", "training_state", "loader_state",
            "next_batch", "next_stream", "metadata", "architecture",
            "source", "model_keys", "gate_duplicate", "b3_gate_duplicate",
            "model_finite", "optimizer_finite", "weight_tying", "passed",
        },
        "persistence_schema": "2d2f_local_stage",
    },
    "2D2G": {
        "targets": 100_139_008,
        "strict_keys": {
            "schema", "stage", "updates", "targets", "metadata",
            "architecture", "model_keys", "no_b2_gate", "rng",
            "strict_model_load", "strict_optimizer_load", "model_finite",
            "optimizer_finite", "loader_next_batch", "loader_next_stream",
            "matched_cursor", "passed",
        },
        "next_batch": "91fa2cae4e6e52cfddd2b470175ec704f0548b447f02861917ec548736fe18e7",
        "next_stream": "4da6fed71755e523030a2d8e9e7cc96d19691a8c9b3ac8c490426bafe3d44e82",
        "persistence_schema": "2d2g_manifest",
    },
    "2D2H": {
        "updates_key": "completed_2d2h_updates",
        "targets_key": "cumulative_2d2_targets",
        "targets": 250_609_664,
        "strict_keys": {
            "fields_exact", "schema", "updates", "additional_targets",
            "cumulative_targets", "training_state", "loader_state",
            "next_batch", "next_stream", "metadata", "architecture",
            "source", "model_keys", "b2_gate_duplicate", "model_finite",
            "optimizer_finite", "weight_tying", "passed",
        },
        "persistence_schema": "2d2h_full_local_stage",
    },
    "2D2I": {
        "updates_key": "completed_2d2i_updates",
        "targets_key": "cumulative_2d2_targets",
        "targets": 450_887_680,
        "strict_keys": {
            "fields_exact", "schema", "updates", "additional_targets",
            "cumulative_targets", "training_state", "loader_state",
            "next_batch", "next_stream", "metadata", "architecture",
            "source", "model_keys", "gate_duplicate", "b2_gate_duplicate",
            "b3_gate_duplicate", "b4_gate_duplicate", "model_finite",
            "optimizer_finite", "weight_tying", "passed",
        },
        "persistence_schema": "2d2i_staged_path",
    },
}
FINALIZATION_EVIDENCE_FILES = {
    "git": "FINAL_GIT_EVIDENCE.json",
    "report": "FINAL_REPORT_EVIDENCE.json",
    "backup": "FINAL_LOCAL_BACKUP_EVIDENCE.json",
    "backup_signature": "FINAL_LOCAL_BACKUP_EVIDENCE.json.sig",
}
FINAL_BACKUP_SIGNER_PUBLIC_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIIXLAAZmfcC7U4EgcyySrSocc4AyMoxLOGhZ2e2ViRqZ"
)
FINAL_BACKUP_SIGNER_FINGERPRINT = (
    "SHA256:X3HEYHE+azIkwSRU/pcV4I4PqKHtIezD2yViCRzSvRM"
)
FINAL_BACKUP_SIGNER_PRINCIPAL = "rahul-local-backup"
FINAL_BACKUP_SIGNER_NAMESPACE = "parallel-2d2-final-backup"
MASTER_CLAIMS_BEGIN = "<!-- PARALLEL_2D2_MASTER_CLAIMS_V1_BEGIN -->"
MASTER_CLAIMS_END = "<!-- PARALLEL_2D2_MASTER_CLAIMS_V1_END -->"
MASTER_MATRIX_BEGIN = "<!-- PARALLEL_2D2_MASTER_MATRIX_V1_BEGIN -->"
MASTER_MATRIX_END = "<!-- PARALLEL_2D2_MASTER_MATRIX_V1_END -->"
MASTER_SCIENTIFIC_QUESTIONS = {
    "M1": "Did large frozen 2D2E confirmation reproduce positive B3 gain?",
    "M2": "Did it reproduce positive Real-vs-Shuffled gap?",
    "M3": "Were both bootstrap CIs strictly positive?",
    "M4": "Does removing B11→B2 during B3 training improve B3 recurrence?",
    "M5": "Does B3 recurrence become stronger with B2 W32 or B2 W1024?",
    "M6": "Does B11→B2 become useful when B12→B1 is removed?",
    "M7": "Is B1 therefore uniquely important as the recurrent-memory entry point?",
    "M8": "Does B9→B4 establish useful recurrence at W128?",
    "M9": "Are deeper recurrent-link gains increasing, decreasing, or irregular?",
    "M10": "Which recurrent rings can now be deleted without measurable degradation?",
    "M11": "Which tested architecture gives the best validation CE?",
    "M12": "Which gives the best CE per byte of inference state?",
    "M13": "Which gives the best CE per recurrent-state ring?",
    "M14": "What architecture should be the new canonical branch?",
    "M15": "What exactly ONE experiment should run next?",
}
MASTER_SCIENTIFIC_ANSWER_CODES = {
    "M1": "YES_DIRECTIONAL",
    "M2": "YES_DIRECTIONAL",
    "M3": "NO_BOTH_INTERVALS_CROSS_ZERO",
    "M4": "YES_2D2F_IMPROVES_TRUE_B3_GAIN",
    "M5": "B2_W32_2D2F",
    "M6": "NO_HARMFUL_TRUE_INCREMENTAL",
    "M7": "YES_FOUNDATIONAL_NOT_EXCLUSIVE",
    "M8": "YES_POSITIVE_UTILITY",
    "M9": "IRREGULAR_NON_MONOTONIC",
    "M10": "DELETE_B11_TO_B2_ONLY",
    "M11": "2D2G",
    "M12": "2D2F",
    "M13": "2D2G",
    "M14": "2D2F",
    "M15": "FROZEN_2D2F_VS_2D2G_LARGE_TRUE_INCREMENTAL_CONFIRMATION",
}
MASTER_SCIENTIFIC_CONCLUSIONS = {
    "M1": "Yes—the large frozen 2D2E confirmation reproduced a positive B3 gain.",
    "M2": "Yes—the large frozen 2D2E confirmation reproduced a positive Real-vs-Shuffled gap.",
    "M3": "No—both paired bootstrap confidence intervals cross zero.",
    "M4": "Yes—removing B11→B2 during B3 training increased the method-matched true B3 gain.",
    "M5": "B2 W32 in 2D2F produced the stronger method-matched true B3 gain.",
    "M6": "No—B11→B2 was harmful in true incremental evaluation after B12→B1 removal.",
    "M7": "Yes as a foundational entry point, but not as the only useful link because B3 and B4 recurrence also established positive utility.",
    "M8": "Yes—B9→B4 at W128 established positive true incremental recurrent utility.",
    "M9": "Destination-depth utility is irregular and non-monotonic.",
    "M10": "Delete only the B11 recurrent-state ring; retain the B12, B10, and B9 rings.",
    "M11": "2D2G has the lowest method-consistent true incremental validation CE.",
    "M12": "2D2F gives the best CE/state tradeoff: it is the smallest tested useful state and pays only a small CE cost versus 2D2G.",
    "M13": "2D2G gives the best CE/ring tradeoff among the tested architectures.",
    "M14": "Use 2D2F as the new canonical branch because it combines the strongest B3 true gain with the smallest useful state.",
    "M15": "Run one frozen, larger true-incremental head-to-head confirmation of 2D2F versus 2D2G; do not adapt either checkpoint.",
}
FINAL_EVIDENCE_FRESHNESS = timedelta(minutes=10)
FINAL_EVIDENCE_CLOCK_SKEW = timedelta(minutes=2)
FINAL_GIT_REPOSITORIES = {
    "MASTER": {
        "worktree": "master",
        "branch": "codex/parallel-2d2-master",
        "minimum_commit": "01355cf3ccd01df5e34b775df4166f4a3c14fd3f",
        "implementation_tag": MASTER_FINALIZATION_IMPLEMENTATION_TAG,
        "implementation_commit": None,
        "tracked_paths": [
            "MASTER_FINAL_REPORT.md",
            "results/experiment_2d2e_c1_large_true_self_confirmation",
        ],
    },
    "2D2F": {
        "worktree": "2d2f",
        "branch": "experiment-2d2f-no-b2-recurrence-b3-w64",
        "minimum_commit": "afdde8b75eb207cc2181821a147a0430413611ea",
        "implementation_tag": None,
        "implementation_commit": "afdde8b75eb207cc2181821a147a0430413611ea",
        "allowed_execution_commits": (
            "afdde8b75eb207cc2181821a147a0430413611ea",
        ),
        "execution_commit_source": "preflight",
        "execution_environment_commit_required": True,
        "implementation_fingerprint_format": "canonical_file_map_sha256",
        "implementation_fingerprint_aggregate": (
            "982f6e0d2cbd4a055b4c53c2363b831403ce54bbe926f69b677749d3eb6ffd23"
        ),
        "implementation_fingerprint_files": (
            "configs/exp2d2f_no_b2_recurrence_b3_w64.json",
            "scripts/experiment_2d0.py",
            "scripts/experiment_2d0d.py",
            "scripts/experiment_2d1.py",
            "scripts/experiment_2d2a.py",
            "scripts/experiment_2d2a_core.py",
            "scripts/experiment_2d2d.py",
            "scripts/experiment_2d2d_core.py",
            "scripts/experiment_2d2f.py",
            "scripts/experiment_2d2f_core.py",
            "scripts/smoke_test.py",
            "tests/test_experiment_2d2f_core.py",
            "tests/test_experiment_2d2f_driver.py",
            "train_gpt2.py",
        ),
        "tracked_paths": ["results/experiment_2d2f_no_b2_recurrence_b3_w64"],
    },
    "2D2G": {
        "worktree": "2d2g",
        "branch": "experiment-2d2g-b2-full-b3-w64",
        "minimum_commit": "41479f75060d7ad8debfaae06418be929281e54d",
        "implementation_tag": None,
        "implementation_commit": "41479f75060d7ad8debfaae06418be929281e54d",
        "allowed_execution_commits": (
            "41479f75060d7ad8debfaae06418be929281e54d",
        ),
        "execution_commit_source": "summary",
        "implementation_fingerprint_format": "raw_digest_concat_sha256",
        "implementation_fingerprint_aggregate": (
            "6c0ca0d078520635e8e366ee4cc8d12042783d43a9043d0d5359b6fb3f7e7f9d"
        ),
        "implementation_fingerprint_files": (
            "configs/exp2d2g_b2_full_b3_w64.json",
            "scripts/experiment_2d2g.py",
            "scripts/experiment_2d2g_core.py",
            "tests/test_experiment_2d2g_core.py",
            "tests/test_experiment_2d2g_driver.py",
        ),
        "tracked_paths": ["results/experiment_2d2g_b2_full_b3_w64"],
    },
    "2D2H": {
        "worktree": "2d2h",
        "branch": "experiment-2d2h-no-b1-recurrence-b2-w32",
        "minimum_commit": "1a6979130994c2959728e0335c30b4fcb9502d24",
        "implementation_tag": None,
        "implementation_commit": "1a6979130994c2959728e0335c30b4fcb9502d24",
        "allowed_execution_commits": (
            "69651f0ba31c55fc175b15c9ed08879868b81866",
        ),
        "execution_commit_source": "preflight",
        "correction_commit": "1a6979130994c2959728e0335c30b4fcb9502d24",
        "correction_fingerprint_aggregate": (
            "68a92dd2baffb8a027204b5fdb31c17c02c79182c9a9dfc02276e41c749eb350"
        ),
        "implementation_fingerprint_format": "canonical_file_map_sha256",
        "implementation_fingerprint_aggregate": (
            "fae91663242240dbfcfe749763e12200a5cc3d76cabcb493c67fb5784dc40e54"
        ),
        "implementation_fingerprint_files": (
            "configs/exp2d2h_no_b1_recurrence_b2_w32.json",
            "scripts/experiment_2d0.py",
            "scripts/experiment_2d0d.py",
            "scripts/experiment_2d1.py",
            "scripts/experiment_2d2a.py",
            "scripts/experiment_2d2a_core.py",
            "scripts/experiment_2d2b.py",
            "scripts/experiment_2d2b_core.py",
            "scripts/experiment_2d2h.py",
            "scripts/experiment_2d2h_core.py",
            "scripts/smoke_test.py",
            "tests/test_experiment_2d2h_core.py",
            "tests/test_experiment_2d2h_driver.py",
            "train_gpt2.py",
        ),
        "tracked_paths": ["results/experiment_2d2h_no_b1_recurrence_b2_w32"],
    },
    "2D2I": {
        "worktree": "2d2i",
        "branch": "experiment-2d2i-b4-w128-b9-recurrent",
        "minimum_commit": "8b05f13c13acd7e517e87044aeafa5dd0fdab911",
        "implementation_tag": None,
        "implementation_commit": "8b05f13c13acd7e517e87044aeafa5dd0fdab911",
        "allowed_execution_commits": (
            "8b05f13c13acd7e517e87044aeafa5dd0fdab911",
        ),
        "execution_commit_source": "preflight",
        "execution_environment_commit_required": True,
        "implementation_fingerprint_format": "canonical_file_map_sha256",
        "implementation_fingerprint_aggregate": (
            "afffbb2dadf7f99d573d62f3c47aa63e8df47620ee0748f8bb19a17e675a5027"
        ),
        "implementation_fingerprint_files": (
            "configs/exp2d2i_b4_w128_b9_recurrent_896.json",
            "scripts/experiment_2d0.py",
            "scripts/experiment_2d0d.py",
            "scripts/experiment_2d1.py",
            "scripts/experiment_2d2a.py",
            "scripts/experiment_2d2a_core.py",
            "scripts/experiment_2d2e.py",
            "scripts/experiment_2d2e_core.py",
            "scripts/experiment_2d2i.py",
            "scripts/experiment_2d2i_core.py",
            "scripts/smoke_test.py",
            "tests/test_experiment_2d2i_core.py",
            "tests/test_experiment_2d2i_driver.py",
            "train_gpt2.py",
        ),
        "tracked_paths": ["results/experiment_2d2i_b4_w128_b9_recurrent"],
    },
}
SCIENTIFIC_PROCESS_MARKERS = (
    "experiment_2d2e_c1.py",
    "experiment_2d2f.py",
    "experiment_2d2g.py",
    "experiment_2d2h.py",
    "experiment_2d2i.py",
    "parallel_2d2_lane0.sh",
    "parallel_2d2_lane1.sh",
    "parallel_2d2_lane2.sh",
    "parallel_2d2_lane3.sh",
)
STOP_PREFLIGHT_KEYS = {
    "authenticated", "authenticated_list_probe", "authenticated_pod_identity_response",
    "checked_utc", "desired_status", "exact_stop_command", "exact_stop_target",
    "gpu_count", "mechanism", "network_volume_preservation_required", "passed",
    "persistent_volume_delete_authorized", "pod_delete_authorized",
    "pod_delete_forbidden", "pod_id", "pod_name", "runtime_status", "schema",
    "secret_recorded", "stop_credential_available", "volume_id",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


UTC_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|\+00:00)"
)


def parse_canonical_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise RuntimeError(f"{label} is not a valid UTC timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise RuntimeError(f"{label} is not UTC")
    return parsed


def require_timestamp_not_future(value: datetime, label: str) -> None:
    if value > datetime.now(timezone.utc) + FINAL_EVIDENCE_CLOCK_SKEW:
        raise RuntimeError(f"{label} is implausibly in the future")


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
    """Atomically create read-only evidence without replacing an existing file."""
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
            os.fchmod(handle.fileno(), 0o444)
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
        published = path.lstat()
        if (
            not stat.S_ISREG(published.st_mode)
            or path.is_symlink()
            or published.st_mode & 0o222
            or path.read_bytes() != content
        ):
            raise RuntimeError(
                f"published immutable evidence identity differs: {path}"
            )
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


def lexical_absolute(path: Path | str) -> Path:
    """Return an absolute path without resolving symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def require_symlink_free_regular_file(
    path: Path, trusted_root: Path, label: str, *, require_nonempty: bool = True
) -> Path:
    """Require a regular file with no symlink below an already trusted root."""
    candidate = lexical_absolute(path)
    root = lexical_absolute(trusted_root)
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its trusted root: {candidate}") from error
    current = root
    try:
        root_stat = current.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} trusted root is unavailable: {root}: {error}") from error
    if not stat.S_ISDIR(root_stat.st_mode) or current.is_symlink():
        raise RuntimeError(f"{label} trusted root is not a real directory: {root}")
    for index, component in enumerate(relative.parts):
        if component in {"", ".", ".."}:
            raise RuntimeError(f"{label} contains an unsafe path component")
        current = current / component
        try:
            observed = current.lstat()
        except OSError as error:
            raise RuntimeError(f"{label} is unavailable: {current}: {error}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(f"{label} contains a symlink component: {current}")
        final = index == len(relative.parts) - 1
        if final:
            if not stat.S_ISREG(observed.st_mode):
                raise RuntimeError(f"{label} is not a regular file: {current}")
            if require_nonempty and observed.st_size <= 0:
                raise RuntimeError(f"{label} is empty: {current}")
        elif not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError(f"{label} parent is not a directory: {current}")
    if not relative.parts:
        raise RuntimeError(f"{label} unexpectedly names its trusted directory")
    return candidate


def symlink_free_tree_files(
    root: Path, trusted_root: Path, label: str, *, require_nonempty: bool = False
) -> set[Path]:
    """Enumerate every nonempty regular file in a tree without following links."""
    tree = lexical_absolute(root)
    trusted = lexical_absolute(trusted_root)
    try:
        tree.relative_to(trusted)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its trusted root: {tree}") from error
    # Anchor the tree itself by validating a temporary sentinel-like child path
    # component-by-component, then walk without following directory symlinks.
    current = trusted
    for component in tree.relative_to(trusted).parts:
        current = current / component
        try:
            observed = current.lstat()
        except OSError as error:
            raise RuntimeError(f"{label} is unavailable: {current}: {error}") from error
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError(f"{label} has a non-directory/symlink component: {current}")
    files: set[Path] = set()
    for base, directories, names in os.walk(tree, followlinks=False):
        base_path = Path(base)
        for name in tuple(directories):
            child = base_path / name
            observed = child.lstat()
            if stat.S_ISLNK(observed.st_mode):
                raise RuntimeError(f"{label} contains a directory symlink: {child}")
            if not stat.S_ISDIR(observed.st_mode):
                raise RuntimeError(f"{label} contains a non-directory entry: {child}")
        for name in names:
            child = base_path / name
            require_symlink_free_regular_file(
                child, trusted, label, require_nonempty=require_nonempty
            )
            files.add(lexical_absolute(child))
    if not files:
        raise RuntimeError(f"{label} contains no files: {tree}")
    return files


def stable_file_identity(path: Path) -> dict:
    """Hash one regular file through a no-follow descriptor and bind its inode."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        lexical = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"cannot open stable evidence file {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lexical.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (lexical.st_dev, lexical.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"stable evidence path changed before open: {path}")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns)
            or size != before.st_size
        ):
            raise RuntimeError(f"stable evidence file changed while hashed: {path}")
    finally:
        os.close(descriptor)
    return {
        "path": str(lexical_absolute(path)),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "mode": stat.S_IMODE(before.st_mode),
    }


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
                NVIDIA_SMI_EXECUTABLE,
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            env=SANITIZED_TOOL_ENVIRONMENT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot audit live NVIDIA compute processes: {error}") from error
    return [line.strip() for line in output.splitlines() if line.strip()]


def _require_exact_true_checks(value: object, expected: set[str], label: str) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or any(flag is not True for flag in value.values())
    ):
        raise RuntimeError(f"{label} is not the exact all-true strict check map")
    return value


def _validate_standard_final_checkpoint_verification(
    experiment: str,
    path: Path,
    observed_sha: str,
    size: int,
    verification: dict,
    contract: dict,
) -> dict:
    base_keys = {
        "checkpoint", "sha256", "bytes", contract["updates_key"],
        contract["targets_key"], "next_global_batch_sha256",
        "next_global_batch_stream_sha256", "strict_reopen", "passed",
    }
    schema = contract["persistence_schema"]
    if schema == "2d2f_local_stage":
        expected_keys = base_keys | {
            "local_stage", "persistent_copy_lock", "persistent_copy_sha_verified"
        }
    elif schema == "2d2h_full_local_stage":
        expected_keys = base_keys | {"local_stage", "persisted_under_global_lock"}
    elif schema == "2d2i_staged_path":
        expected_keys = base_keys | {
            "local_staged_checkpoint", "local_and_persistent_sha_match", "persist_lock"
        }
    else:
        raise RuntimeError(f"unknown persistent checkpoint schema: {experiment}")
    if not isinstance(verification, dict) or set(verification) != expected_keys:
        raise RuntimeError(f"{experiment} final checkpoint verification schema differs")
    cursor_values = (
        verification.get("next_global_batch_sha256"),
        verification.get("next_global_batch_stream_sha256"),
    )
    if (
        verification.get("checkpoint") != str(path.resolve())
        or verification.get("sha256") != observed_sha
        or verification.get("bytes") != size
        or verification.get(contract["updates_key"]) != 191
        or verification.get(contract["targets_key"]) != contract["targets"]
        or verification.get("passed") is not True
        or any(not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
               for value in cursor_values)
    ):
        raise RuntimeError(f"{experiment} final checkpoint verification identity differs")
    strict = _require_exact_true_checks(
        verification.get("strict_reopen"), contract["strict_keys"],
        f"{experiment} strict reopen",
    )
    if schema == "2d2f_local_stage":
        local = verification.get("local_stage")
        expected_local_stage = (
            "/tmp/parallel_2d2_ephemeral/2d2f/"
            ".scientific_update_0191.local-stage.pt"
        )
        if (
            not isinstance(local, dict)
            or set(local) != {"path", "sha256", "bytes", "strict_reopen_passed"}
            or local.get("path") != expected_local_stage
            or local.get("path") == str(path.resolve())
            or local.get("sha256") != observed_sha
            or local.get("bytes") != size
            or local.get("strict_reopen_passed") is not True
            or verification.get("persistent_copy_lock") != CHECKPOINT_PERSIST_LOCK
            or verification.get("persistent_copy_sha_verified") is not True
        ):
            raise RuntimeError(f"{experiment} persistent-copy verification differs")
    elif schema == "2d2h_full_local_stage":
        local = verification.get("local_stage")
        if (
            not isinstance(local, dict)
            or set(local) != base_keys
            or local.get("sha256") != observed_sha
            or local.get("bytes") != size
            or local.get(contract["updates_key"]) != 191
            or local.get(contract["targets_key"]) != contract["targets"]
            or local.get("passed") is not True
            or local.get("checkpoint")
            != "/tmp/parallel_2d2_ephemeral/2d2h/scientific_update_0191.pt"
            or local.get("strict_reopen") != strict
            or local.get("next_global_batch_sha256") != cursor_values[0]
            or local.get("next_global_batch_stream_sha256") != cursor_values[1]
            or verification.get("persisted_under_global_lock") is not True
        ):
            raise RuntimeError(f"{experiment} local-stage/persistent binding differs")
    else:
        local_path = verification.get("local_staged_checkpoint")
        if (
            not isinstance(local_path, str)
            or not local_path.startswith("/tmp/parallel_2d2_ephemeral/2d2i/")
            or Path(local_path).name != path.name
            or verification.get("local_and_persistent_sha_match") is not True
            or verification.get("persist_lock") != CHECKPOINT_PERSIST_LOCK
        ):
            raise RuntimeError(f"{experiment} staged/persistent binding differs")
    output = Path(FINAL_REPORTS[experiment]).resolve().parent
    summary_path = output / "result_summary.json"
    manifest_path = output / "checkpoint_manifest.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    scientific = manifest.get("scientific") if isinstance(manifest, dict) else None
    scientific_191 = scientific.get("191") if isinstance(scientific, dict) else None
    if (
        not isinstance(summary, dict)
        or summary.get("final_checkpoint") != str(path.resolve())
        or summary.get("final_checkpoint_sha256") != observed_sha
    ):
        raise RuntimeError(
            f"{experiment} result summary does not bind the fresh final checkpoint"
        )
    if scientific_191 != verification:
        raise RuntimeError(
            f"{experiment} scientific-191 manifest does not bind the fresh "
            "checkpoint verification"
        )
    return {
        "schema": schema,
        "updates": 191,
        "targets": contract["targets"],
        "strict_checks": strict,
        "result_summary": {
            "path": str(summary_path), "sha256": file_sha256(summary_path),
        },
        "checkpoint_manifest": {
            "path": str(manifest_path), "sha256": file_sha256(manifest_path),
            "scientific_191_exact": True,
        },
        "fresh_identity": {
            "checkpoint": str(path.resolve()),
            "sha256": observed_sha,
            "bytes": size,
            "next_global_batch_sha256": cursor_values[0],
            "next_global_batch_stream_sha256": cursor_values[1],
        },
        "passed": True,
    }


def _validate_2d2g_final_checkpoint_verification(
    path: Path,
    observed_sha: str,
    size: int,
    verification: dict,
    contract: dict,
) -> dict:
    strict = _require_exact_true_checks(
        verification, contract["strict_keys"], "2D2G strict reopen"
    )
    output = Path(FINAL_REPORTS["2D2G"]).resolve().parent
    manifest_path = output / "checkpoint_manifest.json"
    persistent_path = output / "persistent_final_checkpoint.json"
    summary_path = output / "result_summary.json"
    stage_b_metrics_path = output / "stage_b_training_metrics.jsonl"
    generic_metrics_path = output / "training_metrics.jsonl"
    stage_b_match_path = output / "stage_b_data_match.json"
    manifest = read_json(manifest_path)
    persisted = read_json(persistent_path)
    summary = read_json(summary_path)
    try:
        stage_b_metrics_bytes = stage_b_metrics_path.read_bytes()
        generic_metrics_bytes = generic_metrics_path.read_bytes()
        metric_lines = stage_b_metrics_bytes.decode("utf-8").splitlines()
        metrics = [json.loads(line) for line in metric_lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"2D2G Stage-B training metrics are invalid: {error}") from error
    targets_per_update = contract["targets"] // 191
    if (
        len(metric_lines) != 191
        or len(metrics) != 191
        or stage_b_metrics_bytes != generic_metrics_bytes
        or contract["targets"] % 191 != 0
        or any(
            not isinstance(row, dict)
            or row.get("stage") != "b"
            or row.get("local_update") != index
            or row.get("processed_stage_targets") != index * targets_per_update
            for index, row in enumerate(metrics, 1)
        )
        or metrics[-1].get("processed_stage_targets") != contract["targets"]
    ):
        raise RuntimeError("2D2G Stage-B metrics do not bind update 191 and exact targets")
    stage_b_match = read_json(stage_b_match_path)
    final_match = stage_b_match.get("update_191")
    if (
        not isinstance(final_match, dict)
        or set(final_match) != {
            "observed_next_global_batch_sha256",
            "observed_next_global_batch_stream_sha256",
            "expected",
            "exact",
        }
        or final_match.get("observed_next_global_batch_sha256") != contract["next_batch"]
        or final_match.get("observed_next_global_batch_stream_sha256") != contract["next_stream"]
        or final_match.get("expected") != [contract["next_batch"], contract["next_stream"]]
        or final_match.get("exact") is not True
        or stage_b_match.get("pending_stage_a") is not False
        or stage_b_match.get("passed") is not True
    ):
        raise RuntimeError("2D2G Stage-B data-match audit does not bind the final cursor")
    stage_b = manifest.get("stage_b") if isinstance(manifest, dict) else None
    local_191 = stage_b.get("191") if isinstance(stage_b, dict) else None
    persistent_191 = stage_b.get("191_persistent") if isinstance(stage_b, dict) else None
    local_keys = {
        "checkpoint", "sha256", "bytes", "next_global_batch_sha256",
        "next_global_batch_stream_sha256", "strict_reopen",
    }
    if not isinstance(local_191, dict) or set(local_191) != local_keys:
        raise RuntimeError("2D2G Stage-B-191 checkpoint manifest schema differs")
    if (
        local_191.get("sha256") != observed_sha
        or local_191.get("bytes") != size
        or local_191.get("strict_reopen") != strict
        or local_191.get("next_global_batch_sha256") != contract["next_batch"]
        or local_191.get("next_global_batch_stream_sha256") != contract["next_stream"]
        or not isinstance(local_191.get("checkpoint"), str)
        or not local_191["checkpoint"].startswith("/tmp/parallel_2d2_ephemeral/")
        or not isinstance(persisted, dict)
        or local_191.get("checkpoint") != persisted.get("local")
    ):
        raise RuntimeError("2D2G Stage-B-191 manifest does not bind the final checkpoint")
    if persistent_191 != persisted:
        raise RuntimeError("2D2G persistent checkpoint records differ")
    persistent_keys = {
        "local", "persistent", "local_sha256", "persistent_sha256", "bytes",
        "lock", "local_sidecar_audit", "persistent_sidecar_audit",
        "persistent_sha_verified_while_lock_held", "reused_existing_exact_checkpoint",
        "passed", "path_audit",
    }
    if not isinstance(persisted, dict) or set(persisted) != persistent_keys:
        raise RuntimeError("2D2G persistent checkpoint schema differs")
    sidecar_check_keys = {
        "sha_sidecar_present", "verification_sidecar_present",
        "sha_sidecar_matches", "expected_sha_matches", "verification_passed",
    }
    for name in ("local_sidecar_audit", "persistent_sidecar_audit"):
        row = persisted.get(name)
        expected_checkpoint = persisted.get("local" if name.startswith("local") else "persistent")
        if (
            not isinstance(row, dict)
            or set(row) != {"checkpoint", "sha256", "sha_sidecar", "verification_sidecar", "checks", "passed"}
            or row.get("checkpoint") != expected_checkpoint
            or row.get("sha256") != observed_sha
            or row.get("sha_sidecar") != f"{expected_checkpoint}.sha256"
            or row.get("verification_sidecar")
            != f"{expected_checkpoint}.verification.json"
            or row.get("passed") is not True
        ):
            raise RuntimeError(f"2D2G {name} schema/identity differs")
        _require_exact_true_checks(row.get("checks"), sidecar_check_keys, f"2D2G {name}")
    path_audit = persisted.get("path_audit")
    if (
        not isinstance(path_audit, dict)
        or set(path_audit) != {"local_checkpoint", "persistent_directory", "lock_path", "checks", "passed"}
        or path_audit.get("local_checkpoint") != persisted.get("local")
        or Path(path_audit.get("persistent_directory", "")).resolve() != path.parent.resolve()
        or path_audit.get("lock_path") != CHECKPOINT_PERSIST_LOCK
        or path_audit.get("passed") is not True
    ):
        raise RuntimeError("2D2G persistence path audit differs")
    _require_exact_true_checks(
        path_audit.get("checks"),
        {"local_checkpoint_is_ephemeral", "persistent_directory_is_workspace",
         "persistent_directory_not_ephemeral", "shared_lock_exact"},
        "2D2G persistence path audit",
    )
    if (
        persisted.get("persistent") != str(path.resolve())
        or not isinstance(persisted.get("local"), str)
        or not persisted["local"].startswith("/tmp/parallel_2d2_ephemeral/")
        or persisted.get("local_sha256") != observed_sha
        or persisted.get("persistent_sha256") != observed_sha
        or persisted.get("bytes") != size
        or persisted.get("lock") != CHECKPOINT_PERSIST_LOCK
        or persisted.get("persistent_sha_verified_while_lock_held") is not True
        or persisted.get("passed") is not True
        or summary.get("final_checkpoint") != str(path.resolve())
        or summary.get("final_checkpoint_sha256") != observed_sha
    ):
        raise RuntimeError("2D2G persistent/result-summary checkpoint identity differs")
    return {
        "schema": "2d2g_manifest",
        "updates": 191,
        "targets": contract["targets"],
        "strict_checks": strict,
        "checkpoint_manifest": {
            "path": str(manifest_path), "sha256": file_sha256(manifest_path)
        },
        "persistent_record": {
            "path": str(persistent_path), "sha256": file_sha256(persistent_path)
        },
        "result_summary": {
            "path": str(summary_path), "sha256": file_sha256(summary_path)
        },
        "stage_b_metrics": {
            "path": str(stage_b_metrics_path),
            "sha256": hashlib.sha256(stage_b_metrics_bytes).hexdigest(),
            "rows": len(metrics),
            "final_processed_stage_targets": metrics[-1]["processed_stage_targets"],
            "targets_per_update": targets_per_update,
        },
        "stage_b_data_match": {
            "path": str(stage_b_match_path), "sha256": file_sha256(stage_b_match_path),
            "final_cursor_exact": True,
        },
        "passed": True,
    }


def validate_final_checkpoint(
    experiment: str, path: Path, trusted_root: Path | None = None
) -> dict:
    contract = FINAL_CHECKPOINT_CONTRACTS.get(experiment)
    if contract is None:
        raise RuntimeError(f"no exact final checkpoint contract exists: {experiment}")
    trust = trusted_root if trusted_root is not None else lexical_absolute(path).parent
    path = require_symlink_free_regular_file(
        path, trust, f"{experiment} final checkpoint"
    )
    observed_sha = file_sha256(path)
    size = path.stat().st_size
    sha_path = path.with_suffix(path.suffix + ".sha256")
    verification_path = path.with_suffix(path.suffix + ".verification.json")
    require_symlink_free_regular_file(
        sha_path, trust, f"{experiment} final checkpoint SHA sidecar"
    )
    require_symlink_free_regular_file(
        verification_path, trust, f"{experiment} final checkpoint verification"
    )
    try:
        sidecar_fields = sha_path.read_text(encoding="utf-8").split()
    except OSError as error:
        raise RuntimeError(f"{experiment} final checkpoint SHA sidecar failed: {error}") from error
    if (
        len(sidecar_fields) != 2
        or sidecar_fields[0] != observed_sha
        or SHA256_PATTERN.fullmatch(observed_sha) is None
        or sidecar_fields[1].lstrip("*") != path.name
    ):
        raise RuntimeError(f"{experiment} final checkpoint fresh/sidecar SHA mismatch")
    verification = read_json(verification_path)
    if experiment == "2D2G":
        schema_audit = _validate_2d2g_final_checkpoint_verification(
            path, observed_sha, size, verification, contract
        )
    else:
        schema_audit = _validate_standard_final_checkpoint_verification(
            experiment, path, observed_sha, size, verification, contract
        )
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": observed_sha,
        "sha256_sidecar": str(sha_path.resolve()),
        "sha256_sidecar_sha256": file_sha256(sha_path),
        "verification": str(verification_path.resolve()),
        "verification_sha256": file_sha256(verification_path),
        "schema_audit": schema_audit,
        "passed": True,
    }


def validate_final_science_artifacts() -> dict:
    problems = []
    all_paths = [
        lexical_absolute(value)
        for value in (
            *FINAL_REPORTS.values(), *FINAL_AUDITS.values(), *FINAL_CHECKPOINTS.values()
        )
    ]
    common = lexical_absolute(os.path.commonpath([str(path) for path in all_paths]))
    if common in all_paths:
        common = common.parent
    reports = {}
    for experiment, value in FINAL_REPORTS.items():
        path = Path(value)
        try:
            path = require_symlink_free_regular_file(
                path, common, f"{experiment} final report"
            )
            size = path.stat().st_size
        except OSError as error:
            problems.append(f"{experiment} final report is unavailable: {error}")
            continue
        reports[experiment] = {
            "path": str(path),
            "size": size,
            "sha256": file_sha256(path),
        }
    checkpoints = {}
    for experiment, value in FINAL_CHECKPOINTS.items():
        path = Path(value)
        try:
            checkpoints[experiment] = validate_final_checkpoint(
                experiment, path, common
            )
        except (OSError, RuntimeError) as error:
            problems.append(str(error))
    audits = {}
    for experiment, value in FINAL_AUDITS.items():
        path = Path(value)
        try:
            path = require_symlink_free_regular_file(
                path, common, f"{experiment} FINAL_AUDIT"
            )
            size = path.stat().st_size
        except OSError as error:
            raise RuntimeError(f"{experiment} FINAL_AUDIT is unavailable: {error}") from error
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


def _load_finalization_manifest(
    path_value: str,
    expected_path: Path,
    expected_kind: str,
    run_id: str,
) -> tuple[dict, dict]:
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"{expected_kind} evidence path is absent")
    path = lexical_absolute(path_value)
    expected = lexical_absolute(expected_path)
    if path != expected:
        raise RuntimeError(
            f"{expected_kind} evidence must use exact run-scoped path {expected_path}"
        )
    require_symlink_free_regular_file(
        path, lexical_absolute(expected_path).parent,
        f"{expected_kind} evidence",
    )
    if path.stat().st_mode & 0o222:
        raise RuntimeError(
            f"{expected_kind} evidence is not a regular, symlink-free, read-only file: {path}"
        )
    try:
        content = path.read_bytes()
        payload = json.loads(content)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {expected_kind} evidence {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{expected_kind} evidence is not a JSON object")
    common = {
        "schema_version",
        "kind",
        "run_id",
        "pod",
        "created_utc",
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != expected_kind
        or payload.get("run_id") != run_id
        or payload.get("pod") != EXPECTED_POD
        or not isinstance(payload.get("created_utc"), str)
        or not payload["created_utc"]
        or not common.issubset(payload)
    ):
        raise RuntimeError(f"{expected_kind} evidence identity is stale or invalid")
    created = parse_canonical_utc(
        payload["created_utc"], f"{expected_kind} created_utc"
    )
    require_timestamp_not_future(created, f"{expected_kind} created_utc")
    return payload, {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "created_utc": payload["created_utc"],
    }


def _clean_git_environment() -> dict[str, str]:
    environment = dict(SANITIZED_TOOL_ENVIRONMENT)
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _safe_git_argv(*args: str) -> list[str]:
    """Build every Git invocation from one fail-closed configuration."""
    return [
        GIT_EXECUTABLE,
        "--no-replace-objects",
        "-c", "core.fsmonitor=false",
        "-c", "core.hooksPath=/dev/null",
        "-c", "credential.helper=",
        "-c", "core.askPass=/usr/bin/false",
        "-c", "core.pager=cat",
        "-c", "http.sslVerify=true",
        "-c", "http.followRedirects=false",
        *args,
    ]


def _git_bytes(worktree: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            _safe_git_argv(*args), cwd=worktree,
            stderr=subprocess.STDOUT, env=_clean_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "output", b"")
        if isinstance(output, bytes):
            output = output.decode("utf-8", "backslashreplace")
        raise RuntimeError(
            f"Git finalization audit failed in {worktree}: "
            f"git {' '.join(args)}: {output}"
        ) from error


def _git_output(worktree: Path, *args: str) -> str:
    try:
        output = _git_bytes(worktree, *args)
        if isinstance(output, str):
            return output.strip()
        return output.decode("utf-8", "strict").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(
            f"Git finalization audit returned non-UTF-8 output in {worktree}: "
            f"git {' '.join(args)}"
        ) from error


def _git_optional_output(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        _safe_git_argv(*args), cwd=worktree, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=_clean_git_environment(), check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"Git optional finalization audit failed in {worktree}: "
            f"git {' '.join(args)}: {result.stdout}"
        )
    return result.stdout.strip()


def _git_is_ancestor(worktree: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        _safe_git_argv("merge-base", "--is-ancestor", ancestor, descendant),
        cwd=worktree, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=_clean_git_environment(), check=False,
    ).returncode == 0


def _validate_git_relative_path(relative: str, label: str) -> tuple[str, ...]:
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith(("/", "-"))
        or "\\" in relative
        or any(ord(character) < 32 or ord(character) == 127 for character in relative)
    ):
        raise RuntimeError(f"{label} is not a safe repository-relative path: {relative!r}")
    parts = tuple(relative.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"{label} is not a canonical repository path: {relative!r}")
    return parts


def _read_symlink_free_tracked_file(worktree: Path, relative: str) -> tuple[bytes, str]:
    """Read a tracked file without traversing a repository-controlled symlink."""
    parts = _validate_git_relative_path(relative, "tracked Git path")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise RuntimeError("platform lacks symlink-safe file-descriptor traversal")
    descriptors = []
    try:
        current = os.open(worktree, os.O_RDONLY | directory | nofollow)
        descriptors.append(current)
        for component in parts[:-1]:
            current = os.open(
                component, os.O_RDONLY | directory | nofollow, dir_fd=current
            )
            descriptors.append(current)
        descriptor = os.open(parts[-1], os.O_RDONLY | nofollow, dir_fd=current)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if before.st_mode & 0o170000 != 0o100000:
            raise RuntimeError(f"tracked Git path is not a regular file: {relative}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        content = b"".join(chunks)
        stable_fields = (
            "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns"
        )
        if (
            any(getattr(before, key) != getattr(after, key) for key in stable_fields)
            or before.st_size != len(content)
        ):
            raise RuntimeError(f"tracked Git path changed while audited: {relative}")
        mode = "100755" if before.st_mode & 0o111 else "100644"
        return content, mode
    except OSError as error:
        raise RuntimeError(
            f"cannot read symlink-free tracked Git path {relative}: {error}"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _parse_git_tree(tree: bytes) -> dict[str, tuple[str, str]]:
    parsed = {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        fields = metadata.split()
        try:
            relative = encoded_path.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise RuntimeError("Git tree contains a non-UTF-8 path") from error
        if (
            not separator
            or len(fields) != 3
            or fields[1] != b"blob"
            or fields[0] not in (b"100644", b"100755")
            or re.fullmatch(rb"[0-9a-f]{40}", fields[2]) is None
            or relative in parsed
        ):
            raise RuntimeError(f"Git tree contains an unsupported entry: {record!r}")
        _validate_git_relative_path(relative, "Git tree path")
        parsed[relative] = (fields[0].decode(), fields[2].decode())
    if not parsed:
        raise RuntimeError("Git HEAD tree is empty")
    return parsed


def _parse_git_index(index: bytes) -> dict[str, tuple[str, str]]:
    parsed = {}
    for record in index.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        fields = metadata.split()
        try:
            relative = encoded_path.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise RuntimeError("Git index contains a non-UTF-8 path") from error
        if (
            not separator
            or len(fields) != 3
            or fields[0] not in (b"100644", b"100755")
            or re.fullmatch(rb"[0-9a-f]{40}", fields[1]) is None
            or fields[2] != b"0"
            or relative in parsed
        ):
            raise RuntimeError(f"Git index contains an unsupported entry: {record!r}")
        _validate_git_relative_path(relative, "Git index path")
        parsed[relative] = (fields[0].decode(), fields[1].decode())
    return parsed


def _audit_local_git_configuration(worktree: Path) -> dict:
    local_keys = _git_output(
        worktree, "config", "--local", "--includes", "--name-only", "--list"
    ).splitlines()
    dangerous = re.compile(
        r"^(?:include(?:if)?\.|url\.|gpg\.|filter\.|credential\.|http\.|https\."
        r"|protocol\.|diff\.external$|diff\..*\.(?:command|textconv)$"
        r"|merge\..*\.driver$"
        r"|remote\..*\.(?:proxy|proxyauthmethod)$"
        r"|core\.(?:fsmonitor(?:hookversion)?|hookspath|worktree|sshcommand|gitproxy"
        r"|askpass|attributesfile|excludesfile)$|extensions\.worktreeconfig$)"
    )
    forbidden = sorted({key for key in local_keys if dangerous.match(key.lower())})
    replacements = _git_output(worktree, "replace", "-l").splitlines()
    common_value = _git_output(worktree, "rev-parse", "--git-common-dir")
    common_dir = Path(common_value)
    if not common_dir.is_absolute():
        common_dir = worktree / common_dir
    grafts = common_dir / "info" / "grafts"
    grafts_present = grafts.exists() or grafts.is_symlink()
    if forbidden or replacements or grafts_present:
        raise RuntimeError(
            "unsafe repository-local Git controls are present: "
            f"config={forbidden}, replacements={replacements}, "
            f"grafts={str(grafts) if grafts_present else None}"
        )
    return {
        "forbidden_local_config": forbidden,
        "replacement_refs": replacements,
        "legacy_grafts_present": grafts_present,
        "passed": True,
    }


def _audit_git_repository(worktree: Path, head: str) -> dict:
    """Cross-bind raw HEAD, index, and filesystem state without status heuristics."""
    configuration = _audit_local_git_configuration(worktree)
    if _git_output(worktree, "rev-parse", "--show-object-format") != "sha1":
        raise RuntimeError("finalization requires this repository's SHA-1 object format")
    tree_bytes = _git_bytes(worktree, "ls-tree", "-rz", "--full-tree", head)
    index_bytes = _git_bytes(worktree, "ls-files", "--stage", "-z")
    tree = _parse_git_tree(tree_bytes)
    index = _parse_git_index(index_bytes)
    if index != tree:
        raise RuntimeError("Git index blob/mode map differs from the exact HEAD tree")
    for option in ("-v", "-f"):
        flag_rows = _git_bytes(worktree, "ls-files", option, "-z").split(b"\0")
        flags = {}
        for row in flag_rows:
            if not row:
                continue
            tag, separator, encoded_path = row.partition(b" ")
            try:
                relative = encoded_path.decode("utf-8", "strict")
            except UnicodeDecodeError as error:
                raise RuntimeError("Git index flags contain a non-UTF-8 path") from error
            if not separator or len(tag) != 1 or relative in flags:
                raise RuntimeError(f"Git index flag row is malformed: {row!r}")
            flags[relative] = tag.decode("ascii", "strict")
        if set(flags) != set(tree) or any(value != "H" for value in flags.values()):
            raise RuntimeError(
                f"Git index has hidden state flags under ls-files {option}: "
                f"{sorted((path, value) for path, value in flags.items() if value != 'H')}"
            )
    audited_rows = []
    for relative, (expected_mode, expected_oid) in sorted(tree.items()):
        content, observed_mode = _read_symlink_free_tracked_file(worktree, relative)
        header = f"blob {len(content)}\0".encode()
        observed_oid = hashlib.sha1(header + content).hexdigest()
        if observed_mode != expected_mode or observed_oid != expected_oid:
            raise RuntimeError(
                f"tracked file differs from raw HEAD blob/mode: {relative}"
            )
        audited_rows.append(
            {"path": relative, "mode": observed_mode, "blob": observed_oid}
        )
    return {
        "configuration": configuration,
        "head": head,
        "tracked_files": len(audited_rows),
        "head_tree_sha256": hashlib.sha256(tree_bytes).hexdigest(),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "tracked_map_sha256": hashlib.sha256(
            json.dumps(
                audited_rows, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "symlink_free": True,
        "head_index_worktree_exact": True,
        "passed": True,
    }


def _remote_ref_commit(worktree: Path, ref: str) -> str | None:
    lines = _git_output(
        worktree, "ls-remote", "--exit-code", CANONICAL_ORIGIN_URL, ref
    ).splitlines()
    if len(lines) != 1:
        return None
    fields = lines[0].split()
    if len(fields) != 2 or fields[1] != ref:
        return None
    return fields[0]


def _remote_tag_identity(worktree: Path, tag: str) -> dict | None:
    ref = f"refs/tags/{tag}"
    lines = _git_output(
        worktree, "ls-remote", "--exit-code", CANONICAL_ORIGIN_URL,
        ref, ref + "^{}",
    ).splitlines()
    parsed = [fields for fields in map(str.split, lines) if len(fields) == 2]
    if len(parsed) != 2 or {fields[1] for fields in parsed} != {ref, ref + "^{}"}:
        return None
    rows = {fields[1]: fields[0] for fields in parsed}
    if any(SHA256_PATTERN.fullmatch(value) is not None for value in rows.values()):
        # Git object names must be SHA-1 in this repository, not arbitrary SHA-256
        # evidence strings that merely happen to be hexadecimal.
        return None
    if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in rows.values()):
        return None
    return {"tag_object": rows[ref], "peeled_commit": rows[ref + "^{}"]}


def _verify_signed_implementation_tag(worktree: Path, tag: str) -> dict:
    """Verify a direct annotated tag under the pinned local Ed25519 identity."""
    ref = f"refs/tags/{tag}"
    object_type = _git_output(worktree, "cat-file", "-t", ref)
    if object_type != "tag":
        raise RuntimeError(
            "master finalization implementation boundary is not an annotated tag"
        )
    tag_object = _git_output(worktree, "rev-parse", ref)
    implementation_commit = _git_output(
        worktree, "rev-parse", f"{ref}^{{commit}}"
    )
    tag_body = _git_output(worktree, "cat-file", "tag", tag_object)
    header, separator, _ = tag_body.partition("\n\n")
    header_rows = {}
    for line in header.splitlines():
        key, space, value = line.partition(" ")
        if space and key in {"object", "type", "tag"}:
            if key in header_rows:
                raise RuntimeError("implementation tag has duplicate identity headers")
            header_rows[key] = value
    if (
        separator != "\n\n"
        or header_rows != {
            "object": implementation_commit,
            "type": "commit",
            "tag": tag,
        }
        or re.fullmatch(r"[0-9a-f]{40}", tag_object) is None
        or re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None
    ):
        raise RuntimeError("implementation tag does not directly bind its exact commit")
    local_gpg_overrides = _git_optional_output(
        worktree, "config", "--local", "--get-regexp", r"^gpg\."
    )
    if local_gpg_overrides:
        raise RuntimeError(
            "repository-local GPG/SSH verifier configuration is forbidden"
        )
    allowed_path = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            allowed_path = Path(handle.name)
            handle.write(
                (
                    f"{FINAL_BACKUP_SIGNER_PRINCIPAL} "
                    f"{FINAL_BACKUP_SIGNER_PUBLIC_KEY}\n"
                ).encode()
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(allowed_path, 0o600)
        result = subprocess.run(
            _safe_git_argv(
                "-c", "gpg.format=ssh",
                "-c", f"gpg.ssh.program={SSH_KEYGEN_EXECUTABLE}",
                "-c", f"gpg.ssh.allowedSignersFile={allowed_path}",
                "verify-tag", "--raw", ref,
            ),
            cwd=worktree,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_clean_git_environment(),
            check=False,
        )
    except OSError as error:
        raise RuntimeError(f"cannot verify signed implementation tag: {error}") from error
    finally:
        if allowed_path is not None:
            allowed_path.unlink(missing_ok=True)
    output = result.stdout.strip()
    if result.returncode != 0 or FINAL_BACKUP_SIGNER_FINGERPRINT not in output:
        raise RuntimeError(
            "master finalization implementation tag signature is absent or untrusted: "
            + output
        )
    if (
        _git_output(worktree, "cat-file", "-t", ref) != object_type
        or _git_output(worktree, "rev-parse", ref) != tag_object
        or _git_output(worktree, "rev-parse", f"{ref}^{{commit}}")
        != implementation_commit
    ):
        raise RuntimeError("implementation tag changed during signature verification")
    return {
        "tag": tag,
        "tag_object": tag_object,
        "implementation_commit": implementation_commit,
        "signer_principal": FINAL_BACKUP_SIGNER_PRINCIPAL,
        "signer_fingerprint": FINAL_BACKUP_SIGNER_FINGERPRINT,
        "signature_verified": True,
        "passed": True,
    }


def _git_blob_at_commit(worktree: Path, commit: str, relative: str) -> bytes:
    _validate_git_relative_path(relative, "implementation fingerprint path")
    rows = _parse_git_tree(
        _git_bytes(worktree, "ls-tree", "-z", commit, "--", relative)
    )
    if set(rows) != {relative}:
        raise RuntimeError(
            f"implementation fingerprint path is not one exact Git blob: {relative}"
        )
    content = _git_bytes(worktree, "cat-file", "blob", f"{commit}:{relative}")
    mode, expected_oid = rows[relative]
    observed_oid = hashlib.sha1(
        f"blob {len(content)}\0".encode() + content
    ).hexdigest()
    if mode not in ("100644", "100755") or observed_oid != expected_oid:
        raise RuntimeError(
            f"implementation fingerprint Git blob identity differs: {relative}"
        )
    return content


def _fingerprint_against_git_commit(
    name: str,
    worktree: Path,
    commit: str,
    fingerprint: object,
    contract: dict,
    *,
    aggregate_override: str | None = None,
) -> dict:
    expected_paths = contract.get("implementation_fingerprint_files")
    fingerprint_format = contract.get("implementation_fingerprint_format")
    expected_aggregate = (
        aggregate_override
        if aggregate_override is not None
        else contract.get("implementation_fingerprint_aggregate")
    )
    if (
        not isinstance(expected_paths, tuple)
        or not expected_paths
        or len(set(expected_paths)) != len(expected_paths)
        or fingerprint_format not in {
            "canonical_file_map_sha256", "raw_digest_concat_sha256"
        }
        or not isinstance(expected_aggregate, str)
        or SHA256_PATTERN.fullmatch(expected_aggregate) is None
    ):
        raise RuntimeError(f"{name} implementation fingerprint contract is invalid")
    computed_files = {}
    for relative in sorted(expected_paths):
        content = _git_blob_at_commit(worktree, commit, relative)
        computed_files[relative] = hashlib.sha256(content).hexdigest()
    if fingerprint_format == "canonical_file_map_sha256":
        aggregate = hashlib.sha256(
            json.dumps(
                computed_files, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        expected_fingerprint = {
            "algorithm": "sha256",
            "files": computed_files,
            "aggregate_sha256": aggregate,
        }
    else:
        digest = hashlib.sha256()
        for relative in sorted(computed_files):
            digest.update(bytes.fromhex(computed_files[relative]))
        aggregate = digest.hexdigest()
        expected_fingerprint = {
            "files": computed_files,
            "aggregate_sha256": aggregate,
        }
    if aggregate != expected_aggregate or fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"{name} implementation fingerprint does not match exact Git blobs/schema"
        )
    return {
        "commit": commit,
        "format": fingerprint_format,
        "files": len(computed_files),
        "aggregate_sha256": aggregate,
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(
                expected_fingerprint, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "passed": True,
    }


def _read_symlink_free_json(worktree: Path, relative: str, label: str) -> tuple[dict, bytes]:
    content, _ = _read_symlink_free_tracked_file(worktree, relative)
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot parse {label}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload, content


def _validate_h_correction_provenance(
    worktree: Path,
    result_prefix: str,
    summary: dict,
    preflight_bytes: bytes,
    training_fingerprint: dict,
    implementation_commit: str,
    execution_commit: str,
    contract: dict,
) -> dict:
    authorization_relative = (
        result_prefix + "/POST_TRAINING_AUDIT_CORRECTION_AUTHORIZATION.json"
    )
    authorization, authorization_bytes = _read_symlink_free_json(
        worktree, authorization_relative, "2D2H correction authorization"
    )
    current_fingerprint = authorization.get("current_implementation_fingerprint")
    current_audit = _fingerprint_against_git_commit(
        "2D2H correction",
        worktree,
        implementation_commit,
        current_fingerprint,
        contract,
        aggregate_override=contract.get("correction_fingerprint_aggregate"),
    )
    old_preflight = authorization.get("old_preflight")
    summary_correction = summary.get("post_training_audit_correction")
    summary_authorization = (
        summary_correction.get("authorization")
        if isinstance(summary_correction, dict) else None
    )
    expected_remote_path = str(
        Path(FINAL_REPORTS["2D2H"]).with_name(
            "POST_TRAINING_AUDIT_CORRECTION_AUTHORIZATION.json"
        )
    )
    authorization_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
    if (
        contract.get("correction_commit") != implementation_commit
        or authorization.get("schema_version") != 1
        or authorization.get("correction_kind") != "evaluation_only"
        or authorization.get("base_commit") != execution_commit
        or authorization.get("current_commit") != implementation_commit
        or authorization.get("origin_commit") != implementation_commit
        or authorization.get("training_or_checkpoint_change_authorized") is not False
        or authorization.get("passed") is not True
        or not isinstance(authorization.get("checks"), dict)
        or not authorization["checks"]
        or not all(value is True for value in authorization["checks"].values())
        or authorization.get("changed_files")
        != ["scripts/experiment_2d2h.py", "tests/test_experiment_2d2h_driver.py"]
        or not isinstance(old_preflight, dict)
        or old_preflight.get("sha256") != hashlib.sha256(preflight_bytes).hexdigest()
        or old_preflight.get("implementation_fingerprint") != training_fingerprint
        or summary_authorization
        != {
            "current_commit": implementation_commit,
            "path": expected_remote_path,
            "sha256": authorization_sha256,
        }
        or summary_correction.get("correction_kind") != "evaluation_only"
        or summary_correction.get("training_changed") is not False
        or summary_correction.get("checkpoint_changed") is not False
        or summary_correction.get("data_or_primary_scientific_metrics_changed") is not False
        or summary_correction.get("passed") is not True
    ):
        raise RuntimeError("2D2H correction provenance is not exactly cross-bound")
    return {
        "authorization": authorization_relative,
        "authorization_sha256": authorization_sha256,
        "base_commit": execution_commit,
        "correction_commit": implementation_commit,
        "current_fingerprint": current_audit,
        "passed": True,
    }


def _validate_embedded_implementation(
    name: str, worktree: Path, implementation_commit: str
) -> dict:
    if name == "MASTER":
        return {"applicable": False, "passed": True}
    contract = FINAL_GIT_REPOSITORIES[name]
    result_prefix = contract["tracked_paths"][0]
    summary_relative = result_prefix + "/result_summary.json"
    preflight_relative = result_prefix + "/preflight_audit.json"
    summary, _ = _read_symlink_free_json(
        worktree, summary_relative, f"{name} result summary"
    )
    preflight, preflight_bytes = _read_symlink_free_json(
        worktree, preflight_relative, f"{name} preflight audit"
    )
    git_row = summary.get("git")
    summary_commit = (
        git_row.get("implementation_commit") if isinstance(git_row, dict) else None
    )
    preflight_commit = preflight.get("implementation_git_commit")
    fingerprint = preflight.get("implementation_fingerprint")
    allowed = contract.get("allowed_execution_commits")
    if allowed is None:
        # Backward-compatible path for isolated synthetic fixtures.  Every
        # production lane above has an explicit immutable execution contract.
        commits = [
            value for value in (summary_commit, preflight_commit) if value is not None
        ]
        if (
            not commits
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{40}", value) is None
                for value in commits
            )
            or any(
                not _git_is_ancestor(worktree, value, implementation_commit)
                for value in commits
            )
            or (
                summary_commit is not None
                and preflight_commit is not None
                and summary_commit != preflight_commit
            )
            or not isinstance(fingerprint, dict)
            or not fingerprint
        ):
            raise RuntimeError(
                f"{name} embedded implementation provenance is not bound"
            )
        return {
            "summary": str((worktree / summary_relative).resolve()),
            "preflight": str((worktree / preflight_relative).resolve()),
            "summary_implementation_commit": summary_commit,
            "preflight_implementation_commit": preflight_commit,
            "implementation_fingerprint_sha256": hashlib.sha256(
                json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "synthetic_contract_compatibility": True,
            "passed": True,
        }
    source = contract.get("execution_commit_source")
    if (
        not isinstance(allowed, tuple)
        or not allowed
        or any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in allowed)
        or source not in {"summary", "preflight"}
    ):
        raise RuntimeError(f"{name} execution-commit contract is invalid")
    execution_commit = summary_commit if source == "summary" else preflight_commit
    other_commit = preflight_commit if source == "summary" else summary_commit
    environment = preflight.get("environment")
    environment_commit = (
        environment.get("git_commit") if isinstance(environment, dict) else None
    )
    environment_required = contract.get(
        "execution_environment_commit_required", False
    )
    if (
        execution_commit not in allowed
        or other_commit is not None
        or not _git_is_ancestor(worktree, execution_commit, implementation_commit)
        or (
            environment_required
            and environment_commit != execution_commit
        )
        or (
            not environment_required and environment_commit is not None
        )
    ):
        raise RuntimeError(
            f"{name} embedded execution commit is not an explicitly allowed commit"
        )
    fingerprint_audit = _fingerprint_against_git_commit(
        name, worktree, execution_commit, fingerprint, contract
    )
    correction_audit = {"applicable": False, "passed": True}
    if name == "2D2H":
        correction_audit = _validate_h_correction_provenance(
            worktree,
            result_prefix,
            summary,
            preflight_bytes,
            fingerprint,
            implementation_commit,
            execution_commit,
            contract,
        )
    return {
        "summary": str((worktree / summary_relative).resolve()),
        "preflight": str((worktree / preflight_relative).resolve()),
        "summary_implementation_commit": summary_commit,
        "preflight_implementation_commit": preflight_commit,
        "execution_environment_commit": environment_commit,
        "allowed_execution_commits": list(allowed),
        "execution_commit": execution_commit,
        "implementation_fingerprint": fingerprint_audit,
        "correction_provenance": correction_audit,
        "passed": True,
    }


def validate_final_git_evidence(
    master_root: Path, run_root: Path, run_id: str, path_value: str
) -> dict:
    path = run_root / FINALIZATION_EVIDENCE_FILES["git"]
    payload, identity = _load_finalization_manifest(
        path_value, path, "parallel_2d2_final_git_evidence_v1", run_id
    )
    if set(payload) != {
        "schema_version",
        "kind",
        "run_id",
        "pod",
        "created_utc",
        "repositories",
    }:
        raise RuntimeError("Git finalization evidence has unexpected keys")
    rows = payload.get("repositories")
    if not isinstance(rows, dict) or set(rows) != set(FINAL_GIT_REPOSITORIES):
        raise RuntimeError("Git finalization evidence repository set is not exact")
    audited = {}
    for name, contract in FINAL_GIT_REPOSITORIES.items():
        row = rows.get(name)
        worktree = (master_root / "worktrees" / contract["worktree"]).resolve()
        expected_paths = contract["tracked_paths"]
        if not isinstance(row, dict) or set(row) != {
            "worktree",
            "branch",
            "origin_url",
            "implementation_tag",
            "implementation_tag_object",
            "implementation_commit",
            "commit",
            "origin_commit",
            "tracked_paths",
        }:
            raise RuntimeError(f"Git evidence row is not exact: {name}")
        if (
            Path(row.get("worktree", "")).resolve() != worktree
            or row.get("branch") != contract["branch"]
            or row.get("origin_url") != CANONICAL_ORIGIN_URL
            or row.get("implementation_tag") != contract.get("implementation_tag")
            or row.get("tracked_paths") != expected_paths
        ):
            raise RuntimeError(f"Git evidence contract differs: {name}")
        fetch_urls = _git_output(worktree, "remote", "get-url", "--all", "origin").splitlines()
        push_urls = _git_output(
            worktree, "remote", "get-url", "--push", "--all", "origin"
        ).splitlines()
        local_rewrites = _git_optional_output(
            worktree, "config", "--local", "--get-regexp",
            r"^url\..*\.(insteadof|pushinsteadof)$",
        )
        branch = _git_output(worktree, "branch", "--show-current")
        head = _git_output(worktree, "rev-parse", "HEAD")
        origin = _git_output(worktree, "rev-parse", f"origin/{contract['branch']}")
        if re.fullmatch(r"[0-9a-f]{40}", head) is None:
            raise RuntimeError(f"Git HEAD is not a full commit ID: {name}")
        repository_integrity = _audit_git_repository(worktree, head)
        ancestor = _git_is_ancestor(worktree, contract["minimum_commit"], head)
        implementation_tag = contract.get("implementation_tag")
        implementation_commit = contract.get("implementation_commit")
        implementation_tag_object = None
        signed_tag_audit = {"applicable": False, "passed": True}
        remote_implementation = None
        if implementation_tag:
            signed_tag_audit = _verify_signed_implementation_tag(
                worktree, implementation_tag
            )
            implementation_commit = signed_tag_audit["implementation_commit"]
            implementation_tag_object = signed_tag_audit["tag_object"]
            remote_implementation = _remote_tag_identity(
                worktree, implementation_tag
            )
        if (
            not isinstance(implementation_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None
            or not _git_is_ancestor(worktree, implementation_commit, head)
        ):
            raise RuntimeError(f"Git implementation boundary is invalid: {name}")
        clean = _git_output(
            worktree, "status", "--porcelain=v1", "--untracked-files=all"
        ) == ""
        remote_ref = f"refs/heads/{contract['branch']}"
        live = _remote_ref_commit(worktree, remote_ref)
        changed_paths = _git_output(
            worktree, "diff", "--no-ext-diff", "--name-only",
            f"{implementation_commit}..{head}"
        ).splitlines()
        unauthorized_changes = [
            value for value in changed_paths
            if not any(value == prefix or value.startswith(prefix + "/")
                       for prefix in expected_paths)
        ]
        tracked = {}
        for tracked_path in expected_paths:
            files = _git_output(
                worktree, "ls-files", "--", tracked_path
            ).splitlines()
            exact = bool(files) and all(
                value == tracked_path or value.startswith(tracked_path + "/")
                for value in files
            )
            tracked[tracked_path] = {
                "files": len(files),
                "exact_path_scope": exact,
            }
        checks = {
            "worktree_present": worktree.is_dir(),
            "raw_head_index_worktree_exact": repository_integrity["passed"],
            "origin_fetch_url_exact": fetch_urls == [CANONICAL_ORIGIN_URL],
            "origin_push_url_exact": push_urls == [CANONICAL_ORIGIN_URL],
            "no_local_url_rewrite": local_rewrites == "",
            "branch_exact": branch == contract["branch"],
            "implementation_commit_is_ancestor": ancestor,
            "finalization_implementation_boundary_is_ancestor": _git_is_ancestor(
                worktree, implementation_commit, head
            ),
            "remote_implementation_tag_matches": (
                remote_implementation
                == {
                    "tag_object": implementation_tag_object,
                    "peeled_commit": implementation_commit,
                }
                if implementation_tag else True
            ),
            "manifest_implementation_tag_object_exact": (
                row.get("implementation_tag_object") == implementation_tag_object
            ),
            "manifest_implementation_commit_exact": (
                row.get("implementation_commit") == implementation_commit
            ),
            "post_implementation_changes_authorized": not unauthorized_changes,
            "manifest_commit_exact": row.get("commit") == head,
            "manifest_origin_exact": row.get("origin_commit") == origin,
            "local_origin_ref_matches": origin == head,
            "live_origin_matches": live == head,
            "worktree_clean_including_untracked": clean,
            "result_paths_tracked": all(
                value["exact_path_scope"] for value in tracked.values()
            ),
            "head_stable_during_audit": _git_output(
                worktree, "rev-parse", "HEAD"
            ) == head,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Git finalization evidence failed for {name}: {checks}")
        embedded = _validate_embedded_implementation(
            name, worktree, implementation_commit
        )
        audited[name] = {
            "worktree": str(worktree),
            "branch": branch,
            "origin_url": CANONICAL_ORIGIN_URL,
            "implementation_commit": implementation_commit,
            "implementation_tag": implementation_tag,
            "implementation_tag_object": implementation_tag_object,
            "signed_implementation_tag": signed_tag_audit,
            "commit": head,
            "origin_commit": origin,
            "live_origin_commit": live,
            "tracked_paths": tracked,
            "repository_integrity": repository_integrity,
            "post_implementation_changed_paths": changed_paths,
            "unauthorized_changed_paths": unauthorized_changes,
            "embedded_implementation": embedded,
            "checks": checks,
            "passed": True,
        }
    return {"manifest": identity, "repositories": audited, "passed": True}


def _validate_file_evidence(
    row: dict, expected_path: Path, label: str, trusted_root: Path | None = None
) -> dict:
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{label} file evidence has unexpected keys")
    path = lexical_absolute(row.get("path", ""))
    raw_trust = lexical_absolute(
        trusted_root if trusted_root is not None else lexical_absolute(expected_path).parent
    )
    canonical_trust = raw_trust.resolve()
    raw_expected = lexical_absolute(expected_path)
    try:
        expected_relative = raw_expected.relative_to(raw_trust)
    except ValueError:
        # The caller may already have canonicalized the expected path.
        expected_relative = raw_expected.relative_to(canonical_trust)
    expected = canonical_trust / expected_relative
    if path != expected:
        raise RuntimeError(f"{label} file path is unavailable or differs: {path}")
    require_symlink_free_regular_file(
        path,
        canonical_trust,
        label,
    )
    size = path.stat().st_size
    digest = file_sha256(path)
    if (
        size <= 0
        or row.get("bytes") != size
        or row.get("sha256") != digest
        or SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise RuntimeError(f"{label} file identity differs")
    return {"path": str(path), "bytes": size, "sha256": digest}


def _claim_json_source(path: Path, label: str, sources: dict) -> dict:
    path = path.resolve()
    payload = read_json(path)
    content = path.read_bytes()
    sources[label] = {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return payload


def _claim_number(value: object, label: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RuntimeError(f"master scientific claim is not finite numeric: {label}")
    return value


def _claim_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"master scientific claim is not a nonnegative integer: {label}")
    return value


def _checkpoint_claim(summary: dict, label: str) -> dict:
    path = summary.get("checkpoint" if label == "2D2E-C1" else "final_checkpoint")
    sha = summary.get(
        "checkpoint_sha256" if label == "2D2E-C1" else "final_checkpoint_sha256"
    )
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or not isinstance(sha, str)
        or SHA256_PATTERN.fullmatch(sha) is None
    ):
        raise RuntimeError(f"{label} checkpoint claim is absent or invalid")
    return {"path": path, "sha256": sha}


def _assert_summary_artifact(
    summary: dict, summary_key: str, artifact: dict, label: str
) -> None:
    if summary.get(summary_key) != artifact:
        raise RuntimeError(
            f"{label} result summary does not exactly embed its structured artifact"
        )


def _validate_master_scientific_decisions(
    matrix: dict, answers: dict
) -> dict:
    """Prove that the fixed M1-M15 decisions follow from the sealed numbers."""
    expected_names = set(MASTER_SCIENTIFIC_QUESTIONS)
    if not isinstance(answers, dict) or set(answers) != expected_names:
        raise RuntimeError("master scientific decision set is not exact")
    for name in expected_names:
        if answers[name].get("answer_code") != MASTER_SCIENTIFIC_ANSWER_CODES[name]:
            raise RuntimeError(f"master scientific decision code differs: {name}")

    def true_row(name: str) -> dict:
        try:
            row = matrix[name]["true_incremental"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"master scientific decision lacks true-incremental row: {name}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(
                f"master scientific decision true-incremental row is invalid: {name}"
            )
        return row

    true = {
        name: true_row(name)
        for name in (
            "2D2B", "2D2D", "2D2E-C1", "2D2E", "2D2F", "2D2G", "2D2H", "2D2I"
        )
    }
    gains = {
        name: _claim_number(row.get("gain"), f"{name}.decision gain")
        for name, row in true.items()
    }
    gaps = {
        name: _claim_number(row.get("sequence_gap"), f"{name}.decision gap")
        for name, row in true.items()
    }
    common_names = ("2D2E", "2D2F", "2D2G", "2D2H", "2D2I")
    validation_ce = {
        name: _claim_number(
            true[name].get("real_validation_ce"), f"{name}.decision validation CE"
        )
        for name in common_names
    }
    state_bytes = {
        name: _claim_int(
            matrix[name].get("inference_state_bytes_b1"),
            f"{name}.decision state bytes",
        )
        for name in common_names
    }
    ring_count = {
        name: _claim_int(
            matrix[name].get("recurrent_ring_count"),
            f"{name}.decision recurrent ring count",
        )
        for name in common_names
    }
    ci = matrix.get("2D2E-C1", {}).get("bootstrap_95_percent")
    if not isinstance(ci, dict):
        raise RuntimeError("2D2E-C1 decision bootstrap evidence is absent")
    ci_values = {}
    for comparison in ("off_minus_real", "shuffled_minus_real"):
        row = ci.get(comparison)
        if not isinstance(row, dict):
            raise RuntimeError(
                f"2D2E-C1 decision bootstrap comparison is absent: {comparison}"
            )
        ci_values[comparison] = {
            "lower": _claim_number(
                row.get("lower"), f"C1 {comparison} bootstrap lower"
            ),
            "upper": _claim_number(
                row.get("upper"), f"C1 {comparison} bootstrap upper"
            ),
        }

    useful_common = tuple(name for name in common_names if gains[name] > 0.0)
    minimum_ce = min(validation_ce, key=validation_ce.get)
    smallest_useful_state = min(useful_common, key=state_bytes.get)
    minimum_useful_rings = min(ring_count[name] for name in useful_common)
    minimum_ring_candidates = tuple(
        name for name in useful_common if ring_count[name] == minimum_useful_rings
    )
    best_minimum_ring_ce = min(
        minimum_ring_candidates, key=validation_ce.get
    )
    strongest_useful_gain = max(useful_common, key=gains.get)
    c1_intervals_cross_zero = all(
        row["lower"] <= 0.0 <= row["upper"] for row in ci_values.values()
    )
    ordered_depth_gains = [
        gains["2D2B"], gains["2D2D"], gains["2D2F"], gains["2D2I"]
    ]
    slopes = [
        right - left
        for left, right in zip(ordered_depth_gains, ordered_depth_gains[1:])
    ]
    f_ce_cost = validation_ce["2D2F"] - validation_ce["2D2G"]
    f_state_saving = state_bytes["2D2G"] - state_bytes["2D2F"]
    checks = {
        "M1_c1_gain_positive": gains["2D2E-C1"] > 0.0,
        "M2_c1_sequence_gap_positive": gaps["2D2E-C1"] > 0.0,
        "M3_both_bootstrap_intervals_cross_zero": c1_intervals_cross_zero,
        "M4_2d2f_gain_exceeds_2d2e": gains["2D2F"] > gains["2D2E"],
        "M5_b2_w32_gain_exceeds_b2_w1024": gains["2D2F"] > gains["2D2G"],
        "M6_2d2h_b2_gain_harmful_and_below_2d2d": (
            gains["2D2H"] < 0.0 and gains["2D2H"] < gains["2D2D"]
        ),
        "M7_b1_foundational_but_other_links_useful": (
            gains["2D2B"] > 0.0
            and gains["2D2F"] > 0.0
            and gains["2D2I"] > 0.0
        ),
        "M8_2d2i_b4_gain_and_gap_positive": (
            gains["2D2I"] > 0.0 and gaps["2D2I"] > 0.0
        ),
        "M9_depth_gains_non_monotonic": (
            any(value > 0.0 for value in slopes)
            and any(value < 0.0 for value in slopes)
        ),
        "M10_only_b11_link_nonpositive": (
            gains["2D2H"] < 0.0
            and gains["2D2D"] < gains["2D2F"]
            and gains["2D2B"] > 0.0
            and gains["2D2F"] > 0.0
            and gains["2D2I"] > 0.0
        ),
        "M11_2d2g_is_unique_minimum_validation_ce": (
            minimum_ce == "2D2G"
            and sum(value == validation_ce["2D2G"] for value in validation_ce.values())
            == 1
        ),
        "M12_2d2f_smallest_useful_state": smallest_useful_state == "2D2F",
        "M12_2d2f_saves_state_vs_2d2g": f_state_saving > 0,
        "M12_2d2f_ce_cost_vs_2d2g_below_explicit_tradeoff_bound": (
            0.0 < f_ce_cost <= 0.005
        ),
        "M13_2d2g_best_ce_at_minimum_useful_ring_count": (
            minimum_ring_candidates == ("2D2F", "2D2G")
            and best_minimum_ring_ce == "2D2G"
        ),
        "M14_2d2f_strongest_gain_and_smallest_useful_state": (
            strongest_useful_gain == "2D2F" and smallest_useful_state == "2D2F"
        ),
        "M15_exactly_one_frozen_head_to_head": (
            answers["M15"].get("evidence")
            == {
                "experiment_count": 1,
                "adaptation": "none",
                "candidates": ["2D2F", "2D2G"],
                "method": "larger_frozen_true_incremental_head_to_head",
            }
        ),
    }
    if not all(value is True for value in checks.values()):
        raise RuntimeError(
            "master scientific decision is not supported by sealed artifacts: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "checks": checks,
        "quantitative_policy": {
            "ce_state_tradeoff_max_2d2f_ce_cost_vs_2d2g": 0.005,
            "useful_link_requires_true_gain_gt": 0.0,
            "method_consistent_subset_targets_per_control": 262_144,
        },
        "passed": True,
    }


def _validate_master_matrix_numeric_contract(matrix: dict) -> dict:
    expected = {
        "2D2B", "2D2D", "2D2E", "2D2E-C1", "2D2F", "2D2G", "2D2H", "2D2I"
    }
    if not isinstance(matrix, dict) or set(matrix) != expected:
        raise RuntimeError("master comparison matrix row set is not exact")
    checks = {}
    for name in sorted(expected):
        row = matrix[name]
        classification = row.get("classification")
        checkpoint = row.get("checkpoint")
        runtime = _claim_number(row.get("runtime_seconds"), f"{name}.runtime")
        gpu_hours = _claim_number(row.get("gpu_hours"), f"{name}.gpu-hours")
        state_bytes = _claim_int(
            row.get("inference_state_bytes_b1"), f"{name}.state bytes"
        )
        ring_count = _claim_int(
            row.get("recurrent_ring_count"), f"{name}.ring count"
        )
        if (
            not isinstance(classification, str)
            or not classification.strip()
            or not isinstance(checkpoint, dict)
            or set(checkpoint) != {"path", "sha256"}
            or not isinstance(checkpoint.get("path"), str)
            or not Path(checkpoint["path"]).is_absolute()
            or not isinstance(checkpoint.get("sha256"), str)
            or SHA256_PATTERN.fullmatch(checkpoint["sha256"]) is None
            or runtime <= 0.0
            or gpu_hours <= 0.0
            or not math.isclose(gpu_hours, runtime / 3600.0, rel_tol=0.0, abs_tol=1e-12)
            or state_bytes <= 0
            or ring_count <= 0
        ):
            raise RuntimeError(f"master comparison matrix identity is invalid: {name}")
        for kind in ("parallel", "true_incremental"):
            if kind == "parallel" and name == "2D2E-C1":
                continue
            metrics = row.get(kind)
            required = {
                "real_validation_ce", "gain", "sequence_gap",
                "wins_vs_off", "wins_vs_shuffled",
            }
            if kind == "true_incremental":
                required |= {"targets_per_control", "subset_sha256"}
            if not isinstance(metrics, dict) or set(metrics) != required:
                raise RuntimeError(
                    f"master comparison matrix {kind} schema is invalid: {name}"
                )
            ce = _claim_number(metrics.get("real_validation_ce"), f"{name}.{kind}.CE")
            _claim_number(metrics.get("gain"), f"{name}.{kind}.gain")
            _claim_number(metrics.get("sequence_gap"), f"{name}.{kind}.gap")
            wins_off = _claim_int(metrics.get("wins_vs_off"), f"{name}.{kind}.wins off")
            wins_shuffled = _claim_int(
                metrics.get("wins_vs_shuffled"), f"{name}.{kind}.wins shuffled"
            )
            if ce <= 0.0 or wins_off > 1_048_576 or wins_shuffled > 1_048_576:
                raise RuntimeError(
                    f"master comparison matrix {kind} numeric range is invalid: {name}"
                )
            if kind == "true_incremental":
                targets = _claim_int(
                    metrics.get("targets_per_control"), f"{name}.targets per control"
                )
                subset = metrics.get("subset_sha256")
                if (
                    targets <= 0
                    or not isinstance(subset, str)
                    or SHA256_PATTERN.fullmatch(subset) is None
                ):
                    raise RuntimeError(
                        f"master comparison matrix true subset is invalid: {name}"
                    )
        architecture = row.get("comparison_architecture")
        gate = architecture.get("new_link_gate_tanh") if isinstance(architecture, dict) else None
        gate = _claim_number(gate, f"{name}.new-link gate")
        if architecture.get("passed") is not True or not -1.0 <= gate <= 1.0:
            raise RuntimeError(f"master comparison architecture is invalid: {name}")
        checks[name] = True
    c1_ci = matrix["2D2E-C1"].get("bootstrap_95_percent")
    if not isinstance(c1_ci, dict) or set(c1_ci) != {
        "off_minus_real", "shuffled_minus_real"
    }:
        raise RuntimeError("master C1 bootstrap interval set is not exact")
    for comparison, interval in c1_ci.items():
        if not isinstance(interval, dict) or set(interval) != {
            "bootstrap_mean", "confidence_level", "lower", "percentile_method",
            "resamples", "sampling_unit", "seed", "statistic", "upper",
        }:
            raise RuntimeError(f"master C1 bootstrap interval is invalid: {comparison}")
        lower = _claim_number(interval.get("lower"), f"C1.{comparison}.lower")
        upper = _claim_number(interval.get("upper"), f"C1.{comparison}.upper")
        _claim_number(interval.get("bootstrap_mean"), f"C1.{comparison}.mean")
        if (
            lower > upper
            or interval.get("confidence_level") != 0.95
            or interval.get("resamples") != 10_000
            or interval.get("seed") != 20260221
            or not all(
                isinstance(interval.get(key), str) and interval[key]
                for key in ("percentile_method", "sampling_unit", "statistic")
            )
        ):
            raise RuntimeError(f"master C1 bootstrap interval is reversed: {comparison}")
    return {"checks": checks, "passed": True}


def derive_master_scientific_claims(master_root: Path) -> dict:
    """Derive the authoritative matrix and M1-M15 answers from sealed JSON."""
    roots = {
        name: Path(report).resolve().parent for name, report in FINAL_REPORTS.items()
    }
    historical_root = master_root / "worktrees/master/results"
    roots.update(
        {
            "2D2D": historical_root / "experiment_2d2d_b2_w32_b11_recurrent_992",
            "2D2E": historical_root / "experiment_2d2e_b3_w64_b10_recurrent_960",
            "2D2B": historical_root / "experiment_2d2b_full_b12_b1_recurrent_bank",
        }
    )
    sources = {}
    bundles = {}
    for name, root in roots.items():
        summary = _claim_json_source(
            root / "result_summary.json", f"{name}.result_summary", sources
        )
        bundle = {"summary": summary}
        if name == "2D2E-C1":
            paired = _claim_json_source(
                root / "paired_results.json", f"{name}.paired_results", sources
            )
            bootstrap = _claim_json_source(
                root / "bootstrap_results.json", f"{name}.bootstrap_results", sources
            )
            subset = _claim_json_source(
                root / "subset_manifest.json", f"{name}.subset_manifest", sources
            )
            if (
                summary.get("paired")
                != {
                    "off_minus_real": paired.get("off_minus_real"),
                    "shuffled_minus_real": paired.get("shuffled_minus_real"),
                }
                or summary.get("controls") != paired.get("controls")
                or summary.get("bootstrap") != bootstrap
            ):
                raise RuntimeError("2D2E-C1 paired/bootstrap artifacts are not summary-bound")
            bundle.update(paired=paired, bootstrap=bootstrap, subset=subset)
        else:
            preflight = _claim_json_source(
                root / "preflight_audit.json", f"{name}.preflight_audit", sources
            )
            incremental = _claim_json_source(
                root / "incremental_validation.json",
                f"{name}.incremental_validation",
                sources,
            )
            memory = _claim_json_source(
                root / "memory_accounting.json",
                f"{name}.memory_accounting",
                sources,
            )
            performance = _claim_json_source(
                root / "performance.json", f"{name}.performance", sources
            )
            _assert_summary_artifact(
                summary,
                "true_incremental" if name == "2D2H" else "incremental",
                incremental,
                name,
            )
            _assert_summary_artifact(summary, "memory_accounting", memory, name)
            if name in {"2D2E", "2D2F", "2D2I"}:
                _assert_summary_artifact(summary, "performance", performance, name)
            architecture = preflight.get("architecture")
            if architecture is None:
                architecture = summary.get("architecture")
            if not isinstance(architecture, dict) or not architecture:
                raise RuntimeError(f"{name} structured architecture is absent")
            bundle.update(
                preflight=preflight,
                architecture=architecture,
                incremental=incremental,
                memory=memory,
                performance=performance,
            )
        bundles[name] = bundle

    def standard_row(
        name: str,
        link: str,
        ring_count: int,
        state_key: str,
        parallel_key: str,
        real_key: str,
        off_pair_key: str,
        shuffled_pair_key: str,
        true_real_key: str,
        true_off_pair_key: str,
        true_shuffled_pair_key: str,
        runtime_path: tuple[str, ...],
        subset_key: str,
    ) -> dict:
        bundle = bundles[name]
        summary = bundle["summary"]
        incremental = bundle["incremental"]
        memory = bundle["memory"]
        performance = bundle["performance"]

        def nested(value: dict, path: tuple[str, ...], label: str) -> object:
            current = value
            try:
                for key in path:
                    current = current[key]
            except (KeyError, TypeError) as error:
                raise RuntimeError(f"{name} structured claim path is absent: {label}") from error
            return current

        parallel = summary[parallel_key]
        runtime_seconds = _claim_number(
            nested(performance, runtime_path, "runtime"), f"{name}.runtime"
        )
        if name == "2D2G":
            runtime_seconds *= 3600.0
        state_bytes = _claim_int(
            nested(memory, ("B1", state_key), "state bytes"),
            f"{name}.state bytes",
        )
        true_gain_key = f"true_{link}_recurrent_gain"
        true_gap_key = f"true_{link}_sequence_gap"
        parallel_gain_key = f"{link}_recurrent_gain"
        parallel_gap_key = f"{link}_sequence_gap"
        if name == "2D2G":
            parallel_gain_key = "b3_gain"
        return {
            "classification": _classification_from_summary(name, summary),
            "checkpoint": _checkpoint_claim(summary, name),
            "new_link": link.upper(),
            "recurrent_ring_count": ring_count,
            "parallel": {
                "real_validation_ce": _claim_number(
                    nested(parallel, ("controls", real_key, "validation_loss"), "parallel CE"),
                    f"{name}.parallel CE",
                ),
                "gain": _claim_number(parallel[parallel_gain_key], f"{name}.parallel gain"),
                "sequence_gap": _claim_number(
                    parallel[parallel_gap_key], f"{name}.parallel gap"
                ),
                "wins_vs_off": _claim_int(
                    nested(parallel, (off_pair_key, "wins"), "parallel wins off"),
                    f"{name}.parallel wins off",
                ),
                "wins_vs_shuffled": _claim_int(
                    nested(
                        parallel,
                        (shuffled_pair_key, "wins"),
                        "parallel wins shuffled",
                    ),
                    f"{name}.parallel wins shuffled",
                ),
            },
            "true_incremental": {
                "real_validation_ce": _claim_number(
                    nested(
                        incremental,
                        ("controls", true_real_key, "validation_loss"),
                        "true CE",
                    ),
                    f"{name}.true CE",
                ),
                "gain": _claim_number(incremental[true_gain_key], f"{name}.true gain"),
                "sequence_gap": _claim_number(
                    incremental[true_gap_key], f"{name}.true gap"
                ),
                "wins_vs_off": _claim_int(
                    nested(
                        incremental,
                        (true_off_pair_key, "wins"),
                        "true wins off",
                    ),
                    f"{name}.true wins off",
                ),
                "wins_vs_shuffled": _claim_int(
                    nested(
                        incremental,
                        (true_shuffled_pair_key, "wins"),
                        "true wins shuffled",
                    ),
                    f"{name}.true wins shuffled",
                ),
                "targets_per_control": _claim_int(
                    incremental["targets_per_control"], f"{name}.targets per control"
                ),
                "subset_sha256": incremental[subset_key],
            },
            "inference_state_bytes_b1": state_bytes,
            "runtime_seconds": runtime_seconds,
            "gpu_hours": runtime_seconds / 3600.0,
            "runtime_artifact": performance,
        }

    def comparison_architecture(name: str) -> dict:
        bundle = bundles[name]
        summary = bundle["summary"]
        architecture = bundle["architecture"]
        parents = {
            "2D2B": "2D2A",
            "2D2D": "2D2B",
            "2D2E": "2D2D",
            "2D2F": "2D2D",
            "2D2G": "2D2G-A matched-age 2D2B continuation",
            "2D2H": "2D2B",
            "2D2I": "2D2E",
        }
        contracts = {
            "2D2B": {
                "windows": {"B1": 2, "B2": 1024, "B3": 1024, "B4": 1024},
                "recurrence": {"B1": "B12→B1", "B2": None, "B3": None, "B4": None},
                "new_link": "B12→B1",
            },
            "2D2D": {
                "windows": {"B1": 2, "B2": 32, "B3": 1024, "B4": 1024},
                "recurrence": {"B1": "B12→B1", "B2": "B11→B2", "B3": None, "B4": None},
                "new_link": "B11→B2",
            },
            "2D2E": {
                "windows": {"B1": 2, "B2": 32, "B3": 64, "B4": 1024},
                "recurrence": {"B1": "B12→B1", "B2": "B11→B2", "B3": "B10→B3", "B4": None},
                "new_link": "B10→B3",
            },
            "2D2F": {
                "windows": {"B1": 2, "B2": 32, "B3": 64, "B4": 1024},
                "recurrence": {"B1": "B12→B1", "B2": None, "B3": "B10→B3", "B4": None},
                "new_link": "B10→B3",
            },
            "2D2G": {
                "windows": {"B1": 2, "B2": 1024, "B3": 64, "B4": 1024},
                "recurrence": {"B1": "B12→B1", "B2": None, "B3": "B10→B3", "B4": None},
                "new_link": "B10→B3",
            },
            "2D2H": {
                "windows": {"B1": 2, "B2": 32, "B3": 1024, "B4": 1024},
                "recurrence": {"B1": None, "B2": "B11→B2", "B3": None, "B4": None},
                "new_link": "B11→B2",
            },
            "2D2I": {
                "windows": {"B1": 2, "B2": 32, "B3": 64, "B4": 128},
                "recurrence": {"B1": "B12→B1", "B2": "B11→B2", "B3": "B10→B3", "B4": "B9→B4"},
                "new_link": "B9→B4",
            },
        }
        contract = contracts[name]
        if name == "2D2B":
            architecture_checks = {
                "b1_window": architecture.get("b1_local_window") == 2,
                "later_windows": architecture.get("b2_b12_windows") == [1024] * 11,
                "b1_source": architecture.get("source")
                == "B12 post-MLP residual immediately before final LayerNorm",
            }
            gate = summary.get("final_gate_effective")
        elif name == "2D2G":
            architecture_checks = {
                "b1_window": architecture.get("b1", {}).get("local_window") == 2,
                "b1_recurrence": architecture.get("b1", {}).get("source") == "B12",
                "b2_window": architecture.get("b2", {}).get("local_window") == 1024,
                "b2_recurrence_absent": architecture.get("b2", {}).get("recurrent") is False
                and architecture.get("b2", {}).get("b11_ring") is False,
                "b3_window": architecture.get("b3", {}).get("local_window") == 64,
                "b3_recurrence": architecture.get("b3", {}).get("source")
                == "B10 post-MLP residual before B11",
                "b4_later_windows": architecture.get("b4_b12_local_window") == 1024,
            }
            gate = summary.get("final_gates", {}).get("b3")
        else:
            expected_links = {
                "2D2D": {"B12_to_B1", "B11_to_B2"},
                "2D2E": {"B12_to_B1", "B11_to_B2", "B10_to_B3"},
                "2D2F": {"B12_to_B1", "B10_to_B3"},
                "2D2H": {"B12_to_B1", "B11_to_B2"},
                "2D2I": {"B12_to_B1", "B11_to_B2", "B10_to_B3", "B9_to_B4"},
            }[name]
            links = architecture.get("links")
            architecture_checks = {
                "b1_window": architecture.get("b1_local_window") == contract["windows"]["B1"],
                "b2_window": architecture.get("b2_local_window") == contract["windows"]["B2"],
                "link_set": isinstance(links, dict) and set(links) == expected_links,
            }
            if name in {"2D2E", "2D2F", "2D2I"}:
                architecture_checks["b3_window"] = (
                    architecture.get("b3_local_window") == contract["windows"]["B3"]
                )
            else:
                architecture_checks["b3_later_windows"] = (
                    architecture.get("b3_b12_windows") == [1024] * 10
                )
            if name == "2D2I":
                architecture_checks.update(
                    b4_window=architecture.get("b4_local_window") == 128,
                    b5_later_windows=architecture.get("b5_b12_windows") == [1024] * 8,
                )
            elif name in {"2D2E", "2D2F"}:
                architecture_checks["b4_later_windows"] = (
                    architecture.get("b4_b12_windows") == [1024] * 9
                )
            if name == "2D2D":
                gate = summary.get("final_tanh_g_rec_b2")
            elif name == "2D2H":
                architecture_checks["b1_recurrence_physically_absent"] = (
                    links.get("B12_to_B1", {}).get("present") is False
                    and links.get("B12_to_B1", {}).get("gate_parameter_present") is False
                )
                gate = summary.get("gate_diagnostics", {}).get("final_tanh_g_rec_b2")
            else:
                gate = summary.get("final_gates", {}).get(
                    {"2D2E": "b3", "2D2F": "b3", "2D2I": "b4"}[name]
                )
        if not all(value is True for value in architecture_checks.values()):
            raise RuntimeError(
                f"{name} comparison architecture differs from structured artifact: "
                + json.dumps(architecture_checks, sort_keys=True)
            )
        return {
            "parent": parents[name],
            "local_window": contract["windows"],
            "recurrence": contract["recurrence"],
            "new_link": contract["new_link"],
            "new_link_gate_tanh": _claim_number(gate, f"{name}.new-link gate"),
            "structured_architecture_sha256": hashlib.sha256(
                _canonical_json(architecture).encode()
            ).hexdigest(),
            "checks": architecture_checks,
            "passed": True,
        }

    matrix = {
        "2D2D": standard_row(
            "2D2D", "b2", 2, "total_experimental_inference_state_bytes",
            "parallel", "new_real", "new_real_vs_b2_off",
            "new_real_vs_b2_shuffled", "both_real",
            "both_real_vs_b2_off_sequences",
            "both_real_vs_b2_shuffled_sequences", ("training", "total_wall_seconds"),
            "canonical_subset_sha256",
        ),
        "2D2F": standard_row(
            "2D2F", "b3", 2, "total_experimental_inference_state_bytes",
            "parallel", "all_real", "all_real_vs_b3_off",
            "all_real_vs_b3_shuffled", "all_real", "all_real_vs_b3_off_sequences",
            "all_real_vs_b3_shuffled_sequences", ("training_wall_seconds",),
            "canonical_subset_sha256",
        ),
        "2D2G": standard_row(
            "2D2G", "b3", 2, "total_inference_state_bytes", "parallel",
            "real", "real_vs_off_batches", "real_vs_shuffled_batches", "real",
            "real_vs_off_sequences", "real_vs_shuffled_sequences",
            ("recorded_lane_gpu_hours",), "subset_sha256",
        ),
        "2D2H": standard_row(
            "2D2H", "b2", 1, "total_experimental_inference_state_bytes",
            "parallel_final", "new_real", "new_real_vs_b2_off",
            "new_real_vs_b2_shuffled", "real", "real_vs_b2_off_sequences",
            "real_vs_b2_shuffled_sequences", ("training", "total_wall_seconds"),
            "canonical_subset_sha256",
        ),
        "2D2I": standard_row(
            "2D2I", "b4", 4, "total_experimental_inference_state_bytes",
            "parallel", "all_real", "all_real_vs_b4_off",
            "all_real_vs_b4_shuffled", "all_real", "all_real_vs_b4_off_sequences",
            "all_real_vs_b4_shuffled_sequences", ("training_wall_seconds",),
            "canonical_subset_sha256",
        ),
        "2D2E": standard_row(
            "2D2E", "b3", 3, "total_experimental_inference_state_bytes",
            "parallel", "all_real", "all_real_vs_b3_off",
            "all_real_vs_b3_shuffled", "all_real", "all_real_vs_b3_off_sequences",
            "all_real_vs_b3_shuffled_sequences", ("training_wall_seconds",),
            "canonical_subset_sha256",
        ),
    }
    c1 = bundles["2D2E-C1"]["summary"]
    matrix["2D2E-C1"] = {
        "classification": _classification_from_summary("2D2E-C1", c1),
        "checkpoint": _checkpoint_claim(c1, "2D2E-C1"),
        "new_link": "B3",
        "recurrent_ring_count": 3,
        "true_incremental": {
            "real_validation_ce": _claim_number(
                c1["controls"]["all_real"]["validation_loss"], "C1 true CE"
            ),
            "gain": _claim_number(c1["confirm_gain"], "C1 gain"),
            "sequence_gap": _claim_number(c1["confirm_sequence_gap"], "C1 gap"),
            "wins_vs_off": _claim_int(
                c1["paired"]["off_minus_real"]["wins"], "C1 wins off"
            ),
            "wins_vs_shuffled": _claim_int(
                c1["paired"]["shuffled_minus_real"]["wins"],
                "C1 wins shuffled",
            ),
            "targets_per_control": _claim_int(
                c1["controls"]["all_real"]["validation_targets"], "C1 targets"
            ),
            "subset_sha256": bundles["2D2E-C1"]["subset"]["c1_subset"][
                "batch_collection_sha256"
            ],
        },
        "bootstrap_95_percent": c1["bootstrap"],
        "inference_state_bytes_b1": matrix["2D2E"]["inference_state_bytes_b1"],
        "runtime_seconds": _claim_number(
            c1["performance"]["wall_seconds"], "C1 runtime"
        ),
        "gpu_hours": c1["performance"]["wall_seconds"] / 3600.0,
        "runtime_artifact": c1["performance"],
    }
    b = bundles["2D2B"]
    b_summary = b["summary"]
    b_runtime_seconds = _claim_number(
        b["performance"]["training"]["total_wall_seconds"], "2D2B.runtime"
    )
    matrix["2D2B"] = {
        "classification": _classification_from_summary("2D2B", b_summary),
        "checkpoint": _checkpoint_claim(b_summary, "2D2B"),
        "new_link": "B1",
        "recurrent_ring_count": 1,
        "parallel": {
            "real_validation_ce": _claim_number(
                b_summary["parallel"]["controls"]["full_real"]["validation_loss"],
                "2D2B.parallel CE",
            ),
            "gain": _claim_number(
                b_summary["parallel"]["full_bank_gain"], "2D2B.parallel gain"
            ),
            "sequence_gap": _claim_number(
                b_summary["parallel"]["sequence_gap"], "2D2B.parallel gap"
            ),
            "wins_vs_off": _claim_int(
                b_summary["parallel"]["full_vs_plain"]["wins"],
                "2D2B.parallel wins off",
            ),
            "wins_vs_shuffled": _claim_int(
                b_summary["parallel"]["full_vs_shuffled"]["wins"],
                "2D2B.parallel wins shuffled",
            ),
        },
        "true_incremental": {
            "real_validation_ce": _claim_number(
                b["incremental"]["controls"]["full_real"]["validation_loss"],
                "2D2B.true CE",
            ),
            "gain": _claim_number(
                b["incremental"]["true_full_gain"], "2D2B.true gain"
            ),
            "sequence_gap": _claim_number(
                b["incremental"]["true_sequence_gap"], "2D2B.true gap"
            ),
            "wins_vs_off": _claim_int(
                b["incremental"]["full_vs_plain_sequences"]["wins"],
                "2D2B.true wins off",
            ),
            "wins_vs_shuffled": _claim_int(
                b["incremental"]["full_vs_shuffled_sequences"]["wins"],
                "2D2B.true wins shuffled",
            ),
            "targets_per_control": _claim_int(
                b["incremental"]["targets_per_control"], "2D2B.targets per control"
            ),
            "subset_sha256": b["incremental"]["canonical_subset_sha256"],
        },
        "inference_state_bytes_b1": _claim_int(
            b["memory"]["B1"]["total_experimental_inference_state_bytes"],
            "2D2B.state bytes",
        ),
        "runtime_seconds": b_runtime_seconds,
        "gpu_hours": b_runtime_seconds / 3600.0,
        "runtime_artifact": b["performance"],
    }
    for name in ("2D2B", "2D2D", "2D2E", "2D2F", "2D2G", "2D2H", "2D2I"):
        matrix[name]["comparison_architecture"] = comparison_architecture(name)
    if matrix["2D2E-C1"]["checkpoint"] != matrix["2D2E"]["checkpoint"]:
        raise RuntimeError("2D2E-C1 did not use the exact frozen 2D2E checkpoint")
    c1_architecture = json.loads(
        json.dumps(matrix["2D2E"]["comparison_architecture"])
    )
    c1_architecture.update(
        parent="2D2E frozen",
        new_link="B10→B3 (frozen confirmation)",
    )
    matrix["2D2E-C1"]["comparison_architecture"] = c1_architecture
    matrix_numeric_audit = _validate_master_matrix_numeric_contract(matrix)
    common = {
        name: matrix[name]["true_incremental"] for name in ("2D2E", "2D2F", "2D2G", "2D2H", "2D2I")
    }
    common_subsets = {row["subset_sha256"] for row in common.values()}
    if (
        common_subsets != {
            "8befbf790b3e522747cd39da306ec124464bf8dde1604caf64f299efa7e36216"
        }
        or {row["targets_per_control"] for row in common.values()} != {262_144}
    ):
        raise RuntimeError("method-consistent CE comparison subset is not exact")

    c1_true = matrix["2D2E-C1"]["true_incremental"]
    ci = matrix["2D2E-C1"]["bootstrap_95_percent"]
    f_true = matrix["2D2F"]["true_incremental"]
    g_true = matrix["2D2G"]["true_incremental"]
    h_true = matrix["2D2H"]["true_incremental"]
    i_true = matrix["2D2I"]["true_incremental"]
    e_true = matrix["2D2E"]["true_incremental"]
    d_true = matrix["2D2D"]["true_incremental"]
    b_true = matrix["2D2B"]["true_incremental"]
    state_saving_f_vs_g = (
        matrix["2D2G"]["inference_state_bytes_b1"]
        - matrix["2D2F"]["inference_state_bytes_b1"]
    )
    answers = {
        "M1": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M1"], "evidence": {"c1_gain": c1_true["gain"]}},
        "M2": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M2"], "evidence": {"c1_sequence_gap": c1_true["sequence_gap"]}},
        "M3": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M3"], "evidence": {"off_minus_real_ci_lower": ci["off_minus_real"]["lower"], "off_minus_real_ci_upper": ci["off_minus_real"]["upper"], "shuffled_minus_real_ci_lower": ci["shuffled_minus_real"]["lower"], "shuffled_minus_real_ci_upper": ci["shuffled_minus_real"]["upper"]}},
        "M4": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M4"], "evidence": {"2d2e_true_b3_gain": e_true["gain"], "2d2f_true_b3_gain": f_true["gain"], "gain_delta_2d2f_minus_2d2e": f_true["gain"] - e_true["gain"]}},
        "M5": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M5"], "evidence": {"2d2f_b2_w32_true_b3_gain": f_true["gain"], "2d2g_b2_w1024_true_b3_gain": g_true["gain"], "gain_delta_w32_minus_w1024": f_true["gain"] - g_true["gain"]}},
        "M6": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M6"], "evidence": {"historical_2d2d_with_b1_true_b2_gain": d_true["gain"], "historical_2d2d_with_b1_true_b2_sequence_gap": d_true["sequence_gap"], "2d2h_without_b1_true_b2_gain": h_true["gain"], "2d2h_without_b1_true_b2_sequence_gap": h_true["sequence_gap"], "gain_delta_2d2h_minus_2d2d": h_true["gain"] - d_true["gain"]}},
        "M7": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M7"], "evidence": {"b1_2d2b_true_gain": b_true["gain"], "b2_2d2h_true_gain": h_true["gain"], "b3_2d2f_true_gain": f_true["gain"], "b4_2d2i_true_gain": i_true["gain"]}},
        "M8": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M8"], "evidence": {"2d2i_true_b4_gain": i_true["gain"], "2d2i_true_b4_sequence_gap": i_true["sequence_gap"]}},
        "M9": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M9"], "evidence": {"B1_2d2b": b_true["gain"], "B2_historical_2d2d": d_true["gain"], "B2_without_b1_2d2h": h_true["gain"], "B3_2d2f": f_true["gain"], "B4_2d2i": i_true["gain"]}},
        "M10": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M10"], "evidence": {"delete_b11_to_b2": True, "historical_b11_to_b2_true_gain": d_true["gain"], "b11_to_b2_without_b1_true_gain": h_true["gain"], "retain_b12_to_b1": b_true["gain"], "retain_b10_to_b3": f_true["gain"], "retain_b9_to_b4": i_true["gain"]}},
        "M11": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M11"], "evidence": {"method": "true_incremental_common_262144_target_subset", "validation_ce": {name: row["real_validation_ce"] for name, row in common.items()}, "minimum": "2D2G"}},
        "M12": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M12"], "evidence": {"validation_ce": {name: row["real_validation_ce"] for name, row in common.items()}, "state_bytes_b1": {name: matrix[name]["inference_state_bytes_b1"] for name in common}, "2d2f_state_saving_bytes_vs_2d2g": state_saving_f_vs_g, "2d2f_state_saving_fraction_vs_2d2g": state_saving_f_vs_g / matrix["2D2G"]["inference_state_bytes_b1"], "2d2f_ce_cost_vs_2d2g": f_true["real_validation_ce"] - g_true["real_validation_ce"]}},
        "M13": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M13"], "evidence": {"validation_ce": {name: row["real_validation_ce"] for name, row in common.items()}, "ring_count": {name: matrix[name]["recurrent_ring_count"] for name in common}, "same_ring_count_2d2f_2d2g": True, "lower_ce_at_two_rings": "2D2G"}},
        "M14": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M14"], "evidence": {"2d2f_true_gain": f_true["gain"], "2d2g_true_gain": g_true["gain"], "2d2f_state_bytes": matrix["2D2F"]["inference_state_bytes_b1"], "2d2g_state_bytes": matrix["2D2G"]["inference_state_bytes_b1"], "2d2g_ce_advantage": f_true["real_validation_ce"] - g_true["real_validation_ce"], "canonical": "2D2F"}},
        "M15": {"answer_code": MASTER_SCIENTIFIC_ANSWER_CODES["M15"], "evidence": {"experiment_count": 1, "adaptation": "none", "candidates": ["2D2F", "2D2G"], "method": "larger_frozen_true_incremental_head_to_head"}},
    }
    for name, answer in answers.items():
        answer["conclusion"] = MASTER_SCIENTIFIC_CONCLUSIONS[name]
    decision_audit = _validate_master_scientific_decisions(matrix, answers)
    return {
        "schema": "parallel_2d2_master_scientific_claims_v1",
        "source_artifacts": dict(sorted(sources.items())),
        "matrix": matrix,
        "matrix_numeric_audit": matrix_numeric_audit,
        "answers": answers,
        "decision_audit": decision_audit,
        "recommended_next_experiment": MASTER_SCIENTIFIC_CONCLUSIONS["M15"],
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _master_matrix_markdown(claims: dict) -> str:
    matrix = claims.get("matrix")
    order = ("2D2B", "2D2D", "2D2E", "2D2E-C1", "2D2F", "2D2G", "2D2H", "2D2I")
    if not isinstance(matrix, dict) or not set(order).issubset(matrix):
        raise RuntimeError("master comparison matrix experiment set is incomplete")
    headers = (
        "Experiment", "Parent", "B1 local window", "B1 recurrence",
        "B2 local window", "B2 recurrence", "B3 local window",
        "B3 recurrence", "B4 local window", "B4 recurrence",
        "New-link gate tanh", "Parallel gain", "Parallel sequence gap",
        "True gain", "True sequence gap", "Paired wins vs Off",
        "Paired wins vs Shuffled", "Final validation CE",
        "Memory state bytes", "Runtime seconds", "GPU-hours", "Classification",
    )

    def cell(value: object) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError("master comparison matrix contains nonfinite value")
            rendered = json.dumps(value, allow_nan=False)
        elif isinstance(value, str):
            rendered = value
        else:
            raise RuntimeError("master comparison matrix cell has unsupported type")
        return rendered.replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for name in order:
        row = matrix[name]
        architecture = row.get("comparison_architecture")
        true = row.get("true_incremental")
        parallel = row.get("parallel", {})
        if (
            not isinstance(row, dict)
            or not isinstance(architecture, dict)
            or not isinstance(true, dict)
            or not isinstance(parallel, dict)
            or architecture.get("passed") is not True
        ):
            raise RuntimeError(f"master comparison matrix row is incomplete: {name}")
        windows = architecture.get("local_window")
        recurrence = architecture.get("recurrence")
        if (
            not isinstance(windows, dict)
            or set(windows) != {"B1", "B2", "B3", "B4"}
            or not isinstance(recurrence, dict)
            or set(recurrence) != {"B1", "B2", "B3", "B4"}
        ):
            raise RuntimeError(f"master comparison architecture row differs: {name}")
        values = (
            name,
            architecture.get("parent"),
            windows["B1"], recurrence["B1"] or "OFF",
            windows["B2"], recurrence["B2"] or "OFF",
            windows["B3"], recurrence["B3"] or "OFF",
            windows["B4"], recurrence["B4"] or "OFF",
            architecture.get("new_link_gate_tanh"),
            parallel.get("gain"), parallel.get("sequence_gap"),
            true.get("gain"), true.get("sequence_gap"),
            true.get("wins_vs_off"), true.get("wins_vs_shuffled"),
            true.get("real_validation_ce"), row.get("inference_state_bytes_b1"),
            row.get("runtime_seconds"), row.get("gpu_hours"),
            row.get("classification"),
        )
        if len(values) != len(headers):
            raise AssertionError("master comparison matrix internal column mismatch")
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    return "\n".join(lines)


def _validate_master_matrix_block(text: str, claims: dict) -> str:
    markdown = _master_matrix_markdown(claims)
    exact = f"{MASTER_MATRIX_BEGIN}\n{markdown}\n{MASTER_MATRIX_END}"
    if text.count(MASTER_MATRIX_BEGIN) != 1 or text.count(MASTER_MATRIX_END) != 1:
        raise RuntimeError("master report comparison matrix block is not unique")
    if exact not in text:
        raise RuntimeError("master report comparison matrix differs from artifacts")
    return hashlib.sha256(markdown.encode()).hexdigest()


def _validate_master_claim_block(text: str, claims: dict) -> str:
    canonical = _canonical_json(claims)
    exact = f"{MASTER_CLAIMS_BEGIN}\n{canonical}\n{MASTER_CLAIMS_END}"
    if text.count(MASTER_CLAIMS_BEGIN) != 1 or text.count(MASTER_CLAIMS_END) != 1:
        raise RuntimeError("master report scientific claims block is not unique")
    if exact not in text:
        raise RuntimeError("master report scientific claims block differs from artifacts")
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_master_claim_sections(
    sections: dict[str, str], claims: dict
) -> dict[str, str]:
    answers = claims.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(MASTER_SCIENTIFIC_QUESTIONS):
        raise RuntimeError("master scientific answer claim set is not exact")
    digests = {}
    for name, question in MASTER_SCIENTIFIC_QUESTIONS.items():
        body = sections[name]
        lines = body.splitlines()
        if not lines or lines[0] != question:
            raise RuntimeError(f"master report {name} question text differs")
        answer = answers[name]
        evidence_json = _canonical_json(answer.get("evidence"))
        expected = {
            "ANSWER_CODE": answer.get("answer_code"),
            "EVIDENCE_JSON": evidence_json,
            "CONCLUSION": answer.get("conclusion"),
        }
        for label, value in expected.items():
            matches = [
                line[len(label) + 2 :]
                for line in lines
                if line.startswith(label + ": ")
            ]
            if matches != [value]:
                raise RuntimeError(
                    f"master report {name} {label} differs from structured claims"
                )
        if answer.get("answer_code") != MASTER_SCIENTIFIC_ANSWER_CODES[name]:
            raise RuntimeError(f"master report {name} scientific answer is false")
        if answer.get("conclusion") != MASTER_SCIENTIFIC_CONCLUSIONS[name]:
            raise RuntimeError(f"master report {name} conclusion differs")
        digests[name] = hashlib.sha256(_canonical_json(answer).encode()).hexdigest()
    return digests


def _master_protocol_sections(text: str) -> dict[str, str]:
    matches = list(
        re.finditer(r"(?m)^M(?P<number>[1-9]|1[0-5])\.\s*(?P<first>.*)$", text)
    )
    if [int(match.group("number")) for match in matches] != list(range(1, 16)):
        raise RuntimeError("master report does not contain exactly ordered M1-M15 sections")
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = (match.group("first") + "\n" + text[match.end():end]).strip()
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_→-]*", body)
        lowered = body.casefold()
        if (
            len(body) < 24
            or len(words) < 5
            or body.rstrip().endswith("?")
            or any(token in lowered for token in ("tbd", "todo", "placeholder", "not yet answered"))
        ):
            raise RuntimeError(
                f"master report M{match.group('number')} body is not substantive"
            )
        sections[f"M{match.group('number')}"] = body
    return sections


def _classification_from_summary(name: str, summary: dict) -> str:
    key = "classification" if name == "2D2E-C1" else "primary_classification"
    value = summary.get(key)
    if not isinstance(value, str) or len(value.strip()) < 4:
        raise RuntimeError(f"{name} structured classification is absent")
    return value.strip()


def _validate_final_audit_outcome_binding(
    name: str, audit: dict, classification: str
) -> dict:
    """Bind the heterogeneous, published FINAL_AUDIT schemas fail-closed."""
    audit_classification = audit.get("classification")
    audit_experiment = audit.get("experiment")
    if name == "2D2H":
        # The sealed 2D2H FINAL_AUDIT predates a top-level classification
        # field.  Its exact outcome identity is instead the experiment field;
        # the classification remains independently bound by the structured
        # result summary and both final reports below.
        if "classification" in audit or audit_experiment != name:
            raise RuntimeError(
                "2D2H FINAL_AUDIT outcome binding differs from its published schema"
            )
        return {
            "schema": "experiment_identity_without_classification",
            "audit_classification_field_present": False,
            "audit_experiment_field_verified": True,
            "passed": True,
        }
    if audit_classification != classification:
        raise RuntimeError(
            f"{name} FINAL_AUDIT classification differs from its result summary"
        )
    if name == "2D2E-C1" and audit_experiment != name:
        raise RuntimeError(
            "2D2E-C1 FINAL_AUDIT experiment identity differs from its published schema"
        )
    return {
        "schema": "classification_bound"
        + ("_with_experiment_identity" if name == "2D2E-C1" else ""),
        "audit_classification_field_present": True,
        "audit_experiment_field_verified": name == "2D2E-C1",
        "passed": True,
    }


def validate_required_result_artifacts(master_root: Path) -> dict:
    """Validate the Section-54 minimum as actual nonempty, link-free files."""
    audited = {}
    production_contract = set(FINAL_REPORTS) == {
        "2D2E-C1", "2D2F", "2D2G", "2D2H", "2D2I"
    }
    for name, report_value in FINAL_REPORTS.items():
        report_path = lexical_absolute(report_value)
        result_root = report_path.parent
        if production_contract and name == "2D2E-C1":
            required = set(C1_ARTIFACT_MINIMUM)
        elif production_contract and name in {"2D2F", "2D2G", "2D2H", "2D2I"}:
            required = set(TRAINING_ARTIFACT_MINIMUM)
            required.add(report_path.name)
        else:
            # Synthetic/unit-test contracts still require their three sealed
            # outcome artifacts without pretending to be a protocol lane.
            required = {report_path.name, "FINAL_AUDIT.json", "result_summary.json"}
        rows = {}
        for filename in sorted(required):
            candidate = result_root / filename
            require_symlink_free_regular_file(
                candidate,
                master_root,
                f"{name} required Section-54 artifact {filename}",
            )
            rows[filename] = {
                "path": str(lexical_absolute(candidate)),
                "bytes": candidate.stat().st_size,
                "sha256": file_sha256(candidate),
            }
        audited[name] = {
            "result_directory": str(result_root),
            "required_names": sorted(required),
            "artifacts": rows,
            "passed": True,
        }
    return {"experiments": audited, "passed": True}


def _markdown_h2_section(text: str, heading: str) -> str:
    matches = list(
        re.finditer(
            rf"(?ms)^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text
        )
    )
    if len(matches) != 1 or not matches[0].group(1).strip():
        raise RuntimeError(f"master report lacks one substantive '{heading}' section")
    return matches[0].group(1)


def _required_master_report_opening(claims: dict) -> str:
    matrix = claims.get("matrix")
    if not isinstance(matrix, dict):
        raise RuntimeError("master report opening lacks a structured matrix")
    rows = (
        ("2D2E-C1 CONFIRMATION", "2D2E-C1"),
        ("2D2F PRIMARY CLASSIFICATION", "2D2F"),
        ("2D2G PRIMARY CLASSIFICATION", "2D2G"),
        ("2D2H PRIMARY CLASSIFICATION", "2D2H"),
        ("2D2I PRIMARY CLASSIFICATION", "2D2I"),
    )
    lines = ["PARALLEL 4-GPU EXPERIMENT BATCH COMPLETE", ""]
    for heading, name in rows:
        classification = matrix.get(name, {}).get("classification")
        if not isinstance(classification, str) or not classification.strip():
            raise RuntimeError(f"master report opening classification is absent: {name}")
        lines.extend((heading + ":", classification, ""))
    return "\n".join(lines)


def _validate_master_human_provenance(master_text: str, claims: dict) -> dict:
    expected_opening = _required_master_report_opening(claims)
    if not master_text.startswith(expected_opening):
        raise RuntimeError("master report does not begin with the exact Section-62 opening")
    headings = (
        "Compact architecture matrix",
        "Final checkpoint provenance",
        "Git provenance",
        "Artifact directories",
        "Pod and volume status",
        "Runtime and cost accounting",
        "Scientific answers M1–M15",
        "Recommended next experiment",
    )
    sections = {heading: _markdown_h2_section(master_text, heading) for heading in headings}
    checkpoint_body = sections["Final checkpoint provenance"]
    for name in ("2D2E-C1", "2D2F", "2D2G", "2D2H", "2D2I"):
        checkpoint = claims["matrix"][name].get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("path") not in checkpoint_body
            or checkpoint.get("sha256") not in checkpoint_body
            or name not in checkpoint_body
        ):
            raise RuntimeError(
                f"master report human checkpoint provenance is incomplete: {name}"
            )
    git_body = sections["Git provenance"]
    for name, contract in FINAL_GIT_REPOSITORIES.items():
        if (
            name not in git_body
            or contract["branch"] not in git_body
            or re.search(
                rf"(?m)^- {re.escape(name)}: .*commit `([0-9a-f]{{40}})`.*$",
                git_body,
            ) is None
        ):
            raise RuntimeError(f"master report human Git provenance is incomplete: {name}")
    if MASTER_FINALIZATION_IMPLEMENTATION_TAG not in git_body:
        raise RuntimeError("master report omits the signed implementation tag")
    artifact_body = sections["Artifact directories"]
    for name, report in FINAL_REPORTS.items():
        if name not in artifact_body or str(lexical_absolute(report).parent) not in artifact_body:
            raise RuntimeError(
                f"master report human artifact provenance is incomplete: {name}"
            )
    pod_body = sections["Pod and volume status"]
    if not all(
        token in pod_body
        for token in (POD_ID, POD_NAME, VOLUME_ID, "retained")
    ) or not any(status in pod_body for status in ("RUNNING", "EXITED")):
        raise RuntimeError("master report pod/volume provenance is incomplete")
    runtime_body = sections["Runtime and cost accounting"]
    if not all(
        token.lower() in runtime_body.lower()
        for token in (
            "GPU0", "GPU1", "GPU2", "GPU3", "total wall-clock", "GPU-hours"
        )
    ):
        raise RuntimeError("master report runtime/cost provenance is incomplete")
    recommended = sections["Recommended next experiment"]
    if MASTER_SCIENTIFIC_CONCLUSIONS["M15"] not in recommended:
        raise RuntimeError("master report recommended experiment is not exact")
    return {
        "exact_section_62_opening": True,
        "required_human_sections": list(headings),
        "checkpoint_provenance": True,
        "git_provenance": True,
        "artifact_provenance": True,
        "pod_volume_provenance": True,
        "runtime_cost_provenance": True,
        "passed": True,
    }


def validate_final_report_evidence(
    master_root: Path, run_root: Path, run_id: str, path_value: str
) -> dict:
    path = run_root / FINALIZATION_EVIDENCE_FILES["report"]
    payload, identity = _load_finalization_manifest(
        path_value, path, "parallel_2d2_final_report_evidence_v1", run_id
    )
    if set(payload) != {
        "schema_version",
        "kind",
        "run_id",
        "pod",
        "created_utc",
        "master_report",
        "tracked_master_report",
        "experiment_reports",
        "result_summaries",
        "final_audits",
        "scientific_claims",
    }:
        raise RuntimeError("report finalization evidence has unexpected keys")
    master_report = _validate_file_evidence(
        payload.get("master_report"),
        master_root / "MASTER_FINAL_REPORT.md",
        "coordinator master report",
        master_root,
    )
    tracked_master = _validate_file_evidence(
        payload.get("tracked_master_report"),
        master_root / "worktrees/master/MASTER_FINAL_REPORT.md",
        "Git-tracked master report",
        master_root,
    )
    if master_report["sha256"] != tracked_master["sha256"]:
        raise RuntimeError("coordinator and Git-tracked master report bytes differ")
    master_text = Path(master_report["path"]).read_text(encoding="utf-8")
    protocol_sections = _master_protocol_sections(master_text)
    expected_claims = derive_master_scientific_claims(master_root)
    if payload.get("scientific_claims") != expected_claims:
        raise RuntimeError(
            "master scientific claims differ from the sealed structured artifacts"
        )
    scientific_claims_sha256 = _validate_master_claim_block(
        master_text, expected_claims
    )
    comparison_matrix_sha256 = _validate_master_matrix_block(
        master_text, expected_claims
    )
    scientific_answer_sha256 = _validate_master_claim_sections(
        protocol_sections, expected_claims
    )
    production_contract = set(FINAL_REPORTS) == {
        "2D2E-C1", "2D2F", "2D2G", "2D2H", "2D2I"
    }
    human_provenance = (
        _validate_master_human_provenance(master_text, expected_claims)
        if production_contract
        else {"synthetic_contract": True, "passed": True}
    )
    master_report_checks = {
        "all_experiments_named": all(
            name in master_text for name in ("2D2E-C1", "2D2F", "2D2G", "2D2H", "2D2I")
        ),
        "all_scientific_questions_answered": all(
            re.search(rf"(?m)^M{index}\.", master_text) is not None
            for index in range(1, 16)
        ),
        "exactly_one_recommended_next_experiment": master_text.lower().count(
            "recommended next experiment"
        )
        == 1,
        "exact_ordered_substantive_M1_through_M15": len(protocol_sections) == 15,
        "machine_readable_claims_exact": True,
        "exact_protocol_comparison_matrix_bound": bool(comparison_matrix_sha256),
        "M1_through_M15_structurally_bound": len(scientific_answer_sha256) == 15,
        "exact_protocol_completion_line": master_text.rstrip().endswith(
            "# PARALLEL 4-GPU BATCH COMPLETE"
        ),
        "exact_section_62_human_provenance": human_provenance.get("passed") is True,
    }
    if not all(master_report_checks.values()):
        raise RuntimeError(
            f"master final report protocol content is incomplete: {master_report_checks}"
        )
    reports = payload.get("experiment_reports")
    summaries = payload.get("result_summaries")
    audits = payload.get("final_audits")
    if (
        not isinstance(reports, dict)
        or set(reports) != set(FINAL_REPORTS)
        or not isinstance(summaries, dict)
        or set(summaries) != set(FINAL_REPORTS)
        or not isinstance(audits, dict)
        or set(audits) != set(FINAL_AUDITS)
    ):
        raise RuntimeError("final report/audit experiment set is not exact")
    audited_reports = {
        name: _validate_file_evidence(
            reports[name], Path(expected), f"{name} final report", master_root
        )
        for name, expected in FINAL_REPORTS.items()
    }
    required_result_artifacts = validate_required_result_artifacts(master_root)
    audited_summaries = {}
    audited_audits = {}
    structured_outcomes = {}
    for name, expected in FINAL_AUDITS.items():
        result_path = Path(FINAL_REPORTS[name]).resolve().parent / "result_summary.json"
        summary_row = _validate_file_evidence(
            summaries[name], result_path, f"{name} result summary", master_root
        )
        summary = read_json(result_path)
        row = _validate_file_evidence(
            audits[name], Path(expected), f"{name} FINAL_AUDIT", master_root
        )
        audit = read_json(Path(expected))
        if audit.get("passed") is not True:
            raise RuntimeError(f"{name} FINAL_AUDIT is not passing")
        classification = _classification_from_summary(name, summary)
        audit_outcome_binding = _validate_final_audit_outcome_binding(
            name, audit, classification
        )
        if (
            summary.get("experiment") != name
            or classification not in Path(FINAL_REPORTS[name]).read_text(encoding="utf-8")
            or classification not in master_text
        ):
            raise RuntimeError(
                f"{name} classification differs across summary/final reports"
            )
        audited_summaries[name] = summary_row
        audited_audits[name] = row
        structured_outcomes[name] = {
            "classification": classification,
            "summary_sha256": summary_row["sha256"],
            "audit_sha256": row["sha256"],
            "audit_outcome_binding": audit_outcome_binding,
            "experiment_report_contains_classification": True,
            "master_report_contains_classification": True,
            "passed": True,
        }
    return {
        "manifest": identity,
        "master_report": master_report,
        "tracked_master_report": tracked_master,
        "master_report_checks": master_report_checks,
        "human_provenance": human_provenance,
        "required_result_artifacts": required_result_artifacts,
        "master_protocol_sections": {
            name: hashlib.sha256(body.encode()).hexdigest()
            for name, body in protocol_sections.items()
        },
        "scientific_claims_sha256": scientific_claims_sha256,
        "comparison_matrix_sha256": comparison_matrix_sha256,
        "scientific_answer_sha256": scientific_answer_sha256,
        "experiment_reports": audited_reports,
        "result_summaries": audited_summaries,
        "final_audits": audited_audits,
        "structured_outcomes": structured_outcomes,
        "passed": True,
    }


def validate_master_report_git_binding(report_audit: dict, git_audit: dict) -> dict:
    report_path = Path(report_audit["master_report"]["path"])
    text = report_path.read_text(encoding="utf-8")
    body = _markdown_h2_section(text, "Git provenance")
    rows = git_audit.get("repositories")
    if not isinstance(rows, dict) or set(rows) != set(FINAL_GIT_REPOSITORIES):
        raise RuntimeError("Git/report cross-binding repository set is not exact")
    checks = {}
    for name, row in rows.items():
        # The report cannot contain the object ID of the commit that first
        # contains its own bytes.  For MASTER, bind the signed implementation
        # commit; the final report descendant is bound by Git evidence itself.
        report_commit = (
            row["implementation_commit"] if name == "MASTER" else row["commit"]
        )
        pattern = (
            rf"(?m)^- {re.escape(name)}: branch `"
            rf"{re.escape(row['branch'])}`, commit `{re.escape(report_commit)}`"
            rf"(?:, tag `[^`]+`)?\.$"
        )
        if re.search(pattern, body) is None:
            raise RuntimeError(
                f"master report Git line does not bind the audited commit: {name}"
            )
        checks[name] = True
    master = rows["MASTER"]
    for token in (
        MASTER_FINALIZATION_IMPLEMENTATION_TAG,
        master["implementation_tag_object"],
        master["implementation_commit"],
    ):
        if not isinstance(token, str) or token not in body:
            raise RuntimeError("master report signed implementation provenance differs")
    return {"checks": checks, "passed": True}


def required_final_backup_sources(master_root: Path, run_root: Path) -> set[Path]:
    trusted_root = lexical_absolute(master_root).parent
    sources = {
        lexical_absolute(master_root / "MASTER_FINAL_REPORT.md"),
        lexical_absolute(master_root / "worktrees/master/MASTER_FINAL_REPORT.md"),
        lexical_absolute(master_root / "AUTO_STOP_PREFLIGHT.json"),
        lexical_absolute(run_root / "AUTO_STOP_PREFLIGHT.json"),
    }
    for value in (*FINAL_REPORTS.values(), *FINAL_AUDITS.values()):
        sources.add(lexical_absolute(value))
    checkpoint_values = list(FINAL_CHECKPOINTS.values())
    production_contract = set(FINAL_REPORTS) == {
        "2D2E-C1", "2D2F", "2D2G", "2D2H", "2D2I"
    }
    if production_contract:
        checkpoint_values.extend(SOURCE_CHECKPOINTS.values())
    for value in checkpoint_values:
        checkpoint = lexical_absolute(value)
        sources.update(
            {
                checkpoint,
                checkpoint.with_suffix(checkpoint.suffix + ".sha256"),
                checkpoint.with_suffix(checkpoint.suffix + ".verification.json"),
            }
        )
    result_roots = {
        lexical_absolute(value).parent for value in FINAL_REPORTS.values()
    }
    if production_contract:
        historical_base = lexical_absolute(master_root / "worktrees/master")
        result_roots.update(
            historical_base / relative
            for relative in HISTORICAL_RESULT_DIRECTORIES.values()
        )
    for root in result_roots:
        sources.update(
            symlink_free_tree_files(
                root, trusted_root, f"final/historical result directory {root}"
            )
        )
    excluded_run_files = {
        "MASTER_HEARTBEAT.json",
        "MASTER_FINALIZATION_COMPLETE",
        FINALIZATION_EVIDENCE_FILES["backup"],
        FINALIZATION_EVIDENCE_FILES["backup_signature"],
    }
    for candidate in symlink_free_tree_files(
        run_root, trusted_root, "run-scoped finalization evidence"
    ):
        if candidate.name in excluded_run_files:
            continue
        if ".tmp." in candidate.name:
            raise RuntimeError(f"run evidence contains an incomplete temporary: {candidate}")
        sources.add(lexical_absolute(candidate))
    for path in sorted(sources):
        require_symlink_free_regular_file(
            path, trusted_root, f"required final backup source {path}",
            require_nonempty=False,
        )
    return sources


def _backup_inventory_sha256(rows: list[dict]) -> str:
    content = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def verify_backup_manifest_signature(manifest_path: Path, run_root: Path) -> dict:
    signature_path = run_root / FINALIZATION_EVIDENCE_FILES["backup_signature"]
    require_symlink_free_regular_file(
        signature_path, run_root, "detached local-backup signature"
    )
    if (
        signature_path.stat().st_mode & 0o222
        or signature_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            f"detached local-backup signature is absent or mutable: {signature_path}"
        )
    signature_bytes = signature_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    allowed = (
        f"{FINAL_BACKUP_SIGNER_PRINCIPAL} {FINAL_BACKUP_SIGNER_PUBLIC_KEY}\n"
    ).encode()
    allowed_path = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            allowed_path = Path(handle.name)
            handle.write(allowed)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(allowed_path, 0o600)
        result = subprocess.run(
            [
                SSH_KEYGEN_EXECUTABLE, "-Y", "verify", "-f", str(allowed_path),
                "-I", FINAL_BACKUP_SIGNER_PRINCIPAL,
                "-n", FINAL_BACKUP_SIGNER_NAMESPACE,
                "-s", str(signature_path),
            ],
            input=manifest_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=SANITIZED_TOOL_ENVIRONMENT,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(f"cannot execute detached backup signature verifier: {error}") from error
    finally:
        if allowed_path is not None:
            allowed_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            "detached local-backup signature verification failed: "
            + result.stdout.decode(errors="replace").strip()
        )
    if (
        not signature_path.is_file()
        or signature_path.is_symlink()
        or signature_path.stat().st_mode & 0o222
        or signature_path.read_bytes() != signature_bytes
    ):
        raise RuntimeError("detached local-backup signature changed during verification")
    return {
        "signature_path": str(signature_path.resolve()),
        "signature_bytes": len(signature_bytes),
        "signature_sha256": hashlib.sha256(signature_bytes).hexdigest(),
        "signed_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "principal": FINAL_BACKUP_SIGNER_PRINCIPAL,
        "namespace": FINAL_BACKUP_SIGNER_NAMESPACE,
        "public_key_fingerprint": FINAL_BACKUP_SIGNER_FINGERPRINT,
        "passed": True,
    }


def validate_final_backup_evidence(
    master_root: Path, run_root: Path, run_id: str, path_value: str
) -> dict:
    path = run_root / FINALIZATION_EVIDENCE_FILES["backup"]
    payload, identity = _load_finalization_manifest(
        path_value, path, "parallel_2d2_final_local_backup_evidence_v1", run_id
    )
    signature = verify_backup_manifest_signature(path, run_root)
    if signature["signed_manifest_sha256"] != identity["sha256"]:
        raise RuntimeError("detached backup signature does not bind loaded manifest bytes")
    if set(payload) != {
        "schema_version",
        "kind",
        "run_id",
        "pod",
        "created_utc",
        "backup_root",
        "verification_host",
        "all_backup_files_opened_and_hashed",
        "inventory_sha256",
        "authenticated_pod_query",
        "files",
    }:
        raise RuntimeError("local-backup evidence has unexpected keys")
    backup_root_value = payload.get("backup_root")
    backup_root = lexical_absolute(backup_root_value or ".")
    verification_host = payload.get("verification_host")
    rows = payload.get("files")
    query = payload.get("authenticated_pod_query")
    if (
        not isinstance(backup_root_value, str)
        or not Path(backup_root_value).is_absolute()
        or run_id not in backup_root.name
        or not isinstance(verification_host, dict)
        or set(verification_host) != {"hostname", "platform", "verified_utc"}
        or not isinstance(verification_host.get("hostname"), str)
        or not verification_host["hostname"]
        or verification_host.get("platform") != "Darwin"
        or not isinstance(verification_host.get("verified_utc"), str)
        or not verification_host["verified_utc"]
        or payload.get("all_backup_files_opened_and_hashed") is not True
        or not isinstance(rows, list)
        or not rows
    ):
        raise RuntimeError("local-backup verification identity is invalid")
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("source_path"), str)
        or not Path(row.get("source_path", "")).is_absolute()
        or not isinstance(row.get("backup_path"), str)
        or not Path(row.get("backup_path", "")).is_absolute()
        or not isinstance(row.get("bytes"), int)
        or isinstance(row.get("bytes"), bool)
        or not isinstance(row.get("backup_bytes"), int)
        or isinstance(row.get("backup_bytes"), bool)
        or not isinstance(row.get("sha256"), str)
        or not isinstance(row.get("backup_sha256"), str)
        for row in rows
    ):
        raise RuntimeError("local-backup inventory row types are invalid")
    if rows != sorted(rows, key=lambda row: row.get("source_path", "")):
        raise RuntimeError("local-backup inventory is not in canonical source order")
    if payload.get("inventory_sha256") != _backup_inventory_sha256(rows):
        raise RuntimeError("local-backup inventory aggregate SHA differs")
    required_sources = required_final_backup_sources(master_root, run_root)
    observed_sources = set()
    observed_backups = set()
    audited = []
    accessible = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "source_path",
            "backup_path",
            "bytes",
            "sha256",
            "backup_bytes",
            "backup_sha256",
        }:
            raise RuntimeError("local-backup inventory row has unexpected keys")
        source = lexical_absolute(row.get("source_path", ""))
        backup = lexical_absolute(row.get("backup_path", ""))
        try:
            backup.relative_to(backup_root)
        except ValueError as error:
            raise RuntimeError(f"backup path escapes declared local root: {backup}") from error
        if source in observed_sources or backup in observed_backups:
            raise RuntimeError("local-backup inventory contains duplicate paths")
        observed_sources.add(source)
        observed_backups.add(backup)
        require_symlink_free_regular_file(
            source,
            lexical_absolute(master_root).parent,
            f"backup source {source}",
            require_nonempty=False,
        )
        source_bytes = source.stat().st_size
        source_sha = file_sha256(source)
        if (
            row.get("bytes") != source_bytes
            or row.get("sha256") != source_sha
            or row.get("backup_bytes") != source_bytes
            or row.get("backup_sha256") != source_sha
        ):
            raise RuntimeError(f"source/local-backup hash evidence differs: {source}")
        backup_accessible = backup.is_file() and not backup.is_symlink()
        accessible.append(backup_accessible)
        if backup_accessible:
            require_symlink_free_regular_file(
                backup,
                backup_root,
                f"accessible local backup {backup}",
                require_nonempty=False,
            )
        if backup_accessible and (
            backup.stat().st_size != source_bytes or file_sha256(backup) != source_sha
        ):
            raise RuntimeError(f"accessible local backup differs: {backup}")
        audited.append(
            {
                "source_path": str(source),
                "backup_path": str(backup),
                "bytes": source_bytes,
                "sha256": source_sha,
                "backup_accessible_to_coordinator": backup_accessible,
            }
        )
    if observed_sources != required_sources:
        missing = required_sources - observed_sources
        extra = observed_sources - required_sources
        raise RuntimeError(
            "local-backup source set differs; missing="
            + repr(sorted(str(path) for path in missing))
            + " extra="
            + repr(sorted(str(path) for path in extra))
        )
    if any(accessible) and not all(accessible):
        raise RuntimeError("only part of the declared local backup is accessible")
    return {
        "manifest": identity,
        "detached_signature": signature,
        "backup_root": str(backup_root),
        "verification_host": verification_host,
        "authenticated_pod_query": query,
        "inventory_sha256": payload["inventory_sha256"],
        "files": audited,
        "all_backup_files_accessible_to_coordinator": bool(accessible)
        and all(accessible),
        "all_sources_rehashed": True,
        "passed": True,
    }


def running_scientific_processes() -> list[dict]:
    try:
        output = subprocess.check_output(
            [PS_EXECUTABLE, "-axo", "pid=,pgid=,comm=,args="],
            text=True, stderr=subprocess.STDOUT,
            env=SANITIZED_TOOL_ENVIRONMENT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot audit scientific processes: {error}") from error
    rows = []
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if (
            len(fields) != 4
            or not fields[0].isdigit()
            or not fields[1].isdigit()
        ):
            continue
        pid = int(fields[0])
        pgid = int(fields[1])
        comm = fields[2]
        command = fields[3]
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = []
        matched = sorted(
            {
                Path(token).name
                for token in argv
                if Path(token).name in SCIENTIFIC_PROCESS_MARKERS
            }
        )
        if pid != os.getpid() and matched:
            rows.append(
                {
                    "pid": pid,
                    "process_group_id": pgid,
                    "comm_basename": Path(comm).name,
                    "argv": argv,
                    "matched_scientific_basenames": matched,
                }
            )
    return rows


RECORDED_LANE_STATUS_PATTERN = re.compile(
    r"^lane_(gpu[0-3])\.(status|error)\."
    r"(?:attempt[1-9][0-9]*|recovery_attempt_[0-9]{4})\.json$"
)


def _archived_lane_process_records(run_root: Path, run_id: str) -> list[dict]:
    rows = []
    paths = set(run_root.glob("lane_gpu*.status.*.json")) | set(
        run_root.glob("lane_gpu*.error.*.json")
    )
    for path in sorted(paths):
        match = RECORDED_LANE_STATUS_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unrecognized archived lane status filename: {path}")
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"archived lane status is not a regular file: {path}")
        payload = read_json(path)
        lane = match.group(1).upper()
        pid = payload.get("shell_pid")
        pgid = payload.get("process_group_id")
        exit_code = payload.get("exit_code")
        if (
            payload.get("schema_version") != 1
            or payload.get("run_id") != run_id
            or payload.get("lane") != lane
            or payload.get("status") != "HARD_FAILURE"
            or not isinstance(exit_code, int)
            or exit_code == 0
            or not isinstance(pid, int)
            or pid <= 0
            or not isinstance(pgid, int)
            or pgid <= 0
            or pid != pgid
        ):
            raise RuntimeError(f"archived lane status process identity is invalid: {path}")
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "lane": lane,
                "record_type": match.group(2),
                "shell_pid": pid,
                "shell_process_group_id": pgid,
            }
        )
    return rows


def validate_recorded_lane_processes(
    run_root: Path, terminal: dict, supervisor_state: dict
) -> dict:
    state_lanes = supervisor_state.get("lanes")
    canonical_lanes = terminal.get("lanes")
    if (
        not isinstance(state_lanes, dict)
        or set(state_lanes) != set(LANES)
        or not isinstance(canonical_lanes, dict)
        or set(canonical_lanes) != set(LANES)
    ):
        raise RuntimeError("recorded lane process set is not exact")
    rows = {}
    recorded_identities = set()
    for lane, script_name in LANES.items():
        state = state_lanes[lane]
        canonical = canonical_lanes[lane]
        pid = state.get("pid")
        pgid = state.get("process_group_id")
        script = state.get("script")
        if (
            not isinstance(pid, int) or pid <= 0
            or not isinstance(pgid, int) or pgid <= 0
            or pid != pgid
            or not isinstance(script, str)
            or Path(script).name != script_name
            or process_is_alive(pid)
            or process_group_is_alive(pgid)
        ):
            raise RuntimeError(f"{lane} original lane PID/PGID is not canonically exited")
        row = {
            "original_shell_pid": pid,
            "original_shell_process_group_id": pgid,
            "script_basename": script_name,
            "original_shell_and_group_absent": True,
        }
        recorded_identities.add((lane, pid, pgid))
        if canonical.get("status") == "RECOVERABLE_FAILURE_RESUMED":
            effective_pid = canonical.get("effective_shell_pid")
            effective_pgid = canonical.get("effective_shell_process_group_id")
            if (
                not isinstance(effective_pid, int) or effective_pid <= 0
                or not isinstance(effective_pgid, int) or effective_pgid <= 0
                or effective_pid != effective_pgid
                or (effective_pid, effective_pgid) == (pid, pgid)
                or process_is_alive(effective_pid)
                or process_group_is_alive(effective_pgid)
            ):
                raise RuntimeError(
                    f"{lane} recovery lane PID/PGID is not canonically exited"
                )
            row.update(
                effective_shell_pid=effective_pid,
                effective_shell_process_group_id=effective_pgid,
                effective_shell_and_group_absent=True,
            )
            recorded_identities.add((lane, effective_pid, effective_pgid))
        rows[lane] = row
    run_id = terminal.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("canonical terminal run identity is absent")
    archived = _archived_lane_process_records(run_root, run_id)
    for record in archived:
        pid = record["shell_pid"]
        pgid = record["shell_process_group_id"]
        if process_is_alive(pid) or process_group_is_alive(pgid):
            raise RuntimeError(
                f"{record['lane']} archived recovery PID/PGID is still alive: "
                f"{pid}/{pgid}"
            )
        recorded_identities.add((record["lane"], pid, pgid))
    return {
        "lanes": rows,
        "archived_attempt_processes": archived,
        "unique_recorded_processes": [
            {"lane": lane, "shell_pid": pid, "shell_process_group_id": pgid}
            for lane, pid, pgid in sorted(recorded_identities)
        ],
        "all_recorded_lane_process_groups_absent": True,
        "passed": True,
    }


def validate_authenticated_stop_identity(
    master_root: Path,
    run_root: Path,
    run_id: str,
    backup_audit: dict,
    not_before: datetime,
) -> dict:
    top_path = master_root / "AUTO_STOP_PREFLIGHT.json"
    run_path = run_root / "AUTO_STOP_PREFLIGHT.json"
    try:
        top_bytes = top_path.read_bytes()
        run_bytes = run_path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"cannot read authenticated stop preflight: {error}") from error
    if top_bytes != run_bytes:
        raise RuntimeError("top-level and run-scoped stop preflights differ")
    payload = read_json(run_path)
    if set(payload) != STOP_PREFLIGHT_KEYS:
        raise RuntimeError("authenticated stop preflight key set is not exact")
    response = payload.get("authenticated_pod_identity_response")
    response_keys = {
        "desiredStatus", "gpuCount", "id", "name", "networkVolumeId",
        "runtimeStatus",
    }
    query = backup_audit.get("authenticated_pod_query")
    query_keys = {
        "run_id", "command", "authenticated", "queried_utc",
        "preflight_path", "preflight_sha256", "response",
    }
    if not isinstance(query, dict) or set(query) != query_keys:
        raise RuntimeError("signed authenticated pod query key set is not exact")
    queried = parse_canonical_utc(query.get("queried_utc"), "live pod query queried_utc")
    checked = parse_canonical_utc(payload.get("checked_utc"), "stop preflight checked_utc")
    require_timestamp_not_future(queried, "live pod query queried_utc")
    if (
        datetime.now(timezone.utc) - queried > FINAL_EVIDENCE_FRESHNESS
        or queried < not_before
        or checked > queried
    ):
        raise RuntimeError("authenticated live pod query is stale or out of order")
    preflight_sha = hashlib.sha256(run_bytes).hexdigest()
    expected = {
        "schema": payload.get("schema") == "parallel_2d2_runpod_stop_capability_v1",
        "passed": payload.get("passed") is True,
        "pod_id": payload.get("pod_id") == POD_ID,
        "pod_name": payload.get("pod_name") == POD_NAME,
        "gpu_count": payload.get("gpu_count") == 4,
        "volume_id": payload.get("volume_id") == VOLUME_ID,
        "authenticated": payload.get("authenticated") is True,
        "authenticated_list_probe": payload.get("authenticated_list_probe") is True,
        "stop_credential_available": payload.get("stop_credential_available") is True,
        "exact_stop_target": payload.get("exact_stop_target") == POD_ID,
        "exact_stop_command": payload.get("exact_stop_command")
        == f"runpodctl pod stop {POD_ID} -o json",
        "runtime_running": payload.get("runtime_status") == "running"
        and payload.get("desired_status") == "RUNNING",
        "volume_preserved": payload.get("network_volume_preservation_required") is True,
        "deletion_forbidden": payload.get("pod_delete_forbidden") is True
        and payload.get("pod_delete_authorized") is False
        and payload.get("persistent_volume_delete_authorized") is False,
        "no_secret_recorded": payload.get("secret_recorded") is False,
        "response_keys_exact": isinstance(response, dict) and set(response) == response_keys,
        "response_identity_exact": response
        == {
            "desiredStatus": "RUNNING",
            "gpuCount": 4,
            "id": POD_ID,
            "name": POD_NAME,
            "networkVolumeId": VOLUME_ID,
            "runtimeStatus": "running",
        },
        "signed_query_run_exact": query.get("run_id") == run_id,
        "signed_query_command_exact": query.get("command")
        == f"runpodctl pod get {POD_ID} -o json",
        "signed_query_authenticated": query.get("authenticated") is True,
        "signed_query_preflight_path_exact": query.get("preflight_path")
        == str(run_path.resolve()),
        "signed_query_preflight_sha_exact": query.get("preflight_sha256")
        == preflight_sha,
        "signed_query_response_exact": query.get("response") == response,
        "backup_signature_passed": backup_audit.get("detached_signature", {}).get("passed")
        is True,
    }
    if not all(expected.values()):
        raise RuntimeError(f"authenticated exact-pod stop identity failed: {expected}")
    return {
        "path": str(run_path),
        "sha256": preflight_sha,
        "run_id": run_id,
        "pod": EXPECTED_POD,
        "live_query": query,
        "checks": expected,
        "passed": True,
    }


def validate_finalization_boundary(
    master_root: Path,
    run_root: Path,
    run_id: str,
    git_evidence: str,
    report_evidence: str,
    backup_evidence: str,
    terminal: dict,
    supervisor_state: dict,
) -> dict:
    terminal_created = parse_canonical_utc(
        terminal.get("created_utc"), "canonical terminal created_utc"
    )
    require_timestamp_not_future(terminal_created, "canonical terminal created_utc")
    final_science = validate_final_science_artifacts()
    cpu_processes = running_scientific_processes()
    if cpu_processes:
        raise RuntimeError(
            f"scientific processes remain at finalization boundary: {cpu_processes}"
        )
    lane_processes = validate_recorded_lane_processes(
        run_root, terminal, supervisor_state
    )
    git = validate_final_git_evidence(
        master_root, run_root, run_id, git_evidence
    )
    report = validate_final_report_evidence(
        master_root, run_root, run_id, report_evidence
    )
    report_git_binding = validate_master_report_git_binding(report, git)
    backup = validate_final_backup_evidence(
        master_root, run_root, run_id, backup_evidence
    )
    git_created = parse_canonical_utc(
        git["manifest"]["created_utc"], "Git evidence created_utc"
    )
    report_created = parse_canonical_utc(
        report["manifest"]["created_utc"], "report evidence created_utc"
    )
    backup_created = parse_canonical_utc(
        backup["manifest"]["created_utc"], "backup evidence created_utc"
    )
    backup_verified = parse_canonical_utc(
        backup["verification_host"].get("verified_utc"),
        "backup verification verified_utc",
    )
    pod_queried = parse_canonical_utc(
        backup.get("authenticated_pod_query", {}).get("queried_utc"),
        "authenticated pod query queried_utc",
    )
    if (
        min(git_created, report_created, backup_created, backup_verified, pod_queried)
        < terminal_created
        or backup_created < max(git_created, report_created, backup_verified, pod_queried)
        or backup_created - min(backup_verified, pod_queried) > FINAL_EVIDENCE_FRESHNESS
    ):
        raise RuntimeError("finalization evidence timestamps are stale or out of order")
    stop_identity = validate_authenticated_stop_identity(
        master_root, run_root, run_id, backup, terminal_created
    )
    for name, audit in (("git", git), ("report", report), ("backup", backup)):
        manifest = audit["manifest"]
        manifest_path = Path(manifest["path"])
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or manifest_path.stat().st_mode & 0o222
            or manifest_path.stat().st_size != manifest["bytes"]
            or file_sha256(manifest_path) != manifest["sha256"]
        ):
            raise RuntimeError(
                f"{name} finalization evidence changed during boundary audit"
            )
    return {
        "git": git,
        "report": report,
        "report_git_binding": report_git_binding,
        "backup": backup,
        "final_science": final_science,
        "scientific_processes": cpu_processes,
        "recorded_lane_processes": lane_processes,
        "authenticated_stop_identity": stop_identity,
        "timestamp_order": {
            "terminal_created_utc": terminal["created_utc"],
            "git_created_utc": git["manifest"]["created_utc"],
            "report_created_utc": report["manifest"]["created_utc"],
            "backup_verified_utc": backup["verification_host"]["verified_utc"],
            "pod_queried_utc": backup["authenticated_pod_query"]["queried_utc"],
            "backup_created_utc": backup["manifest"]["created_utc"],
            "passed": True,
        },
        "evidence_manifests_unchanged": True,
        "pod_stop_invoked": False,
        "passed": True,
    }


def revalidate_finalization_snapshot(
    master_root: Path,
    run_root: Path,
    run_id: str,
    git_evidence: str,
    report_evidence: str,
    backup_evidence: str,
    terminal: dict,
    supervisor_state: dict,
    captured: dict,
) -> dict:
    current_terminal = read_json(run_root / "MASTER_TERMINAL_STATUS.json")
    current_terminal_sentinel = read_json(run_root / "MASTER_ALL_LANES_TERMINAL")
    current_top_terminal = read_json(master_root / "MASTER_TERMINAL_STATUS.json")
    current_top_sentinel = read_json(master_root / "MASTER_ALL_LANES_TERMINAL")
    current_state = read_json(run_root / "MASTER_SUPERVISOR.json")
    current_top_state = read_json(master_root / "MASTER_STATUS.json")
    if (
        current_terminal != terminal
        or current_terminal_sentinel != terminal
        or current_top_terminal != terminal
        or current_top_sentinel != terminal
        or current_state != supervisor_state
        or current_top_state != supervisor_state
    ):
        raise RuntimeError("canonical terminal/supervisor state changed before publication")
    fresh = validate_finalization_boundary(
        master_root, run_root, run_id, git_evidence, report_evidence,
        backup_evidence, terminal, supervisor_state,
    )
    fields = (
        "git", "report", "report_git_binding", "backup", "final_science", "scientific_processes",
        "recorded_lane_processes", "authenticated_stop_identity", "timestamp_order",
    )
    changed = [name for name in fields if fresh.get(name) != captured.get(name)]
    if changed:
        raise RuntimeError(
            "finalization state changed before sentinel publication: "
            + ", ".join(changed)
        )
    return {
        "rechecked_immediately_before_publication": True,
        "captured_fields": list(fields),
        "snapshot_sha256": hashlib.sha256(
            json.dumps(
                {name: fresh[name] for name in fields},
                sort_keys=True, separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "passed": True,
    }


def validate_final_lightweight_process_gate(
    run_root: Path, terminal: dict, supervisor_state: dict
) -> dict:
    gpu_processes = nvidia_compute_processes()
    cpu_processes = running_scientific_processes()
    recorded = validate_recorded_lane_processes(
        run_root, terminal, supervisor_state
    )
    if gpu_processes or cpu_processes:
        raise RuntimeError(
            "scientific process appeared immediately before finalization publication: "
            f"gpu={gpu_processes!r} cpu={cpu_processes!r}"
        )
    return {
        "nvidia_compute_processes": gpu_processes,
        "scientific_cpu_processes": cpu_processes,
        "recorded_lane_processes": recorded,
        "all_gpus_compute_idle": True,
        "passed": True,
    }


def capture_final_prepublication_fingerprint(
    master_root: Path, run_root: Path, boundary: dict
) -> dict:
    """Rehash the signed source inventory after the last process-side effect."""
    expected_rows = boundary.get("backup", {}).get("files")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise RuntimeError("final backup audit lacks the exact source inventory")
    expected = {
        lexical_absolute(row["source_path"]): {
            "bytes": row["bytes"], "sha256": row["sha256"]
        }
        for row in expected_rows
    }
    if len(expected) != len(expected_rows):
        raise RuntimeError("final backup audit source inventory is duplicated")
    required = required_final_backup_sources(master_root, run_root)
    if set(expected) != required:
        raise RuntimeError("final backup source set changed after process gate")
    current_rows = []
    for path in sorted(required):
        identity = stable_file_identity(path)
        recorded = expected[path]
        if (
            identity["bytes"] != recorded["bytes"]
            or identity["sha256"] != recorded["sha256"]
        ):
            raise RuntimeError(
                f"final evidence source changed after full validation: {path}"
            )
        current_rows.append(identity)
    extra_paths = (
        run_root / FINALIZATION_EVIDENCE_FILES["backup"],
        run_root / FINALIZATION_EVIDENCE_FILES["backup_signature"],
    )
    extra = []
    for path in extra_paths:
        require_symlink_free_regular_file(path, run_root, f"final evidence {path.name}")
        extra.append(stable_file_identity(path))
    backup_manifest = boundary["backup"]["manifest"]
    backup_signature = boundary["backup"]["detached_signature"]
    if (
        extra[0]["bytes"] != backup_manifest["bytes"]
        or extra[0]["sha256"] != backup_manifest["sha256"]
        or extra[1]["bytes"] != backup_signature["signature_bytes"]
        or extra[1]["sha256"] != backup_signature["signature_sha256"]
    ):
        raise RuntimeError("signed backup evidence changed after full validation")
    rows = current_rows + extra
    snapshot = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    boundary_sha = hashlib.sha256(
        json.dumps(boundary, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "captured_after_final_process_gate": True,
        "all_required_sources_match_signed_backup_inventory": True,
        "stable_file_count": len(rows),
        "stable_file_inventory_sha256": snapshot,
        "validated_boundary_serialized_sha256": boundary_sha,
        "backup_manifest_sha256": extra[0]["sha256"],
        "backup_signature_sha256": extra[1]["sha256"],
        "passed": True,
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
    top_finalization = master_root / "MASTER_FINALIZATION_COMPLETE"
    if top_finalization.exists():
        top_finalization_record = read_json(top_finalization)
        if top_finalization_record.get("run_id") == args.run_id:
            raise RuntimeError(
                "cannot reconcile after current-run master finalization publication has begun"
            )
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
        or any(lane not in recovered_lanes for lane in retained_lanes)
        or len(retained_lanes) != len(set(retained_lanes))
        or not legacy_lanes.issubset(set(retained_lanes))
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


def _mark_finalization_complete_locked(args) -> int:
    master_root = Path(args.master_root).resolve()
    _, run_root = validate_master_preflight(master_root, args.run_id)
    run_finalization = run_root / "MASTER_FINALIZATION_COMPLETE"
    top_finalization = master_root / "MASTER_FINALIZATION_COMPLETE"
    if os.path.lexists(run_finalization) or os.path.lexists(top_finalization):
        raise RuntimeError(
            "finalization sentinel already exists; refusing any partial or replacement publication"
        )
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
    boundary = validate_finalization_boundary(
        master_root,
        run_root,
        args.run_id,
        args.git_evidence,
        args.report_evidence,
        args.backup_evidence,
        terminal,
        supervisor_state,
    )
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "MASTER_FINALIZATION_COMPLETE",
        "created_utc": now_utc(),
        "evidence": {
            "git": boundary["git"],
            "report": boundary["report"],
            "backup": boundary["backup"],
        },
        "report_git_binding": boundary["report_git_binding"],
        "final_science_evidence": boundary["final_science"],
        "scientific_processes": boundary["scientific_processes"],
        "recorded_lane_processes": boundary["recorded_lane_processes"],
        "authenticated_stop_identity": boundary["authenticated_stop_identity"],
        "timestamp_order": boundary["timestamp_order"],
        "terminal_record": str(run_root / "MASTER_TERMINAL_STATUS.json"),
        "terminal_record_sha256": file_sha256(run_root / "MASTER_TERMINAL_STATUS.json"),
        "recovery_reconciled": bool(recovered_by_status),
        "recovered_lanes": recovered_by_status,
        "pod_stop_automated": False,
        "pod_stop_separate_required": True,
        "finalization_lock": str(master_root / "locks/finalize.lock"),
    }
    payload["final_revalidation"] = revalidate_finalization_snapshot(
        master_root,
        run_root,
        args.run_id,
        args.git_evidence,
        args.report_evidence,
        args.backup_evidence,
        terminal,
        supervisor_state,
        boundary,
    )
    payload["final_process_gate"] = validate_final_lightweight_process_gate(
        run_root, terminal, supervisor_state
    )
    payload["final_prepublication_fingerprint"] = (
        capture_final_prepublication_fingerprint(master_root, run_root, boundary)
    )
    durable_json_exclusive(top_finalization, payload)
    # This run-scoped path is the heartbeat trigger and is deliberately last.
    durable_bytes_exclusive(run_finalization, top_finalization.read_bytes())
    print(str(run_finalization), flush=True)
    return 0


def mark_finalization_complete(args) -> int:
    master_root = Path(args.master_root).resolve()
    lock_path = master_root / "locks/finalize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_handle = lock_path.open("a+")
    except OSError as error:
        raise RuntimeError(f"cannot open exact finalization lock {lock_path}: {error}") from error
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another finalizer holds the exact finalization lock {lock_path}"
            ) from error
        try:
            return _mark_finalization_complete_locked(args)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    finally:
        lock_handle.close()


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
    current.add_argument(
        "--git-evidence",
        required=True,
        help="exact run-scoped FINAL_GIT_EVIDENCE.json path",
    )
    current.add_argument(
        "--report-evidence",
        required=True,
        help="exact run-scoped FINAL_REPORT_EVIDENCE.json path",
    )
    current.add_argument(
        "--backup-evidence",
        required=True,
        help="exact run-scoped FINAL_LOCAL_BACKUP_EVIDENCE.json path",
    )
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
