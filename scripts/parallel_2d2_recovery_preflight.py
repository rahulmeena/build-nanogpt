#!/usr/bin/env python3
"""Authorize narrowly scoped, checkpoint-exact recovery of failed lanes.

The original passing shared preflight remains immutable.  This audit proves
that only the registered lane implementations changed, that each lane's
declared restart checkpoint and sidecars remain exact, and that the assigned
GPUs are idle.  Tracked worktree state must be clean; only untracked artifacts
inside the lane's exact experiment result root are tolerated.
It never launches a process or controls a pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


LANES = {
    "GPU0": {
        "branch": "codex/parallel-2d2-master",
        "worktree": "/workspace/parallel_2d2_master/worktrees/master",
        "base_checkpoint": "/workspace/exp2d2e_run/checkpoints/scientific_update_0191.pt",
        "base_sha256": "dea5e76b55d1ad7281fe3cf3893713392343b875182ba186a0049904e61de790",
        "allowed_changed_files": {
            "scripts/experiment_2d2e_c1.py",
            "tests/test_experiment_2d2e_c1.py",
            "scripts/parallel_2d2_lane_common.sh",
            "scripts/parallel_2d2_recovery_preflight.py",
            "scripts/parallel_2d2_lane1_stage_b_recovery.sh",
            "scripts/parallel_2d2_lane2_finalize_recovery.sh",
            "scripts/parallel_2d2_supervisor.py",
            "scripts/test_parallel_2d2_orchestration.py",
            "scripts/test_parallel_2d2_recovery_preflight.py",
        },
        "allowed_untracked_result_roots": {
            "results/experiment_2d2e_c1_large_true_self_confirmation",
        },
        "dependent_worktree_patches": [
            {
                "branch": "experiment-2d2f-no-b2-recurrence-b3-w64",
                "worktree": "/workspace/parallel_2d2_master/worktrees/2d2f",
                "allowed_changed_files": {
                    "scripts/experiment_2d2f.py",
                    "tests/test_experiment_2d2f_core.py",
                },
                "allowed_untracked_result_roots": {
                    "results/experiment_2d2f_no_b2_recurrence_b3_w64",
                },
                "tests": [
                    "tests/test_experiment_2d2f_core.py",
                    "tests/test_experiment_2d2f_driver.py",
                ],
            }
        ],
        "recovery_reason": (
            "2D2E-C1 completed its evaluation but artifact publication failed on "
            "non-JSON CUDA UUID metadata; its exact recovery rerun then exposed a "
            "preflight-only 2D2F diagnostic assumption that inherited B1 supplies "
            "local-attention weights, so deterministically rerun frozen C1 and the "
            "audited diagnostic-corrected 2D2F sequence from exact checkpoints"
        ),
        "tests": [
            "tests/test_experiment_2d2e_c1.py",
            "scripts/test_parallel_2d2_recovery_preflight.py",
        ],
    },
    "GPU1": {
        "branch": "experiment-2d2g-b2-full-b3-w64",
        "worktree": "/workspace/parallel_2d2_master/worktrees/2d2g",
        "base_checkpoint": (
            "/tmp/parallel_2d2_ephemeral/2d2g/checkpoints/"
            "stage_a_scientific_update_0191.pt"
        ),
        # This internal scientific staging checkpoint was created during the
        # live run.  Its exact identity is sealed by its colocated SHA sidecar
        # and strict-reopen sidecar rather than a value known before Stage A.
        "base_sha256": None,
        "allowed_changed_files": {
            "scripts/experiment_2d2g.py",
            "tests/test_experiment_2d2g_driver.py",
        },
        "allowed_untracked_result_roots": {
            "results/experiment_2d2g_b2_full_b3_w64",
        },
        "recovery_reason": (
            "2D2G-B recovery attempt 1 stopped before science because its "
            "experiment preflight fingerprint named the pre-device-fix implementation; "
            "rerun the 2D2G preflight from frozen 2D2B, retain exact Stage-A-191, "
            "then rerun smoke-B plus Stage B"
        ),
        "tests": [
            "tests/test_experiment_2d2g_core.py",
            "tests/test_experiment_2d2g_driver.py",
        ],
    },
    "GPU2": {
        "branch": "experiment-2d2h-no-b1-recurrence-b2-w32",
        "worktree": "/workspace/parallel_2d2_master/worktrees/2d2h",
        "base_checkpoint": "/workspace/exp2d2h_run/checkpoints/scientific_update_0191.pt",
        "base_sha256": None,
        "allowed_changed_files": {
            "scripts/experiment_2d2h.py",
            "tests/test_experiment_2d2h_driver.py",
        },
        "allowed_untracked_result_roots": {
            "results/experiment_2d2h_no_b1_recurrence_b2_w32",
        },
        "recovery_reason": (
            "2D2H completed training and final metrics, then its evaluation-only "
            "parallel/incremental equivalence audit used default TF32 while enforcing "
            "an FP32 1e-4 tolerance; seal the exact prior passing preflight, approved "
            "audit-only implementation correction, failed-audit evidence, clean pushed "
            "HEAD, and persisted H update-191 checkpoint before finalize-only recovery"
        ),
        "tests": [
            "tests/test_experiment_2d2h_core.py",
            "tests/test_experiment_2d2h_driver.py",
        ],
    },
}

BASH_PERCENT_Q_SAFE = re.compile(r"[A-Za-z0-9_@%+=:,./-]+")
EXPECTED_POD = {
    "id": "7i2zyd53ytspwz",
    "name": "empirical_tan_panda",
    "gpu_count": 4,
    "volume_id": "yhzyb27fb5",
}

GPU0_FAILED_F_PREFLIGHT_FILES = {
    "2d2d_reference_manifest.json",
    "architecture_manifest.json",
    "parameter_manifest.json",
    "semantic_diff_audit.json",
    "source_manifest.json",
}
GPU0_COMPLETED_C1_FILES = {
    "C1_FINAL_REPORT.md",
    "FINAL_AUDIT.json",
    "HEARTBEAT.json",
    "bootstrap_results.json",
    "c1_validation_subset_manifest.json",
    "paired_results.json",
    "result_summary.json",
    "subset_manifest.json",
}


def render_bash_percent_q(argv) -> str:
    """Render the fixed argv subset exactly as lane_common's Bash ``%q``."""

    rendered = []
    for value in argv:
        word = str(value)
        if BASH_PERCENT_Q_SAFE.fullmatch(word):
            rendered.append(word)
        elif word and all(
            character == " " or BASH_PERCENT_Q_SAFE.fullmatch(character)
            for character in word
        ):
            rendered.append(word.replace(" ", "\\ "))
        else:
            raise RuntimeError(f"unsupported recovery command word for exact %q: {word!r}")
    return " ".join(rendered)


def expected_recovery_argv(master_root: Path, run_root: Path, lane: str) -> list[list[str]]:
    master_worktree = master_root / "worktrees" / "master"
    stop_audit = run_root / "AUTO_STOP_PREFLIGHT.json"
    data_root = "/workspace/build-nanogpt/edu_fineweb10B"
    if lane == "GPU0":
        c1_output = master_worktree / "results/experiment_2d2e_c1_large_true_self_confirmation"
        f_worktree = master_root / "worktrees/2d2f"
        f_output = f_worktree / "results/experiment_2d2f_no_b2_recurrence_b3_w64"
        f_run_root = "/workspace/exp2d2f_run"
        f_ephemeral = "/tmp/parallel_2d2_ephemeral/2d2f"
        f_common = [
            "--source-checkpoint", "/workspace/exp2d2d_run/checkpoints/scientific_update_0191.pt",
            "--data-root", data_root,
            "--output-dir", str(f_output),
            "--run-root", f_run_root,
            "--ephemeral-checkpoint-dir", f_ephemeral,
            "--pod-id", "7i2zyd53ytspwz",
            "--pod-name", "empirical_tan_panda",
            "--gpu-type", "NVIDIA A100-SXM4-80GB",
            "--persistent-volume-identity", "yhzyb27fb5",
            "--stop-mechanism", "runpodctl_exact_pod_stop",
            "--stop-authenticated",
            "--stop-audit-path", str(stop_audit),
        ]
        driver = ["python", "scripts/experiment_2d2f.py"]
        return [
            [
                "python", str(master_worktree / "scripts/experiment_2d2e_c1.py"),
                "--checkpoint", "/workspace/exp2d2e_run/checkpoints/scientific_update_0191.pt",
                "--validation-shard", f"{data_root}/edufineweb_val_000000.npy",
                "--prior-incremental", (
                    "/workspace/build-nanogpt-exp2d2e/results/"
                    "experiment_2d2e_b3_w64_b10_recurrent_960/incremental_validation.json"
                ),
                "--output-dir", str(c1_output),
                "--pod-id", "7i2zyd53ytspwz",
                "--pod-name", "empirical_tan_panda",
                "--stop-audit", str(stop_audit),
            ],
            [*driver, "preflight", *f_common],
            [*driver, "smoke", *f_common],
            [*driver, "train", *f_common, "--end-update", "96"],
            [
                *driver, "train", *f_common, "--end-update", "191",
                "--resume", f"{f_ephemeral}/scientific_update_0096.pt",
            ],
            [
                *driver, "finalize", *f_common, "--final-checkpoint",
                f"{f_run_root}/checkpoints/scientific_update_0191.pt",
            ],
        ]
    if lane == "GPU1":
        worktree = master_root / "worktrees/2d2g"
        output = worktree / "results/experiment_2d2g_b2_full_b3_w64"
        ephemeral = "/tmp/parallel_2d2_ephemeral/2d2g/checkpoints"
        smoke = "/tmp/parallel_2d2_ephemeral/2d2g/smoke"
        persistent = "/workspace/exp2d2g_run/checkpoints"
        source = "/workspace/exp2d2b_run/checkpoints/scientific_update_0191.pt"
        a191 = f"{ephemeral}/stage_a_scientific_update_0191.pt"
        b96 = f"{ephemeral}/stage_b_scientific_update_0096.pt"
        b191 = f"{ephemeral}/stage_b_scientific_update_0191.pt"
        common = [
            "--output-dir", str(output),
            "--pod-id", "7i2zyd53ytspwz",
            "--pod-name", "empirical_tan_panda",
        ]
        driver = ["python", "scripts/experiment_2d2g.py"]
        return [
            [
                *driver, "preflight", *common, "--source-checkpoint", source,
                "--data-root", data_root,
            ],
            [
                *driver, "smoke-b", *common, "--stage-a-checkpoint", a191,
                "--checkpoint-dir", smoke, "--data-root", data_root,
            ],
            [
                *driver, "train-b", *common, "--stage-a-checkpoint", a191,
                "--checkpoint-dir", ephemeral, "--end-update", "96",
                "--data-root", data_root,
            ],
            [
                *driver, "train-b", *common, "--resume", b96,
                "--checkpoint-dir", ephemeral, "--end-update", "191",
                "--data-root", data_root,
            ],
            [
                *driver, "persist-final", "--output-dir", str(output),
                "--local-checkpoint", b191, "--persistent-dir", persistent,
                "--lock-path", str(master_root / "locks/checkpoint_persist.lock"),
            ],
            [
                *driver, "finalize", *common, "--stage-b-checkpoint",
                f"{persistent}/stage_b_scientific_update_0191.pt",
                "--data-root", data_root,
            ],
        ]
    if lane == "GPU2":
        worktree = master_root / "worktrees/2d2h"
        output = worktree / "results/experiment_2d2h_no_b1_recurrence_b2_w32"
        run_root_2d2h = "/workspace/exp2d2h_run"
        ephemeral = "/tmp/parallel_2d2_ephemeral/2d2h"
        common = [
            "--source-checkpoint", "/workspace/exp2d2b_run/checkpoints/scientific_update_0191.pt",
            "--data-root", data_root,
            "--output-dir", str(output),
            "--run-root", run_root_2d2h,
            "--ephemeral-checkpoint-root", ephemeral,
            "--checkpoint-persist-lock", str(master_root / "locks/checkpoint_persist.lock"),
            "--pod-id", "7i2zyd53ytspwz",
            "--pod-name", "empirical_tan_panda",
            "--gpu-type", "NVIDIA A100-SXM4-80GB",
            "--persistent-volume-identity", "yhzyb27fb5",
            "--stop-mechanism", "runpodctl_exact_pod_stop",
            "--stop-authenticated",
            "--stop-audit-path", str(stop_audit),
        ]
        driver = ["python", "scripts/experiment_2d2h.py"]
        return [
            [
                *driver, "authorize-audit-correction", *common, "--final-checkpoint",
                f"{run_root_2d2h}/checkpoints/scientific_update_0191.pt",
            ],
            [
                *driver, "finalize", *common, "--final-checkpoint",
                f"{run_root_2d2h}/checkpoints/scientific_update_0191.pt",
            ],
        ]
    raise RuntimeError(f"no explicit recovery command specification for {lane}")


def recovery_command_plan(
    master_root: Path,
    run_root: Path,
    lanes: list[str],
    retained_active_lanes=(),
    recovery_attempt: int = 1,
    recovery_evidence_schemas: dict[str, str] | None = None,
) -> dict:
    retained = set(retained_active_lanes)
    schemas = recovery_evidence_schemas or {
        lane: (
            "legacy_v1_without_recovery_reason"
            if lane in retained
            else "v2_with_recovery_reason"
        )
        for lane in lanes
    }
    if set(schemas) != set(lanes) or any(
        schema not in {
            "v2_with_recovery_reason",
            "legacy_v1_without_recovery_reason",
        }
        for schema in schemas.values()
    ):
        raise RuntimeError("recovery evidence schema mapping is not lane-exact")
    return {
        "schema_version": 1,
        "run_id": run_root.name,
        "recovery_attempt": int(recovery_attempt),
        "recovered_lanes": {
            lane: {
                "recovery_reason": LANES[lane]["recovery_reason"],
                "recovery_evidence_schema": schemas[lane],
                "expected_resumed_command_records": [
                    render_bash_percent_q(argv)
                    for argv in expected_recovery_argv(master_root, run_root, lane)
                ],
            }
            for lane in lanes
        },
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def durable_json(path: Path, payload: dict) -> None:
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


def preserve_exact_json(path: Path, payload: dict) -> None:
    expected = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    preserve_exact_bytes(path, expected)


def preserve_exact_bytes(path: Path, expected: bytes) -> None:
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"refusing to replace changed recovery evidence: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def versioned_plan_path(run_root: Path, recovery_attempt: int) -> Path:
    if recovery_attempt == 1:
        return run_root / "RECOVERY_COMMAND_PLAN.json"
    return run_root / f"RECOVERY_COMMAND_PLAN_ATTEMPT_{recovery_attempt:04d}.json"


def versioned_preflight_path(run_root: Path, recovery_attempt: int) -> Path:
    return run_root / f"RECOVERY_PREFLIGHT_ATTEMPT_{recovery_attempt:04d}.json"


def prepare_prior_attempt_evidence(
    run_root: Path,
    lanes: list[str],
    recovery_attempt: int,
    original_terminal_gate: dict | None = None,
) -> dict | None:
    """Seal and validate the immediately preceding failed recovery attempt."""

    if recovery_attempt == 1:
        return None
    previous_attempt = recovery_attempt - 1
    canonical_preflight = run_root / "RECOVERY_PREFLIGHT.json"
    previous_preflight_bytes = canonical_preflight.read_bytes()
    previous_preflight = json.loads(previous_preflight_bytes)
    if not isinstance(previous_preflight, dict):
        raise RuntimeError("previous recovery preflight is not a JSON object")
    observed_previous_attempt = previous_preflight.get("recovery_attempt", 1)
    if (
        previous_preflight.get("passed") is not True
        or previous_preflight.get("run_id") != run_root.name
        or observed_previous_attempt != previous_attempt
    ):
        raise RuntimeError("previous recovery preflight is not the exact prior attempt")

    plan_metadata = previous_preflight.get("recovery_command_plan")
    if not isinstance(plan_metadata, dict) or not isinstance(
        plan_metadata.get("path"), str
    ):
        raise RuntimeError("previous recovery plan metadata is missing")
    prior_plan_path = Path(plan_metadata["path"]).resolve()
    expected_prior_plan_path = versioned_plan_path(
        run_root, previous_attempt
    ).resolve()
    if prior_plan_path != expected_prior_plan_path:
        raise RuntimeError("previous recovery plan path is not attempt-exact")
    prior_plan_bytes = prior_plan_path.read_bytes()
    prior_plan_sha = hashlib.sha256(prior_plan_bytes).hexdigest()
    if plan_metadata.get("sha256") != prior_plan_sha:
        raise RuntimeError("previous recovery plan SHA does not match sealed bytes")
    prior_plan = json.loads(prior_plan_bytes)
    if not isinstance(prior_plan, dict):
        raise RuntimeError("previous recovery plan is not a JSON object")
    prior_plan_attempt = prior_plan.get("recovery_attempt", 1)
    prior_rows = prior_plan.get("recovered_lanes")
    if (
        prior_plan.get("schema_version") != 1
        or prior_plan.get("run_id") != run_root.name
        or prior_plan_attempt != previous_attempt
        or not isinstance(prior_rows, dict)
        or any(lane not in prior_rows for lane in lanes)
        or any(
            not isinstance(prior_rows[lane].get("expected_resumed_command_records"), list)
            or not prior_rows[lane]["expected_resumed_command_records"]
            for lane in lanes
        )
    ):
        raise RuntimeError("previous recovery plan does not authorize the retried lanes")

    archived_preflight = versioned_preflight_path(run_root, previous_attempt)
    preserve_exact_bytes(archived_preflight, previous_preflight_bytes)
    lane_files = {}
    for lane in lanes:
        lower = lane.lower()
        sources = {
            "error": run_root / f"lane_{lower}.error.json",
            "status": run_root / f"lane_{lower}.status.json",
            "recovery_commands": run_root / f"lane_{lower}.recovery_commands.jsonl",
        }
        error = read_json(sources["error"])
        status = read_json(sources["status"])
        if (
            error.get("run_id") != run_root.name
            or error.get("lane") != lane
            or error.get("status") != "HARD_FAILURE"
            or not isinstance(error.get("exit_code"), int)
            or error["exit_code"] == 0
            or status.get("run_id") != run_root.name
            or status.get("lane") != lane
            or status.get("status") != "HARD_FAILURE"
            or not isinstance(status.get("exit_code"), int)
            or status["exit_code"] == 0
            or (run_root / f"lane_{lower}.science_complete.json").exists()
            or not terminal_is_sealed_for_lane(
                original_terminal_gate or {},
                lane,
                run_root / f"lane_{lower}.terminal.json",
            )
        ):
            raise RuntimeError(f"{lane} prior recovery outcome is not an exact hard failure")
        lane_files[lane] = {}
        for kind, source in sources.items():
            content = source.read_bytes()
            suffix = ".jsonl" if source.suffix == ".jsonl" else ".json"
            destination = run_root / (
                f"lane_{lower}.{kind}.recovery_attempt_{previous_attempt:04d}{suffix}"
            )
            preserve_exact_bytes(destination, content)
            lane_files[lane][kind] = {
                "source": str(source),
                "preserved_path": str(destination),
                "sha256": hashlib.sha256(content).hexdigest(),
            }

    manifest = {
        "schema_version": 1,
        "run_id": run_root.name,
        "failed_recovery_attempt": previous_attempt,
        "next_recovery_attempt": recovery_attempt,
        "retried_lanes": lanes,
        "prior_preflight": {
            "preserved_path": str(archived_preflight),
            "sha256": hashlib.sha256(previous_preflight_bytes).hexdigest(),
        },
        "prior_command_plan": {
            "path": str(prior_plan_path),
            "sha256": prior_plan_sha,
        },
        "lane_failure_evidence": lane_files,
    }
    manifest_path = run_root / (
        f"RECOVERY_ATTEMPT_{previous_attempt:04d}_EVIDENCE.json"
    )
    effective_manifest_path = manifest_path
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        # A failed preflight-generation command may already have sealed a
        # strict subset of the lanes before discovering that another retained
        # lane also exited.  Preserve that immutable first manifest and add a
        # separately sealed supplement; never replace either set of bytes.
        existing = read_json(manifest_path)
        existing_lanes = existing.get("retried_lanes")
        requested = list(lanes)
        existing_set = set(existing_lanes or [])
        requested_set = set(requested)
        common_exact = (
            existing.get("schema_version") == manifest["schema_version"]
            and existing.get("run_id") == manifest["run_id"]
            and existing.get("failed_recovery_attempt")
            == manifest["failed_recovery_attempt"]
            and existing.get("next_recovery_attempt")
            == manifest["next_recovery_attempt"]
            and existing.get("prior_preflight") == manifest["prior_preflight"]
            and existing.get("prior_command_plan")
            == manifest["prior_command_plan"]
            and isinstance(existing_lanes, list)
            and existing_set < requested_set
            and all(
                existing.get("lane_failure_evidence", {}).get(lane)
                == lane_files[lane]
                for lane in existing_lanes
            )
        )
        if not common_exact:
            raise RuntimeError(
                "existing prior-attempt evidence is not an exact subset"
            )
        supplemented_lanes = [lane for lane in requested if lane not in existing_set]
        supplement = {
            "schema_version": 1,
            "run_id": run_root.name,
            "failed_recovery_attempt": previous_attempt,
            "next_recovery_attempt": recovery_attempt,
            "base_manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
                "retried_lanes": existing_lanes,
            },
            "supplemented_lanes": supplemented_lanes,
            "prior_preflight": manifest["prior_preflight"],
            "prior_command_plan": manifest["prior_command_plan"],
            "lane_failure_evidence": {
                lane: lane_files[lane] for lane in supplemented_lanes
            },
        }
        supplement_path = run_root / (
            f"RECOVERY_ATTEMPT_{previous_attempt:04d}_EVIDENCE_"
            f"SUPPLEMENT_FOR_ATTEMPT_{recovery_attempt:04d}.json"
        )
        preserve_exact_json(supplement_path, supplement)
        index = {
            "schema_version": 1,
            "run_id": run_root.name,
            "failed_recovery_attempt": previous_attempt,
            "next_recovery_attempt": recovery_attempt,
            "retried_lanes": requested,
            "components": [
                {
                    "path": str(manifest_path),
                    "sha256": file_sha256(manifest_path),
                    "lanes": existing_lanes,
                },
                {
                    "path": str(supplement_path),
                    "sha256": file_sha256(supplement_path),
                    "lanes": supplemented_lanes,
                },
            ],
            "prior_preflight": manifest["prior_preflight"],
            "prior_command_plan": manifest["prior_command_plan"],
        }
        effective_manifest_path = run_root / (
            f"RECOVERY_ATTEMPT_{previous_attempt:04d}_EVIDENCE_"
            f"INDEX_FOR_ATTEMPT_{recovery_attempt:04d}.json"
        )
        preserve_exact_json(effective_manifest_path, index)
    else:
        preserve_exact_json(manifest_path, manifest)
    return {
        "failed_recovery_attempt": previous_attempt,
        "manifest_path": str(effective_manifest_path),
        "manifest_sha256": file_sha256(effective_manifest_path),
        "base_manifest_path": str(manifest_path),
        "base_manifest_sha256": file_sha256(manifest_path),
        "prior_command_plan_path": str(prior_plan_path),
        "prior_command_plan_sha256": prior_plan_sha,
        "prior_preflight_path": str(archived_preflight),
        "prior_preflight_sha256": hashlib.sha256(previous_preflight_bytes).hexdigest(),
    }


def exact_tree_inventory(root: Path, expected_files: set[str]) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"exact archived tree is unavailable: {root}")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError(f"archived tree contains a symlink: {root}")
    relative_files = {
        path.relative_to(root).as_posix() for path in paths if path.is_file()
    }
    if relative_files != set(expected_files):
        raise RuntimeError(
            f"archived tree file set differs: {root}: {sorted(relative_files)}"
        )
    return {
        relative: {
            "bytes": (root / relative).stat().st_size,
            "sha256": file_sha256(root / relative),
        }
        for relative in sorted(relative_files)
    }


def move_or_verify_exact_tree(
    source: Path, destination: Path, expected_files: set[str]
) -> dict:
    if source.exists() and destination.exists():
        raise RuntimeError(
            f"refusing ambiguous source and archive trees: {source}, {destination}"
        )
    if source.exists():
        exact_tree_inventory(source, expected_files)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        for parent in {source.parent, destination.parent}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return exact_tree_inventory(destination, expected_files)


def archive_gpu0_attempt2_science(
    master_root: Path,
    run_root: Path,
    recovery_attempt: int,
    prior_attempt_evidence: dict,
) -> dict:
    """Move completed C1 and partial F bytes into immutable attempt-2 evidence."""

    if recovery_attempt != 3 or prior_attempt_evidence.get(
        "failed_recovery_attempt"
    ) != 2:
        raise RuntimeError("GPU0 failed-science archive is attempt-3 specific")
    previous_attempt = 2
    plan_path = versioned_plan_path(run_root, previous_attempt)
    plan = read_json(plan_path)
    expected = plan["recovered_lanes"]["GPU0"][
        "expected_resumed_command_records"
    ]
    command_path = run_root / "lane_gpu0.recovery_commands.jsonl"
    completed = [
        json.loads(line)
        for line in command_path.read_text().splitlines()
        if line
    ]
    error_path = run_root / "lane_gpu0.error.json"
    status_path = run_root / "lane_gpu0.status.json"
    error = read_json(error_path)
    status = read_json(status_path)
    checks = {
        "expected_plan_has_remaining_command": 0 <= len(completed) < len(expected),
        "successful_commands_are_exact_prefix": completed
        == expected[: len(completed)],
        "failed_command_is_exact_next_command": error.get("command")
        == expected[len(completed)]
        if len(completed) < len(expected)
        else False,
        "status_command_matches_error": status.get("command")
        == error.get("command"),
        "failure_identity_exact": error.get("run_id") == run_root.name
        and status.get("run_id") == run_root.name
        and error.get("lane") == status.get("lane") == "GPU0"
        and error.get("status") == status.get("status") == "HARD_FAILURE"
        and error.get("phase") == status.get("phase") == "2D2F_PREFLIGHT"
        and error.get("exit_code") == status.get("exit_code") == 1,
        "prior_attempt_evidence_present": Path(
            prior_attempt_evidence["manifest_path"]
        ).is_file()
        and file_sha256(Path(prior_attempt_evidence["manifest_path"]))
        == prior_attempt_evidence["manifest_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"GPU0 attempt-2 command lineage is not exact: {checks}")

    archive_root = (
        run_root
        / "failed_science_attempts"
        / "gpu0_recovery_attempt_0002"
    )
    archive_root.mkdir(parents=True, exist_ok=True)
    c1_source = (
        master_root
        / "worktrees/master/results/experiment_2d2e_c1_large_true_self_confirmation"
    )
    f_source = (
        master_root
        / "worktrees/2d2f/results/experiment_2d2f_no_b2_recurrence_b3_w64"
    )
    c1_destination = archive_root / "completed_c1"
    f_destination = archive_root / "partial_2d2f_preflight"
    c1_inventory = move_or_verify_exact_tree(
        c1_source, c1_destination, GPU0_COMPLETED_C1_FILES
    )
    f_inventory = move_or_verify_exact_tree(
        f_source, f_destination, GPU0_FAILED_F_PREFLIGHT_FILES
    )
    c1_audit = read_json(c1_destination / "FINAL_AUDIT.json")
    f_semantic = read_json(f_destination / "semantic_diff_audit.json")

    logs = {}
    logs_root = archive_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    for name in ("lane_gpu0.log", "lane_gpu0.recovery.console.log"):
        source = run_root / name
        destination = logs_root / name
        preserve_exact_bytes(destination, source.read_bytes())
        logs[name] = {
            "source": str(source),
            "preserved_path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
        }
    console = (logs_root / "lane_gpu0.recovery.console.log").read_text()
    checks.update(
        {
            "completed_c1_audit_passed": c1_audit.get("passed") is True,
            "completed_c1_classification_exact": c1_audit.get("classification")
            == "DIRECTIONAL CONFIRMATION",
            "partial_f_semantic_diff_passed": f_semantic.get("passed") is True,
            "diagnostic_failure_trace_exact": "KeyError: 'local_attention_weights'"
            in console,
            "source_output_roots_moved_not_deleted": not c1_source.exists()
            and not f_source.exists(),
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"GPU0 failed-science archive is not exact: {checks}")
    payload = {
        "schema_version": 1,
        "run_id": run_root.name,
        "lane": "GPU0",
        "failed_recovery_attempt": previous_attempt,
        "next_recovery_attempt": recovery_attempt,
        "phase": "2D2F_PREFLIGHT",
        "successful_command_records": completed,
        "failed_command_record": error["command"],
        "prior_attempt_evidence": prior_attempt_evidence,
        "error_marker": {
            "path": str(error_path),
            "sha256": file_sha256(error_path),
        },
        "status_marker": {
            "path": str(status_path),
            "sha256": file_sha256(status_path),
        },
        "completed_c1": {
            "original_path": str(c1_source),
            "archive_path": str(c1_destination),
            "files": c1_inventory,
        },
        "partial_2d2f_preflight": {
            "original_path": str(f_source),
            "archive_path": str(f_destination),
            "files": f_inventory,
        },
        "logs": logs,
        "moved_not_deleted": True,
        "checks": checks,
        "passed": True,
    }
    manifest_path = run_root / "GPU0_FAILED_SCIENCE_RECOVERY_ATTEMPT_0002.json"
    preserve_exact_json(manifest_path, payload)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        **payload,
    }


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def process_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_group_alive(process_group_id) -> bool:
    if not isinstance(process_group_id, int) or process_group_id <= 0:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def audit_original_terminal_recovery(
    master_root: Path, run_root: Path, explicitly_authorized: bool
) -> dict:
    master_paths = {
        "run_terminal": run_root / "MASTER_TERMINAL_STATUS.json",
        "run_all_lanes": run_root / "MASTER_ALL_LANES_TERMINAL",
        "top_terminal": master_root / "MASTER_TERMINAL_STATUS.json",
        "top_all_lanes": master_root / "MASTER_ALL_LANES_TERMINAL",
    }
    any_terminal = any(path.exists() for path in master_paths.values()) or any(
        (run_root / f"lane_gpu{index}.terminal.json").exists()
        for index in range(4)
    )
    if not explicitly_authorized:
        return {
            "mode": "original_supervisor_terminal_recovery",
            "explicit_cli_authorization": False,
            "terminal_state_present": any_terminal,
            "checks": {
                "explicitly_authorized_if_terminal_state_present": not any_terminal,
            },
            "lanes": {},
            "passed": not any_terminal,
        }

    master_bytes = {name: path.read_bytes() for name, path in master_paths.items()}
    canonical_bytes = master_bytes["run_terminal"]
    terminal = json.loads(canonical_bytes)
    if not isinstance(terminal, dict):
        raise RuntimeError("original master terminal is not a JSON object")
    embedded = terminal.get("lanes")
    lane_rows = {}
    for index in range(4):
        lane = f"GPU{index}"
        path = run_root / f"lane_{lane.lower()}.terminal.json"
        row = read_json(path)
        expected_status = "SUCCESS" if lane == "GPU3" else "HARD_FAILURE"
        expected_returncode = 0 if lane == "GPU3" else None
        checks = {
            "identity_exact": row.get("run_id") == run_root.name
            and row.get("lane") == lane,
            "embedded_exact": isinstance(embedded, dict)
            and embedded.get(lane) == row,
            "status_exact": row.get("status") == expected_status,
            "returncode_exact": row.get("returncode") == expected_returncode
            if lane == "GPU3"
            else isinstance(row.get("returncode"), int)
            and row["returncode"] != 0,
        }
        lane_rows[lane] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "status": row.get("status"),
            "returncode": row.get("returncode"),
            "checks": checks,
            "passed": all(checks.values()),
        }

    heartbeat_pid = terminal.get("heartbeat_pid")
    heartbeat_pgid = terminal.get("heartbeat_process_group_id")
    checks = {
        "explicit_cli_authorization": explicitly_authorized,
        "all_master_terminal_bytes_exact": all(
            content == canonical_bytes for content in master_bytes.values()
        ),
        "run_identity_exact": terminal.get("run_id") == run_root.name,
        "pod_identity_exact": terminal.get("pod") == EXPECTED_POD,
        "overall_hard_failure": terminal.get("status") == "HARD_FAILURE",
        "all_four_original_shells_exited": terminal.get(
            "all_four_lane_shells_exited"
        )
        is True,
        "all_lanes_terminal": terminal.get("all_lanes_terminal") is True,
        "lane_set_exact": isinstance(embedded, dict)
        and set(embedded) == {"GPU0", "GPU1", "GPU2", "GPU3"},
        "standalone_and_embedded_lanes_exact": all(
            row["passed"] for row in lane_rows.values()
        ),
        "heartbeat_declared_retained": terminal.get(
            "heartbeat_left_running_for_finalization"
        )
        is True,
        "heartbeat_pid_alive": process_alive(heartbeat_pid),
        "heartbeat_process_group_alive": process_group_alive(heartbeat_pgid),
    }
    return {
        "mode": "original_supervisor_terminal_recovery",
        "explicit_cli_authorization": explicitly_authorized,
        "terminal_state_present": any_terminal,
        "master_records": {
            name: {"path": str(master_paths[name]), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in master_bytes.items()
        },
        "master_terminal_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "heartbeat": {
            "pid": heartbeat_pid,
            "process_group_id": heartbeat_pgid,
            "pid_alive": checks["heartbeat_pid_alive"],
            "process_group_alive": checks["heartbeat_process_group_alive"],
        },
        "lanes": lane_rows,
        "checks": checks,
        "passed": all(checks.values()),
    }


def terminal_is_sealed_for_lane(gate: dict, lane: str, path: Path) -> bool:
    if not path.exists():
        return True
    row = gate.get("lanes", {}).get(lane, {}) if isinstance(gate, dict) else {}
    return (
        gate.get("passed") is True
        and gate.get("explicit_cli_authorization") is True
        and row.get("passed") is True
        and row.get("path") == str(path)
        and row.get("sha256") == file_sha256(path)
    )


def git_output(worktree: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=worktree, text=True).strip()


def untracked_files(worktree: Path) -> list[str]:
    raw = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=worktree,
    )
    return sorted(os.fsdecode(item) for item in raw.split(b"\0") if item)


def path_is_below_exact_root(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    exact_root = PurePosixPath(root)
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    try:
        candidate.relative_to(exact_root)
    except ValueError:
        return False
    return candidate != exact_root


def audit_worktree_artifacts(worktree: Path, allowed_roots) -> dict:
    roots = sorted(allowed_roots)
    tracked_status = git_output(
        worktree, "status", "--porcelain=v1", "--untracked-files=no"
    )
    untracked = untracked_files(worktree)
    disallowed = [
        path
        for path in untracked
        if not any(path_is_below_exact_root(path, root) for root in roots)
    ]
    checks = {
        "tracked_worktree_clean": tracked_status == "",
        "untracked_only_in_exact_result_roots": not disallowed,
    }
    return {
        "allowed_untracked_result_roots": roots,
        "tracked_status": tracked_status,
        "untracked_files": untracked,
        "disallowed_untracked_files": disallowed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_patched_worktree(spec: dict, original_git: dict) -> dict:
    """Seal one additional lane worktree whose implementation needed recovery."""

    worktree = Path(spec["worktree"]).resolve()
    branch = spec["branch"]
    old = original_git["branches"][branch]["local"]
    current = git_output(worktree, "rev-parse", "HEAD")
    origin = git_output(worktree, "rev-parse", f"origin/{branch}")
    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", old, current], cwd=worktree
    ) == 0
    changed = set(
        filter(
            None,
            git_output(
                worktree, "diff", "--name-only", f"{old}..{current}"
            ).splitlines(),
        )
    )
    worktree_audit = audit_worktree_artifacts(
        worktree, spec["allowed_untracked_result_roots"]
    )
    test_command = [sys.executable, "-m", "pytest", "-q", *spec["tests"]]
    test = subprocess.run(
        test_command, cwd=worktree, text=True, capture_output=True
    )
    checks = {
        "old_commit_is_ancestor": ancestor,
        "current_commit_pushed": current == origin,
        "tracked_worktree_clean": worktree_audit["checks"][
            "tracked_worktree_clean"
        ],
        "untracked_only_in_exact_result_roots": worktree_audit["checks"][
            "untracked_only_in_exact_result_roots"
        ],
        "changed_files_narrow": changed == set(spec["allowed_changed_files"]),
        "focused_tests_passed": test.returncode == 0,
    }
    return {
        "branch": branch,
        "worktree": str(worktree),
        "old_commit": old,
        "current_commit": current,
        "origin_commit": origin,
        "changed_files": sorted(changed),
        "worktree_artifact_audit": worktree_audit,
        "focused_test_command": test_command,
        "focused_test_stdout": test.stdout,
        "focused_test_stderr": test.stderr,
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_checkpoint_sidecars(checkpoint: Path, expected_sha: str | None) -> dict:
    sha_path = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    verification_path = checkpoint.with_suffix(
        checkpoint.suffix + ".verification.json"
    )
    observed_sha = file_sha256(checkpoint)
    sha_tokens = sha_path.read_text().split()
    verification = read_json(verification_path)
    checks = {
        "sha_sidecar_exact": sha_tokens == [observed_sha, checkpoint.name],
        "expected_sha_exact": expected_sha is None or observed_sha == expected_sha,
        "strict_reopen_sidecar_passed": verification.get("passed") is True,
    }
    return {
        "checkpoint": str(checkpoint),
        "sha256": observed_sha,
        "sha_sidecar": str(sha_path),
        "verification_sidecar": str(verification_path),
        "checks": checks,
        "passed": all(checks.values()),
    }


def gpu_idle(index: int) -> dict:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    row = next(fields for fields in ([item.strip() for item in line.split(",")] for line in rows) if int(fields[0]) == index)
    compute = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    assigned = [line for line in compute if line.split(",", 1)[0].strip() == row[1]]
    checks = {
        "no_compute_process": not assigned,
        "utilization_zero": int(row[2]) == 0,
        "memory_used_at_most_driver_baseline": int(row[3]) <= 16,
    }
    return {
        "index": index,
        "uuid": row[1],
        "utilization_percent": int(row[2]),
        "memory_used_mib": int(row[3]),
        "compute_processes": assigned,
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_lane(
    master_root: Path,
    run_root: Path,
    lane: str,
    original_git: dict,
    original_terminal_gate: dict,
) -> dict:
    spec = LANES[lane]
    worktree = Path(spec["worktree"]).resolve()
    branch = spec["branch"]
    old = original_git["branches"][branch]["local"]
    current = git_output(worktree, "rev-parse", "HEAD")
    origin = git_output(worktree, "rev-parse", f"origin/{branch}")
    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", old, current], cwd=worktree
    ) == 0
    changed = set(
        filter(None, git_output(worktree, "diff", "--name-only", f"{old}..{current}").splitlines())
    )
    error_path = run_root / f"lane_{lane.lower()}.error.json"
    terminal_path = run_root / f"lane_{lane.lower()}.terminal.json"
    error = read_json(error_path)
    checkpoint = Path(spec["base_checkpoint"])
    checkpoint_audit = audit_checkpoint_sidecars(
        checkpoint, spec.get("base_sha256")
    )
    worktree_audit = audit_worktree_artifacts(
        worktree, spec["allowed_untracked_result_roots"]
    )
    test_command = [sys.executable, "-m", "pytest", "-q", *spec["tests"]]
    test = subprocess.run(test_command, cwd=worktree, text=True, capture_output=True)
    dependent_worktree_patches = [
        audit_patched_worktree(patch_spec, original_git)
        for patch_spec in spec.get("dependent_worktree_patches", [])
    ]
    checks = {
        "prior_failure_exact": error.get("run_id") == run_root.name
        and error.get("lane") == lane
        and error.get("status") == "HARD_FAILURE"
        and isinstance(error.get("exit_code"), int)
        and error["exit_code"] != 0,
        "no_success_marker": not (run_root / f"lane_{lane.lower()}.science_complete.json").exists(),
        "terminal_absent_or_exactly_sealed": terminal_is_sealed_for_lane(
            original_terminal_gate, lane, terminal_path
        ),
        "old_commit_is_ancestor": ancestor,
        "current_commit_pushed": current == origin,
        "tracked_worktree_clean": worktree_audit["checks"][
            "tracked_worktree_clean"
        ],
        "untracked_only_in_exact_result_roots": worktree_audit["checks"][
            "untracked_only_in_exact_result_roots"
        ],
        "changed_files_narrow": changed == set(spec["allowed_changed_files"]),
        "base_checkpoint_and_sidecars_exact": checkpoint_audit["passed"],
        "focused_tests_passed": test.returncode == 0,
        "dependent_worktree_patches_exact": all(
            row["passed"] for row in dependent_worktree_patches
        ),
    }
    idle = gpu_idle(int(lane.removeprefix("GPU")))
    checks["assigned_gpu_idle"] = idle["passed"]
    return {
        "lane": lane,
        "branch": branch,
        "old_commit": old,
        "current_commit": current,
        "origin_commit": origin,
        "changed_files": sorted(changed),
        "recovery_reason": spec["recovery_reason"],
        "prior_failure_marker": str(error_path),
        "prior_failure_marker_sha256": file_sha256(error_path),
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": checkpoint_audit["sha256"],
        "checkpoint_sidecar_audit": checkpoint_audit,
        "strict_checkpoint_reopen_passed": checkpoint_audit["checks"][
            "strict_reopen_sidecar_passed"
        ],
        "worktree_artifact_audit": worktree_audit,
        "original_terminal": (
            original_terminal_gate.get("lanes", {}).get(lane)
            if terminal_path.exists()
            else None
        ),
        "focused_test_command": test_command,
        "focused_test_stdout": test.stdout,
        "focused_test_stderr": test.stderr,
        "dependent_worktree_patches": dependent_worktree_patches,
        "assigned_gpu": idle,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(args) -> dict:
    master_root = Path(args.master_root).resolve()
    run_root = (master_root / "runs" / args.run_id).resolve()
    recovery_attempt = int(args.recovery_attempt)
    if run_root.name != args.run_id or not run_root.is_dir():
        raise RuntimeError("exact run directory is missing")
    if recovery_attempt < 1:
        raise RuntimeError("recovery attempt must be a positive integer")
    lanes = list(dict.fromkeys(args.lane))
    retained_lanes = list(dict.fromkeys(args.retain_active_lane or []))
    if not lanes or any(lane not in LANES for lane in lanes):
        raise RuntimeError("only the registered failed lanes can be recovered")
    if (
        any(lane not in LANES for lane in retained_lanes)
        or set(lanes) & set(retained_lanes)
    ):
        raise RuntimeError("retained recovery lanes must be registered and disjoint")
    original_terminal_gate = audit_original_terminal_recovery(
        master_root, run_root, args.allow_original_terminal_recovery
    )
    if original_terminal_gate.get("passed") is not True:
        raise RuntimeError(
            "original-supervisor terminal recovery gate did not pass exactly"
        )
    prior_attempt_evidence = prepare_prior_attempt_evidence(
        run_root, lanes, recovery_attempt, original_terminal_gate
    )
    failed_science_archives = {}
    if recovery_attempt == 3 and "GPU0" in lanes:
        failed_science_archives["GPU0"] = archive_gpu0_attempt2_science(
            master_root,
            run_root,
            recovery_attempt,
            prior_attempt_evidence,
        )
    top_preflight = read_json(master_root / "MASTER_PREFLIGHT.json")
    scoped_preflight = read_json(run_root / "MASTER_PREFLIGHT.json")
    original_git = read_json(run_root / "git_worktree_manifest.json")
    immutable_names = (
        "hardware_manifest.json",
        "storage_preflight.json",
        "source_checkpoint_manifest.json",
        "shared_dataset_manifest.json",
        "AUTO_STOP_PREFLIGHT.json",
    )
    immutable = {}
    for name in immutable_names:
        top = master_root / name
        scoped = run_root / name
        immutable[name] = {
            "top_sha256": file_sha256(top),
            "run_scoped_sha256": file_sha256(scoped),
            "exact_copy": top.read_bytes() == scoped.read_bytes(),
            "passed": read_json(scoped).get("passed") is True,
        }
    lane_evidence = {
        lane: audit_lane(
            master_root,
            run_root,
            lane,
            original_git,
            original_terminal_gate,
        )
        for lane in lanes
    }
    retained_checks = {}
    retained_schemas = {}
    if retained_lanes:
        previous = read_json(run_root / "RECOVERY_PREFLIGHT.json")
        previous_schemas = previous.get("recovery_evidence_schemas", {})
        for lane in retained_lanes:
            evidence = previous.get("lane_evidence", {}).get(lane)
            status = read_json(run_root / f"lane_{lane.lower()}.status.json")
            error_path = run_root / f"lane_{lane.lower()}.error.json"
            valid = (
                previous.get("passed") is True
                and lane in previous.get("authorized_lanes", [])
                and isinstance(evidence, dict)
                and evidence.get("passed") is True
                and evidence.get("prior_failure_marker_sha256")
                == file_sha256(error_path)
                and status.get("run_id") == args.run_id
                and status.get("status") == "RUNNING"
                and not (run_root / f"lane_{lane.lower()}.science_complete.json").exists()
                and terminal_is_sealed_for_lane(
                    original_terminal_gate,
                    lane,
                    run_root / f"lane_{lane.lower()}.terminal.json",
                )
            )
            retained_checks[lane] = valid
            if not valid:
                raise RuntimeError(f"active retained recovery lane is not exact: {lane}")
            schema = previous_schemas.get(lane)
            if schema not in {
                "v2_with_recovery_reason",
                "legacy_v1_without_recovery_reason",
            }:
                raise RuntimeError(f"retained recovery schema is invalid: {lane}")
            retained_schemas[lane] = schema
            # Older passing recovery evidence predates lane-specific reasons.
            # Seal the registered reason into the new combined preflight so
            # completion and reconciliation can require one exact value.
            evidence = dict(evidence)
            evidence["recovery_reason"] = LANES[lane]["recovery_reason"]
            terminal_path = run_root / f"lane_{lane.lower()}.terminal.json"
            evidence["original_terminal"] = (
                original_terminal_gate.get("lanes", {}).get(lane)
                if terminal_path.exists()
                else None
            )
            lane_evidence[lane] = evidence
    authorized_lanes = [*retained_lanes, *lanes]
    recovery_evidence_schemas = {
        **retained_schemas,
        **{lane: "v2_with_recovery_reason" for lane in lanes},
    }
    recovery_reasons = {
        lane: lane_evidence[lane].get(
            "recovery_reason", LANES[lane]["recovery_reason"]
        )
        for lane in authorized_lanes
    }
    unaffected = {}
    for lane in ({"GPU0", "GPU1", "GPU2", "GPU3"} - set(authorized_lanes)):
        status = read_json(run_root / f"lane_{lane.lower()}.status.json")
        unaffected[lane] = {
            "status": status.get("status"),
            "phase": status.get("phase"),
            "same_run": status.get("run_id") == args.run_id,
            "no_error_marker": not (run_root / f"lane_{lane.lower()}.error.json").exists(),
        }
    checks = {
        "original_shared_preflight_still_exact": top_preflight == scoped_preflight
        and scoped_preflight.get("passed") is True
        and scoped_preflight.get("run_id") == args.run_id,
        "immutable_shared_manifests_unchanged": all(
            row["exact_copy"] and row["passed"] for row in immutable.values()
        ),
        "failed_lane_patches_narrow_and_pushed": all(
            row["passed"] for row in lane_evidence.values()
        ),
        "unaffected_lanes_same_run_without_error": all(
            row["same_run"] and row["no_error_marker"] for row in unaffected.values()
        ),
        "retained_active_recovery_lanes_exact": all(retained_checks.values()),
        "original_supervisor_terminal_recovery_exact": original_terminal_gate.get(
            "passed"
        )
        is True,
    }
    payload = {
        "schema_version": 1,
        "created_utc": now_utc(),
        "run_id": args.run_id,
        "recovery_attempt": recovery_attempt,
        "authorized_lanes": authorized_lanes,
        "retained_active_lanes": retained_lanes,
        "recovery_evidence_schemas": recovery_evidence_schemas,
        "reason": (
            next(iter(recovery_reasons.values()))
            if len(recovery_reasons) == 1
            else "lane-specific audited recovery; see recovery_reasons"
        ),
        "recovery_reasons": recovery_reasons,
        "original_master_preflight": str(run_root / "MASTER_PREFLIGHT.json"),
        "original_terminal_recovery_gate": original_terminal_gate,
        "immutable_manifest_audit": immutable,
        "lane_evidence": lane_evidence,
        "unaffected_lanes": unaffected,
        "checks": checks,
        "passed": all(checks.values()),
        "pod_stop_automated": False,
    }
    if prior_attempt_evidence is not None:
        payload["prior_attempt_evidence"] = prior_attempt_evidence
    if failed_science_archives:
        payload["failed_science_archives"] = failed_science_archives
    plan = recovery_command_plan(
        master_root,
        run_root,
        authorized_lanes,
        retained_lanes,
        recovery_attempt,
        recovery_evidence_schemas,
    )
    plan_path = versioned_plan_path(run_root, recovery_attempt)
    payload["recovery_command_plan"] = {
        "path": str(plan_path),
        "sha256": hashlib.sha256(
            (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "authorized_lanes": authorized_lanes,
        "recovery_attempt": recovery_attempt,
    }
    if not payload["passed"]:
        durable_json(
            run_root
            / f"RECOVERY_PREFLIGHT_ATTEMPT_{recovery_attempt:04d}_FAILED.json",
            payload,
        )
        raise RuntimeError(f"recovery preflight failed: {checks}")
    preserve_exact_json(plan_path, plan)
    if recovery_attempt > 1:
        preserve_exact_json(
            versioned_preflight_path(run_root, recovery_attempt), payload
        )
    durable_json(run_root / "RECOVERY_PREFLIGHT.json", payload)
    print("PARALLEL_2D2_RECOVERY_PREFLIGHT_PASS", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lane", action="append", required=True)
    parser.add_argument("--retain-active-lane", action="append")
    parser.add_argument("--recovery-attempt", type=int, default=1)
    parser.add_argument("--allow-original-terminal-recovery", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
