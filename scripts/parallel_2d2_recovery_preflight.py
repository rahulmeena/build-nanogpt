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
            "scripts/parallel_2d2_supervisor.py",
            "scripts/test_parallel_2d2_orchestration.py",
            "scripts/test_parallel_2d2_recovery_preflight.py",
        },
        "allowed_untracked_result_roots": {
            "results/experiment_2d2e_c1_large_true_self_confirmation",
        },
        "recovery_reason": (
            "2D2E-C1 completed its evaluation but artifact publication failed on "
            "non-JSON CUDA UUID metadata; deterministically rerun frozen C1 from "
            "the exact 2D2E checkpoint, then execute the unchanged 2D2F sequence"
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
            "2D2G-B disposable smoke evaluation failed on an evaluation-device "
            "implementation defect after exact 2D2G-A update 191 completed; "
            "retain Stage-A-191 and rerun only smoke-B plus Stage B"
        ),
        "tests": [
            "tests/test_experiment_2d2g_core.py",
            "tests/test_experiment_2d2g_driver.py",
        ],
    },
    "GPU2": {
        "branch": "experiment-2d2h-no-b1-recurrence-b2-w32",
        "worktree": "/workspace/parallel_2d2_master/worktrees/2d2h",
        "base_checkpoint": "/workspace/exp2d2b_run/checkpoints/scientific_update_0191.pt",
        "base_sha256": "8c39f47248e3a5f4dc69f5e8e97c8a1cd1bcdfa91154eba5804c448942075326",
        "allowed_changed_files": {
            "scripts/experiment_2d2h.py",
            "tests/test_experiment_2d2h_driver.py",
        },
        "allowed_untracked_result_roots": {
            "results/experiment_2d2h_no_b1_recurrence_b2_w32",
        },
        "recovery_reason": (
            "2D2H failed before scientific update 1 on B2 gate-freshness audit "
            "ordering; restart from the exact frozen 2D2B checkpoint"
        ),
        "tests": [
            "tests/test_experiment_2d2h_core.py",
            "tests/test_experiment_2d2h_driver.py",
        ],
    },
}

BASH_PERCENT_Q_SAFE = re.compile(r"[A-Za-z0-9_@%+=:,./-]+")


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
            [*driver, "preflight", *common],
            [*driver, "smoke", *common],
            [*driver, "train", *common, "--end-update", "96"],
            [
                *driver, "train", *common, "--end-update", "191",
                "--resume", f"{ephemeral}/scientific_update_0096.pt",
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
) -> dict:
    retained = set(retained_active_lanes)
    return {
        "schema_version": 1,
        "run_id": run_root.name,
        "recovered_lanes": {
            lane: {
                "recovery_reason": LANES[lane]["recovery_reason"],
                "recovery_evidence_schema": (
                    "legacy_v1_without_recovery_reason"
                    if lane in retained
                    else "v2_with_recovery_reason"
                ),
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
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"refusing to replace changed recovery plan: {path}")
        return
    durable_json(path, payload)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


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


def audit_lane(master_root: Path, run_root: Path, lane: str, original_git: dict) -> dict:
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
    error = read_json(error_path)
    checkpoint = Path(spec["base_checkpoint"])
    checkpoint_audit = audit_checkpoint_sidecars(
        checkpoint, spec.get("base_sha256")
    )
    worktree_audit = audit_worktree_artifacts(
        worktree, spec["allowed_untracked_result_roots"]
    )
    test_command = ["python", "-m", "pytest", "-q", *spec["tests"]]
    test = subprocess.run(test_command, cwd=worktree, text=True, capture_output=True)
    checks = {
        "prior_failure_exact": error.get("run_id") == run_root.name
        and error.get("lane") == lane
        and error.get("status") == "HARD_FAILURE"
        and isinstance(error.get("exit_code"), int)
        and error["exit_code"] != 0,
        "no_success_marker": not (run_root / f"lane_{lane.lower()}.science_complete.json").exists(),
        "no_terminal_marker": not (run_root / f"lane_{lane.lower()}.terminal.json").exists(),
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
        "focused_test_command": test_command,
        "focused_test_stdout": test.stdout,
        "focused_test_stderr": test.stderr,
        "assigned_gpu": idle,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(args) -> dict:
    master_root = Path(args.master_root).resolve()
    run_root = (master_root / "runs" / args.run_id).resolve()
    if run_root.name != args.run_id or not run_root.is_dir():
        raise RuntimeError("exact run directory is missing")
    lanes = list(dict.fromkeys(args.lane))
    retained_lanes = list(dict.fromkeys(args.retain_active_lane or []))
    if not lanes or any(lane not in LANES for lane in lanes):
        raise RuntimeError("only the registered failed lanes can be recovered")
    if (
        any(lane not in LANES for lane in retained_lanes)
        or set(lanes) & set(retained_lanes)
    ):
        raise RuntimeError("retained recovery lanes must be registered and disjoint")
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
        lane: audit_lane(master_root, run_root, lane, original_git) for lane in lanes
    }
    retained_checks = {}
    if retained_lanes:
        previous = read_json(run_root / "RECOVERY_PREFLIGHT.json")
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
                and not (run_root / f"lane_{lane.lower()}.terminal.json").exists()
            )
            retained_checks[lane] = valid
            if not valid:
                raise RuntimeError(f"active retained recovery lane is not exact: {lane}")
            # Older passing recovery evidence predates lane-specific reasons.
            # Seal the registered reason into the new combined preflight so
            # completion and reconciliation can require one exact value.
            evidence = dict(evidence)
            evidence["recovery_reason"] = LANES[lane]["recovery_reason"]
            lane_evidence[lane] = evidence
    authorized_lanes = [*retained_lanes, *lanes]
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
    }
    payload = {
        "schema_version": 1,
        "created_utc": now_utc(),
        "run_id": args.run_id,
        "authorized_lanes": authorized_lanes,
        "retained_active_lanes": retained_lanes,
        "recovery_evidence_schemas": {
            lane: (
                "legacy_v1_without_recovery_reason"
                if lane in retained_lanes
                else "v2_with_recovery_reason"
            )
            for lane in authorized_lanes
        },
        "reason": (
            next(iter(recovery_reasons.values()))
            if len(recovery_reasons) == 1
            else "lane-specific audited recovery; see recovery_reasons"
        ),
        "recovery_reasons": recovery_reasons,
        "original_master_preflight": str(run_root / "MASTER_PREFLIGHT.json"),
        "immutable_manifest_audit": immutable,
        "lane_evidence": lane_evidence,
        "unaffected_lanes": unaffected,
        "checks": checks,
        "passed": all(checks.values()),
        "pod_stop_automated": False,
    }
    plan = recovery_command_plan(
        master_root, run_root, authorized_lanes, retained_lanes
    )
    plan_path = run_root / "RECOVERY_COMMAND_PLAN.json"
    payload["recovery_command_plan"] = {
        "path": str(plan_path),
        "sha256": hashlib.sha256(
            (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "authorized_lanes": authorized_lanes,
    }
    if payload["passed"]:
        preserve_exact_json(plan_path, plan)
    durable_json(run_root / "RECOVERY_PREFLIGHT.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"recovery preflight failed: {checks}")
    print("PARALLEL_2D2_RECOVERY_PREFLIGHT_PASS", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lane", action="append", required=True)
    parser.add_argument("--retain-active-lane", action="append")
    args = parser.parse_args()
    try:
        run(args)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
