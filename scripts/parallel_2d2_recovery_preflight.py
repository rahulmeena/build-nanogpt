#!/usr/bin/env python3
"""Authorize narrowly scoped recovery of lanes that failed before update 1.

The original passing shared preflight remains immutable.  This audit proves
that only the failed lane implementations changed, that their frozen
update-zero checkpoints remain exact, and that the assigned GPUs are idle.
It never launches a process or controls a pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


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
        },
        "tests": ["tests/test_experiment_2d2e_c1.py"],
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
        "tests": [
            "tests/test_experiment_2d2h_core.py",
            "tests/test_experiment_2d2h_driver.py",
        ],
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


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def git_output(worktree: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=worktree, text=True).strip()


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
    sha_path = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    verification_path = checkpoint.with_suffix(checkpoint.suffix + ".verification.json")
    observed_sha = file_sha256(checkpoint)
    verification = read_json(verification_path)
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
        "worktree_clean": git_output(worktree, "status", "--porcelain") == "",
        "changed_files_narrow": changed == set(spec["allowed_changed_files"]),
        "base_checkpoint_sha_exact": observed_sha == spec["base_sha256"],
        "base_sha_sidecar_exact": sha_path.read_text().split()[0] == spec["base_sha256"],
        "base_strict_reopen_sidecar_passed": verification.get("passed") is True,
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
        "prior_failure_marker": str(error_path),
        "prior_failure_marker_sha256": file_sha256(error_path),
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": observed_sha,
        "strict_checkpoint_reopen_passed": verification.get("passed") is True,
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
    if not lanes or any(lane not in LANES for lane in lanes):
        raise RuntimeError("only the registered failed lanes can be recovered")
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
    unaffected = {}
    for lane in ({"GPU0", "GPU1", "GPU2", "GPU3"} - set(lanes)):
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
    }
    payload = {
        "schema_version": 1,
        "created_utc": now_utc(),
        "run_id": args.run_id,
        "authorized_lanes": lanes,
        "reason": "launch-time implementation defects before scientific update 1; exact frozen update-zero restart",
        "original_master_preflight": str(run_root / "MASTER_PREFLIGHT.json"),
        "immutable_manifest_audit": immutable,
        "lane_evidence": lane_evidence,
        "unaffected_lanes": unaffected,
        "checks": checks,
        "passed": all(checks.values()),
        "pod_stop_automated": False,
    }
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
    args = parser.parse_args()
    try:
        run(args)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
