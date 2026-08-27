#!/usr/bin/env python3
"""Coordinator utilities for the four-GPU recurrent-KV architecture matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch


POD_ID = "7i2zyd53ytspwz"
POD_NAME = "empirical_tan_panda"
VOLUME_ID = "yhzyb27fb5"
VOLUME_CAPACITY_DECIMAL_BYTES = 100_000_000_000
EXPECTED_SOURCES = {
    "2D2B": {
        "path": "/workspace/exp2d2b_run/checkpoints/scientific_update_0191.pt",
        "sha256": "8c39f47248e3a5f4dc69f5e8e97c8a1cd1bcdfa91154eba5804c448942075326",
        "bytes": 1_493_937_033,
        "commit": "976b92927e698afd27d68eabc78db5a0b6714fef",
    },
    "2D2D": {
        "path": "/workspace/exp2d2d_run/checkpoints/scientific_update_0191.pt",
        "sha256": "d38e8282cca4df395204b860d17e2cd9b89ff7ad07319fe744bbdc46fb945063",
        "bytes": 1_493_940_151,
        "commit": "a9300a9800f2e2c46f3892cff52b0a4a2a547d11",
    },
    "2D2E": {
        "path": "/workspace/exp2d2e_run/checkpoints/scientific_update_0191.pt",
        "sha256": "dea5e76b55d1ad7281fe3cf3893713392343b875182ba186a0049904e61de790",
        "bytes": 1_493_942_757,
        "commit": "406bef0dc0f375d6783cfc364a935e62bb54d982",
        "tag": "experiment-2d2e-b3-w64-b10-recurrent-960-final",
    },
}
VALIDATION_SHA256 = "8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f"
FINAL_CHECKPOINT_COUNT = 4
PERSISTENT_ARTIFACT_ALLOWANCE = 500_000_000
PERSISTENT_SAFETY_MARGIN = 8_000_000_000
# Four update-96 files, four local finals, G-A update-96/final, and bounded
# smoke/recovery headroom.  Superseded disposable files are removed only after
# their successor has been strictly reopened and hashed.
EPHEMERAL_CHECKPOINT_EQUIVALENTS = 12
EPHEMERAL_SAFETY_MARGIN = 2_000_000_000


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def durable_json(path: Path, payload) -> None:
    path = Path(path)
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


def run_output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hardware_manifest() -> dict:
    query = run_output(
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,driver_version,temperature.gpu,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    rows = []
    for line in query.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 9:
            raise SystemExit(f"unexpected nvidia-smi row: {line}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "pci_bus_id": fields[2],
                "name": fields[3],
                "driver_version": fields[4],
                "temperature_c": int(fields[5]),
                "memory_total_mib": int(fields[6]),
                "memory_free_mib": int(fields[7]),
                "utilization_percent": int(fields[8]),
            }
        )
    checks = {
        "exactly_four_nvidia_devices": len(rows) == 4,
        "torch_sees_four_devices": torch.cuda.is_available() and torch.cuda.device_count() == 4,
        "all_A100_SXM4_80GB": all(
            "A100-SXM4-80GB" in row["name"] and row["memory_total_mib"] >= 81_000
            for row in rows
        ),
        "all_initially_idle": all(row["utilization_percent"] == 0 for row in rows),
        "all_have_at_least_80GB_free": all(row["memory_free_mib"] >= 80_000 for row in rows),
        "unique_UUIDs": len({row["uuid"] for row in rows}) == 4,
        "unique_PCI_bus_IDs": len({row["pci_bus_id"] for row in rows}) == 4,
    }
    return {
        "created_utc": utc_now(),
        "gpus": rows,
        "nvidia_smi": run_output("nvidia-smi"),
        "topology": run_output("nvidia-smi", "topo", "-m"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "checks": checks,
        "passed": all(checks.values()),
    }


def logical_workspace_bytes(workspace: Path) -> dict:
    total = 0
    files = 0
    links = 0
    for base, _, names in os.walk(workspace, followlinks=False):
        for name in names:
            path = Path(base) / name
            stat = path.lstat()
            if path.is_symlink():
                links += 1
            else:
                files += 1
                total += stat.st_size
    return {"logical_file_bytes": total, "file_count": files, "symlink_count": links}


def filesystem_report(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path.resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "findmnt": run_output("findmnt", "-T", str(path), "-J", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"),
    }


def storage_manifest(workspace: Path, ephemeral: Path) -> dict:
    logical = logical_workspace_bytes(workspace)
    logical_free = VOLUME_CAPACITY_DECIMAL_BYTES - logical["logical_file_bytes"]
    largest = max(item["bytes"] for item in EXPECTED_SOURCES.values())
    persistent_required = FINAL_CHECKPOINT_COUNT * largest + PERSISTENT_ARTIFACT_ALLOWANCE
    ephemeral_required = EPHEMERAL_CHECKPOINT_EQUIVALENTS * largest
    persistent = filesystem_report(workspace)
    ephemeral.parent.mkdir(parents=True, exist_ok=True)
    ephemeral.mkdir(parents=True, exist_ok=True)
    local = filesystem_report(ephemeral)
    checks = {
        "workspace_is_FUSE_network_volume": json.loads(persistent["findmnt"])["filesystems"][0]["fstype"] == "fuse",
        "logical_quota_free_positive": logical_free > 0,
        "persistent_projection_with_margin_fits": logical_free >= persistent_required + PERSISTENT_SAFETY_MARGIN,
        "ephemeral_is_not_workspace_mount": json.loads(local["findmnt"])["filesystems"][0]["target"] != str(workspace),
        "ephemeral_projection_with_margin_fits": local["free_bytes"] >= ephemeral_required + EPHEMERAL_SAFETY_MARGIN,
    }
    return {
        "created_utc": utc_now(),
        "network_volume_id": VOLUME_ID,
        "network_volume_capacity_decimal_bytes": VOLUME_CAPACITY_DECIMAL_BYTES,
        "logical_workspace": logical,
        "logical_quota_free_decimal_bytes": logical_free,
        "fuse_df_note": "Backend aggregate df values are recorded but are not treated as the 100-GB volume quota.",
        "persistent_df": persistent,
        "ephemeral_df": local,
        "projected": {
            "final_checkpoint_count": FINAL_CHECKPOINT_COUNT,
            "largest_source_checkpoint_bytes": largest,
            "persistent_checkpoint_and_artifact_bytes": persistent_required,
            "persistent_safety_margin_bytes": PERSISTENT_SAFETY_MARGIN,
            "ephemeral_checkpoint_equivalents": EPHEMERAL_CHECKPOINT_EQUIVALENTS,
            "ephemeral_required_bytes": ephemeral_required,
            "ephemeral_safety_margin_bytes": EPHEMERAL_SAFETY_MARGIN,
            "update96_and_GA_staging_location": str(ephemeral.resolve()),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def source_manifest(repo: Path) -> dict:
    rows = {}
    checks = {}
    for name, expected in EXPECTED_SOURCES.items():
        path = Path(expected["path"])
        sidecar = path.with_suffix(path.suffix + ".sha256")
        verification = path.with_suffix(path.suffix + ".verification.json")
        observed = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "sha_sidecar": sidecar.read_text().split()[0] if sidecar.is_file() else None,
            "verification_passed": json.loads(verification.read_text()).get("passed") if verification.is_file() else False,
            "commit_present": subprocess.call(
                ["git", "cat-file", "-e", expected["commit"] + "^{commit}"], cwd=repo
            ) == 0,
        }
        rows[name] = observed
        checks[f"{name}_bytes"] = observed["bytes"] == expected["bytes"]
        checks[f"{name}_sha256"] = observed["sha256"] == expected["sha256"]
        checks[f"{name}_sidecar"] = observed["sha_sidecar"] == expected["sha256"]
        checks[f"{name}_strict_reopen_sidecar"] = observed["verification_passed"]
        checks[f"{name}_commit_present"] = observed["commit_present"]
    tag_commit = run_output(
        "git", "rev-parse", EXPECTED_SOURCES["2D2E"]["tag"] + "^{commit}", cwd=repo
    )
    checks["2D2E_final_tag_exact"] = tag_commit == EXPECTED_SOURCES["2D2E"]["commit"]
    return {"created_utc": utc_now(), "sources": rows, "checks": checks, "passed": all(checks.values())}


def dataset_manifest(data_root: Path) -> dict:
    validation = data_root / "edufineweb_val_000000.npy"
    training = sorted(data_root.glob("edufineweb_train_*.npy"))
    if not validation.is_file() or not training:
        raise SystemExit("FineWeb training/validation shards are missing")
    files = []
    for index, path in enumerate([validation, *training]):
        started = time.monotonic()
        files.append(
            {
                "kind": "validation" if index == 0 else "training",
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "hash_wall_seconds": time.monotonic() - started,
            }
        )
        print(f"dataset manifest hashed {index + 1}/{len(training) + 1}: {path.name}", flush=True)
    checks = {
        "validation_SHA_exact": files[0]["sha256"] == VALIDATION_SHA256,
        "training_shards_nonempty": len(training) > 0,
        "all_shards_nonzero": all(row["bytes"] > 0 for row in files),
        "all_hashes_unique": len({row["sha256"] for row in files}) == len(files),
    }
    collection = hashlib.sha256(
        "\n".join(f"{row['sha256']}  {row['bytes']}  {row['path']}" for row in files).encode()
    ).hexdigest()
    return {
        "created_utc": utc_now(),
        "data_root": str(data_root.resolve()),
        "files": files,
        "training_shard_count": len(training),
        "collection_sha256": collection,
        "checks": checks,
        "passed": all(checks.values()),
    }


def git_manifest(repo: Path) -> dict:
    branches = (
        "experiment-2d2f-no-b2-recurrence-b3-w64",
        "experiment-2d2g-b2-full-b3-w64",
        "experiment-2d2h-no-b1-recurrence-b2-w32",
        "experiment-2d2i-b4-w128-b9-recurrent",
        "codex/parallel-2d2-master",
    )
    rows = {}
    for branch in branches:
        local = run_output("git", "rev-parse", branch, cwd=repo)
        origin = run_output("git", "rev-parse", "origin/" + branch, cwd=repo)
        rows[branch] = {"local": local, "origin": origin, "synchronized": local == origin}
    checks = {
        "all_branches_synchronized": all(row["synchronized"] for row in rows.values()),
        "repository_connectivity": subprocess.call(["git", "fsck", "--connectivity-only"], cwd=repo) == 0,
    }
    return {"created_utc": utc_now(), "repository": str(repo.resolve()), "branches": rows, "checks": checks, "passed": all(checks.values())}


def stale_legacy_execution_markers(root: Path) -> list[Path]:
    """Return unscoped markers that could be mistaken for the new run.

    Historical run directories are intentionally retained.  New orchestration
    only consumes artifacts below ``runs/<run_id>``; old top-level execution
    markers are therefore rejected rather than silently deleted or reused.
    """
    paths = []
    for pattern in (
        "lane_gpu*.error.json",
        "lane_gpu*.science_complete.json",
        "lane_gpu*.terminal.json",
        "MASTER_ALL_LANES_TERMINAL*",
        "MASTER_FINALIZATION_COMPLETE*",
        "MASTER_TERMINAL_STATUS.json",
        "MASTER_SUPERVISOR.json",
    ):
        paths.extend(sorted(root.glob(pattern)))
    return paths


def initialize_master_files(root: Path, run_id: str) -> Path:
    (root / "locks").mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint_persist.lock", "finalize.lock", "git_push.lock"):
        (root / "locks" / name).touch(exist_ok=True)
    run_root = (root / "runs" / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "run_root": str(run_root),
        "created_utc": utc_now(),
        "pod": {"id": POD_ID, "name": POD_NAME, "gpu_count": 4, "volume_id": VOLUME_ID},
        "lanes": {
            "GPU0": {"experiment": "2D2E-C1 then 2D2F", "status": "PREFLIGHT"},
            "GPU1": {"experiment": "2D2G-A then 2D2G-B", "status": "PREFLIGHT"},
            "GPU2": {"experiment": "2D2H", "status": "PREFLIGHT"},
            "GPU3": {"experiment": "2D2I", "status": "PREFLIGHT"},
        },
    }
    durable_json(root / "MASTER_STATUS.json", status)
    durable_json(run_root / "MASTER_STATUS.json", status)
    initial_heartbeat = {**status, "heartbeat_utc": utc_now(), "status": "PREFLIGHT"}
    durable_json(root / "MASTER_HEARTBEAT.json", initial_heartbeat)
    durable_json(run_root / "MASTER_HEARTBEAT.json", initial_heartbeat)
    commands = run_root / "MASTER_COMMANDS.log"
    commands.touch(exist_ok=False)
    return run_root


def preflight(args) -> dict:
    root = Path(args.master_root).resolve()
    workspace = Path(args.workspace).resolve()
    repo = Path(args.repo).resolve()
    data_root = Path(args.data_root).resolve()
    ephemeral = Path(args.ephemeral_root).resolve()
    stop = json.loads(Path(args.stop_audit).read_text())
    if args.pod_id != POD_ID or args.pod_name != POD_NAME or args.volume_id != VOLUME_ID:
        raise SystemExit("master pod/volume identity does not match preregistration")
    root.mkdir(parents=True, exist_ok=True)
    stale = stale_legacy_execution_markers(root)
    if stale:
        raise SystemExit(
            "refusing unscoped stale execution markers: "
            + ", ".join(str(path) for path in stale)
        )
    run_id = str(uuid.uuid4())
    run_root = initialize_master_files(root, run_id)
    # Invalidate any earlier passing top-level preflight before expensive
    # hashing begins.  A supervisor can therefore never launch from an older
    # run while this new preflight is still in progress or after it crashes.
    pending = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "run_id": run_id,
        "run_root": str(run_root),
        "pod": {"id": POD_ID, "name": POD_NAME, "gpu_count": 4, "volume_id": VOLUME_ID},
        "checks": {
            "hardware": False,
            "storage": False,
            "sources": False,
            "dataset": False,
            "git": False,
            "authenticated_stop": False,
        },
        "passed": False,
        "status": "PREFLIGHT_RUNNING",
        "execution_markers_are_run_scoped": True,
        "pod_stop_automated_by_supervisor": False,
    }
    durable_json(root / "MASTER_PREFLIGHT.json", pending)
    durable_json(run_root / "MASTER_PREFLIGHT.json", pending)
    hardware = hardware_manifest()
    storage = storage_manifest(workspace, ephemeral)
    sources = source_manifest(repo)
    dataset = dataset_manifest(data_root)
    git = git_manifest(repo)
    stop_checks = {
        "passed": bool(stop.get("passed")),
        "pod_id_exact": stop.get("pod_id") == POD_ID,
        "pod_name_exact": stop.get("pod_name") == POD_NAME,
        "gpu_count_four": stop.get("gpu_count") == 4,
        "volume_id_exact": stop.get("volume_id") == VOLUME_ID,
        "authenticated": bool(stop.get("authenticated")),
    }
    stop_record = {**stop, "checks": stop_checks, "passed": all(stop_checks.values())}
    durable_json(root / "hardware_manifest.json", hardware)
    durable_json(root / "storage_preflight.json", storage)
    durable_json(root / "source_checkpoint_manifest.json", sources)
    durable_json(root / "shared_dataset_manifest.json", dataset)
    durable_json(root / "git_worktree_manifest.json", git)
    durable_json(root / "AUTO_STOP_PREFLIGHT.json", stop_record)
    durable_json(run_root / "hardware_manifest.json", hardware)
    durable_json(run_root / "storage_preflight.json", storage)
    durable_json(run_root / "source_checkpoint_manifest.json", sources)
    durable_json(run_root / "shared_dataset_manifest.json", dataset)
    durable_json(run_root / "git_worktree_manifest.json", git)
    durable_json(run_root / "AUTO_STOP_PREFLIGHT.json", stop_record)
    checks = {
        "hardware": hardware["passed"],
        "storage": storage["passed"],
        "sources": sources["passed"],
        "dataset": dataset["passed"],
        "git": git["passed"],
        "authenticated_stop": stop_record["passed"],
    }
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "run_id": run_id,
        "run_root": str(run_root),
        "pod": {"id": POD_ID, "name": POD_NAME, "gpu_count": 4, "volume_id": VOLUME_ID},
        "checks": checks,
        "passed": all(checks.values()),
        "execution_markers_are_run_scoped": True,
        "pod_stop_automated_by_supervisor": False,
    }
    durable_json(root / "MASTER_PREFLIGHT.json", result)
    durable_json(run_root / "MASTER_PREFLIGHT.json", result)
    if not result["passed"]:
        raise SystemExit(f"master preflight failed: {checks}")
    print(f"PARALLEL_2D2_MASTER_PREFLIGHT_PASS run_id={run_id}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    current = subparsers.add_parser("preflight")
    current.add_argument("--master-root", required=True)
    current.add_argument("--workspace", required=True)
    current.add_argument("--repo", required=True)
    current.add_argument("--data-root", required=True)
    current.add_argument("--ephemeral-root", required=True)
    current.add_argument("--stop-audit", required=True)
    current.add_argument("--pod-id", required=True)
    current.add_argument("--pod-name", required=True)
    current.add_argument("--volume-id", required=True)
    current.set_defaults(function=preflight)
    return parser


def main():
    args = build_parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    main()
