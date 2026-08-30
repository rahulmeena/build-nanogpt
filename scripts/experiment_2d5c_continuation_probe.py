#!/usr/bin/env python3
"""Disposable fresh-process full-tensor probe for Experiment 2D5C.

This utility never writes an official checkpoint and never advances official
state.  Each invocation strictly reopens the retained disposable update-2
checkpoint, executes only disposable update 3 with the inherited training
recipe, and captures all gradient/model/optimizer tensors on CPU.  A reference
snapshot can be retained so later fresh processes emit complete pairwise
max-absolute and L2 drift evidence without retaining additional raw snapshots.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time
import uuid
from pathlib import Path


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import experiment_2d5c as driver


DISPOSABLE_CHECKPOINT_SHA256 = (
    "d04a074aedcc0647186f28da96540b7b155db54f7b2ca9ef5623ac500e3e63a2"
)
DISPOSABLE_VERIFICATION_SHA256 = (
    "09f5ed08dd07edd5feee081e9dcf89bd9456ac50ce79b31c63ebd002c7f802f2"
)
PROBE_SCHEMA = "experiment_2d5c_continuation_full_tensor_probe_v1"
OFFICIAL_OUTPUT_NAMES = ("preflight", "results", "checkpoints")
OFFICIAL_RUN_ROOT = Path("/workspace/exp2d5c_w2w2_100m")
DIAGNOSTICS_ROOT = OFFICIAL_RUN_ROOT / "diagnostics"


def atomic_torch_save(value, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        driver.fsync_path(temporary)
        os.replace(temporary, path)
        driver.fsync_path(path.parent)
    except BaseException:
        if temporary.exists():
            failed = path.with_name(
                f"FAILED_DISPOSABLE_PARTIAL_{path.name}.{os.getpid()}"
            )
            os.replace(temporary, failed)
            driver.fsync_path(path.parent)
        raise


def load_snapshot(path: Path):
    return driver.base.d0.torch_load(path.resolve(), mmap=False)


def official_state_checks(run_root: Path) -> dict:
    checks = {}
    for name in OFFICIAL_OUTPUT_NAMES:
        path = run_root / name
        checks[f"{name}_fresh"] = (
            not path.exists()
            or (path.is_dir() and not any(path.iterdir()))
        )
    return checks


def require_diagnostics_output(path: Path, label: str) -> None:
    path = path.resolve()
    diagnostics = DIAGNOSTICS_ROOT.resolve()
    if diagnostics not in path.parents:
        raise SystemExit(f"{label} must be beneath the exact diagnostics root")


def boundary_manifest(model, optimizer, loader, payload, accumulation):
    group_definitions = [
        {
            key: driver.canonical_rng_value(value)
            for key, value in sorted(group.items()) if key != "params"
        }
        for group in optimizer.param_groups
    ]
    return {
        "model_state_sha256": driver.parameter_manifest(model)["aggregate_sha256"],
        "optimizer_state_sha256": driver.optimizer_manifest(
            model, optimizer
        )["state_aggregate_sha256"],
        "optimizer_groups_sha256": driver.canonical_sha(
            driver.optimizer_group_manifest(model, optimizer)
        ),
        "optimizer_group_definitions": group_definitions,
        "loader_state": copy.deepcopy(loader.state_dict()),
        "rng_digests": driver.rng_digests(driver.base.capture_rng()),
        "next_global_batch_sha256": driver.base.next_batch_hash(
            loader, accumulation
        ),
        "next_global_batch_stream_sha256": driver.base.next_stream_hash(
            loader, accumulation
        ),
        "weight_tie_exact": (
            model.base.transformer.wte.weight is model.base.lm_head.weight
        ),
        "payload_local_updates": payload.get("local_updates"),
        "payload_global_update": payload.get("global_update"),
        "payload_cumulative_targets": payload.get("cumulative_targets"),
        "payload_parent_checkpoint_sha256": payload.get(
            "parent_checkpoint_sha256"
        ),
        "payload_architecture_fingerprint": payload.get(
            "architecture_fingerprint"
        ),
    }


def boundary_checks(boundary, payload, verification):
    payload_groups = [
        {
            key: driver.canonical_rng_value(value)
            for key, value in sorted(group.items())
        }
        for group in payload.get("optimizer_group_definitions", [])
    ]
    return {
        "model_exact": boundary["model_state_sha256"]
        == verification.get("model_state_sha256"),
        "optimizer_exact": boundary["optimizer_state_sha256"]
        == verification.get("optimizer_state_sha256"),
        "optimizer_groups_exact": boundary["optimizer_group_definitions"]
        == payload_groups,
        "loader_exact": boundary["loader_state"] == payload.get("loader_state"),
        "rng_exact": boundary["rng_digests"] == payload.get("rng_digests"),
        "batch_exact": boundary["next_global_batch_sha256"]
        == payload.get("next_global_batch_sha256"),
        "stream_exact": boundary["next_global_batch_stream_sha256"]
        == payload.get("next_global_batch_stream_sha256"),
        "weight_tie_exact": boundary["weight_tie_exact"],
        "local_update_exact": boundary["payload_local_updates"] == 2,
        "global_update_exact": boundary["payload_global_update"]
        == driver.SOURCE_UPDATES + 2,
        "cumulative_targets_exact": boundary["payload_cumulative_targets"]
        == driver.SOURCE_TARGETS + 2 * driver.base.GLOBAL_TARGETS,
        "parent_exact": boundary["payload_parent_checkpoint_sha256"]
        == driver.SOURCE_SHA256,
        "architecture_exact": boundary["payload_architecture_fingerprint"]
        == driver.ARCHITECTURE_FINGERPRINT_C,
    }


def comparison_checks(reference, row, boundary, pairwise):
    if reference is None:
        return {}
    reference_row = reference["scientific_row"]
    reference_boundary = reference["boundary"]
    return {
        "boundary_exact": boundary == reference_boundary,
        "consumed_batch_exact": row["consumed_batch_sha256"]
        == reference_row["consumed_batch_sha256"],
        "consumed_stream_exact": row["consumed_stream_sha256"]
        == reference_row["consumed_stream_sha256"],
        "pass_count_exact": row["pass_count"] == reference_row["pass_count"],
        "forward_losses_exact": row["pass_losses"]
        == reference_row["pass_losses"],
        "optimizer_step_increment_exact": row["optimizer_step_increment_exact"]
        and reference_row["optimizer_step_increment_exact"],
        "optimizer_step_summaries_exact": (
            row["optimizer_steps_before_summary"]
            == reference_row["optimizer_steps_before_summary"]
            and row["optimizer_steps_after_summary"]
            == reference_row["optimizer_steps_after_summary"]
        ),
        "post_rng_exact": driver.rng_digests(driver.base.capture_rng())
        == reference["post_rng_digests"],
        "full_tensor_structure_exact": pairwise is not None
        and pairwise["passed"],
    }


def run(args) -> None:
    run_uuid = str(uuid.uuid4())
    checkpoint = args.checkpoint.resolve()
    source = args.source_checkpoint.resolve()
    verification_path = args.verification.resolve()
    if args.official_run_root != OFFICIAL_RUN_ROOT:
        raise SystemExit("probe official run root is not exact")
    require_diagnostics_output(args.output_report, "probe report")
    if args.output_snapshot is not None:
        require_diagnostics_output(args.output_snapshot, "raw snapshot")
    if args.reference_snapshot is not None:
        require_diagnostics_output(args.reference_snapshot, "reference snapshot")
        require_diagnostics_output(args.reference_report, "reference report")
    git_identity = {
        "branch": driver.git("branch", "--show-current"),
        "head": driver.git("rev-parse", "HEAD"),
        "origin_branch": driver.git(
            "rev-parse", f"origin/{driver.BRANCH}"
        ),
        "clean": not bool(driver.git("status", "--porcelain")),
    }
    if not (
        git_identity["branch"] == driver.BRANCH
        and git_identity["clean"]
        and git_identity["head"] == git_identity["origin_branch"]
    ):
        raise SystemExit(f"probe Git implementation is not frozen: {git_identity}")
    if not args.label.startswith("DISPOSABLE_"):
        raise SystemExit("probe label must begin with DISPOSABLE_")
    if "DISPOSABLE" not in args.output_report.name:
        raise SystemExit("probe report filename must contain DISPOSABLE")
    if args.output_report.exists():
        raise SystemExit(f"refusing to overwrite probe report: {args.output_report}")
    if args.output_snapshot is not None:
        if "DISPOSABLE" not in args.output_snapshot.name:
            raise SystemExit("raw snapshot filename must contain DISPOSABLE")
        if args.output_snapshot.exists():
            raise SystemExit(
                f"refusing to overwrite raw snapshot: {args.output_snapshot}"
            )
    if args.reference_snapshot is not None and (
        "DISPOSABLE" not in args.reference_snapshot.name
        or "DISPOSABLE" not in args.reference_report.name
    ):
        raise SystemExit("reference filenames must contain DISPOSABLE")
    official_before = official_state_checks(args.official_run_root)
    if not all(official_before.values()):
        raise SystemExit(f"official state is not fresh: {official_before}")
    source_before = driver.file_identity(source)
    checkpoint_identity = driver.file_identity(checkpoint)
    verification_identity = driver.file_identity(verification_path)
    if checkpoint_identity["sha256"] != DISPOSABLE_CHECKPOINT_SHA256:
        raise SystemExit("wrong disposable update-2 checkpoint SHA-256")
    if source_before["sha256"] != driver.SOURCE_SHA256:
        raise SystemExit("wrong 2D3A source checkpoint SHA-256")
    if verification_identity["sha256"] != DISPOSABLE_VERIFICATION_SHA256:
        raise SystemExit("wrong disposable update-2 verification SHA-256")
    verification = json.loads(verification_path.read_text())
    if verification.get("sha256") != DISPOSABLE_CHECKPOINT_SHA256:
        raise SystemExit("verification does not bind the disposable checkpoint")
    device = driver.base.require_a100()
    model, optimizer, loader, payload, source_payload = driver.load_c_checkpoint(
        checkpoint, source, device, restore=True
    )
    accumulation = int(source_payload["gradient_accumulation"])
    boundary = boundary_manifest(
        model, optimizer, loader, payload, accumulation
    )
    checks = boundary_checks(boundary, payload, verification)
    if not all(checks.values()):
        raise SystemExit(f"disposable probe boundary failure: {checks}")
    started = time.time()
    row = driver.smoke_update(
        model, optimizer, loader, accumulation, 3, device
    )
    snapshot = driver.continuation_tensor_snapshot(model, optimizer)
    summary = driver.continuation_tensor_snapshot_summary(snapshot)
    del model, optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    reference_snapshot = None
    reference_report = None
    reference_snapshot_identity = None
    reference_report_identity = None
    pairwise = None
    if args.reference_snapshot is not None:
        reference_snapshot_identity = driver.file_identity(
            args.reference_snapshot
        )
        reference_report_identity = driver.file_identity(args.reference_report)
        reference_snapshot = load_snapshot(args.reference_snapshot)
        pairwise = driver.continuation_tensor_pairwise_metrics(
            reference_snapshot, snapshot
        )
        reference_report = json.loads(args.reference_report.read_text())
        if reference_report.get("raw_snapshot") != reference_snapshot_identity:
            raise SystemExit("reference report does not bind the raw snapshot")
    compare_checks = comparison_checks(
        reference_report, row, boundary, pairwise
    )
    source_after = driver.file_identity(source)
    report = {
        "schema": PROBE_SCHEMA,
        "experiment": driver.EXPERIMENT,
        "label": args.label,
        "run_uuid": run_uuid,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "git": git_identity,
        "disposable": True,
        "created_before_official_training": True,
        "official_training_updates_executed": 0,
        "official_run_root": str(args.official_run_root),
        "official_state_checks_before": official_before,
        "official_state_checks_after": {},
        "disposable_checkpoint": checkpoint_identity,
        "disposable_verification": verification_identity,
        "source_checkpoint": source_after,
        "source_unchanged": source_after == source_before,
        "probe_source": driver.file_identity(Path(__file__)),
        "driver_source": driver.file_identity(Path(driver.__file__)),
        "pod": {
            "id": driver.POD_ID,
            "name": driver.POD_NAME,
            "volume_id": driver.VOLUME_ID,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_count": torch.cuda.device_count(),
        },
        "runtime": driver.continuation_runtime_identity(),
        "boundary": boundary,
        "boundary_checks": checks,
        "scientific_row": row,
        "post_rng_digests": driver.rng_digests(driver.base.capture_rng()),
        "snapshot_summary": summary,
        "pairwise_against_reference": pairwise,
        "reference_snapshot": reference_snapshot_identity,
        "reference_report": reference_report_identity,
        "comparison_checks": compare_checks,
        "wall_seconds": time.time() - started,
        "passed": False,
    }
    if args.output_snapshot is not None:
        atomic_torch_save(snapshot, args.output_snapshot)
        report["raw_snapshot"] = driver.file_identity(args.output_snapshot)
    report["official_state_checks_after"] = official_state_checks(
        args.official_run_root
    )
    report["passed"] = (
        all(checks.values())
        and all(official_before.values())
        and all(report["official_state_checks_after"].values())
        and source_after == source_before
        and (not compare_checks or all(compare_checks.values()))
    )
    # Recheck after the first durable report write. Outputs are constrained to
    # diagnostics, so neither write is permitted to alter protected state.
    driver.durable_json(args.output_report, report)
    report["official_state_checks_after"] = official_state_checks(
        args.official_run_root
    )
    report["passed"] = report["passed"] and all(
        report["official_state_checks_after"].values()
    )
    driver.durable_json(args.output_report, report)
    if not report["passed"]:
        raise SystemExit(f"disposable continuation probe failed: {report}")
    print(
        f"EXPERIMENT_2D5C_DISPOSABLE_TENSOR_PROBE_PASS {args.label}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-snapshot", type=Path)
    parser.add_argument("--reference-snapshot", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument(
        "--official-run-root", type=Path,
        default=Path("/workspace/exp2d5c_w2w2_100m"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "checkpoint", "source_checkpoint", "verification", "output_report",
        "output_snapshot", "reference_snapshot", "reference_report",
        "official_run_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if (args.reference_snapshot is None) != (args.reference_report is None):
        raise SystemExit(
            "--reference-snapshot and --reference-report must be supplied together"
        )
    if args.output_snapshot is None and args.reference_snapshot is None:
        raise SystemExit(
            "reference run requires --output-snapshot; comparison requires reference"
        )
    run(args)


if __name__ == "__main__":
    main()
