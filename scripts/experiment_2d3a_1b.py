#!/usr/bin/env python3
"""Experiment 2D3A Stage D: exact M500 -> one-billion continuation.

Training semantics are inherited unchanged from the sealed 2D3A lineage.
This module adds only continuation bookkeeping, frozen evaluations,
checkpointing, reporting, and terminal evidence synthesis.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import experiment_2d3a as base
import experiment_2d3a_250m as stage_b
import experiment_2d3a_500m as stage_c


EXPERIMENT = "2D3A-1B"
PROTOCOL = "alternating_integration_recurrent_pyramid_500m_to_1b_v1"
BRANCH = "experiment-2d3a-alternating-integration-pyramid-1b"
SOURCE_COMMIT = "a81a7984468cb89bbf6b6e633e6fa3670068041c"
SOURCE_TAG = "experiment-2d3a-alternating-integration-pyramid-500m-final"
SOURCE_SHA256 = "a0cdf438ae44b76fcee1badc67faa1025203914af9645a86043adf1d332ca619"
SOURCE_NEXT_BATCH = "32aa16f47b5cf39e1071e6037f0363e9c8f098aad649e54b5de3617a36be848b"
SOURCE_NEXT_STREAM = "b5f4072ff5add3455efb60b9e4d1f4c4514dbf124f2892566b36800f4abefe1d"
SOURCE_UPDATE = 954
SOURCE_TARGETS = 500_170_752
FINAL_UPDATE = 1908
FINAL_TARGETS = 1_000_341_504
RESTART_UPDATE = 1431
MILESTONES = {1192: "625m", 1431: "750m", 1669: "875m", 1908: "1b"}
MILESTONE_TARGETS = {
    1192: 624_951_296,
    1431: 750_256_128,
    1669: 875_036_672,
    1908: FINAL_TARGETS,
}
RECOVERY_UPDATES = (1050, 1146, 1288, 1384, 1527, 1623, 1765, 1861)
CANONICAL_SHA = base.CANONICAL_COLLECTION_SHA
MATURATION_CORE_SHA = stage_b.MATURATION_CORE_SHA
MATCHED_BOOTSTRAP_SEED = 20_260_829
FRESH_BOOTSTRAP_SEED = 20_260_829
LINKS = stage_b.LINKS
CONTROL_NAMES = stage_b.CONTROL_NAMES
ATTENTION_FILES = stage_b.ATTENTION_FILES
GRADIENT_FILES = stage_b.GRADIENT_FILES
AGES = ("100m", "150m", "200m", "250m", "300m", "375m", "500m",
        "625m", "750m", "875m", "1b")
TARGETS_BY_AGE = {
    "100m": 100_139_008, "150m": 149_946_368, "200m": 199_753_728,
    "250m": 250_085_376, "300m": 299_892_736, "375m": 374_865_920,
    "500m": SOURCE_TARGETS, "625m": 624_951_296, "750m": 750_256_128,
    "875m": 875_036_672, "1b": FINAL_TARGETS,
}
FRESH_CONTROLS = (
    "all_real", "b3_off", "b3_shuffled", "b5_off", "b5_shuffled",
    "b6_off", "b6_shuffled", "b3_b6_off", "new_links_off",
)
REQUIRED_ARTIFACTS = (
    "EXPERIMENT_2D3A_1B_FINAL_REPORT.md", "FINAL_AUDIT.json", "result_summary.json",
    "source_500m_manifest.json", "continuation_semantic_diff_500m_to_1b.json",
    "optimizer_resume_500m_manifest.json", "scheduler_resume_500m_manifest.json",
    "data_resume_500m_manifest.json", "training_metrics_500m_to_1b.jsonl",
    "maturation_table_100m_to_1b.json", "maturation_interval_deltas.json",
    "milestone_625m_validation.json", "milestone_750m_validation.json",
    "milestone_875m_validation.json", "milestone_1b_validation.json",
    "true_incremental_625m.json", "true_incremental_750m.json",
    "true_incremental_875m.json", "true_incremental_1b.json",
    "paired_maturation_controls.json", "gate_maturation.json",
    *ATTENTION_FILES.values(), *GRADIENT_FILES.values(),
    "boundary_memory_maturation.json", "b6_representation_maturation.json",
    "position_bin_maturation.json", "factorial_maturation.json",
    "m1000_factorial.json", "m1000_factorial_interactions.json",
    "best_subset_maturation.json", "m1000_matched_large.json",
    "m1000_matched_large_bootstrap.json", "m1000_fresh_subset_manifest.json",
    "m1000_disjointness_audit.json", "m1000_fresh_final_confirmation.json",
    "m1000_fresh_final_bootstrap.json", "stability_8pass_maturation.json",
    "stability_16pass_terminal.json", "incremental_cache_audit.json",
    "memory_accounting.json", "performance.json", "checkpoint_manifest.json",
    "ONE_BILLION_BASELINE_MANIFEST.json", "storage_cleanup_manifest.json",
    "commands_and_runtime.json", "HEARTBEAT.json", "UNATTENDED_FINAL_HANDOFF.md",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def durable_json(path, value):
    base.durable_json(Path(path), value)


def durable_text(path, value):
    base.durable_text(Path(path), value)


def append_jsonl(path, value):
    base.append_jsonl(Path(path), value)


def sha256(path):
    return base.file_sha256(Path(path))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=base.REPO_ROOT, text=True).strip()


def require_branch(clean=False):
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit(f"must run on {BRANCH}")
    if clean and git("status", "--porcelain"):
        raise SystemExit("Git worktree must be clean")


def optimizer_steps(optimizer):
    values = []
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            values.append(int(step.item() if torch.is_tensor(step) else step))
    return values


def optimizer_steps_exact(values, cumulative_update):
    return collections.Counter(values) == collections.Counter({
        int(cumulative_update): 3,
        int(cumulative_update) + 478: 149,
    })


def heartbeat(output, update, row, checkpoint=None, status="training"):
    durable_json(Path(output) / "HEARTBEAT.json", {
        "experiment": EXPERIMENT,
        "status": status,
        "cumulative_update": int(update),
        "cumulative_targets": int(update) * base.GLOBAL_TARGETS,
        "latest_metrics": row,
        "checkpoint": checkpoint,
        "pid": os.getpid(),
        "updated_at_unix": time.time(),
    })


def record_command(output, kind):
    path = Path(output) / "commands_and_runtime.json"
    value = read_json(path) if path.exists() else {"commands": []}
    value.setdefault("commands", []).append({
        "kind": kind,
        "command": " ".join(sys.argv),
        "pid": os.getpid(),
        "started_at_unix": time.time(),
    })
    durable_json(path, value)


def output_args(parser):
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--pod-name", required=True)


def checkpoint_name(update):
    if update not in MILESTONE_TARGETS:
        raise ValueError(f"update {update} is not a scientific milestone")
    return f"scientific_cumulative_{MILESTONE_TARGETS[update]:012d}.pt"


def continuation_metadata(args, payload, accumulation):
    old = payload["metadata"]
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "branch": BRANCH,
        "source_500m_checkpoint_path": str(Path(args.source_checkpoint).resolve()),
        "source_500m_checkpoint_sha256": SOURCE_SHA256,
        "source_500m_commit": SOURCE_COMMIT,
        "source_500m_tag": SOURCE_TAG,
        "git_implementation_commit": git("rev-parse", "HEAD"),
        "targets_per_update": base.GLOBAL_TARGETS,
        "micro_batch": int(payload["loader_state"]["batch_size"]),
        "gradient_accumulation": int(accumulation),
        "sequence_length": base.T,
        "two_pass_weights": [.25, .75],
        "three_pass_weights": [.2, .4, .4],
        "pass3_every_cumulative_update": 32,
        "mandatory_restart_cumulative_update": RESTART_UPDATE,
        "data_manifest": copy.deepcopy(old["data_manifest"]),
        "canonical_validation_manifest": copy.deepcopy(old["canonical_validation_manifest"]),
        "maturation_core_subset_manifest": copy.deepcopy(old["maturation_core_subset_manifest"]),
        "hardware_metadata": copy.deepcopy(old["hardware_metadata"]),
        "precision_settings": copy.deepcopy(old["precision_settings"]),
        "semantic_changes": [],
        "architecture_changes": [],
        "optimizer_resets": [],
        "scheduler_resets": [],
        "data_stream_restarts": [],
    }


def continuation_payload(source, model, optimizer, loader, completed, accumulation, metadata):
    payload = {key: value for key, value in source.items() if key not in {
        "model", "optimizer", "loader_state", "loader_states", "rng_state", "metadata",
    }}
    payload.update({
        "schema_version": base.SCHEMA,
        "schema": base.SCHEMA,
        "experiment_name": EXPERIMENT,
        "architecture_version": base.ARCHITECTURE_VERSION,
        "parent_experiment": "2D3A-500M",
        "parent_checkpoint_path": metadata["source_500m_checkpoint_path"],
        "parent_checkpoint_sha256": SOURCE_SHA256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": copy.deepcopy(source["scheduler"]),
        "d3a_completed_updates": int(completed),
        "d3a_processed_targets": int(completed) * base.GLOBAL_TARGETS,
        "loader_state": loader.state_dict(),
        "loader_states": [loader.state_dict()],
        "rng_state": base.capture_rng(),
        "targets_per_update": base.GLOBAL_TARGETS,
        "gradient_accumulation": int(accumulation),
        "next_global_batch_sha256": base.next_batch_hash(loader, accumulation),
        "next_global_batch_stream_sha256": base.next_stream_hash(loader, accumulation),
        "raw_gate_values": base.gate_values(model),
        "optimizer_group_definitions": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
        "current_lr_per_group": {group["name"]: group["lr"] for group in optimizer.param_groups},
        "git_implementation_commit": metadata["git_implementation_commit"],
        "data_manifest": metadata["data_manifest"],
        "canonical_validation_manifest": metadata["canonical_validation_manifest"],
        "true_self_maturation_core_subset_manifest": metadata["maturation_core_subset_manifest"],
        "hardware_metadata": metadata["hardware_metadata"],
        "precision_settings": metadata["precision_settings"],
        "metadata": metadata,
        "continuation_source_update": SOURCE_UPDATE,
        "continuation_source_targets": SOURCE_TARGETS,
        "saved_process_id": os.getpid(),
        "saved_at_unix": time.time(),
    })
    return payload


def strict_reopen(path, completed, metadata, device):
    model, optimizer, loader, payload = base.load_d3a_checkpoint(path, device, restore=False)
    accumulation = int(payload["gradient_accumulation"])
    checks = {
        "schema": payload.get("schema") == base.SCHEMA,
        "architecture": payload.get("architecture_version") == base.ARCHITECTURE_VERSION,
        "updates": payload.get("d3a_completed_updates") == completed,
        "targets": payload.get("d3a_processed_targets") == completed * base.GLOBAL_TARGETS,
        "metadata": payload.get("metadata") == metadata,
        "parameters": sum(parameter.numel() for parameter in model.parameters()) == base.MODEL_PARAMETERS,
        "model_finite": base.model_finite(model),
        "optimizer_finite": base.optimizer_finite(optimizer),
        "optimizer_steps": optimizer_steps_exact(optimizer_steps(optimizer), completed),
        "next_batch": base.next_batch_hash(loader, accumulation) == payload["next_global_batch_sha256"],
        "next_stream": base.next_stream_hash(loader, accumulation) == payload["next_global_batch_stream_sha256"],
        "rng_complete": set(payload.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "optimizer_present": bool(payload.get("optimizer")),
        "scheduler_present": "scheduler" in payload,
        "source_lineage": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256,
    }
    del model, optimizer, loader, payload
    torch.cuda.empty_cache()
    return {"checks": checks, "passed": all(checks.values())}


def save_checkpoint(path, source, model, optimizer, loader, completed, accumulation, metadata, device):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = continuation_payload(source, model, optimizer, loader, completed, accumulation, metadata)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    digest = sha256(path)
    audit = strict_reopen(path, completed, metadata, device)
    if not audit["passed"]:
        raise SystemExit(f"strict checkpoint reopen failed: {audit}")
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), audit)
    return {
        "checkpoint": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "next_global_batch_stream_sha256": payload["next_global_batch_stream_sha256"],
        "strict_reopen": audit,
    }


def save_scientific(args, source, model, optimizer, loader, update, accumulation, metadata, device):
    path = Path(args.scientific_checkpoint_dir) / checkpoint_name(update)
    verification = save_checkpoint(path, source, model, optimizer, loader, update,
                                   accumulation, metadata, device)
    verification["persistent"] = {
        "checkpoint": str(path.resolve()),
        "sha256": verification["sha256"],
        "bytes": verification["bytes"],
        "passed": True,
        "saved_atomically_on_persistent_volume": True,
    }
    manifest_path = Path(args.output_dir) / "checkpoint_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest[str(update)] = verification
    durable_json(manifest_path, manifest)
    return verification


def save_recovery(args, source, model, optimizer, loader, update, accumulation, metadata, device):
    path = Path(args.recovery_dir) / "rotating_recovery.pt"
    verification = save_checkpoint(path, source, model, optimizer, loader, update,
                                   accumulation, metadata, device)
    manifest_path = Path(args.output_dir) / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    manifest["recovery"] = {**verification, "rotation_update": update}
    durable_json(manifest_path, manifest)
    return verification


def validate_source(model, optimizer, loader, payload, args):
    accumulation = int(payload["gradient_accumulation"])
    metadata = payload["metadata"]
    return {
        "checkpoint_sha256": sha256(args.source_checkpoint) == SOURCE_SHA256,
        "schema": payload.get("schema") == base.SCHEMA,
        "updates": payload.get("d3a_completed_updates") == SOURCE_UPDATE,
        "targets": payload.get("d3a_processed_targets") == SOURCE_TARGETS,
        "parameters": sum(parameter.numel() for parameter in model.parameters()) == base.MODEL_PARAMETERS,
        "architecture": payload.get("architecture_version") == base.ARCHITECTURE_VERSION,
        "next_batch": base.next_batch_hash(loader, accumulation) == SOURCE_NEXT_BATCH,
        "next_stream": base.next_stream_hash(loader, accumulation) == SOURCE_NEXT_STREAM,
        "stored_next_batch": payload.get("next_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "stored_next_stream": payload.get("next_global_batch_stream_sha256") == SOURCE_NEXT_STREAM,
        "model_finite": base.model_finite(model),
        "optimizer_finite": base.optimizer_finite(optimizer),
        "optimizer_steps_exact": optimizer_steps_exact(optimizer_steps(optimizer), SOURCE_UPDATE),
        "optimizer_state_complete": len(optimizer.state) == sum(
            1 for parameter in model.parameters() if parameter.requires_grad
        ),
        "scheduler_present": "scheduler" in payload,
        "rng_complete": set(payload.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "maturation_core_geometry": metadata["maturation_core_subset_manifest"] == {
            "batches": 4, "batch_size": 64, "sequence_length": 1024,
            "targets_per_control": 262144,
            "source": "first four canonical validation batches",
        },
        "canonical_sha": metadata["canonical_validation_manifest"]["collection_sha256"] == CANONICAL_SHA,
    }


def m500_regression(model, args):
    frozen = Path(args.source_results)
    val = base.validation_path(Path(args.data_root))
    names = ["all_real", "b3_off", "b5_off", "b6_off"]
    expected_parallel = read_json(frozen / "milestone_500m_validation.json")
    parallel = base.evaluate_parallel(model, val, names)
    parallel_delta = {
        name: parallel["controls"][name]["validation_loss"]
        - expected_parallel["controls"][name]["validation_loss"]
        for name in names
    }
    expected_incremental = read_json(frozen / "true_incremental_500m.json")
    loader = base.d1.ExplicitShardLoader([val], base.VALIDATION_B, base.T)
    cpu_x, cpu_y = loader.next_batch()
    device = base.model_device(model)
    x, y = cpu_x.to(device), cpu_y.to(device)
    permutation = torch.arange(base.VALIDATION_B, device=device).roll(1)
    incremental = {}
    with torch.no_grad():
        for name in names:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                row = base.incremental_control(model, x, y, name, permutation)
            expected = expected_incremental["controls"][name]["per_sequence_losses"][:base.VALIDATION_B]
            deltas = [float(left) - float(right)
                      for left, right in zip(row["per_sequence_losses"], expected)]
            incremental[name] = {
                "max_abs_per_sequence_ce_delta": max(abs(value) for value in deltas),
                "exact": all(value == 0.0 for value in deltas),
            }
    del x, y, cpu_x, cpu_y
    torch.cuda.empty_cache()
    return {
        "parallel": {
            "controls": parallel["controls"],
            "loss_deltas": parallel_delta,
            "exact": all(value == 0.0 for value in parallel_delta.values()),
        },
        "true_incremental_short": {
            "controls": incremental,
            "exact": all(row["exact"] for row in incremental.values()),
        },
    }


def run_preflight(args):
    require_branch(clean=True)
    device = base.require_a100()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, optimizer, loader, payload = base.load_d3a_checkpoint(
        args.source_checkpoint, device, restore=False
    )
    checks = validate_source(model, optimizer, loader, payload, args)
    source_reopen = stage_c.strict_reopen(
        args.source_checkpoint, SOURCE_UPDATE, payload["metadata"], device
    )
    optimizer_manifest = stage_b.optimizer_tensor_manifest(optimizer)
    architecture = base.architecture_manifest()
    causality = base.causality_audit(model)
    isolation = base.future_and_row_isolation(model)
    cache_smoke = base.incremental_smoke(model)
    regressions = m500_regression(model, args)
    persistent_free = shutil.disk_usage(Path(args.source_checkpoint).parent).free
    ephemeral_free = shutil.disk_usage(output).free
    checks.update({
        "branch": git("branch", "--show-current") == BRANCH,
        "source_commit_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
            cwd=base.REPO_ROOT,
        ).returncode == 0,
        "tag_exact": git("rev-parse", f"{SOURCE_TAG}^{{}}") == SOURCE_COMMIT,
        "stop_capability": bool(args.stop_capability_verified),
        "storage_inventory_verified": bool(args.storage_inventory_verified),
        "network_volume_free_ge_10gb": (
            args.network_volume_size_gb * 1_000_000_000 - args.network_volume_used_bytes
            >= 10_000_000_000
        ),
        "persistent_free_ge_10gb": persistent_free >= 10 * 1024**3,
        "ephemeral_free_ge_2gb": ephemeral_free >= 2 * 1024**3,
        "parallel_regression_exact": regressions["parallel"]["exact"],
        "incremental_regression_exact": regressions["true_incremental_short"]["exact"],
        "architecture_manifest_exact": architecture["parameter_count"] == base.MODEL_PARAMETERS,
        "causality": causality["passed"],
        "row_isolation": isolation["passed"],
        "incremental_cache_smoke": cache_smoke["passed"],
        "source_strict_reopen": source_reopen["passed"],
        "one_a100_sxm4_80gb": (
            torch.cuda.device_count() == 1
            and torch.cuda.get_device_name(device) == "NVIDIA A100-SXM4-80GB"
        ),
    })
    lrs = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
    source = {
        "checkpoint": str(Path(args.source_checkpoint).resolve()),
        "sha256": sha256(args.source_checkpoint),
        "commit": SOURCE_COMMIT,
        "tag": SOURCE_TAG,
        "updates": SOURCE_UPDATE,
        "targets": SOURCE_TARGETS,
        "next_batch_sha256": SOURCE_NEXT_BATCH,
        "next_stream_sha256": SOURCE_NEXT_STREAM,
        "parameter_count": base.MODEL_PARAMETERS,
        "gates": base.gate_values(model),
        "checks": checks,
    }
    semantic = {
        "architecture_changes": 0,
        "window_changes": 0,
        "recurrent_source_changes": 0,
        "recurrent_lag_changes": 0,
        "parameter_additions": 0,
        "parameter_deletions": 0,
        "optimizer_group_changes": 0,
        "scheduler_semantic_changes": 0,
        "objective_changes": 0,
        "pass_schedule_changes": 0,
        "temporal_detach_changes": 0,
        "incremental_cache_semantic_changes": 0,
        "training_precision_semantic_changes": 0,
        "permissible_additions": [
            "1B milestone bookkeeping", "continuation reports",
            "additional frozen diagnostics", "additional evaluation subsets",
            "checkpoint naming", "non-semantic infrastructure fixes",
        ],
        "semantic_diff_zero": True,
    }
    durable_json(output / "source_500m_manifest.json", source)
    durable_json(output / "continuation_semantic_diff_500m_to_1b.json", semantic)
    durable_json(output / "optimizer_resume_500m_manifest.json", {
        "source": optimizer_manifest,
        "step_values": optimizer_steps(optimizer),
        "reset": False,
        "state_loaded_strictly": True,
        "group_lrs": lrs,
        "update_955_will_use": lrs,
        "betas": [list(group["betas"]) for group in optimizer.param_groups],
        "eps": [group["eps"] for group in optimizer.param_groups],
        "weight_decay": [group["weight_decay"] for group in optimizer.param_groups],
        "gradient_clip": base.GRAD_CLIP,
    })
    durable_json(output / "scheduler_resume_500m_manifest.json", {
        "stored_scheduler_state": payload.get("scheduler"),
        "scheduler_step": SOURCE_UPDATE,
        "reset": False,
        "warmup_restarted": False,
        "cadence_uses_cumulative_update": True,
        "lr_per_parameter_group": lrs,
        "expected_update_955_lr": lrs,
    })
    durable_json(output / "data_resume_500m_manifest.json", {
        "source_loader_state": payload["loader_state"],
        "expected_update_955_batch_sha256": SOURCE_NEXT_BATCH,
        "expected_update_955_stream_sha256": SOURCE_NEXT_STREAM,
        "reset": False,
    })
    durable_json(output / "pre_resume_regression.json", regressions)
    durable_json(output / "pre_resume_architecture_manifest.json", architecture)
    durable_json(output / "pre_resume_causality.json", causality)
    durable_json(output / "pre_resume_row_isolation.json", isolation)
    durable_json(output / "pre_resume_incremental_cache_smoke.json", cache_smoke)
    durable_json(output / "source_500m_strict_reopen.json", source_reopen)
    audit = {
        "experiment": EXPERIMENT,
        "checks": checks,
        "authorized": all(checks.values()),
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "gpu_count": torch.cuda.device_count(),
        },
        "storage": {
            "persistent_free_bytes": persistent_free,
            "ephemeral_free_bytes": ephemeral_free,
            "network_volume_size_gb": args.network_volume_size_gb,
            "network_volume_used_bytes": args.network_volume_used_bytes,
        },
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "stop_command": f"runpodctl pod stop {args.pod_id} -o json",
    }
    durable_json(output / "preflight_audit.json", audit)
    durable_json(output / "checkpoint_manifest.json", {str(SOURCE_UPDATE): {
        "checkpoint": str(Path(args.source_checkpoint).resolve()),
        "sha256": SOURCE_SHA256,
        "next_global_batch_sha256": SOURCE_NEXT_BATCH,
        "next_global_batch_stream_sha256": SOURCE_NEXT_STREAM,
        "source_500m": True,
    }})
    durable_json(output / "commands_and_runtime.json", {
        "started_at_unix": time.time(),
        "commands": [{"kind": "preflight", "command": " ".join(sys.argv), "pid": os.getpid()}],
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "preflight_complete": True,
    })
    if not audit["authorized"]:
        raise SystemExit(f"preflight failed: {checks}")
    print("EXPERIMENT_2D3A_1B_PREFLIGHT_PASS", flush=True)


def merge_keyed(path, key, value):
    payload = read_json(path) if Path(path).exists() else {}
    payload[str(key)] = value
    durable_json(path, payload)


def factorial_from_losses(loss):
    mean = lambda keys: float(np.mean([loss[key] for key in keys]))
    main = {
        "B3": mean(["000", "001", "010", "011"]) - mean(["100", "101", "110", "111"]),
        "B5": mean(["000", "001", "100", "101"]) - mean(["010", "011", "110", "111"]),
        "B6": mean(["000", "010", "100", "110"]) - mean(["001", "011", "101", "111"]),
    }
    pair = {
        "B3xB5": .5 * ((loss["100"] + loss["010"] - loss["000"] - loss["110"])
                        + (loss["101"] + loss["011"] - loss["001"] - loss["111"])),
        "B3xB6": .5 * ((loss["100"] + loss["001"] - loss["000"] - loss["101"])
                        + (loss["110"] + loss["011"] - loss["010"] - loss["111"])),
        "B5xB6": .5 * ((loss["010"] + loss["001"] - loss["000"] - loss["011"])
                        + (loss["110"] + loss["101"] - loss["100"] - loss["111"])),
    }
    third = -(loss["111"] - loss["110"] - loss["101"] - loss["011"]
              + loss["100"] + loss["010"] + loss["001"] - loss["000"])
    return main, pair, third


def run_factorial(model, val_path, output, age):
    path = Path(output) / f"factorial_{age}.json"
    if path.exists():
        return read_json(path)
    incremental = read_json(Path(output) / f"true_incremental_{age}.json")
    extras = stage_c.evaluate_incremental_subset(
        model, val_path, ["b3_b5_off", "b3_b6_off", "b5_b6_off"], 0, 4
    )
    if extras["subset_sha256"] != MATURATION_CORE_SHA:
        raise SystemExit(f"{age} factorial maturation-core SHA mismatch")
    controls = incremental["controls"]
    loss = {
        "111": controls["all_real"]["validation_loss"],
        "011": controls["b3_off"]["validation_loss"],
        "101": controls["b5_off"]["validation_loss"],
        "110": controls["b6_off"]["validation_loss"],
        "001": extras["controls"]["b3_b5_off"]["validation_loss"],
        "010": extras["controls"]["b3_b6_off"]["validation_loss"],
        "100": extras["controls"]["b5_b6_off"]["validation_loss"],
        "000": controls["new_links_off"]["validation_loss"],
    }
    mapping = {
        "111": "all_real", "011": "b3_off", "101": "b5_off", "110": "b6_off",
        "001": "b3_b5_off", "010": "b3_b6_off", "100": "b5_b6_off",
        "000": "new_links_off",
    }
    main, pair, third = factorial_from_losses(loss)
    simple = incremental["combined_new_link_gain"] - sum(
        incremental[f"true_{link}_gain"] for link in ("b3", "b5", "b6")
    )
    best = min(loss, key=loss.get)
    result = {
        "age": age,
        "signed_convention": (
            "positive effect means lower CE from enabling recurrence; "
            "positive interaction means super-additive CE reduction"
        ),
        "subset_sha256": extras["subset_sha256"],
        "targets_per_condition": extras["targets_per_control"],
        "conditions": {
            bits: {"control": mapping[bits], "validation_loss": loss[bits]}
            for bits in mapping
        },
        "factorial_main_effects": main,
        "pairwise_interactions": pair,
        "three_way_interaction": third,
        "historical_combined_minus_leave_one_out_marginals": simple,
        "best_bits_B3_B5_B6": best,
        "best_control": mapping[best],
        "best_ce": loss[best],
        "performance": extras["performance"],
    }
    durable_json(path, result)
    return result


def stability_npass(model, val_path, passes):
    model.eval()
    device = base.model_device(model)
    loader = base.d1.ExplicitShardLoader([val_path], 2, base.T)
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    sources = {"h12": None, "h10": None, "h8": None, "h7": None}
    rows = []
    with torch.no_grad():
        for index in range(1, passes + 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                row = model.forward_pass(
                    x, targets=y,
                    b1_recurrent_source=sources["h12"],
                    b3_recurrent_source=sources["h10"],
                    b5_recurrent_source=sources["h8"],
                    b6_recurrent_source=sources["h7"],
                    return_diagnostics=True,
                )
            current = {
                "pass": index,
                "ce": row["loss"].item(),
                **{
                    f"{key}_rms": row[key].float().square().mean().sqrt().item()
                    for key in ("h12", "h10", "h8", "h7")
                },
            }
            for link, block in base.GATE_BLOCKS.items():
                diagnostic = row["diagnostics"].get(block)
                current[f"{link}_recurrent_output_rms"] = (
                    0.0 if diagnostic is None else diagnostic["recurrent_output_rms"].item()
                )
            current["finite"] = all(
                math.isfinite(value)
                for key, value in current.items()
                if key not in ("pass", "finite")
            )
            rows.append(current)
            sources = {key: row[key] for key in sources}
    return {
        "passes": rows,
        "passed": all(row["finite"] for row in rows),
        "first_divergence_pass": next(
            (row["pass"] for row in rows if not row["finite"]), None
        ),
    }


def update_boundary_memory(output, age):
    boundaries = {"b3": "32-63", "b5": "64-127", "b6": "512-639"}
    result = {}
    for link, bin_name in boundaries.items():
        attention = read_json(Path(output) / ATTENTION_FILES[link])[age]
        result[link] = {
            "bin": bin_name,
            **attention["recurrent"]["bins"][bin_name],
        }
    merge_keyed(Path(output) / "boundary_memory_maturation.json", age, result)


def milestone_complete(output, age):
    required = [
        f"milestone_{age}_validation.json",
        f"true_incremental_{age}.json",
        f"factorial_{age}.json",
    ]
    if age == "1b":
        required.append("stability_16pass_terminal.json")
    return all((Path(output) / name).exists() for name in required)


def run_milestone(args, model, update):
    output = Path(args.output_dir)
    age = MILESTONES[update]
    stage_b.EXPERIMENT = EXPERIMENT
    stage_b.MILESTONES = MILESTONES
    stage_b.MILESTONE_TARGETS = MILESTONE_TARGETS
    stage_b.heartbeat = heartbeat
    parallel, incremental = stage_b.run_milestone(args, model, update)
    val = base.validation_path(Path(args.data_root))
    run_factorial(model, val, output, age)
    update_boundary_memory(output, age)
    if age == "1b" and not (output / "stability_16pass_terminal.json").exists():
        terminal = stability_npass(model, val, 16)
        durable_json(output / "stability_16pass_terminal.json", terminal)
    heartbeat(output, update, {"milestone": age, "diagnostics_complete": True},
              status="milestone_complete")
    return parallel, incremental


def allowed_segment(start, end):
    if start == end == FINAL_UPDATE:
        return True
    if start >= end:
        return False
    allowed_starts = {SOURCE_UPDATE, *MILESTONES, *RECOVERY_UPDATES}
    if start not in allowed_starts:
        return False
    if start < RESTART_UPDATE:
        return end == RESTART_UPDATE
    return end == FINAL_UPDATE


def run_train(args):
    output = Path(args.output_dir)
    require_branch()
    device = base.require_a100()
    if not read_json(output / "preflight_audit.json").get("authorized"):
        raise SystemExit("preflight did not authorize result training")
    record_command(output, "train")
    resume = args.resume_checkpoint or args.source_checkpoint
    model, optimizer, loader, loaded = base.load_d3a_checkpoint(resume, device, restore=True)
    start = int(loaded["d3a_completed_updates"])
    accumulation = int(loaded["gradient_accumulation"])
    if args.resume_checkpoint:
        parent_source = base.d0.torch_load(Path(args.source_checkpoint), mmap=False)
        restart = {
            "loaded_update": start,
            "saved_process_id": loaded.get("saved_process_id"),
            "resumed_process_id": os.getpid(),
            "fresh_process": loaded.get("saved_process_id") != os.getpid(),
            "next_batch_sha256": base.next_batch_hash(loader, accumulation),
            "expected_next_batch_sha256": loaded["next_global_batch_sha256"],
            "next_stream_sha256": base.next_stream_hash(loader, accumulation),
            "expected_next_stream_sha256": loaded["next_global_batch_stream_sha256"],
        }
        restart["passed"] = (
            restart["fresh_process"]
            and restart["next_batch_sha256"] == restart["expected_next_batch_sha256"]
            and restart["next_stream_sha256"] == restart["expected_next_stream_sha256"]
        )
        if start == RESTART_UPDATE:
            restart["required_update"] = RESTART_UPDATE
            durable_json(output / "mandatory_fresh_process_restart_update_1431.json", restart)
        else:
            restart["reason"] = "fresh-process recovery from strict checkpoint"
            durable_json(output / f"scientific_recovery_update_{start}.json", restart)
        if not restart["passed"]:
            raise SystemExit(f"fresh-process restart failed: {restart}")
        metadata = loaded["metadata"]
    else:
        parent_source = loaded
        source_checks = validate_source(model, optimizer, loader, loaded, args)
        if not all(source_checks.values()):
            raise SystemExit(f"source checks failed at train start: {source_checks}")
        metadata = continuation_metadata(args, loaded, accumulation)
    end = int(args.end_update)
    if not allowed_segment(start, end):
        raise SystemExit(f"unauthorized segment {start}->{end}")
    if start in MILESTONES and not milestone_complete(output, MILESTONES[start]):
        run_milestone(args, model, start)
    for update in range(start + 1, end + 1):
        consumed_batch = base.next_batch_hash(loader, accumulation) if update in (955, 1432) else None
        consumed_stream = base.next_stream_hash(loader, accumulation) if update in (955, 1432) else None
        row = base.train_update(model, optimizer, loader, accumulation, update, device)
        row["cumulative_update"] = update
        row["cumulative_targets"] = update * base.GLOBAL_TARGETS
        row["optimizer_lrs"] = {
            group["name"]: float(group["lr"]) for group in optimizer.param_groups
        }
        if consumed_batch:
            row["consumed_global_batch_sha256"] = consumed_batch
            row["consumed_global_stream_sha256"] = consumed_stream
        append_jsonl(output / "training_metrics_500m_to_1b.jsonl", row)
        heartbeat(output, update, row)
        if update in RECOVERY_UPDATES:
            recovery = save_recovery(
                args, parent_source, model, optimizer, loader, update,
                accumulation, metadata, device,
            )
            heartbeat(output, update, row, recovery["checkpoint"], "recovery_verified")
        if update in MILESTONES:
            verification = save_scientific(
                args, parent_source, model, optimizer, loader, update,
                accumulation, metadata, device,
            )
            heartbeat(
                output, update, row, verification["persistent"]["checkpoint"],
                "checkpoint_verified",
            )
            run_milestone(args, model, update)
    print(f"EXPERIMENT_2D3A_1B_SEGMENT_COMPLETE {start}->{end}", flush=True)


def paired_metrics(evaluated, links=("b3", "b5", "b6")):
    controls = evaluated["controls"]
    real = controls["all_real"]
    result = {}
    for link in links:
        off = controls[f"{link}_off"]
        shuffled = controls[f"{link}_shuffled"]
        result[link] = {
            "gain_off_minus_real": off["validation_loss"] - real["validation_loss"],
            "sequence_gap_shuffled_minus_real": (
                shuffled["validation_loss"] - real["validation_loss"]
            ),
            "paired_real_vs_off": stage_c.paired_stats(
                real["per_sequence_losses"], off["per_sequence_losses"]
            ),
            "paired_real_vs_shuffled": stage_c.paired_stats(
                real["per_sequence_losses"], shuffled["per_sequence_losses"]
            ),
        }
    return result


def bootstrap_links(evaluated, seed, resamples):
    controls = evaluated["controls"]
    real = np.asarray(controls["all_real"]["per_sequence_losses"])
    rng = np.random.default_rng(seed)
    result = {"seed": seed, "resamples": resamples}
    for link in ("b3", "b5", "b6"):
        off = np.asarray(controls[f"{link}_off"]["per_sequence_losses"]) - real
        shuffled = np.asarray(controls[f"{link}_shuffled"]["per_sequence_losses"]) - real
        off_ci = stage_c.bootstrap_ci(off, rng, resamples=resamples)
        shuffled_ci = stage_c.bootstrap_ci(shuffled, rng, resamples=resamples)
        if off_ci["lower_2_5"] > 0 and shuffled_ci["lower_2_5"] > 0:
            label = "ROBUSTLY ESTABLISHED"
        elif off_ci["lower_2_5"] > 0 and shuffled_ci["mean"] > 0:
            label = "UTILITY ESTABLISHED / SEQUENCE SPECIFIC UNCERTAIN"
        elif shuffled_ci["lower_2_5"] > 0 and off_ci["mean"] > 0:
            label = "SEQUENCE-SPECIFIC / UTILITY UNCERTAIN"
        elif off_ci["mean"] > 0 and shuffled_ci["mean"] > 0:
            label = "DIRECTIONAL ONLY"
        else:
            label = "NOT ESTABLISHED"
        result[link] = {
            "off_minus_real": off_ci,
            "shuffled_minus_real": shuffled_ci,
            "classification": label,
        }
    return result


def run_matched_large(model, val_path, output, source_results):
    output = Path(output)
    path = output / "m1000_matched_large.json"
    boot_path = output / "m1000_matched_large_bootstrap.json"
    if path.exists() and boot_path.exists():
        return read_json(path), read_json(boot_path)
    names = [
        "all_real", "b3_off", "b3_shuffled", "b5_off", "b5_shuffled",
        "b6_off", "b6_shuffled",
    ]
    evaluated = stage_c.evaluate_incremental_subset(model, val_path, names, 4, 16)
    source = read_json(Path(source_results) / "m500_large_confirmation.json")
    if (
        evaluated["subset_sha256"] != source["subset_sha256"]
        or evaluated["batch_identities"] != source["batch_identities"]
        or evaluated["targets_per_control"] != 1_048_576
    ):
        raise SystemExit("M1000 matched-large subset does not exactly match M500")
    result = {
        **evaluated,
        "metrics": paired_metrics(evaluated),
        "matched_m500_subset_sha256": source["subset_sha256"],
        "matched_m500_exact_sequences": True,
        "m500_metrics": source["metrics"],
        "position_bins": position_bins_for_available_links(evaluated),
    }
    durable_json(path, result)
    boot = bootstrap_links(evaluated, MATCHED_BOOTSTRAP_SEED, 20_000)
    boot["m500_bootstrap"] = read_json(
        Path(source_results) / "m500_large_confirmation_bootstrap.json"
    )
    durable_json(boot_path, boot)
    return result, boot


def position_bins_for_available_links(evaluated):
    controls = evaluated["controls"]
    real = np.asarray(controls["all_real"]["per_position_loss"])
    result = {}
    for link in ("b3", "b5", "b6"):
        off = np.asarray(controls[f"{link}_off"]["per_position_loss"])
        shuffled = np.asarray(controls[f"{link}_shuffled"]["per_position_loss"])
        result[link] = {}
        for name, low, high in base.POSITION_BINS:
            selected = slice(low, high + 1)
            result[link][name] = {
                "off_minus_real": float((off[selected] - real[selected]).mean()),
                "shuffled_minus_real": float((shuffled[selected] - real[selected]).mean()),
            }
    return result


def run_fresh_final(model, val_path, output, matched):
    output = Path(output)
    result_path = output / "m1000_fresh_final_confirmation.json"
    boot_path = output / "m1000_fresh_final_bootstrap.json"
    if result_path.exists() and boot_path.exists():
        return read_json(result_path), read_json(boot_path)
    evaluated = stage_c.evaluate_incremental_subset(
        model, val_path, list(FRESH_CONTROLS), 20, 32
    )
    if evaluated["targets_per_control"] != 2_097_152:
        raise SystemExit("fresh M1000 confirmation target count mismatch")
    core = read_json(output / "true_incremental_1b.json")
    core_hashes = {row["combined_sha256"] for row in core["batch_identities"]}
    matched_hashes = {row["combined_sha256"] for row in matched["batch_identities"]}
    fresh_hashes = {row["combined_sha256"] for row in evaluated["batch_identities"]}
    disjoint = {
        "longitudinal_core_intersection": sorted(core_hashes & fresh_hashes),
        "m500_m1000_matched_large_intersection": sorted(matched_hashes & fresh_hashes),
        "older_F_G_provenance_available": False,
        "older_F_G_note": (
            "No directly comparable prior F/G batch-identity manifest exists in the sealed 2D3A lineage; "
            "no broader disjointness is claimed."
        ),
    }
    disjoint["passed"] = not disjoint["longitudinal_core_intersection"] and not disjoint[
        "m500_m1000_matched_large_intersection"
    ]
    if not disjoint["passed"]:
        raise SystemExit("fresh M1000 subset is not disjoint")
    manifest = {
        "validation_shard": str(val_path),
        "start_batch": 20,
        "batches": 32,
        "batch_size": base.VALIDATION_B,
        "sequence_length": base.T,
        "actual_sequences": evaluated["paired_sequences"],
        "actual_targets_per_control": evaluated["targets_per_control"],
        "subset_sha256": evaluated["subset_sha256"],
        "batch_identities": evaluated["batch_identities"],
    }
    durable_json(output / "m1000_fresh_subset_manifest.json", manifest)
    durable_json(output / "m1000_disjointness_audit.json", disjoint)
    result = {
        **evaluated,
        "metrics": paired_metrics(evaluated),
        "cleaned_subset_ce": {
            "111": evaluated["controls"]["all_real"]["validation_loss"],
            "011": evaluated["controls"]["b3_off"]["validation_loss"],
            "101": evaluated["controls"]["b5_off"]["validation_loss"],
            "110": evaluated["controls"]["b6_off"]["validation_loss"],
            "010": evaluated["controls"]["b3_b6_off"]["validation_loss"],
            "000": evaluated["controls"]["new_links_off"]["validation_loss"],
        },
        "position_bins": position_bins_for_available_links(evaluated),
        "fresh_disjoint_confirmation": True,
    }
    durable_json(result_path, result)
    boot = bootstrap_links(evaluated, FRESH_BOOTSTRAP_SEED, 50_000)
    real = np.asarray(evaluated["controls"]["all_real"]["per_sequence_losses"])
    b5_only = np.asarray(evaluated["controls"]["b3_b6_off"]["per_sequence_losses"])
    all_new_off = np.asarray(evaluated["controls"]["new_links_off"]["per_sequence_losses"])
    rng = np.random.default_rng(FRESH_BOOTSTRAP_SEED)
    boot["010_minus_111"] = stage_c.bootstrap_ci(
        b5_only - real, rng, resamples=50_000
    )
    boot["000_minus_111"] = stage_c.bootstrap_ci(
        all_new_off - real, rng, resamples=50_000
    )
    boot["orientation"] = {
        "010_minus_111": "positive means full 111 is better",
        "000_minus_111": "positive means the complete secondary recurrent subsystem improves CE",
    }
    durable_json(boot_path, boot)
    return result, boot


def copy_history(source_results, output):
    source = Path(source_results)
    output = Path(output)
    names = (
        "paired_maturation_controls.json", "gate_maturation.json",
        "b6_representation_maturation.json", "position_bin_maturation.json",
        "stability_8pass_maturation.json", "incremental_cache_audit.json",
        *ATTENTION_FILES.values(), *GRADIENT_FILES.values(),
    )
    for name in names:
        old = read_json(source / name)
        new = read_json(output / name) if (output / name).exists() else {}
        durable_json(output / name, {**old, **new})


def build_maturation(source_results, output):
    source = Path(source_results)
    output = Path(output)
    copy_history(source, output)
    rows = dict(read_json(source / "maturation_table_100m_to_500m.json")["rows"])
    gates = read_json(output / "gate_maturation.json")
    increments = {}
    parallels = {}
    for age in ("625m", "750m", "875m", "1b"):
        incremental = read_json(output / f"true_incremental_{age}.json")
        parallel = read_json(output / f"milestone_{age}_validation.json")
        increments[age] = incremental
        parallels[age] = parallel
        row = {
            "all_real_ce": parallel["controls"]["all_real"]["validation_loss"],
            "new_links_off_ce": parallel["controls"]["new_links_off"]["validation_loss"],
            "combined_gain": incremental["combined_new_link_gain"],
            "combined_gap": incremental["combined_new_sequence_gap"],
        }
        for link in LINKS:
            row[f"{link}_gain"] = incremental[f"true_{link}_gain"]
            row[f"{link}_gap"] = incremental[f"true_{link}_sequence_gap"]
            row[f"{link}_gate"] = gates[age][link]["effective"]
        rows[age] = row
    for age in AGES[:7]:
        rows[age].setdefault("new_links_off_ce", rows[age]["all_real_ce"] + rows[age]["combined_gain"])
    intervals = {
        f"{left}_to_{right}": {
            key: rows[right][key] - rows[left][key] for key in rows[left]
        }
        for left, right in zip(AGES, AGES[1:])
    }
    interactions = {}
    for age in AGES:
        marginal = sum(rows[age][f"{link}_gain"] for link in ("b3", "b5", "b6"))
        interactions[age] = {
            "combined_gain": rows[age]["combined_gain"],
            "sum_marginals": marginal,
            "simple_interaction": rows[age]["combined_gain"] - marginal,
        }
    durable_json(output / "maturation_table_100m_to_1b.json", {"rows": rows})
    durable_json(output / "maturation_interval_deltas.json", intervals)
    durable_json(output / "combined_link_interaction.json", interactions)
    return rows, increments, parallels, interactions


def build_factorial_maturation(source_results, output):
    source = Path(source_results)
    output = Path(output)
    factors = {
        "500m": read_json(source / "m500_recurrent_subset_factorial.json"),
        **{
            age: read_json(output / f"factorial_{age}.json")
            for age in ("625m", "750m", "875m", "1b")
        },
    }
    best = {
        age: {
            "bits": row["best_bits_B3_B5_B6"],
            "control": row["best_control"],
            "ce": row["best_ce"],
        }
        for age, row in factors.items()
    }
    durable_json(output / "factorial_maturation.json", factors)
    durable_json(output / "best_subset_maturation.json", best)
    durable_json(output / "m1000_factorial.json", factors["1b"])
    durable_json(output / "m1000_factorial_interactions.json", {
        "factorial_main_effects": factors["1b"]["factorial_main_effects"],
        "pairwise_interactions": factors["1b"]["pairwise_interactions"],
        "three_way_interaction": factors["1b"]["three_way_interaction"],
        "simple_interaction": factors["1b"][
            "historical_combined_minus_leave_one_out_marginals"
        ],
    })
    return factors, best


def build_boundary_history(output):
    output = Path(output)
    boundaries = {"b3": "32-63", "b5": "64-127", "b6": "512-639"}
    result = {}
    attentions = {link: read_json(output / ATTENTION_FILES[link]) for link in LINKS}
    for age in AGES:
        result[age] = {}
        for link, bin_name in boundaries.items():
            result[age][link] = {
                "bin": bin_name,
                **attentions[link][age]["recurrent"]["bins"][bin_name],
            }
    durable_json(output / "boundary_memory_maturation.json", result)
    return result


def longitudinal_fate(rows, link, utility, fresh_label, factorial):
    gains = [rows[age][f"{link}_gain"] for age in AGES]
    gaps = [rows[age][f"{link}_gap"] for age in AGES]
    signs = [value > 0 for value in gains]
    flips = sum(left != right for left, right in zip(signs, signs[1:]))
    final_pairwise = factorial["pairwise_interactions"]
    interactions = {
        "b3": (final_pairwise["B3xB5"], final_pairwise["B3xB6"]),
        "b5": (final_pairwise["B3xB5"], final_pairwise["B5xB6"]),
        "b6": (final_pairwise["B3xB6"], final_pairwise["B5xB6"]),
    }[link]
    if fresh_label == "ROBUSTLY ESTABLISHED" and all(signs[-4:]):
        if gains[-1] > gains[-2] and (not all(signs[:4]) or link == "b3"):
            return "RECOVERED AND STRENGTHENING"
        if abs(gains[-1] - gains[-2]) <= max(1e-5, .2 * abs(gains[-2])):
            return "SATURATED"
        return "DURABLY POSITIVE"
    if utility in ("POSITIVE UTILITY", "STRONG POSITIVE") and flips >= 2:
        return "OSCILLATORY POSITIVE"
    if fresh_label in ("DIRECTIONAL ONLY", "NOT ESTABLISHED") and max(interactions) > 0:
        return "SYNERGISTIC ONLY"
    if gains[-1] < 0 and gaps[-1] < 0:
        return "HARMFUL" if sum(value < 0 for value in gains[-4:]) >= 3 else "REVERSED"
    if gains[-1] > 0 and gains[-1] < max(gains[-4:-1]):
        return "WEAKENING"
    return "UNRESOLVED"


def classify_and_recommend(rows, factorial, fresh_boot):
    best = factorial["best_bits_B3_B5_B6"]
    combined_positive = rows["1b"]["combined_gain"] > 0
    b5_robust = fresh_boot["b5"]["classification"] == "ROBUSTLY ESTABLISHED"
    weak = [
        link for link in ("b3", "b6")
        if fresh_boot[link]["classification"] != "ROBUSTLY ESTABLISHED"
    ]
    pairwise = factorial["pairwise_interactions"]
    positive_interaction = max(
        pairwise["B3xB5"], pairwise["B3xB6"], pairwise["B5xB6"],
        factorial["three_way_interaction"],
    ) > 0
    recurrence_harmful = fresh_boot["000_minus_111"]["upper_97_5"] < 0
    cleaned_b5_only = fresh_boot["010_minus_111"]["upper_97_5"] < 0
    if recurrence_harmful:
        return (
            "RECURRENT PYRAMID HARMFUL",
            "STOP THIS RECURRENT PYRAMID LINE AND REDESIGN THE RECURRENT READOUT",
        )
    if cleaned_b5_only and not positive_interaction:
        return (
            "B5-DOMINANT RECURRENT PYRAMID",
            "RUN A MATCHED CLEANED-PYRAMID ADAPTATION EXPERIMENT FROM THE 1B BASELINE",
        )
    if best == "111" and combined_positive and b5_robust and weak and positive_interaction:
        return (
            "MATURE SYNERGISTIC RECURRENT PYRAMID",
            "PRESERVE THE FULL RECURRENT CIRCUIT AND TEST LEARNED SOURCE-DEPTH ROUTING FROM THE FROZEN 1B BASELINE",
        )
    if best == "111" and combined_positive and b5_robust:
        return (
            "MATURE MULTI-LINK POSITIVE RECURRENT PYRAMID",
            "FREEZE 2D3A-1B AS THE RECURRENT BASELINE AND RUN A MATCHED ATTNRES-STYLE LEARNED SOURCE-DEPTH ROUTING EXPERIMENT",
        )
    if b5_robust:
        return (
            "B5-DOMINANT RECURRENT PYRAMID",
            "RUN A MATCHED CLEANED-PYRAMID ADAPTATION EXPERIMENT FROM THE 1B BASELINE",
        )
    if combined_positive:
        return (
            "PARTIAL RECURRENT PYRAMID",
            "FREEZE 2D3A-1B AS THE RECURRENT BASELINE AND RUN A MATCHED ATTNRES-STYLE LEARNED SOURCE-DEPTH ROUTING EXPERIMENT",
        )
    if abs(rows["1b"]["combined_gain"]) < 1e-4:
        return (
            "RECURRENT PYRAMID NEAR ZERO",
            "STOP THIS RECURRENT PYRAMID LINE AND REDESIGN THE RECURRENT READOUT",
        )
    return (
        "RECURRENT PYRAMID HARMFUL",
        "STOP THIS RECURRENT PYRAMID LINE AND REDESIGN THE RECURRENT READOUT",
    )


def build_plots(output, rows, interactions, factors, matched_boot, fresh_boot, performance):
    import matplotlib.pyplot as plt

    output = Path(output)
    x = [TARGETS_BY_AGE[age] for age in AGES]

    def save(number, draw):
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        draw(axis)
        figure.tight_layout()
        figure.savefig(output / f"plot_p{number:02d}.png", dpi=160)
        plt.close(figure)

    save(1, lambda axis: (
        axis.plot(x, [rows[age]["all_real_ce"] for age in AGES], marker="o", label="ALL_REAL"),
        axis.plot(x, [rows[age]["new_links_off_ce"] for age in AGES], marker="o", label="NEW_LINKS_OFF"),
        axis.legend(), axis.set(xlabel="cumulative targets", ylabel="CE"),
    ))
    save(2, lambda axis: (
        [axis.plot(x, [rows[age][f"{link}_gain"] for age in AGES], marker="o", label=link.upper()) for link in LINKS],
        axis.axhline(0, color="black", lw=.7), axis.legend(),
        axis.set(xlabel="cumulative targets", ylabel="recurrent gain"),
    ))
    save(3, lambda axis: (
        [axis.plot(x, [rows[age][f"{link}_gap"] for age in AGES], marker="o", label=link.upper()) for link in LINKS],
        axis.axhline(0, color="black", lw=.7), axis.legend(),
        axis.set(xlabel="cumulative targets", ylabel="sequence gap"),
    ))
    save(4, lambda axis: (
        [axis.plot(x, [rows[age][f"{link}_gate"] for age in AGES], marker="o", label=link.upper()) for link in LINKS],
        axis.legend(), axis.set(xlabel="cumulative targets", ylabel="effective gate"),
    ))
    save(5, lambda axis: (
        [axis.plot([rows[age][f"{link}_gate"] for age in AGES],
                   [rows[age][f"{link}_gain"] for age in AGES], marker="o", label=link.upper()) for link in LINKS],
        axis.axhline(0, color="black", lw=.7), axis.legend(),
        axis.set(xlabel="effective gate", ylabel="recurrent gain"),
    ))
    save(6, lambda axis: (
        [axis.plot([rows[age][f"{link}_gate"] for age in AGES],
                   [rows[age][f"{link}_gap"] for age in AGES], marker="o", label=link.upper()) for link in LINKS],
        axis.axhline(0, color="black", lw=.7), axis.legend(),
        axis.set(xlabel="effective gate", ylabel="sequence gap"),
    ))
    attention = {link: read_json(output / ATTENTION_FILES[link]) for link in LINKS}
    for number, link in ((7, "b3"), (8, "b5"), (9, "b6")):
        bins = list(attention[link]["1b"]["recurrent"]["bins"])
        save(number, lambda axis, link=link, bins=bins: (
            [axis.plot(bins, [attention[link][age]["recurrent"]["bins"][name]["raw_mass"] for name in bins],
                       marker="o", label=age.upper()) for age in AGES],
            axis.legend(fontsize=6), axis.tick_params(axis="x", rotation=35),
            axis.set(ylabel=f"{link.upper()} recurrent attention mass"),
        ))
    boundary = read_json(output / "boundary_memory_maturation.json")
    save(10, lambda axis: (
        [axis.plot(x, [boundary[age][link]["raw_mass"] for age in AGES], marker="o", label=link.upper())
         for link in ("b3", "b5", "b6")],
        axis.legend(), axis.set(xlabel="cumulative targets", ylabel="boundary-bin raw mass"),
    ))
    gradients = {link: read_json(output / GRADIENT_FILES[link]) for link in LINKS}
    save(11, lambda axis: (
        [axis.plot(list(gradients[link]["1b"]["bins"]),
                   [row["mean_gradient_rms"] for row in gradients[link]["1b"]["bins"].values()],
                   marker="o", label=link.upper()) for link in LINKS],
        axis.legend(), axis.tick_params(axis="x", rotation=35),
        axis.set(ylabel="writer-gradient RMS @1B"),
    ))
    position = read_json(output / "position_bin_maturation.json")
    bins = list(position["1b"]["b3"])
    save(12, lambda axis: (
        [axis.plot(bins, [position["1b"][link][name]["off_minus_real"] for name in bins],
                   marker="o", label=link.upper()) for link in LINKS],
        axis.axhline(0, color="black", lw=.7), axis.legend(),
        axis.tick_params(axis="x", rotation=35), axis.set(ylabel="position-binned gain @1B"),
    ))
    save(13, lambda axis: (
        axis.plot(x, [rows[age]["combined_gain"] for age in AGES], marker="o", label="combined gain"),
        axis.plot(x, [interactions[age]["simple_interaction"] for age in AGES], marker="o", label="simple interaction"),
        axis.axhline(0, color="black", lw=.7), axis.legend(), axis.set(xlabel="cumulative targets"),
    ))
    factor_ages = ("500m", "625m", "750m", "875m", "1b")
    bits_value = {bits: index for index, bits in enumerate(("000", "001", "010", "011", "100", "101", "110", "111"))}
    save(14, lambda axis: (
        axis.plot([TARGETS_BY_AGE[age] for age in factor_ages],
                  [bits_value[factors[age]["best_bits_B3_B5_B6"]] for age in factor_ages], marker="o"),
        axis.set_yticks(list(bits_value.values()), list(bits_value)),
        axis.set(xlabel="cumulative targets", ylabel="best B3/B5/B6 configuration"),
    ))
    final_factor = factors["1b"]
    save(15, lambda axis: (
        axis.bar(list(final_factor["conditions"]),
                 [row["validation_loss"] for row in final_factor["conditions"].values()]),
        axis.set(xlabel="B3/B5/B6 bits", ylabel="M1000 CE"),
    ))
    source_boot = matched_boot["m500_bootstrap"]
    save(16, lambda axis: (
        [axis.plot([500, 1000], [source_boot[link]["off_minus_real"]["mean"],
                                 matched_boot[link]["off_minus_real"]["mean"]],
                   marker="o", label=f"{link.upper()} Off-Real") for link in ("b3", "b5", "b6")],
        axis.axhline(0, color="black", lw=.7), axis.legend(),
        axis.set(xlabel="M targets", ylabel="matched-large paired effect"),
    ))

    def draw_fresh_ci(axis):
        for index, link in enumerate(("b3", "b5", "b6")):
            for offset, metric in enumerate(("off_minus_real", "shuffled_minus_real")):
                row = fresh_boot[link][metric]
                location = index * 3 + offset
                axis.errorbar(location, row["mean"], yerr=[
                    [row["mean"] - row["lower_2_5"]],
                    [row["upper_97_5"] - row["mean"]],
                ], fmt="o")
        axis.axhline(0, color="black", lw=.7)
        axis.set_xticks([.5, 3.5, 6.5], ["B3", "B5", "B6"])
        axis.set(ylabel="fresh paired effect (95% CI)")

    save(17, draw_fresh_ci)
    b6 = read_json(output / "b6_representation_maturation.json")
    save(18, lambda axis: (
        axis.plot(x, [b6[age]["primary_O_minus_R"] for age in AGES], marker="o"),
        axis.axhline(0, color="black", lw=.7),
        axis.set(xlabel="cumulative targets", ylabel="B6 representation gain"),
    ))
    stability = read_json(output / "stability_8pass_maturation.json")
    save(19, lambda axis: (
        axis.plot(x, [stability[age]["passes"][-1]["ce"] for age in AGES], marker="o"),
        axis.set(xlabel="cumulative targets", ylabel="8th-pass CE"),
    ))
    terminal = read_json(output / "stability_16pass_terminal.json")
    save(20, lambda axis: (
        axis.plot([row["pass"] for row in terminal["passes"]],
                  [row["ce"] for row in terminal["passes"]], marker="o"),
        axis.set(xlabel="self-composition pass", ylabel="CE @1B"),
    ))
    training = performance["training"]
    save(21, lambda axis: (
        axis.plot([row["cumulative_targets"] for row in training],
                  [row["targets_per_second"] for row in training]),
        axis.set(xlabel="cumulative targets", ylabel="targets/s"),
    ))


def make_questions(rows, increments, interactions, factors, matched, matched_boot,
                   fresh, fresh_boot, checkpoint, output, payload, fates, recommendation):
    output = Path(output)
    training = [
        json.loads(line)
        for line in (output / "training_metrics_500m_to_1b.jsonl").read_text().splitlines()
    ]
    first = training[0]
    new_ages = ("625m", "750m", "875m", "1b")
    values = lambda key: {age: rows[age][key] for age in new_ages}
    attention = {link: read_json(output / ATTENTION_FILES[link]) for link in LINKS}
    gradients = {link: read_json(output / GRADIENT_FILES[link]) for link in LINKS}
    b6rep = read_json(output / "b6_representation_maturation.json")
    stability = read_json(output / "stability_8pass_maturation.json")
    terminal = read_json(output / "stability_16pass_terminal.json")
    final_factor = factors["1b"]

    def boundary_focused(link, bin_name):
        bins = attention[link]["1b"]["recurrent"]["bins"]
        return bins[bin_name]["raw_mass"] == max(row["raw_mass"] for row in bins.values())

    mean_lag_shift = {}
    for link in LINKS:
        old = np.mean([row["mean_lag"] for row in attention[link]["500m"]["recurrent"]["per_head"]])
        new = np.mean([row["mean_lag"] for row in attention[link]["1b"]["recurrent"]["per_head"]])
        mean_lag_shift[link] = float(new - old)
    most_older = max(mean_lag_shift, key=mean_lag_shift.get)
    source_matched = matched["m500_metrics"]
    matched_delta = {
        link: {
            "off_minus_real_change": matched["metrics"][link]["gain_off_minus_real"]
            - source_matched[link]["gain_off_minus_real"],
            "shuffled_minus_real_change": matched["metrics"][link]["sequence_gap_shuffled_minus_real"]
            - source_matched[link]["sequence_gap_shuffled_minus_real"],
        }
        for link in ("b3", "b5", "b6")
    }
    b1_prior_delta = rows["875m"]["b1_gain"] - rows["750m"]["b1_gain"]
    b1_final_delta = rows["1b"]["b1_gain"] - rows["875m"]["b1_gain"]
    q = {
        "Q1": sha256(Path(checkpoint["checkpoint"])) == checkpoint["sha256"],
        "Q2": first.get("consumed_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "Q3": first.get("consumed_global_stream_sha256") == SOURCE_NEXT_STREAM,
        "Q4": not read_json(output / "optimizer_resume_500m_manifest.json")["reset"],
        "Q5": not read_json(output / "scheduler_resume_500m_manifest.json")["reset"],
        "Q6": not read_json(output / "data_resume_500m_manifest.json")["reset"],
        "Q7": first["optimizer_lrs"],
        "Q8": first.get("consumed_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "Q9": base.architecture_manifest()["parameter_count"] == base.MODEL_PARAMETERS,
        "Q10": rows["625m"]["all_real_ce"],
        "Q11": rows["750m"]["all_real_ce"],
        "Q12": rows["875m"]["all_real_ce"],
        "Q13": rows["1b"]["all_real_ce"],
        "Q14": {"gain": values("b1_gain"), "gap": values("b1_gap")},
        "Q15": {"gain": values("b3_gain"), "gap": values("b3_gap")},
        "Q16": {"gain": values("b5_gain"), "gap": values("b5_gap")},
        "Q17": {"gain": values("b6_gain"), "gap": values("b6_gap")},
        "Q18": values("combined_gain"),
        "Q19": values("combined_gap"),
        "Q20": rows["1b"]["b1_gate"],
        "Q21": rows["1b"]["b3_gate"],
        "Q22": rows["1b"]["b5_gate"],
        "Q23": rows["1b"]["b6_gate"],
        "Q24": [link.upper() for link in LINKS if rows["1b"][f"{link}_gate"] > rows["500m"][f"{link}_gate"]],
        "Q25": [link.upper() for link in LINKS if rows["1b"][f"{link}_gate"] < rows["500m"][f"{link}_gate"]],
        "Q26": rows["1b"]["b1_gain"] > rows["500m"]["b1_gain"] and rows["1b"]["b1_gap"] > rows["500m"]["b1_gap"],
        "Q27": abs(b1_final_delta) < abs(b1_prior_delta),
        "Q28": base.classify_link(increments["1b"], "b3") in ("POSITIVE UTILITY", "STRONG POSITIVE"),
        "Q29": fates["b3"],
        "Q30": fresh_boot["b5"]["classification"] == "ROBUSTLY ESTABLISHED",
        "Q31": fates["b5"],
        "Q32": rows["1b"]["b6_gain"] > 0 and rows["1b"]["b6_gap"] > 0,
        "Q33": fates["b6"],
        "Q34": b6rep["1b"]["primary_O_minus_R"],
        "Q35": b6rep["1b"]["primary_O_minus_R"] > b6rep["500m"]["primary_O_minus_R"],
        "Q36": boundary_focused("b3", "32-63"),
        "Q37": boundary_focused("b5", "64-127"),
        "Q38": boundary_focused("b6", "512-639"),
        "Q39": {"link": most_older.upper(), "mean_lag_shift": mean_lag_shift[most_older]},
        "Q40": all(gradients[link][age]["all_eligible_bins_nonzero"] for link in LINKS for age in new_ages),
        "Q41": interactions["1b"]["simple_interaction"],
        "Q42": final_factor["factorial_main_effects"],
        "Q43": final_factor["pairwise_interactions"],
        "Q44": final_factor["three_way_interaction"],
        "Q45": {"bits": final_factor["best_bits_B3_B5_B6"], "ce": final_factor["best_ce"]},
        "Q46": final_factor["best_bits_B3_B5_B6"] == "111",
        "Q47": {"marginal_core_gain": rows["1b"]["b3_gain"], "utility": base.classify_link(increments["1b"], "b3"),
                "fresh_label": fresh_boot["b3"]["classification"]},
        "Q48": {"marginal_core_gain": rows["1b"]["b6_gain"], "utility": base.classify_link(increments["1b"], "b6"),
                "fresh_label": fresh_boot["b6"]["classification"]},
        "Q49": final_factor["best_bits_B3_B5_B6"][1] == "1" and final_factor["factorial_main_effects"]["B5"] > 0,
        "Q50": matched["metrics"]["b3"]["gain_off_minus_real"],
        "Q51": matched["metrics"]["b3"]["sequence_gap_shuffled_minus_real"],
        "Q52": matched_delta["b3"],
        "Q53": {"off_minus_real": matched["metrics"]["b5"]["gain_off_minus_real"],
                "shuffled_minus_real": matched["metrics"]["b5"]["sequence_gap_shuffled_minus_real"],
                "change_from_m500": matched_delta["b5"]},
        "Q54": {"off_minus_real": matched["metrics"]["b6"]["gain_off_minus_real"],
                "shuffled_minus_real": matched["metrics"]["b6"]["sequence_gap_shuffled_minus_real"],
                "change_from_m500": matched_delta["b6"]},
        "Q55": fresh_boot["b3"]["off_minus_real"],
        "Q56": fresh_boot["b3"]["shuffled_minus_real"],
        "Q57": fresh_boot["b3"]["classification"],
        "Q58": fresh_boot["b5"]["off_minus_real"],
        "Q59": fresh_boot["b5"]["shuffled_minus_real"],
        "Q60": fresh_boot["b5"]["classification"],
        "Q61": fresh_boot["b6"]["off_minus_real"],
        "Q62": fresh_boot["b6"]["shuffled_minus_real"],
        "Q63": fresh_boot["b6"]["classification"],
        "Q64": fresh_boot["010_minus_111"],
        "Q65": fresh_boot["000_minus_111"],
        "Q66": fresh_boot["010_minus_111"]["mean"] < 0,
        "Q67": all(stability[age]["passed"] for age in AGES),
        "Q68": terminal["passed"],
        "Q69": {"sha256": checkpoint["sha256"], "strict_reopen": checkpoint["strict_reopen"]["passed"],
                "updates": payload["d3a_completed_updates"], "targets": payload["d3a_processed_targets"]},
        "Q70": recommendation,
    }
    return q


def render_report(summary):
    rows = summary["maturation_table"]
    questions = summary["questions"]
    lines = [
        "EXPERIMENT 2D3A — 1B COMPLETE", "",
        "PRIMARY 1B CLASSIFICATION:", summary["primary_classification"], "",
        "CUMULATIVE 2D3A TARGETS:", f"{FINAL_TARGETS:,}", "",
        "B3 TRUE RECURRENT GAIN @1B:", str(rows["1b"]["b3_gain"]), "",
        "B5 TRUE RECURRENT GAIN @1B:", str(rows["1b"]["b5_gain"]), "",
        "B6 TRUE RECURRENT GAIN @1B:", str(rows["1b"]["b6_gain"]), "",
        "B3 TRUE SEQUENCE GAP @1B:", str(rows["1b"]["b3_gap"]), "",
        "B5 TRUE SEQUENCE GAP @1B:", str(rows["1b"]["b5_gap"]), "",
        "B6 TRUE SEQUENCE GAP @1B:", str(rows["1b"]["b6_gap"]), "",
        "## Full maturation table", "",
        "| Metric | " + " | ".join(age.upper() for age in AGES) + " |",
        "|---|" + "---:|" * len(AGES),
    ]
    metrics = (
        "all_real_ce", "b1_gain", "b1_gap", "b1_gate", "b3_gain", "b3_gap", "b3_gate",
        "b5_gain", "b5_gap", "b5_gate", "b6_gain", "b6_gap", "b6_gate",
        "combined_gain", "combined_gap",
    )
    for metric in metrics:
        lines.append(f"| {metric} | " + " | ".join(f"{rows[age][metric]:.12g}" for age in AGES) + " |")
    lines += [
        "", "## Continuity and frozen analyses", "",
        f"Source M500 checkpoint SHA: `{SOURCE_SHA256}` (exact match).",
        "Optimizer, scheduler, loader, gate values, and all RNG states were restored without reset.",
        "Architecture and parameter count remained unchanged at 124,475,908 parameters.",
        f"M1000 checkpoint SHA: `{summary['checkpoint']['sha256']}`.",
        f"M1000 next-batch SHA: `{summary['checkpoint']['next_global_batch_sha256']}`.",
        f"M1000 next-stream SHA: `{summary['checkpoint']['next_global_batch_stream_sha256']}`.",
        "All attention, boundary-memory, writer-gradient, representation, position-bin, factorial,",
        "matched-large, fresh-final, bootstrap, stability, memory, and runtime artifacts are preserved.",
        "", "## Scientific questions", "",
    ]
    for index in range(1, 71):
        lines.append(f"Q{index}. {json.dumps(questions[f'Q{index}'], sort_keys=True)}")
    lines += [
        "", "## Exactly one next recommendation", "", summary["recommendation"], "",
        "NO TRAINING BEYOND 1,000,341,504 TARGETS WAS RUN.", "",
        "# EXPERIMENT 2D3A 1B COMPLETE", "",
    ]
    return "\n".join(lines)


def run_finalize(args):
    require_branch()
    device = base.require_a100()
    output = Path(args.output_dir)
    record_command(output, "finalize")
    model, optimizer, loader, payload = base.load_d3a_checkpoint(
        args.final_checkpoint, device, restore=False
    )
    if (
        payload["d3a_completed_updates"] != FINAL_UPDATE
        or payload["d3a_processed_targets"] != FINAL_TARGETS
    ):
        raise SystemExit("final checkpoint is not the exact M1000 endpoint")
    manifest = read_json(output / "checkpoint_manifest.json")
    checkpoint = manifest[str(FINAL_UPDATE)]
    if sha256(args.final_checkpoint) != checkpoint["sha256"]:
        raise SystemExit("final checkpoint SHA mismatch")
    if not checkpoint["strict_reopen"]["passed"]:
        raise SystemExit("final checkpoint strict reopen did not pass")
    rows, increments, parallels, interactions = build_maturation(args.source_results, output)
    factors, best = build_factorial_maturation(args.source_results, output)
    boundary = build_boundary_history(output)
    val = base.validation_path(Path(args.data_root))
    matched, matched_boot = run_matched_large(model, val, output, args.source_results)
    fresh, fresh_boot = run_fresh_final(model, val, output, matched)
    utilities = {
        link: base.classify_link(increments["1b"], link) for link in LINKS
    }
    fates = {
        link: longitudinal_fate(
            rows, link, utilities[link], fresh_boot[link]["classification"], factors["1b"]
        )
        for link in ("b3", "b5", "b6")
    }
    primary, recommendation = classify_and_recommend(rows, factors["1b"], fresh_boot)
    training = [
        json.loads(line)
        for line in (output / "training_metrics_500m_to_1b.jsonl").read_text().splitlines()
    ]
    pass_times = {"pass1_seconds": 0.0, "pass2_seconds": 0.0, "pass3_seconds": 0.0}
    for row in training:
        for index, value in enumerate(row.get("approximate_pass_forward_seconds", []), 1):
            pass_times[f"pass{index}_seconds"] += float(value)
    evaluation_times = {
        "milestone_incremental_seconds": {
            age: increments[age]["performance"]["wall_seconds"]
            for age in ("625m", "750m", "875m", "1b")
        },
        "factorial_extra_seconds": {
            age: factors[age]["performance"]["wall_seconds"]
            for age in ("625m", "750m", "875m", "1b")
        },
        "matched_large_seconds": matched["performance"]["wall_seconds"],
        "fresh_final_seconds": fresh["performance"]["wall_seconds"],
    }
    performance = {
        "training": training,
        "updates": len(training),
        "training_wall_seconds": sum(row["wall_seconds"] for row in training),
        "mean_wall_seconds_per_update": float(np.mean([row["wall_seconds"] for row in training])),
        "mean_targets_per_second": float(np.mean([row["targets_per_second"] for row in training])),
        "max_peak_allocated_vram_mb": max(row["peak_allocated_vram_mb"] for row in training),
        "max_peak_reserved_vram_mb": max(row["peak_reserved_vram_mb"] for row in training),
        "evaluation": evaluation_times,
        **pass_times,
    }
    memory = base.memory_accounting()
    durable_json(output / "memory_accounting.json", memory)
    durable_json(output / "performance.json", performance)
    gates = base.gate_values(model)
    gate_divergence = {
        link: {
            "gate": rows["1b"][f"{link}_gate"],
            "gain": rows["1b"][f"{link}_gain"],
            "gap": rows["1b"][f"{link}_gap"],
            "gate_utility_divergence": (
                abs(rows["1b"][f"{link}_gate"]) > 0
                and (rows["1b"][f"{link}_gain"] <= 0 or rows["1b"][f"{link}_gap"] <= 0)
            ),
        }
        for link in LINKS
    }
    baseline_hash_files = [
        "maturation_table_100m_to_1b.json", "maturation_interval_deltas.json",
        "paired_maturation_controls.json", "gate_maturation.json",
        "boundary_memory_maturation.json", "factorial_maturation.json",
        "m1000_matched_large.json", "m1000_matched_large_bootstrap.json",
        "m1000_fresh_final_confirmation.json", "m1000_fresh_final_bootstrap.json",
        "stability_8pass_maturation.json", "stability_16pass_terminal.json",
    ]
    baseline = {
        "checkpoint_path": checkpoint["persistent"]["checkpoint"],
        "checkpoint_sha256": checkpoint["sha256"],
        "architecture": base.architecture_manifest(),
        "parameter_count": base.MODEL_PARAMETERS,
        "gates": gates,
        "optimizer_state_summary": {
            "state_tensors": len(optimizer.state),
            "step_values": optimizer_steps(optimizer),
            "group_lrs": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
            "reset": False,
        },
        "scheduler_state_summary": payload.get("scheduler"),
        "training_targets": FINAL_TARGETS,
        "training_updates": FINAL_UPDATE,
        "next_batch_sha256": checkpoint["next_global_batch_sha256"],
        "next_stream_sha256": checkpoint["next_global_batch_stream_sha256"],
        "git_implementation_commit": payload["metadata"]["git_implementation_commit"],
        "planned_final_tag": "experiment-2d3a-alternating-integration-pyramid-1b-final",
        "longitudinal_artifact_hashes": {
            name: sha256(output / name) for name in baseline_hash_files
        },
        "canonical_fixed_2d3a_1b_baseline": True,
    }
    durable_json(output / "ONE_BILLION_BASELINE_MANIFEST.json", baseline)
    questions = make_questions(
        rows, increments, interactions, factors, matched, matched_boot,
        fresh, fresh_boot, checkpoint, output, payload, fates, recommendation,
    )
    summary = {
        "experiment": EXPERIMENT,
        "primary_classification": primary,
        "cumulative_updates": FINAL_UPDATE,
        "cumulative_targets": FINAL_TARGETS,
        "architecture_unchanged": True,
        "parameter_count": base.MODEL_PARAMETERS,
        "maturation_table": rows,
        "utilities_1b": utilities,
        "fate_labels": fates,
        "gate_utility_diagnostic": gate_divergence,
        "combined_interaction": interactions,
        "factorial_maturation": factors,
        "best_subset_maturation": best,
        "matched_large": matched,
        "matched_large_bootstrap": matched_boot,
        "fresh_final_confirmation": fresh,
        "fresh_final_bootstrap": fresh_boot,
        "boundary_memory": boundary,
        "recommendation": recommendation,
        "checkpoint": checkpoint,
        "one_billion_baseline_manifest": baseline,
        "performance": performance,
        "memory": memory,
        "questions": questions,
        "no_training_beyond_1b": True,
    }
    durable_json(output / "result_summary.json", summary)
    build_plots(output, rows, interactions, factors, matched_boot, fresh_boot, performance)
    durable_json(output / "storage_cleanup_manifest.json", {
        "M100_retained": True,
        "M250_retained": True,
        "M500_retained": True,
        "M625_retained": True,
        "M750_retained": True,
        "M875_retained": True,
        "M1000_retained": True,
        "dataset_retained": True,
        "persistent_volume_retained": True,
        "deleted": [],
        "passed": True,
    })
    report = render_report(summary)
    durable_text(output / "EXPERIMENT_2D3A_1B_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report + "\nFinal pod may be stopped after Git and local-backup verification.\n",
    )
    checks = {
        "exact_final_update": payload["d3a_completed_updates"] == FINAL_UPDATE,
        "exact_final_targets": payload["d3a_processed_targets"] == FINAL_TARGETS,
        "exact_954_new_updates": (
            len(training) == 954
            and training[0]["cumulative_update"] == 955
            and training[-1]["cumulative_update"] == 1908
        ),
        "architecture_unchanged": sum(parameter.numel() for parameter in model.parameters()) == base.MODEL_PARAMETERS,
        "source_sha_exact": read_json(output / "source_500m_manifest.json")["sha256"] == SOURCE_SHA256,
        "optimizer_not_reset": not read_json(output / "optimizer_resume_500m_manifest.json")["reset"],
        "scheduler_not_reset": not read_json(output / "scheduler_resume_500m_manifest.json")["reset"],
        "data_not_reset": not read_json(output / "data_resume_500m_manifest.json")["reset"],
        "mandatory_restart": read_json(output / "mandatory_fresh_process_restart_update_1431.json")["passed"],
        "final_checkpoint_sha": sha256(args.final_checkpoint) == checkpoint["sha256"],
        "final_checkpoint_strict": checkpoint["strict_reopen"]["passed"],
        "canonical_milestones": all(parallels[age]["subset_sha256"] == CANONICAL_SHA for age in ("625m", "750m", "875m", "1b")),
        "maturation_core_milestones": all(increments[age]["subset_sha256"] == MATURATION_CORE_SHA for age in ("625m", "750m", "875m", "1b")),
        "matched_large_exact": matched["matched_m500_exact_sequences"] and matched["targets_per_control"] == 1_048_576,
        "fresh_final_targets": fresh["targets_per_control"] == 2_097_152,
        "fresh_disjoint": read_json(output / "m1000_disjointness_audit.json")["passed"],
        "longitudinal_stability": all(read_json(output / "stability_8pass_maturation.json")[age]["passed"] for age in AGES),
        "terminal_stability_recorded": len(read_json(output / "stability_16pass_terminal.json")["passes"]) == 16,
        "memory_exact": memory["B1"]["total_inference_state_bytes"] == 33_288_192,
        "no_training_beyond": max(row["cumulative_update"] for row in training) == FINAL_UPDATE,
        "single_recommendation": isinstance(recommendation, str) and bool(recommendation),
    }
    inventory = {
        name: ((output / name).is_file() or name == "FINAL_AUDIT.json")
        for name in REQUIRED_ARTIFACTS
    }
    inventory.update({
        f"plot_p{number:02d}.png": (output / f"plot_p{number:02d}.png").is_file()
        for number in range(1, 22)
    })
    checks["required_artifacts"] = all(inventory.values())
    audit = {
        "experiment": EXPERIMENT,
        "checks": checks,
        "passed": all(checks.values()),
        "artifact_inventory": inventory,
        "final_checkpoint_sha256": checkpoint["sha256"],
    }
    durable_json(output / "FINAL_AUDIT.json", audit)
    heartbeat(
        output, FINAL_UPDATE, {"final_audit_passed": audit["passed"]},
        checkpoint["persistent"]["checkpoint"], "complete",
    )
    if not audit["passed"]:
        raise SystemExit(f"final audit failed: {checks}")
    print("EXPERIMENT_2D3A_1B_FINALIZE_PASS", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    output_args(preflight)
    preflight.add_argument("--data-root", required=True)
    preflight.add_argument("--stop-capability-verified", action="store_true")
    preflight.add_argument("--storage-inventory-verified", action="store_true")
    preflight.add_argument("--network-volume-size-gb", type=int, required=True)
    preflight.add_argument("--network-volume-used-bytes", type=int, required=True)
    preflight.set_defaults(func=run_preflight)
    train = subparsers.add_parser("train")
    output_args(train)
    train.add_argument("--data-root", required=True)
    train.add_argument("--scientific-checkpoint-dir", required=True)
    train.add_argument("--recovery-dir", required=True)
    train.add_argument("--resume-checkpoint")
    train.add_argument("--end-update", type=int, required=True)
    train.set_defaults(func=run_train)
    finalize = subparsers.add_parser("finalize")
    output_args(finalize)
    finalize.add_argument("--data-root", required=True)
    finalize.add_argument("--final-checkpoint", required=True)
    finalize.set_defaults(func=run_finalize)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
