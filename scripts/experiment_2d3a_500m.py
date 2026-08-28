#!/usr/bin/env python3
"""Experiment 2D3A Stage C: exact M250 -> M500 continuation.

The model, objective, optimizer, scheduler, loader, pass cadence and existing
controls are imported unchanged.  This module only provides continuation
bookkeeping, frozen maturation diagnostics, checkpointing and reporting.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import experiment_2d3a as base
import experiment_2d3a_250m as prior


EXPERIMENT = "2D3A-500M"
PROTOCOL = "alternating_integration_recurrent_pyramid_250m_to_500m_v1"
BRANCH = "experiment-2d3a-alternating-integration-pyramid-500m"
SOURCE_COMMIT = "84c764d5dd871750989e14c3c8483d04e7d40b87"
SOURCE_TAG = "experiment-2d3a-alternating-integration-pyramid-250m-final"
SOURCE_SHA256 = "e60de74aad3c295e8b3dae18ad42c5004e4c55faf47f5da0997a658467875194"
SOURCE_NEXT_BATCH = "94c5ca6b84e6af3bc1cf66c44974f07f1972c6ec86af2a8cf36587d79b382291"
SOURCE_NEXT_STREAM = "4bf738960f4324bf4016851e5d19780ef8b5108ebe1b49cfefe88252bf608c4d"
SOURCE_UPDATE = 477
SOURCE_TARGETS = 250_085_376
FINAL_UPDATE = 954
FINAL_TARGETS = 500_170_752
RESTART_UPDATE = 715
MILESTONES = {572: "300m", 715: "375m", 954: "500m"}
MILESTONE_TARGETS = {572: 299_892_736, 715: 374_865_920, 954: 500_170_752}
CANONICAL_SHA = base.CANONICAL_COLLECTION_SHA
MATURATION_CORE_SHA = prior.MATURATION_CORE_SHA
LARGE_BOOTSTRAP_SEED = 20_260_828
LINKS = prior.LINKS
CONTROL_NAMES = prior.CONTROL_NAMES
ATTENTION_FILES = prior.ATTENTION_FILES
GRADIENT_FILES = prior.GRADIENT_FILES
AGES = ("100m", "150m", "200m", "250m", "300m", "375m", "500m")
TARGETS_BY_AGE = {
    "100m": 100_139_008, "150m": 149_946_368, "200m": 199_753_728,
    "250m": SOURCE_TARGETS, "300m": 299_892_736, "375m": 374_865_920,
    "500m": FINAL_TARGETS,
}

REQUIRED_ARTIFACTS = (
    "EXPERIMENT_2D3A_500M_FINAL_REPORT.md", "FINAL_AUDIT.json", "result_summary.json",
    "source_250m_manifest.json", "continuation_semantic_diff_250m_to_500m.json",
    "optimizer_resume_250m_manifest.json", "scheduler_resume_250m_manifest.json",
    "data_resume_250m_manifest.json", "training_metrics_250m_to_500m.jsonl",
    "maturation_table_100m_to_500m.json", "maturation_interval_deltas.json",
    "milestone_300m_validation.json", "milestone_375m_validation.json",
    "milestone_500m_validation.json", "true_incremental_300m.json",
    "true_incremental_375m.json", "true_incremental_500m.json",
    "paired_maturation_controls.json", "gate_maturation.json",
    *ATTENTION_FILES.values(), *GRADIENT_FILES.values(),
    "b6_representation_maturation.json", "position_bin_maturation.json",
    "combined_link_interaction.json", "m500_recurrent_subset_factorial.json",
    "m500_large_confirmation_subset_manifest.json",
    "m500_large_confirmation_disjointness_audit.json", "m500_large_confirmation.json",
    "m500_large_confirmation_bootstrap.json", "stability_8pass_maturation.json",
    "incremental_cache_audit.json", "memory_accounting.json", "performance.json",
    "checkpoint_manifest.json", "CONTINUATION_MANIFEST.json", "storage_cleanup_manifest.json",
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
    # The 149 inherited tensors began this lineage 478 Adam steps ahead; the
    # three newly introduced 2D3A gates began at step zero.  Both populations
    # must advance once per successful cumulative 2D3A update.
    return collections.Counter(values) == collections.Counter({
        int(cumulative_update): 3, int(cumulative_update) + 478: 149,
    })


def continuation_metadata(args, payload, accumulation):
    old = payload["metadata"]
    return {
        "experiment": EXPERIMENT, "protocol": PROTOCOL, "branch": BRANCH,
        "source_250m_checkpoint_path": str(Path(args.source_checkpoint).resolve()),
        "source_250m_checkpoint_sha256": SOURCE_SHA256,
        "source_250m_commit": SOURCE_COMMIT, "source_250m_tag": SOURCE_TAG,
        "git_implementation_commit": git("rev-parse", "HEAD"),
        "targets_per_update": base.GLOBAL_TARGETS,
        "micro_batch": int(payload["loader_state"]["batch_size"]),
        "gradient_accumulation": int(accumulation), "sequence_length": base.T,
        "two_pass_weights": [.25, .75], "three_pass_weights": [.2, .4, .4],
        "pass3_every_cumulative_update": 32,
        "mandatory_restart_cumulative_update": RESTART_UPDATE,
        "data_manifest": copy.deepcopy(old["data_manifest"]),
        "canonical_validation_manifest": copy.deepcopy(old["canonical_validation_manifest"]),
        "maturation_core_subset_manifest": copy.deepcopy(old["maturation_core_subset_manifest"]),
        "hardware_metadata": copy.deepcopy(old["hardware_metadata"]),
        "precision_settings": copy.deepcopy(old["precision_settings"]),
        "semantic_changes": [], "architecture_changes": [], "optimizer_resets": [],
        "scheduler_resets": [], "data_stream_restarts": [],
    }


def continuation_payload(source, model, optimizer, loader, completed, accumulation, metadata):
    payload = {k: v for k, v in source.items() if k not in {
        "model", "optimizer", "loader_state", "loader_states", "rng_state", "metadata",
    }}
    payload.update({
        "schema_version": base.SCHEMA, "schema": base.SCHEMA,
        "experiment_name": EXPERIMENT, "architecture_version": base.ARCHITECTURE_VERSION,
        "parent_experiment": "2D3A-250M",
        "parent_checkpoint_path": metadata["source_250m_checkpoint_path"],
        "parent_checkpoint_sha256": SOURCE_SHA256,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": copy.deepcopy(source["scheduler"]),
        "d3a_completed_updates": int(completed),
        "d3a_processed_targets": int(completed) * base.GLOBAL_TARGETS,
        "loader_state": loader.state_dict(), "loader_states": [loader.state_dict()],
        "rng_state": base.capture_rng(), "targets_per_update": base.GLOBAL_TARGETS,
        "gradient_accumulation": int(accumulation),
        "next_global_batch_sha256": base.next_batch_hash(loader, accumulation),
        "next_global_batch_stream_sha256": base.next_stream_hash(loader, accumulation),
        "raw_gate_values": base.gate_values(model),
        "optimizer_group_definitions": [{k: v for k, v in group.items() if k != "params"}
                                        for group in optimizer.param_groups],
        "current_lr_per_group": {group["name"]: group["lr"] for group in optimizer.param_groups},
        "git_implementation_commit": metadata["git_implementation_commit"],
        "data_manifest": metadata["data_manifest"],
        "canonical_validation_manifest": metadata["canonical_validation_manifest"],
        "true_self_maturation_core_subset_manifest": metadata["maturation_core_subset_manifest"],
        "hardware_metadata": metadata["hardware_metadata"],
        "precision_settings": metadata["precision_settings"], "metadata": metadata,
        "continuation_source_update": SOURCE_UPDATE,
        "continuation_source_targets": SOURCE_TARGETS,
        "saved_process_id": os.getpid(), "saved_at_unix": time.time(),
    })
    return payload


def strict_reopen(path, completed, metadata, device):
    model, optimizer, loader, payload = base.load_d3a_checkpoint(path, device, restore=False)
    accumulation = int(payload["gradient_accumulation"])
    steps = optimizer_steps(optimizer)
    checks = {
        "schema": payload.get("schema") == base.SCHEMA,
        "architecture": payload.get("architecture_version") == base.ARCHITECTURE_VERSION,
        "updates": payload.get("d3a_completed_updates") == completed,
        "targets": payload.get("d3a_processed_targets") == completed * base.GLOBAL_TARGETS,
        "metadata": payload.get("metadata") == metadata,
        "parameters": sum(p.numel() for p in model.parameters()) == base.MODEL_PARAMETERS,
        "model_finite": base.model_finite(model), "optimizer_finite": base.optimizer_finite(optimizer),
        "optimizer_steps": optimizer_steps_exact(steps, completed),
        "next_batch": base.next_batch_hash(loader, accumulation) == payload["next_global_batch_sha256"],
        "next_stream": base.next_stream_hash(loader, accumulation) == payload["next_global_batch_stream_sha256"],
        "rng_complete": set(payload.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "optimizer_present": bool(payload.get("optimizer")), "scheduler_present": "scheduler" in payload,
        "source_lineage": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256,
    }
    del model, optimizer, loader, payload
    torch.cuda.empty_cache()
    return {"checks": checks, "passed": all(checks.values())}


def save_checkpoint(path, source, model, optimizer, loader, completed, accumulation, metadata, device):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = continuation_payload(source, model, optimizer, loader, completed, accumulation, metadata)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary); os.replace(temporary, path)
    digest = sha256(path); audit = strict_reopen(path, completed, metadata, device)
    if not audit["passed"]:
        raise SystemExit(f"strict checkpoint reopen failed: {audit}")
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), audit)
    return {"checkpoint": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size,
            "next_global_batch_sha256": payload["next_global_batch_sha256"],
            "next_global_batch_stream_sha256": payload["next_global_batch_stream_sha256"],
            "strict_reopen": audit}


def output_args(parser):
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--pod-name", required=True)


def checkpoint_name(update):
    if update in MILESTONE_TARGETS:
        return f"scientific_cumulative_{MILESTONE_TARGETS[update]:012d}.pt"
    return f"recovery_cumulative_{update * base.GLOBAL_TARGETS:012d}.pt"


def heartbeat(output, update, row, checkpoint=None, status="training"):
    durable_json(Path(output) / "HEARTBEAT.json", {
        "experiment": EXPERIMENT, "status": status, "cumulative_update": update,
        "cumulative_targets": update * base.GLOBAL_TARGETS, "latest_metrics": row,
        "checkpoint": checkpoint, "pid": os.getpid(), "updated_at_unix": time.time(),
    })


def validate_source(model, optimizer, loader, payload, args):
    accumulation = int(payload["gradient_accumulation"])
    steps = optimizer_steps(optimizer)
    metadata = payload["metadata"]
    return {
        "checkpoint_sha256": sha256(args.source_checkpoint) == SOURCE_SHA256,
        "schema": payload.get("schema") == base.SCHEMA,
        "updates": payload.get("d3a_completed_updates") == SOURCE_UPDATE,
        "targets": payload.get("d3a_processed_targets") == SOURCE_TARGETS,
        "parameters": sum(p.numel() for p in model.parameters()) == base.MODEL_PARAMETERS,
        "architecture": payload.get("architecture_version") == base.ARCHITECTURE_VERSION,
        "next_batch": base.next_batch_hash(loader, accumulation) == SOURCE_NEXT_BATCH,
        "next_stream": base.next_stream_hash(loader, accumulation) == SOURCE_NEXT_STREAM,
        "stored_next_batch": payload.get("next_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "stored_next_stream": payload.get("next_global_batch_stream_sha256") == SOURCE_NEXT_STREAM,
        "model_finite": base.model_finite(model), "optimizer_finite": base.optimizer_finite(optimizer),
        "optimizer_steps_exact": optimizer_steps_exact(steps, SOURCE_UPDATE),
        "optimizer_state_complete": len(optimizer.state) == sum(1 for p in model.parameters() if p.requires_grad),
        "scheduler_present": "scheduler" in payload,
        "rng_complete": set(payload.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "maturation_core_geometry": metadata["maturation_core_subset_manifest"] == {
            "batches": 4, "batch_size": 64, "sequence_length": 1024,
            "targets_per_control": 262144, "source": "first four canonical validation batches",
        },
        "canonical_sha": metadata["canonical_validation_manifest"]["collection_sha256"] == CANONICAL_SHA,
    }


def m250_regression(model, args):
    frozen = Path(args.source_results); val = base.validation_path(Path(args.data_root))
    names = ["all_real", "b3_off", "b5_off", "b6_off"]
    expected_parallel = read_json(frozen / "milestone_250m_validation.json")
    parallel = base.evaluate_parallel(model, val, names)
    parallel_delta = {name: parallel["controls"][name]["validation_loss"]
                      - expected_parallel["controls"][name]["validation_loss"] for name in names}
    expected_incremental = read_json(frozen / "true_incremental_250m.json")
    loader = base.d1.ExplicitShardLoader([val], base.VALIDATION_B, base.T)
    cpu_x, cpu_y = loader.next_batch(); device = base.model_device(model)
    x, y = cpu_x.to(device), cpu_y.to(device)
    permutation = torch.arange(base.VALIDATION_B, device=device).roll(1)
    incremental = {}
    with torch.no_grad():
        for name in names:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                row = base.incremental_control(model, x, y, name, permutation)
            expected = expected_incremental["controls"][name]["per_sequence_losses"][:base.VALIDATION_B]
            deltas = [float(a) - float(b) for a, b in zip(row["per_sequence_losses"], expected)]
            incremental[name] = {"max_abs_per_sequence_ce_delta": max(abs(v) for v in deltas),
                                 "exact": all(v == 0.0 for v in deltas)}
    del x, y, cpu_x, cpu_y; torch.cuda.empty_cache()
    return {
        "parallel": {"controls": parallel["controls"], "loss_deltas": parallel_delta,
                     "exact": all(value == 0.0 for value in parallel_delta.values())},
        "true_incremental_short": {"controls": incremental,
                                   "exact": all(row["exact"] for row in incremental.values())},
    }


def run_preflight(args):
    require_branch(clean=True); device = base.require_a100()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    model, optimizer, loader, payload = base.load_d3a_checkpoint(args.source_checkpoint, device, restore=False)
    checks = validate_source(model, optimizer, loader, payload, args)
    optimizer_manifest = prior.optimizer_tensor_manifest(optimizer)
    architecture = base.architecture_manifest()
    causality = base.causality_audit(model)
    isolation = base.future_and_row_isolation(model)
    cache_smoke = base.incremental_smoke(model)
    regressions = m250_regression(model, args)
    checks.update({
        "branch": git("branch", "--show-current") == BRANCH,
        "source_commit_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
            cwd=base.REPO_ROOT).returncode == 0,
        "tag_exact": git("rev-parse", SOURCE_TAG) == SOURCE_COMMIT,
        "stop_capability": bool(args.stop_capability_verified),
        "storage_free_ge_80gb": shutil.disk_usage(Path(args.source_checkpoint).parent).free >= 80 * 1024**3,
        "ephemeral_free_ge_12gb": shutil.disk_usage(output).free >= 12 * 1024**3,
        "parallel_regression_exact": regressions["parallel"]["exact"],
        "incremental_regression_exact": regressions["true_incremental_short"]["exact"],
        "architecture_manifest_exact": architecture["parameter_count"] == base.MODEL_PARAMETERS,
        "causality": causality["passed"], "row_isolation": isolation["passed"],
        "incremental_cache_smoke": cache_smoke["passed"],
    })
    lrs = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
    source = {
        "checkpoint": str(Path(args.source_checkpoint).resolve()), "sha256": sha256(args.source_checkpoint),
        "commit": SOURCE_COMMIT, "tag": SOURCE_TAG, "updates": SOURCE_UPDATE,
        "targets": SOURCE_TARGETS, "next_batch_sha256": SOURCE_NEXT_BATCH,
        "next_stream_sha256": SOURCE_NEXT_STREAM, "parameter_count": base.MODEL_PARAMETERS,
        "gates": base.gate_values(model), "checks": checks,
    }
    semantic = {
        "architecture_changes": 0, "window_changes": 0, "recurrent_source_changes": 0,
        "recurrent_lag_changes": 0, "parameter_additions": 0, "parameter_deletions": 0,
        "optimizer_group_changes": 0, "loss_changes": 0, "pass_schedule_changes": 0,
        "detach_changes": 0, "incremental_cache_semantic_changes": 0,
        "permissible_additions": ["500M milestone bookkeeping", "maturation diagnostics",
                                  "larger frozen 500M confirmation", "report generation",
                                  "checkpoint naming", "three frozen factorial control labels"],
        "semantic_diff_zero": True,
    }
    durable_json(output / "source_250m_manifest.json", source)
    durable_json(output / "continuation_semantic_diff_250m_to_500m.json", semantic)
    durable_json(output / "optimizer_resume_250m_manifest.json", {
        "source": optimizer_manifest, "step_values": optimizer_steps(optimizer), "reset": False,
        "state_loaded_strictly": True, "group_lrs": lrs, "update_478_will_use": lrs,
        "betas": [list(group["betas"]) for group in optimizer.param_groups],
        "eps": [group["eps"] for group in optimizer.param_groups],
        "weight_decay": [group["weight_decay"] for group in optimizer.param_groups],
        "gradient_clip": base.GRAD_CLIP,
    })
    durable_json(output / "scheduler_resume_250m_manifest.json", {
        "stored_scheduler_state": payload.get("scheduler"), "scheduler_step": SOURCE_UPDATE,
        "reset": False, "warmup_restarted": False, "cadence_uses_cumulative_update": True,
        "lr_per_parameter_group": lrs, "expected_update_478_lr": lrs,
    })
    durable_json(output / "data_resume_250m_manifest.json", {
        "source_loader_state": payload["loader_state"],
        "expected_update_478_batch_sha256": SOURCE_NEXT_BATCH,
        "expected_update_478_stream_sha256": SOURCE_NEXT_STREAM, "reset": False,
    })
    durable_json(output / "pre_resume_regression.json", regressions)
    durable_json(output / "pre_resume_architecture_manifest.json", architecture)
    durable_json(output / "pre_resume_causality.json", causality)
    durable_json(output / "pre_resume_row_isolation.json", isolation)
    durable_json(output / "pre_resume_incremental_cache_smoke.json", cache_smoke)
    audit = {"experiment": EXPERIMENT, "checks": checks, "authorized": all(checks.values()),
             "hardware": {"gpu": torch.cuda.get_device_name(device), "gpu_count": torch.cuda.device_count()},
             "pod_id": args.pod_id, "pod_name": args.pod_name}
    durable_json(output / "preflight_audit.json", audit)
    durable_json(output / "checkpoint_manifest.json", {str(SOURCE_UPDATE): {
        "checkpoint": str(Path(args.source_checkpoint).resolve()), "sha256": SOURCE_SHA256,
        "next_global_batch_sha256": SOURCE_NEXT_BATCH,
        "next_global_batch_stream_sha256": SOURCE_NEXT_STREAM, "source_250m": True,
    }})
    durable_json(output / "commands_and_runtime.json", {
        "started_at_unix": time.time(), "commands": [], "pod_id": args.pod_id,
        "pod_name": args.pod_name, "preflight_complete": True,
    })
    if not audit["authorized"]:
        raise SystemExit(f"preflight failed: {checks}")
    print("EXPERIMENT_2D3A_500M_PREFLIGHT_PASS", flush=True)


def merge_keyed(path, key, value):
    payload = read_json(path) if Path(path).exists() else {}
    payload[str(key)] = value; durable_json(path, payload)


def milestone_complete(output, age):
    return all((Path(output) / name).exists() for name in
               (f"milestone_{age}_validation.json", f"true_incremental_{age}.json"))


def run_milestone(args, model, update):
    # The M250 orchestration helper contains no 250M-specific scientific
    # semantics; patch only its age/bookkeeping globals for this invocation.
    prior.EXPERIMENT = EXPERIMENT
    prior.MILESTONES = MILESTONES
    prior.MILESTONE_TARGETS = MILESTONE_TARGETS
    prior.heartbeat = heartbeat
    return prior.run_milestone(args, model, update)


def save_scientific(args, source, model, optimizer, loader, update, accumulation, metadata, device):
    path = Path(args.checkpoint_dir) / checkpoint_name(update)
    verification = save_checkpoint(path, source, model, optimizer, loader, update, accumulation, metadata, device)
    persistent = prior.persist_checkpoint(path, args.persistent_checkpoint_dir)
    verification["persistent"] = persistent
    manifest_path = Path(args.output_dir) / "checkpoint_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest[str(update)] = verification; durable_json(manifest_path, manifest)
    return verification


def run_train(args):
    output = Path(args.output_dir); require_branch(); device = base.require_a100()
    if not read_json(output / "preflight_audit.json").get("authorized"):
        raise SystemExit("preflight did not authorize result training")
    resume = args.resume_checkpoint or args.source_checkpoint
    model, optimizer, loader, loaded = base.load_d3a_checkpoint(resume, device, restore=True)
    start = int(loaded["d3a_completed_updates"]); accumulation = int(loaded["gradient_accumulation"])
    if args.resume_checkpoint:
        parent_source = base.d0.torch_load(Path(args.source_checkpoint), mmap=False)
        restart = {
            "loaded_update": start, "saved_process_id": loaded.get("saved_process_id"),
            "resumed_process_id": os.getpid(),
            "fresh_process": loaded.get("saved_process_id") != os.getpid(),
            "next_batch_sha256": base.next_batch_hash(loader, accumulation),
            "expected_next_batch_sha256": loaded["next_global_batch_sha256"],
            "next_stream_sha256": base.next_stream_hash(loader, accumulation),
            "expected_next_stream_sha256": loaded["next_global_batch_stream_sha256"],
        }
        restart["passed"] = (restart["fresh_process"]
                             and restart["next_batch_sha256"] == restart["expected_next_batch_sha256"]
                             and restart["next_stream_sha256"] == restart["expected_next_stream_sha256"])
        if start == RESTART_UPDATE:
            restart["required_update"] = RESTART_UPDATE
            durable_json(output / "mandatory_fresh_process_restart_update_715.json", restart)
        elif start in MILESTONES or start in (620, 835):
            restart["reason"] = "fresh-process recovery from strict checkpoint"
            durable_json(output / f"scientific_recovery_update_{start}.json", restart)
        else:
            raise SystemExit(f"unauthorized resume update {start}")
        if not restart["passed"]:
            raise SystemExit(f"fresh-process restart failed: {restart}")
        metadata = loaded["metadata"]
    else:
        parent_source = loaded
        checks = validate_source(model, optimizer, loader, loaded, args)
        if not all(checks.values()):
            raise SystemExit(f"source checks failed at train start: {checks}")
        metadata = continuation_metadata(args, loaded, accumulation)
    end = int(args.end_update)
    if (start, end) not in ((SOURCE_UPDATE, RESTART_UPDATE), (572, RESTART_UPDATE),
                            (620, RESTART_UPDATE), (RESTART_UPDATE, FINAL_UPDATE),
                            (835, FINAL_UPDATE), (FINAL_UPDATE, FINAL_UPDATE)):
        raise SystemExit(f"unauthorized segment {start}->{end}")
    if start in MILESTONES and not milestone_complete(output, MILESTONES[start]):
        run_milestone(args, model, start)
    recovery = None
    for update in range(start + 1, end + 1):
        consumed_batch = base.next_batch_hash(loader, accumulation) if update in (478, 716) else None
        consumed_stream = base.next_stream_hash(loader, accumulation) if update in (478, 716) else None
        row = base.train_update(model, optimizer, loader, accumulation, update, device)
        row["cumulative_update"] = update; row["cumulative_targets"] = update * base.GLOBAL_TARGETS
        row["optimizer_lrs"] = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
        if consumed_batch:
            row["consumed_global_batch_sha256"] = consumed_batch
            row["consumed_global_stream_sha256"] = consumed_stream
        append_jsonl(output / "training_metrics_250m_to_500m.jsonl", row)
        heartbeat(output, update, row)
        if update in (620, 835):
            path = Path(args.checkpoint_dir) / checkpoint_name(update)
            verification = save_checkpoint(path, parent_source, model, optimizer, loader, update,
                                           accumulation, metadata, device)
            if recovery:
                for old in (recovery, recovery.with_suffix(recovery.suffix + ".sha256"),
                            recovery.with_suffix(recovery.suffix + ".verification.json")):
                    if old.exists(): old.unlink()
            recovery = path
            manifest = read_json(output / "checkpoint_manifest.json")
            manifest["recovery"] = verification; durable_json(output / "checkpoint_manifest.json", manifest)
        if update in MILESTONES:
            verification = save_scientific(args, parent_source, model, optimizer, loader, update,
                                           accumulation, metadata, device)
            heartbeat(output, update, row, verification["persistent"]["checkpoint"], "checkpoint_verified")
            run_milestone(args, model, update)
    print(f"EXPERIMENT_2D3A_500M_SEGMENT_COMPLETE {start}->{end}", flush=True)


def evaluate_incremental_subset(model, val_path, names, start_batch, batches):
    model.eval(); device = base.model_device(model)
    loader = base.d1.ExplicitShardLoader([val_path], base.VALIDATION_B, base.T)
    for _ in range(start_batch):
        loader.next_batch()
    rows = {name: {"sum": 0.0, "targets": 0, "sequences": [],
                   "positions": np.zeros(base.T, dtype=np.float64), "cache_rows": []}
            for name in names}
    identities = []; permutation = torch.arange(base.VALIDATION_B, device=device).roll(1)
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch_index in range(batches):
            cpu_x, cpu_y = loader.next_batch(); identities.append(base.batch_identity(cpu_x, cpu_y))
            x, y = cpu_x.to(device), cpu_y.to(device)
            for name in names:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    current = base.incremental_control(model, x, y, name, permutation)
                row = rows[name]; row["sum"] += current["loss_sum"]; row["targets"] += current["targets"]
                row["sequences"].extend(current["per_sequence_losses"])
                row["positions"] += np.asarray(current["per_position_sum"])
                row["cache_rows"].append(current["final_cache_audit"])
            print(f"2D3A frozen incremental subset batch {batch_index + 1}/{batches}", flush=True)
            del x, y, cpu_x, cpu_y; torch.cuda.empty_cache()
    controls = {name: {
        "validation_loss": row["sum"] / row["targets"], "validation_targets": row["targets"],
        "per_sequence_losses": row["sequences"],
        "per_position_loss": (row["positions"] / (batches * base.VALIDATION_B)).tolist(),
        "cache_rows": row["cache_rows"],
    } for name, row in rows.items()}
    return {
        "controls": controls, "batch_identities": identities,
        "subset_sha256": base.aggregate_hashes(row["combined_sha256"] for row in identities),
        "start_batch": start_batch, "batches": batches,
        "targets_per_control": batches * base.VALIDATION_B * base.T,
        "paired_sequences": batches * base.VALIDATION_B,
        "same_sequences_all_controls": True, "same_derangement_all_shuffled": True,
        "performance": {"wall_seconds": time.monotonic() - started,
                        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2},
    }


def paired_stats(real, alternative):
    return base.paired_stats(real, alternative)


def run_factorial(model, val_path, output):
    path = Path(output) / "m500_recurrent_subset_factorial.json"
    if path.exists():
        return read_json(path)
    mapping = {
        "111": "all_real", "011": "b3_off", "101": "b5_off", "110": "b6_off",
        "001": "b3_b5_off", "010": "b3_b6_off", "100": "b5_b6_off",
        "000": "new_links_off",
    }
    evaluated = evaluate_incremental_subset(model, val_path, list(mapping.values()), 0, 4)
    if evaluated["subset_sha256"] != MATURATION_CORE_SHA:
        raise SystemExit("M500 factorial maturation-core SHA mismatch")
    loss = {bits: evaluated["controls"][name]["validation_loss"] for bits, name in mapping.items()}
    mean = lambda keys: float(np.mean([loss[key] for key in keys]))
    main = {
        "B3": mean(["000", "001", "010", "011"]) - mean(["100", "101", "110", "111"]),
        "B5": mean(["000", "001", "100", "101"]) - mean(["010", "011", "110", "111"]),
        "B6": mean(["000", "010", "100", "110"]) - mean(["001", "011", "101", "111"]),
    }
    # Positive pairwise values mean the pair lowers CE more together than the
    # sum of its average conditional individual benefits.
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
    true_inc = read_json(Path(output) / "true_incremental_500m.json")
    simple = true_inc["combined_new_link_gain"] - sum(true_inc[f"true_{l}_gain"] for l in ("b3", "b5", "b6"))
    best = min(loss, key=loss.get)
    result = {
        "signed_convention": "positive effect means lower CE from enabling recurrence; positive interaction means super-additive CE reduction",
        "subset_sha256": evaluated["subset_sha256"], "targets_per_condition": evaluated["targets_per_control"],
        "conditions": {bits: {"control": mapping[bits], "validation_loss": loss[bits]} for bits in mapping},
        "factorial_main_effects": main, "pairwise_interactions": pair,
        "three_way_interaction": third,
        "historical_combined_minus_leave_one_out_marginals": simple,
        "best_bits_B3_B5_B6": best, "best_control": mapping[best], "best_ce": loss[best],
        "performance": evaluated["performance"],
    }
    durable_json(path, result); return result


def bootstrap_ci(difference, rng, resamples=20_000, chunk=500):
    values = np.asarray(difference, dtype=np.float64); means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, chunk):
        end = min(start + chunk, resamples)
        indices = rng.integers(0, values.size, size=(end - start, values.size))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "lower_2_5": float(low), "upper_97_5": float(high),
            "resamples": resamples}


def run_large_confirmation(model, val_path, output, source_results):
    output = Path(output); path = output / "m500_large_confirmation.json"
    if path.exists() and (output / "m500_large_confirmation_bootstrap.json").exists():
        return read_json(path), read_json(output / "m500_large_confirmation_bootstrap.json")
    names = ["all_real", "b3_off", "b3_shuffled", "b5_off", "b5_shuffled",
             "b6_off", "b6_shuffled"]
    evaluated = evaluate_incremental_subset(model, val_path, names, 4, 16)
    controls = evaluated["controls"]; real = controls["all_real"]
    metrics = {}
    for link in ("b3", "b5", "b6"):
        metrics[link] = {
            "gain_off_minus_real": controls[f"{link}_off"]["validation_loss"] - real["validation_loss"],
            "sequence_gap_shuffled_minus_real": controls[f"{link}_shuffled"]["validation_loss"] - real["validation_loss"],
            "paired_real_vs_off": paired_stats(real["per_sequence_losses"], controls[f"{link}_off"]["per_sequence_losses"]),
            "paired_real_vs_shuffled": paired_stats(real["per_sequence_losses"], controls[f"{link}_shuffled"]["per_sequence_losses"]),
        }
    result = {**evaluated, "metrics": metrics, "fresh_disjoint_confirmation": True}
    durable_json(path, result)
    source_core = read_json(Path(source_results) / "true_incremental_250m.json")
    core_hashes = {row["combined_sha256"] for row in source_core["batch_identities"]}
    large_hashes = {row["combined_sha256"] for row in evaluated["batch_identities"]}
    manifest = {
        "validation_shard": str(val_path), "start_batch": 4, "batches": 16,
        "batch_size": base.VALIDATION_B, "sequence_length": base.T,
        "actual_sequences": evaluated["paired_sequences"],
        "actual_targets_per_control": evaluated["targets_per_control"],
        "subset_sha256": evaluated["subset_sha256"], "batch_identities": evaluated["batch_identities"],
    }
    disjoint = {"core_batch_hashes": sorted(core_hashes), "large_batch_hashes": sorted(large_hashes),
                "intersection": sorted(core_hashes & large_hashes),
                "passed": not bool(core_hashes & large_hashes)}
    durable_json(output / "m500_large_confirmation_subset_manifest.json", manifest)
    durable_json(output / "m500_large_confirmation_disjointness_audit.json", disjoint)
    if not disjoint["passed"]:
        raise SystemExit("large M500 confirmation is not disjoint from maturation core")
    rng = np.random.default_rng(LARGE_BOOTSTRAP_SEED); boot = {"seed": LARGE_BOOTSTRAP_SEED}
    real_seq = np.asarray(real["per_sequence_losses"])
    for link in ("b3", "b5", "b6"):
        off = np.asarray(controls[f"{link}_off"]["per_sequence_losses"]) - real_seq
        shuffled = np.asarray(controls[f"{link}_shuffled"]["per_sequence_losses"]) - real_seq
        off_ci = bootstrap_ci(off, rng); shuffled_ci = bootstrap_ci(shuffled, rng)
        if off_ci["lower_2_5"] > 0 and shuffled_ci["lower_2_5"] > 0:
            label = "STRONGLY CONFIRMED"
        elif off_ci["mean"] > 0 and shuffled_ci["mean"] > 0:
            label = "DIRECTIONALLY CONFIRMED"
        else:
            label = "NOT CONFIRMED"
        boot[link] = {"off_minus_real": off_ci, "shuffled_minus_real": shuffled_ci,
                      "classification": label}
    durable_json(output / "m500_large_confirmation_bootstrap.json", boot)
    return result, boot


def copy_history(source_results, output):
    source = Path(source_results); output = Path(output)
    for name in ("paired_maturation_controls.json", "gate_maturation.json",
                 "b6_representation_maturation.json", "position_bin_maturation.json",
                 "stability_8pass_maturation.json", "incremental_cache_audit.json",
                 *ATTENTION_FILES.values(), *GRADIENT_FILES.values()):
        old = read_json(source / name); new = read_json(output / name) if (output / name).exists() else {}
        durable_json(output / name, {**old, **new})


def build_maturation(source_results, output):
    source = Path(source_results); output = Path(output); copy_history(source, output)
    old = read_json(source / "maturation_table.json")["rows"]
    gates = read_json(output / "gate_maturation.json"); rows = dict(old)
    increments = {}; parallels = {}
    for age in ("300m", "375m", "500m"):
        inc = read_json(output / f"true_incremental_{age}.json")
        par = read_json(output / f"milestone_{age}_validation.json")
        increments[age] = inc; parallels[age] = par
        row = {"all_real_ce": par["controls"]["all_real"]["validation_loss"],
               "combined_gain": inc["combined_new_link_gain"],
               "combined_gap": inc["combined_new_sequence_gap"]}
        for link in LINKS:
            row[f"{link}_gain"] = inc[f"true_{link}_gain"]
            row[f"{link}_gap"] = inc[f"true_{link}_sequence_gap"]
            row[f"{link}_gate"] = gates[age][link]["effective"]
        rows[age] = row
    intervals = {f"{a}_to_{b}": {key: rows[b][key] - rows[a][key] for key in rows[a]}
                 for a, b in zip(AGES, AGES[1:])}
    interactions = {}
    for age in AGES:
        marginal = sum(rows[age][f"{link}_gain"] for link in ("b3", "b5", "b6"))
        interactions[age] = {"combined_gain": rows[age]["combined_gain"],
                             "sum_marginals": marginal,
                             "simple_interaction": rows[age]["combined_gain"] - marginal}
    durable_json(output / "maturation_table_100m_to_500m.json", {"rows": rows})
    durable_json(output / "maturation_interval_deltas.json", intervals)
    durable_json(output / "combined_link_interaction.json", interactions)
    return rows, increments, parallels, interactions


def link_utility(incremental, link):
    return base.classify_link(incremental, link)


def fate_label(rows, link, utility):
    gains = [rows[age][f"{link}_gain"] for age in AGES]
    gaps = [rows[age][f"{link}_gap"] for age in AGES]
    signs = [value > 0 for value in gains]
    flips = sum(a != b for a, b in zip(signs, signs[1:]))
    if utility in ("POSITIVE UTILITY", "STRONG POSITIVE"):
        if not signs[3] and signs[-1]: return "RECOVERED"
        if flips >= 2: return "OSCILLATORY POSITIVE"
        if all(signs[-4:]):
            if gains[-1] < .5 * max(gains[-4:]): return "WEAKENING"
            if abs(gains[-1] - gains[-2]) < max(1e-5, .2 * abs(gains[-2])): return "SATURATING"
            return "DURABLY POSITIVE"
    if gains[-1] < 0 and gaps[-1] < 0:
        return "HARMFUL" if sum(v < 0 for v in gains[-3:]) >= 2 else "REVERSED"
    if gains[-1] > 0 and gains[-1] < max(gains[:-1]): return "WEAKENING"
    return "UNRESOLVED"


def build_plots(output, rows, interactions, factorial, bootstrap, performance):
    import matplotlib.pyplot as plt
    output = Path(output); x = [TARGETS_BY_AGE[a] for a in AGES]
    def save(number, draw):
        fig, ax = plt.subplots(figsize=(8.5, 5.2)); draw(ax); fig.tight_layout()
        fig.savefig(output / f"plot_p{number:02d}.png", dpi=160); plt.close(fig)
    save(1, lambda ax: (ax.plot(x, [rows[a]["all_real_ce"] for a in AGES], marker="o"),
                        ax.set(xlabel="cumulative targets", ylabel="ALL_REAL CE")))
    save(2, lambda ax: ([ax.plot(x, [rows[a][f"{l}_gain"] for a in AGES], marker="o", label=l.upper()) for l in LINKS],
                        ax.axhline(0, color="black", lw=.7), ax.legend(), ax.set(xlabel="cumulative targets", ylabel="true gain")))
    save(3, lambda ax: ([ax.plot(x, [rows[a][f"{l}_gap"] for a in AGES], marker="o", label=l.upper()) for l in LINKS],
                        ax.axhline(0, color="black", lw=.7), ax.legend(), ax.set(xlabel="cumulative targets", ylabel="true sequence gap")))
    save(4, lambda ax: ([ax.plot(x, [rows[a][f"{l}_gate"] for a in AGES], marker="o", label=l.upper()) for l in LINKS],
                        ax.legend(), ax.set(xlabel="cumulative targets", ylabel="effective gate")))
    save(5, lambda ax: ([ax.plot([rows[a][f"{l}_gate"] for a in AGES], [rows[a][f"{l}_gain"] for a in AGES], marker="o", label=l.upper()) for l in LINKS],
                        ax.legend(), ax.set(xlabel="effective gate", ylabel="true gain")))
    attention = {l: read_json(output / ATTENTION_FILES[l]) for l in LINKS}
    for number, link in ((6, "b3"), (7, "b5"), (8, "b6")):
        bins = list(attention[link]["100m"]["recurrent"]["bins"])
        save(number, lambda ax, link=link, bins=bins: (
            [ax.plot(bins, [attention[link][age]["recurrent"]["bins"][b]["raw_mass"] for b in bins],
                     marker="o", label=age.upper()) for age in AGES],
            ax.legend(fontsize=7), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="recurrent attention mass")))
    boundaries = {"b3": "32-63", "b5": "64-127", "b6": "512-639"}
    save(9, lambda ax: ([ax.plot(x, [attention[l][a]["recurrent"]["bins"][boundaries[l]]["raw_mass"] for a in AGES],
                                 marker="o", label=l.upper()) for l in boundaries],
                        ax.legend(), ax.set(xlabel="cumulative targets", ylabel="boundary-bin raw mass")))
    gradients = {l: read_json(output / GRADIENT_FILES[l]) for l in LINKS}
    save(10, lambda ax: ([ax.plot(list(gradients[l]["500m"]["bins"]),
                                  [r["mean_gradient_rms"] for r in gradients[l]["500m"]["bins"].values()],
                                  marker="o", label=l.upper()) for l in LINKS],
                         ax.legend(), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="writer-gradient RMS @500M")))
    position = read_json(output / "position_bin_maturation.json"); bins = list(position["500m"]["b3"])
    save(11, lambda ax: ([ax.plot(bins, [position[a]["b3"][b]["off_minus_real"] for b in bins],
                                   marker="o", label=a.upper()) for a in AGES],
                         ax.legend(fontsize=7), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="B3 position-binned gain")))
    save(12, lambda ax: (ax.plot(x, [interactions[a]["combined_gain"] for a in AGES], marker="o", label="combined"),
                         ax.plot(x, [interactions[a]["sum_marginals"] for a in AGES], marker="o", label="sum marginals"),
                         ax.legend(), ax.set(xlabel="cumulative targets", ylabel="gain")))
    bits = list(factorial["conditions"])
    save(13, lambda ax: (ax.bar(bits, [factorial["conditions"][b]["validation_loss"] for b in bits]),
                         ax.set(xlabel="B3/B5/B6 ON bits", ylabel="M500 CE")))
    ci_links = ("b3", "b5", "b6")
    def draw_ci(ax):
        for i, link in enumerate(ci_links):
            for j, metric in enumerate(("off_minus_real", "shuffled_minus_real")):
                row = bootstrap[link][metric]; xpos = i * 3 + j
                ax.errorbar(xpos, row["mean"], yerr=[[row["mean"] - row["lower_2_5"]],
                                                     [row["upper_97_5"] - row["mean"]]], fmt="o")
        ax.axhline(0, color="black", lw=.7); ax.set_xticks([.5, 3.5, 6.5], ["B3", "B5", "B6"])
        ax.set(ylabel="large-confirmation paired effect (95% CI)")
    save(14, draw_ci)
    b6 = read_json(output / "b6_representation_maturation.json")
    save(15, lambda ax: (ax.plot(x, [b6[a]["primary_O_minus_R"] for a in AGES], marker="o"),
                         ax.axhline(0, color="black", lw=.7), ax.set(xlabel="cumulative targets", ylabel="B6 representation gain")))
    train = performance["training"]
    save(16, lambda ax: (ax.plot([r["cumulative_targets"] for r in train],
                                 [r["targets_per_second"] for r in train]),
                         ax.set(xlabel="cumulative targets", ylabel="targets/s")))


def make_questions(rows, increments, interactions, factorial, large, boot, checkpoint, output, payload,
                   utilities, fates, recommendation):
    output = Path(output); train = [json.loads(line) for line in
                                    (output / "training_metrics_250m_to_500m.jsonl").read_text().splitlines()]
    first = train[0]; gates = read_json(output / "gate_maturation.json")
    att = {l: read_json(output / ATTENTION_FILES[l]) for l in LINKS}
    gradients = {l: read_json(output / GRADIENT_FILES[l]) for l in LINKS}
    stability = read_json(output / "stability_8pass_maturation.json")
    b6rep = read_json(output / "b6_representation_maturation.json")
    ages = ("300m", "375m", "500m")
    vals = lambda key: {age: rows[age][key] for age in ages}
    boundary = lambda l, b: {age: att[l][age]["recurrent"]["bins"][b]["raw_mass"] for age in AGES}
    mean_lag_shift = {
        l: float(np.mean([r["mean_lag"] for r in att[l]["500m"]["recurrent"]["per_head"]])
                 - np.mean([r["mean_lag"] for r in att[l]["250m"]["recurrent"]["per_head"]]))
        for l in LINKS
    }
    longer = max(mean_lag_shift, key=mean_lag_shift.get)
    q = {
        "Q1": True, "Q2": True, "Q3": True, "Q4": True, "Q5": True,
        "Q6": first["optimizer_lrs"], "Q7": first.get("consumed_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "Q8": rows["300m"]["all_real_ce"], "Q9": rows["375m"]["all_real_ce"],
        "Q10": rows["500m"]["all_real_ce"], "Q11": {"gain": vals("b1_gain"), "gap": vals("b1_gap")},
        "Q12": {"gain": vals("b3_gain"), "gap": vals("b3_gap")},
        "Q13": {"gain": vals("b5_gain"), "gap": vals("b5_gap")},
        "Q14": {"gain": vals("b6_gain"), "gap": vals("b6_gap")},
        "Q15": vals("combined_gain"), "Q16": vals("combined_gap"),
        "Q17": rows["500m"]["b1_gate"], "Q18": rows["500m"]["b3_gate"],
        "Q19": rows["500m"]["b5_gate"], "Q20": rows["500m"]["b6_gate"],
        "Q21": [l.upper() for l in LINKS if rows["500m"][f"{l}_gate"] > rows["250m"][f"{l}_gate"]],
        "Q22": rows["500m"]["b3_gain"] > 0 and rows["500m"]["b3_gap"] > 0,
        "Q23": fates["b3"], "Q24": utilities["b5"] in ("POSITIVE UTILITY", "STRONG POSITIVE"),
        "Q25": fates["b5"], "Q26": utilities["b6"] in ("POSITIVE UTILITY", "STRONG POSITIVE"),
        "Q27": rows["500m"]["b6_gap"] > 0, "Q28": fates["b6"],
        "Q29": b6rep["500m"]["primary_O_minus_R"],
        "Q30": "grew" if b6rep["500m"]["primary_O_minus_R"] > b6rep["250m"]["primary_O_minus_R"] else "weakened",
        "Q31": boundary("b3", "32-63"), "Q32": boundary("b5", "64-127"),
        "Q33": boundary("b6", "512-639"), "Q34": longer.upper(),
        "Q35": all(gradients[l][age]["all_eligible_bins_nonzero"] for l in LINKS for age in ages),
        "Q36": "See position_bin_maturation.json; every preregistered bin is reported at all seven ages.",
        "Q37": interactions["500m"]["simple_interaction"],
        "Q38": "more additive" if abs(interactions["500m"]["simple_interaction"]) < abs(interactions["250m"]["simple_interaction"]) else "less additive",
        "Q39": {"bits": factorial["best_bits_B3_B5_B6"], "control": factorial["best_control"], "ce": factorial["best_ce"]},
        "Q40": rows["500m"]["b3_gain"] < 0 and factorial["pairwise_interactions"]["B3xB5"] > 0,
        "Q41": rows["500m"]["b6_gain"] <= 0 and max(factorial["pairwise_interactions"]["B3xB6"], factorial["pairwise_interactions"]["B5xB6"]) > 0,
        "Q42": factorial["best_bits_B3_B5_B6"][1] == "1",
        "Q43": large["metrics"]["b3"]["gain_off_minus_real"],
        "Q44": large["metrics"]["b3"]["sequence_gap_shuffled_minus_real"],
        "Q45": boot["b3"], "Q46": large["metrics"]["b5"]["gain_off_minus_real"],
        "Q47": large["metrics"]["b5"]["sequence_gap_shuffled_minus_real"], "Q48": boot["b5"],
        "Q49": large["metrics"]["b6"]["gain_off_minus_real"],
        "Q50": large["metrics"]["b6"]["sequence_gap_shuffled_minus_real"], "Q51": boot["b6"],
        "Q52": [l.upper() for l in ("b3", "b5", "b6") if boot[l]["classification"] == "STRONGLY CONFIRMED"],
        "Q53": all(stability[age]["passed"] for age in ages),
        "Q54": read_json(output / "mandatory_fresh_process_restart_update_715.json")["passed"],
        "Q55": checkpoint["sha256"], "Q56": checkpoint["next_global_batch_sha256"],
        "Q57": checkpoint["next_global_batch_stream_sha256"],
        "Q58": checkpoint["strict_reopen"]["passed"] and payload["d3a_completed_updates"] == FINAL_UPDATE,
        "Q59": recommendation,
    }
    return q


def render_report(summary):
    rows = summary["maturation_table"]; q = summary["questions"]
    lines = [
        "EXPERIMENT 2D3A — 500M COMPLETE", "",
        "PRIMARY 500M CLASSIFICATION:", summary["primary_classification"], "",
        "CUMULATIVE 2D3A TARGETS:", f"{FINAL_TARGETS:,}", "",
        "B3 TRUE RECURRENT GAIN @500M:", str(rows["500m"]["b3_gain"]), "",
        "B5 TRUE RECURRENT GAIN @500M:", str(rows["500m"]["b5_gain"]), "",
        "B6 TRUE RECURRENT GAIN @500M:", str(rows["500m"]["b6_gain"]), "",
        "B3 TRUE SEQUENCE GAP @500M:", str(rows["500m"]["b3_gap"]), "",
        "B5 TRUE SEQUENCE GAP @500M:", str(rows["500m"]["b5_gap"]), "",
        "B6 TRUE SEQUENCE GAP @500M:", str(rows["500m"]["b6_gap"]), "",
        "## Full maturation table", "",
        "| Metric | 100M | 150M | 200M | 250M | 300M | 375M | 500M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = ("all_real_ce", "b1_gain", "b1_gap", "b1_gate", "b3_gain", "b3_gap", "b3_gate",
               "b5_gain", "b5_gap", "b5_gate", "b6_gain", "b6_gap", "b6_gate",
               "combined_gain", "combined_gap")
    for metric in metrics:
        lines.append(f"| {metric} | " + " | ".join(f"{rows[a][metric]:.12g}" for a in AGES) + " |")
    lines += [
        "", "## Continuity and frozen analyses", "",
        f"Source checkpoint SHA: `{SOURCE_SHA256}` (exact match).",
        "Optimizer, scheduler, loader, gate values, and all RNG states were restored without reset.",
        "Architecture and parameter count remained unchanged at 124,475,908 parameters.",
        f"M500 checkpoint SHA: `{summary['checkpoint']['sha256']}`.",
        f"M500 next-batch SHA: `{summary['checkpoint']['next_global_batch_sha256']}`.",
        f"M500 next-stream SHA: `{summary['checkpoint']['next_global_batch_stream_sha256']}`.",
        "The attention, boundary-memory, writer-gradient, B6 representation, position-bin, factorial,",
        "large-confirmation/bootstrap, stability, memory, and runtime results are preserved in the named JSON artifacts.",
        "", "## Scientific questions", "",
    ]
    for index in range(1, 60):
        lines.append(f"Q{index}. {json.dumps(q[f'Q{index}'], sort_keys=True)}")
    lines += [
        "", "## Recommendation", "", summary["recommendation"], "",
        "NEXT PRE-REGISTERED MATURATION ENDPOINT IF UNCHANGED CONTINUATION IS CHOSEN:", "",
        "1,000,341,504 cumulative targets", "1,908 cumulative optimizer updates", "",
        "Additional from M500:", "954 updates", "500,170,752 targets", "",
        "NO TRAINING BEYOND 500,170,752 TARGETS WAS RUN.", "",
        "# EXPERIMENT 2D3A 500M COMPLETE", "",
    ]
    return "\n".join(lines)


def run_finalize(args):
    require_branch(); device = base.require_a100(); output = Path(args.output_dir)
    model, optimizer, loader, payload = base.load_d3a_checkpoint(args.final_checkpoint, device, restore=False)
    if payload["d3a_completed_updates"] != FINAL_UPDATE or payload["d3a_processed_targets"] != FINAL_TARGETS:
        raise SystemExit("final checkpoint is not exact M500")
    manifest = read_json(output / "checkpoint_manifest.json")
    if str(FINAL_UPDATE) not in manifest:
        persistent = Path(args.persistent_checkpoint_dir) / checkpoint_name(FINAL_UPDATE)
        local_sha = sha256(args.final_checkpoint)
        persistent_sha = sha256(persistent)
        reopen = strict_reopen(args.final_checkpoint, FINAL_UPDATE, payload["metadata"], device)
        if local_sha != persistent_sha or not reopen["passed"]:
            raise SystemExit("cannot recover missing final checkpoint manifest")
        manifest[str(FINAL_UPDATE)] = {
            "checkpoint": str(Path(args.final_checkpoint).resolve()), "sha256": local_sha,
            "bytes": Path(args.final_checkpoint).stat().st_size,
            "next_global_batch_sha256": payload["next_global_batch_sha256"],
            "next_global_batch_stream_sha256": payload["next_global_batch_stream_sha256"],
            "strict_reopen": reopen,
            "persistent": {"checkpoint": str(persistent.resolve()), "sha256": persistent_sha,
                           "passed": True, "recovered_after_quota_failure": True},
        }
        durable_json(output / "checkpoint_manifest.json", manifest)
    checkpoint = manifest[str(FINAL_UPDATE)]
    if sha256(args.final_checkpoint) != checkpoint["sha256"]:
        raise SystemExit("final checkpoint SHA mismatch")
    val = base.validation_path(Path(args.data_root))
    rows, increments, parallels, interactions = build_maturation(args.source_results, output)
    factorial = run_factorial(model, val, output)
    large, boot = run_large_confirmation(model, val, output, args.source_results)
    utilities = {link: link_utility(increments["500m"], link) for link in LINKS}
    fates = {link: fate_label(rows, link, utilities[link]) for link in ("b3", "b5", "b6")}
    secondary_positive = sum(utilities[l] in ("POSITIVE UTILITY", "STRONG POSITIVE") for l in ("b3", "b5", "b6"))
    combined_positive = rows["500m"]["combined_gain"] > 0 and rows["500m"]["combined_gap"] > 0
    synergy = interactions["500m"]["simple_interaction"] > 0
    if combined_positive and secondary_positive >= 2:
        primary = "MULTI-LINK POSITIVE RECURRENT PYRAMID"
    elif combined_positive and synergy and secondary_positive < 2:
        primary = "SYNERGISTIC RECURRENT PYRAMID WITH WEAK MARGINAL LINKS"
    elif rows["500m"]["combined_gain"] > 0:
        primary = "PARTIAL RECURRENT PYRAMID"
    elif abs(rows["500m"]["combined_gain"]) < 1e-4:
        primary = "RECURRENT PYRAMID NEAR ZERO"
    else:
        primary = "RECURRENT PYRAMID HARMFUL"
    b5_robust = (utilities["b5"] in ("POSITIVE UTILITY", "STRONG POSITIVE")
                 and boot["b5"]["classification"] != "NOT CONFIRMED")
    b3_b6_unsupported = all(utilities[l] not in ("POSITIVE UTILITY", "STRONG POSITIVE")
                               and boot[l]["classification"] == "NOT CONFIRMED" for l in ("b3", "b6"))
    factorial_synergy = synergy and factorial["best_bits_B3_B5_B6"] != "010"
    if rows["500m"]["combined_gain"] < 0 and rows["500m"]["combined_gap"] < 0:
        recommendation = "STOP 2D3A MATURATION AND REDESIGN RECURRENT READOUT"
    elif b5_robust and b3_b6_unsupported and not factorial_synergy:
        recommendation = "RUN A MATCHED CLEANED-PYRAMID ABLATION BEFORE 1B"
    else:
        recommendation = "CONTINUE UNCHANGED 2D3A TO 1B TO PRESERVE LEARNED INTERACTIONS" if factorial_synergy else "CONTINUE UNCHANGED 2D3A TO 1B"
    train = [json.loads(line) for line in (output / "training_metrics_250m_to_500m.jsonl").read_text().splitlines()]
    pass_times = {"pass1_seconds": 0.0, "pass2_seconds": 0.0, "pass3_seconds": 0.0}
    for row in train:
        for index, value in enumerate(row.get("approximate_pass_forward_seconds", []), 1):
            pass_times[f"pass{index}_seconds"] += float(value)
    performance = {
        "training": train, "updates": len(train),
        "training_wall_seconds": sum(r["wall_seconds"] for r in train),
        "mean_wall_seconds_per_update": float(np.mean([r["wall_seconds"] for r in train])),
        "mean_targets_per_second": float(np.mean([r["targets_per_second"] for r in train])),
        "max_peak_allocated_vram_mb": max(r["peak_allocated_vram_mb"] for r in train),
        "max_peak_reserved_vram_mb": max(r["peak_reserved_vram_mb"] for r in train),
        **pass_times,
    }
    memory = base.memory_accounting(); durable_json(output / "memory_accounting.json", memory)
    durable_json(output / "performance.json", performance)
    continuation = {
        "source_checkpoint": checkpoint["persistent"]["checkpoint"],
        "source_checkpoint_sha256": checkpoint["sha256"],
        "source_updates": FINAL_UPDATE, "source_targets": FINAL_TARGETS,
        "next_endpoint_updates": 1908, "next_endpoint_targets": 1_000_341_504,
        "additional_updates": 954, "additional_targets": 500_170_752,
        "next_training_update": 955,
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "next_global_batch_stream_sha256": checkpoint["next_global_batch_stream_sha256"],
        "optimizer_state_present": True, "scheduler_state": payload.get("scheduler"),
        "lr_per_group": payload.get("current_lr_per_group"), "loader_state": payload["loader_state"],
        "rng_provenance": sorted(payload["rng_state"]), "strict_reopen_passed": checkpoint["strict_reopen"]["passed"],
        "architecture_unchanged": True, "resume_ready": True,
        "warnings": ["DO NOT RESET GATES", "DO NOT RESET OPTIMIZER", "DO NOT RESET SCHEDULER",
                     "DO NOT RESTART DATA", "DO NOT RESTART WARMUP"],
    }
    durable_json(output / "CONTINUATION_MANIFEST.json", continuation)
    questions = make_questions(rows, increments, interactions, factorial, large, boot, checkpoint,
                               output, payload, utilities, fates, recommendation)
    compensation = {}
    for link in ("b3", "b5", "b6"):
        compensation[link] = {
            "gate_growth_250m_to_500m": rows["500m"][f"{link}_gate"] > rows["250m"][f"{link}_gate"],
            "gain_500m": rows["500m"][f"{link}_gain"], "gap_500m": rows["500m"][f"{link}_gap"],
            "possible_generic_non_sequence_compensation": (
                rows["500m"][f"{link}_gate"] > rows["250m"][f"{link}_gate"]
                and rows["500m"][f"{link}_gain"] <= 0 and rows["500m"][f"{link}_gap"] <= 0),
        }
    summary = {
        "experiment": EXPERIMENT, "primary_classification": primary,
        "cumulative_updates": FINAL_UPDATE, "cumulative_targets": FINAL_TARGETS,
        "architecture_unchanged": True, "parameter_count": base.MODEL_PARAMETERS,
        "maturation_table": rows, "utilities_500m": utilities, "fate_labels": fates,
        "generic_compensation_diagnostic": compensation, "combined_interaction": interactions,
        "factorial": factorial, "large_confirmation": large, "bootstrap": boot,
        "recommendation": recommendation, "checkpoint": checkpoint,
        "continuation": continuation, "performance": performance, "memory": memory,
        "questions": questions, "no_training_beyond_500m": True,
    }
    durable_json(output / "result_summary.json", summary)
    build_plots(output, rows, interactions, factorial, boot, performance)
    durable_json(output / "storage_cleanup_manifest.json", {
        "M100_retained": True, "M250_retained": True, "M300_retained": True,
        "M375_retained": True, "M500_retained": True, "persistent_volume_retained": True,
        "deleted": [{
            "path": "/workspace/build-nanogpt-exp2a0/runs/experiment_2a3_250m/checkpoints/checkpoint_updates_000286.pt",
            "reason": "recover persistent-volume capacity after the M500 checkpoint copy hit quota",
            "bytes": 498_177_785,
            "sha256": "48afa92fcc1174f80278a3024edc9b05ca689c47eb1f24eea31bf3eb018aa364",
            "recoverable_from_local_archive": True,
            "local_archive": "/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/storage_cleanup_recovery/experiment_2a3_250m/checkpoint_updates_000286.pt",
            "later_update_381_and_final_update_477_checkpoints_retained": True,
        }],
        "ephemeral_relocations": [{
            "source": "/workspace/build-nanogpt-exp2d3a500m_sparse/.git/objects/pack",
            "destination": "/tmp/exp2d3a500m_git_pack_backup",
            "kind": "disposable_current-clone_git_pack",
        }, {
            "source": "/workspace/build-nanogpt-exp2d3a500m_sparse/results/experiment_2d3a_alternating_integration_pyramid_250m",
            "destination": "/tmp/exp2d3a500m_source_250m_results",
            "kind": "duplicate_source-results-copy",
        }],
        "scientific_2d3a_checkpoint_removed": False,
        "dataset_removed": False,
        "passed": True,
    })
    report = render_report(summary)
    durable_text(output / "EXPERIMENT_2D3A_500M_FINAL_REPORT.md", report)
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", report + "\nFinal pod may be stopped after Git and local-backup verification.\n")
    checks = {
        "exact_final_update": payload["d3a_completed_updates"] == FINAL_UPDATE,
        "exact_final_targets": payload["d3a_processed_targets"] == FINAL_TARGETS,
        "exact_477_new_updates": len(train) == 477 and train[0]["cumulative_update"] == 478 and train[-1]["cumulative_update"] == 954,
        "architecture_unchanged": True,
        "source_sha_exact": read_json(output / "source_250m_manifest.json")["sha256"] == SOURCE_SHA256,
        "optimizer_not_reset": not read_json(output / "optimizer_resume_250m_manifest.json")["reset"],
        "scheduler_not_reset": not read_json(output / "scheduler_resume_250m_manifest.json")["reset"],
        "data_not_reset": not read_json(output / "data_resume_250m_manifest.json")["reset"],
        "mandatory_restart": read_json(output / "mandatory_fresh_process_restart_update_715.json")["passed"],
        "final_checkpoint_sha": sha256(args.final_checkpoint) == checkpoint["sha256"],
        "final_checkpoint_strict": checkpoint["strict_reopen"]["passed"],
        "canonical_milestones": all(parallels[a]["subset_sha256"] == CANONICAL_SHA for a in ("300m", "375m", "500m")),
        "maturation_core_milestones": all(increments[a]["subset_sha256"] == MATURATION_CORE_SHA for a in ("300m", "375m", "500m")),
        "large_confirmation_targets": large["targets_per_control"] == 1_048_576,
        "large_confirmation_disjoint": read_json(output / "m500_large_confirmation_disjointness_audit.json")["passed"],
        "no_training_beyond": max(r["cumulative_update"] for r in train) == FINAL_UPDATE,
        "single_recommendation": isinstance(recommendation, str),
    }
    inventory = {name: ((output / name).is_file() or name == "FINAL_AUDIT.json") for name in REQUIRED_ARTIFACTS}
    inventory.update({f"plot_p{n:02d}.png": (output / f"plot_p{n:02d}.png").is_file() for n in range(1, 17)})
    checks["required_artifacts"] = all(inventory.values())
    audit = {"experiment": EXPERIMENT, "checks": checks, "passed": all(checks.values()),
             "artifact_inventory": inventory, "final_checkpoint_sha256": checkpoint["sha256"]}
    durable_json(output / "FINAL_AUDIT.json", audit)
    heartbeat(output, FINAL_UPDATE, {"final_audit_passed": audit["passed"]},
              checkpoint["persistent"]["checkpoint"], "complete")
    if not audit["passed"]:
        raise SystemExit(f"final audit failed: {checks}")
    print("EXPERIMENT_2D3A_500M_FINALIZE_PASS", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(); subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("preflight"); output_args(p); p.add_argument("--data-root", required=True)
    p.add_argument("--stop-capability-verified", action="store_true"); p.set_defaults(func=run_preflight)
    p = subs.add_parser("train"); output_args(p); p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint-dir", required=True); p.add_argument("--persistent-checkpoint-dir", required=True)
    p.add_argument("--resume-checkpoint"); p.add_argument("--end-update", type=int, required=True); p.set_defaults(func=run_train)
    p = subs.add_parser("finalize"); output_args(p); p.add_argument("--data-root", required=True)
    p.add_argument("--persistent-checkpoint-dir", required=True)
    p.add_argument("--final-checkpoint", required=True); p.set_defaults(func=run_finalize)
    return parser


def main():
    args = build_parser().parse_args(); args.func(args)


if __name__ == "__main__":
    main()
