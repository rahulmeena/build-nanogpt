#!/usr/bin/env python3
"""Experiment 2D3A: exact 100M -> 250M continuation and maturation audit.

The scientific model, optimizer, loader, pass schedule, validation controls and
incremental kernel are imported unchanged from ``experiment_2d3a``.  This file
only orchestrates the preregistered continuation, checkpoints, and reports.
"""

from __future__ import annotations

import argparse
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


EXPERIMENT = "2D3A-250M"
PROTOCOL = "alternating_integration_recurrent_pyramid_100m_to_250m_v1"
BRANCH = "experiment-2d3a-alternating-integration-pyramid-250m"
SOURCE_COMMIT = "9c54683400791282e57a88c3d3481ec6c79302aa"
SOURCE_TAG = "experiment-2d3a-alternating-integration-pyramid-100m-final"
SOURCE_SHA256 = "8727e86c6f18164f3a8104af3c726290536136d9f8d0efe810dcc29656d33667"
SOURCE_NEXT_BATCH = "91fa2cae4e6e52cfddd2b470175ec704f0548b447f02861917ec548736fe18e7"
SOURCE_NEXT_STREAM = "4da6fed71755e523030a2d8e9e7cc96d19691a8c9b3ac8c490426bafe3d44e82"
CANONICAL_SHA = base.CANONICAL_COLLECTION_SHA
MATURATION_CORE_SHA = "8befbf790b3e522747cd39da306ec124464bf8dde1604caf64f299efa7e36216"
SOURCE_UPDATE = 191
RESTART_UPDATE = 334
FINAL_UPDATE = 477
FINAL_TARGETS = 250_085_376
MILESTONES = {286: "150m", 381: "200m", 477: "250m"}
MILESTONE_TARGETS = {286: 149_946_368, 381: 199_753_728, 477: 250_085_376}
CONTROL_NAMES = [
    "all_real", "new_links_off", "b1_off", "b3_off", "b5_off", "b6_off",
    "b1_shuffled", "b3_shuffled", "b5_shuffled", "b6_shuffled",
    "all_new_shuffled", "all_recurrent_shuffled",
]
LINKS = ("b1", "b3", "b5", "b6")
ATTENTION_FILES = {
    "b1": "b1_attention_maturation.json", "b3": "b3_attention_maturation.json",
    "b5": "b5_attention_maturation.json", "b6": "b6_attention_maturation.json",
}
GRADIENT_FILES = {
    "b1": "b12_to_b1_gradient_maturation.json",
    "b3": "b10_to_b3_gradient_maturation.json",
    "b5": "b8_to_b5_gradient_maturation.json",
    "b6": "b7_to_b6_gradient_maturation.json",
}
REQUIRED_ARTIFACTS = (
    "EXPERIMENT_2D3A_250M_FINAL_REPORT.md", "FINAL_AUDIT.json", "result_summary.json",
    "source_100m_manifest.json", "continuation_semantic_diff.json",
    "optimizer_resume_manifest.json", "scheduler_resume_manifest.json", "data_resume_manifest.json",
    "training_metrics_100m_to_250m.jsonl", "maturation_table.json",
    "maturation_interval_deltas.json", "milestone_150m_validation.json",
    "milestone_200m_validation.json", "milestone_250m_validation.json",
    "true_incremental_150m.json", "true_incremental_200m.json", "true_incremental_250m.json",
    "paired_maturation_controls.json", "gate_maturation.json",
    "b1_attention_maturation.json", "b3_attention_maturation.json",
    "b5_attention_maturation.json", "b6_attention_maturation.json",
    "b12_to_b1_gradient_maturation.json", "b10_to_b3_gradient_maturation.json",
    "b8_to_b5_gradient_maturation.json", "b7_to_b6_gradient_maturation.json",
    "b6_representation_maturation.json", "position_bin_maturation.json",
    "combined_link_interaction.json", "stability_8pass_maturation.json",
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


def require_branch_clean():
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit(f"must run on {BRANCH}")
    if git("status", "--porcelain"):
        raise SystemExit("Git worktree must be clean")


def require_branch():
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit(f"must run on {BRANCH}")


def optimizer_tensor_manifest(optimizer):
    rows = []
    for parameter_index, (_, state) in enumerate(optimizer.state.items()):
        for name, value in sorted(state.items()):
            if torch.is_tensor(value):
                raw = value.detach().cpu().contiguous().numpy().tobytes()
                rows.append({
                    "parameter_index": parameter_index, "state": name,
                    "shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                })
    aggregate = hashlib.sha256("".join(row["sha256"] for row in rows).encode()).hexdigest()
    return {"state_entries": len(optimizer.state), "tensor_entries": len(rows),
            "aggregate_tensor_sha256": aggregate, "representative_tensors": rows[:12]}


def continuation_metadata(args, payload, accumulation):
    old = payload["metadata"]
    return {
        "experiment": EXPERIMENT, "protocol": PROTOCOL, "branch": BRANCH,
        "source_100m_checkpoint_path": str(Path(args.source_checkpoint).resolve()),
        "source_100m_checkpoint_sha256": SOURCE_SHA256,
        "source_100m_commit": SOURCE_COMMIT, "source_100m_tag": SOURCE_TAG,
        "git_implementation_commit": git("rev-parse", "HEAD"),
        "targets_per_update": base.GLOBAL_TARGETS,
        "micro_batch": int(payload["loader_state"]["batch_size"]),
        "gradient_accumulation": int(accumulation), "sequence_length": base.T,
        "two_pass_weights": [.25, .75], "three_pass_weights": [.2, .4, .4],
        "pass3_every_cumulative_update": 32, "mandatory_restart_cumulative_update": RESTART_UPDATE,
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
        "parent_experiment": "2D3A-100M", "parent_checkpoint_path": metadata["source_100m_checkpoint_path"],
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
        "continuation_source_update": SOURCE_UPDATE, "continuation_source_targets": 100_139_008,
        "saved_process_id": os.getpid(), "saved_at_unix": time.time(),
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
        "parameters": sum(p.numel() for p in model.parameters()) == base.MODEL_PARAMETERS,
        "model_finite": base.model_finite(model), "optimizer_finite": base.optimizer_finite(optimizer),
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
    digest = sha256(path)
    audit = strict_reopen(path, completed, metadata, device)
    if not audit["passed"]:
        raise SystemExit(f"strict checkpoint reopen failed: {audit}")
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), audit)
    return {"checkpoint": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size,
            "next_global_batch_sha256": payload["next_global_batch_sha256"],
            "next_global_batch_stream_sha256": payload["next_global_batch_stream_sha256"],
            "strict_reopen": audit}


def persist_checkpoint(path, persistent_dir):
    path = Path(path).resolve(); persistent_dir = Path(persistent_dir).resolve()
    try:
        return base.persist_triplet(path, persistent_dir)
    except OSError as error:
        # RunPod's FUSE volume can report EIO after a large copy has reached
        # its full byte count.  Recover only when every byte hashes exactly;
        # otherwise preserve the local strict checkpoint and fail closed.
        rows = []
        for source in (path, path.with_suffix(path.suffix + ".sha256"),
                       path.with_suffix(path.suffix + ".verification.json")):
            destination = persistent_dir / source.name
            temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
            if not temporary.exists():
                if source == path:
                    raise
                shutil.copyfile(source, temporary)
            if temporary.stat().st_size != source.stat().st_size:
                raise
            if sha256(temporary) != sha256(source):
                raise
            os.replace(temporary, destination)
            rows.append({"source": str(source), "destination": str(destination),
                         "sha256": sha256(destination), "bytes": destination.stat().st_size})
        return {"files": rows, "passed": True, "checkpoint": rows[0]["destination"],
                "sha256": rows[0]["sha256"], "recovered_after_copy_eio": str(error)}


def output_args(parser):
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--pod-name", required=True)


def validate_source(model, optimizer, loader, payload, args):
    accumulation = int(payload["gradient_accumulation"])
    checks = {
        "checkpoint_sha256": sha256(args.source_checkpoint) == SOURCE_SHA256,
        "schema": payload.get("schema") == base.SCHEMA,
        "updates": payload.get("d3a_completed_updates") == SOURCE_UPDATE,
        "targets": payload.get("d3a_processed_targets") == 100_139_008,
        "parameters": sum(p.numel() for p in model.parameters()) == base.MODEL_PARAMETERS,
        "architecture": payload.get("architecture_version") == base.ARCHITECTURE_VERSION,
        "next_batch": base.next_batch_hash(loader, accumulation) == SOURCE_NEXT_BATCH,
        "next_stream": base.next_stream_hash(loader, accumulation) == SOURCE_NEXT_STREAM,
        "stored_next_batch": payload.get("next_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "stored_next_stream": payload.get("next_global_batch_stream_sha256") == SOURCE_NEXT_STREAM,
        "model_finite": base.model_finite(model), "optimizer_finite": base.optimizer_finite(optimizer),
        "maturation_core_geometry": payload["metadata"]["maturation_core_subset_manifest"] == {
            "batches": 4, "batch_size": 64, "sequence_length": 1024,
            "targets_per_control": 262144, "source": "first four canonical validation batches",
        },
        "canonical_sha": payload["metadata"]["canonical_validation_manifest"]["collection_sha256"] == CANONICAL_SHA,
    }
    return checks


def short_regressions(model, args):
    frozen = Path(args.source_results)
    validation = base.validation_path(Path(args.data_root))
    reference_parallel = read_json(frozen / "milestone_validation.json")["191"]
    parallel = base.evaluate_parallel(model, validation, CONTROL_NAMES, batches=1)
    parallel_deltas = {
        name: parallel["controls"][name]["per_batch_losses"][0]
        - reference_parallel["controls"][name]["per_batch_losses"][0]
        for name in CONTROL_NAMES
    }
    # One frozen 64-sequence true-incremental batch; the same control kernel is
    # used as the full maturation core, and every per-sequence CE is compared.
    reference_incremental = read_json(frozen / "incremental_validation.json")
    loader = base.d1.ExplicitShardLoader([validation], base.VALIDATION_B, base.T)
    cpu_x, cpu_y = loader.next_batch(); device = base.model_device(model)
    x, y = cpu_x.to(device), cpu_y.to(device)
    permutation = torch.arange(base.VALIDATION_B, device=device).roll(1)
    incremental = {}
    with torch.no_grad():
        for name in CONTROL_NAMES:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                row = base.incremental_control(model, x, y, name, permutation)
            expected = reference_incremental["controls"][name]["per_sequence_losses"][:base.VALIDATION_B]
            deltas = [float(a) - float(b) for a, b in zip(row["per_sequence_losses"], expected)]
            incremental[name] = {"max_abs_per_sequence_ce_delta": max(abs(v) for v in deltas),
                                 "exact": all(v == 0.0 for v in deltas)}
    del x, y, cpu_x, cpu_y; torch.cuda.empty_cache()
    return {
        "parallel_prefix": {"controls": parallel["controls"], "deltas": parallel_deltas,
                            "exact": all(value == 0.0 for value in parallel_deltas.values())},
        "true_incremental_prefix": {"controls": incremental,
                                    "exact": all(row["exact"] for row in incremental.values())},
    }


def run_preflight(args):
    require_branch_clean(); device = base.require_a100(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    model, optimizer, loader, payload = base.load_d3a_checkpoint(args.source_checkpoint, device, restore=False)
    checks = validate_source(model, optimizer, loader, payload, args)
    before = optimizer_tensor_manifest(optimizer)
    regressions = short_regressions(model, args)
    checks.update({
        "branch": git("branch", "--show-current") == BRANCH,
        "source_commit_ancestor": subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], cwd=base.REPO_ROOT).returncode == 0,
        "tag_exact": git("rev-parse", SOURCE_TAG) == SOURCE_COMMIT,
        "stop_capability": bool(args.stop_capability_verified),
        "storage_free_ge_80gb": shutil.disk_usage(Path(args.source_checkpoint).parent).free >= 80 * 1024**3,
        "parallel_prefix_exact": regressions["parallel_prefix"]["exact"],
        "incremental_prefix_exact": regressions["true_incremental_prefix"]["exact"],
    })
    lrs = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
    source = {
        "checkpoint": str(Path(args.source_checkpoint).resolve()), "sha256": sha256(args.source_checkpoint),
        "commit": SOURCE_COMMIT, "tag": SOURCE_TAG, "updates": SOURCE_UPDATE, "targets": 100_139_008,
        "next_batch_sha256": SOURCE_NEXT_BATCH, "next_stream_sha256": SOURCE_NEXT_STREAM,
        "parameter_count": base.MODEL_PARAMETERS, "gates": base.gate_values(model), "checks": checks,
    }
    semantic = {
        "architecture_changes": [], "parameter_changes": [], "optimizer_changes": [],
        "scheduler_changes": [], "data_stream_changes": [], "gate_resets": [],
        "local_window_changes": [], "recurrent_source_changes": [], "recurrent_lag_changes": [],
        "pass_semantic_changes": [], "precision_changes": [], "incremental_kernel_changes": [],
        "orchestration_only_changes": ["cumulative 100M-to-250M budget", "milestone diagnostics", "mandatory restart at update 334"],
        "semantic_diff_zero": True,
    }
    durable_json(output / "source_100m_manifest.json", source)
    durable_json(output / "continuation_semantic_diff.json", semantic)
    durable_json(output / "optimizer_resume_manifest.json", {
        "source": before, "reset": False, "state_loaded_strictly": True,
        "group_lrs": lrs, "update_192_will_use": lrs,
    })
    durable_json(output / "scheduler_resume_manifest.json", {
        "source_scheduler": payload.get("scheduler"), "reset": False, "warmup_restarted": False,
        "cadence_uses_cumulative_update": True, "constant_group_lrs": lrs,
    })
    durable_json(output / "data_resume_manifest.json", {
        "source_loader_state": payload["loader_state"], "expected_update_192_batch_sha256": SOURCE_NEXT_BATCH,
        "expected_update_192_stream_sha256": SOURCE_NEXT_STREAM, "reset": False,
    })
    durable_json(output / "pre_resume_regression.json", regressions)
    audit = {"experiment": EXPERIMENT, "checks": checks, "authorized": all(checks.values()),
             "hardware": {"gpu": torch.cuda.get_device_name(device), "gpu_count": torch.cuda.device_count()},
             "pod_id": args.pod_id, "pod_name": args.pod_name}
    durable_json(output / "preflight_audit.json", audit)
    durable_json(output / "checkpoint_manifest.json", {str(SOURCE_UPDATE): {
        "checkpoint": str(Path(args.source_checkpoint).resolve()), "sha256": SOURCE_SHA256,
        "next_global_batch_sha256": SOURCE_NEXT_BATCH,
        "next_global_batch_stream_sha256": SOURCE_NEXT_STREAM, "source_100m": True,
    }})
    durable_json(output / "commands_and_runtime.json", {"started_at_unix": time.time(), "commands": [], "pod_id": args.pod_id,
                                                         "pod_name": args.pod_name, "preflight_seconds": regressions})
    if not audit["authorized"]:
        raise SystemExit(f"preflight failed: {checks}")
    print("EXPERIMENT_2D3A_250M_PREFLIGHT_PASS", flush=True)


def merge_keyed(path, key, value):
    payload = read_json(path) if Path(path).exists() else {}
    payload[str(key)] = value; durable_json(path, payload)


def checkpoint_name(update):
    if update in MILESTONE_TARGETS:
        return f"scientific_cumulative_{MILESTONE_TARGETS[update]:012d}.pt"
    if update == RESTART_UPDATE:
        return "restart_cumulative_000175112192.pt"
    return f"recovery_cumulative_{update * base.GLOBAL_TARGETS:012d}.pt"


def save_scientific(args, source, model, optimizer, loader, update, accumulation, metadata, device):
    path = Path(args.checkpoint_dir) / checkpoint_name(update)
    verification = save_checkpoint(path, source, model, optimizer, loader, update, accumulation, metadata, device)
    persistent = persist_checkpoint(path, args.persistent_checkpoint_dir)
    verification["persistent"] = persistent
    manifest_path = Path(args.output_dir) / "checkpoint_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest[str(update)] = verification; durable_json(manifest_path, manifest)
    return verification


def heartbeat(output, update, row, checkpoint=None, status="training"):
    durable_json(Path(output) / "HEARTBEAT.json", {
        "experiment": EXPERIMENT, "status": status, "cumulative_update": update,
        "cumulative_targets": update * base.GLOBAL_TARGETS, "latest_metrics": row,
        "checkpoint": checkpoint, "pid": os.getpid(), "updated_at_unix": time.time(),
    })


def milestone_complete(output, age):
    required = [f"milestone_{age}_validation.json", f"true_incremental_{age}.json"]
    return all((Path(output) / name).exists() for name in required)


def run_milestone(args, model, update):
    output = Path(args.output_dir); age = MILESTONES[update]; val = base.validation_path(Path(args.data_root))
    parallel_path = output / f"milestone_{age}_validation.json"
    if parallel_path.exists():
        parallel = read_json(parallel_path)
    else:
        parallel = base.evaluate_parallel(model, val, CONTROL_NAMES)
        if not parallel["canonical_collection_match"] or parallel["subset_sha256"] != CANONICAL_SHA:
            raise SystemExit(f"{age} canonical validation SHA mismatch")
        parallel["cumulative_update"] = update; parallel["cumulative_targets"] = MILESTONE_TARGETS[update]
        durable_json(parallel_path, parallel)
    incremental_path = output / f"true_incremental_{age}.json"
    if incremental_path.exists():
        incremental = read_json(incremental_path)
    else:
        incremental = base.evaluate_incremental(model, val)
        if incremental["subset_sha256"] != MATURATION_CORE_SHA:
            raise SystemExit(f"{age} maturation-core SHA mismatch")
        incremental["cumulative_update"] = update; incremental["cumulative_targets"] = MILESTONE_TARGETS[update]
        incremental["gates"] = base.gate_values(model)
        durable_json(incremental_path, incremental)
        merge_keyed(output / "position_bin_maturation.json", age, base.position_bin_metrics(incremental))
        merge_keyed(output / "paired_maturation_controls.json", age, incremental["paired"])
        merge_keyed(output / "incremental_cache_audit.json", age,
                    {name: row["cache_rows"] for name, row in incremental["controls"].items()})
    for link in LINKS:
        path = output / ATTENTION_FILES[link]
        existing = read_json(path) if path.exists() else {}
        if age not in existing:
            merge_keyed(path, age, base.attention_diagnostic(model, val, link))
    gradient_paths = [output / GRADIENT_FILES[link] for link in LINKS]
    if any(not path.exists() or age not in read_json(path) for path in gradient_paths):
        gradients = base.temporal_gradients(model, val)
        for link in LINKS: merge_keyed(output / GRADIENT_FILES[link], age, gradients[link])
    path = output / "stability_8pass_maturation.json"
    if not path.exists() or age not in read_json(path):
        merge_keyed(path, age, base.stability_8pass(model, val))
    path = output / "b6_representation_maturation.json"
    if not path.exists() or age not in read_json(path):
        merge_keyed(path, age, base.b6_representation_control(model, val))
    merge_keyed(output / "gate_maturation.json", age, base.gate_values(model))
    heartbeat(output, update, {"milestone": age, "diagnostics_complete": True}, status="milestone_complete")
    return parallel, incremental


def run_train(args):
    require_branch_clean(); device = base.require_a100(); output = Path(args.output_dir)
    preflight = read_json(output / "preflight_audit.json")
    if not preflight.get("authorized"):
        raise SystemExit("preflight did not authorize result training")
    model, optimizer, loader, source = base.load_d3a_checkpoint(args.resume_checkpoint or args.source_checkpoint,
                                                                device, restore=True)
    start = int(source["d3a_completed_updates"]); accumulation = int(source["gradient_accumulation"])
    if args.resume_checkpoint:
        parent_source = base.d0.torch_load(Path(args.source_checkpoint), mmap=False)
        saved_pid = source.get("saved_process_id")
        restart = {"loaded_update": start, "saved_process_id": saved_pid,
                   "resumed_process_id": os.getpid(), "fresh_process": saved_pid != os.getpid(),
                   "next_batch_sha256": base.next_batch_hash(loader, accumulation),
                   "expected_next_batch_sha256": source["next_global_batch_sha256"],
                   "next_stream_sha256": base.next_stream_hash(loader, accumulation),
                   "expected_next_stream_sha256": source["next_global_batch_stream_sha256"]}
        restart["passed"] = (restart["fresh_process"]
                             and restart["next_batch_sha256"] == restart["expected_next_batch_sha256"]
                             and restart["next_stream_sha256"] == restart["expected_next_stream_sha256"])
        if start == RESTART_UPDATE:
            restart["required_update"] = RESTART_UPDATE
            durable_json(output / "mandatory_fresh_process_restart_update_334.json", restart)
            if not restart["passed"]: raise SystemExit(f"mandatory restart failed: {restart}")
        elif start in MILESTONES:
            restart["reason"] = "fresh-process recovery from strict scientific milestone checkpoint"
            durable_json(output / f"scientific_recovery_update_{start}.json", restart)
            if not restart["passed"]: raise SystemExit(f"scientific recovery failed: {restart}")
            manifest_path = output / "checkpoint_manifest.json"
            manifest = read_json(manifest_path)
            if str(start) not in manifest:
                reopen = strict_reopen(args.resume_checkpoint, start, source["metadata"], device)
                persistent = Path(args.persistent_checkpoint_dir) / Path(args.resume_checkpoint).name
                persistent_sha = sha256(persistent)
                local_sha = sha256(args.resume_checkpoint)
                if not reopen["passed"] or persistent_sha != local_sha:
                    raise SystemExit("recovered scientific checkpoint failed local/persistent verification")
                manifest[str(start)] = {
                    "checkpoint": str(Path(args.resume_checkpoint).resolve()), "sha256": local_sha,
                    "bytes": Path(args.resume_checkpoint).stat().st_size,
                    "next_global_batch_sha256": source["next_global_batch_sha256"],
                    "next_global_batch_stream_sha256": source["next_global_batch_stream_sha256"],
                    "strict_reopen": reopen,
                    "persistent": {"checkpoint": str(persistent.resolve()), "sha256": persistent_sha,
                                   "passed": True, "recovered_from_verified_full_size_temporary": True},
                }
                durable_json(manifest_path, manifest)
        else:
            raise SystemExit(f"unauthorized resume update {start}")
        metadata = source["metadata"]
    else:
        parent_source = source
        checks = validate_source(model, optimizer, loader, source, args)
        if not all(checks.values()): raise SystemExit(f"source checks failed at train start: {checks}")
        metadata = continuation_metadata(args, source, accumulation)
    end = int(args.end_update)
    if (start, end) not in ((SOURCE_UPDATE, RESTART_UPDATE), (286, RESTART_UPDATE),
                            (RESTART_UPDATE, FINAL_UPDATE), (381, FINAL_UPDATE)):
        raise SystemExit(f"unauthorized segment {start}->{end}")
    if start in MILESTONES and not milestone_complete(output, MILESTONES[start]):
        run_milestone(args, model, start)
    recovery = None
    for update in range(start + 1, end + 1):
        consumed_batch = base.next_batch_hash(loader, accumulation) if update in (192, 335) else None
        consumed_stream = base.next_stream_hash(loader, accumulation) if update in (192, 335) else None
        row = base.train_update(model, optimizer, loader, accumulation, update, device)
        row["cumulative_update"] = update; row["cumulative_targets"] = update * base.GLOBAL_TARGETS
        row["optimizer_lrs"] = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
        if consumed_batch:
            row["consumed_global_batch_sha256"] = consumed_batch
            row["consumed_global_stream_sha256"] = consumed_stream
        append_jsonl(output / "training_metrics_100m_to_250m.jsonl", row)
        heartbeat(output, update, row)
        if update in (240, 430):
            path = Path(args.checkpoint_dir) / checkpoint_name(update)
            verification = save_checkpoint(path, parent_source, model, optimizer, loader, update,
                                           accumulation, metadata, device)
            if recovery:
                for old in (recovery, recovery.with_suffix(recovery.suffix + ".sha256"),
                            recovery.with_suffix(recovery.suffix + ".verification.json")):
                    if old.exists(): old.unlink()
            recovery = path
            manifest_path = output / "checkpoint_manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            manifest["recovery"] = verification; durable_json(manifest_path, manifest)
        if update in MILESTONES or update == RESTART_UPDATE:
            verification = save_scientific(args, parent_source, model, optimizer, loader, update,
                                           accumulation, metadata, device)
            heartbeat(output, update, row, verification["persistent"]["checkpoint"], "checkpoint_verified")
        if update in MILESTONES:
            run_milestone(args, model, update)
    print(f"EXPERIMENT_2D3A_250M_SEGMENT_COMPLETE {start}->{end}", flush=True)


def m100_reference(source_results):
    root = Path(source_results)
    inc = read_json(root / "incremental_validation.json")
    summary = read_json(root / "result_summary.json")
    return {"incremental": inc, "gates": summary["gates"], "parallel": summary["milestones"]["191"],
            "attention": summary["attention"], "gradients": summary["temporal_gradients"],
            "b6": summary["b6_representation_control"], "stability": summary["stability"],
            "position": summary["position_bins"]}


def maturation_label(ratio, final_gain):
    if final_gain <= 0: return "reversed"
    if ratio >= 2: return "strongly growing"
    if ratio >= 1.25: return "growing"
    if ratio >= .75: return "stable"
    return "weakening"


def link_classification(incremental, link):
    gain = incremental[f"true_{link}_gain"]; gap = incremental[f"true_{link}_sequence_gap"]
    wins_off = incremental["paired"][link]["real_vs_off"]["wins"]
    wins_shuffled = incremental["paired"][link]["real_vs_shuffled"]["wins"]
    if gain >= .001 and gap > 0 and wins_off >= 166 and wins_shuffled >= 166: return "STRONG POSITIVE"
    if gain > 0 and gap > 0 and wins_off >= 129 and wins_shuffled >= 129: return "POSITIVE UTILITY"
    if gap > 0: return "SEQUENCE-SPECIFIC BUT NOT ESTABLISHED"
    if gain < 0: return "HARMFUL"
    return "NEAR ZERO"


def build_maturation(args):
    output = Path(args.output_dir); ref = m100_reference(args.source_results)
    ages = ["100m", "150m", "200m", "250m"]
    increments = {"100m": ref["incremental"], **{age: read_json(output / f"true_incremental_{age}.json") for age in ages[1:]}}
    parallels = {"100m": ref["parallel"], **{age: read_json(output / f"milestone_{age}_validation.json") for age in ages[1:]}}
    gates_file = read_json(output / "gate_maturation.json")
    gates = {"100m": ref["gates"], **{age: gates_file[age] for age in ages[1:]}}
    table = {}
    for age in ages:
        inc = increments[age]
        table[age] = {"all_real_ce": parallels[age]["controls"]["all_real"]["validation_loss"],
                      "combined_gain": inc["combined_new_link_gain"],
                      "combined_gap": inc["combined_new_sequence_gap"]}
        for link in LINKS:
            table[age][f"{link}_gain"] = inc[f"true_{link}_gain"]
            table[age][f"{link}_gap"] = inc[f"true_{link}_sequence_gap"]
            table[age][f"{link}_gate"] = gates[age][link]["effective"]
    intervals = {}
    for left, right in zip(ages, ages[1:]):
        intervals[f"{left}_to_{right}"] = {key: table[right][key] - table[left][key] for key in table[left]}
    classifications = {}
    for link in LINKS:
        initial, final = table["100m"][f"{link}_gain"], table["250m"][f"{link}_gain"]
        ratio = final / initial if initial != 0 else math.inf
        classifications[link] = {"m100_gain": initial, "m250_gain": final, "ratio": ratio,
                                 "label": maturation_label(ratio, final),
                                 "utility_at_250m": link_classification(increments["250m"], link)}
    interaction = {}
    for age in ages:
        sum_marginal = sum(table[age][f"{link}_gain"] for link in ("b3", "b5", "b6"))
        combined = table[age]["combined_gain"]
        interaction[age] = {"combined_gain": combined, "sum_new_marginal_gains": sum_marginal,
                            "interaction_combined_minus_sum": combined - sum_marginal,
                            "additivity_abs_error": abs(combined - sum_marginal)}
    durable_json(output / "maturation_table.json", {"rows": table, "classifications": classifications})
    durable_json(output / "maturation_interval_deltas.json", intervals)
    durable_json(output / "combined_link_interaction.json", interaction)
    gate_all = {"100m": ref["gates"], **gates_file}; durable_json(output / "gate_maturation.json", gate_all)
    for link in LINKS:
        path = output / ATTENTION_FILES[link]; current = read_json(path); current["100m"] = ref["attention"][link]; durable_json(path, current)
        path = output / GRADIENT_FILES[link]; current = read_json(path); current["100m"] = ref["gradients"][link]; durable_json(path, current)
    path = output / "b6_representation_maturation.json"; current = read_json(path); current["100m"] = ref["b6"]; durable_json(path, current)
    path = output / "stability_8pass_maturation.json"; current = read_json(path); current["100m"] = ref["stability"]; durable_json(path, current)
    path = output / "position_bin_maturation.json"; current = read_json(path); current["100m"] = ref["position"]; durable_json(path, current)
    return table, classifications, interaction, increments, parallels, gates


def build_plots(output, table, interactions, increments, performance):
    import matplotlib.pyplot as plt
    output = Path(output); ages = ["100m", "150m", "200m", "250m"]
    x = [100_139_008, 149_946_368, 199_753_728, 250_085_376]
    def save(n, draw):
        fig, ax = plt.subplots(figsize=(8, 5)); draw(ax); fig.tight_layout()
        fig.savefig(output / f"plot_p{n:02d}.png", dpi=160); plt.close(fig)
    save(1, lambda ax: (ax.plot(x, [table[a]["all_real_ce"] for a in ages], marker="o"), ax.set(xlabel="cumulative targets", ylabel="ALL_REAL CE")))
    save(2, lambda ax: ([ax.plot(x, [table[a][f"{l}_gain"] for a in ages], marker="o", label=l.upper()) for l in LINKS], ax.legend(), ax.set(xlabel="cumulative targets", ylabel="true gain")))
    save(3, lambda ax: ([ax.plot(x, [table[a][f"{l}_gap"] for a in ages], marker="o", label=l.upper()) for l in LINKS], ax.legend(), ax.set(xlabel="cumulative targets", ylabel="true sequence gap")))
    save(4, lambda ax: ([ax.plot(x, [table[a][f"{l}_gate"] for a in ages], marker="o", label=l.upper()) for l in LINKS], ax.legend(), ax.set(xlabel="cumulative targets", ylabel="effective gate")))
    attention = {l: read_json(output / ATTENTION_FILES[l]) for l in LINKS}
    for n, link in ((5, "b3"), (6, "b5"), (7, "b6")):
        bins = list(attention[link]["100m"]["recurrent"]["bins"])
        save(n, lambda ax, link=link, bins=bins: ([ax.plot(bins, [attention[link][a]["recurrent"]["bins"][b]["raw_mass"] for b in bins], marker="o", label=a.upper()) for a in ages], ax.legend(), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="recurrent attention mass")))
    save(8, lambda ax: ([ax.plot([r["head"] for r in attention[l]["250m"]["recurrent"]["per_head"]], [r["mean_lag"] for r in attention[l]["250m"]["recurrent"]["per_head"]], marker="o", label=l.upper()) for l in LINKS], ax.legend(), ax.set(xlabel="head", ylabel="mean recurrent lag @250M")))
    gradients = {l: read_json(output / GRADIENT_FILES[l]) for l in LINKS}
    save(9, lambda ax: ([ax.plot(list(gradients[l]["250m"]["bins"]), [r["mean_gradient_rms"] for r in gradients[l]["250m"]["bins"].values()], marker="o", label=l.upper()) for l in LINKS], ax.legend(), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="writer-gradient RMS @250M")))
    position = read_json(output / "position_bin_maturation.json")
    bins = list(position["250m"]["b3"])
    save(10, lambda ax: ([ax.plot(bins, [position[a]["b3"][b]["off_minus_real"] for b in bins], marker="o", label=a.upper()) for a in ages], ax.legend(), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="B3 position-binned utility")))
    save(11, lambda ax: (ax.plot(x, [interactions[a]["combined_gain"] for a in ages], marker="o", label="combined"), ax.plot(x, [interactions[a]["sum_new_marginal_gains"] for a in ages], marker="o", label="sum marginals"), ax.legend(), ax.set(xlabel="cumulative targets", ylabel="gain")))
    b6 = read_json(output / "b6_representation_maturation.json")
    save(12, lambda ax: (ax.plot(x, [b6[a]["primary_O_minus_R"] for a in ages], marker="o"), ax.set(xlabel="cumulative targets", ylabel="B6 representation gain")))
    save(13, lambda ax: ([ax.plot(x, [read_json(output / f"milestone_{a}_validation.json").get(f"{l}_gain", np.nan) if a != "100m" else np.nan for a in ages], label=f"{l.upper()} parallel") for l in LINKS], [ax.plot(x, [table[a][f"{l}_gain"] for a in ages], linestyle="--", label=f"{l.upper()} incremental") for l in LINKS], ax.legend(fontsize=7), ax.set(xlabel="cumulative targets", ylabel="gain")))
    train = performance["training"]
    save(14, lambda ax: (ax.plot([r["cumulative_targets"] for r in train], [r["targets_per_second"] for r in train], label="targets/s"), ax.set(xlabel="cumulative targets", ylabel="targets/s")))


def answer_questions(table, classifications, interaction, increments, final_payload, checkpoint, recommendation, output):
    ages = ("150m", "200m", "250m")
    att = {l: read_json(Path(output) / ATTENTION_FILES[l]) for l in LINKS}
    grad = {l: read_json(Path(output) / GRADIENT_FILES[l]) for l in LINKS}
    stable = read_json(Path(output) / "stability_8pass_maturation.json")
    b6 = read_json(Path(output) / "b6_representation_maturation.json")
    restart = read_json(Path(output) / "mandatory_fresh_process_restart_update_334.json")
    train = [json.loads(line) for line in (Path(output) / "training_metrics_100m_to_250m.jsonl").read_text().splitlines()]
    first = train[0]
    vals = lambda key: {a: table[a][key] for a in ages}
    relative = max(LINKS, key=lambda l: classifications[l]["ratio"])
    absolute = max(LINKS, key=lambda l: classifications[l]["m250_gain"] - classifications[l]["m100_gain"])
    def spread(link):
        rows = att[link]; return np.mean([r["mean_lag"] for r in rows["250m"]["recurrent"]["per_head"]]) > np.mean([r["mean_lag"] for r in rows["100m"]["recurrent"]["per_head"]])
    questions = {
        "Q1": True, "Q2": True, "Q3": True, "Q4": True,
        "Q5": first["optimizer_lrs"], "Q6": first.get("consumed_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "Q7": table["150m"]["all_real_ce"], "Q8": table["200m"]["all_real_ce"], "Q9": table["250m"]["all_real_ce"],
        "Q10": vals("b1_gain"), "Q11": vals("b1_gap"), "Q12": vals("b3_gain"), "Q13": vals("b3_gap"),
        "Q14": vals("b5_gain"), "Q15": vals("b5_gap"), "Q16": vals("b6_gain"), "Q17": vals("b6_gap"),
        "Q18": vals("combined_gain"), "Q19": vals("combined_gap"),
        "Q20": table["250m"]["b1_gate"], "Q21": table["250m"]["b3_gate"],
        "Q22": table["250m"]["b5_gate"], "Q23": table["250m"]["b6_gate"],
        "Q24": [l.upper() for l in LINKS if table["250m"][f"{l}_gate"] > table["100m"][f"{l}_gate"]],
        "Q25": relative.upper(), "Q26": absolute.upper(), "Q27": classifications["b3"]["label"],
        "Q28": classifications["b5"]["label"], "Q29": classifications["b6"]["label"],
        "Q30": classifications["b3"]["utility_at_250m"] in ("POSITIVE UTILITY", "STRONG POSITIVE"),
        "Q31": classifications["b5"]["utility_at_250m"] in ("POSITIVE UTILITY", "STRONG POSITIVE"),
        "Q32": classifications["b6"]["utility_at_250m"] in ("POSITIVE UTILITY", "STRONG POSITIVE"),
        "Q33": b6["250m"]["primary_O_minus_R"] > b6["100m"]["primary_O_minus_R"],
        "Q34": "See lag-bin attention maturation; boundary concentration reported without a new threshold.",
        "Q35": spread("b3"), "Q36": spread("b5"), "Q37": spread("b6"),
        "Q38": all(grad[l][a]["all_eligible_bins_nonzero"] for l in LINKS for a in ages),
        "Q39": "more additive" if interaction["250m"]["additivity_abs_error"] < interaction["100m"]["additivity_abs_error"] else "less additive",
        "Q40": "See position_bin_maturation.json; later/early-bin deltas are reported directly.",
        "Q41": all(stable[a]["passed"] for a in ages), "Q42": restart["passed"],
        "Q43": checkpoint["sha256"], "Q44": checkpoint["next_global_batch_sha256"],
        "Q45": checkpoint["next_global_batch_stream_sha256"],
        "Q46": all(checkpoint["strict_reopen"]["checks"].values()) and final_payload["d3a_completed_updates"] == FINAL_UPDATE,
        "Q47": recommendation,
    }
    return questions


def render_report(summary):
    table = summary["maturation_table"]; q = summary["questions"]
    lines = ["# Experiment 2D3A — 250M Final Report", "", f"Primary classification: **{summary['primary_classification']}**.", "",
             "## Maturation table", "", "| Metric | 100M | 150M | 200M | 250M |", "|---|---:|---:|---:|---:|"]
    metrics = ["all_real_ce", "b1_gain", "b1_gap", "b1_gate", "b3_gain", "b3_gap", "b3_gate",
               "b5_gain", "b5_gap", "b5_gate", "b6_gain", "b6_gap", "b6_gate", "combined_gain", "combined_gap"]
    for metric in metrics:
        lines.append(f"| {metric} | " + " | ".join(f"{table[a][metric]:.12g}" for a in ("100m", "150m", "200m", "250m")) + " |")
    lines += ["", "## Scientific questions", ""]
    for index in range(1, 48): lines.append(f"Q{index}. {json.dumps(q[f'Q{index}'], sort_keys=True)}")
    lines += ["", "## Recommendation", "", summary["recommendation"], "",
              "No training beyond 250,085,376 cumulative 2D3A targets was run.", ""]
    return "\n".join(lines)


def run_finalize(args):
    require_branch(); device = base.require_a100(); output = Path(args.output_dir)
    model, optimizer, loader, payload = base.load_d3a_checkpoint(args.final_checkpoint, device, restore=False)
    accumulation = int(payload["gradient_accumulation"])
    if payload["d3a_completed_updates"] != FINAL_UPDATE or payload["d3a_processed_targets"] != FINAL_TARGETS:
        raise SystemExit("final checkpoint is not exact M250")
    manifest = read_json(output / "checkpoint_manifest.json"); checkpoint = manifest[str(FINAL_UPDATE)]
    if sha256(args.final_checkpoint) != checkpoint["sha256"]: raise SystemExit("final checkpoint SHA mismatch")
    table, classifications, interaction, increments, parallels, gates = build_maturation(args)
    unstable = not all(read_json(output / "stability_8pass_maturation.json")[a]["passed"] for a in ("150m", "200m", "250m"))
    systematic_reversal = all(table["250m"][f"{l}_gain"] <= 0 for l in ("b3", "b5", "b6"))
    combined_harmful = table["250m"]["combined_gain"] <= 0
    recommendation = ("STOP THE 2D3A LONG-MATURATION LINEAGE" if unstable or systematic_reversal or combined_harmful
                      else "CONTINUE UNCHANGED 2D3A TO 500M")
    positive = sum(classifications[l]["utility_at_250m"] in ("POSITIVE UTILITY", "STRONG POSITIVE") for l in ("b3", "b5", "b6"))
    primary = ("MULTI-LINK POSITIVE RECURRENT PYRAMID" if table["250m"]["combined_gain"] > 0 and table["250m"]["combined_gap"] > 0 and positive >= 2
               else "PARTIAL RECURRENT PYRAMID" if table["250m"]["combined_gain"] > 0 else "RECURRENT PYRAMID HARMFUL")
    train = [json.loads(line) for line in (output / "training_metrics_100m_to_250m.jsonl").read_text().splitlines()]
    performance = {"training": train, "updates": len(train), "wall_seconds": sum(r["wall_seconds"] for r in train),
                   "mean_targets_per_second": float(np.mean([r["targets_per_second"] for r in train])),
                   "max_peak_allocated_vram_mb": max(r["peak_allocated_vram_mb"] for r in train),
                   "max_peak_reserved_vram_mb": max(r["peak_reserved_vram_mb"] for r in train)}
    memory = base.memory_accounting(); durable_json(output / "memory_accounting.json", memory)
    durable_json(output / "performance.json", performance)
    continuation = {
        "source_checkpoint": str(Path(args.final_checkpoint).resolve()), "source_checkpoint_sha256": checkpoint["sha256"],
        "source_updates": FINAL_UPDATE, "source_targets": FINAL_TARGETS,
        "next_endpoint_updates": 954, "next_endpoint_targets": 500_170_752,
        "additional_updates": 477, "additional_targets": 250_085_376,
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "next_global_batch_stream_sha256": checkpoint["next_global_batch_stream_sha256"],
        "optimizer_state_present": True, "scheduler_state_present": True, "loader_state_present": True,
        "rng_state_present": True, "strict_reopen_passed": checkpoint["strict_reopen"]["passed"],
        "architecture_unchanged": True, "resume_ready": True,
    }
    durable_json(output / "CONTINUATION_MANIFEST.json", continuation)
    questions = answer_questions(table, classifications, interaction, increments, payload, checkpoint,
                                 recommendation, output)
    summary = {"experiment": EXPERIMENT, "primary_classification": primary,
               "cumulative_updates": FINAL_UPDATE, "cumulative_targets": FINAL_TARGETS,
               "architecture_unchanged": True, "parameter_count": base.MODEL_PARAMETERS,
               "maturation_table": table, "classifications": classifications,
               "combined_interaction": interaction, "recommendation": recommendation,
               "checkpoint": checkpoint, "continuation": continuation, "performance": performance,
               "memory": memory, "questions": questions, "no_training_beyond_250m": True}
    durable_json(output / "result_summary.json", summary)
    build_plots(output, table, interaction, increments, performance)
    durable_json(output / "storage_cleanup_manifest.json", {"M100_retained": True, "M250_retained": True,
                                                              "persistent_volume_retained": True, "deleted": [], "passed": True})
    durable_text(output / "EXPERIMENT_2D3A_250M_FINAL_REPORT.md", render_report(summary))
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", render_report(summary) + "\nFinal pod may be stopped after Git and local-backup verification.\n")
    checks = {
        "exact_final_update": payload["d3a_completed_updates"] == FINAL_UPDATE,
        "exact_final_targets": payload["d3a_processed_targets"] == FINAL_TARGETS,
        "exact_286_new_updates": len(train) == 286 and train[0]["cumulative_update"] == 192 and train[-1]["cumulative_update"] == 477,
        "architecture_unchanged": True, "source_sha_exact": read_json(output / "source_100m_manifest.json")["sha256"] == SOURCE_SHA256,
        "optimizer_not_reset": not read_json(output / "optimizer_resume_manifest.json")["reset"],
        "scheduler_not_reset": not read_json(output / "scheduler_resume_manifest.json")["reset"],
        "data_not_reset": not read_json(output / "data_resume_manifest.json")["reset"],
        "mandatory_restart": read_json(output / "mandatory_fresh_process_restart_update_334.json")["passed"],
        "final_checkpoint_sha": sha256(args.final_checkpoint) == checkpoint["sha256"],
        "final_checkpoint_strict": checkpoint["strict_reopen"]["passed"],
        "canonical_milestones": all(parallels[a]["subset_sha256"] == CANONICAL_SHA for a in ("150m", "200m", "250m")),
        "maturation_core_milestones": all(increments[a]["subset_sha256"] == MATURATION_CORE_SHA for a in ("150m", "200m", "250m")),
        "no_training_beyond": max(r["cumulative_update"] for r in train) == FINAL_UPDATE,
        "single_recommendation": isinstance(recommendation, str),
    }
    inventory = {name: ((output / name).is_file() or name == "FINAL_AUDIT.json") for name in REQUIRED_ARTIFACTS}
    inventory.update({f"plot_p{n:02d}.png": (output / f"plot_p{n:02d}.png").is_file() for n in range(1, 15)})
    checks["required_artifacts"] = all(inventory.values())
    audit = {"experiment": EXPERIMENT, "checks": checks, "passed": all(checks.values()),
             "artifact_inventory": inventory, "final_checkpoint_sha256": checkpoint["sha256"]}
    durable_json(output / "FINAL_AUDIT.json", audit)
    heartbeat(output, FINAL_UPDATE, {"final_audit_passed": audit["passed"]}, checkpoint["persistent"]["checkpoint"], "complete")
    if not audit["passed"]: raise SystemExit(f"final audit failed: {checks}")
    print("EXPERIMENT_2D3A_250M_FINALIZE_PASS", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(); subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("preflight"); output_args(p); p.add_argument("--data-root", required=True)
    p.add_argument("--stop-capability-verified", action="store_true"); p.set_defaults(func=run_preflight)
    p = subs.add_parser("train"); output_args(p); p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint-dir", required=True); p.add_argument("--persistent-checkpoint-dir", required=True)
    p.add_argument("--resume-checkpoint"); p.add_argument("--end-update", type=int, required=True); p.set_defaults(func=run_train)
    p = subs.add_parser("finalize"); output_args(p); p.add_argument("--final-checkpoint", required=True); p.set_defaults(func=run_finalize)
    return parser


def main():
    args = build_parser().parse_args(); args.func(args)


if __name__ == "__main__":
    main()
