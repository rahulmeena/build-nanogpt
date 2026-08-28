#!/usr/bin/env python3
"""Experiment 2D3A: Alternating-Integration Recurrent Pyramid, 100M stage."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0 as d0  # noqa: E402
import experiment_2d1 as d1  # noqa: E402
import experiment_2d2g as parent  # noqa: E402
from experiment_2d3a_core import (  # noqa: E402
    INCREMENTAL_CONTROLS,
    LOCAL_WINDOWS,
    MAX_RECURRENT_ENTRIES,
    MIN_LAGS,
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
    AlternatingIntegrationRecurrentPyramidGPT,
)


EXPERIMENT = "2D3A"
PROTOCOL = "alternating_integration_recurrent_pyramid_100m_v1"
BRANCH = "experiment-2d3a-alternating-integration-pyramid-100m"
OUTPUT_NAME = "experiment_2d3a_alternating_integration_pyramid_100m"
SCHEMA = "exp2d3a_alternating_integration_pyramid_checkpoint_v1"
ARCHITECTURE_VERSION = "2D3A-AIRP-v1"
SOURCE_SHA256 = "cb5dd5904779617959b5619982a9dfe69f0c4d705679652f4f99a8285879b5e8"
SOURCE_SCHEMA = parent.STAGE_A_SCHEMA
SOURCE_PARAMETERS = 124_475_905
MODEL_PARAMETERS = 124_475_908
SOURCE_NEXT_BATCH = parent.STAGE_A_FINAL_BATCH
SOURCE_NEXT_STREAM = parent.STAGE_A_FINAL_STREAM
UPDATE96_NEXT_BATCH = parent.STAGE_B_UPDATE96_BATCH
UPDATE96_NEXT_STREAM = parent.STAGE_B_UPDATE96_STREAM
FINAL_NEXT_BATCH = parent.STAGE_B_FINAL_BATCH
FINAL_NEXT_STREAM = parent.STAGE_B_FINAL_STREAM
CANONICAL_COLLECTION_SHA = "3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb"
VALIDATION_SHARD_SHA = parent.VALIDATION_SHARD_SHA256

T = 1024
N_LAYER = 12
N_EMBD = 768
N_HEAD = 12
GLOBAL_TARGETS = 524_288
MAX_UPDATES = 191
MAX_TARGETS = 100_139_008
PARENT_HISTORICAL_TARGETS = 350_748_672
MILESTONES = (0, 20, 48, 96, 143, 191)
VALIDATION_B = 64
VALIDATION_BATCHES = 20
INCREMENTAL_BATCHES = 4
SMOKE_UPDATES = 3
BASE_LR = parent.BASE_LR
GATE_LR = parent.GATE_LR
WEIGHT_DECAY = parent.WEIGHT_DECAY
BETAS = parent.BETAS
ADAM_EPS = parent.ADAM_EPS
GRAD_CLIP = parent.GRAD_CLIP
SEED = 2_026_0301

GATE_BLOCKS = {"b1": 0, "b3": 2, "b5": 4, "b6": 5}
SOURCE_KEYS = {"b1": "h12", "b3": "h10", "b5": "h8", "b6": "h7"}
SOURCE_BLOCKS = {"b1": 12, "b3": 10, "b5": 8, "b6": 7}
RECURRENT_BINS = {
    "b1": (("2-7", 2, 7), ("8-15", 8, 15), ("16-31", 16, 31),
           ("32-63", 32, 63), ("64-127", 64, 127), ("128-255", 128, 255),
           ("256-511", 256, 511), ("512-1023", 512, 1023)),
    "b3": (("32-63", 32, 63), ("64-127", 64, 127), ("128-255", 128, 255),
           ("256-511", 256, 511), ("512-1023", 512, 1023)),
    "b5": (("64-127", 64, 127), ("128-255", 128, 255),
           ("256-511", 256, 511), ("512-1023", 512, 1023)),
    "b6": (("512-639", 512, 639), ("640-767", 640, 767),
           ("768-895", 768, 895), ("896-1023", 896, 1023)),
}
LOCAL_BINS = {
    "b1": (("0-1", 0, 1),),
    "b3": (("0-3", 0, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16-31", 16, 31)),
    "b5": (("0-7", 0, 7), ("8-15", 8, 15), ("16-31", 16, 31), ("32-63", 32, 63)),
    "b6": (("0-31", 0, 31), ("32-63", 32, 63), ("64-127", 64, 127),
           ("128-255", 128, 255), ("256-511", 256, 511)),
}
POSITION_BINS = (("1-31", 1, 31), ("32-63", 32, 63), ("64-127", 64, 127),
                 ("128-255", 128, 255), ("256-511", 256, 511),
                 ("512-767", 512, 767), ("768-1023", 768, 1023))

REQUIRED_ARTIFACTS = (
    "EXPERIMENT_2D3A_100M_FINAL_REPORT.md", "FINAL_AUDIT.json", "result_summary.json",
    "source_manifest.json", "architecture_manifest.json", "parameter_manifest.json",
    "named_parameter_diff.json", "optimizer_continuity_manifest.json",
    "scheduler_continuity_manifest.json", "training_metrics.jsonl",
    "milestone_validation.json", "initial_compression_diagnostics.json",
    "gate_diagnostics.json", "b1_attention_diagnostics.json",
    "b3_attention_diagnostics.json", "b5_attention_diagnostics.json",
    "b6_attention_diagnostics.json", "b12_to_b1_temporal_gradients.json",
    "b10_to_b3_temporal_gradients.json", "b8_to_b5_temporal_gradients.json",
    "b7_to_b6_temporal_gradients.json", "paired_controls.json",
    "position_bin_metrics.json", "incremental_validation.json",
    "incremental_cache_audit.json", "b6_representation_control.json",
    "memory_accounting.json", "stability_8pass.json", "performance.json",
    "checkpoint_manifest.json", "maturation_core_subset_manifest.json",
    "CONTINUATION_MANIFEST.json", "storage_cleanup_manifest.json",
    "commands_and_runtime.json", "HEARTBEAT.json", "UNATTENDED_FINAL_HANDOFF.md",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def durable_json(path, value):
    parent.durable_json(path, value)


def durable_text(path, value):
    parent.durable_text(path, value)


def read_json(path):
    return parent.read_json(path)


def append_jsonl(path, value):
    parent.append_jsonl(path, value)


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git_clean():
    branch = git_output("branch", "--show-current")
    if branch != BRANCH:
        raise SystemExit(f"expected branch {BRANCH}, found {branch}")
    dirty = git_output("status", "--porcelain")
    if dirty:
        raise SystemExit(f"working tree must be clean before scientific work:\n{dirty}")


def require_a100():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("2D3A requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    total = torch.cuda.get_device_properties(device).total_memory
    if "A100" not in name or total < 79 * 1024**3:
        raise SystemExit(f"expected one A100 80GB, found {name} {total}")
    return device


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng():
    return parent.capture_rng_state()


def restore_rng(value):
    parent.restore_rng_state(value)


def model_device(model):
    # The inherited 2D2G-A scalar gate is historically CPU-resident while the
    # GPT-2 base is CUDA-resident.  Model execution follows the embedding.
    return model.base.transformer.wte.weight.device


def model_finite(model):
    return parent.model_finite(model)


def optimizer_finite(optimizer):
    return parent.optimizer_finite(optimizer)


def next_batch_hash(loader, accumulation):
    return parent.next_global_batch_hash(loader, accumulation)


def next_stream_hash(loader, accumulation):
    return parent.global_batch_stream_hash(loader, accumulation)


def batch_identity(x, y):
    return parent.batch_identity(x, y)


def aggregate_hashes(values):
    return parent.aggregate_hashes(values)


def validation_path(data_root):
    return parent.validation_path(data_root)


def training_shards(data_root):
    return parent.training_shards(data_root)


def instantiate_base(device):
    return parent.instantiate_base(device)


def architecture_manifest():
    blocks = {}
    for block in range(1, 13):
        if block in (1, 3, 5, 6):
            source = {1: 12, 3: 10, 5: 8, 6: 7}[block]
            window = {1: 2, 3: 32, 5: 64, 6: 512}[block]
            blocks[f"B{block}"] = {
                "local_window": window,
                "source": f"B{source} post-MLP residual",
                "recurrent_lags": [window, 1023],
                "separate_softmax": True,
                "destination_ln_qkv_reused": True,
                "c_proj_applications": 1,
            }
        else:
            blocks[f"B{block}"] = {"local_window": 1024, "recurrent": False}
    return {
        "experiment": EXPERIMENT,
        "architecture_version": ARCHITECTURE_VERSION,
        "sequence_length": T,
        "blocks": blocks,
        "absent_links": ["B11->B2", "B9->B4"],
        "raw_source_rings": ["B12", "B10", "B8", "B7"],
        "parameter_count": MODEL_PARAMETERS,
        "no_attnres": True,
        "no_teacher": True,
        "no_auxiliary_loss": True,
        "no_dedicated_recurrent_projection": True,
    }


def configure_optimizer(model, device_type="cuda"):
    excluded = {"g_rec", "g_rec_b3", "g_rec_b5", "g_rec_b6"}
    decay, nodecay = [], []
    for name, parameter in model.named_parameters():
        if name in excluded:
            continue
        (decay if parameter.dim() >= 2 else nodecay).append(parameter)
    groups = [
        {"name": "base_decay", "params": decay, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
        {"name": "base_nodecay", "params": nodecay, "lr": BASE_LR, "weight_decay": 0.0},
        {"name": "gate", "params": [model.g_rec], "lr": GATE_LR, "weight_decay": 0.0},
        {"name": "b3_gate", "params": [model.g_rec_b3], "lr": GATE_LR, "weight_decay": 0.0},
        {"name": "b5_gate", "params": [model.g_rec_b5], "lr": GATE_LR, "weight_decay": 0.0},
        {"name": "b6_gate", "params": [model.g_rec_b6], "lr": GATE_LR, "weight_decay": 0.0},
    ]
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
    return torch.optim.AdamW(groups, betas=BETAS, eps=ADAM_EPS, fused=fused)


def load_source(path, device, restore=False):
    path = Path(path).resolve()
    if file_sha256(path) != SOURCE_SHA256:
        raise SystemExit("exact 2D2G-A source SHA is unavailable")
    model, optimizer, loader, payload = parent.load_stage_a_checkpoint(path, device, restore_rng=restore)
    if payload.get("schema") != SOURCE_SCHEMA:
        raise SystemExit("source schema mismatch")
    if sum(p.numel() for p in model.parameters()) != SOURCE_PARAMETERS:
        raise SystemExit("source parameter count mismatch")
    names = set(dict(model.named_parameters()))
    if "g_rec" not in names or any(name in names for name in ("g_rec_b3", "g_rec_b5", "g_rec_b6")):
        raise SystemExit("source recurrent-gate inventory mismatch")
    return model, optimizer, loader, payload


def transplant(source_model, source_optimizer, device):
    model = AlternatingIntegrationRecurrentPyramidGPT(source_model.base).to(device)
    model.g_rec = source_model.g_rec
    optimizer = configure_optimizer(model, device.type)
    fresh = {model.g_rec_b3, model.g_rec_b5, model.g_rec_b6}
    shared = set(model.parameters()) - fresh
    source_parameters = {p for group in source_optimizer.param_groups for p in group["params"]}
    if shared != source_parameters:
        raise SystemExit("source/2D3A parameter identity mismatch")
    for parameter in shared:
        if parameter in source_optimizer.state:
            optimizer.state[parameter] = copy.deepcopy(source_optimizer.state[parameter])
    source_groups = {group["name"]: group for group in source_optimizer.param_groups}
    for group in optimizer.param_groups:
        if group["name"] not in source_groups:
            continue
        source = source_groups[group["name"]]
        for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "capturable", "differentiable"):
            if key in source:
                group[key] = source[key]
    checks = {
        "source_parameter_identity": shared == source_parameters,
        "source_optimizer_state_entries": len(optimizer.state) == len(source_optimizer.state),
        "parameters": sum(p.numel() for p in model.parameters()) == MODEL_PARAMETERS,
        "three_fresh_zero_gates": all(p.detach().float().item() == 0.0 for p in fresh),
        "fresh_gate_state_absent": all(p not in optimizer.state for p in fresh),
        "weight_tying": model.base.transformer.wte.weight is model.base.lm_head.weight,
    }
    if not all(checks.values()):
        raise SystemExit(f"source transplant failed: {checks}")
    return model, optimizer, checks


def loader_at_cursor(loader_state, micro_batch):
    state = copy.deepcopy(loader_state)
    state["batch_size"] = int(micro_batch)
    return d1.ExplicitShardLoader(state["shards"], int(micro_batch), T, state=state)


def accumulation_for(micro_batch):
    denominator = int(micro_batch) * T
    if GLOBAL_TARGETS % denominator:
        raise ValueError("microbatch does not divide the logical global batch")
    return GLOBAL_TARGETS // denominator


def pass_count(update):
    return 3 if int(update) % 32 == 0 else 2


def control_kwargs(name, permutation=None):
    kwargs = {}
    if name == "new_links_off":
        kwargs.update(b3_gate_override=0.0, b5_gate_override=0.0, b6_gate_override=0.0)
    elif name in {"b1_off", "b3_off", "b5_off", "b6_off"}:
        kwargs[f"{name[:2]}_gate_override"] = 0.0
    elif name in {"b1_shuffled", "b3_shuffled", "b5_shuffled", "b6_shuffled"}:
        kwargs[f"{name[:2]}_recurrent_permutation"] = permutation
    elif name == "all_new_shuffled":
        for link in ("b3", "b5", "b6"):
            kwargs[f"{link}_recurrent_permutation"] = permutation
    elif name == "all_recurrent_shuffled":
        for link in ("b1", "b3", "b5", "b6"):
            kwargs[f"{link}_recurrent_permutation"] = permutation
    elif name == "b6_full_native":
        kwargs.update(b6_gate_override=0.0, full_counterfactual_blocks=(5,))
    elif name != "all_real":
        raise ValueError(f"unknown control {name}")
    return kwargs


def paired_stats(left, right):
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    return {
        "count": len(differences),
        "mean": statistics.fmean(differences),
        "median": statistics.median(differences),
        "sample_std": statistics.stdev(differences) if len(differences) > 1 else 0.0,
        "wins": sum(value < 0 for value in differences),
        "losses": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "differences_left_minus_right": differences,
    }


def evaluate_source(model, val_path, batches=VALIDATION_BATCHES):
    model.eval()
    device = model_device(model)
    loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    total = 0.0
    targets = 0
    identities = []
    losses = []
    with torch.no_grad():
        for _ in range(int(batches)):
            cpu_x, cpu_y = loader.next_batch()
            identities.append(batch_identity(cpu_x, cpu_y))
            x, y = cpu_x.to(device), cpu_y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                row = model.forward_multi_pass(x, targets=y, num_passes=2)
            loss = row["loss"].detach().float().item()
            total += loss * y.numel()
            targets += y.numel()
            losses.append(loss)
    subset = aggregate_hashes(row["combined_sha256"] for row in identities)
    return {"validation_loss": total / targets, "validation_targets": targets,
            "per_batch_losses": losses, "batch_identities": identities,
            "subset_sha256": subset}


def evaluate_parallel(model, val_path, names, batches=VALIDATION_BATCHES):
    model.eval()
    device = model_device(model)
    loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    rows = {name: {"sum": 0.0, "targets": 0, "batches": [], "positions": torch.zeros(T, dtype=torch.float64)} for name in names}
    identities = []
    permutation = torch.arange(VALIDATION_B, device=device).roll(1)
    with torch.no_grad():
        for _ in range(int(batches)):
            cpu_x, cpu_y = loader.next_batch()
            identities.append(batch_identity(cpu_x, cpu_y))
            x, y = cpu_x.to(device), cpu_y.to(device)
            for name in names:
                kwargs = control_kwargs(name, permutation)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = model.forward_multi_pass(x, targets=y, num_passes=2, **kwargs)
                token_loss = F.cross_entropy(
                    result["logits"].float().reshape(-1, result["logits"].size(-1)),
                    y.reshape(-1), reduction="none"
                ).view(y.shape)
                loss = result["loss"].detach().float().item()
                rows[name]["sum"] += loss * y.numel()
                rows[name]["targets"] += y.numel()
                rows[name]["batches"].append(loss)
                rows[name]["positions"] += token_loss.double().sum(0).cpu()
    subset = aggregate_hashes(row["combined_sha256"] for row in identities)
    controls = {}
    for name, row in rows.items():
        controls[name] = {
            "validation_loss": row["sum"] / row["targets"],
            "validation_targets": row["targets"],
            "per_batch_losses": row["batches"],
            "per_position_loss": (row["positions"] / (int(batches) * VALIDATION_B)).tolist(),
        }
    real = controls["all_real"]
    metrics = {"controls": controls, "batch_identities": identities,
               "subset_sha256": subset, "canonical_collection_match": subset == CANONICAL_COLLECTION_SHA}
    for link in ("b1", "b3", "b5", "b6"):
        off, shuffled = f"{link}_off", f"{link}_shuffled"
        if off in controls:
            metrics[f"{link}_gain"] = controls[off]["validation_loss"] - real["validation_loss"]
        if shuffled in controls:
            metrics[f"{link}_sequence_gap"] = controls[shuffled]["validation_loss"] - real["validation_loss"]
    if "new_links_off" in controls:
        metrics["combined_new_link_gain"] = controls["new_links_off"]["validation_loss"] - real["validation_loss"]
    if "all_new_shuffled" in controls:
        metrics["combined_new_sequence_gap"] = controls["all_new_shuffled"]["validation_loss"] - real["validation_loss"]
    metrics["gates"] = gate_values(model)
    return metrics


def gate_values(model):
    return {link: {"raw": model.gate_parameter(block).detach().float().item(),
                   "effective": model.recurrent_scale(block).detach().float().item()}
            for link, block in GATE_BLOCKS.items()}


def gradient_groups(model):
    excluded = {"g_rec", "g_rec_b3", "g_rec_b5", "g_rec_b6"}
    values = {
        "base": [p.grad for name, p in model.named_parameters() if name not in excluded and p.grad is not None],
        "b1_gate": [] if model.g_rec.grad is None else [model.g_rec.grad],
        "b3_gate": [] if model.g_rec_b3.grad is None else [model.g_rec_b3.grad],
        "b5_gate": [] if model.g_rec_b5.grad is None else [model.g_rec_b5.grad],
        "b6_gate": [] if model.g_rec_b6.grad is None else [model.g_rec_b6.grad],
    }
    report = {}
    for name, tensors in values.items():
        squared = sum(value.float().square().sum() for value in tensors) if tensors else torch.tensor(0.0)
        report[name] = {"tensors": len(tensors), "norm": squared.sqrt().item(),
                        "finite": all(bool(torch.isfinite(v).all()) for v in tensors),
                        "nonzero": bool(tensors) and bool(squared.gt(0).item())}
    return report


def train_update(model, optimizer, loader, accumulation, update, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    passes = pass_count(update)
    totals = [0.0] * passes
    pass_seconds = [0.0] * passes
    before = gate_values(model)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(accumulation):
        cpu_x, cpu_y = loader.next_batch()
        x, y = cpu_x.to(device, non_blocking=True), cpu_y.to(device, non_blocking=True)
        pass_started = time.monotonic()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model.forward_multi_pass(
                x, targets=y, num_passes=passes, activation_checkpointing=True
            )
            loss = result["loss"] / accumulation
        forward_elapsed = time.monotonic() - pass_started
        for index, value in enumerate(result["pass_losses"]):
            totals[index] += value.detach().float().item()
            pass_seconds[index] += forward_elapsed / passes
        loss.backward()
        del x, y, cpu_x, cpu_y, result, loss
    groups = gradient_groups(model)
    if not all(row["finite"] and row["nonzero"] for row in groups.values()):
        raise SystemExit(f"missing or nonfinite gradient group: {groups}")
    gate_grads = {
        link: model.gate_parameter(block).grad.detach().float().item()
        for link, block in GATE_BLOCKS.items()
    }
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not torch.isfinite(norm):
        raise SystemExit("nonfinite gradient norm")
    optimizer.step()
    if not model_finite(model) or not optimizer_finite(optimizer):
        raise SystemExit("nonfinite model or optimizer after update")
    elapsed = time.monotonic() - started
    return {
        "local_update": int(update), "d3a_processed_targets": int(update) * GLOBAL_TARGETS,
        "pass_count": passes, "pass_weights": list((.25, .75) if passes == 2 else (.2, .4, .4)),
        "pass_losses": [value / accumulation for value in totals],
        "approximate_pass_forward_seconds": [value for value in pass_seconds],
        "gradient_groups": groups, "gate_gradients": gate_grads,
        "gate_before": before, "gate_after": gate_values(model),
        "gradient_norm_before_clip": norm.detach().float().item(),
        "wall_seconds": elapsed, "targets_per_second": GLOBAL_TARGETS / elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def scheduler_manifest(source_payload, optimizer):
    groups = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
    return {
        "inherited_scheduler_present_in_source": "scheduler" in source_payload,
        "policy": "constant optimizer-group learning rates; no scheduler object in 2D2G-A checkpoint",
        "scheduler_state": source_payload.get("scheduler"),
        "scheduler_step": None,
        "current_lrs": groups,
        "expected_lrs": {age: groups for age in ("100M", "250M", "500M", "1B")},
        "warmup_restarted": False,
        "well_defined_through_100m": True,
    }


def checkpoint_payload(model, optimizer, loader, completed, accumulation, metadata, source_payload):
    return {
        "schema_version": SCHEMA, "schema": SCHEMA, "experiment_name": EXPERIMENT,
        "architecture_version": ARCHITECTURE_VERSION, "parent_experiment": "2D2G-A",
        "parent_checkpoint_path": metadata["parent_checkpoint_path"],
        "parent_checkpoint_sha256": SOURCE_SHA256, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler_manifest(source_payload, optimizer),
        "d3a_completed_updates": int(completed),
        "d3a_processed_targets": int(completed) * GLOBAL_TARGETS,
        "parent_historical_targets": PARENT_HISTORICAL_TARGETS,
        "loader_state": loader.state_dict(), "loader_states": [loader.state_dict()],
        "rng_state": capture_rng(), "targets_per_update": GLOBAL_TARGETS,
        "gradient_accumulation": int(accumulation),
        "next_global_batch_sha256": next_batch_hash(loader, accumulation),
        "next_global_batch_stream_sha256": next_stream_hash(loader, accumulation),
        "local_windows": {"B1": 2, "B2": 1024, "B3": 32, "B4": 1024,
                          "B5": 64, "B6": 512, **{f"B{i}": 1024 for i in range(7, 13)}},
        "recurrent_sources": {"B1": "B12", "B3": "B10", "B5": "B8", "B6": "B7"},
        "recurrent_lag_ranges": {"B1": [2, 1023], "B3": [32, 1023],
                                 "B5": [64, 1023], "B6": [512, 1023]},
        "raw_gate_values": gate_values(model),
        "optimizer_group_definitions": [{k: v for k, v in group.items() if k != "params"}
                                        for group in optimizer.param_groups],
        "current_lr_per_group": {group["name"]: group["lr"] for group in optimizer.param_groups},
        "git_implementation_commit": metadata["git_implementation_commit"],
        "data_manifest": metadata["data_manifest"],
        "canonical_validation_manifest": metadata["canonical_validation_manifest"],
        "true_self_maturation_core_subset_manifest": metadata["maturation_core_subset_manifest"],
        "hardware_metadata": metadata["hardware_metadata"],
        "precision_settings": metadata["precision_settings"],
        "metadata": metadata, "saved_process_id": os.getpid(), "saved_at_unix": time.time(),
    }


def expected_cursor(completed):
    return {96: (UPDATE96_NEXT_BATCH, UPDATE96_NEXT_STREAM),
            191: (FINAL_NEXT_BATCH, FINAL_NEXT_STREAM)}.get(int(completed))


def strict_reopen(path, completed, metadata, device):
    payload = d0.torch_load(Path(path), mmap=False)
    _, base = instantiate_base(device)
    model = AlternatingIntegrationRecurrentPyramidGPT(base).to(device)
    # Reproduce the source line's mixed-device optimizer geometry exactly.
    model.g_rec = torch.nn.Parameter(torch.zeros(()))
    model.load_state_dict(payload["model"], strict=True)
    optimizer = configure_optimizer(model, device.type)
    optimizer.load_state_dict(payload["optimizer"])
    loader = d1.ExplicitShardLoader(
        payload["loader_state"]["shards"], payload["loader_state"]["batch_size"], T,
        state=payload["loader_state"]
    )
    accumulation = int(payload["gradient_accumulation"])
    checks = {
        "schema": payload.get("schema") == SCHEMA,
        "architecture": payload.get("architecture_version") == ARCHITECTURE_VERSION,
        "updates": payload.get("d3a_completed_updates") == int(completed),
        "targets": payload.get("d3a_processed_targets") == int(completed) * GLOBAL_TARGETS,
        "metadata": payload.get("metadata") == metadata,
        "model_parameters": sum(p.numel() for p in model.parameters()) == MODEL_PARAMETERS,
        "model_finite": model_finite(model), "optimizer_finite": optimizer_finite(optimizer),
        "loader_next_batch": next_batch_hash(loader, accumulation) == payload["next_global_batch_sha256"],
        "loader_next_stream": next_stream_hash(loader, accumulation) == payload["next_global_batch_stream_sha256"],
        "rng_complete": set(payload.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "scheduler_present": "scheduler" in payload,
        "continuation_fields": all(key in payload for key in (
            "optimizer", "scheduler", "loader_state", "rng_state", "raw_gate_values",
            "recurrent_sources", "recurrent_lag_ranges", "git_implementation_commit")),
    }
    expected = expected_cursor(completed)
    if expected:
        checks["preregistered_next_cursor"] = (
            payload["next_global_batch_sha256"], payload["next_global_batch_stream_sha256"]
        ) == expected
    audit = {"checks": checks, "passed": all(checks.values())}
    del model, optimizer, loader, base, payload
    gc.collect(); torch.cuda.empty_cache()
    return audit


def save_checkpoint(path, model, optimizer, loader, completed, accumulation, metadata, source_payload, device):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, optimizer, loader, completed, accumulation, metadata, source_payload)
    expected = expected_cursor(completed)
    if expected and (payload["next_global_batch_sha256"], payload["next_global_batch_stream_sha256"]) != expected:
        raise SystemExit(f"update {completed} next-batch continuity mismatch")
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    sha = file_sha256(path)
    audit = strict_reopen(path, completed, metadata, device)
    if not audit["passed"]:
        raise SystemExit(f"strict checkpoint reopen failed: {audit}")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), audit)
    return {"checkpoint": str(path.resolve()), "sha256": sha, "bytes": path.stat().st_size,
            "next_global_batch_sha256": payload["next_global_batch_sha256"],
            "next_global_batch_stream_sha256": payload["next_global_batch_stream_sha256"],
            "strict_reopen": audit}


def load_d3a_checkpoint(path, device, restore=False):
    payload = d0.torch_load(Path(path), mmap=False)
    if payload.get("schema") != SCHEMA:
        raise SystemExit("not a 2D3A checkpoint")
    _, base = instantiate_base(device)
    model = AlternatingIntegrationRecurrentPyramidGPT(base).to(device)
    model.g_rec = torch.nn.Parameter(torch.zeros(()))
    model.load_state_dict(payload["model"], strict=True)
    optimizer = configure_optimizer(model, device.type)
    optimizer.load_state_dict(payload["optimizer"])
    loader = d1.ExplicitShardLoader(payload["loader_state"]["shards"],
                                    payload["loader_state"]["batch_size"], T,
                                    state=payload["loader_state"])
    if restore:
        restore_rng(payload["rng_state"])
    return model, optimizer, loader, payload


def persist_triplet(source_path, persistent_dir):
    source_path = Path(source_path).resolve()
    persistent_dir = Path(persistent_dir).resolve()
    persistent_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in (source_path, source_path.with_suffix(source_path.suffix + ".sha256"),
                   source_path.with_suffix(source_path.suffix + ".verification.json")):
        destination = persistent_dir / source.name
        temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        rows.append({"source": str(source), "destination": str(destination),
                     "sha256": file_sha256(destination), "bytes": destination.stat().st_size})
    if rows[0]["sha256"] != file_sha256(source_path):
        raise SystemExit("persistent checkpoint copy SHA mismatch")
    return {"files": rows, "passed": True, "checkpoint": rows[0]["destination"], "sha256": rows[0]["sha256"]}


def training_metadata(args, micro_batch, accumulation):
    val_path = validation_path(args.data_root)
    return {
        "experiment": EXPERIMENT, "protocol": PROTOCOL, "branch": BRANCH,
        "parent_checkpoint_path": str(Path(args.source_checkpoint).resolve()),
        "parent_checkpoint_sha256": SOURCE_SHA256,
        "git_implementation_commit": git_output("rev-parse", "HEAD"),
        "targets_per_update": GLOBAL_TARGETS, "micro_batch": micro_batch,
        "gradient_accumulation": accumulation, "sequence_length": T,
        "two_pass_weights": [.25, .75], "three_pass_weights": [.2, .4, .4],
        "pass3_every": 32, "mandatory_restart_update": 96,
        "data_manifest": {"training_shards": [str(p.resolve()) for p in training_shards(args.data_root)],
                          "validation_shard": str(val_path.resolve()),
                          "validation_sha256": file_sha256(val_path)},
        "canonical_validation_manifest": {"batches": 20, "batch_size": 64,
                                          "sequence_length": 1024,
                                          "collection_sha256": CANONICAL_COLLECTION_SHA},
        "maturation_core_subset_manifest": {"batches": 4, "batch_size": 64,
                                            "sequence_length": 1024,
                                            "targets_per_control": 262144,
                                            "source": "first four canonical validation batches"},
        "hardware_metadata": {"pod_id": args.pod_id, "pod_name": args.pod_name,
                              "gpu": torch.cuda.get_device_name(0), "gpu_count": 1},
        "precision_settings": {"training": "BF16 autocast", "optimizer": "AdamW",
                               "loss_accumulation": "FP32", "incremental": "BF16 autocast"},
    }


def parameter_manifests(source_model, model):
    source = dict(source_model.named_parameters())
    destination = dict(model.named_parameters())
    new = sorted(set(destination) - set(source))
    changed = sorted(name for name in set(source) & set(destination)
                     if source[name] is not destination[name])
    manifest = {"source_parameters": sum(p.numel() for p in source_model.parameters()),
                "d3a_parameters": sum(p.numel() for p in model.parameters()),
                "new_parameter_tensors": new, "new_tensor_count": len(new),
                "new_parameter_count": sum(destination[name].numel() for name in new),
                "passed": new == ["g_rec_b3", "g_rec_b5", "g_rec_b6"] and not changed}
    diff = {"new": {name: {"shape": list(destination[name].shape),
                            "initial_value": destination[name].detach().float().item()} for name in new},
            "removed": sorted(set(source) - set(destination)),
            "shared_parameter_identity_changed": changed,
            "passed": manifest["passed"]}
    return manifest, diff


def causality_audit(model, length=640):
    device = model_device(model)
    rows = {}
    for link, block in GATE_BLOCKS.items():
        local = model.local_mask(block, length, device)
        recurrent = model.recurrent_mask(block, length, length, device)
        query = torch.arange(length, device=device).view(length, 1)
        source = torch.arange(length, device=device).view(1, length)
        lag = query - source
        rows[link] = {
            "local_exact": bool(torch.equal(local, (lag >= 0) & (lag < LOCAL_WINDOWS[block]))),
            "recurrent_exact": bool(torch.equal(recurrent, (lag >= MIN_LAGS[block]) & (lag <= 1023))),
            "nonoverlap": not bool((local & recurrent).any()),
            "no_future": not bool((recurrent & (lag <= 0)).any()),
        }
    return {"links": rows, "passed": all(all(row.values()) for row in rows.values())}


def future_and_row_isolation(model):
    device = model_device(model)
    torch.manual_seed(71)
    tokens = torch.randint(0, 50257, (2, 160), device=device)
    mutated = tokens.clone(); mutated[:, 100:] = torch.randint(0, 50257, mutated[:, 100:].shape, device=device)
    row_mutated = tokens.clone(); row_mutated[1] = torch.randint(0, 50257, row_mutated[1].shape, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        left = model.forward_multi_pass(tokens, num_passes=2)["logits"]
        future = model.forward_multi_pass(mutated, num_passes=2)["logits"]
        isolated = model.forward_multi_pass(row_mutated, num_passes=2)["logits"]
    prefix = (left[:, :100].float() - future[:, :100].float()).abs().max().item()
    row = (left[0].float() - isolated[0].float()).abs().max().item()
    return {"future_prefix_max_abs": prefix, "row0_max_abs": row,
            "future_token_invariance": prefix == 0.0, "row_isolation": row == 0.0,
            "passed": prefix == 0.0 and row == 0.0}


def zero_gate_identities(model):
    device = model_device(model)
    tokens = torch.randint(0, 50257, (2, 160), device=device)
    with torch.no_grad():
        first = model.forward_pass(tokens)
        sources = {"b1_recurrent_source": first["h12"], "b3_recurrent_source": first["h10"],
                   "b5_recurrent_source": first["h8"], "b6_recurrent_source": first["h7"]}
        rows = {}
        for link in ("b3", "b5", "b6"):
            with_bank = model.forward_pass(tokens, **sources, **{f"{link}_gate_override": 0.0})["logits"]
            absent_sources = dict(sources); absent_sources[f"{link}_recurrent_source"] = None
            absent = model.forward_pass(tokens, **absent_sources)["logits"]
            rows[link] = {"exact": bool(torch.equal(with_bank, absent)),
                          "max_abs": (with_bank.float() - absent.float()).abs().max().item()}
    return {"links": rows, "passed": all(row["exact"] for row in rows.values())}


def incremental_smoke(model):
    device = model_device(model)
    tokens = torch.randint(0, 50257, (2, 136), device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parallel = model.forward_multi_pass(tokens, num_passes=2)["logits"]
        incremental = model.incremental_logits(tokens, control="all_real")
    delta = (parallel.float() - incremental["logits"].float()).abs()
    return {"cache_audit": incremental["cache_audit"],
            "parallel_incremental_max_abs": delta.max().item(),
            "parallel_incremental_mean_abs": delta.mean().item(),
            "finite": bool(torch.isfinite(incremental["logits"]).all()),
            "passed": incremental["cache_audit"]["passed"] and bool(torch.isfinite(incremental["logits"]).all())}


def initial_compression_diagnostics(source_model, model, val_path):
    source = evaluate_source(source_model, val_path)
    names = ["all_real", "new_links_off"]
    compressed = evaluate_parallel(model, val_path, names)
    def evaluate_geometry(full_blocks):
        model.eval(); device = model_device(model)
        loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
        total = 0.0; targets = 0; identities = []
        with torch.no_grad():
            for _ in range(VALIDATION_BATCHES):
                cpu_x, cpu_y = loader.next_batch(); identities.append(batch_identity(cpu_x, cpu_y))
                x, y = cpu_x.to(device), cpu_y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    row = model.forward_multi_pass(
                        x, targets=y, num_passes=2, b3_gate_override=0.0,
                        b5_gate_override=0.0, b6_gate_override=0.0,
                        full_counterfactual_blocks=full_blocks
                    )
                total += row["loss"].item() * y.numel(); targets += y.numel()
        return {"validation_loss": total / targets, "validation_targets": targets,
                "subset_sha256": aggregate_hashes(row["combined_sha256"] for row in identities)}
    source_reproduction = evaluate_geometry((2, 4, 5))
    source_reproduction["ce_delta_vs_source"] = source_reproduction["validation_loss"] - source["validation_loss"]
    source_reproduction["passed"] = (
        source_reproduction["ce_delta_vs_source"] == 0.0
        and source_reproduction["subset_sha256"] == source["subset_sha256"]
    )
    singles = {}
    for link, full_others in {
        "B3_W32_ONLY": (4, 5), "B5_W64_ONLY": (2, 5), "B6_W512_ONLY": (2, 4)
    }.items():
        row = evaluate_geometry(full_others)
        singles[link] = {**row, "damage_vs_source": row["validation_loss"] - source["validation_loss"]}
    return {"SOURCE_2D2GA": source, "SOURCE_2D2GA_REPRODUCED_BY_2D3A_KERNEL": source_reproduction,
            "COMPRESSED_NEW_OFF": compressed["controls"]["new_links_off"],
            "initial_joint_compression_damage": compressed["controls"]["new_links_off"]["validation_loss"] - source["validation_loss"],
            "single_change_diagnostics": singles,
            "canonical_collection_match": source["subset_sha256"] == CANONICAL_COLLECTION_SHA == compressed["subset_sha256"]}


def attention_diagnostic(model, val_path, link):
    model.eval(); device = model_device(model)
    loader = d1.ExplicitShardLoader([val_path], 1, T)
    x, y = loader.next_batch(); x, y = x.to(device), y.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first = model.forward_pass(x, targets=y)
        second = model.forward_pass(x, targets=y, b1_recurrent_source=first["h12"],
                                    b3_recurrent_source=first["h10"], b5_recurrent_source=first["h8"],
                                    b6_recurrent_source=first["h7"], return_diagnostics=True)
    block = GATE_BLOCKS[link]
    diag = second["diagnostics"][block]
    rec = diag["recurrent_attention_weights"].detach().float().cpu()
    rec_mask = diag["recurrent_valid_mask"].cpu()
    local = diag["local_attention_weights"].detach().float().cpu()
    local_mask = diag["local_valid_mask"].cpu()

    def summarize(weights, mask, bins):
        per_head_num = torch.zeros(weights.size(1)); per_head_den = torch.zeros(weights.size(1))
        lag_mass = {name: 0.0 for name, _, _ in bins}; lag_count = {name: 0 for name, _, _ in bins}
        # Aggregate mass by lag directly.  This is mathematically identical to
        # sorting every (lag, mass) tuple, because the requested quantiles are
        # over lag and all equal-lag entries are contiguous in that ordering.
        # It avoids materializing and sorting millions of Python tuples.
        head_lag_mass = torch.zeros((weights.size(1), T), dtype=torch.float64)
        entropy = 0.0; rows = 0
        for query in range(weights.size(2)):
            source = torch.where(mask[query])[0]
            if not source.numel(): continue
            current = weights[0, :, query, source]
            lags = query - source
            mean = current.mean(0)
            for name, low, high in bins:
                selected = (lags >= low) & (lags <= high)
                lag_mass[name] += mean[selected].sum().item(); lag_count[name] += int(selected.sum())
            per_head_num += (current * lags.float()).sum(1); per_head_den += current.sum(1)
            head_lag_mass[:, lags] += current.double()
            entropy += (-(current.clamp_min(1e-30).log() * current).sum(1)).mean().item(); rows += 1
        means = (per_head_num / per_head_den.clamp_min(1e-30)).tolist()
        quantiles = []
        for sample in head_lag_mass:
            total = sample.sum().item()
            cumulative = sample.cumsum(0)
            values = []
            for q in (.5, .9):
                threshold = total * q
                answer = int(torch.searchsorted(cumulative, torch.tensor(threshold, dtype=cumulative.dtype)).clamp_max(T - 1).item())
                values.append(answer)
            quantiles.append({"median_lag": values[0], "p90_lag": values[1]})
        return {"bins": {name: {"raw_mass": lag_mass[name] / max(rows, 1),
                                 "normalized_mass_per_available_token": lag_mass[name] / max(lag_count[name], 1)}
                         for name, _, _ in bins},
                "entropy": entropy / max(rows, 1), "effective_positions": math.exp(entropy / max(rows, 1)),
                "per_head": [{"head": index, "mean_lag": mean, **quantiles[index]}
                             for index, mean in enumerate(means)]}
    return {"link": link, "source_block": SOURCE_BLOCKS[link],
            "destination_block": block + 1, "recurrent": summarize(rec, rec_mask, RECURRENT_BINS[link]),
            "local": summarize(local, local_mask, LOCAL_BINS[link]),
            "gate": gate_values(model)[link]}


def temporal_gradients(model, val_path):
    model.train(); device = model_device(model)
    loader = d1.ExplicitShardLoader([val_path], 2, T)
    x, y = loader.next_batch(); x, y = x.to(device), y.to(device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first = model.forward_pass(x, targets=y)
        for key in ("h12", "h10", "h8", "h7"): first[key].retain_grad()
        second = model.forward_pass(x, b1_recurrent_source=first["h12"],
                                    b3_recurrent_source=first["h10"], b5_recurrent_source=first["h8"],
                                    b6_recurrent_source=first["h7"])
        loss = F.cross_entropy(second["logits"][:, -1].float(), y[:, -1])
    loss.backward()
    result = {}
    for link, key in SOURCE_KEYS.items():
        grad = first[key].grad.detach().float()
        by_position = grad.square().mean((0, 2)).sqrt()
        rows = {}
        for name, low, high in RECURRENT_BINS[link]:
            source_low = T - 1 - high; source_high = T - 1 - low
            selected = by_position[max(0, source_low): min(T - 1, source_high) + 1]
            elements = grad[:, max(0, source_low): min(T - 1, source_high) + 1]
            rows[name] = {"mean_gradient_rms": selected.mean().item(),
                          "max_gradient_rms": selected.max().item(),
                          "fraction_nonzero_source_positions": (selected > 0).float().mean().item(),
                          "fraction_nonzero_elements": (elements != 0).float().mean().item()}
        result[link] = {"source": f"B{SOURCE_BLOCKS[link]}", "destination": link.upper(),
                        "last_token_ce": loss.detach().item(), "bins": rows,
                        "all_eligible_bins_nonzero": all(row["mean_gradient_rms"] > 0 for row in rows.values())}
    model.zero_grad(set_to_none=True)
    return result


def stability_8pass(model, val_path):
    model.eval(); device = model_device(model)
    loader = d1.ExplicitShardLoader([val_path], 2, T)
    x, y = loader.next_batch(); x, y = x.to(device), y.to(device)
    sources = {"h12": None, "h10": None, "h8": None, "h7": None}; rows = []
    with torch.no_grad():
        for index in range(1, 9):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                row = model.forward_pass(x, targets=y, b1_recurrent_source=sources["h12"],
                                         b3_recurrent_source=sources["h10"], b5_recurrent_source=sources["h8"],
                                         b6_recurrent_source=sources["h7"], return_diagnostics=True)
            current = {"pass": index, "ce": row["loss"].item(),
                       **{f"{key}_rms": row[key].float().square().mean().sqrt().item()
                          for key in ("h12", "h10", "h8", "h7")}}
            for link, block in GATE_BLOCKS.items():
                diag = row["diagnostics"].get(block)
                current[f"{link}_recurrent_output_rms"] = 0.0 if diag is None else diag["recurrent_output_rms"].item()
            current["finite"] = all(math.isfinite(value) for key, value in current.items() if key not in ("pass", "finite"))
            rows.append(current); sources = {key: row[key] for key in sources}
    return {"passes": rows, "passed": all(row["finite"] for row in rows)}


def incremental_control(model, x, y, name, permutation):
    state = model.init_incremental_state(x.size(0), device=x.device,
                                         b6_full_native=name == "b6_full_native")
    per_sequence = torch.zeros(x.size(0), dtype=torch.float64)
    per_position = torch.zeros(T, dtype=torch.float64)
    cache_rows = []
    shuffled = "shuffled" in name
    for position in range(T):
        logits, state, diag = model.incremental_step(
            x[:, position], state, control=name,
            recurrent_permutation=permutation if shuffled else None,
            return_diagnostics=True, diagnostic_attention_weights=False
        )
        losses = F.cross_entropy(logits[:, 0].float(), y[:, position], reduction="none").double().cpu()
        per_sequence += losses; per_position[position] += losses.sum()
        if position in (0, 1, 31, 63, 511, 1023): cache_rows.append(diag["cache_audit"])
    return {"loss_sum": per_sequence.sum().item(), "targets": x.numel(),
            "per_sequence_losses": (per_sequence / T).tolist(),
            "per_position_sum": per_position.tolist(), "cache_rows": cache_rows,
            "final_cache_audit": model.incremental_cache_audit(state)}


def evaluate_incremental(model, val_path):
    model.eval(); device = model_device(model)
    loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    names = [name for name in INCREMENTAL_CONTROLS if name != "b6_full_native"]
    rows = {name: {"sum": 0.0, "targets": 0, "sequences": [],
                   "positions": np.zeros(T, dtype=np.float64), "cache_rows": []} for name in names}
    identities = []; permutation = torch.arange(VALIDATION_B, device=device).roll(1)
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch_index in range(INCREMENTAL_BATCHES):
            cpu_x, cpu_y = loader.next_batch(); identities.append(batch_identity(cpu_x, cpu_y))
            x, y = cpu_x.to(device), cpu_y.to(device)
            for name in names:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    current = incremental_control(model, x, y, name, permutation)
                row = rows[name]; row["sum"] += current["loss_sum"]; row["targets"] += current["targets"]
                row["sequences"].extend(current["per_sequence_losses"])
                row["positions"] += np.array(current["per_position_sum"])
                row["cache_rows"].append(current["final_cache_audit"])
            print(f"2D3A incremental batch {batch_index + 1}/4", flush=True)
            del x, y, cpu_x, cpu_y; torch.cuda.empty_cache()
    controls = {name: {"validation_loss": row["sum"] / row["targets"],
                       "validation_targets": row["targets"],
                       "per_sequence_losses": row["sequences"],
                       "per_position_loss": (row["positions"] / (INCREMENTAL_BATCHES * VALIDATION_B)).tolist(),
                       "cache_rows": row["cache_rows"]} for name, row in rows.items()}
    real = controls["all_real"]
    paired = {}; metrics = {}
    for link in ("b1", "b3", "b5", "b6"):
        off, shuffled = controls[f"{link}_off"], controls[f"{link}_shuffled"]
        metrics[f"true_{link}_gain"] = off["validation_loss"] - real["validation_loss"]
        metrics[f"true_{link}_sequence_gap"] = shuffled["validation_loss"] - real["validation_loss"]
        paired[link] = {"real_vs_off": paired_stats(real["per_sequence_losses"], off["per_sequence_losses"]),
                        "real_vs_shuffled": paired_stats(real["per_sequence_losses"], shuffled["per_sequence_losses"])}
    metrics["combined_new_link_gain"] = controls["new_links_off"]["validation_loss"] - real["validation_loss"]
    metrics["combined_new_sequence_gap"] = controls["all_new_shuffled"]["validation_loss"] - real["validation_loss"]
    subset = aggregate_hashes(row["combined_sha256"] for row in identities)
    return {"controls": controls, **metrics, "paired": paired, "batch_identities": identities,
            "subset_sha256": subset, "targets_per_control": 262144, "paired_sequences": 256,
            "same_sequences_all_controls": True, "same_derangement_all_shuffled": True,
            "no_complete_prefix_recomputation": True,
            "performance": {"wall_seconds": time.monotonic() - started,
                            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2}}


def b6_representation_control(model, val_path):
    parallel = evaluate_parallel(model, val_path, ["all_real", "b6_off", "b6_full_native"])
    controls = parallel["controls"]
    return {"R_B6": controls["all_real"], "O_B6": controls["b6_off"],
            "F_B6": controls["b6_full_native"],
            "primary_O_minus_R": controls["b6_off"]["validation_loss"] - controls["all_real"]["validation_loss"],
            "descriptive_F_minus_R": controls["b6_full_native"]["validation_loss"] - controls["all_real"]["validation_loss"],
            "interpretation": "representation utility; F_B6 is descriptive and not a matched training architecture"}


def memory_accounting():
    def one(batch):
        bf16 = 2
        local_caps = {1: 1, 3: 31, 5: 63, 6: 511}
        rows = {}
        for block in range(1, 13):
            if block in local_caps:
                local = batch * local_caps[block] * N_EMBD * 2 * bf16
                ring = batch * RECURRENT_RING_CAPACITY * N_EMBD * bf16
                rows[f"B{block}"] = {"local_historical_kv_bytes": local,
                                     "raw_source_ring_bytes": ring, "total_bytes": local + ring}
            else:
                full = batch * 1023 * N_EMBD * 2 * bf16
                rows[f"B{block}"] = {"full_historical_kv_bytes": full, "total_bytes": full}
        total = sum(row["total_bytes"] for row in rows.values())
        standard = batch * 12 * 1023 * N_EMBD * 2 * bf16
        return {"layers": rows, "total_inference_state_bytes": total,
                "standard_gpt2_bytes": standard, "saving_bytes_vs_standard": standard - total,
                "saving_mib": (standard - total) / 1024**2,
                "saving_percent": 100 * (standard - total) / standard,
                "b6_full_native_kv_bytes": batch * 1023 * N_EMBD * 2 * bf16,
                "b6_recurrent_design_bytes": rows["B6"]["total_bytes"],
                "b6_saving_bytes": batch * 1023 * N_EMBD * 2 * bf16 - rows["B6"]["total_bytes"]}
    return {"dtype": "BF16", "formula_uses_historical_entries": True, "B1": one(1), "B64": one(64)}


def position_bin_metrics(incremental):
    real = np.array(incremental["controls"]["all_real"]["per_position_loss"])
    result = {}
    for link in ("b1", "b3", "b5", "b6"):
        off = np.array(incremental["controls"][f"{link}_off"]["per_position_loss"])
        shuffled = np.array(incremental["controls"][f"{link}_shuffled"]["per_position_loss"])
        result[link] = {}
        for name, low, high in POSITION_BINS:
            sl = slice(low, high + 1)
            result[link][name] = {"off_minus_real": float((off[sl] - real[sl]).mean()),
                                  "shuffled_minus_real": float((shuffled[sl] - real[sl]).mean())}
    return result


def classify_link(incremental, link):
    gain = incremental[f"true_{link}_gain"]; gap = incremental[f"true_{link}_sequence_gap"]
    off = incremental["paired"][link]["real_vs_off"]["wins"]
    shuffled = incremental["paired"][link]["real_vs_shuffled"]["wins"]
    if gain >= .001 and gap > 0 and off >= 166 and shuffled >= 166: return "STRONG POSITIVE"
    if gain > 0 and gap > 0 and off >= 129 and shuffled >= 129: return "POSITIVE UTILITY"
    if gap > 0: return "SEQUENCE-SPECIFIC BUT NOT ESTABLISHED"
    if abs(gain) < 1e-4 and abs(gap) < 1e-4: return "NEAR ZERO"
    if gain < 0: return "HARMFUL"
    return "NEAR ZERO"


def classify_overall(incremental, b6_control, integrity):
    if not integrity: return "EXPERIMENT 2D3A INVALID"
    positives = sum(classify_link(incremental, link) in ("STRONG POSITIVE", "POSITIVE UTILITY")
                    for link in ("b3", "b5", "b6"))
    gain = incremental["combined_new_link_gain"]; gap = incremental["combined_new_sequence_gap"]
    if gain > 0 and gap > 0 and positives >= 2: return "MULTI-LINK POSITIVE RECURRENT PYRAMID"
    if gain > 0: return "PARTIAL RECURRENT PYRAMID"
    if b6_control["primary_O_minus_R"] > 0: return "REPRESENTATION UTILITY WITHOUT BROAD RECURRENT GAIN"
    if abs(gain) < 1e-4 and all(abs(incremental[f"true_{link}_gain"]) < 1e-4 for link in ("b3", "b5", "b6")):
        return "RECURRENT PYRAMID NEAR ZERO"
    return "RECURRENT PYRAMID HARMFUL"


def probe_microbatch(source_path, device, candidates=(32, 16, 8, 4, 2, 1)):
    rows = []
    for candidate in candidates:
        try:
            source_model, source_optimizer, _, payload = load_source(source_path, device, restore=False)
            model, optimizer, _ = transplant(source_model, source_optimizer, device)
            loader = loader_at_cursor(payload["loader_state"], candidate)
            x, y = loader.next_batch(); x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model.forward_multi_pass(x, targets=y, num_passes=2, activation_checkpointing=True)
            result["loss"].backward()
            row = {"micro_batch": candidate, "fit": True,
                   "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                   "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2}
            rows.append(row)
            del source_model, source_optimizer, model, optimizer, loader, x, y, result; gc.collect(); torch.cuda.empty_cache()
            return candidate, {"candidates": rows, "selected": candidate, "passed": True}
        except torch.cuda.OutOfMemoryError:
            rows.append({"micro_batch": candidate, "fit": False, "reason": "CUDA OOM"})
            gc.collect(); torch.cuda.empty_cache()
    raise SystemExit(f"2D3A cannot fit even microbatch 1: {rows}")


def run_preflight(args):
    require_git_clean(); device = require_a100(); seed_all(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    source_model, source_optimizer, source_loader, source_payload = load_source(args.source_checkpoint, device)
    model, optimizer, transplant_audit = transplant(source_model, source_optimizer, device)
    parameter_manifest, parameter_diff = parameter_manifests(source_model, model)
    causality = causality_audit(model); isolation = future_and_row_isolation(model)
    identities = zero_gate_identities(model); incremental = incremental_smoke(model)
    selected, probe = probe_microbatch(args.source_checkpoint, device)
    val = validation_path(args.data_root)
    source_manifest = {
        "resolved_path": str(Path(args.source_checkpoint).resolve()), "sha256": file_sha256(args.source_checkpoint),
        "expected_sha256": SOURCE_SHA256, "schema": source_payload.get("schema"),
        "parameter_count": sum(p.numel() for p in source_model.parameters()),
        "named_gate_inventory": [name for name in dict(source_model.named_parameters()) if name.startswith("g_rec")],
        "inherited_b1_gate": source_model.g_rec.detach().float().item(),
        "inherited_b1_effective_gate": source_model.g_rec.detach().float().tanh().item(),
        "next_global_batch_sha256": next_batch_hash(source_loader, source_payload["gradient_accumulation"]),
        "next_global_batch_stream_sha256": next_stream_hash(source_loader, source_payload["gradient_accumulation"]),
        "strict_reopen_sidecars": parent.checkpoint_sidecar_audit(args.source_checkpoint),
        "git_implementation_lineage": source_payload.get("git_commit"),
    }
    stop_audit = {"pod_id": args.pod_id, "pod_name": args.pod_name,
                  "exact_stop_command": f"runpodctl pod stop {args.pod_id} -o json",
                  "authenticated_external_driver_required": True,
                  "driver_verified_before_launch": args.stop_capability_verified}
    checks = {
        "source_sha": source_manifest["sha256"] == SOURCE_SHA256,
        "source_schema": source_manifest["schema"] == SOURCE_SCHEMA,
        "source_parameters": source_manifest["parameter_count"] == SOURCE_PARAMETERS,
        "source_gate_inventory": source_manifest["named_gate_inventory"] == ["g_rec"],
        "source_cursor": source_manifest["next_global_batch_sha256"] == SOURCE_NEXT_BATCH and source_manifest["next_global_batch_stream_sha256"] == SOURCE_NEXT_STREAM,
        "transplant": all(transplant_audit.values()), "parameter_diff": parameter_manifest["passed"],
        "causality": causality["passed"], "future_and_row_isolation": isolation["passed"],
        "zero_gate_identities": identities["passed"], "incremental_cache": incremental["passed"],
        "canonical_validation_shard": file_sha256(val) == VALIDATION_SHARD_SHA,
        "microbatch_fit": probe["passed"], "stop_capability": args.stop_capability_verified,
    }
    audit = {"experiment": EXPERIMENT, "command": " ".join(sys.argv), "source": source_manifest,
             "architecture": architecture_manifest(), "transplant": transplant_audit,
             "parameter_manifest": parameter_manifest, "causality": causality,
             "future_and_row_isolation": isolation, "zero_gate_identities": identities,
             "incremental_smoke": incremental, "microbatch_probe": probe,
             "runpod_stop_capability": stop_audit, "checks": checks,
             "passed": all(checks.values()), "authorized": all(checks.values())}
    durable_json(output / "preflight_audit.json", audit); durable_json(output / "source_manifest.json", source_manifest)
    durable_json(output / "architecture_manifest.json", architecture_manifest())
    durable_json(output / "parameter_manifest.json", parameter_manifest); durable_json(output / "named_parameter_diff.json", parameter_diff)
    optimizer_manifest = {"source_groups": [{k: v for k, v in group.items() if k != "params"} for group in source_optimizer.param_groups],
                          "d3a_groups": [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
                          "transplant": transplant_audit, "only_new_gate_states_fresh": True,
                          "new_gate_lr": GATE_LR, "betas": list(BETAS), "eps": ADAM_EPS, "weight_decay": 0.0}
    durable_json(output / "optimizer_continuity_manifest.json", optimizer_manifest)
    durable_json(output / "scheduler_continuity_manifest.json", scheduler_manifest(source_payload, optimizer))
    durable_json(output / "runpod_stop_capability.json", stop_audit)
    durable_json(output / "commands_and_runtime.json", {"commands": [{"command": " ".join(sys.argv), "kind": "preflight"}]})
    if not audit["passed"]: raise SystemExit(f"2D3A preflight failed: {checks}")
    print(f"EXPERIMENT_2D3A_PREFLIGHT_PASS micro_batch={selected}", flush=True)


def run_smoke(args):
    require_git_clean(); device = require_a100(); output = Path(args.output_dir)
    preflight = read_json(output / "preflight_audit.json")
    if not preflight.get("authorized"): raise SystemExit("preflight did not authorize smoke")
    source_sha_before = file_sha256(args.source_checkpoint)
    source_model, source_optimizer, _, source_payload = load_source(args.source_checkpoint, device, restore=True)
    model, optimizer, _ = transplant(source_model, source_optimizer, device)
    micro = int(args.micro_batch or preflight["microbatch_probe"]["selected"]); accumulation = accumulation_for(micro)
    loader = loader_at_cursor(source_payload["loader_state"], micro); rows = []
    for update in range(1, SMOKE_UPDATES + 1):
        rows.append(train_update(model, optimizer, loader, accumulation, update, device))
    causality = causality_audit(model); isolation = future_and_row_isolation(model); incremental = incremental_smoke(model)
    temporal = temporal_gradients(model, validation_path(args.data_root))
    metadata = training_metadata(args, micro, accumulation)
    smoke_path = Path(args.checkpoint_dir) / "smoke_cumulative_000001572864.pt"
    verification = save_checkpoint(smoke_path, model, optimizer, loader, 3, accumulation, metadata, source_payload, device)
    reopened_model, reopened_optimizer, reopened_loader, reopened_payload = load_d3a_checkpoint(smoke_path, device)
    reload_checks = {"model_finite": model_finite(reopened_model), "optimizer_finite": optimizer_finite(reopened_optimizer),
                     "next_batch": next_batch_hash(reopened_loader, accumulation) == verification["next_global_batch_sha256"],
                     "updates": reopened_payload["d3a_completed_updates"] == 3,
                     "gates_exact": gate_values(reopened_model) == gate_values(model)}
    removed = []
    for path in (smoke_path, smoke_path.with_suffix(smoke_path.suffix + ".sha256"), smoke_path.with_suffix(smoke_path.suffix + ".verification.json")):
        if path.exists(): path.unlink(); removed.append(str(path))
    checks = {"exactly_three_updates": len(rows) == 3,
              "finite": all(math.isfinite(row["pass_losses"][-1]) for row in rows),
              "all_gates_move": all(rows[0]["gate_before"][link]["raw"] != rows[0]["gate_after"][link]["raw"] for link in ("b3", "b5", "b6")),
              "b1_functional": rows[0]["gradient_groups"]["b1_gate"]["nonzero"],
              "writer_gradients": all(temporal[link]["all_eligible_bins_nonzero"] for link in ("b3", "b5", "b6")),
              "causality": causality["passed"], "isolation": isolation["passed"],
              "cache": incremental["passed"], "checkpoint_reload": all(reload_checks.values()),
              "smoke_discarded": not smoke_path.exists(), "source_unchanged": file_sha256(args.source_checkpoint) == source_sha_before}
    audit = {"kind": "exactly three disposable optimizer updates", "rows": rows,
             "temporal_gradients": temporal, "causality": causality, "isolation": isolation,
             "incremental": incremental, "checkpoint_verification": verification,
             "reload_checks": reload_checks, "removed": removed, "checks": checks,
             "passed": all(checks.values()), "disposition": "discarded; science reloads immutable 2D2G-A"}
    durable_json(output / "smoke_audit.json", audit)
    if not audit["passed"]: raise SystemExit(f"2D3A smoke failed: {checks}")
    print("EXPERIMENT_2D3A_SMOKE_PASS", flush=True)


def milestone_names(update):
    names = ["all_real", "new_links_off", "b3_off", "b5_off", "b6_off"]
    if update in (0, 96, 191): names.append("b1_off")
    if update in (48, 96, 143, 191): names += ["b3_shuffled", "b5_shuffled", "b6_shuffled"]
    if update == 191: names += ["b1_shuffled", "all_new_shuffled", "all_recurrent_shuffled"]
    return names


def merge_keyed(path, key, value):
    payload = read_json(path) if Path(path).exists() else {}; payload[str(key)] = value; durable_json(path, payload)


def run_milestone(output, model, update, val_path):
    result = evaluate_parallel(model, val_path, milestone_names(update))
    if not result["canonical_collection_match"]:
        raise SystemExit("canonical validation collection SHA mismatch")
    if update in (20, 48, 96, 143, 191):
        gradients = temporal_gradients(model, val_path)
        for link, row in gradients.items():
            merge_keyed(output / {"b1": "b12_to_b1_temporal_gradients.json", "b3": "b10_to_b3_temporal_gradients.json",
                                  "b5": "b8_to_b5_temporal_gradients.json", "b6": "b7_to_b6_temporal_gradients.json"}[link], update, row)
    if update in (96, 191): merge_keyed(output / "stability_8pass.json", update, stability_8pass(model, val_path))
    merge_keyed(output / "milestone_validation.json", update, result)
    return result


def write_heartbeat(output, update, row, checkpoint=None):
    durable_json(Path(output) / "HEARTBEAT.json", {"experiment": EXPERIMENT, "status": "training" if update < 191 else "training_complete",
                                                   "local_update": update, "d3a_processed_targets": update * GLOBAL_TARGETS,
                                                   "latest_metrics": row, "checkpoint": checkpoint, "pid": os.getpid(),
                                                   "updated_at_unix": time.time()})


def run_train(args):
    require_git_clean(); device = require_a100(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    preflight, smoke = read_json(output / "preflight_audit.json"), read_json(output / "smoke_audit.json")
    if not preflight.get("authorized") or not smoke.get("passed"): raise SystemExit("preflight/smoke gates are not passed")
    if args.resume_checkpoint:
        model, optimizer, loader, payload = load_d3a_checkpoint(args.resume_checkpoint, device, restore=True)
        start = int(payload["d3a_completed_updates"]); source_payload = d0.torch_load(Path(args.source_checkpoint), mmap=False)
        metadata = payload["metadata"]; accumulation = int(payload["gradient_accumulation"]); micro = payload["loader_state"]["batch_size"]
        restart = {"fresh_process": payload.get("saved_process_id") != os.getpid(),
                   "saved_process_id": payload.get("saved_process_id"), "resumed_process_id": os.getpid(),
                   "next_batch": next_batch_hash(loader, accumulation), "expected_next_batch": payload["next_global_batch_sha256"],
                   "passed": payload.get("saved_process_id") != os.getpid() and next_batch_hash(loader, accumulation) == payload["next_global_batch_sha256"]}
        durable_json(output / "mandatory_fresh_process_restart_update_96.json", restart)
        if not restart["passed"]: raise SystemExit("mandatory fresh-process restart failed")
    else:
        source_model, source_optimizer, _, source_payload = load_source(args.source_checkpoint, device, restore=True)
        model, optimizer, _ = transplant(source_model, source_optimizer, device)
        micro = int(args.micro_batch or preflight["microbatch_probe"]["selected"]); accumulation = accumulation_for(micro)
        loader = loader_at_cursor(source_payload["loader_state"], micro); metadata = training_metadata(args, micro, accumulation); start = 0
        if next_batch_hash(loader, accumulation) != SOURCE_NEXT_BATCH: raise SystemExit("2D3A update 1 data cursor mismatch")
        if not (output / "initial_compression_diagnostics.json").exists():
            initial = initial_compression_diagnostics(source_model, model, validation_path(args.data_root))
            if not initial["canonical_collection_match"]: raise SystemExit("source canonical regression collection mismatch")
            if not initial["SOURCE_2D2GA_REPRODUCED_BY_2D3A_KERNEL"]["passed"]:
                raise SystemExit("update-0 2D2G-A source architecture regression failed")
            durable_json(output / "initial_compression_diagnostics.json", initial)
        if not (output / "milestone_validation.json").exists(): run_milestone(output, model, 0, validation_path(args.data_root))
    end = int(args.end_update)
    if not (0 <= start < end <= MAX_UPDATES): raise SystemExit(f"invalid segment {start}->{end}")
    if start == 0 and end > 96: raise SystemExit("first process must stop at update 96")
    if start == 96 and end != 191: raise SystemExit("fresh restart must continue exactly to 191")
    checkpoint_manifest = read_json(output / "checkpoint_manifest.json") if (output / "checkpoint_manifest.json").exists() else {}
    recovery_path = None
    for update in range(start + 1, end + 1):
        row = train_update(model, optimizer, loader, accumulation, update, device)
        append_jsonl(output / "training_metrics.jsonl", row); write_heartbeat(output, update, row)
        if update in MILESTONES[1:]: run_milestone(output, model, update, validation_path(args.data_root))
        if update in (50, 100, 150) and update != end:
            path = Path(args.checkpoint_dir) / f"recovery_update_{update:04d}.pt"
            verification = save_checkpoint(path, model, optimizer, loader, update, accumulation, metadata, source_payload, device)
            if recovery_path:
                for old in (recovery_path, recovery_path.with_suffix(recovery_path.suffix + ".sha256"), recovery_path.with_suffix(recovery_path.suffix + ".verification.json")):
                    if old.exists(): old.unlink()
            recovery_path = path; checkpoint_manifest[f"recovery_{update}"] = verification
        if update in (96, 191):
            targets = update * GLOBAL_TARGETS
            filename = f"scientific_cumulative_{targets:012d}.pt"
            path = Path(args.checkpoint_dir) / filename
            verification = save_checkpoint(path, model, optimizer, loader, update, accumulation, metadata, source_payload, device)
            persistent = persist_triplet(path, args.persistent_checkpoint_dir)
            verification["persistent"] = persistent; checkpoint_manifest[str(update)] = verification
            durable_json(output / "checkpoint_manifest.json", checkpoint_manifest)
            write_heartbeat(output, update, row, persistent["checkpoint"])
    print(f"EXPERIMENT_2D3A_SEGMENT_COMPLETE {start}->{end}", flush=True)


def build_plots(output, milestones, training, incremental, attention, temporal, memory, performance, b6):
    import matplotlib.pyplot as plt
    output = Path(output); updates = sorted(int(key) for key in milestones)
    targets = [update * GLOBAL_TARGETS for update in updates]
    def save(number, draw):
        fig, ax = plt.subplots(figsize=(8, 5)); draw(ax); fig.tight_layout();
        fig.savefig(output / f"plot_p{number:02d}.png", dpi=160); plt.close(fig)
    save(1, lambda ax: (ax.plot(targets, [milestones[str(u)]["controls"]["all_real"]["validation_loss"] for u in updates], label="ALL_REAL"),
                        ax.plot(targets, [milestones[str(u)]["controls"]["new_links_off"]["validation_loss"] for u in updates], label="NEW_LINKS_OFF"), ax.legend(), ax.set(xlabel="2D3A targets", ylabel="CE")))
    save(2, lambda ax: ([ax.plot(targets, [milestones[str(u)].get(f"{link}_gain", np.nan) for u in updates], label=link.upper()) for link in GATE_BLOCKS], ax.legend(), ax.set(xlabel="targets", ylabel="marginal gain")))
    save(3, lambda ax: ([ax.plot(targets, [milestones[str(u)].get(f"{link}_sequence_gap", np.nan) for u in updates], label=link.upper()) for link in GATE_BLOCKS], ax.legend(), ax.set(xlabel="targets", ylabel="sequence gap")))
    save(4, lambda ax: ([ax.plot([0] + [row["d3a_processed_targets"] for row in training],
                                     [milestones["0"]["gates"][link]["effective"]] + [row["gate_after"][link]["effective"] for row in training], label=link.upper()) for link in GATE_BLOCKS], ax.legend(), ax.set(xlabel="targets", ylabel="tanh(gate)")))
    for number, link in ((5, "b3"), (6, "b5"), (7, "b6")):
        save(number, lambda ax, link=link: (ax.bar(list(attention[link]["recurrent"]["bins"]), [r["raw_mass"] for r in attention[link]["recurrent"]["bins"].values()]), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="attention mass", title=f"{link.upper()} recurrent lag")))
    save(8, lambda ax: ([ax.plot([row["head"] for row in attention[link]["recurrent"]["per_head"]], [row["mean_lag"] for row in attention[link]["recurrent"]["per_head"]], marker="o", label=link.upper()) for link in ("b3", "b5", "b6")], ax.legend(), ax.set(xlabel="head", ylabel="mean recurrent lag")))
    save(9, lambda ax: ([ax.plot(list(temporal[link]["bins"]), [row["mean_gradient_rms"] for row in temporal[link]["bins"].values()], marker="o", label=link.upper()) for link in ("b3", "b5", "b6")], ax.legend(), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="writer gradient RMS")))
    positions = position_bin_metrics(incremental)
    save(10, lambda ax: ([ax.plot(list(positions[link]), [row["off_minus_real"] for row in positions[link].values()], marker="o", label=link.upper()) for link in ("b3", "b5", "b6")], ax.legend(), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="off-real")))
    save(11, lambda ax: (ax.bar(["B3", "B5", "B6"], [incremental[f"true_{link}_gain"] for link in ("b3", "b5", "b6")]), ax.set(ylabel="true incremental gain")))
    save(12, lambda ax: (ax.bar(list(memory["B1"]["layers"]), [row["total_bytes"] / 1024**2 for row in memory["B1"]["layers"].values()]), ax.tick_params(axis="x", rotation=35), ax.set(ylabel="BF16 MiB (B=1)")))
    save(13, lambda ax: (ax.bar(["R_B6", "O_B6", "F_B6"], [b6[key]["validation_loss"] for key in ("R_B6", "O_B6", "F_B6")]), ax.set(ylabel="validation CE")))
    save(14, lambda ax: (ax.plot([row["local_update"] for row in training], [row["targets_per_second"] for row in training]), ax.set(xlabel="update", ylabel="targets/sec", title=f"peak VRAM {performance['peak_reserved_vram_mb']:.0f} MiB")))


def render_report(summary, questions):
    inc = summary["incremental"]
    lines = ["EXPERIMENT 2D3A — 100M COMPLETE", "", "PRIMARY CLASSIFICATION:", summary["primary_classification"], "",
             "2D3A CUMULATIVE TARGETS:", "100,139,008", "", "B3 TRUE RECURRENT GAIN:", str(inc["true_b3_gain"]), "",
             "B5 TRUE RECURRENT GAIN:", str(inc["true_b5_gain"]), "", "B6 TRUE RECURRENT GAIN:", str(inc["true_b6_gain"]), "",
             "B3 TRUE SEQUENCE GAP:", str(inc["true_b3_sequence_gap"]), "", "B5 TRUE SEQUENCE GAP:", str(inc["true_b5_sequence_gap"]), "",
             "B6 TRUE SEQUENCE GAP:", str(inc["true_b6_sequence_gap"]), "", "## Source", f"- Path: `{summary['source']['resolved_path']}`",
             f"- SHA-256: `{summary['source']['sha256']}`", "", "## Architecture and training", "- B1 W2 + B12 recurrence; B2 W1024; B3 W32 + B10 recurrence; B4 W1024; B5 W64 + B8 recurrence; B6 W512 + B7 recurrence; B7-B12 W1024.",
             f"- Parameters: {MODEL_PARAMETERS:,}", f"- Runtime: {summary['performance']['training_wall_seconds']:.3f}s; mean {summary['performance']['mean_seconds_per_update']:.3f}s/update.",
             "", "## Link classifications", *[f"- {link.upper()}: {summary['link_classifications'][link]}" for link in ("b3", "b5", "b6")],
             "", "## Continuation", f"- Final checkpoint: `{summary['continuation_manifest']['checkpoint_path']}`", f"- SHA: `{summary['continuation_manifest']['checkpoint_sha256']}`",
             f"- Exact next-batch SHA: `{summary['continuation_manifest']['next_global_batch_sha256']}`", "- The checkpoint is resume-ready; no gate, optimizer, scheduler, loader, warmup, or RNG reset is permitted.",
             "", "## Q1–Q45"]
    for key in sorted(questions, key=lambda value: int(value[1:])):
        lines += [f"- **{key}. {questions[key]['question']}**", f"  {questions[key]['answer']}"]
    lines += ["", "FUTURE 2D3A MATURATION TARGETS:", "", "- 250M: cumulative updates 477; cumulative targets 250,085,376",
              "- 500M: cumulative updates 954; cumulative targets 500,170,752",
              "- 1B: cumulative updates 1908; cumulative targets 1,000,341,504", "", "NO FURTHER TRAINING WAS RUN AFTER 100M.", "", "# EXPERIMENT 2D3A 100M COMPLETE", ""]
    return "\n".join(lines)


def run_finalize(args):
    require_git_clean(); device = require_a100(); output = Path(args.output_dir)
    model, optimizer, loader, payload = load_d3a_checkpoint(args.final_checkpoint, device)
    if payload["d3a_completed_updates"] != 191 or payload["d3a_processed_targets"] != MAX_TARGETS: raise SystemExit("final checkpoint is not exact 100M endpoint")
    milestones = read_json(output / "milestone_validation.json")
    if set(milestones) != {str(value) for value in MILESTONES}: raise SystemExit("milestone set incomplete")
    incremental_path = output / "incremental_validation.json"
    if incremental_path.exists():
        incremental = read_json(incremental_path)
    else:
        incremental = evaluate_incremental(model, validation_path(args.data_root))
        durable_json(incremental_path, incremental)
        durable_json(output / "incremental_cache_audit.json", {
            name: control["cache_rows"] for name, control in incremental["controls"].items()
        })
    attention = {}
    for link in GATE_BLOCKS:
        path = output / f"{link}_attention_diagnostics.json"
        if path.exists():
            attention[link] = read_json(path)
        else:
            attention[link] = attention_diagnostic(model, validation_path(args.data_root), link)
            durable_json(path, attention[link])
    temporal = temporal_gradients(model, validation_path(args.data_root)); stable = stability_8pass(model, validation_path(args.data_root))
    b6 = b6_representation_control(model, validation_path(args.data_root)); memory = memory_accounting()
    position = position_bin_metrics(incremental); training = [json.loads(line) for line in (output / "training_metrics.jsonl").read_text().splitlines() if line]
    performance = {"training_wall_seconds": sum(row["wall_seconds"] for row in training),
                   "mean_seconds_per_update": statistics.fmean(row["wall_seconds"] for row in training),
                   "aggregate_targets_per_second": MAX_TARGETS / sum(row["wall_seconds"] for row in training),
                   "peak_allocated_vram_mb": max(row["peak_allocated_vram_mb"] for row in training),
                   "peak_reserved_vram_mb": max(row["peak_reserved_vram_mb"] for row in training),
                   "incremental": incremental["performance"]}
    checkpoint_manifest = read_json(output / "checkpoint_manifest.json"); final = checkpoint_manifest["191"]
    continuation = {"current_stage": "100M", "checkpoint_path": final["persistent"]["checkpoint"],
                    "checkpoint_sha256": final["sha256"], "next_global_batch_sha256": final["next_global_batch_sha256"],
                    "next_update": 192, "next_target_count_after_one_update": 100_663_296,
                    "current_cumulative_updates": 191, "current_cumulative_targets": 100_139_008,
                    "next_stage_updates": 477, "next_stage_targets": 250_085_376,
                    "updates_needed": 286, "additional_targets": 149_946_368,
                    "future": {"500M": {"updates": 954, "targets": 500_170_752},
                               "1B": {"updates": 1908, "targets": 1_000_341_504}},
                    "instructions": ["DO NOT REINITIALIZE ANY GATE", "DO NOT RESET OPTIMIZER", "DO NOT RESET SCHEDULER", "DO NOT RESTART DATA", "DO NOT RESTART WARMUP"]}
    source = read_json(output / "source_manifest.json")
    integrity = {
        "source": source["sha256"] == SOURCE_SHA256 and source["parameter_count"] == SOURCE_PARAMETERS,
        "architecture": payload["architecture_version"] == ARCHITECTURE_VERSION,
        "parameters": sum(p.numel() for p in model.parameters()) == MODEL_PARAMETERS,
        "training": len(training) == 191 and payload["d3a_processed_targets"] == MAX_TARGETS,
        "gradients": all(temporal[link]["all_eligible_bins_nonzero"] for link in GATE_BLOCKS),
        "causality": read_json(output / "preflight_audit.json")["causality"]["passed"],
        "cache": all(row["passed"] for row in incremental["controls"]["all_real"]["cache_rows"]),
        "stability": stable["passed"], "restart": read_json(output / "mandatory_fresh_process_restart_update_96.json")["passed"],
        "continuation": final["strict_reopen"]["passed"] and final["next_global_batch_sha256"] == FINAL_NEXT_BATCH,
        "persistence": final["persistent"]["passed"], "no_training_after_191": True,
    }
    primary = classify_overall(incremental, b6, all(integrity.values()))
    link_classes = {link: classify_link(incremental, link) for link in ("b3", "b5", "b6")}
    initial = read_json(output / "initial_compression_diagnostics.json")
    gates = gate_values(model)
    milestone_gain = lambda link, update: milestones[str(update)].get(f"{link}_gain")
    questions = {
        "Q1": {"question": "What exact Stage-A source path was resolved?", "answer": source["resolved_path"]},
        "Q2": {"question": "Did its SHA equal the required SHA?", "answer": source["sha256"] == SOURCE_SHA256},
        "Q3": {"question": "What was inherited B1 gate at update 0?", "answer": source["inherited_b1_gate"]},
        "Q4": {"question": "What was source 2D2G-A canonical CE?", "answer": initial["SOURCE_2D2GA"]["validation_loss"]},
        "Q5": {"question": "What was initial joint compression damage?", "answer": initial["initial_joint_compression_damage"]},
        "Q6": {"question": "What was B3-W32-only damage?", "answer": initial["single_change_diagnostics"]["B3_W32_ONLY"]["damage_vs_source"]},
        "Q7": {"question": "What was B5-W64-only damage?", "answer": initial["single_change_diagnostics"]["B5_W64_ONLY"]["damage_vs_source"]},
        "Q8": {"question": "What was B6-W512-only damage?", "answer": initial["single_change_diagnostics"]["B6_W512_ONLY"]["damage_vs_source"]},
        "Q9": {"question": "Did B3 gate open?", "answer": gates["b3"]["raw"] != 0}, "Q10": {"question": "At what update/sign?", "answer": "update 1 / " + ("positive" if training[0]["gate_after"]["b3"]["raw"] > 0 else "negative")},
        "Q11": {"question": "Did B5 gate open?", "answer": gates["b5"]["raw"] != 0}, "Q12": {"question": "At what update/sign?", "answer": "update 1 / " + ("positive" if training[0]["gate_after"]["b5"]["raw"] > 0 else "negative")},
        "Q13": {"question": "Did B6 gate open?", "answer": gates["b6"]["raw"] != 0}, "Q14": {"question": "At what update/sign?", "answer": "update 1 / " + ("positive" if training[0]["gate_after"]["b6"]["raw"] > 0 else "negative")},
        "Q15": {"question": "Final B1 gate?", "answer": gates["b1"]}, "Q16": {"question": "Final B3 gate?", "answer": gates["b3"]},
        "Q17": {"question": "Final B5 gate?", "answer": gates["b5"]}, "Q18": {"question": "Final B6 gate?", "answer": gates["b6"]},
    }
    q = 19
    for link in ("b3", "b5", "b6"):
        for update in (48, 96, 143, 191):
            questions[f"Q{q}"] = {"question": f"{link.upper()} gain at update {update}?", "answer": milestone_gain(link, update)}; q += 1
    questions.update({
        "Q31": {"question": "Final true B1 gain/gap?", "answer": [incremental["true_b1_gain"], incremental["true_b1_sequence_gap"]]},
        "Q32": {"question": "Final true B3 gain/gap?", "answer": [incremental["true_b3_gain"], incremental["true_b3_sequence_gap"]]},
        "Q33": {"question": "Final true B5 gain/gap?", "answer": [incremental["true_b5_gain"], incremental["true_b5_sequence_gap"]]},
        "Q34": {"question": "Final true B6 gain/gap?", "answer": [incremental["true_b6_gain"], incremental["true_b6_sequence_gap"]]},
        "Q35": {"question": "True paired wins for each link vs Off?", "answer": {link: incremental["paired"][link]["real_vs_off"]["wins"] for link in GATE_BLOCKS}},
        "Q36": {"question": "True paired wins for each link vs Shuffled?", "answer": {link: incremental["paired"][link]["real_vs_shuffled"]["wins"] for link in GATE_BLOCKS}},
        "Q37": {"question": "Combined new-link true gain?", "answer": incremental["combined_new_link_gain"]},
        "Q38": {"question": "Combined new-link sequence gap?", "answer": incremental["combined_new_sequence_gap"]},
        "Q39": {"question": "Did long-lag writer gradients reach all eligible bins?", "answer": integrity["gradients"]},
        "Q40": {"question": "Did B6 provide positive representation utility?", "answer": b6["primary_O_minus_R"] > 0},
        "Q41": {"question": "What is exact theoretical BF16 inference state?", "answer": memory["B1"]["total_inference_state_bytes"]},
        "Q42": {"question": "What is exact saving vs Standard?", "answer": memory["B1"]["saving_bytes_vs_standard"]},
        "Q43": {"question": "Did 8-pass self-composition remain stable?", "answer": stable["passed"]},
        "Q44": {"question": "What checkpoint should be used for 250M continuation?", "answer": continuation["checkpoint_path"]},
        "Q45": {"question": "Is it proven resume-ready?", "answer": integrity["continuation"]},
    })
    summary = {"experiment": EXPERIMENT, "primary_classification": primary, "link_classifications": link_classes,
               "source": source, "architecture": architecture_manifest(), "parameter_count": MODEL_PARAMETERS,
               "initial_compression": initial, "training": {"updates": 191, "targets": MAX_TARGETS},
               "milestones": milestones, "gates": gates, "attention": attention, "temporal_gradients": temporal,
               "incremental": incremental, "b6_representation_control": b6, "stability": stable,
               "memory": memory, "performance": performance, "checkpoint": final,
               "continuation_manifest": continuation, "position_bins": position, "questions": questions,
               "git": {"branch": BRANCH, "implementation_commit": payload["git_implementation_commit"], "current": git_output("rev-parse", "HEAD")},
               "no_further_training_after_100m": True}
    audit = {"experiment": EXPERIMENT, "checks": integrity, "passed": all(integrity.values()),
             "classification": primary, "final_checkpoint_sha256": final["sha256"],
             "no_further_training_after_100m": True}
    durable_json(output / "incremental_validation.json", incremental)
    durable_json(output / "incremental_cache_audit.json", {name: control["cache_rows"] for name, control in incremental["controls"].items()})
    durable_json(output / "paired_controls.json", incremental["paired"]); durable_json(output / "position_bin_metrics.json", position)
    durable_json(output / "b6_representation_control.json", b6); durable_json(output / "memory_accounting.json", memory)
    durable_json(output / "performance.json", performance); durable_json(output / "gate_diagnostics.json", {"final": gates, "trajectory": [row["gate_after"] for row in training]})
    for link in GATE_BLOCKS: durable_json(output / f"{link}_attention_diagnostics.json", attention[link])
    durable_json(output / "stability_8pass.json", {**(read_json(output / "stability_8pass.json") if (output / "stability_8pass.json").exists() else {}), "191_final": stable})
    durable_json(output / "maturation_core_subset_manifest.json", {"subset_sha256": incremental["subset_sha256"], "targets_per_control": 262144,
                                                                    "batch_identities": incremental["batch_identities"], "immutable_for_future_stages": True})
    durable_json(output / "CONTINUATION_MANIFEST.json", continuation); durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit); durable_json(output / "questions_q1_q45.json", questions)
    durable_json(output / "storage_cleanup_manifest.json", {"persistent_volume_retained": True, "source_checkpoint_retained": True,
                                                              "final_checkpoint_retained": True, "deleted": [], "passed": True})
    durable_text(output / "EXPERIMENT_2D3A_100M_FINAL_REPORT.md", render_report(summary, questions))
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", render_report(summary, questions) + "\nPod may be stopped only after local backup, Git push, and SHA verification.\n")
    build_plots(output, milestones, training, incremental, attention, temporal, memory, performance, b6)
    inventory = {name: {"present": (output / name).is_file(), "sha256": file_sha256(output / name) if (output / name).is_file() else None}
                 for name in REQUIRED_ARTIFACTS}
    durable_json(output / "artifact_inventory.json", inventory)
    if not audit["passed"] or not all(row["present"] for row in inventory.values()): raise SystemExit("final integrity or artifact inventory failed")
    print("EXPERIMENT_2D3A_FINALIZE_PASS", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    def common(p):
        p.add_argument("--source-checkpoint", required=True); p.add_argument("--data-root", required=True)
        p.add_argument("--output-dir", required=True); p.add_argument("--pod-id", required=True); p.add_argument("--pod-name", required=True)
    p = sub.add_parser("preflight"); common(p); p.add_argument("--stop-capability-verified", action="store_true"); p.set_defaults(func=run_preflight)
    p = sub.add_parser("smoke"); common(p); p.add_argument("--checkpoint-dir", required=True); p.add_argument("--micro-batch", type=int); p.set_defaults(func=run_smoke)
    p = sub.add_parser("train"); common(p); p.add_argument("--checkpoint-dir", required=True); p.add_argument("--persistent-checkpoint-dir", required=True)
    p.add_argument("--resume-checkpoint"); p.add_argument("--end-update", type=int, required=True); p.add_argument("--micro-batch", type=int); p.set_defaults(func=run_train)
    p = sub.add_parser("finalize"); common(p); p.add_argument("--final-checkpoint", required=True); p.set_defaults(func=run_finalize)
    return parser


def main():
    args = build_parser().parse_args(); return args.func(args)


if __name__ == "__main__":
    main()
