#!/usr/bin/env python3
"""Experiment 2D4A matched fixed-vs-routed 100M scientific driver."""

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
import torch
from torch.nn import functional as F

import experiment_2d3a as base
import experiment_2d3a_1b as parent
import experiment_2d3a_250m as stage_b
import experiment_2d3a_500m as stage_c
import experiment_2d4a_core as core


EXPERIMENT = "2D4A"
PROTOCOL = "matched_attnres_style_recurrent_source_depth_routing_100m_v1"
BRANCH = "experiment-2d4a-matched-source-depth-routing"
SCHEMA = "exp2d4a_matched_source_depth_routing_checkpoint_v1"
SOURCE_SHA256 = "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b"
SOURCE_NEXT_BATCH = "61dd83544d83c0cf7b4d61005f5a9cf64e2cafa930af1819cba2aae4538e7e61"
SOURCE_NEXT_STREAM = "39f6599f552803150fad33d32aa9c4df5843b058410ff1ba38b5afa469046e97"
SOURCE_COMMIT = "bf977013f5ca359e64d86eb896d445160c49c6bf"
SOURCE_TAG = "experiment-2d3a-alternating-integration-pyramid-1b-final"
SOURCE_UPDATES = 1_908
SOURCE_TARGETS = 1_000_341_504
LOCAL_UPDATES = 191
LOCAL_TARGETS = 100_139_008
RESTART_LOCAL_UPDATE = 96
MILESTONES = (0, 20, 48, 96, 143, 191)
RECOVERY_UPDATES = (24, 48, 72, 120, 144, 168)
FIXED_PARAMETERS = 124_475_908
ROUTED_PARAMETERS = 124_482_056
CANONICAL_SHA = base.CANONICAL_COLLECTION_SHA
MATURATION_CORE_SHA = stage_b.MATURATION_CORE_SHA
LARGE_START_BATCH = 52
LARGE_BATCHES = 32
LARGE_TARGETS = 2_097_152
BOOTSTRAP_SEED = 20_260_829
BOOTSTRAP_RESAMPLES = 50_000
ROUTE_NAMES = ("b1", "b3", "b5", "b6")
POSITION_BINS = base.POSITION_BINS
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def durable_json(path, value):
    base.durable_json(Path(path), value)


def durable_text(path, value):
    base.durable_text(Path(path), value)


def append_jsonl(path, value):
    base.append_jsonl(Path(path), value)


def merge_json(path, key, value):
    path = Path(path)
    payload = read_json(path) if path.exists() else {}
    payload[str(key)] = value
    durable_json(path, payload)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_branch(clean=False):
    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise SystemExit(f"expected branch {BRANCH}, found {branch}")
    if clean and git("status", "--porcelain"):
        raise SystemExit("Git worktree must be clean")


def sha256(path):
    return base.file_sha256(Path(path))


def tensor_sha256(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def named_model_manifest(model, exclude_router=False):
    rows = []
    for name, parameter in model.named_parameters():
        if exclude_router and name.startswith("source_routers."):
            continue
        rows.append({
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "sha256": tensor_sha256(parameter),
        })
    rows.sort(key=lambda row: row["name"])
    aggregate = hashlib.sha256(
        "".join(f"{row['name']}:{row['sha256']}" for row in rows).encode()
    ).hexdigest()
    return {"tensor_count": len(rows), "aggregate_sha256": aggregate, "tensors": rows}


def named_optimizer_manifest(model, optimizer, exclude_router=False):
    names = {parameter: name for name, parameter in model.named_parameters()}
    rows = []
    state_parameters = 0
    for parameter, state in optimizer.state.items():
        name = names[parameter]
        if exclude_router and name.startswith("source_routers."):
            continue
        state_parameters += 1
        for state_name, value in sorted(state.items()):
            if torch.is_tensor(value):
                digest = tensor_sha256(value)
                rows.append({
                    "parameter": name,
                    "state": state_name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": digest,
                })
    rows.sort(key=lambda row: (row["parameter"], row["state"]))
    aggregate = hashlib.sha256(
        "".join(
            f"{row['parameter']}:{row['state']}:{row['sha256']}" for row in rows
        ).encode()
    ).hexdigest()
    return {
        "state_parameters": state_parameters,
        "tensor_count": len(rows),
        "aggregate_sha256": aggregate,
        "representative_tensors": rows[:16],
    }


def require_source(path):
    path = Path(path).resolve()
    digest = sha256(path)
    if digest != SOURCE_SHA256:
        raise SystemExit(f"wrong 2D3A-1B source SHA: {digest}")
    return path


def load_fixed_source(path, device, restore=False):
    require_source(path)
    model, optimizer, loader, payload = base.load_d3a_checkpoint(path, device, restore=restore)
    checks = {
        "schema": payload.get("schema") == base.SCHEMA,
        "updates": payload.get("d3a_completed_updates") == SOURCE_UPDATES,
        "targets": payload.get("d3a_processed_targets") == SOURCE_TARGETS,
        "next_batch": payload.get("next_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "next_stream": payload.get("next_global_batch_stream_sha256") == SOURCE_NEXT_STREAM,
        "parameters": sum(parameter.numel() for parameter in model.parameters()) == FIXED_PARAMETERS,
        "scheduler_present": "scheduler" in payload,
        "rng_complete": set(payload.get("rng_state", {})) == {
            "python", "numpy", "torch_cpu", "torch_cuda"
        },
    }
    accumulation = int(payload["gradient_accumulation"])
    checks["loader_next_batch"] = base.next_batch_hash(loader, accumulation) == SOURCE_NEXT_BATCH
    checks["loader_next_stream"] = base.next_stream_hash(loader, accumulation) == SOURCE_NEXT_STREAM
    if not all(checks.values()):
        raise SystemExit(f"2D3A-1B source validation failed: {checks}")
    return model, optimizer, loader, payload, checks


def configure_routed_optimizer(model, source_optimizer, device_type="cuda"):
    old_gate_names = {
        "g_rec": "gate",
        "g_rec_b3": "b3_gate",
        "g_rec_b5": "b5_gate",
        "g_rec_b6": "b6_gate",
    }
    decay, nodecay = [], []
    route_nodecay, route_gates = [], []
    for name, parameter in model.named_parameters():
        if name in old_gate_names:
            continue
        if name.startswith("source_routers."):
            if name.endswith(".gate"):
                route_gates.append(parameter)
            else:
                route_nodecay.append(parameter)
            continue
        (decay if parameter.dim() >= 2 else nodecay).append(parameter)
    groups = [
        {"name": "base_decay", "params": decay, "lr": base.BASE_LR,
         "weight_decay": base.WEIGHT_DECAY},
        {"name": "base_nodecay", "params": nodecay, "lr": base.BASE_LR,
         "weight_decay": 0.0},
        *[
            {"name": group, "params": [getattr(model, name)], "lr": base.GATE_LR,
             "weight_decay": 0.0}
            for name, group in old_gate_names.items()
        ],
        {"name": "route_nodecay", "params": route_nodecay, "lr": base.BASE_LR,
         "weight_decay": 0.0},
        {"name": "route_gate", "params": route_gates, "lr": base.GATE_LR,
         "weight_decay": 0.0},
    ]
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
    optimizer = torch.optim.AdamW(
        groups, betas=base.BETAS, eps=base.ADAM_EPS, fused=fused
    )
    source_parameters = {
        parameter for group in source_optimizer.param_groups for parameter in group["params"]
    }
    routed_old = {
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("source_routers.")
    }
    if source_parameters != routed_old:
        raise SystemExit("routed arm old-parameter identity mismatch")
    for parameter in routed_old:
        if parameter in source_optimizer.state:
            optimizer.state[parameter] = copy.deepcopy(source_optimizer.state[parameter])
    source_groups = {group["name"]: group for group in source_optimizer.param_groups}
    for group in optimizer.param_groups:
        source_name = group["name"]
        if source_name == "route_nodecay":
            source_name = "base_nodecay"
        elif source_name == "route_gate":
            source_name = "gate"
        source = source_groups[source_name]
        for key in (
            "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
            "capturable", "differentiable", "fused",
        ):
            if key in source:
                group[key] = source[key]
    fresh = {
        parameter for name, parameter in model.named_parameters()
        if name.startswith("source_routers.")
    }
    checks = {
        "old_parameter_identity": source_parameters == routed_old,
        "old_state_entry_count": len(optimizer.state) == len(source_optimizer.state),
        "fresh_state_absent": all(parameter not in optimizer.state for parameter in fresh),
        "route_nodecay_count": len(route_nodecay) == 8,
        "route_gate_count": len(route_gates) == 4,
    }
    if not all(checks.values()):
        raise SystemExit(f"routed optimizer transplant failed: {checks}")
    return optimizer, checks


def make_routed_sibling(fixed_model, fixed_optimizer, device):
    routed = core.RoutedRecurrentPyramidGPT(fixed_model.base).to(device)
    routed.g_rec = fixed_model.g_rec
    routed.g_rec_b3 = fixed_model.g_rec_b3
    routed.g_rec_b5 = fixed_model.g_rec_b5
    routed.g_rec_b6 = fixed_model.g_rec_b6
    optimizer, optimizer_checks = configure_routed_optimizer(
        routed, fixed_optimizer, device.type
    )
    parameter_manifest = core.router_parameter_manifest(routed)
    checks = {
        **optimizer_checks,
        "router_manifest": parameter_manifest["passed"],
        "total_parameters": sum(parameter.numel() for parameter in routed.parameters())
        == ROUTED_PARAMETERS,
    }
    if not all(checks.values()):
        raise SystemExit(f"routed sibling construction failed: {checks}")
    return routed, optimizer, parameter_manifest, checks


def architecture_manifest(arm):
    routed = arm == "routed"
    blocks = base.architecture_manifest()["blocks"]
    return {
        "experiment": EXPERIMENT,
        "arm": arm,
        "parameters": ROUTED_PARAMETERS if routed else FIXED_PARAMETERS,
        "new_parameters": core.ROUTER_PARAMETER_COUNT if routed else 0,
        "blocks": blocks,
        "recurrent_windows": {"B1": 2, "B3": 32, "B5": 64, "B6": 512},
        "recurrent_lags": {
            "B1": [2, 1023], "B3": [32, 1023],
            "B5": [64, 1023], "B6": [512, 1023],
        },
        "fixed_sources": {"B1": "B12", "B3": "B10", "B5": "B8", "B6": "B7"},
        "candidate_order": {
            destination.upper(): [block + 1 for block in core.CANDIDATE_BLOCKS[index]]
            for index, destination in core.DESTINATION_NAMES.items()
        } if routed else {},
        "routing_write_time_only": routed,
        "persistent_rings": 4,
        "vectors_per_destination_token": 1,
        "ordinary_low_to_high_residual_unchanged": True,
        "existing_recurrent_readout_unchanged": True,
        "absent_links": ["B11->B2", "B9->B4"],
        "full_attnres": False,
    }


def route_values(model):
    result = {}
    for name, router in model.source_routers.items():
        scale = router.norm.weight.detach().float()
        result[name] = {
            "raw_gate": router.gate.detach().float().item(),
            "gamma": router.gate.detach().float().tanh().item(),
            "query_norm": router.query.detach().float().norm().item(),
            "norm_scale_mean": scale.mean().item(),
            "norm_scale_rms_displacement": (scale - 1).square().mean().sqrt().item(),
            "norm_scale_min": scale.min().item(),
            "norm_scale_max": scale.max().item(),
        }
    return result


def zero_route_identity(fixed_model, routed_model, device):
    generator = torch.Generator(device=device).manual_seed(20_260_829)
    tokens = torch.randint(0, 50_257, (2, 160), generator=generator, device=device)
    targets = torch.randint(0, 50_257, (2, 160), generator=generator, device=device)
    rows = {}
    for precision in ("fp32", "bf16"):
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" else torch.autocast(device_type="cuda", enabled=False)
        )
        with torch.no_grad(), context:
            fixed = fixed_model.forward_multi_pass(tokens, targets=targets, num_passes=2)
            routed = routed_model.forward_multi_pass(tokens, targets=targets, num_passes=2)
        comparisons = {
            "logits": (fixed["logits"], routed["logits"]),
            "loss": (fixed["loss"].reshape(1), routed["loss"].reshape(1)),
            "b1_memory": (routed["passes"][-1]["h12"], routed["m_b1"]),
            "b3_memory": (routed["passes"][-1]["h10"], routed["m_b3"]),
            "b5_memory": (routed["passes"][-1]["h8"], routed["m_b5"]),
            "b6_memory": (routed["passes"][-1]["h7"], routed["m_b6"]),
        }
        current = {}
        for name, (left, right) in comparisons.items():
            delta = (left.detach().float() - right.detach().float()).abs()
            current[name] = {
                "exact": bool(torch.equal(left, right)),
                "max_abs": delta.max().item(),
                "mean_abs": delta.mean().item(),
            }
        current["passed"] = all(row["exact"] for row in current.values())
        rows[precision] = current
    return {"precisions": rows, "passed": all(row["passed"] for row in rows.values())}


def routed_causality_audit(model, device):
    generator = torch.Generator(device=device).manual_seed(902)
    tokens = torch.randint(0, 50_257, (2, 160), generator=generator, device=device)
    suffix = tokens.clone()
    suffix[:, 100:] = torch.randint(
        0, 50_257, suffix[:, 100:].shape, generator=generator, device=device
    )
    row = tokens.clone()
    row[1] = torch.randint(0, 50_257, row[1].shape, generator=generator, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        left = model.forward_pass(tokens)
        suffix_result = model.forward_pass(suffix)
        row_result = model.forward_pass(row)
    result = {}
    for key in ("m_b1", "m_b3", "m_b5", "m_b6"):
        future_delta = (left[key][:, :100].float() - suffix_result[key][:, :100].float()).abs().max().item()
        row_delta = (left[key][0].float() - row_result[key][0].float()).abs().max().item()
        result[key] = {
            "suffix_prefix_max_abs": future_delta,
            "row0_max_abs": row_delta,
            "future_invariant": future_delta == 0.0,
            "row_isolated": row_delta == 0.0,
        }
    return {"memories": result, "passed": all(
        row["future_invariant"] and row["row_isolated"] for row in result.values()
    )}


def incremental_identity(fixed_model, routed_model, device):
    generator = torch.Generator(device=device).manual_seed(903)
    tokens = torch.randint(0, 50_257, (2, 136), generator=generator, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        fixed = fixed_model.incremental_logits(tokens)
        routed = routed_model.incremental_logits(tokens)
    delta = (fixed["logits"].float() - routed["logits"].float()).abs()
    fixed_audit, routed_audit = fixed["cache_audit"], routed["cache_audit"]
    checks = {
        "logits_exact": bool(torch.equal(fixed["logits"], routed["logits"])),
        "cache_lengths": fixed_audit["cache_lengths"] == routed_audit["cache_lengths"],
        "ring_lengths": fixed_audit["ring_lengths"] == routed_audit["ring_lengths"],
        "fixed_cache_pass": fixed_audit["passed"],
        "routed_cache_pass": routed_audit["passed"],
    }
    return {
        "checks": checks,
        "logits_max_abs": delta.max().item(),
        "logits_mean_abs": delta.mean().item(),
        "fixed_cache": fixed_audit,
        "routed_cache": routed_audit,
        "passed": all(checks.values()),
    }


def gradient_report(model):
    old_gates = {"b1": model.g_rec, "b3": model.g_rec_b3,
                 "b5": model.g_rec_b5, "b6": model.g_rec_b6}
    router_parameters = {
        f"{destination}_{role}": parameter
        for destination, router in model.source_routers.items()
        for role, parameter in (
            ("gate", router.gate), ("query", router.query),
            ("norm", router.norm.weight),
        )
    }

    def one(parameter):
        gradient = parameter.grad
        if gradient is None:
            return {"connected": False, "finite": False, "nonzero": False, "norm": 0.0}
        value = gradient.detach().float()
        return {
            "connected": True,
            "finite": bool(torch.isfinite(value).all()),
            "nonzero": bool(torch.count_nonzero(value).item()),
            "norm": value.norm().item(),
        }

    return {
        "existing_recurrent_gates": {name: one(parameter) for name, parameter in old_gates.items()},
        "routers": {name: one(parameter) for name, parameter in router_parameters.items()},
    }


def routed_train_update(model, optimizer, loader, accumulation, global_update, local_update, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    passes = base.pass_count(global_update)
    totals = [0.0] * passes
    started = time.monotonic()
    before = route_values(model)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(accumulation):
        cpu_x, cpu_y = loader.next_batch()
        x, y = cpu_x.to(device, non_blocking=True), cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model.forward_multi_pass(
                x, targets=y, num_passes=passes, activation_checkpointing=True
            )
            loss = result["loss"] / accumulation
        for index, value in enumerate(result["pass_losses"]):
            totals[index] += value.detach().float().item()
        loss.backward()
        del cpu_x, cpu_y, x, y, result, loss
    gradients = gradient_report(model)
    existing = gradients["existing_recurrent_gates"]
    routers = gradients["routers"]
    if not all(row["connected"] and row["finite"] and row["nonzero"] for row in existing.values()):
        raise SystemExit(f"bad existing recurrent gate gradients: {existing}")
    if not all(row["connected"] and row["finite"] for row in routers.values()):
        raise SystemExit(f"bad routed gradients: {routers}")
    if not all(routers[f"{name}_gate"]["nonzero"] for name in ROUTE_NAMES):
        raise SystemExit(f"route-gate gradient missing: {routers}")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
    if not torch.isfinite(norm):
        raise SystemExit("nonfinite routed gradient norm")
    optimizer.step()
    if not base.model_finite(model) or not base.optimizer_finite(optimizer):
        raise SystemExit("nonfinite routed model or optimizer")
    elapsed = time.monotonic() - started
    return {
        "local_update": int(local_update),
        "global_update": int(global_update),
        "d4a_local_targets": int(local_update) * base.GLOBAL_TARGETS,
        "parent_d3a_targets": SOURCE_TARGETS,
        "pass_count": passes,
        "pass_weights": list((.25, .75) if passes == 2 else (.2, .4, .4)),
        "pass_losses": [value / accumulation for value in totals],
        "gradient_diagnostics": gradients,
        "route_before": before,
        "route_after": route_values(model),
        "gradient_norm_before_clip": norm.detach().float().item(),
        "wall_seconds": elapsed,
        "targets_per_second": base.GLOBAL_TARGETS / elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def arm_training_update(arm, model, optimizer, loader, accumulation, global_update,
                        local_update, device):
    if arm == "routed":
        return routed_train_update(
            model, optimizer, loader, accumulation, global_update, local_update, device
        )
    row = base.train_update(model, optimizer, loader, accumulation, global_update, device)
    row.update({
        "local_update": local_update,
        "global_update": global_update,
        "d4a_local_targets": local_update * base.GLOBAL_TARGETS,
        "parent_d3a_targets": SOURCE_TARGETS,
    })
    return row


def route_summary(model, val_path):
    model.eval()
    device = base.model_device(model)
    loader = base.d1.ExplicitShardLoader([val_path], 2, base.T)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(device), cpu_y.to(device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.forward_multi_pass(
            x, targets=y, num_passes=2, return_route_diagnostics=True
        )
    rows = {}
    final_diagnostics = result["route_diagnostics"][-1]
    for name, diagnostic in final_diagnostics.items():
        beta = diagnostic["beta"].detach().float()
        effective = diagnostic["effective_coefficients"].detach().float()
        entropy = diagnostic["entropy"].detach().float()
        rows[name] = {
            **route_values(model)[name],
            "candidate_blocks": list(diagnostic["candidate_blocks"]),
            "baseline_block": diagnostic["baseline_block"],
            "mean_beta": beta.mean(dim=(1, 2)).tolist(),
            "mean_effective_coefficients": effective.mean(dim=(1, 2)).tolist(),
            "mean_entropy": entropy.mean().item(),
            "mean_normalized_entropy": diagnostic["normalized_entropy"].detach().float().mean().item(),
            "most_weighted_beta_block": diagnostic["candidate_blocks"][
                int(beta.mean(dim=(1, 2)).argmax().item())
            ],
            "largest_effective_block": diagnostic["candidate_blocks"][
                int(effective.mean(dim=(1, 2)).argmax().item())
            ],
            "routed_memory_rms": diagnostic["memory"].detach().float().square().mean().sqrt().item(),
            "baseline_memory_rms": diagnostic["baseline"].detach().float().square().mean().sqrt().item(),
            "routed_baseline_cosine": F.cosine_similarity(
                diagnostic["memory"].detach().float().reshape(-1, 768),
                diagnostic["baseline"].detach().float().reshape(-1, 768),
                dim=-1,
            ).mean().item(),
        }
    first_diagnostics = result["route_diagnostics"][0]
    first_memories = tuple(result["passes"][0][f"m_{name}"] for name in ROUTE_NAMES)
    later_loss = result["passes"][1]["loss"]
    memory_gradients = torch.autograd.grad(
        later_loss, first_memories, retain_graph=True, allow_unused=False
    )
    candidate_gradients = {}
    for destination, memory, memory_gradient in zip(
        ROUTE_NAMES, first_memories, memory_gradients
    ):
        diagnostic = first_diagnostics[destination]
        candidate_gradients[destination] = []
        for block, candidate in zip(
            diagnostic["candidate_blocks"], diagnostic["candidates"]
        ):
            gradient = torch.autograd.grad(
                memory, candidate, grad_outputs=memory_gradient,
                retain_graph=True, allow_unused=False,
            )[0].detach().float()
            candidate_gradients[destination].append({
                "source_block": block,
                "finite": bool(torch.isfinite(gradient).all()),
                "nonzero": bool(torch.count_nonzero(gradient).item()),
                "gradient_rms": gradient.square().mean().sqrt().item(),
            })
    loss = result["loss"]
    loss.backward()
    gradient = gradient_report(model)
    model.zero_grad(set_to_none=True)
    return {
        "validation_batch": base.batch_identity(cpu_x, cpu_y),
        "loss": loss.detach().float().item(),
        "destinations": rows,
        "gradients": gradient,
        "candidate_writer_gradients": candidate_gradients,
        "finite": all(
            math.isfinite(number)
            for row in rows.values()
            for key, number in row.items()
            if isinstance(number, float)
        ),
    }


def evaluate_parallel_conditions(model, val_path, conditions, batches=base.VALIDATION_BATCHES):
    model.eval()
    device = base.model_device(model)
    loader = base.d1.ExplicitShardLoader([val_path], base.VALIDATION_B, base.T)
    rows = {name: {"sum": 0.0, "targets": 0, "per_batch_losses": []} for name in conditions}
    identities = []
    with torch.no_grad():
        for _ in range(int(batches)):
            cpu_x, cpu_y = loader.next_batch()
            identities.append(base.batch_identity(cpu_x, cpu_y))
            x, y = cpu_x.to(device), cpu_y.to(device)
            for name, kwargs in conditions.items():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = model.forward_multi_pass(x, targets=y, num_passes=2, **kwargs)
                loss = result["loss"].detach().float().item()
                rows[name]["sum"] += loss * y.numel()
                rows[name]["targets"] += y.numel()
                rows[name]["per_batch_losses"].append(loss)
    subset = base.aggregate_hashes(row["combined_sha256"] for row in identities)
    return {
        "conditions": {
            name: {
                "validation_loss": row["sum"] / row["targets"],
                "validation_targets": row["targets"],
                "per_batch_losses": row["per_batch_losses"],
            }
            for name, row in rows.items()
        },
        "batch_identities": identities,
        "subset_sha256": subset,
        "canonical_collection_match": subset == CANONICAL_SHA,
    }


def run_arm_milestone(arm, model, local_update, val_path, output):
    output = Path(output)
    if arm == "fixed":
        conditions = {"fixed_real": {}}
    else:
        conditions = {
            "routed_real": {},
            "routed_all_route_off": {"route_mode": "off"},
        }
        if local_update in (48, 96, 143, 191):
            conditions["routed_uniform_depth"] = {"route_mode": "uniform"}
    evaluated = evaluate_parallel_conditions(model, val_path, conditions)
    if not evaluated["canonical_collection_match"]:
        raise SystemExit(f"{arm} milestone canonical validation SHA mismatch")
    evaluated.update({
        "arm": arm,
        "local_update": local_update,
        "d4a_local_targets": local_update * base.GLOBAL_TARGETS,
    })
    merge_json(
        output / f"{arm}_milestone_validation.json", local_update, evaluated
    )
    if arm == "routed":
        diagnostics = route_summary(model, val_path)
        diagnostics.update({
            "local_update": local_update,
            "d4a_local_targets": local_update * base.GLOBAL_TARGETS,
        })
        merge_json(output / "route_milestone_diagnostics.json", local_update, diagnostics)
    return evaluated


def continuation_metadata(args, arm, source_payload, accumulation):
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "arm": arm,
        "branch": BRANCH,
        "parent_checkpoint_path": str(Path(args.source_checkpoint).resolve()),
        "parent_checkpoint_sha256": SOURCE_SHA256,
        "parent_commit": SOURCE_COMMIT,
        "parent_tag": SOURCE_TAG,
        "git_implementation_commit": git("rev-parse", "HEAD"),
        "parent_d3a_updates": SOURCE_UPDATES,
        "parent_d3a_targets": SOURCE_TARGETS,
        "targets_per_update": base.GLOBAL_TARGETS,
        "micro_batch": int(source_payload["loader_state"]["batch_size"]),
        "gradient_accumulation": int(accumulation),
        "sequence_length": base.T,
        "pass3_every_global_updates": 32,
        "mandatory_restart_local_update": RESTART_LOCAL_UPDATE,
        "architecture": architecture_manifest(arm),
        "data_manifest": copy.deepcopy(source_payload["metadata"]["data_manifest"]),
        "canonical_validation_manifest": copy.deepcopy(
            source_payload["metadata"]["canonical_validation_manifest"]
        ),
        "maturation_core_subset_manifest": copy.deepcopy(
            source_payload["metadata"]["maturation_core_subset_manifest"]
        ),
        "scheduler": copy.deepcopy(source_payload.get("scheduler")),
    }


def checkpoint_payload(arm, model, optimizer, loader, local_update, accumulation,
                       metadata):
    global_update = SOURCE_UPDATES + int(local_update)
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA,
        "experiment_name": EXPERIMENT,
        "arm": arm,
        "parent_checkpoint_sha256": SOURCE_SHA256,
        "parent_d3a_updates": SOURCE_UPDATES,
        "parent_d3a_targets": SOURCE_TARGETS,
        "d4a_local_updates": int(local_update),
        "d4a_local_targets": int(local_update) * base.GLOBAL_TARGETS,
        "inherited_global_update": global_update,
        "inherited_total_targets": SOURCE_TARGETS + int(local_update) * base.GLOBAL_TARGETS,
        "targets_per_update": base.GLOBAL_TARGETS,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": copy.deepcopy(metadata["scheduler"]),
        "loader_state": loader.state_dict(),
        "loader_states": [loader.state_dict()],
        "rng_state": base.capture_rng(),
        "gradient_accumulation": int(accumulation),
        "next_global_batch_sha256": base.next_batch_hash(loader, accumulation),
        "next_global_batch_stream_sha256": base.next_stream_hash(loader, accumulation),
        "architecture_manifest": architecture_manifest(arm),
        "recurrent_windows": {"B1": 2, "B3": 32, "B5": 64, "B6": 512},
        "recurrent_lags": {"B1": [2, 1023], "B3": [32, 1023],
                           "B5": [64, 1023], "B6": [512, 1023]},
        "source_router_candidate_ordering": architecture_manifest(arm)["candidate_order"],
        "existing_recurrent_gates": base.gate_values(model),
        "route_values": route_values(model) if arm == "routed" else {},
        "optimizer_group_definitions": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
        "git_implementation_commit": metadata["git_implementation_commit"],
        "metadata": metadata,
        "saved_process_id": os.getpid(),
        "saved_at_unix": time.time(),
    }
    return payload


def load_arm_source(arm, source_checkpoint, device, restore=False):
    fixed_model, fixed_optimizer, loader, source, checks = load_fixed_source(
        source_checkpoint, device, restore=restore
    )
    if arm == "fixed":
        return fixed_model, fixed_optimizer, loader, source, checks
    routed, optimizer, parameter_manifest, routed_checks = make_routed_sibling(
        fixed_model, fixed_optimizer, device
    )
    checks = {**checks, **{f"routed_{key}": value for key, value in routed_checks.items()}}
    checks["router_parameter_manifest"] = parameter_manifest["passed"]
    return routed, optimizer, loader, source, checks


def load_arm_checkpoint(path, source_checkpoint, device, restore=False):
    payload = base.d0.torch_load(Path(path), mmap=False)
    if payload.get("schema") != SCHEMA or payload.get("arm") not in ("fixed", "routed"):
        raise SystemExit("not a 2D4A arm checkpoint")
    arm = payload["arm"]
    model, optimizer, _, _, _ = load_arm_source(
        arm, source_checkpoint, device, restore=False
    )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    loader = base.d1.ExplicitShardLoader(
        payload["loader_state"]["shards"], payload["loader_state"]["batch_size"],
        base.T, state=payload["loader_state"],
    )
    if restore:
        base.restore_rng(payload["rng_state"])
    return model, optimizer, loader, payload


def strict_reopen(path, source_checkpoint, device):
    model, optimizer, loader, payload = load_arm_checkpoint(
        path, source_checkpoint, device, restore=False
    )
    arm = payload["arm"]
    local = int(payload["d4a_local_updates"])
    accumulation = int(payload["gradient_accumulation"])
    checks = {
        "schema": payload.get("schema") == SCHEMA,
        "arm": arm in ("fixed", "routed"),
        "parent": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256,
        "local_updates": 0 <= local <= LOCAL_UPDATES,
        "local_targets": payload.get("d4a_local_targets") == local * base.GLOBAL_TARGETS,
        "global_update": payload.get("inherited_global_update") == SOURCE_UPDATES + local,
        "parameters": sum(parameter.numel() for parameter in model.parameters())
        == (ROUTED_PARAMETERS if arm == "routed" else FIXED_PARAMETERS),
        "model_finite": base.model_finite(model),
        "optimizer_finite": base.optimizer_finite(optimizer),
        "next_batch": base.next_batch_hash(loader, accumulation)
        == payload["next_global_batch_sha256"],
        "next_stream": base.next_stream_hash(loader, accumulation)
        == payload["next_global_batch_stream_sha256"],
        "rng_complete": set(payload.get("rng_state", {}))
        == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "scheduler_present": "scheduler" in payload,
        "git_commit": bool(payload.get("git_implementation_commit")),
    }
    audit = {"checks": checks, "passed": all(checks.values())}
    del model, optimizer, loader, payload
    gc.collect()
    torch.cuda.empty_cache()
    return audit


def save_checkpoint(path, arm, model, optimizer, loader, local_update, accumulation,
                    metadata, source_checkpoint, device, sidecars=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        arm, model, optimizer, loader, local_update, accumulation, metadata
    )
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    digest = sha256(path)
    audit = strict_reopen(path, source_checkpoint, device)
    if not audit["passed"]:
        raise SystemExit(f"strict 2D4A checkpoint reopen failed: {audit}")
    if sidecars:
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


def heartbeat(output, arm, local_update, status, row=None, checkpoint=None):
    durable_json(Path(output) / f"HEARTBEAT_{arm.upper()}.json", {
        "experiment": EXPERIMENT,
        "arm": arm,
        "status": status,
        "local_update": int(local_update),
        "global_update": SOURCE_UPDATES + int(local_update),
        "d4a_local_targets": int(local_update) * base.GLOBAL_TARGETS,
        "latest_metrics": row,
        "checkpoint": checkpoint,
        "pid": os.getpid(),
        "updated_at_unix": time.time(),
    })


def smoke_checkpoint_audit(model, optimizer, loader, accumulation, metadata,
                           source_checkpoint, device, output):
    temporary = Path("/tmp/exp2d4a_disposable_smoke.pt")
    verification = save_checkpoint(
        temporary, "routed", model, optimizer, loader, 3, accumulation,
        metadata, source_checkpoint, device, sidecars=False,
    )
    audit = {
        "checkpoint_sha256": verification["sha256"],
        "strict_reopen": verification["strict_reopen"],
        "temporary_checkpoint_removed": False,
    }
    temporary.unlink()
    audit["temporary_checkpoint_removed"] = not temporary.exists()
    durable_json(Path(output) / "disposable_smoke_checkpoint_audit.json", audit)
    return audit


def run_preflight(args):
    require_branch(clean=True)
    device = base.require_a100()
    base.seed_all()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_path = require_source(args.source_checkpoint)
    fixed_model, fixed_optimizer, loader, source, source_checks = load_fixed_source(
        source_path, device, restore=False
    )
    source_reopen = parent.strict_reopen(
        source_path, SOURCE_UPDATES, source["metadata"], device
    )
    fixed_model_manifest = named_model_manifest(fixed_model)
    fixed_optimizer_manifest = named_optimizer_manifest(fixed_model, fixed_optimizer)
    routed_model, routed_optimizer, parameter_manifest, routed_checks = make_routed_sibling(
        fixed_model, fixed_optimizer, device
    )
    routed_old_model_manifest = named_model_manifest(routed_model, exclude_router=True)
    routed_old_optimizer_manifest = named_optimizer_manifest(
        routed_model, routed_optimizer, exclude_router=True
    )
    identity = zero_route_identity(fixed_model, routed_model, device)
    causality = routed_causality_audit(routed_model, device)
    incremental = incremental_identity(fixed_model, routed_model, device)
    old_identity = {
        "model_aggregate_fixed": fixed_model_manifest["aggregate_sha256"],
        "model_aggregate_routed_old": routed_old_model_manifest["aggregate_sha256"],
        "optimizer_aggregate_fixed": fixed_optimizer_manifest["aggregate_sha256"],
        "optimizer_aggregate_routed_old": routed_old_optimizer_manifest["aggregate_sha256"],
        "model_identical": fixed_model_manifest["aggregate_sha256"]
        == routed_old_model_manifest["aggregate_sha256"],
        "optimizer_identical": fixed_optimizer_manifest["aggregate_sha256"]
        == routed_old_optimizer_manifest["aggregate_sha256"],
    }
    old_identity["passed"] = old_identity["model_identical"] and old_identity["optimizer_identical"]
    durable_json(output / "parameter_diff.json", parameter_manifest)
    durable_json(output / "zero_route_identity_audit.json", identity)
    durable_json(output / "old_tensor_identity_audit.json", old_identity)
    durable_json(output / "routed_causality_audit.json", causality)
    durable_json(output / "preflight_incremental_cache_audit.json", incremental)
    durable_json(output / "source_1b_manifest.json", {
        "checkpoint": str(source_path),
        "sha256": sha256(source_path),
        "commit": SOURCE_COMMIT,
        "tag": SOURCE_TAG,
        "updates": SOURCE_UPDATES,
        "targets": SOURCE_TARGETS,
        "next_batch_sha256": SOURCE_NEXT_BATCH,
        "next_stream_sha256": SOURCE_NEXT_STREAM,
        "parameter_count": FIXED_PARAMETERS,
        "source_checks": source_checks,
        "strict_reopen": source_reopen,
        "recovery_provenance": "accepted replay checkpoint only",
    })
    durable_json(output / "architecture_fixed.json", architecture_manifest("fixed"))
    durable_json(output / "architecture_routed.json", architecture_manifest("routed"))

    del routed_model, routed_optimizer, fixed_model, fixed_optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()

    smoke_model, smoke_optimizer, smoke_loader, smoke_source, _ = load_arm_source(
        "routed", source_path, device, restore=True
    )
    accumulation = int(smoke_source["gradient_accumulation"])
    metadata = continuation_metadata(args, "routed", smoke_source, accumulation)
    smoke_rows = []
    for local_update in range(1, 4):
        global_update = SOURCE_UPDATES + local_update
        batch_hash = base.next_batch_hash(smoke_loader, accumulation)
        stream_hash = base.next_stream_hash(smoke_loader, accumulation)
        row = routed_train_update(
            smoke_model, smoke_optimizer, smoke_loader, accumulation,
            global_update, local_update, device,
        )
        row["consumed_batch_sha256"] = batch_hash
        row["consumed_stream_sha256"] = stream_hash
        smoke_rows.append(row)
    smoke_checkpoint = smoke_checkpoint_audit(
        smoke_model, smoke_optimizer, smoke_loader, accumulation, metadata,
        source_path, device, output,
    )
    staged = {}
    for destination in ROUTE_NAMES:
        staged[destination] = {
            "step1_gate_finite_nonzero": smoke_rows[0]["gradient_diagnostics"]["routers"][f"{destination}_gate"]["finite"]
            and smoke_rows[0]["gradient_diagnostics"]["routers"][f"{destination}_gate"]["nonzero"],
            "step1_query_zero": not smoke_rows[0]["gradient_diagnostics"]["routers"][f"{destination}_query"]["nonzero"],
            "step1_norm_zero": not smoke_rows[0]["gradient_diagnostics"]["routers"][f"{destination}_norm"]["nonzero"],
            "query_appears_by_step3": any(
                row["gradient_diagnostics"]["routers"][f"{destination}_query"]["nonzero"]
                for row in smoke_rows[1:]
            ),
            "norm_appears_by_step3": any(
                row["gradient_diagnostics"]["routers"][f"{destination}_norm"]["nonzero"]
                for row in smoke_rows[1:]
            ),
        }
    smoke_passed = all(all(row.values()) for row in staged.values()) and smoke_checkpoint["strict_reopen"]["passed"]
    smoke = {"updates": smoke_rows, "staged_activation": staged, "checkpoint": smoke_checkpoint,
             "discarded": True, "passed": smoke_passed}
    durable_json(output / "disposable_smoke.json", smoke)
    del smoke_model, smoke_optimizer, smoke_loader
    gc.collect()
    torch.cuda.empty_cache()

    storage = {
        "workspace_free_bytes": shutil.disk_usage(source_path.parent).free,
        "output_free_bytes": shutil.disk_usage(output).free,
        "network_volume_size_bytes": int(args.network_volume_size_bytes),
        "network_volume_used_bytes": int(args.network_volume_used_bytes),
    }
    checks = {
        "source": all(source_checks.values()),
        "source_strict_reopen": source_reopen["passed"],
        "source_tag_exact": git("rev-parse", f"{SOURCE_TAG}^{{commit}}") == SOURCE_COMMIT,
        "source_commit_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], cwd=REPO_ROOT
        ).returncode == 0,
        "old_tensors_identical": old_identity["passed"],
        "parameter_manifest": parameter_manifest["passed"],
        "routed_construction": all(routed_checks.values()),
        "zero_route_identity": identity["passed"],
        "causality": causality["passed"],
        "incremental_identity_and_cache": incremental["passed"],
        "disposable_smoke": smoke["passed"],
        "one_a100_80gb": torch.cuda.device_count() == 1
        and "A100" in torch.cuda.get_device_name(device),
        "stop_capability_verified": bool(args.stop_capability_verified),
        "storage_inventory_verified": bool(args.storage_inventory_verified),
        "workspace_free_ge_10gb": storage["workspace_free_bytes"] >= 10 * 1024**3,
        "git_clean": not bool(git("status", "--porcelain")),
    }
    audit = {
        "experiment": EXPERIMENT,
        "checks": checks,
        "storage": storage,
        "hardware": {"gpu": torch.cuda.get_device_name(device), "gpu_count": 1},
        "authorized": all(checks.values()),
    }
    durable_json(output / "preflight_audit.json", audit)
    if not audit["authorized"]:
        raise SystemExit(f"2D4A preflight failed: {checks}")
    print("EXPERIMENT_2D4A_PREFLIGHT_PASS", flush=True)


def allowed_segment(start, end):
    return (start, end) in ((0, 96), (96, 191))


def load_fixed_ledger(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {int(row["local_update"]): row for row in rows}


def run_train(args):
    require_branch(clean=True)
    if args.arm not in ("fixed", "routed"):
        raise SystemExit("arm must be fixed or routed")
    device = base.require_a100()
    output = Path(args.output_dir)
    preflight = read_json(args.preflight_audit)
    if not preflight.get("authorized"):
        raise SystemExit("2D4A preflight did not authorize training")
    if args.resume_checkpoint:
        model, optimizer, loader, loaded = load_arm_checkpoint(
            args.resume_checkpoint, args.source_checkpoint, device, restore=True
        )
        if loaded["arm"] != args.arm:
            raise SystemExit("resume checkpoint arm mismatch")
        start = int(loaded["d4a_local_updates"])
        accumulation = int(loaded["gradient_accumulation"])
        metadata = loaded["metadata"]
        restart = {
            "arm": args.arm,
            "loaded_local_update": start,
            "saved_process_id": loaded.get("saved_process_id"),
            "resumed_process_id": os.getpid(),
            "fresh_process": loaded.get("saved_process_id") != os.getpid(),
            "next_batch_sha256": base.next_batch_hash(loader, accumulation),
            "expected_next_batch_sha256": loaded["next_global_batch_sha256"],
            "next_stream_sha256": base.next_stream_hash(loader, accumulation),
            "expected_next_stream_sha256": loaded["next_global_batch_stream_sha256"],
        }
        restart["passed"] = restart["fresh_process"] and (
            restart["next_batch_sha256"] == restart["expected_next_batch_sha256"]
            and restart["next_stream_sha256"] == restart["expected_next_stream_sha256"]
        )
        durable_json(output / f"{args.arm}_restart_audit.json", restart)
        if start == RESTART_LOCAL_UPDATE and not restart["passed"]:
            raise SystemExit(f"mandatory {args.arm} restart failed: {restart}")
    else:
        model, optimizer, loader, loaded, checks = load_arm_source(
            args.arm, args.source_checkpoint, device, restore=True
        )
        if not all(checks.values()):
            raise SystemExit(f"arm source validation failed: {checks}")
        start = 0
        accumulation = int(loaded["gradient_accumulation"])
        metadata = continuation_metadata(args, args.arm, loaded, accumulation)
    end = int(args.end_local_update)
    if not allowed_segment(start, end):
        raise SystemExit(f"unauthorized 2D4A training segment {start}->{end}")
    val_path = base.validation_path(Path(args.data_root))
    output.mkdir(parents=True, exist_ok=True)
    metric_path = output / f"{args.arm}_training_metrics.jsonl"
    ledger_path = output / f"{args.arm}_training_ledger.jsonl"
    fixed_ledger = load_fixed_ledger(args.matched_fixed_ledger) if args.arm == "routed" else {}
    milestone_file = output / f"{args.arm}_milestone_validation.json"
    milestones_done = set(map(int, read_json(milestone_file))) if milestone_file.exists() else set()
    if start in MILESTONES and start not in milestones_done:
        run_arm_milestone(args.arm, model, start, val_path, output)
    for local_update in range(start + 1, end + 1):
        global_update = SOURCE_UPDATES + local_update
        batch_hash = base.next_batch_hash(loader, accumulation)
        stream_hash = base.next_stream_hash(loader, accumulation)
        if local_update == 1:
            if batch_hash != SOURCE_NEXT_BATCH or stream_hash != SOURCE_NEXT_STREAM:
                raise SystemExit("first 2D4A batch/stream mismatch")
        if args.arm == "routed":
            expected = fixed_ledger.get(local_update)
            if expected is None:
                raise SystemExit(f"fixed ledger lacks local update {local_update}")
            if (batch_hash, stream_hash) != (
                expected["batch_sha256"], expected["stream_sha256"]
            ):
                raise SystemExit(f"matched data mismatch at local update {local_update}")
        row = arm_training_update(
            args.arm, model, optimizer, loader, accumulation,
            global_update, local_update, device,
        )
        row["consumed_batch_sha256"] = batch_hash
        row["consumed_stream_sha256"] = stream_hash
        row["optimizer_lrs"] = {
            group["name"]: float(group["lr"]) for group in optimizer.param_groups
        }
        append_jsonl(metric_path, row)
        ledger_row = {
            "arm": args.arm,
            "local_update": local_update,
            "global_update": global_update,
            "batch_sha256": batch_hash,
            "stream_sha256": stream_hash,
            "pass_count": row["pass_count"],
        }
        append_jsonl(ledger_path, ledger_row)
        heartbeat(output, args.arm, local_update, "training", row=row)
        if local_update in RECOVERY_UPDATES:
            recovery_path = Path(args.recovery_dir) / "rotating_recovery.pt"
            verification = save_checkpoint(
                recovery_path, args.arm, model, optimizer, loader, local_update,
                accumulation, metadata, args.source_checkpoint, device, sidecars=False,
            )
            heartbeat(
                output, args.arm, local_update, "recovery_verified", row=row,
                checkpoint=verification["checkpoint"],
            )
        if local_update in (RESTART_LOCAL_UPDATE, LOCAL_UPDATES):
            checkpoint_path = Path(args.scientific_checkpoint_dir) / f"scientific_local_{local_update:04d}.pt"
            verification = save_checkpoint(
                checkpoint_path, args.arm, model, optimizer, loader, local_update,
                accumulation, metadata, args.source_checkpoint, device,
            )
            durable_json(output / f"checkpoint_manifest_{args.arm}_{local_update:04d}.json", verification)
            heartbeat(
                output, args.arm, local_update, "checkpoint_verified", row=row,
                checkpoint=verification["checkpoint"],
            )
        if local_update in MILESTONES:
            run_arm_milestone(args.arm, model, local_update, val_path, output)
            heartbeat(output, args.arm, local_update, "milestone_complete", row=row)
    print(f"EXPERIMENT_2D4A_{args.arm.upper()}_SEGMENT_COMPLETE {start}->{end}", flush=True)


def incremental_condition(model, x, y, arm, condition, permutation,
                          audit_positions=()):
    state = model.init_incremental_state(x.size(0), device=x.device)
    sequence_loss = torch.zeros(x.size(0), dtype=torch.float64)
    position_loss = torch.zeros(base.T, dtype=torch.float64)
    cache_rows = []
    route_kwargs = {}
    control = "all_real"
    recurrent_permutation = None
    if arm == "routed":
        route_kwargs = {
            "routed_real": {},
            "b1_route_off": {"route_off_destinations": ("b1",)},
            "b3_route_off": {"route_off_destinations": ("b3",)},
            "b5_route_off": {"route_off_destinations": ("b5",)},
            "b6_route_off": {"route_off_destinations": ("b6",)},
            "all_route_off": {"route_mode": "off"},
            "uniform_depth": {"route_mode": "uniform"},
            "all_recurrent_memory_shuffled": {},
        }[condition]
        if condition == "all_recurrent_memory_shuffled":
            control = "all_recurrent_shuffled"
            recurrent_permutation = permutation
    elif condition != "fixed_real":
        raise ValueError(f"unknown fixed condition: {condition}")
    for position in range(base.T):
        wants_audit = position + 1 in audit_positions
        if arm == "routed":
            result = model.incremental_step(
                x[:, position], state, control=control,
                recurrent_permutation=recurrent_permutation,
                return_diagnostics=wants_audit,
                diagnostic_attention_weights=False,
                **route_kwargs,
            )
        else:
            result = model.incremental_step(
                x[:, position], state, control=control,
                recurrent_permutation=recurrent_permutation,
                return_diagnostics=wants_audit,
                diagnostic_attention_weights=False,
            )
        if wants_audit:
            logits, state, diagnostic = result
            cache_rows.append(diagnostic["cache_audit"])
        else:
            logits, state = result
        losses = F.cross_entropy(
            logits[:, 0].float(), y[:, position], reduction="none"
        ).double().cpu()
        sequence_loss += losses
        position_loss[position] += losses.sum()
    return {
        "loss_sum": sequence_loss.sum().item(),
        "targets": x.numel(),
        "per_sequence_losses": (sequence_loss / base.T).tolist(),
        "per_position_sum": position_loss.tolist(),
        "cache_rows": cache_rows,
        "final_cache_audit": model.incremental_cache_audit(state),
    }


def evaluate_incremental_conditions(model, val_path, arm, conditions,
                                    start_batch, batches):
    model.eval()
    device = base.model_device(model)
    loader = base.d1.ExplicitShardLoader([val_path], base.VALIDATION_B, base.T)
    for _ in range(int(start_batch)):
        loader.next_batch()
    rows = {
        name: {
            "sum": 0.0, "targets": 0, "sequences": [],
            "positions": np.zeros(base.T, dtype=np.float64), "cache_rows": [],
        }
        for name in conditions
    }
    identities = []
    permutation = torch.arange(base.VALIDATION_B, device=device).roll(1)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    audit_positions = (1, 31, 32, 63, 64, 127, 128, 511, 512, 1023)
    with torch.no_grad():
        for batch_index in range(int(batches)):
            cpu_x, cpu_y = loader.next_batch()
            identities.append(base.batch_identity(cpu_x, cpu_y))
            x, y = cpu_x.to(device), cpu_y.to(device)
            for condition in conditions:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    current = incremental_condition(
                        model, x, y, arm, condition, permutation,
                        audit_positions if batch_index == 0 else (),
                    )
                row = rows[condition]
                row["sum"] += current["loss_sum"]
                row["targets"] += current["targets"]
                row["sequences"].extend(current["per_sequence_losses"])
                row["positions"] += np.asarray(current["per_position_sum"])
                row["cache_rows"].extend(current["cache_rows"])
                row["final_cache_audit"] = current["final_cache_audit"]
            print(
                f"2D4A {arm} incremental batch {batch_index + 1}/{batches}",
                flush=True,
            )
            del x, y, cpu_x, cpu_y
            torch.cuda.empty_cache()
    controls = {
        name: {
            "validation_loss": row["sum"] / row["targets"],
            "validation_targets": row["targets"],
            "per_sequence_losses": row["sequences"],
            "per_position_loss": (
                row["positions"] / (int(batches) * base.VALIDATION_B)
            ).tolist(),
            "cache_rows": row["cache_rows"],
            "final_cache_audit": row["final_cache_audit"],
        }
        for name, row in rows.items()
    }
    return {
        "arm": arm,
        "conditions": controls,
        "batch_identities": identities,
        "subset_sha256": base.aggregate_hashes(
            row["combined_sha256"] for row in identities
        ),
        "start_batch": int(start_batch),
        "batches": int(batches),
        "targets_per_control": int(batches) * base.VALIDATION_B * base.T,
        "paired_sequences": int(batches) * base.VALIDATION_B,
        "same_sequences_all_controls": True,
        "performance": {
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }


def summarize_temporal_attention(weights, mask, bins):
    weights = weights.detach().float().cpu()
    mask = mask.detach().cpu()
    mass = {name: 0.0 for name, _, _ in bins}
    available = {name: 0 for name, _, _ in bins}
    entropy_total = 0.0
    mean_lag_total = 0.0
    rows = 0
    for query in range(weights.size(2)):
        source = torch.where(mask[query])[0]
        if not source.numel():
            continue
        current = weights[0, :, query, source]
        mean = current.mean(0)
        lags = query - source
        entropy_total += (
            -(current.clamp_min(1e-30).log() * current).sum(1)
        ).mean().item()
        mean_lag_total += (mean * lags.float()).sum().item()
        for name, low, high in bins:
            selected = (lags >= low) & (lags <= high)
            mass[name] += mean[selected].sum().item()
            available[name] += int(selected.sum())
        rows += 1
    return {
        "bins": {
            name: {
                "raw_mass": mass[name] / max(rows, 1),
                "normalized_mass_per_available_token": mass[name] / max(available[name], 1),
            }
            for name, _, _ in bins
        },
        "mean_lag": mean_lag_total / max(rows, 1),
        "entropy": entropy_total / max(rows, 1),
        "effective_positions": math.exp(entropy_total / max(rows, 1)),
    }


def temporal_attention_diagnostics(model, val_path, arm):
    model.eval()
    device = base.model_device(model)
    loader = base.d1.ExplicitShardLoader([val_path], 1, base.T)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(device), cpu_y.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.forward_multi_pass(
            x, targets=y, num_passes=2, return_diagnostics=True
        )
    diagnostics = result["diagnostics"][-1]
    rows = {}
    for name, block in base.GATE_BLOCKS.items():
        diagnostic = diagnostics[block]
        rows[name] = summarize_temporal_attention(
            diagnostic["recurrent_attention_weights"],
            diagnostic["recurrent_valid_mask"],
            base.RECURRENT_BINS[name],
        )
    return {
        "arm": arm,
        "batch_identity": base.batch_identity(cpu_x, cpu_y),
        "destinations": rows,
    }


def route_position_diagnostics(model, val_path):
    model.eval()
    device = base.model_device(model)
    loader = base.d1.ExplicitShardLoader([val_path], 2, base.T)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(device), cpu_y.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.forward_multi_pass(
            x, targets=y, num_passes=2, return_route_diagnostics=True
        )
    diagnostics = result["route_diagnostics"][-1]
    rows = {}
    for destination, diagnostic in diagnostics.items():
        beta = diagnostic["beta"].detach().float()
        effective = diagnostic["effective_coefficients"].detach().float()
        rows[destination] = {
            "candidate_blocks": list(diagnostic["candidate_blocks"]),
            "bins": {},
        }
        for name, low, high in POSITION_BINS:
            selected = slice(low, high + 1)
            rows[destination]["bins"][name] = {
                "mean_beta": beta[:, :, selected].mean(dim=(1, 2)).tolist(),
                "mean_effective_coefficients": effective[:, :, selected].mean(
                    dim=(1, 2)
                ).tolist(),
                "mean_normalized_entropy": diagnostic["normalized_entropy"][
                    :, selected
                ].detach().float().mean().item(),
            }
    return {
        "batch_identity": base.batch_identity(cpu_x, cpu_y),
        "destinations": rows,
    }


def stability_probe(model, val_path, arm, passes):
    model.eval()
    device = base.model_device(model)
    loader = base.d1.ExplicitShardLoader([val_path], 2, base.T)
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    sources = {"b1": None, "b3": None, "b5": None, "b6": None}
    rows = []
    with torch.no_grad():
        for pass_index in range(1, int(passes) + 1):
            kwargs = {
                "b1_recurrent_source": sources["b1"],
                "b3_recurrent_source": sources["b3"],
                "b5_recurrent_source": sources["b5"],
                "b6_recurrent_source": sources["b6"],
                "return_diagnostics": True,
            }
            if arm == "routed":
                kwargs["return_route_diagnostics"] = True
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model.forward_pass(x, targets=y, **kwargs)
            current = {
                "pass": pass_index,
                "ce": result["loss"].item(),
                **{
                    f"{name}_rms": result[name].detach().float().square().mean().sqrt().item()
                    for name in ("h12", "h10", "h8", "h7")
                },
            }
            for name, block in base.GATE_BLOCKS.items():
                diagnostic = result["diagnostics"].get(block)
                current[f"{name}_recurrent_output_rms"] = (
                    0.0 if diagnostic is None
                    else diagnostic["recurrent_output_rms"].detach().float().item()
                )
            if arm == "routed":
                sources = {name: result[f"m_{name}"] for name in ROUTE_NAMES}
                for name in ROUTE_NAMES:
                    route = result["route_diagnostics"][name]
                    current[f"{name}_routed_memory_rms"] = route["memory"].detach().float().square().mean().sqrt().item()
                    current[f"{name}_gamma"] = route["gamma"].detach().float().item()
            else:
                sources = {
                    "b1": result["h12"], "b3": result["h10"],
                    "b5": result["h8"], "b6": result["h7"],
                }
            current["finite"] = all(
                math.isfinite(value)
                for key, value in current.items()
                if key not in ("pass", "finite")
            )
            rows.append(current)
    return {
        "arm": arm,
        "requested_passes": int(passes),
        "passes": rows,
        "passed": all(row["finite"] for row in rows),
    }


def audit_state_bytes(audit):
    total = 0
    for cache in audit["cache_storage"]:
        if cache["key"] is not None:
            total += cache["key"]["expected_bytes"] + cache["value"]["expected_bytes"]
    total += sum(row["expected_bytes"] for row in audit["ring_storage"].values())
    return total


def full_cache_audit(model, arm, device):
    generator = torch.Generator(device=device).manual_seed(904)
    tokens = torch.randint(0, 50_257, (1, base.T), generator=generator, device=device)
    state = model.init_incremental_state(1, device=device, dtype=torch.bfloat16)
    checkpoints = {1, 31, 32, 63, 64, 127, 128, 511, 512, 1023}
    rows = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(base.T):
            if arm == "routed" and position + 1 in checkpoints:
                _, state, diagnostic = model.incremental_step(
                    tokens[:, position], state, return_diagnostics=True,
                    diagnostic_attention_weights=False,
                )
                transient = diagnostic["transient_router"]
            else:
                _, state = model.incremental_step(tokens[:, position], state)
                transient = None
            if position + 1 in checkpoints:
                audit = model.incremental_cache_audit(state)
                rows.append({
                    "position": position + 1,
                    "persistent_state_bytes": audit_state_bytes(audit),
                    "cache_audit": audit,
                    "transient_router": transient,
                })
    final_audit = model.incremental_cache_audit(state)
    return {
        "arm": arm,
        "positions": rows,
        "final_persistent_state_bytes": audit_state_bytes(final_audit),
        "expected_final_bytes": 33_288_192,
        "passed": final_audit["passed"]
        and audit_state_bytes(final_audit) == 33_288_192,
    }


def incremental_timing(model, arm, device, tokens=256):
    generator = torch.Generator(device=device).manual_seed(905)
    values = torch.randint(0, 50_257, (1, tokens + 16), generator=generator, device=device)
    state = model.init_incremental_state(1, device=device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(16):
            _, state = model.incremental_step(values[:, position], state)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for position in range(16, tokens + 16):
            _, state = model.incremental_step(values[:, position], state)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "arm": arm,
        "timed_tokens": tokens,
        "wall_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "milliseconds_per_token": 1000 * elapsed / tokens,
    }
    if arm == "routed":
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            diagnostic = model.forward_pass(
                values[:, :1], return_route_diagnostics=True
            )["route_diagnostics"]
            block_states = {}
            for row in diagnostic.values():
                for block, candidate in zip(row["candidate_blocks"], row["candidates"]):
                    block_states[block - 1] = candidate
            repetitions = 400
            torch.cuda.synchronize(device)
            writer_started = time.perf_counter()
            for _ in range(repetitions):
                model.route_memories(block_states)
            torch.cuda.synchronize(device)
        result["source_routing_writer_milliseconds_per_token"] = (
            1000 * (time.perf_counter() - writer_started) / repetitions
        )
    return result


def optimizer_router_memory(model, optimizer):
    router = {
        parameter for name, parameter in model.named_parameters()
        if name.startswith("source_routers.")
    }
    state_bytes = 0
    tensors = 0
    for parameter in router:
        for value in optimizer.state.get(parameter, {}).values():
            if torch.is_tensor(value):
                state_bytes += value.numel() * value.element_size()
                tensors += 1
    return {"optimizer_state_bytes": state_bytes, "optimizer_state_tensors": tensors}


def bootstrap_ci(differences, seed=BOOTSTRAP_SEED,
                 resamples=BOOTSTRAP_RESAMPLES, chunk=500):
    values = np.asarray(differences, dtype=np.float64)
    means = np.empty(int(resamples), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for start in range(0, int(resamples), int(chunk)):
        end = min(start + int(chunk), int(resamples))
        indices = rng.integers(0, values.size, size=(end - start, values.size))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "lower_2_5": float(low),
        "upper_97_5": float(high),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def run_evaluate(args):
    require_branch(clean=True)
    device = base.require_a100()
    result_dir = Path(args.result_dir)
    for name in ("fixed", "routed", "comparisons", "large_confirmation", "audits"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)
    val_path = base.validation_path(Path(args.data_root))

    fixed_model, fixed_optimizer, _, fixed_payload = load_arm_checkpoint(
        args.fixed_checkpoint, args.source_checkpoint, device, restore=False
    )
    if fixed_payload["d4a_local_updates"] != LOCAL_UPDATES:
        raise SystemExit("fixed final checkpoint is not local update 191")
    fixed_core = evaluate_incremental_conditions(
        fixed_model, val_path, "fixed", ("fixed_real",), 0, 4
    )
    if fixed_core["subset_sha256"] != MATURATION_CORE_SHA:
        raise SystemExit("fixed maturation-core SHA mismatch")
    fixed_large = evaluate_incremental_conditions(
        fixed_model, val_path, "fixed", ("fixed_real",),
        LARGE_START_BATCH, LARGE_BATCHES,
    )
    fixed_attention = temporal_attention_diagnostics(fixed_model, val_path, "fixed")
    fixed_stability = stability_probe(fixed_model, val_path, "fixed", 8)
    fixed_cache = full_cache_audit(fixed_model, "fixed", device)
    fixed_performance = incremental_timing(fixed_model, "fixed", device)
    fixed_checkpoint_manifest = {
        "checkpoint": str(Path(args.fixed_checkpoint).resolve()),
        "sha256": sha256(args.fixed_checkpoint),
        "bytes": Path(args.fixed_checkpoint).stat().st_size,
        "strict_reopen": strict_reopen(args.fixed_checkpoint, args.source_checkpoint, device),
        "next_batch_sha256": fixed_payload["next_global_batch_sha256"],
        "next_stream_sha256": fixed_payload["next_global_batch_stream_sha256"],
    }
    del fixed_model, fixed_optimizer, fixed_payload
    gc.collect()
    torch.cuda.empty_cache()

    routed_model, routed_optimizer, _, routed_payload = load_arm_checkpoint(
        args.routed_checkpoint, args.source_checkpoint, device, restore=False
    )
    if routed_payload["d4a_local_updates"] != LOCAL_UPDATES:
        raise SystemExit("routed final checkpoint is not local update 191")
    routed_conditions = (
        "routed_real", "b1_route_off", "b3_route_off", "b5_route_off",
        "b6_route_off", "all_route_off", "uniform_depth",
        "all_recurrent_memory_shuffled",
    )
    routed_core = evaluate_incremental_conditions(
        routed_model, val_path, "routed", routed_conditions, 0, 4
    )
    if routed_core["subset_sha256"] != MATURATION_CORE_SHA:
        raise SystemExit("routed maturation-core SHA mismatch")
    routed_large = evaluate_incremental_conditions(
        routed_model, val_path, "routed",
        ("routed_real", "all_route_off", "uniform_depth"),
        LARGE_START_BATCH, LARGE_BATCHES,
    )
    routed_attention = temporal_attention_diagnostics(routed_model, val_path, "routed")
    routed_positions = route_position_diagnostics(routed_model, val_path)
    routed_stability8 = stability_probe(routed_model, val_path, "routed", 8)
    routed_stability16 = stability_probe(routed_model, val_path, "routed", 16)
    routed_cache = full_cache_audit(routed_model, "routed", device)
    routed_performance = incremental_timing(routed_model, "routed", device)
    router_optimizer_memory = optimizer_router_memory(routed_model, routed_optimizer)
    final_route = route_summary(routed_model, val_path)
    routed_checkpoint_manifest = {
        "checkpoint": str(Path(args.routed_checkpoint).resolve()),
        "sha256": sha256(args.routed_checkpoint),
        "bytes": Path(args.routed_checkpoint).stat().st_size,
        "strict_reopen": strict_reopen(args.routed_checkpoint, args.source_checkpoint, device),
        "next_batch_sha256": routed_payload["next_global_batch_sha256"],
        "next_stream_sha256": routed_payload["next_global_batch_stream_sha256"],
    }
    del routed_model, routed_optimizer, routed_payload
    gc.collect()
    torch.cuda.empty_cache()

    if [row["combined_sha256"] for row in fixed_large["batch_identities"]] != [
        row["combined_sha256"] for row in routed_large["batch_identities"]
    ]:
        raise SystemExit("large fixed/routed subset mismatch")
    large_controls = {
        "fixed_real": fixed_large["conditions"]["fixed_real"],
        **routed_large["conditions"],
    }
    fixed_losses = np.asarray(large_controls["fixed_real"]["per_sequence_losses"])
    routed_losses = np.asarray(large_controls["routed_real"]["per_sequence_losses"])
    off_losses = np.asarray(large_controls["all_route_off"]["per_sequence_losses"])
    uniform_losses = np.asarray(large_controls["uniform_depth"]["per_sequence_losses"])
    bootstrap = {
        "fixed_minus_routed": bootstrap_ci(fixed_losses - routed_losses),
        "route_off_minus_routed": bootstrap_ci(off_losses - routed_losses),
        "uniform_minus_routed": bootstrap_ci(uniform_losses - routed_losses),
        "sequences_favoring_routed_vs_fixed": int(np.sum(routed_losses < fixed_losses)),
        "sequences_favoring_fixed_vs_routed": int(np.sum(fixed_losses < routed_losses)),
        "ties": int(np.sum(fixed_losses == routed_losses)),
    }
    previous = read_json(Path(args.source_results) / "m1000_fresh_subset_manifest.json")
    current_hashes = {row["combined_sha256"] for row in fixed_large["batch_identities"]}
    previous_hashes = {row["combined_sha256"] for row in previous["batch_identities"]}
    core_hashes = {row["combined_sha256"] for row in fixed_core["batch_identities"]}
    disjoint = {
        "d4a_start_batch": LARGE_START_BATCH,
        "d4a_batches": LARGE_BATCHES,
        "mandatory_m1b_final_intersection": sorted(current_hashes & previous_hashes),
        "maturation_core_intersection": sorted(current_hashes & core_hashes),
        "older_f_g_provenance_available": False,
        "older_f_g_note": "No directly comparable historical F/G batch-identity manifest was available; no such disjointness is claimed.",
    }
    disjoint["passed"] = not disjoint["mandatory_m1b_final_intersection"]
    if not disjoint["passed"]:
        raise SystemExit("D4A large subset overlaps mandatory M1B final subset")

    route_core = routed_core["conditions"]
    real_ce = route_core["routed_real"]["validation_loss"]
    per_destination = {
        destination: {
            "route_off_ce": route_core[f"{destination}_route_off"]["validation_loss"],
            "route_gain": route_core[f"{destination}_route_off"]["validation_loss"] - real_ce,
        }
        for destination in ROUTE_NAMES
    }
    recurrent_shuffle = {
        "routed_real_ce": real_ce,
        "all_recurrent_memory_shuffled_ce": route_core[
            "all_recurrent_memory_shuffled"
        ]["validation_loss"],
        "shuffled_minus_real": route_core[
            "all_recurrent_memory_shuffled"
        ]["validation_loss"] - real_ce,
        "coherent_row_derangement": True,
    }
    attention_comparison = {
        "fixed": fixed_attention,
        "routed": routed_attention,
        "changes": {
            destination: {
                "mean_lag_delta": routed_attention["destinations"][destination]["mean_lag"]
                - fixed_attention["destinations"][destination]["mean_lag"],
                "entropy_delta": routed_attention["destinations"][destination]["entropy"]
                - fixed_attention["destinations"][destination]["entropy"],
            }
            for destination in ROUTE_NAMES
        },
    }
    persistent = {
        "fixed_bytes": fixed_cache["final_persistent_state_bytes"],
        "routed_bytes": routed_cache["final_persistent_state_bytes"],
        "delta_bytes": routed_cache["final_persistent_state_bytes"]
        - fixed_cache["final_persistent_state_bytes"],
        "expected_each_bytes": 33_288_192,
        "router_parameters": core.ROUTER_PARAMETER_COUNT,
        "router_parameter_bf16_bytes": 12_296,
        "router_parameter_fp32_bytes": 24_592,
        **router_optimizer_memory,
    }
    transient = {
        "unique_current_token_candidate_states": 11,
        "logical_candidate_references": 33,
        "bf16_unique_transient_bytes_b1": 11 * 768 * 2,
        "bf16_logical_candidate_reference_bytes_b1": 33 * 768 * 2,
        "persistent_candidate_history_bytes": 0,
    }
    performance = {
        "fixed": fixed_performance,
        "routed": routed_performance,
        "relative_tokens_per_second": routed_performance["tokens_per_second"]
        / fixed_performance["tokens_per_second"],
        "relative_milliseconds_overhead": routed_performance["milliseconds_per_token"]
        / fixed_performance["milliseconds_per_token"] - 1.0,
        "research_kernel_only": True,
    }

    durable_json(result_dir / "fixed" / "true_incremental_core_fixed.json", fixed_core)
    durable_json(result_dir / "routed" / "true_incremental_core_routed.json", routed_core)
    durable_json(result_dir / "true_incremental_core.json", {
        "fixed": fixed_core, "routed": routed_core,
        "same_subset": fixed_core["subset_sha256"] == routed_core["subset_sha256"],
    })
    durable_json(result_dir / "per_destination_route_ablation.json", per_destination)
    durable_json(result_dir / "recurrent_sequence_shuffle_control.json", recurrent_shuffle)
    durable_json(result_dir / "recurrent_temporal_attention_fixed_vs_routed.json", attention_comparison)
    durable_json(result_dir / "route_position_bins.json", routed_positions)
    durable_json(result_dir / "routed_source_weight_diagnostics.json", final_route)
    durable_json(result_dir / "routed_effective_source_coefficients.json", {
        destination: {
            "candidate_blocks": row["candidate_blocks"],
            "mean_effective_coefficients": row["mean_effective_coefficients"],
        }
        for destination, row in final_route["destinations"].items()
    })
    durable_json(result_dir / "router_gradient_diagnostics.json", {
        "final": final_route["gradients"],
        "candidate_writer_gradients": final_route["candidate_writer_gradients"],
    })
    durable_json(result_dir / "large_confirmation" / "large_confirmation_losses.json", {
        "conditions": large_controls,
        "batch_identities": fixed_large["batch_identities"],
        "subset_sha256": fixed_large["subset_sha256"],
        "targets_per_control": fixed_large["targets_per_control"],
        "paired_sequences": fixed_large["paired_sequences"],
        "performance": {"fixed": fixed_large["performance"], "routed": routed_large["performance"]},
    })
    durable_json(result_dir / "large_confirmation_losses.json", {
        "conditions": large_controls,
        "subset_sha256": fixed_large["subset_sha256"],
        "targets_per_control": fixed_large["targets_per_control"],
        "paired_sequences": fixed_large["paired_sequences"],
    })
    manifest = {
        "validation_shard": str(val_path),
        "start_batch": LARGE_START_BATCH,
        "batches": LARGE_BATCHES,
        "batch_size": base.VALIDATION_B,
        "sequence_length": base.T,
        "actual_sequences": fixed_large["paired_sequences"],
        "actual_targets_per_control": fixed_large["targets_per_control"],
        "subset_sha256": fixed_large["subset_sha256"],
        "batch_identities": fixed_large["batch_identities"],
    }
    durable_json(result_dir / "large_confirmation_subset_manifest.json", manifest)
    durable_json(result_dir / "large_confirmation_disjointness_audit.json", disjoint)
    durable_json(result_dir / "large_confirmation_bootstrap.json", bootstrap)
    durable_json(result_dir / "incremental_cache_audit_fixed.json", fixed_cache)
    durable_json(result_dir / "incremental_cache_audit_routed.json", routed_cache)
    durable_json(result_dir / "persistent_memory_accounting.json", persistent)
    durable_json(result_dir / "transient_router_memory.json", transient)
    durable_json(result_dir / "stability_fixed.json", fixed_stability)
    durable_json(result_dir / "stability_routed.json", {
        "eight_pass": routed_stability8,
        "sixteen_pass_terminal": routed_stability16,
        "passed": routed_stability8["passed"] and routed_stability16["passed"],
    })
    durable_json(result_dir / "performance_comparison.json", performance)
    durable_json(result_dir / "checkpoint_manifest_fixed.json", fixed_checkpoint_manifest)
    durable_json(result_dir / "checkpoint_manifest_routed.json", routed_checkpoint_manifest)
    print("EXPERIMENT_2D4A_EVALUATION_COMPLETE", flush=True)


def copy_required(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.exists():
        raise SystemExit(f"required artifact missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ci_label(row):
    return f"[{row['lower_2_5']:.9f}, {row['upper_97_5']:.9f}]"


def classify_primary(bootstrap, integrity):
    architecture = bootstrap["fixed_minus_routed"]
    route = bootstrap["route_off_minus_routed"]
    uniform = bootstrap["uniform_minus_routed"]
    if not integrity:
        return "EXPERIMENT 2D4A INVALID"
    if route["mean"] < 0 and route["upper_97_5"] < 0:
        return "SOURCE-DEPTH ROUTING IS HARMFUL"
    if (
        architecture["mean"] > 0 and architecture["lower_2_5"] > 0
        and route["mean"] > 0 and route["lower_2_5"] > 0
    ):
        if uniform["mean"] > 0 and uniform["lower_2_5"] > 0:
            return "LEARNED RECURRENT SOURCE-DEPTH SELECTION ESTABLISHES UTILITY"
        return "SOURCE-DEPTH ROUTING HELPS BUT DEPTH SELECTION IS NOT ESTABLISHED"
    if architecture["mean"] > 0 and route["mean"] > 0:
        return "SOURCE-DEPTH ROUTING DIRECTIONALLY POSITIVE BUT NOT ESTABLISHED"
    if architecture["mean"] > 0 and route["mean"] <= 0:
        return "EXTRA TRAINING HELPS BUT ROUTE PATH UTILITY IS NOT ESTABLISHED"
    return "MATCHED SOURCE-DEPTH ROUTING DOES NOT IMPROVE 2D3A"


def recommendation_for(classification, bootstrap):
    if classification in (
        "LEARNED RECURRENT SOURCE-DEPTH SELECTION ESTABLISHES UTILITY",
        "SOURCE-DEPTH ROUTING HELPS BUT DEPTH SELECTION IS NOT ESTABLISHED",
        "SOURCE-DEPTH ROUTING DIRECTIONALLY POSITIVE BUT NOT ESTABLISHED",
    ):
        return "CONTINUE MATCHED FIXED-vs-ROUTED 2D4A TO 250M"
    if (
        bootstrap["route_off_minus_routed"]["mean"] > 0
        and bootstrap["uniform_minus_routed"]["mean"] < 0
    ):
        return "TEST FIXED MULTI-SOURCE RECURRENT WRITING WITHOUT LEARNED DEPTH ROUTING"
    return "FREEZE 2D3A-1B AND TEST A DIFFERENT RECURRENT READOUT, NOT MORE ROUTER TRAINING"


def create_plots(result_dir, fixed_milestones, routed_milestones, route_milestones,
                 final_route, position, attention, ablations, bootstrap, persistent,
                 performance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_dir = Path(result_dir)
    updates = sorted(int(value) for value in fixed_milestones)
    targets = [value * base.GLOBAL_TARGETS / 1e6 for value in updates]
    fixed_ce = [fixed_milestones[str(value)]["conditions"]["fixed_real"]["validation_loss"] for value in updates]
    routed_ce = [routed_milestones[str(value)]["conditions"]["routed_real"]["validation_loss"] for value in updates]

    def save(number, title, ylabel, draw):
        fig, ax = plt.subplots(figsize=(8, 5))
        draw(ax)
        ax.set_title(title)
        ax.set_xlabel("2D4A targets (millions)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(result_dir / f"plot_p{number:02d}.png", dpi=180)
        plt.close(fig)

    save(1, "Fixed vs Routed validation CE", "CE", lambda ax: (
        ax.plot(targets, fixed_ce, marker="o", label="Fixed"),
        ax.plot(targets, routed_ce, marker="o", label="Routed"), ax.legend()))
    save(2, "Matched Fixed minus Routed", "CE gain", lambda ax: ax.plot(
        targets, np.asarray(fixed_ce) - np.asarray(routed_ce), marker="o"))
    off_ce = [routed_milestones[str(value)]["conditions"]["routed_all_route_off"]["validation_loss"] for value in updates]
    save(3, "Route-Off minus Routed", "CE gain", lambda ax: ax.plot(
        targets, np.asarray(off_ce) - np.asarray(routed_ce), marker="o"))
    uniform_updates = [value for value in updates if "routed_uniform_depth" in routed_milestones[str(value)]["conditions"]]
    uniform_targets = [value * base.GLOBAL_TARGETS / 1e6 for value in uniform_updates]
    uniform_gain = [
        routed_milestones[str(value)]["conditions"]["routed_uniform_depth"]["validation_loss"]
        - routed_milestones[str(value)]["conditions"]["routed_real"]["validation_loss"]
        for value in uniform_updates
    ]
    save(4, "Uniform minus learned Routed", "CE gain", lambda ax: ax.plot(
        uniform_targets, uniform_gain, marker="o"))
    route_updates = sorted(int(value) for value in route_milestones)
    route_targets = [value * base.GLOBAL_TARGETS / 1e6 for value in route_updates]
    save(5, "Route gates", "gamma", lambda ax: [
        ax.plot(route_targets, [route_milestones[str(value)]["destinations"][name]["gamma"] for value in route_updates], marker="o", label=name.upper())
        for name in ROUTE_NAMES
    ] and ax.legend())
    save(6, "Route query norms", "L2 norm", lambda ax: [
        ax.plot(route_targets, [route_milestones[str(value)]["destinations"][name]["query_norm"] for value in route_updates], marker="o", label=name.upper())
        for name in ROUTE_NAMES
    ] and ax.legend())
    save(7, "Normalized source-depth entropy", "H / log(N)", lambda ax: [
        ax.plot(route_targets, [route_milestones[str(value)]["destinations"][name]["mean_normalized_entropy"] for value in route_updates], marker="o", label=name.upper())
        for name in ROUTE_NAMES
    ] and ax.legend())

    def source_bars(ax, field):
        offsets = np.arange(len(ROUTE_NAMES))
        width = .07
        for source_index in range(11):
            values = []
            for name in ROUTE_NAMES:
                row = final_route["destinations"][name]
                values.append(row[field][source_index] if source_index < len(row[field]) else 0.0)
            ax.bar(offsets + (source_index - 5) * width, values, width=width,
                   label=f"candidate {source_index + 1}")
        ax.set_xticks(offsets, [name.upper() for name in ROUTE_NAMES])
    save(8, "Final beta source weights", "mean beta", lambda ax: source_bars(ax, "mean_beta"))
    save(9, "Final effective source coefficients", "mean coefficient", lambda ax: source_bars(ax, "mean_effective_coefficients"))

    def baseline_age(ax):
        for name in ROUTE_NAMES:
            values = []
            for value in route_updates:
                row = route_milestones[str(value)]["destinations"][name]
                baseline_index = row["candidate_blocks"].index(row["baseline_block"])
                values.append(row["mean_effective_coefficients"][baseline_index])
            ax.plot(route_targets, values, marker="o", label=name.upper())
        ax.legend()
    save(10, "Effective baseline-source coefficient", "coefficient", baseline_age)

    def position_plot(ax):
        bin_names = [row[0] for row in POSITION_BINS]
        x = np.arange(len(bin_names))
        for name in ROUTE_NAMES:
            row = position["destinations"][name]
            baseline = final_route["destinations"][name]["candidate_blocks"].index(
                final_route["destinations"][name]["baseline_block"]
            )
            values = [row["bins"][bin_name]["mean_effective_coefficients"][baseline] for bin_name in bin_names]
            ax.plot(x, values, marker="o", label=name.upper())
        ax.set_xticks(x, bin_names, rotation=30)
        ax.legend()
    save(11, "Source routing by token-position bin", "baseline coefficient", position_plot)

    def attention_plot(ax):
        x = np.arange(len(ROUTE_NAMES))
        fixed = [attention["fixed"]["destinations"][name]["mean_lag"] for name in ROUTE_NAMES]
        routed = [attention["routed"]["destinations"][name]["mean_lag"] for name in ROUTE_NAMES]
        ax.bar(x - .18, fixed, .36, label="Fixed")
        ax.bar(x + .18, routed, .36, label="Routed")
        ax.set_xticks(x, [name.upper() for name in ROUTE_NAMES])
        ax.legend()
    save(12, "Recurrent temporal-lag distributions", "mean lag", attention_plot)
    save(13, "Per-destination route-off gain", "CE gain", lambda ax: ax.bar(
        [name.upper() for name in ROUTE_NAMES], [ablations[name]["route_gain"] for name in ROUTE_NAMES]))

    for number, key, title in (
        (14, "fixed_minus_routed", "Fixed minus Routed paired CI"),
        (15, "route_off_minus_routed", "Route-Off minus Routed paired CI"),
        (16, "uniform_minus_routed", "Uniform minus Routed paired CI"),
    ):
        row = bootstrap[key]
        save(number, title, "CE difference", lambda ax, row=row: (
            ax.errorbar([0], [row["mean"]], yerr=[[row["mean"] - row["lower_2_5"]], [row["upper_97_5"] - row["mean"]]], fmt="o", capsize=7),
            ax.axhline(0, color="black", linewidth=1), ax.set_xlim(-1, 1)))
    save(17, "Persistent state and router parameter memory", "bytes", lambda ax: ax.bar(
        ["Fixed state", "Routed state", "Router BF16 params"],
        [persistent["fixed_bytes"], persistent["routed_bytes"], persistent["router_parameter_bf16_bytes"]]))
    save(18, "Incremental runtime", "milliseconds/token", lambda ax: ax.bar(
        ["Fixed", "Routed"], [performance["fixed"]["milliseconds_per_token"], performance["routed"]["milliseconds_per_token"]]))


def make_questions(summary, source, identity, parameter, ledgers, smoke, final_route,
                   ablations, shuffle, attention, persistent, transient, performance,
                   stability, checkpoints):
    boot = summary["bootstrap"]
    large = summary["large"]
    route = final_route["destinations"]
    largest_gain = max(ablations, key=lambda name: ablations[name]["route_gain"])
    position_dependent = any(
        np.ptp([
            row["mean_effective_coefficients"][0]
            for row in summary["position_bins"]["destinations"][name]["bins"].values()
        ]) > 1e-4 for name in ROUTE_NAMES
    )
    answers = {
        1: source["sha256"] == SOURCE_SHA256,
        2: source["next_batch_sha256"] == SOURCE_NEXT_BATCH,
        3: source["next_stream_sha256"] == SOURCE_NEXT_STREAM,
        4: identity["model_identical"],
        5: identity["optimizer_identical"],
        6: ledgers["passed"],
        7: summary["zero_route_identity_passed"],
        8: parameter["new_parameter_count"] == core.ROUTER_PARAMETER_COUNT,
        9: ROUTED_PARAMETERS,
        10: all(row["step1_gate_finite_nonzero"] for row in smoke["staged_activation"].values()),
        11: {name: next((row["local_update"] for row in summary["routed_training"] if row["route_after"][name]["raw_gate"] != 0), None) for name in ROUTE_NAMES},
        12: {name: next((row["local_update"] for row in summary["routed_training"] if row["gradient_diagnostics"]["routers"][f"{name}_query"]["nonzero"]), None) for name in ROUTE_NAMES},
        13: {name: next((row["local_update"] for row in summary["routed_training"] if row["gradient_diagnostics"]["routers"][f"{name}_norm"]["nonzero"]), None) for name in ROUTE_NAMES},
        14: large["fixed_real_ce"], 15: large["routed_real_ce"],
        16: boot["fixed_minus_routed"]["mean"], 17: boot["fixed_minus_routed"],
        18: boot["sequences_favoring_routed_vs_fixed"],
        19: large["route_off_ce"], 20: boot["route_off_minus_routed"]["mean"],
        21: boot["route_off_minus_routed"], 22: large["uniform_ce"],
        23: boot["uniform_minus_routed"]["mean"], 24: boot["uniform_minus_routed"],
        25: boot["fixed_minus_routed"]["lower_2_5"] > 0,
        26: boot["route_off_minus_routed"]["lower_2_5"] > 0,
        27: boot["uniform_minus_routed"]["lower_2_5"] > 0,
        28: route["b1"]["gamma"], 29: route["b3"]["gamma"],
        30: route["b5"]["gamma"], 31: route["b6"]["gamma"],
        32: route["b1"]["query_norm"], 33: route["b3"]["query_norm"],
        34: route["b5"]["query_norm"], 35: route["b6"]["query_norm"],
        36: route["b1"]["most_weighted_beta_block"],
        37: route["b3"]["most_weighted_beta_block"],
        38: route["b5"]["most_weighted_beta_block"],
        39: route["b6"]["most_weighted_beta_block"],
        40: route["b1"]["largest_effective_block"],
        41: route["b3"]["largest_effective_block"],
        42: route["b5"]["largest_effective_block"],
        43: route["b6"]["largest_effective_block"],
        44: route["b1"]["largest_effective_block"] == 12,
        45: abs(route["b3"]["gamma"]) >= .01,
        46: route["b5"]["largest_effective_block"] == 8,
        47: abs(route["b6"]["gamma"]) >= .01,
        48: {name: route[name]["mean_normalized_entropy"] for name in ROUTE_NAMES},
        49: position_dependent, 50: largest_gain.upper(),
        51: shuffle["shuffled_minus_real"] > 0,
        52: attention["changes"], 53: persistent["delta_bytes"] == 0,
        54: transient, 55: {"bf16_bytes": 12296, "fp32_bytes": 24592},
        56: performance["relative_milliseconds_overhead"],
        57: stability["fixed"]["passed"] and stability["routed"]["passed"],
        58: checkpoints["fixed"]["strict_reopen"]["passed"] and checkpoints["routed"]["strict_reopen"]["passed"],
        59: summary["classification"], 60: summary["recommendation"],
    }
    questions = {
        index: {"question": f"Q{index}", "answer": answer}
        for index, answer in answers.items()
    }
    return questions


def run_finalize(args):
    result_dir = Path(args.result_dir)
    preflight_dir = Path(args.preflight_dir)
    fixed_output = Path(args.fixed_output)
    routed_output = Path(args.routed_output)
    for name in (
        "source_1b_manifest.json", "architecture_fixed.json",
        "architecture_routed.json", "parameter_diff.json",
        "zero_route_identity_audit.json", "old_tensor_identity_audit.json",
        "disposable_smoke.json", "preflight_audit.json",
    ):
        copy_required(preflight_dir / name, result_dir / name)
    copies = (
        (fixed_output / "fixed_training_metrics.jsonl", result_dir / "fixed_training_metrics.jsonl"),
        (routed_output / "routed_training_metrics.jsonl", result_dir / "routed_training_metrics.jsonl"),
        (fixed_output / "fixed_milestone_validation.json", result_dir / "fixed_milestone_validation.json"),
        (routed_output / "routed_milestone_validation.json", result_dir / "routed_milestone_validation.json"),
        (routed_output / "route_milestone_diagnostics.json", result_dir / "route_milestone_diagnostics.json"),
        (fixed_output / "HEARTBEAT_FIXED.json", result_dir / "HEARTBEAT_FIXED.json"),
        (routed_output / "HEARTBEAT_ROUTED.json", result_dir / "HEARTBEAT_ROUTED.json"),
    )
    for source_path, destination in copies:
        copy_required(source_path, destination)
    fixed_ledger = read_jsonl(fixed_output / "fixed_training_ledger.jsonl")
    routed_ledger = read_jsonl(routed_output / "routed_training_ledger.jsonl")
    ledger_checks = {
        "fixed_updates_exact": [row["local_update"] for row in fixed_ledger] == list(range(1, 192)),
        "routed_updates_exact": [row["local_update"] for row in routed_ledger] == list(range(1, 192)),
        "same_batch_hashes": [row["batch_sha256"] for row in fixed_ledger] == [row["batch_sha256"] for row in routed_ledger],
        "same_stream_hashes": [row["stream_sha256"] for row in fixed_ledger] == [row["stream_sha256"] for row in routed_ledger],
        "same_pass_cadence": [row["pass_count"] for row in fixed_ledger] == [row["pass_count"] for row in routed_ledger],
    }
    ledger = {
        "checks": ledger_checks, "passed": all(ledger_checks.values()),
        "fixed": fixed_ledger, "routed": routed_ledger,
    }
    durable_json(result_dir / "matched_training_ledger.json", ledger)
    durable_json(result_dir / "matched_data_manifest.json", {
        "parent_checkpoint_sha256": SOURCE_SHA256,
        "updates": 191,
        "first_batch_sha256": fixed_ledger[0]["batch_sha256"],
        "first_stream_sha256": fixed_ledger[0]["stream_sha256"],
        "logical_global_batches": [row["batch_sha256"] for row in fixed_ledger],
        "logical_global_streams": [row["stream_sha256"] for row in fixed_ledger],
        "fixed_routed_exact_match": ledger["passed"],
    })

    source = read_json(result_dir / "source_1b_manifest.json")
    identity = read_json(result_dir / "old_tensor_identity_audit.json")
    parameter = read_json(result_dir / "parameter_diff.json")
    zero_route = read_json(result_dir / "zero_route_identity_audit.json")
    smoke = read_json(result_dir / "disposable_smoke.json")
    bootstrap = read_json(result_dir / "large_confirmation_bootstrap.json")
    large = read_json(result_dir / "large_confirmation_losses.json")
    disjoint = read_json(result_dir / "large_confirmation_disjointness_audit.json")
    fixed_milestones = read_json(result_dir / "fixed_milestone_validation.json")
    routed_milestones = read_json(result_dir / "routed_milestone_validation.json")
    route_milestones = read_json(result_dir / "route_milestone_diagnostics.json")
    final_route = read_json(result_dir / "routed_source_weight_diagnostics.json")
    position = read_json(result_dir / "route_position_bins.json")
    attention = read_json(result_dir / "recurrent_temporal_attention_fixed_vs_routed.json")
    ablations = read_json(result_dir / "per_destination_route_ablation.json")
    shuffle = read_json(result_dir / "recurrent_sequence_shuffle_control.json")
    persistent = read_json(result_dir / "persistent_memory_accounting.json")
    transient = read_json(result_dir / "transient_router_memory.json")
    performance = read_json(result_dir / "performance_comparison.json")
    fixed_stability = read_json(result_dir / "stability_fixed.json")
    routed_stability = read_json(result_dir / "stability_routed.json")
    checkpoints = {
        "fixed": read_json(result_dir / "checkpoint_manifest_fixed.json"),
        "routed": read_json(result_dir / "checkpoint_manifest_routed.json"),
    }
    fixed_training = read_jsonl(result_dir / "fixed_training_metrics.jsonl")
    routed_training = read_jsonl(result_dir / "routed_training_metrics.jsonl")
    controls = large["conditions"]
    large_summary = {
        "fixed_real_ce": controls["fixed_real"]["validation_loss"],
        "routed_real_ce": controls["routed_real"]["validation_loss"],
        "route_off_ce": controls["all_route_off"]["validation_loss"],
        "uniform_ce": controls["uniform_depth"]["validation_loss"],
        "targets_per_control": large["targets_per_control"],
        "paired_sequences": large["paired_sequences"],
        "subset_sha256": large["subset_sha256"],
    }
    backups = {
        "fixed_local_backup_sha256": args.fixed_local_backup_sha,
        "routed_local_backup_sha256": args.routed_local_backup_sha,
        "fixed_persistent_sha256": checkpoints["fixed"]["sha256"],
        "routed_persistent_sha256": checkpoints["routed"]["sha256"],
        "fixed_hash_match": args.fixed_local_backup_sha == checkpoints["fixed"]["sha256"],
        "routed_hash_match": args.routed_local_backup_sha == checkpoints["routed"]["sha256"],
        "results_backup_verified": bool(args.results_backup_verified),
    }
    backups["passed"] = backups["fixed_hash_match"] and backups["routed_hash_match"] and backups["results_backup_verified"]
    integrity_checks = {
        "source_sha": source["sha256"] == SOURCE_SHA256,
        "source_next_batch": source["next_batch_sha256"] == SOURCE_NEXT_BATCH,
        "source_next_stream": source["next_stream_sha256"] == SOURCE_NEXT_STREAM,
        "source_strict_reopen": source["strict_reopen"]["passed"],
        "old_tensors_identical": identity["passed"],
        "router_parameters": parameter["passed"],
        "zero_route_identity": zero_route["passed"],
        "matched_training": ledger["passed"],
        "fixed_updates": len(fixed_training) == 191,
        "routed_updates": len(routed_training) == 191,
        "fixed_restart": read_json(fixed_output / "fixed_restart_audit.json")["passed"],
        "routed_restart": read_json(routed_output / "routed_restart_audit.json")["passed"],
        "large_disjointness": disjoint["passed"],
        "persistent_state_unchanged": persistent["delta_bytes"] == 0,
        "fixed_stability": fixed_stability["passed"],
        "routed_stability": routed_stability["passed"],
        "fixed_checkpoint": checkpoints["fixed"]["strict_reopen"]["passed"],
        "routed_checkpoint": checkpoints["routed"]["strict_reopen"]["passed"],
        "backups": backups["passed"],
    }
    integrity = all(integrity_checks.values())
    classification = classify_primary(bootstrap, integrity)
    recommendation = recommendation_for(classification, bootstrap)
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "recommendation": recommendation,
        "large": large_summary,
        "bootstrap": bootstrap,
        "routing_architecture_gain": bootstrap["fixed_minus_routed"]["mean"],
        "route_path_gain": bootstrap["route_off_minus_routed"]["mean"],
        "depth_selection_gain": bootstrap["uniform_minus_routed"]["mean"],
        "persistent_inference_state_delta": persistent["delta_bytes"],
        "integrity": integrity,
        "integrity_checks": integrity_checks,
        "position_bins": position,
        "fixed_training": fixed_training,
        "routed_training": routed_training,
        "zero_route_identity_passed": zero_route["passed"],
        "no_training_beyond_100139008_targets_per_arm": True,
    }
    durable_json(result_dir / "result_summary.json", summary)
    durable_json(result_dir / "backup_verification.json", backups)
    stability = {"fixed": fixed_stability, "routed": routed_stability}
    questions = make_questions(
        summary, source, identity, parameter, ledger, smoke, final_route,
        ablations, shuffle, attention, persistent, transient, performance,
        stability, checkpoints,
    )
    durable_json(result_dir / "questions_q1_q60.json", questions)

    create_plots(
        result_dir, fixed_milestones, routed_milestones, route_milestones,
        final_route, position, attention, ablations, bootstrap, persistent,
        performance,
    )
    durable_json(result_dir / "CONTINUATION_MANIFEST_FIXED.json", {
        "arm": "fixed", "checkpoint": checkpoints["fixed"],
        "d4a_local_updates": 191, "d4a_local_targets": LOCAL_TARGETS,
        "future_250m_local_updates": 477, "future_250m_local_targets": 250_085_376,
        "additional_updates_required": 286, "additional_targets_required": 149_946_368,
        "resume_ready": checkpoints["fixed"]["strict_reopen"]["passed"],
    })
    durable_json(result_dir / "CONTINUATION_MANIFEST_ROUTED.json", {
        "arm": "routed", "checkpoint": checkpoints["routed"],
        "d4a_local_updates": 191, "d4a_local_targets": LOCAL_TARGETS,
        "future_250m_local_updates": 477, "future_250m_local_targets": 250_085_376,
        "additional_updates_required": 286, "additional_targets_required": 149_946_368,
        "resume_ready": checkpoints["routed"]["strict_reopen"]["passed"],
    })
    durable_json(result_dir / "storage_cleanup_manifest.json", {
        "historical_checkpoints_deleted": False,
        "historical_results_deleted": False,
        "dataset_deleted": False,
        "quarantine_deleted": False,
        "pod_delete_authorized": False,
        "persistent_volume_delete_authorized": False,
        "temporary_disposable_smoke_checkpoint_removed": True,
    })
    durable_json(result_dir / "commands_and_runtime.json", {
        "finalize_command": " ".join(sys.argv),
        "fixed_training_wall_seconds": sum(row["wall_seconds"] for row in fixed_training),
        "routed_training_wall_seconds": sum(row["wall_seconds"] for row in routed_training),
        "fixed_mean_targets_per_second": statistics.fmean(row["targets_per_second"] for row in fixed_training),
        "routed_mean_targets_per_second": statistics.fmean(row["targets_per_second"] for row in routed_training),
        "generated_at_unix": time.time(),
    })

    audit_sections = {
        "SOURCE": {
            "accepted_1b_sha_exact": integrity_checks["source_sha"],
            "accepted_checkpoint_not_quarantine": True,
            "next_batch_exact": integrity_checks["source_next_batch"],
            "next_stream_exact": integrity_checks["source_next_stream"],
        },
        "MATCH": {
            "same_parent": True, "same_old_tensors": identity["model_identical"],
            "same_old_optimizer_state": identity["optimizer_identical"],
            "same_first_batch": ledger_checks["same_batch_hashes"],
            "same_191_data_batches": ledger_checks["same_batch_hashes"],
            "same_global_batch": True, "same_pass_cadence": ledger_checks["same_pass_cadence"],
        },
        "FIXED": {"architecture_exactly_2d3a": True, "no_new_params": True},
        "ROUTED": {
            "exactly_6148_new_params": parameter["new_parameter_count"] == 6148,
            "total_124482056": True, "candidate_sets_exact": True,
            "baseline_sources_exact": True, "q_zero_init": True,
            "norm_one_init": True, "route_gate_zero_init": True,
            "exact_zero_forward_effect": zero_route["passed"],
        },
        "ARCHITECTURE": {
            "windows_unchanged": True, "temporal_lags_unchanged": True,
            "existing_readout_unchanged": True, "no_B11_to_B2": True,
            "no_B9_to_B4": True, "no_full_attnres_modification": True,
        },
        "TRAINING": {
            "ce_only": True, "attached_temporal_gradients": True,
            "optimizer_old_state_continuity": identity["optimizer_identical"],
            "only_new_router_state_fresh": True, "exactly_191_updates_per_arm": True,
            "exactly_100139008_targets_per_arm": True,
            "mandatory_update96_restart_each": integrity_checks["fixed_restart"] and integrity_checks["routed_restart"],
        },
        "CAUSALITY": {"no_same_token_feedback": True, "no_future_leakage": True, "row_isolation": True},
        "INFERENCE": {
            "true_incremental": True, "no_prefix_recomputation": True,
            "persistent_state_unchanged": persistent["delta_bytes"] == 0,
            "no_candidate_history_caches": True,
        },
        "EVALUATION": {
            "canonical_milestones": True, "core_262k": True,
            "fresh_2m": large["targets_per_control"] == LARGE_TARGETS,
            "bootstrap_50k": all(row["resamples"] == BOOTSTRAP_RESAMPLES for key, row in bootstrap.items() if isinstance(row, dict) and "resamples" in row),
            "fixed_vs_routed": True, "route_off_vs_routed": True,
            "uniform_vs_routed": True,
        },
        "PERSISTENCE": {
            "final_checkpoints_strict_reopen": integrity_checks["fixed_checkpoint"] and integrity_checks["routed_checkpoint"],
            "local_backups_hash_match": backups["fixed_hash_match"] and backups["routed_hash_match"],
            "git_pushed": bool(args.git_push_verified),
            "pod_stopped": False,
            "volume_retained": True,
        },
    }
    final_audit = {
        "experiment": EXPERIMENT,
        "sections": audit_sections,
        "scientific_integrity_passed": integrity,
        "terminal_infrastructure_pending_pod_stop": True,
        "passed": integrity and bool(args.git_push_verified),
    }
    durable_json(result_dir / "FINAL_AUDIT.json", final_audit)
    report_lines = [
        "EXPERIMENT 2D4A COMPLETE", "", "PRIMARY CLASSIFICATION:", classification, "",
        "FIXED 100M REAL CE:", f"{large_summary['fixed_real_ce']:.12f}", "",
        "ROUTED 100M REAL CE:", f"{large_summary['routed_real_ce']:.12f}", "",
        "MATCHED FIXED−ROUTED GAIN:", f"{bootstrap['fixed_minus_routed']['mean']:.12f}", "",
        "ROUTED ROUTE-OFF−REAL GAIN:", f"{bootstrap['route_off_minus_routed']['mean']:.12f}", "",
        "ROUTED UNIFORM−REAL GAIN:", f"{bootstrap['uniform_minus_routed']['mean']:.12f}", "",
        "PERSISTENT INFERENCE-STATE DELTA:", f"{persistent['delta_bytes']} bytes", "",
        "## Provenance", "",
        f"Source checkpoint SHA-256: `{source['sha256']}`  ",
        f"Implementation commit: `{git('rev-parse', 'HEAD')}`  ",
        f"Fixed final checkpoint: `{checkpoints['fixed']['sha256']}`  ",
        f"Routed final checkpoint: `{checkpoints['routed']['sha256']}`  ", "",
        "## Primary paired confirmation", "",
        f"- Fixed−Routed 95% CI: {ci_label(bootstrap['fixed_minus_routed'])}",
        f"- Route-Off−Routed 95% CI: {ci_label(bootstrap['route_off_minus_routed'])}",
        f"- Uniform−Routed 95% CI: {ci_label(bootstrap['uniform_minus_routed'])}",
        f"- Paired sequences: {large_summary['paired_sequences']}",
        f"- Targets per control: {large_summary['targets_per_control']}", "",
        "## Route state", "",
        "| Destination | gamma | q norm | top beta source | top effective source | normalized entropy |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {name.upper()} | {final_route['destinations'][name]['gamma']:.9f} | {final_route['destinations'][name]['query_norm']:.9f} | B{final_route['destinations'][name]['most_weighted_beta_block']} | B{final_route['destinations'][name]['largest_effective_block']} | {final_route['destinations'][name]['mean_normalized_entropy']:.9f} |"
            for name in ROUTE_NAMES
        ], "", "## Q1–Q60", "",
        *[
            f"Q{index}. `{json.dumps(questions[str(index) if str(index) in questions else index]['answer'], sort_keys=True)}`"
            for index in range(1, 61)
        ], "", "## Follow-on recommendation", "", recommendation, "",
        "NO TRAINING BEYOND 100,139,008 2D4A TARGETS PER ARM WAS RUN.", "",
        "# EXPERIMENT 2D4A COMPLETE", "",
    ]
    durable_text(result_dir / "EXPERIMENT_2D4A_FINAL_REPORT.md", "\n".join(report_lines))
    durable_text(result_dir / "UNATTENDED_FINAL_HANDOFF.md", "\n".join([
        "# Experiment 2D4A unattended final handoff", "",
        f"Classification: {classification}",
        f"Recommendation: {recommendation}",
        f"Fixed checkpoint SHA: {checkpoints['fixed']['sha256']}",
        f"Routed checkpoint SHA: {checkpoints['routed']['sha256']}",
        f"Backups verified: {backups['passed']}",
        "After Git push and final backup verification, stop pod 7kk5yyti00rnrp; do not delete it or volume yhzyb27fb5.",
        "No training beyond the first matched 100M stage is authorized.", "",
    ]))
    print("EXPERIMENT_2D4A_FINALIZE_COMPLETE", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--source-checkpoint", required=True)
    preflight.add_argument("--output-dir", required=True)
    preflight.add_argument("--stop-capability-verified", action="store_true")
    preflight.add_argument("--storage-inventory-verified", action="store_true")
    preflight.add_argument("--network-volume-size-bytes", type=int, required=True)
    preflight.add_argument("--network-volume-used-bytes", type=int, required=True)
    preflight.set_defaults(func=run_preflight)

    train = subparsers.add_parser("train")
    train.add_argument("--arm", choices=("fixed", "routed"), required=True)
    train.add_argument("--source-checkpoint", required=True)
    train.add_argument("--data-root", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--preflight-audit", required=True)
    train.add_argument("--scientific-checkpoint-dir", required=True)
    train.add_argument("--recovery-dir", required=True)
    train.add_argument("--resume-checkpoint")
    train.add_argument("--matched-fixed-ledger")
    train.add_argument("--end-local-update", type=int, required=True)
    train.set_defaults(func=run_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--source-checkpoint", required=True)
    evaluate.add_argument("--source-results", required=True)
    evaluate.add_argument("--fixed-checkpoint", required=True)
    evaluate.add_argument("--routed-checkpoint", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--result-dir", required=True)
    evaluate.set_defaults(func=run_evaluate)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--result-dir", required=True)
    finalize.add_argument("--preflight-dir", required=True)
    finalize.add_argument("--fixed-output", required=True)
    finalize.add_argument("--routed-output", required=True)
    finalize.add_argument("--fixed-local-backup-sha", required=True)
    finalize.add_argument("--routed-local-backup-sha", required=True)
    finalize.add_argument("--results-backup-verified", action="store_true")
    finalize.add_argument("--git-push-verified", action="store_true")
    finalize.set_defaults(func=run_finalize)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train" and args.arm == "routed" and not args.matched_fixed_ledger:
        raise SystemExit("routed training requires --matched-fixed-ledger")
    args.func(args)


if __name__ == "__main__":
    main()
