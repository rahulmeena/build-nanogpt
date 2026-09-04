#!/usr/bin/env python3
"""Bounded 2D9 driver. Reuses the sealed O1 kernels, loader and state helpers."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from torch.nn import functional as F
import experiment_2d3a as base
import experiment_2d5c as d5c
import experiment_2d6 as d6
import experiment_2d7 as prior
import experiment_2d7_core as o1_core
import experiment_2d9_core as core

EXPERIMENT = "2D9"
BRANCH = "codex/experiment-2d9-token-conditioned-recurrent-gating"
SCHEMA = "experiment_2d9_checkpoint_v1"
PARENT_SHA256 = "c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6"
PARENT_GLOBAL_UPDATE = 2290
PARENT_TARGETS = 1_200_619_520
LOCAL_UPDATES = 191
TARGETS_PER_UPDATE = 524_288
LOCAL_TARGETS = 100_139_008
FINAL_GLOBAL_UPDATE = 2481
FINAL_TARGETS = 1_300_758_528
FINAL_CHECKPOINT_NAME = "scientific_cumulative_001300758528.pt"
ACCUMULATION = 16
MICROBATCH = 32
PANEL_SEQUENCES = 4096
PANEL_TARGETS = 4_194_304
PANEL_BATCHES = 64
PANEL_SEED = 20260905
DATASET_SHA256 = prior.DATASET_SHA256
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CURSOR_SHA = "682abcbcc8db8886274ccbb604927683af38d010ecae31b1494987513ae982d3"
SOURCE_NEXT_BATCH = "a021ce09f7a25b6617632e2a76da1acb0980ac1dda888df5f1c8eb65b3939fbe"
SOURCE_NEXT_STREAM = "7918ea7e6f979b8e49fca89c60dd68ace44b768854d34bc96dd751bee07b2567"


def implementation_sha256():
    paths = [*sorted((REPO_ROOT / "scripts").glob("experiment_2d9*.py")),
             REPO_ROOT / "tests/test_experiment_2d9.py",
             REPO_ROOT / "configs/exp2d9_token_conditioned_dynamic_recurrent_gating.json"]
    return {str(p.relative_to(REPO_ROOT)): sha256(p) for p in paths}


def optimizer_names(model, optimizer):
    names = {p: n for n, p in model.named_parameters()}
    return [[names[p] for p in group["params"]] for group in optimizer.param_groups]


def append_dynamic(model, optimizer):
    model.enable_dynamic()
    group = next(g for g in optimizer.param_groups if g["name"] == "base_nodecay")
    optimizer.add_param_group({**{k: v for k, v in group.items() if k != "params"},
                               "name": "dynamic_nodecay",
                               "params": [getattr(model, n) for n in core.W_NAMES.values()]})


def parent_payload(path, mmap=True):
    source = require_file(path, PARENT_SHA256, "sealed 2D7 O1 parent")
    p = base.d0.torch_load(source, mmap=mmap)
    checks = {
        "schema": p.get("schema") == "experiment_2d7_checkpoint_v1",
        "experiment": p.get("experiment") == "2D7",
        "arm": p.get("arm") == "O",
        "global_update": p.get("global_update") == PARENT_GLOBAL_UPDATE,
        "cumulative_targets": p.get("cumulative_targets") == PARENT_TARGETS,
        "parameters": p.get("parameter_count") == core.PARAMETER_COUNTS["S"],
        "architecture": p.get("architecture_fingerprint") == o1_core.architecture_fingerprint("O"),
        "accumulation": p.get("gradient_accumulation") == ACCUMULATION,
        "microbatch": p.get("loader_state", {}).get("batch_size") == MICROBATCH,
        "targets_per_update": p.get("targets_per_update") == TARGETS_PER_UPDATE,
        "cursor_sha": canonical_sha(p["loader_state"]) == SOURCE_CURSOR_SHA,
        "next_batch": p.get("next_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "next_stream": p.get("next_global_batch_stream_sha256") == SOURCE_NEXT_STREAM,
        "rng": set(p.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "scheduler": p.get("scheduler", {}).get("warmup_restarted") is False,
        "old_optimizer_groups": len(p["optimizer"]["param_groups"]) == 6,
    }
    for group in p["optimizer"]["param_groups"]:
        name = group["name"]
        checks[f"settings_{name}"] = (group["lr"] == (3e-5 if name.startswith("base_") else 3e-4)
            and group["weight_decay"] == (0.1 if name == "base_decay" else 0.0)
            and tuple(group["betas"]) == (0.9, 0.95) and group["eps"] == 1e-8)
    if not all(checks.values()):
        raise SystemExit(f"source audit failed: {checks}")
    return source, p, checks


def load_parent_model(path, arm, device, restore=False):
    _, p, checks = parent_payload(path)
    metadata = source_metadata(p)
    _, foundation = base.instantiate_base(device)
    model = core.DynamicGatingGPT(foundation, "S").to(device)
    model.load_state_dict(p["model"], strict=True)
    optimizer = base.configure_optimizer(model, device.type)
    optimizer.load_state_dict(p["optimizer"])
    metadata["source_optimizer_steps_by_name"] = d6.optimizer_steps_by_name(model, optimizer)
    metadata["source_dormant_state"] = d6.dormant_state(model, optimizer)
    # Prove exact old state before altering the inventory. No resized positional load.
    old_names = optimizer_names(model, optimizer)
    checks["all_old_tensors_exact"] = all(torch.equal(t.cpu(), p["model"][n]) for n, t in model.state_dict().items())
    state = optimizer.state_dict()
    checks["old_optimizer_groups_exact"] = state["param_groups"] == p["optimizer"]["param_groups"]
    checks["old_optimizer_states_exact"] = all(
        all(torch.equal(v.cpu(), p["optimizer"]["state"][i][k].cpu()) if torch.is_tensor(v)
            else v == p["optimizer"]["state"][i][k] for k, v in row.items())
        for i, row in state["state"].items())
    if arm == "D":
        append_dynamic(model, optimizer)
    elif arm != "S":
        raise ValueError(arm)
    checks["old_optimizer_names_preserved"] = optimizer_names(model, optimizer)[:6] == old_names
    checks["only_expected_new_parameters"] = set(model.state_dict()) - set(p["model"]) == (set(core.W_NAMES.values()) if arm == "D" else set())
    checks["parameter_count"] = sum(v.numel() for v in model.parameters()) == core.PARAMETER_COUNTS[arm]
    checks["tied_weights"] = model.base.transformer.wte.weight is model.base.lm_head.weight
    checks["master_parameters_fp32"] = all(v.dtype == torch.float32 for v in model.parameters())
    checks["new_zero_vectors_fresh_state"] = arm == "S" or all(
        bool(getattr(model, n).eq(0).all()) and getattr(model, n) not in optimizer.state for n in core.W_NAMES.values())
    loader = loader_from_state(p["loader_state"])
    checks["loader_exact"] = loader.state_dict() == p["loader_state"]
    checks["next_batch_exact"] = base.next_batch_hash(loader, ACCUMULATION) == SOURCE_NEXT_BATCH
    checks["next_stream_exact"] = base.next_stream_hash(loader, ACCUMULATION) == SOURCE_NEXT_STREAM
    if restore:
        base.restore_rng(p["rng_state"])
    checks["rng_exact"] = not restore or d5c.rng_digests(base.capture_rng()) == metadata["rng_digests"]
    if not all(checks.values()):
        raise SystemExit(f"initialization failed {arm}: {checks}")
    del p, state
    gc.collect()
    return model, optimizer, loader, metadata, checks


def load_final_checkpoint(path, device, restore=False):
    p = base.d0.torch_load(Path(path).resolve(), mmap=True)
    if p.get("schema") != SCHEMA or p.get("arm") not in core.PARAMETER_COUNTS:
        raise SystemExit("not a 2D9 checkpoint")
    _, foundation = base.instantiate_base(device)
    model = core.DynamicGatingGPT(foundation, "S").to(device)
    optimizer = base.configure_optimizer(model, device.type)
    if p["arm"] == "D":
        append_dynamic(model, optimizer)
    model.load_state_dict(p["model"], strict=True)
    if p["optimizer_parameter_names"] != optimizer_names(model, optimizer):
        raise SystemExit("optimizer name mapping mismatch")
    optimizer.load_state_dict(p["optimizer"])
    loader = loader_from_state(p["loader_state"])
    if restore:
        base.restore_rng(p["rng_state"])
    return model, optimizer, loader, p


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()

def require_branch(clean=False):
    current = git("branch", "--show-current")
    if current != BRANCH:
        raise SystemExit(f"expected branch {BRANCH}, found {current}")
    if clean and git("status", "--porcelain"):
        raise SystemExit("2D9 requires a clean worktree")

def read_json(path):
    return json.loads(Path(path).read_text())

def durable_json(path, value):
    d5c.durable_json(Path(path), value)

def durable_text(path, value):
    d5c.durable_text(Path(path), value)

def append_jsonl(path, value):
    d5c.append_jsonl(Path(path), value)

def sha256(path):
    return d5c.sha256(Path(path))

def canonical_sha(value):
    return d5c.canonical_sha(value)

def require_file(path, expected=None, label="file"):
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    observed = sha256(path)
    if expected is not None and observed != expected:
        raise SystemExit(f"{label} SHA mismatch: {observed}")
    return path

def loader_from_state(state):
    return base.d1.ExplicitShardLoader(state["shards"], state["batch_size"], base.T, state=state)

def build_continuation(args):
    require_branch(clean=True)
    source, payload, parent_checks = parent_payload(args.parent_checkpoint, mmap=True)
    loader = loader_from_state(payload["loader_state"])
    rows = []
    previous_chain = PARENT_SHA256
    for local_update in range(1, LOCAL_UPDATES + 1):
        global_update = PARENT_GLOBAL_UPDATE + local_update
        start_cursor = loader.state_dict()
        batch_hash = base.next_batch_hash(loader, ACCUMULATION)
        stream_hash = base.next_stream_hash(loader, ACCUMULATION)
        for _ in range(ACCUMULATION):
            loader.next_batch()
        row = {
            "local_update": local_update,
            "global_update": global_update,
            "start_cursor": start_cursor,
            "end_cursor": loader.state_dict(),
            "logical_global_batch_sha256": batch_hash,
            "logical_global_stream_sha256": stream_hash,
            "pass_count": base.pass_count(global_update),
            "target_count": TARGETS_PER_UPDATE,
            "previous_chain_sha256": previous_chain,
        }
        row["chain_sha256"] = canonical_sha(row)
        previous_chain = row["chain_sha256"]
        rows.append(row)
    terminal_batch = base.next_batch_hash(loader, ACCUMULATION)
    terminal_stream = base.next_stream_hash(loader, ACCUMULATION)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "MATCHED_BATCH_LEDGER.jsonl"
    if ledger.exists():
        raise SystemExit("refusing to overwrite continuation ledger")
    for row in rows:
        append_jsonl(ledger, row)
    manifest = {
        "experiment": EXPERIMENT,
        "parent": {"path": str(source), "sha256": PARENT_SHA256, "checks": parent_checks},
        "starting_global_update": PARENT_GLOBAL_UPDATE,
        "first_global_update": PARENT_GLOBAL_UPDATE + 1,
        "updates": LOCAL_UPDATES,
        "targets_per_update": TARGETS_PER_UPDATE,
        "new_targets": LOCAL_TARGETS,
        "final_global_update": FINAL_GLOBAL_UPDATE,
        "final_cumulative_targets": FINAL_TARGETS,
        "gradient_accumulation": ACCUMULATION,
        "microbatch": MICROBATCH,
        "first_loader_cursor": rows[0]["start_cursor"],
        "first_global_batch_sha256": rows[0]["logical_global_batch_sha256"],
        "first_stream_sha256": rows[0]["logical_global_stream_sha256"],
        "final_loader_cursor": loader.state_dict(),
        "next_global_batch_sha256": terminal_batch,
        "next_stream_sha256": terminal_stream,
        "terminal_chain_sha256": previous_chain,
        "ledger_sha256": sha256(ledger),
        "rows": len(rows),
        "two_pass_updates": sum(row["pass_count"] == 2 for row in rows),
        "three_pass_updates": sum(row["pass_count"] == 3 for row in rows),
    }
    manifest["passed"] = (
        len(rows) == LOCAL_UPDATES
        and rows[0]["global_update"] == PARENT_GLOBAL_UPDATE + 1
        and rows[-1]["global_update"] == FINAL_GLOBAL_UPDATE
        and LOCAL_UPDATES * TARGETS_PER_UPDATE == LOCAL_TARGETS
    )
    durable_json(output / "CONTINUATION_MANIFEST.json", manifest)
    if not manifest["passed"]:
        raise SystemExit("continuation manifest failed")
    print(f"EXPERIMENT_2D9_CONTINUATION_FROZEN {manifest['ledger_sha256']}", flush=True)

def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

def tensor_sha(tensor):
    return d5c.tensor_sha256(tensor)

def source_metadata(payload):
    return {
        "schema": payload["schema"],
        "global_update": payload["global_update"],
        "cumulative_targets": payload["cumulative_targets"],
        "gradient_accumulation": payload["gradient_accumulation"],
        "scheduler": copy.deepcopy(payload["scheduler"]),
        "rng_digests": d5c.rng_digests(payload["rng_state"]),
        "loader_state": copy.deepcopy(payload["loader_state"]),
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "next_stream_sha256": payload["next_global_batch_stream_sha256"],
        "gate_sha256": {
            name: tensor_sha(payload["model"][name])
            for name in ("g_rec", "g_rec_b3", "g_rec_b5", "g_rec_b6")
        },
    }

def active_gradient_report(model):
    groups = {
        "base": [parameter for name, parameter in model.named_parameters() if name not in {"g_rec", "g_rec_b3", "g_rec_b5", "g_rec_b6"}],
        "b1_gate": [model.g_rec], "b3_gate": [model.g_rec_b3], "b5_gate": [model.g_rec_b5],
        **({n: [getattr(model, n)] for n in core.W_NAMES.values()} if model.arm == "D" else {}),
    }
    result = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        squared = sum((gradient.float().square().sum() for gradient in gradients), torch.tensor(0.0, device=base.model_device(model)))
        result[name] = {
            "gradient_tensors": len(gradients),
            "finite": bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients),
            "nonzero": bool(gradients) and bool(squared.gt(0).item()),
            "norm": float(squared.sqrt().item()),
        }
    return result

def train_update(model, optimizer, loader, local_update, device):
    global_update = PARENT_GLOBAL_UPDATE + int(local_update)
    passes = base.pass_count(global_update)
    before_steps = d6.optimizer_steps_by_name(model, optimizer)
    before_dormant = d6.dormant_state(model, optimizer)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = [0.0] * passes
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(ACCUMULATION):
        cpu_x, cpu_y = loader.next_batch()
        x, y = cpu_x.to(device, non_blocking=True), cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model.forward_multi_pass(x, targets=y, num_passes=passes, activation_checkpointing=True)
            loss = result["loss"] / ACCUMULATION
        for index, current in enumerate(result["pass_losses"]):
            totals[index] += float(current.detach().float())
        loss.backward()
        del cpu_x, cpu_y, x, y, result, loss
    gradients = active_gradient_report(model)
    if not all(row["finite"] and row["nonzero"] for row in gradients.values()):
        raise SystemExit(f"missing/nonfinite active gradients: {gradients}")
    if model.g_rec_b6.grad is not None:
        raise SystemExit("dormant B6 gate received a gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
    if not torch.isfinite(norm):
        raise SystemExit("nonfinite gradient norm")
    optimizer.step()
    after_steps = d6.optimizer_steps_by_name(model, optimizer)
    after_dormant = d6.dormant_state(model, optimizer)
    active = set(after_steps) - {"g_rec_b6"}
    step_checks = {
        "active_incremented_once": all(after_steps[name] == before_steps.get(name, 0) + 1 for name in active),
        "dormant_step_unchanged": after_steps["g_rec_b6"] == before_steps["g_rec_b6"],
        "dormant_parameter_unchanged": after_dormant["parameter_sha256"] == before_dormant["parameter_sha256"],
        "dormant_optimizer_unchanged": after_dormant["optimizer_state_sha256"] == before_dormant["optimizer_state_sha256"],
    }
    if not all(step_checks.values()) or not base.model_finite(model) or not base.optimizer_finite(optimizer):
        raise SystemExit(f"optimizer state failure: {step_checks}")
    elapsed = time.monotonic() - started
    return {
        "local_update": int(local_update), "global_update": global_update,
        "pass_count": passes, "target_count": TARGETS_PER_UPDATE,
        "new_targets": int(local_update) * TARGETS_PER_UPDATE,
        "cumulative_targets": PARENT_TARGETS + int(local_update) * TARGETS_PER_UPDATE,
        "pass_losses": [value / ACCUMULATION for value in totals],
        "ce": totals[-1] / ACCUMULATION,
        "gradient_norm_before_clip": float(norm.detach().float()),
        "active_gradient_groups": gradients, "optimizer_checks": step_checks,
        "wall_seconds": elapsed, "targets_per_second": TARGETS_PER_UPDATE / elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }

def checkpoint_payload(model, optimizer, loader, metadata, arm, ledger_sha):
    rng = base.capture_rng()
    return {
        "schema": SCHEMA, "experiment": EXPERIMENT, "arm": arm,
        "condition": core.CONDITIONS[arm],
        "parent_checkpoint_sha256": PARENT_SHA256,
        "parent_global_update": PARENT_GLOBAL_UPDATE,
        "parent_cumulative_targets": PARENT_TARGETS,
        "local_updates": LOCAL_UPDATES, "global_update": FINAL_GLOBAL_UPDATE,
        "new_targets": LOCAL_TARGETS, "cumulative_targets": FINAL_TARGETS,
        "targets_per_update": TARGETS_PER_UPDATE,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": copy.deepcopy(metadata["scheduler"]),
        "loader_state": loader.state_dict(), "loader_states": [loader.state_dict()],
        "rng_state": rng, "rng_digests": d5c.rng_digests(rng),
        "gradient_accumulation": ACCUMULATION,
        "next_global_batch_sha256": base.next_batch_hash(loader, ACCUMULATION),
        "next_global_batch_stream_sha256": base.next_stream_hash(loader, ACCUMULATION),
        "architecture_manifest": core.architecture_manifest(arm),
        "architecture_fingerprint": core.architecture_fingerprint(arm),
        "continuation_ledger_sha256": ledger_sha,
        "parameter_count": core.PARAMETER_COUNTS[arm],
        "optimizer_group_definitions": [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups],
        "current_optimizer_steps_by_name": d6.optimizer_steps_by_name(model, optimizer),
        "git_implementation_commit": git("rev-parse", "HEAD"),
        "parameter_names": list(dict(model.named_parameters())),
        "optimizer_parameter_names": optimizer_names(model, optimizer),
        "source_optimizer_steps_by_name": metadata["source_optimizer_steps_by_name"],
        "source_dormant_state": metadata["source_dormant_state"],
        "saved_process_id": os.getpid(), "saved_at_unix": time.time(),
    }

def strict_reopen(path, device):
    model, optimizer, loader, payload = load_final_checkpoint(path, device, restore=False)
    arm = payload["arm"]
    checks = {
        "schema": payload.get("schema") == SCHEMA,
        "parent": payload.get("parent_checkpoint_sha256") == PARENT_SHA256,
        "global_update": payload.get("global_update") == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": payload.get("cumulative_targets") == FINAL_TARGETS,
        "local_updates": payload.get("local_updates") == LOCAL_UPDATES,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()) == core.PARAMETER_COUNTS[arm],
        "architecture": payload.get("architecture_fingerprint") == core.architecture_fingerprint(arm),
        "model_finite": base.model_finite(model), "optimizer_finite": base.optimizer_finite(optimizer),
        "model_snapshot_exact": all(torch.equal(v.cpu(), payload["model"][n]) for n, v in model.state_dict().items()),
        "optimizer_snapshot_exact": all(
            all(torch.equal(v.cpu(), payload["optimizer"]["state"][i][k].cpu()) if torch.is_tensor(v)
                else v == payload["optimizer"]["state"][i][k] for k, v in row.items())
            for i, row in optimizer.state_dict()["state"].items()),
        "optimizer_groups_exact": optimizer.state_dict()["param_groups"] == payload["optimizer"]["param_groups"],
        "optimizer_inventory_complete": set(d6.optimizer_steps_by_name(model, optimizer)) == set(dict(model.named_parameters())),
        "loader_cursor": loader.state_dict() == payload["loader_state"],
        "next_batch": base.next_batch_hash(loader, ACCUMULATION) == payload["next_global_batch_sha256"],
        "next_stream": base.next_stream_hash(loader, ACCUMULATION) == payload["next_global_batch_stream_sha256"],
        "rng": payload.get("rng_digests") == d5c.rng_digests(payload["rng_state"]),
        "scheduler": "scheduler" in payload,
        "parameter_names": payload["parameter_names"] == list(dict(model.named_parameters())),
        "optimizer_names": payload["optimizer_parameter_names"] == optimizer_names(model, optimizer),
        "dormant_unchanged": d6.dormant_state(model, optimizer) == payload["source_dormant_state"],
        "optimizer_progression": all(step == payload["source_optimizer_steps_by_name"].get(name, 0) + (0 if name == "g_rec_b6" else LOCAL_UPDATES) for name, step in d6.optimizer_steps_by_name(model, optimizer).items()),
    }
    result = {"arm": arm, "checks": checks, "passed": all(checks.values())}
    del model, optimizer, loader, payload
    gc.collect()
    torch.cuda.empty_cache()
    return result

def save_final(path, model, optimizer, loader, metadata, arm, ledger_sha, device):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, optimizer, loader, metadata, arm, ledger_sha)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    d5c.fsync_path(temporary)
    os.replace(temporary, path)
    d5c.fsync_path(path.parent)
    digest = sha256(path)
    reopen = strict_reopen(path, device)
    if not reopen["passed"]:
        raise SystemExit(f"strict final reopen failed: {reopen}")
    verification = {
        "checkpoint": str(path), "sha256": digest, "bytes": path.stat().st_size,
        "arm": arm, "global_update": FINAL_GLOBAL_UPDATE,
        "cumulative_targets": FINAL_TARGETS,
        "final_loader_cursor": payload["loader_state"],
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "next_stream_sha256": payload["next_global_batch_stream_sha256"],
        "strict_reopen": reopen,
    }
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), verification)
    return verification

def run_train(args):
    require_branch(clean=True)
    arm = args.arm.upper()
    if arm not in core.PARAMETER_COUNTS:
        raise SystemExit("arm must be S or D")
    device = base.require_a100()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"TRAINING_{arm}.jsonl"
    if log_path.exists():
        raise SystemExit("refusing to resume or overwrite an official arm; restart into a fresh directory")
    preflight = read_json(args.preflight_audit)
    continuation = read_json(args.continuation_manifest)
    rows = load_rows(args.continuation_ledger)
    code_checks = {
        "preflight_authorized": preflight.get("authorized") is True,
        "git_commit": preflight.get("git_commit") == git("rev-parse", "HEAD"),
        "implementation": preflight.get("implementation_sha256") == implementation_sha256(),
        "parent": sha256(args.parent_checkpoint) == PARENT_SHA256,
        "ledger": sha256(args.continuation_ledger) == continuation.get("ledger_sha256"),
        "rows": len(rows) == LOCAL_UPDATES,
    }
    if not all(code_checks.values()):
        raise SystemExit(f"training binding failed: {code_checks}")
    model, optimizer, loader, metadata, construction = load_parent_model(args.parent_checkpoint, arm, device, restore=True)
    source_steps = d6.optimizer_steps_by_name(model, optimizer)
    first = {
        "cursor": loader.state_dict() == rows[0]["start_cursor"],
        "batch": base.next_batch_hash(loader, ACCUMULATION) == rows[0]["logical_global_batch_sha256"],
        "stream": base.next_stream_hash(loader, ACCUMULATION) == rows[0]["logical_global_stream_sha256"],
        "global_update": rows[0]["global_update"] == PARENT_GLOBAL_UPDATE + 1,
    }
    if not all(first.values()):
        raise SystemExit(f"first matched batch failure for {arm}: {first}")
    for local_update, expected in enumerate(rows, 1):
        before_batch = base.next_batch_hash(loader, ACCUMULATION)
        before_stream = base.next_stream_hash(loader, ACCUMULATION)
        invariants = {
            "local_update": expected["local_update"] == local_update,
            "global_update": expected["global_update"] == PARENT_GLOBAL_UPDATE + local_update,
            "cursor": loader.state_dict() == expected["start_cursor"],
            "batch": before_batch == expected["logical_global_batch_sha256"],
            "stream": before_stream == expected["logical_global_stream_sha256"],
            "pass": expected["pass_count"] == base.pass_count(PARENT_GLOBAL_UPDATE + local_update),
            "targets": expected["target_count"] == TARGETS_PER_UPDATE,
        }
        if not all(invariants.values()):
            raise SystemExit(f"replay invariant failed at {arm}/{local_update}: {invariants}")
        row = train_update(model, optimizer, loader, local_update, device)
        row.update({
            "arm": arm, "batch_sha256": before_batch, "stream_sha256": before_stream,
            "end_cursor_exact": loader.state_dict() == expected["end_cursor"],
            "pre_forward_invariants": invariants, "ce_only": True,
            "lr": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
        })
        if not row["end_cursor_exact"]:
            raise SystemExit(f"post-update cursor mismatch at {arm}/{local_update}")
        append_jsonl(log_path, row)
        durable_json(output / f"HEARTBEAT_{arm}.json", {
            "status": "training", "arm": arm, "local_update": local_update,
            "global_update": row["global_update"], "latest": row,
            "updated_at_unix": time.time(),
        })
        print(f"2D9 {arm} update {local_update}/{LOCAL_UPDATES} CE {row['ce']:.6f}", flush=True)
    checkpoint_path = Path(args.checkpoint_dir) / arm / FINAL_CHECKPOINT_NAME
    verification = save_final(checkpoint_path, model, optimizer, loader, metadata, arm, continuation["ledger_sha256"], device)
    metrics = load_rows(log_path)
    final_steps = d6.optimizer_steps_by_name(model, optimizer)
    active = set(final_steps) - {"g_rec_b6"}
    checks = {
        **{f"code_{key}": value for key, value in code_checks.items()},
        **{f"first_{key}": value for key, value in first.items()},
        "construction": all(construction.values()),
        "updates": len(metrics) == LOCAL_UPDATES and metrics[-1]["global_update"] == FINAL_GLOBAL_UPDATE,
        "targets": metrics[-1]["cumulative_targets"] == FINAL_TARGETS,
        "batches": [row["batch_sha256"] for row in metrics] == [row["logical_global_batch_sha256"] for row in rows],
        "streams": [row["stream_sha256"] for row in metrics] == [row["logical_global_stream_sha256"] for row in rows],
        "passes": [row["pass_count"] for row in metrics] == [row["pass_count"] for row in rows],
        "final_cursor": loader.state_dict() == continuation["final_loader_cursor"] == verification["final_loader_cursor"],
        "final_next_batch": verification["next_global_batch_sha256"] == continuation["next_global_batch_sha256"],
        "final_next_stream": verification["next_stream_sha256"] == continuation["next_stream_sha256"],
        "active_optimizer_progression": all(final_steps[name] == source_steps.get(name, 0) + LOCAL_UPDATES for name in active),
        "dormant_optimizer_unchanged": final_steps["g_rec_b6"] == source_steps["g_rec_b6"],
        "only_final_checkpoint": checkpoint_path.name == FINAL_CHECKPOINT_NAME,
    }
    summary = {
        "schema": "experiment_2d9_training_complete_v1", "experiment": EXPERIMENT,
        "arm": arm, "checks": checks, "final_checkpoint": verification,
        "training_wall_seconds": sum(row["wall_seconds"] for row in metrics),
        "mean_targets_per_second": statistics.fmean(row["targets_per_second"] for row in metrics),
        "final_loader_cursor": verification["final_loader_cursor"],
        "next_global_batch_sha256": verification["next_global_batch_sha256"],
        "next_stream_sha256": verification["next_stream_sha256"],
        "passed": all(checks.values()),
    }
    durable_json(output / f"TRAINING_COMPLETE_{arm}.json", summary)
    if not summary["passed"]:
        raise SystemExit(f"terminal training audit failed for {arm}: {checks}")
    print(f"EXPERIMENT_2D9_TRAINING_COMPLETE {arm}", flush=True)


def run_prepare_panel(args):
    require_branch(clean=True)
    dataset = require_file(args.dataset, DATASET_SHA256, "validation dataset")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokens = np.load(dataset, mmap_mode="r")
    if (output / "EVALUATION_PANEL_MANIFEST.json").exists():
        raise SystemExit("refusing panel reselection")
    historical_spans = set()
    historical_files = []
    for root in (REPO_ROOT / "configs", REPO_ROOT / "results"):
        for path in sorted(root.rglob("*.json")):
            if output == path.parent or output in path.parents:
                continue
            try:
                value = read_json(path)
            except (OSError, ValueError):
                continue
            recovered = set()
            prior._collect_historical_spans(value, recovered)
            before = len(historical_spans)
            historical_spans.update(recovered)
            if recovered:
                historical_files.append({
                    "path": str(path.resolve()), "sha256": sha256(path),
                    "new_spans": len(historical_spans) - before,
                })
    for index in range(128):
        start = index * 64 * base.T
        historical_spans.add((start + 1, start + 64 * base.T + 1))

    def intersects(span):
        return any(max(span[0], old[0]) < min(span[1], old[1]) for old in historical_spans)

    excluded_spans = sorted(historical_spans)
    available = (int(tokens.shape[0]) - 1) // (64 * base.T)
    order = np.random.default_rng(PANEL_SEED).permutation(available)
    selected, identities, sequences, selected_spans = [], [], [], []
    for raw in order:
        index = int(raw)
        start = index * 64 * base.T
        span = (start + 1, start + 64 * base.T + 1)
        if intersects(span):
            continue
        x, y = prior.batch_arrays(tokens, start)
        identity = prior.batch_identity_numpy(x, y)
        rows = prior.sequence_identities(x, y, start, index)
        selected.append(index)
        identities.append(identity)
        sequences.extend(rows)
        selected_spans.append(list(span))
        historical_spans.add(span)
        if len(selected) == PANEL_BATCHES:
            break
    if len(selected) != PANEL_BATCHES:
        raise SystemExit("could not construct one fresh 2D9 panel")
    panel_hash = prior.aggregate_hashes(row["combined_sha256"] for row in identities)
    manifest = {
        "experiment": EXPERIMENT,
        "panel_name": "fresh disjoint 2D9 matched panel",
        "dataset": "edu_fineweb10B/edufineweb_val_000000.npy",
        "dataset_sha256": DATASET_SHA256,
        "dataset_split": "validation",
        "selection_seed": PANEL_SEED,
        "selection_algorithm": "seeded permutation of complete canonical 64x1024 validation batches; reject every recovered historical target span and reserved prefix 0..127; accept first 64",
        "candidate_panels_constructed": 1,
        "checkpoint_losses_inspected_during_selection": False,
        "sealed_before_training_and_scoring": True,
        "training_split_disjoint_by_dataset_split": True,
        "batch_size": 64,
        "sequence_length": base.T,
        "batch_indices_in_evaluation_order": selected,
        "canonical_target_spans_half_open": selected_spans,
        "batch_identities": identities,
        "sequence_identities": sequences,
        "sequence_count": len(sequences),
        "targets_per_sequence": base.T,
        "targets_per_condition": len(sequences) * base.T,
        "panel_sha256": panel_hash,
    }
    checks = {
        "one_panel": manifest["candidate_panels_constructed"] == 1,
        "sealed_before_results": manifest["sealed_before_training_and_scoring"],
        "no_checkpoint_loss_inspection": not manifest["checkpoint_losses_inspected_during_selection"],
        "training_split_disjoint": manifest["training_split_disjoint_by_dataset_split"],
        "sequences": len(sequences) == PANEL_SEQUENCES,
        "targets": len(sequences) * base.T == PANEL_TARGETS,
        "no_historical_overlap": all(not any(max(a, c) < min(b, d) for c, d in excluded_spans) for a, b in selected_spans),
        "recent_panels_recovered": all(any(term in row["path"] for row in historical_files) for term in ("experiment_2d8", "experiment_2d7", "experiment_2d6_fresh_panel_confirmation")),
        "unique_batches": len(set(selected)) == PANEL_BATCHES,
        "unique_sequence_hashes": len({row["combined_sha256"] for row in sequences}) == PANEL_SEQUENCES,
    }
    audit = {
        "schema": "experiment_2d9_panel_disjointness_v1",
        "experiment": EXPERIMENT,
        "panel_sha256": panel_hash,
        "historical_json_files_used": historical_files,
        "recovered_historical_spans_including_reserved_prefix": len(historical_spans) - len(selected_spans),
        "selected_target_spans": selected_spans,
        "excluded_target_spans": excluded_spans,
        "checks": checks,
        "passed": all(checks.values()),
    }
    d5c.durable_json(output / "EVALUATION_PANEL_MANIFEST.json", manifest)
    d5c.durable_json(output / "DISJOINTNESS_AUDIT.json", audit)
    if not audit["passed"]:
        raise SystemExit(f"fresh 2D9 panel audit failed: {checks}")
    print(f"EXPERIMENT_2D9_PANEL_FROZEN {panel_hash}", flush=True)


class GateCollector:
    """One batch on device, one transfer; exact full-panel FP32 scalar arrays."""
    def __init__(self, path):
        self.array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                             shape=(3, 3, PANEL_SEQUENCES, base.T))
        self.pending = {b: [] for b in core.W_NAMES}

    def record(self, block, gate, cast):
        self.pending[block].append(torch.stack((gate.preactivation.detach(),
            gate.coefficient.detach(), cast.detach().squeeze(1).float()), dim=0))

    def finish_batch(self, ordinal):
        assert all(len(rows) == base.T for rows in self.pending.values())
        values = torch.stack([torch.cat(self.pending[b], dim=2).squeeze(-1)
                              for b in core.W_NAMES], dim=0).cpu().numpy()
        assert values.shape == (3, 3, 64, base.T)
        self.array[:, :, ordinal * 64:(ordinal + 1) * 64] = values
        self.array.flush()
        self.pending = {b: [] for b in core.W_NAMES}


def run_evaluate(args):
    require_branch(clean=True)
    device = base.require_a100()
    panel = read_json(args.panel_manifest)
    assert panel["sequence_count"] == PANEL_SEQUENCES and panel["targets_per_condition"] == PANEL_TARGETS
    assert panel["sealed_before_training_and_scoring"] is True
    condition = args.condition
    arm = "S" if condition == "STATIC_REAL" else "D"
    checkpoint = Path(args.checkpoint).resolve()
    verification = read_json(str(checkpoint) + ".verification.json")
    assert verification["strict_reopen"]["passed"] and verification["sha256"] == sha256(checkpoint)
    model, optimizer, loader, p = load_final_checkpoint(checkpoint, device)
    assert p["arm"] == arm and p["global_update"] == FINAL_GLOBAL_UPDATE
    assert p["git_implementation_commit"] == git("rev-parse", "HEAD")
    del optimizer, loader
    model.set_gate_mode("staticized" if condition == "DYNAMIC_STATICIZED" else "real")
    output = Path(args.output_path)
    if output.exists():
        raise SystemExit("refusing fourth, repeated, or overwritten final evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    collector = None
    if condition == "DYNAMIC_REAL":
        scalar_path = output.parent / "GATE_SCALARS.npy"
        if scalar_path.exists():
            raise SystemExit("refusing to overwrite gate scalars")
        collector = GateCollector(scalar_path)
        model.gate_collector = collector
    state = {
        "schema": "experiment_2d9_incremental_losses_v1", "experiment": EXPERIMENT,
        "condition": condition, "arm": arm, "checkpoint_sha256": verification["sha256"],
        "panel_sha256": panel["panel_sha256"], "panel_manifest_sha256": sha256(args.panel_manifest),
        "condition_sha256": canonical_sha({"checkpoint": verification["sha256"], "panel": panel["panel_sha256"], "condition": condition}),
        "sequence_identities": panel["sequence_identities"],
        "batch_indices_in_evaluation_order": panel["batch_indices_in_evaluation_order"],
        "completed_batch_indices": [], "batch_identities": [],
        "per_sequence_nll": [], "per_sequence_ce": [], "targets": 0, "nll_sum": 0.0,
        "precision": "BF16 model; FP32 token CE; FP64 accumulation",
        "mode_reset_per_sequence_batch": True, "status": "running",
        "model_gate_parameters": {f"B{b+1}": {"g0": float(model.gate_parameter(b).detach()),
            "tanh_g0": float(model.gate_parameter(b).detach().tanh()),
            "w_norm": float(getattr(model, n).detach().norm()) if arm == "D" else 0.0}
            for b, n in core.W_NAMES.items()},
    }
    val_path = require_file(base.validation_path(Path(args.data_root)), DATASET_SHA256)
    model.eval()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for ordinal, batch_index in enumerate(panel["batch_indices_in_evaluation_order"]):
            cpu_x, cpu_y = d5c.batch_at_index(val_path, int(batch_index))
            observed = base.batch_identity(cpu_x, cpu_y)
            assert observed == panel["batch_identities"][ordinal]
            x, y = cpu_x.to(device), cpu_y.to(device)
            cache = model.init_incremental_state(64, device=device, dtype=torch.bfloat16)
            nll = torch.zeros(64, device=device, dtype=torch.float64)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for position in range(base.T):
                    logits, cache = model.incremental_step(x[:, position], cache)
                    nll.add_(F.cross_entropy(logits[:, 0].float(), y[:, position], reduction="none").double())
            if collector is not None:
                collector.finish_batch(ordinal)
            row_nll = nll.cpu().numpy()
            state["per_sequence_nll"].extend(row_nll.tolist())
            state["per_sequence_ce"].extend((row_nll / base.T).tolist())
            state["nll_sum"] += float(row_nll.sum())
            state["targets"] += int(y.numel())
            state["completed_batch_indices"].append(int(batch_index))
            state["batch_identities"].append(observed)
            if ordinal == 0:
                state["full_length_cache_audit"] = model.incremental_cache_audit(cache)
                assert state["full_length_cache_audit"]["passed"]
            durable_json(output, state)
            print(f"2D9 {condition} batch {ordinal+1}/{PANEL_BATCHES}", flush=True)
            del cpu_x, cpu_y, x, y, cache, logits, nll
    model.gate_collector = None
    state["wall_seconds"] = time.monotonic() - started
    state["peak_allocated_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    state["checkpoint_file_unchanged"] = sha256(checkpoint) == verification["sha256"]
    state["model_tensors_unchanged"] = all(torch.equal(t.cpu(), p["model"][n]) for n, t in model.state_dict().items())
    state["aggregate_ce"] = state["nll_sum"] / state["targets"]
    state["perplexity"] = math.exp(state["aggregate_ce"])
    state["persistent_state"] = {"logical_bytes_per_sequence": state["full_length_cache_audit"]["logical_payload_bytes"] // 64,
        "physical_bytes_per_sequence": state["full_length_cache_audit"]["actual_unique_storage_bytes"] // 64}
    state["passed"] = (state["targets"] == PANEL_TARGETS and len(state["per_sequence_ce"]) == PANEL_SEQUENCES
        and state["batch_identities"] == panel["batch_identities"] and state["checkpoint_file_unchanged"]
        and state["model_tensors_unchanged"] and all(v == 33_289_728 for v in state["persistent_state"].values()))
    state["status"] = "complete" if state["passed"] else "failed"
    durable_json(output, state)
    if not state["passed"]:
        raise SystemExit(f"final evaluation audit failed: {condition}")
    print(f"EXPERIMENT_2D9_EVALUATION_COMPLETE {condition}", flush=True)


def run_preflight(args):
    require_branch(clean=True)
    device = base.require_a100()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_experiment_2d9.py"],
                          cwd=REPO_ROOT, env={**os.environ, "EXP2D9_TEST_DEVICE": "cuda"}, capture_output=True, text=True)
    durable_text(output / "TARGETED_TESTS.txt", test.stdout + test.stderr)
    if test.returncode:
        raise SystemExit("targeted tests failed")
    audits, smoke = {}, {}
    for arm in ("S", "D"):
        model, optimizer, loader, metadata, checks = load_parent_model(args.parent_checkpoint, arm, device, restore=True)
        audits[arm] = {"checks": checks, "source_metadata": metadata,
            "parameter_inventory": [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)} for n, p in model.named_parameters()],
            "optimizer_parameter_names": optimizer_names(model, optimizer),
            "optimizer_groups": [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]}
        # One full inherited microbatch establishes memory fit and first gradients.
        cpu_x, cpu_y = loader.next_batch()
        x, y = cpu_x.to(device), cpu_y.to(device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        before_steps = d6.optimizer_steps_by_name(model, optimizer)
        dormant = d6.dormant_state(model, optimizer)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.forward_multi_pass(x, targets=y, num_passes=2, activation_checkpointing=True)
        result["loss"].backward()
        gradients = active_gradient_report(model)
        assert all(r["finite"] and r["nonzero"] for r in gradients.values())
        assert model.g_rec_b6.grad is None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        after_steps = d6.optimizer_steps_by_name(model, optimizer)
        assert all(step == before_steps.get(n, 0) + (0 if n == "g_rec_b6" else 1) for n, step in after_steps.items())
        assert d6.dormant_state(model, optimizer) == dormant
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        del x, y, cpu_x, cpu_y, result
        payload = checkpoint_payload(model, optimizer, loader, metadata, arm, "DISPOSABLE")
        payload.update(disposable_smoke=True, local_updates=1, global_update=2291,
                       new_targets=32768, cumulative_targets=PARENT_TARGETS+32768)
        with tempfile.TemporaryDirectory(prefix="exp2d9-smoke-") as directory:
            path = Path(directory) / "smoke.pt"; torch.save(payload, path)
            reopened, opt, _, saved = load_final_checkpoint(path, device, restore=True)
            assert all(torch.equal(v, reopened.state_dict()[n]) for n, v in model.state_dict().items())
            assert optimizer_names(model, optimizer) == optimizer_names(reopened, opt)
            assert all(d6.tensor_state_digest(optimizer.state[p]) == d6.tensor_state_digest(opt.state[dict(reopened.named_parameters())[n]])
                       for n, p in model.named_parameters())
            assert d5c.rng_digests(base.capture_rng()) == saved["rng_digests"]
            del reopened, opt, saved
        smoke[arm] = {"passed": True, "batch_size": 32, "sequence_length": 1024,
                      "gradients": gradients, "peak_allocated_vram_mb": peak,
                      "state_reload_exact": True, "dormant_unchanged": True}
        del model, optimizer, loader, payload
        gc.collect(); torch.cuda.empty_cache()
    audit = {"experiment": EXPERIMENT, "git_commit": git("rev-parse", "HEAD"),
             "implementation_sha256": implementation_sha256(), "targeted_tests_passed": True,
             "disposable_smoke": smoke, "source_checks": {a: r["checks"] for a, r in audits.items()},
             "scientific_state_independently_reloaded": True, "authorized": True,
             "torch_version": torch.__version__, "numpy_version": np.__version__}
    durable_json(output / "SOURCE_AND_PARAMETER_AUDIT.json", audits)
    durable_json(output / "PREFLIGHT_AUDIT.json", audit)
    print("EXPERIMENT_2D9_PREFLIGHT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("continuation", "preflight", "train", "panel", "evaluate"):
        p = sub.add_parser(command)
        if command in ("continuation", "preflight", "train"):
            p.add_argument("--parent-checkpoint", required=True)
        if command != "evaluate":
            p.add_argument("--output-dir", required=True)
        if command == "train":
            p.add_argument("--arm", choices=("S", "D"), required=True)
            p.add_argument("--preflight-audit", required=True)
            p.add_argument("--continuation-manifest", required=True)
            p.add_argument("--continuation-ledger", required=True)
            p.add_argument("--checkpoint-dir", required=True)
        if command == "panel":
            p.add_argument("--dataset", required=True)
        if command == "evaluate":
            p.add_argument("--condition", choices=("STATIC_REAL", "DYNAMIC_REAL", "DYNAMIC_STATICIZED"), required=True)
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--panel-manifest", required=True)
            p.add_argument("--data-root", required=True)
            p.add_argument("--output-path", required=True)
    args = parser.parse_args()
    {"continuation": build_continuation, "preflight": run_preflight,
     "train": run_train, "panel": run_prepare_panel, "evaluate": run_evaluate}[args.command](args)


if __name__ == "__main__":
    import sys
    main()
