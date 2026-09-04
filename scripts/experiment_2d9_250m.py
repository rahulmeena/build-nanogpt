#!/usr/bin/env python3
"""Resume sealed S/D independently for exactly 286 additional matched updates."""
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
import experiment_2d9 as sealed
from experiment_2d9 import append_dynamic, optimizer_names, GateCollector

EXPERIMENT = "2D9_250M"
BRANCH = "codex/experiment-2d9-matched-continuation-250m"
SCHEMA = "experiment_2d9_checkpoint_v1"
SOURCE_SHA256 = {"S": "676762f2523703167df61f6acda483ae04f7db14a2f918dfd4171911fa5e911b",
                 "D": "c9d859813d1cc2b2df33527d9a07cba32f3901e64febe752cf95a30bb9a73b44"}
SEALED_COMMIT = "482ad55637c2a0adb5c7c268b37c7be243ac15c8"
PRIOR_UPDATES = 191
PRIOR_TARGETS = 100_139_008
PARENT_GLOBAL_UPDATE = 2481
PARENT_TARGETS = 1_300_758_528
LOCAL_UPDATES = 286
TARGETS_PER_UPDATE = 524_288
LOCAL_TARGETS = 149_946_368
FINAL_GLOBAL_UPDATE = 2767
FINAL_TARGETS = 1_450_704_896
FINAL_CHECKPOINT_NAME = "scientific_cumulative_001450704896.pt"
ACCUMULATION = 16
MICROBATCH = 32
PANEL_SEQUENCES = 4096
PANEL_TARGETS = 4_194_304
PANEL_BATCHES = 64
PANEL_SEED = 20260907
DATASET_SHA256 = prior.DATASET_SHA256
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CURSOR_SHA = "d5de64a96c5dd33e9a97ed48ba76cd1d1bc36b6d5bae49aa8673d5cbe6c5e07d"
SOURCE_NEXT_BATCH = "400223a0240720bd6a202a6c9c74a8e2a9c8d80d4e3a5f6db2ea0f51721d4649"
SOURCE_NEXT_STREAM = "0fd6648d5fe2a6d03af41036cb26f7539c96dca41f7cfa343c8035811670e642"


SEALED_RESULT = REPO_ROOT / "results/experiment_2d9_token_conditioned_dynamic_recurrent_gating"

def implementation_sha256():
    paths = [*sorted((REPO_ROOT / "scripts").glob("experiment_2d9*.py")),
             REPO_ROOT / "tests/test_experiment_2d9_250m.py",
             REPO_ROOT / "configs/exp2d9_token_conditioned_dynamic_recurrent_gating_250m.json"]
    return {str(p.relative_to(REPO_ROOT)): sha256(p) for p in paths}


def source_checks(p, arm):
    checks = {
        "schema": p.get("schema") == SCHEMA,
        "experiment": p.get("experiment") == "2D9",
        "arm": p.get("arm") == arm,
        "global_update": p.get("global_update") == PARENT_GLOBAL_UPDATE,
        "cumulative_targets": p.get("cumulative_targets") == PARENT_TARGETS,
        "completed_100m_updates": p.get("local_updates") == PRIOR_UPDATES,
        "completed_100m_targets": p.get("new_targets") == PRIOR_TARGETS,
        "architecture": p.get("architecture_fingerprint") == core.architecture_fingerprint(arm),
        "parameters": p.get("parameter_count") == core.PARAMETER_COUNTS[arm],
        "accumulation": p.get("gradient_accumulation") == ACCUMULATION,
        "targets_per_update": p.get("targets_per_update") == TARGETS_PER_UPDATE,
        "microbatch": p["loader_state"]["batch_size"] == MICROBATCH,
        "sequence_length": p["loader_state"]["sequence_length"] == 1024,
        "cursor_sha": canonical_sha(p["loader_state"]) == SOURCE_CURSOR_SHA,
        "next_batch": p.get("next_global_batch_sha256") == SOURCE_NEXT_BATCH,
        "next_stream": p.get("next_global_batch_stream_sha256") == SOURCE_NEXT_STREAM,
        "rng_inventory": set(p["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "rng_digests": p["rng_digests"] == d5c.rng_digests(p["rng_state"]),
        "scheduler": p["scheduler"] == read_json(SEALED_RESULT / "SOURCE_AND_PARAMETER_AUDIT.json")[arm]["source_metadata"]["scheduler"],
        "optimizer_groups": len(p["optimizer"]["param_groups"]) == (7 if arm == "D" else 6),
        "original_implementation": p["git_implementation_commit"] == read_json(SEALED_RESULT / "FINAL_AUDIT.json")["implementation_commit"],
        "dynamic_steps_191": arm == "S" or all(p["current_optimizer_steps_by_name"][n] == PRIOR_UPDATES for n in core.W_NAMES.values()),
        "learned_dynamic_vectors_retained": arm == "S" or all(bool(p["model"][n].ne(0).any()) for n in core.W_NAMES.values()),
    }
    for group in p["optimizer"]["param_groups"]:
        name = group["name"]
        checks[f"settings_{name}"] = (group["lr"] == (3e-5 if name.startswith("base_") or name == "dynamic_nodecay" else 3e-4)
            and group["weight_decay"] == (0.1 if name == "base_decay" else 0.0)
            and tuple(group["betas"]) == (0.9, 0.95) and group["eps"] == 1e-8)
    return checks


def parent_payload(path, arm):
    source = require_file(path, SOURCE_SHA256[arm], f"sealed 100M {arm} source")
    manifest = read_json(SEALED_RESULT / "CHECKPOINT_MANIFESTS.json")[arm]
    assert manifest["sha256"] == SOURCE_SHA256[arm] and manifest["verification"]["strict_reopen"]["passed"]
    p = base.d0.torch_load(source, mmap=True)
    checks = source_checks(p, arm)
    if not all(checks.values()):
        raise SystemExit(f"sealed source failed {arm}: {checks}")
    return source, p, checks


def state_equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(state_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return left == right


def exact_restore_checks(model, optimizer, loader, p, restore):
    state = optimizer.state_dict()
    return {
        "all_model_tensors_exact": set(model.state_dict()) == set(p["model"]) and all(torch.equal(t.cpu(), p["model"][n]) for n, t in model.state_dict().items()),
        "optimizer_names_exact": optimizer_names(model, optimizer) == p["optimizer_parameter_names"],
        "optimizer_groups_exact": state["param_groups"] == p["optimizer"]["param_groups"],
        "optimizer_state_exact": state_equal(state, p["optimizer"]),
        "individual_steps_exact": d6.optimizer_steps_by_name(model, optimizer) == p["current_optimizer_steps_by_name"],
        "parameter_names_exact": list(dict(model.named_parameters())) == p["parameter_names"],
        "tied_weights": model.base.transformer.wte.weight is model.base.lm_head.weight,
        "master_parameters_fp32": all(v.dtype == torch.float32 for v in model.parameters()),
        "loader_exact": loader.state_dict() == p["loader_state"],
        "next_batch_exact": base.next_batch_hash(loader, ACCUMULATION) == p["next_global_batch_sha256"],
        "next_stream_exact": base.next_stream_hash(loader, ACCUMULATION) == p["next_global_batch_stream_sha256"],
        "rng_exact": not restore or d5c.rng_digests(base.capture_rng()) == p["rng_digests"],
    }


def load_parent_model(path, arm, device, restore=False):
    _, p, checks = parent_payload(path, arm)
    model, optimizer, loader, loaded = load_final_checkpoint(path, device, restore=restore)
    checks.update(exact_restore_checks(model, optimizer, loader, p, restore))
    metadata = source_metadata(p)
    metadata["source_optimizer_steps_by_name"] = d6.optimizer_steps_by_name(model, optimizer)
    metadata["source_dormant_state"] = d6.dormant_state(model, optimizer)
    metadata["source_checkpoint_sha256"] = SOURCE_SHA256[arm]
    if not all(checks.values()):
        raise SystemExit(f"resume state failed {arm}: {checks}")
    del p, loaded
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
    source, payload, parent_checks = parent_payload(args.parent_checkpoint, "S")
    dsource, dpayload, dchecks = parent_payload(args.dynamic_parent_checkpoint, "D")
    assert payload["loader_state"] == dpayload["loader_state"]
    assert payload["next_global_batch_sha256"] == dpayload["next_global_batch_sha256"] == SOURCE_NEXT_BATCH
    assert payload["next_global_batch_stream_sha256"] == dpayload["next_global_batch_stream_sha256"] == SOURCE_NEXT_STREAM
    del dpayload
    loader = loader_from_state(payload["loader_state"])
    rows = []
    previous_chain = canonical_sha(SOURCE_SHA256)
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
        "parents": {"S": {"path": str(source), "sha256": SOURCE_SHA256["S"], "checks": parent_checks},
                    "D": {"path": str(dsource), "sha256": SOURCE_SHA256["D"], "checks": dchecks}},
        "experiment_total_updates": PRIOR_UPDATES + LOCAL_UPDATES,
        "experiment_total_targets": PRIOR_TARGETS + LOCAL_TARGETS,
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
        "local_update": int(local_update), "continuation_local_update": int(local_update),
        "experiment_total_update": PRIOR_UPDATES + int(local_update), "global_update": global_update,
        "experiment_total_targets": PRIOR_TARGETS + int(local_update) * TARGETS_PER_UPDATE,
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

def checkpoint_payload(model, optimizer, loader, metadata, arm, ledger_sha, local_updates=LOCAL_UPDATES):
    rng = base.capture_rng()
    return {
        "schema": SCHEMA, "experiment": EXPERIMENT, "arm": arm,
        "condition": core.CONDITIONS[arm],
        "parent_checkpoint_sha256": SOURCE_SHA256[arm],
        "parent_global_update": PARENT_GLOBAL_UPDATE,
        "parent_cumulative_targets": PARENT_TARGETS,
        "local_updates": local_updates, "continuation_local_updates": local_updates,
        "experiment_total_updates": PRIOR_UPDATES + local_updates,
        "experiment_total_targets": PRIOR_TARGETS + local_updates * TARGETS_PER_UPDATE,
        "global_update": PARENT_GLOBAL_UPDATE + local_updates,
        "new_targets": local_updates * TARGETS_PER_UPDATE,
        "cumulative_targets": PARENT_TARGETS + local_updates * TARGETS_PER_UPDATE,
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

def strict_reopen(path, device, expected_updates=LOCAL_UPDATES):
    model, optimizer, loader, payload = load_final_checkpoint(path, device, restore=False)
    arm = payload["arm"]
    checks = {
        "schema": payload.get("schema") == SCHEMA,
        "parent": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256[arm],
        "global_update": payload.get("global_update") == PARENT_GLOBAL_UPDATE + expected_updates,
        "cumulative_targets": payload.get("cumulative_targets") == PARENT_TARGETS + expected_updates * TARGETS_PER_UPDATE,
        "local_updates": payload.get("local_updates") == expected_updates,
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
        "scheduler": payload["scheduler"] == read_json(SEALED_RESULT / "SOURCE_AND_PARAMETER_AUDIT.json")[arm]["source_metadata"]["scheduler"],
        "experiment_total_updates": payload["experiment_total_updates"] == PRIOR_UPDATES + expected_updates,
        "experiment_total_targets": payload["experiment_total_targets"] == PRIOR_TARGETS + expected_updates * TARGETS_PER_UPDATE,
        "dynamic_steps": arm == "S" or all(d6.optimizer_steps_by_name(model, optimizer)[n] == PRIOR_UPDATES + expected_updates for n in core.W_NAMES.values()),
        "parameter_names": payload["parameter_names"] == list(dict(model.named_parameters())),
        "optimizer_names": payload["optimizer_parameter_names"] == optimizer_names(model, optimizer),
        "dormant_unchanged": d6.dormant_state(model, optimizer) == payload["source_dormant_state"],
        "optimizer_progression": all(step == payload["source_optimizer_steps_by_name"].get(name, 0) + (0 if name == "g_rec_b6" else expected_updates) for name, step in d6.optimizer_steps_by_name(model, optimizer).items()),
    }
    result = {"arm": arm, "checks": checks, "passed": all(checks.values())}
    del model, optimizer, loader, payload
    gc.collect()
    torch.cuda.empty_cache()
    return result

def save_final(path, model, optimizer, loader, metadata, arm, ledger_sha, device, local_updates=LOCAL_UPDATES):
    path = Path(path).resolve()
    if path.exists():
        raise SystemExit(f"refusing checkpoint overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, optimizer, loader, metadata, arm, ledger_sha, local_updates)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    d5c.fsync_path(temporary)
    os.replace(temporary, path)
    d5c.fsync_path(path.parent)
    digest = sha256(path)
    reopen = strict_reopen(path, device, local_updates)
    # Reconstructing a model consumes RNG; a recovery save must not change training.
    base.restore_rng(payload["rng_state"])
    assert d5c.rng_digests(base.capture_rng()) == payload["rng_digests"]
    if not reopen["passed"]:
        raise SystemExit(f"strict final reopen failed: {reopen}")
    verification = {
        "checkpoint": str(path), "sha256": digest, "bytes": path.stat().st_size,
        "arm": arm, "global_update": PARENT_GLOBAL_UPDATE + local_updates,
        "local_updates": local_updates, "experiment_total_updates": PRIOR_UPDATES + local_updates,
        "cumulative_targets": PARENT_TARGETS + local_updates * TARGETS_PER_UPDATE,
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
    if log_path.exists() and not args.resume_checkpoint:
        raise SystemExit("existing arm requires an explicit verified complete-state checkpoint")
    preflight = read_json(args.preflight_audit)
    continuation = read_json(args.continuation_manifest)
    rows = load_rows(args.continuation_ledger)
    code_checks = {
        "preflight_authorized": preflight.get("authorized") is True,
        "git_commit": preflight.get("git_commit") == git("rev-parse", "HEAD"),
        "implementation": preflight.get("implementation_sha256") == implementation_sha256(),
        "parent": sha256(args.parent_checkpoint) == SOURCE_SHA256[arm],
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
    start_update = 0
    if args.resume_checkpoint:
        path = Path(args.resume_checkpoint).resolve()
        record = read_json(path.with_suffix(path.suffix + ".verification.json"))
        assert sha256(path) == record["sha256"] and record["strict_reopen"]["passed"]
        start_update = record["local_updates"]
        assert 0 < start_update <= LOCAL_UPDATES
        assert strict_reopen(path, device, start_update)["passed"]
        del model, optimizer, loader
        gc.collect(); torch.cuda.empty_cache()
        model, optimizer, loader, recovery = load_final_checkpoint(path, device, restore=True)
        assert recovery["arm"] == arm and recovery["git_implementation_commit"] == git("rev-parse", "HEAD")
        assert recovery["continuation_ledger_sha256"] == continuation["ledger_sha256"]
        assert recovery["source_optimizer_steps_by_name"] == source_steps
        assert all(exact_restore_checks(model, optimizer, loader, recovery, True).values())
        assert loader.state_dict() == rows[start_update-1]["end_cursor"]
        metrics = load_rows(log_path)
        assert len(metrics) >= start_update
        for number, row in enumerate(metrics[:start_update], 1):
            assert row["local_update"] == number and row["batch_sha256"] == rows[number-1]["logical_global_batch_sha256"]
        if len(metrics) > start_update:
            durable_text(output / f"UNSEALED_LOG_BEFORE_RECOVERY_{time.time_ns()}.jsonl", log_path.read_text())
            durable_text(log_path, "".join(json.dumps(r) + "\n" for r in metrics[:start_update]))
        durable_json(output / f"RECOVERY_{arm}_{start_update}.json", {
            "checkpoint": str(path), "sha256": record["sha256"], "resumed_after_continuation_update": start_update,
            "model_optimizer_rng_loader_restored": True, "discarded_unsealed_log_rows_preserved": max(0, len(metrics)-start_update)})
        del recovery
    for local_update in range(start_update + 1, LOCAL_UPDATES + 1):
        expected = rows[local_update - 1]
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
        if local_update == 144:
            recovery_path = Path(args.checkpoint_dir) / arm / "recovery_continuation_0144.pt"
            save_final(recovery_path, model, optimizer, loader, metadata, arm, continuation["ledger_sha256"], device, local_update)
    checkpoint_path = Path(args.checkpoint_dir) / arm / FINAL_CHECKPOINT_NAME
    if start_update == LOCAL_UPDATES:
        assert checkpoint_path.resolve() == Path(args.resume_checkpoint).resolve()
        verification = read_json(checkpoint_path.with_suffix(checkpoint_path.suffix + ".verification.json"))
    else:
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
        "final_checkpoint_name": checkpoint_path.name == FINAL_CHECKPOINT_NAME,
        "experiment_total_budget": metrics[-1]["experiment_total_update"] == 477 and metrics[-1]["experiment_total_targets"] == 250085376,
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
        "panel_name": "fresh disjoint 2D9 250M matched panel",
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
        "sealed_100m_panel_excluded": all(tuple(span) in excluded_spans for span in read_json(SEALED_RESULT / "EVALUATION_PANEL_MANIFEST.json")["canonical_target_spans_half_open"]),
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
    evidence = read_json(args.targeted_test_evidence)
    assert evidence["returncode"] == 0
    assert evidence["test_sha256"] == sha256(REPO_ROOT / "tests/test_experiment_2d9_250m.py")
    assert evidence["driver_sha256"] == sha256(Path(__file__))
    durable_json(output / "TARGETED_RESUME_TESTS.json", evidence)
    inherited = read_json(SEALED_RESULT / "FINAL_AUDIT.json")
    kernel_checks = {}
    for name in ("experiment_2d3a_core.py", "experiment_2d6_core.py", "experiment_2d7_core.py", "experiment_2d9_core.py"):
        relative = "scripts/" + name
        sealed_bytes = subprocess.check_output(["git", "show", SEALED_COMMIT + ":" + relative], cwd=REPO_ROOT)
        kernel_checks[relative] = hashlib.sha256(sealed_bytes).hexdigest() == sha256(REPO_ROOT / relative)
    assert inherited["passed"] and all(kernel_checks.values())
    audits, roundtrips = {}, {}
    for arm, source in (("S", args.parent_checkpoint), ("D", args.dynamic_parent_checkpoint)):
        model, optimizer, loader, metadata, checks = load_parent_model(source, arm, device, restore=True)
        audits[arm] = {"checks": checks, "source_checkpoint_sha256": SOURCE_SHA256[arm],
            "source_metadata": metadata,
            "parameter_inventory": [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)} for n, p in model.named_parameters()],
            "optimizer_parameter_names": optimizer_names(model, optimizer),
            "optimizer_groups": [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups],
            "source_gate_parameters": {f"B{b+1}": {"g0": float(model.gate_parameter(b).detach()),
                "w_norm": float(getattr(model, n).detach().norm()) if arm == "D" else 0.0} for b, n in core.W_NAMES.items()}}
        # A full source round-trip, without a scientific or disposable training step.
        # The small resume tests already compare the next optimizer update exactly.
        with tempfile.TemporaryDirectory(prefix="exp2d9-250m-roundtrip-") as directory:
            path = Path(directory) / "disposable_state_roundtrip.pt"
            verification = save_final(path, model, optimizer, loader, metadata, arm, "DISPOSABLE_ROUNDTRIP", device, 0)
            reopened, opt, restored_loader, payload = load_final_checkpoint(path, device, restore=True)
            checks2 = exact_restore_checks(reopened, opt, restored_loader, payload, True)
            checks2["source_model_still_exact"] = state_equal(model.state_dict(), reopened.state_dict())
            checks2["source_optimizer_still_exact"] = state_equal(optimizer.state_dict(), opt.state_dict())
            assert all(checks2.values())
            roundtrips[arm] = {"passed": True, "strict_reopen": verification["strict_reopen"], "checks": checks2,
                               "scientific_updates_performed": 0}
            del reopened, opt, restored_loader, payload
        del model, optimizer, loader
        gc.collect(); torch.cuda.empty_cache()
    assert audits["S"]["source_metadata"]["loader_state"] == audits["D"]["source_metadata"]["loader_state"]
    audit = {"experiment": EXPERIMENT, "git_commit": git("rev-parse", "HEAD"),
        "implementation_sha256": implementation_sha256(), "targeted_resume_tests_passed": True,
        "inherited_100m_audit_sha256": sha256(SEALED_RESULT / "FINAL_AUDIT.json"),
        "inherited_implementation_evidence_passed": inherited["passed"], "unchanged_kernel_checks": kernel_checks,
        "source_roundtrips": roundtrips, "source_checks": {a: r["checks"] for a, r in audits.items()},
        "scientific_state_independently_reloaded": True, "authorized": True,
        "torch_version": torch.__version__, "numpy_version": np.__version__}
    durable_json(output / "SOURCE_AND_PARAMETER_AUDIT.json", audits)
    durable_json(output / "PREFLIGHT_AUDIT.json", audit)
    print("EXPERIMENT_2D9_250M_PREFLIGHT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("continuation", "preflight", "train", "panel", "evaluate"):
        p = sub.add_parser(command)
        if command in ("continuation", "preflight", "train"):
            p.add_argument("--parent-checkpoint", required=True)
        if command in ("continuation", "preflight"):
            p.add_argument("--dynamic-parent-checkpoint", required=True)
        if command == "preflight":
            p.add_argument("--targeted-test-evidence", required=True)
        if command != "evaluate":
            p.add_argument("--output-dir", required=True)
        if command == "train":
            p.add_argument("--resume-checkpoint")
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
    main()
