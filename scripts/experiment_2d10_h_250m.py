#!/usr/bin/env python3
"""Bounded 2D10 driver. Reuses the sealed O1 kernels, loader and state helpers."""
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
import experiment_2d10_core as core
import experiment_2d9 as dynamic
import experiment_2d10 as sealed
import experiment_2d9_250m as historical
from experiment_2d9_250m import state_equal, exact_restore_checks
from experiment_2d10 import optimizer_names, compatibility_state, retired, active_gradient_report, GateCollector

EXPERIMENT = "2D10_H_250M"
BRANCH = "codex/experiment-2d10-h-250m"
SCHEMA = "experiment_2d10_h_250m_checkpoint_v1"
PARENT_SHA256 = "c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6"
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
PANEL_SEED = 20260912
DATASET_SHA256 = prior.DATASET_SHA256
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CURSOR_SHA = "d5de64a96c5dd33e9a97ed48ba76cd1d1bc36b6d5bae49aa8673d5cbe6c5e07d"
SOURCE_NEXT_BATCH = "400223a0240720bd6a202a6c9c74a8e2a9c8d80d4e3a5f6db2ea0f51721d4649"
SOURCE_NEXT_STREAM = "0fd6648d5fe2a6d03af41036cb26f7539c96dca41f7cfa343c8035811670e642"


PRIOR_UPDATES=191
PRIOR_TARGETS=100_139_008
SOURCE_SHA256={"H":"d9c0eea937b4e4726a4963a4586a4c6eb3de8f6a40ac72c4d3959a3f21a2415c"}
CONTROL_SHA256="9714b2e3f53a8c15dfecfed3e9b56c358176c1f9f609bcce7e28c35b8a358a9b"
LEDGER_SHA256="0875d5533a4a8ae753f2e0aec661d81f314609c16d4de053a6ffd48df8e751ec"
SEALED_RESULT=REPO_ROOT/"results/experiment_2d10_retrieval_aware_gating_100m"
HISTORICAL_RESULT=REPO_ROOT/"results/experiment_2d9_token_conditioned_dynamic_recurrent_gating_250m"


def implementation_sha256():
    paths=[REPO_ROOT/"scripts/experiment_2d10_h_250m.py",REPO_ROOT/"scripts/experiment_2d10_core.py",
           REPO_ROOT/"tests/test_experiment_2d10_h_250m.py",REPO_ROOT/"configs/exp2d10_h_250m.json"]
    return {str(p.relative_to(REPO_ROOT)):sha256(p) for p in paths}


def restore_h_payload(p, device, foundation=None):
    """Allocate named tensor slots, strictly load trained H; no Router initializer."""
    if foundation is None:
        _,foundation=base.instantiate_base(device)
    model=core.RetrievalGatingGPT(foundation).to(device)
    model.arm="H"
    routers={}
    for block in core.BLOCKS:
        router=torch.nn.Module()
        for name in ("W1","b1","W2","b2"):
            tensor=p["model"][f"routers.{block}.{name}"]
            router.register_parameter(name,torch.nn.Parameter(torch.empty_like(tensor,device=device)))
        routers[str(block)]=router
    model.routers=torch.nn.ModuleDict(routers)
    assert list(dict(model.named_parameters()))==p["parameter_names"]
    model.load_state_dict(p["model"],strict=True)
    named=dict(model.named_parameters());groups=[]
    assert len(p["optimizer_parameter_names"])==len(p["optimizer"]["param_groups"])
    for saved,names in zip(p["optimizer"]["param_groups"],p["optimizer_parameter_names"]):
        assert len(names)==len(saved["params"])
        groups.append({**{k:copy.deepcopy(v) for k,v in saved.items() if k!="params"},"params":[named[n] for n in names]})
    optimizer=torch.optim.AdamW(groups)
    assert optimizer_names(model,optimizer)==p["optimizer_parameter_names"]
    optimizer.load_state_dict(p["optimizer"])
    return model,optimizer


def load_final_checkpoint(path,device,restore=False):
    p=base.d0.torch_load(Path(path).resolve(),mmap=True)
    assert p["schema"] in (SCHEMA,"experiment_2d10_checkpoint_v1") and p["arm"]=="H"
    model,optimizer=restore_h_payload(p,device)
    loader=loader_from_state(p["loader_state"])
    if restore:base.restore_rng(p["rng_state"])
    return model,optimizer,loader,p


def source_checks(p):
    checks={"schema":p["schema"]=="experiment_2d10_checkpoint_v1","arm":p["arm"]=="H",
        "source_global_update":p["global_update"]==2481,"source_targets":p["cumulative_targets"]==1300758528,
        "original_O1_ancestry":p["parent_checkpoint_sha256"]==PARENT_SHA256,
        "source_adaptation":p["local_updates"]==191 and p["new_targets"]==100139008,
        "architecture":p["architecture_fingerprint"]==core.architecture_fingerprint("H"),
        "parameters":p["parameter_count"]==124697386,"source_cursor":canonical_sha(p["loader_state"])==SOURCE_CURSOR_SHA,
        "next_batch":p["next_global_batch_sha256"]==SOURCE_NEXT_BATCH,"next_stream":p["next_global_batch_stream_sha256"]==SOURCE_NEXT_STREAM,
        "router_counters_191":all(v==191 for n,v in p["current_optimizer_steps_by_name"].items() if n.startswith("routers.")),
        "batch_shape":p["loader_state"]["batch_size"]==32 and p["loader_state"]["sequence_length"]==1024 and p["gradient_accumulation"]==16,
        "scientific_commit":p["git_implementation_commit"]==read_json(SEALED_RESULT/"FINAL_AUDIT.json")["scientific_implementation_commit"]}
    for g in p["optimizer"]["param_groups"]:
        name=g["name"]
        lr=3e-5 if name in ("base_decay","base_nodecay","router_decay","router_nodecay") else 3e-4
        checks['settings_'+name]=g['lr']==lr and g['weight_decay']==(.1 if name in ('base_decay','router_decay') else 0) and tuple(g['betas'])==(.9,.95) and g['eps']==1e-8
    return checks


def load_parent_model(path,arm,device,restore=False):
    assert arm=="H"
    require_file(path,SOURCE_SHA256["H"],"sealed H100M source")
    model,optimizer,loader,p=load_final_checkpoint(path,device,restore)
    checks=source_checks(p);checks.update(exact_restore_checks(model,optimizer,loader,p,restore))
    checks['compatibility_exact']=compatibility_state(model,optimizer)==p['source_compatibility_state']
    assert all(checks.values()),checks
    metadata=source_metadata(p)
    metadata.update(source_optimizer_steps_by_name=d6.optimizer_steps_by_name(model,optimizer),
        source_dormant_state=d6.dormant_state(model,optimizer),source_compatibility_state=compatibility_state(model,optimizer),
        source_checkpoint_sha256=SOURCE_SHA256['H'],initial_hidden_hashes=p['initial_hidden_hashes'])
    del p;gc.collect()
    return model,optimizer,loader,metadata,checks

def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()

def require_branch(clean=False):
    current = git("branch", "--show-current")
    if current != BRANCH:
        raise SystemExit(f"expected branch {BRANCH}, found {current}")
    if clean and git("status", "--porcelain"):
        raise SystemExit("2D10 requires a clean worktree")

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

def train_update(model, optimizer, loader, local_update, device):
    global_update = PARENT_GLOBAL_UPDATE + int(local_update)
    passes = base.pass_count(global_update)
    before_steps = d6.optimizer_steps_by_name(model, optimizer)
    before_dormant = d6.dormant_state(model, optimizer)
    before_compatibility = compatibility_state(model, optimizer)
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
    if not all(row["finite"] for row in gradients.values()):
        raise SystemExit(f"missing/nonfinite active gradients: {gradients}")
    if model.g_rec_b6.grad is not None:
        raise SystemExit("dormant B6 gate received a gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
    if not torch.isfinite(norm):
        raise SystemExit("nonfinite gradient norm")
    optimizer.step()
    after_steps = d6.optimizer_steps_by_name(model, optimizer)
    after_dormant = d6.dormant_state(model, optimizer)
    active = set(after_steps) - retired(model.arm)
    step_checks = {
        "compatibility_unchanged": compatibility_state(model, optimizer)==before_compatibility,
        "active_incremented_once": all(after_steps[name] == before_steps.get(name, 0) + 1 for name in active),
        "dormant_step_unchanged": after_steps["g_rec_b6"] == before_steps["g_rec_b6"],
        "dormant_parameter_unchanged": after_dormant["parameter_sha256"] == before_dormant["parameter_sha256"],
        "dormant_optimizer_unchanged": after_dormant["optimizer_state_sha256"] == before_dormant["optimizer_state_sha256"],
    }
    if not all(step_checks.values()) or not base.model_finite(model) or not base.optimizer_finite(optimizer):
        raise SystemExit(f"optimizer state failure: {step_checks}")
    elapsed = time.monotonic() - started
    return {
        "local_update": int(local_update),
        "experiment_total_update": PRIOR_UPDATES+int(local_update),
        "experiment_total_targets": PRIOR_TARGETS+int(local_update)*TARGETS_PER_UPDATE, "global_update": global_update,
        "pass_count": passes, "target_count": TARGETS_PER_UPDATE,
        "new_targets": int(local_update) * TARGETS_PER_UPDATE,
        "cumulative_targets": PARENT_TARGETS + int(local_update) * TARGETS_PER_UPDATE,
        "pass_losses": [value / ACCUMULATION for value in totals],
        "ce": totals[-1] / ACCUMULATION,
        "weighted_multipass_ce": sum(a*b for a,b in zip((.25,.75) if passes==2 else (.20,.40,.40),totals))/ACCUMULATION,
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
        "source_compatibility_state":metadata["source_compatibility_state"],
        "original_O1_checkpoint_sha256":PARENT_SHA256,
        "initial_hidden_hashes":metadata["initial_hidden_hashes"],
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
        "router_steps": all(v == PRIOR_UPDATES + expected_updates for n,v in d6.optimizer_steps_by_name(model, optimizer).items() if n.startswith("routers.")),
        "compatibility_unchanged":compatibility_state(model,optimizer)==payload["source_compatibility_state"],
        "parameter_names": payload["parameter_names"] == list(dict(model.named_parameters())),
        "optimizer_names": payload["optimizer_parameter_names"] == optimizer_names(model, optimizer),
        "dormant_unchanged": d6.dormant_state(model, optimizer) == payload["source_dormant_state"],
        "optimizer_progression": all(step == payload["source_optimizer_steps_by_name"].get(name, 0) + (0 if name in retired(arm) else expected_updates) for name, step in d6.optimizer_steps_by_name(model, optimizer).items()),
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
    if arm != "H":
        raise SystemExit("only H training is authorized")
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
        "ledger": sha256(args.continuation_ledger) == continuation.get("ledger_sha256") == LEDGER_SHA256,
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
        print(f"2D10_H_250M {arm} update {local_update}/{LOCAL_UPDATES} CE {row['ce']:.6f}", flush=True)
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
    active = set(final_steps) - retired(arm)
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
    print(f"EXPERIMENT_2D10_H_250M_TRAINING_COMPLETE {arm}", flush=True)

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
        raise SystemExit("could not construct one fresh 2D10 panel")
    panel_hash = prior.aggregate_hashes(row["combined_sha256"] for row in identities)
    manifest = {
        "experiment": EXPERIMENT,
        "panel_name": "fresh disjoint H/D 250M matched panel",
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
        "H100M_panel_excluded": all(tuple(span) in excluded_spans for span in read_json(SEALED_RESULT/"EVALUATION_PANEL_MANIFEST.json")["canonical_target_spans_half_open"]),
        "sealed_controls_panels_excluded": all(tuple(span) in excluded_spans for path in REPO_ROOT.glob("results/experiment_2d9_token_conditioned_dynamic_recurrent_gating*/EVALUATION_PANEL_MANIFEST.json") for span in read_json(path)["canonical_target_spans_half_open"]),
        "recent_panels_recovered": all(any(term in row["path"] for row in historical_files) for term in ("experiment_2d8", "experiment_2d7", "experiment_2d6_fresh_panel_confirmation")),
        "unique_batches": len(set(selected)) == PANEL_BATCHES,
        "unique_sequence_hashes": len({row["combined_sha256"] for row in sequences}) == PANEL_SEQUENCES,
    }
    audit = {
        "schema": "experiment_2d10_panel_disjointness_v1",
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
        raise SystemExit(f"fresh 2D10 panel audit failed: {checks}")
    print(f"EXPERIMENT_2D10_PANEL_FROZEN {panel_hash}", flush=True)


def run_evaluate(args):
    require_branch(clean=True)
    device = base.require_a100()
    panel = read_json(args.panel_manifest)
    assert panel["sequence_count"] == PANEL_SEQUENCES and panel["targets_per_condition"] == PANEL_TARGETS
    assert panel["sealed_before_training_and_scoring"] is True
    condition = args.condition
    assert condition in ("D_REAL","H_REAL")
    arm = condition[0]
    checkpoint = Path(args.checkpoint).resolve()
    if arm == "D": require_file(checkpoint, CONTROL_SHA256, "sealed D250M control")
    verification = read_json(str(checkpoint) + ".verification.json")
    assert verification["strict_reopen"]["passed"] and verification["sha256"] == sha256(checkpoint)
    model, optimizer, loader, p = (dynamic.load_final_checkpoint if arm == "D" else load_final_checkpoint)(checkpoint, device)
    assert p["arm"] == arm and p["global_update"] == FINAL_GLOBAL_UPDATE
    assert arm == "D" or p["git_implementation_commit"] == git("rev-parse", "HEAD")
    del optimizer, loader
    output = Path(args.output_path)
    if output.exists():
        raise SystemExit("refusing fourth, repeated, or overwritten final evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    collector = None
    if arm == "H":
        scalar_path = output.parent / f"GATE_SCALARS_{arm}.npy"
        if scalar_path.exists():
            raise SystemExit("refusing to overwrite gate scalars")
        collector = GateCollector(scalar_path,arm)
        model.gate_collector = collector
    state = {
        "schema": "experiment_2d10_incremental_losses_v1", "experiment": EXPERIMENT,
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
            "w_norm": float(getattr(model, n).detach().norm()) if arm in "DT" else 0.0}
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
            print(f"2D10 {condition} batch {ordinal+1}/{PANEL_BATCHES}", flush=True)
            del cpu_x, cpu_y, x, y, cache, logits, nll
    model.gate_collector = None
    state["parameter_count"] = sum(v.numel() for v in model.parameters())
    state["router_norms"] = {n:float(p.detach().norm()) for n,p in model.named_parameters() if n.startswith("routers.")}
    state["H_output_biases"] = {str(b):model.routers[str(b)].b2.detach().cpu().tolist() for b in core.BLOCKS} if arm=="H" else None
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
        and all(math.isfinite(v) for v in state["per_sequence_ce"]) and state["model_tensors_unchanged"] and all(v == 33_289_728 for v in state["persistent_state"].values()))
    state["status"] = "complete" if state["passed"] else "failed"
    durable_json(output, state)
    if not state["passed"]:
        raise SystemExit(f"final evaluation audit failed: {condition}")
    print(f"EXPERIMENT_2D10_EVALUATION_COMPLETE {condition}", flush=True)


def run_preflight(args):
    require_branch(clean=True)
    device=base.require_a100();output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    kernel=REPO_ROOT/'scripts/experiment_2d10_core.py'
    kernel_checks={'unchanged_H_kernel':subprocess.check_output(['git','show','ef9d33ca5ad89a1b0b18db159310fc3e3dbe3a9c:scripts/experiment_2d10_core.py'],cwd=REPO_ROOT)==kernel.read_bytes(),
                   'inherited_100m_audit_passed':read_json(SEALED_RESULT/'FINAL_AUDIT.json')['passed']}
    assert all(kernel_checks.values())
    test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_experiment_2d10_h_250m.py'],cwd=REPO_ROOT,
        env={**os.environ,'EXP2D10_TEST_DEVICE':'cuda'},capture_output=True,text=True)
    durable_text(output/'TARGETED_TESTS.txt',test.stdout+test.stderr)
    assert test.returncode==0,test.stdout+test.stderr
    control=Path(args.control_checkpoint);require_file(control,CONTROL_SHA256,'sealed D250M control')
    d=base.d0.torch_load(control,mmap=True)
    cv=read_json(HISTORICAL_RESULT/'CHECKPOINT_MANIFESTS.json')['D']
    control_checks={'sha_matches_manifest':cv['sha256']==CONTROL_SHA256,'prior_strict_reopen':cv['verification']['strict_reopen']['passed'],
        'correct_endpoint':d['global_update']==2767 and d['cumulative_targets']==1450704896,
        'correct_architecture':d['arm']=='D' and d['parameter_count']==124478212,
        'total_adaptation':d['experiment_total_updates']==477 and d['experiment_total_targets']==250085376,
        'original_O1_ancestry_via_sealed_D100M':d['parent_checkpoint_sha256']=='c9d859813d1cc2b2df33527d9a07cba32f3901e64febe752cf95a30bb9a73b44'}
    ledger=HISTORICAL_RESULT/'MATCHED_BATCH_LEDGER.jsonl';assert sha256(ledger)==LEDGER_SHA256
    rows=load_rows(ledger);manifest=read_json(HISTORICAL_RESULT/'CONTINUATION_MANIFEST.json')
    dlog=load_rows(HISTORICAL_RESULT/'TRAINING_D.jsonl')
    assert len(rows)==len(dlog)==286
    assert [x['logical_global_batch_sha256'] for x in rows]==[x['batch_sha256'] for x in dlog]
    assert [x['logical_global_stream_sha256'] for x in rows]==[x['stream_sha256'] for x in dlog]
    assert [x['pass_count'] for x in rows]==[x['pass_count'] for x in dlog]
    control_checks['terminal_cursor']=d['loader_state']==manifest['final_loader_cursor']
    control_checks['terminal_batch']=d['next_global_batch_sha256']==manifest['next_global_batch_sha256']
    control_checks['terminal_stream']=d['next_global_batch_stream_sha256']==manifest['next_stream_sha256']
    assert all(control_checks.values()),control_checks
    del d
    model,optimizer,loader,metadata,checks=load_parent_model(args.parent_checkpoint,'H',device,restore=True)
    assert sum(p.numel() for p in model.parameters())==124697386
    assert optimizer_names(model,optimizer)==read_json(SEALED_RESULT/'SOURCE_AND_PARAMETER_AUDIT.json')['H']['optimizer_parameter_names']
    # Strict self-reload of H100M, with no fresh-router initializer, before smoke.
    other,opt,other_loader,p=load_final_checkpoint(args.parent_checkpoint,device,restore=True)
    assert all(exact_restore_checks(other,opt,other_loader,p,True).values())
    state=loader.state_dict();cpu_x,cpu_y=loader.next_batch();loader=loader_from_state(state)
    x,y=cpu_x[:1,:70].to(device),cpu_y[:1,:70].to(device)
    equivalence={}
    for mode in ('parallel','incremental'):
        values=[]
        for m in (model,other):
            m.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                if mode=='parallel':
                    result=m.forward_multi_pass(x,targets=y,num_passes=2);z=result['logits'];loss=result['loss']
                else:
                    cache=m.init_incremental_state(1,device=device,dtype=torch.bfloat16);zs=[]
                    for i in range(70):
                        z,cache=m.incremental_step(x[:,i],cache);zs.append(z)
                    z=torch.cat(zs,dim=1);loss=F.cross_entropy(z.float().flatten(0,1),y.flatten())
                values.append((z.detach().cpu(),float(loss)))
        assert torch.equal(values[0][0],values[1][0]) and values[0][1]==values[1][1]
        equivalence[mode]={'logits_exact':True,'CE_exact':True,'CE':values[0][1]}
    del other,opt,other_loader,p,z,values,x,y,cpu_x,cpu_y
    gc.collect();torch.cuda.empty_cache()
    # Reload scientific RNG immediately before one disposable complete update.
    p=base.d0.torch_load(Path(args.parent_checkpoint),mmap=True);base.restore_rng(p['rng_state']);del p
    smoke=train_update(model,optimizer,loader,1,device)
    assert smoke['global_update']==2482 and smoke['experiment_total_update']==192
    with tempfile.TemporaryDirectory(prefix='exp2d10-h250-smoke-') as folder:
        path=Path(folder)/'smoke.pt'
        verification=save_final(path,model,optimizer,loader,metadata,'H','DISPOSABLE',device,1)
        reopened,opt,rl,pl=load_final_checkpoint(path,device,restore=True)
        assert all(exact_restore_checks(reopened,opt,rl,pl,True).values())
        assert state_equal(model.state_dict(),reopened.state_dict()) and state_equal(optimizer.state_dict(),opt.state_dict())
        del reopened,opt,rl,pl
    source_audit={'H':{'checks':checks,'source_metadata':metadata,'optimizer_parameter_names':optimizer_names(model,optimizer),
        'parameter_count':124697386},'D_control':{'checks':control_checks,'sha256':CONTROL_SHA256},
        'H100M_self_equivalence':equivalence,'disposable_update':smoke,'disposable_reload':verification['strict_reopen'],
        'historical_ledger_sha256':LEDGER_SHA256,'actual_H_parent_sha256':SOURCE_SHA256['H'],
        'historical_ledger_provenance_preserved':True,'no_fresh_router_initialization':True}
    audit={'authorized':True,'git_commit':git('rev-parse','HEAD'),'implementation_sha256':implementation_sha256(),
        'kernel_checks':kernel_checks,'targeted_tests_passed':True,'complete_resume_checks_passed':True,
        'source_H_sha256':SOURCE_SHA256['H'],'control_D_sha256':CONTROL_SHA256,
        'scientific_state_independently_reloaded_for_training':True,'torch_version':torch.__version__,'numpy_version':np.__version__}
    durable_json(output/'SOURCE_AND_RESUME_AUDIT.json',source_audit);durable_json(output/'PREFLIGHT_AUDIT.json',audit)
    print('EXPERIMENT_2D10_H_250M_PREFLIGHT_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    for name in ('panel','preflight','train','evaluate'):
        p=sub.add_parser(name)
        if name!='evaluate':p.add_argument('--output-dir',required=True)
        if name in ('train','preflight'):p.add_argument('--parent-checkpoint',required=True)
        if name=='preflight':p.add_argument('--control-checkpoint',required=True)
        if name=='panel':p.add_argument('--dataset',required=True)
        if name=='train':
            p.add_argument('--arm',choices=('H',),default='H');p.add_argument('--resume-checkpoint')
            for flag in ('preflight-audit','continuation-manifest','continuation-ledger','checkpoint-dir'):p.add_argument('--'+flag,required=True)
        if name=='evaluate':
            p.add_argument('--condition',choices=('D_REAL','H_REAL'),required=True)
            for flag in ('checkpoint','panel-manifest','data-root','output-path'):p.add_argument('--'+flag,required=True)
    args=parser.parse_args()
    {'panel':run_prepare_panel,'preflight':run_preflight,'train':run_train,'evaluate':run_evaluate}[args.command](args)

if __name__=='__main__':main()
