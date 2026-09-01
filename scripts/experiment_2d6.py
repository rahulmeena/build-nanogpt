#!/usr/bin/env python3
"""Lean, fail-closed Experiment 2D6 scientific driver."""

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
import time

import numpy as np
import torch
from torch.nn import functional as F

import experiment_2d3a as base
import experiment_2d4a as d4a
import experiment_2d5c as d5c
import experiment_2d6_core as core


EXPERIMENT = "2D6"
PROTOCOL = "b6_native_w1024_no_b7_recurrence_matched_100m_v1"
BRANCH = "experiment-2d6-b6-w1024-no-b7-recurrence-matched-100m"
FINAL_TAG = BRANCH + "-final"
TOOLING_BASE = "3240d98420a1989aca00e2514b2d3aa6195c6dbd"
SCHEMA = "experiment_2d6_checkpoint_v1"
SOURCE_SHA256 = core.SOURCE_CHECKPOINT_SHA256
CONTROL_SHA256 = core.FIXED_CONTROL_CHECKPOINT_SHA256
SOURCE_UPDATES = 1_908
SOURCE_TARGETS = 1_000_341_504
LOCAL_UPDATES = 191
LOCAL_TARGETS = 100_139_008
FINAL_GLOBAL_UPDATE = 2_099
FINAL_CUMULATIVE_TARGETS = 1_100_480_512
RESTART_LOCAL_UPDATE = 96
PARAMETERS = core.EXPECTED_PARAMETER_COUNT
SOURCE_NEXT_BATCH = "61dd83544d83c0cf7b4d61005f5a9cf64e2cafa930af1819cba2aae4538e7e61"
SOURCE_NEXT_STREAM = "39f6599f552803150fad33d32aa9c4df5843b058410ff1ba38b5afa469046e97"
FINAL_NEXT_BATCH = "62800455f294aaf110fbfc024abaa601c30f45d96175acb795a4d162d53da097"
FINAL_NEXT_STREAM = "cdfd4afb20c268d69e3e3fbc1c39076af21719ba6d2a6636180f12b1afd5a157"
REPLAY_LEDGER_SHA256 = "429f1d11b2af285fafab8aaf48341f6098a983b7f89598b1465974f4e969b6c0"
REPLAY_CHAIN_SHA256 = "6a5ab6dd6aee669d7e415b6ba456bf5e7e940940ab624cf1a957b103cee72d0e"
FIXED_EXPECTED_CE = 3.044323022936
BOOTSTRAP_SEED = 2_026_090_2
BOOTSTRAP_RESAMPLES = 50_000
DELTA_CE = 0.0001
RUN_ROOT = Path("/workspace/exp2d6_b6_native_100m")
REPO_ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def require_branch(clean=False):
    current = git("branch", "--show-current")
    if current != BRANCH:
        raise SystemExit(f"expected branch {BRANCH}, found {current}")
    if clean and git("status", "--porcelain"):
        raise SystemExit("Git worktree must be clean")


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


def require_exact_file(path, expected, label):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} is missing: {resolved}")
    actual = sha256(resolved)
    if actual != expected:
        raise SystemExit(f"{label} SHA mismatch: {actual}")
    return resolved


def implementation_sha256():
    paths = (
        REPO_ROOT / "scripts" / "experiment_2d6.py",
        REPO_ROOT / "scripts" / "experiment_2d6_core.py",
        REPO_ROOT / "configs" / "exp2d6_b6_w1024_no_b7_recurrence_matched_100m.json",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


def bind_parameters(fixed, model):
    model.g_rec = fixed.g_rec
    model.g_rec_b3 = fixed.g_rec_b3
    model.g_rec_b5 = fixed.g_rec_b5
    model.g_rec_b6 = fixed.g_rec_b6


def make_new_model(source_checkpoint, device, restore=False):
    fixed, optimizer, loader, payload, source_checks = d4a.load_fixed_source(
        require_exact_file(source_checkpoint, SOURCE_SHA256, "accepted source"),
        device,
        restore=False,
    )
    model = core.B6NativeNoB7RecurrenceGPT(fixed.base).to(device)
    bind_parameters(fixed, model)
    fixed_named = dict(fixed.named_parameters())
    new_named = dict(model.named_parameters())
    optimizer_parameters = {
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    checks = {
        **source_checks,
        "source_sha256": sha256(source_checkpoint) == SOURCE_SHA256,
        "parameter_names_exact": fixed_named.keys() == new_named.keys(),
        "parameter_shapes_exact": all(
            fixed_named[name].shape == new_named[name].shape for name in fixed_named
        ),
        "parameter_identity_exact": all(
            fixed_named[name] is new_named[name] for name in fixed_named
        ),
        "parameter_count_exact": sum(p.numel() for p in model.parameters())
        == PARAMETERS,
        "optimizer_parameter_identity": optimizer_parameters
        == set(model.parameters()),
        "architecture_fingerprint": model.architecture_fingerprint()
        == core.ARCHITECTURE_FINGERPRINT,
        "state_dict_keys_exact": fixed.state_dict().keys()
        == model.state_dict().keys(),
        "b6_gate_identity_preserved": model.g_rec_b6 is fixed.g_rec_b6,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D6 construction failed: {checks}")
    if restore:
        base.restore_rng(payload["rng_state"])
    del fixed
    return model, optimizer, loader, payload, checks


def load_fixed_control(control_checkpoint, source_checkpoint, device):
    path = require_exact_file(control_checkpoint, CONTROL_SHA256, "Fixed-100M")
    fixed, optimizer, loader, payload = d4a.load_arm_checkpoint(
        path, source_checkpoint, device, restore=False
    )
    checks = {
        "sha256": sha256(path) == CONTROL_SHA256,
        "arm_fixed": payload.get("arm") == "fixed",
        "parent": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256,
        "local_updates": payload.get("d4a_local_updates") == LOCAL_UPDATES,
        "global_update": payload.get("inherited_global_update")
        == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": payload.get("inherited_total_targets")
        == FINAL_CUMULATIVE_TARGETS,
        "parameter_count": sum(p.numel() for p in fixed.parameters()) == PARAMETERS,
        "next_batch": payload.get("next_global_batch_sha256") == FINAL_NEXT_BATCH,
        "next_stream": payload.get("next_global_batch_stream_sha256")
        == FINAL_NEXT_STREAM,
        "not_routed": not any(
            name.startswith("source_routers.")
            for name, _ in fixed.named_parameters()
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"Fixed control provenance failed: {checks}")
    return fixed, optimizer, loader, payload, checks


def optimizer_steps_by_name(model, optimizer):
    names = {parameter: name for name, parameter in model.named_parameters()}
    result = {}
    for parameter, state in optimizer.state.items():
        step = state.get("step")
        if step is None:
            continue
        result[names[parameter]] = int(step.item() if torch.is_tensor(step) else step)
    return result


def tensor_state_digest(value):
    rows = []
    if torch.is_tensor(value):
        return d5c.tensor_sha256(value)
    for name, current in sorted(value.items()):
        rows.append(
            {
                "name": name,
                "value": d5c.tensor_sha256(current)
                if torch.is_tensor(current)
                else current,
            }
        )
    return canonical_sha(rows)


def dormant_state(model, optimizer):
    return {
        "parameter_sha256": d5c.tensor_sha256(model.g_rec_b6),
        "optimizer_state_sha256": tensor_state_digest(
            optimizer.state[model.g_rec_b6]
        ),
        "optimizer_step": optimizer_steps_by_name(model, optimizer)["g_rec_b6"],
    }


def active_gradient_report(model):
    rows = {}
    for name, parameters in {
        "base": [
            parameter
            for current_name, parameter in model.named_parameters()
            if current_name not in {"g_rec", "g_rec_b3", "g_rec_b5", "g_rec_b6"}
        ],
        "b1_gate": [model.g_rec],
        "b3_gate": [model.g_rec_b3],
        "b5_gate": [model.g_rec_b5],
    }.items():
        gradients = [p.grad for p in parameters if p.grad is not None]
        squared = sum(
            (gradient.float().square().sum() for gradient in gradients),
            torch.tensor(0.0, device=next(model.parameters()).device),
        )
        rows[name] = {
            "gradient_tensors": len(gradients),
            "finite": bool(gradients)
            and all(bool(torch.isfinite(g).all()) for g in gradients),
            "nonzero": bool(gradients) and bool(squared.gt(0).item()),
            "norm": float(squared.sqrt().item()),
        }
    return rows


def train_update(model, optimizer, loader, accumulation, local_update, device):
    if not 1 <= int(local_update) <= LOCAL_UPDATES:
        raise SystemExit("2D6 entry point refuses an update outside 1..191")
    global_update = SOURCE_UPDATES + int(local_update)
    pass_count = base.pass_count(global_update)
    before_steps = optimizer_steps_by_name(model, optimizer)
    before_dormant = dormant_state(model, optimizer)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = [0.0] * pass_count
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(int(accumulation)):
        cpu_x, cpu_y = loader.next_batch()
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model.forward_multi_pass(
                x,
                targets=y,
                num_passes=pass_count,
                activation_checkpointing=True,
            )
            loss = result["loss"] / int(accumulation)
        for index, current in enumerate(result["pass_losses"]):
            totals[index] += float(current.detach().float())
        loss.backward()
        del cpu_x, cpu_y, x, y, result, loss
    active_gradients = active_gradient_report(model)
    if not all(row["finite"] and row["nonzero"] for row in active_gradients.values()):
        raise SystemExit(f"missing/nonfinite active gradients: {active_gradients}")
    if model.g_rec_b6.grad is not None:
        raise SystemExit("dormant g_rec_b6 unexpectedly received a gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
    if not torch.isfinite(gradient_norm):
        raise SystemExit("nonfinite gradient norm")
    optimizer.step()
    after_dormant = dormant_state(model, optimizer)
    after_steps = optimizer_steps_by_name(model, optimizer)
    active_names = set(after_steps) - {"g_rec_b6"}
    step_checks = {
        "active_incremented_once": all(
            after_steps[name] == before_steps[name] + 1 for name in active_names
        ),
        "dormant_step_unchanged": after_steps["g_rec_b6"]
        == before_steps["g_rec_b6"],
        "dormant_parameter_unchanged": after_dormant["parameter_sha256"]
        == before_dormant["parameter_sha256"],
        "dormant_optimizer_state_unchanged": after_dormant[
            "optimizer_state_sha256"
        ]
        == before_dormant["optimizer_state_sha256"],
    }
    if not all(step_checks.values()):
        raise SystemExit(f"optimizer/dormant-state failure: {step_checks}")
    if not base.model_finite(model) or not base.optimizer_finite(optimizer):
        raise SystemExit("nonfinite model or optimizer after update")
    elapsed = time.monotonic() - started
    return {
        "local_update": int(local_update),
        "global_update": global_update,
        "pass_count": pass_count,
        "target_count": base.GLOBAL_TARGETS,
        "new_targets": int(local_update) * base.GLOBAL_TARGETS,
        "cumulative_targets": SOURCE_TARGETS
        + int(local_update) * base.GLOBAL_TARGETS,
        "pass_losses": [value / int(accumulation) for value in totals],
        "ce": totals[-1] / int(accumulation),
        "gradient_norm_before_clip": float(gradient_norm.detach().float()),
        "active_gradient_groups": active_gradients,
        "dormant_b6_compatibility": {
            "gradient_is_none": model.g_rec_b6.grad is None,
            "before": before_dormant,
            "after": after_dormant,
            "checks": step_checks,
        },
        "optimizer_step_success": True,
        "wall_seconds": elapsed,
        "targets_per_second": base.GLOBAL_TARGETS / elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device)
        / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device)
        / 1024**2,
    }


def loader_from_state(state):
    return base.d1.ExplicitShardLoader(
        state["shards"], state["batch_size"], base.T, state=state
    )


def checkpoint_name(local_update):
    targets = SOURCE_TARGETS + int(local_update) * base.GLOBAL_TARGETS
    return f"scientific_cumulative_{targets:012d}.pt"


def checkpoint_payload(
    model,
    optimizer,
    loader,
    source_payload,
    local_update,
    replay_ledger_sha256,
):
    local_update = int(local_update)
    rng = base.capture_rng()
    accumulation = int(source_payload["gradient_accumulation"])
    current_steps = optimizer_steps_by_name(model, optimizer)
    source_steps = {
        name: step if name == "g_rec_b6" else step - local_update
        for name, step in current_steps.items()
    }
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "description": "b6-w1024-no-b7-recurrence",
        "arm": "NEW",
        "parent_checkpoint_sha256": SOURCE_SHA256,
        "fixed_control_checkpoint_sha256": CONTROL_SHA256,
        "local_updates": local_update,
        "global_update": SOURCE_UPDATES + local_update,
        "new_targets": local_update * base.GLOBAL_TARGETS,
        "cumulative_targets": SOURCE_TARGETS + local_update * base.GLOBAL_TARGETS,
        "targets_per_update": base.GLOBAL_TARGETS,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": copy.deepcopy(source_payload.get("scheduler")),
        "loader_state": loader.state_dict(),
        "loader_states": [loader.state_dict()],
        "rng_state": rng,
        "rng_digests": d5c.rng_digests(rng),
        "gradient_accumulation": accumulation,
        "next_global_batch_sha256": base.next_batch_hash(loader, accumulation),
        "next_global_batch_stream_sha256": base.next_stream_hash(
            loader, accumulation
        ),
        "next_pass_count": base.pass_count(SOURCE_UPDATES + local_update + 1)
        if local_update < LOCAL_UPDATES
        else None,
        "architecture_manifest": core.ARCHITECTURE_MANIFEST,
        "architecture_fingerprint": core.ARCHITECTURE_FINGERPRINT,
        "replay_ledger_sha256": replay_ledger_sha256,
        "inactive_b6_compatibility_state": {
            **core.ARCHITECTURE_MANIFEST["inactive_compatibility_state"],
            "observed": dormant_state(model, optimizer),
        },
        "parameter_count": PARAMETERS,
        "optimizer_group_definitions": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
        "source_optimizer_steps_by_name": source_steps,
        "current_optimizer_steps_by_name": current_steps,
        "git_implementation_commit": git("rev-parse", "HEAD"),
        "tooling_base": TOOLING_BASE,
        "saved_process_id": os.getpid(),
        "saved_at_unix": time.time(),
    }


def load_new_checkpoint(path, source_checkpoint, device, restore=False):
    payload = base.d0.torch_load(Path(path), mmap=False)
    if payload.get("schema") != SCHEMA or payload.get("arm") != "NEW":
        raise SystemExit("not a 2D6 New checkpoint")
    model, optimizer, _, source, _ = make_new_model(
        source_checkpoint, device, restore=False
    )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    loader = loader_from_state(payload["loader_state"])
    if restore:
        base.restore_rng(payload["rng_state"])
    return model, optimizer, loader, payload, source


def strict_reopen(path, source_checkpoint, device):
    entry_rng = base.capture_rng()
    model = optimizer = loader = payload = None
    try:
        model, optimizer, loader, payload, source = load_new_checkpoint(
            path, source_checkpoint, device, restore=False
        )
        local_update = int(payload["local_updates"])
        accumulation = int(payload["gradient_accumulation"])
        steps = optimizer_steps_by_name(model, optimizer)
        active = set(steps) - {"g_rec_b6"}
        source_steps = payload.get("source_optimizer_steps_by_name", {})
        checks = {
            "schema": payload.get("schema") == SCHEMA,
            "parent": payload.get("parent_checkpoint_sha256") == SOURCE_SHA256,
            "control_provenance": payload.get("fixed_control_checkpoint_sha256")
            == CONTROL_SHA256,
            "local_update_range": 0 <= local_update <= LOCAL_UPDATES,
            "targets": payload.get("new_targets")
            == local_update * base.GLOBAL_TARGETS,
            "global_update": payload.get("global_update")
            == SOURCE_UPDATES + local_update,
            "cumulative_targets": payload.get("cumulative_targets")
            == SOURCE_TARGETS + local_update * base.GLOBAL_TARGETS,
            "parameter_count": sum(p.numel() for p in model.parameters())
            == PARAMETERS,
            "architecture": payload.get("architecture_fingerprint")
            == core.ARCHITECTURE_FINGERPRINT,
            "model_finite": base.model_finite(model),
            "optimizer_finite": base.optimizer_finite(optimizer),
            "next_batch": base.next_batch_hash(loader, accumulation)
            == payload.get("next_global_batch_sha256"),
            "next_stream": base.next_stream_hash(loader, accumulation)
            == payload.get("next_global_batch_stream_sha256"),
            "rng_complete": set(payload.get("rng_state", {}))
            == {"python", "numpy", "torch_cpu", "torch_cuda"},
            "rng_digests": payload.get("rng_digests")
            == d5c.rng_digests(payload.get("rng_state", {})),
            "scheduler_exact": payload.get("scheduler")
            == source.get("scheduler"),
            "source_optimizer_step_names": set(source_steps) == set(steps),
            "active_optimizer_progression": all(
                steps[name] == source_steps.get(name, -1) + local_update
                for name in active
            ),
            "dormant_optimizer_step": steps.get("g_rec_b6")
            == source_steps.get("g_rec_b6")
            == SOURCE_UPDATES,
            "saved_current_steps_exact": payload.get(
                "current_optimizer_steps_by_name"
            )
            == steps,
            "dormant_gradient_absent": model.g_rec_b6.grad is None,
            "b7_ring_absent": not hasattr(
                model.init_incremental_state(1, device=device), "h7_ring"
            ),
        }
        return {
            "checkpoint": str(Path(path).resolve()),
            "local_update": local_update,
            "optimizer_steps": {
                "active_unique": sorted({steps[name] for name in active}),
                "dormant_b6": steps.get("g_rec_b6"),
                "source_active_unique": sorted(
                    {source_steps[name] for name in active}
                ),
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
    finally:
        del model, optimizer, loader, payload
        gc.collect()
        torch.cuda.empty_cache()
        base.restore_rng(entry_rng)


def save_checkpoint(
    path,
    model,
    optimizer,
    loader,
    source_payload,
    local_update,
    replay_sha,
    source_checkpoint,
    device,
):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model,
        optimizer,
        loader,
        source_payload,
        local_update,
        replay_sha,
    )
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    d5c.fsync_path(temporary)
    os.replace(temporary, path)
    d5c.fsync_path(path.parent)
    digest = sha256(path)
    reopen = strict_reopen(path, source_checkpoint, device)
    if not reopen["passed"]:
        raise SystemExit(f"strict 2D6 checkpoint reopen failed: {reopen}")
    verification = {
        "checkpoint": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "local_update": int(local_update),
        "global_update": SOURCE_UPDATES + int(local_update),
        "cumulative_targets": SOURCE_TARGETS
        + int(local_update) * base.GLOBAL_TARGETS,
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "next_global_batch_stream_sha256": payload[
            "next_global_batch_stream_sha256"
        ],
        "architecture_fingerprint": core.ARCHITECTURE_FINGERPRINT,
        "dormant_b6_compatibility": payload["inactive_b6_compatibility_state"],
        "saved_process_id": payload["saved_process_id"],
        "strict_reopen": reopen,
    }
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), verification)
    return verification


def panel_batch_indices(panel):
    return [int(value) for value in panel["batch_indices_in_evaluation_order"]]


def incremental_condition(model, x, y, control="all_real", audit=False):
    state = model.init_incremental_state(
        x.size(0), device=x.device, dtype=torch.bfloat16
    )
    per_sequence_nll = torch.zeros(x.size(0), dtype=torch.float64)
    cache_rows = []
    audit_positions = {1, 2, 31, 32, 63, 64, 511, 512, 513, 1023, 1024}
    for position in range(base.T):
        wants = audit and position + 1 in audit_positions
        result = model.incremental_step(
            x[:, position],
            state,
            control=control,
            return_diagnostics=wants,
            diagnostic_attention_weights=False,
        )
        if wants:
            logits, state, diagnostic = result
            cache_rows.append(
                {
                    "position": position + 1,
                    "cache_audit": diagnostic["cache_audit"],
                }
            )
        else:
            logits, state = result
        losses = F.cross_entropy(
            logits[:, 0].float(), y[:, position], reduction="none"
        ).double().cpu()
        per_sequence_nll += losses
    return {
        "nll_sum": float(per_sequence_nll.sum()),
        "targets": int(y.numel()),
        "per_sequence_nll": per_sequence_nll.tolist(),
        "per_sequence_ce": (per_sequence_nll / base.T).tolist(),
        "cache_rows": cache_rows,
        "final_cache_audit": model.incremental_cache_audit(state),
    }


def evaluate_batches(
    model,
    val_path,
    panel,
    control,
    output_path,
    identity,
    max_batches=None,
):
    output_path = Path(output_path).resolve()
    indices = panel_batch_indices(panel)
    if max_batches is not None:
        indices = indices[: int(max_batches)]
    if output_path.exists():
        state = read_json(output_path)
        if state.get("identity") != identity:
            raise SystemExit("evaluation resume identity mismatch")
        if state.get("status") == "complete":
            return state
    else:
        state = {
            "schema": "experiment_2d6_true_incremental_losses_v1",
            "experiment": EXPERIMENT,
            "identity": identity,
            "evaluation_set_label": "reused sealed matched panel",
            "panel_sha256": panel["panel_sha256"],
            "batch_indices_in_evaluation_order": indices,
            "completed_batch_indices": [],
            "batch_identities": [],
            "nll_sum": 0.0,
            "targets": 0,
            "per_sequence_nll": [],
            "per_sequence_ce": [],
            "cache_rows": [],
            "status": "running",
        }
    completed = set(state["completed_batch_indices"])
    model.eval()
    device = base.model_device(model)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for ordinal, batch_index in enumerate(indices):
            if batch_index in completed:
                continue
            cpu_x, cpu_y = d5c.batch_at_index(val_path, batch_index)
            observed_identity = base.batch_identity(cpu_x, cpu_y)
            expected = panel["batch_identities"][ordinal]
            if observed_identity != expected:
                raise SystemExit(f"panel identity mismatch at batch ordinal {ordinal}")
            x, y = cpu_x.to(device), cpu_y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                row = incremental_condition(
                    model, x, y, control=control, audit=ordinal == 0
                )
            state["nll_sum"] += row["nll_sum"]
            state["targets"] += row["targets"]
            state["per_sequence_nll"].extend(row["per_sequence_nll"])
            state["per_sequence_ce"].extend(row["per_sequence_ce"])
            state["cache_rows"].extend(row["cache_rows"])
            state["final_cache_audit"] = row["final_cache_audit"]
            state["completed_batch_indices"].append(batch_index)
            state["batch_identities"].append(observed_identity)
            durable_json(output_path, state)
            print(
                f"2D6 {identity['condition']} true-incremental batch "
                f"{ordinal + 1}/{len(indices)}",
                flush=True,
            )
            del cpu_x, cpu_y, x, y
            torch.cuda.empty_cache()
    state["aggregate_ce"] = state["nll_sum"] / state["targets"]
    state["paired_sequences"] = len(state["per_sequence_ce"])
    state["wall_seconds_this_process"] = time.monotonic() - started
    state["peak_allocated_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    state["peak_reserved_vram_mb"] = torch.cuda.max_memory_reserved(device) / 1024**2
    state["status"] = "complete"
    state["passed"] = (
        len(state["completed_batch_indices"]) == len(indices)
        and state["paired_sequences"] == 64 * len(indices)
        and state["targets"] == 64 * base.T * len(indices)
        and state["final_cache_audit"]["passed"]
    )
    durable_json(output_path, state)
    if not state["passed"]:
        raise SystemExit("true-incremental evaluation audit failed")
    return state


def cache_reload_test(model, device):
    generator = torch.Generator(device=device).manual_seed(2_026_090_201)
    tokens = torch.randint(0, 50_257, (2, 160), generator=generator, device=device)
    full = model.init_incremental_state(2, device=device, dtype=torch.bfloat16)
    split = model.init_incremental_state(2, device=device, dtype=torch.bfloat16)
    full_logits, split_logits = [], []
    temporary = Path(f"/tmp/exp2d6_incremental_{os.getpid()}.pt")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            logits, full = model.incremental_step(tokens[:, position], full)
            full_logits.append(logits)
        for start, end in ((0, 37), (37, 91), (91, 160)):
            for position in range(start, end):
                logits, split = model.incremental_step(tokens[:, position], split)
                split_logits.append(logits)
            if end == 91:
                torch.save(split, temporary)
                split = torch.load(temporary, map_location=device, weights_only=False)
    temporary.unlink(missing_ok=True)
    left, right = torch.cat(full_logits, 1), torch.cat(split_logits, 1)
    cache_exact = all(
        (a is None and b is None)
        or (
            a is not None
            and b is not None
            and torch.equal(a.key, b.key)
            and torch.equal(a.value, b.value)
        )
        for a, b in zip(full.caches, split.caches)
    )
    rings_exact = all(
        getattr(full, f"{name}_positions") == getattr(split, f"{name}_positions")
        and torch.equal(
            getattr(full, f"{name}_ring"), getattr(split, f"{name}_ring")
        )
        for name in ("h8", "h10", "h12")
    )
    checks = {
        "logits_exact": torch.equal(left, right),
        "caches_exact": cache_exact,
        "rings_exact": rings_exact,
        "state_position_exact": full.position == split.position == 160,
        "temporary_removed": not temporary.exists(),
        "no_h7_ring": not hasattr(full, "h7_ring") and not hasattr(split, "h7_ring"),
    }
    return {"checks": checks, "passed": all(checks.values())}


def targeted_architecture_test(model, optimizer, device):
    generator = torch.Generator(device=device).manual_seed(2_026_090_202)
    tokens = torch.randint(0, 50_257, (1, 96), generator=generator, device=device)
    targets = torch.randint(0, 50_257, (1, 96), generator=generator, device=device)
    before_dormant = dormant_state(model, optimizer)
    model.zero_grad(set_to_none=True)
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.forward_multi_pass(tokens, targets=targets, num_passes=2)
    result["loss"].backward()
    active = {
        "b1": model.g_rec.grad,
        "b3": model.g_rec_b3.grad,
        "b5": model.g_rec_b5.grad,
    }
    checks = {
        "b6_recurrent_branch_calls_zero": model._b6_recurrent_branch_calls == 0,
        "active_recurrent_branches_called": all(
            model._active_special_branch_calls[index] > 0 for index in (0, 2, 4)
        ),
        "dormant_b6_gradient_none": model.g_rec_b6.grad is None,
        "active_gate_gradients_present": all(
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and bool(torch.count_nonzero(gradient))
            for gradient in active.values()
        ),
        "dormant_parameter_unchanged": dormant_state(model, optimizer)
        == before_dormant,
        "state_dict_parameter_count": sum(p.numel() for p in model.parameters())
        == PARAMETERS,
    }
    model.zero_grad(set_to_none=True)
    state = model.init_incremental_state(1, device=device, dtype=torch.bfloat16)
    sequence = (torch.arange(base.T, device=device) * 3571 + 11).remainder(50_257)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(base.T):
            _, state = model.incremental_step(sequence[position : position + 1], state)
    audit = model.incremental_cache_audit(state)
    checks.update(
        {
            "b6_1023_historical_native_kv": audit["b6_historical_local_kv"]
            == 1023,
            "b6_cache_never_exceeds_1023": audit["cache_lengths"][5] <= 1023,
            "b7_ring_absent": audit["b7_ring_present"] is False
            and not hasattr(state, "h7_ring"),
            "active_rings_unchanged": audit["ring_lengths"]
            == {"h8": 1023, "h10": 1023, "h12": 1023},
            "physical_cache_audit": audit["passed"],
            "reset_complete": all(
                value == 0
                for value in model.incremental_cache_lengths(
                    model.init_incremental_state(
                        1, device=device, dtype=torch.bfloat16
                    )
                )
            ),
        }
    )
    return {
        "branch_call_counts": {
            f"B{index + 1}": count
            for index, count in model._active_special_branch_calls.items()
        },
        "b6_recurrent_branch_calls": model._b6_recurrent_branch_calls,
        "final_cache_audit": audit,
        "checks": checks,
        "passed": all(checks.values()),
    }


def causality_test(model, device):
    generator = torch.Generator(device=device).manual_seed(2_026_090_203)
    rows = []
    for length, cutoff in ((80, 47), (1024, 700)):
        tokens = torch.randint(0, 50_257, (1, length), generator=generator, device=device)
        changed = tokens.clone()
        changed[:, cutoff + 1 :] = torch.randint(
            0,
            50_257,
            changed[:, cutoff + 1 :].shape,
            generator=generator,
            device=device,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            parallel_left = model.forward_multi_pass(tokens, num_passes=2)["logits"]
            parallel_right = model.forward_multi_pass(changed, num_passes=2)["logits"]
            incremental_left = model.incremental_logits(tokens)["logits"]
            incremental_right = model.incremental_logits(changed)["logits"]
        parallel_delta = float(
            (parallel_left[:, : cutoff + 1].float() - parallel_right[:, : cutoff + 1].float())
            .abs()
            .max()
        )
        incremental_delta = float(
            (incremental_left[:, : cutoff + 1].float() - incremental_right[:, : cutoff + 1].float())
            .abs()
            .max()
        )
        rows.append(
            {
                "length": length,
                "cutoff": cutoff,
                "parallel_prefix_max_abs": parallel_delta,
                "incremental_prefix_max_abs": incremental_delta,
                "passed": parallel_delta == 0.0 and incremental_delta == 0.0,
            }
        )
    reload = cache_reload_test(model, device)
    return {
        "suffix_perturbation": rows,
        "cache_reference_reload": reload,
        "passed": all(row["passed"] for row in rows) and reload["passed"],
    }


def geometry_shock(source_checkpoint, core_panel, data_root, device):
    panel = copy.deepcopy(core_panel)
    panel["batch_indices_in_evaluation_order"] = panel[
        "batch_indices_in_evaluation_order"
    ][:1]
    panel["batch_identities"] = panel["batch_identities"][:1]
    fixed, fixed_optimizer, fixed_loader, _, _ = d4a.load_fixed_source(
        source_checkpoint, device, restore=False
    )
    new, new_optimizer, new_loader, _, _ = make_new_model(
        source_checkpoint, device, restore=False
    )
    val_path = base.validation_path(Path(data_root))
    cpu_x, cpu_y = d5c.batch_at_index(
        val_path, panel["batch_indices_in_evaluation_order"][0]
    )
    subset_hash = base.batch_identity(cpu_x, cpu_y)["combined_sha256"]
    x, y = cpu_x.to(device), cpu_y.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        original = incremental_condition(fixed, x, y, "all_real")
        changed = incremental_condition(new, x, y, "all_real")
    fixed_ce = original["nll_sum"] / original["targets"]
    new_ce = changed["nll_sum"] / changed["targets"]
    differences = np.asarray(changed["per_sequence_ce"]) - np.asarray(
        original["per_sequence_ce"]
    )
    result = {
        "schema": "experiment_2d6_zero_training_shock_v1",
        "subset": "first 64 sequences in existing immutable true-incremental core order",
        "subset_sha256": subset_hash,
        "sequences": 64,
        "targets": 65_536,
        "parent_original_real_ce": fixed_ce,
        "parent_new_geometry_ce": new_ce,
        "parent_new_minus_original_ce": new_ce - fixed_ce,
        "paired_standard_error": float(differences.std(ddof=1) / math.sqrt(64)),
        "passed": original["final_cache_audit"]["passed"]
        and changed["final_cache_audit"]["passed"],
    }
    del fixed, fixed_optimizer, fixed_loader, new, new_optimizer, new_loader
    del cpu_x, cpu_y, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return result


def disposable_smoke(source_checkpoint, replay_rows, device, output):
    model, optimizer, loader, source, checks = make_new_model(
        source_checkpoint, device, restore=True
    )
    accumulation = int(source["gradient_accumulation"])
    expected = replay_rows[0]
    before = {
        "batch": base.next_batch_hash(loader, accumulation),
        "stream": base.next_stream_hash(loader, accumulation),
    }
    if (
        before["batch"] != expected["logical_global_batch_sha256"]
        or before["stream"] != expected["logical_global_stream_sha256"]
    ):
        raise SystemExit("disposable smoke replay mismatch")
    row = train_update(model, optimizer, loader, accumulation, 1, device)
    temporary = Path(output) / "DISPOSABLE_UPDATE_0001.pt"
    verification = save_checkpoint(
        temporary,
        model,
        optimizer,
        loader,
        source,
        1,
        REPLAY_LEDGER_SHA256,
        source_checkpoint,
        device,
    )
    reopened = strict_reopen(temporary, source_checkpoint, device)
    result = {
        "label": "DISPOSABLE",
        "official_training_updates": 0,
        "construction": checks,
        "before": before,
        "row": row,
        "checkpoint": verification,
        "reopen": reopened,
        "next_batch_matches_replay_update_2": verification[
            "next_global_batch_sha256"
        ]
        == replay_rows[1]["logical_global_batch_sha256"],
        "passed": reopened["passed"]
        and row["dormant_b6_compatibility"]["gradient_is_none"],
    }
    durable_json(Path(output) / "DISPOSABLE_SMOKE_REPORT.json", result)
    for path in (
        temporary,
        temporary.with_suffix(temporary.suffix + ".sha256"),
        temporary.with_suffix(temporary.suffix + ".verification.json"),
    ):
        path.unlink(missing_ok=True)
    result["discarded"] = all(
        not path.exists()
        for path in (
            temporary,
            temporary.with_suffix(temporary.suffix + ".sha256"),
            temporary.with_suffix(temporary.suffix + ".verification.json"),
        )
    )
    result["passed"] = result["passed"] and result["discarded"]
    durable_json(Path(output) / "DISPOSABLE_SMOKE_REPORT.json", result)
    del model, optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_preflight(args):
    require_branch(clean=True)
    device = base.require_a100()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = require_exact_file(args.source_checkpoint, SOURCE_SHA256, "source")
    control = require_exact_file(args.control_checkpoint, CONTROL_SHA256, "control")
    ledger = require_exact_file(
        args.replay_ledger, REPLAY_LEDGER_SHA256, "sealed replay ledger"
    )
    replay_audit = read_json(args.replay_audit)
    replay_rows = d5c.load_replay_rows(ledger)
    large_panel = read_json(args.large_panel)
    core_panel = read_json(args.core_panel)
    fixed_large = read_json(args.fixed_large)
    stop = read_json(args.stop_status_json)
    head = git("rev-parse", "HEAD")
    provenance_checks = {
        "source_sha": sha256(source) == SOURCE_SHA256,
        "control_sha": sha256(control) == CONTROL_SHA256,
        "ledger_sha": sha256(ledger) == REPLAY_LEDGER_SHA256,
        "replay_audit_passed": replay_audit.get("passed") is True,
        "replay_audit_ledger": replay_audit.get("ledger_sha256")
        == REPLAY_LEDGER_SHA256,
        "replay_chain": replay_audit.get("terminal_chain_sha256")
        == REPLAY_CHAIN_SHA256
        == replay_rows[-1]["chain_sha256"],
        "replay_rows": len(replay_rows) == LOCAL_UPDATES,
        "first_batch": replay_rows[0]["logical_global_batch_sha256"]
        == SOURCE_NEXT_BATCH,
        "first_stream": replay_rows[0]["logical_global_stream_sha256"]
        == SOURCE_NEXT_STREAM,
        "terminal_batch": replay_audit.get("terminal_next_batch_sha256")
        == FINAL_NEXT_BATCH,
        "terminal_stream": replay_audit.get("terminal_next_stream_sha256")
        == FINAL_NEXT_STREAM,
        "targets_per_update": all(
            row["target_count"] == base.GLOBAL_TARGETS for row in replay_rows
        ),
        "cadence": sum(row["pass_count"] == 2 for row in replay_rows) == 185
        and sum(row["pass_count"] == 3 for row in replay_rows) == 6,
        "large_panel": large_panel.get("sequence_count") == 2048
        and large_panel.get("targets_per_condition") == 2_097_152
        and large_panel.get("all_required_disjointness_passed") is True,
        "fixed_stored_losses": fixed_large.get("status") == "complete"
        and fixed_large.get("passed") is True
        and fixed_large.get("panel_sha256") == large_panel.get("panel_sha256")
        and abs(
            fixed_large["conditions"]["all_real"]["validation_loss"]
            - FIXED_EXPECTED_CE
        )
        < 5e-13,
        "stop_capability_authenticated": stop.get("authenticated") is True,
        "stop_exact_pod": stop.get("pod", {}).get("id") == args.pod_id
        and stop.get("pod", {}).get("name") == args.pod_name,
        "stop_status_running": stop.get("pod", {}).get("desiredStatus") == "RUNNING"
        and stop.get("pod", {}).get("runtimeStatus") == "running",
        "volume_retained": stop.get("network_volume", {}).get("id")
        == args.volume_id,
        "one_a100": torch.cuda.device_count() == 1
        and "A100" in torch.cuda.get_device_name(device),
        "branch_pushed": git("rev-parse", f"origin/{BRANCH}") == head,
        "tooling_base_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", TOOLING_BASE, "HEAD"],
            cwd=REPO_ROOT,
        ).returncode
        == 0,
    }
    durable_json(
        output / "SCOPE_LOCK.json",
        {
            "experiment": EXPERIMENT,
            "exactly_one_newly_trained_arm": ["B6 W1024 / NO B7->B6 RECURRENCE"],
            "fixed_optimizer_steps": 0,
            "local_update_limit": LOCAL_UPDATES,
            "targets_limit": LOCAL_TARGETS,
            "entry_point_refuses_update_192": True,
            "prohibited_continuations": ["2D5C", "Fixed", "Routed", "250M"],
        },
    )
    durable_json(
        output / "SOURCE_CONTROL_PROVENANCE.json",
        {
            "source": {"path": str(source), "sha256": sha256(source)},
            "control": {"path": str(control), "sha256": sha256(control)},
            "replay": {
                "path": str(ledger),
                "sha256": sha256(ledger),
                "terminal_chain_sha256": replay_rows[-1]["chain_sha256"],
            },
            "large_panel": {
                "path": str(Path(args.large_panel).resolve()),
                "file_sha256": sha256(args.large_panel),
                "panel_sha256": large_panel["panel_sha256"],
                "label": "reused sealed matched panel",
            },
            "checks": provenance_checks,
            "passed": all(provenance_checks.values()),
        },
    )
    durable_json(output / "ARCHITECTURE_MANIFEST.json", core.ARCHITECTURE_MANIFEST)
    model, optimizer, loader, _, construction = make_new_model(
        source, device, restore=False
    )
    targeted = targeted_architecture_test(model, optimizer, device)
    causality = causality_test(model, device)
    tests = {
        "construction": {"checks": construction, "passed": all(construction.values())},
        "architecture": targeted,
        "causality_cache": causality,
    }
    tests["passed"] = all(row["passed"] for row in tests.values())
    durable_json(output / "TARGETED_PREFLIGHT.json", tests)
    del model, optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    if not tests["passed"]:
        raise SystemExit("targeted 2D6 architecture/causality/cache preflight failed")
    smoke = disposable_smoke(source, replay_rows, device, output)
    if not smoke["passed"]:
        raise SystemExit("disposable smoke failed")
    shock = geometry_shock(source, core_panel, args.data_root, device)
    durable_json(output / "ZERO_TRAINING_SHOCK.json", shock)
    checks = {
        **provenance_checks,
        "targeted_preflight": tests["passed"],
        "disposable_smoke": smoke["passed"],
        "zero_training_shock": shock["passed"],
        "source_unchanged": sha256(source) == SOURCE_SHA256,
        "control_unchanged": sha256(control) == CONTROL_SHA256,
        "implementation_sha": implementation_sha256()
        == implementation_sha256(),
    }
    audit = {
        "schema": "experiment_2d6_preflight_authorization_v1",
        "experiment": EXPERIMENT,
        "git_commit": head,
        "tooling_base": TOOLING_BASE,
        "implementation_file_sha256": implementation_sha256(),
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "volume_id": args.volume_id,
        "replay_ledger_sha256": REPLAY_LEDGER_SHA256,
        "large_panel_file_sha256": sha256(args.large_panel),
        "checks": checks,
        "authorized": all(checks.values()),
    }
    durable_json(output / "PREFLIGHT_AUTHORIZATION.json", audit)
    if not audit["authorized"]:
        raise SystemExit(f"2D6 preflight did not authorize training: {checks}")
    print("EXPERIMENT_2D6_PREFLIGHT_PASS", flush=True)


def replay_rows(path):
    rows = d5c.load_replay_rows(path)
    if sha256(path) != REPLAY_LEDGER_SHA256:
        raise SystemExit("replay ledger changed")
    return rows


def midpoint_preexit(model, optimizer, loader, verification, source):
    return {
        "schema": "experiment_2d6_midpoint_preexit_v1",
        "local_update": RESTART_LOCAL_UPDATE,
        "saved_process_id": os.getpid(),
        "checkpoint": verification,
        "model_sha256": d5c.parameter_manifest(model)["aggregate_sha256"],
        "optimizer_sha256": d5c.optimizer_manifest(model, optimizer)[
            "state_aggregate_sha256"
        ],
        "optimizer_steps": optimizer_steps_by_name(model, optimizer),
        "dormant_b6": dormant_state(model, optimizer),
        "scheduler": copy.deepcopy(source.get("scheduler")),
        "loader_state": loader.state_dict(),
        "rng_digests": d5c.rng_digests(base.capture_rng()),
        "next_batch": verification["next_global_batch_sha256"],
        "next_stream": verification["next_global_batch_stream_sha256"],
        "next_pass_count": base.pass_count(SOURCE_UPDATES + 97),
        "status": "fresh_process_required",
    }


def midpoint_restart(preexit_path, model, optimizer, loader, payload, checkpoint, source, device):
    before = read_json(preexit_path)
    accumulation = int(payload["gradient_accumulation"])
    checks = {
        "fresh_process": before["saved_process_id"] != os.getpid(),
        "saved_process_exact": payload.get("saved_process_id")
        == before["saved_process_id"],
        "checkpoint_sha": sha256(checkpoint) == before["checkpoint"]["sha256"],
        "strict_reopen": strict_reopen(checkpoint, source, device)["passed"],
        "local_update": payload.get("local_updates") == RESTART_LOCAL_UPDATE,
        "model": d5c.parameter_manifest(model)["aggregate_sha256"]
        == before["model_sha256"],
        "optimizer": d5c.optimizer_manifest(model, optimizer)[
            "state_aggregate_sha256"
        ]
        == before["optimizer_sha256"],
        "optimizer_steps": optimizer_steps_by_name(model, optimizer)
        == before["optimizer_steps"],
        "dormant_b6": dormant_state(model, optimizer) == before["dormant_b6"],
        "scheduler": payload.get("scheduler") == before["scheduler"],
        "loader": loader.state_dict() == before["loader_state"],
        "rng": d5c.rng_digests(base.capture_rng()) == before["rng_digests"],
        "next_batch": base.next_batch_hash(loader, accumulation)
        == before["next_batch"],
        "next_stream": base.next_stream_hash(loader, accumulation)
        == before["next_stream"],
        "next_pass": base.pass_count(SOURCE_UPDATES + 97)
        == before["next_pass_count"],
    }
    return {
        "schema": "experiment_2d6_midpoint_restart_audit_v1",
        "preexit_process_id": before["saved_process_id"],
        "resumed_process_id": os.getpid(),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_train(args):
    require_branch(clean=True)
    end = int(args.end_local_update)
    if end not in (RESTART_LOCAL_UPDATE, LOCAL_UPDATES):
        raise SystemExit("2D6 training may end only at update 96 or 191")
    device = base.require_a100()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = read_json(args.preflight_authorization)
    head = git("rev-parse", "HEAD")
    code_checks = {
        "authorized": preflight.get("authorized") is True,
        "commit": preflight.get("git_commit") == head,
        "origin": git("rev-parse", f"origin/{BRANCH}") == head,
        "implementation": preflight.get("implementation_file_sha256")
        == implementation_sha256(),
        "replay": sha256(args.replay_ledger) == REPLAY_LEDGER_SHA256,
        "source": sha256(args.source_checkpoint) == SOURCE_SHA256,
    }
    if not all(code_checks.values()):
        raise SystemExit(f"official training binding failed: {code_checks}")
    ledger = replay_rows(args.replay_ledger)
    source_path = require_exact_file(args.source_checkpoint, SOURCE_SHA256, "source")
    if args.resume_checkpoint:
        model, optimizer, loader, loaded, source = load_new_checkpoint(
            args.resume_checkpoint, source_path, device, restore=True
        )
        start = int(loaded["local_updates"])
        source_step_baseline = loaded["source_optimizer_steps_by_name"]
        if (start, end) != (RESTART_LOCAL_UPDATE, LOCAL_UPDATES):
            raise SystemExit(f"unauthorized resume segment {start}->{end}")
        audit = midpoint_restart(
            args.midpoint_preexit,
            model,
            optimizer,
            loader,
            loaded,
            args.resume_checkpoint,
            source_path,
            device,
        )
        durable_json(output / "MIDPOINT_RESTART_AUDIT.json", audit)
        if not audit["passed"]:
            raise SystemExit(f"midpoint restart failed: {audit['checks']}")
    else:
        if end != RESTART_LOCAL_UPDATE:
            raise SystemExit("official training must begin with segment 0->96")
        model, optimizer, loader, source, checks = make_new_model(
            source_path, device, restore=True
        )
        if not all(checks.values()):
            raise SystemExit("source validation failed at official start")
        start = 0
        source_step_baseline = optimizer_steps_by_name(model, optimizer)
    accumulation = int(source["gradient_accumulation"])
    if accumulation != 16 or int(source["loader_state"]["batch_size"]) != 32:
        raise SystemExit("single-GPU recipe is not microbatch 32 / accumulation 16")
    log_path = output / "TRAINING_LOG.jsonl"
    actual_path = output / "TRAINING_REPLAY_ACTUAL.jsonl"
    existing = (
        [json.loads(line) for line in log_path.read_text().splitlines() if line]
        if log_path.exists()
        else []
    )
    if len(existing) != start:
        raise SystemExit("training log length does not match segment start")
    for local_update in range(start + 1, end + 1):
        if local_update > LOCAL_UPDATES:
            raise SystemExit("hard stop: update 192 is forbidden")
        expected = ledger[local_update - 1]
        batch_hash = base.next_batch_hash(loader, accumulation)
        stream_hash = base.next_stream_hash(loader, accumulation)
        invariants = {
            "local_update": expected["local_update"] == local_update,
            "global_update": expected["global_update"]
            == SOURCE_UPDATES + local_update,
            "cursor": loader.state_dict() == expected["start_cursor"],
            "batch": batch_hash == expected["logical_global_batch_sha256"],
            "stream": stream_hash == expected["logical_global_stream_sha256"],
            "pass": expected["pass_count"]
            == base.pass_count(SOURCE_UPDATES + local_update),
            "targets": expected["target_count"] == base.GLOBAL_TARGETS,
        }
        if not all(invariants.values()):
            raise SystemExit(
                f"replay invariant failed at update {local_update}: {invariants}"
            )
        row = train_update(
            model, optimizer, loader, accumulation, local_update, device
        )
        row.update(
            {
                "batch_sha256": batch_hash,
                "stream_sha256": stream_hash,
                "expected_replay_chain_sha256": expected["chain_sha256"],
                "end_cursor_exact": loader.state_dict() == expected["end_cursor"],
                "lr": {
                    group["name"]: float(group["lr"])
                    for group in optimizer.param_groups
                },
                "process_id": os.getpid(),
                "ce_only": True,
                "pre_forward_invariants": invariants,
            }
        )
        if not row["end_cursor_exact"]:
            raise SystemExit("post-update loader cursor mismatch")
        append_jsonl(log_path, row)
        append_jsonl(
            actual_path,
            {
                "local_update": local_update,
                "global_update": SOURCE_UPDATES + local_update,
                "batch_sha256": batch_hash,
                "stream_sha256": stream_hash,
                "pass_count": row["pass_count"],
                "target_count": base.GLOBAL_TARGETS,
                "chain_sha256": expected["chain_sha256"],
                "process_id": os.getpid(),
            },
        )
        durable_json(
            output / "HEARTBEAT_NEW.json",
            {
                "status": "training",
                "local_update": local_update,
                "global_update": SOURCE_UPDATES + local_update,
                "latest": row,
                "updated_at_unix": time.time(),
            },
        )
        if local_update in (RESTART_LOCAL_UPDATE, LOCAL_UPDATES):
            checkpoint = Path(args.checkpoint_dir) / checkpoint_name(local_update)
            verification = save_checkpoint(
                checkpoint,
                model,
                optimizer,
                loader,
                source,
                local_update,
                REPLAY_LEDGER_SHA256,
                source_path,
                device,
            )
            milestones_path = output / "MILESTONE_CHECKPOINTS.json"
            milestones = read_json(milestones_path) if milestones_path.exists() else {}
            milestones[str(local_update)] = verification
            durable_json(milestones_path, milestones)
            if local_update == RESTART_LOCAL_UPDATE:
                durable_json(
                    output / "MIDPOINT_RESTART_PREEXIT.json",
                    midpoint_preexit(model, optimizer, loader, verification, source),
                )
    if end == RESTART_LOCAL_UPDATE:
        print("EXPERIMENT_2D6_SEGMENT_COMPLETE 0->96 FRESH_PROCESS_REQUIRED", flush=True)
        return
    metrics = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    actual = [json.loads(line) for line in actual_path.read_text().splitlines() if line]
    restart = read_json(output / "MIDPOINT_RESTART_AUDIT.json")
    preexit = read_json(output / "MIDPOINT_RESTART_PREEXIT.json")
    milestones = read_json(output / "MILESTONE_CHECKPOINTS.json")
    steps = optimizer_steps_by_name(model, optimizer)
    active = set(steps) - {"g_rec_b6"}
    checks = {
        "updates_exact": len(metrics) == LOCAL_UPDATES
        and [row["local_update"] for row in metrics] == list(range(1, 192)),
        "targets_exact": len(metrics) * base.GLOBAL_TARGETS == LOCAL_TARGETS,
        "global_update": metrics[-1]["global_update"] == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": metrics[-1]["cumulative_targets"]
        == FINAL_CUMULATIVE_TARGETS,
        "replay_batches": [row["batch_sha256"] for row in actual]
        == [row["logical_global_batch_sha256"] for row in ledger],
        "replay_streams": [row["stream_sha256"] for row in actual]
        == [row["logical_global_stream_sha256"] for row in ledger],
        "cadence": [row["pass_count"] for row in actual]
        == [row["pass_count"] for row in ledger],
        "restart": restart["passed"],
        "two_processes": {row["process_id"] for row in metrics[:96]}
        == {preexit["saved_process_id"]}
        and {row["process_id"] for row in metrics[96:]}
        == {restart["resumed_process_id"]}
        and preexit["saved_process_id"] != restart["resumed_process_id"],
        "active_optimizer_progression": all(
            steps[name] == source_step_baseline[name] + LOCAL_UPDATES
            for name in active
        ),
        "dormant_optimizer_step": steps["g_rec_b6"]
        == source_step_baseline["g_rec_b6"]
        == SOURCE_UPDATES,
        "dormant_gradient_none": model.g_rec_b6.grad is None,
        "milestones_exact": set(milestones) == {"96", "191"},
        "final_next_batch": milestones["191"]["next_global_batch_sha256"]
        == FINAL_NEXT_BATCH,
        "final_next_stream": milestones["191"]["next_global_batch_stream_sha256"]
        == FINAL_NEXT_STREAM,
        "no_update_192": max(row["local_update"] for row in metrics)
        == LOCAL_UPDATES,
    }
    final = milestones["191"]
    complete = {
        "schema": "experiment_2d6_training_complete_v1",
        "experiment": EXPERIMENT,
        "checks": checks,
        "final_checkpoint": final,
        "optimizer_steps": {
            "active_unique": sorted({steps[name] for name in active}),
            "source_active_unique": sorted(
                {source_step_baseline[name] for name in active}
            ),
            "dormant_b6": steps["g_rec_b6"],
        },
        "training_wall_seconds": sum(row["wall_seconds"] for row in metrics),
        "mean_targets_per_second": statistics.fmean(
            row["targets_per_second"] for row in metrics
        ),
        "passed": all(checks.values()),
    }
    durable_json(output / "TRAINING_COMPLETE.json", complete)
    if not complete["passed"]:
        raise SystemExit(f"terminal training audit failed: {checks}")
    print("EXPERIMENT_2D6_TRAINING_COMPLETE 191 UPDATES", flush=True)


def run_evaluate(args):
    require_branch(clean=True)
    device = base.require_a100()
    panel = read_json(args.panel_manifest)
    if (
        panel.get("sequence_count") != 2048
        or panel.get("targets_per_condition") != 2_097_152
    ):
        raise SystemExit("evaluation is not the sealed 2,048-sequence panel")
    source = require_exact_file(args.source_checkpoint, SOURCE_SHA256, "source")
    if args.family == "new":
        if args.control != "all_real" or not args.checkpoint:
            raise SystemExit("New evaluation requires checkpoint and all_real")
        model, optimizer, loader, payload, _ = load_new_checkpoint(
            args.checkpoint, source, device, restore=False
        )
        if int(payload["local_updates"]) != LOCAL_UPDATES:
            raise SystemExit("New evaluation requires update-191 checkpoint")
        checkpoint_sha = sha256(args.checkpoint)
        architecture = core.ARCHITECTURE_FINGERPRINT
        condition = "NEW_REAL"
        reopen = strict_reopen(args.checkpoint, source, device)
        if not reopen["passed"]:
            raise SystemExit("final New checkpoint failed strict reopen")
    else:
        if args.family != "fixed" or args.control not in ("all_real", "b6_off"):
            raise SystemExit("Fixed evaluation permits all_real or b6_off")
        model, optimizer, loader, _, checks = load_fixed_control(
            args.control_checkpoint, source, device
        )
        checkpoint_sha = CONTROL_SHA256
        architecture = "accepted-2d3a-fixed-w512-b7-to-b6-real"
        condition = "FIXED_REAL" if args.control == "all_real" else "FIXED_B6_RECURRENCE_OFF"
        if not all(checks.values()):
            raise SystemExit("Fixed control load failed")
    identity = {
        "condition": condition,
        "checkpoint_sha256": checkpoint_sha,
        "architecture_fingerprint": architecture,
        "panel_manifest_sha256": sha256(args.panel_manifest),
        "panel_sha256": panel["panel_sha256"],
        "evaluation_set_label": "reused sealed matched panel",
        "precision": "BF16 model execution; FP32 CE logits; FP64 accumulation",
        "scoring": "1024 targets per sequence",
    }
    identity["condition_fingerprint"] = canonical_sha(identity)
    result = evaluate_batches(
        model,
        base.validation_path(Path(args.data_root)),
        panel,
        args.control,
        args.output_path,
        identity,
        max_batches=args.max_batches,
    )
    result["checkpoint_sha256"] = checkpoint_sha
    result["panel_manifest_sha256"] = sha256(args.panel_manifest)
    result["condition_fingerprint"] = identity["condition_fingerprint"]
    durable_json(args.output_path, result)
    del model, optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    print(f"EXPERIMENT_2D6_EVALUATION_COMPLETE {condition}", flush=True)


def storage_audit_fixed(state, model):
    audit = model.incremental_cache_audit(state)
    tensors = []
    components = {}
    for index, cache in enumerate(state.caches):
        if cache is None:
            continue
        tensors.extend((cache.key, cache.value))
        components[f"B{index + 1}_local_kv"] = (
            cache.key.numel() * cache.key.element_size()
            + cache.value.numel() * cache.value.element_size()
        )
    for name in ("h7", "h8", "h10", "h12"):
        tensor = getattr(state, f"{name}_ring")
        tensors.append(tensor)
        components[f"{name}_ring"] = tensor.numel() * tensor.element_size()
    unique = {}
    for tensor in tensors:
        raw = tensor.untyped_storage()
        unique[(str(tensor.device), raw.data_ptr(), raw.nbytes())] = raw.nbytes()
    return {
        "logical_bytes": sum(t.numel() * t.element_size() for t in tensors),
        "physical_unique_bytes": sum(unique.values()),
        "components": components,
        "cache_audit": audit,
        "b7_ring_present": True,
    }


def storage_audit_new(state, model):
    audit = model.incremental_cache_audit(state)
    components = {}
    for index, cache in enumerate(state.caches):
        if cache is None:
            continue
        components[f"B{index + 1}_local_kv"] = (
            cache.key.numel() * cache.key.element_size()
            + cache.value.numel() * cache.value.element_size()
        )
    for name in ("h8", "h10", "h12"):
        tensor = getattr(state, f"{name}_ring")
        components[f"{name}_ring"] = tensor.numel() * tensor.element_size()
    return {
        "logical_bytes": audit["logical_payload_bytes"],
        "physical_unique_bytes": audit["actual_unique_storage_bytes"],
        "components": components,
        "cache_audit": audit,
        "b7_ring_present": False,
    }


def run_one_incremental(model, tokens, control="all_real"):
    state = model.init_incremental_state(1, device=tokens.device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            _, state = model.incremental_step(
                tokens[:, position], state, control=control
            )
    return state


def benchmark_incremental(model, tokens, control, device, repeats=3):
    run_one_incremental(model, tokens, control)
    torch.cuda.synchronize(device)
    durations = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        state = run_one_incremental(model, tokens, control)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        del state
        torch.cuda.empty_cache()
    median = statistics.median(durations)
    return {
        "warmup_runs": 1,
        "timed_repeats": int(repeats),
        "sequence_length": int(tokens.size(1)),
        "batch_size": 1,
        "durations_seconds": durations,
        "median_latency_seconds": median,
        "median_tokens_per_second": tokens.numel() / median,
    }


def run_memory_speed(args):
    require_branch(clean=True)
    device = base.require_a100()
    source = require_exact_file(args.source_checkpoint, SOURCE_SHA256, "source")
    new, new_optimizer, new_loader, payload, _ = load_new_checkpoint(
        args.new_checkpoint, source, device, restore=False
    )
    fixed, fixed_optimizer, fixed_loader, _, _ = load_fixed_control(
        args.control_checkpoint, source, device
    )
    if int(payload["local_updates"]) != LOCAL_UPDATES:
        raise SystemExit("memory/speed audit requires final New checkpoint")
    tokens = (torch.arange(base.T, device=device) * 7919 + 17).remainder(50_257).view(1, -1)
    new.eval()
    fixed.eval()
    fixed_state = run_one_incremental(fixed, tokens)
    fixed_memory = storage_audit_fixed(fixed_state, fixed)
    del fixed_state
    torch.cuda.empty_cache()
    new_state = run_one_incremental(new, tokens)
    new_memory = storage_audit_new(new_state, new)
    del new_state
    torch.cuda.empty_cache()
    fixed_speed = benchmark_incremental(fixed, tokens, "all_real", device)
    new_speed = benchmark_incremental(new, tokens, "all_real", device)
    logical_delta = new_memory["logical_bytes"] - fixed_memory["logical_bytes"]
    physical_delta = (
        new_memory["physical_unique_bytes"] - fixed_memory["physical_unique_bytes"]
    )
    result = {
        "schema": "experiment_2d6_memory_speed_audit_v1",
        "deployment_equivalent": {
            "batch_size": 1,
            "sequence_length": 1024,
            "persistent_dtype": "torch.bfloat16",
        },
        "fixed": {"memory": fixed_memory, "speed": fixed_speed},
        "new": {"memory": new_memory, "speed": new_speed},
        "new_minus_fixed": {
            "logical_persistent_bytes": logical_delta,
            "physical_unique_bytes": physical_delta,
            "latency_seconds": new_speed["median_latency_seconds"]
            - fixed_speed["median_latency_seconds"],
            "latency_percent": 100
            * (
                new_speed["median_latency_seconds"]
                / fixed_speed["median_latency_seconds"]
                - 1
            ),
            "tokens_per_second": new_speed["median_tokens_per_second"]
            - fixed_speed["median_tokens_per_second"],
            "throughput_percent": 100
            * (
                new_speed["median_tokens_per_second"]
                / fixed_speed["median_tokens_per_second"]
                - 1
            ),
        },
        "checks": {
            "fixed_b7_ring_present": fixed_memory["b7_ring_present"] is True,
            "new_b7_ring_absent": new_memory["b7_ring_present"] is False,
            "fixed_physical_exact": fixed_memory["logical_bytes"]
            == fixed_memory["physical_unique_bytes"],
            "new_physical_exact": new_memory["logical_bytes"]
            == new_memory["physical_unique_bytes"],
            "three_repeats_each": len(fixed_speed["durations_seconds"])
            == len(new_speed["durations_seconds"])
            == 3,
        },
    }
    result["passed"] = all(result["checks"].values())
    durable_json(args.output_path, result)
    if not result["passed"]:
        raise SystemExit("memory/speed audit failed")
    del new, new_optimizer, new_loader, fixed, fixed_optimizer, fixed_loader
    gc.collect()
    torch.cuda.empty_cache()
    print("EXPERIMENT_2D6_MEMORY_SPEED_COMPLETE", flush=True)


def run_final_reopen(args):
    require_branch(clean=True)
    device = base.require_a100()
    reopen = strict_reopen(args.checkpoint, args.source_checkpoint, device)
    checkpoint_sha = sha256(args.checkpoint)
    output = {
        "schema": "experiment_2d6_final_checkpoint_provenance_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
        "strict_reopen": reopen,
        "passed": reopen["passed"],
    }
    durable_json(args.output_path, output)
    if not output["passed"]:
        raise SystemExit("final checkpoint strict reopen failed")
    print("EXPERIMENT_2D6_FINAL_REOPEN_PASS", flush=True)


def paired_bootstrap(arrays, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    names = list(arrays)
    length = len(next(iter(arrays.values())))
    if length != 2048 or any(len(values) != length for values in arrays.values()):
        raise SystemExit("paired bootstrap requires 2,048 aligned sequences")
    generator = np.random.default_rng(int(seed))
    distributions = {name: np.empty(int(resamples), dtype=np.float64) for name in names}
    batch = 250
    cursor = 0
    while cursor < int(resamples):
        count = min(batch, int(resamples) - cursor)
        indices = generator.integers(0, length, size=(count, length), dtype=np.int32)
        for name, values in arrays.items():
            distributions[name][cursor : cursor + count] = np.asarray(values)[indices].mean(axis=1)
        cursor += count
    result = {}
    for name, values in arrays.items():
        distribution = distributions[name]
        result[name] = {
            "estimate": float(np.mean(values)),
            "lower_95": float(np.percentile(distribution, 2.5)),
            "upper_95": float(np.percentile(distribution, 97.5)),
        }
    return {
        "method": "paired per-sequence percentile bootstrap",
        "seed": int(seed),
        "resamples": int(resamples),
        "paired_sequences": length,
        "identical_resample_indices_all_contrasts": True,
        "contrasts": result,
    }


def interval_overlaps(row, boundary):
    return row["lower_95"] <= boundary <= row["upper_95"]


def classification(primary_penalty):
    borderline = any(
        interval_overlaps(primary_penalty, boundary)
        for boundary in (-DELTA_CE, 0.0, DELTA_CE)
    )
    if borderline:
        label = "B6 REPRESENTATION COMPARISON UNRESOLVED"
        reason = "reused-panel CI overlaps zero or a binding practical boundary"
    elif primary_penalty["upper_95"] < 0.0:
        label = "B6 NATIVE W1024 SUPERIOR; B7→B6 RECURRENCE NOT JUSTIFIED"
        reason = "complete Fixed-minus-New CI is above zero"
    elif primary_penalty["lower_95"] > DELTA_CE:
        label = "B7→B6 RECURRENCE PROVIDES MATERIAL REPRESENTATION ADVANTAGE"
        reason = "complete New-minus-Fixed CI is above +0.0001"
    elif (
        primary_penalty["lower_95"] > 0.0
        and primary_penalty["upper_95"] < DELTA_CE
    ):
        label = "SMALL RECURRENT QUALITY ADVANTAGE; NATIVE W1024 REMAINS PRACTICALLY NONINFERIOR"
        reason = "small Fixed advantage is wholly inside the +0.0001 margin"
    elif (
        primary_penalty["lower_95"] > -DELTA_CE
        and primary_penalty["upper_95"] < DELTA_CE
    ):
        label = "B6 NATIVE W1024 PRACTICALLY EQUIVALENT; PREFER SIMPLER PATH IF SPEED IS NOT WORSE"
        reason = "complete two-sided CI lies inside ±0.0001"
    elif primary_penalty["upper_95"] < DELTA_CE:
        label = "B6 NATIVE W1024 PRACTICALLY NONINFERIOR"
        reason = "upper penalty bound is below +0.0001"
    else:
        label = "B6 REPRESENTATION COMPARISON UNRESOLVED"
        reason = "binding superiority/noninferiority/equivalence rules are not established"
    return {
        "classification": label,
        "reason": reason,
        "delta_ce": DELTA_CE,
        "borderline_reused_panel_rule_applied": borderline,
        "fresh_panel_confirmation_needed": borderline,
    }


def render_report(summary):
    primary = summary["bootstrap"]["contrasts"]["new_minus_fixed_penalty"]
    d = summary["bootstrap"]["contrasts"]["fixed_minus_new"]
    off = summary["bootstrap"]["contrasts"]["fixed_off_minus_fixed_real"]
    memory = summary["memory_speed"]["new_minus_fixed"]
    shock = summary["zero_training_shock"]
    classification_row = summary["classification"]
    fresh = "Yes" if classification_row["fresh_panel_confirmation_needed"] else "No"
    further = "No further training is warranted."
    return f"""# Experiment 2D6 Final Report

## Result

**{classification_row['classification']}**

Evaluation set: **reused sealed matched panel** (2,048 paired sequences; 2,097,152 targets per condition).

- Fixed REAL CE: `{summary['fixed_real_ce']:.12f}`
- New REAL CE: `{summary['new_real_ce']:.12f}`
- Fixed − New: `{d['estimate']:+.12f}`; paired 95% CI `[{d['lower_95']:+.12f}, {d['upper_95']:+.12f}]`
- New penalty (New − Fixed): `{primary['estimate']:+.12f}`; paired 95% CI `[{primary['lower_95']:+.12f}, {primary['upper_95']:+.12f}]`
- Binding practical margin: `delta_CE = {DELTA_CE}`
- Zero-training geometry shock (New geometry − Original): `{shock['parent_new_minus_original_ce']:+.12f}`
- Fixed B6-OFF effect (OFF − REAL): `{off['estimate']:+.12f}`; paired 95% CI `[{off['lower_95']:+.12f}, {off['upper_95']:+.12f}]`
- Persistent state, New − Fixed: `{memory['physical_unique_bytes']:+,d}` bytes physical (`{memory['logical_persistent_bytes']:+,d}` logical)
- Median latency, New − Fixed: `{memory['latency_seconds']:+.6f}` s (`{memory['latency_percent']:+.3f}%`)
- Median throughput, New − Fixed: `{memory['tokens_per_second']:+.3f}` token/s (`{memory['throughput_percent']:+.3f}%`)
- Final checkpoint SHA-256: `{summary['final_checkpoint']['sha256']}`
- Audit: `{summary['audit_status']}`
- Git branch: `{BRANCH}`
- Git tag: `{FINAL_TAG}`
- Pod status: `STOPPED`; volume retained

## Answers

1. Native B6 W1024 is classified as: **{classification_row['classification']}**.
2. B7→B6 inside mature W512 changes CE by `{off['estimate']:+.12f}` when removed (OFF − REAL); positive means recurrence helps.
3. The approximately equal-memory native architecture is preferable only if the classification and measured speed support it; the measured physical memory delta is `{memory['physical_unique_bytes']:+,d}` bytes.
4. Exact tradeoff: quality `{primary['estimate']:+.12f}` CE penalty, memory `{memory['physical_unique_bytes']:+,d}` bytes, throughput `{memory['throughput_percent']:+.3f}%`.
5. Fresh-panel confirmation needed: **{fresh}**.
6. {further}

The Fixed stored losses were reused only after sentinel reproduction. This was not a fresh confirmation set.

STOPPED AFTER ONE NEW ARM AT EXACTLY 191 UPDATES / 100,139,008 TARGETS
"""


def run_analyze(args):
    fixed_stored = read_json(args.fixed_stored)
    sentinel = read_json(args.fixed_sentinel)
    new = read_json(args.new_real)
    fixed_off = read_json(args.fixed_off)
    shock = read_json(args.zero_training_shock)
    memory = read_json(args.memory_speed)
    final_checkpoint = read_json(args.final_checkpoint_provenance)
    training = read_json(args.training_complete)
    stop = read_json(args.stop_verification)
    fixed_row = fixed_stored["conditions"]["all_real"]
    fixed_values = np.asarray(fixed_row["per_sequence_ce"], dtype=np.float64)
    new_values = np.asarray(new["per_sequence_ce"], dtype=np.float64)
    off_values = np.asarray(fixed_off["per_sequence_ce"], dtype=np.float64)
    sentinel_expected = fixed_values[: len(sentinel["per_sequence_ce"])]
    sentinel_actual = np.asarray(sentinel["per_sequence_ce"], dtype=np.float64)
    sentinel_max_abs = float(np.max(np.abs(sentinel_expected - sentinel_actual)))
    fixed_ce = float(fixed_values.mean())
    new_ce = float(new_values.mean())
    fixed_off_ce = float(off_values.mean())
    arrays = {
        "fixed_minus_new": fixed_values - new_values,
        "new_minus_fixed_penalty": new_values - fixed_values,
        "fixed_off_minus_fixed_real": off_values - fixed_values,
        "fixed_off_minus_new": off_values - new_values,
    }
    bootstrap = paired_bootstrap(arrays)
    classification_row = classification(
        bootstrap["contrasts"]["new_minus_fixed_penalty"]
    )
    losses = {
        "schema": "experiment_2d6_large_paired_losses_v1",
        "evaluation_set_label": "reused sealed matched panel",
        "panel_sha256": new["panel_sha256"],
        "paired_sequences": 2048,
        "targets_per_condition": 2_097_152,
        "fixed_real": fixed_values.tolist(),
        "new_real": new_values.tolist(),
        "fixed_b6_recurrence_off": off_values.tolist(),
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    durable_json(output / "LARGE_PAIRED_LOSSES.json", losses)
    durable_json(output / "PAIRED_BOOTSTRAP.json", bootstrap)
    checks = {
        "training_complete": training.get("passed") is True,
        "new_complete": new.get("passed") is True
        and new.get("paired_sequences") == 2048,
        "fixed_off_complete": fixed_off.get("passed") is True
        and fixed_off.get("paired_sequences") == 2048,
        "stored_fixed_complete": fixed_stored.get("passed") is True
        and len(fixed_values) == 2048,
        "fixed_ce_anchor": abs(fixed_ce - FIXED_EXPECTED_CE) < 5e-13,
        "fixed_sentinel_reproduced": sentinel_max_abs <= 1e-12,
        "panel_exact": fixed_stored["panel_sha256"]
        == new["panel_sha256"]
        == fixed_off["panel_sha256"],
        "pairing_exact": len(fixed_values) == len(new_values) == len(off_values) == 2048,
        "bootstrap": bootstrap["resamples"] == BOOTSTRAP_RESAMPLES,
        "memory_speed": memory.get("passed") is True,
        "zero_training_shock": shock.get("passed") is True,
        "final_checkpoint_reopen": final_checkpoint.get("passed") is True,
        "local_remote_checkpoint_sha": final_checkpoint["checkpoint_sha256"]
        == args.local_checkpoint_sha256,
        "pod_stopped": stop.get("pod", {}).get("desiredStatus") == "EXITED"
        and stop.get("pod", {}).get("runtimeStatus") == "stopped",
        "volume_retained": stop.get("network_volume_retained") is True,
    }
    audit_status = "PASS" if all(checks.values()) else "INVALID — NO SCIENTIFIC CONCLUSION"
    if audit_status != "PASS":
        classification_row = {
            **classification_row,
            "classification": "INVALID — NO SCIENTIFIC CONCLUSION",
            "reason": "critical scientific or operational audit failed",
        }
    final_checkpoint_row = {
        "path": args.local_checkpoint_path,
        "sha256": args.local_checkpoint_sha256,
    }
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification_row,
        "fixed_real_ce": fixed_ce,
        "new_real_ce": new_ce,
        "fixed_b6_off_ce": fixed_off_ce,
        "bootstrap": bootstrap,
        "zero_training_shock": shock,
        "memory_speed": memory,
        "fixed_sentinel_max_abs_ce": sentinel_max_abs,
        "final_checkpoint": final_checkpoint_row,
        "audit_status": audit_status,
        "git_branch": BRANCH,
        "git_tag": FINAL_TAG,
        "pod_status": "STOPPED",
    }
    durable_json(output / "SCIENTIFIC_RESULT_SUMMARY.json", summary)
    final_audit = {
        "schema": "experiment_2d6_final_audit_v1",
        "experiment": EXPERIMENT,
        "checks": checks,
        "audit_status": audit_status,
        "passed": all(checks.values()),
    }
    durable_json(output / "FINAL_AUDIT.json", final_audit)
    durable_text(output / "EXPERIMENT_2D6_FINAL_REPORT.md", render_report(summary))
    if not final_audit["passed"]:
        raise SystemExit(f"final 2D6 audit failed: {checks}")
    print(f"EXPERIMENT_2D6_ANALYSIS_COMPLETE {classification_row['classification']}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Experiment 2D6 lean scientific driver")
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.set_defaults(handler=run_preflight)
    for name in (
        "output_dir",
        "source_checkpoint",
        "control_checkpoint",
        "replay_ledger",
        "replay_audit",
        "core_panel",
        "large_panel",
        "fixed_large",
        "data_root",
        "stop_status_json",
        "pod_id",
        "pod_name",
        "volume_id",
    ):
        preflight.add_argument(f"--{name.replace('_', '-')}", required=True)

    train = commands.add_parser("train")
    train.set_defaults(handler=run_train)
    for name in (
        "output_dir",
        "preflight_authorization",
        "replay_ledger",
        "source_checkpoint",
        "checkpoint_dir",
    ):
        train.add_argument(f"--{name.replace('_', '-')}", required=True)
    train.add_argument("--end-local-update", required=True, type=int)
    train.add_argument("--resume-checkpoint")
    train.add_argument("--midpoint-preexit")

    evaluate = commands.add_parser("evaluate")
    evaluate.set_defaults(handler=run_evaluate)
    evaluate.add_argument("--family", required=True, choices=("new", "fixed"))
    evaluate.add_argument("--control", required=True, choices=("all_real", "b6_off"))
    for name in (
        "source_checkpoint",
        "control_checkpoint",
        "panel_manifest",
        "data_root",
        "output_path",
    ):
        evaluate.add_argument(f"--{name.replace('_', '-')}", required=True)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--max-batches", type=int)

    memory = commands.add_parser("memory-speed")
    memory.set_defaults(handler=run_memory_speed)
    for name in (
        "source_checkpoint",
        "control_checkpoint",
        "new_checkpoint",
        "output_path",
    ):
        memory.add_argument(f"--{name.replace('_', '-')}", required=True)

    reopen = commands.add_parser("final-reopen")
    reopen.set_defaults(handler=run_final_reopen)
    for name in ("checkpoint", "source_checkpoint", "output_path"):
        reopen.add_argument(f"--{name.replace('_', '-')}", required=True)

    analyze = commands.add_parser("analyze")
    analyze.set_defaults(handler=run_analyze)
    for name in (
        "fixed_stored",
        "fixed_sentinel",
        "new_real",
        "fixed_off",
        "zero_training_shock",
        "memory_speed",
        "final_checkpoint_provenance",
        "training_complete",
        "stop_verification",
        "local_checkpoint_path",
        "local_checkpoint_sha256",
        "output_dir",
    ):
        analyze.add_argument(f"--{name.replace('_', '-')}", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        if args.end_local_update == RESTART_LOCAL_UPDATE and (
            args.resume_checkpoint or args.midpoint_preexit
        ):
            raise SystemExit("segment 0->96 cannot accept resume artifacts")
        if args.end_local_update == LOCAL_UPDATES and not (
            args.resume_checkpoint and args.midpoint_preexit
        ):
            raise SystemExit("segment 96->191 requires midpoint artifacts")
    args.handler(args)


if __name__ == "__main__":
    main()
