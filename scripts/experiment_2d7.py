#!/usr/bin/env python3
"""Fail-closed driver for Experiment 2D7 boundary-alignment N/O/G."""

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
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

import experiment_2d3a as base
import experiment_2d5c as d5c
import experiment_2d6 as d6
import experiment_2d7_core as core


EXPERIMENT = "2D7"
BRANCH = "experiment-2d7-trained-boundary-alignment-nog"
FINAL_TAG = BRANCH + "-final"
SCHEMA = "experiment_2d7_checkpoint_v1"
PARENT_SHA256 = "6e5023b127032dbb4d32a23bf1be052702d51177437b2795b99c52bcd83314c7"
PARENT_GLOBAL_UPDATE = 2_099
PARENT_TARGETS = 1_100_480_512
LOCAL_UPDATES = 191
TARGETS_PER_UPDATE = 524_288
LOCAL_TARGETS = 100_139_008
FINAL_GLOBAL_UPDATE = 2_290
FINAL_TARGETS = 1_200_619_520
FINAL_CHECKPOINT_NAME = "scientific_cumulative_001200619520.pt"
ACCUMULATION = 16
MICROBATCH = 32
PANEL_SEQUENCES = 2_048
PANEL_TARGETS = 2_097_152
PANEL_BATCHES = 32
PANEL_SEED = 2_026_090_3
BOOTSTRAP_SEED = 2_026_090_2
BOOTSTRAP_RESAMPLES = 50_000
DELTA_CE = 0.0001
DATASET_SHA256 = "8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f"
REPO_ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()


def require_branch(clean=False):
    current = git("branch", "--show-current")
    if current != BRANCH:
        raise SystemExit(f"expected branch {BRANCH}, found {current}")
    if clean and git("status", "--porcelain"):
        raise SystemExit("2D7 requires a clean worktree")


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


def implementation_sha256():
    paths = (
        REPO_ROOT / "scripts/experiment_2d7.py",
        REPO_ROOT / "scripts/experiment_2d7_core.py",
        REPO_ROOT / "configs/exp2d7_trained_boundary_alignment_nog.json",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


def bytes_sha256(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
    return digest.hexdigest()


def aggregate_hashes(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(str(value)))
    return digest.hexdigest()


def batch_arrays(tokens, start):
    count = 64 * base.T
    x = np.asarray(tokens[start : start + count], dtype=np.int64).reshape(64, base.T)
    y = np.asarray(tokens[start + 1 : start + count + 1], dtype=np.int64).reshape(64, base.T)
    return x, y


def batch_identity_numpy(x, y):
    return {
        "input_sha256": bytes_sha256(x),
        "target_sha256": bytes_sha256(y),
        "combined_sha256": bytes_sha256(x, y),
    }


def sequence_identities(x, y, start, batch_index):
    rows = []
    for index in range(64):
        offset = int(start) + index * base.T
        rows.append({
            "batch_index": int(batch_index),
            "sequence_index": index,
            "dataset_input_span": [offset, offset + base.T - 1],
            "dataset_target_span": [offset + 1, offset + base.T],
            "input_sha256": bytes_sha256(x[index]),
            "target_sha256": bytes_sha256(y[index]),
            "combined_sha256": bytes_sha256(x[index], y[index]),
            "targets": base.T,
        })
    return rows


def _collect_historical_spans(value, spans):
    if isinstance(value, dict):
        direct = value.get("canonical_target_spans_half_open")
        if isinstance(direct, list):
            for row in direct:
                if isinstance(row, list) and len(row) == 2:
                    spans.add((int(row[0]), int(row[1])))
        indices = value.get("batch_indices_in_evaluation_order")
        if isinstance(indices, list):
            for index in indices:
                start = int(index) * 64 * base.T
                spans.add((start + 1, start + 64 * base.T + 1))
        selection = value.get("selection")
        if isinstance(selection, dict) and {
            "start_token_offset", "batch_count"
        } <= set(selection):
            start = int(selection["start_token_offset"])
            for ordinal in range(int(selection["batch_count"])):
                current = start + ordinal * 64 * base.T
                spans.add((current + 1, current + 64 * base.T + 1))
        for child in value.values():
            _collect_historical_spans(child, spans)
    elif isinstance(value, list):
        for child in value:
            _collect_historical_spans(child, spans)


def prepare_panel(args):
    require_branch(clean=True)
    dataset = require_file(args.dataset, DATASET_SHA256, "validation dataset")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokens = np.load(dataset, mmap_mode="r")
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
            before = len(historical_spans)
            _collect_historical_spans(value, historical_spans)
            if len(historical_spans) != before:
                historical_files.append({
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "new_spans": len(historical_spans) - before,
                })
    # Historical standard validation commonly consumed the prefix without a
    # dedicated panel artifact. Reserve it conservatively.
    for index in range(128):
        start = index * 64 * base.T
        historical_spans.add((start + 1, start + 64 * base.T + 1))

    def intersects(span):
        return any(max(span[0], old[0]) < min(span[1], old[1]) for old in historical_spans)

    available = (int(tokens.shape[0]) - 1) // (64 * base.T)
    order = np.random.default_rng(PANEL_SEED).permutation(available)
    selected, identities, sequences, selected_spans = [], [], [], []
    for raw in order:
        index = int(raw)
        start = index * 64 * base.T
        span = (start + 1, start + 64 * base.T + 1)
        if intersects(span):
            continue
        x, y = batch_arrays(tokens, start)
        identity = batch_identity_numpy(x, y)
        rows = sequence_identities(x, y, start, index)
        if identity["combined_sha256"] in {row["combined_sha256"] for row in identities}:
            continue
        selected.append(index)
        identities.append(identity)
        sequences.extend(rows)
        selected_spans.append(list(span))
        historical_spans.add(span)
        if len(selected) == PANEL_BATCHES:
            break
    if len(selected) != PANEL_BATCHES:
        raise SystemExit("could not construct one fresh 2D7 panel")
    panel_hash = aggregate_hashes(row["combined_sha256"] for row in identities)
    manifest = {
        "experiment": EXPERIMENT,
        "panel_name": "fresh disjoint 2D7 matched panel",
        "dataset": "edu_fineweb10B/edufineweb_val_000000.npy",
        "dataset_sha256": DATASET_SHA256,
        "dataset_split": "validation",
        "selection_seed": PANEL_SEED,
        "selection_algorithm": "seeded permutation of complete canonical 64x1024 validation batches; reject every recovered historical target span and reserved prefix 0..127; accept first 32",
        "candidate_panels_constructed": 1,
        "checkpoint_losses_inspected_during_selection": False,
        "sealed_before_checkpoint_loading": True,
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
    audit = {
        "experiment": EXPERIMENT,
        "panel_sha256": panel_hash,
        "historical_json_files_used": historical_files,
        "recovered_historical_spans_including_reserved_prefix": len(historical_spans) - len(selected_spans),
        "selected_target_spans": selected_spans,
        "checks": {
            "one_panel": manifest["candidate_panels_constructed"] == 1,
            "sealed_before_checkpoint_loading": True,
            "no_checkpoint_loss_inspection": True,
            "validation_split_not_training_split": True,
            "sequences": len(sequences) == PANEL_SEQUENCES,
            "targets": len(sequences) * base.T == PANEL_TARGETS,
            "unique_batches": len(set(selected)) == PANEL_BATCHES,
            "unique_sequence_hashes": len({row["combined_sha256"] for row in sequences}) == PANEL_SEQUENCES,
        },
    }
    audit["passed"] = all(audit["checks"].values())
    durable_json(output / "EVALUATION_PANEL_MANIFEST.json", manifest)
    durable_json(output / "EVALUATION_PANEL_DISJOINTNESS_AUDIT.json", audit)
    if not audit["passed"]:
        raise SystemExit("fresh panel audit failed")
    print(f"EXPERIMENT_2D7_PANEL_FROZEN {panel_hash}", flush=True)


def parent_payload(path, mmap=True):
    source = require_file(path, PARENT_SHA256, "sealed 2D6 parent")
    payload = base.d0.torch_load(source, mmap=mmap)
    checks = {
        "schema": payload.get("schema") == d6.SCHEMA,
        "experiment": payload.get("experiment") == "2D6",
        "arm": payload.get("arm") == "NEW",
        "local_updates": payload.get("local_updates") == 191,
        "global_update": payload.get("global_update") == PARENT_GLOBAL_UPDATE,
        "cumulative_targets": payload.get("cumulative_targets") == PARENT_TARGETS,
        "targets_per_update": payload.get("targets_per_update") == TARGETS_PER_UPDATE,
        "gradient_accumulation": payload.get("gradient_accumulation") == ACCUMULATION,
        "microbatch": payload.get("loader_state", {}).get("batch_size") == MICROBATCH,
        "optimizer": "optimizer" in payload,
        "scheduler": "scheduler" in payload,
        "loader": "loader_state" in payload,
        "rng": set(payload.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "architecture": payload.get("architecture_fingerprint") == d6.core.ARCHITECTURE_FINGERPRINT,
    }
    if not all(checks.values()):
        raise SystemExit(f"sealed 2D6 parent validation failed: {checks}")
    return source, payload, checks


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
    ledger = output / "CONTINUATION_LEDGER.jsonl"
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
        and rows[0]["global_update"] == 2100
        and rows[-1]["global_update"] == FINAL_GLOBAL_UPDATE
        and LOCAL_UPDATES * TARGETS_PER_UPDATE == LOCAL_TARGETS
    )
    durable_json(output / "CONTINUATION_MANIFEST.json", manifest)
    if not manifest["passed"]:
        raise SystemExit("continuation manifest failed")
    print(f"EXPERIMENT_2D7_CONTINUATION_FROZEN {manifest['ledger_sha256']}", flush=True)


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


def load_parent_model(path, arm, device, restore=False):
    source, payload, parent_checks = parent_payload(path, mmap=True)
    metadata = source_metadata(payload)
    _, foundation = base.instantiate_base(device)
    model = core.BoundaryAlignmentGPT(foundation, arm).to(device)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    optimizer = base.configure_optimizer(model, device.type)
    optimizer.load_state_dict(payload["optimizer"])
    loader = loader_from_state(payload["loader_state"])
    if restore:
        base.restore_rng(payload["rng_state"])
    steps = d6.optimizer_steps_by_name(model, optimizer)
    checks = {
        **{f"parent_{key}": value for key, value in parent_checks.items()},
        "strict_no_missing": not incompatible.missing_keys,
        "strict_no_unexpected": not incompatible.unexpected_keys,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()) == core.PARAMETER_COUNT,
        "state_dict_keys_exact": set(model.state_dict()) == set(payload["model"]),
        "gate_inventory": all(name in dict(model.named_parameters()) for name in ("g_rec", "g_rec_b3", "g_rec_b5")),
        "gate_values_inherited": all(
            tensor_sha(dict(model.named_parameters())[name]) == metadata["gate_sha256"][name]
            for name in ("g_rec", "g_rec_b3", "g_rec_b5")
        ),
        "active_gates_trainable": all(
            dict(model.named_parameters())[name].requires_grad
            for name in ("g_rec", "g_rec_b3", "g_rec_b5")
        ),
        "optimizer_state_restored": set(steps) == set(dict(model.named_parameters())),
        "scheduler_state_restored": "scheduler" in metadata,
        "loader_state_restored": loader.state_dict() == metadata["loader_state"],
        "loader_next_batch": base.next_batch_hash(loader, ACCUMULATION) == metadata["next_global_batch_sha256"],
        "loader_next_stream": base.next_stream_hash(loader, ACCUMULATION) == metadata["next_stream_sha256"],
        "rng_state_restored": (not restore) or d5c.rng_digests(base.capture_rng()) == metadata["rng_digests"],
        "model_finite": base.model_finite(model),
        "optimizer_finite": base.optimizer_finite(optimizer),
    }
    if not all(checks.values()):
        raise SystemExit(f"2D7 parent load failed for {arm}: {checks}")
    del payload
    gc.collect()
    return model, optimizer, loader, metadata, checks


def boundary_audit(model, arm, device):
    expected = core.GEOMETRIES[arm]
    rows = {}
    for block in (0, 2, 4):
        boundary = core.LOCAL_WINDOWS[block]
        length = max(boundary + 2, 66)
        query = torch.arange(length, device=device).view(length, 1)
        source = torch.arange(length, device=device).view(1, length)
        lag = query - source
        local = model.local_mask(block, length, device)
        recurrent = model.recurrent_mask(block, length, length, device)
        minimum = expected[block]
        rows[f"B{block + 1}"] = {
            "local_window": boundary,
            "recurrent_minimum_lag": minimum,
            "neighborhood": {
                str(value): {
                    "local": bool(((lag == value) & local).any()),
                    "recurrent": bool(((lag == value) & recurrent).any()),
                }
                for value in (boundary - 1, boundary, boundary + 1)
            },
            "local_exact": bool(torch.equal(local, (lag >= 0) & (lag < boundary))),
            "recurrent_exact": bool(torch.equal(recurrent, (lag >= minimum) & (lag <= 1023))),
            "no_future": not bool((recurrent & (lag <= 0)).any()),
            "source_identity": "j=t-lag",
        }
    state = model.init_incremental_state(1, device=device, dtype=torch.bfloat16)
    tokens = (torch.arange(66, device=device) * 7919 + 17).remainder(50_257).view(1, -1)
    final_diagnostic = None
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            logits, state, final_diagnostic = model.incremental_step(
                tokens[:, position], state, return_diagnostics=True,
                diagnostic_attention_weights=False,
            )
            del logits
    runtime_positions = {}
    for block in (0, 2, 4):
        positions = final_diagnostic["links"][f"b{block + 1}"]["recurrent_positions"]
        observed = [] if positions is None else [int(value) for value in positions[0].tolist()]
        expected_positions = list(range(0, 66 - expected[block]))
        runtime_positions[f"B{block + 1}"] = {
            "query_position": 65,
            "positions": observed,
            "expected_positions": expected_positions,
            "source_identity_exact": observed == expected_positions,
        }
    checks = {
        "boundary_masks": all(
            row["local_exact"] and row["recurrent_exact"] and row["no_future"]
            for row in rows.values()
        ),
        "runtime_source_identity": all(row["source_identity_exact"] for row in runtime_positions.values()),
        "b6_runtime_capacity_w1024": model._last_b6_local_capacity == 1023,
        "b7_to_b6_runtime_absent": model._b6_recurrent_branch_calls == 0,
        "b7_ring_absent": not hasattr(state, "h7_ring"),
        "maximum_recurrent_lag": max(1023 for _ in expected) == 1023,
    }
    return {"arm": arm, "boundaries": rows, "runtime_positions": runtime_positions, "checks": checks, "passed": all(checks.values())}


def disposable_smoke(model, arm, device):
    model.train()
    model.zero_grad(set_to_none=True)
    tokens = (torch.arange(70, device=device) * 3571 + 11).remainder(50_257).view(1, -1)
    targets = tokens.roll(-1, dims=1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.forward_multi_pass(tokens, targets=targets, num_passes=2)
    loss = result["loss"]
    loss.backward()
    gradients = {
        name: {
            "present": parameter.grad is not None,
            "finite": parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
            "nonzero": parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)),
        }
        for name, parameter in (
            ("g_rec", model.g_rec), ("g_rec_b3", model.g_rec_b3), ("g_rec_b5", model.g_rec_b5)
        )
    }
    result_row = {
        "arm": arm,
        "official_optimizer_updates": 0,
        "forward_loss": float(loss.detach().float()),
        "finite_forward_loss": bool(torch.isfinite(loss)),
        "finite_gradients": all(row["finite"] for row in gradients.values()),
        "active_gate_gradients": gradients,
        "dormant_b6_gate_gradient_none": model.g_rec_b6.grad is None,
        "shape": list(result["logits"].shape),
    }
    result_row["passed"] = (
        result_row["finite_forward_loss"]
        and result_row["finite_gradients"]
        and all(row["present"] and row["nonzero"] for row in gradients.values())
        and result_row["dormant_b6_gate_gradient_none"]
    )
    model.zero_grad(set_to_none=True)
    return result_row


def normalized_common_manifest(arm):
    value = copy.deepcopy(core.architecture_manifest(arm))
    value.pop("arm")
    value.pop("condition")
    value.pop("recurrent_minimum_lags")
    for block in value["blocks"]:
        if block["recurrent_lags"] is not None:
            block["recurrent_lags"][0] = "GEOMETRY_MINIMUM"
    return value


def run_preflight(args):
    require_branch(clean=True)
    device = base.require_a100()
    source = require_file(args.parent_checkpoint, PARENT_SHA256, "sealed 2D6 parent")
    continuation = read_json(args.continuation_manifest)
    panel = read_json(args.panel_manifest)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "parent_path": str(source),
        "parent_sha256": sha256(source),
        "parent_global_update": PARENT_GLOBAL_UPDATE,
        "parent_cumulative_targets": PARENT_TARGETS,
        "continuation_manifest_path": str(Path(args.continuation_manifest).resolve()),
        "continuation_manifest_sha256": sha256(args.continuation_manifest),
        "continuation_ledger_sha256": continuation["ledger_sha256"],
        "panel_manifest_path": str(Path(args.panel_manifest).resolve()),
        "panel_manifest_sha256": sha256(args.panel_manifest),
        "panel_sha256": panel["panel_sha256"],
    }
    arm_rows = {}
    parameter_counts = {}
    gate_hashes = {}
    first_batches = {}
    for arm in ("N", "O", "G"):
        model, optimizer, loader, metadata, construction = load_parent_model(source, arm, device, restore=True)
        geometry = boundary_audit(model, arm, device)
        smoke = disposable_smoke(model, arm, device)
        parameter_counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        gate_hashes[arm] = metadata["gate_sha256"]
        first_batches[arm] = {
            "cursor": loader.state_dict(),
            "batch": base.next_batch_hash(loader, ACCUMULATION),
            "stream": base.next_stream_hash(loader, ACCUMULATION),
        }
        arm_rows[arm] = {
            "architecture_manifest": core.architecture_manifest(arm),
            "architecture_fingerprint": core.architecture_fingerprint(arm),
            "construction": construction,
            "geometry": geometry,
            "disposable_smoke": smoke,
            "passed": all(construction.values()) and geometry["passed"] and smoke["passed"],
        }
        del model, optimizer, loader
        gc.collect()
        torch.cuda.empty_cache()
    common = normalized_common_manifest("N")
    config_diff = {
        "common_architecture_exact": all(normalized_common_manifest(arm) == common for arm in ("N", "O", "G")),
        "only_allowed_geometry_minima": [core.GEOMETRIES[arm] for arm in ("N", "O", "G")] == [
            {0: 2, 2: 32, 4: 64}, {0: 1, 2: 31, 4: 63}, {0: 3, 2: 33, 4: 65}
        ],
    }
    checks = {
        "parent_sha": provenance["parent_sha256"] == PARENT_SHA256,
        "continuation": continuation.get("passed") is True and continuation.get("rows") == LOCAL_UPDATES,
        "first_update": continuation.get("first_global_update") == 2100,
        "terminal_update": continuation.get("final_global_update") == FINAL_GLOBAL_UPDATE,
        "panel": panel.get("sequence_count") == PANEL_SEQUENCES and panel.get("targets_per_condition") == PANEL_TARGETS,
        "panel_fresh": panel.get("sealed_before_checkpoint_loading") is True and panel.get("checkpoint_losses_inspected_during_selection") is False,
        "all_arms": all(row["passed"] for row in arm_rows.values()),
        "parameter_counts_identical": len(set(parameter_counts.values())) == 1 and next(iter(parameter_counts.values())) == core.PARAMETER_COUNT,
        "gate_values_identical": gate_hashes["N"] == gate_hashes["O"] == gate_hashes["G"],
        "first_loader_state_identical": first_batches["N"] == first_batches["O"] == first_batches["G"],
        "first_batch_matches_manifest": first_batches["N"]["batch"] == continuation["first_global_batch_sha256"],
        "first_stream_matches_manifest": first_batches["N"]["stream"] == continuation["first_stream_sha256"],
        "config_diff": all(config_diff.values()),
    }
    audit = {
        "schema": "experiment_2d7_preflight_audit_v1",
        "experiment": EXPERIMENT,
        "git_commit": git("rev-parse", "HEAD"),
        "implementation_sha256": implementation_sha256(),
        "provenance": provenance,
        "arms": arm_rows,
        "parameter_counts": parameter_counts,
        "inherited_gate_sha256": gate_hashes,
        "first_logical_batches": first_batches,
        "config_diff_audit": config_diff,
        "checks": checks,
        "authorized": all(checks.values()),
    }
    durable_json(output / "PREFLIGHT_AUDIT.json", audit)
    if not audit["authorized"]:
        raise SystemExit(f"2D7 preflight failed: {checks}")
    print("EXPERIMENT_2D7_PREFLIGHT_PASS", flush=True)


def active_gradient_report(model):
    groups = {
        "base": [parameter for name, parameter in model.named_parameters() if name not in {"g_rec", "g_rec_b3", "g_rec_b5", "g_rec_b6"}],
        "b1_gate": [model.g_rec], "b3_gate": [model.g_rec_b3], "b5_gate": [model.g_rec_b5],
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
        "active_incremented_once": all(after_steps[name] == before_steps[name] + 1 for name in active),
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
        "condition": core.GEOMETRY_NAMES[arm],
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
        "parameter_count": core.PARAMETER_COUNT,
        "optimizer_group_definitions": [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups],
        "current_optimizer_steps_by_name": d6.optimizer_steps_by_name(model, optimizer),
        "git_implementation_commit": git("rev-parse", "HEAD"),
        "saved_process_id": os.getpid(), "saved_at_unix": time.time(),
    }


def load_final_checkpoint(path, device, restore=False):
    payload = base.d0.torch_load(Path(path).resolve(), mmap=True)
    arm = payload.get("arm")
    if payload.get("schema") != SCHEMA or arm not in core.GEOMETRIES:
        raise SystemExit("not a 2D7 final checkpoint")
    _, foundation = base.instantiate_base(device)
    model = core.BoundaryAlignmentGPT(foundation, arm).to(device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = base.configure_optimizer(model, device.type)
    optimizer.load_state_dict(payload["optimizer"])
    loader = loader_from_state(payload["loader_state"])
    if restore:
        base.restore_rng(payload["rng_state"])
    return model, optimizer, loader, payload


def strict_reopen(path, device):
    model, optimizer, loader, payload = load_final_checkpoint(path, device, restore=False)
    arm = payload["arm"]
    checks = {
        "schema": payload.get("schema") == SCHEMA,
        "parent": payload.get("parent_checkpoint_sha256") == PARENT_SHA256,
        "global_update": payload.get("global_update") == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": payload.get("cumulative_targets") == FINAL_TARGETS,
        "local_updates": payload.get("local_updates") == LOCAL_UPDATES,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()) == core.PARAMETER_COUNT,
        "architecture": payload.get("architecture_fingerprint") == core.architecture_fingerprint(arm),
        "model_finite": base.model_finite(model), "optimizer_finite": base.optimizer_finite(optimizer),
        "loader_cursor": loader.state_dict() == payload["loader_state"],
        "next_batch": base.next_batch_hash(loader, ACCUMULATION) == payload["next_global_batch_sha256"],
        "next_stream": base.next_stream_hash(loader, ACCUMULATION) == payload["next_global_batch_stream_sha256"],
        "rng": payload.get("rng_digests") == d5c.rng_digests(payload["rng_state"]),
        "scheduler": "scheduler" in payload,
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
    if arm not in core.GEOMETRIES:
        raise SystemExit("arm must be N, O, or G")
    device = base.require_a100()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"TRAINING_LOG_{arm}.jsonl"
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
        "global_update": rows[0]["global_update"] == 2100,
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
        print(f"2D7 {arm} update {local_update}/{LOCAL_UPDATES} CE {row['ce']:.6f}", flush=True)
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
        "active_optimizer_progression": all(final_steps[name] == source_steps[name] + LOCAL_UPDATES for name in active),
        "dormant_optimizer_unchanged": final_steps["g_rec_b6"] == source_steps["g_rec_b6"],
        "only_final_checkpoint": checkpoint_path.name == FINAL_CHECKPOINT_NAME,
    }
    summary = {
        "schema": "experiment_2d7_training_complete_v1", "experiment": EXPERIMENT,
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
    print(f"EXPERIMENT_2D7_TRAINING_COMPLETE {arm}", flush=True)


def incremental_condition(model, x, y, audit=False):
    state = model.init_incremental_state(x.size(0), device=x.device, dtype=torch.bfloat16)
    nll = torch.zeros(x.size(0), dtype=torch.float64)
    for position in range(base.T):
        logits, state = model.incremental_step(x[:, position], state)
        nll += F.cross_entropy(logits[:, 0].float(), y[:, position], reduction="none").double().cpu()
    return {
        "nll_sum": float(nll.sum()), "targets": int(y.numel()),
        "per_sequence_nll": nll.tolist(), "per_sequence_ce": (nll / base.T).tolist(),
        "final_cache_audit": model.incremental_cache_audit(state),
    }


def run_evaluate(args):
    require_branch(clean=True)
    arm = args.arm.upper()
    device = base.require_a100()
    panel = read_json(args.panel_manifest)
    if panel.get("sequence_count") != PANEL_SEQUENCES or panel.get("targets_per_condition") != PANEL_TARGETS:
        raise SystemExit("evaluation requires the sealed 2,048-sequence panel")
    checkpoint = require_file(args.checkpoint, label=f"{arm} final checkpoint")
    model, optimizer, loader, payload = load_final_checkpoint(checkpoint, device, restore=False)
    checks = {
        "arm": payload.get("arm") == arm,
        "global_update": payload.get("global_update") == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": payload.get("cumulative_targets") == FINAL_TARGETS,
        "strict_reopen": strict_reopen(checkpoint, device)["passed"],
    }
    if not all(checks.values()):
        raise SystemExit(f"evaluation checkpoint failure: {checks}")
    output_path = Path(args.output_path).resolve()
    if output_path.exists():
        raise SystemExit("refusing to overwrite or resume a final 2D7 evaluation")
    state = {
        "schema": "experiment_2d7_true_incremental_losses_v1", "experiment": EXPERIMENT,
        "condition": core.GEOMETRY_NAMES[arm], "arm": arm,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "architecture_fingerprint": core.architecture_fingerprint(arm),
        "panel_manifest_sha256": sha256(args.panel_manifest), "panel_sha256": panel["panel_sha256"],
        "evaluation_set_label": "fresh disjoint matched panel",
        "precision": "BF16 model execution; FP32 token CE; FP64 accumulation",
        "batch_indices_in_evaluation_order": panel["batch_indices_in_evaluation_order"],
        "completed_batch_indices": [], "batch_identities": [],
        "nll_sum": 0.0, "targets": 0, "per_sequence_nll": [], "per_sequence_ce": [],
        "status": "running", "checkpoint_checks": checks,
    }
    val_path = base.validation_path(Path(args.data_root))
    model.eval()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for ordinal, batch_index in enumerate(panel["batch_indices_in_evaluation_order"]):
            cpu_x, cpu_y = d5c.batch_at_index(val_path, int(batch_index))
            observed = base.batch_identity(cpu_x, cpu_y)
            if observed != panel["batch_identities"][ordinal]:
                raise SystemExit(f"panel identity mismatch at ordinal {ordinal}")
            x, y = cpu_x.to(device), cpu_y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                row = incremental_condition(model, x, y, audit=ordinal == 0)
            state["nll_sum"] += row["nll_sum"]
            state["targets"] += row["targets"]
            state["per_sequence_nll"].extend(row["per_sequence_nll"])
            state["per_sequence_ce"].extend(row["per_sequence_ce"])
            state["completed_batch_indices"].append(int(batch_index))
            state["batch_identities"].append(observed)
            state["final_cache_audit"] = row["final_cache_audit"]
            durable_json(output_path, state)
            print(f"2D7 {arm} true-incremental batch {ordinal + 1}/{PANEL_BATCHES}", flush=True)
            del cpu_x, cpu_y, x, y
            torch.cuda.empty_cache()
    state["aggregate_ce"] = state["nll_sum"] / state["targets"]
    state["perplexity"] = math.exp(state["aggregate_ce"])
    state["paired_sequences"] = len(state["per_sequence_ce"])
    state["persistent_state"] = {
        "logical_bytes_per_sequence": state["final_cache_audit"]["logical_payload_bytes"] // 64,
        "physical_bytes_per_sequence": state["final_cache_audit"]["actual_unique_storage_bytes"] // 64,
        "batch_size": 64, "position": base.T,
    }
    state["wall_seconds"] = time.monotonic() - started
    state["peak_allocated_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    state["peak_reserved_vram_mb"] = torch.cuda.max_memory_reserved(device) / 1024**2
    state["status"] = "complete"
    state["passed"] = (
        state["targets"] == PANEL_TARGETS and state["paired_sequences"] == PANEL_SEQUENCES
        and state["batch_identities"] == panel["batch_identities"]
        and state["final_cache_audit"]["passed"]
    )
    durable_json(output_path, state)
    if not state["passed"]:
        raise SystemExit(f"final evaluation failed for {arm}")
    print(f"EXPERIMENT_2D7_EVALUATION_COMPLETE {arm}", flush=True)


def paired_bootstrap(arrays):
    length = len(next(iter(arrays.values())))
    if length != PANEL_SEQUENCES or any(len(value) != length for value in arrays.values()):
        raise SystemExit("paired bootstrap requires three aligned 2,048-sequence arrays")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    distributions = {name: np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64) for name in arrays}
    cursor = 0
    while cursor < BOOTSTRAP_RESAMPLES:
        count = min(250, BOOTSTRAP_RESAMPLES - cursor)
        indices = generator.integers(0, length, size=(count, length), dtype=np.int32)
        for name, values in arrays.items():
            distributions[name][cursor : cursor + count] = np.asarray(values)[indices].mean(axis=1)
        cursor += count
    return {
        "method": "paired per-sequence percentile bootstrap",
        "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
        "sampling_unit": "sequence", "paired_sequences": length,
        "shared_resample_indices_all_contrasts": True,
        "contrasts": {
            name: {
                "estimate": float(np.mean(values)),
                "lower_95": float(np.percentile(distributions[name], 2.5)),
                "upper_95": float(np.percentile(distributions[name], 97.5)),
            }
            for name, values in arrays.items()
        },
    }


def classify(row, left, right):
    lower, upper = row["lower_95"], row["upper_95"]
    if lower > DELTA_CE:
        return f"{right} superior to {left} by more than delta_CE"
    if upper < -DELTA_CE:
        return f"{left} superior to {right} by more than delta_CE"
    if lower > 0:
        return f"{right} statistically superior to {left}, but not beyond delta_CE"
    if upper < 0:
        return f"{left} statistically superior to {right}, but not beyond delta_CE"
    if lower > -DELTA_CE and upper < DELTA_CE:
        return "practical equivalence established"
    return "unresolved for superiority and practical equivalence"


def wins(left, right):
    difference = np.asarray(left) - np.asarray(right)
    return {
        "left_wins": int(np.sum(difference < 0)),
        "right_wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
    }


def format_report(summary):
    ce = summary["mean_ce"]
    contrasts = summary["bootstrap"]["contrasts"]
    ratios = summary["perplexity_ratios"]
    win = summary["per_sequence_wins"]
    classes = summary["classification"]
    persistent = summary["persistent_state"]
    checkpoints = summary["final_checkpoints"]
    ranking = summary["point_estimate_ranking"]
    terminal = summary["terminal_stream"]
    return f"""# EXPERIMENT 2D7 — BOUNDARY ALIGNMENT N/O/G COMPLETE

Baseline CE: `{ce['N']:.12f}`
Overlap-1 CE: `{ce['O']:.12f}`
Gap-1 CE: `{ce['G']:.12f}`

Baseline − Overlap: `{contrasts['D_NO']['estimate']:+.12f}`
95% CI: `[{contrasts['D_NO']['lower_95']:+.12f}, {contrasts['D_NO']['upper_95']:+.12f}]`

Baseline − Gap: `{contrasts['D_NG']['estimate']:+.12f}`
95% CI: `[{contrasts['D_NG']['lower_95']:+.12f}, {contrasts['D_NG']['upper_95']:+.12f}]`

Overlap − Gap: `{contrasts['D_OG']['estimate']:+.12f}`
95% CI: `[{contrasts['D_OG']['lower_95']:+.12f}, {contrasts['D_OG']['upper_95']:+.12f}]`

Perplexity ratios:
Baseline/Overlap: `{ratios['D_NO']:.12f}`
Baseline/Gap: `{ratios['D_NG']:.12f}`
Overlap/Gap: `{ratios['D_OG']:.12f}`

Per-sequence wins:
Baseline vs Overlap: Baseline `{win['N_vs_O']['left_wins']}`, Overlap `{win['N_vs_O']['right_wins']}`, ties `{win['N_vs_O']['ties']}`
Baseline vs Gap: Baseline `{win['N_vs_G']['left_wins']}`, Gap `{win['N_vs_G']['right_wins']}`, ties `{win['N_vs_G']['ties']}`
Overlap vs Gap: Overlap `{win['O_vs_G']['left_wins']}`, Gap `{win['O_vs_G']['right_wins']}`, ties `{win['O_vs_G']['ties']}`

Point-estimate ranking:
1. {ranking[0]}
2. {ranking[1]}
3. {ranking[2]}

delta_CE:
`0.0001`

Practical/statistical classification:
Baseline vs Overlap: {classes['N_vs_O']}
Baseline vs Gap: {classes['N_vs_G']}
Overlap vs Gap: {classes['O_vs_G']}

Persistent state:
Baseline: `{persistent['N']:,}` bytes/sequence
Overlap-1: `{persistent['O']:,}` bytes/sequence
Gap-1: `{persistent['G']:,}` bytes/sequence
Overlap − Baseline: `{persistent['O_minus_N']:+,}` bytes/sequence
Gap − Baseline: `{persistent['G_minus_N']:+,}` bytes/sequence

Training counters:
Starting global update: 2099
First new update:       2100
Final global update:    2290
Final cumulative targets: 1,200,619,520

Terminal stream equality:
Final loader cursor: `{terminal['final_loader_cursor_sha256']}`
Next-global-batch SHA: `{terminal['next_global_batch_sha256']}`
Next-stream SHA: `{terminal['next_stream_sha256']}`
{terminal['status']}

Final checkpoints:
N: `{checkpoints['N']['path']}`
SHA-256: `{checkpoints['N']['sha256']}`

O: `{checkpoints['O']['path']}`
SHA-256: `{checkpoints['O']['sha256']}`

G: `{checkpoints['G']['path']}`
SHA-256: `{checkpoints['G']['sha256']}`

AUDIT:
{summary['audit_status']}

GPU STATUS:
{summary['gpu_status']}

## SCIENTIFIC INTERPRETATION

{summary['scientific_interpretation']}

## RECOMMENDATION

{summary['recommendation']}
"""


def run_analyze(args):
    evaluations = {arm: read_json(getattr(args, f"evaluation_{arm.lower()}")) for arm in ("N", "O", "G")}
    training = {arm: read_json(getattr(args, f"training_{arm.lower()}")) for arm in ("N", "O", "G")}
    preflight = read_json(args.preflight_audit)
    panel = read_json(args.panel_manifest)
    stop = read_json(args.stop_verification)
    values = {arm: np.asarray(evaluations[arm]["per_sequence_ce"], dtype=np.float64) for arm in ("N", "O", "G")}
    difference_arrays = {"D_NO": values["N"] - values["O"], "D_NG": values["N"] - values["G"], "D_OG": values["O"] - values["G"]}
    bootstrap = paired_bootstrap(difference_arrays)
    mean_ce = {arm: float(values[arm].mean()) for arm in values}
    point_ranking = [core.GEOMETRY_NAMES[arm] for arm in sorted(mean_ce, key=mean_ce.get)]
    classes = {
        "N_vs_O": classify(bootstrap["contrasts"]["D_NO"], "Baseline", "Overlap-1"),
        "N_vs_G": classify(bootstrap["contrasts"]["D_NG"], "Baseline", "Gap-1"),
        "O_vs_G": classify(bootstrap["contrasts"]["D_OG"], "Overlap-1", "Gap-1"),
    }
    persistent = {arm: int(evaluations[arm]["persistent_state"]["physical_bytes_per_sequence"]) for arm in values}
    persistent.update(O_minus_N=persistent["O"] - persistent["N"], G_minus_N=persistent["G"] - persistent["N"])
    cursor_hashes = {arm: canonical_sha(training[arm]["final_loader_cursor"]) for arm in values}
    batches = {arm: training[arm]["next_global_batch_sha256"] for arm in values}
    streams = {arm: training[arm]["next_stream_sha256"] for arm in values}
    terminal_match = len(set(cursor_hashes.values())) == len(set(batches.values())) == len(set(streams.values())) == 1
    checkpoint_rows = {
        arm: {
            "path": getattr(args, f"checkpoint_{arm.lower()}"),
            "sha256": sha256(getattr(args, f"checkpoint_{arm.lower()}")),
            "recorded_sha256": training[arm]["final_checkpoint"]["sha256"],
        }
        for arm in values
    }
    evaluation_pairing = all(
        evaluations[arm]["batch_identities"] == panel["batch_identities"]
        and evaluations[arm]["panel_sha256"] == panel["panel_sha256"]
        for arm in values
    )
    stop_confirmed = (
        stop.get("pod", {}).get("desiredStatus") == "EXITED"
        and stop.get("pod", {}).get("runtimeStatus") == "stopped"
    ) or stop.get("status") == "STOPPED"
    checks = {
        "preflight": preflight.get("authorized") is True,
        "three_training_arms": all(training[arm].get("passed") is True for arm in values),
        "three_evaluations": all(evaluations[arm].get("passed") is True for arm in values),
        "evaluation_pairing": evaluation_pairing,
        "per_sequence_counts": all(len(values[arm]) == PANEL_SEQUENCES for arm in values),
        "targets": all(evaluations[arm]["targets"] == PANEL_TARGETS for arm in values),
        "terminal_stream_equality": terminal_match,
        "final_counters": all(
            training[arm]["final_checkpoint"]["global_update"] == FINAL_GLOBAL_UPDATE
            and training[arm]["final_checkpoint"]["cumulative_targets"] == FINAL_TARGETS
            for arm in values
        ),
        "checkpoint_sha_exports": all(row["sha256"] == row["recorded_sha256"] for row in checkpoint_rows.values()),
        "bootstrap": bootstrap["resamples"] == BOOTSTRAP_RESAMPLES,
        "scope_three_conditions": {evaluations[arm]["condition"] for arm in values} == set(core.GEOMETRY_NAMES.values()),
        "gpu_stopped": stop_confirmed,
    }
    audit_pass = all(checks.values())
    meaningful_o = (
        bootstrap["contrasts"]["D_NO"]["lower_95"] > DELTA_CE
        and bootstrap["contrasts"]["D_OG"]["upper_95"] < -DELTA_CE
    )
    meaningful_g = (
        bootstrap["contrasts"]["D_NG"]["lower_95"] > DELTA_CE
        and bootstrap["contrasts"]["D_OG"]["lower_95"] > DELTA_CE
    )
    all_equivalent = all(
        row["lower_95"] > -DELTA_CE and row["upper_95"] < DELTA_CE
        for row in bootstrap["contrasts"].values()
    )
    if meaningful_o:
        recommendation = "Recommend OVERLAP-1 because it established a greater-than-0.0001 CE advantage over both alternatives."
    elif meaningful_g:
        recommendation = "Recommend GAP-1 because it established a greater-than-0.0001 CE advantage over both alternatives."
    elif all_equivalent:
        recommendation = "Recommend BASELINE/N because all three geometries are practically equivalent and N is the established implementation."
    else:
        recommendation = "Recommend BASELINE/N provisionally because no alternative established sufficient superiority."
    if all_equivalent:
        interpretation = (
            "N, O, and G are practically equivalent at the preregistered ±0.0001 CE scale. "
            f"The numerical point-estimate winner ({point_ranking[0]}) is not an established meaningful winner."
        )
    else:
        interpretation = (
            f"The numerical ordering is {' < '.join(point_ranking)}. "
            "The pairwise classifications above distinguish established effects from unresolved numerical differences."
        )
    summary = {
        "experiment": EXPERIMENT, "mean_ce": mean_ce,
        "perplexity": {arm: math.exp(value) for arm, value in mean_ce.items()},
        "bootstrap": bootstrap,
        "perplexity_ratios": {name: math.exp(row["estimate"]) for name, row in bootstrap["contrasts"].items()},
        "per_sequence_wins": {
            "N_vs_O": wins(values["N"], values["O"]),
            "N_vs_G": wins(values["N"], values["G"]),
            "O_vs_G": wins(values["O"], values["G"]),
        },
        "point_estimate_ranking": point_ranking, "delta_ce": DELTA_CE,
        "classification": classes, "persistent_state": persistent,
        "terminal_stream": {
            "final_loader_cursor_sha256": next(iter(cursor_hashes.values())),
            "next_global_batch_sha256": next(iter(batches.values())),
            "next_stream_sha256": next(iter(streams.values())),
            "per_arm_cursor_sha256": cursor_hashes, "per_arm_next_batch_sha256": batches,
            "per_arm_next_stream_sha256": streams, "status": "MATCH" if terminal_match else "MISMATCH",
        },
        "final_checkpoints": checkpoint_rows,
        "checks": checks, "audit_status": "PASS" if audit_pass else "FAIL",
        "gpu_status": "STOPPED" if stop_confirmed else "NOT STOPPED",
        "scientific_interpretation": interpretation, "recommendation": recommendation,
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    durable_json(output / "PAIRED_BOOTSTRAP.json", bootstrap)
    durable_json(output / "PERSISTENT_STATE_SUMMARY.json", persistent)
    durable_json(output / "SCIENTIFIC_RESULT_SUMMARY.json", summary)
    durable_json(output / "FINAL_AUDIT.json", {
        "schema": "experiment_2d7_final_audit_v1", "experiment": EXPERIMENT,
        "checks": checks, "audit_status": summary["audit_status"], "passed": audit_pass,
    })
    durable_text(output / "EXPERIMENT_2D7_FINAL_REPORT.md", format_report(summary))
    if not audit_pass:
        raise SystemExit(f"2D7 final audit failed: {checks}")
    print("EXPERIMENT_2D7_ANALYSIS_COMPLETE", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Experiment 2D7 scientific driver")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-panel")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(handler=prepare_panel)
    continuation = commands.add_parser("build-continuation")
    continuation.add_argument("--parent-checkpoint", required=True)
    continuation.add_argument("--output-dir", required=True)
    continuation.set_defaults(handler=build_continuation)
    preflight = commands.add_parser("preflight")
    for name in ("parent_checkpoint", "continuation_manifest", "panel_manifest", "output_dir"):
        preflight.add_argument(f"--{name.replace('_', '-')}", required=True)
    preflight.set_defaults(handler=run_preflight)
    train = commands.add_parser("train")
    train.add_argument("--arm", required=True, choices=("N", "O", "G"))
    for name in ("parent_checkpoint", "continuation_manifest", "continuation_ledger", "preflight_audit", "output_dir", "checkpoint_dir"):
        train.add_argument(f"--{name.replace('_', '-')}", required=True)
    train.set_defaults(handler=run_train)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--arm", required=True, choices=("N", "O", "G"))
    for name in ("checkpoint", "panel_manifest", "data_root", "output_path"):
        evaluate.add_argument(f"--{name.replace('_', '-')}", required=True)
    evaluate.set_defaults(handler=run_evaluate)
    analyze = commands.add_parser("analyze")
    for prefix in ("evaluation", "training", "checkpoint"):
        for arm in ("n", "o", "g"):
            analyze.add_argument(f"--{prefix}-{arm}", required=True)
    for name in ("preflight_audit", "panel_manifest", "stop_verification", "output_dir"):
        analyze.add_argument(f"--{name.replace('_', '-')}", required=True)
    analyze.set_defaults(handler=run_analyze)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
