#!/usr/bin/env python3
"""Fail-closed driver for Experiment 2D8 overlap width N/O1/O2."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import experiment_2d3a as base
import experiment_2d5c as d5c
import experiment_2d7 as engine
import experiment_2d7_core as d7_core
import experiment_2d8_core as core


EXPERIMENT = "2D8"
BRANCH = "experiment-2d8-trained-overlap-width-n-o1-o2"
SCHEMA = "experiment_2d8_o2_checkpoint_v1"
PARENT_SHA256 = "6e5023b127032dbb4d32a23bf1be052702d51177437b2795b99c52bcd83314c7"
N_SHA256 = "57e62a2094693205b520e2986047d46c28d042d4ec34d6e65b2135f474adec20"
O1_SHA256 = "c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6"
CONTINUATION_MANIFEST_SHA256 = "f15a5de4b5428031adfe8877f01e6487dcdfc6749e337f552feb7c6f92e9cc4d"
CONTINUATION_LEDGER_SHA256 = "555ac4b4425fcd711edf2e923412ecfac1db49802653570fe56b02ae4139c1aa"
TERMINAL_CURSOR_SHA256 = "682abcbcc8db8886274ccbb604927683af38d010ecae31b1494987513ae982d3"
TERMINAL_NEXT_BATCH_SHA256 = "a021ce09f7a25b6617632e2a76da1acb0980ac1dda888df5f1c8eb65b3939fbe"
TERMINAL_NEXT_STREAM_SHA256 = "7918ea7e6f979b8e49fca89c60dd68ace44b768854d34bc96dd751bee07b2567"
PARENT_GLOBAL_UPDATE = 2_099
PARENT_TARGETS = 1_100_480_512
LOCAL_UPDATES = 191
TARGETS_PER_UPDATE = 524_288
LOCAL_TARGETS = 100_139_008
FINAL_GLOBAL_UPDATE = 2_290
FINAL_TARGETS = 1_200_619_520
FINAL_CHECKPOINT_NAME = "scientific_cumulative_001200619520.pt"
PANEL_SEQUENCES = 4_096
PANEL_TARGETS = 4_194_304
PANEL_BATCHES = 64
PANEL_SEED = 2_026_090_4
DATASET_SHA256 = engine.DATASET_SHA256
REPO_ROOT = Path(__file__).resolve().parents[1]


# Reuse the sealed 2D7 continuation/training engine with 2D8 bindings.
engine.EXPERIMENT = EXPERIMENT
engine.BRANCH = BRANCH
engine.SCHEMA = SCHEMA
engine.core = core
engine.PANEL_SEQUENCES = PANEL_SEQUENCES
engine.PANEL_TARGETS = PANEL_TARGETS
engine.PANEL_BATCHES = PANEL_BATCHES
engine.PANEL_SEED = PANEL_SEED


def read_json(path):
    return json.loads(Path(path).read_text())


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
        REPO_ROOT / "scripts/experiment_2d8.py",
        REPO_ROOT / "scripts/experiment_2d8_core.py",
        REPO_ROOT / "configs/exp2d8_trained_overlap_width_n_o1_o2.json",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


engine.implementation_sha256 = implementation_sha256


def normalized_manifest(arm):
    value = copy.deepcopy(core.architecture_manifest(arm))
    value.pop("experiment")
    value.pop("arm")
    value.pop("condition")
    value.pop("description")
    value.pop("recurrent_minimum_lags")
    for block in value["blocks"]:
        if block["recurrent_lags"] is not None:
            block["recurrent_lags"][0] = "GEOMETRY_MINIMUM"
    return value


def boundary_audit(model, device):
    neighborhoods = {
        0: (0, 1, 2),
        2: (29, 30, 31, 32),
        4: (61, 62, 63, 64),
    }
    boundaries = {}
    for block in (0, 2, 4):
        length = 1024
        query = torch.arange(length, device=device).view(length, 1)
        source = torch.arange(length, device=device).view(1, length)
        lag = query - source
        local = model.local_mask(block, length, device)
        recurrent = model.recurrent_mask(block, length, length, device)
        window = core.LOCAL_WINDOWS[block]
        minimum = core.GEOMETRIES["O2"][block]
        boundaries[f"B{block + 1}"] = {
            "local_lags": [0, window - 1],
            "recurrent_lags": [minimum, 1023],
            "neighborhood": {
                str(value): {
                    "local": bool(((lag == value) & local).any()),
                    "recurrent": bool(((lag == value) & recurrent).any()),
                }
                for value in neighborhoods[block]
            },
            "local_exact": bool(torch.equal(local, (lag >= 0) & (lag < window))),
            "recurrent_exact": bool(
                torch.equal(recurrent, (lag >= minimum) & (lag <= 1023))
            ),
            "lag_1023_recurrent": bool(recurrent[1023, 0]),
            "no_future_or_current_recurrent": not bool((recurrent & (lag <= 0)).any()),
            "source_identity": "j=t-lag",
        }
        del query, source, lag, local, recurrent
    state = model.init_incremental_state(1, device=device, dtype=torch.bfloat16)
    tokens = (torch.arange(66, device=device) * 7919 + 17).remainder(50_257).view(1, -1)
    diagnostic = None
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            logits, state, diagnostic = model.incremental_step(
                tokens[:, position], state, return_diagnostics=True,
                diagnostic_attention_weights=False,
            )
            del logits
    runtime_positions = {}
    for block in (0, 2, 4):
        observed = diagnostic["links"][f"b{block + 1}"]["recurrent_positions"]
        positions = [] if observed is None else [int(value) for value in observed[0].tolist()]
        expected = list(range(0, 66 - core.GEOMETRIES["O2"][block]))
        runtime_positions[f"B{block + 1}"] = {
            "query_position": 65,
            "positions": positions,
            "expected_positions": expected,
            "source_identity_exact": positions == expected,
        }
    expected_neighborhoods = {
        "B1": {"0": (True, False), "1": (True, True), "2": (False, True)},
        "B3": {"29": (True, False), "30": (True, True), "31": (True, True), "32": (False, True)},
        "B5": {"61": (True, False), "62": (True, True), "63": (True, True), "64": (False, True)},
    }
    checks = {
        "masks_exact": all(
            row["local_exact"] and row["recurrent_exact"]
            for row in boundaries.values()
        ),
        "boundary_neighborhoods_exact": all(
            all(
                (row["neighborhood"][lag]["local"], row["neighborhood"][lag]["recurrent"])
                == expected
                for lag, expected in expected_neighborhoods[name].items()
            )
            for name, row in boundaries.items()
        ),
        "maximum_recurrent_lag_1023": all(
            row["lag_1023_recurrent"] for row in boundaries.values()
        ),
        "no_future_or_source_shift": all(
            row["no_future_or_current_recurrent"] and row["source_identity"] == "j=t-lag"
            for row in boundaries.values()
        ) and all(row["source_identity_exact"] for row in runtime_positions.values()),
        "b6_runtime_w1024": model._last_b6_local_capacity == 1023,
        "b7_to_b6_absent": model._b6_recurrent_branch_calls == 0 and not hasattr(state, "h7_ring"),
    }
    return {
        "arm": "O2", "boundaries": boundaries,
        "runtime_positions": runtime_positions, "checks": checks,
        "passed": all(checks.values()),
    }


def strict_existing_checkpoint(path, requested_arm, device):
    expected_sha = {"N": N_SHA256, "O1": O1_SHA256}[requested_arm]
    source = require_file(path, expected_sha, f"sealed 2D7 {requested_arm} checkpoint")
    payload = base.d0.torch_load(source, mmap=True)
    stored_arm = "N" if requested_arm == "N" else "O"
    _, foundation = base.instantiate_base(device)
    model = core.OverlapWidthGPT(foundation, requested_arm).to(device)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    optimizer = base.configure_optimizer(model, device.type)
    optimizer.load_state_dict(payload["optimizer"])
    loader = engine.loader_from_state(payload["loader_state"])
    checks = {
        "sha256": sha256(source) == expected_sha,
        "schema": payload.get("schema") == "experiment_2d7_checkpoint_v1",
        "stored_arm": payload.get("arm") == stored_arm,
        "strict_no_missing": not incompatible.missing_keys,
        "strict_no_unexpected": not incompatible.unexpected_keys,
        "parent_sha": payload.get("parent_checkpoint_sha256") == PARENT_SHA256,
        "global_update": payload.get("global_update") == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": payload.get("cumulative_targets") == FINAL_TARGETS,
        "architecture": payload.get("architecture_fingerprint")
        == d7_core.architecture_fingerprint(stored_arm),
        "optimizer": base.optimizer_finite(optimizer),
        "scheduler": "scheduler" in payload,
        "rng": payload.get("rng_digests") == d5c.rng_digests(payload["rng_state"]),
        "loader": loader.state_dict() == payload["loader_state"],
        "next_batch": base.next_batch_hash(loader, engine.ACCUMULATION)
        == payload["next_global_batch_sha256"],
        "next_stream": base.next_stream_hash(loader, engine.ACCUMULATION)
        == payload["next_global_batch_stream_sha256"],
    }
    metadata = {
        "stored_arm": payload["arm"], "global_update": payload["global_update"],
        "cumulative_targets": payload["cumulative_targets"],
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "next_stream_sha256": payload["next_global_batch_stream_sha256"],
    }
    del optimizer, loader, payload
    gc.collect()
    if not all(checks.values()):
        del model
        torch.cuda.empty_cache()
        raise SystemExit(f"existing {requested_arm} checkpoint audit failed: {checks}")
    return model, metadata, checks


def strict_o2_checkpoint(path, device):
    source = require_file(path, label="sealed O2 checkpoint")
    model, optimizer, loader, payload = engine.load_final_checkpoint(source, device)
    checks = {
        "schema": payload.get("schema") == SCHEMA,
        "arm": payload.get("arm") == "O2",
        "parent_sha": payload.get("parent_checkpoint_sha256") == PARENT_SHA256,
        "global_update": payload.get("global_update") == FINAL_GLOBAL_UPDATE,
        "cumulative_targets": payload.get("cumulative_targets") == FINAL_TARGETS,
        "architecture": payload.get("architecture_fingerprint") == core.architecture_fingerprint("O2"),
        "optimizer": base.optimizer_finite(optimizer),
        "scheduler": "scheduler" in payload,
        "rng": payload.get("rng_digests") == d5c.rng_digests(payload["rng_state"]),
        "loader": loader.state_dict() == payload["loader_state"],
        "next_batch": base.next_batch_hash(loader, engine.ACCUMULATION) == TERMINAL_NEXT_BATCH_SHA256,
        "next_stream": base.next_stream_hash(loader, engine.ACCUMULATION) == TERMINAL_NEXT_STREAM_SHA256,
        "cursor_sha": canonical_sha(loader.state_dict()) == TERMINAL_CURSOR_SHA256,
    }
    metadata = {
        "stored_arm": payload["arm"], "global_update": payload["global_update"],
        "cumulative_targets": payload["cumulative_targets"],
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "next_stream_sha256": payload["next_global_batch_stream_sha256"],
    }
    del optimizer, loader, payload
    gc.collect()
    if not all(checks.values()):
        del model
        torch.cuda.empty_cache()
        raise SystemExit(f"O2 checkpoint audit failed: {checks}")
    return model, metadata, checks


def run_preflight(args):
    engine.require_branch(clean=True)
    device = base.require_a100()
    parent = require_file(args.parent_checkpoint, PARENT_SHA256, "sealed 2D6 parent")
    continuation_path = require_file(
        args.continuation_manifest, CONTINUATION_MANIFEST_SHA256,
        "sealed 2D7 continuation manifest",
    )
    ledger_path = require_file(
        args.continuation_ledger, CONTINUATION_LEDGER_SHA256,
        "sealed 2D7 continuation ledger",
    )
    continuation = read_json(continuation_path)
    rows = engine.load_rows(ledger_path)
    model, optimizer, loader, parent_metadata, construction = engine.load_parent_model(
        parent, "O2", device, restore=True
    )
    geometry = boundary_audit(model, device)
    smoke = engine.disposable_smoke(model, "O2", device)
    first = {
        "cursor": loader.state_dict() == continuation["first_loader_cursor"],
        "batch": base.next_batch_hash(loader, engine.ACCUMULATION)
        == continuation["first_global_batch_sha256"],
        "stream": base.next_stream_hash(loader, engine.ACCUMULATION)
        == continuation["first_stream_sha256"],
        "global_update": rows[0]["global_update"] == 2100,
    }
    o2_preflight = {
        "construction": construction,
        "geometry": geometry,
        "smoke": smoke,
        "first_logical_batch": first,
        "architecture_manifest": core.architecture_manifest("O2"),
        "architecture_fingerprint": core.architecture_fingerprint("O2"),
        "inherited_gate_sha256": parent_metadata["gate_sha256"],
    }
    del model, optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    n_model, n_metadata, n_checks = strict_existing_checkpoint(args.n_checkpoint, "N", device)
    del n_model
    gc.collect()
    torch.cuda.empty_cache()
    o1_model, o1_metadata, o1_checks = strict_existing_checkpoint(args.o1_checkpoint, "O1", device)
    del o1_model
    gc.collect()
    torch.cuda.empty_cache()
    checks = {
        "parent_sha_exact": sha256(parent) == PARENT_SHA256,
        "parent_counters_exact": construction["parent_global_update"]
        and construction["parent_cumulative_targets"],
        "o2_independent_parent_load": all(construction.values()),
        "n_checkpoint_exact_strict": all(n_checks.values()),
        "o1_checkpoint_exact_strict": all(o1_checks.values()),
        "n_o1_not_retrained": True,
        "sealed_2d7_continuation_exact": sha256(continuation_path) == CONTINUATION_MANIFEST_SHA256
        and sha256(ledger_path) == CONTINUATION_LEDGER_SHA256
        and continuation["ledger_sha256"] == CONTINUATION_LEDGER_SHA256,
        "continuation_191_exact": len(rows) == LOCAL_UPDATES
        and rows[0]["global_update"] == 2100
        and rows[-1]["global_update"] == 2290
        and all(row["target_count"] == TARGETS_PER_UPDATE for row in rows),
        "runtime_geometry": geometry["passed"],
        "disposable_smoke": smoke["passed"] and smoke["official_optimizer_updates"] == 0,
        "first_batch_exact": all(first.values()),
        "only_allowed_geometry_delta": normalized_manifest("O1") == normalized_manifest("O2")
        and core.GEOMETRIES["O2"] == {0: 1, 2: 30, 4: 62},
    }
    audit = {
        "schema": "experiment_2d8_preflight_audit_v1",
        "experiment": EXPERIMENT,
        "git_commit": engine.git("rev-parse", "HEAD"),
        "implementation_sha256": implementation_sha256(),
        "source": {
            "path": str(parent), "sha256": sha256(parent),
            "global_update": PARENT_GLOBAL_UPDATE,
            "cumulative_targets": PARENT_TARGETS,
        },
        "continuation": {
            "manifest_path": str(continuation_path),
            "manifest_sha256": sha256(continuation_path),
            "ledger_path": str(ledger_path),
            "ledger_sha256": sha256(ledger_path),
        },
        "existing_models": {
            "N": {"path": str(Path(args.n_checkpoint).resolve()), "sha256": N_SHA256, "metadata": n_metadata, "checks": n_checks},
            "O1": {"path": str(Path(args.o1_checkpoint).resolve()), "sha256": O1_SHA256, "metadata": o1_metadata, "checks": o1_checks},
        },
        "o2": o2_preflight,
        "checks": checks,
        "authorized": all(checks.values()),
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    d5c.durable_json(output / "PREFLIGHT_AUDIT.json", audit)
    if not audit["authorized"]:
        raise SystemExit(f"2D8 preflight failed: {checks}")
    print("EXPERIMENT_2D8_PREFLIGHT_PASS", flush=True)


def run_train(args):
    if args.arm.upper() != "O2":
        raise SystemExit("Experiment 2D8 trains exactly one arm: O2")
    engine.run_train(args)
    path = Path(args.output_dir).resolve() / "TRAINING_COMPLETE_O2.json"
    summary = read_json(path)
    summary["schema"] = "experiment_2d8_training_complete_v1"
    summary["experiment"] = EXPERIMENT
    summary["existing_models_retrained"] = []
    d5c.durable_json(path, summary)
    print("EXPERIMENT_2D8_TRAINING_COMPLETE O2", flush=True)


def run_prepare_panel(args):
    engine.require_branch(clean=True)
    require_file(args.o2_checkpoint, label="sealed O2 checkpoint barrier")
    verification = read_json(str(Path(args.o2_checkpoint).resolve()) + ".verification.json")
    if not (
        verification.get("strict_reopen", {}).get("passed") is True
        and verification.get("global_update") == FINAL_GLOBAL_UPDATE
        and verification.get("cumulative_targets") == FINAL_TARGETS
        and verification.get("sha256") == sha256(args.o2_checkpoint)
    ):
        raise SystemExit("O2 checkpoint barrier failed before fresh-panel construction")
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
            engine._collect_historical_spans(value, historical_spans)
            if len(historical_spans) != before:
                historical_files.append({
                    "path": str(path.resolve()), "sha256": sha256(path),
                    "new_spans": len(historical_spans) - before,
                })
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
        x, y = engine.batch_arrays(tokens, start)
        identity = engine.batch_identity_numpy(x, y)
        rows = engine.sequence_identities(x, y, start, index)
        selected.append(index)
        identities.append(identity)
        sequences.extend(rows)
        selected_spans.append(list(span))
        historical_spans.add(span)
        if len(selected) == PANEL_BATCHES:
            break
    if len(selected) != PANEL_BATCHES:
        raise SystemExit("could not construct one fresh 2D8 panel")
    panel_hash = engine.aggregate_hashes(row["combined_sha256"] for row in identities)
    manifest = {
        "experiment": EXPERIMENT,
        "panel_name": "fresh disjoint 2D8 matched panel",
        "dataset": "edu_fineweb10B/edufineweb_val_000000.npy",
        "dataset_sha256": DATASET_SHA256,
        "dataset_split": "validation",
        "selection_seed": PANEL_SEED,
        "selection_algorithm": "seeded permutation of complete canonical 64x1024 validation batches; reject every recovered historical target span and reserved prefix 0..127; accept first 64",
        "candidate_panels_constructed": 1,
        "checkpoint_losses_inspected_during_selection": False,
        "sealed_after_o2_checkpoint_before_any_scoring": True,
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
        "o2_checkpoint_barrier_sha256": verification["sha256"],
    }
    checks = {
        "one_panel": manifest["candidate_panels_constructed"] == 1,
        "sealed_before_results": manifest["sealed_after_o2_checkpoint_before_any_scoring"],
        "no_checkpoint_loss_inspection": not manifest["checkpoint_losses_inspected_during_selection"],
        "training_split_disjoint": manifest["training_split_disjoint_by_dataset_split"],
        "sequences": len(sequences) == PANEL_SEQUENCES,
        "targets": len(sequences) * base.T == PANEL_TARGETS,
        "unique_batches": len(set(selected)) == PANEL_BATCHES,
        "unique_sequence_hashes": len({row["combined_sha256"] for row in sequences}) == PANEL_SEQUENCES,
    }
    audit = {
        "schema": "experiment_2d8_panel_disjointness_v1",
        "experiment": EXPERIMENT,
        "panel_sha256": panel_hash,
        "historical_json_files_used": historical_files,
        "recovered_historical_spans_including_reserved_prefix": len(historical_spans) - len(selected_spans),
        "selected_target_spans": selected_spans,
        "checks": checks,
        "passed": all(checks.values()),
    }
    d5c.durable_json(output / "EVALUATION_PANEL_MANIFEST.json", manifest)
    d5c.durable_json(output / "EVALUATION_PANEL_DISJOINTNESS_AUDIT.json", audit)
    if not audit["passed"]:
        raise SystemExit(f"fresh 2D8 panel audit failed: {checks}")
    print(f"EXPERIMENT_2D8_PANEL_FROZEN {panel_hash}", flush=True)


def run_evaluate(args):
    engine.require_branch(clean=True)
    arm = args.arm.upper()
    if arm not in core.GEOMETRIES:
        raise SystemExit("evaluation arm must be N, O1, or O2")
    device = base.require_a100()
    panel = read_json(args.panel_manifest)
    if not (
        panel.get("sequence_count") == PANEL_SEQUENCES
        and panel.get("targets_per_condition") == PANEL_TARGETS
        and panel.get("sealed_after_o2_checkpoint_before_any_scoring") is True
    ):
        raise SystemExit("evaluation requires the sealed fresh 4,096-sequence panel")
    checkpoint = Path(args.checkpoint).resolve()
    if arm in {"N", "O1"}:
        model, metadata, checkpoint_checks = strict_existing_checkpoint(checkpoint, arm, device)
    else:
        model, metadata, checkpoint_checks = strict_o2_checkpoint(checkpoint, device)
    output_path = Path(args.output_path).resolve()
    if output_path.exists():
        raise SystemExit("refusing to overwrite or resume a final 2D8 evaluation")
    state = {
        "schema": "experiment_2d8_true_incremental_losses_v1",
        "experiment": EXPERIMENT,
        "condition": core.CONDITIONS[arm], "arm": arm,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_metadata": metadata, "checkpoint_checks": checkpoint_checks,
        "architecture_fingerprint": core.architecture_fingerprint(arm),
        "panel_manifest_sha256": sha256(args.panel_manifest),
        "panel_sha256": panel["panel_sha256"],
        "evaluation_set_label": "fresh previously unexamined disjoint matched panel",
        "precision": "BF16 model execution; FP32 token CE; FP64 accumulation",
        "batch_indices_in_evaluation_order": panel["batch_indices_in_evaluation_order"],
        "completed_batch_indices": [], "batch_identities": [],
        "nll_sum": 0.0, "targets": 0,
        "per_sequence_nll": [], "per_sequence_ce": [], "status": "running",
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
                row = engine.incremental_condition(model, x, y, audit=ordinal == 0)
            state["nll_sum"] += row["nll_sum"]
            state["targets"] += row["targets"]
            state["per_sequence_nll"].extend(row["per_sequence_nll"])
            state["per_sequence_ce"].extend(row["per_sequence_ce"])
            state["completed_batch_indices"].append(int(batch_index))
            state["batch_identities"].append(observed)
            state["final_cache_audit"] = row["final_cache_audit"]
            d5c.durable_json(output_path, state)
            print(f"2D8 {arm} true-incremental batch {ordinal + 1}/{PANEL_BATCHES}", flush=True)
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
        state["targets"] == PANEL_TARGETS
        and state["paired_sequences"] == PANEL_SEQUENCES
        and state["batch_identities"] == panel["batch_identities"]
        and state["final_cache_audit"]["passed"]
        and all(checkpoint_checks.values())
    )
    d5c.durable_json(output_path, state)
    if not state["passed"]:
        raise SystemExit(f"final evaluation failed for {arm}")
    print(f"EXPERIMENT_2D8_EVALUATION_COMPLETE {arm}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Experiment 2D8 scientific driver")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    for name in (
        "parent_checkpoint", "n_checkpoint", "o1_checkpoint",
        "continuation_manifest", "continuation_ledger", "output_dir",
    ):
        preflight.add_argument(f"--{name.replace('_', '-')}", required=True)
    preflight.set_defaults(handler=run_preflight)
    train = commands.add_parser("train")
    train.add_argument("--arm", choices=("O2",), default="O2")
    for name in (
        "parent_checkpoint", "continuation_manifest", "continuation_ledger",
        "preflight_audit", "output_dir", "checkpoint_dir",
    ):
        train.add_argument(f"--{name.replace('_', '-')}", required=True)
    train.set_defaults(handler=run_train)
    panel = commands.add_parser("prepare-panel")
    panel.add_argument("--dataset", required=True)
    panel.add_argument("--o2-checkpoint", required=True)
    panel.add_argument("--output-dir", required=True)
    panel.set_defaults(handler=run_prepare_panel)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--arm", choices=("N", "O1", "O2"), required=True)
    for name in ("checkpoint", "panel_manifest", "data_root", "output_path"):
        evaluate.add_argument(f"--{name.replace('_', '-')}", required=True)
    evaluate.set_defaults(handler=run_evaluate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
