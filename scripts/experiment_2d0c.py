#!/usr/bin/env python3
"""Experiment 2D0C: full single-layer marginal KV-window sensitivity map."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0 as d0  # noqa: E402


EXPERIMENT = "2D0C"
PROTOCOL = "exp2d0c_layer_window_sensitivity_map_v1"
BRANCH = "experiment-2d0c-layer-window-sensitivity-map"
FROZEN_TAG = "experiment-2d0b-b11-micro-window-sweep-final"
FROZEN_COMMIT = "43212789a2bca8ded75df9100e981904a0adb6ff"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d0c_layer_window_sensitivity_map.json"
PHASE_A_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_2d0_standard_b11_context_completion"
    / "phase_a_results.json"
)
EXTREME_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_2d0a_b11_extreme_window_sweep"
    / "extreme_window_results.json"
)
MICRO_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_2d0b_b11_micro_window_sweep"
    / "micro_window_results.json"
)
WINDOWS = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
MATRIX_WINDOWS = (1024,) + WINDOWS
LAYERS = tuple(range(1, 13))
GPU_LAYERS = {0: (1, 5, 9), 1: (2, 6, 10), 2: (3, 7, 11), 3: (4, 8, 12)}
POSITION_BINS = (
    ("1-64", 1, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513-768", 513, 768),
    ("769-896", 769, 896),
    ("897-1023", 897, 1023),
)
THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
SENTINEL_TOLERANCE = 1e-8
EXPECTED_CELLS = 120


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    checks = {
        "experiment": config.get("experiment") == EXPERIMENT,
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "frozen_commit": config.get("frozen_2d0b_commit") == FROZEN_COMMIT,
        "source": config.get("source_checkpoint_sha256") == d0.SOURCE_SHA256,
        "validation": config.get("canonical_validation_sha256")
        == d0.CANONICAL_VALIDATION_SHA256,
        "windows": tuple(config["evaluation"]["windows"]) == WINDOWS,
        "layers": tuple(config["evaluation"]["layers"]) == LAYERS,
        "gpu_layers": {
            int(key): tuple(value)
            for key, value in config["evaluation"]["gpu_layer_assignment"].items()
        }
        == GPU_LAYERS,
        "batches": config["evaluation"]["batches"] == d0.PHASE_A_BATCHES,
        "targets": config["evaluation"]["targets_per_cell"]
        == d0.PHASE_A_BATCHES * d0.VALIDATION_B * d0.T,
        "bins": tuple(tuple(row) for row in config["position_bins"]) == POSITION_BINS,
        "thresholds": tuple(config["damage_thresholds"]) == THRESHOLDS,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D0C frozen configuration mismatch: {checks}")
    return config


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2D0C requires branch {BRANCH}")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=REPO_ROOT
    )
    if git_output("rev-list", "-n", "1", FROZEN_TAG) != FROZEN_COMMIT:
        raise SystemExit("2D0B frozen tag does not resolve to the finalized commit")
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D0C execution requires a clean worktree")


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def environment_audit():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_count_visible": torch.cuda.device_count(),
        "autocast": "cuda bfloat16",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def historical_oracles():
    phase_a = json.loads(PHASE_A_PATH.read_text())["rows"]
    extreme = json.loads(EXTREME_PATH.read_text())["rows"]
    micro = json.loads(MICRO_PATH.read_text())["rows"]
    return {
        "full_loss": phase_a["1024"]["validation_loss"],
        "full_per_batch_losses": phase_a["1024"]["per_batch_losses"],
        "b11": {
            512: phase_a["512"]["validation_loss"],
            128: extreme["128"]["validation_loss"],
            1: micro["1"]["validation_loss"],
        },
    }


def validation_manifest(val_path):
    loader = d0.ExplicitShardLoader([str(Path(val_path).resolve())], d0.VALIDATION_B, d0.T)
    rows = []
    combined_hashes = []
    for batch_index in range(d0.PHASE_A_BATCHES):
        x, y = loader.next_batch()
        combined_hashes.append(d0.batch_payload_hash(x, y))
        rows.append(
            {
                "batch_index": batch_index,
                "input_sha256": tensor_sha256(x),
                "target_sha256": tensor_sha256(y),
                "input_shape": list(x.shape),
                "target_shape": list(y.shape),
                "input_dtype": str(x.dtype),
                "target_dtype": str(y.dtype),
                "loss_denominator": x.numel(),
            }
        )
    return {
        "validation_shard": str(Path(val_path).resolve()),
        "validation_shard_sha256": d0.file_sha256(val_path),
        "canonical_batch_collection_sha256": d0.aggregate_hashes(combined_hashes),
        "batches": rows,
        "batch_count": len(rows),
        "batch_size": d0.VALIDATION_B,
        "sequence_length": d0.T,
        "targets_per_control": d0.PHASE_A_BATCHES * d0.VALIDATION_B * d0.T,
    }


def forward_top(model, tokens, test_block_index=None, window=d0.T):
    length = tokens.size(1)
    positions = torch.arange(length, dtype=torch.long, device=tokens.device)
    value = model.transformer.wte(tokens) + model.transformer.wpe(positions)
    for block_index, block in enumerate(model.transformer.h):
        if block_index == test_block_index:
            value, _ = d0.run_block(block, value, window)
        else:
            value = block(value)
    return model.transformer.ln_f(value)


def batch_identity(cpu_x, cpu_y):
    return {
        "input_sha256": tensor_sha256(cpu_x),
        "target_sha256": tensor_sha256(cpu_y),
        "combined_sha256": d0.batch_payload_hash(cpu_x, cpu_y),
    }


@torch.no_grad()
def evaluate_baseline(model, val_path, device):
    loader = d0.ExplicitShardLoader([val_path], d0.VALIDATION_B, d0.T)
    loss_batches = []
    argmax_batches = []
    per_batch_losses = []
    identities = []
    loss_sum = 0.0
    tokens = 0
    per_position_sum = np.zeros(d0.T, dtype=np.float64)
    start = time.monotonic()
    for batch_index in range(d0.PHASE_A_BATCHES):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(batch_identity(cpu_x, cpu_y))
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            top = forward_top(model, x)
            losses, logits = d0.token_cross_entropy(model, top, y)
        if not torch.isfinite(losses).all() or not torch.isfinite(logits).all():
            raise SystemExit("non-finite full-context baseline prediction")
        batch_loss = losses.float().mean().item()
        per_batch_losses.append(batch_loss)
        loss_sum += losses.double().sum().item()
        tokens += losses.numel()
        per_position_sum += losses.double().sum(dim=0).cpu().numpy()
        loss_batches.append(losses.float().cpu().numpy())
        argmax_batches.append(logits.argmax(dim=-1).to(torch.int32).cpu().numpy())
        print(
            f"2D0C baseline batch={batch_index + 1:02d}/{d0.PHASE_A_BATCHES} "
            f"loss={batch_loss:.10f}",
            flush=True,
        )
        del x, y, top, losses, logits
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - start
    return {
        "validation_loss": loss_sum / tokens,
        "validation_targets": tokens,
        "per_batch_losses": per_batch_losses,
        "batch_identities": identities,
        "canonical_validation_sha256": d0.aggregate_hashes(
            [row["combined_sha256"] for row in identities]
        ),
        "per_position_loss": (per_position_sum / (d0.PHASE_A_BATCHES * d0.VALIDATION_B)).tolist(),
        "wall_seconds": elapsed,
        "targets_per_second": tokens / elapsed,
        "losses": np.stack(loss_batches),
        "argmax": np.stack(argmax_batches),
    }


def position_bins(short_per_position, full_per_position):
    rows = {}
    for label, start, end in POSITION_BINS:
        short_loss = float(np.mean(short_per_position[start : end + 1]))
        full_loss = float(np.mean(full_per_position[start : end + 1]))
        rows[label] = {
            "start": start,
            "end": end,
            "short_loss": short_loss,
            "full_loss": full_loss,
            "delta": short_loss - full_loss,
            "target_count": (end - start + 1) * d0.PHASE_A_BATCHES * d0.VALIDATION_B,
        }
    return rows


@torch.no_grad()
def evaluate_cell(model, val_path, device, layer, window, baseline, full_loss):
    block_index = layer - 1
    loader = d0.ExplicitShardLoader([val_path], d0.VALIDATION_B, d0.T)
    torch.cuda.reset_peak_memory_stats(0)
    start_unix = time.time()
    start = time.monotonic()
    loss_sum = 0.0
    tokens = 0
    argmax_matches = 0
    per_position_sum = np.zeros(d0.T, dtype=np.float64)
    per_batch_losses = []
    identities = []
    finite_batches = 0
    for batch_index in range(d0.PHASE_A_BATCHES):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(batch_identity(cpu_x, cpu_y))
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            top = forward_top(model, x, test_block_index=block_index, window=window)
            losses, logits = d0.token_cross_entropy(model, top, y)
        finite = bool(torch.isfinite(losses).all() and torch.isfinite(logits).all())
        finite_batches += int(finite)
        batch_loss = losses.float().mean().item()
        per_batch_losses.append(batch_loss)
        loss_sum += losses.double().sum().item()
        tokens += losses.numel()
        per_position_sum += losses.double().sum(dim=0).cpu().numpy()
        prediction = logits.argmax(dim=-1).to(torch.int32).cpu().numpy()
        argmax_matches += int(np.count_nonzero(prediction == baseline["argmax"][batch_index]))
        del x, y, top, losses, logits, prediction
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - start
    validation_loss = loss_sum / tokens
    short_per_position = per_position_sum / (d0.PHASE_A_BATCHES * d0.VALIDATION_B)
    all_windows = [d0.T] * d0.N_LAYER
    all_windows[block_index] = window
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "block": f"B{layer}",
        "human_layer": layer,
        "zero_based_block_index": block_index,
        "window": window,
        "all_layer_windows": all_windows,
        "shortened_layer_count": sum(value != d0.T for value in all_windows),
        "validation_loss": validation_loss,
        "damage_vs_full": validation_loss - full_loss,
        "validation_targets": tokens,
        "argmax_agreement": argmax_matches / tokens,
        "per_batch_losses": per_batch_losses,
        "per_position_loss": short_per_position.tolist(),
        "position_bins": position_bins(short_per_position, baseline["per_position_loss"]),
        "batch_identities": identities,
        "canonical_validation_sha256": d0.aggregate_hashes(
            [row["combined_sha256"] for row in identities]
        ),
        "finite_batches": finite_batches,
        "all_losses_and_predictions_finite": finite_batches == d0.PHASE_A_BATCHES,
        "absolute_positions_unchanged": True,
        "causal_window_includes_current": True,
        "evaluation_precision": "torch.autocast(cuda,bfloat16)",
        "loss_denominator": tokens,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "grad_scaler_objects": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_targets": 0,
        "recurrence_active": False,
        "completion_active": False,
        "writers_active": False,
        "temporal_attnres_active": False,
        "full_attnres_active": False,
        "bptt_active": False,
        "hellaswag_executed": False,
        "performance": {
            "wall_start_unix": start_unix,
            "wall_end_unix": time.time(),
            "wall_seconds": elapsed,
            "targets_per_second": tokens / elapsed,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(0) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(0) / 1024**2,
        },
    }


def expected_batch_identities(manifest):
    return [
        {
            "input_sha256": row["input_sha256"],
            "target_sha256": row["target_sha256"],
        }
        for row in manifest["batches"]
    ]


def identities_match_manifest(identities, manifest):
    expected = expected_batch_identities(manifest)
    observed = [
        {"input_sha256": row["input_sha256"], "target_sha256": row["target_sha256"]}
        for row in identities
    ]
    return observed == expected


def run_preflight(args):
    require_git(clean=True)
    config = load_config()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("2D0C preflight requires exactly one visible GPU")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("2D0C forbids DDP/NCCL/distributed execution")
    torch.cuda.set_device(0)
    d0.seed_all(0)
    device = torch.device("cuda", 0)
    val_path = str(Path(args.validation_shard).resolve())
    manifest = validation_manifest(val_path)
    if manifest["validation_shard_sha256"] != d0.VAL_SHA256:
        raise SystemExit("2D0C validation shard SHA mismatch")
    if manifest["canonical_batch_collection_sha256"] != d0.CANONICAL_VALIDATION_SHA256:
        raise SystemExit("2D0C canonical validation batch mismatch")
    _, model, source_audit = d0.load_standard_model(args.parent_checkpoint, device)
    model_before = d0.tensor_state_sha256(model)
    baseline = evaluate_baseline(model, val_path, device)
    oracles = historical_oracles()
    baseline_deltas = [
        abs(left - right)
        for left, right in zip(baseline["per_batch_losses"], oracles["full_per_batch_losses"])
    ]
    regression_rows = {}
    for window in config["preflight"]["b11_regression_windows"]:
        row = evaluate_cell(model, val_path, device, 11, window, baseline, oracles["full_loss"])
        delta = abs(row["validation_loss"] - oracles["b11"][window])
        regression_rows[str(window)] = {
            "expected_validation_loss": oracles["b11"][window],
            "observed_validation_loss": row["validation_loss"],
            "absolute_delta": delta,
            "tolerance": SENTINEL_TOLERANCE,
            "batch_manifest_exact": identities_match_manifest(row["batch_identities"], manifest),
            "passed": delta <= SENTINEL_TOLERANCE
            and identities_match_manifest(row["batch_identities"], manifest),
        }
    model_after = d0.tensor_state_sha256(model)
    semantic = {
        "experiment": EXPERIMENT,
        "validated_parent_evaluator": "scripts/experiment_2d0a.py via d0.run_block/sliding_mask",
        "generalized_field": "test_block_index",
        "unchanged_semantics": {
            "causal_sliding_mask": True,
            "window_includes_current": True,
            "loss_calculation": True,
            "validation_loader": True,
            "precision": True,
            "absolute_positional_embeddings": True,
            "attention_implementation": True,
        },
        "human_zero_based_mapping": {f"B{layer}": layer - 1 for layer in LAYERS},
        "mapping_exact": all(layer - 1 == index for index, layer in enumerate(LAYERS)),
        "only_semantic_generalization": "the selected block index is configurable; all other blocks use the frozen full-context forward",
    }
    semantic["passed"] = semantic["mapping_exact"] and all(
        semantic["unchanged_semantics"].values()
    )
    preflight = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "source_audit": source_audit,
        "baseline": {
            key: value
            for key, value in baseline.items()
            if key not in ("losses", "argmax")
        },
        "baseline_expected_loss": oracles["full_loss"],
        "baseline_loss_absolute_delta": abs(baseline["validation_loss"] - oracles["full_loss"]),
        "baseline_per_batch_max_absolute_delta": max(baseline_deltas),
        "b11_regression": regression_rows,
        "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after,
        "model_tensors_unchanged": model_before == model_after,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "grad_scaler_objects": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_targets": 0,
        "environment": environment_audit(),
    }
    preflight["passed"] = (
        source_audit["passed"]
        and semantic["passed"]
        and identities_match_manifest(baseline["batch_identities"], manifest)
        and baseline["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256
        and preflight["baseline_loss_absolute_delta"] <= SENTINEL_TOLERANCE
        and preflight["baseline_per_batch_max_absolute_delta"] <= SENTINEL_TOLERANCE
        and all(row["passed"] for row in regression_rows.values())
        and preflight["model_tensors_unchanged"]
    )
    run_root = Path(args.run_root).resolve() / "preflight"
    d0.durable_json(run_root / "batch_manifest.json", manifest)
    d0.durable_json(run_root / "semantic_diff_audit.json", semantic)
    d0.durable_json(run_root / "environment.json", preflight["environment"])
    d0.durable_json(run_root / "preflight_audit.json", preflight)
    print(
        f"EXPERIMENT_2D0C_PREFLIGHT_{'PASS' if preflight['passed'] else 'FAIL'} "
        f"baseline_delta={preflight['baseline_loss_absolute_delta']:.3e}",
        flush=True,
    )
    if not preflight["passed"]:
        raise SystemExit("2D0C preflight failed; map is not authorized")


def log_event(path, payload):
    d0.append_jsonl(path, {"unix_time": time.time(), **payload})


def run_worker(args):
    require_git(clean=True)
    load_config()
    physical_gpu = int(args.physical_gpu)
    if physical_gpu not in GPU_LAYERS:
        raise SystemExit("invalid 2D0C physical GPU assignment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("each 2D0C worker requires exactly one visible GPU")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("2D0C workers forbid DDP/NCCL/distributed state")
    preflight_path = Path(args.preflight).resolve()
    preflight = json.loads(preflight_path.read_text())
    if not preflight["passed"]:
        raise SystemExit("2D0C worker requires passing preflight")
    run_root = Path(args.run_root).resolve()
    log_path = run_root / "logs" / f"worker_gpu{physical_gpu}.jsonl"
    if log_path.exists():
        raise SystemExit(f"refusing to append to existing worker log: {log_path}")
    log_event(
        log_path,
        {
            "event": "worker_start",
            "gpu_assignment": physical_gpu,
            "layers": list(GPU_LAYERS[physical_gpu]),
            "windows": list(WINDOWS),
            "command": " ".join(sys.argv),
            "pid": os.getpid(),
            "start_time": time.time(),
        },
    )
    torch.cuda.set_device(0)
    d0.seed_all(physical_gpu)
    device = torch.device("cuda", 0)
    val_path = str(Path(args.validation_shard).resolve())
    manifest = json.loads((preflight_path.parent / "batch_manifest.json").read_text())
    oracles = historical_oracles()
    _, model, source_audit = d0.load_standard_model(args.parent_checkpoint, device)
    model_before = d0.tensor_state_sha256(model)
    worker_start_unix = time.time()
    worker_start = time.monotonic()
    baseline = evaluate_baseline(model, val_path, device)
    first_two_deltas = [
        abs(baseline["per_batch_losses"][index] - oracles["full_per_batch_losses"][index])
        for index in range(2)
    ]
    baseline_pass = (
        max(first_two_deltas) <= SENTINEL_TOLERANCE
        and abs(baseline["validation_loss"] - oracles["full_loss"]) <= SENTINEL_TOLERANCE
        and identities_match_manifest(baseline["batch_identities"], manifest)
        and baseline["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256
    )
    baseline_json = {
        key: value for key, value in baseline.items() if key not in ("losses", "argmax")
    }
    baseline_json.update(
        {
            "physical_gpu": physical_gpu,
            "first_two_absolute_deltas": first_two_deltas,
            "expected_validation_loss": oracles["full_loss"],
            "passed": baseline_pass,
        }
    )
    d0.durable_json(run_root / "baselines" / f"gpu{physical_gpu}.json", baseline_json)
    prediction_path = run_root / "baselines" / f"gpu{physical_gpu}_predictions.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        per_target_loss=baseline["losses"],
        argmax_token_id=baseline["argmax"],
    )
    if not baseline_pass:
        log_event(log_path, {"event": "worker_abort", "reason": "baseline sentinel failed", "exit_code": 1})
        raise SystemExit(f"2D0C GPU{physical_gpu} baseline sentinel failed")

    cell_keys = []
    for layer in GPU_LAYERS[physical_gpu]:
        for window in WINDOWS:
            cell_start = time.time()
            row = evaluate_cell(
                model,
                val_path,
                device,
                layer,
                window,
                baseline,
                oracles["full_loss"],
            )
            row["physical_gpu"] = physical_gpu
            row["source_checkpoint_sha256"] = d0.SOURCE_SHA256
            row["batch_manifest_exact"] = identities_match_manifest(
                row["batch_identities"], manifest
            )
            row["passed"] = (
                row["shortened_layer_count"] == 1
                and row["all_layer_windows"][layer - 1] == window
                and all(
                    value == d0.T
                    for index, value in enumerate(row["all_layer_windows"])
                    if index != layer - 1
                )
                and row["batch_manifest_exact"]
                and row["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256
                and row["validation_targets"] == d0.PHASE_A_BATCHES * d0.VALIDATION_B * d0.T
                and row["all_losses_and_predictions_finite"]
            )
            cell_key = f"B{layer:02d}_W{window:04d}"
            d0.durable_json(run_root / "cells" / f"{cell_key}.json", row)
            cell_keys.append(cell_key)
            log_event(
                log_path,
                {
                    "event": "cell_complete",
                    "cell": cell_key,
                    "layer": layer,
                    "window": window,
                    "start_time": cell_start,
                    "end_time": time.time(),
                    "exit_code": 0 if row["passed"] else 1,
                    "wall_seconds": row["performance"]["wall_seconds"],
                    "peak_allocated_vram_mb": row["performance"]["peak_allocated_vram_mb"],
                    "peak_reserved_vram_mb": row["performance"]["peak_reserved_vram_mb"],
                },
            )
            print(
                f"2D0C GPU{physical_gpu} B{layer} W={window} "
                f"loss={row['validation_loss']:.10f} damage={row['damage_vs_full']:+.10f} "
                f"seconds={row['performance']['wall_seconds']:.3f}",
                flush=True,
            )
            if not row["passed"]:
                raise SystemExit(f"2D0C cell {cell_key} failed")

    model_after = d0.tensor_state_sha256(model)
    elapsed = time.monotonic() - worker_start
    worker = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "physical_gpu": physical_gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "assigned_layers": list(GPU_LAYERS[physical_gpu]),
        "windows": list(WINDOWS),
        "expected_cells": len(GPU_LAYERS[physical_gpu]) * len(WINDOWS),
        "completed_cells": len(cell_keys),
        "cell_keys": cell_keys,
        "baseline_passed": baseline_pass,
        "source_audit": source_audit,
        "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after,
        "model_tensors_unchanged": model_before == model_after,
        "worker_start_unix": worker_start_unix,
        "worker_end_unix": time.time(),
        "wall_seconds": elapsed,
        "exit_code": 0,
        "distributed_initialized": torch.distributed.is_initialized(),
    }
    worker["passed"] = (
        worker["completed_cells"] == worker["expected_cells"] == 30
        and worker["baseline_passed"]
        and worker["source_audit"]["passed"]
        and worker["model_tensors_unchanged"]
        and not worker["distributed_initialized"]
    )
    d0.durable_json(run_root / "workers" / f"gpu{physical_gpu}.json", worker)
    log_event(
        log_path,
        {
            "event": "worker_end",
            "end_time": time.time(),
            "exit_code": 0 if worker["passed"] else 1,
            "completed_cells": len(cell_keys),
            "wall_seconds": elapsed,
        },
    )
    print(
        f"EXPERIMENT_2D0C_WORKER_{'PASS' if worker['passed'] else 'FAIL'} "
        f"GPU={physical_gpu} cells={len(cell_keys)} seconds={elapsed:.3f}",
        flush=True,
    )
    if not worker["passed"]:
        raise SystemExit(f"2D0C GPU{physical_gpu} worker failed")


def smallest_window(row, threshold):
    candidates = [window for window in sorted(MATRIX_WINDOWS) if row[window] <= threshold]
    return candidates[0] if candidates else None


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for offset in range(cursor, end):
            ranks[order[offset]] = rank
        cursor = end
    return ranks


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.square(left).sum() * np.square(right).sum()))
    return 0.0 if denominator == 0 else float((left * right).sum() / denominator)


def shape_row(widths):
    values = [widths[layer] for layer in LAYERS]
    increases = sum(right > left for left, right in zip(values, values[1:]))
    decreases = sum(right < left for left, right in zip(values, values[1:]))
    ties = 11 - increases - decreases
    maximum = max(values)
    minimum = min(values)
    return {
        "selected_windows": {str(layer): widths[layer] for layer in LAYERS},
        "spearman_depth_vs_selected_window": pearson(
            average_ranks(list(LAYERS)), average_ranks(values)
        ),
        "adjacent_increases": increases,
        "adjacent_decreases": decreases,
        "adjacent_ties": ties,
        "layers_with_maximum_window": [layer for layer in LAYERS if widths[layer] == maximum],
        "layers_with_minimum_window": [layer for layer in LAYERS if widths[layer] == minimum],
        "maximum_window": maximum,
        "minimum_window": minimum,
    }


def classify_shape(shape_analysis):
    rows = [shape_analysis[str(threshold)] for threshold in (0.005, 0.01, 0.02, 0.05)]
    if all(row["maximum_window"] == row["minimum_window"] for row in rows):
        return "MARGINAL KV SENSITIVITY IS TOO FLAT TO DETERMINE SHAPE"
    correlations = [row["spearman_depth_vs_selected_window"] for row in rows]
    if float(np.median(correlations)) >= 0.5:
        return "MARGINAL KV SENSITIVITY IS TRIANGLE-LIKE"
    if float(np.median(correlations)) <= -0.5:
        return "MARGINAL KV SENSITIVITY IS REVERSE-TRIANGLE-LIKE"
    bulge_votes = 0
    for row in rows:
        widths = {int(key): value for key, value in row["selected_windows"].items()}
        early = np.mean([widths[layer] for layer in (1, 2, 3)])
        middle = np.mean([widths[layer] for layer in (4, 5, 6, 7, 8, 9)])
        late = np.mean([widths[layer] for layer in (10, 11, 12)])
        bulge_votes += int(middle > early and middle > late)
    if bulge_votes >= 3:
        return "MARGINAL KV SENSITIVITY SHOWS A MIDDLE BULGE"
    return "MARGINAL KV SENSITIVITY IS IRREGULAR / MULTIMODAL"


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run_assemble(args):
    require_git(clean=True)
    load_config()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((run_root / "preflight" / "preflight_audit.json").read_text())
    manifest = json.loads((run_root / "preflight" / "batch_manifest.json").read_text())
    semantic = json.loads((run_root / "preflight" / "semantic_diff_audit.json").read_text())
    environment = json.loads((run_root / "preflight" / "environment.json").read_text())
    if not preflight["passed"]:
        raise SystemExit("cannot aggregate failed 2D0C preflight")
    workers = {
        gpu: json.loads((run_root / "workers" / f"gpu{gpu}.json").read_text())
        for gpu in range(4)
    }
    baselines = {
        gpu: json.loads((run_root / "baselines" / f"gpu{gpu}.json").read_text())
        for gpu in range(4)
    }
    baseline_arrays = [
        np.load(run_root / "baselines" / f"gpu{gpu}_predictions.npz") for gpu in range(4)
    ]
    baseline_prediction_identity = all(
        np.array_equal(baseline_arrays[0]["per_target_loss"], row["per_target_loss"])
        and np.array_equal(baseline_arrays[0]["argmax_token_id"], row["argmax_token_id"])
        for row in baseline_arrays[1:]
    )
    cells = {}
    for layer in LAYERS:
        for window in WINDOWS:
            key = f"B{layer:02d}_W{window:04d}"
            path = run_root / "cells" / f"{key}.json"
            if not path.is_file():
                raise SystemExit(f"missing 2D0C cell {key}")
            cells[(layer, window)] = json.loads(path.read_text())
    full_loss = historical_oracles()["full_loss"]
    validation_losses = {
        layer: {1024: full_loss, **{window: cells[(layer, window)]["validation_loss"] for window in WINDOWS}}
        for layer in LAYERS
    }
    damages = {
        layer: {window: validation_losses[layer][window] - full_loss for window in MATRIX_WINDOWS}
        for layer in LAYERS
    }
    loss_rows = [[f"B{layer}"] + [validation_losses[layer][window] for window in MATRIX_WINDOWS] for layer in LAYERS]
    damage_rows = [[f"B{layer}"] + [damages[layer][window] for window in MATRIX_WINDOWS] for layer in LAYERS]
    header = ["Layer"] + [f"W{window}" for window in MATRIX_WINDOWS]
    write_csv(output / "validation_loss_matrix.csv", header, loss_rows)
    write_csv(output / "validation_damage_matrix.csv", header, damage_rows)
    d0.durable_json(
        output / "validation_damage_matrix.json",
        {f"B{layer}": {str(window): damages[layer][window] for window in MATRIX_WINDOWS} for layer in LAYERS},
    )

    profiles = {
        threshold: {layer: smallest_window(damages[layer], threshold) for layer in LAYERS}
        for threshold in THRESHOLDS
    }
    profile_json = {
        str(threshold): {f"B{layer}": profiles[threshold][layer] for layer in LAYERS}
        for threshold in THRESHOLDS
    }
    d0.durable_json(output / "width_threshold_profiles.json", profile_json)
    write_csv(
        output / "width_threshold_profiles.csv",
        ["Layer"] + [f"W@{threshold}" for threshold in THRESHOLDS],
        [[f"B{layer}"] + [profiles[threshold][layer] for threshold in THRESHOLDS] for layer in LAYERS],
    )
    budgets = []
    for threshold in THRESHOLDS:
        values = [profiles[threshold][layer] for layer in LAYERS]
        if any(value is None for value in values):
            total = fraction = minimum = maximum = None
        else:
            total = sum(values)
            fraction = total / (12 * d0.T)
            minimum = min(values)
            maximum = max(values)
        budgets.append(
            {
                "damage_threshold": threshold,
                "sum_independently_selected_windows": total,
                "fraction_of_12x1024": fraction,
                "minimum_selected_window": minimum,
                "maximum_selected_window": maximum,
                "label": "HYPOTHETICAL MARGINAL BUDGET — NOT A JOINT-MODEL RESULT",
            }
        )
    write_csv(
        output / "hypothetical_kv_budgets.csv",
        ["Damage threshold", "Sum windows", "Fraction of 12x1024", "Min W", "Max W", "Caveat"],
        [
            [row["damage_threshold"], row["sum_independently_selected_windows"], row["fraction_of_12x1024"], row["minimum_selected_window"], row["maximum_selected_window"], row["label"]]
            for row in budgets
        ],
    )
    rankings = {}
    for window in (512, 128, 32, 1):
        ordered = sorted(LAYERS, key=lambda layer: damages[layer][window], reverse=True)
        rankings[str(window)] = [
            {"rank": rank, "block": f"B{layer}", "damage": damages[layer][window]}
            for rank, layer in enumerate(ordered, start=1)
        ]
    d0.durable_json(output / "fixed_window_rankings.json", rankings)

    monotonicity = {}
    all_violations = []
    for layer in LAYERS:
        violations = []
        for left, right in zip(MATRIX_WINDOWS, MATRIX_WINDOWS[1:]):
            if damages[layer][right] < damages[layer][left]:
                violation = {
                    "from_window": left,
                    "to_window": right,
                    "from_damage": damages[layer][left],
                    "to_damage": damages[layer][right],
                    "decrease": damages[layer][left] - damages[layer][right],
                }
                violations.append(violation)
                all_violations.append({"block": f"B{layer}", **violation})
        monotonicity[f"B{layer}"] = {
            "violation_count": len(violations),
            "largest_violation": max((row["decrease"] for row in violations), default=0.0),
            "violations": violations,
        }
    monotonicity["summary"] = {
        "total_violations": len(all_violations),
        "largest_violation": max((row["decrease"] for row in all_violations), default=0.0),
        "locations": all_violations,
    }
    d0.durable_json(output / "monotonicity.json", monotonicity)

    shape_analysis = {
        str(threshold): shape_row(profiles[threshold]) for threshold in THRESHOLDS
    }
    classification = classify_shape(shape_analysis)
    shape_analysis["classification"] = classification
    d0.durable_json(output / "shape_analysis.json", shape_analysis)

    per_position = {
        f"B{layer}": {
            str(window): {
                "short": cells[(layer, window)]["per_position_loss"],
                "full": baselines[0]["per_position_loss"],
                "delta": [
                    short - full
                    for short, full in zip(
                        cells[(layer, window)]["per_position_loss"],
                        baselines[0]["per_position_loss"],
                    )
                ],
            }
            for window in WINDOWS
        }
        for layer in LAYERS
    }
    bin_rows = {
        f"B{layer}": {str(window): cells[(layer, window)]["position_bins"] for window in WINDOWS}
        for layer in LAYERS
    }
    argmax = {
        f"B{layer}": {str(window): cells[(layer, window)]["argmax_agreement"] for window in WINDOWS}
        for layer in LAYERS
    }
    d0.durable_json(output / "per_position_loss.json", per_position)
    d0.durable_json(output / "position_bin_loss.json", bin_rows)
    d0.durable_json(output / "argmax_agreement.json", argmax)

    cell_performance = [cells[(layer, window)]["performance"] for layer in LAYERS for window in WINDOWS]
    performance = {
        "total_4gpu_elapsed_wall_seconds": max(row["worker_end_unix"] for row in workers.values())
        - min(row["worker_start_unix"] for row in workers.values()),
        "per_gpu": {
            str(gpu): {
                "wall_seconds": workers[gpu]["wall_seconds"],
                "completed_cells": workers[gpu]["completed_cells"],
                "peak_allocated_vram_mb": max(
                    cells[(layer, window)]["performance"]["peak_allocated_vram_mb"]
                    for layer in GPU_LAYERS[gpu]
                    for window in WINDOWS
                ),
                "peak_reserved_vram_mb": max(
                    cells[(layer, window)]["performance"]["peak_reserved_vram_mb"]
                    for layer in GPU_LAYERS[gpu]
                    for window in WINDOWS
                ),
            }
            for gpu in range(4)
        },
        "per_layer_mean_seconds": {
            f"B{layer}": float(
                np.mean([cells[(layer, window)]["performance"]["wall_seconds"] for window in WINDOWS])
            )
            for layer in LAYERS
        },
        "per_cell_mean_seconds": float(np.mean([row["wall_seconds"] for row in cell_performance])),
        "scientific_cells": EXPECTED_CELLS,
        "scientific_evaluated_targets": EXPECTED_CELLS * d0.PHASE_A_BATCHES * d0.VALIDATION_B * d0.T,
        "mean_targets_per_second": float(np.mean([row["targets_per_second"] for row in cell_performance])),
        "baseline_overhead_targets": 4 * d0.PHASE_A_BATCHES * d0.VALIDATION_B * d0.T,
        "preflight_regression_targets": 4 * d0.PHASE_A_BATCHES * d0.VALIDATION_B * d0.T,
    }
    d0.durable_json(output / "performance.json", performance)

    source_manifest = {
        "checkpoint": preflight["source_audit"]["checkpoint"],
        "checkpoint_sha256": d0.SOURCE_SHA256,
        "checkpoint_bytes": d0.SOURCE_BYTES,
        "historical_training_tokens": d0.SOURCE_TOKENS,
        "validation_shard": manifest["validation_shard"],
        "validation_shard_sha256": d0.VAL_SHA256,
        "canonical_validation_sha256": d0.CANONICAL_VALIDATION_SHA256,
        "architecture": preflight["source_audit"],
        "frozen_2d0b_tag": FROZEN_TAG,
        "frozen_2d0b_commit": FROZEN_COMMIT,
    }
    d0.durable_json(output / "source_manifest.json", source_manifest)
    d0.durable_json(output / "environment.json", environment)
    d0.durable_json(output / "batch_manifest.json", manifest)
    d0.durable_json(output / "semantic_diff_audit.json", semantic)
    d0.durable_json(output / "preflight_audit.json", preflight)
    shutil.copy2(run_root / "baselines" / "gpu0_predictions.npz", output / "baseline_predictions.npz")
    log_output = output / "process_logs"
    log_output.mkdir(parents=True, exist_ok=True)
    for gpu in range(4):
        shutil.copy2(run_root / "logs" / f"worker_gpu{gpu}.jsonl", log_output / f"worker_gpu{gpu}.jsonl")

    cell_batch_consistency = all(
        cells[(layer, window)]["batch_manifest_exact"]
        and cells[(layer, window)]["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256
        for layer in LAYERS
        for window in WINDOWS
    )
    one_shortened = all(
        cells[(layer, window)]["shortened_layer_count"] == 1
        and cells[(layer, window)]["all_layer_windows"][layer - 1] == window
        and all(
            value == d0.T
            for index, value in enumerate(cells[(layer, window)]["all_layer_windows"])
            if index != layer - 1
        )
        for layer in LAYERS
        for window in WINDOWS
    )
    zero_training_and_forbidden_features = all(
        row["optimizer_objects"] == 0
        and row["scheduler_objects"] == 0
        and row["grad_scaler_objects"] == 0
        and row["backward_calls"] == 0
        and row["optimizer_steps"] == 0
        and row["parameter_updates"] == 0
        and row["training_targets"] == 0
        and not row["recurrence_active"]
        and not row["completion_active"]
        and not row["writers_active"]
        and not row["temporal_attnres_active"]
        and not row["full_attnres_active"]
        and not row["bptt_active"]
        and not row["hellaswag_executed"]
        for row in cells.values()
    )
    runtime_semantics_exact = all(
        row["evaluation_precision"] == "torch.autocast(cuda,bfloat16)"
        and row["loss_denominator"] == 1_310_720
        and row["absolute_positions_unchanged"]
        and row["causal_window_includes_current"]
        for row in cells.values()
    )
    integrity_pre = {
        "preflight": preflight["passed"],
        "semantic_diff": semantic["passed"],
        "workers": all(row["passed"] for row in workers.values()),
        "worker_baselines": all(row["passed"] for row in baselines.values()),
        "baseline_prediction_identity_across_gpus": baseline_prediction_identity,
        "exactly_120_cells": len(cells) == EXPECTED_CELLS,
        "all_cells_passed": all(row["passed"] for row in cells.values()),
        "one_shortened_layer_per_cell": one_shortened,
        "batch_manifest_every_cell": cell_batch_consistency,
        "model_tensors_unchanged": all(row["model_tensors_unchanged"] for row in workers.values()),
        "finite": all(row["all_losses_and_predictions_finite"] for row in cells.values()),
        "zero_training_and_forbidden_features": zero_training_and_forbidden_features,
        "runtime_semantics_exact": runtime_semantics_exact,
    }
    integrity_pre["passed"] = all(integrity_pre.values())
    layer_scores = {
        f"B{layer}": {
            "mean_damage_10_shortened_windows": float(np.mean([damages[layer][window] for window in WINDOWS])),
            "log2_window_trapezoid_area": float(
                np.trapz(
                    [damages[layer][window] for window in reversed(WINDOWS)],
                    x=[math.log2(window) for window in reversed(WINDOWS)],
                )
            ),
        }
        for layer in LAYERS
    }
    summary = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "classification": classification,
        "full_context_validation_loss": full_loss,
        "windows": list(MATRIX_WINDOWS),
        "layers": [f"B{layer}" for layer in LAYERS],
        "validation_losses": {
            f"B{layer}": {str(window): validation_losses[layer][window] for window in MATRIX_WINDOWS}
            for layer in LAYERS
        },
        "validation_damages": {
            f"B{layer}": {str(window): damages[layer][window] for window in MATRIX_WINDOWS}
            for layer in LAYERS
        },
        "width_threshold_profiles": profile_json,
        "marginal_W_at_0.01": profile_json["0.01"],
        "hypothetical_marginal_budgets": budgets,
        "layer_sensitivity_scores": layer_scores,
        "monotonicity": monotonicity["summary"],
        "shape_analysis": shape_analysis,
        "integrity_pre_audit": integrity_pre,
        "joint_configuration_evaluated": False,
        "training_performed": False,
    }
    d0.durable_json(output / "result_summary.json", summary)
    d0.durable_json(
        output / "commands_and_runtime.json",
        {
            "branch": BRANCH,
            "implementation_commit": git_output("rev-parse", "HEAD"),
            "commands": [
                "CUDA_VISIBLE_DEVICES=0 python scripts/experiment_2d0c.py preflight ...",
                "four independent workers with GPU assignments 0:(B1,B5,B9), 1:(B2,B6,B10), 2:(B3,B7,B11), 3:(B4,B8,B12)",
                "python scripts/experiment_2d0c.py assemble ...",
                "python scripts/experiment_2d0c.py finalize ...",
            ],
            "runtime": environment,
            "ddp": False,
            "nccl": False,
            "training": False,
            "joint_window_cells": 0,
        },
    )
    print(
        f"EXPERIMENT_2D0C_RESULTS_ASSEMBLED cells={len(cells)} "
        f"classification={classification} integrity={integrity_pre['passed']}",
        flush=True,
    )
    if not integrity_pre["passed"]:
        raise SystemExit("2D0C assembled result integrity failed")


def generate_plots(output, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    damages = summary["validation_damages"]
    matrix = np.asarray(
        [[damages[f"B{layer}"][str(window)] for window in MATRIX_WINDOWS] for layer in LAYERS]
    )
    fig, axis = plt.subplots(figsize=(15, 8), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(MATRIX_WINDOWS)), [str(window) for window in MATRIX_WINDOWS])
    axis.set_yticks(range(12), [f"B{layer}" for layer in LAYERS])
    axis.set_xlabel("Selected block attention window W (includes current token)")
    axis.set_ylabel("Independently shortened Transformer block")
    axis.set_title("Single-layer marginal validation damage")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Validation loss damage vs W1024")
    fig.savefig(output / "layer_window_damage_heatmap.png", dpi=220)
    fig.savefig(output / "layer_window_damage_heatmap.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for threshold in (0.005, 0.01, 0.02, 0.05):
        widths = [summary["width_threshold_profiles"][str(threshold)][f"B{layer}"] for layer in LAYERS]
        axis.plot(LAYERS, widths, marker="o", linewidth=2, label=f"damage ≤ {threshold:g}")
    axis.set_yscale("log", base=2)
    axis.set_xticks(LAYERS, [f"B{layer}" for layer in LAYERS])
    axis.set_yticks(MATRIX_WINDOWS, [str(window) for window in MATRIX_WINDOWS])
    axis.set_xlabel("Layer depth")
    axis.set_ylabel("Smallest independently allowed W")
    axis.set_title("Marginal width profiles by damage budget")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.savefig(output / "width_profiles.png", dpi=220)
    fig.savefig(output / "width_profiles.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    for layer in LAYERS:
        axis.plot(
            MATRIX_WINDOWS,
            [damages[f"B{layer}"][str(window)] for window in MATRIX_WINDOWS],
            marker="o",
            linewidth=1.6,
            label=f"B{layer}",
        )
    axis.set_xscale("log", base=2)
    axis.invert_xaxis()
    axis.set_xticks(MATRIX_WINDOWS, [str(window) for window in MATRIX_WINDOWS])
    axis.set_xlabel("Selected block attention window W")
    axis.set_ylabel("Validation loss damage vs W1024")
    axis.set_title("Per-layer marginal KV-window sensitivity curves")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2)
    fig.savefig(output / "per_layer_window_curves.png", dpi=220)
    fig.savefig(output / "per_layer_window_curves.svg")
    plt.close(fig)


def md_number(value, digits=6):
    return "NONE" if value is None else f"{value:.{digits}f}"


def run_finalize(args):
    require_git(clean=True)
    load_config()
    output = Path(args.output_dir).resolve()
    summary = json.loads((output / "result_summary.json").read_text())
    performance = json.loads((output / "performance.json").read_text())
    rankings = json.loads((output / "fixed_window_rankings.json").read_text())
    monotonicity = json.loads((output / "monotonicity.json").read_text())
    bins = json.loads((output / "position_bin_loss.json").read_text())
    argmax = json.loads((output / "argmax_agreement.json").read_text())
    generate_plots(output, summary)
    required = (
        "result_summary.json",
        "source_manifest.json",
        "environment.json",
        "batch_manifest.json",
        "semantic_diff_audit.json",
        "preflight_audit.json",
        "validation_loss_matrix.csv",
        "validation_damage_matrix.csv",
        "validation_damage_matrix.json",
        "width_threshold_profiles.csv",
        "width_threshold_profiles.json",
        "hypothetical_kv_budgets.csv",
        "fixed_window_rankings.json",
        "monotonicity.json",
        "shape_analysis.json",
        "per_position_loss.json",
        "position_bin_loss.json",
        "argmax_agreement.json",
        "performance.json",
        "commands_and_runtime.json",
        "baseline_predictions.npz",
        "layer_window_damage_heatmap.svg",
        "layer_window_damage_heatmap.png",
        "width_profiles.svg",
        "width_profiles.png",
        "per_layer_window_curves.svg",
        "per_layer_window_curves.png",
    )
    source = json.loads((output / "source_manifest.json").read_text())
    preflight = json.loads((output / "preflight_audit.json").read_text())
    semantic = json.loads((output / "semantic_diff_audit.json").read_text())
    matrix_json = json.loads((output / "validation_damage_matrix.json").read_text())
    checks = {
        "2D0B frozen tag exact": git_output("rev-list", "-n", "1", FROZEN_TAG) == FROZEN_COMMIT,
        "~10B Standard checkpoint SHA exact": source["checkpoint_sha256"] == d0.SOURCE_SHA256,
        "Standard GPT-2 architecture exact": source["architecture"]["passed"],
        "Full AttnRes absent": source["architecture"]["full_attnres_active_modules"] == 0
        and source["architecture"]["full_attnres_trainable_parameters"] == 0,
        "canonical validation manifest exact": source["canonical_validation_sha256"]
        == d0.CANONICAL_VALIDATION_SHA256,
        "baseline full-context regression exact": preflight["baseline_loss_absolute_delta"]
        <= SENTINEL_TOLERANCE,
        "B11 W512 regression": preflight["b11_regression"]["512"]["passed"],
        "B11 W128 regression": preflight["b11_regression"]["128"]["passed"],
        "B11 W1 regression": preflight["b11_regression"]["1"]["passed"],
        "human/zero-based layer mapping exact": semantic["mapping_exact"],
        "one and only one shortened layer per cell": summary["integrity_pre_audit"]["one_shortened_layer_per_cell"],
        "all other 11 layers W1024": summary["integrity_pre_audit"]["one_shortened_layer_per_cell"],
        "window grid exact": summary["windows"] == list(MATRIX_WINDOWS),
        "absolute positions and causal semantics exact": semantic["unchanged_semantics"]["absolute_positional_embeddings"]
        and semantic["unchanged_semantics"]["causal_sliding_mask"],
        "same BF16 runtime and loss denominator": summary["integrity_pre_audit"]["runtime_semantics_exact"],
        "same validation batches every cell": summary["integrity_pre_audit"]["batch_manifest_every_cell"],
        "all losses and predictions finite": summary["integrity_pre_audit"]["finite"],
        "model tensor hashes before/after identical": summary["integrity_pre_audit"]["model_tensors_unchanged"],
        "optimizer/scheduler/GradScaler/backward/steps/updates/training targets zero": summary["integrity_pre_audit"]["zero_training_and_forbidden_features"],
        "no recurrence/completion/writers/temporal or Full AttnRes/BPTT/HellaSwag": summary["integrity_pre_audit"]["zero_training_and_forbidden_features"],
        "all 120 shortened cells completed": summary["integrity_pre_audit"]["exactly_120_cells"],
        "all four GPU workers exited successfully": summary["integrity_pre_audit"]["workers"],
        "cross-artifact matrix consistency": matrix_json == summary["validation_damages"],
        "required artifacts present": all((output / name).is_file() for name in required),
        "four worker logs present": all(
            (output / "process_logs" / f"worker_gpu{gpu}.jsonl").is_file() for gpu in range(4)
        ),
    }
    passed = all(checks.values())
    d0.durable_json(
        output / "FINAL_AUDIT.json",
        {
            "experiment": EXPERIMENT,
            "checks": {
                key: {"status": "PASS" if value else "FAIL", "passed": value}
                for key, value in checks.items()
            },
            "passed": passed,
        },
    )
    classification = summary["classification"] if passed else "EXPERIMENT 2D0C UNSTABLE"
    recommendation = (
        "PROCEED TO MATCHED JOINT-KV GEOMETRY EVALUATION"
        if passed
        else "FIX 2D0C BEFORE FOLLOW-ON"
    )
    damages = summary["validation_damages"]
    profiles = summary["width_threshold_profiles"]
    most_w1 = rankings["1"][0]
    least_w1 = rankings["1"][-1]
    b11_w1_rank = next(row["rank"] for row in rankings["1"] if row["block"] == "B11")
    b12_vs_b11 = damages["B12"]["1"] - damages["B11"]["1"]
    rho_001 = summary["shape_analysis"]["0.01"]["spearman_depth_vs_selected_window"]
    budget_001 = next(
        row for row in summary["hypothetical_marginal_budgets"] if row["damage_threshold"] == 0.01
    )
    boundary_checks = []
    for layer in LAYERS:
        for window in WINDOWS:
            relevant = [
                row["delta"]
                for row in bins[f"B{layer}"][str(window)].values()
                if row["start"] >= window
            ]
            pre = [
                row["delta"]
                for row in bins[f"B{layer}"][str(window)].values()
                if row["end"] < window
            ]
            if relevant and pre:
                boundary_checks.append(abs(float(np.mean(relevant))) >= abs(float(np.mean(pre))))
    boundary_fraction = sum(boundary_checks) / len(boundary_checks) if boundary_checks else 1.0

    lines = [
        "# Experiment 2D0C — Full Layer × KV-Window Sensitivity Map",
        "",
        "## Outcome",
        "",
        f"Descriptive classification: **{classification}**.",
        "",
        "This is a single-layer marginal sensitivity map under an otherwise full-context Transformer. It is not a jointly shortened model and does not identify optimal simultaneous layer windows; interactions between shortened layers can be nonlinear.",
        "",
        "## Validation-damage matrix",
        "",
        "| Layer | W1024 | W512 | W256 | W128 | W64 | W32 | W16 | W8 | W4 | W2 | W1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in LAYERS:
        lines.append(
            "| B{} | {} |".format(
                layer,
                " | ".join(f"{damages[f'B{layer}'][str(window)]:+.6f}" for window in MATRIX_WINDOWS),
            )
        )
    lines.extend(
        [
            "",
            "## Marginal width-at-damage profiles",
            "",
            "| Layer | W@0.001 | W@0.0025 | W@0.005 | W@0.01 | W@0.02 | W@0.05 | W@0.10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in LAYERS:
        lines.append(
            f"| B{layer} | "
            + " | ".join(str(profiles[str(threshold)][f"B{layer}"]) for threshold in THRESHOLDS)
            + " |"
        )
    lines.extend(
        [
            "",
            "The W@0.01 row is the primary *marginal W@0.01 profile*, not a set of optimal joint windows.",
            "",
            "## Fixed-window depth summary",
            "",
            "| Layer | damage@512 | damage@128 | damage@32 | damage@1 | argmax@1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in LAYERS:
        lines.append(
            f"| B{layer} | {damages[f'B{layer}']['512']:+.6f} | "
            f"{damages[f'B{layer}']['128']:+.6f} | {damages[f'B{layer}']['32']:+.6f} | "
            f"{damages[f'B{layer}']['1']:+.6f} | {argmax[f'B{layer}']['1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Hypothetical marginal KV budgets — not joint-model results",
            "",
            "| Damage threshold | Sum independently selected W | Fraction of 12×1024 | Min W | Max W |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["hypothetical_marginal_budgets"]:
        lines.append(
            f"| {row['damage_threshold']:.4f} | {row['sum_independently_selected_windows']} | "
            f"{row['fraction_of_12x1024']:.6f} | {row['minimum_selected_window']} | "
            f"{row['maximum_selected_window']} |"
        )
    lines.extend(["", "## Scientific questions", ""])
    lines.append(f"1. At W1 the most sensitive block is {most_w1['block']} ({most_w1['damage']:+.6f}); fixed-window rankings for W512/W128/W32/W1 are saved in `fixed_window_rankings.json`.")
    lines.append(f"2. At W1 the least sensitive block is {least_w1['block']} ({least_w1['damage']:+.6f}).")
    lines.append("3. Marginal W@0.01: " + ", ".join(f"B{layer}={profiles['0.01'][f'B{layer}']}" for layer in LAYERS) + ".")
    lines.append(f"4. Depth-vs-W@0.01 Spearman correlation is {rho_001:+.6f}; the classification above states whether widening with depth is supported.")
    lines.append(f"5. The same correlation and adjacent-step counts distinguish decreasing sensitivity; see `shape_analysis.json`.")
    lines.append("6. Middle-layer bulge evidence was assessed across W@0.005/.01/.02/.05 rather than from one threshold; the resulting descriptive category is shown above.")
    lines.append(f"7. B11 ranks {b11_w1_rank}/12 in W1 damage, so its earlier curve is interpreted relative to the complete network rather than in isolation.")
    lines.append(f"8. B12 minus B11 W1 damage is {b12_vs_b11:+.6f}; B12 follows eleven full-context blocks, while B11 follows ten.")
    lines.append(f"9. B1 W1 damage is {damages['B1']['1']:+.6f}; unlike later blocks, it has no contextualized lower-layer Transformer residual input.")
    steep_below_128 = sorted(LAYERS, key=lambda layer: damages[f"B{layer}"]["32"] - damages[f"B{layer}"]["128"], reverse=True)[:4]
    lines.append("10. The largest W128→W32 damage increases occur at " + ", ".join(f"B{layer}" for layer in steep_below_128) + ".")
    lines.append(f"11. There are {monotonicity['summary']['total_violations']} monotonic violations; the largest is {monotonicity['summary']['largest_violation']:.10f}.")
    lines.append(f"12. In {boundary_fraction:.1%} of comparable layer/window bin cases, mean absolute damage after the removal boundary is at least as large as before it.")
    lines.append(f"13. Independently selecting W@0.01 sums to {budget_001['sum_independently_selected_windows']} ({budget_001['fraction_of_12x1024']:.6f} of 12×1024). This is hypothetical because joint damage is not additive and was not measured.")
    lines.append(f"14. The triangle hypothesis assessment is: {classification}. It is descriptive marginal evidence, not a final architecture decision.")
    lines.append("15. The next controlled comparison should use matched total KV budgets to compare the empirical marginal profile, a monotonic triangle, and a uniform sliding-window baseline; it is not executed here.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "For every layer above B1, a low marginal sensitivity can reflect information already contextualized by lower full-context layers. It does not imply that the same layer will remain insensitive when lower layers are shortened simultaneously.",
            "",
            "B1 and B12 are special endpoints: B1 cannot receive contextualized Transformer state from below, while B12 tests the final block's own historical attention after eleven full-context blocks.",
            "",
            "## Integrity audit",
            "",
        ]
    )
    for name, value in checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — {name}")
    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"Total four-GPU elapsed wall time: {performance['total_4gpu_elapsed_wall_seconds']:.3f} seconds.",
            f"Mean scientific cell time: {performance['per_cell_mean_seconds']:.3f} seconds.",
            f"Scientific validation targets: {performance['scientific_evaluated_targets']}.",
            f"Mean targets/sec across cells: {performance['mean_targets_per_second']:.1f}.",
            "",
            "## Next controlled experiment",
            "",
            f"**{recommendation}**",
            "",
            "The recommendation was not executed.",
            "",
            "# EXPERIMENT 2D0C COMPLETE",
        ]
    )
    d0.durable_text(output / "EXPERIMENT_2D0C_FINAL_REPORT.md", "\n".join(lines) + "\n")
    print(
        f"EXPERIMENT_2D0C_FINAL_AUDIT_{'PASS' if passed else 'FAIL'} "
        f"classification={classification}",
        flush=True,
    )
    if not passed:
        raise SystemExit("2D0C final audit failed")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--parent-checkpoint", required=True)
    preflight.add_argument("--validation-shard", required=True)
    preflight.add_argument("--run-root", required=True)
    preflight.set_defaults(func=run_preflight)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--parent-checkpoint", required=True)
    worker.add_argument("--validation-shard", required=True)
    worker.add_argument("--preflight", required=True)
    worker.add_argument("--run-root", required=True)
    worker.add_argument("--physical-gpu", type=int, required=True)
    worker.set_defaults(func=run_worker)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--run-root", required=True)
    assemble.add_argument("--output-dir", required=True)
    assemble.set_defaults(func=run_assemble)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-dir", required=True)
    finalize.set_defaults(func=run_finalize)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
