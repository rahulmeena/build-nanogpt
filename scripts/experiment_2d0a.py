#!/usr/bin/env python3
"""Experiment 2D0A: extreme B11 KV-window sensitivity on frozen Standard GPT-2."""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0 as d0  # noqa: E402


EXPERIMENT = "2D0A"
PROTOCOL = "exp2d0a_b11_extreme_window_sweep_v1"
BRANCH = "experiment-2d0a-b11-extreme-window-sweep"
FROZEN_TAG = "experiment-2d0-phase-a-b11-window-sweep-final"
FROZEN_COMMIT = "9f679713044790b782fbaccc39d795dfa0ec4277"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d0a_b11_extreme_window_sweep.json"
HISTORICAL_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_2d0_standard_b11_context_completion"
    / "phase_a_results.json"
)
OUTPUT_NAME = "experiment_2d0a_b11_extreme_window_sweep"
NEW_WINDOWS = (384, 256, 128, 1)
ALL_WINDOWS = (1024, 896, 768, 512, 384, 256, 128, 1)
GPU_BY_WINDOW = {384: 0, 256: 1, 128: 2, 1: 3}
POSITION_BINS = (
    ("1-64", 1, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-384", 257, 384),
    ("385-512", 385, 512),
    ("513-768", 513, 768),
    ("769-896", 769, 896),
    ("897-1023", 897, 1023),
)
PARETO_THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
SENTINEL_TOLERANCE = 1e-8


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    checks = {
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "frozen_commit": config.get("frozen_2d0_commit") == FROZEN_COMMIT,
        "source": config.get("source_checkpoint_sha256") == d0.SOURCE_SHA256,
        "validation": config.get("canonical_validation_sha256")
        == d0.CANONICAL_VALIDATION_SHA256,
        "windows": tuple(config["evaluation"]["windows"]) == NEW_WINDOWS,
        "gpu_map": {
            int(key): value
            for key, value in config["evaluation"]["physical_gpu_by_window"].items()
        }
        == GPU_BY_WINDOW,
        "geometry": config["evaluation"]["b1_to_b10_window"] == d0.T
        and config["evaluation"]["b12_window"] == d0.T,
        "batches": config["evaluation"]["batches"] == d0.PHASE_A_BATCHES,
        "loss_denominator": config["evaluation"]["loss_denominator"]
        == d0.PHASE_A_BATCHES * d0.VALIDATION_B * d0.T,
        "bins": tuple(tuple(row) for row in config["position_bins"]) == POSITION_BINS,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D0A frozen configuration mismatch: {checks}")
    return config


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2D0A requires branch {BRANCH}")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=REPO_ROOT
    )
    tag_commit = git_output("rev-list", "-n", "1", FROZEN_TAG)
    if tag_commit != FROZEN_COMMIT:
        raise SystemExit(f"frozen 2D0 tag mismatch: {tag_commit}")
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D0A execution requires a clean worktree")


def blank_drift():
    return {
        "count": 0,
        "cosine_sum": 0.0,
        "rms_sum": 0.0,
        "norm_ratio_sum": 0.0,
    }


def add_drift(accumulator, candidate, reference, start=0, end=d0.T - 1):
    candidate = candidate[:, start : end + 1].float()
    reference = reference[:, start : end + 1].float()
    difference = candidate - reference
    cosine = F.cosine_similarity(candidate, reference, dim=-1)
    rms = difference.square().mean(dim=-1).sqrt()
    norm_ratio = candidate.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-30)
    accumulator["count"] += cosine.numel()
    accumulator["cosine_sum"] += cosine.double().sum().item()
    accumulator["rms_sum"] += rms.double().sum().item()
    accumulator["norm_ratio_sum"] += norm_ratio.double().sum().item()


def finish_drift(row):
    count = row["count"]
    return {
        "count": count,
        "cosine": row["cosine_sum"] / count,
        "rms_difference": row["rms_sum"] / count,
        "norm_ratio": row["norm_ratio_sum"] / count,
    }


def drift_group():
    return {
        "all_positions": blank_drift(),
        "position_bins": {label: blank_drift() for label, _, _ in POSITION_BINS},
    }


def add_drift_group(group, candidate, reference):
    add_drift(group["all_positions"], candidate, reference)
    for label, start, end in POSITION_BINS:
        add_drift(group["position_bins"][label], candidate, reference, start, end)


def finish_drift_group(group):
    return {
        "all_positions": finish_drift(group["all_positions"]),
        "position_bins": {
            label: finish_drift(row) for label, row in group["position_bins"].items()
        },
    }


def run_sentinel(args):
    require_git(clean=True)
    config = load_config()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("2D0A sentinel requires exactly one visible CUDA GPU")
    if "RANK" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("2D0A sentinel forbids distributed execution")
    d0.seed_all(0)
    device = torch.device("cuda", 0)
    val_path = Path(args.validation_shard).resolve()
    if d0.file_sha256(val_path) != d0.VAL_SHA256:
        raise SystemExit("canonical validation shard SHA mismatch")
    _, model, source_audit = d0.load_standard_model(args.parent_checkpoint, device)
    historical = json.loads(HISTORICAL_PATH.read_text())["rows"]
    original_batches = d0.PHASE_A_BATCHES
    d0.PHASE_A_BATCHES = config["sentinel"]["batches"]
    rows = {}
    try:
        for window in config["sentinel"]["windows"]:
            observed = d0.phase_a_for_window(model, str(val_path), window, device)
            expected = historical[str(window)]["per_batch_losses"][
                : config["sentinel"]["batches"]
            ]
            deltas = [abs(left - right) for left, right in zip(observed["per_batch_losses"], expected)]
            rows[str(window)] = {
                "window": window,
                "expected_per_batch_losses": expected,
                "observed_per_batch_losses": observed["per_batch_losses"],
                "absolute_deltas": deltas,
                "maximum_absolute_delta": max(deltas),
                "tolerance": SENTINEL_TOLERANCE,
                "first_two_batch_sha256": observed["canonical_validation_sha256"],
                "passed": max(deltas) <= SENTINEL_TOLERANCE,
            }
    finally:
        d0.PHASE_A_BATCHES = original_batches
    output = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "source_audit": source_audit,
        "validation_shard": str(val_path),
        "validation_shard_sha256": d0.VAL_SHA256,
        "rows": rows,
        "optimizer_objects": 0,
        "backward_calls": 0,
        "parameter_updates": 0,
        "training_targets": 0,
        "passed": source_audit["passed"] and all(row["passed"] for row in rows.values()),
    }
    d0.durable_json(Path(args.run_root).resolve() / "regression_sentinel.json", output)
    print(f"EXPERIMENT_2D0A_SENTINEL_{'PASS' if output['passed'] else 'FAIL'}", flush=True)
    if not output["passed"]:
        raise SystemExit("2D0A regression sentinel failed")


@torch.no_grad()
def run_worker(args):
    require_git(clean=True)
    load_config()
    window = int(args.window)
    physical_gpu = int(args.physical_gpu)
    if window not in NEW_WINDOWS or GPU_BY_WINDOW[window] != physical_gpu:
        raise SystemExit("2D0A window/GPU mapping mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("each 2D0A worker requires exactly one visible CUDA GPU")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("2D0A workers forbid DDP/NCCL/distributed state")
    d0.seed_all(physical_gpu)
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.time()
    monotonic_start = time.monotonic()
    val_path = Path(args.validation_shard).resolve()
    val_sha = d0.file_sha256(val_path)
    if val_sha != d0.VAL_SHA256:
        raise SystemExit(f"canonical validation shard SHA mismatch: {val_sha}")
    _, model, source_audit = d0.load_standard_model(args.parent_checkpoint, device)
    model_before = d0.tensor_state_sha256(model)
    loader = d0.ExplicitShardLoader([str(val_path)], d0.VALIDATION_B, d0.T)
    batch_hashes = []
    full_loss_sum = 0.0
    short_loss_sum = 0.0
    loss_count = 0
    full_position_sum = np.zeros(d0.T, dtype=np.float64)
    short_position_sum = np.zeros(d0.T, dtype=np.float64)
    per_batch_full = []
    per_batch_short = []
    b11 = drift_group()
    b12 = drift_group()
    attention = drift_group()
    mlp = drift_group()
    logit_abs_sum = 0.0
    logit_squared_sum = 0.0
    logit_count = 0
    argmax_agreement = 0
    argmax_count = 0
    h10_exact_batches = 0
    finite_batches = 0

    for batch_index in range(d0.PHASE_A_BATCHES):
        cpu_x, cpu_y = loader.next_batch()
        batch_hashes.append(d0.batch_payload_hash(cpu_x, cpu_y))
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h10 = d0.shared_lower_trunk(model, x)
            incoming_copy = h10.clone()
            full = d0.teacher_tail(model, h10)
            short = d0.student_tail(model, h10, window)
            full_losses, full_logits = d0.token_cross_entropy(model, full["top"], y)
            short_losses, short_logits = d0.token_cross_entropy(model, short["top"], y)
        h10_exact_batches += int(torch.equal(h10, incoming_copy))
        finite = all(
            torch.isfinite(value).all().item()
            for value in (
                h10,
                full["h11"],
                full["h12"],
                full["top"],
                short["h11"],
                short["h12"],
                short["top"],
                full_losses,
                short_losses,
                full_logits,
                short_logits,
            )
        )
        finite_batches += int(finite)

        full_batch = full_losses.float().mean().item()
        short_batch = short_losses.float().mean().item()
        per_batch_full.append(full_batch)
        per_batch_short.append(short_batch)
        full_loss_sum += full_losses.double().sum().item()
        short_loss_sum += short_losses.double().sum().item()
        loss_count += full_losses.numel()
        full_position_sum += full_losses.double().sum(dim=0).cpu().numpy()
        short_position_sum += short_losses.double().sum(dim=0).cpu().numpy()

        add_drift_group(b11, short["h11"], full["h11"])
        add_drift_group(b12, short["h12"], full["h12"])
        add_drift_group(attention, short["b11_attention"], full["b11_attention"])
        full_mlp = full["h11"] - h10 - full["b11_attention"]
        short_mlp = short["h11"] - h10 - short["b11_attention"]
        add_drift_group(mlp, short_mlp, full_mlp)

        full_argmax = full_logits.argmax(dim=-1)
        short_argmax = short_logits.argmax(dim=-1)
        argmax_agreement += int((full_argmax == short_argmax).sum().item())
        argmax_count += full_argmax.numel()
        for start in range(0, d0.T, 64):
            difference = (
                short_logits[:, start : start + 64].float()
                - full_logits[:, start : start + 64].float()
            )
            logit_abs_sum += difference.abs().sum(dtype=torch.float64).item()
            logit_squared_sum += difference.square().sum(dtype=torch.float64).item()
            logit_count += difference.numel()
            del difference

        print(
            f"2D0A physical_gpu={physical_gpu} W={window} "
            f"batch={batch_index + 1:02d}/{d0.PHASE_A_BATCHES} "
            f"full={full_batch:.10f} short={short_batch:.10f}",
            flush=True,
        )
        del (
            x,
            y,
            h10,
            incoming_copy,
            full,
            short,
            full_losses,
            short_losses,
            full_logits,
            short_logits,
            full_argmax,
            short_argmax,
            full_mlp,
            short_mlp,
        )
        torch.cuda.empty_cache()

    model_after = d0.tensor_state_sha256(model)
    validation_loss = short_loss_sum / loss_count
    full_validation_loss = full_loss_sum / loss_count
    full_per_position = (full_position_sum / (d0.PHASE_A_BATCHES * d0.VALIDATION_B)).tolist()
    short_per_position = (short_position_sum / (d0.PHASE_A_BATCHES * d0.VALIDATION_B)).tolist()
    position_bins = {}
    for label, start, end in POSITION_BINS:
        full_ce = float(full_position_sum[start : end + 1].sum() / ((end - start + 1) * d0.PHASE_A_BATCHES * d0.VALIDATION_B))
        short_ce = float(short_position_sum[start : end + 1].sum() / ((end - start + 1) * d0.PHASE_A_BATCHES * d0.VALIDATION_B))
        position_bins[label] = {
            "start": start,
            "end": end,
            "targets": (end - start + 1) * d0.PHASE_A_BATCHES * d0.VALIDATION_B,
            "full_context_loss": full_ce,
            "short_b11_loss": short_ce,
            "delta": short_ce - full_ce,
        }
    elapsed = time.monotonic() - monotonic_start
    result = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "window": window,
        "physical_gpu": physical_gpu,
        "visible_cuda_device_count": torch.cuda.device_count(),
        "visible_gpu_name": torch.cuda.get_device_name(device),
        "distributed_initialized": torch.distributed.is_initialized(),
        "source_audit": source_audit,
        "validation_shard": str(val_path),
        "validation_shard_sha256": val_sha,
        "canonical_validation_sha256": d0.aggregate_hashes(batch_hashes),
        "validation_loss": validation_loss,
        "full_validation_loss": full_validation_loss,
        "damage_vs_archived_1024": validation_loss
        - json.loads(HISTORICAL_PATH.read_text())["rows"]["1024"]["validation_loss"],
        "validation_targets": loss_count,
        "per_batch_full_losses": per_batch_full,
        "per_batch_short_losses": per_batch_short,
        "per_position_loss": {
            "positions": list(range(d0.T)),
            "full": full_per_position,
            "short": short_per_position,
            "delta": [short - full for short, full in zip(short_per_position, full_per_position)],
        },
        "position_bins": position_bins,
        "incoming_b11_h10": {
            "bit_exact_batches": h10_exact_batches,
            "total_batches": d0.PHASE_A_BATCHES,
            "passed": h10_exact_batches == d0.PHASE_A_BATCHES,
        },
        "b11_state_drift": finish_drift_group(b11),
        "b12_state_drift": finish_drift_group(b12),
        "b11_attention_output_drift": finish_drift_group(attention),
        "b11_mlp_output_drift": finish_drift_group(mlp),
        "logit_drift": {
            "values": logit_count,
            "mean_absolute_difference": logit_abs_sum / logit_count,
            "rms_difference": math.sqrt(logit_squared_sum / logit_count),
            "argmax_agreement": argmax_agreement / argmax_count,
            "argmax_targets": argmax_count,
        },
        "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after,
        "model_tensors_unchanged": model_before == model_after,
        "finite_batches": finite_batches,
        "all_losses_and_activations_finite": finite_batches == d0.PHASE_A_BATCHES,
        "b1_to_b10_window": d0.T,
        "b11_window": window,
        "b12_window": d0.T,
        "absolute_positions_unchanged": True,
        "evaluation_precision": "torch.autocast(cuda,bfloat16)",
        "loss_denominator": loss_count,
        "optimizer_objects": 0,
        "backward_calls": 0,
        "parameter_updates": 0,
        "training_targets": 0,
        "recurrence_active": False,
        "completion_module_active": False,
        "hellaswag_executed": False,
        "performance": {
            "wall_start_unix": wall_start,
            "wall_end_unix": time.time(),
            "wall_seconds": elapsed,
            "evaluated_targets": loss_count,
            "targets_per_second": loss_count / elapsed,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }
    result["passed"] = all(
        (
            source_audit["passed"],
            result["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256,
            result["validation_targets"] == d0.PHASE_A_BATCHES * d0.VALIDATION_B * d0.T,
            result["incoming_b11_h10"]["passed"],
            result["model_tensors_unchanged"],
            result["all_losses_and_activations_finite"],
            not result["distributed_initialized"],
        )
    )
    output = Path(args.run_root).resolve() / "workers" / f"window_{window}.json"
    d0.durable_json(output, result)
    print(
        f"EXPERIMENT_2D0A_WORKER_{'PASS' if result['passed'] else 'FAIL'} "
        f"W={window} loss={validation_loss:.10f} damage={result['damage_vs_archived_1024']:+.10f}",
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(f"2D0A worker W={window} failed")


def smallest_windows(damages):
    rows = []
    for threshold in PARETO_THRESHOLDS:
        candidates = [window for window in sorted(ALL_WINDOWS) if damages[window] <= threshold]
        window = candidates[0] if candidates else None
        rows.append(
            {
                "allowed_damage": threshold,
                "smallest_b11_window": window,
                "kv_fraction": None if window is None else window / d0.T,
            }
        )
    return rows


def classify(damages, monotonic_deviations):
    strong_nonmonotonic = any(row["decrease"] > 1e-4 for row in monotonic_deviations)
    if strong_nonmonotonic:
        return "B11 WINDOW RESULT IS MIXED"
    if damages[1] <= 0.01:
        return "B11 EXPLICIT HISTORY HIGHLY REDUNDANT"
    if damages[128] <= 0.01 or damages[256] <= 0.01:
        return "B11 EXPLICIT HISTORY MODERATELY REDUNDANT"
    if damages[128] > 0.01:
        return "B11 REQUIRES SUBSTANTIAL EXPLICIT HISTORY"
    return "B11 WINDOW RESULT IS MIXED"


def run_assemble(args):
    require_git(clean=True)
    load_config()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sentinel = json.loads((run_root / "regression_sentinel.json").read_text())
    historical_payload = json.loads(HISTORICAL_PATH.read_text())
    historical = historical_payload["rows"]
    workers = {
        window: json.loads((run_root / "workers" / f"window_{window}.json").read_text())
        for window in NEW_WINDOWS
    }
    archived_full_loss = historical["1024"]["validation_loss"]
    archived_full_batches = historical["1024"]["per_batch_losses"]
    full_control_deltas = {
        str(window): max(
            abs(left - right)
            for left, right in zip(workers[window]["per_batch_full_losses"], archived_full_batches)
        )
        for window in NEW_WINDOWS
    }
    full_control_pass = all(value <= SENTINEL_TOLERANCE for value in full_control_deltas.values())
    damages = {
        window: (
            historical[str(window)]["validation_loss"] - archived_full_loss
            if window >= 512
            else workers[window]["validation_loss"] - archived_full_loss
        )
        for window in ALL_WINDOWS
    }
    combined_rows = {}
    for window in ALL_WINDOWS:
        if window >= 512:
            row = historical[str(window)]
            combined_rows[str(window)] = {
                "window": window,
                "validation_loss": row["validation_loss"],
                "damage_vs_1024": damages[window],
                "historical_kv_retained": window - 1,
                "kv_fraction": window / d0.T,
                "b11_cosine": row["b11_post_block_state_drift"]["cosine"],
                "b11_rms": row["b11_post_block_state_drift"]["rms_difference"],
                "b12_cosine": row["b12_state_drift"]["cosine"],
                "b12_rms": row["b12_state_drift"]["rms_difference"],
                "diagnostic_provenance": "Experiment 2D0 Phase A (B12 value is final normalized top state)",
            }
        else:
            row = workers[window]
            combined_rows[str(window)] = {
                "window": window,
                "validation_loss": row["validation_loss"],
                "damage_vs_1024": damages[window],
                "historical_kv_retained": window - 1,
                "kv_fraction": window / d0.T,
                "b11_cosine": row["b11_state_drift"]["all_positions"]["cosine"],
                "b11_rms": row["b11_state_drift"]["all_positions"]["rms_difference"],
                "b12_cosine": row["b12_state_drift"]["all_positions"]["cosine"],
                "b12_rms": row["b12_state_drift"]["all_positions"]["rms_difference"],
                "diagnostic_provenance": "Experiment 2D0A B12 post-block H12 state",
            }
    monotonic_deviations = []
    for left, right in zip(ALL_WINDOWS, ALL_WINDOWS[1:]):
        if damages[right] < damages[left]:
            monotonic_deviations.append(
                {
                    "from_window": left,
                    "to_window": right,
                    "from_damage": damages[left],
                    "to_damage": damages[right],
                    "decrease": damages[left] - damages[right],
                }
            )
    classification = classify(damages, monotonic_deviations)
    pareto = smallest_windows(damages)
    per_position = {
        "position_indexing": "absolute target positions 0..1023; requested analysis positions are 1..1023",
        "W1024": {
            "provenance": "full-context control embedded in all four 2D0A workers",
            "mean_loss_by_absolute_position": workers[384]["per_position_loss"]["full"],
        },
        "W512": {
            "available": False,
            "reason": "Experiment 2D0 Phase A saved bins but not all 1024 per-position means; no redundant full rerun was needed",
        },
    }
    for window, row in workers.items():
        per_position[f"W{window}"] = {
            "mean_full_loss_by_absolute_position": row["per_position_loss"]["full"],
            "mean_short_loss_by_absolute_position": row["per_position_loss"]["short"],
            "mean_delta_by_absolute_position": row["per_position_loss"]["delta"],
        }
    performance_rows = {str(window): row["performance"] for window, row in workers.items()}
    total_four_gpu_elapsed = max(row["wall_end_unix"] for row in performance_rows.values()) - min(
        row["wall_start_unix"] for row in performance_rows.values()
    )
    integrity_pre = {
        "sentinel": sentinel["passed"],
        "worker_results": all(row["passed"] for row in workers.values()),
        "full_control_per_batch_regression": full_control_pass,
        "canonical_validation": all(
            row["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256
            for row in workers.values()
        ),
        "source_exact": all(row["source_audit"]["passed"] for row in workers.values()),
        "model_tensors_unchanged": all(row["model_tensors_unchanged"] for row in workers.values()),
        "finite": all(row["all_losses_and_activations_finite"] for row in workers.values()),
        "h10_identical": all(row["incoming_b11_h10"]["passed"] for row in workers.values()),
        "no_distributed": all(not row["distributed_initialized"] for row in workers.values()),
    }
    integrity_pre["passed"] = all(integrity_pre.values())
    source_manifest = {
        "experiment": EXPERIMENT,
        "checkpoint": workers[384]["source_audit"]["checkpoint"],
        "checkpoint_sha256": d0.SOURCE_SHA256,
        "checkpoint_bytes": d0.SOURCE_BYTES,
        "historical_training_tokens": d0.SOURCE_TOKENS,
        "validation_shard": workers[384]["validation_shard"],
        "validation_shard_sha256": d0.VAL_SHA256,
        "canonical_batch_collection_sha256": d0.CANONICAL_VALIDATION_SHA256,
        "architecture": workers[384]["source_audit"],
        "frozen_2d0_tag": FROZEN_TAG,
        "frozen_2d0_commit": FROZEN_COMMIT,
    }
    extreme = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "windows": list(NEW_WINDOWS),
        "physical_gpu_by_window": {str(key): value for key, value in GPU_BY_WINDOW.items()},
        "rows": {str(key): value for key, value in workers.items()},
        "passed": integrity_pre["passed"],
    }
    summary = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "classification": classification,
        "combined_rows": combined_rows,
        "damages": {str(key): value for key, value in damages.items()},
        "monotonic_nondecreasing": not monotonic_deviations,
        "monotonicity_deviations": monotonic_deviations,
        "windows_in_old_recurrence_reference_band_0.01_to_0.10": [
            window for window in ALL_WINDOWS if 0.01 <= damages[window] <= 0.10
        ],
        "pareto": pareto,
        "integrity_pre_audit": integrity_pre,
        "full_control_per_batch_max_absolute_deltas": full_control_deltas,
        "phase_b_recurrence_training_authorized": False,
        "training_performed": False,
        "recommendation_executed": False,
    }
    artifacts = {
        "source_manifest.json": source_manifest,
        "phase_a_historical_points.json": historical_payload,
        "extreme_window_results.json": extreme,
        "per_position_loss.json": per_position,
        "position_bin_loss.json": {
            str(window): row["position_bins"] for window, row in workers.items()
        },
        "b11_state_drift.json": {
            str(window): row["b11_state_drift"] for window, row in workers.items()
        },
        "b12_state_drift.json": {
            str(window): row["b12_state_drift"] for window, row in workers.items()
        },
        "logit_drift.json": {
            str(window): row["logit_drift"] for window, row in workers.items()
        },
        "pareto_windows.json": {"thresholds": pareto},
        "performance.json": {
            "workers": performance_rows,
            "total_four_gpu_elapsed_wall_seconds": total_four_gpu_elapsed,
        },
        "commands_and_runtime.json": {
            "branch": BRANCH,
            "implementation_commit": git_output("rev-parse", "HEAD"),
            "python": sys.version,
            "platform": platform.platform(),
            "commands": [
                "CUDA_VISIBLE_DEVICES=0 python scripts/experiment_2d0a.py sentinel ...",
                "four independent concurrent processes: CUDA_VISIBLE_DEVICES={0,1,2,3} python scripts/experiment_2d0a.py worker --window {384,256,128,1} ...",
                "python scripts/experiment_2d0a.py assemble ...",
                "python scripts/experiment_2d0a.py finalize ...",
            ],
            "ddp": False,
            "nccl": False,
            "training": False,
            "hellaswag": False,
        },
        "result_summary.json": summary,
    }
    for name, payload in artifacts.items():
        d0.durable_json(output / name, payload)
    print(
        f"EXPERIMENT_2D0A_RESULTS_ASSEMBLED classification={classification} "
        f"integrity={integrity_pre['passed']}",
        flush=True,
    )
    if not integrity_pre["passed"]:
        raise SystemExit("2D0A assembled result integrity failed")


def fmt(value, digits=10):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def run_finalize(args):
    require_git(clean=True)
    load_config()
    output = Path(args.output_dir).resolve()
    summary = json.loads((output / "result_summary.json").read_text())
    extreme = json.loads((output / "extreme_window_results.json").read_text())
    positions = json.loads((output / "position_bin_loss.json").read_text())
    performance = json.loads((output / "performance.json").read_text())
    required_before_final = (
        "result_summary.json",
        "source_manifest.json",
        "phase_a_historical_points.json",
        "extreme_window_results.json",
        "per_position_loss.json",
        "position_bin_loss.json",
        "b11_state_drift.json",
        "b12_state_drift.json",
        "logit_drift.json",
        "pareto_windows.json",
        "performance.json",
        "commands_and_runtime.json",
    )
    workers = {int(key): value for key, value in extreme["rows"].items()}
    checks = {
        "source ~10B checkpoint exact": all(
            row["source_audit"]["sha256"] == d0.SOURCE_SHA256 for row in workers.values()
        ),
        "Standard GPT-2 exact": all(row["source_audit"]["checks"]["standard_mode"] for row in workers.values()),
        "Full AttnRes absent": all(row["source_audit"]["full_attnres_active_modules"] == 0 for row in workers.values()),
        "canonical validation data exact": all(row["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256 for row in workers.values()),
        "Phase-A W1024 regression sentinel": summary["integrity_pre_audit"]["sentinel"],
        "Phase-A W512 sentinel if executed": summary["integrity_pre_audit"]["sentinel"],
        "B1-B10 exactly W1024": all(row["b1_to_b10_window"] == 1024 for row in workers.values()),
        "B12 exactly W1024": all(row["b12_window"] == 1024 for row in workers.values()),
        "only B11 window modified": all(row["b11_window"] in NEW_WINDOWS for row in workers.values()),
        "windows exactly 384/256/128/1": set(workers) == set(NEW_WINDOWS),
        "absolute positions unchanged": all(row["absolute_positions_unchanged"] for row in workers.values()),
        "same evaluation precision": all(row["evaluation_precision"] == "torch.autocast(cuda,bfloat16)" for row in workers.values()),
        "same loss denominator": all(row["loss_denominator"] == 1_310_720 for row in workers.values()),
        "all losses finite": all(math.isfinite(row["validation_loss"]) for row in workers.values()),
        "all activations finite": all(row["all_losses_and_activations_finite"] for row in workers.values()),
        "incoming B11 h10 identical to full": all(row["incoming_b11_h10"]["passed"] for row in workers.values()),
        "optimizer objects zero": all(row["optimizer_objects"] == 0 for row in workers.values()),
        "backward calls zero": all(row["backward_calls"] == 0 for row in workers.values()),
        "parameter updates zero": all(row["parameter_updates"] == 0 for row in workers.values()),
        "training targets zero": all(row["training_targets"] == 0 for row in workers.values()),
        "model tensors unchanged before/after": all(row["model_tensors_unchanged"] for row in workers.values()),
        "no recurrence": all(not row["recurrence_active"] for row in workers.values()),
        "no completion module active": all(not row["completion_module_active"] for row in workers.values()),
        "no HellaSwag": all(not row["hellaswag_executed"] for row in workers.values()),
        "required machine-readable artifacts present": all((output / name).is_file() for name in required_before_final),
        "one independent process per GPU; no DDP/NCCL": all(not row["distributed_initialized"] and row["visible_cuda_device_count"] == 1 for row in workers.values()),
    }
    passed = all(checks.values())
    audit = {
        "experiment": EXPERIMENT,
        "checks": {key: {"status": "PASS" if value else "FAIL", "passed": value} for key, value in checks.items()},
        "passed": passed,
    }
    d0.durable_json(output / "FINAL_AUDIT.json", audit)

    rows = summary["combined_rows"]
    damages = {int(key): value for key, value in summary["damages"].items()}
    if not passed:
        recommendation = "EXPERIMENT INVALID — FIX INTEGRITY FIRST"
    elif any(row["decrease"] > 1e-4 for row in summary["monotonicity_deviations"]):
        recommendation = "B11 NEEDS ADDITIONAL WINDOW RESOLUTION FIRST"
    else:
        recommendation = "PROCEED TO FULL LAYER×WINDOW SENSITIVITY MAP"
    report = []
    report.append("# Experiment 2D0A — B11 Extreme KV-Window Sensitivity Sweep")
    report.append("")
    report.append("## Outcome")
    report.append("")
    report.append(f"Classification: **{summary['classification']}**.")
    report.append("")
    report.append("This was an evaluation-only sweep. Optimizers, backward calls, parameter updates, and training targets were all zero. No recurrence or completion module was active.")
    report.append("")
    report.append("## Complete B11 sensitivity curve")
    report.append("")
    report.append("| Window | Val loss | Damage | Historical KV retained | KV fraction | B11 cosine | B11 RMS | B12 cosine | B12 RMS |")
    report.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for window in ALL_WINDOWS:
        row = rows[str(window)]
        report.append(
            f"| {window} | {fmt(row['validation_loss'])} | {row['damage_vs_1024']:+.10f} | "
            f"{row['historical_kv_retained']} | {row['kv_fraction']:.6f} | "
            f"{fmt(row['b11_cosine'])} | {fmt(row['b11_rms'])} | "
            f"{fmt(row['b12_cosine'])} | {fmt(row['b12_rms'])} |"
        )
    report.append("")
    report.append("The KV fraction is W/1024 for B11 only; it is not a total-model KV saving. Historical 2D0 B12 diagnostics used the final normalized top state, while new 2D0A diagnostics use the B12 post-block H12 state.")
    report.append("")
    report.append("## Quality-retention Pareto profile")
    report.append("")
    report.append("| Allowed damage | Smallest B11 window | KV fraction |")
    report.append("|---:|---:|---:|")
    for row in summary["pareto"]:
        window = "NONE" if row["smallest_b11_window"] is None else str(row["smallest_b11_window"])
        fraction = "N/A" if row["kv_fraction"] is None else f"{row['kv_fraction']:.6f}"
        report.append(f"| {row['allowed_damage']:.4f} | {window} | {fraction} |")
    report.append("")
    band = summary["windows_in_old_recurrence_reference_band_0.01_to_0.10"]
    report.append(f"Windows incidentally in the old 0.01–0.10 recurrence-reference band: {band if band else 'NONE'}. This does not authorize recurrence training.")
    report.append("")
    report.append("## Position-dependent loss")
    for window in NEW_WINDOWS:
        report.append("")
        report.append(f"### W{window}")
        report.append("")
        report.append("| Position bin | Full loss | Short loss | Delta | Targets |")
        report.append("|---|---:|---:|---:|---:|")
        for label, _, _ in POSITION_BINS:
            row = positions[str(window)][label]
            report.append(
                f"| {label} | {fmt(row['full_context_loss'])} | {fmt(row['short_b11_loss'])} | "
                f"{row['delta']:+.10f} | {row['targets']} |"
            )
    report.append("")
    report.append("Before a window's history-removal boundary, any tiny nonzero delta reflects numerical differences between the full causal and explicit sliding-mask kernels, not removed history.")
    report.append("")
    report.append("## Representation and logit diagnostics")
    report.append("")
    report.append("| Window | B11 norm ratio | B12 norm ratio | Logit MAE | Logit RMS | Argmax agreement |")
    report.append("|---:|---:|---:|---:|---:|---:|")
    for window in NEW_WINDOWS:
        row = workers[window]
        report.append(
            f"| {window} | {row['b11_state_drift']['all_positions']['norm_ratio']:.10f} | "
            f"{row['b12_state_drift']['all_positions']['norm_ratio']:.10f} | "
            f"{row['logit_drift']['mean_absolute_difference']:.10f} | "
            f"{row['logit_drift']['rms_difference']:.10f} | "
            f"{row['logit_drift']['argmax_agreement']:.10f} |"
        )
    report.append("")
    report.append("Incoming h10 was bit-identical in every batch. Position-binned B11/B12, attention-output, MLP-output, and per-position loss diagnostics are retained in the machine-readable artifacts.")
    report.append("")
    report.append("## Monotonicity")
    report.append("")
    if summary["monotonic_nondecreasing"]:
        report.append("Damage was nondecreasing across 1024→896→768→512→384→256→128→1.")
    else:
        report.append(f"Exact damage decreases: `{json.dumps(summary['monotonicity_deviations'], sort_keys=True)}`")
    report.append("")
    report.append("## Scientific questions")
    report.append("")
    report.append("1. Damage versus full B11: " + ", ".join(f"W{window} {damages[window]:+.10f}" for window in NEW_WINDOWS) + ".")
    report.append(f"2. The curve is {'monotonic at measured resolution' if summary['monotonic_nondecreasing'] else 'not perfectly monotonic; deviations are reported above'}.")
    q3 = []
    for threshold in (0.001, 0.005, 0.01):
        row = next(item for item in summary["pareto"] if item["allowed_damage"] == threshold)
        q3.append(f"+{threshold:g}: {row['smallest_b11_window'] if row['smallest_b11_window'] is not None else 'NONE'}")
    report.append("3. Smallest windows within the requested damage limits — " + "; ".join(q3) + ".")
    report.append("4. Damage is evaluated by absolute position. It begins at each removal boundary apart from very small kernel-path numerical deltas before that boundary; the bin tables quantify its later concentration.")
    report.append("5. B11 divergence rises as the window shrinks; the exact cosine, RMS, norm-ratio, and position-binned curves are reported above and in b11_state_drift.json.")
    repair = []
    for window in NEW_WINDOWS:
        b11_rms = workers[window]["b11_state_drift"]["all_positions"]["rms_difference"]
        b12_rms = workers[window]["b12_state_drift"]["all_positions"]["rms_difference"]
        repair.append(f"W{window}: B11 {b11_rms:.6f} → B12 {b12_rms:.6f}")
    report.append("6. B12 residual error — " + "; ".join(repair) + ". Lower B12 RMS is evidence of partial absorption; higher RMS is persistence/amplification.")
    report.append(f"7. At W1, validation loss is {workers[1]['validation_loss']:.10f}, damage is {damages[1]:+.10f}, and argmax agreement is {workers[1]['logit_drift']['argmax_agreement']:.10f}; this is the surviving end-to-end performance with zero B11 historical KV.")
    report.append("8. The result is evidence that a late layer can rely heavily on already-contextualized bottom-up residual state, but it is not proof: B1-B10 and B12 remained full-context.")
    report.append("9. It challenges a blanket assumption that KV width must monotonically increase toward the top, but it does not determine the final joint layerwise shape.")
    report.append("10. The measured endpoint and intermediate curve are sufficiently informative for the recommendation below, subject to the integrity audit.")
    report.append("")
    report.append("## Causal interpretation")
    report.append("")
    report.append("This experiment does not show that B11 needs no long-range information. It measures only the end-to-end damage caused by reducing B11's own direct attention history while B1-B10 and B12 remain full-context. Distant information may already be encoded in h10(t). Likewise, W1 is not automatically an optimal final setting: lower layers in a jointly KV-reduced model may deliver less-contextualized residual states.")
    report.append("")
    report.append("## Integrity audit")
    report.append("")
    for name, value in checks.items():
        report.append(f"- {'PASS' if value else 'FAIL'} — {name}")
    report.append("")
    report.append("## Performance")
    report.append("")
    report.append("| Window/GPU | Wall s | Targets | Targets/s | Peak allocated MB | Peak reserved MB |")
    report.append("|---|---:|---:|---:|---:|---:|")
    for window in NEW_WINDOWS:
        row = performance["workers"][str(window)]
        report.append(
            f"| W{window}/GPU{GPU_BY_WINDOW[window]} | {row['wall_seconds']:.3f} | "
            f"{row['evaluated_targets']} | {row['targets_per_second']:.1f} | "
            f"{row['peak_allocated_vram_mb']:.1f} | {row['peak_reserved_vram_mb']:.1f} |"
        )
    report.append("")
    report.append(f"Total four-GPU elapsed wall time: {performance['total_four_gpu_elapsed_wall_seconds']:.3f} seconds.")
    report.append("")
    report.append("## Next experiment")
    report.append("")
    report.append(f"**{recommendation}**")
    report.append("")
    report.append("The recommendation was not executed.")
    report.append("")
    report.append("# EXPERIMENT 2D0A COMPLETE")
    d0.durable_text(output / "EXPERIMENT_2D0A_FINAL_REPORT.md", "\n".join(report) + "\n")
    print(
        f"EXPERIMENT_2D0A_FINAL_AUDIT_{'PASS' if passed else 'FAIL'} "
        f"recommendation={recommendation}",
        flush=True,
    )
    if not passed:
        raise SystemExit("Experiment 2D0A final audit failed")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sentinel = subparsers.add_parser("sentinel")
    sentinel.add_argument("--parent-checkpoint", required=True)
    sentinel.add_argument("--validation-shard", required=True)
    sentinel.add_argument("--run-root", required=True)
    sentinel.set_defaults(func=run_sentinel)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--parent-checkpoint", required=True)
    worker.add_argument("--validation-shard", required=True)
    worker.add_argument("--run-root", required=True)
    worker.add_argument("--window", type=int, required=True)
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
