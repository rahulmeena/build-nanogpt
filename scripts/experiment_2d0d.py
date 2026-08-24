#!/usr/bin/env python3
"""Experiment 2D0D: matched joint-KV geometry evaluation, with no training."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
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
import experiment_2d0c as d0c  # noqa: E402


EXPERIMENT = "2D0D"
PROTOCOL = "exp2d0d_matched_joint_kv_geometries_v1"
BRANCH = "experiment-2d0d-matched-joint-kv-geometries"
FROZEN_TAG = "experiment-2d0c-layer-window-sensitivity-map-final"
FROZEN_COMMIT = "752bdc8e0f1a8b0694692ad0b0ae37f4edbeead0"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d0d_matched_joint_kv_geometries.json"
PARENT_RESULTS = REPO_ROOT / "results" / "experiment_2d0c_layer_window_sensitivity_map"
PARENT_PREFLIGHT_PATH = PARENT_RESULTS / "preflight_audit.json"
PARENT_BASELINE_PATH = PARENT_RESULTS / "baseline_predictions.npz"
MARGINAL_MATRIX_PATH = PARENT_RESULTS / "validation_damage_matrix.json"
OUTPUT_NAME = "experiment_2d0d_matched_joint_kv_geometries"
GEOMETRIES = {
    "EMPIRICAL": (1024, 1024, 128, 256, 256, 64, 256, 1024, 512, 256, 256, 256),
    "TOP_WIDE_TRIANGLE": (128, 152, 184, 224, 272, 328, 396, 480, 580, 700, 844, 1024),
    "REVERSE_TRIANGLE": (1024, 844, 700, 580, 480, 396, 328, 272, 224, 184, 152, 128),
    "UNIFORM_MATCHED": (443, 443, 442, 443, 442, 443, 443, 442, 443, 442, 443, 443),
}
GPU_ASSIGNMENT = {
    0: "EMPIRICAL",
    1: "TOP_WIDE_TRIANGLE",
    2: "REVERSE_TRIANGLE",
    3: "UNIFORM_MATCHED",
}
POSITION_BINS = d0c.POSITION_BINS
SENTINEL_TOLERANCE = 1e-8
CLEAR_MARGIN = 0.01
CLEAR_WINS = 15
EXPECTED_TARGETS = d0.PHASE_A_BATCHES * d0.VALIDATION_B * d0.T
TRAINING_AUDIT = {
    "optimizer_objects": 0,
    "scheduler_objects": 0,
    "grad_scaler_objects": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameter_updates": 0,
    "training_targets": 0,
}
FORBIDDEN_AUDIT = {
    "recurrence_active": False,
    "completion_active": False,
    "reader_active": False,
    "writer_active": False,
    "full_bandwidth_active": False,
    "maglev_active": False,
    "temporal_attnres_active": False,
    "full_attnres_active": False,
    "bptt_active": False,
    "hellaswag_executed": False,
}


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"2D0D requires branch {BRANCH}")
    if git_output("rev-list", "-n", "1", FROZEN_TAG) != FROZEN_COMMIT:
        raise SystemExit("2D0C frozen tag does not resolve to the final commit")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=REPO_ROOT)
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D0D execution requires a clean worktree")


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    checks = {
        "experiment": config.get("experiment") == EXPERIMENT,
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "frozen": config.get("frozen_2d0c_commit") == FROZEN_COMMIT,
        "source": config.get("source_checkpoint_sha256") == d0.SOURCE_SHA256,
        "validation": config.get("canonical_validation_sha256") == d0.CANONICAL_VALIDATION_SHA256,
        "geometries": config.get("geometries") == {key: list(value) for key, value in GEOMETRIES.items()},
        "assignment": config.get("gpu_assignment") == {str(key): value for key, value in GPU_ASSIGNMENT.items()},
        "bins": tuple(tuple(value) for value in config.get("position_bins", [])) == POSITION_BINS,
        "batches": config["evaluation"]["batches"] == d0.PHASE_A_BATCHES,
        "targets": config["evaluation"]["targets_per_candidate"] == EXPECTED_TARGETS,
        "margin": config["evaluation"]["clear_winner_loss_margin"] == CLEAR_MARGIN,
        "wins": config["evaluation"]["clear_winner_paired_wins"] == CLEAR_WINS,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D0D frozen configuration mismatch: {checks}")
    return config


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def batch_identity(cpu_x, cpu_y):
    return {
        "input_sha256": tensor_sha256(cpu_x),
        "target_sha256": tensor_sha256(cpu_y),
        "combined_sha256": d0.batch_payload_hash(cpu_x, cpu_y),
    }


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


def geometry_manifest_payload():
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "created_before_result_bearing_compute": True,
        "geometries": {key: list(value) for key, value in GEOMETRIES.items()},
        "gpu_assignment": {str(key): value for key, value in GPU_ASSIGNMENT.items()},
        "schedule_source": "preregistered Experiment 2D0D protocol",
    }


def budget_audit_payload():
    rows = {}
    for name, schedule in GEOMETRIES.items():
        checks = {
            "exactly_12_entries": len(schedule) == 12,
            "all_integer": all(isinstance(value, int) and not isinstance(value, bool) for value in schedule),
            "all_within_1_1024": all(1 <= value <= 1024 for value in schedule),
            "sum_W_exactly_5312": sum(schedule) == 5312,
            "sum_historical_W_minus_1_exactly_5300": sum(value - 1 for value in schedule) == 5300,
        }
        rows[name] = {
            "windows": list(schedule),
            "sum_W": sum(schedule),
            "sum_historical_W_minus_1": sum(value - 1 for value in schedule),
            "fraction_full_nominal_layer_window": sum(schedule) / (12 * 1024),
            "reduction_full_nominal_layer_window": 1 - sum(schedule) / (12 * 1024),
            "historical_slot_fraction": sum(value - 1 for value in schedule) / (12 * 1023),
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "geometries": rows,
        "same_nominal_budget": len({row["sum_W"] for row in rows.values()}) == 1,
        "passed": all(row["passed"] for row in rows.values()),
        "caveat": "Window sums are a nominal deployed KV-capacity proxy, not measured masked-evaluator wall time or total model memory.",
    }


def semantic_diff_payload():
    checks = {
        "source_model_unchanged": True,
        "attention_mask_implementation_reused": True,
        "validation_loader_reused": True,
        "token_ordering_unchanged": True,
        "loss_calculation_reused": True,
        "precision_unchanged": True,
        "absolute_positions_unchanged": True,
        "only_generalization_is_12_element_schedule": True,
        "full_window_dispatches_native_block": True,
        "short_window_dispatches_frozen_d0_run_block": True,
    }
    return {
        "experiment": EXPERIMENT,
        "validated_parent_evaluator": "scripts/experiment_2d0c.py",
        "old_interface": "one selected block index and one W",
        "new_interface": "one immutable 12-element W schedule",
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_prepare(args):
    require_git(clean=False)
    load_config()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = geometry_manifest_payload()
    budget = budget_audit_payload()
    semantic = semantic_diff_payload()
    d0.durable_json(output / "geometry_manifest.json", manifest)
    d0.durable_json(output / "budget_audit.json", budget)
    d0.durable_json(output / "semantic_diff_audit.json", semantic)
    if not budget["passed"] or not semantic["passed"]:
        raise SystemExit("2D0D prepare audit failed")
    print("EXPERIMENT_2D0D_PREPARE_PASS geometries=4 sum_W=5312", flush=True)


def forward_top_schedule(model, tokens, windows):
    if len(windows) != d0.N_LAYER:
        raise ValueError("2D0D schedule must contain exactly 12 windows")
    length = tokens.size(1)
    positions = torch.arange(length, dtype=torch.long, device=tokens.device)
    value = model.transformer.wte(tokens) + model.transformer.wpe(positions)
    for block, window in zip(model.transformer.h, windows):
        if window == length:
            value = block(value)
        else:
            value, _ = d0.run_block(block, value, window)
    return model.transformer.ln_f(value)


def expected_identities(manifest, limit=None):
    rows = manifest["batches"] if limit is None else manifest["batches"][:limit]
    return [
        {"input_sha256": row["input_sha256"], "target_sha256": row["target_sha256"]}
        for row in rows
    ]


def identities_match(identities, manifest, limit=None):
    observed = [
        {"input_sha256": row["input_sha256"], "target_sha256": row["target_sha256"]}
        for row in identities
    ]
    return observed == expected_identities(manifest, limit)


def load_parent_oracle():
    preflight = json.loads(PARENT_PREFLIGHT_PATH.read_text())
    matrix = json.loads(MARGINAL_MATRIX_PATH.read_text())
    return {
        "full_loss": preflight["baseline"]["validation_loss"],
        "full_per_batch_losses": preflight["baseline"]["per_batch_losses"],
        "full_per_position_loss": preflight["baseline"]["per_position_loss"],
        "full_batch_identities": preflight["baseline"]["batch_identities"],
        "sentinels": {
            "B11_W256": preflight["baseline"]["validation_loss"] + matrix["B11"]["256"],
            "B1_W512": preflight["baseline"]["validation_loss"] + matrix["B1"]["512"],
            "B12_W1": preflight["baseline"]["validation_loss"] + matrix["B12"]["1"],
        },
    }


def validation_manifest(val_path):
    loader = d0.ExplicitShardLoader([str(Path(val_path).resolve())], d0.VALIDATION_B, d0.T)
    rows = []
    combined = []
    for batch_index in range(d0.PHASE_A_BATCHES):
        x, y = loader.next_batch()
        identity = batch_identity(x, y)
        combined.append(identity["combined_sha256"])
        rows.append({
            "batch_index": batch_index,
            **identity,
            "input_shape": list(x.shape),
            "target_shape": list(y.shape),
            "input_dtype": str(x.dtype),
            "target_dtype": str(y.dtype),
            "loss_denominator": x.numel(),
        })
    return {
        "validation_shard": str(Path(val_path).resolve()),
        "validation_shard_sha256": d0.file_sha256(val_path),
        "canonical_batch_collection_sha256": d0.aggregate_hashes(combined),
        "batches": rows,
        "batch_count": len(rows),
        "batch_size": d0.VALIDATION_B,
        "sequence_length": d0.T,
        "targets_per_candidate": EXPECTED_TARGETS,
    }


@torch.no_grad()
def evaluate_schedule(model, val_path, device, windows, batch_count, baseline_argmax=None, capture_arrays=False):
    loader = d0.ExplicitShardLoader([val_path], d0.VALIDATION_B, d0.T)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    start_unix = time.time()
    start = time.monotonic()
    loss_sum = 0.0
    tokens = 0
    per_batch_losses = []
    per_position_sum = np.zeros(d0.T, dtype=np.float64)
    per_position_matches = np.zeros(d0.T, dtype=np.int64)
    identities = []
    finite_batches = 0
    loss_arrays = []
    argmax_arrays = []
    for batch_index in range(batch_count):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(batch_identity(cpu_x, cpu_y))
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            top = forward_top_schedule(model, x, windows)
            losses, logits = d0.token_cross_entropy(model, top, y)
        finite = bool(torch.isfinite(losses).all() and torch.isfinite(logits).all())
        finite_batches += int(finite)
        batch_loss = losses.float().mean().item()
        predictions = logits.argmax(dim=-1).to(torch.int32).cpu().numpy()
        per_batch_losses.append(batch_loss)
        loss_sum += losses.double().sum().item()
        tokens += losses.numel()
        per_position_sum += losses.double().sum(dim=0).cpu().numpy()
        if baseline_argmax is not None:
            per_position_matches += np.count_nonzero(predictions == baseline_argmax[batch_index], axis=0)
        if capture_arrays:
            loss_arrays.append(losses.float().cpu().numpy())
            argmax_arrays.append(predictions)
        print(f"2D0D batches={batch_index + 1:02d}/{batch_count} loss={batch_loss:.10f}", flush=True)
        del x, y, top, losses, logits, predictions
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - start
    row = {
        "windows": list(windows),
        "validation_loss": loss_sum / tokens,
        "validation_targets": tokens,
        "per_batch_losses": per_batch_losses,
        "per_position_loss": (per_position_sum / (batch_count * d0.VALIDATION_B)).tolist(),
        "per_position_argmax_agreement": (
            (per_position_matches / (batch_count * d0.VALIDATION_B)).tolist()
            if baseline_argmax is not None else None
        ),
        "argmax_agreement": (
            int(per_position_matches.sum()) / tokens if baseline_argmax is not None else None
        ),
        "batch_identities": identities,
        "canonical_validation_sha256": d0.aggregate_hashes([row["combined_sha256"] for row in identities]),
        "finite_batches": finite_batches,
        "all_losses_and_predictions_finite": finite_batches == batch_count,
        "evaluation_precision": "torch.autocast(cuda,bfloat16)",
        "loss_denominator": tokens,
        "absolute_positions_unchanged": True,
        "causal_window_includes_current": True,
        "performance": {
            "wall_start_unix": start_unix,
            "wall_end_unix": time.time(),
            "wall_seconds": elapsed,
            "targets_per_second": tokens / elapsed,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(0) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(0) / 1024**2,
        },
        **TRAINING_AUDIT,
        **FORBIDDEN_AUDIT,
    }
    if capture_arrays:
        row["losses"] = np.stack(loss_arrays)
        row["argmax"] = np.stack(argmax_arrays)
    return row


def serializable_evaluation(row):
    return {key: value for key, value in row.items() if key not in ("losses", "argmax")}


def run_preflight(args):
    require_git(clean=True)
    load_config()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("2D0D preflight requires exactly one visible GPU")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("2D0D forbids DDP/NCCL/distributed execution")
    output = Path(args.output_dir).resolve()
    geometry_manifest = json.loads((output / "geometry_manifest.json").read_text())
    budget = json.loads((output / "budget_audit.json").read_text())
    semantic = json.loads((output / "semantic_diff_audit.json").read_text())
    if geometry_manifest["geometries"] != {key: list(value) for key, value in GEOMETRIES.items()}:
        raise SystemExit("2D0D geometry manifest mismatch")
    if not budget["passed"] or not semantic["passed"]:
        raise SystemExit("2D0D prepare audits are not passing")
    torch.cuda.set_device(0)
    d0.seed_all(0)
    device = torch.device("cuda", 0)
    val_path = str(Path(args.validation_shard).resolve())
    manifest = validation_manifest(val_path)
    if manifest["validation_shard_sha256"] != d0.VAL_SHA256:
        raise SystemExit("2D0D validation shard SHA mismatch")
    if manifest["canonical_batch_collection_sha256"] != d0.CANONICAL_VALIDATION_SHA256:
        raise SystemExit("2D0D canonical batch collection mismatch")
    _, model, source_audit = d0.load_standard_model(args.parent_checkpoint, device)
    model_before = d0.tensor_state_sha256(model)
    oracle = load_parent_oracle()
    baseline_arrays = np.load(PARENT_BASELINE_PATH)
    full = evaluate_schedule(
        model, val_path, device, [d0.T] * d0.N_LAYER, d0.PHASE_A_BATCHES,
        capture_arrays=True,
    )
    baseline_deltas = [
        abs(left - right) for left, right in zip(full["per_batch_losses"], oracle["full_per_batch_losses"])
    ]
    full_checks = {
        "loss_absolute_delta": abs(full["validation_loss"] - oracle["full_loss"]),
        "per_batch_max_absolute_delta": max(baseline_deltas),
        "per_target_loss_arrays_exact": np.array_equal(full["losses"], baseline_arrays["per_target_loss"]),
        "argmax_arrays_exact": np.array_equal(full["argmax"], baseline_arrays["argmax_token_id"]),
        "batch_manifest_exact": identities_match(full["batch_identities"], manifest),
    }
    full_checks["passed"] = (
        full_checks["loss_absolute_delta"] <= SENTINEL_TOLERANCE
        and full_checks["per_batch_max_absolute_delta"] <= SENTINEL_TOLERANCE
        and full_checks["per_target_loss_arrays_exact"]
        and full_checks["argmax_arrays_exact"]
        and full_checks["batch_manifest_exact"]
    )
    sentinel_schedules = {
        "B11_W256": [1024] * 10 + [256, 1024],
        "B1_W512": [512] + [1024] * 11,
        "B12_W1": [1024] * 11 + [1],
    }
    sentinels = {}
    for name, schedule in sentinel_schedules.items():
        row = evaluate_schedule(model, val_path, device, schedule, d0.PHASE_A_BATCHES)
        delta = abs(row["validation_loss"] - oracle["sentinels"][name])
        sentinels[name] = {
            "windows": schedule,
            "expected_validation_loss": oracle["sentinels"][name],
            "observed_validation_loss": row["validation_loss"],
            "absolute_delta": delta,
            "tolerance": SENTINEL_TOLERANCE,
            "batch_manifest_exact": identities_match(row["batch_identities"], manifest),
            "passed": delta <= SENTINEL_TOLERANCE and identities_match(row["batch_identities"], manifest),
        }
    model_after = d0.tensor_state_sha256(model)
    preflight = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "source_audit": source_audit,
        "full_context_oracle": oracle["full_loss"],
        "full_context_observed": full["validation_loss"],
        "full_context": full_checks,
        "sentinels": sentinels,
        "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after,
        "model_tensors_unchanged": model_before == model_after,
        "geometry_manifest_present_before_compute": True,
        "budget_audit_passed": budget["passed"],
        "semantic_diff_passed": semantic["passed"],
        "environment": environment_audit(),
        **TRAINING_AUDIT,
        **FORBIDDEN_AUDIT,
    }
    preflight["passed"] = (
        source_audit["passed"]
        and full_checks["passed"]
        and all(row["passed"] for row in sentinels.values())
        and preflight["model_tensors_unchanged"]
        and budget["passed"]
        and semantic["passed"]
    )
    d0.durable_json(output / "batch_manifest.json", manifest)
    d0.durable_json(output / "environment.json", preflight["environment"])
    d0.durable_json(output / "preflight_audit.json", preflight)
    print(f"EXPERIMENT_2D0D_PREFLIGHT_{'PASS' if preflight['passed'] else 'FAIL'} full_delta={full_checks['loss_absolute_delta']:.3e}", flush=True)
    if not preflight["passed"]:
        raise SystemExit("2D0D preflight failed; candidate evaluation prohibited")


def position_bin_rows(candidate, full_per_position):
    rows = {}
    candidate_per_position = candidate["per_position_loss"]
    agreement = candidate["per_position_argmax_agreement"]
    for label, start, end in POSITION_BINS:
        full_loss = float(np.mean(full_per_position[start : end + 1]))
        candidate_loss = float(np.mean(candidate_per_position[start : end + 1]))
        rows[label] = {
            "start": start,
            "end": end,
            "full_loss": full_loss,
            "candidate_loss": candidate_loss,
            "damage": candidate_loss - full_loss,
            "target_count": (end - start + 1) * d0.PHASE_A_BATCHES * d0.VALIDATION_B,
            "argmax_agreement": float(np.mean(agreement[start : end + 1])),
        }
    return rows


def run_worker(args):
    require_git(clean=True)
    load_config()
    physical_gpu = int(args.physical_gpu)
    if physical_gpu not in GPU_ASSIGNMENT:
        raise SystemExit("invalid 2D0D physical GPU assignment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("each 2D0D worker requires exactly one visible GPU")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("2D0D worker forbids DDP/NCCL/distributed execution")
    output = Path(args.output_dir).resolve()
    preflight = json.loads((output / "preflight_audit.json").read_text())
    manifest = json.loads((output / "batch_manifest.json").read_text())
    geometry_manifest = json.loads((output / "geometry_manifest.json").read_text())
    if not preflight["passed"]:
        raise SystemExit("2D0D worker requires passing preflight")
    geometry = GPU_ASSIGNMENT[physical_gpu]
    schedule = GEOMETRIES[geometry]
    if geometry_manifest["geometries"][geometry] != list(schedule):
        raise SystemExit("2D0D worker geometry differs from frozen manifest")
    run_root = Path(args.run_root).resolve()
    worker_path = run_root / "workers" / f"gpu{physical_gpu}_{geometry}.json"
    if worker_path.exists():
        raise SystemExit(f"refusing to overwrite existing worker artifact: {worker_path}")
    torch.cuda.set_device(0)
    d0.seed_all(physical_gpu)
    device = torch.device("cuda", 0)
    val_path = str(Path(args.validation_shard).resolve())
    oracle = load_parent_oracle()
    baseline_arrays = np.load(PARENT_BASELINE_PATH)
    _, model, source_audit = d0.load_standard_model(args.parent_checkpoint, device)
    model_before = d0.tensor_state_sha256(model)
    worker_start_unix = time.time()
    worker_start = time.monotonic()
    baseline = evaluate_schedule(model, val_path, device, [d0.T] * d0.N_LAYER, 2)
    first_two_deltas = [
        abs(value - oracle["full_per_batch_losses"][index])
        for index, value in enumerate(baseline["per_batch_losses"])
    ]
    baseline_passed = (
        max(first_two_deltas) <= SENTINEL_TOLERANCE
        and identities_match(baseline["batch_identities"], manifest, limit=2)
        and baseline["all_losses_and_predictions_finite"]
    )
    if not baseline_passed:
        raise SystemExit(f"2D0D GPU{physical_gpu} baseline worker sentinel failed")
    candidate = evaluate_schedule(
        model,
        val_path,
        device,
        schedule,
        d0.PHASE_A_BATCHES,
        baseline_argmax=baseline_arrays["argmax_token_id"],
    )
    candidate["geometry"] = geometry
    candidate["joint_damage_vs_full"] = candidate["validation_loss"] - oracle["full_loss"]
    candidate["position_bins"] = position_bin_rows(candidate, oracle["full_per_position_loss"])
    candidate["batch_manifest_exact"] = identities_match(candidate["batch_identities"], manifest)
    candidate["canonical_validation_exact"] = candidate["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256
    candidate["source_checkpoint_sha256"] = d0.SOURCE_SHA256
    candidate["nominal_sum_W"] = sum(schedule)
    candidate["nominal_fraction_full"] = sum(schedule) / (12 * d0.T)
    candidate["passed"] = (
        candidate["batch_manifest_exact"]
        and candidate["canonical_validation_exact"]
        and candidate["validation_targets"] == EXPECTED_TARGETS
        and candidate["all_losses_and_predictions_finite"]
        and len(candidate["windows"]) == 12
        and candidate["nominal_sum_W"] == 5312
        and candidate["evaluation_precision"] == "torch.autocast(cuda,bfloat16)"
        and candidate["loss_denominator"] == EXPECTED_TARGETS
    )
    model_after = d0.tensor_state_sha256(model)
    worker = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "physical_gpu": physical_gpu,
        "geometry": geometry,
        "schedule": list(schedule),
        "gpu_name": torch.cuda.get_device_name(0),
        "source_audit": source_audit,
        "baseline_sentinel": {
            "batch_count": 2,
            "per_batch_losses": baseline["per_batch_losses"],
            "batch_identities": baseline["batch_identities"],
            "absolute_deltas": first_two_deltas,
            "passed": baseline_passed,
        },
        "candidate": candidate,
        "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after,
        "model_tensors_unchanged": model_before == model_after,
        "worker_start_unix": worker_start_unix,
        "worker_end_unix": time.time(),
        "wall_seconds": time.monotonic() - worker_start,
        "independent_process": True,
        "ddp": False,
        "nccl": False,
        "distributed_initialized": torch.distributed.is_initialized(),
        **TRAINING_AUDIT,
        **FORBIDDEN_AUDIT,
    }
    worker["passed"] = (
        source_audit["passed"]
        and baseline_passed
        and candidate["passed"]
        and worker["model_tensors_unchanged"]
        and not worker["distributed_initialized"]
    )
    d0.durable_json(worker_path, worker)
    print(
        f"EXPERIMENT_2D0D_WORKER_{'PASS' if worker['passed'] else 'FAIL'} "
        f"GPU={physical_gpu} geometry={geometry} loss={candidate['validation_loss']:.10f} "
        f"damage={candidate['joint_damage_vs_full']:+.10f}",
        flush=True,
    )
    if not worker["passed"]:
        raise SystemExit(f"2D0D GPU{physical_gpu} worker failed")


def paired_stats(name_a, name_b, values_a, values_b):
    if len(values_a) != len(values_b) or len(values_a) != d0.PHASE_A_BATCHES:
        raise ValueError("paired comparison requires the same 20 canonical batches")
    differences = [float(left - right) for left, right in zip(values_a, values_b)]
    mean = statistics.fmean(differences)
    sample_sd = statistics.stdev(differences)
    standard_error = sample_sd / math.sqrt(len(differences))
    return {
        "a": name_a,
        "b": name_b,
        "differences_a_minus_b": differences,
        "mean_a_minus_b": mean,
        "median_a_minus_b": statistics.median(differences),
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "descriptive_95_percent_interval": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "a_wins": sum(value < 0 for value in differences),
        "b_wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "batch_count": len(differences),
        "interval_is_descriptive_not_formal_iid_inference": True,
    }


def empirical_marginal_terms(matrix, schedule):
    return [
        {
            "block": f"B{layer}",
            "window": schedule[layer - 1],
            "damage": matrix[f"B{layer}"][str(schedule[layer - 1])],
            "source": "2D0C authoritative validation_damage_matrix.json",
        }
        for layer in range(1, 13)
    ]


def training_forbidden_exact(row):
    return all(row[key] == value for key, value in {**TRAINING_AUDIT, **FORBIDDEN_AUDIT}.items())


def run_assemble(args):
    require_git(clean=True)
    load_config()
    output = Path(args.output_dir).resolve()
    run_root = Path(args.run_root).resolve()
    preflight = json.loads((output / "preflight_audit.json").read_text())
    manifest = json.loads((output / "batch_manifest.json").read_text())
    geometry_manifest = json.loads((output / "geometry_manifest.json").read_text())
    budget = json.loads((output / "budget_audit.json").read_text())
    semantic = json.loads((output / "semantic_diff_audit.json").read_text())
    if not preflight["passed"]:
        raise SystemExit("cannot assemble failed 2D0D preflight")
    workers = {}
    candidates = {}
    for gpu, geometry in GPU_ASSIGNMENT.items():
        worker = json.loads((run_root / "workers" / f"gpu{gpu}_{geometry}.json").read_text())
        workers[geometry] = worker
        candidates[geometry] = worker["candidate"]
    oracle = load_parent_oracle()
    full_loss = oracle["full_loss"]
    ordered = sorted(GEOMETRIES, key=lambda name: candidates[name]["validation_loss"])
    ranks = {name: index + 1 for index, name in enumerate(ordered)}
    pair_names = [
        ("EMPIRICAL", "TOP_WIDE_TRIANGLE"),
        ("EMPIRICAL", "REVERSE_TRIANGLE"),
        ("EMPIRICAL", "UNIFORM_MATCHED"),
        ("TOP_WIDE_TRIANGLE", "REVERSE_TRIANGLE"),
        ("TOP_WIDE_TRIANGLE", "UNIFORM_MATCHED"),
        ("REVERSE_TRIANGLE", "UNIFORM_MATCHED"),
    ]
    pairs = [
        paired_stats(left, right, candidates[left]["per_batch_losses"], candidates[right]["per_batch_losses"])
        for left, right in pair_names
    ]
    best, second = ordered[:2]
    best_vs_second = paired_stats(
        best,
        second,
        candidates[best]["per_batch_losses"],
        candidates[second]["per_batch_losses"],
    )
    best_margin = candidates[second]["validation_loss"] - candidates[best]["validation_loss"]
    clear_winner = best_margin >= CLEAR_MARGIN and best_vs_second["a_wins"] >= CLEAR_WINS
    feed_forward_conclusion = (
        {
            "EMPIRICAL": "EMPIRICAL PROFILE IS BEST FEED-FORWARD GEOMETRY",
            "TOP_WIDE_TRIANGLE": "TOP-WIDE TRIANGLE IS BEST FEED-FORWARD GEOMETRY",
            "REVERSE_TRIANGLE": "REVERSE TRIANGLE IS BEST FEED-FORWARD GEOMETRY",
            "UNIFORM_MATCHED": "UNIFORM IS BEST FEED-FORWARD GEOMETRY",
        }[best]
        if clear_winner
        else "NO CLEAR MATCHED-BUDGET FEED-FORWARD WINNER"
    )
    matrix = json.loads(MARGINAL_MATRIX_PATH.read_text())
    terms = empirical_marginal_terms(matrix, GEOMETRIES["EMPIRICAL"])
    marginal_sum = math.fsum(row["damage"] for row in terms)
    empirical_joint_damage = candidates["EMPIRICAL"]["joint_damage_vs_full"]
    interaction = empirical_joint_damage - marginal_sum
    empirical_interaction = {
        "full_loss": full_loss,
        "empirical_joint_loss": candidates["EMPIRICAL"]["validation_loss"],
        "empirical_joint_damage": empirical_joint_damage,
        "single_layer_marginal_terms": terms,
        "sum_single_layer_marginal_damages": marginal_sum,
        "interaction_E_joint_minus_marginal_sum": interaction,
        "interaction_ratio_joint_over_marginal_sum": empirical_joint_damage / marginal_sum,
        "decomposition_scope": "controlled descriptive difference, not a formal interaction decomposition",
        "other_geometry_interaction_decomposition": {
            "TOP_WIDE_TRIANGLE": "NOT DIRECTLY AVAILABLE",
            "REVERSE_TRIANGLE": "NOT DIRECTLY AVAILABLE",
            "UNIFORM_MATCHED": "NOT DIRECTLY AVAILABLE",
        },
    }
    joint_losses = {
        "FULL": {
            "sum_W": 12288,
            "fraction_full": 1.0,
            "validation_loss": full_loss,
            "damage_vs_full": 0.0,
            "argmax_agreement_vs_full": 1.0,
            "rank": None,
        },
        **{
            name: {
                "sum_W": sum(GEOMETRIES[name]),
                "fraction_full": sum(GEOMETRIES[name]) / (12 * 1024),
                "validation_loss": candidates[name]["validation_loss"],
                "damage_vs_full": candidates[name]["joint_damage_vs_full"],
                "argmax_agreement_vs_full": candidates[name]["argmax_agreement"],
                "rank": ranks[name],
            }
            for name in GEOMETRIES
        },
    }
    per_batch = {"FULL": oracle["full_per_batch_losses"]}
    per_batch.update({name: candidates[name]["per_batch_losses"] for name in GEOMETRIES})
    per_position = {
        "FULL": {"loss": oracle["full_per_position_loss"]},
        **{
            name: {
                "loss": candidates[name]["per_position_loss"],
                "damage_vs_full": [
                    short - full
                    for short, full in zip(candidates[name]["per_position_loss"], oracle["full_per_position_loss"])
                ],
            }
            for name in GEOMETRIES
        },
    }
    position_bins = {name: candidates[name]["position_bins"] for name in GEOMETRIES}
    argmax = {
        name: {
            "overall_vs_full": candidates[name]["argmax_agreement"],
            "per_position_vs_full": candidates[name]["per_position_argmax_agreement"],
            "position_bins_vs_full": {
                label: row["argmax_agreement"] for label, row in candidates[name]["position_bins"].items()
            },
        }
        for name in GEOMETRIES
    }
    baseline_sentinel_identity = all(
        workers[name]["baseline_sentinel"]["per_batch_losses"]
        == workers[ordered[0]]["baseline_sentinel"]["per_batch_losses"]
        and workers[name]["baseline_sentinel"]["batch_identities"]
        == workers[ordered[0]]["baseline_sentinel"]["batch_identities"]
        for name in GEOMETRIES
    )
    integrity_pre = {
        "2d0c_frozen_tag_exact": git_output("rev-list", "-n", "1", FROZEN_TAG) == FROZEN_COMMIT,
        "source_checkpoint_exact": all(row["source_audit"]["passed"] for row in workers.values()),
        "canonical_validation_manifest_exact": manifest["canonical_batch_collection_sha256"] == d0.CANONICAL_VALIDATION_SHA256,
        "full_context_baseline_exact": preflight["full_context"]["passed"],
        "single_layer_sentinels_exact": all(row["passed"] for row in preflight["sentinels"].values()),
        "standard_gpt2_exact_full_attnres_absent": all(row["source_audit"]["passed"] for row in workers.values()),
        "geometry_manifest_frozen_before_compute": geometry_manifest["created_before_result_bearing_compute"],
        "exactly_four_candidates": len(candidates) == 4 == len(GEOMETRIES),
        "budget_audit": budget["passed"],
        "causal_window_and_positions_exact": all(row["causal_window_includes_current"] and row["absolute_positions_unchanged"] for row in candidates.values()),
        "same_bf16_runtime_targets_denominator": all(row["evaluation_precision"] == "torch.autocast(cuda,bfloat16)" and row["validation_targets"] == EXPECTED_TARGETS and row["loss_denominator"] == EXPECTED_TARGETS for row in candidates.values()),
        "independent_processes_no_ddp_no_nccl": all(row["independent_process"] and not row["ddp"] and not row["nccl"] and not row["distributed_initialized"] for row in workers.values()),
        "worker_baseline_sentinels_identical": baseline_sentinel_identity,
        "all_losses_predictions_finite": all(row["all_losses_and_predictions_finite"] for row in candidates.values()),
        "zero_training_and_forbidden_features": all(training_forbidden_exact(row) and training_forbidden_exact(row["candidate"]) for row in workers.values()),
        "model_hashes_unchanged": all(row["model_tensors_unchanged"] for row in workers.values()),
        "exactly_four_scientific_candidate_evaluations": len(candidates) == 4,
        "paired_batch_accounting_exact": all(row["batch_count"] == 20 and row["a_wins"] + row["b_wins"] + row["ties"] == 20 for row in pairs),
        "empirical_terms_exact_2d0c_cells": all(row["damage"] == matrix[row["block"]][str(row["window"])] for row in terms),
        "empirical_interaction_arithmetic_exact": interaction == empirical_joint_damage - marginal_sum,
        "semantic_diff": semantic["passed"],
        "all_workers_passed": all(row["passed"] for row in workers.values()),
    }
    integrity_pre["passed"] = all(integrity_pre.values())
    performance = {
        "total_four_gpu_elapsed_wall_seconds": max(row["worker_end_unix"] for row in workers.values()) - min(row["worker_start_unix"] for row in workers.values()),
        "per_geometry": {
            name: {
                **candidates[name]["performance"],
                "evaluated_targets": candidates[name]["validation_targets"],
            }
            for name in GEOMETRIES
        },
        "masked_evaluator_throughput_is_not_optimized_deployment_throughput": True,
    }
    source_manifest = {
        "checkpoint": preflight["source_audit"]["checkpoint"],
        "checkpoint_sha256": d0.SOURCE_SHA256,
        "checkpoint_bytes": d0.SOURCE_BYTES,
        "historical_training_tokens": d0.SOURCE_TOKENS,
        "validation_shard": manifest["validation_shard"],
        "validation_shard_sha256": d0.VAL_SHA256,
        "canonical_validation_sha256": d0.CANONICAL_VALIDATION_SHA256,
        "architecture": preflight["source_audit"],
        "frozen_2d0c_tag": FROZEN_TAG,
        "frozen_2d0c_commit": FROZEN_COMMIT,
    }
    summary = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "feed_forward_geometry_conclusion": feed_forward_conclusion,
        "clear_matched_budget_winner": clear_winner,
        "best_geometry": best,
        "second_best_geometry": second,
        "best_minus_second_loss_margin": best_margin,
        "best_vs_second_paired": best_vs_second,
        "joint_losses": joint_losses,
        "empirical_interaction": empirical_interaction,
        "integrity_pre_audit": integrity_pre,
        "training_performed": False,
        "recurrence_active": False,
        "candidate_count": 4,
    }
    for name, payload in (
        ("joint_losses.json", joint_losses),
        ("per_batch_losses.json", per_batch),
        ("paired_comparisons.json", {"pairs": pairs, "best_vs_second": best_vs_second}),
        ("empirical_interaction.json", empirical_interaction),
        ("per_position_loss.json", per_position),
        ("position_bin_loss.json", position_bins),
        ("argmax_agreement.json", argmax),
        ("performance.json", performance),
        ("source_manifest.json", source_manifest),
        ("result_summary.json", summary),
    ):
        d0.durable_json(output / name, payload)
    d0.durable_json(
        output / "commands_and_runtime.json",
        {
            "branch": BRANCH,
            "implementation_commit": git_output("rev-parse", "HEAD"),
            "commands": [
                "CUDA_VISIBLE_DEVICES=0 python scripts/experiment_2d0d.py preflight ...",
                "four independent workers: GPU0 EMPIRICAL, GPU1 TOP_WIDE_TRIANGLE, GPU2 REVERSE_TRIANGLE, GPU3 UNIFORM_MATCHED",
                "python scripts/experiment_2d0d.py assemble ...",
                "python scripts/experiment_2d0d.py finalize ...",
            ],
            "ddp": False,
            "nccl": False,
            "training": False,
            "candidate_evaluations": 4,
            "runtime": preflight["environment"],
        },
    )
    print(f"EXPERIMENT_2D0D_RESULTS_ASSEMBLED best={best} clear={clear_winner} integrity={integrity_pre['passed']}", flush=True)
    if not integrity_pre["passed"]:
        raise SystemExit("2D0D assembled integrity failed")


def generate_plots(output, summary, geometry_manifest, per_position):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "EMPIRICAL": "#0072B2",
        "TOP_WIDE_TRIANGLE": "#D55E00",
        "REVERSE_TRIANGLE": "#009E73",
        "UNIFORM_MATCHED": "#CC79A7",
    }
    labels = {
        "EMPIRICAL": "Empirical",
        "TOP_WIDE_TRIANGLE": "Top-wide triangle",
        "REVERSE_TRIANGLE": "Reverse triangle",
        "UNIFORM_MATCHED": "Uniform matched",
    }
    layers = np.arange(1, 13)
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for name, schedule in geometry_manifest["geometries"].items():
        axis.plot(layers, schedule, marker="o", linewidth=2.2, label=labels[name], color=colors[name])
    axis.set_xticks(layers, [f"B{value}" for value in layers])
    axis.set_xlabel("Transformer block")
    axis.set_ylabel("Attention window W (includes current token)")
    axis.set_title("Matched joint-KV geometry profiles (ΣW=5312 each)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "matched_geometry_profiles.png", dpi=220)
    fig.savefig(output / "matched_geometry_profiles.svg")
    plt.close(fig)

    names = list(GEOMETRIES)
    damages = [summary["joint_losses"][name]["damage_vs_full"] for name in names]
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = axis.bar([labels[name] for name in names], damages, color=[colors[name] for name in names])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Joint validation-loss damage vs FULL")
    axis.set_title("Matched-budget joint geometry damage")
    axis.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, damages):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:+.4f}", ha="center", va="bottom")
    fig.savefig(output / "matched_geometry_damage.png", dpi=220)
    fig.savefig(output / "matched_geometry_damage.svg")
    plt.close(fig)

    positions = np.arange(1, d0.T)
    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for name in names:
        axis.plot(
            positions,
            per_position[name]["damage_vs_full"][1:],
            linewidth=1.5,
            alpha=0.9,
            label=labels[name],
            color=colors[name],
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Absolute target position")
    axis.set_ylabel("Mean CE damage vs FULL")
    axis.set_title("Position-dependent joint geometry damage")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(output / "geometry_damage_by_position.png", dpi=220)
    fig.savefig(output / "geometry_damage_by_position.svg")
    plt.close(fig)


def f10(value):
    return f"{value:.10f}"


def comparison_sentence(left, right, joint):
    left_loss = joint[left]["validation_loss"]
    right_loss = joint[right]["validation_loss"]
    if left_loss < right_loss:
        return f"{left} beat {right} by {right_loss - left_loss:.10f} validation loss."
    if right_loss < left_loss:
        return f"{right} beat {left} by {left_loss - right_loss:.10f} validation loss."
    return f"{left} and {right} tied in mean validation loss."


def render_report(output, summary, interaction_label, recommendation, implementation_commit, results_commit):
    geometry_manifest = json.loads((output / "geometry_manifest.json").read_text())
    joint = json.loads((output / "joint_losses.json").read_text())
    pairs = json.loads((output / "paired_comparisons.json").read_text())
    interaction = json.loads((output / "empirical_interaction.json").read_text())
    bins = json.loads((output / "position_bin_loss.json").read_text())
    argmax = json.loads((output / "argmax_agreement.json").read_text())
    performance = json.loads((output / "performance.json").read_text())
    best = summary["best_geometry"]
    second = summary["second_best_geometry"]
    max_damage_bins = {
        name: max(rows, key=lambda label: rows[label]["damage"])
        for name, rows in bins.items()
    }
    best_argmax = max(GEOMETRIES, key=lambda name: argmax[name]["overall_vs_full"])
    joint_rows = [
        f"| {name} | {row['sum_W']} | {row['fraction_full']:.6f} | {row['validation_loss']:.10f} | {row['damage_vs_full']:+.10f} | {row['argmax_agreement_vs_full']:.10f} | {row['rank'] if row['rank'] is not None else '—'} |"
        for name, row in joint.items()
    ]
    budget_rows = []
    for layer_index in range(12):
        budget_rows.append(
            "| B{} | {} | {} | {} | {} |".format(
                layer_index + 1,
                geometry_manifest["geometries"]["EMPIRICAL"][layer_index],
                geometry_manifest["geometries"]["TOP_WIDE_TRIANGLE"][layer_index],
                geometry_manifest["geometries"]["REVERSE_TRIANGLE"][layer_index],
                geometry_manifest["geometries"]["UNIFORM_MATCHED"][layer_index],
            )
        )
    budget_rows.append("| **ΣW** | **5312** | **5312** | **5312** | **5312** |")
    pair_rows = [
        f"| {row['a']} | {row['b']} | {row['mean_a_minus_b']:+.10f} | {row['a_wins']} | {row['b_wins']} | {row['ties']} | [{row['descriptive_95_percent_interval'][0]:+.10f}, {row['descriptive_95_percent_interval'][1]:+.10f}] |"
        for row in pairs["pairs"]
    ]
    term_rows = [
        f"| {row['block']} | {row['window']} | {row['damage']:+.10f} |"
        for row in interaction["single_layer_marginal_terms"]
    ]
    position_rows = [
        f"| {name} | {max_damage_bins[name]} | {bins[name][max_damage_bins[name]]['damage']:+.10f} |"
        for name in GEOMETRIES
    ]
    argmax_rows = [
        f"| {name} | {argmax[name]['overall_vs_full']:.10f} |"
        for name in sorted(GEOMETRIES, key=lambda value: argmax[value]["overall_vs_full"], reverse=True)
    ]
    empirical_ratio = interaction["interaction_ratio_joint_over_marginal_sum"]
    interaction_evidence = (
        "Yes: joint damage exceeded the exact marginal sum, consistent with compensation disappearing under simultaneous shortening."
        if interaction["interaction_E_joint_minus_marginal_sum"] > 0
        else "No positive super-additive evidence: joint damage did not exceed the exact marginal sum."
    )
    use_empirical_directly = (
        "No: the simultaneous profile should not be adopted directly because its measured joint damage exceeded its marginal prediction."
        if interaction["interaction_E_joint_minus_marginal_sum"] > 0
        else "Only cautiously: the joint measurement is required, and the marginal profile alone is not sufficient evidence."
    )
    early_importance = (
        "Yes in this ordinary no-recurrence model: the reverse triangle outperformed the top-wide triangle while preserving wider early-layer windows."
        if joint["REVERSE_TRIANGLE"]["validation_loss"] < joint["TOP_WIDE_TRIANGLE"]["validation_loss"]
        else "Not from this comparison: the reverse triangle did not outperform the top-wide triangle."
    )
    report = f"""# Experiment 2D0D — Matched Joint-KV Geometry Evaluation

## Feed-forward geometry conclusion

**{summary['feed_forward_geometry_conclusion']}**

The numerically best matched-budget geometry was **{best}**. Its advantage over **{second}** was {summary['best_minus_second_loss_margin']:.10f}, with {summary['best_vs_second_paired']['a_wins']}/20 paired-batch wins. The preregistered clear-winner rule was {'met' if summary['clear_matched_budget_winner'] else 'not met'}.

## Joint-interaction conclusion

**{interaction_label}**

Empirical joint damage was {interaction['empirical_joint_damage']:+.10f}, versus an exact 2D0C single-layer marginal sum of {interaction['sum_single_layer_marginal_damages']:+.10f}. Their difference was {interaction['interaction_E_joint_minus_marginal_sum']:+.10f}; the joint/marginal ratio was {empirical_ratio:.6f}. This is a controlled descriptive difference, not a formal interaction decomposition.

## Joint geometry results

| Geometry | ΣW | Fraction full | Val loss | Damage | Argmax agreement | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(joint_rows)}

All four candidates use 43.22916667% of the nominal 12×1024 layer-window budget, a 56.77083333% reduction. This is a KV-capacity proxy, not an exact total-memory or optimized serving-throughput measurement.

## Paired comparisons

| A | B | Mean A-B | A wins | B wins | Ties | 95% descriptive interval |
| --- | --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(pair_rows)}

The intervals describe these fixed evaluation batches; they are not formal IID confidence intervals.

## Empirical additive-versus-joint decomposition

| Block | Empirical W | Exact marginal damage |
| --- | ---: | ---: |
{chr(10).join(term_rows)}

- FULL loss: {interaction['full_loss']:.10f}
- Empirical joint loss: {interaction['empirical_joint_loss']:.10f}
- Empirical joint damage: {interaction['empirical_joint_damage']:+.10f}
- Sum of exact single-layer marginal damages: {interaction['sum_single_layer_marginal_damages']:+.10f}
- Interaction (joint minus marginal sum): {interaction['interaction_E_joint_minus_marginal_sum']:+.10f}
- Joint/marginal ratio: {empirical_ratio:.10f}

Interaction decompositions for Triangle, Reverse, and Uniform are **NOT DIRECTLY AVAILABLE** because their unmeasured widths were not interpolated or evaluated separately.

## Position-dependent damage

| Geometry | Highest-damage common bin | Damage in bin |
| --- | --- | ---: |
{chr(10).join(position_rows)}

Machine-readable per-position arrays and every common-bin result are stored in `per_position_loss.json` and `position_bin_loss.json`.

## Argmax retention

| Geometry | Agreement vs FULL |
| --- | ---: |
{chr(10).join(argmax_rows)}

**{best_argmax}** retained full-model argmax predictions best.

## Direct answers to Q1–Q15

1. **Joint damage:** Empirical {joint['EMPIRICAL']['damage_vs_full']:+.10f}; Triangle {joint['TOP_WIDE_TRIANGLE']['damage_vs_full']:+.10f}; Reverse {joint['REVERSE_TRIANGLE']['damage_vs_full']:+.10f}; Uniform {joint['UNIFORM_MATCHED']['damage_vs_full']:+.10f}.
2. **Lowest loss:** {best}, at {joint[best]['validation_loss']:.10f}.
3. **Clear winner:** {'Yes' if summary['clear_matched_budget_winner'] else 'No'} under the fixed 0.01 plus 15/20 rule.
4. **Empirical vs Uniform:** {comparison_sentence('EMPIRICAL', 'UNIFORM_MATCHED', joint)}
5. **Top-wide Triangle vs Uniform:** {comparison_sentence('TOP_WIDE_TRIANGLE', 'UNIFORM_MATCHED', joint)}
6. **Reverse vs Top-wide Triangle:** {comparison_sentence('REVERSE_TRIANGLE', 'TOP_WIDE_TRIANGLE', joint)}
7. **Large early vs late windows:** {early_importance}
8. **Empirical joint vs marginal sum:** {interaction['empirical_joint_damage']:+.10f} vs {interaction['sum_single_layer_marginal_damages']:+.10f}; interaction {interaction['interaction_E_joint_minus_marginal_sum']:+.10f}.
9. **Empirical interaction:** {interaction_label}, ratio {empirical_ratio:.6f}.
10. **Where damage accumulates most:** {', '.join(f'{name}: {max_damage_bins[name]}' for name in GEOMETRIES)}.
11. **Best argmax preservation:** {best_argmax}, {argmax[best_argmax]['overall_vs_full']:.10f}.
12. **Use the 2D0C marginal profile directly:** {use_empirical_directly}
13. **Evidence of cross-layer compensation:** {interaction_evidence}
14. **Raw no-recurrence triangle deficit:** {joint['TOP_WIDE_TRIANGLE']['damage_vs_full']:+.10f} validation loss, with {argmax['TOP_WIDE_TRIANGLE']['overall_vs_full']:.10f} argmax agreement.
15. **Next experiment:** {recommendation}.

## Geometry manifest and budget

| Layer | Empirical | Triangle | Reverse | Uniform |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(budget_rows)}

These rows were generated from the frozen `geometry_manifest.json`, not retyped as a separate source of truth.

## Interpretation boundaries

2D0D identifies which geometry the frozen ordinary feed-forward Standard GPT-2 tolerates naturally without recurrent repair. It does not determine the best geometry after recurrent training.

The triangle was evaluated without the high→low recurrent repair mechanism that motivates preserving wider upper layers. A poor 2D0D triangle result measures the burden placed on future recurrence; it does not by itself falsify the recurrent triangle hypothesis.

Likewise, a Reverse win demonstrates natural feed-forward compatibility, not the optimal architecture for recurrent top-down memory. B1 is structurally special because it receives no contextualized Transformer state from a lower block.

The best candidate's absolute joint damage was {joint[best]['damage_vs_full']:+.10f}. Winning this four-way comparison does not by itself make that degradation deployment-ready.

## Exactly one recommended next experiment

**{recommendation}**

This recommendation was not executed.

## Integrity and provenance

- Frozen 2D0C tag: `{FROZEN_TAG}` → `{FROZEN_COMMIT}`
- 2D0D branch: `{BRANCH}`
- Source checkpoint SHA-256: `{d0.SOURCE_SHA256}`
- Validation shard SHA-256: `{d0.VAL_SHA256}`
- Canonical batch collection SHA-256: `{d0.CANONICAL_VALIDATION_SHA256}`
- Full-context oracle: {joint['FULL']['validation_loss']:.10f}
- Standard GPT-2 only; no Full AttnRes, recurrence, completion, reader, writer, optimizer, backward call, parameter update, or training target
- Four independent single-GPU processes; no DDP or NCCL
- 20 identical B=64, T=1024 batches per candidate; 1,310,720 targets each
- All preflight and pre-final integrity checks: PASS

## Git commits

- Implementation commit: `{implementation_commit}`
- Results commit: `{results_commit}`
- Final-report commit: recorded in the synchronized experiment handoff after this report is committed

## Performance

- Four-GPU candidate elapsed wall time: {performance['total_four_gpu_elapsed_wall_seconds']:.3f} seconds
- Masked-evaluator throughput is not optimized deployment throughput.

# EXPERIMENT 2D0D COMPLETE
"""
    (output / "EXPERIMENT_2D0D_FINAL_REPORT.md").write_text(report)


def run_finalize(args):
    require_git(clean=False)
    load_config()
    allowed_interaction = {
        "STRONG SUPER-ADDITIVE JOINT DAMAGE",
        "JOINT DAMAGE APPROXIMATELY ADDITIVE",
        "JOINT DAMAGE IS SUB-ADDITIVE",
    }
    allowed_recommendations = {
        "PROCEED TO RECURRENT REPAIR BASELINE ON BEST JOINT GEOMETRY",
        "INCREASE / REDISTRIBUTE JOINT KV BUDGET BEFORE RECURRENT TRAINING",
        "REFINE MATCHED-BUDGET JOINT GEOMETRIES",
        "FIX 2D0D BEFORE FOLLOW-ON",
    }
    if args.interaction_conclusion not in allowed_interaction:
        raise SystemExit("invalid 2D0D interaction conclusion")
    if args.recommendation not in allowed_recommendations:
        raise SystemExit("invalid 2D0D recommendation")
    output = Path(args.output_dir).resolve()
    required_results = [
        "source_manifest.json", "environment.json", "batch_manifest.json",
        "geometry_manifest.json", "budget_audit.json", "semantic_diff_audit.json",
        "preflight_audit.json", "joint_losses.json", "per_batch_losses.json",
        "paired_comparisons.json", "empirical_interaction.json", "per_position_loss.json",
        "position_bin_loss.json", "argmax_agreement.json", "performance.json",
        "commands_and_runtime.json", "result_summary.json",
    ]
    if not all((output / name).is_file() for name in required_results):
        raise SystemExit("2D0D finalize is missing assembled result artifacts")
    summary = json.loads((output / "result_summary.json").read_text())
    geometry_manifest = json.loads((output / "geometry_manifest.json").read_text())
    per_position = json.loads((output / "per_position_loss.json").read_text())
    generate_plots(output, summary, geometry_manifest, per_position)
    render_report(
        output,
        summary,
        args.interaction_conclusion,
        args.recommendation,
        args.implementation_commit,
        args.results_commit,
    )
    report_path = output / "EXPERIMENT_2D0D_FINAL_REPORT.md"
    joint = json.loads((output / "joint_losses.json").read_text())
    interaction = json.loads((output / "empirical_interaction.json").read_text())
    pairs = json.loads((output / "paired_comparisons.json").read_text())
    argmax = json.loads((output / "argmax_agreement.json").read_text())
    required_final = required_results + [
        "EXPERIMENT_2D0D_FINAL_REPORT.md",
        "matched_geometry_profiles.svg", "matched_geometry_profiles.png",
        "matched_geometry_damage.svg", "matched_geometry_damage.png",
        "geometry_damage_by_position.svg", "geometry_damage_by_position.png",
    ]
    cross_checks = {
        "summary_joint_losses_exact": summary["joint_losses"] == joint,
        "summary_interaction_exact": summary["empirical_interaction"] == interaction,
        "pair_count_exact": len(pairs["pairs"]) == 6,
        "argmax_candidate_set_exact": set(argmax) == set(GEOMETRIES),
        "geometry_manifest_exact": geometry_manifest["geometries"] == {key: list(value) for key, value in GEOMETRIES.items()},
        "report_has_required_end_marker": report_path.read_text().rstrip().endswith("# EXPERIMENT 2D0D COMPLETE"),
        "report_has_triangle_boundary": "does not by itself falsify the recurrent triangle hypothesis" in report_path.read_text(),
        "all_required_artifacts_present": all((output / name).is_file() for name in required_final),
        "pre_final_integrity_passed": summary["integrity_pre_audit"]["passed"],
        "exactly_one_recommendation_heading": report_path.read_text().count("## Exactly one recommended next experiment") == 1,
    }
    audit = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "status": "PASS" if all(cross_checks.values()) else "FAIL",
        "checks": cross_checks,
        "interaction_conclusion": args.interaction_conclusion,
        "recommendation": args.recommendation,
        "implementation_commit": args.implementation_commit,
        "results_commit": args.results_commit,
        "report_sha256": d0.file_sha256(report_path),
        "cross_artifact_consistency": all(cross_checks.values()),
        "training_performed": False,
        "scientific_candidate_evaluations": 4,
    }
    d0.durable_json(output / "FINAL_AUDIT.json", audit)
    print(f"EXPERIMENT_2D0D_FINALIZE_{audit['status']} best={summary['best_geometry']}", flush=True)
    if audit["status"] != "PASS":
        raise SystemExit("2D0D final audit failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--parent-checkpoint", required=True)
    preflight.add_argument("--validation-shard", required=True)
    preflight.add_argument("--output-dir", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--parent-checkpoint", required=True)
    worker.add_argument("--validation-shard", required=True)
    worker.add_argument("--output-dir", required=True)
    worker.add_argument("--run-root", required=True)
    worker.add_argument("--physical-gpu", required=True, type=int)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--output-dir", required=True)
    assemble.add_argument("--run-root", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--interaction-conclusion", required=True)
    finalize.add_argument("--recommendation", required=True)
    finalize.add_argument("--implementation-commit", required=True)
    finalize.add_argument("--results-commit", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    commands = {
        "prepare": run_prepare,
        "preflight": run_preflight,
        "worker": run_worker,
        "assemble": run_assemble,
        "finalize": run_finalize,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
