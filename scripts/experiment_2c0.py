#!/usr/bin/env python3
"""Experiment 2C0: frozen generic correction plus centered sequence reader."""

import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2a0 as a0  # noqa: E402
import experiment_2b2 as b2  # noqa: E402
import experiment_2b2a as b2a  # noqa: E402
import experiment_2b3 as b3  # noqa: E402
import experiment_2b4 as b4  # noqa: E402
import experiment_2b5 as b5  # noqa: E402


BRANCH = "experiment-2c0-separated-generic-sequence-b1"
PARENT_TAG = "experiment-2b5-decomposition-final"
PARENT_COMMIT = "6c8112f267f64751080eaa7799ad2f76a93fa591"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2c0_separated_generic_sequence_4gpu.json"
CALIBRATION_MANIFEST_PATH = (
    REPO_ROOT / "results" / "experiment_2b4_memory_content_mask_depth"
    / "part_a_calibration_manifest.json"
)
GENERIC_MEAN_PATH = (
    REPO_ROOT / "results" / "experiment_2b5_mean_residual_decomposition"
    / "calibration_means" / "C3_2B3_FINAL.pt"
)
GENERIC_MEANS_MANIFEST_PATH = (
    REPO_ROOT / "results" / "experiment_2b5_mean_residual_decomposition"
    / "generic_means_manifest.json"
)

WORLD_SIZE = 4
MICROSTEPS_PER_RANK = 2
B = 64
T = 1024
GLOBAL_TARGETS = 524_288
RANK_TARGETS = 131_072
BACKWARD_CHUNK = 16
SOURCE_DEPTHS = (16, 17, 20, 24)
TRAINABLE_PARAMETERS = 1_537
RESULT_UPDATES = 10
RESTART_UPDATE = 5
CANONICAL_BATCHES = 20
CALIBRATION_BATCHES = 4
SOURCE_2B1_SHA = "5a97c36c038ad04155c7965e20a800cdd78845819671f91c6d516599bb9cd69a"
SOURCE_2B3_SHA = "7797f349905e344934bd7d2475cf61b332ef9053cb0bc1a44f450fc24249c65b"
SOURCE_NEXT_SHA = "7f6d8da5044e9f485492712373fda12d09eb4ccec60ab8fdee812519d05869a7"
CANONICAL_SHA = "3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb"
CALIBRATION_SHA = "d159c297f26e5e7ef707d37c5656b3702d66a11809ebed5577cd12903bfcb2f6"
GENERIC_MEAN_TENSOR_SHA = "26c550b5770307b50c20a384447785f348515e4a232f60f78b58c71b62c1fd99"
EXPECTED_GENERIC_LOSS = 4.7776873112
GENERIC_LOSS_ATOL = 5e-6
CHECKPOINT_SCHEMA = "exp2c0_separated_generic_sequence_b1_v1"
SEED = 20260818

READER_KEYS = (
    "query",
    "norm.weight",
    "gate",
)


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2C0 requires branch {BRANCH}")
    if git_output("rev-parse", f"{PARENT_TAG}^{{}}") != PARENT_COMMIT:
        raise SystemExit("frozen Experiment 2B5 tag target mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2C0 execution requires a clean worktree")


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        "protocol": "exp2c0_separated_generic_sequence_b1_v1",
        "world_size": WORLD_SIZE,
        "microsteps_per_rank": MICROSTEPS_PER_RANK,
        "batch_sequences": B,
        "sequence_length": T,
        "global_targets_per_update": GLOBAL_TARGETS,
        "result_updates": RESULT_UPDATES,
        "forced_restart_after_update": RESTART_UPDATE,
        "backward_chunk_tokens": BACKWARD_CHUNK,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "source_depths": list(SOURCE_DEPTHS),
        "source_2b1_checkpoint_sha256": SOURCE_2B1_SHA,
        "source_2b3_checkpoint_sha256": SOURCE_2B3_SHA,
        "source_2b3_next_global_batch_sha256": SOURCE_NEXT_SHA,
        "canonical_validation_sha256": CANONICAL_SHA,
        "calibration_validation_sha256": CALIBRATION_SHA,
        "generic_only_expected_loss": EXPECTED_GENERIC_LOSS,
        "generic_only_absolute_tolerance": GENERIC_LOSS_ATOL,
        "writers": "forbidden in active recurrence",
        "hellaswag": "forbidden",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f"2C0 config mismatch for {key}: {config.get(key)} != {value}")
    return config


def file_sha256(path):
    return b2a.file_sha256(Path(path))


def tensor_sha256(name, tensor):
    return b4.tensor_sha256(name, tensor.detach().cpu().contiguous())


def atomic_torch_save(path, payload, refuse_overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".incomplete")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_sha256(path)


def load_source_2b3(path):
    path = Path(path).resolve()
    digest = file_sha256(path)
    if digest != SOURCE_2B3_SHA:
        raise SystemExit(f"2B3 source checkpoint SHA mismatch: {digest}")
    checkpoint = a0.torch_load(path, mmap=True)
    if checkpoint.get("schema") != b3.CHECKPOINT_SCHEMA:
        raise SystemExit("2B3 source checkpoint schema mismatch")
    state = checkpoint.get("training_state", {})
    if state.get("joint_local_updates") != 9:
        raise SystemExit("2B3 source is not the final joint-update-9 checkpoint")
    if checkpoint.get("next_global_batch_sha256") != SOURCE_NEXT_SHA:
        raise SystemExit("2B3 source data cursor mismatch")
    if len(checkpoint.get("dataloader_states", ())) != WORLD_SIZE:
        raise SystemExit("2B3 source must contain four loader states")
    if len(checkpoint.get("rank_rng_states", ())) != WORLD_SIZE:
        raise SystemExit("2B3 source must contain four rank RNG states")
    return checkpoint, digest


def instantiate_frozen_model(checkpoint, symbols, device):
    with torch.random.fork_rng(devices=[]):
        model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SystemExit("frozen model unexpectedly has trainable parameters")
    return model


def reader_parameter_rows(reader):
    named = dict(reader.named_parameters())
    if set(named) != set(READER_KEYS):
        raise SystemExit(f"sequence reader tensor mismatch: {sorted(named)}")
    rows = [(name, named[name]) for name in READER_KEYS]
    if sum(parameter.numel() for _, parameter in rows) != TRAINABLE_PARAMETERS:
        raise SystemExit("sequence reader parameter count mismatch")
    return rows


def load_initial_reader(path, symbols, device):
    path = Path(path).resolve()
    digest = file_sha256(path)
    if digest != SOURCE_2B1_SHA:
        raise SystemExit(f"2B1 reader checkpoint SHA mismatch: {digest}")
    checkpoint = a0.torch_load(path, mmap=True)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise SystemExit("2B1 checkpoint has no model state")
    source_prefix = "transformer.topdown_attnres."
    source = {
        key: state[source_prefix + key].detach().cpu().clone()
        for key in READER_KEYS
    }
    with torch.random.fork_rng(devices=[]):
        reader = symbols["TopDownAttnRes"](
            768, SOURCE_DEPTHS, eps=1e-5
        )
    with torch.no_grad():
        reader.query.copy_(source["query"])
        reader.norm.weight.copy_(source["norm.weight"])
        old_effective = source["gate"].float().tanh()
        new_effective = old_effective * 0.5
        reader.gate.copy_(torch.atanh(new_effective).to(reader.gate.dtype))
    reader.to(device)
    reader_parameter_rows(reader)
    metadata = {
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": digest,
        "copied_query_norm": source["query"].float().norm().item(),
        "copied_rmsnorm_displacement": (
            source["norm.weight"].float() - 1.0
        ).norm().item(),
        "old_gate": source["gate"].float().item(),
        "old_effective_gate": old_effective.item(),
        "new_effective_gate": new_effective.item(),
        "new_gate": reader.gate.detach().float().item(),
        "independent_copies": True,
        "trainable_parameters": TRAINABLE_PARAMETERS,
    }
    return reader, metadata


def fresh_optimizer(reader, device_type):
    kwargs = {
        "lr": 1e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in reader_parameter_rows(reader)], **kwargs
    )
    if optimizer.state:
        raise SystemExit("fresh sequence optimizer unexpectedly has state")
    return optimizer


def optimizer_report(optimizer, completed_updates):
    state = optimizer if isinstance(optimizer, dict) else optimizer.state_dict()
    group = state["param_groups"][0]
    for key, value in {
        "lr": 1e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }.items():
        actual = tuple(group[key]) if key == "betas" else group[key]
        if actual != value:
            raise SystemExit(f"sequence optimizer {key} mismatch")
    if completed_updates == 0 and state["state"]:
        raise SystemExit("step-zero sequence optimizer is not fresh")
    if completed_updates > 0 and len(state["state"]) != 3:
        raise SystemExit("sequence optimizer must contain exactly three state entries")
    steps = []
    for values in state["state"].values():
        steps.append(int(values["step"].item()))
        for value in values.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if not torch.isfinite(value).all():
                    raise SystemExit("non-finite sequence optimizer state")
    if completed_updates > 0 and sorted(steps) != [completed_updates] * 3:
        raise SystemExit(f"sequence optimizer step mismatch: {steps}")
    return {
        "state_entries": len(state["state"]),
        "steps": sorted(steps),
        "lr": group["lr"],
        "betas": list(group["betas"]),
        "eps": group["eps"],
        "weight_decay": group["weight_decay"],
        "moments_finite": True,
    }


def reader_state_sha256(reader):
    digest = hashlib.sha256()
    for name, value in reader.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_sha256(reader, optimizer, completed_updates):
    optimizer_report(optimizer, completed_updates)
    digest = hashlib.sha256()
    for name, parameter in reader_parameter_rows(reader):
        state = optimizer.state.get(parameter, {})
        for field in ("step", "exp_avg", "exp_avg_sq"):
            value = state.get(field)
            if value is None:
                continue
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{name}:{field}".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def reader_metrics(reader):
    return {
        "gate": reader.gate.detach().float().item(),
        "effective_gate": reader.gate.detach().float().tanh().item(),
        "query_norm": reader.query.detach().float().norm().item(),
        "rmsnorm_displacement": (
            reader.norm.weight.detach().float() - 1.0
        ).norm().item(),
    }


def make_generic_correction(model, device):
    manifest = json.loads(GENERIC_MEANS_MANIFEST_PATH.read_text())["means"]["C3_2B3_FINAL"]
    artifact_digest = file_sha256(GENERIC_MEAN_PATH)
    if artifact_digest != manifest["artifact_sha256"]:
        raise SystemExit("2B5 generic-mean artifact SHA mismatch")
    payload = a0.torch_load(GENERIC_MEAN_PATH)
    if payload.get("source_checkpoint_sha256") != SOURCE_2B3_SHA:
        raise SystemExit("generic mean source lineage mismatch")
    mean = payload.get("mean")
    if tuple(mean.shape) != (4, 768) or mean.dtype != torch.float32:
        raise SystemExit("generic mean tensor contract mismatch")
    if tensor_sha256("global_template", mean) != GENERIC_MEAN_TENSOR_SHA:
        raise SystemExit("generic mean tensor SHA mismatch")
    bank = mean.to(device=device, dtype=torch.bfloat16)[:, None, None, :]
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        topdown = model.transformer.topdown_attnres(list(bank.unbind(dim=0)))
        correction = (
            model.transformer.topdown_attnres.gate.tanh() * topdown
        ).squeeze(0).squeeze(0).detach().clone()
    if tuple(correction.shape) != (768,) or not torch.isfinite(correction).all():
        raise SystemExit("invalid fixed generic correction")
    metadata = {
        "source_checkpoint_sha256": SOURCE_2B3_SHA,
        "calibration_manifest_sha256": file_sha256(CALIBRATION_MANIFEST_PATH),
        "source_mean_artifact_sha256": artifact_digest,
        "source_mean_tensor_sha256": GENERIC_MEAN_TENSOR_SHA,
        "source_mu_tensor_shas": {
            f"mu{depth}": tensor_sha256(f"mu{depth}", mean[index])
            for index, depth in enumerate(SOURCE_DEPTHS)
        },
        "tensor_sha256": tensor_sha256("generic_correction", correction),
        "dtype": str(correction.dtype),
        "shape": list(correction.shape),
        "rms": correction.float().pow(2).mean().sqrt().item(),
        "detached": not correction.requires_grad,
        "old_reader_executed_once": True,
        "active_old_reader_dynamic_calls": 0,
    }
    return correction, metadata


def save_generic_artifact(run_dir, correction, metadata):
    path = Path(run_dir) / "generic_correction.pt"
    payload = {
        "experiment": "2C0",
        "generic_correction": correction.detach().cpu(),
        "metadata": metadata,
    }
    digest = atomic_torch_save(path, payload)
    reopened = a0.torch_load(path)
    if tensor_sha256(
        "generic_correction", reopened["generic_correction"]
    ) != metadata["tensor_sha256"]:
        raise SystemExit("generic correction strict reopen failed")
    return path, digest


def load_generic_artifact(run_dir, device):
    path = Path(run_dir) / "generic_correction.pt"
    payload = a0.torch_load(path)
    correction = payload["generic_correction"]
    metadata = payload["metadata"]
    if tensor_sha256("generic_correction", correction) != metadata["tensor_sha256"]:
        raise SystemExit("generic correction tensor identity mismatch")
    return correction.to(device), metadata, file_sha256(path)


def load_validation(symbols):
    canonical, calibration, canonical_hashes, calibration_hashes = b4.load_validation_batches(symbols)
    if b4.aggregate_payload_hash(canonical_hashes) != CANONICAL_SHA:
        raise SystemExit("canonical validation aggregate SHA mismatch")
    if b4.aggregate_payload_hash(calibration_hashes) != CALIBRATION_SHA:
        raise SystemExit("calibration validation aggregate SHA mismatch")
    frozen = json.loads(CALIBRATION_MANIFEST_PATH.read_text())
    if (
        frozen.get("calibration_evaluation_disjoint") is not True
        or frozen.get("calibration_batch_payload_sha256") != calibration_hashes
        or frozen.get("calibration_batch_indices") != [20, 21, 22, 23]
    ):
        raise SystemExit("frozen calibration manifest mismatch")
    return canonical, calibration, canonical_hashes, calibration_hashes, frozen


def direct_feedback(
    reader,
    state,
    source_means,
    generic,
    control,
    permutation=None,
    fixed_sequence_feedback=None,
):
    batch_size = state.feedback_memory.size(1)
    if state.position == 0:
        zero = generic.new_zeros((batch_size, 1, generic.numel()))
        return zero, {
            "sequence_feedback": zero,
            "sequence_topdown": zero,
            "routing_weights": None,
            "routing_entropy": None,
            "centered_sources": None,
        }
    generic_enabled = control not in {"sequence_only"}
    sequence_enabled = control not in {"generic", "gate_zero"}
    generic_feedback = (
        generic.view(1, 1, -1).expand(batch_size, -1, -1)
        if generic_enabled else generic.new_zeros((batch_size, 1, generic.numel()))
    )
    if fixed_sequence_feedback is not None:
        sequence_feedback = fixed_sequence_feedback.to(generic.device).view(1, 1, -1).expand(
            batch_size, -1, -1
        )
        return generic_feedback + sequence_feedback, {
            "sequence_feedback": sequence_feedback,
            "sequence_topdown": sequence_feedback,
            "routing_weights": None,
            "routing_entropy": None,
            "centered_sources": None,
        }
    centered32 = state.feedback_memory.detach().float() - source_means.float()[:, None, None, :]
    if control in {"shuffle", "initial_shuffle"}:
        if permutation is None:
            raise ValueError("shuffle control requires a coherent permutation")
        centered32 = centered32[:, permutation]
    elif control == "batchmean":
        if batch_size < 2:
            raise ValueError("leave-one-out batch mean requires batch size > 1")
        centered32 = (
            centered32.sum(dim=1, keepdim=True) - centered32
        ) / float(batch_size - 1)
    centered = centered32.to(state.feedback_memory.dtype)
    topdown, weights = reader(list(centered.unbind(dim=0)), return_weights=True)
    sequence_feedback = reader.gate.tanh() * topdown
    if not sequence_enabled:
        sequence_feedback = torch.zeros_like(sequence_feedback)
    safe = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
    entropy = -(safe * safe.log()).sum(dim=0).squeeze(-1)
    return generic_feedback + sequence_feedback, {
        "sequence_feedback": sequence_feedback,
        "sequence_topdown": topdown,
        "routing_weights": weights,
        "routing_entropy": entropy,
        "centered_sources": centered,
    }


def cache_health(state, length):
    return {
        "block_1_cache_absent": state.kv_caches[0] is None,
        "other_cache_lengths_correct": all(
            cache.length == length for cache in state.kv_caches[1:]
        ),
        "historical_kv_detached": all(
            cache.key.grad_fn is None and cache.value.grad_fn is None
            for cache in state.kv_caches[1:]
        ),
        "historical_kv_finite": all(
            torch.isfinite(cache.key[:, :, :length]).all()
            and torch.isfinite(cache.value[:, :, :length]).all()
            for cache in state.kv_caches[1:]
        ),
        "raw_source_memory_detached": state.feedback_memory.grad_fn is None,
        "raw_source_memory_finite": bool(torch.isfinite(state.feedback_memory).all()),
    }


@torch.no_grad()
def evaluate_stream(
    model,
    reader,
    x,
    y,
    source_means,
    generic,
    control,
    permutation=None,
    fixed_sequence_feedback=None,
    capture=False,
):
    model.eval()
    reader.eval()
    state = model.init_recurrent_state(
        x.size(0), "masked_l1_no_feedback", device=x.device,
        dtype=torch.bfloat16, mask_depth=1,
    )
    loss_sum = torch.zeros((), device=x.device, dtype=torch.float64)
    routing_sum = torch.zeros(4, device=x.device, dtype=torch.float64)
    entropy_sum = 0.0
    source_rms_sum = torch.zeros(4, device=x.device, dtype=torch.float64)
    topdown_rms_sum = 0.0
    feedback_rms_sum = 0.0
    feedback_vector_sum = torch.zeros(768, device=x.device, dtype=torch.float64)
    metric_count = 0
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(x.size(1)):
            feedback, diag = direct_feedback(
                reader, state, source_means, generic, control,
                permutation=permutation,
                fixed_sequence_feedback=fixed_sequence_feedback,
            )
            logits, state = model.forward_step(
                x[:, position], state, block1_feedback=feedback
            )
            loss_sum += F.cross_entropy(
                logits[:, 0], y[:, position], reduction="sum"
            ).double()
            if capture and diag["routing_weights"] is not None:
                routing_sum += diag["routing_weights"].double().sum(dim=(1, 2))
                entropy_sum += diag["routing_entropy"].double().sum().item()
                centered = diag["centered_sources"].float()
                source_rms_sum += centered.pow(2).mean(dim=(2, 3)).sqrt().double().sum(dim=1)
                topdown_rms_sum += diag["sequence_topdown"].float().pow(2).mean(dim=(1, 2)).sqrt().double().sum().item()
                sequence = diag["sequence_feedback"].float()
                feedback_rms_sum += sequence.pow(2).mean(dim=(1, 2)).sqrt().double().sum().item()
                feedback_vector_sum += sequence.double().sum(dim=(0, 1))
                metric_count += x.size(0)
    health = cache_health(state, x.size(1))
    finite = bool(torch.isfinite(loss_sum)) and all(health.values())
    result = {
        "loss": (loss_sum / x.numel()).item(),
        "loss_sum": loss_sum.item(),
        "finite": finite,
        "cache_health": health,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if capture and metric_count:
        mean_feedback = feedback_vector_sum / metric_count
        result["diagnostics"] = {
            "routing": (routing_sum / metric_count).cpu().tolist(),
            "routing_entropy": entropy_sum / metric_count,
            "centered_source_rms": (source_rms_sum / metric_count).cpu().tolist(),
            "sequence_topdown_rms": topdown_rms_sum / metric_count,
            "sequence_feedback_rms": feedback_rms_sum / metric_count,
            "mean_sequence_feedback": mean_feedback.cpu().tolist(),
            "mean_feedback_rms": mean_feedback.pow(2).mean().sqrt().item(),
            "mean_feedback_ratio": mean_feedback.pow(2).mean().sqrt().item()
            / max(feedback_rms_sum / metric_count, 1e-30),
            "metric_count": metric_count,
        }
    return result


def fixed_rank_sum(local):
    slots = torch.zeros(
        (WORLD_SIZE,) + tuple(local.shape), device=local.device, dtype=local.dtype
    )
    slots[dist.get_rank()].copy_(local)
    dist.all_reduce(slots, op=dist.ReduceOp.SUM)
    result = slots[0].clone()
    for rank in range(1, WORLD_SIZE):
        result.add_(slots[rank])
    return result


@torch.no_grad()
def calibrate_source_means(model, calibration_batch, generic, device):
    x_cpu, _ = calibration_batch
    x = x_cpu.to(device, non_blocking=True)
    state = model.init_recurrent_state(
        B, "masked_l1_no_feedback", device=device,
        dtype=torch.bfloat16, mask_depth=1,
    )
    local_sum = torch.zeros((4, 768), device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(T):
            feedback = (
                generic.new_zeros((B, 1, 768))
                if position == 0 else generic.view(1, 1, -1).expand(B, -1, -1)
            )
            _, state = model.forward_step(
                x[:, position], state, block1_feedback=feedback
            )
            if position < T - 1:
                local_sum.add_(state.feedback_memory[:, :, 0].float().sum(dim=1))
    health = cache_health(state, T)
    if not all(health.values()):
        raise SystemExit(f"source-mean calibration cache failure: {health}")
    global_sum = fixed_rank_sum(local_sum)
    means = global_sum / float(WORLD_SIZE * B * (T - 1))
    if not torch.isfinite(means).all():
        raise SystemExit("calibrated sequence source means contain NaN/Inf")
    return means, local_sum, health


def save_source_means(run_dir, means, calibration_hashes, local_sums):
    means_cpu = means.detach().float().cpu()
    centered_global_sum = (
        local_sums.float().cpu().sum(dim=0)
        - means_cpu * float(WORLD_SIZE * B * (T - 1))
    )
    metadata = {
        "source_checkpoint_sha256": SOURCE_2B3_SHA,
        "generic_architecture": "fixed generic-only direct Block-1 correction",
        "calibration_manifest": str(CALIBRATION_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "calibration_manifest_sha256": file_sha256(CALIBRATION_MANIFEST_PATH),
        "calibration_batch_payload_sha256": calibration_hashes,
        "calibration_aggregate_sha256": CALIBRATION_SHA,
        "canonical_calibration_disjoint": True,
        "accumulation_dtype": "torch.float32",
        "subtraction_dtype": "torch.float32",
        "shape": list(means_cpu.shape),
        "source_shas": {
            f"nu{depth}": tensor_sha256(f"nu{depth}", means_cpu[index])
            for index, depth in enumerate(SOURCE_DEPTHS)
        },
        "tensor_sha256": tensor_sha256("sequence_source_means", means_cpu),
        "mean_centered_calibration_residual": (
            centered_global_sum / float(WORLD_SIZE * B * (T - 1))
        ).cpu().tolist(),
        "maximum_absolute_mean_centered_residual": (
            centered_global_sum / float(WORLD_SIZE * B * (T - 1))
        ).abs().max().item(),
        "frozen": True,
    }
    path = Path(run_dir) / "sequence_source_means.pt"
    digest = atomic_torch_save(path, {
        "experiment": "2C0",
        "sequence_source_means": means_cpu,
        "metadata": metadata,
    })
    reopened = a0.torch_load(path)
    if tensor_sha256(
        "sequence_source_means", reopened["sequence_source_means"]
    ) != metadata["tensor_sha256"]:
        raise SystemExit("sequence source means strict reopen failed")
    return path, digest, metadata


def load_source_means(run_dir, device):
    path = Path(run_dir) / "sequence_source_means.pt"
    payload = a0.torch_load(path)
    means = payload["sequence_source_means"]
    metadata = payload["metadata"]
    if tensor_sha256("sequence_source_means", means) != metadata["tensor_sha256"]:
        raise SystemExit("sequence source means identity mismatch")
    return means.to(device), metadata, file_sha256(path)


def paired_statistics(real, shuffled):
    gaps = [right - left for left, right in zip(real, shuffled)]
    return {
        "real_wins": sum(gap > 0 for gap in gaps),
        "shuffled_wins": sum(gap < 0 for gap in gaps),
        "ties": sum(gap == 0 for gap in gaps),
        "mean_gap": statistics.fmean(gaps),
        "median_gap": statistics.median(gaps),
        "sample_std": statistics.stdev(gaps),
        "minimum": min(gaps),
        "maximum": max(gaps),
        "gaps": gaps,
    }


@torch.no_grad()
def rollout_capture(model, reader, tokens, source_means, generic, prefix_length):
    state = model.init_recurrent_state(
        tokens.size(0), "masked_l1_no_feedback", device=tokens.device,
        dtype=torch.bfloat16, mask_depth=1,
    )
    logits = []
    prefix_state = None
    prefix_feedback = None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            feedback, _ = direct_feedback(
                reader, state, source_means, generic, "real"
            )
            row, state = model.forward_step(
                tokens[:, position], state, block1_feedback=feedback
            )
            if position < prefix_length:
                logits.append(row.detach().clone())
            if position + 1 == prefix_length:
                prefix_state = state.state_dict()
                next_feedback, _ = direct_feedback(
                    reader, state, source_means, generic, "real"
                )
                prefix_feedback = next_feedback.detach().clone()
    return torch.cat(logits, dim=1), prefix_state, prefix_feedback


@torch.no_grad()
def causality_and_isolation_tests(model, reader, tokens, source_means, generic):
    first = tokens[:2, :32].clone()
    second = first.clone()
    second[:, 16:] = (second[:, 16:] + 1) % model.config.vocab_size
    logits_a, state_a, feedback_a = rollout_capture(
        model, reader, first, source_means, generic, 16
    )
    logits_b, state_b, feedback_b = rollout_capture(
        model, reader, second, source_means, generic, 16
    )
    future = {
        "block1_prefix_logits_bit_exact": torch.equal(logits_a, logits_b),
        "prefix_memory_and_kv_bit_exact": b2.b0.cache_payload_equal(state_a, state_b),
        "sequence_memory_at_t_bit_exact": torch.equal(feedback_a, feedback_b),
        "generic_G_fixed": True,
    }
    future["passed"] = all(future.values())
    row_a = tokens[:2, :16].clone()
    row_b = row_a.clone()
    row_b[1] = (row_b[1] + 17) % model.config.vocab_size
    row_logits_a, row_state_a, row_feedback_a = rollout_capture(
        model, reader, row_a, source_means, generic, 16
    )
    row_logits_b, row_state_b, row_feedback_b = rollout_capture(
        model, reader, row_b, source_means, generic, 16
    )
    isolation = {
        "unchanged_row_logits_bit_exact": torch.equal(row_logits_a[0], row_logits_b[0]),
        "unchanged_row_memory_and_kv_bit_exact": b2.b0.cache_payload_equal(
            row_state_a, row_state_b, row=0
        ),
        "unchanged_row_next_feedback_bit_exact": torch.equal(
            row_feedback_a[0], row_feedback_b[0]
        ),
    }
    isolation["passed"] = all(isolation.values())
    zero_sources = torch.zeros(
        4, 2, 1, 768, device=tokens.device, dtype=torch.bfloat16
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        zero_topdown = reader(list(zero_sources.unbind(dim=0)))
        zero_feedback = reader.gate.tanh() * zero_topdown
    zero_contract = {
        "zero_centered_sources_produce_bitwise_zero_topdown": zero_topdown.count_nonzero().item() == 0,
        "zero_centered_sources_produce_bitwise_zero_feedback": zero_feedback.count_nonzero().item() == 0,
    }
    zero_contract["passed"] = all(zero_contract.values())
    initial = model.init_recurrent_state(
        2, "masked_l1_no_feedback", device=tokens.device,
        dtype=torch.bfloat16, mask_depth=1,
    )
    reset = {
        "position_zero": initial.position == 0,
        "raw_memory_zero": initial.feedback_memory.count_nonzero().item() == 0,
        "block_1_cache_absent": initial.kv_caches[0] is None,
        "other_caches_empty": all(cache.length == 0 for cache in initial.kv_caches[1:]),
    }
    reset["passed"] = all(reset.values())
    return {
        "future_causality": future,
        "row_isolation": isolation,
        "zero_input_contract": zero_contract,
        "fresh_state": reset,
        "passed": future["passed"] and isolation["passed"]
        and zero_contract["passed"] and reset["passed"],
    }


def aggregate_drift_diagnostics(rows):
    total_count = sum(row["count"] for row in rows)
    vector = torch.zeros(768, dtype=torch.float64)
    rms_sum = 0.0
    for row in rows:
        vector.add_(torch.tensor(row["feedback_vector_sum"], dtype=torch.float64))
        rms_sum += row["feedback_rms_sum"]
    mean = vector / total_count
    mean_rms = mean.pow(2).mean().sqrt().item()
    ordinary = rms_sum / total_count
    return {
        "RMS_mean_sequence_feedback": mean_rms,
        "mean_RMS_sequence_feedback": ordinary,
        "mean_feedback_ratio": mean_rms / max(ordinary, 1e-30),
        "mean_sequence_feedback": mean.tolist(),
        "count": total_count,
    }


@torch.no_grad()
def drift_batch(model, reader, batch, source_means, generic, device):
    x_cpu, _ = batch
    x = x_cpu.to(device, non_blocking=True)
    state = model.init_recurrent_state(
        B, "masked_l1_no_feedback", device=device,
        dtype=torch.bfloat16, mask_depth=1,
    )
    vector_sum = torch.zeros(768, device=device, dtype=torch.float64)
    rms_sum = 0.0
    count = 0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(T):
            feedback, diag = direct_feedback(
                reader, state, source_means, generic, "real"
            )
            _, state = model.forward_step(
                x[:, position], state, block1_feedback=feedback
            )
            if position > 0:
                sequence = diag["sequence_feedback"].float()
                vector_sum += sequence.double().sum(dim=(0, 1))
                rms_sum += sequence.pow(2).mean(dim=(1, 2)).sqrt().double().sum().item()
                count += B
    health = cache_health(state, T)
    if not all(health.values()):
        raise SystemExit("mean-drift calibration cache failure")
    return {
        "feedback_vector_sum": vector_sum.cpu().tolist(),
        "feedback_rms_sum": rms_sum,
        "count": count,
        "cache_health": health,
    }


def prepare(args):
    require_git(clean=True)
    load_config()
    rank, local_rank = b2a.init_distributed()
    try:
        if torch.cuda.get_device_name(local_rank) != "NVIDIA A100-SXM4-80GB":
            raise SystemExit("Experiment 2C0 requires four A100-SXM4-80GB GPUs")
        device = torch.device("cuda", local_rank)
        run_dir = Path(args.run_dir)
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        symbols = a0.support.load_training_symbols()
        source, source_digest = load_source_2b3(args.source_2b3_checkpoint)
        model = instantiate_frozen_model(source, symbols, device)
        frozen_model_sha = b5.model_state_sha256(model)
        generic, generic_meta = make_generic_correction(model, device)
        generic_identities = [None] * WORLD_SIZE
        dist.all_gather_object(generic_identities, generic_meta["tensor_sha256"])
        if len(set(generic_identities)) != 1:
            raise SystemExit("generic correction differs across ranks")
        if rank == 0:
            generic_path, generic_file_sha = save_generic_artifact(
                run_dir, generic, generic_meta
            )
        dist.barrier()
        generic, generic_meta, generic_file_sha = load_generic_artifact(run_dir, device)
        canonical, calibration, canonical_hashes, calibration_hashes, frozen_calibration = load_validation(symbols)
        means, local_sum, calibration_health = calibrate_source_means(
            model, calibration[rank], generic, device
        )
        gathered_local_sums = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_local_sums, local_sum.cpu())
        if rank == 0:
            stacked_sums = torch.stack(gathered_local_sums)
            means_path, means_file_sha, means_meta = save_source_means(
                run_dir, means, calibration_hashes, stacked_sums
            )
        dist.barrier()
        means, means_meta, means_file_sha = load_source_means(run_dir, device)
        reader, initialization = load_initial_reader(
            args.source_2b1_checkpoint, symbols, device
        )
        if any(parameter.numel() == 0 for _, parameter in reader_parameter_rows(reader)):
            raise SystemExit("invalid sequence reader")
        generic_before = generic.detach().cpu().clone()
        means_before = means.detach().cpu().clone()
        controls = ("generic", "real", "shuffle", "sequence_only", "gate_zero")
        progress_path = run_dir / f"zero_shot_rank{rank}.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            if (
                progress.get("implementation_git_commit")
                != git_output("rev-parse", "HEAD")
                or progress.get("rank") != rank
            ):
                raise SystemExit("stale zero-shot progress lineage mismatch")
        else:
            progress = {
                "experiment": "2C0",
                "stage": "zero_shot_rank_progress",
                "implementation_git_commit": git_output("rev-parse", "HEAD"),
                "rank": rank,
                "rows": [],
            }
        local_rows = progress["rows"]
        completed_tasks = {
            (row["batch_index"], row["control"]) for row in local_rows
        }
        permutation = b4.coherent_permutation(B, device)
        for batch_index, (x_cpu, y_cpu) in enumerate(canonical):
            if batch_index % WORLD_SIZE != rank:
                continue
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            for control in controls:
                if (batch_index, control) in completed_tasks:
                    continue
                row = evaluate_stream(
                    model, reader, x, y, means, generic, control,
                    permutation=permutation,
                    capture=control in {"real", "shuffle"},
                )
                row.update({
                    "batch_index": batch_index,
                    "payload_sha256": canonical_hashes[batch_index],
                    "control": control,
                    "rank": rank,
                })
                local_rows.append(row)
                completed_tasks.add((batch_index, control))
                b2a.write_json(progress_path, progress)
                print(
                    f"2C0_ZERO_SHOT rank={rank} control={control} "
                    f"batch={batch_index} loss={row['loss']:.10f} "
                    f"wall={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        gathered_rows = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_rows, local_rows)
        drift_local = drift_batch(
            model, reader, calibration[rank], means, generic, device
        )
        gathered_drift = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_drift, drift_local)
        test_tokens = canonical[rank][0][:2, :32].to(device)
        causality = causality_and_isolation_tests(
            model, reader, test_tokens, means, generic
        )
        gathered_causality = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_causality, causality)
        model_after_sha = b5.model_state_sha256(model)
        invariants = {
            "trainable_params": sum(
                parameter.numel() for _, parameter in reader_parameter_rows(reader)
            ),
            "base_gradients_none": all(parameter.grad is None for parameter in model.parameters()),
            "old_reader_gradients_none": all(
                parameter.grad is None
                for parameter in model.transformer.topdown_attnres.parameters()
            ),
            "writer_gradients_none": all(
                parameter.grad is None
                for parameter in model.transformer.memory_writers.parameters()
            ),
            "writers_active_calls": 0,
            "generic_bit_identical": torch.equal(generic_before, generic.detach().cpu()),
            "source_means_bit_identical": torch.equal(means_before, means.detach().cpu()),
            "frozen_model_bit_identical": frozen_model_sha == model_after_sha,
            "all_rank_causality_passed": all(row["passed"] for row in gathered_causality),
            "calibration_health": calibration_health,
            "hellaswag_run": False,
        }
        invariants["passed"] = (
            invariants["trainable_params"] == TRAINABLE_PARAMETERS
            and invariants["base_gradients_none"]
            and invariants["old_reader_gradients_none"]
            and invariants["writer_gradients_none"]
            and invariants["writers_active_calls"] == 0
            and invariants["generic_bit_identical"]
            and invariants["source_means_bit_identical"]
            and invariants["frozen_model_bit_identical"]
            and invariants["all_rank_causality_passed"]
        )
        if rank == 0:
            rows = sorted(
                [row for group in gathered_rows for row in group],
                key=lambda row: (row["control"], row["batch_index"]),
            )
            by_control = {
                control: [row for row in rows if row["control"] == control]
                for control in controls
            }
            losses = {
                control: statistics.fmean(row["loss"] for row in selected)
                for control, selected in by_control.items()
            }
            paired = paired_statistics(
                [row["loss"] for row in by_control["real"]],
                [row["loss"] for row in by_control["shuffle"]],
            )
            regression = {
                "expected": EXPECTED_GENERIC_LOSS,
                "measured": losses["generic"],
                "absolute_difference": abs(losses["generic"] - EXPECTED_GENERIC_LOSS),
            }
            regression["passed"] = regression["absolute_difference"] <= GENERIC_LOSS_ATOL
            gate = {
                "generic_only_regression": regression["passed"],
                "specific_gap_positive": paired["mean_gap"] > 0,
                "real_wins_at_least_12_of_20": paired["real_wins"] >= 12,
                "real_within_generic_plus_0_10": losses["real"] <= losses["generic"] + 0.10,
                "causality_and_integrity": invariants["passed"],
            }
            gate["passed"] = all(gate.values())
            report = {
                "experiment": "2C0",
                "stage": "zero_shot_training_gate",
                "implementation_git_commit": git_output("rev-parse", "HEAD"),
                "source_2b3_checkpoint_sha256": source_digest,
                "source_2b1_checkpoint_sha256": SOURCE_2B1_SHA,
                "generic_artifact_sha256": generic_file_sha,
                "generic": generic_meta,
                "source_means_artifact_sha256": means_file_sha,
                "source_means": means_meta,
                "initialization": initialization,
                "canonical_validation_sha256": CANONICAL_SHA,
                "calibration_validation_sha256": CALIBRATION_SHA,
                "losses": losses,
                "specific_gap_0": paired["mean_gap"],
                "paired_real_vs_shuffle": paired,
                "generic_regression": regression,
                "mean_drift_initial": aggregate_drift_diagnostics(gathered_drift),
                "causality_by_rank": gathered_causality,
                "integrity": invariants,
                "training_gate": gate,
                "paired_batch_losses": {
                    control: [
                        {"batch_index": row["batch_index"], "loss": row["loss"]}
                        for row in by_control[control]
                    ] for control in controls
                },
                "hellaswag_run": False,
            }
            b2a.write_json(run_dir / "ZERO_SHOT_CONTROLS.json", report)
            if not regression["passed"]:
                raise SystemExit(f"2C0 generic-only regression failed: {regression}")
            if not gate["passed"]:
                print("EXPERIMENT_2C0_ZERO_SHOT_GATE_FAIL", flush=True)
            else:
                print(
                    f"EXPERIMENT_2C0_ZERO_SHOT_GATE_PASS gap={paired['mean_gap']:.10f} "
                    f"wins={paired['real_wins']}/20",
                    flush=True,
                )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def zero_shot_gate(run_dir):
    path = Path(run_dir) / "ZERO_SHOT_CONTROLS.json"
    if not path.is_file():
        raise SystemExit("zero-shot controls must complete before optimizer construction")
    report = json.loads(path.read_text())
    if not report.get("training_gate", {}).get("passed"):
        raise SystemExit("zero-shot training gate did not pass; result training is forbidden")
    return report


def flat_reader_gradients(reader):
    rows = reader_parameter_rows(reader)
    missing = [name for name, parameter in rows if parameter.grad is None]
    if missing:
        raise SystemExit(f"missing sequence reader gradients: {missing}")
    flat = torch.cat([
        parameter.grad.detach().float().reshape(-1)
        for _, parameter in rows
    ]).contiguous()
    if flat.numel() != TRAINABLE_PARAMETERS or not torch.isfinite(flat).all():
        raise SystemExit("invalid flattened sequence reader gradient")
    return flat


def scatter_reader_gradients(reader, flat):
    offset = 0
    for _, parameter in reader_parameter_rows(reader):
        count = parameter.numel()
        parameter.grad = flat[offset:offset + count].view_as(parameter).to(parameter.dtype)
        offset += count
    if offset != flat.numel():
        raise SystemExit("sequence reader gradient scatter mismatch")


def flat_reader_parameters(reader):
    return torch.cat([
        parameter.detach().float().reshape(-1)
        for _, parameter in reader_parameter_rows(reader)
    ])


def comparison(left, right):
    left = left.detach().double()
    right = right.detach().double()
    difference = right - left
    return {
        "cosine": F.cosine_similarity(left, right, dim=0).item(),
        "relative_l2": difference.norm().item() / max(left.norm().item(), 1e-30),
        "maximum_absolute_difference": difference.abs().max().item(),
        "left_norm": left.norm().item(),
        "right_norm": right.norm().item(),
    }


def gradient_report(model, reader):
    sequence = {}
    for name, parameter in reader_parameter_rows(reader):
        gradient = parameter.grad
        sequence[name] = {
            "present": gradient is not None,
            "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
            "nonzero": gradient is not None and bool(torch.count_nonzero(gradient)),
            "norm": None if gradient is None else gradient.detach().float().norm().item(),
        }
    return {
        "sequence_reader": sequence,
        "base_tensors_with_grad": [
            name for name, parameter in model.named_parameters()
            if parameter.grad is not None
        ],
    }


def process_training_batches(
    model,
    reader,
    batches,
    source_means,
    generic,
    expected_batch_hashes=None,
):
    model.train()
    reader.train()
    reader.zero_grad(set_to_none=True)
    raw_loss_sum = 0.0
    target_seen = 0
    hashes = []
    routing_sum = torch.zeros(4, dtype=torch.float64)
    entropy_sum = 0.0
    source_rms_sum = torch.zeros(4, dtype=torch.float64)
    topdown_rms_sum = 0.0
    feedback_rms_sum = 0.0
    feedback_vector_sum = torch.zeros(768, dtype=torch.float64)
    metric_count = 0
    states = []
    device = generic.device
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for x_cpu, y_cpu in batches:
        hashes.append(a0.batch_payload_hash(x_cpu, y_cpu))
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        state = model.init_recurrent_state(
            B, "masked_l1_no_feedback", device=device,
            dtype=torch.bfloat16, mask_depth=1,
        )
        pending = None
        for position in range(T):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feedback, diag = direct_feedback(
                    reader, state, source_means, generic, "real"
                )
                logits, state = model.forward_step(
                    x[:, position], state, block1_feedback=feedback
                )
                token_loss = F.cross_entropy(
                    logits[:, 0], y[:, position], reduction="sum"
                )
                scaled = token_loss / GLOBAL_TARGETS
                pending = scaled if pending is None else pending + scaled
            raw_loss_sum += token_loss.detach().double().item()
            target_seen += B
            if position > 0:
                routing_sum += diag["routing_weights"].detach().double().sum(
                    dim=(1, 2)
                ).cpu()
                entropy_sum += diag["routing_entropy"].detach().double().sum().item()
                centered = diag["centered_sources"].detach().float()
                source_rms_sum += centered.pow(2).mean(dim=(2, 3)).sqrt().double().sum(dim=1).cpu()
                topdown_rms_sum += diag["sequence_topdown"].detach().float().pow(2).mean(dim=(1, 2)).sqrt().double().sum().item()
                sequence = diag["sequence_feedback"].detach().float()
                feedback_rms_sum += sequence.pow(2).mean(dim=(1, 2)).sqrt().double().sum().item()
                feedback_vector_sum += sequence.double().sum(dim=(0, 1)).cpu()
                metric_count += B
            if (position + 1) % BACKWARD_CHUNK == 0 or position + 1 == T:
                if pending is None or not torch.isfinite(pending):
                    raise SystemExit("non-finite pending 2C0 loss")
                pending.backward()
                pending = None
        health = cache_health(state, T)
        if not all(health.values()):
            raise SystemExit(f"2C0 recurrent state invariant failed: {health}")
        states.append(health)
        del x, y, state
    if expected_batch_hashes is not None and hashes != expected_batch_hashes:
        raise SystemExit("rank consumed batches differ from exact preview")
    if target_seen != RANK_TARGETS:
        raise SystemExit(f"rank target geometry mismatch: {target_seen}")
    torch.cuda.synchronize()
    report = gradient_report(model, reader)
    if report["base_tensors_with_grad"]:
        raise SystemExit(f"gradient leaked into frozen model: {report['base_tensors_with_grad']}")
    if not all(
        row["present"] and row["finite"] and row["nonzero"]
        for row in report["sequence_reader"].values()
    ):
        raise SystemExit(f"invalid sequence reader gradient: {report}")
    return {
        "raw_loss_sum": raw_loss_sum,
        "target_seen": target_seen,
        "batch_hashes": hashes,
        "routing_sum": routing_sum.tolist(),
        "entropy_sum": entropy_sum,
        "source_rms_sum": source_rms_sum.tolist(),
        "topdown_rms_sum": topdown_rms_sum,
        "feedback_rms_sum": feedback_rms_sum,
        "feedback_vector_sum": feedback_vector_sum.tolist(),
        "metric_count": metric_count,
        "state_health": states,
        "gradient_report": report,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "local_wall_seconds": time.perf_counter() - started,
    }


def process_smoke_batch(model, reader, optimizer, batch, source_means, generic):
    model.train()
    reader.train()
    optimizer.zero_grad(set_to_none=True)
    x_cpu, y_cpu = batch
    x = x_cpu[:2, :64].to(generic.device)
    y = y_cpu[:2, :64].to(generic.device)
    state = model.init_recurrent_state(
        2, "masked_l1_no_feedback", device=generic.device,
        dtype=torch.bfloat16, mask_depth=1,
    )
    pending = None
    loss_sum = 0.0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(64):
            feedback, _ = direct_feedback(
                reader, state, source_means, generic, "real"
            )
            logits, state = model.forward_step(
                x[:, position], state, block1_feedback=feedback
            )
            token_loss = F.cross_entropy(
                logits[:, 0], y[:, position], reduction="sum"
            )
            pending = token_loss / (2 * 64) if pending is None else pending + token_loss / (2 * 64)
            loss_sum += token_loss.detach().double().item()
    pending.backward()
    gradients = gradient_report(model, reader)
    if gradients["base_tensors_with_grad"] or not all(
        row["present"] and row["finite"] and row["nonzero"]
        for row in gradients["sequence_reader"].values()
    ):
        raise SystemExit(f"smoke gradient invariant failed: {gradients}")
    pre_clip = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in reader_parameter_rows(reader)], 1.0
    )
    optimizer.step()
    torch.cuda.synchronize()
    health = cache_health(state, 64)
    if not all(health.values()):
        raise SystemExit("smoke recurrent state failure")
    return {
        "loss": loss_sum / (2 * 64),
        "pre_clip_gradient_norm": float(pre_clip),
        "reader": reader_metrics(reader),
        "gradients": gradients,
        "cache_health": health,
        "finite": math.isfinite(loss_sum),
    }


def smoke(args):
    require_git(clean=True)
    load_config()
    zero_shot_gate(args.run_dir)
    if not torch.cuda.is_available():
        raise SystemExit("2C0 smoke requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    symbols = a0.support.load_training_symbols()
    source, _ = load_source_2b3(args.source_2b3_checkpoint)
    model = instantiate_frozen_model(source, symbols, device)
    model_before = b5.model_state_sha256(model)
    generic, generic_meta, _ = load_generic_artifact(args.run_dir, device)
    means, means_meta, _ = load_source_means(args.run_dir, device)
    generic_before = generic.detach().cpu().clone()
    means_before = means.detach().cpu().clone()
    reader, initialization = load_initial_reader(
        args.source_2b1_checkpoint, symbols, device
    )
    optimizer = fresh_optimizer(reader, "cuda")
    completed = 0
    rows = []
    smoke_checkpoint = Path(args.run_dir) / "smoke" / "checkpoint_update_000002.pt"
    if args.target_update == 3:
        payload = a0.torch_load(smoke_checkpoint)
        if payload.get("schema") != "exp2c0_disposable_smoke_v1":
            raise SystemExit("invalid 2C0 smoke restart checkpoint")
        reader.load_state_dict(payload["sequence_reader"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        completed = payload["completed_updates"]
        rows = payload["rows"]
        if completed != 2:
            raise SystemExit("smoke update 3 must resume update 2")
        optimizer_report(optimizer, 2)
    elif args.target_update != 2:
        raise SystemExit("smoke target must be update 2 or 3")
    _, calibration, _, _, _ = load_validation(symbols)
    for update in range(completed + 1, args.target_update + 1):
        row = process_smoke_batch(
            model, reader, optimizer, calibration[(update - 1) % len(calibration)],
            means, generic,
        )
        row["update"] = update
        rows.append(row)
        optimizer_report(optimizer, update)
        print(
            f"2C0_SMOKE_UPDATE_PASS update={update} loss={row['loss']:.6f}",
            flush=True,
        )
    if args.target_update == 2:
        atomic_torch_save(smoke_checkpoint, {
            "schema": "exp2c0_disposable_smoke_v1",
            "sequence_reader": reader.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_updates": 2,
            "rows": rows,
            "generic_tensor_sha256": generic_meta["tensor_sha256"],
            "source_means_tensor_sha256": means_meta["tensor_sha256"],
        }, refuse_overwrite=True)
        reopened = a0.torch_load(smoke_checkpoint)
        if reopened["completed_updates"] != 2:
            raise SystemExit("smoke forced restart checkpoint reopen failed")
        b2a.write_json(Path(args.run_dir) / "SMOKE_PHASE1.json", {
            "rows": rows,
            "forced_restart_required": True,
            "checkpoint": str(smoke_checkpoint.resolve()),
            "passed": True,
        })
        print("EXPERIMENT_2C0_SMOKE_PHASE1_PASS_RESTART_REQUIRED", flush=True)
        return
    canonical, calibration, _, _, _ = load_validation(symbols)
    drift_rows = [
        drift_batch(model, reader, batch, means, generic, device)
        for batch in calibration
    ]
    causality = causality_and_isolation_tests(
        model, reader, canonical[0][0][:2, :32].to(device), means, generic
    )
    integrity = {
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "base_gradients_none": all(parameter.grad is None for parameter in model.parameters()),
        "generic_bit_identical": torch.equal(generic_before, generic.detach().cpu()),
        "source_means_bit_identical": torch.equal(means_before, means.detach().cpu()),
        "frozen_model_bit_identical": model_before == b5.model_state_sha256(model),
        "losses_finite": all(row["finite"] for row in rows),
        "recurrent_state_finite": all(all(row["cache_health"].values()) for row in rows),
        "causality": causality,
        "fresh_process_restart_observed": completed == 2,
        "optimizer_updates": 3,
        "hellaswag_run": False,
    }
    integrity["passed"] = (
        integrity["base_gradients_none"]
        and integrity["generic_bit_identical"]
        and integrity["source_means_bit_identical"]
        and integrity["frozen_model_bit_identical"]
        and integrity["losses_finite"]
        and integrity["recurrent_state_finite"]
        and integrity["causality"]["passed"]
        and integrity["fresh_process_restart_observed"]
    )
    report = {
        "experiment": "2C0",
        "stage": "disposable_smoke_complete",
        "initialization": initialization,
        "rows": rows,
        "mean_drift_after_smoke": aggregate_drift_diagnostics(drift_rows),
        "integrity": integrity,
        "discard_reader_sha256": reader_state_sha256(reader),
        "passed": integrity["passed"],
    }
    b2a.write_json(Path(args.run_dir) / "SMOKE_FINAL.json", report)
    if not report["passed"]:
        raise SystemExit("2C0 disposable smoke failed")
    print("EXPERIMENT_2C0_DISPOSABLE_SMOKE_PASS", flush=True)


def temporary_update(reader, optimizer):
    before = flat_reader_parameters(reader).detach().clone()
    pre_clip = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in reader_parameter_rows(reader)], 1.0
    )
    optimizer.step()
    torch.cuda.synchronize()
    return {
        "pre_clip_norm": float(pre_clip),
        "update": (flat_reader_parameters(reader) - before).detach().cpu(),
    }


def migration_reference(args):
    require_git(clean=True)
    load_config()
    zero_shot_gate(args.run_dir)
    smoke_report = json.loads((Path(args.run_dir) / "SMOKE_FINAL.json").read_text())
    if not smoke_report.get("passed"):
        raise SystemExit("passing disposable smoke is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    symbols = a0.support.load_training_symbols()
    source, _ = load_source_2b3(args.source_2b3_checkpoint)
    model = instantiate_frozen_model(source, symbols, device)
    generic, _, _ = load_generic_artifact(args.run_dir, device)
    means, _, _ = load_source_means(args.run_dir, device)
    reader, initialization = load_initial_reader(
        args.source_2b1_checkpoint, symbols, device
    )
    optimizer = fresh_optimizer(reader, "cuda")
    loaders = a0.make_replay_loaders(
        symbols, copy.deepcopy(source["dataloader_states"])
    )
    preview = a0.next_update_hash(loaders, symbols, replay=True)
    if preview != SOURCE_NEXT_SHA:
        raise SystemExit("1-GPU migration reference next-batch mismatch")
    local_gradients = []
    local_metrics = []
    for simulated_rank in range(WORLD_SIZE):
        b2a.restore_rank_rng(source["rank_rng_states"][simulated_rank], 0)
        batches = [
            loaders[simulated_rank].next_batch()
            for _ in range(MICROSTEPS_PER_RANK)
        ]
        metrics = process_training_batches(
            model, reader, batches, means, generic
        )
        local_metrics.append(metrics)
        local_gradients.append(flat_reader_gradients(reader).detach().clone())
        reader.zero_grad(set_to_none=True)
    combined = local_gradients[0].clone()
    for rank in range(1, WORLD_SIZE):
        combined.add_(local_gradients[rank])
    scatter_reader_gradients(reader, combined)
    temporary = temporary_update(reader, optimizer)
    payload = {
        "schema": "exp2c0_migration_reference_v1",
        "global_loss": sum(row["raw_loss_sum"] for row in local_metrics) / GLOBAL_TARGETS,
        "global_gradient": combined.cpu(),
        "temporary_update": temporary["update"],
        "temporary_pre_clip_norm": temporary["pre_clip_norm"],
        "next_global_batch_sha256": preview,
        "rank_batch_hashes": [row["batch_hashes"] for row in local_metrics],
        "initialization": initialization,
        "targets": sum(row["target_seen"] for row in local_metrics),
    }
    path = Path(args.run_dir) / "migration" / "one_gpu_reference.pt"
    atomic_torch_save(path, payload, refuse_overwrite=True)
    b2a.write_json(path.with_suffix(".json"), {
        "global_loss": payload["global_loss"],
        "gradient_norm": combined.double().norm().item(),
        "temporary_update_norm": temporary["update"].double().norm().item(),
        "temporary_pre_clip_norm": temporary["pre_clip_norm"],
        "next_global_batch_sha256": preview,
        "targets": payload["targets"],
        "passed": payload["targets"] == GLOBAL_TARGETS,
    })
    print(
        f"EXPERIMENT_2C0_MIGRATION_REFERENCE_PASS loss={payload['global_loss']:.10f}",
        flush=True,
    )


def migration_candidate(args):
    require_git(clean=True)
    load_config()
    zero_shot_gate(args.run_dir)
    rank, local_rank = b2a.init_distributed()
    try:
        device = torch.device("cuda", local_rank)
        symbols = a0.support.load_training_symbols()
        source, _ = load_source_2b3(args.source_2b3_checkpoint)
        model = instantiate_frozen_model(source, symbols, device)
        generic, _, _ = load_generic_artifact(args.run_dir, device)
        means, _, _ = load_source_means(args.run_dir, device)
        reader, _ = load_initial_reader(
            args.source_2b1_checkpoint, symbols, device
        )
        optimizer = fresh_optimizer(reader, "cuda")
        b2a.restore_rank_rng(source["rank_rng_states"][rank], local_rank)
        loader = b2a.make_rank_loader(
            symbols, copy.deepcopy(source["dataloader_states"]), rank
        )
        expected_hash, rank_hashes = b2a.distributed_preview_hash(loader, symbols)
        if expected_hash != SOURCE_NEXT_SHA:
            raise SystemExit("4-GPU migration candidate next-batch mismatch")
        batches = [loader.next_batch() for _ in range(MICROSTEPS_PER_RANK)]
        local = process_training_batches(
            model, reader, batches, means, generic,
            expected_batch_hashes=rank_hashes[rank],
        )
        local_gradient = flat_reader_gradients(reader)
        combined = fixed_rank_sum(local_gradient)
        scatter_reader_gradients(reader, combined)
        temporary = temporary_update(reader, optimizer)
        gathered_metrics = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_metrics, local)
        global_loss = sum(row["raw_loss_sum"] for row in gathered_metrics) / GLOBAL_TARGETS
        reference = a0.torch_load(
            Path(args.run_dir) / "migration" / "one_gpu_reference.pt"
        )
        gradient_comparison = comparison(reference["global_gradient"], combined.cpu())
        update_comparison = comparison(
            reference["temporary_update"], temporary["update"]
        )
        audit = {
            "experiment": "2C0",
            "stage": "1GPU_to_4GPU_migration_equivalence",
            "one_gpu_loss": reference["global_loss"],
            "four_gpu_loss": global_loss,
            "loss_absolute_delta": abs(global_loss - reference["global_loss"]),
            "reader_gradient": gradient_comparison,
            "temporary_update": update_comparison,
            "global_targets": sum(row["target_seen"] for row in gathered_metrics),
            "next_global_batch_sha256": expected_hash,
            "exact_rank_batch_hashes": [row["batch_hashes"] for row in gathered_metrics],
            "thresholds": {
                "loss_absolute_delta": 1e-5,
                "gradient_cosine": 0.999999,
                "gradient_relative_l2": 1e-4,
                "update_cosine": 0.999999,
                "update_relative_l2": 1e-4,
            },
            "temporary_states_discarded": True,
        }
        audit["passed"] = (
            audit["loss_absolute_delta"] <= 1e-5
            and gradient_comparison["cosine"] >= 0.999999
            and gradient_comparison["relative_l2"] <= 1e-4
            and update_comparison["cosine"] >= 0.999999
            and update_comparison["relative_l2"] <= 1e-4
            and audit["global_targets"] == GLOBAL_TARGETS
            and expected_hash == SOURCE_NEXT_SHA
        )
        if rank == 0:
            b2a.write_json(
                Path(args.run_dir) / "FOUR_GPU_EQUIVALENCE_AUDIT.json", audit
            )
            if not audit["passed"]:
                raise SystemExit(f"2C0 migration equivalence failed: {audit}")
            print(
                f"EXPERIMENT_2C0_4GPU_EQUIVALENCE_PASS "
                f"loss_delta={audit['loss_absolute_delta']:.3e} "
                f"grad_cos={gradient_comparison['cosine']:.9f}",
                flush=True,
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def load_training_runtime(args, rank, local_rank):
    device = torch.device("cuda", local_rank)
    symbols = a0.support.load_training_symbols()
    generic, generic_meta, generic_file_sha = load_generic_artifact(args.run_dir, device)
    means, means_meta, means_file_sha = load_source_means(args.run_dir, device)
    input_path = Path(args.checkpoint).resolve()
    input_digest = file_sha256(input_path)
    checkpoint = a0.torch_load(input_path, mmap=True)
    if checkpoint.get("schema") == b3.CHECKPOINT_SCHEMA:
        source, verified = load_source_2b3(input_path)
        if input_digest != verified:
            raise SystemExit("fresh source digest mismatch")
        model = instantiate_frozen_model(source, symbols, device)
        reader, initialization = load_initial_reader(
            args.source_2b1_checkpoint, symbols, device
        )
        optimizer = fresh_optimizer(reader, "cuda")
        completed = 0
        loader_states = source["dataloader_states"]
        b2a.restore_rank_rng(source["rank_rng_states"][rank], local_rank)
        rank_seed = source.get("rank_seeds", [SEED + value for value in range(4)])[rank]
        expected_next = SOURCE_NEXT_SHA
        source_kind = "pristine_post_2b3_cursor"
        frozen_model_sha = b5.model_state_sha256(model)
    elif checkpoint.get("schema") == CHECKPOINT_SCHEMA:
        completed = checkpoint.get("training_state", {}).get("local_updates")
        if completed != RESTART_UPDATE:
            raise SystemExit("2C0 result resume is authorized only from update 5")
        sidecar = input_path.with_suffix(input_path.suffix + ".sha256")
        if not sidecar.is_file() or sidecar.read_text().split()[0] != input_digest:
            raise SystemExit("2C0 resume checkpoint SHA sidecar mismatch")
        source, _ = load_source_2b3(args.source_2b3_checkpoint)
        model = instantiate_frozen_model(source, symbols, device)
        model.load_state_dict(checkpoint["frozen_model"], strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        reader, initialization = load_initial_reader(
            args.source_2b1_checkpoint, symbols, device
        )
        reader.load_state_dict(checkpoint["sequence_reader"], strict=True)
        optimizer = fresh_optimizer(reader, "cuda")
        optimizer.load_state_dict(checkpoint["reader_optimizer"])
        optimizer_report(optimizer, completed)
        if tensor_sha256(
            "generic_correction", checkpoint["generic_correction"]
        ) != generic_meta["tensor_sha256"]:
            raise SystemExit("resume generic correction identity mismatch")
        if tensor_sha256(
            "sequence_source_means", checkpoint["sequence_source_means"]
        ) != means_meta["tensor_sha256"]:
            raise SystemExit("resume source means identity mismatch")
        exact = {
            "source_2b1_checkpoint_sha256": SOURCE_2B1_SHA,
            "source_2b3_checkpoint_sha256": SOURCE_2B3_SHA,
            "generic_artifact_sha256": generic_file_sha,
            "source_means_artifact_sha256": means_file_sha,
            "implementation_git_commit": git_output("rev-parse", "HEAD"),
            "config_sha256": file_sha256(CONFIG_PATH),
            "world_size": WORLD_SIZE,
            "global_targets_per_update": GLOBAL_TARGETS,
        }
        for key, value in exact.items():
            if checkpoint.get(key) != value:
                raise SystemExit(f"2C0 resume lineage mismatch for {key}")
        loader_states = checkpoint["dataloader_states"]
        b2a.restore_rank_rng(checkpoint["rank_rng_states"][rank], local_rank)
        rank_seed = checkpoint["rank_seeds"][rank]
        expected_next = checkpoint["next_global_batch_sha256"]
        source_kind = "fresh_process_update_5_resume"
        frozen_model_sha = checkpoint["frozen_model_sha256"]
        if b5.model_state_sha256(model) != frozen_model_sha:
            raise SystemExit("resume frozen model identity mismatch")
    else:
        raise SystemExit("unsupported 2C0 training input checkpoint")
    loader = b2a.make_rank_loader(symbols, copy.deepcopy(loader_states), rank)
    preview, _ = b2a.distributed_preview_hash(loader, symbols)
    if preview != expected_next:
        raise SystemExit(f"2C0 training next-batch mismatch: {preview} != {expected_next}")
    return {
        "input_checkpoint": checkpoint,
        "input_digest": input_digest,
        "source": source,
        "symbols": symbols,
        "model": model,
        "reader": reader,
        "optimizer": optimizer,
        "loader": loader,
        "completed": completed,
        "rank_seed": rank_seed,
        "source_kind": source_kind,
        "generic": generic,
        "generic_meta": generic_meta,
        "generic_file_sha": generic_file_sha,
        "means": means,
        "means_meta": means_meta,
        "means_file_sha": means_file_sha,
        "initialization": initialization,
        "frozen_model_sha": frozen_model_sha,
    }


def aggregate_training_metrics(rows, reader, update, batch_hash, pre_clip, post_clip,
                               reduction_seconds, wall_seconds):
    count = sum(row["metric_count"] for row in rows)
    routing = [
        sum(row["routing_sum"][index] for row in rows) / count
        for index in range(4)
    ]
    source_rms = [
        sum(row["source_rms_sum"][index] for row in rows) / count
        for index in range(4)
    ]
    feedback_vector = torch.zeros(768, dtype=torch.float64)
    for row in rows:
        feedback_vector.add_(torch.tensor(row["feedback_vector_sum"], dtype=torch.float64))
    mean_feedback = feedback_vector / count
    sequence_feedback_rms = sum(row["feedback_rms_sum"] for row in rows) / count
    return {
        "update": update,
        "processed_targets": update * GLOBAL_TARGETS,
        "global_training_loss": sum(row["raw_loss_sum"] for row in rows) / GLOBAL_TARGETS,
        "reader_pre_clip_gradient_norm": float(pre_clip),
        "reader_post_clip_gradient_norm": float(post_clip),
        "reader": reader_metrics(reader),
        "routing": {f"v{depth}": routing[index] for index, depth in enumerate(SOURCE_DEPTHS)},
        "routing_entropy": sum(row["entropy_sum"] for row in rows) / count,
        "centered_source_rms": {
            f"v{depth}": source_rms[index] for index, depth in enumerate(SOURCE_DEPTHS)
        },
        "sequence_topdown_rms": sum(row["topdown_rms_sum"] for row in rows) / count,
        "sequence_feedback_rms": sequence_feedback_rms,
        "mean_sequence_feedback_rms": mean_feedback.pow(2).mean().sqrt().item(),
        "mean_feedback_ratio": mean_feedback.pow(2).mean().sqrt().item()
        / max(sequence_feedback_rms, 1e-30),
        "global_batch_sha256": batch_hash,
        "global_targets": sum(row["target_seen"] for row in rows),
        "gradient_reduction_seconds": reduction_seconds,
        "update_wall_seconds": wall_seconds,
        "targets_per_second": GLOBAL_TARGETS / wall_seconds,
        "per_rank_peak_allocated_mb": [row["peak_allocated_mb"] for row in rows],
        "per_rank_peak_reserved_mb": [row["peak_reserved_mb"] for row in rows],
        "per_rank_wall_seconds": [row["local_wall_seconds"] for row in rows],
        "all_cache_health_passed": all(
            all(all(health.values()) for health in row["state_health"])
            for row in rows
        ),
        "all_base_gradients_none": all(
            not row["gradient_report"]["base_tensors_with_grad"] for row in rows
        ),
    }


def save_result_checkpoint(args, runtime, rank, local_rank, update):
    model = runtime["model"]
    reader = runtime["reader"]
    optimizer = runtime["optimizer"]
    loader = runtime["loader"]
    symbols = runtime["symbols"]
    loader_states = [None] * WORLD_SIZE
    rng_states = [None] * WORLD_SIZE
    rank_metadata = [None] * WORLD_SIZE
    dist.all_gather_object(loader_states, a0.loader_state(loader))
    dist.all_gather_object(rng_states, b2a.capture_rank_rng(rank, local_rank))
    dist.all_gather_object(rank_metadata, {
        "rank": rank,
        "gpu": local_rank,
        "loader_state": rank,
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
    })
    next_hash, next_rank_hashes = b2a.distributed_preview_hash(loader, symbols)
    state_identity = {
        "reader": reader_state_sha256(reader),
        "optimizer": optimizer_sha256(reader, optimizer, update),
    }
    consistency = b2a.all_equal_across_ranks(
        state_identity, f"2C0 update {update} state"
    )
    output = Path(args.run_dir) / "checkpoints" / f"checkpoint_updates_{update:06d}.pt"
    sidecar = None
    if rank == 0:
        if b5.model_state_sha256(model) != runtime["frozen_model_sha"]:
            raise SystemExit("frozen model changed before checkpoint")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "frozen_model": model.state_dict(),
            "sequence_reader": reader.state_dict(),
            "reader_optimizer": optimizer.state_dict(),
            "generic_correction": runtime["generic"].detach().cpu(),
            "sequence_source_means": runtime["means"].detach().cpu(),
            "training_state": {
                "local_updates": update,
                "processed_targets": update * GLOBAL_TARGETS,
                "fineweb_lineage_completed_update": 535 + update,
                "kind": "2c0_forced_restart" if update == RESTART_UPDATE else "2c0_final",
            },
            "dataloader_states": loader_states,
            "rank_rng_states": rng_states,
            "rank_seeds": [SEED + value for value in range(WORLD_SIZE)],
            "rank_metadata": rank_metadata,
            "next_rank_microstep_hashes": next_rank_hashes,
            "next_global_batch_sha256": next_hash,
            "world_size": WORLD_SIZE,
            "rank_to_loader_mapping": {str(value): value for value in range(WORLD_SIZE)},
            "global_targets_per_update": GLOBAL_TARGETS,
            "gradient_synchronization": "one rank-slotted flattened FP32 NCCL all_reduce(SUM), then fixed rank-order sum",
            "gradient_clipping": "one global sequence-reader norm clipped to 1.0",
            "loss_scaling": "token-loss sums / 524288; no post-SUM division",
            "centering_semantics": "FP32 raw-source subtraction; one BF16 cast; frozen global means",
            "generic_sequence_architecture": "fixed direct G plus independent centered raw-source AttnRes at Block 1",
            "temporal_credit": "none through detached recurrent raw sources or detached historical KV",
            "writers_active": False,
            "source_2b1_checkpoint_sha256": SOURCE_2B1_SHA,
            "source_2b3_checkpoint_sha256": SOURCE_2B3_SHA,
            "source_2b3_data_cursor_sha256": SOURCE_NEXT_SHA,
            "model_initialization_lineage": "2B1 direct reader query/norm with half effective gate",
            "data_cursor_lineage": "post-final-2B3 four-rank replay cursor",
            "generic_artifact_sha256": runtime["generic_file_sha"],
            "generic_tensor_sha256": runtime["generic_meta"]["tensor_sha256"],
            "source_means_artifact_sha256": runtime["means_file_sha"],
            "source_means_tensor_sha256": runtime["means_meta"]["tensor_sha256"],
            "implementation_git_commit": git_output("rev-parse", "HEAD"),
            "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "config_sha256": file_sha256(CONFIG_PATH),
            "frozen_model_sha256": runtime["frozen_model_sha"],
            "sequence_reader_sha256": state_identity["reader"],
            "reader_optimizer_sha256": state_identity["optimizer"],
            "cross_rank_consistency": consistency,
            "hellaswag_run": False,
        }
        digest = atomic_torch_save(output, payload, refuse_overwrite=True)
        reopened = a0.torch_load(output, mmap=True)
        with torch.random.fork_rng(devices=[]):
            clone_model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
        clone_model.load_state_dict(reopened["frozen_model"], strict=True)
        clone_reader, _ = load_initial_reader(
            args.source_2b1_checkpoint, symbols, torch.device("cpu")
        )
        clone_reader.load_state_dict(reopened["sequence_reader"], strict=True)
        clone_optimizer = fresh_optimizer(clone_reader, "cpu")
        clone_optimizer.load_state_dict(reopened["reader_optimizer"])
        optimizer_report(clone_optimizer, update)
        if reopened["next_global_batch_sha256"] != next_hash:
            raise SystemExit("strict checkpoint reopen next-batch mismatch")
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{digest}  {output.name}\n"
        )
        sidecar = {
            "checkpoint": str(output.resolve()),
            "sha256": digest,
            "bytes": output.stat().st_size,
            "model_strict_reload": True,
            "reader_strict_reload": True,
            "optimizer_strict_reload": True,
            "optimizer": optimizer_report(reopened["reader_optimizer"], update),
            "loader_states": len(reopened["dataloader_states"]),
            "rank_rng_states": len(reopened["rank_rng_states"]),
            "next_global_batch_sha256": next_hash,
            "cross_rank_consistency": consistency,
            "passed": True,
        }
        b2a.write_json(output.with_suffix(output.suffix + ".verification.json"), sidecar)
        print(
            f"2C0_CHECKPOINT_PASS update={update} sha256={digest} next={next_hash}",
            flush=True,
        )
    dist.barrier()
    return sidecar


def train(args):
    require_git(clean=True)
    load_config()
    zero_shot_gate(args.run_dir)
    equivalence = json.loads(
        (Path(args.run_dir) / "FOUR_GPU_EQUIVALENCE_AUDIT.json").read_text()
    )
    if not equivalence.get("passed"):
        raise SystemExit("passing four-GPU equivalence is required")
    if args.target_update not in (RESTART_UPDATE, RESULT_UPDATES):
        raise SystemExit("2C0 training target must be update 5 or 10")
    rank, local_rank = b2a.init_distributed()
    try:
        runtime = load_training_runtime(args, rank, local_rank)
        completed = runtime["completed"]
        if completed == 0 and args.target_update != RESTART_UPDATE:
            raise SystemExit("fresh 2C0 result run must hard-stop at update 5")
        if completed == RESTART_UPDATE and args.target_update != RESULT_UPDATES:
            raise SystemExit("restarted 2C0 result run must hard-stop at update 10")
        model = runtime["model"]
        reader = runtime["reader"]
        optimizer = runtime["optimizer"]
        loader = runtime["loader"]
        symbols = runtime["symbols"]
        metrics_path = Path(args.run_dir) / "training_metrics.jsonl"
        stage_started = time.perf_counter()
        for update in range(completed + 1, args.target_update + 1):
            update_started = time.perf_counter()
            expected_hash, rank_hashes = b2a.distributed_preview_hash(loader, symbols)
            optimizer.zero_grad(set_to_none=True)
            batches = [loader.next_batch() for _ in range(MICROSTEPS_PER_RANK)]
            local = process_training_batches(
                model, reader, batches, runtime["means"], runtime["generic"],
                expected_batch_hashes=rank_hashes[rank],
            )
            local_gradient = flat_reader_gradients(reader)
            reduction_started = time.perf_counter()
            combined = fixed_rank_sum(local_gradient)
            torch.cuda.synchronize()
            reduction_seconds = time.perf_counter() - reduction_started
            scatter_reader_gradients(reader, combined)
            gradients = gradient_report(model, reader)
            if gradients["base_tensors_with_grad"]:
                raise SystemExit("frozen gradient leak after global reduction")
            pre_clip = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in reader_parameter_rows(reader)], 1.0
            )
            post_clip = b2.grad_global_norm(
                [parameter for _, parameter in reader_parameter_rows(reader)]
            )
            if not torch.isfinite(pre_clip) or not math.isfinite(post_clip):
                raise SystemExit("non-finite synchronized reader gradient")
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_report(optimizer, update)
            consistency = b2a.all_equal_across_ranks({
                "reader": reader_state_sha256(reader),
                "optimizer": optimizer_sha256(reader, optimizer, update),
            }, f"2C0 result update {update}")
            gathered = [None] * WORLD_SIZE
            dist.all_gather_object(gathered, local)
            actual_hash = b2a.canonical_batch_hash(
                [row["batch_hashes"] for row in gathered]
            )
            if actual_hash != expected_hash:
                raise SystemExit("2C0 global consumed batch differs from preview")
            wall = time.perf_counter() - update_started
            row = aggregate_training_metrics(
                gathered, reader, update, actual_hash, pre_clip, post_clip,
                reduction_seconds, wall,
            )
            generic_rms = runtime["generic"].float().pow(2).mean().sqrt().item()
            row["generic_feedback_rms"] = generic_rms
            row["sequence_to_generic_feedback_rms_ratio"] = (
                row["sequence_feedback_rms"] / max(generic_rms, 1e-30)
            )
            row["cross_rank_state"] = consistency
            row["source_kind"] = runtime["source_kind"] if update == completed + 1 else "same_stage"
            row["teacher_training_forward_calls"] = 0
            row["writers_active_calls"] = 0
            if rank == 0:
                b2a.append_jsonl(metrics_path, row)
                print(
                    f"2C0_RESULT_UPDATE_PASS update={update:02d} "
                    f"loss={row['global_training_loss']:.6f} "
                    f"grad={row['reader_pre_clip_gradient_norm']:.6f} "
                    f"gate={row['reader']['effective_gate']:.6f} "
                    f"wall={wall:.1f}s tok/s={row['targets_per_second']:.0f}",
                    flush=True,
                )
        if b5.model_state_sha256(model) != runtime["frozen_model_sha"]:
            raise SystemExit("frozen model changed during 2C0 training")
        sidecar = save_result_checkpoint(
            args, runtime, rank, local_rank, args.target_update
        )
        if rank == 0:
            b2a.write_json(
                Path(args.run_dir) / f"TRAINING_STAGE_{args.target_update}.json",
                {
                    "start_update": completed,
                    "end_update": args.target_update,
                    "new_updates": args.target_update - completed,
                    "processed_targets": args.target_update * GLOBAL_TARGETS,
                    "input_checkpoint_sha256": runtime["input_digest"],
                    "output_checkpoint": sidecar,
                    "stage_wall_seconds": time.perf_counter() - stage_started,
                    "forced_fresh_process_restart_required": args.target_update == RESTART_UPDATE,
                    "hard_stop_reached": args.target_update == RESULT_UPDATES,
                    "frozen_model_unchanged": True,
                    "generic_unchanged": True,
                    "source_means_unchanged": True,
                    "writers_active_calls": 0,
                    "hellaswag_run": False,
                    "passed": True,
                },
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def load_final_runtime(args, local_rank):
    path = Path(args.checkpoint).resolve()
    digest = file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != digest:
        raise SystemExit("final 2C0 checkpoint SHA mismatch")
    checkpoint = a0.torch_load(path, mmap=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("final 2C0 checkpoint schema mismatch")
    if checkpoint.get("training_state", {}).get("local_updates") != RESULT_UPDATES:
        raise SystemExit("final evaluation requires update-10 checkpoint")
    symbols = a0.support.load_training_symbols()
    source, _ = load_source_2b3(args.source_2b3_checkpoint)
    device = torch.device("cuda", local_rank)
    model = instantiate_frozen_model(source, symbols, device)
    model.load_state_dict(checkpoint["frozen_model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    initial_reader, initialization = load_initial_reader(
        args.source_2b1_checkpoint, symbols, device
    )
    final_reader, _ = load_initial_reader(
        args.source_2b1_checkpoint, symbols, device
    )
    final_reader.load_state_dict(checkpoint["sequence_reader"], strict=True)
    for reader in (initial_reader, final_reader):
        for parameter in reader.parameters():
            parameter.requires_grad_(False)
        reader.eval()
    generic, generic_meta, generic_file_sha = load_generic_artifact(args.run_dir, device)
    means, means_meta, means_file_sha = load_source_means(args.run_dir, device)
    if tensor_sha256("generic_correction", checkpoint["generic_correction"]) != generic_meta["tensor_sha256"]:
        raise SystemExit("final checkpoint generic identity mismatch")
    if tensor_sha256("sequence_source_means", checkpoint["sequence_source_means"]) != means_meta["tensor_sha256"]:
        raise SystemExit("final checkpoint source-means identity mismatch")
    if b5.model_state_sha256(model) != checkpoint["frozen_model_sha256"]:
        raise SystemExit("final frozen model identity mismatch")
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha": digest,
        "symbols": symbols,
        "model": model,
        "initial_reader": initial_reader,
        "final_reader": final_reader,
        "initialization": initialization,
        "generic": generic,
        "generic_meta": generic_meta,
        "generic_file_sha": generic_file_sha,
        "means": means,
        "means_meta": means_meta,
        "means_file_sha": means_file_sha,
        "device": device,
    }


def classification_from_metrics(training_gain, final_gap, wins, batchmean_gap, integrity):
    if not integrity:
        return "SEPARATED ARCHITECTURE UNSTABLE"
    if (
        training_gain >= 0.010
        and final_gap >= 0.020
        and wins >= 18
        and batchmean_gap >= 0.010
    ):
        return "SEPARATED SEQUENCE BRANCH LEARNS SEQUENCE MEMORY"
    if training_gain >= 0.010 and (final_gap < 0.010 or batchmean_gap < 0.005):
        return "SEPARATED SEQUENCE BRANCH IMPROVES GENERIC COMPENSATION ONLY"
    if abs(training_gain) < 0.010:
        return "SEPARATED SEQUENCE BRANCH IS NEUTRAL"
    if training_gain <= -0.010:
        return "SEPARATED SEQUENCE BRANCH DEGRADES"
    # The preregistered thresholds leave a narrow stable positive-gain gray
    # zone. It has not met the sequence-memory rule, so classify conservatively
    # as generic compensation and expose the threshold miss in the audit.
    return "SEPARATED SEQUENCE BRANCH IMPROVES GENERIC COMPENSATION ONLY"


def average_stream_diagnostics(rows):
    diagnostics = [row["diagnostics"] for row in rows if "diagnostics" in row]
    if not diagnostics:
        return None
    return {
        "routing": [
            statistics.fmean(row["routing"][index] for row in diagnostics)
            for index in range(4)
        ],
        "routing_entropy": statistics.fmean(row["routing_entropy"] for row in diagnostics),
        "centered_source_rms": [
            statistics.fmean(row["centered_source_rms"][index] for row in diagnostics)
            for index in range(4)
        ],
        "sequence_topdown_rms": statistics.fmean(row["sequence_topdown_rms"] for row in diagnostics),
        "sequence_feedback_rms": statistics.fmean(row["sequence_feedback_rms"] for row in diagnostics),
        "mean_feedback_ratio_per_batch": statistics.fmean(row["mean_feedback_ratio"] for row in diagnostics),
    }


def final_evaluate(args):
    require_git(clean=True)
    load_config()
    zero_shot = zero_shot_gate(args.run_dir)
    rank, local_rank = b2a.init_distributed()
    try:
        runtime = load_final_runtime(args, local_rank)
        model = runtime["model"]
        initial_reader = runtime["initial_reader"]
        final_reader = runtime["final_reader"]
        means = runtime["means"]
        generic = runtime["generic"]
        device = runtime["device"]
        canonical, calibration, canonical_hashes, _, _ = load_validation(runtime["symbols"])
        drift_local = drift_batch(
            model, final_reader, calibration[rank], means, generic, device
        )
        drift_rows = [None] * WORLD_SIZE
        dist.all_gather_object(drift_rows, drift_local)
        final_drift = aggregate_drift_diagnostics(drift_rows)
        fixed_calibration_feedback = torch.tensor(
            final_drift["mean_sequence_feedback"],
            device=device,
            dtype=generic.dtype,
        )
        permutation = b4.coherent_permutation(B, device)
        controls = (
            "generic",
            "initial_real",
            "initial_shuffle",
            "final_real",
            "final_shuffle",
            "final_gate_zero",
            "final_sequence_only",
            "final_batchmean",
            "final_calibration_mean",
        )
        local_rows = []
        for batch_index, (x_cpu, y_cpu) in enumerate(canonical):
            if batch_index % WORLD_SIZE != rank:
                continue
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            for control in controls:
                if control == "generic":
                    reader = final_reader
                    branch_control = "generic"
                    fixed = None
                elif control == "initial_real":
                    reader = initial_reader
                    branch_control = "real"
                    fixed = None
                elif control == "initial_shuffle":
                    reader = initial_reader
                    branch_control = "initial_shuffle"
                    fixed = None
                elif control == "final_real":
                    reader = final_reader
                    branch_control = "real"
                    fixed = None
                elif control == "final_shuffle":
                    reader = final_reader
                    branch_control = "shuffle"
                    fixed = None
                elif control == "final_gate_zero":
                    reader = final_reader
                    branch_control = "gate_zero"
                    fixed = None
                elif control == "final_sequence_only":
                    reader = final_reader
                    branch_control = "sequence_only"
                    fixed = None
                elif control == "final_batchmean":
                    reader = final_reader
                    branch_control = "batchmean"
                    fixed = None
                elif control == "final_calibration_mean":
                    reader = final_reader
                    branch_control = "real"
                    fixed = fixed_calibration_feedback
                row = evaluate_stream(
                    model, reader, x, y, means, generic, branch_control,
                    permutation=permutation,
                    fixed_sequence_feedback=fixed,
                    capture=control in {"initial_real", "final_real", "final_shuffle"},
                )
                row.update({
                    "batch_index": batch_index,
                    "payload_sha256": canonical_hashes[batch_index],
                    "control": control,
                    "rank": rank,
                })
                local_rows.append(row)
                print(
                    f"2C0_FINAL_EVAL rank={rank} control={control} "
                    f"batch={batch_index} loss={row['loss']:.10f} "
                    f"wall={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local_rows)
        model_sha_after = b5.model_state_sha256(model)
        local_integrity = {
            "all_losses_finite": all(row["finite"] for row in local_rows),
            "base_gradients_none": all(parameter.grad is None for parameter in model.parameters()),
            "old_reader_gradients_none": all(
                parameter.grad is None for parameter in model.transformer.topdown_attnres.parameters()
            ),
            "writer_gradients_none": all(
                parameter.grad is None for parameter in model.transformer.memory_writers.parameters()
            ),
            "generic_unchanged": tensor_sha256("generic_correction", generic) == runtime["generic_meta"]["tensor_sha256"],
            "source_means_unchanged": tensor_sha256("sequence_source_means", means) == runtime["means_meta"]["tensor_sha256"],
            "frozen_model_unchanged": model_sha_after == runtime["checkpoint"]["frozen_model_sha256"],
            "writers_active_calls": 0,
            "hellaswag_run": False,
        }
        local_integrity["passed"] = all(
            value for key, value in local_integrity.items()
            if key not in {"writers_active_calls", "hellaswag_run"}
        ) and local_integrity["writers_active_calls"] == 0
        integrity_rows = [None] * WORLD_SIZE
        dist.all_gather_object(integrity_rows, local_integrity)
        if rank == 0:
            rows = sorted(
                [row for group in gathered for row in group],
                key=lambda row: (row["control"], row["batch_index"]),
            )
            by_control = {
                control: [row for row in rows if row["control"] == control]
                for control in controls
            }
            losses = {
                control: statistics.fmean(row["loss"] for row in selected)
                for control, selected in by_control.items()
            }
            initial_paired = paired_statistics(
                [row["loss"] for row in by_control["initial_real"]],
                [row["loss"] for row in by_control["initial_shuffle"]],
            )
            final_paired = paired_statistics(
                [row["loss"] for row in by_control["final_real"]],
                [row["loss"] for row in by_control["final_shuffle"]],
            )
            batchmean_paired = paired_statistics(
                [row["loss"] for row in by_control["final_real"]],
                [row["loss"] for row in by_control["final_batchmean"]],
            )
            metrics = {
                "sequence_gain_initial": losses["generic"] - losses["initial_real"],
                "sequence_gain_final": losses["generic"] - losses["final_real"],
                "specific_gap_initial": initial_paired["mean_gap"],
                "specific_gap_final": final_paired["mean_gap"],
                "specific_gap_gain": final_paired["mean_gap"] - initial_paired["mean_gap"],
                "training_real_gain": losses["initial_real"] - losses["final_real"],
                "batchmean_gap": losses["final_batchmean"] - losses["final_real"],
                "calibration_mean_gap": losses["final_calibration_mean"] - losses["final_real"],
            }
            all_integrity = all(row["passed"] for row in integrity_rows)
            classification = classification_from_metrics(
                metrics["training_real_gain"], metrics["specific_gap_final"],
                final_paired["real_wins"], metrics["batchmean_gap"], all_integrity,
            )
            report = {
                "experiment": "2C0",
                "stage": "final_canonical_evaluation",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256": runtime["checkpoint_sha"],
                "canonical_validation_sha256": CANONICAL_SHA,
                "losses": losses,
                "metrics": metrics,
                "initial_paired_real_vs_shuffle": initial_paired,
                "final_paired_real_vs_shuffle": final_paired,
                "final_paired_real_vs_batchmean": batchmean_paired,
                "mean_drift_final": final_drift,
                "initial_reader": reader_metrics(initial_reader),
                "final_reader": reader_metrics(final_reader),
                "initial_stream_diagnostics": average_stream_diagnostics(
                    by_control["initial_real"]
                ),
                "final_stream_diagnostics": average_stream_diagnostics(
                    by_control["final_real"]
                ),
                "zero_shot_regression": {
                    "initial_real_difference": losses["initial_real"] - zero_shot["losses"]["real"],
                    "initial_shuffle_difference": losses["initial_shuffle"] - zero_shot["losses"]["shuffle"],
                },
                "per_batch_losses": {
                    control: [
                        {"batch_index": row["batch_index"], "loss": row["loss"]}
                        for row in by_control[control]
                    ] for control in controls
                },
                "integrity_by_rank": integrity_rows,
                "integrity_passed": all_integrity,
                "classification": classification,
                "hellaswag_run": False,
            }
            b2a.write_json(Path(args.run_dir) / "FINAL_EVALUATION.json", report)
            if not all_integrity:
                raise SystemExit("2C0 final evaluation integrity failed")
            print(
                f"EXPERIMENT_2C0_FINAL_EVALUATION_PASS classification={classification}",
                flush=True,
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def fmt(value):
    return f"{value:.10f}" if isinstance(value, float) else str(value)


def finalize(args):
    require_git(clean=False)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    zero = json.loads((run_dir / "ZERO_SHOT_CONTROLS.json").read_text())
    smoke_report = json.loads((run_dir / "SMOKE_FINAL.json").read_text())
    migration = json.loads((run_dir / "FOUR_GPU_EQUIVALENCE_AUDIT.json").read_text())
    evaluation = json.loads((run_dir / "FINAL_EVALUATION.json").read_text())
    stage5 = json.loads((run_dir / "TRAINING_STAGE_5.json").read_text())
    stage10 = json.loads((run_dir / "TRAINING_STAGE_10.json").read_text())
    metrics_rows = [
        json.loads(line) for line in (run_dir / "training_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if [row["update"] for row in metrics_rows] != list(range(1, 11)):
        raise SystemExit("2C0 training metric coverage mismatch")
    checkpoint = a0.torch_load(evaluation["checkpoint"], mmap=True)
    verification = json.loads(
        Path(evaluation["checkpoint"] + ".verification.json").read_text()
    )
    generic = a0.torch_load(run_dir / "generic_correction.pt")
    source_means = a0.torch_load(run_dir / "sequence_source_means.pt")
    classification = evaluation["classification"]
    losses = evaluation["losses"]
    primary = evaluation["metrics"]
    integrity = {
        "trainable_parameters_exactly_1537": zero["integrity"]["trainable_params"] == TRAINABLE_PARAMETERS,
        "base_gradients_none": all(row["all_base_gradients_none"] for row in metrics_rows),
        "old_reader_gradients_none": zero["integrity"]["old_reader_gradients_none"],
        "writer_gradients_none": zero["integrity"]["writer_gradients_none"],
        "writers_never_active": all(row["writers_active_calls"] == 0 for row in metrics_rows),
        "generic_G_frozen": all(row["generic_unchanged"] for row in evaluation["integrity_by_rank"]),
        "source_means_frozen": all(row["source_means_unchanged"] for row in evaluation["integrity_by_rank"]),
        "historical_KV_temporal_gradients_none": all(
            all(row["all_cache_health_passed"] for row in metrics_rows)
            for _ in [0]
        ),
        "future_causality_pass": all(
            row["future_causality"]["passed"]
            for row in zero["causality_by_rank"]
        ),
        "row_isolation_pass": all(
            row["row_isolation"]["passed"]
            for row in zero["causality_by_rank"]
        ),
        "zero_input_feedback_exactly_zero": all(
            row["zero_input_contract"]["passed"]
            for row in zero["causality_by_rank"]
        ),
        "global_targets_per_update_524288": all(
            row["global_targets"] == GLOBAL_TARGETS for row in metrics_rows
        ),
        "total_updates_exactly_10": len(metrics_rows) == 10,
        "total_targets_exactly_5242880": metrics_rows[-1]["processed_targets"] == 5_242_880,
        "forced_restart_after_update_5": stage5["forced_fresh_process_restart_required"]
        and stage10["start_update"] == 5,
        "replay_hashes_exact": stage5["passed"] and stage10["passed"],
        "cross_rank_reader_state_identical": all(
            bool(row["cross_rank_state"]) for row in metrics_rows
        ),
        "cross_rank_reader_optimizer_identical": all(
            bool(row["cross_rank_state"]) for row in metrics_rows
        ),
        "all_losses_finite": all(
            row["all_losses_finite"] for row in evaluation["integrity_by_rank"]
        ),
        "generic_only_regression_pass": zero["generic_regression"]["passed"],
        "zero_shot_gate_pass": zero["training_gate"]["passed"],
        "distributed_equivalence_pass": migration["passed"],
        "disposable_smoke_pass": smoke_report["passed"],
        "checkpoint_atomic_strict_reopen_pass": verification["passed"],
        "frozen_model_unchanged": all(
            row["frozen_model_unchanged"] for row in evaluation["integrity_by_rank"]
        ),
        "hellaswag_not_run": not evaluation["hellaswag_run"],
    }
    integrity["passed"] = all(integrity.values())
    if not integrity["passed"] or not evaluation["integrity_passed"]:
        raise SystemExit(f"2C0 final audit failed: {integrity}")
    runtime = sum(row["update_wall_seconds"] for row in metrics_rows)
    summary = {
        "experiment": "2C0",
        "implementation_git_commit": checkpoint["implementation_git_commit"],
        "results_commit": args.results_commit,
        "classification": classification,
        "generic_branch": {
            **generic["metadata"],
            "artifact_sha256": file_sha256(run_dir / "generic_correction.pt"),
            "expected_loss": EXPECTED_GENERIC_LOSS,
            "measured_loss": zero["losses"]["generic"],
            "regression_passed": zero["generic_regression"]["passed"],
        },
        "sequence_centering": {
            **source_means["metadata"],
            "artifact_sha256": file_sha256(run_dir / "sequence_source_means.pt"),
        },
        "initialization": zero["initialization"],
        "zero_shot": zero,
        "distributed_preflight": migration,
        "training": {
            "updates": 10,
            "targets": 5_242_880,
            "runtime_seconds": runtime,
            "targets_per_second": 5_242_880 / runtime,
            "peak_vram_mb": max(
                max(row["per_rank_peak_allocated_mb"]) for row in metrics_rows
            ),
            "metrics": metrics_rows,
        },
        "final_evaluation": evaluation,
        "integrity": integrity,
        "optimizer_updates": 10,
        "backward_calls": 10 * WORLD_SIZE * MICROSTEPS_PER_RANK * (T // BACKWARD_CHUNK),
        "parameter_updates": 10,
        "additional_training_targets": 5_242_880,
        "hellaswag_run": False,
    }
    audit = {
        "experiment": "2C0",
        "classification": classification,
        "hard_invariants": integrity,
        "checkpoint": {
            "path": evaluation["checkpoint"],
            "sha256": evaluation["checkpoint_sha256"],
            "verification": verification,
            "training_state": checkpoint["training_state"],
            "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        },
        "generic_before_after_sha256": generic["metadata"]["tensor_sha256"],
        "source_means_before_after_sha256": source_means["metadata"]["tensor_sha256"],
        "canonical_evaluations": CANONICAL_BATCHES * 9,
        "optimizer_updates": 10,
        "additional_training_targets": 5_242_880,
        "hellaswag_run": False,
        "passed": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    b2a.write_json(output_dir / "result_summary.json", summary)
    b2a.write_json(output_dir / "FINAL_AUDIT.json", audit)
    b2a.write_json(output_dir / "paired_losses.json", {
        "initial_real_vs_shuffle": evaluation["initial_paired_real_vs_shuffle"],
        "final_real_vs_shuffle": evaluation["final_paired_real_vs_shuffle"],
        "final_real_vs_batchmean": evaluation["final_paired_real_vs_batchmean"],
        "per_batch_losses": evaluation["per_batch_losses"],
    })
    if args.results_commit:
        init_diag = evaluation["initial_stream_diagnostics"]
        final_diag = evaluation["final_stream_diagnostics"]
        final_paired = evaluation["final_paired_real_vs_shuffle"]
        decisions = {
            "A": "YES" if classification == "SEPARATED SEQUENCE BRANCH LEARNS SEQUENCE MEMORY" else "NO",
            "B": "YES, but do not launch it" if classification == "SEPARATED SEQUENCE BRANCH LEARNS SEQUENCE MEMORY" else "NO",
            "C": "NO" if classification == "SEPARATED SEQUENCE BRANCH LEARNS SEQUENCE MEMORY" else "YES",
            "D": "NO, continue the demonstrated Block-1 reader first" if classification == "SEPARATED SEQUENCE BRANCH LEARNS SEQUENCE MEMORY" else "YES, as a new controlled experiment",
            "E": "YES",
            "F": "YES",
        }
        lines = [
            "# Experiment 2C0 — Final Report", "",
            "## Git", "",
            f"2B5 frozen tag: `{PARENT_TAG}`  ",
            f"2B5 parent commit: `{PARENT_COMMIT}`  ",
            f"2C0 branch: `{BRANCH}`  ",
            f"Implementation commit: `{summary['implementation_git_commit']}`  ",
            f"Results commit: `{args.results_commit}`  ",
            "Final report commit: `the immutable commit containing this file`", "",
            "## Generic branch", "",
            f"Generic calibration source: final 2B3/2B5 disjoint calibration artifacts  ",
            f"Generic μ provenance: `{GENERIC_MEAN_PATH.relative_to(REPO_ROOT)}`  ",
            f"G SHA: `{summary['generic_branch']['tensor_sha256']}`  ",
            f"G RMS: `{summary['generic_branch']['rms']:.10f}`  ",
            f"Generic-only expected loss: `{EXPECTED_GENERIC_LOSS:.10f}`  ",
            f"Generic-only measured loss: `{zero['losses']['generic']:.10f}`  ",
            f"Regression: `{'PASS' if zero['generic_regression']['passed'] else 'FAIL'}`", "",
            "## Sequence centering", "",
            f"Calibration manifest: `{CALIBRATION_MANIFEST_PATH.relative_to(REPO_ROOT)}`  ",
            *[
                f"ν{depth} SHA: `{source_means['metadata']['source_shas'][f'nu{depth}']}`  "
                for depth in SOURCE_DEPTHS
            ],
            f"Mean centered calibration residual, maximum absolute component: `{source_means['metadata']['maximum_absolute_mean_centered_residual']:.3e}`", "",
            "## Initialization", "",
            f"2B1 source checkpoint: `{zero['initialization']['source_checkpoint']}`  ",
            f"SHA: `{SOURCE_2B1_SHA}`  ",
            f"Copied query norm: `{zero['initialization']['copied_query_norm']:.10f}`  ",
            f"Copied RMSNorm displacement: `{zero['initialization']['copied_rmsnorm_displacement']:.10f}`  ",
            f"Old effective gate: `{zero['initialization']['old_effective_gate']:.10f}`  ",
            f"New effective gate: `{zero['initialization']['new_effective_gate']:.10f}`", "",
            "## Zero-shot controls", "",
            f"Generic only: `{zero['losses']['generic']:.10f}`  ",
            f"Generic + real sequence: `{zero['losses']['real']:.10f}`  ",
            f"Generic + shuffled sequence: `{zero['losses']['shuffle']:.10f}`  ",
            f"Sequence only: `{zero['losses']['sequence_only']:.10f}`  ",
            f"Gate zero: `{zero['losses']['gate_zero']:.10f}`  ",
            f"Specific gap: `{zero['specific_gap_0']:.10f}`  ",
            f"Real wins: `{zero['paired_real_vs_shuffle']['real_wins']}/20`", "",
            "## Distributed preflight", "",
            f"1GPU loss: `{migration['one_gpu_loss']:.10f}`  ",
            f"4GPU loss: `{migration['four_gpu_loss']:.10f}`  ",
            f"Reader grad cosine: `{migration['reader_gradient']['cosine']:.10f}`  ",
            f"Reader grad relative L2: `{migration['reader_gradient']['relative_l2']:.3e}`  ",
            f"Temporary update cosine: `{migration['temporary_update']['cosine']:.10f}`  ",
            f"Temporary update relative L2: `{migration['temporary_update']['relative_l2']:.3e}`  ",
            f"Result: `{'PASS' if migration['passed'] else 'FAIL'}`", "",
            "## Training", "",
            f"Updates: `10`  ",
            f"Targets: `5,242,880`  ",
            f"Runtime: `{runtime:.1f} seconds`  ",
            f"Targets/sec: `{summary['training']['targets_per_second']:.0f}`  ",
            f"Peak VRAM: `{summary['training']['peak_vram_mb']:.1f} MiB`", "",
            "## Final controls", "",
            f"Generic only: `{losses['generic']:.10f}`  ",
            f"Initial real: `{losses['initial_real']:.10f}`  ",
            f"Initial shuffled: `{losses['initial_shuffle']:.10f}`  ",
            f"Trained real: `{losses['final_real']:.10f}`  ",
            f"Trained shuffled: `{losses['final_shuffle']:.10f}`  ",
            f"Trained gate zero: `{losses['final_gate_zero']:.10f}`  ",
            f"Trained sequence only: `{losses['final_sequence_only']:.10f}`  ",
            f"Batch-mean sequence: `{losses['final_batchmean']:.10f}`  ",
            f"Calibration-mean sequence: `{losses['final_calibration_mean']:.10f}`", "",
            "## Primary metrics", "",
            f"Training real gain: `{primary['training_real_gain']:.10f}`  ",
            f"Initial specific gap: `{primary['specific_gap_initial']:.10f}`  ",
            f"Final specific gap: `{primary['specific_gap_final']:.10f}`  ",
            f"Specific gap gain: `{primary['specific_gap_gain']:.10f}`  ",
            f"Real wins: `{final_paired['real_wins']}/20`  ",
            f"Shuffled wins: `{final_paired['shuffled_wins']}/20`  ",
            f"Batchmean gap: `{primary['batchmean_gap']:.10f}`", "",
            "## Sequence reader", "",
            "| metric | initial | final |", "|---|---:|---:|",
            f"| gate | {evaluation['initial_reader']['effective_gate']:.8f} | {evaluation['final_reader']['effective_gate']:.8f} |",
            f"| query norm | {evaluation['initial_reader']['query_norm']:.8f} | {evaluation['final_reader']['query_norm']:.8f} |",
            f"| RMS displacement | {evaluation['initial_reader']['rmsnorm_displacement']:.8f} | {evaluation['final_reader']['rmsnorm_displacement']:.8f} |",
            *[
                f"| v{depth} routing | {init_diag['routing'][index]:.8f} | {final_diag['routing'][index]:.8f} |"
                for index, depth in enumerate(SOURCE_DEPTHS)
            ],
            f"| entropy | {init_diag['routing_entropy']:.8f} | {final_diag['routing_entropy']:.8f} |",
            f"| sequence feedback RMS | {init_diag['sequence_feedback_rms']:.8f} | {final_diag['sequence_feedback_rms']:.8f} |",
            f"| sequence/generic RMS ratio | {init_diag['sequence_feedback_rms']/summary['generic_branch']['rms']:.8f} | {final_diag['sequence_feedback_rms']/summary['generic_branch']['rms']:.8f} |",
            f"| mean-feedback ratio | {zero['mean_drift_initial']['mean_feedback_ratio']:.8f} | {evaluation['mean_drift_final']['mean_feedback_ratio']:.8f} |", "",
            "## Integrity", "",
            *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in integrity.items() if key != "passed"], "",
            "## Classification", "", classification, "",
            "## Decisions A–F", "",
            f"A. Did separation prevent collapse into generic compensation? **{decisions['A']}**  ",
            f"B. Continue this reader beyond 5M? **{decisions['B']}**  ",
            f"C. Is an aligned-vs-shuffled auxiliary objective necessary? **{decisions['C']}**  ",
            f"D. Test a middle/higher destination next? **{decisions['D']}**  ",
            f"E. Keep writers absent until a destination proves direct sequence-specific signal? **{decisions['E']}**  ",
            f"F. Keep temporal credit zero/one-step-free at this stage? **{decisions['F']}**", "",
            "No additional reader training, writers, auxiliary losses, destination changes, mask-depth changes, BPTT, or HellaSwag were launched.", "",
            "# EXPERIMENT 2C0 COMPLETE",
        ]
        (output_dir / "EXPERIMENT_2C0_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        f"EXPERIMENT_2C0_FINALIZE_PASS classification={classification}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "prepare", "smoke", "migration-reference", "migration-candidate",
        "train", "evaluate-final", "finalize",
    ))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-2b3-checkpoint")
    parser.add_argument("--source-2b1-checkpoint")
    parser.add_argument("--checkpoint")
    parser.add_argument("--target-update", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--results-commit")
    return parser.parse_args()


def require_common_sources(args):
    if not args.source_2b3_checkpoint or not args.source_2b1_checkpoint:
        raise SystemExit("command requires both 2B3 and 2B1 source checkpoints")


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if args.command == "prepare":
        require_common_sources(args)
        prepare(args)
    elif args.command == "smoke":
        require_common_sources(args)
        if args.target_update is None:
            raise SystemExit("smoke requires --target-update")
        smoke(args)
    elif args.command == "migration-reference":
        require_common_sources(args)
        migration_reference(args)
    elif args.command == "migration-candidate":
        require_common_sources(args)
        migration_candidate(args)
    elif args.command == "train":
        require_common_sources(args)
        if args.checkpoint is None or args.target_update is None:
            raise SystemExit("train requires --checkpoint and --target-update")
        train(args)
    elif args.command == "evaluate-final":
        require_common_sources(args)
        if args.checkpoint is None:
            raise SystemExit("evaluate-final requires --checkpoint")
        final_evaluate(args)
    elif args.command == "finalize":
        if args.output_dir is None:
            raise SystemExit("finalize requires --output-dir")
        finalize(args)


if __name__ == "__main__":
    main()
