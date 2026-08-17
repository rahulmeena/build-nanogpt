#!/usr/bin/env python3
"""Experiment 2B5: zero-optimizer generic-mean/residual decomposition."""

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2a0 as a0  # noqa: E402
import experiment_2b2 as b2  # noqa: E402
import experiment_2b2a as b2a  # noqa: E402
import experiment_2b3 as b3  # noqa: E402
import experiment_2b4 as b4  # noqa: E402


BRANCH = "experiment-2b5-mean-residual-decomposition-4gpu"
FROZEN_2B4_TAG = "experiment-2b4-memory-content-mask-depth-final"
FROZEN_2B4_COMMIT = "692fd80ba9fb5e81731397dcd4bf149c3c705d41"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2b5_decomposition_4gpu.json"
FROZEN_CALIBRATION_MANIFEST = (
    REPO_ROOT
    / "results"
    / "experiment_2b4_memory_content_mask_depth"
    / "part_a_calibration_manifest.json"
)
WORLD_SIZE = 4
B = 64
T = 1024
SOURCE_DEPTHS = (16, 17, 20, 24)
CANONICAL_BATCHES = 20
CALIBRATION_BATCHES = 4
CANONICAL_SHA256 = "3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb"
CALIBRATION_SHA256 = "d159c297f26e5e7ef707d37c5656b3702d66a11809ebed5577cd12903bfcb2f6"
DIAGNOSTIC_SEED = 20260817
ALPHAS = (0.25, 0.5, 1.0, 2.0)
CHECKPOINT_LABELS = (
    "C0_2B2_5M",
    "C1_2B2A_10M",
    "C2_2B2A_15M",
    "C3_2B3_FINAL",
)
BASE_SHA256 = "1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd"
FROZEN_READER_SHA256 = "aca8f87518e3728b5d721a48e7729b8e93569a23174f29299d3795227fcd61a7"
C3_GLOBAL_TEMPLATE_SHA256 = "26c550b5770307b50c20a384447785f348515e4a232f60f78b58c71b62c1fd99"

CHECKPOINTS = {
    "C0_2B2_5M": {
        "stage": "2B2 5M",
        "writer_lineage_tokens": 5_242_880,
        "sha256": "a125c81acb9e4ec3395bd8b38dee8fade62012c642b102a1b6c4c0e0997f0637",
        "schema": b2.CHECKPOINT_SCHEMA,
        "training_state": {
            "local_completed_updates": 10,
            "processed_2b2_tokens": 5_242_880,
            "fineweb_lineage_completed_update": 497,
            "kind": "result_5m",
        },
        "next_global_batch_sha256": "e3289bee6ed5a5b2fa1d2c05a615cd3f10f07c51b71aa091ee40380ebeedc21b",
        "subset_sha256": {
            "base": BASE_SHA256,
            "reader": FROZEN_READER_SHA256,
            "writers": "eadff45425a595a79eea10e2ef8313050b338ab2b508c20730df652e0054eafb",
        },
        "historical": {
            "zero": 5.9736480713,
            "real": 5.5900331020,
            "shuffle": 5.6259720802,
        },
    },
    "C1_2B2A_10M": {
        "stage": "2B2A 10M",
        "writer_lineage_tokens": 10_485_760,
        "sha256": "de5e04f817dcfa5dd8a4dcc6e503ec86d8545d558d837b517c7259917218dff3",
        "schema": b2a.CHECKPOINT_SCHEMA,
        "training_state": {
            "writer_updates": 20,
            "writer_training_tokens": 10_485_760,
            "fineweb_lineage_completed_update": 517,
            "kind": "2b2a_10m",
        },
        "next_global_batch_sha256": "ddbb966eff17ddabd102ce4706ccace0e23973f98803478b392b8c4e5f9d32f3",
        "historical": {
            "zero": 5.9736480713,
            "real": 5.3613477468,
            "shuffle": 5.4016084433,
        },
    },
    "C2_2B2A_15M": {
        "stage": "2B2A 15M",
        "writer_lineage_tokens": 15_204_352,
        "sha256": "86c66343141e24d0beffcf8bc98a558f25c82e1dc05582feade2300d30b2ba84",
        "schema": b2a.CHECKPOINT_SCHEMA,
        "training_state": {
            "writer_updates": 29,
            "writer_training_tokens": 15_204_352,
            "fineweb_lineage_completed_update": 526,
            "kind": "2b2a_15m",
        },
        "next_global_batch_sha256": "8b9fe2fa1c2a10ce930caff4d527c48e4f14ab0e1a6f5e4b352e42f61b8b360d",
        "historical": {
            "zero": 5.9736480713,
            "real": 5.0959878206,
            "shuffle": 5.1250012159,
        },
    },
    "C3_2B3_FINAL": {
        "stage": "2B3 final",
        "writer_lineage_tokens": 19_922_944,
        "sha256": "7797f349905e344934bd7d2475cf61b332ef9053cb0bc1a44f450fc24249c65b",
        "schema": b3.CHECKPOINT_SCHEMA,
        "training_state": {
            "writer_lineage_updates": 38,
            "writer_lineage_tokens": 19_922_944,
            "joint_local_updates": 9,
            "joint_training_tokens": 4_718_592,
            "fineweb_lineage_completed_update": 535,
            "kind": "2b3_final",
        },
        "next_global_batch_sha256": "7f6d8da5044e9f485492712373fda12d09eb4ccec60ab8fdee812519d05869a7",
        "historical": {
            "zero": 5.9736480713,
            "real": 4.8141904593,
            "shuffle": 4.8176936150,
            "mu": 4.7776873112,
            "independent": 4.7925686121,
        },
    },
}

CONTROL_EXECUTION_ORDER = (
    "zero",
    "real",
    "mu",
    "alpha_real_1",
    "alpha_shuffle_1",
    "independent_shuffle",
    "residual",
    "residual_shuffle",
    "alpha_real_0.25",
    "alpha_shuffle_0.25",
    "alpha_real_0.5",
    "alpha_shuffle_0.5",
    "alpha_real_2",
    "alpha_shuffle_2",
)

RUNTIME_COUNTS = {
    "optimizer_objects_created": 0,
    "scheduler_objects_created": 0,
    "grad_scalers_created": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
}


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2B5 requires branch {BRANCH}")
    if git_output("rev-parse", f"{FROZEN_2B4_TAG}^{{}}") != FROZEN_2B4_COMMIT:
        raise SystemExit("frozen Experiment 2B4 tag target mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FROZEN_2B4_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing execution requires a clean worktree")


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        "protocol": "exp2b5_mean_residual_decomposition_v1",
        "world_size": WORLD_SIZE,
        "batch_sequences": B,
        "sequence_length": T,
        "canonical_batches": CANONICAL_BATCHES,
        "calibration_batches": CALIBRATION_BATCHES,
        "canonical_validation_sha256": CANONICAL_SHA256,
        "calibration_aggregate_sha256": CALIBRATION_SHA256,
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "alphas": list(ALPHAS),
        "checkpoint_labels": list(CHECKPOINT_LABELS),
        "training": "forbidden",
        "hellaswag": "forbidden",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f"config {key} mismatch: {config.get(key)} != {value}")
    return config


def install_zero_training_guards():
    """Turn every forbidden training primitive into an immediate hard failure."""

    def forbidden_optimizer_init(*args, **kwargs):
        RUNTIME_COUNTS["optimizer_objects_created"] += 1
        raise RuntimeError("Experiment 2B5 forbids optimizer construction")

    def forbidden_optimizer_step(*args, **kwargs):
        RUNTIME_COUNTS["optimizer_steps"] += 1
        raise RuntimeError("Experiment 2B5 forbids optimizer steps")

    def forbidden_backward(*args, **kwargs):
        RUNTIME_COUNTS["backward_calls"] += 1
        raise RuntimeError("Experiment 2B5 forbids backward calls")

    torch.optim.Optimizer.__init__ = forbidden_optimizer_init
    torch.optim.Optimizer.step = forbidden_optimizer_step
    torch.Tensor.backward = forbidden_backward
    torch.autograd.backward = forbidden_backward
    torch.autograd.grad = forbidden_backward

    seen = set()
    for name in ("LRScheduler", "_LRScheduler"):
        scheduler_type = getattr(torch.optim.lr_scheduler, name, None)
        if scheduler_type is None or scheduler_type in seen:
            continue
        seen.add(scheduler_type)

        def forbidden_scheduler_init(*args, **kwargs):
            RUNTIME_COUNTS["scheduler_objects_created"] += 1
            raise RuntimeError("Experiment 2B5 forbids scheduler construction")

        scheduler_type.__init__ = forbidden_scheduler_init

    seen = set()
    for scaler_type in (
        getattr(torch.amp, "GradScaler", None),
        getattr(torch.cuda.amp, "GradScaler", None),
    ):
        if scaler_type is None or scaler_type in seen:
            continue
        seen.add(scaler_type)

        def forbidden_scaler_init(*args, **kwargs):
            RUNTIME_COUNTS["grad_scalers_created"] += 1
            raise RuntimeError("Experiment 2B5 forbids GradScaler construction")

        scaler_type.__init__ = forbidden_scaler_init


def assert_zero_training_counts():
    if any(RUNTIME_COUNTS.values()):
        raise SystemExit(f"forbidden training activity detected: {RUNTIME_COUNTS}")


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def coherent_permutation(batch_size, device):
    return b4.coherent_permutation(batch_size, device)


def independent_source_permutations(batch_size, device):
    return b4.independent_source_permutations(batch_size, device)


def _expanded_mean(mean, memory):
    if tuple(mean.shape) != (len(SOURCE_DEPTHS), memory.size(-1)):
        raise ValueError(f"invalid generic mean shape: {tuple(mean.shape)}")
    return mean.float()[:, None, None, :].to(memory.device).expand_as(memory)


def controlled_memory(control, candidate_memory, mean, position, return_identity=False):
    """Apply one production FP32 decomposition transform to current trajectory state."""

    if position == 0:
        result = torch.zeros_like(candidate_memory)
        identity_reference = result
    else:
        m32 = candidate_memory.float()
        mu32 = _expanded_mean(mean, candidate_memory)
        if control in {"zero", "real"}:
            controlled32 = m32
            identity_reference = candidate_memory
        elif control == "mu":
            controlled32 = mu32
            identity_reference = mu32.to(candidate_memory.dtype)
        elif control == "residual":
            controlled32 = m32 - mu32
            identity_reference = controlled32.to(candidate_memory.dtype)
        elif control == "residual_shuffle":
            permutation = coherent_permutation(candidate_memory.size(1), candidate_memory.device)
            controlled32 = (m32 - mu32)[:, permutation]
            identity_reference = controlled32.to(candidate_memory.dtype)
        elif control.startswith("alpha_real_"):
            alpha = float(control.removeprefix("alpha_real_"))
            controlled32 = mu32 + alpha * (m32 - mu32)
            identity_reference = candidate_memory if alpha == 1.0 else controlled32.to(candidate_memory.dtype)
        elif control.startswith("alpha_shuffle_"):
            alpha = float(control.removeprefix("alpha_shuffle_"))
            permutation = coherent_permutation(candidate_memory.size(1), candidate_memory.device)
            residual32 = (m32 - mu32)[:, permutation]
            controlled32 = mu32 + alpha * residual32
            identity_reference = (
                candidate_memory[:, permutation]
                if alpha == 1.0
                else controlled32.to(candidate_memory.dtype)
            )
        elif control == "independent_shuffle":
            permutations = independent_source_permutations(
                candidate_memory.size(1), candidate_memory.device
            )
            controlled32 = torch.stack(
                [
                    mu32[index]
                    + (m32[index, permutation] - mu32[index, permutation])
                    for index, permutation in enumerate(permutations)
                ],
                dim=0,
            )
            identity_reference = torch.stack(
                [
                    candidate_memory[index, permutation]
                    for index, permutation in enumerate(permutations)
                ],
                dim=0,
            )
        else:
            raise ValueError(f"unknown 2B5 control: {control}")
        result = controlled32.to(candidate_memory.dtype)
    if not return_identity:
        return result
    difference = (result.float() - identity_reference.float()).abs()
    return result, {
        "max_absolute_difference": difference.max().item(),
        "mean_absolute_difference": difference.mean().item(),
        "element_count": difference.numel(),
    }


def validate_checkpoint(path, label):
    spec = CHECKPOINTS[label]
    path = Path(path).resolve()
    digest = b2a.file_sha256(path)
    if digest != spec["sha256"]:
        raise SystemExit(f"{label} checkpoint SHA mismatch: {digest}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != digest:
        raise SystemExit(f"{label} checkpoint SHA sidecar mismatch")
    checkpoint = a0.torch_load(path, mmap=True)
    if checkpoint.get("schema") != spec["schema"]:
        raise SystemExit(f"{label} checkpoint schema mismatch")
    if checkpoint.get("training_state") != spec["training_state"]:
        raise SystemExit(f"{label} checkpoint training-state mismatch")
    if checkpoint.get("next_global_batch_sha256") != spec["next_global_batch_sha256"]:
        raise SystemExit(f"{label} next-batch lineage mismatch")
    return checkpoint, digest


def load_frozen_model(path, label, symbols, device):
    checkpoint, digest = validate_checkpoint(path, label)
    with torch.random.fork_rng(devices=[]):
        model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise SystemExit(f"{label} model is not frozen in eval mode")
    actual_subsets = {
        group: b2.state_subset_sha256(model, group)
        for group in ("base", "reader", "writers")
    }
    if label == "C0_2B2_5M":
        expected_subsets = CHECKPOINTS[label]["subset_sha256"]
    else:
        expected_subsets = {
            "base": checkpoint.get("frozen_base_sha256"),
            "reader": checkpoint.get("reader_sha256")
            if label == "C3_2B3_FINAL"
            else checkpoint.get("frozen_reader_sha256"),
            "writers": checkpoint.get("writer_sha256"),
        }
    if actual_subsets != expected_subsets or actual_subsets["base"] != BASE_SHA256:
        raise SystemExit(f"{label} strict-loaded subset hash mismatch: {actual_subsets}")
    metadata = {
        "label": label,
        "stage": CHECKPOINTS[label]["stage"],
        "path": str(Path(path).resolve()),
        "sha256": digest,
        "schema": checkpoint["schema"],
        "training_state": checkpoint["training_state"],
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "writer_lineage_tokens": CHECKPOINTS[label]["writer_lineage_tokens"],
        "subset_sha256": actual_subsets,
        "strict_load": True,
        "eval": True,
        "all_parameters_frozen": True,
    }
    del checkpoint
    return model, metadata


def load_validation_batches(symbols):
    canonical, calibration, canonical_hashes, calibration_hashes = b4.load_validation_batches(symbols)
    if b4.aggregate_payload_hash(calibration_hashes) != CALIBRATION_SHA256:
        raise SystemExit("calibration aggregate SHA mismatch")
    frozen = json.loads(FROZEN_CALIBRATION_MANIFEST.read_text())
    checks = {
        "calibration_batch_count": CALIBRATION_BATCHES,
        "calibration_batch_indices": [20, 21, 22, 23],
        "calibration_batch_payload_sha256": calibration_hashes,
        "calibration_aggregate_sha256": CALIBRATION_SHA256,
        "canonical_batch_count": CANONICAL_BATCHES,
        "calibration_evaluation_disjoint": True,
        "global_template_shape": [4, 768],
        "position_template_shape": [4, 1024, 768],
        "position_zero_exactly_zero": True,
    }
    for key, expected in checks.items():
        if frozen.get(key) != expected:
            raise SystemExit(f"frozen 2B4 calibration manifest {key} mismatch")
    return canonical, calibration, canonical_hashes, calibration_hashes, frozen


@torch.no_grad()
def build_generic_mean(model, calibration, calibration_hashes, device, label):
    combined = torch.zeros(
        len(SOURCE_DEPTHS), T, model.config.n_embd, device=device, dtype=torch.float32
    )
    started = time.perf_counter()
    batch_health = []
    for batch_index, (x_cpu, _y_cpu) in enumerate(calibration):
        x = x_cpu.to(device, non_blocking=True)
        state = model.init_recurrent_state(
            B,
            "masked_l1_topdown_self",
            device=device,
            dtype=torch.bfloat16,
            mask_depth=1,
        )
        position_sum = torch.zeros_like(combined)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for position in range(T):
                position_sum[:, position].add_(
                    state.feedback_memory[:, :, 0].float().sum(dim=1)
                )
                _, state = model.forward_step(
                    x[:, position], state, use_memory_writers=True
                )
        health = b4.cache_health(state, T)
        batch_health.append(health)
        if not (
            health["masked_cache_absence"]
            and health["unmasked_cache_expected_lengths"]
            and health["unmasked_cache_finite"]
        ):
            raise SystemExit(f"{label} calibration rollout cache-health failure")
        combined.add_(position_sum)
        del x, state, position_sum
    position_template = combined / float(CALIBRATION_BATCHES * B)
    if torch.count_nonzero(position_template[:, 0]).item() != 0:
        raise SystemExit(f"{label} calibration position zero is not exactly zero")
    generic_mean = position_template[:, 1:].mean(dim=1)
    if not torch.isfinite(generic_mean).all():
        raise SystemExit(f"{label} generic mean contains NaN/Inf")
    manifest = {
        "checkpoint_label": label,
        "source_checkpoint_sha256": CHECKPOINTS[label]["sha256"],
        "calibration_batch_payload_sha256": calibration_hashes,
        "calibration_aggregate_sha256": CALIBRATION_SHA256,
        "calibration_batch_count": CALIBRATION_BATCHES,
        "calibration_sequence_count": CALIBRATION_BATCHES * B,
        "token_geometry": [CALIBRATION_BATCHES, B, T],
        "usable_recurrent_positions_per_sequence": T - 1,
        "position_semantics": "feedback generated by positions 0..1022 and supplied at positions 1..1023; position zero excluded",
        "accumulation_dtype": "torch.float32",
        "division_dtype": "torch.float32",
        "stored_mean_dtype": str(generic_mean.dtype),
        "inference_cast_dtype": "torch.bfloat16",
        "tensor_shape": list(generic_mean.shape),
        "tensor_sha256": b4.tensor_sha256("generic_mean", generic_mean),
        "legacy_global_template_sha256": b4.tensor_sha256(
            "global_template", generic_mean
        ),
        "fixed_batch_order_fp32_sum": True,
        "calibration_wall_seconds": time.perf_counter() - started,
        "batch_cache_health": batch_health,
        "passed": True,
    }
    del combined, position_template
    torch.cuda.empty_cache()
    return generic_mean, manifest


@torch.no_grad()
def diagnostic_stream(
    model,
    x,
    y,
    control,
    generic_mean,
    capture_geometry=False,
    use_bf16=True,
):
    batch_size, sequence_length = x.shape
    state = model.init_recurrent_state(
        batch_size,
        "masked_l1_topdown_self",
        device=x.device,
        dtype=torch.bfloat16 if use_bf16 else model.transformer.wte.weight.dtype,
        mask_depth=1,
    )
    loss_sum = torch.zeros((), device=x.device)
    input_rms_sum = torch.zeros(len(SOURCE_DEPTHS), device=x.device)
    routing_sum = torch.zeros(len(SOURCE_DEPTHS), device=x.device)
    entropy_sum = torch.zeros((), device=x.device)
    topdown_sum = torch.zeros((), device=x.device)
    feedback_sum = torch.zeros((), device=x.device)
    memory_rms_sum = torch.zeros(len(SOURCE_DEPTHS), device=x.device)
    residual_rms_sum = torch.zeros(len(SOURCE_DEPTHS), device=x.device)
    cosine_sum = torch.zeros(len(SOURCE_DEPTHS), device=x.device)
    finite = torch.ones((), dtype=torch.bool, device=x.device)
    identity_max = 0.0
    identity_abs_sum = 0.0
    identity_count = 0
    started = time.perf_counter()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )
    with autocast:
        for position in range(sequence_length):
            memory, identity = controlled_memory(
                control,
                state.feedback_memory.detach(),
                generic_mean,
                position,
                return_identity=True,
            )
            identity_max = max(identity_max, identity["max_absolute_difference"])
            identity_abs_sum += (
                identity["mean_absolute_difference"] * identity["element_count"]
            )
            identity_count += identity["element_count"]
            input_rms = memory[:, :, 0].float().pow(2).mean(-1).sqrt()
            input_rms_sum += input_rms.sum(dim=1)
            if capture_geometry and position > 0:
                m32 = state.feedback_memory[:, :, 0].float()
                mu32 = generic_mean.float()[:, None, :]
                residual32 = m32 - mu32
                memory_rms_sum += m32.pow(2).mean(-1).sqrt().sum(dim=1)
                residual_rms_sum += residual32.pow(2).mean(-1).sqrt().sum(dim=1)
                cosine_sum += F.cosine_similarity(
                    m32, mu32.expand_as(m32), dim=-1, eps=1e-12
                ).sum(dim=1)
            input_state = replace(state, feedback_memory=memory.detach())
            logits, state, diagnostics = model.forward_step(
                x[:, position],
                input_state,
                feedback_gate_override=0.0 if control == "zero" else None,
                use_memory_writers=True,
                return_diagnostics=True,
            )
            loss_sum += F.cross_entropy(
                logits[:, 0], y[:, position], reduction="sum"
            )
            routing_sum += diagnostics["routing_weights"].float().sum(dim=(1, 2))
            entropy_sum += diagnostics["routing_entropy"].sum()
            topdown_sum += diagnostics["topdown_rms"].sum()
            feedback_sum += diagnostics["feedback_rms"].sum()
            finite &= (
                torch.isfinite(logits).all()
                & torch.isfinite(memory).all()
                & torch.isfinite(state.feedback_memory).all()
            )
    health = b4.cache_health(state, sequence_length)
    finite &= health["unmasked_cache_finite"]
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    elapsed = time.perf_counter() - started
    count = batch_size * sequence_length
    result = {
        "loss": (loss_sum / count).double().item(),
        "finite": bool(finite.item()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": count / elapsed,
        "input_memory_rms": {
            f"v{depth}": value
            for depth, value in zip(SOURCE_DEPTHS, (input_rms_sum / count).cpu().tolist())
        },
        "routing_weights": {
            f"v{depth}": value
            for depth, value in zip(SOURCE_DEPTHS, (routing_sum / count).cpu().tolist())
        },
        "routing_entropy": (entropy_sum / count).item(),
        "topdown_rms": (topdown_sum / count).item(),
        "feedback_rms": (feedback_sum / count).item(),
        "cache_health": health,
        "state_position": state.position,
        "identity_reconstruction": {
            "max_absolute_difference": identity_max,
            "mean_absolute_difference": identity_abs_sum / identity_count,
            "element_count": identity_count,
        },
    }
    if capture_geometry:
        geometry_count = batch_size * (sequence_length - 1)
        mu_rms = generic_mean.float().pow(2).mean(-1).sqrt().cpu().tolist()
        memory_rms = (memory_rms_sum / geometry_count).cpu().tolist()
        residual_rms = (residual_rms_sum / geometry_count).cpu().tolist()
        cosines = (cosine_sum / geometry_count).cpu().tolist()
        result["memory_geometry"] = {
            f"v{depth}": {
                "mu_rms": mu_value,
                "memory_rms": memory_value,
                "residual_rms": residual_value,
                "residual_to_memory_ratio": residual_value / memory_value,
                "mean_cosine_memory_mu": cosine_value,
            }
            for depth, mu_value, memory_value, residual_value, cosine_value in zip(
                SOURCE_DEPTHS, mu_rms, memory_rms, residual_rms, cosines
            )
        }
    return result


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise SystemExit(f"stale incomplete tensor artifact: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def progress_path(run_dir, label):
    return Path(run_dir) / "workers" / f"{label}_progress.json"


def load_progress(run_dir, label, implementation_commit, checkpoint_sha, mean_sha):
    path = progress_path(run_dir, label)
    expected = {
        "experiment": "2B5",
        "checkpoint_label": label,
        "implementation_git_commit": implementation_commit,
        "checkpoint_sha256": checkpoint_sha,
        "generic_mean_sha256": mean_sha,
    }
    if not path.is_file():
        return expected | {"rows": []}
    progress = json.loads(path.read_text())
    for key, value in expected.items():
        if progress.get(key) != value:
            raise SystemExit(f"{label} progress {key} mismatch")
    ids = [row["task_id"] for row in progress.get("rows", [])]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{label} progress contains duplicate tasks")
    return progress


def control_rows(rows, control):
    selected = sorted(
        (row for row in rows if row["control"] == control),
        key=lambda row: row["batch_index"],
    )
    return selected


def completed_control_mean(rows, control):
    selected = control_rows(rows, control)
    if len(selected) != CANONICAL_BATCHES:
        return None
    if [row["batch_index"] for row in selected] != list(range(CANONICAL_BATCHES)):
        raise SystemExit(f"{control} canonical order mismatch")
    return statistics.fmean(row["loss"] for row in selected)


def validate_available_regressions(label, rows):
    historical = CHECKPOINTS[label]["historical"]
    mappings = {
        "zero": "zero",
        "real": "real",
        "alpha_shuffle_1": "shuffle",
        "mu": "mu",
        "independent_shuffle": "independent",
    }
    checked = {}
    for control, reference_name in mappings.items():
        if reference_name not in historical:
            continue
        measured = completed_control_mean(rows, control)
        if measured is None:
            continue
        difference = measured - historical[reference_name]
        checked[control] = {
            "measured": measured,
            "expected": historical[reference_name],
            "difference": difference,
            "passed": abs(difference) <= 5e-6,
        }
        if not checked[control]["passed"]:
            raise SystemExit(f"{label} {control} historical regression failed: {checked[control]}")
    real = completed_control_mean(rows, "real")
    alpha_real = completed_control_mean(rows, "alpha_real_1")
    if real is not None and alpha_real is not None:
        difference = alpha_real - real
        checked["alpha_real_1_identity"] = {
            "measured": alpha_real,
            "reference": real,
            "difference": difference,
            "passed": abs(difference) <= 1e-5,
        }
        if abs(difference) > 1e-5:
            raise SystemExit(f"{label} alpha=1 real identity failed")
    if label == "C1_2B2A_10M":
        real_rows = control_rows(rows, "real")
        shuffled_rows = control_rows(rows, "alpha_shuffle_1")
        if len(real_rows) == len(shuffled_rows) == CANONICAL_BATCHES:
            paired = b2.paired_statistics(
                [row["loss"] for row in real_rows],
                [row["loss"] for row in shuffled_rows],
            )
            if paired["real_wins"] != 20:
                raise SystemExit("C1 real/shuffled 20/20 regression failed")
            checked["real_wins_20_of_20"] = True
    return checked


def execute_worker(args):
    require_git(clean=True)
    load_config()
    label = args.label
    if label not in CHECKPOINTS or args.rank != CHECKPOINT_LABELS.index(label):
        raise SystemExit("worker rank/checkpoint assignment mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        raise SystemExit("Experiment 2B5 worker requires the four-GPU pod")
    device = torch.device("cuda", args.rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != "NVIDIA A100-SXM4-80GB":
        raise SystemExit("each worker requires NVIDIA A100-SXM4-80GB")
    torch.cuda.reset_peak_memory_stats(device)
    implementation_commit = git_output("rev-parse", "HEAD")
    symbols = a0.support.load_training_symbols()
    canonical, calibration, canonical_hashes, calibration_hashes, frozen_calibration = (
        load_validation_batches(symbols)
    )
    model, checkpoint_metadata = load_frozen_model(
        args.checkpoint, label, symbols, device
    )
    before_hash = model_state_sha256(model)
    worker_started = time.perf_counter()
    generic_mean, mean_manifest = build_generic_mean(
        model, calibration, calibration_hashes, device, label
    )
    if label == "C3_2B3_FINAL" and (
        mean_manifest["legacy_global_template_sha256"] != C3_GLOBAL_TEMPLATE_SHA256
    ):
        raise SystemExit("C3 generic mean does not reproduce the frozen 2B4 template")
    mean_path = Path(args.run_dir) / "calibration_means" / f"{label}.pt"
    mean_payload = {
        "experiment": "2B5",
        "checkpoint_label": label,
        "source_checkpoint_sha256": CHECKPOINTS[label]["sha256"],
        "calibration_manifest_sha256": b2a.file_sha256(FROZEN_CALIBRATION_MANIFEST),
        "mean": generic_mean.detach().cpu(),
        "manifest": mean_manifest,
    }
    atomic_torch_save(mean_path, mean_payload)
    mean_manifest["artifact_path"] = str(mean_path.resolve())
    mean_manifest["artifact_sha256"] = b2a.file_sha256(mean_path)
    progress = load_progress(
        args.run_dir,
        label,
        implementation_commit,
        checkpoint_metadata["sha256"],
        mean_manifest["tensor_sha256"],
    )
    existing = {row["task_id"] for row in progress["rows"]}
    suite_started = time.perf_counter()
    for control in CONTROL_EXECUTION_ORDER:
        for batch_index, (x_cpu, y_cpu) in enumerate(canonical):
            task_id = f"{label}:{control}:{batch_index}"
            if task_id in existing:
                continue
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            result = diagnostic_stream(
                model,
                x,
                y,
                control,
                generic_mean,
                capture_geometry=control == "real",
                use_bf16=True,
            )
            row = {
                "task_id": task_id,
                "checkpoint_label": label,
                "control": control,
                "batch_index": batch_index,
                "payload_sha256": canonical_hashes[batch_index],
                **result,
            }
            progress["rows"].append(row)
            existing.add(task_id)
            b2a.write_json(progress_path(args.run_dir, label), progress)
            print(
                f"rank={args.rank} label={label} control={control} "
                f"batch={batch_index} loss={row['loss']:.10f} "
                f"wall={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
            del x, y
            torch.cuda.empty_cache()
        validate_available_regressions(label, progress["rows"])
    expected_ids = {
        f"{label}:{control}:{batch_index}"
        for control in CONTROL_EXECUTION_ORDER
        for batch_index in range(CANONICAL_BATCHES)
    }
    if existing != expected_ids:
        raise SystemExit(f"{label} canonical task coverage mismatch")
    after_hash = model_state_sha256(model)
    assert_zero_training_counts()
    regressions = validate_available_regressions(label, progress["rows"])
    all_finite = all(row["finite"] for row in progress["rows"])
    cache_health = all(
        row["cache_health"]["masked_cache_absence"]
        and row["cache_health"]["unmasked_cache_expected_lengths"]
        and row["cache_health"]["unmasked_cache_finite"]
        and row["cache_health"]["unmasked_cache_detached"]
        for row in progress["rows"]
    )
    summary = {
        "experiment": "2B5",
        "checkpoint_label": label,
        "rank": args.rank,
        "cuda_device": args.rank,
        "device_name": torch.cuda.get_device_name(device),
        "implementation_git_commit": implementation_commit,
        "checkpoint": checkpoint_metadata,
        "canonical_validation_sha256": b4.aggregate_payload_hash(canonical_hashes),
        "calibration_manifest_reused": frozen_calibration,
        "generic_mean": mean_manifest,
        "parameter_sha256_before": before_hash,
        "parameter_sha256_after": after_hash,
        "model_state_identical": before_hash == after_hash,
        "models_eval": not model.training,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "runtime_counts": dict(RUNTIME_COUNTS),
        "regressions": regressions,
        "task_count": len(progress["rows"]),
        "all_losses_and_memory_finite": all_finite,
        "all_cache_health_passed": cache_health,
        "performance": {
            "calibration_mean_construction_seconds": mean_manifest[
                "calibration_wall_seconds"
            ],
            "canonical_suite_wall_seconds_this_process": time.perf_counter()
            - suite_started,
            "canonical_task_elapsed_seconds_sum": sum(
                row["elapsed_seconds"] for row in progress["rows"]
            ),
            "worker_total_wall_seconds": time.perf_counter() - worker_started,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        },
        "passed": (
            before_hash == after_hash
            and all_finite
            and cache_health
            and all(value == 0 for value in RUNTIME_COUNTS.values())
        ),
    }
    summary_path = Path(args.run_dir) / "workers" / f"{label}.json"
    b2a.write_json(summary_path, summary)
    if not summary["passed"]:
        raise SystemExit(f"{label} worker integrity failure")
    print(f"EXPERIMENT_2B5_WORKER_PASS label={label}", flush=True)


@torch.no_grad()
def preflight_rollout(model, tokens, control, generic_mean, capture_position):
    state = model.init_recurrent_state(
        tokens.size(0),
        "masked_l1_topdown_self",
        device=tokens.device,
        dtype=model.transformer.wte.weight.dtype,
        mask_depth=1,
    )
    logits = []
    captured = None
    for position in range(tokens.size(1)):
        memory = controlled_memory(
            control, state.feedback_memory, generic_mean, position
        )
        logits_row, state = model.forward_step(
            tokens[:, position],
            replace(state, feedback_memory=memory),
            feedback_gate_override=0.0 if control == "zero" else None,
            use_memory_writers=True,
        )
        logits.append(logits_row.detach().cpu())
        if position + 1 == capture_position:
            captured = state.state_dict()
    return torch.cat(logits, dim=1), captured


def preflight(args):
    require_git(clean=True)
    load_config()
    if not torch.cuda.is_available():
        raise SystemExit("2B5 preflight requires CUDA")
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_name(device) != "NVIDIA A100-SXM4-80GB":
        raise SystemExit("2B5 preflight requires NVIDIA A100-SXM4-80GB")
    symbols = a0.support.load_training_symbols()
    model = b4.tiny_diagnostic_model(symbols, device)
    # The tiny constructor inherits the production zero-gate initialization.
    # Open only this synthetic probe's gate before its integrity hash so that
    # intentional donor-row coupling is observable in logits.
    with torch.no_grad():
        model.transformer.topdown_attnres.gate.fill_(1.0)
    before_hash = model_state_sha256(model)
    generator = torch.Generator(device=device)
    generator.manual_seed(DIAGNOSTIC_SEED)
    tokens = torch.randint(0, 32, (8, 12), device=device, generator=generator)
    generic_mean = torch.randn(
        len(SOURCE_DEPTHS), model.config.n_embd, device=device, generator=generator
    )
    controls = (
        "mu",
        "residual",
        "residual_shuffle",
        "alpha_real_0.25",
        "alpha_shuffle_0.25",
        "alpha_real_0.5",
        "alpha_shuffle_0.5",
        "alpha_real_1",
        "alpha_shuffle_1",
        "alpha_real_2",
        "alpha_shuffle_2",
        "independent_shuffle",
    )
    causality = {}
    row_isolation = {}
    prefix = 6
    for control in controls:
        altered = tokens.clone()
        altered[:, prefix:] = torch.flip(altered[:, prefix:], dims=(1,))
        first_logits, first_state = preflight_rollout(
            model, tokens, control, generic_mean, prefix
        )
        second_logits, second_state = preflight_rollout(
            model, altered, control, generic_mean, prefix
        )
        causality[control] = {
            "future_prefix_logits_bit_exact": torch.equal(
                first_logits[:, :prefix], second_logits[:, :prefix]
            ),
            "future_prefix_state_bit_exact": b2.b0.cache_payload_equal(
                first_state, second_state
            ),
            "finite": bool(torch.isfinite(first_logits).all().item()),
        }
        causality[control]["passed"] = all(causality[control].values())
        perturbed = tokens.clone()
        perturbed[0, 0] = (perturbed[0, 0] + 1) % model.config.vocab_size
        baseline_logits, _ = preflight_rollout(
            model, tokens, control, generic_mean, 2
        )
        perturbed_logits, _ = preflight_rollout(
            model, perturbed, control, generic_mean, 2
        )
        changed_by_position = []
        for position in (0, 1):
            difference = (
                baseline_logits[:, position] != perturbed_logits[:, position]
            ).any(dim=1)
            changed_by_position.append(
                torch.nonzero(difference).flatten().cpu().tolist()
            )
        if control == "independent_shuffle":
            expected_position_1 = [0, 1, 2, 3, 4]
        elif "shuffle" in control:
            expected_position_1 = [0, 1]
        else:
            expected_position_1 = [0]
        row_isolation[control] = {
            "perturbed_input_row": 0,
            "changed_output_rows_position_0": changed_by_position[0],
            "expected_changed_rows_position_0": [0],
            "changed_output_rows_position_1": changed_by_position[1],
            "expected_changed_rows_position_1": expected_position_1,
            "intentional_coupling_only": "shuffle" in control,
        }
        row_isolation[control]["passed"] = (
            row_isolation[control]["changed_output_rows_position_0"] == [0]
            and row_isolation[control]["changed_output_rows_position_1"]
            == expected_position_1
        )
    memory = torch.randn(
        len(SOURCE_DEPTHS), 8, 1, model.config.n_embd,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    _, real_identity = controlled_memory(
        "alpha_real_1", memory, generic_mean, 1, return_identity=True
    )
    _, shuffled_identity = controlled_memory(
        "alpha_shuffle_1", memory, generic_mean, 1, return_identity=True
    )
    _, independent_identity = controlled_memory(
        "independent_shuffle", memory, generic_mean, 1, return_identity=True
    )
    row_probe = torch.zeros_like(memory)
    row_probe[:, 0] = 1
    coherent = controlled_memory("alpha_shuffle_1", row_probe, generic_mean * 0, 1)
    independent = controlled_memory(
        "independent_shuffle", row_probe, generic_mean * 0, 1
    )
    row_coupling = {
        "coherent_changed_receivers": torch.nonzero(
            coherent.abs().sum(dim=(0, 2, 3))
        ).flatten().cpu().tolist(),
        "coherent_expected_receivers": [1],
        "independent_changed_receivers": [
            torch.nonzero(independent[source].abs().sum(dim=(1, 2)))
            .flatten()
            .cpu()
            .tolist()
            for source in range(len(SOURCE_DEPTHS))
        ],
        "independent_expected_receivers": [[1], [2], [3], [4]],
    }
    row_coupling["passed"] = (
        row_coupling["coherent_changed_receivers"]
        == row_coupling["coherent_expected_receivers"]
        and row_coupling["independent_changed_receivers"]
        == row_coupling["independent_expected_receivers"]
    )
    after_hash = model_state_sha256(model)
    assert_zero_training_counts()
    report = {
        "experiment": "2B5",
        "stage": "fp32_preflight",
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "device": torch.cuda.get_device_name(device),
        "future_causality": causality,
        "row_isolation_by_control": row_isolation,
        "row_coupling": row_coupling,
        "decomposition_identity": {
            "real_alpha_1": real_identity,
            "coherent_shuffle_alpha_1": shuffled_identity,
            "independent_shuffle_alpha_1": independent_identity,
        },
        "closed_loop_rollout_semantics": True,
        "position_zero_exactly_zero": all(
            torch.count_nonzero(
                controlled_memory(control, memory, generic_mean, 0)
            ).item()
            == 0
            for control in controls
        ),
        "coherent_permutation": coherent_permutation(8, device).cpu().tolist(),
        "independent_permutations": [
            value.cpu().tolist()
            for value in independent_source_permutations(8, device)
        ],
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "model_state_identical": before_hash == after_hash,
        "runtime_counts": dict(RUNTIME_COUNTS),
        "hellaswag_run": False,
    }
    report["passed"] = (
        all(row["passed"] for row in causality.values())
        and all(row["passed"] for row in row_isolation.values())
        and row_coupling["passed"]
        and all(
            row["max_absolute_difference"] == 0.0
            for row in report["decomposition_identity"].values()
        )
        and report["position_zero_exactly_zero"]
        and report["model_state_identical"]
        and all(value == 0 for value in RUNTIME_COUNTS.values())
    )
    b2a.write_json(Path(args.run_dir) / "PREFLIGHT.json", report)
    if not report["passed"]:
        raise SystemExit("Experiment 2B5 preflight failed")
    print("EXPERIMENT_2B5_PREFLIGHT_PASS", flush=True)


def paired_record(real_rows, shuffled_rows):
    real_losses = [row["loss"] for row in real_rows]
    shuffled_losses = [row["loss"] for row in shuffled_rows]
    return {
        "real_batch_losses": real_losses,
        "shuffled_batch_losses": shuffled_losses,
        **b2.paired_statistics(real_losses, shuffled_losses),
    }


def average_diagnostics(rows):
    return {
        "mean_routing_weight": {
            f"v{depth}": statistics.fmean(
                row["routing_weights"][f"v{depth}"] for row in rows
            )
            for depth in SOURCE_DEPTHS
        },
        "routing_entropy": statistics.fmean(row["routing_entropy"] for row in rows),
        "input_memory_rms": {
            f"v{depth}": statistics.fmean(
                row["input_memory_rms"][f"v{depth}"] for row in rows
            )
            for depth in SOURCE_DEPTHS
        },
        "topdown_rms": statistics.fmean(row["topdown_rms"] for row in rows),
        "feedback_rms": statistics.fmean(row["feedback_rms"] for row in rows),
    }


def alpha_control(kind, alpha):
    if alpha == 0:
        return "mu"
    formatted = str(int(alpha)) if float(alpha).is_integer() else str(alpha)
    return f"alpha_{kind}_{formatted}"


def classification_for(longitudinal, paired):
    c3 = longitudinal["C3_2B3_FINAL"]
    c0 = longitudinal["C0_2B2_5M"]
    c1 = longitudinal["C1_2B2A_10M"]
    gap1 = c3["specific_gap_alpha_1"]
    gap2 = c3["specific_gap_alpha_2"]
    if (
        gap1 < 0.010
        and gap2 >= 0.020
        and paired["C3_2B3_FINAL"]["alpha_2"]["real_wins"] >= 18
    ):
        return (
            "SEQUENCE RESIDUAL SURVIVES BUT IS UNDERWEIGHTED",
            "Section 46 precedence rule",
        )
    if (
        c3["generic_recovery_retention"] >= 0.90
        and gap1 <= 0.010
        and gap2 <= 0.015
        and max(c0["specific_gap_alpha_1"], c1["specific_gap_alpha_1"])
        - gap1
        >= 0.020
    ):
        return (
            "GENERIC COMPENSATION DISPLACES SEQUENCE MEMORY",
            "Section 45 frozen rule",
        )
    if c3["generic_recovery_retention"] >= 0.50 and gap1 >= 0.020:
        return "GENERIC AND SEQUENCE COMPONENTS COEXIST", "Section 47 frozen rule"
    if c3["generic_recovery_retention"] < 0.50:
        return (
            "GENERIC COMPONENT DOES NOT EXPLAIN RECURRENT RECOVERY",
            "Section 48 frozen rule",
        )
    return "DECOMPOSITION MIXED OR INCONCLUSIVE", "Section 49 frozen rule"


def aggregate(args):
    require_git(clean=False)
    config = load_config()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    preflight_report = json.loads((run_dir / "PREFLIGHT.json").read_text())
    if not preflight_report.get("passed"):
        raise SystemExit("passed 2B5 preflight is required")
    workers = {}
    progress = {}
    means = {}
    for label in CHECKPOINT_LABELS:
        worker_path = run_dir / "workers" / f"{label}.json"
        progress_file = progress_path(run_dir, label)
        if not worker_path.is_file() or not progress_file.is_file():
            raise SystemExit(f"missing completed worker artifacts for {label}")
        workers[label] = json.loads(worker_path.read_text())
        progress[label] = json.loads(progress_file.read_text())["rows"]
        if not workers[label].get("passed"):
            raise SystemExit(f"worker {label} did not pass")
        expected_ids = {
            f"{label}:{control}:{batch_index}"
            for control in CONTROL_EXECUTION_ORDER
            for batch_index in range(CANONICAL_BATCHES)
        }
        if {row["task_id"] for row in progress[label]} != expected_ids:
            raise SystemExit(f"{label} aggregate task coverage mismatch")
        mean_payload = a0.torch_load(
            run_dir / "calibration_means" / f"{label}.pt"
        )
        means[label] = mean_payload["mean"].float()
        if b4.tensor_sha256("generic_mean", means[label]) != workers[label][
            "generic_mean"
        ]["tensor_sha256"]:
            raise SystemExit(f"{label} generic mean tensor hash mismatch")

    checkpoint_manifest = {
        "experiment": "2B5",
        "assignments": {
            str(index): workers[label]["checkpoint"]
            for index, label in enumerate(CHECKPOINT_LABELS)
        },
        "all_checkpoint_shas_exact": all(
            workers[label]["checkpoint"]["sha256"] == CHECKPOINTS[label]["sha256"]
            for label in CHECKPOINT_LABELS
        ),
    }
    frozen_calibration = json.loads(FROZEN_CALIBRATION_MANIFEST.read_text())
    calibration_manifest = {
        "experiment": "2B5",
        "source": str(FROZEN_CALIBRATION_MANIFEST.relative_to(REPO_ROOT)),
        "source_manifest_sha256": b2a.file_sha256(FROZEN_CALIBRATION_MANIFEST),
        "source_manifest": frozen_calibration,
        "input_tokens_identical_for_all_checkpoints": True,
        "canonical_calibration_disjoint": True,
        "reused_exactly": all(
            workers[label]["calibration_manifest_reused"] == frozen_calibration
            for label in CHECKPOINT_LABELS
        ),
    }
    calibration_manifest["passed"] = (
        calibration_manifest["reused_exactly"]
        and frozen_calibration["calibration_aggregate_sha256"]
        == CALIBRATION_SHA256
        and frozen_calibration["calibration_evaluation_disjoint"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    means_dir = output_dir / "calibration_means"
    means_dir.mkdir(parents=True, exist_ok=True)
    generic_means_manifest = {"experiment": "2B5", "means": {}}
    for label in CHECKPOINT_LABELS:
        destination = means_dir / f"{label}.pt"
        payload = {
            "experiment": "2B5",
            "checkpoint_label": label,
            "source_checkpoint_sha256": CHECKPOINTS[label]["sha256"],
            "calibration_manifest_sha256": calibration_manifest[
                "source_manifest_sha256"
            ],
            "mean": means[label],
            "manifest": workers[label]["generic_mean"],
        }
        atomic_torch_save(destination, payload)
        row = dict(workers[label]["generic_mean"])
        row["artifact_path"] = str(destination.relative_to(output_dir))
        row["artifact_sha256"] = b2a.file_sha256(destination)
        generic_means_manifest["means"][label] = row

    longitudinal = {}
    alpha_sweep = {"experiment": "2B5", "checkpoints": {}}
    paired_losses = {"experiment": "2B5", "checkpoints": {}}
    memory_geometry = {"experiment": "2B5", "checkpoints": {}}
    routing = {"experiment": "2B5", "checkpoints": {}}
    identity = {}
    for label in CHECKPOINT_LABELS:
        rows = progress[label]
        by_control = {
            control: control_rows(rows, control) for control in CONTROL_EXECUTION_ORDER
        }
        losses = {
            control: statistics.fmean(row["loss"] for row in selected)
            for control, selected in by_control.items()
        }
        zero = losses["zero"]
        real = losses["real"]
        real_recovery = zero - real
        generic_recovery = zero - losses["mu"]
        residual_recovery = zero - losses["residual"]
        paired_losses["checkpoints"][label] = {}
        alpha_rows = []
        for alpha in (0.0, *ALPHAS):
            if alpha == 0:
                mu_losses = [row["loss"] for row in by_control["mu"]]
                paired = {
                    "real_batch_losses": mu_losses,
                    "shuffled_batch_losses": mu_losses,
                    "real_wins": 0,
                    "shuffled_wins": 0,
                    "ties": CANONICAL_BATCHES,
                    "mean": 0.0,
                    "median": 0.0,
                    "sample_standard_deviation": 0.0,
                    "minimum": 0.0,
                    "maximum": 0.0,
                    "differences": [0.0] * CANONICAL_BATCHES,
                }
                real_loss = shuffled_loss = losses["mu"]
            else:
                real_control = alpha_control("real", alpha)
                shuffled_control = alpha_control("shuffle", alpha)
                paired = paired_record(
                    by_control[real_control], by_control[shuffled_control]
                )
                real_loss = losses[real_control]
                shuffled_loss = losses[shuffled_control]
            paired_losses["checkpoints"][label][f"alpha_{alpha:g}"] = paired
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "real_loss": real_loss,
                    "shuffled_loss": shuffled_loss,
                    "specific_gap": shuffled_loss - real_loss,
                    "real_wins": paired["real_wins"],
                    "shuffled_wins": paired["shuffled_wins"],
                    "ties": paired["ties"],
                }
            )
        residual_paired = paired_record(
            by_control["residual"], by_control["residual_shuffle"]
        )
        paired_losses["checkpoints"][label]["residual_only"] = residual_paired
        sweep_by_alpha = {row["alpha"]: row for row in alpha_rows}
        gap1 = sweep_by_alpha[1.0]["specific_gap"]
        gap2 = sweep_by_alpha[2.0]["specific_gap"]
        longitudinal[label] = {
            "stage": CHECKPOINTS[label]["stage"],
            "writer_lineage_tokens": CHECKPOINTS[label]["writer_lineage_tokens"],
            "zero_loss": zero,
            "real_loss": real,
            "shuffled_loss": losses["alpha_shuffle_1"],
            "mu_only_loss": losses["mu"],
            "residual_only_loss": losses["residual"],
            "residual_only_shuffled_loss": losses["residual_shuffle"],
            "independent_source_residual_shuffle_loss": losses[
                "independent_shuffle"
            ],
            "real_recovery": real_recovery,
            "generic_recovery": generic_recovery,
            "residual_recovery": residual_recovery,
            "generic_recovery_retention": generic_recovery / real_recovery,
            "residual_recovery_retention": residual_recovery / real_recovery,
            "specific_gap_alpha_0.25": sweep_by_alpha[0.25]["specific_gap"],
            "specific_gap_alpha_0.5": sweep_by_alpha[0.5]["specific_gap"],
            "specific_gap_alpha_1": gap1,
            "specific_gap_alpha_2": gap2,
            "residual_only_specific_gap": losses["residual_shuffle"]
            - losses["residual"],
            "gap_amplification_2x_minus_1x": gap2 - gap1,
            "gap_2x_over_1x": gap2 / gap1 if abs(gap1) >= 0.001 else None,
        }
        alpha_sweep["checkpoints"][label] = alpha_rows
        memory_geometry["checkpoints"][label] = {
            f"v{depth}": {
                field: statistics.fmean(
                    row["memory_geometry"][f"v{depth}"][field]
                    for row in by_control["real"]
                )
                for field in (
                    "mu_rms",
                    "memory_rms",
                    "residual_rms",
                    "residual_to_memory_ratio",
                    "mean_cosine_memory_mu",
                )
            }
            for depth in SOURCE_DEPTHS
        }
        selected_routing = {
            "real": "real",
            "mu_only": "mu",
            "residual_only": "residual",
            "alpha_0.5_real": "alpha_real_0.5",
            "alpha_1_real": "alpha_real_1",
            "alpha_2_real": "alpha_real_2",
        }
        routing["checkpoints"][label] = {
            name: average_diagnostics(by_control[control])
            for name, control in selected_routing.items()
        }
        identity[label] = {
            control: {
                "memory_max_absolute_reconstruction_difference": max(
                    row["identity_reconstruction"]["max_absolute_difference"]
                    for row in by_control[control]
                ),
                "memory_mean_absolute_reconstruction_difference": sum(
                    row["identity_reconstruction"]["mean_absolute_difference"]
                    * row["identity_reconstruction"]["element_count"]
                    for row in by_control[control]
                )
                / sum(
                    row["identity_reconstruction"]["element_count"]
                    for row in by_control[control]
                ),
            }
            for control in (
                "alpha_real_1",
                "alpha_shuffle_1",
                "independent_shuffle",
            )
        }
        identity[label]["alpha_real_1"]["loss_difference_vs_real"] = (
            losses["alpha_real_1"] - losses["real"]
        )
        identity[label]["alpha_shuffle_1"][
            "loss_difference_vs_historical_shuffle"
        ] = losses["alpha_shuffle_1"] - CHECKPOINTS[label]["historical"]["shuffle"]

    generic_cosines = {"experiment": "2B5", "sources": {}}
    for source_index, depth in enumerate(SOURCE_DEPTHS):
        matrix = {}
        for left in CHECKPOINT_LABELS:
            matrix[left] = {}
            for right in CHECKPOINT_LABELS:
                matrix[left][right] = F.cosine_similarity(
                    means[left][source_index],
                    means[right][source_index],
                    dim=0,
                    eps=1e-12,
                ).item()
        generic_cosines["sources"][f"v{depth}"] = {
            "checkpoint_order": list(CHECKPOINT_LABELS),
            "cosine_matrix": matrix,
            "mu_rms": {
                label: means[label][source_index].pow(2).mean().sqrt().item()
                for label in CHECKPOINT_LABELS
            },
        }

    classification, rule = classification_for(
        longitudinal, paired_losses["checkpoints"]
    )
    performance = {
        "experiment": "2B5",
        "checkpoints": {
            label: workers[label]["performance"] for label in CHECKPOINT_LABELS
        },
        "total_four_gpu_wall_seconds": max(
            workers[label]["performance"]["calibration_mean_construction_seconds"]
            + workers[label]["performance"]["canonical_task_elapsed_seconds_sum"]
            for label in CHECKPOINT_LABELS
        ),
        "parallelization": config["parallelization"],
    }

    regression_checks = {}
    for label in CHECKPOINT_LABELS:
        row = longitudinal[label]
        historical = CHECKPOINTS[label]["historical"]
        regression_checks[label] = {
            "zero": abs(row["zero_loss"] - historical["zero"]) <= 5e-6,
            "real": abs(row["real_loss"] - historical["real"]) <= 5e-6,
            "shuffle": abs(row["shuffled_loss"] - historical["shuffle"]) <= 5e-6,
        }
    regression_checks["C3_2B3_FINAL"].update(
        {
            "mu": abs(
                longitudinal["C3_2B3_FINAL"]["mu_only_loss"]
                - CHECKPOINTS["C3_2B3_FINAL"]["historical"]["mu"]
            )
            <= 5e-6,
            "independent": abs(
                longitudinal["C3_2B3_FINAL"][
                    "independent_source_residual_shuffle_loss"
                ]
                - CHECKPOINTS["C3_2B3_FINAL"]["historical"]["independent"]
            )
            <= 5e-6,
        }
    )
    alpha_identity_pass = all(
        abs(identity[label]["alpha_real_1"]["loss_difference_vs_real"]) <= 1e-5
        and identity[label]["alpha_real_1"][
            "memory_max_absolute_reconstruction_difference"
        ]
        == 0.0
        and identity[label]["alpha_shuffle_1"][
            "memory_max_absolute_reconstruction_difference"
        ]
        == 0.0
        and identity[label]["independent_shuffle"][
            "memory_max_absolute_reconstruction_difference"
        ]
        == 0.0
        for label in CHECKPOINT_LABELS
    )
    integrity = {
        "2B2_5M_SHA_exact": workers["C0_2B2_5M"]["checkpoint"]["sha256"]
        == CHECKPOINTS["C0_2B2_5M"]["sha256"],
        "2B2A_10M_SHA_exact": workers["C1_2B2A_10M"]["checkpoint"]["sha256"]
        == CHECKPOINTS["C1_2B2A_10M"]["sha256"],
        "2B2A_15M_SHA_exact": workers["C2_2B2A_15M"]["checkpoint"]["sha256"]
        == CHECKPOINTS["C2_2B2A_15M"]["sha256"],
        "2B3_final_SHA_exact": workers["C3_2B3_FINAL"]["checkpoint"]["sha256"]
        == CHECKPOINTS["C3_2B3_FINAL"]["sha256"],
        "canonical_validation_hash_exact": all(
            workers[label]["canonical_validation_sha256"] == CANONICAL_SHA256
            for label in CHECKPOINT_LABELS
        ),
        "2B4_calibration_manifest_reused_exactly": calibration_manifest["passed"],
        "calibration_canonical_data_disjoint": frozen_calibration[
            "calibration_evaluation_disjoint"
        ],
        "C3_real_regression": regression_checks["C3_2B3_FINAL"]["real"],
        "C3_shuffled_regression": regression_checks["C3_2B3_FINAL"]["shuffle"],
        "C3_global_template_regression": regression_checks["C3_2B3_FINAL"]["mu"],
        "C3_independent_source_regression": regression_checks["C3_2B3_FINAL"][
            "independent"
        ],
        "C1_real_shuffled_regression": regression_checks["C1_2B2A_10M"]["real"]
        and regression_checks["C1_2B2A_10M"]["shuffle"]
        and paired_losses["checkpoints"]["C1_2B2A_10M"]["alpha_1"][
            "real_wins"
        ]
        == 20,
        "alpha_1_decomposition_identity": alpha_identity_pass,
        "alpha_0_mu_identity": all(
            alpha_sweep["checkpoints"][label][0]["specific_gap"] == 0.0
            and alpha_sweep["checkpoints"][label][0]["ties"] == 20
            for label in CHECKPOINT_LABELS
        ),
        "future_causality": all(
            row["passed"] for row in preflight_report["future_causality"].values()
        ),
        "row_isolation": all(
            row["passed"]
            for row in preflight_report["row_isolation_by_control"].values()
        )
        and preflight_report["row_coupling"]["passed"],
        "closed_loop_rollout_semantics": preflight_report[
            "closed_loop_rollout_semantics"
        ],
        "all_losses_finite": all(
            row["finite"]
            for label in CHECKPOINT_LABELS
            for row in progress[label]
        ),
        "all_memory_states_finite": all(
            workers[label]["all_losses_and_memory_finite"]
            for label in CHECKPOINT_LABELS
        ),
        "model_hashes_before_after_identical": all(
            workers[label]["model_state_identical"] for label in CHECKPOINT_LABELS
        ),
        "optimizer_objects_created_zero": all(
            workers[label]["runtime_counts"]["optimizer_objects_created"] == 0
            for label in CHECKPOINT_LABELS
        ),
        "scheduler_objects_created_zero": all(
            workers[label]["runtime_counts"]["scheduler_objects_created"] == 0
            for label in CHECKPOINT_LABELS
        ),
        "grad_scalers_created_zero": all(
            workers[label]["runtime_counts"]["grad_scalers_created"] == 0
            for label in CHECKPOINT_LABELS
        ),
        "optimizer_steps_zero": all(
            workers[label]["runtime_counts"]["optimizer_steps"] == 0
            for label in CHECKPOINT_LABELS
        ),
        "backward_calls_zero": all(
            workers[label]["runtime_counts"]["backward_calls"] == 0
            for label in CHECKPOINT_LABELS
        ),
        "parameter_updates_zero": all(
            workers[label]["parameter_sha256_before"]
            == workers[label]["parameter_sha256_after"]
            for label in CHECKPOINT_LABELS
        ),
        "hellaswag_not_run": True,
    }
    integrity["passed"] = all(integrity.values())
    if not integrity["passed"]:
        classification = "DECOMPOSITION UNSTABLE"
        rule = "Section 50 integrity-failure rule"

    decomposition_controls = {"experiment": "2B5", "checkpoints": {}}
    for label in CHECKPOINT_LABELS:
        row = longitudinal[label]
        real_recovery = row["real_recovery"]
        controls = {
            "zero": row["zero_loss"],
            "real": row["real_loss"],
            "mu_only": row["mu_only_loss"],
            "residual_only": row["residual_only_loss"],
            "residual_only_shuffled": row["residual_only_shuffled_loss"],
            "independent_source_residual_shuffle": row[
                "independent_source_residual_shuffle_loss"
            ],
        }
        decomposition_controls["checkpoints"][label] = {
            name: {
                "loss": loss,
                "delta_vs_real": loss - row["real_loss"],
                "recovery_from_zero": row["zero_loss"] - loss,
                "recovery_retained_vs_real": (
                    (row["zero_loss"] - loss) / real_recovery
                    if name != "zero"
                    else 0.0
                ),
            }
            for name, loss in controls.items()
        }

    result_summary = {
        "experiment": "2B5",
        "protocol": config["protocol"],
        "implementation_git_commit": preflight_report[
            "implementation_git_commit"
        ],
        "longitudinal": longitudinal,
        "classification": classification,
        "classification_rule": rule,
        "decomposition_controls": decomposition_controls["checkpoints"],
        "decomposition_identity": identity,
        "regression_checks": regression_checks,
        "integrity_passed": integrity["passed"],
        "training": {
            "optimizer_objects_created": 0,
            "scheduler_objects_created": 0,
            "grad_scalers_created": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parameter_updates": 0,
            "additional_training_tokens": 0,
        },
        "hellaswag_run": False,
        "passed": integrity["passed"],
    }
    hard_audit = {
        key: "PASS" if value else "FAIL"
        for key, value in integrity.items()
        if key != "passed"
    }
    hard_audit["HellaSwag"] = "NOT RUN"
    final_audit = {
        "experiment": "2B5",
        "implementation_git_commit": preflight_report[
            "implementation_git_commit"
        ],
        "source_checkpoints": checkpoint_manifest,
        "preflight": preflight_report,
        "workers": workers,
        "calibration": calibration_manifest,
        "decomposition_identity": identity,
        "regression_checks": regression_checks,
        "integrity": integrity,
        "hard_audit_checklist": hard_audit,
        "classification": classification,
        "classification_rule": rule,
        "hellaswag_run": False,
        "passed": integrity["passed"],
    }

    b2a.write_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    b2a.write_json(output_dir / "calibration_manifest.json", calibration_manifest)
    b2a.write_json(output_dir / "generic_means_manifest.json", generic_means_manifest)
    b2a.write_json(output_dir / "longitudinal_results.json", {"experiment": "2B5", "checkpoints": longitudinal})
    b2a.write_json(output_dir / "alpha_sweep.json", alpha_sweep)
    b2a.write_json(output_dir / "paired_losses.json", paired_losses)
    b2a.write_json(output_dir / "memory_geometry.json", memory_geometry)
    b2a.write_json(output_dir / "generic_mean_cosines.json", generic_cosines)
    b2a.write_json(output_dir / "routing_diagnostics.json", routing)
    b2a.write_json(output_dir / "decomposition_controls.json", decomposition_controls)
    b2a.write_json(output_dir / "performance.json", performance)
    b2a.write_json(output_dir / "result_summary.json", result_summary)
    b2a.write_json(output_dir / "FINAL_AUDIT.json", final_audit)
    b2a.write_json(output_dir / "PREFLIGHT.json", preflight_report)

    with (output_dir / "alpha_sweep.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "alpha",
                "real_loss",
                "shuffled_loss",
                "specific_gap",
                "real_wins",
                "shuffled_wins",
                "ties",
            ]
        )
        for label in CHECKPOINT_LABELS:
            for row in alpha_sweep["checkpoints"][label]:
                writer.writerow(
                    [
                        label,
                        row["alpha"],
                        row["real_loss"],
                        row["shuffled_loss"],
                        row["specific_gap"],
                        row["real_wins"],
                        row["shuffled_wins"],
                        row["ties"],
                    ]
                )
    generate_plots(output_dir, longitudinal, alpha_sweep)
    print(
        f"EXPERIMENT_2B5_AGGREGATION_PASS classification={classification}",
        flush=True,
    )
    if not integrity["passed"]:
        raise SystemExit("Experiment 2B5 final integrity audit failed")


def generate_plots(output_dir, longitudinal, alpha_sweep):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    stages = [CHECKPOINTS[label]["stage"] for label in CHECKPOINT_LABELS]
    trajectory_rows = [
        {
            "checkpoint": label,
            "stage": CHECKPOINTS[label]["stage"],
            "generic_recovery_retention": longitudinal[label][
                "generic_recovery_retention"
            ],
            "specific_gap_alpha_1": longitudinal[label]["specific_gap_alpha_1"],
            "specific_gap_alpha_2": longitudinal[label]["specific_gap_alpha_2"],
        }
        for label in CHECKPOINT_LABELS
    ]
    b2a.write_json(plots / "trajectory_data.json", trajectory_rows)
    with (plots / "trajectory_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    def line_plot(values, ylabel, filename):
        fig, axis = plt.subplots(figsize=(7.2, 4.5))
        axis.plot(stages, values, marker="o", linewidth=2)
        axis.set_xlabel("Training stage")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=160)
        plt.close(fig)

    line_plot(
        [row["generic_recovery_retention"] for row in trajectory_rows],
        "Generic recovery retention",
        "generic_recovery_retention.png",
    )
    line_plot(
        [row["specific_gap_alpha_1"] for row in trajectory_rows],
        "Specific gap (alpha=1)",
        "specific_gap_alpha_1.png",
    )
    line_plot(
        [row["specific_gap_alpha_2"] for row in trajectory_rows],
        "Specific gap (alpha=2)",
        "specific_gap_alpha_2.png",
    )

    alpha_data = []
    for label in CHECKPOINT_LABELS:
        alpha_data.extend(
            {"checkpoint": label, **row}
            for row in alpha_sweep["checkpoints"][label]
        )
    b2a.write_json(plots / "alpha_plot_data.json", alpha_data)
    with (plots / "alpha_plot_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alpha_data[0]))
        writer.writeheader()
        writer.writerows(alpha_data)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for label in CHECKPOINT_LABELS:
        rows = alpha_sweep["checkpoints"][label]
        axis.plot(
            [row["alpha"] for row in rows],
            [row["real_loss"] for row in rows],
            marker="o",
            label=CHECKPOINTS[label]["stage"],
        )
    axis.set_xlabel("Residual scale alpha")
    axis.set_ylabel("Real-residual loss")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "alpha_vs_real_loss.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for label in CHECKPOINT_LABELS:
        rows = alpha_sweep["checkpoints"][label]
        axis.plot(
            [row["alpha"] for row in rows],
            [row["specific_gap"] for row in rows],
            marker="o",
            label=CHECKPOINTS[label]["stage"],
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Residual scale alpha")
    axis.set_ylabel("Shuffled - real loss")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "alpha_vs_specific_gap.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--run-dir", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--run-dir", required=True)
    worker_parser.add_argument("--rank", type=int, required=True)
    worker_parser.add_argument("--label", choices=CHECKPOINT_LABELS, required=True)
    worker_parser.add_argument("--checkpoint", required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--run-dir", required=True)
    aggregate_parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(DIAGNOSTIC_SEED)
    np.random.seed(DIAGNOSTIC_SEED)
    torch.manual_seed(DIAGNOSTIC_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(DIAGNOSTIC_SEED)
    install_zero_training_guards()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "worker":
        execute_worker(args)
    elif args.command == "aggregate":
        aggregate(args)


if __name__ == "__main__":
    main()
