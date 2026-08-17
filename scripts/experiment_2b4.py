#!/usr/bin/env python3
"""Experiment 2B4: zero-optimizer memory-content and mask-depth diagnostics."""

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
import torch.distributed as dist
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2a0 as a0  # noqa: E402
import experiment_2b2 as b2  # noqa: E402
import experiment_2b2a as b2a  # noqa: E402
import experiment_2b3 as b3  # noqa: E402


BRANCH = "experiment-2b4-memory-content-mask-depth"
FROZEN_2B3_TAG = "experiment-2b3-joint-20m-final"
FROZEN_2B3_COMMIT = "f8eb37d639d8509fb76673aa02db5c01ae9f7fd7"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2b4_diagnostic_4gpu.json"
WORLD_SIZE = 4
B = 64
T = 1024
SOURCE_DEPTHS = (16, 17, 20, 24)
CANONICAL_BATCHES = 20
CALIBRATION_BATCHES = 4
CANONICAL_SHA256 = "3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb"
FINAL_2B3_SHA256 = "7797f349905e344934bd7d2475cf61b332ef9053cb0bc1a44f450fc24249c65b"
FINAL_2B3_NEXT_SHA256 = "7f6d8da5044e9f485492712373fda12d09eb4ccec60ab8fdee812519d05869a7"
WRITER_10M_SHA256 = "de5e04f817dcfa5dd8a4dcc6e503ec86d8545d558d837b517c7259917218dff3"
WRITER_10M_NEXT_SHA256 = "ddbb966eff17ddabd102ce4706ccace0e23973f98803478b392b8c4e5f9d32f3"
BASE_SHA256 = "1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd"
DIAGNOSTIC_SEED = 20260817
FULL_CONTEXT_REFERENCE = 4.078654408454895

PART_A_CONTROLS = (
    "zero",
    "real",
    "coherent_shuffle",
    "independent_source_shuffle",
    "batch_mean",
    "position_template",
    "global_template",
    "norm_random",
    "lag8",
    "lag32",
)
PART_A_BASELINES = (
    "full_context",
    "masked_l1_no_feedback",
    "zero",
    "real",
    "coherent_shuffle",
)
PART_A_NEW = tuple(name for name in PART_A_CONTROLS if name not in PART_A_BASELINES)
PART_B_CONTROLS = ("zero", "real", "coherent_shuffle")
CONTROL_LABELS = {
    "zero": "zero",
    "real": "real",
    "coherent_shuffle": "coherent shuffled",
    "independent_source_shuffle": "independent-source shuffled",
    "batch_mean": "leave-one-out batch mean",
    "position_template": "position-conditioned template",
    "global_template": "global template",
    "norm_random": "norm-matched random",
    "lag8": "same-sequence lag-8",
    "lag32": "same-sequence lag-32",
}

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
        raise SystemExit(f"Experiment 2B4 requires branch {BRANCH}")
    if git_output("rev-parse", f"{FROZEN_2B3_TAG}^{{}}") != FROZEN_2B3_COMMIT:
        raise SystemExit("frozen Experiment 2B3 tag target mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FROZEN_2B3_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing execution requires a clean worktree")


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        "protocol": "exp2b4_zero_optimizer_memory_content_mask_depth_v1",
        "world_size": WORLD_SIZE,
        "batch_sequences": B,
        "sequence_length": T,
        "canonical_batches": CANONICAL_BATCHES,
        "calibration_batches": CALIBRATION_BATCHES,
        "canonical_validation_sha256": CANONICAL_SHA256,
        "final_2b3_checkpoint_sha256": FINAL_2B3_SHA256,
        "writer_10m_checkpoint_sha256": WRITER_10M_SHA256,
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "mask_depths": [1, 2, 3, 4],
        "lag_controls": [8, 32],
        "strong_support_gap_gain": 0.010,
        "strong_support_real_wins": 18,
        "generic_recovery_material_gain": 0.010,
        "hellaswag": "forbidden",
        "training": "forbidden",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f"config {key} mismatch: {config.get(key)} != {value}")
    if config.get("part_a_controls") != list(PART_A_CONTROLS):
        raise SystemExit("Part-A control list mismatch")
    return config


def install_zero_optimizer_guards():
    """Turn forbidden training actions into immediate hard failures."""

    original_optimizer_init = torch.optim.Optimizer.__init__
    original_optimizer_step = torch.optim.Optimizer.step
    original_tensor_backward = torch.Tensor.backward
    original_autograd_backward = torch.autograd.backward
    original_autograd_grad = torch.autograd.grad

    def forbidden_optimizer_init(*args, **kwargs):
        RUNTIME_COUNTS["optimizer_objects_created"] += 1
        raise RuntimeError("Experiment 2B4 forbids optimizer construction")

    def forbidden_optimizer_step(*args, **kwargs):
        RUNTIME_COUNTS["optimizer_steps"] += 1
        raise RuntimeError("Experiment 2B4 forbids optimizer steps")

    def forbidden_backward(*args, **kwargs):
        RUNTIME_COUNTS["backward_calls"] += 1
        raise RuntimeError("Experiment 2B4 forbids backward calls")

    torch.optim.Optimizer.__init__ = forbidden_optimizer_init
    torch.optim.Optimizer.step = forbidden_optimizer_step
    torch.Tensor.backward = forbidden_backward
    torch.autograd.backward = forbidden_backward
    torch.autograd.grad = forbidden_backward

    scheduler_types = []
    for name in ("LRScheduler", "_LRScheduler"):
        scheduler_type = getattr(torch.optim.lr_scheduler, name, None)
        if scheduler_type is not None and scheduler_type not in scheduler_types:
            scheduler_types.append(scheduler_type)
    scheduler_originals = []
    for scheduler_type in scheduler_types:
        original = scheduler_type.__init__

        def forbidden_scheduler_init(*args, **kwargs):
            RUNTIME_COUNTS["scheduler_objects_created"] += 1
            raise RuntimeError("Experiment 2B4 forbids scheduler construction")

        scheduler_type.__init__ = forbidden_scheduler_init
        scheduler_originals.append((scheduler_type, original))

    scaler_types = []
    for candidate in (
        getattr(torch.amp, "GradScaler", None),
        getattr(torch.cuda.amp, "GradScaler", None),
    ):
        if candidate is not None and candidate not in scaler_types:
            scaler_types.append(candidate)
    scaler_originals = []
    for scaler_type in scaler_types:
        original = scaler_type.__init__

        def forbidden_scaler_init(*args, **kwargs):
            RUNTIME_COUNTS["grad_scalers_created"] += 1
            raise RuntimeError("Experiment 2B4 forbids GradScaler construction")

        scaler_type.__init__ = forbidden_scaler_init
        scaler_originals.append((scaler_type, original))

    return {
        "optimizer_init": original_optimizer_init,
        "optimizer_step": original_optimizer_step,
        "tensor_backward": original_tensor_backward,
        "autograd_backward": original_autograd_backward,
        "autograd_grad": original_autograd_grad,
        "scheduler_originals": scheduler_originals,
        "scaler_originals": scaler_originals,
    }


def assert_zero_training_counts():
    if any(RUNTIME_COUNTS.values()):
        raise SystemExit(f"forbidden training activity detected: {RUNTIME_COUNTS}")


def model_parameter_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        tensor = value.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def aggregate_payload_hash(payload_hashes):
    digest = hashlib.sha256()
    for value in payload_hashes:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def coherent_permutation(batch_size, device):
    return torch.arange(batch_size, device=device).roll(1)


def independent_source_permutations(batch_size, device):
    if batch_size < len(SOURCE_DEPTHS) + 1:
        raise ValueError("independent-source shuffling requires batch size at least five")
    permutations = tuple(
        torch.arange(batch_size, device=device).roll(source_index + 1)
        for source_index in range(len(SOURCE_DEPTHS))
    )
    expected = torch.arange(batch_size, device=device)
    if len({tuple(row.cpu().tolist()) for row in permutations}) != len(SOURCE_DEPTHS):
        raise RuntimeError("source permutations are not distinct")
    if any(torch.any(row == expected) for row in permutations):
        raise RuntimeError("source permutation is not fixed-point-free")
    return permutations


def norm_random_memory(real_memory, batch_index, position):
    if position == 0:
        return torch.zeros_like(real_memory)
    random_rows = []
    modulus = 2**63 - 1
    for source_index in range(len(SOURCE_DEPTHS)):
        seed = (
            DIAGNOSTIC_SEED
            + 1_000_003 * int(batch_index)
            + 9_176 * int(position)
            + 104_729 * source_index
        ) % modulus
        generator = torch.Generator(device=real_memory.device)
        generator.manual_seed(seed)
        # Row identity selects its fixed contiguous substream within [B, C].
        noise = torch.randn(
            real_memory.size(1),
            real_memory.size(3),
            device=real_memory.device,
            dtype=torch.float32,
            generator=generator,
        )
        noise = noise * torch.rsqrt(
            noise.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-30)
        )
        rms = real_memory[source_index, :, 0].float().pow(2).mean(-1, keepdim=True).sqrt()
        random_value = noise * rms
        # A second normalization removes the first-pass floating-point drift.
        actual_rms = random_value.pow(2).mean(dim=-1, keepdim=True).sqrt()
        random_value = random_value * (rms / actual_rms.clamp_min(1e-30))
        random_rows.append(random_value.unsqueeze(1))
    return torch.stack(random_rows, dim=0)


def supplied_memory(
    control,
    state,
    position,
    batch_index,
    templates=None,
    lag_history=None,
):
    real_memory = state.feedback_memory.detach()
    if control in {"real", "zero"}:
        return real_memory
    if control == "coherent_shuffle":
        return real_memory[:, coherent_permutation(real_memory.size(1), real_memory.device)]
    if control == "independent_source_shuffle":
        permutations = independent_source_permutations(
            real_memory.size(1), real_memory.device
        )
        return torch.stack(
            [real_memory[index, permutation] for index, permutation in enumerate(permutations)],
            dim=0,
        )
    if control == "batch_mean":
        return (
            real_memory.sum(dim=1, keepdim=True) - real_memory
        ) / (real_memory.size(1) - 1)
    if control == "position_template":
        if templates is None:
            raise ValueError("position template is unavailable")
        value = templates["position"][:, position].to(
            device=real_memory.device, dtype=real_memory.dtype
        )
        return value[:, None, None, :].expand_as(real_memory)
    if control == "global_template":
        if templates is None:
            raise ValueError("global template is unavailable")
        if position == 0:
            return torch.zeros_like(real_memory)
        value = templates["global"].to(
            device=real_memory.device, dtype=real_memory.dtype
        )
        return value[:, None, None, :].expand_as(real_memory)
    if control == "norm_random":
        return norm_random_memory(real_memory, batch_index, position)
    if control in {"lag8", "lag32"}:
        lag = int(control.removeprefix("lag"))
        if lag_history is None:
            raise ValueError("lag history is unavailable")
        if position < lag:
            return torch.zeros_like(real_memory)
        return lag_history[position - lag]
    raise ValueError(f"unknown memory control: {control}")


def cache_health(state, expected_length):
    masked = state.kv_caches[: state.mask_depth]
    unmasked = state.kv_caches[state.mask_depth :]
    finite = True
    detached = True
    lengths = []
    for cache in unmasked:
        if cache is None:
            finite = False
            detached = False
            lengths.append(None)
            continue
        key, value = cache.prefix()
        finite &= bool(torch.isfinite(key).all().item() and torch.isfinite(value).all().item())
        detached &= (
            key.grad_fn is None
            and value.grad_fn is None
            and not key.requires_grad
            and not value.requires_grad
        )
        lengths.append(cache.length)
    return {
        "mask_depth": state.mask_depth,
        "masked_cache_absence": all(cache is None for cache in masked),
        "unmasked_cache_lengths": lengths,
        "unmasked_cache_expected_lengths": all(value == expected_length for value in lengths),
        "unmasked_cache_finite": bool(finite),
        "unmasked_cache_detached": bool(detached),
    }


@torch.no_grad()
def diagnostic_stream(
    model,
    x,
    y,
    control,
    mask_depth,
    batch_index,
    templates=None,
    use_bf16=True,
    capture_logits=False,
    capture_position=None,
):
    batch_size, sequence_length = x.shape
    state = model.init_recurrent_state(
        batch_size,
        "masked_l1_topdown_self",
        device=x.device,
        dtype=torch.bfloat16 if use_bf16 else model.transformer.wte.weight.dtype,
        mask_depth=mask_depth,
    )
    lag_history = []
    loss_sum = torch.zeros((), device=x.device)
    restricted_sums = {8: torch.zeros((), device=x.device), 32: torch.zeros((), device=x.device)}
    input_rms_sum = torch.zeros(len(SOURCE_DEPTHS), device=x.device)
    routing_sum = torch.zeros(len(SOURCE_DEPTHS), device=x.device)
    entropy_sum = torch.zeros((), device=x.device)
    topdown_sum = torch.zeros((), device=x.device)
    feedback_sum = torch.zeros((), device=x.device)
    finite = torch.ones((), dtype=torch.bool, device=x.device)
    logits_rows = []
    captured_state = None
    started = time.perf_counter()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )
    with autocast:
        for position in range(sequence_length):
            memory = supplied_memory(
                control,
                state,
                position,
                batch_index,
                templates=templates,
                lag_history=lag_history,
            ).detach()
            input_rms = memory[:, :, 0].float().pow(2).mean(-1).sqrt()
            input_rms_sum += input_rms.sum(dim=1)
            input_state = replace(state, feedback_memory=memory)
            logits, state, diagnostics = model.forward_step(
                x[:, position],
                input_state,
                feedback_gate_override=0.0 if control == "zero" else None,
                use_memory_writers=True,
                return_diagnostics=True,
            )
            token_loss = F.cross_entropy(
                logits[:, 0], y[:, position], reduction="sum"
            )
            loss_sum += token_loss
            for cutoff in restricted_sums:
                if position >= cutoff:
                    restricted_sums[cutoff] += token_loss
            routing_sum += diagnostics["routing_weights"].float().sum(dim=(1, 2))
            entropy_sum += diagnostics["routing_entropy"].sum()
            topdown_sum += diagnostics["topdown_rms"].sum()
            feedback_sum += diagnostics["feedback_rms"].sum()
            finite &= torch.isfinite(logits).all() & torch.isfinite(state.feedback_memory).all()
            if control in {"lag8", "lag32"}:
                lag_history.append(state.feedback_memory.detach().clone())
            if capture_logits:
                logits_rows.append(logits.detach().float().cpu())
            if capture_position is not None and position + 1 == capture_position:
                captured_state = state.state_dict()
    health = cache_health(state, sequence_length)
    finite &= health["unmasked_cache_finite"]
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    elapsed = time.perf_counter() - started
    count = batch_size * sequence_length
    result = {
        "loss": (loss_sum / count).double().item(),
        "restricted_losses": {
            str(cutoff): (
                restricted_sums[cutoff] / (batch_size * (sequence_length - cutoff))
            ).double().item()
            if sequence_length > cutoff
            else None
            for cutoff in restricted_sums
        },
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
    }
    if capture_logits:
        result["logits"] = torch.cat(logits_rows, dim=1)
    if captured_state is not None:
        result["captured_state"] = captured_state
    return result


def validate_checkpoint(path, kind):
    path = Path(path).resolve()
    expected_digest = FINAL_2B3_SHA256 if kind == "final_2b3" else WRITER_10M_SHA256
    digest = b2a.file_sha256(path)
    if digest != expected_digest:
        raise SystemExit(f"{kind} checkpoint SHA mismatch: {digest}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != digest:
        raise SystemExit(f"{kind} checkpoint SHA sidecar mismatch")
    checkpoint = a0.torch_load(path, mmap=True)
    if kind == "final_2b3":
        if checkpoint.get("schema") != b3.CHECKPOINT_SCHEMA:
            raise SystemExit("final 2B3 checkpoint schema mismatch")
        expected_state = {
            "writer_lineage_updates": 38,
            "writer_lineage_tokens": 19_922_944,
            "joint_local_updates": 9,
            "joint_training_tokens": 4_718_592,
            "fineweb_lineage_completed_update": 535,
            "kind": "2b3_final",
        }
        if checkpoint.get("training_state") != expected_state:
            raise SystemExit("final 2B3 checkpoint training-state mismatch")
        if checkpoint.get("next_global_batch_sha256") != FINAL_2B3_NEXT_SHA256:
            raise SystemExit("final 2B3 next-batch lineage mismatch")
        expected_subsets = {
            "base": checkpoint.get("frozen_base_sha256"),
            "reader": checkpoint.get("reader_sha256"),
            "writers": checkpoint.get("writer_sha256"),
        }
    elif kind == "writer_10m":
        if checkpoint.get("schema") != b2a.CHECKPOINT_SCHEMA:
            raise SystemExit("2B2A 10M checkpoint schema mismatch")
        expected_state = {
            "writer_updates": 20,
            "writer_training_tokens": 10_485_760,
            "fineweb_lineage_completed_update": 517,
            "kind": "2b2a_10m",
        }
        if checkpoint.get("training_state") != expected_state:
            raise SystemExit("2B2A 10M checkpoint training-state mismatch")
        if checkpoint.get("next_global_batch_sha256") != WRITER_10M_NEXT_SHA256:
            raise SystemExit("2B2A 10M next-batch lineage mismatch")
        expected_subsets = {
            "base": checkpoint.get("frozen_base_sha256"),
            "reader": checkpoint.get("frozen_reader_sha256"),
            "writers": checkpoint.get("writer_sha256"),
        }
    else:
        raise ValueError(kind)
    if expected_subsets["base"] != BASE_SHA256 or not all(expected_subsets.values()):
        raise SystemExit(f"{kind} checkpoint subset metadata mismatch")
    return checkpoint, digest, expected_subsets


def load_frozen_model(path, kind, symbols, device):
    checkpoint, digest, expected_subsets = validate_checkpoint(path, kind)
    with torch.random.fork_rng(devices=[]):
        model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise SystemExit(f"{kind} model is not frozen in eval mode")
    actual_subsets = {
        group: b2.state_subset_sha256(model, group)
        for group in ("base", "reader", "writers")
    }
    if actual_subsets != expected_subsets:
        raise SystemExit(
            f"{kind} strict-loaded subset hash mismatch: {actual_subsets}"
        )
    metadata = {
        "path": str(Path(path).resolve()),
        "sha256": digest,
        "schema": checkpoint["schema"],
        "training_state": checkpoint["training_state"],
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "subset_sha256": actual_subsets,
        "strict_load": True,
        "eval": True,
        "all_parameters_frozen": True,
    }
    del checkpoint
    return model, metadata


def tiny_diagnostic_model(symbols, device):
    config = symbols["GPTConfig"](
        block_size=48,
        vocab_size=32,
        n_layer=12,
        n_head=2,
        n_embd=16,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
        enable_memory_writers=True,
        memory_writer_rank=8,
        memory_writer_init_seed=DIAGNOSTIC_SEED,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(991_337)
        model = symbols["GPT"](config)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


@torch.no_grad()
def fp32_real_rollout(model, tokens, mask_depth=None, capture_position=None):
    kwargs = {}
    if mask_depth is not None:
        kwargs["mask_depth"] = mask_depth
    state = model.init_recurrent_state(
        tokens.size(0),
        "masked_l1_topdown_self",
        device=tokens.device,
        dtype=model.transformer.wte.weight.dtype,
        **kwargs,
    )
    logits = []
    captured = None
    for position in range(tokens.size(1)):
        row, state = model.forward_step(
            tokens[:, position], state, use_memory_writers=True
        )
        logits.append(row.detach().float().cpu())
        if capture_position is not None and position + 1 == capture_position:
            captured = state.state_dict()
    return torch.cat(logits, dim=1), state, captured


@torch.no_grad()
def serialization_test(model, tokens, mask_depth, split):
    state = model.init_recurrent_state(
        tokens.size(0),
        "masked_l1_topdown_self",
        device=tokens.device,
        dtype=model.transformer.wte.weight.dtype,
        mask_depth=mask_depth,
    )
    for position in range(split):
        _, state = model.forward_step(
            tokens[:, position], state, use_memory_writers=True
        )
    restored = model.load_recurrent_state(state.state_dict())
    direct_logits = []
    restored_logits = []
    for position in range(split, tokens.size(1)):
        direct, state = model.forward_step(
            tokens[:, position], state, use_memory_writers=True
        )
        resumed, restored = model.forward_step(
            tokens[:, position], restored, use_memory_writers=True
        )
        direct_logits.append(direct.detach().float().cpu())
        restored_logits.append(resumed.detach().float().cpu())
    return {
        "schema": state.state_dict()["schema"],
        "continuation_logits_bit_exact": torch.equal(
            torch.cat(direct_logits, 1), torch.cat(restored_logits, 1)
        ),
        "final_state_bit_exact": b2.b0.cache_payload_equal(
            state.state_dict(), restored.state_dict()
        ),
    }


def preflight(args):
    require_git(clean=True)
    load_config()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("Experiment 2B4 preflight requires CUDA")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_name(0) != "NVIDIA A100-SXM4-80GB":
        raise SystemExit("Experiment 2B4 preflight requires an A100-SXM4-80GB")
    device = torch.device("cuda", 0)
    symbols = a0.support.load_training_symbols()
    model = tiny_diagnostic_model(symbols, device)
    before_hash = model_parameter_sha256(model)
    generator = torch.Generator(device="cpu").manual_seed(44_019)
    tokens = torch.randint(0, 32, (5, 40), generator=generator).to(device)
    targets = tokens.roll(-1, dims=1)
    prefix = 36

    default_logits, default_state, _ = fp32_real_rollout(model, tokens, None)
    explicit_logits, explicit_state, _ = fp32_real_rollout(model, tokens, 1)
    d1_equivalence = {
        "logits_bit_exact": torch.equal(default_logits, explicit_logits),
        "state_bit_exact": b2.b0.cache_payload_equal(
            default_state.state_dict(), explicit_state.state_dict()
        ),
    }
    d1_equivalence["passed"] = all(d1_equivalence.values())

    depth_tests = {}
    for depth in (1, 2, 3, 4):
        altered = tokens.clone()
        altered[:, prefix:] = (altered[:, prefix:] + 7) % 32
        logits_a, state_a, prefix_state_a = fp32_real_rollout(
            model, tokens, depth, capture_position=prefix
        )
        logits_b, state_b, prefix_state_b = fp32_real_rollout(
            model, altered, depth, capture_position=prefix
        )
        isolated_a = tokens[:2, :12].clone()
        isolated_b = isolated_a.clone()
        isolated_b[1] = (isolated_b[1] + 11) % 32
        isolated_logits_a, isolated_state_a, _ = fp32_real_rollout(
            model, isolated_a, depth
        )
        isolated_logits_b, isolated_state_b, _ = fp32_real_rollout(
            model, isolated_b, depth
        )
        fresh_logits_a, fresh_state_a, _ = fp32_real_rollout(
            model, tokens[:2, :12], depth
        )
        fresh_logits_b, fresh_state_b, _ = fp32_real_rollout(
            model, tokens[:2, :12], depth
        )
        health = cache_health(state_a, tokens.size(1))
        serialization = serialization_test(model, tokens[:2, :12], depth, 6)
        row = {
            "masked_cache_absence": health["masked_cache_absence"],
            "unmasked_cache_expected_lengths": health[
                "unmasked_cache_expected_lengths"
            ],
            "unmasked_cache_finite": health["unmasked_cache_finite"],
            "future_prefix_logits_bit_exact": torch.equal(
                logits_a[:, :prefix], logits_b[:, :prefix]
            ),
            "future_prefix_state_bit_exact": b2.b0.cache_payload_equal(
                prefix_state_a, prefix_state_b
            ),
            "row_isolation_logits_bit_exact": torch.equal(
                isolated_logits_a[0], isolated_logits_b[0]
            ),
            "row_isolation_state_bit_exact": b2.b0.cache_payload_equal(
                isolated_state_a.state_dict(), isolated_state_b.state_dict(), row=0
            ),
            "fresh_sequence_logits_bit_exact": torch.equal(
                fresh_logits_a, fresh_logits_b
            ),
            "fresh_sequence_state_bit_exact": b2.b0.cache_payload_equal(
                fresh_state_a.state_dict(), fresh_state_b.state_dict()
            ),
            "serialization": serialization,
            "all_outputs_finite": bool(
                torch.isfinite(logits_a).all().item()
                and torch.isfinite(logits_b).all().item()
            ),
        }
        row["passed"] = all(
            value
            for key, value in row.items()
            if key not in {"serialization", "passed"}
        ) and all(
            serialization[key]
            for key in ("continuation_logits_bit_exact", "final_state_bit_exact")
        )
        depth_tests[str(depth)] = row

    position_template = torch.randn(
        len(SOURCE_DEPTHS), 40, 16, generator=generator
    ).to(device)
    position_template[:, 0].zero_()
    global_template = torch.randn(
        len(SOURCE_DEPTHS), 16, generator=generator
    ).to(device)
    templates = {"position": position_template, "global": global_template}
    control_causality = {}
    for control in (
        "coherent_shuffle",
        "independent_source_shuffle",
        "batch_mean",
        "position_template",
        "global_template",
        "norm_random",
        "lag8",
        "lag32",
    ):
        altered = tokens.clone()
        altered[:, prefix:] = (altered[:, prefix:] + 13) % 32
        first = diagnostic_stream(
            model,
            tokens,
            targets,
            control,
            1,
            7,
            templates=templates,
            use_bf16=False,
            capture_logits=True,
            capture_position=prefix,
        )
        second = diagnostic_stream(
            model,
            altered,
            targets,
            control,
            1,
            7,
            templates=templates,
            use_bf16=False,
            capture_logits=True,
            capture_position=prefix,
        )
        row = {
            "future_prefix_logits_bit_exact": torch.equal(
                first["logits"][:, :prefix], second["logits"][:, :prefix]
            ),
            "future_prefix_state_bit_exact": b2.b0.cache_payload_equal(
                first["captured_state"], second["captured_state"]
            ),
            "finite": first["finite"] and second["finite"],
        }
        row["passed"] = all(row.values())
        control_causality[control] = row

    after_hash = model_parameter_sha256(model)
    assert_zero_training_counts()
    integrity = {
        "fp32": True,
        "d1_default_explicit_equivalence": d1_equivalence["passed"],
        "all_depth_tests_passed": all(row["passed"] for row in depth_tests.values()),
        "all_memory_control_causality_passed": all(
            row["passed"] for row in control_causality.values()
        ),
        "model_parameter_hash_identical": before_hash == after_hash,
        "optimizer_objects_created": RUNTIME_COUNTS["optimizer_objects_created"],
        "optimizer_steps": RUNTIME_COUNTS["optimizer_steps"],
        "backward_calls": RUNTIME_COUNTS["backward_calls"],
        "hellaswag_not_run": True,
    }
    integrity["passed"] = (
        all(
            integrity[key]
            for key in (
                "fp32",
                "d1_default_explicit_equivalence",
                "all_depth_tests_passed",
                "all_memory_control_causality_passed",
                "model_parameter_hash_identical",
                "hellaswag_not_run",
            )
        )
        and integrity["optimizer_objects_created"] == 0
        and integrity["optimizer_steps"] == 0
        and integrity["backward_calls"] == 0
    )
    report = {
        "experiment": "2B4",
        "stage": "fp32_preflight",
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "device": torch.cuda.get_device_name(0),
        "d1_equivalence": d1_equivalence,
        "mask_depth_tests": depth_tests,
        "memory_control_causality": control_causality,
        "independent_source_permutations": {
            f"v{depth}": int(index + 1)
            for index, depth in enumerate(SOURCE_DEPTHS)
        },
        "parameter_sha256_before": before_hash,
        "parameter_sha256_after": after_hash,
        "integrity": integrity,
        "passed": integrity["passed"],
    }
    if not report["passed"]:
        raise SystemExit("Experiment 2B4 preflight failed")
    b2a.write_json(Path(args.run_dir) / "PREFLIGHT.json", report)
    print("EXPERIMENT_2B4_PREFLIGHT_PASS", flush=True)


def load_validation_batches(symbols):
    loader = b2.validation_loader(symbols)
    canonical = []
    calibration = []
    canonical_hashes = []
    calibration_hashes = []
    for batch_index in range(CANONICAL_BATCHES + CALIBRATION_BATCHES):
        x, y = loader.next_batch()
        payload_hash = a0.batch_payload_hash(x, y)
        if batch_index < CANONICAL_BATCHES:
            canonical.append((x, y))
            canonical_hashes.append(payload_hash)
        else:
            calibration.append((x, y))
            calibration_hashes.append(payload_hash)
    digest = aggregate_payload_hash(canonical_hashes)
    if digest != CANONICAL_SHA256:
        raise SystemExit(f"canonical validation SHA mismatch: {digest}")
    if set(canonical_hashes) & set(calibration_hashes):
        raise SystemExit("calibration data overlap canonical validation data")
    return canonical, calibration, canonical_hashes, calibration_hashes


def rank_slotted_sum(local_value, rank):
    if local_value.dtype != torch.float32:
        raise ValueError("rank-slotted evaluation sum requires FP32")
    slots = torch.zeros(
        (WORLD_SIZE, *local_value.shape),
        device=local_value.device,
        dtype=torch.float32,
    )
    slots[rank].copy_(local_value)
    dist.all_reduce(slots, op=dist.ReduceOp.SUM)
    result = slots[0].clone()
    for source_rank in range(1, WORLD_SIZE):
        result.add_(slots[source_rank])
    return result


def tensor_sha256(name, tensor):
    value = tensor.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def build_calibration_templates(model, calibration, calibration_hashes, rank, device):
    x_cpu, _y_cpu = calibration[rank]
    x = x_cpu.to(device, non_blocking=True)
    state = model.init_recurrent_state(
        B,
        "masked_l1_topdown_self",
        device=device,
        dtype=torch.bfloat16,
        mask_depth=1,
    )
    position_sum = torch.zeros(
        len(SOURCE_DEPTHS), T, model.config.n_embd, device=device, dtype=torch.float32
    )
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(T):
            position_sum[:, position] += state.feedback_memory[:, :, 0].float().sum(dim=1)
            _, state = model.forward_step(
                x[:, position], state, use_memory_writers=True
            )
    health = cache_health(state, T)
    if not (
        health["masked_cache_absence"]
        and health["unmasked_cache_expected_lengths"]
        and health["unmasked_cache_finite"]
    ):
        raise SystemExit("calibration rollout cache-health failure")
    combined = rank_slotted_sum(position_sum, rank)
    position_template = combined / float(WORLD_SIZE * B)
    global_template = position_template[:, 1:].mean(dim=1)
    if torch.count_nonzero(position_template[:, 0]).item() != 0:
        raise SystemExit("calibration position-zero template is not exactly zero")
    hashes = {
        "position_template_sha256": tensor_sha256(
            "position_template", position_template
        ),
        "global_template_sha256": tensor_sha256("global_template", global_template),
    }
    gathered_hashes = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_hashes, hashes)
    if any(value != hashes for value in gathered_hashes):
        raise SystemExit("calibration templates differ across ranks")
    manifest = {
        "source": "four validation batches immediately after canonical batch 19",
        "canonical_batch_count": CANONICAL_BATCHES,
        "calibration_batch_count": CALIBRATION_BATCHES,
        "calibration_batch_indices": list(
            range(CANONICAL_BATCHES, CANONICAL_BATCHES + CALIBRATION_BATCHES)
        ),
        "calibration_batch_payload_sha256": calibration_hashes,
        "calibration_aggregate_sha256": aggregate_payload_hash(calibration_hashes),
        "calibration_evaluation_disjoint": True,
        "position_template_shape": list(position_template.shape),
        "global_template_shape": list(global_template.shape),
        "position_zero_exactly_zero": True,
        "rank_slotted_fp32_sum_fixed_rank_order": True,
        "per_rank_calibration_batch": {str(index): CANONICAL_BATCHES + index for index in range(4)},
        "template_hashes": hashes,
        "rank_wall_seconds": time.perf_counter() - started,
        "passed": True,
    }
    del x, position_sum, combined
    torch.cuda.empty_cache()
    return {"position": position_template, "global": global_template}, manifest


def progress_path(run_dir, rank):
    return Path(run_dir) / "progress" / f"rank{rank}.json"


def load_progress(run_dir, rank, implementation_commit):
    path = progress_path(run_dir, rank)
    expected = {
        "schema": "exp2b4_progress_v1",
        "rank": rank,
        "implementation_git_commit": implementation_commit,
        "final_2b3_checkpoint_sha256": FINAL_2B3_SHA256,
        "writer_10m_checkpoint_sha256": WRITER_10M_SHA256,
        "canonical_validation_sha256": CANONICAL_SHA256,
    }
    if not path.is_file():
        return expected | {"rows": []}
    progress = json.loads(path.read_text())
    for key, value in expected.items():
        if progress.get(key) != value:
            raise SystemExit(f"rank {rank} progress {key} mismatch")
    if not isinstance(progress.get("rows"), list):
        raise SystemExit("invalid progress row collection")
    ids = [row.get("task_id") for row in progress["rows"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate task IDs in progress")
    return progress


def task_id(task):
    return ":".join(
        str(task[key])
        for key in ("part", "model", "control", "mask_depth", "batch_index")
    )


def make_tasks(part, model, controls, depths=(1,)):
    return [
        {
            "part": part,
            "model": model,
            "control": control,
            "mask_depth": depth,
            "batch_index": batch_index,
        }
        for depth in depths
        for control in controls
        for batch_index in range(CANONICAL_BATCHES)
    ]


@torch.no_grad()
def execute_task(task, models, canonical, canonical_hashes, templates, device):
    x_cpu, y_cpu = canonical[task["batch_index"]]
    x = x_cpu.to(device, non_blocking=True)
    y = y_cpu.to(device, non_blocking=True)
    model = models[task["model"]]
    control = task["control"]
    started = time.perf_counter()
    if control in {"full_context", "masked_l1_no_feedback"}:
        loss = b2.b0.parallel_loss(model, x, y, control)
        result = {
            "loss": loss,
            "finite": math.isfinite(loss),
            "elapsed_seconds": time.perf_counter() - started,
            "cache_health": None,
        }
    else:
        result = diagnostic_stream(
            model,
            x,
            y,
            control,
            task["mask_depth"],
            task["batch_index"],
            templates=templates,
            use_bf16=True,
        )
    row = dict(task)
    row.update(result)
    row["task_id"] = task_id(task)
    row["payload_sha256"] = canonical_hashes[task["batch_index"]]
    del x, y
    torch.cuda.empty_cache()
    return row


def gathered_progress_rows(progress):
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, progress["rows"])
    rows = [row for group in gathered for row in group]
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate distributed progress task IDs")
    return rows


def execute_phase(
    phase_name,
    tasks,
    models,
    canonical,
    canonical_hashes,
    templates,
    progress,
    run_dir,
    rank,
    device,
):
    existing = {row["task_id"] for row in progress["rows"]}
    for task_index, task in enumerate(tasks):
        if task_index % WORLD_SIZE != rank:
            continue
        identifier = task_id(task)
        if identifier in existing:
            continue
        row = execute_task(
            task, models, canonical, canonical_hashes, templates, device
        )
        progress["rows"].append(row)
        existing.add(identifier)
        b2a.write_json(progress_path(run_dir, rank), progress)
        print(
            f"rank={rank} phase={phase_name} task={identifier} "
            f"loss={row['loss']:.10f} wall={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    dist.barrier()
    rows = gathered_progress_rows(progress)
    wanted = {task_id(task) for task in tasks}
    selected = [row for row in rows if row["task_id"] in wanted]
    if len(selected) != len(tasks) or {row["task_id"] for row in selected} != wanted:
        raise SystemExit(f"{phase_name} task coverage mismatch")
    return selected


def mean_losses(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["control"], row["mask_depth"]), []).append(row)
    return {
        f"{control}:d{depth}": statistics.fmean(row["loss"] for row in values)
        for (control, depth), values in grouped.items()
    }


def distributed_verdict(rank, payload):
    values = [payload if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    if not values[0].get("passed"):
        raise SystemExit(values[0].get("error", "distributed verification failed"))
    return values[0]


def select_rows(rows, part, model, control, depth):
    selected = sorted(
        (
            row
            for row in rows
            if row["part"] == part
            and row["model"] == model
            and row["control"] == control
            and row["mask_depth"] == depth
        ),
        key=lambda row: row["batch_index"],
    )
    if len(selected) != CANONICAL_BATCHES:
        raise SystemExit(
            f"missing result rows for {part}/{model}/{control}/d{depth}: "
            f"{len(selected)}"
        )
    if [row["batch_index"] for row in selected] != list(range(CANONICAL_BATCHES)):
        raise SystemExit("canonical result rows are not in exact batch order")
    return selected


def average_stream_diagnostics(rows):
    return {
        "input_memory_rms": {
            f"v{depth}": statistics.fmean(
                row["input_memory_rms"][f"v{depth}"] for row in rows
            )
            for depth in SOURCE_DEPTHS
        },
        "routing_weights": {
            f"v{depth}": statistics.fmean(
                row["routing_weights"][f"v{depth}"] for row in rows
            )
            for depth in SOURCE_DEPTHS
        },
        "routing_entropy": statistics.fmean(row["routing_entropy"] for row in rows),
        "topdown_rms": statistics.fmean(row["topdown_rms"] for row in rows),
        "feedback_rms": statistics.fmean(row["feedback_rms"] for row in rows),
    }


def aggregate_part_a(rows):
    real_rows = select_rows(rows, "A", "final_2b3", "real", 1)
    zero_rows = select_rows(rows, "A", "final_2b3", "zero", 1)
    real_losses = [row["loss"] for row in real_rows]
    zero_loss = statistics.fmean(row["loss"] for row in zero_rows)
    real_loss = statistics.fmean(real_losses)
    denominator = zero_loss - real_loss
    controls = {}
    paired_payload = {}
    for control in PART_A_CONTROLS:
        selected = select_rows(rows, "A", "final_2b3", control, 1)
        losses = [row["loss"] for row in selected]
        loss = statistics.fmean(losses)
        paired = b2.paired_statistics(real_losses, losses)
        row = {
            "label": CONTROL_LABELS[control],
            "loss": loss,
            "delta_vs_real": loss - real_loss,
            "recovery": zero_loss - loss,
            "fraction_of_real_recovery_retained": (
                (zero_loss - loss) / denominator if denominator > 0 else None
            ),
            "real_wins": paired["real_wins"],
            "control_wins": paired["shuffled_wins"],
            "ties": paired["ties"],
            "paired_statistics_vs_real": paired,
            "batch_losses": losses,
            "routing_state_diagnostics": average_stream_diagnostics(selected),
        }
        if control in {"lag8", "lag32"}:
            cutoff = control.removeprefix("lag")
            restricted_losses = [
                value["restricted_losses"][cutoff] for value in selected
            ]
            real_restricted = [
                value["restricted_losses"][cutoff] for value in real_rows
            ]
            restricted_paired = b2.paired_statistics(
                real_restricted, restricted_losses
            )
            row["restricted"] = {
                "target_positions": f"t >= {cutoff}",
                "loss": statistics.fmean(restricted_losses),
                "real_loss": statistics.fmean(real_restricted),
                "specific_delta": statistics.fmean(restricted_losses)
                - statistics.fmean(real_restricted),
                "paired_statistics_vs_real": restricted_paired,
                "batch_losses": restricted_losses,
                "real_batch_losses": real_restricted,
            }
        controls[control] = row
        paired_payload[control] = {
            "real_batch_losses": real_losses,
            "control_batch_losses": losses,
            "statistics": paired,
        }
    mechanism = {
        "exact_sequence_identity_gap": controls["coherent_shuffle"]["delta_vs_real"],
        "cross_source_coherence_value": controls["independent_source_shuffle"]["loss"]
        - controls["coherent_shuffle"]["loss"],
        "position_template_recovery_fraction": controls["position_template"][
            "fraction_of_real_recovery_retained"
        ],
        "global_template_recovery_fraction": controls["global_template"][
            "fraction_of_real_recovery_retained"
        ],
        "norm_random_recovery_fraction": controls["norm_random"][
            "fraction_of_real_recovery_retained"
        ],
        "lag8_specific_delta": controls["lag8"]["restricted"]["specific_delta"],
        "lag32_specific_delta": controls["lag32"]["restricted"]["specific_delta"],
        "lag8_minus_cross_sequence_identity_gap": controls["lag8"]["restricted"][
            "specific_delta"
        ]
        - controls["coherent_shuffle"]["delta_vs_real"],
    }
    return {
        "zero_loss": zero_loss,
        "real_loss": real_loss,
        "real_recovery": denominator,
        "controls": controls,
        "mechanism_metrics": mechanism,
        "optional_lag32_run": True,
    }, paired_payload


def gap_trajectory_label(gaps):
    values = [gaps[str(depth)] for depth in (1, 2, 3, 4)]
    differences = [right - left for left, right in zip(values, values[1:])]
    if all(value >= 0 for value in differences):
        return "monotonically increasing"
    if sum(value > 0 for value in differences) >= 2 and values[-1] > values[0]:
        return "mostly increasing"
    if max(values) - min(values) <= 0.005:
        return "flat"
    if sum(value < 0 for value in differences) >= 2 and values[-1] < values[0]:
        return "decreasing"
    return "non-monotonic"


def aggregate_part_b(rows, full_context_loss, config, integrity_passed=True):
    depths = {}
    paired_payload = {}
    for depth in (1, 2, 3, 4):
        selected = {
            control: select_rows(rows, "B", "writer_10m", control, depth)
            for control in PART_B_CONTROLS
        }
        losses = {
            control: [row["loss"] for row in values]
            for control, values in selected.items()
        }
        means = {
            control: statistics.fmean(values)
            for control, values in losses.items()
        }
        paired = b2.paired_statistics(losses["real"], losses["coherent_shuffle"])
        mask_damage = means["zero"] - full_context_loss
        real_recovery = means["zero"] - means["real"]
        specific_gap = means["coherent_shuffle"] - means["real"]
        depths[str(depth)] = {
            "mask_depth": depth,
            "zero_loss": means["zero"],
            "real_loss": means["real"],
            "shuffled_loss": means["coherent_shuffle"],
            "mask_damage": mask_damage,
            "real_recovery": real_recovery,
            "real_recovery_fraction": (
                real_recovery / mask_damage if mask_damage > 0 else None
            ),
            "specific_gap": specific_gap,
            "specific_recovery_fraction": (
                specific_gap / mask_damage if mask_damage > 0 else None
            ),
            "specific_share_of_recovery": (
                specific_gap / real_recovery if real_recovery > 0 else None
            ),
            "real_wins": paired["real_wins"],
            "shuffled_wins": paired["shuffled_wins"],
            "ties": paired["ties"],
            "paired_statistics": paired,
            "batch_losses": losses,
            "routing_state_diagnostics": {
                control: average_stream_diagnostics(selected[control])
                for control in PART_B_CONTROLS
            },
        }
        paired_payload[str(depth)] = {
            "real_batch_losses": losses["real"],
            "shuffled_batch_losses": losses["coherent_shuffle"],
            "statistics": paired,
        }
    depth1_gap = depths["1"]["specific_gap"]
    gap_minus_depth1 = {
        str(depth): depths[str(depth)]["specific_gap"] - depth1_gap
        for depth in (2, 3, 4)
    }
    strong_depths = [
        depth
        for depth in (2, 3, 4)
        if depths[str(depth)]["specific_gap"]
        >= depth1_gap + config["strong_support_gap_gain"]
        and depths[str(depth)]["real_wins"] >= config["strong_support_real_wins"]
        and depths[str(depth)]["real_recovery"] > 0
        and integrity_passed
    ]
    strong = bool(strong_depths)
    gaps = {str(depth): depths[str(depth)]["specific_gap"] for depth in (1, 2, 3, 4)}
    recovery_gain = max(
        depths[str(depth)]["real_recovery"] for depth in (2, 3, 4)
    ) - depths["1"]["real_recovery"]
    if not integrity_passed:
        classification = "MASK-DEPTH DIAGNOSTIC UNSTABLE"
    elif strong:
        classification = "MASK PRESSURE INCREASES SEQUENCE-SPECIFIC MEMORY"
    elif recovery_gain >= config["generic_recovery_material_gain"]:
        classification = "MASK PRESSURE INCREASES GENERIC RECURRENT UTILITY ONLY"
    elif any(depths[str(depth)]["real_recovery"] <= 0 for depth in (2, 3, 4)):
        classification = "MASK PRESSURE DESTROYS RECURRENT UTILITY"
    else:
        classification = "MASK PRESSURE DOES NOT INCREASE RECURRENT DEPENDENCE"
    best_depth = max((2, 3, 4), key=lambda value: depths[str(value)]["specific_gap"])
    return {
        "full_context_loss": full_context_loss,
        "depths": depths,
        "gap_minus_depth1": gap_minus_depth1,
        "gap_trajectory": gaps,
        "trajectory_classification": gap_trajectory_label(gaps),
        "strong_support": strong,
        "strong_support_depths": strong_depths,
        "best_deeper_depth": best_depth,
        "real_recovery_gain_over_depth1": recovery_gain,
        "classification": classification,
    }, paired_payload


def aggregate_part_c(rows, part_a, best_depth):
    result = {"status": "RUN", "best_depth": best_depth, "depths": {}}
    for depth in (1, best_depth):
        if depth == 1:
            selected = {
                control: select_rows(rows, "A", "final_2b3", control, 1)
                for control in PART_B_CONTROLS
            }
        else:
            selected = {
                control: select_rows(rows, "C", "final_2b3", control, depth)
                for control in PART_B_CONTROLS
            }
        losses = {
            control: [row["loss"] for row in values]
            for control, values in selected.items()
        }
        means = {control: statistics.fmean(value) for control, value in losses.items()}
        paired = b2.paired_statistics(losses["real"], losses["coherent_shuffle"])
        result["depths"][str(depth)] = {
            "zero_loss": means["zero"],
            "real_loss": means["real"],
            "shuffled_loss": means["coherent_shuffle"],
            "specific_gap": means["coherent_shuffle"] - means["real"],
            "real_recovery": means["zero"] - means["real"],
            "real_wins": paired["real_wins"],
            "paired_statistics": paired,
            "batch_losses": losses,
        }
    result["gap_increase_over_depth1"] = (
        result["depths"][str(best_depth)]["specific_gap"]
        - result["depths"]["1"]["specific_gap"]
    )
    result["depth1_reference_from_part_a"] = (
        result["depths"]["1"]["specific_gap"]
        == part_a["controls"]["coherent_shuffle"]["delta_vs_real"]
    )
    return result


def write_csv_artifacts(output_dir, part_a, part_b):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "part_a_controls.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "control",
                "loss",
                "delta_vs_real",
                "recovery",
                "fraction_of_real_recovery_retained",
                "real_wins",
            ]
        )
        for control in PART_A_CONTROLS:
            row = part_a["controls"][control]
            writer.writerow(
                [
                    control,
                    row["loss"],
                    row["delta_vs_real"],
                    row["recovery"],
                    row["fraction_of_real_recovery_retained"],
                    row["real_wins"],
                ]
            )


def streaming_rows_pass(rows):
    stream_rows = [
        row
        for row in rows
        if row["control"] not in {"full_context", "masked_l1_no_feedback"}
    ]
    return all(
        row["finite"]
        and row["cache_health"]["masked_cache_absence"]
        and row["cache_health"]["unmasked_cache_expected_lengths"]
        and row["cache_health"]["unmasked_cache_finite"]
        and row["cache_health"]["unmasked_cache_detached"]
        for row in stream_rows
    )


def check_expected_means(means, expected, tolerances):
    mismatches = {}
    for name, value in expected.items():
        actual = means.get(name)
        tolerance = tolerances[name]
        if actual is None or abs(actual - value) > tolerance:
            mismatches[name] = {
                "actual": actual,
                "expected": value,
                "tolerance": tolerance,
            }
    return {
        "means": means,
        "expected": expected,
        "tolerances": tolerances,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def evaluate(args):
    require_git(clean=True)
    config = load_config()
    implementation_commit = git_output("rev-parse", "HEAD")
    preflight_path = Path(args.preflight).resolve()
    if not preflight_path.is_file():
        raise SystemExit("canonical evaluation requires the completed preflight artifact")
    preflight_report = json.loads(preflight_path.read_text())
    if (
        preflight_report.get("passed") is not True
        or preflight_report.get("implementation_git_commit") != implementation_commit
    ):
        raise SystemExit("preflight did not pass for this exact implementation commit")

    rank, local_rank = b2a.init_distributed()
    try:
        rank_seed = b2a.seed_rank(rank)
        device = torch.device("cuda", local_rank)
        symbols = a0.support.load_training_symbols()
        canonical, calibration, canonical_hashes, calibration_hashes = (
            load_validation_batches(symbols)
        )
        validation_identity = {
            "canonical_sha256": aggregate_payload_hash(canonical_hashes),
            "canonical_batch_hashes": canonical_hashes,
            "calibration_batch_hashes": calibration_hashes,
        }
        gathered_identity = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_identity, validation_identity)
        if any(value != validation_identity for value in gathered_identity):
            raise SystemExit("validation/calibration identity differs across ranks")

        final_model, final_metadata = load_frozen_model(
            args.final_2b3_checkpoint, "final_2b3", symbols, device
        )
        writer_model, writer_metadata = load_frozen_model(
            args.writer_10m_checkpoint, "writer_10m", symbols, device
        )
        models = {"final_2b3": final_model, "writer_10m": writer_model}
        model_metadata = {
            "final_2b3": final_metadata,
            "writer_10m": writer_metadata,
        }
        before_hashes = {
            name: model_parameter_sha256(model) for name, model in models.items()
        }
        startup_identity = {
            "metadata": model_metadata,
            "parameter_sha256_before": before_hashes,
        }
        gathered_startup = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_startup, startup_identity)
        if any(value != startup_identity for value in gathered_startup):
            raise SystemExit("strict-loaded model identity differs across ranks")

        progress = load_progress(args.run_dir, rank, implementation_commit)

        part_a_baseline_tasks = make_tasks(
            "A", "final_2b3", PART_A_BASELINES, depths=(1,)
        )
        part_a_baseline_rows = execute_phase(
            "part_a_baseline",
            part_a_baseline_tasks,
            models,
            canonical,
            canonical_hashes,
            None,
            progress,
            args.run_dir,
            rank,
            device,
        )
        if rank == 0:
            part_a_baseline_check = check_expected_means(
                mean_losses(part_a_baseline_rows),
                {
                    "full_context:d1": FULL_CONTEXT_REFERENCE,
                    "masked_l1_no_feedback:d1": 5.973674488067627,
                    "zero:d1": 5.9736480712890625,
                    "real:d1": 4.814190459251404,
                    "coherent_shuffle:d1": 4.817693614959717,
                },
                {
                    "full_context:d1": 5e-4,
                    "masked_l1_no_feedback:d1": 5e-4,
                    "zero:d1": 5e-6,
                    "real:d1": 5e-6,
                    "coherent_shuffle:d1": 5e-6,
                },
            )
            verdict = part_a_baseline_check | {
                "error": "Part-A established control reproduction failed"
            }
        else:
            part_a_baseline_check = None
            verdict = None
        part_a_baseline_check = distributed_verdict(rank, verdict)

        templates, calibration_manifest = build_calibration_templates(
            final_model, calibration, calibration_hashes, rank, device
        )
        part_a_new_tasks = make_tasks(
            "A", "final_2b3", PART_A_NEW, depths=(1,)
        )
        execute_phase(
            "part_a_new_controls",
            part_a_new_tasks,
            models,
            canonical,
            canonical_hashes,
            templates,
            progress,
            args.run_dir,
            rank,
            device,
        )

        part_b_depth1_tasks = make_tasks(
            "B", "writer_10m", PART_B_CONTROLS, depths=(1,)
        )
        part_b_depth1_rows = execute_phase(
            "part_b_depth1_regression",
            part_b_depth1_tasks,
            models,
            canonical,
            canonical_hashes,
            None,
            progress,
            args.run_dir,
            rank,
            device,
        )
        if rank == 0:
            part_b_depth1_check = check_expected_means(
                mean_losses(part_b_depth1_rows),
                {
                    "zero:d1": 5.9736480712890625,
                    "real:d1": 5.36134774684906,
                    "coherent_shuffle:d1": 5.4016084432601925,
                },
                {
                    "zero:d1": 5e-6,
                    "real:d1": 5e-6,
                    "coherent_shuffle:d1": 5e-6,
                },
            )
            verdict = part_b_depth1_check | {
                "error": "Part-B depth-1 reproduction failed"
            }
        else:
            part_b_depth1_check = None
            verdict = None
        part_b_depth1_check = distributed_verdict(rank, verdict)

        part_b_deeper_tasks = make_tasks(
            "B", "writer_10m", PART_B_CONTROLS, depths=(2, 3, 4)
        )
        execute_phase(
            "part_b_deeper_masks",
            part_b_deeper_tasks,
            models,
            canonical,
            canonical_hashes,
            None,
            progress,
            args.run_dir,
            rank,
            device,
        )
        interim_hashes = {
            name: model_parameter_sha256(model) for name, model in models.items()
        }
        gathered_interim_hashes = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_interim_hashes, interim_hashes)
        interim_hash_integrity = all(
            value == before_hashes for value in gathered_interim_hashes
        )
        all_rows = gathered_progress_rows(progress)
        if rank == 0:
            preliminary_part_a, _ = aggregate_part_a(all_rows)
            measured_full_context = statistics.fmean(
                row["loss"]
                for row in select_rows(
                    all_rows, "A", "final_2b3", "full_context", 1
                )
            )
            preliminary_integrity = (
                preflight_report["passed"]
                and part_a_baseline_check["passed"]
                and part_b_depth1_check["passed"]
                and calibration_manifest["passed"]
                and interim_hash_integrity
                and streaming_rows_pass(
                    [
                        row
                        for row in all_rows
                        if row["part"] in {"A", "B"}
                    ]
                )
            )
            preliminary_part_b, _ = aggregate_part_b(
                all_rows,
                measured_full_context,
                config,
                integrity_passed=preliminary_integrity,
            )
            conditional_verdict = {
                "passed": True,
                "triggered": preliminary_part_b["strong_support"],
                "best_depth": preliminary_part_b["best_deeper_depth"],
                "preliminary_classification": preliminary_part_b["classification"],
            }
        else:
            conditional_verdict = None
        conditional_verdict = distributed_verdict(rank, conditional_verdict)

        conditional_triggered = conditional_verdict["triggered"]
        best_depth = conditional_verdict["best_depth"]
        if conditional_triggered:
            part_c_tasks = make_tasks(
                "C", "final_2b3", PART_B_CONTROLS, depths=(best_depth,)
            )
            execute_phase(
                "part_c_conditional_confirmation",
                part_c_tasks,
                models,
                canonical,
                canonical_hashes,
                None,
                progress,
                args.run_dir,
                rank,
                device,
            )

        after_hashes = {
            name: model_parameter_sha256(model) for name, model in models.items()
        }
        local_runtime = {
            "rank": rank,
            "rank_seed": rank_seed,
            "parameter_sha256_before": before_hashes,
            "parameter_sha256_after": after_hashes,
            "runtime_counts": dict(RUNTIME_COUNTS),
            "models_eval": all(not model.training for model in models.values()),
            "all_parameters_frozen": all(
                not parameter.requires_grad
                for model in models.values()
                for parameter in model.parameters()
            ),
        }
        gathered_runtime = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_runtime, local_runtime)
        assert_zero_training_counts()
        all_rows = gathered_progress_rows(progress)

        if rank == 0:
            expected_tasks = (
                part_a_baseline_tasks
                + part_a_new_tasks
                + part_b_depth1_tasks
                + part_b_deeper_tasks
            )
            if conditional_triggered:
                expected_tasks += part_c_tasks
            expected_ids = {task_id(task) for task in expected_tasks}
            selected_rows = [row for row in all_rows if row["task_id"] in expected_ids]
            if (
                len(selected_rows) != len(expected_tasks)
                or {row["task_id"] for row in selected_rows} != expected_ids
            ):
                raise SystemExit("final canonical task coverage mismatch")
            measured_full_context = statistics.fmean(
                row["loss"]
                for row in select_rows(
                    selected_rows, "A", "final_2b3", "full_context", 1
                )
            )
            part_a, part_a_paired = aggregate_part_a(selected_rows)
            parameter_hash_integrity = all(
                row["parameter_sha256_before"] == row["parameter_sha256_after"]
                for row in gathered_runtime
            )
            core_integrity = {
                "preflight_passed": preflight_report["passed"],
                "final_2b3_checkpoint_sha256_exact": final_metadata["sha256"]
                == FINAL_2B3_SHA256,
                "writer_10m_checkpoint_sha256_exact": writer_metadata["sha256"]
                == WRITER_10M_SHA256,
                "canonical_validation_sha256_exact": aggregate_payload_hash(
                    canonical_hashes
                )
                == CANONICAL_SHA256,
                "calibration_evaluation_disjoint": calibration_manifest[
                    "calibration_evaluation_disjoint"
                ],
                "part_a_control_regression_passed": part_a_baseline_check["passed"],
                "part_b_depth1_regression_passed": part_b_depth1_check["passed"],
                "all_losses_finite": all(row["finite"] for row in selected_rows),
                "all_stream_cache_health_passed": streaming_rows_pass(selected_rows),
                "model_parameter_hashes_identical": parameter_hash_integrity,
                "all_models_eval_and_frozen": all(
                    row["models_eval"] and row["all_parameters_frozen"]
                    for row in gathered_runtime
                ),
                "optimizer_objects_created_zero": all(
                    row["runtime_counts"]["optimizer_objects_created"] == 0
                    for row in gathered_runtime
                ),
                "scheduler_objects_created_zero": all(
                    row["runtime_counts"]["scheduler_objects_created"] == 0
                    for row in gathered_runtime
                ),
                "grad_scalers_created_zero": all(
                    row["runtime_counts"]["grad_scalers_created"] == 0
                    for row in gathered_runtime
                ),
                "optimizer_steps_zero": all(
                    row["runtime_counts"]["optimizer_steps"] == 0
                    for row in gathered_runtime
                ),
                "backward_calls_zero": all(
                    row["runtime_counts"]["backward_calls"] == 0
                    for row in gathered_runtime
                ),
                "d1_default_explicit_fp32_equivalence": preflight_report[
                    "d1_equivalence"
                ]["passed"],
                "future_causality_passed": all(
                    row["future_prefix_logits_bit_exact"]
                    and row["future_prefix_state_bit_exact"]
                    for row in preflight_report["mask_depth_tests"].values()
                ),
                "row_isolation_passed": all(
                    row["row_isolation_logits_bit_exact"]
                    and row["row_isolation_state_bit_exact"]
                    for row in preflight_report["mask_depth_tests"].values()
                ),
                "fresh_sequence_reset_passed": all(
                    row["fresh_sequence_logits_bit_exact"]
                    and row["fresh_sequence_state_bit_exact"]
                    for row in preflight_report["mask_depth_tests"].values()
                ),
                "masked_cache_absence_passed": all(
                    row["masked_cache_absence"]
                    for row in preflight_report["mask_depth_tests"].values()
                ),
                "unmasked_cache_health_passed": all(
                    row["unmasked_cache_expected_lengths"]
                    and row["unmasked_cache_finite"]
                    for row in preflight_report["mask_depth_tests"].values()
                ),
                "all_memory_control_causality_passed": all(
                    row["passed"]
                    for row in preflight_report["memory_control_causality"].values()
                ),
                "hellaswag_not_run": True,
            }
            core_integrity["passed"] = all(core_integrity.values())
            part_b, part_b_paired = aggregate_part_b(
                selected_rows,
                measured_full_context,
                config,
                integrity_passed=core_integrity["passed"],
            )
            if part_b["strong_support"] != conditional_triggered:
                raise SystemExit("conditional Part-C trigger changed after final audit")
            part_c = (
                aggregate_part_c(selected_rows, part_a, best_depth)
                if conditional_triggered
                else {
                    "status": "NOT TRIGGERED",
                    "reason": "Part B did not satisfy STRONG MASK-PRESSURE SUPPORT",
                }
            )
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            source_checkpoints = {
                "final_2b3": final_metadata,
                "writer_10m": writer_metadata,
            }
            part_a_artifact = {
                "experiment": "2B4",
                "source_checkpoint": source_checkpoints["final_2b3"],
                "canonical_validation_sha256": CANONICAL_SHA256,
                **part_a,
            }
            part_b_artifact = {
                "experiment": "2B4",
                "source_checkpoint": source_checkpoints["writer_10m"],
                "canonical_validation_sha256": CANONICAL_SHA256,
                **part_b,
            }
            status = lambda value: "PASS" if value else "FAIL"
            hard_audit = {
                "2B3 checkpoint SHA": status(
                    core_integrity["final_2b3_checkpoint_sha256_exact"]
                ),
                "2B2A 10M checkpoint SHA": status(
                    core_integrity["writer_10m_checkpoint_sha256_exact"]
                ),
                "canonical validation SHA": status(
                    core_integrity["canonical_validation_sha256_exact"]
                ),
                "no optimizer constructed for result path": status(
                    core_integrity["optimizer_objects_created_zero"]
                ),
                "optimizer steps": sum(
                    row["runtime_counts"]["optimizer_steps"]
                    for row in gathered_runtime
                ),
                "backward calls": sum(
                    row["runtime_counts"]["backward_calls"]
                    for row in gathered_runtime
                ),
                "model parameter hashes before/after": (
                    "IDENTICAL"
                    if core_integrity["model_parameter_hashes_identical"]
                    else "CHANGED"
                ),
                "Block-1 d=1 regression": status(
                    core_integrity["part_b_depth1_regression_passed"]
                ),
                "generalized mask d=1 FP32 equivalence": status(
                    core_integrity["d1_default_explicit_fp32_equivalence"]
                ),
                "future causality": status(core_integrity["future_causality_passed"]),
                "row isolation": status(core_integrity["row_isolation_passed"]),
                "masked-cache absence": status(
                    core_integrity["masked_cache_absence_passed"]
                ),
                "unmasked-cache health": status(
                    core_integrity["unmasked_cache_health_passed"]
                ),
                "all memory-control causality checks": status(
                    core_integrity["all_memory_control_causality_passed"]
                ),
                "calibration/evaluation data disjoint": status(
                    core_integrity["calibration_evaluation_disjoint"]
                ),
                "all losses finite": status(core_integrity["all_losses_finite"]),
                "HellaSwag": "NOT RUN",
            }
            final_audit = {
                "experiment": "2B4",
                "implementation_git_commit": implementation_commit,
                "source_checkpoints": source_checkpoints,
                "runtime_by_rank": gathered_runtime,
                "preflight": preflight_report,
                "part_a_baseline_regression": part_a_baseline_check,
                "part_b_depth1_regression": part_b_depth1_check,
                "calibration": calibration_manifest,
                "integrity": core_integrity,
                "hard_audit_checklist": hard_audit,
                "hellaswag_run": False,
                "passed": core_integrity["passed"],
            }
            result_summary = {
                "experiment": "2B4",
                "protocol": config["protocol"],
                "implementation_git_commit": implementation_commit,
                "source_checkpoints": source_checkpoints,
                "canonical_validation_sha256": CANONICAL_SHA256,
                "part_a": part_a,
                "part_b": part_b,
                "part_c": part_c,
                "classification": part_b["classification"],
                "integrity_passed": core_integrity["passed"],
                "training": {
                    "optimizer_objects_created": 0,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                    "model_parameter_changes": 0,
                },
                "hellaswag_run": False,
                "passed": core_integrity["passed"],
            }
            b2a.write_json(output_dir / "part_a_content_controls.json", part_a_artifact)
            b2a.write_json(output_dir / "part_a_paired_losses.json", part_a_paired)
            b2a.write_json(
                output_dir / "part_a_calibration_manifest.json", calibration_manifest
            )
            b2a.write_json(output_dir / "part_b_mask_depth.json", part_b_artifact)
            b2a.write_json(output_dir / "part_b_paired_losses.json", part_b_paired)
            if conditional_triggered:
                b2a.write_json(output_dir / "part_c_confirmation.json", part_c)
            b2a.write_json(output_dir / "result_summary.json", result_summary)
            b2a.write_json(output_dir / "FINAL_AUDIT.json", final_audit)
            b2a.write_json(output_dir / "PREFLIGHT.json", preflight_report)
            write_csv_artifacts(output_dir, part_a, part_b)
            print(
                f"EXPERIMENT_2B4_EVALUATION_PASS classification={part_b['classification']} "
                f"part_c={part_c['status']}",
                flush=True,
            )
            if not core_integrity["passed"]:
                raise SystemExit("Experiment 2B4 final integrity audit failed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "evaluate"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--preflight")
    parser.add_argument("--final-2b3-checkpoint")
    parser.add_argument("--writer-10m-checkpoint")
    parser.add_argument("--output-dir")
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
    install_zero_optimizer_guards()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "evaluate":
        required = {
            "--preflight": args.preflight,
            "--final-2b3-checkpoint": args.final_2b3_checkpoint,
            "--writer-10m-checkpoint": args.writer_10m_checkpoint,
            "--output-dir": args.output_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"evaluate requires {', '.join(missing)}")
        evaluate(args)


if __name__ == "__main__":
    main()

    with (output_dir / "part_b_mask_depth.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mask_depth",
                "zero_loss",
                "real_loss",
                "shuffled_loss",
                "specific_gap",
                "real_recovery_fraction",
                "specific_share_of_recovery",
                "real_wins",
            ]
        )
        for depth in (1, 2, 3, 4):
            row = part_b["depths"][str(depth)]
            writer.writerow(
                [
                    depth,
                    row["zero_loss"],
                    row["real_loss"],
                    row["shuffled_loss"],
                    row["specific_gap"],
                    row["real_recovery_fraction"],
                    row["specific_share_of_recovery"],
                    row["real_wins"],
                ]
            )
