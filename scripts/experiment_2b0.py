#!/usr/bin/env python3
"""Optimizer-free incremental and self-recurrent evaluation for Experiment 2B0."""

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2a0 as a0  # noqa: E402


BRANCH = "experiment-2b0-self-feedback-l1"
PINNED_PYTHON = Path("/workspace/venvs/exp1b/bin/python")
CONFIG_PATH = REPO_ROOT / "configs" / "exp2b0_zero_shot.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "0702dc09c74b01eee8be504a7f5f89ca61fcc504cda8f34f30865d4ff9653d76"
)
EXPECTED_CHECKPOINT_STATE = {
    "completed_updates": 477,
    "processed_student_tokens": 250_085_376,
}
EXPECTED_VALIDATION_SHA256 = a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256
SOURCE_DEPTHS = a0.SOURCE_DEPTHS
REFERENCE_EVALUATION = (
    REPO_ROOT
    / "results"
    / "experiment_2a3_250m"
    / "evaluations"
    / "evaluation_updates_000477.json"
)
POSITION_BINS = (
    ("1-16", 1, 17),
    ("17-32", 17, 33),
    ("33-64", 33, 65),
    ("65-128", 65, 129),
    ("129-256", 129, 257),
    ("257-512", 257, 513),
    ("513-1023", 513, 1024),
)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def require_environment():
    actual_python = Path(sys.executable).resolve()
    if actual_python != PINNED_PYTHON.resolve():
        raise SystemExit(f"requires pinned Python {PINNED_PYTHON}, got {actual_python}")
    if a0.git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"requires branch {BRANCH}")
    device = a0.require_cuda()
    if torch.cuda.get_device_name(0) != "NVIDIA A100-SXM4-80GB":
        raise SystemExit("Experiment 2B0 requires one NVIDIA A100-SXM4-80GB")
    return device


def validate_config():
    config = json.loads(CONFIG_PATH.read_text())
    required = {
        "protocol": "exp2b0_self_recurrent_zero_shot_v1",
        "seed": a0.SEED,
        "source_depths": list(SOURCE_DEPTHS),
        "destination": "Block 1 Attention input",
        "reader_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "reader_training_updates": 477,
        "reader_training_tokens": 250_085_376,
        "sequence_length": 1024,
        "validation_batches": 20,
        "validation_batch_sequences": 64,
        "initial_diagnostic_batches": 2,
        "short_horizons": [8, 16, 32, 64],
        "reset_horizons": [1, 2, 4, 8, 16, 32, 64, 128, "never"],
        "recurrent_gradient_semantics": "detached inference only",
        "block_1_kv_cache": False,
        "blocks_2_through_12_kv_cache": True,
        "optimizer_steps": 0,
        "hellaswag": "not run without separate approval",
        "canonical_gate": "finite short runs and two-batch self loss <= masked loss",
    }
    if config != required:
        raise SystemExit("Experiment 2B0 config differs from the frozen protocol")
    return config


def load_reference():
    artifact = json.loads(REFERENCE_EVALUATION.read_text())
    if artifact.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("2A3 reference checkpoint mismatch")
    if len(artifact.get("batch_payload_sha256", [])) != a0.VALIDATION_BATCHES:
        raise SystemExit("2A3 reference validation vector is incomplete")
    return artifact


def load_runtime(parent_checkpoint, reader_checkpoint, device):
    reader_checkpoint = Path(reader_checkpoint).resolve()
    checkpoint_hash = file_sha256(reader_checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(
            f"reader checkpoint SHA256 mismatch: {checkpoint_hash} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    symbols, teacher, student, parent_aux = a0.load_models(
        parent_checkpoint, device, include_teacher=True
    )
    checkpoint = a0.torch_load(reader_checkpoint, mmap=True)
    if checkpoint.get("schema") != a0.CHECKPOINT_SCHEMA:
        raise SystemExit("reader checkpoint schema mismatch")
    if checkpoint.get("training_state") != EXPECTED_CHECKPOINT_STATE:
        raise SystemExit("reader checkpoint training-state mismatch")
    if checkpoint.get("parent_checkpoint_sha256") != a0.EXPECTED_PARENT_SHA256:
        raise SystemExit("reader checkpoint parent lineage mismatch")
    optimizer_steps = sorted(
        int(entry["step"].item())
        for entry in checkpoint.get("optimizer", {}).get("state", {}).values()
    )
    if optimizer_steps != [477, 477, 477]:
        raise SystemExit(f"reader checkpoint Adam steps mismatch: {optimizer_steps}")
    student.load_state_dict(checkpoint["model"], strict=True)
    if not a0.nested_equal(student.state_dict(), checkpoint["model"]):
        raise SystemExit("reader model strict reload mismatch")
    del checkpoint
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.eval()
    teacher.eval()
    router = student.transformer.topdown_attnres
    info = {
        "path": str(reader_checkpoint),
        "sha256": checkpoint_hash,
        "training_state": EXPECTED_CHECKPOINT_STATE,
        "optimizer_state_steps_in_checkpoint": optimizer_steps,
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
        "gate": router.gate.detach().float().item(),
        "gate_coefficient": router.gate.detach().float().tanh().item(),
        "query_norm": router.query.detach().float().norm().item(),
        "rmsnorm_displacement": (router.norm.weight.detach().float() - 1).norm().item(),
        "reader_parameters": sum(p.numel() for p in router.parameters()),
        "parent_checkpoint": parent_aux["checkpoint"],
        "parent_checkpoint_sha256": parent_aux["checkpoint_sha256"],
    }
    return symbols, teacher, student, info


def make_validation_loader(symbols):
    return symbols["DataLoaderLite"](
        B=a0.VALIDATION_B,
        T=a0.T,
        process_rank=0,
        num_processes=1,
        split="val",
    )


@torch.no_grad()
def parallel_loss(student, x, y, mode, feedback_sources=None):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = student(
            x, y, mode=mode, feedback_sources=feedback_sources
        )
    return loss.detach().double().item()


@torch.no_grad()
def incremental_logits(student, tokens, mode, feedback_sources=None):
    state = student.init_recurrent_state(
        tokens.size(0), mode, device=tokens.device, dtype=torch.bfloat16
    )
    rows = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            kwargs = {}
            if feedback_sources is not None:
                kwargs["feedback_sources"] = feedback_sources[
                    :, :, position : position + 1
                ]
            logits, state = student.forward_step(
                tokens[:, position : position + 1], state, **kwargs
            )
            rows.append(logits)
    return torch.cat(rows, dim=1), state


def equivalence_metrics(parallel, incremental, targets):
    parallel_float = parallel.float()
    incremental_float = incremental.float()
    difference = (parallel_float - incremental_float).abs()
    parallel_loss_value = F.cross_entropy(
        parallel_float.reshape(-1, parallel.size(-1)), targets.reshape(-1)
    ).double().item()
    incremental_loss_value = F.cross_entropy(
        incremental_float.reshape(-1, incremental.size(-1)), targets.reshape(-1)
    ).double().item()
    return {
        "maximum_absolute_logit_difference": difference.max().item(),
        "mean_absolute_logit_difference": difference.mean().item(),
        "root_mean_square_logit_difference": difference.square().mean().sqrt().item(),
        "mean_absolute_parallel_logit": parallel_float.abs().mean().item(),
        "relative_mean_absolute_logit_difference": (
            difference.mean() / parallel_float.abs().mean().clamp_min(1e-12)
        ).item(),
        "argmax_agreement_fraction": (
            parallel_float.argmax(dim=-1) == incremental_float.argmax(dim=-1)
        ).float().mean().item(),
        "parallel_loss": parallel_loss_value,
        "incremental_loss": incremental_loss_value,
        "absolute_loss_difference": abs(parallel_loss_value - incremental_loss_value),
    }


def cache_payload_equal(left, right, row=None):
    if left["position"] != right["position"] or left["mode"] != right["mode"]:
        return False
    memory_left = left["feedback_memory"]
    memory_right = right["feedback_memory"]
    if row is not None:
        memory_left = memory_left[:, row]
        memory_right = memory_right[:, row]
    if not torch.equal(memory_left, memory_right):
        return False
    for cache_left, cache_right in zip(left["kv_caches"], right["kv_caches"]):
        if cache_left is None or cache_right is None:
            if cache_left is not None or cache_right is not None:
                return False
            continue
        for name in ("key", "value"):
            value_left = cache_left[name]
            value_right = cache_right[name]
            if row is not None:
                value_left = value_left[row]
                value_right = value_right[row]
            if not torch.equal(value_left, value_right):
                return False
    return True


@torch.no_grad()
def run_prefix_capture(student, tokens, prefix_length):
    state = student.init_recurrent_state(
        tokens.size(0),
        "masked_l1_topdown_self",
        device=tokens.device,
        dtype=torch.bfloat16,
    )
    prefix_logits = []
    prefix_state = None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            logits, state = student.forward_step(tokens[:, position], state)
            if position < prefix_length:
                prefix_logits.append(logits.detach().clone())
            if position + 1 == prefix_length:
                prefix_state = state.state_dict()
    return torch.cat(prefix_logits, dim=1), prefix_state


@torch.no_grad()
def causality_tests(student, tokens):
    first = tokens[:2, :32].clone()
    second = first.clone()
    second[:, 16:] = (second[:, 16:] + 1) % student.config.vocab_size
    first_logits, first_prefix = run_prefix_capture(student, first, 16)
    second_logits, second_prefix = run_prefix_capture(student, second, 16)
    future = {
        "prefix_logits_bit_exact": torch.equal(first_logits, second_logits),
        "prefix_memory_and_kv_bit_exact": cache_payload_equal(
            first_prefix, second_prefix
        ),
        "maximum_absolute_logit_difference": (
            first_logits.float() - second_logits.float()
        ).abs().max().item(),
    }
    future["passed"] = all(
        future[key]
        for key in ("prefix_logits_bit_exact", "prefix_memory_and_kv_bit_exact")
    )

    isolated_first = tokens[:2, :16].clone()
    isolated_second = isolated_first.clone()
    isolated_second[1] = (isolated_second[1] + 17) % student.config.vocab_size
    logits_a, state_a = incremental_logits(
        student, isolated_first, "masked_l1_topdown_self"
    )
    logits_b, state_b = incremental_logits(
        student, isolated_second, "masked_l1_topdown_self"
    )
    isolation = {
        "unchanged_row_logits_bit_exact": torch.equal(logits_a[0], logits_b[0]),
        "unchanged_row_memory_and_kv_bit_exact": cache_payload_equal(
            state_a.state_dict(), state_b.state_dict(), row=0
        ),
    }
    isolation["passed"] = all(isolation.values())

    fresh_tokens = tokens[:2, :8]
    incremental_logits(student, tokens[:2, :16], "masked_l1_topdown_self")
    fresh_logits_a, fresh_state_a = incremental_logits(
        student, fresh_tokens, "masked_l1_topdown_self"
    )
    fresh_logits_b, fresh_state_b = incremental_logits(
        student, fresh_tokens, "masked_l1_topdown_self"
    )
    initial = student.init_recurrent_state(
        2,
        "masked_l1_topdown_self",
        device=tokens.device,
        dtype=torch.bfloat16,
    )
    reset = {
        "fresh_logits_bit_exact": torch.equal(fresh_logits_a, fresh_logits_b),
        "fresh_state_bit_exact": cache_payload_equal(
            fresh_state_a.state_dict(), fresh_state_b.state_dict()
        ),
        "initial_position_zero": initial.position == 0,
        "initial_memory_exactly_zero": initial.feedback_memory.count_nonzero().item()
        == 0,
        "block_1_cache_absent": initial.kv_caches[0] is None,
        "other_caches_empty": all(cache.length == 0 for cache in initial.kv_caches[1:]),
    }
    reset["passed"] = all(reset.values())

    resume_tokens = tokens[:2, :16]
    direct = student.init_recurrent_state(
        2,
        "masked_l1_topdown_self",
        device=tokens.device,
        dtype=torch.bfloat16,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(8):
            _, direct = student.forward_step(resume_tokens[:, position], direct)
    buffer = io.BytesIO()
    torch.save(direct.state_dict(), buffer)
    buffer.seek(0)
    restored = student.load_recurrent_state(
        torch.load(buffer, map_location=tokens.device, weights_only=False)
    )
    direct_logits = []
    restored_logits = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(8, 16):
            direct_row, direct = student.forward_step(
                resume_tokens[:, position], direct
            )
            restored_row, restored = student.forward_step(
                resume_tokens[:, position], restored
            )
            direct_logits.append(direct_row)
            restored_logits.append(restored_row)
    resume = {
        "continuation_logits_bit_exact": torch.equal(
            torch.cat(direct_logits, 1), torch.cat(restored_logits, 1)
        ),
        "final_memory_and_kv_bit_exact": cache_payload_equal(
            direct.state_dict(), restored.state_dict()
        ),
        "serialized_schema": "full_attnres_recurrent_state_v1",
    }
    resume["passed"] = all(
        resume[key]
        for key in ("continuation_logits_bit_exact", "final_memory_and_kv_bit_exact")
    )
    report = {
        "future_suffix_invariance": future,
        "sequence_isolation": isolation,
        "sequence_reset": reset,
        "checkpoint_resume": resume,
    }
    report["passed"] = all(row["passed"] for row in report.values())
    return report


def empty_stream_aggregates(device):
    return {
        "source_sum": torch.zeros(4, device=device),
        "source_max": torch.zeros(4, device=device),
        "routing_sum": torch.zeros(4, device=device),
        "entropy_sum": torch.zeros((), device=device),
        "topdown_sum": torch.zeros((), device=device),
        "feedback_sum": torch.zeros((), device=device),
        "count": 0,
    }


def empty_drift_aggregates(device):
    return {
        label: {
            "cosine": torch.zeros(4, device=device),
            "rms_difference": torch.zeros(4, device=device),
            "norm_ratio": torch.zeros(4, device=device),
            "count": 0,
        }
        for label, _, _ in POSITION_BINS
    }


def position_bin(position):
    for label, start, end in POSITION_BINS:
        if start <= position < end:
            return label
    return None


@torch.no_grad()
def stream_loss(
    student,
    x,
    y,
    mode="masked_l1_topdown_self",
    feedback_sources=None,
    permutation=None,
    gate_override=None,
    reset_interval=None,
    teacher_raw=None,
):
    B, T = x.shape
    state = student.init_recurrent_state(
        B, mode, device=x.device, dtype=torch.bfloat16
    )
    loss_sum = torch.zeros((), device=x.device)
    finite = torch.ones((), dtype=torch.bool, device=x.device)
    aggregates = empty_stream_aggregates(x.device)
    drift = empty_drift_aggregates(x.device) if teacher_raw is not None else None
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(T):
            reset = (
                reset_interval is not None
                and position % reset_interval == 0
            )
            kwargs = {
                "feedback_gate_override": gate_override,
                "reset_feedback": reset,
                "return_diagnostics": True,
            }
            if feedback_sources is not None:
                kwargs["feedback_sources"] = feedback_sources[
                    :, :, position : position + 1
                ]
            if permutation is not None:
                kwargs["feedback_permutation"] = permutation
            logits, state, diagnostics = student.forward_step(
                x[:, position], state, **kwargs
            )
            loss_sum = loss_sum + F.cross_entropy(
                logits[:, 0], y[:, position], reduction="sum"
            )
            finite = finite & torch.isfinite(logits).all()
            finite = finite & torch.isfinite(state.feedback_memory).all()
            source_rms = diagnostics["source_rms"]
            aggregates["source_sum"] += source_rms.sum(dim=1)
            aggregates["source_max"] = torch.maximum(
                aggregates["source_max"], source_rms.max(dim=1).values
            )
            if diagnostics["routing_weights"] is not None:
                aggregates["routing_sum"] += diagnostics["routing_weights"].float().sum(
                    dim=(1, 2)
                )
                aggregates["entropy_sum"] += diagnostics["routing_entropy"].sum()
                aggregates["topdown_sum"] += diagnostics["topdown_rms"].sum()
                aggregates["feedback_sum"] += diagnostics["feedback_rms"].sum()
            aggregates["count"] += B

            if teacher_raw is not None:
                label = position_bin(position)
                if label is None:
                    continue
                student_sources = diagnostics["source_memory"][:, :, 0].float()
                teacher_sources = teacher_raw[:, :, position].float()
                cosine = F.cosine_similarity(student_sources, teacher_sources, dim=-1)
                rms_difference = (student_sources - teacher_sources).pow(2).mean(-1).sqrt()
                norm_ratio = student_sources.norm(dim=-1) / teacher_sources.norm(
                    dim=-1
                ).clamp_min(1e-12)
                drift[label]["cosine"] += cosine.sum(dim=1)
                drift[label]["rms_difference"] += rms_difference.sum(dim=1)
                drift[label]["norm_ratio"] += norm_ratio.sum(dim=1)
                drift[label]["count"] += B
    for cache in state.kv_caches:
        if cache is not None:
            key, value = cache.prefix()
            finite = finite & torch.isfinite(key).all() & torch.isfinite(value).all()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    count = aggregates["count"]
    result = {
        "loss": (loss_sum / (B * T)).double().item(),
        "finite": bool(finite.item()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": B * T / elapsed,
        "memory_rms_mean": {
            f"v{depth}": value
            for depth, value in zip(
                SOURCE_DEPTHS, (aggregates["source_sum"] / count).cpu().tolist()
            )
        },
        "memory_rms_max": {
            f"v{depth}": value
            for depth, value in zip(SOURCE_DEPTHS, aggregates["source_max"].cpu().tolist())
        },
        "routing_weights": None,
        "routing_entropy": None,
        "topdown_rms": None,
        "feedback_rms": None,
        "cache_health": {
            "block_1_cache_absent": state.kv_caches[0] is None,
            "blocks_2_through_12_lengths": [
                cache.length for cache in state.kv_caches[1:]
            ],
            "all_expected_lengths": all(
                cache.length == T for cache in state.kv_caches[1:]
            ),
            "finite": bool(finite.item()),
        },
        "state_position": state.position,
    }
    if mode in {
        "masked_l1_topdown_teacher",
        "masked_l1_topdown_self",
        "masked_l1_shuffled_self_feedback",
    }:
        result.update(
            {
                "routing_weights": {
                    f"v{depth}": value
                    for depth, value in zip(
                        SOURCE_DEPTHS,
                        (aggregates["routing_sum"] / count).cpu().tolist(),
                    )
                },
                "routing_entropy": (aggregates["entropy_sum"] / count).item(),
                "topdown_rms": (aggregates["topdown_sum"] / count).item(),
                "feedback_rms": (aggregates["feedback_sum"] / count).item(),
            }
        )
    if drift is not None:
        result["teacher_student_drift"] = {
            label: {
                f"v{depth}": {
                    "cosine_similarity": drift[label]["cosine"][index].item()
                    / drift[label]["count"],
                    "rms_difference": drift[label]["rms_difference"][index].item()
                    / drift[label]["count"],
                    "norm_ratio": drift[label]["norm_ratio"][index].item()
                    / drift[label]["count"],
                }
                for index, depth in enumerate(SOURCE_DEPTHS)
            }
            for label, _, _ in POSITION_BINS
            if drift[label]["count"]
        }
    return result


@torch.no_grad()
def teacher_routing_metrics(student, memory):
    router = student.transformer.topdown_attnres
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        topdown, weights = router(list(memory.unbind(dim=0)), return_weights=True)
        feedback = router.gate.tanh() * topdown
    safe = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
    entropy = -(safe * safe.log()).sum(dim=0)
    return {
        "routing_weights": {
            f"v{depth}": value
            for depth, value in zip(SOURCE_DEPTHS, weights.mean(dim=(1, 2)).cpu().tolist())
        },
        "routing_entropy": entropy.mean().item(),
        "topdown_rms": topdown.float().pow(2).mean(dim=-1).sqrt().mean().item(),
        "feedback_rms": feedback.float().pow(2).mean(dim=-1).sqrt().mean().item(),
    }


def average_dicts(rows, fields):
    return {field: sum(float(row[field]) for row in rows) / len(rows) for field in fields}


def average_routing(rows):
    return {
        "routing_weights": {
            f"v{depth}": sum(row["routing_weights"][f"v{depth}"] for row in rows)
            / len(rows)
            for depth in SOURCE_DEPTHS
        },
        **average_dicts(rows, ("routing_entropy", "topdown_rms", "feedback_rms")),
    }


def average_drift(rows):
    result = {}
    for label, _, _ in POSITION_BINS:
        result[label] = {}
        for depth in SOURCE_DEPTHS:
            values = [row["teacher_student_drift"][label][f"v{depth}"] for row in rows]
            result[label][f"v{depth}"] = {
                field: sum(value[field] for value in values) / len(values)
                for field in ("cosine_similarity", "rms_difference", "norm_ratio")
            }
    return result


def classify(masked, teacher, self_loss, finite=True):
    if not finite or self_loss > masked + 0.01:
        return "SELF-RECURRENT MEMORY IS UNSTABLE"
    if abs(self_loss - masked) <= 0.01:
        return "SELF-RECURRENT MEMORY DOES NOT TRANSFER ZERO-SHOT"
    teacher_recovery = masked - teacher
    self_recovery = masked - self_loss
    ratio = self_recovery / teacher_recovery if teacher_recovery > 0 else float("nan")
    if self_recovery > 0 and ratio >= 0.8 and self_loss <= teacher + 0.05:
        return "SELF-RECURRENT MEMORY TRANSFERS STRONGLY"
    if self_recovery > 0:
        return "SELF-RECURRENT MEMORY TRANSFERS PARTIALLY"
    return "SELF-RECURRENT MEMORY DOES NOT TRANSFER ZERO-SHOT"


def run_preflight(args, device):
    config = validate_config()
    symbols, teacher, student, checkpoint = load_runtime(
        args.parent_checkpoint, args.reader_checkpoint, device
    )
    state_hash_before = model_state_sha256(student)
    checkpoint_hash_before = file_sha256(args.reader_checkpoint)
    loader = make_validation_loader(symbols)
    x_cpu, y_cpu = loader.next_batch()
    x = x_cpu[:4].to(device)
    y = y_cpu[:4].to(device)

    equivalence = {}
    short_x = x[:2, :64]
    short_y = y[:2, :64]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parallel_full, _ = student(short_x, mode="full_context")
    incremental_full, _ = incremental_logits(student, short_x, "full_context")
    equivalence["full_context"] = equivalence_metrics(
        parallel_full, incremental_full, short_y
    )
    del parallel_full, incremental_full
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parallel_masked, _ = student(short_x, mode="masked_l1_no_feedback")
    incremental_masked, _ = incremental_logits(
        student, short_x, "masked_l1_no_feedback"
    )
    equivalence["masked_l1_no_feedback"] = equivalence_metrics(
        parallel_masked, incremental_masked, short_y
    )
    del parallel_masked, incremental_masked
    thresholds = {
        "maximum_absolute_logit_difference": 0.25,
        "mean_absolute_logit_difference": 0.015,
        "absolute_loss_difference": 0.005,
    }
    equivalence["thresholds"] = thresholds
    equivalence["passed"] = all(
        row[field] <= limit
        for row in (
            equivalence["full_context"],
            equivalence["masked_l1_no_feedback"],
        )
        for field, limit in thresholds.items()
    )
    if not equivalence["passed"]:
        raise SystemExit(f"incremental equivalence hard stop: {equivalence}")

    causality = causality_tests(student, x)
    if not causality["passed"]:
        raise SystemExit(f"recurrent causality hard stop: {causality}")

    smoke = {}
    for horizon in config["short_horizons"]:
        result = stream_loss(
            student,
            x[:, :horizon],
            y[:, :horizon],
            mode="masked_l1_topdown_self",
        )
        result["passed"] = (
            result["finite"]
            and result["state_position"] == horizon
            and result["cache_health"]["all_expected_lengths"]
            and max(result["memory_rms_max"].values()) < 10_000
        )
        smoke[str(horizon)] = result
        print(
            f"smoke T={horizon} loss={result['loss']:.6f} "
            f"feedback_rms={result['feedback_rms']:.6f}",
            flush=True,
        )
    smoke_passed = all(row["passed"] for row in smoke.values())
    if not smoke_passed:
        raise SystemExit(f"short-horizon stability hard stop: {smoke}")

    state_hash_after = model_state_sha256(student)
    checkpoint_hash_after = file_sha256(args.reader_checkpoint)
    integrity = {
        "model_state_sha256_before": state_hash_before,
        "model_state_sha256_after": state_hash_after,
        "model_state_unchanged": state_hash_before == state_hash_after,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_file_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
    }
    integrity["passed"] = (
        integrity["model_state_unchanged"] and integrity["checkpoint_file_unchanged"]
    )
    report = {
        "experiment": "2B0",
        "stage": "incremental_preflight",
        "git": {
            "branch": a0.git_output("branch", "--show-current"),
            "commit": a0.git_output("rev-parse", "HEAD"),
            "status": a0.git_output("status", "--short", "--branch"),
        },
        "runtime": {
            "python": str(Path(sys.executable).resolve()),
            "gpu": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": config,
        "starting_checkpoint": checkpoint,
        "equivalence": equivalence,
        "causality": causality,
        "short_horizon_stability": smoke,
        "integrity": integrity,
        "passed": equivalence["passed"] and causality["passed"] and smoke_passed and integrity["passed"],
    }
    write_json(Path(args.out_dir) / "preflight.json", report)
    return report


def run_diagnostic(args, device):
    preflight_path = Path(args.out_dir) / "preflight.json"
    if not preflight_path.is_file() or not json.loads(preflight_path.read_text()).get("passed"):
        raise SystemExit("passing preflight.json is required before diagnostic evaluation")
    reference = load_reference()
    symbols, teacher, student, checkpoint = load_runtime(
        args.parent_checkpoint, args.reader_checkpoint, device
    )
    state_hash_before = model_state_sha256(student)
    loader = make_validation_loader(symbols)
    batches = []
    for batch_index in range(2):
        x_cpu, y_cpu = loader.next_batch()
        payload_hash = a0.batch_payload_hash(x_cpu, y_cpu)
        if payload_hash != reference["batch_payload_sha256"][batch_index]:
            raise SystemExit(f"validation batch {batch_index} payload mismatch")
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        masked = parallel_loss(student, x, y, "masked_l1_no_feedback")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            teacher_raw = teacher.capture_residual_sources(x, SOURCE_DEPTHS)
        memory = symbols["shift_teacher_sources"](teacher_raw)
        teacher_value = parallel_loss(
            student, x, y, "masked_l1_topdown_teacher", feedback_sources=memory
        )
        teacher_routing = teacher_routing_metrics(student, memory)
        self_result = stream_loss(
            student,
            x,
            y,
            mode="masked_l1_topdown_self",
            teacher_raw=teacher_raw,
        )
        batch = {
            "batch_index": batch_index,
            "payload_sha256": payload_hash,
            "masked_l1_no_feedback": masked,
            "teacher_feedback": teacher_value,
            "self_feedback": self_result,
            "teacher_routing": teacher_routing,
        }
        batches.append(batch)
        write_json(Path(args.out_dir) / "diagnostic_progress.json", {"batches": batches})
        print(
            f"diagnostic {batch_index + 1}/2 masked={masked:.6f} "
            f"teacher={teacher_value:.6f} self={self_result['loss']:.6f}",
            flush=True,
        )
        del x, y, teacher_raw, memory
        torch.cuda.empty_cache()

    losses = {
        "masked_l1_no_feedback": sum(row["masked_l1_no_feedback"] for row in batches) / 2,
        "teacher_feedback": sum(row["teacher_feedback"] for row in batches) / 2,
        "self_feedback": sum(row["self_feedback"]["loss"] for row in batches) / 2,
    }
    teacher_recovery = losses["masked_l1_no_feedback"] - losses["teacher_feedback"]
    self_recovery = losses["masked_l1_no_feedback"] - losses["self_feedback"]
    finite = all(row["self_feedback"]["finite"] for row in batches)
    canonical_gate = finite and losses["self_feedback"] <= losses["masked_l1_no_feedback"]
    classification = classify(
        losses["masked_l1_no_feedback"],
        losses["teacher_feedback"],
        losses["self_feedback"],
        finite,
    )
    state_hash_after = model_state_sha256(student)
    report = {
        "experiment": "2B0",
        "stage": "two_batch_zero_shot_diagnostic",
        "starting_checkpoint": checkpoint,
        "validation_batches": 2,
        "B": a0.VALIDATION_B,
        "T": a0.T,
        "batches": batches,
        "losses": losses,
        "teacher_recovery": teacher_recovery,
        "self_recovery": self_recovery,
        "self_teacher_recovery_ratio": self_recovery / teacher_recovery,
        "teacher_memory_routing": average_routing(
            [row["teacher_routing"] for row in batches]
        ),
        "self_memory_routing": average_routing(
            [row["self_feedback"] for row in batches]
        ),
        "teacher_student_drift": average_drift(
            [row["self_feedback"] for row in batches]
        ),
        "canonical_gate_passed": canonical_gate,
        "canonical_gate_rule": "finite self-feedback and two-batch self loss <= masked loss",
        "classification": classification,
        "model_state_sha256_before": state_hash_before,
        "model_state_sha256_after": state_hash_after,
        "model_state_unchanged": state_hash_before == state_hash_after,
        "checkpoint_sha256_after": file_sha256(args.reader_checkpoint),
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
        "passed": finite and state_hash_before == state_hash_after,
    }
    write_json(Path(args.out_dir) / "diagnostic.json", report)
    return report


def load_progress(path):
    path = Path(path)
    if not path.is_file():
        return {"batches": []}
    payload = json.loads(path.read_text())
    if set(payload) != {"batches"} or not isinstance(payload["batches"], list):
        raise SystemExit(f"invalid progress artifact: {path}")
    return payload


def progress_entry(progress, batch_index, payload_hash):
    matches = [row for row in progress["batches"] if row.get("batch_index") == batch_index]
    if len(matches) > 1:
        raise SystemExit(f"duplicate progress batch {batch_index}")
    if matches:
        row = matches[0]
        if row.get("payload_sha256") != payload_hash:
            raise SystemExit(f"progress payload mismatch at batch {batch_index}")
        return row
    row = {"batch_index": batch_index, "payload_sha256": payload_hash}
    progress["batches"].append(row)
    progress["batches"].sort(key=lambda item: item["batch_index"])
    return row


def validate_canonical_progress(progress):
    required = {
        "batch_index",
        "payload_sha256",
        "self_feedback",
        "shuffled_self_feedback",
        "self_feedback_gate_zero",
        "teacher_routing",
    }
    return (
        len(progress["batches"]) == a0.VALIDATION_BATCHES
        and [row["batch_index"] for row in progress["batches"]]
        == list(range(a0.VALIDATION_BATCHES))
        and all(required <= set(row) for row in progress["batches"])
    )


def run_canonical(args, device):
    diagnostic_path = Path(args.out_dir) / "diagnostic.json"
    if not diagnostic_path.is_file():
        raise SystemExit("diagnostic.json is required before canonical evaluation")
    diagnostic = json.loads(diagnostic_path.read_text())
    if not diagnostic.get("canonical_gate_passed"):
        raise SystemExit("canonical expansion gate did not pass")
    reference = load_reference()
    symbols, teacher, student, checkpoint = load_runtime(
        args.parent_checkpoint, args.reader_checkpoint, device
    )
    state_hash_before = model_state_sha256(student)
    progress_path = Path(args.out_dir) / "canonical_progress.json"
    progress = load_progress(progress_path)
    diagnostic_batches = {
        row["batch_index"]: row for row in diagnostic["batches"]
    }
    loader = make_validation_loader(symbols)
    permutation = symbols["fixed_derangement"](a0.VALIDATION_B, device)
    validation_digest = hashlib.sha256()
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        payload_hash = a0.batch_payload_hash(x_cpu, y_cpu)
        if payload_hash != reference["batch_payload_sha256"][batch_index]:
            raise SystemExit(f"canonical payload mismatch at batch {batch_index}")
        validation_digest.update(bytes.fromhex(payload_hash))
        row = progress_entry(progress, batch_index, payload_hash)
        if batch_index in diagnostic_batches and "self_feedback" not in row:
            prior = diagnostic_batches[batch_index]
            row["self_feedback"] = prior["self_feedback"]
            row["teacher_routing"] = prior["teacher_routing"]
            write_json(progress_path, progress)
        if {
            "self_feedback",
            "shuffled_self_feedback",
            "self_feedback_gate_zero",
            "teacher_routing",
        } <= set(row):
            print(f"canonical {batch_index + 1:02d}/20 reused", flush=True)
            continue

        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        if "self_feedback" not in row:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                teacher_raw = teacher.capture_residual_sources(x, SOURCE_DEPTHS)
            memory = symbols["shift_teacher_sources"](teacher_raw)
            row["teacher_routing"] = teacher_routing_metrics(student, memory)
            row["self_feedback"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_topdown_self",
                teacher_raw=teacher_raw,
            )
            del teacher_raw, memory
            write_json(progress_path, progress)
            print(
                f"canonical {batch_index + 1:02d}/20 self="
                f"{row['self_feedback']['loss']:.6f}",
                flush=True,
            )
        if "shuffled_self_feedback" not in row:
            row["shuffled_self_feedback"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_shuffled_self_feedback",
                permutation=permutation,
            )
            write_json(progress_path, progress)
            print(
                f"canonical {batch_index + 1:02d}/20 shuffled="
                f"{row['shuffled_self_feedback']['loss']:.6f}",
                flush=True,
            )
        if "self_feedback_gate_zero" not in row:
            row["self_feedback_gate_zero"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_topdown_self",
                gate_override=0.0,
            )
            write_json(progress_path, progress)
            print(
                f"canonical {batch_index + 1:02d}/20 gate-zero="
                f"{row['self_feedback_gate_zero']['loss']:.6f}",
                flush=True,
            )
        del x, y
        torch.cuda.empty_cache()

    if validation_digest.hexdigest() != EXPECTED_VALIDATION_SHA256:
        raise SystemExit("canonical validation global hash mismatch")
    if not validate_canonical_progress(progress):
        raise SystemExit("canonical progress is incomplete")
    rows = progress["batches"]
    new_losses = {
        "self_feedback_zero_shot": sum(row["self_feedback"]["loss"] for row in rows)
        / len(rows),
        "shuffled_self_feedback": sum(
            row["shuffled_self_feedback"]["loss"] for row in rows
        )
        / len(rows),
        "self_feedback_gate_zero": sum(
            row["self_feedback_gate_zero"]["loss"] for row in rows
        )
        / len(rows),
    }
    losses = {
        "full_context": reference["losses"]["full_context"],
        "masked_l1_no_feedback": reference["losses"]["masked_l1_no_feedback"],
        "teacher_feedback_250m": reference["losses"]["real_feedback"],
        **new_losses,
    }
    teacher_recovery = losses["masked_l1_no_feedback"] - losses["teacher_feedback_250m"]
    self_recovery = losses["masked_l1_no_feedback"] - losses["self_feedback_zero_shot"]
    finite = all(
        row[mode]["finite"]
        for row in rows
        for mode in (
            "self_feedback",
            "shuffled_self_feedback",
            "self_feedback_gate_zero",
        )
    )
    state_hash_after = model_state_sha256(student)
    report = {
        "experiment": "2B0",
        "stage": "canonical_validation",
        "starting_checkpoint": checkpoint,
        "controls_reused_from_audited_2a3": {
            "artifact": str(REFERENCE_EVALUATION.resolve()),
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "full_context": True,
            "masked_l1_no_feedback": True,
            "teacher_feedback_250m": True,
        },
        "validation_batches": a0.VALIDATION_BATCHES,
        "B": a0.VALIDATION_B,
        "T": a0.T,
        "validation_global_batches_sha256": validation_digest.hexdigest(),
        "losses": losses,
        "teacher_recovery": teacher_recovery,
        "self_recovery": self_recovery,
        "self_teacher_recovery_ratio": self_recovery / teacher_recovery,
        "real_minus_shuffled_self_gap": losses["shuffled_self_feedback"]
        - losses["self_feedback_zero_shot"],
        "self_gate_zero_minus_masked": losses["self_feedback_gate_zero"]
        - losses["masked_l1_no_feedback"],
        "teacher_memory_routing": average_routing(
            [row["teacher_routing"] for row in rows]
        ),
        "self_memory_routing": average_routing(
            [row["self_feedback"] for row in rows]
        ),
        "teacher_student_drift": average_drift(
            [row["self_feedback"] for row in rows]
        ),
        "batch_losses": {
            "self_feedback_zero_shot": [row["self_feedback"]["loss"] for row in rows],
            "shuffled_self_feedback": [
                row["shuffled_self_feedback"]["loss"] for row in rows
            ],
            "self_feedback_gate_zero": [
                row["self_feedback_gate_zero"]["loss"] for row in rows
            ],
        },
        "classification": classify(
            losses["masked_l1_no_feedback"],
            losses["teacher_feedback_250m"],
            losses["self_feedback_zero_shot"],
            finite,
        ),
        "model_state_sha256_before": state_hash_before,
        "model_state_sha256_after": state_hash_after,
        "model_state_unchanged": state_hash_before == state_hash_after,
        "checkpoint_sha256_after": file_sha256(args.reader_checkpoint),
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
        "finite": finite,
        "passed": finite and state_hash_before == state_hash_after,
    }
    write_json(Path(args.out_dir) / "canonical.json", report)
    return report


@torch.no_grad()
def stream_reset_horizons(student, x, y, intervals):
    B, T = x.shape
    count = len(intervals)
    expanded_x = x.repeat((count, 1))
    expanded_y = y.repeat((count, 1))
    state = student.init_recurrent_state(
        count * B,
        "masked_l1_topdown_self",
        device=x.device,
        dtype=torch.bfloat16,
    )
    losses = torch.zeros(count, device=x.device)
    finite = torch.ones((), dtype=torch.bool, device=x.device)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(T):
            group_resets = torch.tensor(
                [
                    interval is not None and position % interval == 0
                    for interval in intervals
                ],
                device=x.device,
                dtype=torch.bool,
            )
            reset_mask = group_resets.repeat_interleave(B)
            logits, state = student.forward_step(
                expanded_x[:, position], state, reset_feedback=reset_mask
            )
            token_losses = F.cross_entropy(
                logits[:, 0], expanded_y[:, position], reduction="none"
            ).view(count, B)
            losses += token_losses.sum(dim=1)
            finite = finite & torch.isfinite(logits).all() & torch.isfinite(
                state.feedback_memory
            ).all()
    for cache in state.kv_caches[1:]:
        key, value = cache.prefix()
        finite = finite & torch.isfinite(key).all() & torch.isfinite(value).all()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "losses": {
            "never" if interval is None else str(interval): value
            for interval, value in zip(intervals, (losses / (B * T)).double().cpu().tolist())
        },
        "finite": bool(finite.item()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": count * B * T / elapsed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "cache_lengths_correct": all(cache.length == T for cache in state.kv_caches[1:]),
    }


def run_reset_horizons(args, device):
    canonical_path = Path(args.out_dir) / "canonical.json"
    if not canonical_path.is_file():
        raise SystemExit("canonical.json is required before reset-horizon evaluation")
    canonical = json.loads(canonical_path.read_text())
    if canonical["self_recovery"] <= 0:
        raise SystemExit("reset-horizon diagnostic requires positive self recovery")
    reference = load_reference()
    symbols, _, student, checkpoint = load_runtime(
        args.parent_checkpoint, args.reader_checkpoint, device
    )
    state_hash_before = model_state_sha256(student)
    intervals = (1, 2, 4, 8, 16, 32, 64, 128, None)
    progress_path = Path(args.out_dir) / "reset_horizon_progress.json"
    progress = load_progress(progress_path)
    loader = make_validation_loader(symbols)
    validation_digest = hashlib.sha256()
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        payload_hash = a0.batch_payload_hash(x_cpu, y_cpu)
        if payload_hash != reference["batch_payload_sha256"][batch_index]:
            raise SystemExit(f"reset-horizon payload mismatch at batch {batch_index}")
        validation_digest.update(bytes.fromhex(payload_hash))
        row = progress_entry(progress, batch_index, payload_hash)
        if "reset_horizons" in row:
            print(f"reset horizons {batch_index + 1:02d}/20 reused", flush=True)
            continue
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        row["reset_horizons"] = stream_reset_horizons(
            student, x, y, intervals
        )
        write_json(progress_path, progress)
        print(
            f"reset horizons {batch_index + 1:02d}/20 "
            f"one={row['reset_horizons']['losses']['1']:.6f} "
            f"never={row['reset_horizons']['losses']['never']:.6f}",
            flush=True,
        )
        del x, y
        torch.cuda.empty_cache()
    if validation_digest.hexdigest() != EXPECTED_VALIDATION_SHA256:
        raise SystemExit("reset-horizon validation global hash mismatch")
    rows = progress["batches"]
    if len(rows) != a0.VALIDATION_BATCHES or any(
        "reset_horizons" not in row for row in rows
    ):
        raise SystemExit("reset-horizon progress incomplete")
    losses = {
        key: sum(row["reset_horizons"]["losses"][key] for row in rows) / len(rows)
        for key in ("1", "2", "4", "8", "16", "32", "64", "128", "never")
    }
    finite = all(row["reset_horizons"]["finite"] for row in rows)
    state_hash_after = model_state_sha256(student)
    report = {
        "experiment": "2B0",
        "stage": "reset_horizon",
        "starting_checkpoint": checkpoint,
        "validation_batches": a0.VALIDATION_BATCHES,
        "B": a0.VALIDATION_B,
        "T": a0.T,
        "validation_global_batches_sha256": validation_digest.hexdigest(),
        "reset_interval_losses": losses,
        "batch_losses": {
            key: [row["reset_horizons"]["losses"][key] for row in rows]
            for key in losses
        },
        "finite": finite,
        "cache_health": all(
            row["reset_horizons"]["cache_lengths_correct"] for row in rows
        ),
        "mean_tokens_per_second": sum(
            row["reset_horizons"]["tokens_per_second"] for row in rows
        )
        / len(rows),
        "maximum_peak_allocated_mb": max(
            row["reset_horizons"]["peak_allocated_mb"] for row in rows
        ),
        "model_state_sha256_before": state_hash_before,
        "model_state_sha256_after": state_hash_after,
        "model_state_unchanged": state_hash_before == state_hash_after,
        "checkpoint_sha256_after": file_sha256(args.reader_checkpoint),
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
        "passed": finite and state_hash_before == state_hash_after,
    }
    write_json(Path(args.out_dir) / "reset_horizon.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("preflight", "diagnostic", "canonical", "reset-horizons")
    )
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    device = require_environment()
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(a0.SEED)
    np.random.seed(a0.SEED)
    torch.manual_seed(a0.SEED)
    torch.cuda.manual_seed(a0.SEED)
    if args.stage == "preflight":
        report = run_preflight(args, device)
    elif args.stage == "diagnostic":
        report = run_diagnostic(args, device)
    elif args.stage == "canonical":
        report = run_canonical(args, device)
    else:
        report = run_reset_horizons(args, device)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
