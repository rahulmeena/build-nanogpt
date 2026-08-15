#!/usr/bin/env python3
"""Guarded memory-only writer training with one-step temporal credit for 2B2."""

import argparse
import copy
import gc
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

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2a0 as a0  # noqa: E402
import experiment_2b0 as b0  # noqa: E402
import experiment_2b1 as b1  # noqa: E402


BRANCH = "experiment-2b2-memory-writers-1step"
SOURCE_2B1_TAG = "experiment-2b1-self-reader-5m"
SOURCE_2B1_COMMIT = "ad2eb56b1fdbf20dde515693f6d2c9bd9034f444"
SOURCE_2B1_IMPLEMENTATION = "c2f8082a7e0deff797a59e1f8b78949e7da3d246"
SOURCE_2B0_TAG = "experiment-2b0-zero-shot"
SOURCE_2B0_COMMIT = "a8b271ee71ae1af77da8ddad022ce549be390682"
SOURCE_2B0_IMPLEMENTATION = "ff552a2d662e92b417b0d4cdae295c3a17180ca7"
PINNED_PYTHON = Path("/workspace/venvs/exp1b/bin/python")
CONFIG_PATH = REPO_ROOT / "configs" / "exp2b2_5m.json"
CHECKPOINT_SCHEMA = "exp2b2_memory_writers_1step_v1"
SOURCE_CHECKPOINT_SHA256 = (
    "5a97c36c038ad04155c7965e20a800cdd78845819671f91c6d516599bb9cd69a"
)
SOURCE_NEXT_SHA256 = (
    "3c4d0cd7905f16bfcfdd283cbc0799ff9f85c91ac3c521e886f0e403fc11ae57"
)
SOURCE_STATE = {
    "local_completed_updates": 10,
    "processed_2b1_tokens": 5_242_880,
    "fineweb_lineage_completed_update": 487,
    "kind": "result_5m",
}
EXPECTED_READER = {
    "gate": 0.1594133824110031,
    "gate_coefficient": 0.15807659924030304,
    "query_norm": 1.9570868015289307,
    "rmsnorm_displacement": 1.9478669166564941,
    "reader_parameters": 1537,
}
PINNED = {
    "full_context": 4.078654408454895,
    "masked_l1_no_feedback": 5.973674488067627,
    "source_2b1_self": 5.701308727264404,
    "source_2b1_shuffled": 5.731030297279358,
    "source_2b1_gate_zero": 5.9736480712890625,
    "source_2b1_teacher_memory": 5.572065925598144,
    "source_2b1_specific_gap": 0.02972157001495379,
}
SOURCE_DEPTHS = a0.SOURCE_DEPTHS
WRITER_INIT_SEED = 20260202
WRITER_RANK = 8
WRITER_PARAMETER_COUNT = 49_152
TRAINABLE_NAMES = {
    f"transformer.memory_writers.writer_v{depth}.W_{projection}.weight"
    for depth in SOURCE_DEPTHS
    for projection in ("down", "up")
}
SOURCE_FILES = (
    "train_gpt2.py",
    "scripts/experiment_2b2.py",
    "scripts/test_experiment_2b2_cpu.py",
    "configs/exp2b2_5m.json",
    "EXPERIMENT_2B2_DESIGN.md",
)


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def durable_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        orphan = temporary.with_name(
            temporary.name + f".orphaned.{int(time.time())}.{os.getpid()}"
        )
        os.replace(temporary, orphan)
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def durable_append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    return a0.file_sha256(Path(path))


def source_file_hashes():
    return {name: file_sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def validate_config():
    config = load_json(CONFIG_PATH)
    required = {
        "protocol": "exp2b2_memory_writers_1step_5m_v1",
        "seed": a0.SEED,
        "writer_initialization_seed": WRITER_INIT_SEED,
        "source_depths": list(SOURCE_DEPTHS),
        "destination": "Block 1 Attention input",
        "source_2b1_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_2b1_local_completed_updates": 10,
        "source_2b1_processed_tokens": 5_242_880,
        "source_next_global_batch_sha256": SOURCE_NEXT_SHA256,
        "writer_rank": WRITER_RANK,
        "writer_eps": 1e-5,
        "local_optimizer_updates": 10,
        "checkpoint_after_local_updates": 5,
        "global_batch_tokens": a0.GLOBAL_BATCH_TOKENS,
        "processed_2b2_tokens": 10 * a0.GLOBAL_BATCH_TOKENS,
        "legacy_world_size": a0.LEGACY_WORLD_SIZE,
        "legacy_micro_batch_sequences_per_rank": a0.LEGACY_B,
        "legacy_gradient_accumulation": a0.LEGACY_GRAD_ACCUM,
        "serialized_slices_per_update": 8,
        "sequence_length": a0.T,
        "backward_chunk_tokens": 16,
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
        },
        "trainable_parameters": WRITER_PARAMETER_COUNT,
        "training_memory": "student-generated writer-adapted previous-token v16/v17/v20/v24",
        "writer_input": "detached raw student source",
        "temporal_gradient": "exactly one token through writer memory",
        "block_1_historical_kv": False,
        "blocks_2_through_12_historical_kv": True,
        "teacher_training_forward_calls": 0,
        "smoke": {
            "batch_sequences": 2,
            "sequence_length": 64,
            "optimizer_updates": 3,
            "checkpoint_after_updates": 2,
        },
        "validation": {
            "batches": a0.VALIDATION_BATCHES,
            "batch_sequences": a0.VALIDATION_B,
            "sequence_length": a0.T,
            "global_batches_sha256": a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256,
        },
        "reset_horizons": [1, 2, 4, 8, 16, 32, 64, 128, "never"],
        "hellaswag": "not run without separate approval",
        "classification_equality_band": 0.01,
    }
    if config != required:
        raise SystemExit("Experiment 2B2 config differs from the frozen protocol")
    return config


def require_environment(require_clean=False):
    if Path(sys.executable).resolve() != PINNED_PYTHON.resolve():
        raise SystemExit(f"requires pinned Python {PINNED_PYTHON}")
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"requires branch {BRANCH}")
    if require_clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing training requires a clean committed worktree")
    if git_output("rev-parse", SOURCE_2B0_TAG) != SOURCE_2B0_COMMIT:
        raise SystemExit("frozen Experiment 2B0 tag target mismatch")
    if git_output("rev-parse", SOURCE_2B1_TAG) != SOURCE_2B1_COMMIT:
        raise SystemExit("frozen Experiment 2B1 tag target mismatch")
    device = a0.require_cuda()
    if torch.cuda.get_device_name(0) != "NVIDIA A100-SXM4-80GB":
        raise SystemExit("Experiment 2B2 requires one NVIDIA A100-SXM4-80GB")
    return device


def reader_values(student):
    router = student.transformer.topdown_attnres
    return {
        "gate": router.gate.detach().float().item(),
        "gate_coefficient": router.gate.detach().float().tanh().item(),
        "query_norm": router.query.detach().float().norm().item(),
        "rmsnorm_displacement": (
            router.norm.weight.detach().float() - 1
        ).norm().item(),
        "reader_parameters": sum(p.numel() for p in router.parameters()),
    }


def assert_reader_initialization(student):
    actual = reader_values(student)
    for name, expected in EXPECTED_READER.items():
        if name == "reader_parameters":
            matches = actual[name] == expected
        else:
            matches = math.isclose(actual[name], expected, rel_tol=0.0, abs_tol=2e-7)
        if not matches:
            raise SystemExit(f"source reader {name} mismatch: {actual[name]} != {expected}")
    return actual


def assert_runtime_contract(student):
    trainable = {
        name: parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != TRAINABLE_NAMES:
        raise SystemExit(f"trainable tensor set mismatch: {sorted(trainable)}")
    if sum(parameter.numel() for parameter in trainable.values()) != WRITER_PARAMETER_COUNT:
        raise SystemExit("trainable parameter count is not 49,152")
    return sorted(trainable)


def state_subset_sha256(model, group):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        is_reader = name.startswith("transformer.topdown_attnres.")
        is_writer = name.startswith("transformer.memory_writers.")
        selected = {
            "base": not is_reader and not is_writer,
            "reader": is_reader,
            "writers": is_writer,
            "source_model": not is_writer,
        }[group]
        if not selected:
            continue
        tensor = value.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def model_config(symbols, enable_writers):
    return symbols["GPTConfig"](
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
        enable_memory_writers=enable_writers,
        memory_writer_rank=WRITER_RANK,
        memory_writer_init_seed=WRITER_INIT_SEED,
    )


def load_source_runtime(
    source_checkpoint,
    device,
    include_teacher=False,
    restore_source_rng=False,
):
    source_checkpoint = Path(source_checkpoint).resolve()
    digest = file_sha256(source_checkpoint)
    if digest != SOURCE_CHECKPOINT_SHA256:
        raise SystemExit(f"source 2B1 checkpoint SHA mismatch: {digest}")
    symbols = a0.support.load_training_symbols()
    checkpoint = a0.torch_load(source_checkpoint, mmap=True)
    if checkpoint.get("schema") != b1.CHECKPOINT_SCHEMA:
        raise SystemExit("source 2B1 checkpoint schema mismatch")
    if checkpoint.get("training_state") != SOURCE_STATE:
        raise SystemExit("source 2B1 training state mismatch")
    if checkpoint.get("next_global_batch_sha256") != SOURCE_NEXT_SHA256:
        raise SystemExit("source 2B1 next-batch field mismatch")
    source_optimizer_steps = sorted(
        int(row["step"].item())
        for row in checkpoint.get("optimizer", {}).get("state", {}).values()
    )
    if source_optimizer_steps != [10, 10, 10]:
        raise SystemExit(f"source Adam steps mismatch: {source_optimizer_steps}")
    with torch.random.fork_rng(devices=[]):
        student = symbols["GPT"](model_config(symbols, enable_writers=True))
    missing, unexpected = student.load_state_dict(checkpoint["model"], strict=False)
    if set(missing) != TRAINABLE_NAMES or unexpected:
        raise SystemExit(
            f"source-to-writer load mismatch: missing={missing}, unexpected={unexpected}"
        )
    if not all(
        a0.nested_equal(student.state_dict()[name], value)
        for name, value in checkpoint["model"].items()
    ):
        raise SystemExit("source 2B1 model tensors did not load exactly")
    student.freeze_for_memory_writer_training()
    assert_runtime_contract(student)
    initial_reader = assert_reader_initialization(student)
    teacher = None
    if include_teacher:
        with torch.random.fork_rng(devices=[]):
            teacher = symbols["GPT"](model_config(symbols, enable_writers=False))
        teacher.load_state_dict(checkpoint["model"], strict=True)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
    source = {
        "path": str(source_checkpoint),
        "sha256": digest,
        "training_state": copy.deepcopy(checkpoint["training_state"]),
        "dataloader_states": copy.deepcopy(checkpoint["dataloader_states"]),
        "rng_state": copy.deepcopy(checkpoint["rng_state"]),
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "source_optimizer_steps_ignored": source_optimizer_steps,
        "initial_reader": initial_reader,
        "source_2a3_checkpoint": checkpoint.get("source_2a3_checkpoint_path"),
        "source_2a3_checkpoint_sha256": checkpoint.get("source_2a3_checkpoint_sha256"),
        "source_model_sha256": state_subset_sha256(student, "source_model"),
        "frozen_base_sha256": state_subset_sha256(student, "base"),
        "frozen_reader_sha256": state_subset_sha256(student, "reader"),
        "initial_writer_sha256": state_subset_sha256(student, "writers"),
    }
    if restore_source_rng:
        a0.restore_rng_state(source["rng_state"])
    del checkpoint
    gc.collect()
    student.to(device)
    if teacher is not None:
        teacher.to(device)
    return symbols, teacher, student, source


def source_report(source):
    """Return the JSON-safe source lineage; raw loader/RNG state stays in checkpoints."""
    return {
        name: value
        for name, value in source.items()
        if name not in {"dataloader_states", "rng_state"}
    } | {
        "dataloader_state_count": len(source["dataloader_states"]),
        "rng_fields": sorted(source["rng_state"]),
    }


def fresh_optimizer(student, device_type="cuda"):
    parameters = [p for p in student.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != WRITER_PARAMETER_COUNT:
        raise SystemExit("optimizer parameter count mismatch")
    kwargs = {
        "lr": 1e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **kwargs)
    if optimizer.state:
        raise SystemExit("fresh 2B2 optimizer unexpectedly contains moments")
    return optimizer


def optimizer_report(optimizer, completed_updates):
    state = optimizer if isinstance(optimizer, dict) else optimizer.state_dict()
    if len(state["param_groups"]) != 1:
        raise SystemExit("2B2 requires one optimizer parameter group")
    group = state["param_groups"][0]
    expected = {
        "lr": 1e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    for name, value in expected.items():
        if group.get(name) != value:
            raise SystemExit(f"optimizer {name} mismatch: {group.get(name)} != {value}")
    if completed_updates == 0 and state["state"]:
        raise SystemExit("fresh optimizer must not have state")
    if completed_updates > 0 and len(state["state"]) != 8:
        raise SystemExit("2B2 optimizer must have exactly eight state entries")
    steps = []
    nonfinite = []
    for parameter_id, values in state["state"].items():
        if "step" in values:
            steps.append(int(values["step"].item()))
        for name, value in values.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if not torch.isfinite(value).all():
                    nonfinite.append(f"{parameter_id}:{name}")
    if completed_updates > 0 and sorted(steps) != [completed_updates] * 8:
        raise SystemExit(f"2B2 Adam step mismatch: {steps}")
    if nonfinite:
        raise SystemExit(f"non-finite optimizer moments: {nonfinite}")
    return {
        "state_entries": len(state["state"]),
        "steps": sorted(steps),
        "moments_finite": not nonfinite,
        "lr": group["lr"],
        "betas": list(group["betas"]),
        "eps": group["eps"],
        "weight_decay": group["weight_decay"],
    }


def gradient_report(student):
    rows = {}
    named = dict(student.named_parameters())
    for depth in SOURCE_DEPTHS:
        for projection in ("down", "up"):
            short = f"writer_v{depth}_W_{projection}"
            name = f"transformer.memory_writers.writer_v{depth}.W_{projection}.weight"
            gradient = named[name].grad
            rows[short] = {
                "present": gradient is not None,
                "finite": gradient is not None and bool(torch.isfinite(gradient).all().item()),
                "nonzero": gradient is not None and bool(torch.count_nonzero(gradient).item()),
                "norm": None if gradient is None else gradient.detach().float().norm().item(),
            }
    rows["reader_tensors_with_grad"] = [
        name for name, parameter in student.named_parameters()
        if name.startswith("transformer.topdown_attnres.") and parameter.grad is not None
    ]
    rows["frozen_tensors_with_grad"] = [
        name
        for name, parameter in student.named_parameters()
        if name not in TRAINABLE_NAMES and parameter.grad is not None
    ]
    return rows


def state_health(state, expected_length, expect_writer_graph=True):
    memory_ok = (
        ((state.feedback_memory.grad_fn is not None and state.feedback_memory.requires_grad)
         if expect_writer_graph else
         (state.feedback_memory.grad_fn is None and not state.feedback_memory.requires_grad))
        and bool(torch.isfinite(state.feedback_memory).all().item())
    )
    caches_ok = True
    lengths = []
    for index, cache in enumerate(state.kv_caches):
        if index == 0:
            caches_ok &= cache is None
            continue
        if cache is None:
            caches_ok = False
            continue
        key, value = cache.prefix()
        lengths.append(cache.length)
        caches_ok &= (
            cache.length == expected_length
            and key.grad_fn is None
            and value.grad_fn is None
            and not key.requires_grad
            and not value.requires_grad
            and bool(torch.isfinite(key).all().item())
            and bool(torch.isfinite(value).all().item())
        )
    return {
        "memory_finite_with_expected_graph": memory_ok,
        "memory_requires_grad": state.feedback_memory.requires_grad,
        "memory_has_grad_fn": state.feedback_memory.grad_fn is not None,
        "block_1_cache_absent": state.kv_caches[0] is None,
        "blocks_2_through_12_lengths": lengths,
        "historical_kv_finite_detached": bool(caches_ok),
        "passed": bool(memory_ok and caches_ok),
    }


def grad_global_norm(parameters):
    squares = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            squares += parameter.grad.detach().float().pow(2).sum()
    return squares.sqrt().item()


@torch.no_grad()
def stream_loss(
    student,
    x,
    y,
    mode="masked_l1_topdown_self",
    use_writers=True,
    feedback_sources=None,
    permutation=None,
    gate_override=None,
    reset_interval=None,
    teacher_raw=None,
    disabled_writer_depths=(),
):
    B, T = x.shape
    self_mode = mode in {
        "masked_l1_topdown_self",
        "masked_l1_shuffled_self_feedback",
    }
    if use_writers and not self_mode:
        raise ValueError("writers are applied before shifting teacher sources")
    state = student.init_recurrent_state(B, mode, device=x.device, dtype=torch.bfloat16)
    loss_sum = torch.zeros((), device=x.device)
    finite = torch.ones((), dtype=torch.bool, device=x.device)
    source_sum = torch.zeros(4, device=x.device)
    adapted_sum = torch.zeros(4, device=x.device)
    delta_sum = torch.zeros(4, device=x.device)
    ratio_sum = torch.zeros(4, device=x.device)
    cosine_sum = torch.zeros(4, device=x.device)
    ratio_max = torch.zeros(4, device=x.device)
    routing_sum = torch.zeros(4, device=x.device)
    entropy_sum = torch.zeros((), device=x.device)
    topdown_sum = torch.zeros((), device=x.device)
    feedback_sum = torch.zeros((), device=x.device)
    drift = {
        kind: b0.empty_drift_aggregates(x.device)
        for kind in ("teacher_vs_raw_student", "teacher_vs_adapted_memory")
    } if teacher_raw is not None else None
    count = 0
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(T):
            reset = reset_interval is not None and position % reset_interval == 0
            kwargs = {
                "feedback_gate_override": gate_override,
                "reset_feedback": reset,
                "return_diagnostics": True,
                "use_memory_writers": bool(use_writers and self_mode),
                "disabled_writer_depths": disabled_writer_depths,
            }
            if feedback_sources is not None:
                kwargs["feedback_sources"] = feedback_sources[:, :, position : position + 1]
                kwargs.pop("use_memory_writers")
                kwargs.pop("disabled_writer_depths")
            if permutation is not None:
                kwargs["feedback_permutation"] = permutation
            logits, state, diagnostics = student.forward_step(x[:, position], state, **kwargs)
            loss_sum += F.cross_entropy(logits[:, 0], y[:, position], reduction="sum")
            finite &= torch.isfinite(logits).all() & torch.isfinite(state.feedback_memory).all()
            raw = diagnostics["source_memory"][:, :, 0].float()
            adapted = diagnostics["adapted_memory"][:, :, 0].float()
            delta = adapted - raw
            raw_rms = raw.pow(2).mean(-1).sqrt()
            adapted_rms = adapted.pow(2).mean(-1).sqrt()
            delta_rms = delta.pow(2).mean(-1).sqrt()
            ratio = delta_rms / raw_rms.clamp_min(1e-12)
            cosine = F.cosine_similarity(raw, adapted, dim=-1)
            source_sum += raw_rms.sum(dim=1)
            adapted_sum += adapted_rms.sum(dim=1)
            delta_sum += delta_rms.sum(dim=1)
            ratio_sum += ratio.sum(dim=1)
            cosine_sum += cosine.sum(dim=1)
            ratio_max = torch.maximum(ratio_max, ratio.max(dim=1).values)
            if diagnostics["routing_weights"] is not None:
                routing_sum += diagnostics["routing_weights"].float().sum(dim=(1, 2))
                entropy_sum += diagnostics["routing_entropy"].sum()
                topdown_sum += diagnostics["topdown_rms"].sum()
                feedback_sum += diagnostics["feedback_rms"].sum()
            count += B
            if drift is not None:
                label = b0.position_bin(position)
                if label is not None:
                    teacher_position = teacher_raw[:, :, position].float()
                    for kind, candidate in (
                        ("teacher_vs_raw_student", raw),
                        ("teacher_vs_adapted_memory", adapted),
                    ):
                        row = drift[kind][label]
                        row["cosine"] += F.cosine_similarity(
                            candidate, teacher_position, dim=-1
                        ).sum(dim=1)
                        row["rms_difference"] += (
                            candidate - teacher_position
                        ).pow(2).mean(-1).sqrt().sum(dim=1)
                        row["norm_ratio"] += (
                            candidate.norm(dim=-1)
                            / teacher_position.norm(dim=-1).clamp_min(1e-12)
                        ).sum(dim=1)
                        row["count"] += B
    for cache in state.kv_caches:
        if cache is not None:
            key, value = cache.prefix()
            finite &= torch.isfinite(key).all() & torch.isfinite(value).all()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "loss": (loss_sum / (B * T)).double().item(),
        "finite": bool(finite.item()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": B * T / elapsed,
        "writer_behavior": {
            f"v{depth}": {
                "source_rms": (source_sum[index] / count).item(),
                "adapted_memory_rms": (adapted_sum[index] / count).item(),
                "delta_rms": (delta_sum[index] / count).item(),
                "delta_to_source_rms_ratio": (ratio_sum[index] / count).item(),
                "maximum_delta_to_source_rms_ratio": ratio_max[index].item(),
                "cosine_source_adapted": (cosine_sum[index] / count).item(),
            }
            for index, depth in enumerate(SOURCE_DEPTHS)
        },
        "routing_weights": None,
        "routing_entropy": None,
        "topdown_rms": None,
        "feedback_rms": None,
        "cache_health": {
            "block_1_cache_absent": state.kv_caches[0] is None,
            "blocks_2_through_12_lengths": [cache.length for cache in state.kv_caches[1:]],
            "all_expected_lengths": all(cache.length == T for cache in state.kv_caches[1:]),
            "historical_kv_detached": all(
                cache.key.grad_fn is None and cache.value.grad_fn is None
                and not cache.key.requires_grad and not cache.value.requires_grad
                for cache in state.kv_caches[1:]
            ),
        },
        "state_position": state.position,
    }
    if mode != "masked_l1_no_feedback":
        result.update({
            "routing_weights": {
                f"v{depth}": value
                for depth, value in zip(SOURCE_DEPTHS, (routing_sum / count).cpu().tolist())
            },
            "routing_entropy": (entropy_sum / count).item(),
            "topdown_rms": (topdown_sum / count).item(),
            "feedback_rms": (feedback_sum / count).item(),
        })
    if drift is not None:
        result["representation_drift"] = {
            kind: {
                label: {
                    f"v{depth}": {
                        "cosine_similarity": table[label]["cosine"][index].item()
                        / table[label]["count"],
                        "rms_difference": table[label]["rms_difference"][index].item()
                        / table[label]["count"],
                        "norm_ratio": table[label]["norm_ratio"][index].item()
                        / table[label]["count"],
                    }
                    for index, depth in enumerate(SOURCE_DEPTHS)
                }
                for label, _, _ in b0.POSITION_BINS
                if table[label]["count"]
            }
            for kind, table in drift.items()
        }
    return result


@torch.no_grad()
def writer_adapted_teacher_memory(student, teacher_raw):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        adapted, _ = student.write_recurrent_memory(teacher_raw)
    return torch.cat(
        [torch.zeros_like(adapted[:, :, :1]), adapted[:, :, :-1]], dim=2
    ).detach()


@torch.no_grad()
def stream_writer_reset_horizons(student, x, y, intervals):
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
            reset_groups = torch.tensor(
                [interval is not None and position % interval == 0 for interval in intervals],
                device=x.device,
                dtype=torch.bool,
            )
            reset_mask = reset_groups.repeat_interleave(B)
            logits, state = student.forward_step(
                expanded_x[:, position],
                state,
                reset_feedback=reset_mask,
                use_memory_writers=True,
            )
            token_losses = F.cross_entropy(
                logits[:, 0], expanded_y[:, position], reduction="none"
            ).view(count, B)
            losses += token_losses.sum(dim=1)
            finite &= torch.isfinite(logits).all() & torch.isfinite(
                state.feedback_memory
            ).all()
    for cache in state.kv_caches[1:]:
        key, value = cache.prefix()
        finite &= torch.isfinite(key).all() & torch.isfinite(value).all()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "losses": {
            "never" if interval is None else str(interval): value
            for interval, value in zip(
                intervals, (losses / (B * T)).double().cpu().tolist()
            )
        },
        "finite": bool(finite.item()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": count * B * T / elapsed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "cache_lengths_correct": all(cache.length == T for cache in state.kv_caches[1:]),
        "historical_kv_detached": all(
            cache.key.grad_fn is None and cache.value.grad_fn is None
            for cache in state.kv_caches[1:]
        ),
    }


def train_recurrent_update(
    student,
    optimizer,
    batches,
    target_count,
    backward_chunk,
    local_update,
    expected_batch_sha256,
):
    """Train one update; every yielded BxT slice receives a fresh recurrent state."""
    assert_runtime_contract(student)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    update_hash = hashlib.sha256()
    raw_loss_sum = 0.0
    target_seen = 0
    routing_sum = torch.zeros(4, dtype=torch.float64)
    source_sum = torch.zeros(4, dtype=torch.float64)
    adapted_sum = torch.zeros(4, dtype=torch.float64)
    delta_sum = torch.zeros(4, dtype=torch.float64)
    maximum_residual_ratio = 0.0
    entropy_sum = 0.0
    topdown_sum = 0.0
    feedback_sum = 0.0
    routing_count = 0
    all_state_health = []
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()

    for x_cpu, y_cpu in batches:
        update_hash.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
        x = x_cpu.to("cuda", non_blocking=True)
        y = y_cpu.to("cuda", non_blocking=True)
        B, T = x.shape
        state = student.init_recurrent_state(
            B, "masked_l1_topdown_self", device=x.device, dtype=torch.bfloat16
        )
        chunk_loss = None
        for position in range(T):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, state, diagnostics = student.forward_step(
                    x[:, position],
                    state,
                    use_memory_writers=True,
                    return_diagnostics=True,
                )
                token_loss_sum = F.cross_entropy(
                    logits[:, 0], y[:, position], reduction="sum"
                )
                scaled = token_loss_sum / target_count
                chunk_loss = scaled if chunk_loss is None else chunk_loss + scaled
            raw_loss_sum += token_loss_sum.detach().double().item()
            target_seen += B
            routing_sum += diagnostics["routing_weights"].detach().double().sum(
                dim=(1, 2)
            ).cpu()
            source_sum += diagnostics["source_rms"].detach().double().sum(dim=1).cpu()
            adapted_sum += diagnostics["adapted_rms"].detach().double().sum(dim=1).cpu()
            delta_rms = diagnostics["writer_delta"].float().pow(2).mean(
                dim=(2, 3)
            ).sqrt()
            delta_sum += delta_rms.detach().double().sum(dim=1).cpu()
            residual_ratio = delta_rms / diagnostics["source_rms"].float().clamp_min(1e-12)
            maximum_residual_ratio = max(
                maximum_residual_ratio, residual_ratio.max().detach().item()
            )
            entropy_sum += diagnostics["routing_entropy"].detach().double().sum().item()
            topdown_sum += diagnostics["topdown_rms"].detach().double().sum().item()
            feedback_sum += diagnostics["feedback_rms"].detach().double().sum().item()
            routing_count += B
            end_chunk = (position + 1) % backward_chunk == 0 or position + 1 == T
            if end_chunk:
                if not torch.isfinite(chunk_loss).item():
                    raise SystemExit(f"non-finite loss in update {local_update}")
                chunk_loss.backward()
                chunk_loss = None
        health = state_health(state, T)
        all_state_health.append(health)
        if not health["passed"]:
            raise SystemExit(f"recurrent state invariant failed: {health}")
        del x, y, state

    actual_hash = update_hash.hexdigest()
    if actual_hash != expected_batch_sha256:
        raise SystemExit(
            f"consumed batch hash mismatch at local update {local_update}: "
            f"{actual_hash} != {expected_batch_sha256}"
        )
    if target_seen != target_count:
        raise SystemExit(f"target count mismatch: {target_seen} != {target_count}")
    gradients = gradient_report(student)
    for depth in SOURCE_DEPTHS:
        up = gradients[f"writer_v{depth}_W_up"]
        down = gradients[f"writer_v{depth}_W_down"]
        if not all(up[key] for key in ("present", "finite", "nonzero")):
            raise SystemExit(f"invalid writer_v{depth} W_up gradient: {up}")
        if not down["present"] or not down["finite"]:
            raise SystemExit(f"invalid writer_v{depth} W_down gradient: {down}")
        if local_update == 1 and down["nonzero"]:
            raise SystemExit(f"writer_v{depth} W_down moved before W_up staging")
        if local_update >= 2 and not down["nonzero"]:
            raise SystemExit(f"writer_v{depth} W_down gradient remained zero")
    if gradients["reader_tensors_with_grad"]:
        raise SystemExit(f"reader gradient leak: {gradients['reader_tensors_with_grad']}")
    if gradients["frozen_tensors_with_grad"]:
        raise SystemExit(f"frozen gradient leak: {gradients['frozen_tensors_with_grad']}")
    trainable = [p for p in student.parameters() if p.requires_grad]
    pre_clip = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    if not torch.isfinite(pre_clip):
        raise SystemExit("non-finite pre-clip gradient norm")
    post_clip = grad_global_norm(trainable)
    optimizer.step()
    torch.cuda.synchronize()
    nonfinite_parameters = [
        name
        for name, parameter in student.named_parameters()
        if parameter.requires_grad and not torch.isfinite(parameter).all()
    ]
    if nonfinite_parameters:
        raise SystemExit(f"non-finite writer parameters: {nonfinite_parameters}")
    optimizer_state = optimizer_report(optimizer, local_update)
    values = reader_values(student)
    writer_norms = {"W_up": 0.0, "W_down": 0.0}
    for name, parameter in student.named_parameters():
        if name in TRAINABLE_NAMES:
            projection = "W_up" if ".W_up." in name else "W_down"
            writer_norms[projection] += parameter.detach().float().pow(2).sum().item()
    writer_norms = {name: math.sqrt(value) for name, value in writer_norms.items()}
    row = {
        "kind": "train",
        "local_update": local_update,
        "fineweb_lineage_completed_update": SOURCE_STATE[
            "fineweb_lineage_completed_update"
        ] + local_update,
        "processed_2b2_tokens": local_update * target_count,
        "global_mean_training_loss": raw_loss_sum / target_count,
        "learning_rate": 1e-4,
        "pre_clip_gradient_norm": float(pre_clip),
        "post_clip_gradient_norm": post_clip,
        **values,
        "routing_entropy": entropy_sum / routing_count,
        "routing_weights": {
            f"v{depth}": value
            for depth, value in zip(SOURCE_DEPTHS, (routing_sum / routing_count).tolist())
        },
        "source_state_rms": {
            f"v{depth}": value
            for depth, value in zip(SOURCE_DEPTHS, (source_sum / routing_count).tolist())
        },
        "adapted_memory_rms": {
            f"v{depth}": value
            for depth, value in zip(SOURCE_DEPTHS, (adapted_sum / routing_count).tolist())
        },
        "writer_delta_rms": {
            f"v{depth}": value
            for depth, value in zip(SOURCE_DEPTHS, (delta_sum / routing_count).tolist())
        },
        "maximum_writer_residual_to_source_rms_ratio": maximum_residual_ratio,
        "W_up_aggregate_norm": writer_norms["W_up"],
        "W_down_aggregate_norm": writer_norms["W_down"],
        "topdown_rms": topdown_sum / routing_count,
        "feedback_rms": feedback_sum / routing_count,
        "global_batch_sha256": actual_hash,
        "targets": target_seen,
        "serialized_slices": len(all_state_health),
        "backward_chunk_tokens": backward_chunk,
        "gradients": gradients,
        "optimizer": optimizer_state,
        "teacher_forward_calls_during_training": 0,
        "teacher_constructed_for_training": False,
        "state_health": all_state_health,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "wall_seconds": time.perf_counter() - wall_start,
    }
    scalar_fields = (
        "global_mean_training_loss",
        "learning_rate",
        "pre_clip_gradient_norm",
        "post_clip_gradient_norm",
        "gate",
        "gate_coefficient",
        "query_norm",
        "rmsnorm_displacement",
        "routing_entropy",
        "topdown_rms",
        "feedback_rms",
        "maximum_writer_residual_to_source_rms_ratio",
        "W_up_aggregate_norm",
        "W_down_aggregate_norm",
    )
    if not all(math.isfinite(float(row[name])) for name in scalar_fields):
        raise SystemExit(f"non-finite training metric: {row}")
    if not math.isclose(
        sum(row["routing_weights"].values()), 1.0, rel_tol=0.0, abs_tol=2e-6
    ):
        raise SystemExit("training routing weights do not form a simplex")
    return row


def make_metadata(kind, config, source):
    return {
        "experiment": "2B2",
        "kind": kind,
        "protocol": config["protocol"],
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "source_2b0_tag": SOURCE_2B0_TAG,
        "source_2b0_commit": SOURCE_2B0_COMMIT,
        "source_2b0_implementation_commit": SOURCE_2B0_IMPLEMENTATION,
        "source_2b1_tag": SOURCE_2B1_TAG,
        "source_2b1_commit": SOURCE_2B1_COMMIT,
        "source_2b1_implementation_commit": SOURCE_2B1_IMPLEMENTATION,
        "source_2b1_checkpoint": source["path"],
        "source_2b1_checkpoint_sha256": source["sha256"],
        "source_2b1_training_state": source["training_state"],
        "source_next_global_batch_sha256": source["next_global_batch_sha256"],
        "source_2a3_checkpoint": source["source_2a3_checkpoint"],
        "source_2a3_checkpoint_sha256": source["source_2a3_checkpoint_sha256"],
        "config": config,
        "source_file_sha256": source_file_hashes(),
        "training_mode": "masked_l1_topdown_self",
        "teacher_training_forward_calls": 0,
        "teacher_constructed_for_training": False,
        "fresh_optimizer": "source 2B1 reader Adam moments deliberately ignored",
        "autograd_detach_policy": {
            "writer_input_raw_source": "detach",
            "writer_output_for_next_token": "retain exactly until one use",
            "next_historical_keys": "detach",
            "next_historical_values": "detach",
            "ordinary_same_token_transformer_path": "raw writer sources only",
            "temporal_credit_horizon_tokens": 1,
            "multi_token_bptt": False,
        },
        "sequence_boundary_policy": {
            "each_B64_row": "position zero, zero recurrent memory, empty KV caches",
            "in_row_endoftext": "no reset",
            "block_1_historical_kv": False,
            "blocks_2_through_12_historical_kv": True,
        },
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "determinism": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "hellaswag": "not run",
    }


def checkpoint_payload(
    student,
    optimizer,
    loaders,
    symbols,
    replay,
    training_state,
    metadata,
):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_state": dict(training_state),
        "dataloader_states": a0.snapshot_loaders(loaders),
        "rng_state": a0.capture_rng_state(),
        "next_global_batch_sha256": a0.next_update_hash(loaders, symbols, replay),
        "metadata": copy.deepcopy(metadata),
        "source_2a3_checkpoint_path": metadata["source_2a3_checkpoint"],
        "source_2a3_checkpoint_sha256": metadata["source_2a3_checkpoint_sha256"],
        "source_2b0_tag": SOURCE_2B0_TAG,
        "source_2b0_commit": SOURCE_2B0_COMMIT,
        "source_2b1_tag": SOURCE_2B1_TAG,
        "source_2b1_commit": SOURCE_2B1_COMMIT,
        "source_2b1_checkpoint_path": metadata["source_2b1_checkpoint"],
        "source_2b1_checkpoint_sha256": metadata["source_2b1_checkpoint_sha256"],
        "writer_architecture": {
            "sources": [f"v{depth}" for depth in SOURCE_DEPTHS],
            "rank": WRITER_RANK,
            "eps": 1e-5,
            "bias": False,
            "initialization_seed": WRITER_INIT_SEED,
            "W_down_initialization": "Normal(mean=0,std=0.02)",
            "W_up_initialization": "exact zero",
            "temporal_credit_horizon_tokens": 1,
        },
        "implementation_git_commit": metadata["git_commit"],
        "saved_by_pid": os.getpid(),
    }


def make_loaders_for_checkpoint(symbols, checkpoint, replay):
    return a0.make_loaders_from_states(
        symbols, checkpoint["dataloader_states"], replay=replay
    )


def verify_checkpoint_payload(
    path,
    symbols,
    student,
    optimizer,
    expected_payload,
    replay,
):
    checkpoint = a0.torch_load(path, mmap=True)
    required = {
        "schema",
        "model",
        "optimizer",
        "training_state",
        "dataloader_states",
        "rng_state",
        "next_global_batch_sha256",
        "metadata",
        "source_2a3_checkpoint_path",
        "source_2a3_checkpoint_sha256",
        "source_2b0_tag",
        "source_2b0_commit",
        "source_2b1_tag",
        "source_2b1_commit",
        "source_2b1_checkpoint_path",
        "source_2b1_checkpoint_sha256",
        "writer_architecture",
        "implementation_git_commit",
        "saved_by_pid",
    }
    if set(checkpoint) != required or checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise SystemExit("2B2 checkpoint schema/field mismatch")
    for name in (
        "training_state",
        "dataloader_states",
        "rng_state",
        "metadata",
        "next_global_batch_sha256",
    ):
        if not a0.nested_equal(checkpoint[name], expected_payload[name]):
            raise SystemExit(f"2B2 checkpoint {name} reload mismatch")
    if not a0.nested_equal(checkpoint["model"], student.state_dict()):
        raise SystemExit("2B2 checkpoint live/model mismatch")
    if not a0.nested_equal(checkpoint["optimizer"], optimizer.state_dict()):
        raise SystemExit("2B2 checkpoint live/optimizer mismatch")
    fresh_loaders = make_loaders_for_checkpoint(symbols, checkpoint, replay)
    replayed_next = a0.next_update_hash(fresh_loaders, symbols, replay)
    if replayed_next != checkpoint["next_global_batch_sha256"]:
        raise SystemExit("2B2 serialized-loader next-batch mismatch")

    live_rng = a0.capture_rng_state()
    try:
        with torch.random.fork_rng(devices=[]):
            clone = symbols["GPT"](model_config(symbols, enable_writers=True))
        clone.freeze_for_memory_writer_training()
        clone.load_state_dict(checkpoint["model"], strict=True)
        clone_optimizer = fresh_optimizer(clone, device_type="cpu")
        clone_optimizer.load_state_dict(checkpoint["optimizer"])
        if not a0.nested_equal(clone.state_dict(), checkpoint["model"]):
            raise SystemExit("2B2 strict fresh-object model reload mismatch")
        if not a0.nested_equal(clone_optimizer.state_dict(), checkpoint["optimizer"]):
            raise SystemExit("2B2 strict fresh-object optimizer reload mismatch")
    finally:
        a0.restore_rng_state(live_rng)
    completed = checkpoint["training_state"]["local_completed_updates"]
    optimizer_state = optimizer_report(checkpoint["optimizer"], completed)
    report = {
        "schema": checkpoint["schema"],
        "model_strict_reload": True,
        "optimizer_strict_reload": True,
        "loader_strict_reload": True,
        "rng_strict_reload": True,
        "serialized_loader_replay_match": True,
        "next_global_batch_sha256": replayed_next,
        "local_completed_updates": completed,
        "processed_2b2_tokens": checkpoint["training_state"]["processed_2b2_tokens"],
        "loader_states": len(checkpoint["dataloader_states"]),
        "rng_fields": sorted(checkpoint["rng_state"]),
        "optimizer": optimizer_state,
        "source_2a3_checkpoint_sha256": checkpoint["source_2a3_checkpoint_sha256"],
        "source_2b0_commit": checkpoint["source_2b0_commit"],
        "source_2b1_commit": checkpoint["source_2b1_commit"],
        "source_2b1_checkpoint_sha256": checkpoint[
            "source_2b1_checkpoint_sha256"
        ],
        "writer_architecture": checkpoint["writer_architecture"],
        "implementation_git_commit": checkpoint["implementation_git_commit"],
        "saved_by_pid": checkpoint["saved_by_pid"],
        "passed": True,
    }
    del checkpoint, clone, clone_optimizer, fresh_loaders
    gc.collect()
    return report


def save_checkpoint(
    path,
    student,
    optimizer,
    loaders,
    symbols,
    replay,
    training_state,
    metadata,
):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise SystemExit(f"stale incomplete checkpoint requires inspection: {temporary}")
    payload = checkpoint_payload(
        student, optimizer, loaders, symbols, replay, training_state, metadata
    )
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    verification = verify_checkpoint_payload(
        temporary, symbols, student, optimizer, payload, replay
    )
    digest = file_sha256(temporary)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    sidecar = {
        "checkpoint": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "verification": verification,
    }
    durable_write_json(path.with_suffix(path.suffix + ".verification.json"), sidecar)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return sidecar


def verify_checkpoint_sidecars(path):
    path = Path(path).resolve()
    sha_path = path.with_suffix(path.suffix + ".sha256")
    verification_path = path.with_suffix(path.suffix + ".verification.json")
    if not sha_path.is_file() or not verification_path.is_file():
        raise SystemExit(f"checkpoint sidecars missing for {path}")
    fields = sha_path.read_text().strip().split()
    digest = file_sha256(path)
    if len(fields) != 2 or fields[0] != digest or fields[1] != path.name:
        raise SystemExit("checkpoint SHA sidecar mismatch")
    sidecar = load_json(verification_path)
    if sidecar.get("sha256") != digest or not sidecar["verification"].get("passed"):
        raise SystemExit("checkpoint verification sidecar mismatch")
    return sidecar


def load_checkpoint_into_fresh(
    path,
    student,
    optimizer,
    symbols,
    replay,
    expected_metadata,
):
    sidecar = verify_checkpoint_sidecars(path)
    checkpoint = a0.torch_load(path, mmap=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("resume schema mismatch")
    if checkpoint.get("metadata") != expected_metadata:
        raise SystemExit("resume metadata mismatch")
    if checkpoint.get("source_2b1_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
        raise SystemExit("resume source checkpoint mismatch")
    if checkpoint.get("source_2b0_commit") != SOURCE_2B0_COMMIT:
        raise SystemExit("resume source 2B0 mismatch")
    if checkpoint.get("source_2b1_commit") != SOURCE_2B1_COMMIT:
        raise SystemExit("resume source 2B1 mismatch")
    student.load_state_dict(checkpoint["model"], strict=True)
    student.freeze_for_memory_writer_training()
    optimizer.load_state_dict(checkpoint["optimizer"])
    loaders = make_loaders_for_checkpoint(symbols, checkpoint, replay)
    a0.restore_rng_state(checkpoint["rng_state"])
    expected_next = a0.next_update_hash(loaders, symbols, replay)
    if expected_next != checkpoint["next_global_batch_sha256"]:
        raise SystemExit("resume next-batch mismatch")
    state = copy.deepcopy(checkpoint["training_state"])
    optimizer_report(optimizer, state["local_completed_updates"])
    audit = {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": sidecar["sha256"],
        "saved_by_pid": checkpoint["saved_by_pid"],
        "resumed_by_pid": os.getpid(),
        "fresh_process": checkpoint["saved_by_pid"] != os.getpid(),
        "model_strict_reload": a0.nested_equal(student.state_dict(), checkpoint["model"]),
        "optimizer_strict_reload": a0.nested_equal(
            optimizer.state_dict(), checkpoint["optimizer"]
        ),
        "loader_strict_reload": a0.nested_equal(
            a0.snapshot_loaders(loaders), checkpoint["dataloader_states"]
        ),
        "rng_strict_reload": a0.nested_equal(
            a0.capture_rng_state(), checkpoint["rng_state"]
        ),
        "restored_next_global_batch_sha256": expected_next,
        "local_completed_updates": state["local_completed_updates"],
        "processed_2b2_tokens": state["processed_2b2_tokens"],
    }
    audit["passed"] = all(
        audit[name]
        for name in (
            "fresh_process",
            "model_strict_reload",
            "optimizer_strict_reload",
            "loader_strict_reload",
            "rng_strict_reload",
        )
    )
    if not audit["passed"]:
        raise SystemExit(f"fresh-process resume audit failed: {audit}")
    del checkpoint
    gc.collect()
    return loaders, state, audit


def validation_loader(symbols):
    return symbols["DataLoaderLite"](
        B=a0.VALIDATION_B,
        T=a0.T,
        process_rank=0,
        num_processes=1,
        split="val",
    )


def reference_validation():
    path = REPO_ROOT / "results" / "experiment_2b1_5m" / "evaluation" / "canonical_progress.json"
    progress = load_json(path)
    rows = progress.get("batches", [])
    if len(rows) != a0.VALIDATION_BATCHES:
        raise SystemExit("2B1 validation reference batch vector incomplete")
    return {"batch_payload_sha256": [row["payload_sha256"] for row in rows]}


def derive_replay_oracle(symbols, loader_states, updates=10):
    loaders = a0.make_replay_loaders(symbols, loader_states)
    rows = []
    for local_update in range(1, updates + 1):
        digest = a0.next_update_hash(loaders, symbols, replay=True)
        rows.append({"local_update": local_update, "global_batch_sha256": digest})
        for _ in a0.update_batches(loaders, replay=True):
            pass
    if rows[0]["global_batch_sha256"] != SOURCE_NEXT_SHA256:
        raise SystemExit("replay oracle does not start at the 2B1 continuation")
    return rows


def tiny_gradient_vector(model, tokens, targets, backward_chunk):
    model.eval()
    model.freeze_for_memory_writer_training()
    model.zero_grad(set_to_none=True)
    state = model.init_recurrent_state(
        tokens.size(0), "masked_l1_topdown_self", device=tokens.device, dtype=torch.float32
    )
    pending = None
    for position in range(tokens.size(1)):
        logits, state = model.forward_step(
            tokens[:, position], state, use_memory_writers=True
        )
        loss = F.cross_entropy(logits[:, 0], targets[:, position], reduction="sum")
        loss = loss / targets.numel()
        pending = loss if pending is None else pending + loss
        if (position + 1) % backward_chunk == 0 or position + 1 == tokens.size(1):
            if pending.requires_grad:
                pending.backward()
            pending = None
    health = state_health(state, tokens.size(1))
    if not health["passed"]:
        raise SystemExit(f"tiny gradient state invariant failed: {health}")
    gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return gradients


def chunked_gradient_equivalence(symbols, configured_chunk):
    torch.manual_seed(20260815)
    config = symbols["GPTConfig"](
        block_size=8,
        vocab_size=32,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
        enable_memory_writers=True,
        memory_writer_rank=WRITER_RANK,
        memory_writer_init_seed=WRITER_INIT_SEED,
    )
    reference = symbols["GPT"](config)
    with torch.no_grad():
        router = reference.transformer.topdown_attnres
        router.query.copy_(torch.linspace(-0.3, 0.3, config.n_embd))
        router.norm.weight.copy_(torch.linspace(0.8, 1.2, config.n_embd))
        router.gate.fill_(0.25)
    chunked = symbols["GPT"](config)
    chunked.load_state_dict(reference.state_dict(), strict=True)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    reference_gradients = tiny_gradient_vector(reference, tokens, targets, 1)
    effective_chunk = min(configured_chunk, tokens.size(1))
    chunked_gradients = tiny_gradient_vector(chunked, tokens, targets, effective_chunk)
    rows = {}
    maximum = 0.0
    maximum_relative = 0.0
    for name in sorted(reference_gradients):
        left = reference_gradients[name]
        right = chunked_gradients[name]
        difference = (left - right).abs().max().item()
        relative = difference / left.abs().max().clamp_min(1e-12).item()
        maximum = max(maximum, difference)
        maximum_relative = max(maximum_relative, relative)
        rows[name] = {
            "max_absolute_gradient_difference": difference,
            "relative_gradient_difference": relative,
            "reference_gradient_norm": left.norm().item(),
            "chunked_gradient_norm": right.norm().item(),
        }
    passed = maximum <= 2e-6 and maximum_relative <= 2e-5
    return {
        "dtype": "float32",
        "reference_backward_chunk_tokens": 1,
        "configured_backward_chunk_tokens": configured_chunk,
        "tested_backward_chunk_tokens": effective_chunk,
        "parameters": rows,
        "max_absolute_gradient_difference": maximum,
        "max_relative_gradient_difference": maximum_relative,
        "passed": passed,
    }


def cpu_fp32_writer_regression(symbols):
    """Identity and writer-gradient comparison against an explicit FP32 formula."""
    torch.manual_seed(20260816)
    config = symbols["GPTConfig"](
        block_size=4,
        vocab_size=32,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
        enable_memory_writers=True,
        memory_writer_rank=WRITER_RANK,
        memory_writer_init_seed=WRITER_INIT_SEED,
    )
    model = symbols["GPT"](config).float().eval()
    model.freeze_for_memory_writer_training()
    tokens = torch.randint(0, config.vocab_size, (2, 4))
    raw_state = model.init_recurrent_state(
        2, "masked_l1_topdown_self", device=tokens.device, dtype=torch.float32
    )
    writer_state = model.init_recurrent_state(
        2, "masked_l1_topdown_self", device=tokens.device, dtype=torch.float32
    )
    raw_logits = []
    writer_logits = []
    with torch.no_grad():
        for position in range(tokens.size(1)):
            raw_row, raw_state = model.forward_step(tokens[:, position], raw_state)
            writer_row, writer_state = model.forward_step(
                tokens[:, position], writer_state, use_memory_writers=True
            )
            raw_logits.append(raw_row)
            writer_logits.append(writer_row)
    identity = {
        "logits_bit_exact": torch.equal(torch.cat(raw_logits, 1), torch.cat(writer_logits, 1)),
        "final_memory_bit_exact": torch.equal(
            raw_state.feedback_memory, writer_state.feedback_memory
        ),
        "W_up_exactly_zero": all(
            torch.count_nonzero(
                model.transformer.memory_writers[f"writer_v{depth}"].W_up.weight
            ).item() == 0
            for depth in SOURCE_DEPTHS
        ),
    }
    identity["passed"] = all(identity.values())

    raw = torch.randn(4, 2, 1, config.n_embd)
    target = torch.randn(4, 2, 1, config.n_embd)
    with torch.no_grad():
        router = model.transformer.topdown_attnres
        router.query.copy_(torch.linspace(-0.3, 0.3, config.n_embd))
        router.norm.weight.copy_(torch.linspace(0.8, 1.2, config.n_embd))
        router.gate.fill_(0.25)
        for depth in SOURCE_DEPTHS:
            writer = model.transformer.memory_writers[f"writer_v{depth}"]
            writer.W_up.weight.normal_(mean=0.0, std=0.01)
    raw_path = model.init_recurrent_state(
        2, "masked_l1_topdown_self", device=tokens.device, dtype=torch.float32
    )
    writer_path = model.init_recurrent_state(
        2, "masked_l1_topdown_self", device=tokens.device, dtype=torch.float32
    )
    with torch.no_grad():
        raw_token0, raw_path = model.forward_step(tokens[:, 0], raw_path)
        writer_token0, writer_path = model.forward_step(
            tokens[:, 0], writer_path, use_memory_writers=True
        )
        raw_token1, raw_path = model.forward_step(tokens[:, 1], raw_path)
        writer_token1, writer_path = model.forward_step(
            tokens[:, 1], writer_path, use_memory_writers=True
        )
    memory_only_isolation = {
        "same_token_logits_bit_exact": torch.equal(raw_token0, writer_token0),
        "next_token_logits_changed": not torch.equal(raw_token1, writer_token1),
    }
    memory_only_isolation["passed"] = all(memory_only_isolation.values())

    model.zero_grad(set_to_none=True)
    adapted, _ = model.write_recurrent_memory(raw)
    loss = (adapted * target).sum()
    loss.backward()
    implementation = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if name in TRAINABLE_NAMES
    }
    explicit_parameters = {}
    explicit_loss = 0.0
    for index, depth in enumerate(SOURCE_DEPTHS):
        writer = model.transformer.memory_writers[f"writer_v{depth}"]
        down = writer.W_down.weight.detach().clone().requires_grad_(True)
        up = writer.W_up.weight.detach().clone().requires_grad_(True)
        source = raw[index].detach()
        normalized = source * torch.rsqrt(
            source.pow(2).mean(dim=-1, keepdim=True) + 1e-5
        )
        delta = F.linear(F.silu(F.linear(normalized, down)), up)
        explicit_loss = explicit_loss + ((source + delta) * target[index]).sum()
        explicit_parameters[
            f"transformer.memory_writers.writer_v{depth}.W_down.weight"
        ] = down
        explicit_parameters[
            f"transformer.memory_writers.writer_v{depth}.W_up.weight"
        ] = up
    explicit_loss.backward()
    rows = {}
    max_absolute = 0.0
    max_relative = 0.0
    for name, actual in sorted(implementation.items()):
        expected = explicit_parameters[name].grad
        absolute = (actual - expected).abs().max().item()
        relative = absolute / expected.abs().max().clamp_min(1e-12).item()
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, relative)
        rows[name] = {
            "maximum_absolute_gradient_difference": absolute,
            "maximum_relative_gradient_difference": relative,
        }
    gradient = {
        "parameters": rows,
        "maximum_absolute_gradient_difference": max_absolute,
        "maximum_relative_gradient_difference": max_relative,
        "passed": max_absolute <= 1e-6 and max_relative <= 1e-6,
    }
    differentiable_source = raw.detach().clone().requires_grad_(True)
    differentiable_memory, _ = model.write_recurrent_memory(differentiable_source)
    source_gradient = torch.autograd.grad(
        differentiable_memory.sum(), differentiable_source, allow_unused=True
    )[0]
    detach_boundary = {
        "writer_input_source_gradient_none": source_gradient is None,
        "writer_output_requires_grad": differentiable_memory.requires_grad,
    }
    detach_boundary["passed"] = all(detach_boundary.values())
    return {
        "dtype": "float32",
        "zero_writer_identity": identity,
        "memory_only_same_token_isolation": memory_only_isolation,
        "explicit_writer_gradient": gradient,
        "writer_input_detach_boundary": detach_boundary,
        "passed": identity["passed"]
        and memory_only_isolation["passed"]
        and gradient["passed"]
        and detach_boundary["passed"],
    }


def cpu_one_step_temporal_gradient_test(symbols):
    torch.manual_seed(20260817)
    config = symbols["GPTConfig"](
        block_size=4,
        vocab_size=32,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
        enable_memory_writers=True,
        memory_writer_rank=WRITER_RANK,
        memory_writer_init_seed=WRITER_INIT_SEED,
    )
    model = symbols["GPT"](config).float().eval()
    with torch.no_grad():
        router = model.transformer.topdown_attnres
        router.query.copy_(torch.linspace(-0.3, 0.3, config.n_embd))
        router.norm.weight.copy_(torch.linspace(0.8, 1.2, config.n_embd))
        router.gate.fill_(0.25)
    model.freeze_for_memory_writer_training()
    tokens = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))
    state = model.init_recurrent_state(
        2, "masked_l1_topdown_self", device=tokens.device, dtype=torch.float32
    )
    memories = []
    losses = []
    cache_source = {}

    def hook(_module, _inputs, output):
        if len(memories) == 1:
            cache_source["position_1_block_2_qkv"] = output

    handle = model.transformer.h[1].attn.c_attn.register_forward_hook(hook)
    try:
        for position in range(4):
            logits, state = model.forward_step(
                tokens[:, position], state, use_memory_writers=True
            )
            memories.append(state.feedback_memory)
            losses.append(F.cross_entropy(logits[:, 0], targets[:, position]))
    finally:
        handle.remove()

    def gradient(loss_index, tensor):
        return torch.autograd.grad(
            losses[loss_index], tensor, allow_unused=True, retain_graph=True
        )[0]

    loss1_m0 = gradient(1, memories[0])
    loss2_m1 = gradient(2, memories[1])
    loss2_m0 = gradient(2, memories[0])
    loss3_m2 = gradient(3, memories[2])
    loss3_m1 = gradient(3, memories[1])
    cache_grad = gradient(2, cache_source["position_1_block_2_qkv"])
    losses[1].backward()
    gradients = gradient_report(model)
    report = {
        "dtype": "float32",
        "loss1_to_m0_finite_nonzero": loss1_m0 is not None
        and bool(torch.isfinite(loss1_m0).all())
        and bool(torch.count_nonzero(loss1_m0)),
        "loss2_to_m1_finite_nonzero": loss2_m1 is not None
        and bool(torch.isfinite(loss2_m1).all())
        and bool(torch.count_nonzero(loss2_m1)),
        "loss2_to_m0_none": loss2_m0 is None,
        "loss3_to_m2_finite_nonzero": loss3_m2 is not None
        and bool(torch.isfinite(loss3_m2).all())
        and bool(torch.count_nonzero(loss3_m2)),
        "loss3_to_m1_none": loss3_m1 is None,
        "historical_kv_temporal_gradient_none": cache_grad is None,
        "reader_gradient_none": not gradients["reader_tensors_with_grad"],
        "base_gradient_none": not gradients["frozen_tensors_with_grad"],
        "first_update_W_up_nonzero": all(
            gradients[f"writer_v{depth}_W_up"]["nonzero"] for depth in SOURCE_DEPTHS
        ),
        "first_update_W_down_zero": all(
            gradients[f"writer_v{depth}_W_down"]["present"]
            and gradients[f"writer_v{depth}_W_down"]["finite"]
            and not gradients[f"writer_v{depth}_W_down"]["nonzero"]
            for depth in SOURCE_DEPTHS
        ),
    }
    report["passed"] = all(report.values())
    return report


def production_temporal_gradient_test(student, tokens, targets, expect_initial_staging=True):
    student.eval()
    student.zero_grad(set_to_none=True)
    memories = []
    losses = []
    cached_position_one = {}

    def capture_cache_source(_module, _inputs, output):
        if len(memories) == 1:
            output.retain_grad()
            cached_position_one["block2_qkv"] = output

    handle = student.transformer.h[1].attn.c_attn.register_forward_hook(
        capture_cache_source
    )
    stored_state_checks = []
    try:
        state = student.init_recurrent_state(
            tokens.size(0),
            "masked_l1_topdown_self",
            device=tokens.device,
            dtype=torch.bfloat16,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for position in range(4):
                logits, state = student.forward_step(
                    tokens[:, position], state, use_memory_writers=True
                )
                state.feedback_memory.retain_grad()
                memories.append(state.feedback_memory)
                stored_state_checks.append(state_health(state, position + 1))
                losses.append(F.cross_entropy(logits[:, 0], targets[:, position]))
    finally:
        handle.remove()
    if "block2_qkv" not in cached_position_one:
        raise SystemExit("historical-KV gradient diagnostic did not capture position 1")

    def memory_grad(loss_index, memory_index):
        gradient = torch.autograd.grad(
            losses[loss_index], memories[memory_index], allow_unused=True, retain_graph=True
        )[0]
        return {
            "present": gradient is not None,
            "nonzero": gradient is not None and bool(torch.count_nonzero(gradient).item()),
            "finite": gradient is not None and bool(torch.isfinite(gradient).all().item()),
        }

    loss1_to_m0 = memory_grad(1, 0)
    loss2_to_m1 = memory_grad(2, 1)
    loss2_to_m0 = memory_grad(2, 0)
    loss3_to_m2 = memory_grad(3, 2)
    loss3_to_m1 = memory_grad(3, 1)
    cache_gradient = torch.autograd.grad(
        losses[2], cached_position_one["block2_qkv"], allow_unused=True, retain_graph=True
    )[0]
    losses[1].backward()
    gradients = gradient_report(student)
    up_valid = all(
        all(gradients[f"writer_v{depth}_W_up"][key] for key in ("present", "finite", "nonzero"))
        for depth in SOURCE_DEPTHS
    )
    down_zero = all(
        gradients[f"writer_v{depth}_W_down"]["present"]
        and gradients[f"writer_v{depth}_W_down"]["finite"]
        and not gradients[f"writer_v{depth}_W_down"]["nonzero"]
        for depth in SOURCE_DEPTHS
    )
    report = {
        "loss1_to_m0": loss1_to_m0,
        "loss2_to_m1": loss2_to_m1,
        "loss2_to_m0": loss2_to_m0,
        "loss3_to_m2": loss3_to_m2,
        "loss3_to_m1": loss3_to_m1,
        "historical_kv_temporal_gradient_none": cache_gradient is None,
        "stored_state_after_each_token": stored_state_checks,
        "stored_writer_memory_graph_present": all(row["memory_finite_with_expected_graph"] for row in stored_state_checks),
        "stored_historical_kv_grad_fn_none": all(
            row["historical_kv_finite_detached"] for row in stored_state_checks
        ),
        "first_update_W_up_gradient_finite_nonzero": up_valid,
        "first_update_W_down_gradient_zero": down_zero if expect_initial_staging else None,
        "writer_gradients": gradients,
        "reader_gradients_none": not gradients["reader_tensors_with_grad"],
        "frozen_base_gradients_none": not gradients["frozen_tensors_with_grad"],
    }
    report["passed"] = (
        all(loss1_to_m0.values())
        and all(loss2_to_m1.values())
        and not loss2_to_m0["present"]
        and all(loss3_to_m2.values())
        and not loss3_to_m1["present"]
        and report["historical_kv_temporal_gradient_none"]
        and all(row["passed"] for row in stored_state_checks)
        and up_valid
        and (down_zero if expect_initial_staging else all(
            gradients[f"writer_v{depth}_W_down"]["present"]
            and gradients[f"writer_v{depth}_W_down"]["finite"]
            for depth in SOURCE_DEPTHS
        ))
        and report["reader_gradients_none"]
        and report["frozen_base_gradients_none"]
    )
    student.zero_grad(set_to_none=True)
    if not report["passed"]:
        raise SystemExit(f"temporal gradient boundary failed: {report}")
    return report


def run_preflight(args, device):
    config = validate_config()
    symbols, _, student, source = load_source_runtime(
        args.source_checkpoint, device, include_teacher=False
    )
    hashes_before = {
        group: state_subset_sha256(student, group)
        for group in ("base", "reader", "writers", "source_model")
    }
    loaders = a0.make_replay_loaders(symbols, source["dataloader_states"])
    first_next = a0.next_update_hash(loaders, symbols, replay=True)
    if first_next != SOURCE_NEXT_SHA256:
        raise SystemExit(f"2B1 data continuation mismatch: {first_next}")
    oracle = derive_replay_oracle(symbols, source["dataloader_states"], updates=10)
    durable_write_json(Path(args.run_dir) / "replay_oracle.json", {"updates": oracle})

    loader = validation_loader(symbols)
    x_cpu, y_cpu = loader.next_batch()
    x = x_cpu[:2, :64].to(device)
    y = y_cpu[:2, :64].to(device)
    equivalence = {}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parallel, _ = student(x, mode="full_context")
    incremental, _ = b0.incremental_logits(student, x, "full_context")
    equivalence["full_context"] = b0.equivalence_metrics(parallel, incremental, y)
    del parallel, incremental
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parallel, _ = student(x, mode="masked_l1_no_feedback")
    incremental, _ = b0.incremental_logits(student, x, "masked_l1_no_feedback")
    equivalence["masked_l1_no_feedback"] = b0.equivalence_metrics(
        parallel, incremental, y
    )
    del parallel, incremental
    thresholds = {
        "maximum_absolute_logit_difference": 0.30,
        "mean_absolute_logit_difference": 0.015,
        "relative_mean_absolute_logit_difference": 0.005,
        "argmax_agreement_fraction_minimum": 0.98,
        "absolute_loss_difference": 0.002,
    }
    equivalence_passed = all(
        row["maximum_absolute_logit_difference"]
        <= thresholds["maximum_absolute_logit_difference"]
        and row["mean_absolute_logit_difference"]
        <= thresholds["mean_absolute_logit_difference"]
        and row["relative_mean_absolute_logit_difference"]
        <= thresholds["relative_mean_absolute_logit_difference"]
        and row["argmax_agreement_fraction"]
        >= thresholds["argmax_agreement_fraction_minimum"]
        and row["absolute_loss_difference"]
        <= thresholds["absolute_loss_difference"]
        for row in equivalence.values()
    )
    causality = b0.causality_tests(student, x)
    cpu_regression = cpu_fp32_writer_regression(symbols)
    cpu_temporal = cpu_one_step_temporal_gradient_test(symbols)
    chunked = chunked_gradient_equivalence(symbols, config["backward_chunk_tokens"])
    temporal = production_temporal_gradient_test(student, x[:, :4], y[:, :4])

    stability = {}
    for horizon in (8, 16, 32, 64):
        result = stream_loss(
            student,
            x[:, :horizon],
            y[:, :horizon],
            mode="masked_l1_topdown_self",
            use_writers=True,
        )
        stability[str(horizon)] = result
    stability_passed = all(
        row["finite"]
        and row["cache_health"]["all_expected_lengths"]
        and row["cache_health"]["block_1_cache_absent"]
        for row in stability.values()
    )
    hashes_after = {
        group: state_subset_sha256(student, group)
        for group in ("base", "reader", "writers", "source_model")
    }
    initialization = {
        "rank": WRITER_RANK,
        "seed": WRITER_INIT_SEED,
        "parameters": sum(
            parameter.numel() for name, parameter in student.named_parameters()
            if name in TRAINABLE_NAMES
        ),
        "W_up_exactly_zero": all(
            torch.count_nonzero(
                student.transformer.memory_writers[f"writer_v{depth}"].W_up.weight
            ).item() == 0
            for depth in SOURCE_DEPTHS
        ),
        "W_down_nonzero": all(
            torch.count_nonzero(
                student.transformer.memory_writers[f"writer_v{depth}"].W_down.weight
            ).item() > 0
            for depth in SOURCE_DEPTHS
        ),
    }
    initialization["passed"] = (
        initialization["parameters"] == WRITER_PARAMETER_COUNT
        and initialization["W_up_exactly_zero"]
        and initialization["W_down_nonzero"]
    )
    report = {
        "experiment": "2B2",
        "stage": "preflight",
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": BRANCH,
        "source": source_report(source),
        "checkpoint_sha256_exact": file_sha256(args.source_checkpoint)
        == SOURCE_CHECKPOINT_SHA256,
        "initial_reader": reader_values(student),
        "writer_initialization": initialization,
        "trainable_parameter_names": assert_runtime_contract(student),
        "trainable_parameters": WRITER_PARAMETER_COUNT,
        "fresh_optimizer_constructed": False,
        "teacher_constructed": False,
        "teacher_forward_calls": 0,
        "data_next_global_batch_sha256": first_next,
        "replay_oracle": oracle,
        "dataset": a0.dataset_manifest_report(verify_shards=False),
        "incremental_equivalence": equivalence,
        "incremental_equivalence_thresholds": thresholds,
        "incremental_equivalence_passed": equivalence_passed,
        "causality_and_isolation": causality,
        "cpu_fp32_mathematical_regression": cpu_regression,
        "cpu_fp32_one_step_temporal_gradient": cpu_temporal,
        "chunked_backward_equivalence": chunked,
        "temporal_gradient_boundary": temporal,
        "short_horizon_stability": stability,
        "short_horizon_stability_passed": stability_passed,
        "state_hashes_before": hashes_before,
        "state_hashes_after": hashes_after,
        "model_state_unchanged": hashes_before == hashes_after,
        "source_checkpoint_sha256_after": file_sha256(args.source_checkpoint),
        "hellaswag_run": False,
    }
    report["passed"] = all(
        (
            report["checkpoint_sha256_exact"],
            first_next == SOURCE_NEXT_SHA256,
            equivalence_passed,
            causality["passed"],
            cpu_regression["passed"],
            cpu_temporal["passed"],
            chunked["passed"],
            temporal["passed"],
            stability_passed,
            initialization["passed"],
            report["model_state_unchanged"],
        )
    )
    durable_write_json(Path(args.run_dir) / "preflight.json", report)
    if not report["passed"]:
        raise SystemExit("Experiment 2B2 preflight failed")
    return report


def load_progress(path):
    path = Path(path)
    if not path.is_file():
        return {"batches": []}
    payload = load_json(path)
    if set(payload) != {"batches"} or not isinstance(payload["batches"], list):
        raise SystemExit(f"invalid progress artifact: {path}")
    return payload


def progress_row(progress, batch_index, payload_hash):
    matches = [row for row in progress["batches"] if row.get("batch_index") == batch_index]
    if len(matches) > 1:
        raise SystemExit(f"duplicate progress batch {batch_index}")
    if matches:
        if matches[0].get("payload_sha256") != payload_hash:
            raise SystemExit(f"progress payload mismatch at batch {batch_index}")
        return matches[0]
    row = {"batch_index": batch_index, "payload_sha256": payload_hash}
    progress["batches"].append(row)
    progress["batches"].sort(key=lambda item: item["batch_index"])
    return row


def require_passed(path, label):
    path = Path(path)
    if not path.is_file() or not load_json(path).get("passed"):
        raise SystemExit(f"{label} gate has not passed: {path}")


def run_baseline(args, device):
    require_passed(Path(args.run_dir) / "preflight.json", "preflight")
    symbols, _, student, source = load_source_runtime(
        args.source_checkpoint, device, include_teacher=False
    )
    student.eval()
    hashes_before = {
        group: state_subset_sha256(student, group)
        for group in ("base", "reader", "writers")
    }
    reference = reference_validation()
    loader = validation_loader(symbols)
    progress_path = Path(args.run_dir) / "baseline_progress.json"
    progress = load_progress(progress_path)
    validation_digest = hashlib.sha256()
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        payload_hash = a0.batch_payload_hash(x_cpu, y_cpu)
        if payload_hash != reference["batch_payload_sha256"][batch_index]:
            raise SystemExit(f"baseline validation payload mismatch at {batch_index}")
        validation_digest.update(bytes.fromhex(payload_hash))
        row = progress_row(progress, batch_index, payload_hash)
        required = {
            "masked_l1_no_feedback",
            "source_2b1_self_bypass",
            "zero_writer_self",
            "source_2b1_gate_zero_bypass",
            "zero_writer_gate_zero",
        }
        if required <= set(row):
            print(f"baseline {batch_index + 1:02d}/20 reused", flush=True)
            continue
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        if "masked_l1_no_feedback" not in row:
            row["masked_l1_no_feedback"] = b0.parallel_loss(
                student, x, y, "masked_l1_no_feedback"
            )
            durable_write_json(progress_path, progress)
        if "source_2b1_self_bypass" not in row:
            row["source_2b1_self_bypass"] = stream_loss(
                student, x, y, mode="masked_l1_topdown_self", use_writers=False
            )
            durable_write_json(progress_path, progress)
        if "zero_writer_self" not in row:
            row["zero_writer_self"] = stream_loss(
                student, x, y, mode="masked_l1_topdown_self", use_writers=True
            )
            durable_write_json(progress_path, progress)
        if "source_2b1_gate_zero_bypass" not in row:
            row["source_2b1_gate_zero_bypass"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_topdown_self",
                use_writers=False,
                gate_override=0.0,
            )
            durable_write_json(progress_path, progress)
        if "zero_writer_gate_zero" not in row:
            row["zero_writer_gate_zero"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_topdown_self",
                use_writers=True,
                gate_override=0.0,
            )
            durable_write_json(progress_path, progress)
        print(
            f"baseline {batch_index + 1:02d}/20 "
            f"masked={row['masked_l1_no_feedback']:.6f} "
            f"self={row['zero_writer_self']['loss']:.6f} "
            f"gate0={row['zero_writer_gate_zero']['loss']:.6f}",
            flush=True,
        )
        del x, y
        torch.cuda.empty_cache()
    if validation_digest.hexdigest() != a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256:
        raise SystemExit("baseline canonical validation hash mismatch")
    rows = progress["batches"]
    if len(rows) != a0.VALIDATION_BATCHES:
        raise SystemExit("baseline progress is incomplete")
    losses = {
        "masked_l1_no_feedback": statistics.fmean(
            row["masked_l1_no_feedback"] for row in rows
        ),
        "source_2b1_self_bypass": statistics.fmean(
            row["source_2b1_self_bypass"]["loss"] for row in rows
        ),
        "zero_writer_self": statistics.fmean(
            row["zero_writer_self"]["loss"] for row in rows
        ),
        "source_2b1_gate_zero_bypass": statistics.fmean(
            row["source_2b1_gate_zero_bypass"]["loss"] for row in rows
        ),
        "zero_writer_gate_zero": statistics.fmean(
            row["zero_writer_gate_zero"]["loss"] for row in rows
        ),
    }
    tolerances = {
        "masked_l1_no_feedback": 5e-4,
        "source_2b1_self_bypass": 5e-4,
        "zero_writer_self": 5e-4,
        "source_2b1_gate_zero_bypass": 5e-4,
        "zero_writer_gate_zero": 5e-4,
    }
    expected = {
        "masked_l1_no_feedback": PINNED["masked_l1_no_feedback"],
        "source_2b1_self_bypass": PINNED["source_2b1_self"],
        "zero_writer_self": PINNED["source_2b1_self"],
        "source_2b1_gate_zero_bypass": PINNED["source_2b1_gate_zero"],
        "zero_writer_gate_zero": PINNED["source_2b1_gate_zero"],
    }
    regression = {
        name: {
            "actual": losses[name],
            "expected": expected[name],
            "absolute_difference": abs(losses[name] - expected[name]),
            "tolerance": tolerances[name],
            "passed": abs(losses[name] - expected[name]) <= tolerances[name],
        }
        for name in losses
    }
    hashes_after = {
        group: state_subset_sha256(student, group)
        for group in ("base", "reader", "writers")
    }
    report = {
        "experiment": "2B2",
        "stage": "pre_training_canonical_regression",
        "source": source_report(source),
        "validation_batches": a0.VALIDATION_BATCHES,
        "B": a0.VALIDATION_B,
        "T": a0.T,
        "validation_global_batches_sha256": validation_digest.hexdigest(),
        "losses": losses,
        "regression": regression,
        "batch_losses": {
            "masked_l1_no_feedback": [row["masked_l1_no_feedback"] for row in rows],
            "source_2b1_self_bypass": [
                row["source_2b1_self_bypass"]["loss"] for row in rows
            ],
            "zero_writer_self": [row["zero_writer_self"]["loss"] for row in rows],
            "source_2b1_gate_zero_bypass": [
                row["source_2b1_gate_zero_bypass"]["loss"] for row in rows
            ],
            "zero_writer_gate_zero": [
                row["zero_writer_gate_zero"]["loss"] for row in rows
            ],
        },
        "zero_writer_self_routing": b0.average_routing(
            [row["zero_writer_self"] for row in rows]
        ),
        "writer_disabled_and_zero_initialized_agree": all(
            row["source_2b1_self_bypass"]["loss"] == row["zero_writer_self"]["loss"]
            and row["source_2b1_gate_zero_bypass"]["loss"] == row["zero_writer_gate_zero"]["loss"]
            for row in rows
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "teacher_constructed": False,
        "teacher_training_forward_calls": 0,
        "state_hashes_before": hashes_before,
        "state_hashes_after": hashes_after,
        "model_state_unchanged": hashes_before == hashes_after,
        "source_checkpoint_sha256_after": file_sha256(args.source_checkpoint),
        "hellaswag_run": False,
    }
    report["passed"] = (
        all(row["passed"] for row in regression.values())
        and report["writer_disabled_and_zero_initialized_agree"]
        and report["model_state_unchanged"]
        and report["source_checkpoint_sha256_after"] == SOURCE_CHECKPOINT_SHA256
    )
    durable_write_json(Path(args.run_dir) / "baseline.json", report)
    if not report["passed"]:
        raise SystemExit(f"zero-shot baseline regression failed: {regression}")
    return report


def smoke_loader(symbols):
    return symbols["DataLoaderLite"](
        B=2, T=64, process_rank=0, num_processes=1, split="train"
    )


def run_smoke_phase1(args, device):
    require_passed(Path(args.run_dir) / "preflight.json", "preflight")
    require_passed(Path(args.run_dir) / "baseline.json", "baseline")
    config = validate_config()
    symbols, _, student, source = load_source_runtime(
        args.source_checkpoint,
        device,
        include_teacher=False,
        restore_source_rng=True,
    )
    optimizer = fresh_optimizer(student)
    optimizer_report(optimizer, 0)
    loader = smoke_loader(symbols)
    metadata = make_metadata("disposable_smoke", config, source)
    smoke_dir = Path(args.run_dir) / "smoke"
    if (smoke_dir / "checkpoint_updates_000002.pt").exists():
        raise SystemExit("smoke midpoint checkpoint already exists")
    frozen_before = {
        group: state_subset_sha256(student, group) for group in ("base", "reader")
    }
    rows = []
    for local_update in (1, 2):
        expected = a0.next_update_hash([loader], symbols, replay=False)
        row = train_recurrent_update(
            student,
            optimizer,
            a0.update_batches([loader], replay=False),
            target_count=2 * 64,
            backward_chunk=config["backward_chunk_tokens"],
            local_update=local_update,
            expected_batch_sha256=expected,
        )
        rows.append(row)
        durable_append_jsonl(smoke_dir / "metrics.jsonl", row)
        print(
            f"smoke update {local_update}/3 loss={row['global_mean_training_loss']:.6f}",
            flush=True,
        )
    frozen_after = {
        group: state_subset_sha256(student, group) for group in ("base", "reader")
    }
    if frozen_after != frozen_before:
        raise SystemExit("frozen base or reader changed during smoke phase 1")
    state = {
        "local_completed_updates": 2,
        "processed_2b2_tokens": 2 * 2 * 64,
        "fineweb_lineage_completed_update": None,
        "kind": "disposable_smoke",
    }
    checkpoint_path = smoke_dir / "checkpoint_updates_000002.pt"
    sidecar = save_checkpoint(
        checkpoint_path,
        student,
        optimizer,
        [loader],
        symbols,
        replay=False,
        training_state=state,
        metadata=metadata,
    )
    report = {
        "stage": "smoke_phase1",
        "updates": rows,
        "checkpoint": sidecar,
        "next_smoke_batch_sha256": sidecar["verification"][
            "next_global_batch_sha256"
        ],
        "source_model_sha256": source["source_model_sha256"],
        "frozen_hashes_before": frozen_before,
        "frozen_hashes_after": frozen_after,
        "frozen_base_and_reader_unchanged": True,
        "teacher_training_forward_calls": 0,
        "passed": True,
        "process_exit_required": True,
    }
    durable_write_json(smoke_dir / "phase1.json", report)
    return report


def run_smoke_phase2(args, device):
    config = validate_config()
    require_passed(Path(args.run_dir) / "smoke" / "phase1.json", "smoke phase 1")
    symbols, _, student, source = load_source_runtime(
        args.source_checkpoint, device, include_teacher=False
    )
    optimizer = fresh_optimizer(student)
    metadata = make_metadata("disposable_smoke", config, source)
    checkpoint_path = Path(args.run_dir) / "smoke" / "checkpoint_updates_000002.pt"
    loaders, state, resume_audit = load_checkpoint_into_fresh(
        checkpoint_path,
        student,
        optimizer,
        symbols,
        replay=False,
        expected_metadata=metadata,
    )
    if state["local_completed_updates"] != 2 or len(loaders) != 1:
        raise SystemExit("smoke resume state mismatch")
    expected = resume_audit["restored_next_global_batch_sha256"]
    frozen_before = {
        group: state_subset_sha256(student, group) for group in ("base", "reader")
    }
    row = train_recurrent_update(
        student,
        optimizer,
        a0.update_batches(loaders, replay=False),
        target_count=2 * 64,
        backward_chunk=config["backward_chunk_tokens"],
        local_update=3,
        expected_batch_sha256=expected,
    )
    durable_append_jsonl(Path(args.run_dir) / "smoke" / "metrics.jsonl", row)
    frozen_after = {
        group: state_subset_sha256(student, group) for group in ("base", "reader")
    }
    if frozen_after != frozen_before:
        raise SystemExit("frozen base or reader changed during smoke phase 2")
    phase1 = load_json(Path(args.run_dir) / "smoke" / "phase1.json")
    all_rows = phase1["updates"] + [row]
    diagnostic_loader = symbols["DataLoaderLite"](
        B=2, T=4, process_rank=0, num_processes=1, split="val"
    )
    diagnostic_x, diagnostic_y = diagnostic_loader.next_batch()
    temporal_after_smoke = production_temporal_gradient_test(
        student,
        diagnostic_x.to(device),
        diagnostic_y.to(device),
        expect_initial_staging=False,
    )
    report = {
        "stage": "disposable_smoke_complete",
        "B": 2,
        "T": 64,
        "optimizer_updates": 3,
        "updates": all_rows,
        "resume_audit": resume_audit,
        "update_3_consumed_expected_next_batch": row["global_batch_sha256"] == expected,
        "all_losses_finite": all(
            math.isfinite(item["global_mean_training_loss"]) for item in all_rows
        ),
        "update_1_W_up_nonzero_W_down_zero": all(
            all(all_rows[0]["gradients"][f"writer_v{depth}_W_up"][key]
                for key in ("present", "finite", "nonzero"))
            and all_rows[0]["gradients"][f"writer_v{depth}_W_down"]["present"]
            and all_rows[0]["gradients"][f"writer_v{depth}_W_down"]["finite"]
            and not all_rows[0]["gradients"][f"writer_v{depth}_W_down"]["nonzero"]
            for depth in SOURCE_DEPTHS
        ),
        "updates_2_and_3_writer_gradients_finite_nonzero": all(
            all(
                all(item["gradients"][f"writer_v{depth}_W_{projection}"][key]
                    for key in ("present", "finite", "nonzero"))
                for depth in SOURCE_DEPTHS for projection in ("down", "up")
            )
            for item in all_rows[1:]
        ),
        "all_reader_gradients_none": all(
            not item["gradients"]["reader_tensors_with_grad"] for item in all_rows
        ),
        "all_frozen_gradients_none": all(
            not item["gradients"]["frozen_tensors_with_grad"] for item in all_rows
        ),
        "all_optimizer_moments_finite": all(
            item["optimizer"]["moments_finite"] for item in all_rows
        ),
        "all_recurrent_states_valid": all(
            all(state_row["passed"] for state_row in item["state_health"])
            for item in all_rows
        ),
        "teacher_training_forward_calls": sum(
            item["teacher_forward_calls_during_training"] for item in all_rows
        ),
        "frozen_base_unchanged": frozen_after == frozen_before,
        "frozen_reader_unchanged": frozen_after == frozen_before,
        "one_step_temporal_gradient_after_update_3": temporal_after_smoke,
        "smoke_state_disposition": "discarded; result run must reload untouched 2B1 source",
        "hellaswag_run": False,
    }
    report["passed"] = all(
        report[name]
        for name in (
            "update_3_consumed_expected_next_batch",
            "all_losses_finite",
            "update_1_W_up_nonzero_W_down_zero",
            "updates_2_and_3_writer_gradients_finite_nonzero",
            "all_reader_gradients_none",
            "all_frozen_gradients_none",
            "all_optimizer_moments_finite",
            "all_recurrent_states_valid",
            "frozen_base_unchanged",
            "frozen_reader_unchanged",
        )
    ) and report["teacher_training_forward_calls"] == 0 and temporal_after_smoke["passed"]
    durable_write_json(Path(args.run_dir) / "smoke" / "summary.json", report)
    if not report["passed"]:
        raise SystemExit("disposable smoke failed")
    print(f"smoke update 3/3 loss={row['global_mean_training_loss']:.6f}", flush=True)
    return report


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def oracle_map(run_dir):
    rows = load_json(Path(run_dir) / "replay_oracle.json")["updates"]
    if [row["local_update"] for row in rows] != list(range(1, 11)):
        raise SystemExit("invalid 2B2 replay oracle update vector")
    return {row["local_update"]: row["global_batch_sha256"] for row in rows}


def result_training_state(local_completed_updates):
    return {
        "local_completed_updates": local_completed_updates,
        "processed_2b2_tokens": local_completed_updates * a0.GLOBAL_BATCH_TOKENS,
        "fineweb_lineage_completed_update": (
            SOURCE_STATE["fineweb_lineage_completed_update"] + local_completed_updates
        ),
        "kind": "result_5m",
    }


def run_result_phase1(args, device):
    require_passed(Path(args.run_dir) / "preflight.json", "preflight")
    require_passed(Path(args.run_dir) / "baseline.json", "baseline")
    require_passed(Path(args.run_dir) / "smoke" / "summary.json", "smoke")
    config = validate_config()
    symbols, _, student, source = load_source_runtime(
        args.source_checkpoint,
        device,
        include_teacher=False,
        restore_source_rng=True,
    )
    source_state_hash = state_subset_sha256(student, "source_model")
    if source_state_hash != source["source_model_sha256"]:
        raise SystemExit("result run did not start from untouched 2B1 model state")
    optimizer = fresh_optimizer(student)
    optimizer_report(optimizer, 0)
    loaders = a0.make_replay_loaders(symbols, source["dataloader_states"])
    initial_next = a0.next_update_hash(loaders, symbols, replay=True)
    if initial_next != SOURCE_NEXT_SHA256:
        raise SystemExit("result run initial replay hash mismatch")
    metadata = make_metadata("result_5m", config, source)
    result_dir = Path(args.run_dir) / "result"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise SystemExit("result directory is not empty; refusing a second result run")
    result_dir.mkdir(parents=True, exist_ok=True)
    durable_write_json(result_dir / "config.json", config)
    durable_write_json(result_dir / "metadata.json", metadata)
    durable_write_json(
        result_dir / "initialization.json",
        {
            "source_2b1_checkpoint": source["path"],
            "source_2b1_checkpoint_sha256": source["sha256"],
            "source_model_state_sha256": source_state_hash,
            "source_frozen_base_sha256": source["frozen_base_sha256"],
            "source_frozen_reader_sha256": source["frozen_reader_sha256"],
            "source_reader": source["initial_reader"],
            "initial_writer_sha256": source["initial_writer_sha256"],
            "writer_rank": WRITER_RANK,
            "writer_parameters": WRITER_PARAMETER_COUNT,
            "writer_initialization_seed": WRITER_INIT_SEED,
            "zero_writer_identity_regression_passed": load_json(
                Path(args.run_dir) / "baseline.json"
            )["passed"],
            "source_optimizer_moments_restored": False,
            "fresh_optimizer_state_entries": 0,
            "source_next_global_batch_sha256": source["next_global_batch_sha256"],
            "replayed_next_global_batch_sha256": initial_next,
            "source_rng_restored": True,
            "smoke_state_loaded": False,
            "passed": True,
        },
    )
    expected_hashes = oracle_map(args.run_dir)
    frozen_before = {
        group: state_subset_sha256(student, group) for group in ("base", "reader")
    }
    for local_update in range(1, 6):
        preview = a0.next_update_hash(loaders, symbols, replay=True)
        if preview != expected_hashes[local_update]:
            raise SystemExit(f"pre-step replay mismatch at update {local_update}")
        row = train_recurrent_update(
            student,
            optimizer,
            a0.update_batches(loaders, replay=True),
            target_count=a0.GLOBAL_BATCH_TOKENS,
            backward_chunk=config["backward_chunk_tokens"],
            local_update=local_update,
            expected_batch_sha256=expected_hashes[local_update],
        )
        frozen_live = {
            group: state_subset_sha256(student, group) for group in ("base", "reader")
        }
        if frozen_live != frozen_before:
            raise SystemExit(f"frozen base/reader changed at result update {local_update}")
        durable_append_jsonl(result_dir / "metrics.jsonl", row)
        print(
            f"result update {local_update:02d}/10 "
            f"loss={row['global_mean_training_loss']:.6f} "
            f"writer_up={row['W_up_aggregate_norm']:.6f} wall={row['wall_seconds']:.1f}s",
            flush=True,
        )
    checkpoint_path = result_dir / "checkpoints" / "checkpoint_updates_000005.pt"
    sidecar = save_checkpoint(
        checkpoint_path,
        student,
        optimizer,
        loaders,
        symbols,
        replay=True,
        training_state=result_training_state(5),
        metadata=metadata,
    )
    phase1 = {
        "stage": "result_phase1_complete",
        "process_pid": os.getpid(),
        "completed_updates": 5,
        "processed_2b2_tokens": 5 * a0.GLOBAL_BATCH_TOKENS,
        "checkpoint": sidecar,
        "next_global_batch_sha256": sidecar["verification"][
            "next_global_batch_sha256"
        ],
        "frozen_hashes_before": frozen_before,
        "frozen_hashes_after": {
            group: state_subset_sha256(student, group) for group in ("base", "reader")
        },
        "teacher_training_forward_calls": 0,
        "passed": True,
        "process_exit_required": True,
    }
    durable_write_json(result_dir / "phase1.json", phase1)
    return phase1


def validate_result_metrics(rows, completed):
    if [row.get("local_update") for row in rows] != list(range(1, completed + 1)):
        raise SystemExit("result metrics update sequence mismatch")
    for row in rows:
        local_update = row["local_update"]
        if row["processed_2b2_tokens"] != local_update * a0.GLOBAL_BATCH_TOKENS:
            raise SystemExit("result metrics token counter mismatch")
        if row["targets"] != a0.GLOBAL_BATCH_TOKENS:
            raise SystemExit("result metrics target count mismatch")
        if row["teacher_forward_calls_during_training"] != 0:
            raise SystemExit("teacher participated in result training")


def run_result_phase2(args, device):
    config = validate_config()
    require_passed(Path(args.run_dir) / "result" / "phase1.json", "result phase 1")
    symbols, _, student, source = load_source_runtime(
        args.source_checkpoint, device, include_teacher=False
    )
    optimizer = fresh_optimizer(student)
    expected_metadata = make_metadata("result_5m", config, source)
    stored_metadata = load_json(Path(args.run_dir) / "result" / "metadata.json")
    if stored_metadata != expected_metadata:
        raise SystemExit("live result metadata/source files differ at phase-2 restart")
    checkpoint_path = (
        Path(args.run_dir) / "result" / "checkpoints" / "checkpoint_updates_000005.pt"
    )
    loaders, state, resume_audit = load_checkpoint_into_fresh(
        checkpoint_path,
        student,
        optimizer,
        symbols,
        replay=True,
        expected_metadata=expected_metadata,
    )
    if state != result_training_state(5):
        raise SystemExit(f"result midpoint training state mismatch: {state}")
    phase1 = load_json(Path(args.run_dir) / "result" / "phase1.json")
    if phase1["process_pid"] == os.getpid():
        raise SystemExit("result phase 2 did not start in a fresh process")
    resume_audit["phase1_process_pid"] = phase1["process_pid"]
    resume_audit["phase2_process_pid"] = os.getpid()
    resume_audit["phase_processes_distinct"] = phase1["process_pid"] != os.getpid()
    durable_write_json(Path(args.run_dir) / "result" / "resume_audit.json", resume_audit)
    rows = read_jsonl(Path(args.run_dir) / "result" / "metrics.jsonl")
    validate_result_metrics(rows, 5)
    expected_hashes = oracle_map(args.run_dir)
    if resume_audit["restored_next_global_batch_sha256"] != expected_hashes[6]:
        raise SystemExit("restored update-6 batch hash disagrees with replay oracle")
    frozen_before = {
        group: state_subset_sha256(student, group) for group in ("base", "reader")
    }
    for local_update in range(6, 11):
        preview = a0.next_update_hash(loaders, symbols, replay=True)
        if preview != expected_hashes[local_update]:
            raise SystemExit(f"pre-step replay mismatch at update {local_update}")
        row = train_recurrent_update(
            student,
            optimizer,
            a0.update_batches(loaders, replay=True),
            target_count=a0.GLOBAL_BATCH_TOKENS,
            backward_chunk=config["backward_chunk_tokens"],
            local_update=local_update,
            expected_batch_sha256=expected_hashes[local_update],
        )
        if local_update == 6:
            row["fresh_process_update_6_batch_verified"] = (
                row["global_batch_sha256"]
                == resume_audit["restored_next_global_batch_sha256"]
            )
            if not row["fresh_process_update_6_batch_verified"]:
                raise SystemExit("fresh-process update 6 batch verification failed")
        frozen_live = {
            group: state_subset_sha256(student, group) for group in ("base", "reader")
        }
        if frozen_live != frozen_before:
            raise SystemExit(f"frozen base/reader changed at result update {local_update}")
        durable_append_jsonl(Path(args.run_dir) / "result" / "metrics.jsonl", row)
        print(
            f"result update {local_update:02d}/10 "
            f"loss={row['global_mean_training_loss']:.6f} "
            f"writer_up={row['W_up_aggregate_norm']:.6f} wall={row['wall_seconds']:.1f}s",
            flush=True,
        )
    final_path = (
        Path(args.run_dir) / "result" / "checkpoints" / "checkpoint_updates_000010.pt"
    )
    sidecar = save_checkpoint(
        final_path,
        student,
        optimizer,
        loaders,
        symbols,
        replay=True,
        training_state=result_training_state(10),
        metadata=expected_metadata,
    )
    rows = read_jsonl(Path(args.run_dir) / "result" / "metrics.jsonl")
    validate_result_metrics(rows, 10)
    summary = {
        "stage": "result_training_complete",
        "updates": 10,
        "processed_2b2_tokens": 10 * a0.GLOBAL_BATCH_TOKENS,
        "targets_per_update": a0.GLOBAL_BATCH_TOKENS,
        "final_checkpoint": sidecar,
        "midpoint_checkpoint_sha256": phase1["checkpoint"]["sha256"],
        "resume_audit": resume_audit,
        "total_training_wall_seconds": sum(row["wall_seconds"] for row in rows),
        "peak_allocated_mb": max(row["peak_allocated_mb"] for row in rows),
        "peak_reserved_mb": max(row["peak_reserved_mb"] for row in rows),
        "initial_reader": source["initial_reader"],
        "final_reader": reader_values(student),
        "final_writer_sha256": state_subset_sha256(student, "writers"),
        "frozen_hashes_before_phase2": frozen_before,
        "frozen_base_sha256_after": state_subset_sha256(student, "base"),
        "frozen_reader_sha256_after": state_subset_sha256(student, "reader"),
        "frozen_base_unchanged": state_subset_sha256(student, "base")
        == source["frozen_base_sha256"],
        "frozen_reader_unchanged": state_subset_sha256(student, "reader")
        == source["frozen_reader_sha256"],
        "teacher_training_forward_calls": sum(
            row["teacher_forward_calls_during_training"] for row in rows
        ),
        "source_checkpoint_sha256_after": file_sha256(args.source_checkpoint),
        "hellaswag_run": False,
        "authorized_optimizer_updates_exhausted": True,
    }
    summary["passed"] = (
        summary["frozen_base_unchanged"]
        and summary["frozen_reader_unchanged"]
        and summary["teacher_training_forward_calls"] == 0
        and summary["source_checkpoint_sha256_after"] == SOURCE_CHECKPOINT_SHA256
        and resume_audit["passed"]
        and rows[5].get("fresh_process_update_6_batch_verified", False)
    )
    durable_write_json(Path(args.run_dir) / "result" / "training_summary.json", summary)
    if not summary["passed"]:
        raise SystemExit("result training audit failed")
    return summary


def load_trained_for_evaluation(args, device):
    symbols, teacher, student, source = load_source_runtime(
        args.source_checkpoint, device, include_teacher=True
    )
    final_path = (
        Path(args.run_dir) / "result" / "checkpoints" / "checkpoint_updates_000010.pt"
    )
    sidecar = verify_checkpoint_sidecars(final_path)
    checkpoint = a0.torch_load(final_path, mmap=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("final 2B2 checkpoint schema mismatch")
    if checkpoint.get("training_state") != result_training_state(10):
        raise SystemExit("final 2B2 checkpoint training-state mismatch")
    stored_metadata = load_json(Path(args.run_dir) / "result" / "metadata.json")
    if checkpoint.get("metadata") != stored_metadata:
        raise SystemExit("final 2B2 checkpoint metadata mismatch")
    student.load_state_dict(checkpoint["model"], strict=True)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.eval()
    teacher.eval()
    info = {
        "path": str(final_path.resolve()),
        "sha256": sidecar["sha256"],
        "training_state": copy.deepcopy(checkpoint["training_state"]),
        "reader": reader_values(student),
        "base_sha256": state_subset_sha256(student, "base"),
        "reader_sha256": state_subset_sha256(student, "reader"),
        "writer_sha256": state_subset_sha256(student, "writers"),
    }
    del checkpoint
    gc.collect()
    return symbols, teacher, student, source, info


def paired_statistics(real, shuffled):
    differences = [b - a for a, b in zip(real, shuffled)]
    return {
        "real_wins": sum(value > 0 for value in differences),
        "shuffled_wins": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "mean": statistics.fmean(differences),
        "median": statistics.median(differences),
        "sample_standard_deviation": statistics.stdev(differences),
        "minimum": min(differences),
        "maximum": max(differences),
        "differences": differences,
    }


def average_writer_behavior(rows):
    return {
        f"v{depth}": {
            field: statistics.fmean(
                row["writer_behavior"][f"v{depth}"][field] for row in rows
            )
            for field in (
                "source_rms",
                "adapted_memory_rms",
                "delta_rms",
                "delta_to_source_rms_ratio",
                "cosine_source_adapted",
            )
        } | {
            "maximum_delta_to_source_rms_ratio": max(
                row["writer_behavior"][f"v{depth}"][
                    "maximum_delta_to_source_rms_ratio"
                ]
                for row in rows
            )
        }
        for depth in SOURCE_DEPTHS
    }


def average_representation_drift(rows):
    result = {}
    for kind in ("teacher_vs_raw_student", "teacher_vs_adapted_memory"):
        result[kind] = {}
        for label, _, _ in b0.POSITION_BINS:
            result[kind][label] = {}
            for depth in SOURCE_DEPTHS:
                values = [
                    row["representation_drift"][kind][label][f"v{depth}"]
                    for row in rows
                ]
                result[kind][label][f"v{depth}"] = {
                    field: statistics.fmean(value[field] for value in values)
                    for field in ("cosine_similarity", "rms_difference", "norm_ratio")
                }
    return result


def run_canonical_evaluation(args, device):
    require_passed(
        Path(args.run_dir) / "result" / "training_summary.json", "result training"
    )
    baseline = load_json(Path(args.run_dir) / "baseline.json")
    reference = reference_validation()
    symbols, teacher, student, source, trained = load_trained_for_evaluation(
        args, device
    )
    hashes_before = {
        group: state_subset_sha256(student, group)
        for group in ("base", "reader", "writers")
    }
    loader = validation_loader(symbols)
    permutation = symbols["fixed_derangement"](a0.VALIDATION_B, device)
    progress_path = Path(args.run_dir) / "evaluation" / "canonical_progress.json"
    progress = load_progress(progress_path)
    validation_digest = hashlib.sha256()
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        payload_hash = a0.batch_payload_hash(x_cpu, y_cpu)
        if payload_hash != reference["batch_payload_sha256"][batch_index]:
            raise SystemExit(f"trained canonical payload mismatch at {batch_index}")
        validation_digest.update(bytes.fromhex(payload_hash))
        row = progress_row(progress, batch_index, payload_hash)
        required = {
            "full_context",
            "masked_l1_no_feedback",
            "trained_writer_real",
            "trained_writer_shuffled",
            "trained_writer_bypass",
            "trained_writer_gate_zero",
            "teacher_sources_with_writers",
        }
        if required <= set(row):
            print(f"trained canonical {batch_index + 1:02d}/20 reused", flush=True)
            continue
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        if "full_context" not in row:
            row["full_context"] = b0.parallel_loss(student, x, y, "full_context")
            durable_write_json(progress_path, progress)
        if "masked_l1_no_feedback" not in row:
            row["masked_l1_no_feedback"] = b0.parallel_loss(
                student, x, y, "masked_l1_no_feedback"
            )
            durable_write_json(progress_path, progress)
        teacher_raw = None
        memory = None
        if "trained_writer_real" not in row or "teacher_sources_with_writers" not in row:
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                teacher_raw = teacher.capture_residual_sources(x, SOURCE_DEPTHS)
            memory = writer_adapted_teacher_memory(student, teacher_raw)
        if "trained_writer_real" not in row:
            row["trained_writer_real"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_topdown_self",
                use_writers=True,
                teacher_raw=teacher_raw,
            )
            durable_write_json(progress_path, progress)
        if "trained_writer_shuffled" not in row:
            row["trained_writer_shuffled"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_shuffled_self_feedback",
                use_writers=True,
                permutation=permutation,
            )
            durable_write_json(progress_path, progress)
        if "trained_writer_bypass" not in row:
            row["trained_writer_bypass"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_topdown_self",
                use_writers=False,
            )
            durable_write_json(progress_path, progress)
        if "trained_writer_gate_zero" not in row:
            row["trained_writer_gate_zero"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_topdown_self",
                use_writers=True,
                gate_override=0.0,
            )
            durable_write_json(progress_path, progress)
        if "teacher_sources_with_writers" not in row:
            if memory is None:
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    teacher_raw = teacher.capture_residual_sources(x, SOURCE_DEPTHS)
                memory = writer_adapted_teacher_memory(student, teacher_raw)
            row["teacher_sources_with_writers"] = stream_loss(
                student,
                x,
                y,
                mode="masked_l1_topdown_teacher",
                use_writers=False,
                feedback_sources=memory,
            )
            durable_write_json(progress_path, progress)
        print(
            f"trained canonical {batch_index + 1:02d}/20 "
            f"writer={row['trained_writer_real']['loss']:.6f} "
            f"shuffled={row['trained_writer_shuffled']['loss']:.6f} "
            f"teacher={row['teacher_sources_with_writers']['loss']:.6f}",
            flush=True,
        )
        del x, y, teacher_raw, memory
        torch.cuda.empty_cache()
    if validation_digest.hexdigest() != a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256:
        raise SystemExit("trained canonical global validation hash mismatch")
    rows = progress["batches"]
    if len(rows) != a0.VALIDATION_BATCHES:
        raise SystemExit("trained canonical progress incomplete")
    losses = {
        "full_context": statistics.fmean(row["full_context"] for row in rows),
        "masked_l1_no_feedback": statistics.fmean(
            row["masked_l1_no_feedback"] for row in rows
        ),
        "frozen_2b1_self": baseline["losses"]["source_2b1_self_bypass"],
        "trained_writer_real": statistics.fmean(
            row["trained_writer_real"]["loss"] for row in rows
        ),
        "trained_writer_shuffled": statistics.fmean(
            row["trained_writer_shuffled"]["loss"] for row in rows
        ),
        "trained_writer_bypass": statistics.fmean(
            row["trained_writer_bypass"]["loss"] for row in rows
        ),
        "trained_writer_gate_zero": statistics.fmean(
            row["trained_writer_gate_zero"]["loss"] for row in rows
        ),
        "teacher_sources_with_writers": statistics.fmean(
            row["teacher_sources_with_writers"]["loss"] for row in rows
        ),
    }
    L_masked = losses["masked_l1_no_feedback"]
    L_2b1 = PINNED["source_2b1_self"]
    L_writer = losses["trained_writer_real"]
    L_shuffled = losses["trained_writer_shuffled"]
    writer_recovery = L_masked - L_writer
    writer_gap = L_shuffled - L_writer
    derived = {
        "writer_gain": L_2b1 - L_writer,
        "writer_recovery": writer_recovery,
        "writer_recovery_fraction": writer_recovery
        / (L_masked - losses["full_context"]),
        "source_2b1_specific_gap": PINNED["source_2b1_specific_gap"],
        "writer_specific_gap": writer_gap,
        "specific_gap_gain": writer_gap - PINNED["source_2b1_specific_gap"],
    }
    real_batch = [row["trained_writer_real"]["loss"] for row in rows]
    shuffled_batch = [row["trained_writer_shuffled"]["loss"] for row in rows]
    finite = all(
        row[mode]["finite"]
        for row in rows
        for mode in (
            "trained_writer_real",
            "trained_writer_shuffled",
            "trained_writer_bypass",
            "trained_writer_gate_zero",
            "teacher_sources_with_writers",
        )
    )
    hashes_after = {
        group: state_subset_sha256(student, group)
        for group in ("base", "reader", "writers")
    }
    frozen_regressions = {
        "full_context": abs(losses["full_context"] - PINNED["full_context"]),
        "masked_l1_no_feedback": abs(
            losses["masked_l1_no_feedback"] - PINNED["masked_l1_no_feedback"]
        ),
        "frozen_2b1_self": abs(losses["frozen_2b1_self"] - PINNED["source_2b1_self"]),
        "trained_writer_bypass": abs(
            losses["trained_writer_bypass"] - PINNED["source_2b1_self"]
        ),
        "trained_writer_gate_zero": abs(
            losses["trained_writer_gate_zero"] - PINNED["source_2b1_gate_zero"]
        ),
    }
    frozen_regressions_passed = all(
        value <= 5e-4 for value in frozen_regressions.values()
    )
    report = {
        "experiment": "2B2",
        "stage": "canonical_validation",
        "source": source_report(source),
        "trained_checkpoint": trained,
        "validation_batches": a0.VALIDATION_BATCHES,
        "B": a0.VALIDATION_B,
        "T": a0.T,
        "validation_global_batches_sha256": validation_digest.hexdigest(),
        "losses": losses,
        "derived": derived,
        "frozen_regression_absolute_differences": frozen_regressions,
        "frozen_regressions_passed": frozen_regressions_passed,
        "paired_writer_real_vs_shuffled": paired_statistics(real_batch, shuffled_batch),
        "batch_losses": {
            "full_context": [row["full_context"] for row in rows],
            "masked_l1_no_feedback": [row["masked_l1_no_feedback"] for row in rows],
            "frozen_2b1_self": baseline["batch_losses"]["source_2b1_self_bypass"],
            "trained_writer_real": real_batch,
            "trained_writer_shuffled": shuffled_batch,
            "trained_writer_bypass": [
                row["trained_writer_bypass"]["loss"] for row in rows
            ],
            "trained_writer_gate_zero": [
                row["trained_writer_gate_zero"]["loss"] for row in rows
            ],
            "teacher_sources_with_writers": [
                row["teacher_sources_with_writers"]["loss"] for row in rows
            ],
        },
        "trained_writer_routing": b0.average_routing(
            [row["trained_writer_real"] for row in rows]
        ),
        "teacher_sources_with_writers_routing": b0.average_routing(
            [row["teacher_sources_with_writers"] for row in rows]
        ),
        "per_source_writer_behavior": average_writer_behavior(
            [row["trained_writer_real"] for row in rows]
        ),
        "representation_drift": average_representation_drift(
            [row["trained_writer_real"] for row in rows]
        ),
        "writer_gain_positive": derived["writer_gain"] > 0,
        "writer_gain_at_least_0_01": derived["writer_gain"] >= 0.01,
        "finite": finite,
        "state_hashes_before": hashes_before,
        "state_hashes_after": hashes_after,
        "model_state_unchanged": hashes_before == hashes_after,
        "checkpoint_sha256_after": file_sha256(trained["path"]),
        "hellaswag_run": False,
    }
    report["passed"] = (
        finite
        and frozen_regressions_passed
        and report["model_state_unchanged"]
        and report["checkpoint_sha256_after"] == trained["sha256"]
    )
    durable_write_json(Path(args.run_dir) / "evaluation" / "canonical.json", report)
    if not report["passed"]:
        raise SystemExit("trained canonical evaluation failed")
    return report


def run_conditional_diagnostics(args, device):
    canonical_path = Path(args.run_dir) / "evaluation" / "canonical.json"
    require_passed(canonical_path, "canonical evaluation")
    canonical = load_json(canonical_path)
    run_reset = canonical["derived"]["writer_gain"] > 0
    run_writer_ablation = canonical["derived"]["writer_gain"] >= 0.01
    if not run_reset and not run_writer_ablation:
        report = {
            "stage": "conditional_diagnostics",
            "condition": "writer_gain > 0 for reset; writer_gain >= 0.01 for writer ablation",
            "condition_met": False,
            "reset_horizon_run": False,
            "writer_residual_ablation_run": False,
            "whole_source_ablation_run": False,
            "hellaswag_run": False,
            "passed": True,
        }
        durable_write_json(
            Path(args.run_dir) / "evaluation" / "conditional_diagnostics.json", report
        )
        return report
    symbols, _, student, _, trained = load_trained_for_evaluation(args, device)
    hashes_before = {
        group: state_subset_sha256(student, group)
        for group in ("base", "reader", "writers")
    }
    reference = reference_validation()
    loader = validation_loader(symbols)
    progress_path = Path(args.run_dir) / "evaluation" / "diagnostics_progress.json"
    progress = load_progress(progress_path)
    intervals = (1, 2, 4, 8, 16, 32, 64, 128, None)
    validation_digest = hashlib.sha256()
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        payload_hash = a0.batch_payload_hash(x_cpu, y_cpu)
        if payload_hash != reference["batch_payload_sha256"][batch_index]:
            raise SystemExit(f"conditional diagnostic payload mismatch at {batch_index}")
        validation_digest.update(bytes.fromhex(payload_hash))
        row = progress_row(progress, batch_index, payload_hash)
        required = set()
        if run_reset:
            required.add("reset_horizons")
        if run_writer_ablation:
            required.add("writer_residual_ablations")
        if required <= set(row):
            print(f"conditional diagnostics {batch_index + 1:02d}/20 reused", flush=True)
            continue
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        if run_reset and "reset_horizons" not in row:
            row["reset_horizons"] = stream_writer_reset_horizons(
                student, x, y, intervals
            )
            durable_write_json(progress_path, progress)
        if run_writer_ablation and "writer_residual_ablations" not in row:
            ablations = {}
            for depth in SOURCE_DEPTHS:
                ablations[f"v{depth}"] = stream_loss(
                    student,
                    x,
                    y,
                    mode="masked_l1_topdown_self",
                    use_writers=True,
                    disabled_writer_depths=(depth,),
                )
            row["writer_residual_ablations"] = ablations
            durable_write_json(progress_path, progress)
        message = f"conditional diagnostics {batch_index + 1:02d}/20"
        if run_reset:
            message += (
                f" reset1={row['reset_horizons']['losses']['1']:.6f}"
                f" never={row['reset_horizons']['losses']['never']:.6f}"
            )
        print(message, flush=True)
        del x, y
        torch.cuda.empty_cache()
    if validation_digest.hexdigest() != a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256:
        raise SystemExit("conditional diagnostic validation hash mismatch")
    rows = progress["batches"]
    baseline_batch = canonical["batch_losses"]["trained_writer_real"]
    reset_losses = {
        key: statistics.fmean(row["reset_horizons"]["losses"][key] for row in rows)
        for key in ("1", "2", "4", "8", "16", "32", "64", "128", "never")
    } if run_reset else None
    ablations = {}
    if run_writer_ablation:
        for depth in SOURCE_DEPTHS:
            name = f"v{depth}"
            values = [row["writer_residual_ablations"][name]["loss"] for row in rows]
            deltas = [value - base for value, base in zip(values, baseline_batch)]
            ablations[name] = {
                "ablated_loss": statistics.fmean(values),
                "delta": statistics.fmean(deltas),
                "positive_batches": sum(delta > 0 for delta in deltas),
                "negative_batches": sum(delta < 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "batch_losses": values,
                "batch_deltas": deltas,
            }
    finite = all(
        (not run_reset or row["reset_horizons"]["finite"])
        and (
            not run_writer_ablation
            or all(
                row["writer_residual_ablations"][f"v{depth}"]["finite"]
                for depth in SOURCE_DEPTHS
            )
        )
        for row in rows
    )
    hashes_after = {
        group: state_subset_sha256(student, group)
        for group in ("base", "reader", "writers")
    }
    report = {
        "stage": "conditional_diagnostics",
        "condition": "writer_gain > 0 for reset; writer_gain >= 0.01 for writer ablation",
        "condition_met": True,
        "trained_checkpoint": trained,
        "validation_global_batches_sha256": validation_digest.hexdigest(),
        "reset_horizon_run": run_reset,
        "reset_interval_losses": reset_losses,
        "reset_batch_losses": {
            key: [row["reset_horizons"]["losses"][key] for row in rows]
            for key in reset_losses
        } if run_reset else None,
        "frozen_2b1_reset_interval_losses": {
            "1": 5.9737106323,
            "2": 5.8261914968,
            "4": 5.7499492407,
            "8": 5.7218983650,
            "16": 5.7104103088,
            "32": 5.7059802055,
            "64": 5.7035196781,
            "128": 5.7021847963,
            "never": 5.7013566494,
        } if run_reset else None,
        "writer_residual_ablation_run": run_writer_ablation,
        "writer_residual_ablations": ablations,
        "whole_source_ablation_run": False,
        "finite": finite,
        "cache_health": all(
            not run_reset or row["reset_horizons"]["cache_lengths_correct"]
            for row in rows
        ),
        "state_hashes_before": hashes_before,
        "state_hashes_after": hashes_after,
        "model_state_unchanged": hashes_before == hashes_after,
        "hellaswag_run": False,
    }
    report["passed"] = (
        finite and report["cache_health"] and report["model_state_unchanged"]
    )
    durable_write_json(
        Path(args.run_dir) / "evaluation" / "conditional_diagnostics.json", report
    )
    if not report["passed"]:
        raise SystemExit("conditional diagnostics failed")
    return report


def classification(writer_gain, writer_specific_gap, invariants_passed=True):
    if not invariants_passed:
        return "MEMORY-WRITER LEARNING IS UNSTABLE"
    if writer_gain >= 0.01:
        if writer_specific_gap >= PINNED["source_2b1_specific_gap"] - 0.005:
            return "MEMORY-WRITER LEARNING IMPROVES RECURRENT MEMORY"
        return "MEMORY-WRITER LEARNING IMPROVES GENERIC COMPENSATION ONLY"
    if abs(writer_gain) < 0.01:
        return "MEMORY-WRITER LEARNING IS NEUTRAL"
    return "MEMORY-WRITER LEARNING DEGRADES"


def mean_drift(table, field):
    values = [
        source[field]
        for position in table.values()
        for source in position.values()
    ]
    return statistics.fmean(values)


def drift_comparison(trained_table):
    zero = load_json(
        REPO_ROOT / "results" / "experiment_2b0_zero_shot" / "canonical.json"
    )["teacher_student_drift"]
    zero_rms = mean_drift(zero, "rms_difference")
    trained_rms = mean_drift(trained_table, "rms_difference")
    zero_cosine = mean_drift(zero, "cosine_similarity")
    trained_cosine = mean_drift(trained_table, "cosine_similarity")
    relative_rms_change = (trained_rms - zero_rms) / zero_rms
    if abs(relative_rms_change) <= 0.01:
        summary = "stayed similar"
    elif relative_rms_change < 0:
        summary = "decreased"
    else:
        summary = "increased"
    return {
        "summary": summary,
        "rule": "mean RMS-difference change within 1% is similar; lower is decreased",
        "zero_shot_mean_rms_difference": zero_rms,
        "trained_mean_rms_difference": trained_rms,
        "relative_mean_rms_difference_change": relative_rms_change,
        "zero_shot_mean_cosine_similarity": zero_cosine,
        "trained_mean_cosine_similarity": trained_cosine,
        "mean_cosine_similarity_change": trained_cosine - zero_cosine,
        "zero_shot_raw": zero,
        "trained_raw": trained_table,
    }


def hellaswag_runtime_estimate(run_dir, canonical):
    if canonical["derived"]["writer_gain"] < 0.01:
        return {
            "prepared": False,
            "reason": "runtime estimate is conditional on at least 0.01 writer gain",
            "launched": False,
        }
    progress = load_json(
        Path(run_dir) / "evaluation" / "canonical_progress.json"
    )["batches"]
    seconds_per_step = statistics.fmean(
        row["trained_writer_real"]["elapsed_seconds"] / a0.T for row in progress
    )
    # HellaSwag candidates are isolated as a B=4 group. Without touching the
    # dataset, bracket average padded candidate length at 48-96 token steps.
    examples = 10_042
    low_steps = examples * 48
    high_steps = examples * 96
    one_control_hours = (
        low_steps * seconds_per_step / 3600,
        high_steps * seconds_per_step / 3600,
    )
    return {
        "prepared": True,
        "launched": False,
        "examples": examples,
        "candidate_isolation": "four alternatives isolated as batch rows",
        "example_reset": True,
        "cross_candidate_kv": False,
        "cross_candidate_recurrent_memory": False,
        "observed_B64_recurrent_seconds_per_token_step": seconds_per_step,
        "assumed_average_candidate_steps_range": [48, 96],
        "estimated_one_recurrent_control_gpu_hours_range": list(one_control_hours),
        "estimated_five_control_gpu_hours_range": [
            one_control_hours[0] * 5,
            one_control_hours[1] * 5,
        ],
        "historical_parallel_2a3_full_matrix_gpu_hours": 5597.372182846069
        / 3600,
        "recommended_timing": (
            "after 5M"
            if canonical["derived"]["writer_gain"] >= 0.02
            and canonical["derived"]["writer_specific_gap"]
            >= PINNED["source_2b1_specific_gap"] - 0.005
            else "after a later 25M writer experiment"
        ),
        "caveat": (
            "planning estimate only; HellaSwag B=4 kernel utilization and actual "
            "padded lengths were not benchmarked because no HellaSwag run was authorized"
        ),
    }


def run_final_audit(args, _device):
    config = validate_config()
    required_artifacts = {
        "preflight": Path(args.run_dir) / "preflight.json",
        "baseline": Path(args.run_dir) / "baseline.json",
        "smoke": Path(args.run_dir) / "smoke" / "summary.json",
        "training": Path(args.run_dir) / "result" / "training_summary.json",
        "canonical": Path(args.run_dir) / "evaluation" / "canonical.json",
        "diagnostics": Path(args.run_dir)
        / "evaluation"
        / "conditional_diagnostics.json",
    }
    artifacts = {name: load_json(path) for name, path in required_artifacts.items()}
    invariants_passed = all(row.get("passed") for row in artifacts.values())
    rows = read_jsonl(Path(args.run_dir) / "result" / "metrics.jsonl")
    validate_result_metrics(rows, 10)
    expected_hashes = oracle_map(args.run_dir)
    replay_exact = all(
        row["global_batch_sha256"] == expected_hashes[row["local_update"]]
        for row in rows
    )
    checkpoint5 = verify_checkpoint_sidecars(
        Path(args.run_dir) / "result" / "checkpoints" / "checkpoint_updates_000005.pt"
    )
    checkpoint10 = verify_checkpoint_sidecars(
        Path(args.run_dir) / "result" / "checkpoints" / "checkpoint_updates_000010.pt"
    )
    resume = load_json(Path(args.run_dir) / "result" / "resume_audit.json")
    canonical = artifacts["canonical"]
    derived = canonical["derived"]
    final_classification = classification(
        derived["writer_gain"],
        derived["writer_specific_gap"],
        invariants_passed=invariants_passed and replay_exact,
    )
    runtime_estimate = hellaswag_runtime_estimate(args.run_dir, canonical)
    continue_25m = final_classification == "MEMORY-WRITER LEARNING IMPROVES RECURRENT MEMORY"
    if continue_25m:
        continuation_reason = (
            "The writer cleared the 0.01 loss gate while preserving the sequence-specific gap."
        )
    elif final_classification == "MEMORY-WRITER LEARNING IMPROVES GENERIC COMPENSATION ONLY":
        continuation_reason = (
            "Do not continue: total loss improved but the sequence-specific signal collapsed."
        )
    else:
        continuation_reason = (
            "Do not continue: the 5M gate did not establish a stable sequence-memory gain."
        )
    checks = {
        "all_stage_artifacts_passed": invariants_passed,
        "source_2b0_tag_exact": git_output("rev-parse", SOURCE_2B0_TAG)
        == SOURCE_2B0_COMMIT,
        "source_2b1_tag_exact": git_output("rev-parse", SOURCE_2B1_TAG)
        == SOURCE_2B1_COMMIT,
        "source_checkpoint_sha256_exact": file_sha256(args.source_checkpoint)
        == SOURCE_CHECKPOINT_SHA256,
        "ten_updates_exact": len(rows) == 10,
        "five_million_target_geometry_exact": rows[-1]["processed_2b2_tokens"]
        == 5_242_880,
        "all_replay_hashes_exact": replay_exact,
        "all_updates_524288_targets": all(
            row["targets"] == a0.GLOBAL_BATCH_TOKENS for row in rows
        ),
        "teacher_training_forward_calls_zero": sum(
            row["teacher_forward_calls_during_training"] for row in rows
        )
        == 0,
        "all_frozen_gradients_none": all(
            not row["gradients"]["frozen_tensors_with_grad"] for row in rows
        ),
        "all_reader_gradients_none": all(
            not row["gradients"]["reader_tensors_with_grad"] for row in rows
        ),
        "writer_gradient_staging_exact": all(
            all(
                row["gradients"][f"writer_v{depth}_W_up"]["present"]
                and row["gradients"][f"writer_v{depth}_W_up"]["finite"]
                and row["gradients"][f"writer_v{depth}_W_up"]["nonzero"]
                and row["gradients"][f"writer_v{depth}_W_down"]["present"]
                and row["gradients"][f"writer_v{depth}_W_down"]["finite"]
                and (
                    not row["gradients"][f"writer_v{depth}_W_down"]["nonzero"]
                    if row["local_update"] == 1
                    else row["gradients"][f"writer_v{depth}_W_down"]["nonzero"]
                )
                for depth in SOURCE_DEPTHS
            )
            for row in rows
        ),
        "all_stored_state_valid": all(
            all(item["passed"] for item in row["state_health"]) for row in rows
        ),
        "fresh_process_restart": resume["fresh_process"]
        and resume["phase_processes_distinct"],
        "update_6_hash_matches_update_5_checkpoint": rows[5].get(
            "fresh_process_update_6_batch_verified", False
        ),
        "checkpoint_5_verified": checkpoint5["verification"]["passed"],
        "checkpoint_10_verified": checkpoint10["verification"]["passed"],
        "frozen_base_unchanged": artifacts["training"]["frozen_base_unchanged"],
        "frozen_reader_unchanged": artifacts["training"]["frozen_reader_unchanged"],
        "one_step_temporal_gradient_exact": artifacts["preflight"][
            "temporal_gradient_boundary"
        ]["passed"],
        "cpu_fp32_mathematical_regression": artifacts["preflight"][
            "cpu_fp32_mathematical_regression"
        ]["passed"],
        "canonical_validation_hash_exact": canonical[
            "validation_global_batches_sha256"
        ]
        == a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256,
        "hellaswag_not_run": not any(Path(args.run_dir).rglob("hellaswag*.json")),
        "authorized_optimizer_updates_exhausted": artifacts["training"][
            "authorized_optimizer_updates_exhausted"
        ],
    }
    audit_passed = all(checks.values())
    if not audit_passed:
        final_classification = "MEMORY-WRITER LEARNING IS UNSTABLE"
        continue_25m = False
        continuation_reason = "Do not continue: the final integrity audit failed."
    report = {
        "experiment": "2B2",
        "protocol": config["protocol"],
        "git": {
            "source_2b0_tag": SOURCE_2B0_TAG,
            "source_2b0_commit": SOURCE_2B0_COMMIT,
            "source_2b1_tag": SOURCE_2B1_TAG,
            "source_2b1_commit": SOURCE_2B1_COMMIT,
            "branch": BRANCH,
            "implementation_commit": artifacts["training"]["final_checkpoint"][
                "verification"
            ]["implementation_git_commit"],
            "audit_worktree_head": git_output("rev-parse", "HEAD"),
        },
        "checks": checks,
        "initialization": load_json(
            Path(args.run_dir) / "result" / "initialization.json"
        ),
        "training": artifacts["training"],
        "resume_audit": resume,
        "checkpoint_5": checkpoint5,
        "checkpoint_10": checkpoint10,
        "validation": canonical,
        "conditional_diagnostics": artifacts["diagnostics"],
        "representation_drift": canonical["representation_drift"],
        "hellaswag": {
            "run": False,
            "runtime_estimate": runtime_estimate,
        },
        "classification": final_classification,
        "continue_same_checkpoint_to_approximately_25m": (
            "YES" if continue_25m else "NO"
        ),
        "continuation_reason": continuation_reason,
        "decision_a_continue_writers_to_approximately_25m": (
            "YES" if continue_25m else "NO"
        ),
        "decision_b_jointly_train_writers_and_reader": (
            "YES" if final_classification
            == "MEMORY-WRITER LEARNING IMPROVES RECURRENT MEMORY" else "NO"
        ),
        "decision_c_increase_temporal_credit_horizon": "NO",
        "optimizer_updates_beyond_10": 0,
        "passed": audit_passed,
        "terminal_line": "# EXPERIMENT 2B2 5M COMPLETE",
    }
    durable_write_json(Path(args.run_dir) / "FINAL_AUDIT.json", report)
    if not report["passed"]:
        raise SystemExit("Experiment 2B2 final audit failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "baseline",
            "smoke-phase1",
            "smoke-phase2",
            "result-phase1",
            "result-phase2",
            "canonical",
            "diagnostics",
            "audit",
        ),
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-optimizer-steps", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    optimizer_stage = args.stage in {
        "smoke-phase1",
        "smoke-phase2",
        "result-phase1",
        "result-phase2",
    }
    if optimizer_stage and not args.allow_optimizer_steps:
        raise SystemExit("optimizer stages require explicit --allow-optimizer-steps")
    device = require_environment(require_clean=args.stage == "result-phase1")
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(a0.SEED)
    np.random.seed(a0.SEED)
    torch.manual_seed(a0.SEED)
    torch.cuda.manual_seed(a0.SEED)
    functions = {
        "preflight": run_preflight,
        "baseline": run_baseline,
        "smoke-phase1": run_smoke_phase1,
        "smoke-phase2": run_smoke_phase2,
        "result-phase1": run_result_phase1,
        "result-phase2": run_result_phase2,
        "canonical": run_canonical_evaluation,
        "diagnostics": run_conditional_diagnostics,
        "audit": run_final_audit,
    }
    report = functions[args.stage](args, device)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
