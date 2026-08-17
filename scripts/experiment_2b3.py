#!/usr/bin/env python3
"""Experiment 2B3: joint writer/reader learning with one-step credit."""

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
import numpy as np
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2a0 as a0  # noqa: E402
import experiment_2b2 as b2  # noqa: E402
import experiment_2b2a as b2a  # noqa: E402


BRANCH = "experiment-2b3-joint-writer-reader-1step"
PARENT_TAG = "experiment-2b2a-writers-15m-canonical"
PARENT_COMMIT = "c47e4c7619d4f7507f1c05bba8557d6c4712ab73"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2b3_joint_4gpu.json"
SOURCE_SHA256 = "86c66343141e24d0beffcf8bc98a558f25c82e1dc05582feade2300d30b2ba84"
SOURCE_NEXT_SHA256 = "8b9fe2fa1c2a10ce930caff4d527c48e4f14ab0e1a6f5e4b352e42f61b8b360d"
FINAL_NEXT_SHA256 = "7f6d8da5044e9f485492712373fda12d09eb4ccec60ab8fdee812519d05869a7"
SOURCE_SCHEMA = b2a.CHECKPOINT_SCHEMA
CHECKPOINT_SCHEMA = "exp2b3_joint_writer_reader_1step_v1"
SOURCE_IMPLEMENTATION_COMMIT = "ede5eb245f44d19c53d7fcbc5187f020d40f5ffc"
SOURCE_BASE_SHA256 = "1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd"
SOURCE_READER_SHA256 = "aca8f87518e3728b5d721a48e7729b8e93569a23174f29299d3795227fcd61a7"
SOURCE_WRITER_SHA256 = "d2e23a5f07f45f15847d70d112be5c6ae63b06000c82f5eb624ac4d60501b45c"
SOURCE_OPTIMIZER_HASHES = {
    "writer": "6d0302981ec1714ce05c034a6099872b4869fb0d40805186ad791b6fa50fcaf7",
    "adam_steps": [29] * 8,
    "adam_m1": "736e0e1d20a21347de98f279dab3b9bdd220f03f84655a7cf3d85ce8bb0db156",
    "adam_m2": "2ab9397fe0f6a3ab8c78f7ba55bc5922e3b4621fcda2f747334daeabdb9b2c18",
}
WORLD_SIZE = 4
MICROSTEPS_PER_RANK = 2
B = 64
T = 1024
GLOBAL_TARGETS = 524_288
RANK_TARGETS = 131_072
SOURCE_WRITER_UPDATE = 29
SOURCE_WRITER_TOKENS = 15_204_352
FINAL_JOINT_UPDATE = 9
RESTART_JOINT_UPDATE = 5
WRITER_PARAMETER_COUNT = 49_152
READER_PARAMETER_COUNT = 1_537
TRAINABLE_PARAMETER_COUNT = 50_689
READER_NAMES = {
    "transformer.topdown_attnres.query",
    "transformer.topdown_attnres.norm.weight",
    "transformer.topdown_attnres.gate",
}
TRAINABLE_NAMES = set(b2.TRAINABLE_NAMES) | READER_NAMES


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2B3 requires branch {BRANCH}")
    if git_output("rev-parse", f"{PARENT_TAG}^{{}}") != PARENT_COMMIT:
        raise SystemExit("frozen Experiment 2B2A tag target mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing execution requires a clean worktree")


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    checks = {
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_next_global_batch_sha256": SOURCE_NEXT_SHA256,
        "final_next_global_batch_sha256": FINAL_NEXT_SHA256,
        "source_writer_updates": SOURCE_WRITER_UPDATE,
        "source_writer_tokens": SOURCE_WRITER_TOKENS,
        "joint_updates": FINAL_JOINT_UPDATE,
        "forced_restart_after_joint_update": RESTART_JOINT_UPDATE,
        "world_size": WORLD_SIZE,
        "microsteps_per_rank": MICROSTEPS_PER_RANK,
        "batch_sequences": B,
        "sequence_length": T,
        "rank_targets_per_update": RANK_TARGETS,
        "global_targets_per_update": GLOBAL_TARGETS,
        "backward_chunk_tokens": b2a.BACKWARD_CHUNK,
        "writer_parameters": WRITER_PARAMETER_COUNT,
        "reader_parameters": READER_PARAMETER_COUNT,
        "trainable_parameters": TRAINABLE_PARAMETER_COUNT,
    }
    for name, expected in checks.items():
        if config.get(name) != expected:
            raise SystemExit(f"config {name} mismatch: {config.get(name)} != {expected}")
    if config.get("temporal_credit") != "exactly one token":
        raise SystemExit("temporal credit must remain exactly one token")
    if config.get("hellaswag") != "forbidden":
        raise SystemExit("HellaSwag must remain forbidden")
    return config


def configure_joint_training(model):
    model.freeze_for_joint_writer_reader_training()
    actual = {name for name, value in model.named_parameters() if value.requires_grad}
    if actual != TRAINABLE_NAMES:
        raise SystemExit(f"joint trainable tensor mismatch: {sorted(actual)}")
    count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if count != TRAINABLE_PARAMETER_COUNT:
        raise SystemExit(f"joint trainable parameter mismatch: {count}")


def parameter_rows(model, group):
    if group not in {"writer", "reader"}:
        raise ValueError(group)
    names = b2.TRAINABLE_NAMES if group == "writer" else READER_NAMES
    named = dict(model.named_parameters())
    rows = [(name, named[name]) for name in sorted(names)]
    expected = WRITER_PARAMETER_COUNT if group == "writer" else READER_PARAMETER_COUNT
    if sum(value.numel() for _, value in rows) != expected:
        raise SystemExit(f"{group} parameter count mismatch")
    return rows


def make_optimizer(model, group, device_type):
    parameters = [value for _, value in parameter_rows(model, group)]
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
        raise SystemExit(f"fresh {group} optimizer unexpectedly contains state")
    return optimizer


def reader_optimizer_report(optimizer, completed_updates):
    state = optimizer if isinstance(optimizer, dict) else optimizer.state_dict()
    if len(state["param_groups"]) != 1:
        raise SystemExit("reader optimizer must have one parameter group")
    group = state["param_groups"][0]
    expected = {
        "lr": 1e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    for name, value in expected.items():
        if group.get(name) != value:
            raise SystemExit(f"reader optimizer {name} mismatch")
    if completed_updates == 0 and state["state"]:
        raise SystemExit("reader optimizer step 0 must have fresh state")
    if completed_updates > 0 and len(state["state"]) != 3:
        raise SystemExit("reader optimizer must have exactly three state entries")
    steps = []
    for parameter_id, values in state["state"].items():
        if "step" in values:
            steps.append(int(values["step"].item()))
        for name, value in values.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if not torch.isfinite(value).all():
                    raise SystemExit(f"non-finite reader optimizer state {parameter_id}:{name}")
    if completed_updates > 0 and sorted(steps) != [completed_updates] * 3:
        raise SystemExit(f"reader Adam step mismatch: {steps}")
    return {
        "state_entries": len(state["state"]),
        "steps": sorted(steps),
        "moments_finite": True,
        "lr": group["lr"],
        "betas": list(group["betas"]),
        "eps": group["eps"],
        "weight_decay": group["weight_decay"],
    }


def flat_parameters(model, group):
    return torch.cat(
        [value.detach().float().reshape(-1) for _, value in parameter_rows(model, group)]
    )


def flat_optimizer_moment(model, optimizer, group, field):
    values = []
    for _, parameter in parameter_rows(model, group):
        state = optimizer.state.get(parameter)
        if state is None or field not in state:
            raise SystemExit(f"missing {group} optimizer field {field}")
        values.append(state[field].detach().float().reshape(-1))
    return torch.cat(values)


def group_state_hashes(model, optimizer, group, completed_updates):
    rows = parameter_rows(model, group)
    if group == "writer":
        b2.optimizer_report(optimizer, completed_updates)
    else:
        reader_optimizer_report(optimizer, completed_updates)
    steps = []
    m1 = []
    m2 = []
    for name, parameter in rows:
        state = optimizer.state.get(parameter, {})
        if "step" in state:
            steps.append(int(state["step"].item()))
            m1.append((name, state["exp_avg"]))
            m2.append((name, state["exp_avg_sq"]))
    return {
        "parameters": b2a.tensor_digest(rows),
        "adam_steps": steps,
        "adam_m1": b2a.tensor_digest(m1),
        "adam_m2": b2a.tensor_digest(m2),
    }


def joint_state_hashes(model, writer_optimizer, reader_optimizer, writer_step, reader_step):
    return {
        "writer": group_state_hashes(model, writer_optimizer, "writer", writer_step),
        "reader": group_state_hashes(model, reader_optimizer, "reader", reader_step),
    }


def flatten_group_gradients(model, group):
    rows = parameter_rows(model, group)
    missing = [name for name, value in rows if value.grad is None]
    if missing:
        raise SystemExit(f"missing {group} gradients: {missing}")
    flat = torch.cat([value.grad.detach().float().reshape(-1) for _, value in rows])
    if not torch.isfinite(flat).all():
        raise SystemExit(f"non-finite {group} gradients")
    return flat.contiguous()


def flatten_joint_gradients(model):
    writer = flatten_group_gradients(model, "writer")
    reader = flatten_group_gradients(model, "reader")
    combined = torch.cat((writer, reader)).contiguous()
    if combined.numel() != TRAINABLE_PARAMETER_COUNT:
        raise SystemExit("combined gradient length mismatch")
    return combined, writer.numel()


def scatter_group_gradients(model, group, flat):
    offset = 0
    for _, parameter in parameter_rows(model, group):
        count = parameter.numel()
        parameter.grad = flat[offset : offset + count].view_as(parameter).to(parameter.dtype)
        offset += count
    if offset != flat.numel():
        raise SystemExit(f"{group} gradient scatter mismatch")


def scatter_joint_gradients(model, combined, writer_elements):
    scatter_group_gradients(model, "writer", combined[:writer_elements])
    scatter_group_gradients(model, "reader", combined[writer_elements:])


def canonical_rank_slotted_all_reduce(local_gradient, rank):
    """Synchronize once, then sum exact per-rank slots in a fixed FP32 order."""
    if local_gradient.dtype != torch.float32 or local_gradient.ndim != 1:
        raise SystemExit("canonical gradient reduction requires a flat FP32 tensor")
    slots = torch.zeros(
        (WORLD_SIZE, local_gradient.numel()),
        dtype=torch.float32,
        device=local_gradient.device,
    )
    slots[rank].copy_(local_gradient)
    started = time.perf_counter()
    dist.all_reduce(slots, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = slots[0].clone()
    for source_rank in range(1, WORLD_SIZE):
        result.add_(slots[source_rank])
    return result.contiguous(), elapsed


def canonical_local_rank_sum(local_gradients):
    if len(local_gradients) != WORLD_SIZE:
        raise SystemExit("canonical local sum requires four rank gradients")
    result = local_gradients[0].clone()
    for source_rank in range(1, WORLD_SIZE):
        result.add_(local_gradients[source_rank])
    return result.contiguous()


def gradient_report(model):
    named = dict(model.named_parameters())
    result = {"writer": {}, "reader": {}, "frozen_tensors_with_grad": []}
    for group, names in (("writer", b2.TRAINABLE_NAMES), ("reader", READER_NAMES)):
        for name in sorted(names):
            gradient = named[name].grad
            result[group][name] = {
                "present": gradient is not None,
                "finite": gradient is not None and bool(torch.isfinite(gradient).all().item()),
                "nonzero": gradient is not None and bool(torch.count_nonzero(gradient).item()),
                "norm": None if gradient is None else gradient.detach().float().norm().item(),
            }
    result["frozen_tensors_with_grad"] = [
        name for name, value in model.named_parameters()
        if name not in TRAINABLE_NAMES and value.grad is not None
    ]
    return result


def validate_gradient_report(report):
    if report["frozen_tensors_with_grad"]:
        raise SystemExit(f"gradient leaked into frozen base: {report['frozen_tensors_with_grad']}")
    for group in ("writer", "reader"):
        bad = [
            name for name, row in report[group].items()
            if not (row["present"] and row["finite"] and row["nonzero"])
        ]
        if bad:
            raise SystemExit(f"invalid {group} gradients: {bad}")


def load_source_checkpoint(path):
    path = Path(path).resolve()
    digest = b2a.file_sha256(path)
    if digest != SOURCE_SHA256:
        raise SystemExit(f"canonical 15M checkpoint SHA mismatch: {digest}")
    checkpoint = a0.torch_load(path, mmap=True)
    if checkpoint.get("schema") != SOURCE_SCHEMA:
        raise SystemExit("canonical 15M checkpoint schema mismatch")
    expected_state = {
        "writer_updates": SOURCE_WRITER_UPDATE,
        "writer_training_tokens": SOURCE_WRITER_TOKENS,
        "fineweb_lineage_completed_update": 526,
        "kind": "2b2a_15m",
    }
    if checkpoint.get("training_state") != expected_state:
        raise SystemExit(f"canonical 15M training state mismatch: {checkpoint.get('training_state')}")
    exact = {
        "next_global_batch_sha256": SOURCE_NEXT_SHA256,
        "world_size": WORLD_SIZE,
        "rank_to_loader_mapping": {str(value): value for value in range(WORLD_SIZE)},
        "gradient_synchronization": "one flattened FP32 NCCL all_reduce(SUM) per update",
        "temporal_credit": "exactly one token; historical Blocks 2-12 KV detached",
        "implementation_git_commit": SOURCE_IMPLEMENTATION_COMMIT,
        "frozen_base_sha256": SOURCE_BASE_SHA256,
        "frozen_reader_sha256": SOURCE_READER_SHA256,
        "writer_sha256": SOURCE_WRITER_SHA256,
        "optimizer_consistency": SOURCE_OPTIMIZER_HASHES,
    }
    for name, expected in exact.items():
        if checkpoint.get(name) != expected:
            raise SystemExit(f"canonical checkpoint {name} mismatch")
    if len(checkpoint.get("dataloader_states", ())) != WORLD_SIZE:
        raise SystemExit("canonical checkpoint must contain four loader states")
    if len(checkpoint.get("rank_rng_states", ())) != WORLD_SIZE:
        raise SystemExit("canonical checkpoint must contain four rank RNG states")
    if checkpoint.get("rank_seeds") != [2_026_082_000 + value for value in range(4)]:
        raise SystemExit("canonical rank seeds mismatch")
    if [row.get("rank") for row in checkpoint["rank_rng_states"]] != list(range(4)):
        raise SystemExit("canonical rank RNG mapping mismatch")
    b2.optimizer_report(checkpoint["optimizer"], SOURCE_WRITER_UPDATE)
    return checkpoint, digest


def instantiate_source(checkpoint, device):
    symbols = a0.support.load_training_symbols()
    with torch.random.fork_rng(devices=[]):
        model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
    model.load_state_dict(checkpoint["model"], strict=True)
    configure_joint_training(model)
    b2.assert_reader_initialization(model)
    hashes = {
        "base": b2.state_subset_sha256(model, "base"),
        "reader": b2.state_subset_sha256(model, "reader"),
        "writers": b2.state_subset_sha256(model, "writers"),
    }
    expected = {
        "base": SOURCE_BASE_SHA256,
        "reader": SOURCE_READER_SHA256,
        "writers": SOURCE_WRITER_SHA256,
    }
    if hashes != expected:
        raise SystemExit(f"canonical model subset hash mismatch: {hashes}")
    model.to(device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    writer_optimizer = make_optimizer(model, "writer", device_type)
    writer_optimizer.load_state_dict(checkpoint["optimizer"])
    b2.optimizer_report(writer_optimizer, SOURCE_WRITER_UPDATE)
    writer_hashes = group_state_hashes(
        model, writer_optimizer, "writer", SOURCE_WRITER_UPDATE
    )
    expected_writer_hashes = {
        "parameters": SOURCE_OPTIMIZER_HASHES["writer"],
        "adam_steps": SOURCE_OPTIMIZER_HASHES["adam_steps"],
        "adam_m1": SOURCE_OPTIMIZER_HASHES["adam_m1"],
        "adam_m2": SOURCE_OPTIMIZER_HASHES["adam_m2"],
    }
    if writer_hashes != expected_writer_hashes:
        raise SystemExit(f"restored writer optimizer hash mismatch: {writer_hashes}")
    reader_optimizer = make_optimizer(model, "reader", device_type)
    reader_optimizer_report(reader_optimizer, 0)
    return symbols, model, writer_optimizer, reader_optimizer


def save_starting_components(args, checkpoint):
    output = Path(args.run_dir) / "starting_15m_components.pt"
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_checkpoint_sha256": SOURCE_SHA256,
            "writer": {
                name: value.detach().cpu().clone()
                for name, value in checkpoint["model"].items()
                if name in b2.TRAINABLE_NAMES
            },
            "reader": {
                name: value.detach().cpu().clone()
                for name, value in checkpoint["model"].items()
                if name in READER_NAMES
            },
        }
        temporary = output.with_name(output.name + ".incomplete")
        torch.save(payload, temporary)
        os.replace(temporary, output)
    if payload.get("source_checkpoint_sha256") != SOURCE_SHA256:
        raise SystemExit("starting component artifact source mismatch")
    if set(payload["writer"]) != b2.TRAINABLE_NAMES or set(payload["reader"]) != READER_NAMES:
        raise SystemExit("starting component artifact tensor mismatch")
    return {"path": str(output.resolve()), "sha256": b2a.file_sha256(output)}


def source_audit(args):
    require_git(clean=True)
    load_config()
    checkpoint, digest = load_source_checkpoint(args.source_checkpoint)
    symbols = a0.support.load_training_symbols()
    loaders = a0.make_replay_loaders(symbols, copy.deepcopy(checkpoint["dataloader_states"]))
    next_hash = a0.next_update_hash(loaders, symbols, replay=True)
    if next_hash != SOURCE_NEXT_SHA256:
        raise SystemExit(f"canonical source replay mismatch: {next_hash}")
    _, model, writer_optimizer, reader_optimizer = instantiate_source(
        checkpoint, torch.device("cpu")
    )
    components = save_starting_components(args, checkpoint)
    report = {
        "experiment": "2B3",
        "stage": "canonical_15m_source_audit",
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": digest,
        "source_schema": checkpoint["schema"],
        "training_state": checkpoint["training_state"],
        "next_global_batch_sha256": next_hash,
        "model_subset_sha256": {
            group: b2.state_subset_sha256(model, group)
            for group in ("base", "reader", "writers")
        },
        "writer_optimizer": b2.optimizer_report(writer_optimizer, 29),
        "reader_optimizer": reader_optimizer_report(reader_optimizer, 0),
        "reader_weights": b2.reader_values(model),
        "loader_states": len(checkpoint["dataloader_states"]),
        "rank_rng_states": len(checkpoint["rank_rng_states"]),
        "rank_to_loader_mapping": checkpoint["rank_to_loader_mapping"],
        "starting_components": components,
        "trainable_parameters": TRAINABLE_PARAMETER_COUNT,
        "source_git_lineage": {
            "frozen_tag": PARENT_TAG,
            "frozen_commit": PARENT_COMMIT,
            "checkpoint_implementation_commit": SOURCE_IMPLEMENTATION_COMMIT,
        },
        "passed": True,
    }
    b2a.write_json(Path(args.run_dir) / "SOURCE_CHECKPOINT_AUDIT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _memory_gradient(loss, memory):
    gradient = torch.autograd.grad(
        loss, memory, allow_unused=True, retain_graph=True
    )[0]
    return {
        "present": gradient is not None,
        "finite": gradient is not None and bool(torch.isfinite(gradient).all().item()),
        "nonzero": gradient is not None and bool(torch.count_nonzero(gradient).item()),
    }


def joint_temporal_gradient_test(model, tokens, targets):
    model.eval()
    model.zero_grad(set_to_none=True)
    memories = []
    losses = []
    cached_position_one = {}

    def capture_cache_source(_module, _inputs, output):
        if len(memories) == 1:
            output.retain_grad()
            cached_position_one["block2_qkv"] = output

    handle = model.transformer.h[1].attn.c_attn.register_forward_hook(capture_cache_source)
    state_checks = []
    try:
        state = model.init_recurrent_state(
            tokens.size(0), "masked_l1_topdown_self",
            device=tokens.device, dtype=torch.float32,
        )
        for position in range(4):
            logits, state = model.forward_step(
                tokens[:, position], state, use_memory_writers=True
            )
            state.feedback_memory.retain_grad()
            memories.append(state.feedback_memory)
            state_checks.append(b2.state_health(state, position + 1))
            losses.append(F.cross_entropy(logits[:, 0], targets[:, position]))
    finally:
        handle.remove()
    if "block2_qkv" not in cached_position_one:
        raise SystemExit("joint temporal test did not capture Block-2 QKV")
    loss1_to_memory0 = _memory_gradient(losses[1], memories[0])
    loss2_to_memory1 = _memory_gradient(losses[2], memories[1])
    loss2_to_memory0 = _memory_gradient(losses[2], memories[0])
    historical_gradient = torch.autograd.grad(
        losses[2], cached_position_one["block2_qkv"],
        allow_unused=True, retain_graph=True,
    )[0]
    losses[1].backward()
    gradients = gradient_report(model)
    reader_short = {
        "query": gradients["reader"]["transformer.topdown_attnres.query"],
        "rmsnorm": gradients["reader"]["transformer.topdown_attnres.norm.weight"],
        "gate": gradients["reader"]["transformer.topdown_attnres.gate"],
    }
    report = {
        "precision": "FP32",
        "loss_t_plus_1_to_writer_memory_t": loss1_to_memory0,
        "loss_t_plus_2_to_writer_memory_t": loss2_to_memory0,
        "loss_t_plus_2_to_writer_memory_t_plus_1": loss2_to_memory1,
        "historical_kv_temporal_gradient_none": historical_gradient is None,
        "writer_gradients": gradients["writer"],
        "reader_gradients": reader_short,
        "frozen_base_gradients_none": not gradients["frozen_tensors_with_grad"],
        "stored_state_after_each_token": state_checks,
    }
    report["passed"] = (
        all(loss1_to_memory0.values())
        and not loss2_to_memory0["present"]
        and all(loss2_to_memory1.values())
        and report["historical_kv_temporal_gradient_none"]
        and all(row["present"] and row["finite"] and row["nonzero"]
                for row in gradients["writer"].values())
        and all(row["present"] and row["finite"] and row["nonzero"]
                for row in reader_short.values())
        and report["frozen_base_gradients_none"]
        and all(row["passed"] for row in state_checks)
    )
    model.zero_grad(set_to_none=True)
    if not report["passed"]:
        raise SystemExit(f"joint temporal gradient boundary failed: {report}")
    return report


@torch.no_grad()
def _writer_prefix_capture(model, tokens, prefix_length):
    state = model.init_recurrent_state(
        tokens.size(0), "masked_l1_topdown_self",
        device=tokens.device, dtype=torch.bfloat16,
    )
    logits = []
    prefix_state = None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            row, state = model.forward_step(
                tokens[:, position], state, use_memory_writers=True
            )
            if position < prefix_length:
                logits.append(row.detach().clone())
            if position + 1 == prefix_length:
                prefix_state = state.state_dict()
    return torch.cat(logits, dim=1), prefix_state


@torch.no_grad()
def joint_causality_tests(model, tokens):
    first = tokens[:2, :32].clone()
    second = first.clone()
    second[:, 16:] = (second[:, 16:] + 1) % model.config.vocab_size
    logits_a, state_a = _writer_prefix_capture(model, first, 16)
    logits_b, state_b = _writer_prefix_capture(model, second, 16)
    future = {
        "prefix_logits_bit_exact": torch.equal(logits_a, logits_b),
        "prefix_memory_and_kv_bit_exact": b2.b0.cache_payload_equal(state_a, state_b),
        "maximum_absolute_logit_difference": (
            logits_a.float() - logits_b.float()
        ).abs().max().item(),
    }
    future["passed"] = future["prefix_logits_bit_exact"] and future["prefix_memory_and_kv_bit_exact"]
    row_a = tokens[:2, :16].clone()
    row_b = row_a.clone()
    row_b[1] = (row_b[1] + 17) % model.config.vocab_size
    row_logits_a, row_state_a = _writer_prefix_capture(model, row_a, 16)
    row_logits_b, row_state_b = _writer_prefix_capture(model, row_b, 16)
    isolation = {
        "unchanged_row_logits_bit_exact": torch.equal(row_logits_a[0], row_logits_b[0]),
        "unchanged_row_memory_and_kv_bit_exact": b2.b0.cache_payload_equal(
            row_state_a, row_state_b, row=0
        ),
    }
    isolation["passed"] = all(isolation.values())
    initial = model.init_recurrent_state(
        2, "masked_l1_topdown_self", device=tokens.device, dtype=torch.bfloat16
    )
    reset = {
        "initial_position_zero": initial.position == 0,
        "initial_memory_exactly_zero": initial.feedback_memory.count_nonzero().item() == 0,
        "block_1_cache_absent": initial.kv_caches[0] is None,
        "other_caches_empty": all(cache.length == 0 for cache in initial.kv_caches[1:]),
    }
    reset["passed"] = all(reset.values())
    return {
        "future_causality": future,
        "row_isolation": isolation,
        "fresh_state": reset,
        "passed": future["passed"] and isolation["passed"] and reset["passed"],
    }


def preflight(args):
    require_git(clean=True)
    load_config()
    if not torch.cuda.is_available():
        raise SystemExit("joint preflight requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    checkpoint, digest = load_source_checkpoint(args.source_checkpoint)
    symbols, model, writer_optimizer, reader_optimizer = instantiate_source(checkpoint, device)
    loaders = a0.make_replay_loaders(symbols, copy.deepcopy(checkpoint["dataloader_states"]))
    next_hash = a0.next_update_hash(loaders, symbols, replay=True)
    if next_hash != SOURCE_NEXT_SHA256:
        raise SystemExit("joint preflight next-batch mismatch")
    validation = b2.validation_loader(symbols)
    x_cpu, y_cpu = validation.next_batch()
    x = x_cpu[:2, :32].to(device)
    y = y_cpu[:2, :32].to(device)
    hashes_before = {
        group: b2.state_subset_sha256(model, group)
        for group in ("base", "reader", "writers")
    }
    temporal = joint_temporal_gradient_test(model, x[:, :4], y[:, :4])
    causality = joint_causality_tests(model, x)
    hashes_after = {
        group: b2.state_subset_sha256(model, group)
        for group in ("base", "reader", "writers")
    }
    report = {
        "experiment": "2B3",
        "stage": "joint_gradient_and_integrity_preflight",
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "source_checkpoint_sha256": digest,
        "source_next_global_batch_sha256": next_hash,
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "writer_optimizer": b2.optimizer_report(writer_optimizer, 29),
        "reader_optimizer": reader_optimizer_report(reader_optimizer, 0),
        "temporal_gradient_boundary": temporal,
        "causality": causality,
        "model_state_unchanged": hashes_before == hashes_after,
        "teacher_training_forward_calls": 0,
        "hellaswag_run": False,
    }
    report["passed"] = (
        report["trainable_parameters"] == TRAINABLE_PARAMETER_COUNT
        and temporal["passed"] and causality["passed"]
        and report["model_state_unchanged"]
    )
    b2a.write_json(Path(args.run_dir) / "JOINT_PREFLIGHT.json", report)
    if not report["passed"]:
        raise SystemExit("joint preflight failed")
    print("JOINT_PREFLIGHT_PASS", flush=True)


def temporary_optimizer_vectors(model, writer_optimizer, reader_optimizer):
    before = {group: flat_parameters(model, group).detach().clone() for group in ("writer", "reader")}
    pre_writer = torch.nn.utils.clip_grad_norm_(
        [value for _, value in parameter_rows(model, "writer")], 1.0
    )
    pre_reader = torch.nn.utils.clip_grad_norm_(
        [value for _, value in parameter_rows(model, "reader")], 1.0
    )
    if not torch.isfinite(pre_writer) or not torch.isfinite(pre_reader):
        raise SystemExit("non-finite temporary optimizer gradient norm")
    writer_optimizer.step()
    reader_optimizer.step()
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    result = {}
    for group, optimizer in (("writer", writer_optimizer), ("reader", reader_optimizer)):
        after = flat_parameters(model, group).detach().clone()
        result[group] = {
            "pre_clip_norm": float(pre_writer if group == "writer" else pre_reader),
            "update": (after - before[group]).cpu(),
            "parameters": after.cpu(),
            "m1": flat_optimizer_moment(model, optimizer, group, "exp_avg").cpu(),
            "m2": flat_optimizer_moment(model, optimizer, group, "exp_avg_sq").cpu(),
        }
    return result


def migration_reference(args):
    require_git(clean=True)
    load_config()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    checkpoint, digest = load_source_checkpoint(args.source_checkpoint)
    symbols, model, writer_optimizer, reader_optimizer = instantiate_source(checkpoint, device)
    loaders = a0.make_replay_loaders(symbols, copy.deepcopy(checkpoint["dataloader_states"]))
    expected = a0.next_update_hash(loaders, symbols, replay=True)
    if expected != SOURCE_NEXT_SHA256:
        raise SystemExit("1-GPU joint reference next-batch mismatch")
    rank_metrics = []
    local_gradients = []
    for simulated_rank in range(WORLD_SIZE):
        b2a.restore_rank_rng(checkpoint["rank_rng_states"][simulated_rank], 0)
        batches = [
            loaders[simulated_rank].next_batch()
            for _ in range(MICROSTEPS_PER_RANK)
        ]
        metrics = b2a.process_batches(model, batches, SOURCE_WRITER_UPDATE + 1)
        local_gradient, _writer_elements = flatten_joint_gradients(model)
        rank_metrics.append(metrics)
        local_gradients.append(local_gradient.detach().clone())
    actual = b2a.canonical_batch_hash(
        [row["batch_hashes"] for row in rank_metrics]
    )
    if (
        actual != expected
        or sum(row["target_seen"] for row in rank_metrics) != GLOBAL_TARGETS
    ):
        raise SystemExit("1-GPU joint reference batch mismatch")
    combined = canonical_local_rank_sum(local_gradients)
    writer_elements = WRITER_PARAMETER_COUNT
    scatter_joint_gradients(model, combined, writer_elements)
    gradients = {
        "writer": combined[:writer_elements].detach().cpu(),
        "reader": combined[writer_elements:].detach().cpu(),
    }
    validate_gradient_report(gradient_report(model))
    temporary = temporary_optimizer_vectors(model, writer_optimizer, reader_optimizer)
    artifact = {
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "source_checkpoint_sha256": digest,
        "global_batch_sha256": actual,
        "global_loss": sum(row["raw_loss_sum"] for row in rank_metrics) / GLOBAL_TARGETS,
        "gradients": gradients,
        "temporary": temporary,
        "rank_batch_hashes": [row["batch_hashes"] for row in rank_metrics],
        "targets": sum(row["target_seen"] for row in rank_metrics),
        "gradient_accumulation": "two local microsteps per simulated rank, then fixed rank-order FP32 sum",
    }
    output = Path(args.run_dir) / "migration" / "one_gpu_joint_reference.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite joint migration reference: {output}")
    torch.save(artifact, output)
    summary = {
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "source_checkpoint_sha256": digest,
        "global_batch_sha256": actual,
        "global_loss": artifact["global_loss"],
        "writer_gradient_norm": gradients["writer"].double().norm().item(),
        "reader_gradient_norm": gradients["reader"].double().norm().item(),
        "targets": GLOBAL_TARGETS,
        "artifact_sha256": b2a.file_sha256(output),
        "authoritative_checkpoint_unchanged": b2a.file_sha256(args.source_checkpoint) == SOURCE_SHA256,
        "passed": True,
    }
    b2a.write_json(output.with_suffix(".json"), summary)
    print(
        f"1GPU_JOINT_REFERENCE_PASS loss={artifact['global_loss']:.10f} "
        f"writer_grad={summary['writer_gradient_norm']:.10f} "
        f"reader_grad={summary['reader_gradient_norm']:.10f}",
        flush=True,
    )


def migration_candidate(args):
    require_git(clean=True)
    load_config()
    rank, local_rank = b2a.init_distributed()
    try:
        device = torch.device("cuda", local_rank)
        checkpoint, digest = load_source_checkpoint(args.source_checkpoint)
        symbols, model, writer_optimizer, reader_optimizer = instantiate_source(checkpoint, device)
        b2a.restore_rank_rng(checkpoint["rank_rng_states"][rank], local_rank)
        loader = b2a.make_rank_loader(
            symbols, copy.deepcopy(checkpoint["dataloader_states"]), rank
        )
        expected, preview_hashes = b2a.distributed_preview_hash(loader, symbols)
        if expected != SOURCE_NEXT_SHA256:
            raise SystemExit("4-GPU joint candidate next-batch mismatch")
        batches = [loader.next_batch() for _ in range(MICROSTEPS_PER_RANK)]
        metrics = b2a.process_batches(model, batches, SOURCE_WRITER_UPDATE + 1)
        local_combined, writer_elements = flatten_joint_gradients(model)
        combined, reduction_seconds = canonical_rank_slotted_all_reduce(
            local_combined, rank
        )
        scatter_joint_gradients(model, combined, writer_elements)
        validate_gradient_report(gradient_report(model))
        b2a.all_equal_across_ranks(
            b2a.tensor_digest([("joint_gradient", combined)]),
            "joint synchronized gradient",
        )
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, metrics)
        actual = b2a.canonical_batch_hash([row["batch_hashes"] for row in gathered])
        if actual != expected or metrics["batch_hashes"] != preview_hashes[rank]:
            raise SystemExit("4-GPU joint candidate consumed-batch mismatch")
        reference_path = Path(args.run_dir) / "migration" / "one_gpu_joint_reference.pt"
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        if reference["global_batch_sha256"] != actual:
            raise SystemExit("joint migration paths used different global batches")
        global_loss = sum(row["raw_loss_sum"] for row in gathered) / GLOBAL_TARGETS
        gradients = {
            "writer": combined[:writer_elements].detach().cpu(),
            "reader": combined[writer_elements:].detach().cpu(),
        }
        gradient_comparison = {
            group: b2a.comparison(reference["gradients"][group], gradients[group])
            for group in ("writer", "reader")
        }
        temporary = temporary_optimizer_vectors(model, writer_optimizer, reader_optimizer)
        consistency = b2a.all_equal_across_ranks(
            joint_state_hashes(model, writer_optimizer, reader_optimizer, 30, 1),
            "temporary joint optimizer step",
        )
        optimizer_comparison = {}
        for group in ("writer", "reader"):
            optimizer_comparison[group] = {
                field: b2a.comparison(
                    reference["temporary"][group][field], temporary[group][field]
                )
                for field in ("update", "parameters", "m1", "m2")
            }
        passed = (
            abs(global_loss - reference["global_loss"]) <= 1e-5
            and all(
                gradient_comparison[group]["cosine_similarity"] >= 0.999999
                and gradient_comparison[group]["relative_l2_difference"] <= 1e-4
                and gradient_comparison[group]["relative_norm_difference"] <= 1e-4
                for group in ("writer", "reader")
            )
            and all(
                optimizer_comparison[group]["update"]["cosine_similarity"] >= 0.999999
                and optimizer_comparison[group]["update"]["relative_l2_difference"] <= 1e-4
                and optimizer_comparison[group]["m1"]["relative_l2_difference"] <= 1e-4
                and optimizer_comparison[group]["m2"]["relative_l2_difference"] <= 1e-4
                for group in ("writer", "reader")
            )
            and digest == SOURCE_SHA256
            and b2a.file_sha256(args.source_checkpoint) == SOURCE_SHA256
        )
        if rank == 0:
            audit = {
                "experiment": "2B3",
                "stage": "1GPU_to_4GPU_joint_gradient_equivalence",
                "implementation_git_commit": git_output("rev-parse", "HEAD"),
                "source_checkpoint_sha256": digest,
                "source_next_global_batch_sha256": SOURCE_NEXT_SHA256,
                "consumed_global_batch_sha256": actual,
                "gpu_model": torch.cuda.get_device_name(local_rank),
                "world_size": WORLD_SIZE,
                "rank_to_loader_mapping": {str(value): value for value in range(WORLD_SIZE)},
                "microsteps_per_rank": MICROSTEPS_PER_RANK,
                "loss_scaling": "each token-loss sum / 524288; no division after SUM",
                "gradient_reduction": "one combined rank-slotted flattened FP32 NCCL all_reduce(SUM), then fixed rank-order local sum",
                "gradient_elements": combined.numel(),
                "communication_buffer_elements": WORLD_SIZE * combined.numel(),
                "writer_gradient_elements": writer_elements,
                "reader_gradient_elements": combined.numel() - writer_elements,
                "gradient_all_reduce_seconds": reduction_seconds,
                "one_gpu_loss": reference["global_loss"],
                "four_gpu_loss": global_loss,
                "absolute_loss_difference": abs(global_loss - reference["global_loss"]),
                "gradient_comparison": gradient_comparison,
                "temporary_optimizer_comparison": optimizer_comparison,
                "temporary_cross_rank_state": consistency,
                "acceptance": {
                    "loss_absolute_difference_max": 1e-5,
                    "gradient_cosine_min": 0.999999,
                    "gradient_relative_l2_max": 1e-4,
                    "gradient_norm_relative_difference_max": 1e-4,
                    "parameter_update_cosine_min": 0.999999,
                    "parameter_update_relative_l2_max": 1e-4,
                },
                "temporary_states_disposition": "discarded",
                "authoritative_checkpoint_unchanged": b2a.file_sha256(args.source_checkpoint) == SOURCE_SHA256,
                "status": "PASS" if passed else "FAIL",
                "passed": passed,
            }
            b2a.write_json(Path(args.run_dir) / "FOUR_GPU_JOINT_EQUIVALENCE_AUDIT.json", audit)
            print(
                f"FOUR_GPU_JOINT_EQUIVALENCE_{audit['status']} "
                f"loss_diff={audit['absolute_loss_difference']:.3e} "
                f"writer_cos={gradient_comparison['writer']['cosine_similarity']:.10f} "
                f"reader_cos={gradient_comparison['reader']['cosine_similarity']:.10f}",
                flush=True,
            )
        verdict = torch.tensor([1 if passed else 0], device=device)
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
        if verdict.item() != 1:
            raise SystemExit("four-GPU joint equivalence failed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def load_training_runtime(args, rank, local_rank):
    path = Path(args.checkpoint).resolve()
    digest = b2a.file_sha256(path)
    checkpoint = a0.torch_load(path, mmap=True)
    device = torch.device("cuda", local_rank)
    if checkpoint.get("schema") == SOURCE_SCHEMA:
        if digest != SOURCE_SHA256:
            raise SystemExit("result source must be the canonical 15M checkpoint")
        source, _ = load_source_checkpoint(path)
        symbols, model, writer_optimizer, reader_optimizer = instantiate_source(source, device)
        local_update = 0
        writer_update = SOURCE_WRITER_UPDATE
        loader_states = source["dataloader_states"]
        b2a.restore_rank_rng(source["rank_rng_states"][rank], local_rank)
        rank_seed = source["rank_seeds"][rank]
        source_kind = "canonical_15m_fresh_start"
    elif checkpoint.get("schema") == CHECKPOINT_SCHEMA:
        resume_exact = {
            "parent_2b2a_checkpoint_sha256": SOURCE_SHA256,
            "parent_2b2a_tag": PARENT_TAG,
            "parent_2b2a_results_commit": PARENT_COMMIT,
            "world_size": WORLD_SIZE,
            "rank_to_loader_mapping": {str(value): value for value in range(4)},
            "global_targets_per_update": GLOBAL_TARGETS,
            "gradient_synchronization": "one combined rank-slotted flattened FP32 NCCL all_reduce(SUM), then fixed rank-order local sum",
            "gradient_clipping": "separate synchronized writer and reader norms, each clipped to 1.0",
            "temporal_credit": "exactly one token; writer sources and historical Blocks 2-12 KV detached",
            "implementation_git_commit": git_output("rev-parse", "HEAD"),
            "config_sha256": b2a.file_sha256(CONFIG_PATH),
            "frozen_base_sha256": SOURCE_BASE_SHA256,
            "starting_reader_sha256": SOURCE_READER_SHA256,
            "starting_writer_sha256": SOURCE_WRITER_SHA256,
        }
        for name, expected in resume_exact.items():
            if checkpoint.get(name) != expected:
                raise SystemExit(f"2B3 resume {name} mismatch")
        if len(checkpoint.get("dataloader_states", ())) != WORLD_SIZE:
            raise SystemExit("2B3 resume loader-state count mismatch")
        if len(checkpoint.get("rank_rng_states", ())) != WORLD_SIZE:
            raise SystemExit("2B3 resume rank-RNG count mismatch")
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.is_file() or sidecar.read_text().split()[0] != digest:
            raise SystemExit("2B3 checkpoint SHA sidecar mismatch")
        symbols = a0.support.load_training_symbols()
        with torch.random.fork_rng(devices=[]):
            model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
        model.load_state_dict(checkpoint["model"], strict=True)
        configure_joint_training(model)
        model_hashes = {
            "base": b2.state_subset_sha256(model, "base"),
            "reader": b2.state_subset_sha256(model, "reader"),
            "writers": b2.state_subset_sha256(model, "writers"),
        }
        if model_hashes != {
            "base": SOURCE_BASE_SHA256,
            "reader": checkpoint["reader_sha256"],
            "writers": checkpoint["writer_sha256"],
        }:
            raise SystemExit("2B3 resume model subset hash mismatch")
        model.to(device)
        writer_optimizer = make_optimizer(model, "writer", "cuda")
        reader_optimizer = make_optimizer(model, "reader", "cuda")
        writer_optimizer.load_state_dict(checkpoint["writer_optimizer"])
        reader_optimizer.load_state_dict(checkpoint["reader_optimizer"])
        local_update = checkpoint["training_state"]["joint_local_updates"]
        writer_update = checkpoint["training_state"]["writer_lineage_updates"]
        b2.optimizer_report(writer_optimizer, writer_update)
        reader_optimizer_report(reader_optimizer, local_update)
        if writer_update != SOURCE_WRITER_UPDATE + local_update:
            raise SystemExit("2B3 resume writer/local update mismatch")
        loader_states = checkpoint["dataloader_states"]
        b2a.restore_rank_rng(checkpoint["rank_rng_states"][rank], local_rank)
        rank_seed = checkpoint["rank_seeds"][rank]
        source_kind = "fresh_joint_update_5_resume"
    else:
        raise SystemExit("unsupported 2B3 training checkpoint schema")
    loader = b2a.make_rank_loader(symbols, loader_states, rank)
    expected_next, _ = b2a.distributed_preview_hash(loader, symbols)
    if expected_next != checkpoint["next_global_batch_sha256"]:
        raise SystemExit("2B3 training resume next-batch mismatch")
    return (
        checkpoint, digest, symbols, model, writer_optimizer, reader_optimizer,
        loader, local_update, writer_update, rank_seed, source_kind,
    )


def writer_weight_norms(model):
    named = dict(model.named_parameters())
    return {
        f"v{depth}": {
            "W_down": named[f"transformer.memory_writers.writer_v{depth}.W_down.weight"].detach().float().norm().item(),
            "W_up": named[f"transformer.memory_writers.writer_v{depth}.W_up.weight"].detach().float().norm().item(),
        }
        for depth in b2.SOURCE_DEPTHS
    }


def aggregate_training_metrics(rows, local_update, writer_update, global_hash,
                               writer_pre, writer_post, reader_pre, reader_post,
                               reduction_seconds, total_wall, consistency, model):
    base = b2a.aggregate_update_metrics(
        rows, writer_update, global_hash, writer_pre, writer_post,
        reduction_seconds, total_wall, consistency,
    )
    base["joint_local_update"] = local_update
    base["writer_lineage_update"] = writer_update
    base["writer_lineage_tokens"] = writer_update * GLOBAL_TARGETS
    base["joint_training_tokens"] = local_update * GLOBAL_TARGETS
    base["writer_pre_clip_gradient_norm"] = base.pop("pre_clip_global_gradient_norm")
    base["writer_post_clip_gradient_norm"] = base.pop("post_clip_global_gradient_norm")
    base["reader_pre_clip_gradient_norm"] = float(reader_pre)
    base["reader_post_clip_gradient_norm"] = reader_post
    base["reader"] = b2.reader_values(model)
    base["writer_parameter_norms"] = writer_weight_norms(model)
    base["separate_gradient_clipping"] = {"writer": 1.0, "reader": 1.0}
    base["gradient_synchronization"] = "one combined rank-slotted flattened FP32 all_reduce(SUM), then fixed rank-order local sum"
    return base


def save_distributed_checkpoint(args, source, symbols, model, writer_optimizer,
                                reader_optimizer, loader, local_update, writer_update,
                                rank, local_rank, rank_seed):
    loader_states = [None] * WORLD_SIZE
    rng_states = [None] * WORLD_SIZE
    rank_meta = [None] * WORLD_SIZE
    dist.all_gather_object(loader_states, a0.loader_state(loader))
    dist.all_gather_object(rng_states, b2a.capture_rank_rng(rank, local_rank))
    dist.all_gather_object(rank_meta, {
        "rank": rank, "gpu": local_rank, "loader_state": rank,
        "hostname": os.uname().nodename, "pid": os.getpid(),
    })
    next_hash, next_rank_hashes = b2a.distributed_preview_hash(loader, symbols)
    consistency = b2a.all_equal_across_ranks(
        joint_state_hashes(
            model, writer_optimizer, reader_optimizer, writer_update, local_update
        ),
        "pre-checkpoint joint state",
    )
    if local_update == FINAL_JOINT_UPDATE and next_hash != FINAL_NEXT_SHA256:
        raise SystemExit(f"final matched-counterfactual cursor mismatch: {next_hash}")
    output = Path(args.run_dir) / "checkpoints" / f"checkpoint_joint_updates_{local_update:06d}.pt"
    sidecar = None
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise SystemExit(f"refusing to overwrite 2B3 checkpoint: {output}")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "model": model.state_dict(),
            "writer_optimizer": writer_optimizer.state_dict(),
            "reader_optimizer": reader_optimizer.state_dict(),
            "training_state": {
                "writer_lineage_updates": writer_update,
                "writer_lineage_tokens": writer_update * GLOBAL_TARGETS,
                "joint_local_updates": local_update,
                "joint_training_tokens": local_update * GLOBAL_TARGETS,
                "fineweb_lineage_completed_update": 526 + local_update,
                "kind": "2b3_forced_restart" if local_update == 5 else "2b3_final",
            },
            "dataloader_states": loader_states,
            "rank_rng_states": rng_states,
            "rank_seeds": [2_026_082_000 + value for value in range(WORLD_SIZE)],
            "rank_metadata": rank_meta,
            "next_rank_microstep_hashes": next_rank_hashes,
            "next_global_batch_sha256": next_hash,
            "world_size": WORLD_SIZE,
            "rank_to_loader_mapping": {str(value): value for value in range(WORLD_SIZE)},
            "global_targets_per_update": GLOBAL_TARGETS,
            "gradient_synchronization": "one combined rank-slotted flattened FP32 NCCL all_reduce(SUM), then fixed rank-order local sum",
            "gradient_clipping": "separate synchronized writer and reader norms, each clipped to 1.0",
            "loss_scaling": "token-loss sums / 524288; no post-SUM division",
            "temporal_credit": "exactly one token; writer sources and historical Blocks 2-12 KV detached",
            "teacher_training_forward_calls": 0,
            "parent_2b2a_checkpoint_path": str(Path(args.source_checkpoint).resolve()),
            "parent_2b2a_checkpoint_sha256": SOURCE_SHA256,
            "parent_2b2a_tag": PARENT_TAG,
            "parent_2b2a_results_commit": PARENT_COMMIT,
            "parent_2b2a_checkpoint_implementation_commit": SOURCE_IMPLEMENTATION_COMMIT,
            "implementation_git_commit": git_output("rev-parse", "HEAD"),
            "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "config_sha256": b2a.file_sha256(CONFIG_PATH),
            "frozen_base_sha256": b2.state_subset_sha256(model, "base"),
            "starting_reader_sha256": SOURCE_READER_SHA256,
            "starting_writer_sha256": SOURCE_WRITER_SHA256,
            "reader_sha256": b2.state_subset_sha256(model, "reader"),
            "writer_sha256": b2.state_subset_sha256(model, "writers"),
            "optimizer_consistency": consistency,
            "writer_architecture": source["writer_architecture"],
            "saved_by_pid": os.getpid(),
        }
        if payload["frozen_base_sha256"] != SOURCE_BASE_SHA256:
            raise SystemExit("frozen base changed before checkpoint")
        temporary = output.with_name(output.name + ".incomplete")
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        reopened = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
        with torch.random.fork_rng(devices=[]):
            clone = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
        clone.load_state_dict(reopened["model"], strict=True)
        configure_joint_training(clone)
        clone_writer = make_optimizer(clone, "writer", "cpu")
        clone_reader = make_optimizer(clone, "reader", "cpu")
        clone_writer.load_state_dict(reopened["writer_optimizer"])
        clone_reader.load_state_dict(reopened["reader_optimizer"])
        b2.optimizer_report(clone_writer, writer_update)
        reader_optimizer_report(clone_reader, local_update)
        if reopened["next_global_batch_sha256"] != next_hash:
            raise SystemExit("2B3 strict reopen next-batch mismatch")
        digest = b2a.file_sha256(temporary)
        os.replace(temporary, output)
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{digest}  {output.name}\n"
        )
        sidecar = {
            "checkpoint": str(output.resolve()),
            "sha256": digest,
            "bytes": output.stat().st_size,
            "model_strict_reload": True,
            "writer_optimizer_strict_reload": True,
            "reader_optimizer_strict_reload": True,
            "writer_optimizer": b2.optimizer_report(reopened["writer_optimizer"], writer_update),
            "reader_optimizer": reader_optimizer_report(reopened["reader_optimizer"], local_update),
            "loader_states": len(reopened["dataloader_states"]),
            "rank_rng_states": len(reopened["rank_rng_states"]),
            "next_global_batch_sha256": next_hash,
            "cross_rank_consistency": consistency,
            "passed": True,
        }
        b2a.write_json(output.with_suffix(output.suffix + ".verification.json"), sidecar)
        print(
            f"JOINT_CHECKPOINT_PASS local={local_update} writer={writer_update} "
            f"sha256={digest} next={next_hash}", flush=True,
        )
    dist.barrier()
    return sidecar


def train(args):
    require_git(clean=True)
    load_config()
    audit_path = Path(args.run_dir) / "FOUR_GPU_JOINT_EQUIVALENCE_AUDIT.json"
    if not audit_path.is_file() or not json.loads(audit_path.read_text()).get("passed"):
        raise SystemExit("passing four-GPU joint equivalence is required")
    if args.target_joint_update not in (RESTART_JOINT_UPDATE, FINAL_JOINT_UPDATE):
        raise SystemExit("2B3 target must be joint update 5 or 9")
    rank, local_rank = b2a.init_distributed()
    try:
        stage_started = time.perf_counter()
        (
            checkpoint, checkpoint_digest, symbols, model, writer_optimizer,
            reader_optimizer, loader, completed, writer_completed, rank_seed,
            source_kind,
        ) = load_training_runtime(args, rank, local_rank)
        if completed == 0 and args.target_joint_update != 5:
            raise SystemExit("fresh 2B3 result training must stop at joint update 5")
        if completed == 5 and args.target_joint_update != 9:
            raise SystemExit("fresh restart from joint update 5 must stop at update 9")
        if completed not in (0, 5):
            raise SystemExit(f"unauthorized 2B3 stage start: {completed}")
        frozen_base = b2.state_subset_sha256(model, "base")
        metrics_path = Path(args.run_dir) / "training_metrics.jsonl"
        for local_update in range(completed + 1, args.target_joint_update + 1):
            writer_update = SOURCE_WRITER_UPDATE + local_update
            update_started = time.perf_counter()
            expected_hash, expected_rank_hashes = b2a.distributed_preview_hash(loader, symbols)
            writer_optimizer.zero_grad(set_to_none=True)
            reader_optimizer.zero_grad(set_to_none=True)
            batches = [loader.next_batch() for _ in range(MICROSTEPS_PER_RANK)]
            local = b2a.process_batches(model, batches, writer_update)
            if local["batch_hashes"] != expected_rank_hashes[rank]:
                raise SystemExit("rank consumed batch differs from exact preview")
            local_combined, writer_elements = flatten_joint_gradients(model)
            combined, reduction_seconds = canonical_rank_slotted_all_reduce(
                local_combined, rank
            )
            scatter_joint_gradients(model, combined, writer_elements)
            gradients = gradient_report(model)
            validate_gradient_report(gradients)
            writer_pre = torch.nn.utils.clip_grad_norm_(
                [value for _, value in parameter_rows(model, "writer")], 1.0
            )
            reader_pre = torch.nn.utils.clip_grad_norm_(
                [value for _, value in parameter_rows(model, "reader")], 1.0
            )
            if not torch.isfinite(writer_pre) or not torch.isfinite(reader_pre):
                raise SystemExit("non-finite synchronized gradient norm")
            writer_post = b2.grad_global_norm(
                [value for _, value in parameter_rows(model, "writer")]
            )
            reader_post = b2.grad_global_norm(
                [value for _, value in parameter_rows(model, "reader")]
            )
            writer_optimizer.step()
            reader_optimizer.step()
            torch.cuda.synchronize()
            b2.optimizer_report(writer_optimizer, writer_update)
            reader_optimizer_report(reader_optimizer, local_update)
            consistency = b2a.all_equal_across_ranks(
                joint_state_hashes(
                    model, writer_optimizer, reader_optimizer,
                    writer_update, local_update,
                ),
                f"joint update {local_update} state",
            )
            gathered = [None] * WORLD_SIZE
            dist.all_gather_object(gathered, local)
            actual_hash = b2a.canonical_batch_hash(
                [row["batch_hashes"] for row in gathered]
            )
            if actual_hash != expected_hash:
                raise SystemExit("global consumed batch differs from preview")
            total_wall = time.perf_counter() - update_started
            row = aggregate_training_metrics(
                gathered, local_update, writer_update, actual_hash,
                writer_pre, writer_post, reader_pre, reader_post,
                reduction_seconds, total_wall, consistency, model,
            )
            row["gradients"] = gradients
            row["source_kind"] = source_kind if local_update == completed + 1 else "same_stage"
            row["teacher_training_forward_calls"] = 0
            if rank == 0:
                b2a.append_jsonl(metrics_path, row)
                print(
                    f"JOINT_RESULT_UPDATE_PASS local={local_update} writer={writer_update} "
                    f"loss={row['global_training_loss']:.6f} "
                    f"wgrad={row['writer_pre_clip_gradient_norm']:.6f} "
                    f"rgrad={row['reader_pre_clip_gradient_norm']:.6f} "
                    f"wall={total_wall:.1f}s tok/s={row['effective_tokens_per_second']:.0f}",
                    flush=True,
                )
        if b2.state_subset_sha256(model, "base") != frozen_base or frozen_base != SOURCE_BASE_SHA256:
            raise SystemExit("frozen base changed during joint stage")
        sidecar = save_distributed_checkpoint(
            args, checkpoint, symbols, model, writer_optimizer, reader_optimizer,
            loader, args.target_joint_update,
            SOURCE_WRITER_UPDATE + args.target_joint_update,
            rank, local_rank, rank_seed,
        )
        if rank == 0:
            summary = {
                "stage": "forced_restart_after_5" if args.target_joint_update == 5 else "final_after_9",
                "start_joint_update": completed,
                "end_joint_update": args.target_joint_update,
                "new_joint_updates": args.target_joint_update - completed,
                "writer_lineage_update": SOURCE_WRITER_UPDATE + args.target_joint_update,
                "writer_lineage_tokens": (SOURCE_WRITER_UPDATE + args.target_joint_update) * GLOBAL_TARGETS,
                "joint_training_tokens": args.target_joint_update * GLOBAL_TARGETS,
                "source_kind": source_kind,
                "input_checkpoint_sha256": checkpoint_digest,
                "output_checkpoint": sidecar,
                "frozen_base_unchanged": True,
                "teacher_training_forward_calls": 0,
                "stage_wall_seconds": time.perf_counter() - stage_started,
                "fresh_process_restart_required": args.target_joint_update == 5,
                "hard_stop_reached": args.target_joint_update == 9,
                "passed": True,
            }
            b2a.write_json(
                Path(args.run_dir) / f"training_joint_{args.target_joint_update}.json",
                summary,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def component_state(model, names):
    named = dict(model.named_parameters())
    return {name: named[name].detach().cpu().clone() for name in sorted(names)}


def apply_component_state(model, state):
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def load_evaluation_runtime(args, local_rank, include_teacher=True):
    path = Path(args.checkpoint).resolve()
    checkpoint = a0.torch_load(path, mmap=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("final evaluation checkpoint schema mismatch")
    state = checkpoint.get("training_state", {})
    if state.get("joint_local_updates") != 9 or state.get("writer_lineage_updates") != 38:
        raise SystemExit("final evaluation requires the joint-update-9 checkpoint")
    if checkpoint.get("next_global_batch_sha256") != FINAL_NEXT_SHA256:
        raise SystemExit("final evaluation checkpoint cursor mismatch")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest = b2a.file_sha256(path)
    if not sidecar.is_file() or sidecar.read_text().split()[0] != digest:
        raise SystemExit("final evaluation checkpoint SHA mismatch")
    source, source_digest = load_source_checkpoint(args.source_checkpoint)
    symbols = a0.support.load_training_symbols()
    device = torch.device("cuda", local_rank)
    with torch.random.fork_rng(devices=[]):
        model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)
    teacher = None
    if include_teacher:
        with torch.random.fork_rng(devices=[]):
            teacher = symbols["GPT"](b2.model_config(symbols, enable_writers=False))
        missing, unexpected = teacher.load_state_dict(source["model"], strict=False)
        expected_unexpected = {
            name for name in source["model"]
            if name.startswith("transformer.memory_writers.")
        }
        if missing or set(unexpected) != expected_unexpected:
            raise SystemExit(
                f"teacher evaluation load mismatch: missing={missing}, unexpected={unexpected}"
            )
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval().to(device)
    return checkpoint, digest, source, source_digest, symbols, model, teacher, device


def _all_stream_results_finite(rows, names):
    return all(
        row["conditions"][name]["finite"]
        and row["conditions"][name]["cache_health"]["all_expected_lengths"]
        and row["conditions"][name]["cache_health"]["block_1_cache_absent"]
        for row in rows for name in names
    )


def average_routing(results):
    return {
        f"v{depth}": sum(
            result["routing_weights"][f"v{depth}"] for result in results
        ) / len(results)
        for depth in b2.SOURCE_DEPTHS
    } | {
        "entropy": sum(result["routing_entropy"] for result in results) / len(results)
    }


def final_evaluate(args):
    require_git(clean=True)
    load_config()
    rank, local_rank = b2a.init_distributed()
    try:
        (
            checkpoint, checkpoint_digest, source, source_digest, symbols,
            model, teacher, device,
        ) = load_evaluation_runtime(args, local_rank, include_teacher=True)
        start_writer = {
            name: source["model"][name].detach().cpu().clone()
            for name in sorted(b2.TRAINABLE_NAMES)
        }
        start_reader = {
            name: source["model"][name].detach().cpu().clone()
            for name in sorted(READER_NAMES)
        }
        final_writer = component_state(model, b2.TRAINABLE_NAMES)
        final_reader = component_state(model, READER_NAMES)
        final_hashes = {
            group: b2.state_subset_sha256(model, group)
            for group in ("base", "reader", "writers")
        }
        final_reader_metrics = b2.reader_values(model)
        final_writer_norms = writer_weight_norms(model)
        apply_component_state(model, start_reader)
        start_reader_metrics = b2.reader_values(model)
        apply_component_state(model, final_reader)
        apply_component_state(model, start_writer)
        start_writer_norms = writer_weight_norms(model)
        apply_component_state(model, final_writer)
        local_model_metrics = {
            "start_reader": start_reader_metrics,
            "final_reader": final_reader_metrics,
            "start_writer_parameter_norms": start_writer_norms,
            "final_writer_parameter_norms": final_writer_norms,
        }
        b2a.all_equal_across_ranks(local_model_metrics, "evaluation component metrics")
        progress_path = Path(args.run_dir) / "evaluation" / f"final_rank{rank}.json"
        progress = b2a.rank_progress(progress_path)
        loader = b2.validation_loader(symbols)
        permutation = symbols["fixed_derangement"](B, device)
        payload_hashes = []
        stream_names = (
            "start_writer_start_reader",
            "final_writer_final_reader",
            "final_writer_final_reader_shuffled",
            "final_writer_final_reader_gate_zero",
            "final_writer_start_reader",
            "start_writer_final_reader",
            "teacher_sources_final_writer_final_reader",
        )
        for batch_index in range(a0.VALIDATION_BATCHES):
            x_cpu, y_cpu = loader.next_batch()
            payload_hash = a0.batch_payload_hash(x_cpu, y_cpu)
            payload_hashes.append(payload_hash)
            if batch_index % WORLD_SIZE != rank:
                continue
            row = b2a.progress_lookup(progress, batch_index)
            if row.get("payload_sha256") not in (None, payload_hash):
                raise SystemExit("final evaluation payload mismatch")
            row["payload_sha256"] = payload_hash
            row.setdefault("conditions", {})
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            apply_component_state(model, final_writer)
            apply_component_state(model, final_reader)
            if "full_context" not in row["conditions"]:
                row["conditions"]["full_context"] = {
                    "loss": b2.b0.parallel_loss(model, x, y, "full_context"),
                    "finite": True,
                }
                b2a.write_json(progress_path, progress)
            if "masked_l1_no_feedback" not in row["conditions"]:
                row["conditions"]["masked_l1_no_feedback"] = {
                    "loss": b2.b0.parallel_loss(model, x, y, "masked_l1_no_feedback"),
                    "finite": True,
                }
                b2a.write_json(progress_path, progress)
            if "final_writer_final_reader" not in row["conditions"]:
                row["conditions"]["final_writer_final_reader"] = b2.stream_loss(
                    model, x, y, mode="masked_l1_topdown_self", use_writers=True
                )
                b2a.write_json(progress_path, progress)
            if "final_writer_final_reader_shuffled" not in row["conditions"]:
                row["conditions"]["final_writer_final_reader_shuffled"] = b2.stream_loss(
                    model, x, y, mode="masked_l1_shuffled_self_feedback",
                    use_writers=True, permutation=permutation,
                )
                b2a.write_json(progress_path, progress)
            if "final_writer_final_reader_gate_zero" not in row["conditions"]:
                row["conditions"]["final_writer_final_reader_gate_zero"] = b2.stream_loss(
                    model, x, y, mode="masked_l1_topdown_self",
                    use_writers=True, gate_override=0.0,
                )
                b2a.write_json(progress_path, progress)
            apply_component_state(model, start_writer)
            apply_component_state(model, start_reader)
            if "start_writer_start_reader" not in row["conditions"]:
                row["conditions"]["start_writer_start_reader"] = b2.stream_loss(
                    model, x, y, mode="masked_l1_topdown_self", use_writers=True
                )
                b2a.write_json(progress_path, progress)
            apply_component_state(model, final_writer)
            if "final_writer_start_reader" not in row["conditions"]:
                row["conditions"]["final_writer_start_reader"] = b2.stream_loss(
                    model, x, y, mode="masked_l1_topdown_self", use_writers=True
                )
                b2a.write_json(progress_path, progress)
            apply_component_state(model, start_writer)
            apply_component_state(model, final_reader)
            if "start_writer_final_reader" not in row["conditions"]:
                row["conditions"]["start_writer_final_reader"] = b2.stream_loss(
                    model, x, y, mode="masked_l1_topdown_self", use_writers=True
                )
                b2a.write_json(progress_path, progress)
            apply_component_state(model, final_writer)
            if "teacher_sources_final_writer_final_reader" not in row["conditions"]:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    teacher_raw = teacher.capture_residual_sources(x, b2.SOURCE_DEPTHS)
                memory = b2.writer_adapted_teacher_memory(model, teacher_raw)
                row["conditions"]["teacher_sources_final_writer_final_reader"] = b2.stream_loss(
                    model, x, y, mode="masked_l1_topdown_teacher",
                    use_writers=False, feedback_sources=memory,
                )
                b2a.write_json(progress_path, progress)
                del teacher_raw, memory
            apply_component_state(model, final_reader)
            print(
                f"rank={rank} final-eval={batch_index:02d} "
                f"real={row['conditions']['final_writer_final_reader']['loss']:.6f} "
                f"shuffled={row['conditions']['final_writer_final_reader_shuffled']['loss']:.6f}",
                flush=True,
            )
            del x, y
            torch.cuda.empty_cache()
        digest = hashlib.sha256()
        for value in payload_hashes:
            digest.update(bytes.fromhex(value))
        validation_digest = digest.hexdigest()
        if validation_digest != a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256:
            raise SystemExit("final canonical validation hash mismatch")
        apply_component_state(model, final_writer)
        apply_component_state(model, final_reader)
        after_hashes = {
            group: b2.state_subset_sha256(model, group)
            for group in ("base", "reader", "writers")
        }
        if after_hashes != final_hashes:
            raise SystemExit("final component state not restored after cross-swap evaluation")
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, progress["rows"])
        if rank == 0:
            rows = sorted(
                (row for group in gathered for row in group),
                key=lambda row: row["batch_index"],
            )
            if [row["batch_index"] for row in rows] != list(range(20)):
                raise SystemExit("final evaluation coverage mismatch")
            condition_losses = {
                name: [row["conditions"][name]["loss"] for row in rows]
                for name in ("full_context", "masked_l1_no_feedback", *stream_names)
            }
            means = {
                name: sum(values) / len(values)
                for name, values in condition_losses.items()
            }
            real = condition_losses["final_writer_final_reader"]
            shuffled = condition_losses["final_writer_final_reader_shuffled"]
            paired = b2.paired_statistics(real, shuffled)
            gap = means["final_writer_final_reader_shuffled"] - means["final_writer_final_reader"]
            real_gain = 5.0959878206253055 - means["final_writer_final_reader"]
            specific_change = gap - 0.029013395309448242
            integrity = {
                "checkpoint_sha256_verified": checkpoint_digest == b2a.file_sha256(args.checkpoint),
                "parent_checkpoint_sha256_verified": source_digest == SOURCE_SHA256,
                "final_next_global_batch_sha256_exact": checkpoint["next_global_batch_sha256"] == FINAL_NEXT_SHA256,
                "canonical_validation_sha256_exact": validation_digest == a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256,
                "all_losses_finite": all(
                    math.isfinite(value)
                    for values in condition_losses.values() for value in values
                ),
                "all_stream_cache_health_passed": _all_stream_results_finite(rows, stream_names),
                "frozen_base_sha256_exact": after_hashes["base"] == SOURCE_BASE_SHA256,
                "final_reader_sha256_restored": after_hashes["reader"] == checkpoint["reader_sha256"],
                "final_writer_sha256_restored": after_hashes["writers"] == checkpoint["writer_sha256"],
                "teacher_training_forward_calls_zero": checkpoint["teacher_training_forward_calls"] == 0,
                "hellaswag_not_run": True,
                "start_start_reproduces_canonical_15m": abs(
                    means["start_writer_start_reader"] - 5.0959878206253055
                ) <= 5e-6,
                "full_context_regression_within_tolerance": abs(
                    means["full_context"] - 4.078654408454895
                ) <= 5e-4,
                "masked_regression_within_tolerance": abs(
                    means["masked_l1_no_feedback"] - 5.973674488067627
                ) <= 5e-4,
                "gate_zero_regression_within_tolerance": abs(
                    means["final_writer_final_reader_gate_zero"] - 5.9736480712890625
                ) <= 5e-4,
            }
            integrity["passed"] = all(integrity.values())
            if not integrity["passed"]:
                classification = "JOINT CO-ADAPTATION IS UNSTABLE"
            elif (
                real_gain >= 0.010
                and gap >= 0.0240133953
                and paired["real_wins"] >= 18
            ):
                classification = "JOINT CO-ADAPTATION IMPROVES RECURRENT MEMORY"
            elif real_gain >= 0.010 and gap < 0.0240133953:
                classification = "JOINT CO-ADAPTATION IMPROVES GENERIC COMPENSATION ONLY"
            elif abs(real_gain) < 0.010:
                classification = "JOINT CO-ADAPTATION IS NEUTRAL"
            else:
                classification = "JOINT CO-ADAPTATION DEGRADES"
            start_results = [row["conditions"]["start_writer_start_reader"] for row in rows]
            final_results = [row["conditions"]["final_writer_final_reader"] for row in rows]
            report = {
                "experiment": "2B3",
                "stage": "canonical_final_validation_and_cross_swap",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256": checkpoint_digest,
                "validation_global_batches_sha256": validation_digest,
                "losses": means,
                "canonical_names": {
                    "full_context": means["full_context"],
                    "masked": means["masked_l1_no_feedback"],
                    "starting_15m_real": means["start_writer_start_reader"],
                    "final_joint_real": means["final_writer_final_reader"],
                    "final_joint_shuffled": means["final_writer_final_reader_shuffled"],
                    "final_joint_gate_zero": means["final_writer_final_reader_gate_zero"],
                    "teacher_sources_final_joint": means["teacher_sources_final_writer_final_reader"],
                },
                "cross_swap": {
                    "start_writer_start_reader": means["start_writer_start_reader"],
                    "final_writer_start_reader": means["final_writer_start_reader"],
                    "start_writer_final_reader": means["start_writer_final_reader"],
                    "final_writer_final_reader": means["final_writer_final_reader"],
                },
                "paired": paired,
                "specific_gap_joint": gap,
                "specific_gap_gain_over_writer_only": gap - 0.008815073966979448,
                "joint_vs_writer_only_real_delta": means["final_writer_final_reader"] - 4.81970694065094,
                "joint_real_gain_from_start": real_gain,
                "specific_gap_change_from_start": specific_change,
                "restores_memory_specificity_vs_writer_only": gap >= 0.0188150740,
                "reader_evolution": {
                    "parameter_metrics": {
                        "start": start_reader_metrics,
                        "final": final_reader_metrics,
                    },
                    "routing_start": average_routing(start_results),
                    "routing_final": average_routing(final_results),
                },
                "writer_evolution": {
                    "start_behavior": b2.average_writer_behavior(start_results),
                    "final_behavior": b2.average_writer_behavior(final_results),
                    "start_parameter_norms": start_writer_norms,
                    "final_parameter_norms": final_writer_norms,
                },
                "integrity": integrity,
                "classification": classification,
                "conditional_diagnostics_required": classification == "JOINT CO-ADAPTATION IMPROVES RECURRENT MEMORY",
                "teacher_training_forward_calls": 0,
                "hellaswag_run": False,
                "batch_rows": rows,
                "passed": integrity["passed"],
            }
            b2a.write_json(Path(args.run_dir) / "FINAL_EVALUATION.json", report)
            print(
                f"FINAL_EVALUATION_PASS real={means['final_writer_final_reader']:.10f} "
                f"shuffled={means['final_writer_final_reader_shuffled']:.10f} "
                f"gap={gap:.10f} wins={paired['real_wins']}/20 "
                f"classification={classification}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _diagnostic_progress(path):
    path = Path(path)
    if not path.is_file():
        return {"rows": []}
    payload = json.loads(path.read_text())
    if set(payload) != {"rows"} or not isinstance(payload["rows"], list):
        raise SystemExit(f"invalid conditional diagnostic progress: {path}")
    return payload


def conditional_diagnostics(args):
    require_git(clean=True)
    load_config()
    rank, local_rank = b2a.init_distributed()
    try:
        evaluation_path = Path(args.run_dir) / "FINAL_EVALUATION.json"
        if not evaluation_path.is_file():
            raise SystemExit("conditional diagnostics require FINAL_EVALUATION.json")
        evaluation = json.loads(evaluation_path.read_text())
        required = evaluation.get("conditional_diagnostics_required") is True
        if not required:
            if rank == 0:
                report = {
                    "experiment": "2B3",
                    "stage": "conditional_reset_and_ablation_diagnostics",
                    "status": "SKIPPED_AS_REQUIRED",
                    "classification": evaluation["classification"],
                    "reason": "primary IMPROVES RECURRENT MEMORY criterion did not pass",
                    "hellaswag_run": False,
                    "passed": True,
                }
                b2a.write_json(Path(args.run_dir) / "CONDITIONAL_DIAGNOSTICS.json", report)
                print("CONDITIONAL_DIAGNOSTICS_SKIPPED_AS_REQUIRED", flush=True)
            return
        if args.writer_only_checkpoint is None:
            raise SystemExit("passing classification requires --writer-only-checkpoint")
        (
            checkpoint, checkpoint_digest, source, _source_digest, symbols,
            model, _teacher, device,
        ) = load_evaluation_runtime(args, local_rank, include_teacher=False)
        writer_only_path = Path(args.writer_only_checkpoint).resolve()
        if b2a.file_sha256(writer_only_path) != "d65ff192e037862008d85253a215d3112922c9c8365a576671462f3eaf56a838":
            raise SystemExit("writer-only matched-token checkpoint SHA mismatch")
        writer_only = a0.torch_load(writer_only_path, mmap=True)
        if (
            writer_only.get("schema") != SOURCE_SCHEMA
            or writer_only.get("training_state", {}).get("writer_updates") != 38
            or writer_only.get("next_global_batch_sha256") != FINAL_NEXT_SHA256
        ):
            raise SystemExit("writer-only matched-token checkpoint metadata mismatch")
        start_writer = {
            name: source["model"][name].detach().cpu().clone()
            for name in sorted(b2.TRAINABLE_NAMES)
        }
        start_reader = {
            name: source["model"][name].detach().cpu().clone()
            for name in sorted(READER_NAMES)
        }
        writer_only_writer = {
            name: writer_only["model"][name].detach().cpu().clone()
            for name in sorted(b2.TRAINABLE_NAMES)
        }
        final_writer = component_state(model, b2.TRAINABLE_NAMES)
        final_reader = component_state(model, READER_NAMES)
        final_hashes = {
            group: b2.state_subset_sha256(model, group)
            for group in ("base", "reader", "writers")
        }
        loader = b2.validation_loader(symbols)
        batches = []
        validation_digest = hashlib.sha256()
        for _ in range(20):
            x_cpu, y_cpu = loader.next_batch()
            validation_digest.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
            batches.append((x_cpu, y_cpu))
        if validation_digest.hexdigest() != a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256:
            raise SystemExit("conditional diagnostic validation hash mismatch")
        intervals = (1, 2, 4, 8, 16, 32, 64, 128, None)
        interval_names = ["never" if value is None else str(value) for value in intervals]
        model_states = {
            "start_15m": (start_writer, start_reader),
            "writer_only_20m": (writer_only_writer, start_reader),
            "final_joint": (final_writer, final_reader),
        }
        tasks = []
        for model_name in model_states:
            for interval, interval_name in zip(intervals, interval_names):
                for batch_index in range(20):
                    tasks.append({
                        "kind": "reset",
                        "model": model_name,
                        "interval": interval,
                        "interval_name": interval_name,
                        "batch_index": batch_index,
                    })
        for depth in b2.SOURCE_DEPTHS:
            for batch_index in range(20):
                tasks.append({
                    "kind": "ablation",
                    "writer": f"v{depth}",
                    "depth": depth,
                    "batch_index": batch_index,
                })
        progress_path = Path(args.run_dir) / "conditional" / f"rank{rank}.json"
        progress = _diagnostic_progress(progress_path)
        existing = {row["task_id"] for row in progress["rows"]}
        for task_index, task in enumerate(tasks):
            if task_index % WORLD_SIZE != rank:
                continue
            task_id = (
                f"reset:{task['model']}:{task['interval_name']}:{task['batch_index']}"
                if task["kind"] == "reset"
                else f"ablation:{task['writer']}:{task['batch_index']}"
            )
            if task_id in existing:
                continue
            x_cpu, y_cpu = batches[task["batch_index"]]
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            if task["kind"] == "reset":
                writer_state, reader_state = model_states[task["model"]]
                apply_component_state(model, writer_state)
                apply_component_state(model, reader_state)
                result = b2.stream_loss(
                    model, x, y, mode="masked_l1_topdown_self", use_writers=True,
                    reset_interval=task["interval"],
                )
                row = {
                    "task_id": task_id,
                    "kind": "reset",
                    "model": task["model"],
                    "interval": task["interval_name"],
                    "batch_index": task["batch_index"],
                    "loss": result["loss"],
                    "finite": result["finite"],
                    "cache_health": result["cache_health"],
                }
            else:
                apply_component_state(model, final_writer)
                apply_component_state(model, final_reader)
                result = b2.stream_loss(
                    model, x, y, mode="masked_l1_topdown_self", use_writers=True,
                    disabled_writer_depths=(task["depth"],),
                )
                row = {
                    "task_id": task_id,
                    "kind": "ablation",
                    "writer": task["writer"],
                    "batch_index": task["batch_index"],
                    "loss": result["loss"],
                    "finite": result["finite"],
                    "cache_health": result["cache_health"],
                }
            progress["rows"].append(row)
            existing.add(task_id)
            b2a.write_json(progress_path, progress)
            print(f"rank={rank} conditional={task_id} loss={row['loss']:.6f}", flush=True)
            del x, y, result
            torch.cuda.empty_cache()
        apply_component_state(model, final_writer)
        apply_component_state(model, final_reader)
        after_hashes = {
            group: b2.state_subset_sha256(model, group)
            for group in ("base", "reader", "writers")
        }
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, progress["rows"])
        if rank == 0:
            rows = [row for group in gathered for row in group]
            if len(rows) != len(tasks) or len({row["task_id"] for row in rows}) != len(tasks):
                raise SystemExit("conditional diagnostic task coverage mismatch")
            reset_losses = {
                model_name: {
                    interval_name: sum(
                        row["loss"] for row in rows
                        if row["kind"] == "reset"
                        and row["model"] == model_name
                        and row["interval"] == interval_name
                    ) / 20
                    for interval_name in interval_names
                }
                for model_name in model_states
            }
            baseline_rows = {
                row["batch_index"]: row["conditions"]["final_writer_final_reader"]["loss"]
                for row in evaluation["batch_rows"]
            }
            ablations = {}
            for depth in b2.SOURCE_DEPTHS:
                name = f"v{depth}"
                selected = sorted(
                    (row for row in rows if row["kind"] == "ablation" and row["writer"] == name),
                    key=lambda row: row["batch_index"],
                )
                deltas = [
                    row["loss"] - baseline_rows[row["batch_index"]] for row in selected
                ]
                ablations[name] = {
                    "ablated_loss": sum(row["loss"] for row in selected) / 20,
                    "delta": sum(deltas) / 20,
                    "positive_batches": sum(value > 0 for value in deltas),
                    "negative_batches": sum(value < 0 for value in deltas),
                    "ties": sum(value == 0 for value in deltas),
                    "batch_losses": [row["loss"] for row in selected],
                    "batch_deltas": deltas,
                }
            integrity = {
                "checkpoint_sha256_verified": checkpoint_digest == b2a.file_sha256(args.checkpoint),
                "writer_only_checkpoint_sha256_verified": b2a.file_sha256(writer_only_path) == "d65ff192e037862008d85253a215d3112922c9c8365a576671462f3eaf56a838",
                "validation_sha256_exact": validation_digest.hexdigest() == a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256,
                "all_results_finite": all(row["finite"] for row in rows),
                "all_cache_health_passed": all(
                    row["cache_health"]["all_expected_lengths"]
                    and row["cache_health"]["block_1_cache_absent"]
                    for row in rows
                ),
                "final_model_state_restored": after_hashes == final_hashes,
                "hellaswag_not_run": True,
            }
            integrity["passed"] = all(integrity.values())
            report = {
                "experiment": "2B3",
                "stage": "conditional_reset_and_ablation_diagnostics",
                "status": "COMPLETED",
                "classification": evaluation["classification"],
                "validation_global_batches_sha256": validation_digest.hexdigest(),
                "reset_interval_losses": reset_losses,
                "writer_residual_ablations": ablations,
                "integrity": integrity,
                "hellaswag_run": False,
                "passed": integrity["passed"],
            }
            b2a.write_json(Path(args.run_dir) / "CONDITIONAL_DIAGNOSTICS.json", report)
            if not report["passed"]:
                raise SystemExit("conditional diagnostics failed")
            print("CONDITIONAL_DIAGNOSTICS_PASS", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "source-audit", "preflight", "migration-reference",
        "migration-candidate", "train", "evaluate-final",
        "conditional-diagnostics",
    ))
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--target-joint-update", type=int)
    parser.add_argument("--writer-only-checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(a0.SEED)
    np.random.seed(a0.SEED)
    torch.manual_seed(a0.SEED)
    torch.cuda.manual_seed(a0.SEED)
    if args.command == "source-audit":
        source_audit(args)
    elif args.command == "preflight":
        preflight(args)
    elif args.command == "migration-reference":
        migration_reference(args)
    elif args.command == "migration-candidate":
        migration_candidate(args)
    elif args.command == "train":
        if args.checkpoint is None or args.target_joint_update is None:
            raise SystemExit("train requires --checkpoint and --target-joint-update")
        train(args)
    elif args.command == "evaluate-final":
        if args.checkpoint is None:
            raise SystemExit("evaluate-final requires --checkpoint")
        final_evaluate(args)
    elif args.command == "conditional-diagnostics":
        if args.checkpoint is None:
            raise SystemExit("conditional-diagnostics requires --checkpoint")
        conditional_diagnostics(args)


if __name__ == "__main__":
    main()
