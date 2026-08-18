#!/usr/bin/env python3
"""Experiment 2C1: four independent teacher-reader destination workers."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
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
import experiment_2b4 as b4  # noqa: E402


BRANCH = "experiment-2c1-destination-depth-sweep"
PARENT_TAG = "experiment-2c0-separated-b1-final"
PARENT_COMMIT = "677d711bc00dba0da1b80cb6369f33841ec29a51"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2c1_destination_depth.json"
PROTOCOL = "exp2c1_destination_depth_teacher_reader_v1"
CHECKPOINT_SCHEMA = "exp2c1_destination_reader_v1"
DESTINATIONS = {"D1": 1, "D5": 5, "D9": 9, "D12": 12}
GPU_MAPPING = {"D1": 0, "D5": 1, "D9": 2, "D12": 3}
MILESTONES = (10, 20, 29, 48)
TARGET_UPDATE = 48
RESTART_UPDATE = 20
GLOBAL_TARGETS = 524_288
CANONICAL_SHA = a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256
CALIBRATION_SHA = "d159c297f26e5e7ef707d37c5656b3702d66a11809ebed5577cd12903bfcb2f6"
BASE_MODEL_SHA = "1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd"
PINNED_FULL = 4.0786544085
PINNED_D1_REAL = 5.8353391409
PINNED_D1_SHUFFLED = 5.8765912533
PINNED_D1_GAP = 0.0412521124
D1_ATOL = 5e-6
SOURCE_DEPTHS = tuple(a0.SOURCE_DEPTHS)
TRAINABLE_PARAMETERS = 1_537
T19_975 = 2.093024054408263


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2C1 requires branch {BRANCH}")
    if git_output("rev-parse", f"{PARENT_TAG}^{{}}") != PARENT_COMMIT:
        raise SystemExit("frozen Experiment 2C0 tag target mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2C1 execution requires a clean worktree")


def file_sha256(path):
    return a0.file_sha256(Path(path))


def tensor_bytes(tensor):
    value = tensor.detach().cpu().contiguous()
    # PyTorch 2.8 rejects a dtype-changing view on a rank-0 tensor. Flattening
    # preserves the exact storage bytes while making scalar parameters (the
    # reader gate) hashable by the same path as vectors and matrices.
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(name, tensor):
    digest = hashlib.sha256()
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(tensor_bytes(value))
    return digest.hexdigest()


def durable_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def durable_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path, payload):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        "protocol": PROTOCOL,
        "destinations": DESTINATIONS,
        "gpu_mapping": GPU_MAPPING,
        "parent_checkpoint_sha256": a0.EXPECTED_PARENT_SHA256,
        "source_depths": list(SOURCE_DEPTHS),
        "global_targets_per_update": GLOBAL_TARGETS,
        "optimizer_updates": TARGET_UPDATE,
        "milestone_updates": list(MILESTONES),
        "forced_process_restart_after_update": RESTART_UPDATE,
        "canonical_validation_sha256": CANONICAL_SHA,
        "writers": "forbidden",
        "hellaswag": "forbidden",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"2C1 config mismatch: {mismatches}")
    return config


def destination_number(destination):
    if destination not in DESTINATIONS:
        raise SystemExit(f"unknown 2C1 destination: {destination}")
    return DESTINATIONS[destination]


def destination_index(destination):
    return destination_number(destination) - 1


def paired_statistics(real, shuffled):
    if len(real) != 20 or len(shuffled) != 20:
        raise ValueError("canonical paired statistics require exactly 20 batches")
    gaps = [float(right) - float(left) for left, right in zip(real, shuffled)]
    if not all(math.isfinite(value) for value in [*real, *shuffled, *gaps]):
        raise ValueError("paired losses must be finite")
    sample_std = statistics.stdev(gaps)
    standard_error = sample_std / math.sqrt(len(gaps))
    return {
        "real_wins": sum(value > 0 for value in gaps),
        "shuffled_wins": sum(value < 0 for value in gaps),
        "ties": sum(value == 0 for value in gaps),
        "mean_gap": statistics.fmean(gaps),
        "median_gap": statistics.median(gaps),
        "sample_std": sample_std,
        "minimum": min(gaps),
        "maximum": max(gaps),
        "standard_error": standard_error,
        "ci95_lower": statistics.fmean(gaps) - T19_975 * standard_error,
        "ci95_upper": statistics.fmean(gaps) + T19_975 * standard_error,
        "gaps": gaps,
    }


def reader_parameters(student):
    reader = student.transformer.topdown_attnres
    named = dict(reader.named_parameters())
    if set(named) != {"query", "norm.weight", "gate"}:
        raise SystemExit(f"unexpected reader parameters: {sorted(named)}")
    if sum(value.numel() for value in named.values()) != TRAINABLE_PARAMETERS:
        raise SystemExit("reader parameter count mismatch")
    return reader, named


def reader_metrics(student):
    reader, _ = reader_parameters(student)
    return {
        "gate": reader.gate.detach().float().item(),
        "effective_gate": reader.gate.detach().float().tanh().item(),
        "query_norm": reader.query.detach().float().norm().item(),
        "rmsnorm_displacement": (
            reader.norm.weight.detach().float() - 1.0
        ).norm().item(),
    }


def reader_state(student):
    return {
        key: value.detach().cpu().clone()
        for key, value in student.transformer.topdown_attnres.state_dict().items()
    }


def reader_state_sha(student):
    digest = hashlib.sha256()
    for name, value in sorted(reader_state(student).items()):
        digest.update(name.encode())
        digest.update(tensor_bytes(value))
    return digest.hexdigest()


def frozen_hashes(student, teacher):
    return {
        "student_base": a0.state_tensor_sha256(student, include_topdown=False),
        "teacher": a0.state_tensor_sha256(teacher, include_topdown=False),
    }


def validate_frozen_hashes(student, teacher):
    hashes = frozen_hashes(student, teacher)
    if hashes != {"student_base": BASE_MODEL_SHA, "teacher": BASE_MODEL_SHA}:
        raise SystemExit(f"frozen model hash mismatch: {hashes}")
    return hashes


def make_runtime(parent_checkpoint, include_optimizer):
    symbols, teacher, student, parent_aux = a0.load_models(
        parent_checkpoint, torch.device("cuda", 0), include_teacher=True
    )
    validate_frozen_hashes(student, teacher)
    reader, named = reader_parameters(student)
    initialization = {
        "query_exact_zero": reader.query.count_nonzero().item() == 0,
        "rmsnorm_exact_one": torch.equal(
            reader.norm.weight.detach(), torch.ones_like(reader.norm.weight)
        ),
        "gate_exact_zero": reader.gate.detach().item() == 0.0,
        "trainable_parameters": sum(value.numel() for value in named.values()),
    }
    initialization["passed"] = all(
        initialization[key]
        for key in ("query_exact_zero", "rmsnorm_exact_one", "gate_exact_zero")
    ) and initialization["trainable_parameters"] == TRAINABLE_PARAMETERS
    if not initialization["passed"]:
        raise SystemExit(f"fresh reader initialization mismatch: {initialization}")
    optimizer = a0.feedback_optimizer(student) if include_optimizer else None
    loaders = a0.make_replay_loaders(
        symbols, copy.deepcopy(parent_aux["dataloader_states"])
    )
    return symbols, teacher, student, optimizer, loaders, parent_aux, initialization


def destination_forward(
    student,
    x,
    y=None,
    destination=None,
    control="real",
    memory=None,
    permutation=None,
):
    block = destination_index(destination)
    if control == "full_context":
        return student(x, y, mode="full_context")
    if control == "masked":
        return student(
            x,
            y,
            mode="masked_destination_no_feedback",
            feedback_destination_block=block,
        )
    if control in {"real", "generic", "zero"}:
        return student(
            x,
            y,
            mode="masked_destination_topdown_teacher",
            feedback_sources=memory,
            feedback_gate_override=0.0 if control == "zero" else None,
            feedback_destination_block=block,
        )
    if control == "shuffle":
        return student(
            x,
            y,
            mode="masked_destination_shuffled_feedback",
            feedback_sources=memory,
            feedback_permutation=permutation,
            feedback_destination_block=block,
        )
    raise ValueError(control)


def cache_policy(student, destination, batch_size=2, dtype=torch.float32):
    block = destination_index(destination)
    state = student.init_recurrent_state(
        batch_size,
        "masked_single_no_feedback",
        device="cuda",
        dtype=dtype,
        mask_depth=0,
        masked_block_index=block,
    )
    report = {
        "destination_block": destination_number(destination),
        "destination_cache_absent": state.kv_caches[block] is None,
        "only_destination_cache_absent": all(
            (cache is None) == (index == block)
            for index, cache in enumerate(state.kv_caches)
        ),
        "non_destination_caches_empty": all(
            cache.length == 0
            for index, cache in enumerate(state.kv_caches)
            if index != block
        ),
        "fresh_memory_zero": state.feedback_memory.count_nonzero().item() == 0,
    }
    report["passed"] = all(value for key, value in report.items() if key != "destination_block")
    return report


@torch.no_grad()
def causal_and_isolation_preflight(student, teacher, symbols, destination):
    loader = symbols["DataLoaderLite"](
        B=2, T=32, process_rank=0, num_processes=1, split="val"
    )
    first, _ = loader.next_batch()
    first = first.cuda()
    future = first.clone()
    future[:, 16:] = (future[:, 16:] + 17) % student.config.vocab_size
    row_changed = first.clone()
    row_changed[1] = (row_changed[1] + 29) % student.config.vocab_size
    reader = student.transformer.topdown_attnres
    saved_gate = reader.gate.detach().clone()
    reader.gate.fill_(math.atanh(0.25))
    try:
        memory_first = symbols["shift_teacher_sources"](
            teacher.capture_residual_sources(first, SOURCE_DEPTHS)
        )
        memory_future = symbols["shift_teacher_sources"](
            teacher.capture_residual_sources(future, SOURCE_DEPTHS)
        )
        memory_row = symbols["shift_teacher_sources"](
            teacher.capture_residual_sources(row_changed, SOURCE_DEPTHS)
        )
        logits_first, _ = destination_forward(
            student, first, destination=destination, memory=memory_first
        )
        logits_future, _ = destination_forward(
            student, future, destination=destination, memory=memory_future
        )
        logits_row, _ = destination_forward(
            student, row_changed, destination=destination, memory=memory_row
        )
        masked_logits, _ = destination_forward(
            student, first, destination=destination, control="masked"
        )
        zero_logits, _ = destination_forward(
            student,
            first,
            destination=destination,
            control="zero",
            memory=memory_first,
        )
        legacy = None
        if destination == "D1":
            legacy, _ = student(
                first,
                mode="masked_l1_topdown_teacher",
                feedback_sources=memory_first,
            )
        report = {
            "future_prefix_logits_bit_exact": torch.equal(
                logits_first[:, :16], logits_future[:, :16]
            ),
            "teacher_memory_prefix_bit_exact": torch.equal(
                memory_first[:, :, :16], memory_future[:, :, :16]
            ),
            "unchanged_row_logits_bit_exact": torch.equal(
                logits_first[0], logits_row[0]
            ),
            "zero_gate_equals_masked_bit_exact": torch.equal(
                masked_logits, zero_logits
            ),
            "all_outputs_finite": all(
                torch.isfinite(value).all()
                for value in (logits_first, logits_future, logits_row)
            ),
            "d1_legacy_equivalence": (
                True if legacy is None else torch.equal(legacy, logits_first)
            ),
        }
        report["passed"] = all(report.values())
        return report
    finally:
        reader.gate.copy_(saved_gate)


def validation_loader(symbols):
    return symbols["DataLoaderLite"](
        B=a0.VALIDATION_B,
        T=a0.T,
        process_rank=0,
        num_processes=1,
        split="val",
    )


def generic_bank(means, batch_size, sequence_length, dtype):
    time_mask = torch.ones(
        (1, 1, sequence_length, 1), device=means.device, dtype=torch.float32
    )
    time_mask[:, :, 0] = 0
    template = means.float()[:, None, None, :] * time_mask
    return template.to(dtype).expand(-1, batch_size, -1, -1)


@torch.no_grad()
def evaluate_teacher_controls(
    student,
    teacher,
    symbols,
    destination,
    completed_updates,
    include_full=False,
    generic_means=None,
):
    student.eval()
    teacher.eval()
    loader = validation_loader(symbols)
    controls = ["masked", "real", "shuffle", "zero"]
    if include_full:
        controls.insert(0, "full_context")
    if generic_means is not None:
        controls.append("generic")
    losses = {name: [] for name in controls}
    validation_hash = hashlib.sha256()
    permutation = symbols["fixed_derangement"](a0.VALIDATION_B, "cuda")
    routing_sum = torch.zeros(4, dtype=torch.float64)
    entropy_sum = 0.0
    topdown_rms_sum = 0.0
    feedback_rms_sum = 0.0
    zero_equals_masked = True
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        validation_hash.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        memory = a0.teacher_memory(teacher, x, symbols)
        if include_full:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = destination_forward(
                    student, x, y, destination, "full_context"
                )
            losses["full_context"].append(loss.detach().double().item())
            del logits, loss
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, masked_loss = destination_forward(
                student, x, y, destination, "masked"
            )
        losses["masked"].append(masked_loss.detach().double().item())
        del logits
        student.set_topdown_instrumentation(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, real_loss = destination_forward(
                student, x, y, destination, "real", memory=memory
            )
        del logits
        stats = student.get_topdown_stats()
        student.set_topdown_instrumentation(False)
        routing_sum += torch.tensor(stats["mean_weights"], dtype=torch.float64)
        entropy_sum += stats["mean_entropy"]
        topdown_rms_sum += stats["topdown_rms"]
        feedback_rms_sum += stats["feedback_rms"]
        losses["real"].append(real_loss.detach().double().item())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, shuffled_loss = destination_forward(
                student,
                x,
                y,
                destination,
                "shuffle",
                memory=memory,
                permutation=permutation,
            )
        del logits
        losses["shuffle"].append(shuffled_loss.detach().double().item())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, zero_loss = destination_forward(
                student, x, y, destination, "zero", memory=memory
            )
        del logits
        losses["zero"].append(zero_loss.detach().double().item())
        zero_equals_masked &= torch.equal(masked_loss, zero_loss)
        if generic_means is not None:
            template = generic_bank(
                generic_means, x.size(0), x.size(1), memory.dtype
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, generic_loss = destination_forward(
                    student, x, y, destination, "generic", memory=template
                )
            del logits
            losses["generic"].append(generic_loss.detach().double().item())
            del generic_loss, template
        del x, y, memory, masked_loss, real_loss, shuffled_loss, zero_loss
        print(
            f"{destination} eval@{completed_updates} batch "
            f"{batch_index + 1:02d}/{a0.VALIDATION_BATCHES}",
            flush=True,
        )
    digest = validation_hash.hexdigest()
    if digest != CANONICAL_SHA:
        raise SystemExit(f"canonical validation hash mismatch: {digest}")
    means = {name: statistics.fmean(values) for name, values in losses.items()}
    paired = paired_statistics(losses["real"], losses["shuffle"])
    full = means.get("full_context", PINNED_FULL)
    damage = means["masked"] - full
    recovery = means["masked"] - means["real"]
    specific_gap = paired["mean_gap"]
    result = {
        "experiment": "2C1",
        "destination": destination,
        "destination_block": destination_number(destination),
        "completed_updates": completed_updates,
        "processed_reader_tokens": completed_updates * GLOBAL_TARGETS,
        "canonical_validation_sha256": digest,
        "losses": means,
        "per_batch_losses": losses,
        "paired_real_vs_shuffled": paired,
        "damage": damage,
        "low_mask_damage": damage < 0.02,
        "recovery": recovery,
        "recovery_fraction": recovery / damage if damage > 0 else None,
        "specific_gap": specific_gap,
        "specific_fraction": specific_gap / damage if damage >= 0.02 else None,
        "specific_share": specific_gap / recovery if recovery > 0 else None,
        "zero_gate_equals_masked": zero_equals_masked,
        "reader": reader_metrics(student),
        "router": {
            "routing_weights": {
                f"v{depth}": value
                for depth, value in zip(
                    SOURCE_DEPTHS,
                    (routing_sum / a0.VALIDATION_BATCHES).tolist(),
                )
            },
            "routing_entropy": entropy_sum / a0.VALIDATION_BATCHES,
            "topdown_rms": topdown_rms_sum / a0.VALIDATION_BATCHES,
            "feedback_rms": feedback_rms_sum / a0.VALIDATION_BATCHES,
        },
        "generic_means_tensor_sha256": (
            None
            if generic_means is None
            else tensor_sha256("generic_teacher_source_means", generic_means)
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "hellaswag_run": False,
    }
    result["passed"] = (
        zero_equals_masked
        and all(math.isfinite(value) for value in means.values())
        and paired["real_wins"] + paired["shuffled_wins"] + paired["ties"] == 20
    )
    if not result["passed"]:
        raise SystemExit(f"teacher control evaluation failed: {result}")
    return result


@torch.no_grad()
def compute_generic_means(teacher, symbols):
    loader = validation_loader(symbols)
    hashes = []
    for _ in range(20):
        x, y = loader.next_batch()
        hashes.append(a0.batch_payload_hash(x, y))
    total = torch.zeros((4, 768), device="cuda", dtype=torch.float64)
    count = 0
    for _ in range(4):
        x_cpu, y_cpu = loader.next_batch()
        hashes.append(a0.batch_payload_hash(x_cpu, y_cpu))
        x = x_cpu.cuda(non_blocking=True)
        memory = a0.teacher_memory(teacher, x, symbols)
        total += memory[:, :, 1:].double().sum(dim=(1, 2))
        count += memory.size(1) * (memory.size(2) - 1)
        del x, memory
    means = (total / count).float()
    if not torch.isfinite(means).all() or count != 4 * 64 * 1023:
        raise SystemExit("invalid generic teacher source means")
    calibration_aggregate = b4.aggregate_payload_hash(hashes[20:])
    if calibration_aggregate != CALIBRATION_SHA:
        raise SystemExit(
            f"generic calibration aggregate mismatch: {calibration_aggregate}"
        )
    return means, {
        "calibration_batch_indices": [20, 21, 22, 23],
        "batch_payload_sha256": hashes[20:],
        "calibration_aggregate_sha256": calibration_aggregate,
        "count": count,
        "tensor_sha256": tensor_sha256("generic_teacher_source_means", means),
        "source_shas": {
            f"mu{depth}": tensor_sha256(f"mu{depth}", means[index])
            for index, depth in enumerate(SOURCE_DEPTHS)
        },
    }


def run_preflight(args):
    require_git(clean=True)
    config = load_config()
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_device = str(GPU_MAPPING[args.destination])
    if visible_device != expected_device:
        raise SystemExit(
            f"{args.destination} requires CUDA_VISIBLE_DEVICES={expected_device}, "
            f"got {visible_device!r}"
        )
    if file_sha256(args.parent_checkpoint) != a0.EXPECTED_PARENT_SHA256:
        raise SystemExit("Experiment 1B parent checkpoint SHA mismatch")
    symbols, teacher, student, _, _, parent_aux, initialization = make_runtime(
        args.parent_checkpoint, include_optimizer=False
    )
    before = {
        **frozen_hashes(student, teacher),
        "reader": reader_state_sha(student),
    }
    cache = cache_policy(student, args.destination)
    causality = causal_and_isolation_preflight(
        student, teacher, symbols, args.destination
    )
    evaluation = evaluate_teacher_controls(
        student,
        teacher,
        symbols,
        args.destination,
        completed_updates=0,
        include_full=True,
    )
    after = {
        **frozen_hashes(student, teacher),
        "reader": reader_state_sha(student),
    }
    trainable = [
        name for name, value in student.named_parameters() if value.requires_grad
    ]
    integrity = {
        "initialization": initialization["passed"],
        "cache_policy": cache["passed"],
        "causality_and_isolation": causality["passed"],
        "full_context_reference": abs(
            evaluation["losses"]["full_context"] - PINNED_FULL
        ) <= D1_ATOL,
        "zero_gate_equivalence": evaluation["zero_gate_equals_masked"],
        "trainable_parameters_exactly_1537": sum(
            value.numel() for value in student.parameters() if value.requires_grad
        ) == TRAINABLE_PARAMETERS,
        "trainable_names_exact": set(trainable) == {
            "transformer.topdown_attnres.query",
            "transformer.topdown_attnres.norm.weight",
            "transformer.topdown_attnres.gate",
        },
        "base_gradients_none": all(
            value.grad is None
            for name, value in student.named_parameters()
            if not name.startswith("transformer.topdown_attnres.")
        ),
        "teacher_gradients_none": all(value.grad is None for value in teacher.parameters()),
        "teacher_eval": not teacher.training,
        "frozen_hashes_unchanged": before == after,
        "all_losses_finite": all(
            math.isfinite(value) for value in evaluation["losses"].values()
        ),
        "hellaswag_not_run": True,
    }
    integrity["passed"] = all(integrity.values())
    report = {
        "experiment": "2C1",
        "stage": "preflight",
        "destination": args.destination,
        "destination_block": destination_number(args.destination),
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "config": config,
        "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "parent_checkpoint_sha256": parent_aux["checkpoint_sha256"],
        "optimizer_constructions": 0,
        "hardware": {
            "expected_physical_gpu": GPU_MAPPING[args.destination],
            "cuda_visible_devices": visible_device,
            "device_name": torch.cuda.get_device_name(0),
        },
        "initialization": initialization,
        "cache_policy": cache,
        "causality": causality,
        "evaluation": evaluation,
        "integrity": integrity,
        "passed": integrity["passed"],
    }
    if not report["passed"]:
        raise SystemExit(f"2C1 preflight failed: {report}")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    durable_json(run_dir / "preflight.json", report)
    print(f"EXPERIMENT_2C1_PREFLIGHT_PASS destination={args.destination}", flush=True)
    return report


def checkpoint_path(run_dir, completed_updates):
    return Path(run_dir) / "checkpoints" / f"checkpoint_updates_{completed_updates:06d}.pt"


def evaluation_path(run_dir, completed_updates):
    return Path(run_dir) / f"evaluation_updates_{completed_updates:06d}.json"


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_sha256(path)


def optimizer_integrity(optimizer, completed_updates):
    state = optimizer.state_dict()
    groups = state.get("param_groups", [])
    if len(groups) != 1:
        raise SystemExit("2C1 requires one reader optimizer group")
    steps = sorted(
        int(values["step"].detach().cpu().item())
        for values in state.get("state", {}).values()
    )
    expected_steps = [] if completed_updates == 0 else [completed_updates] * 3
    finite = all(
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or torch.isfinite(value).all()
        for values in state.get("state", {}).values()
        for value in values.values()
    )
    report = {
        "state_entries": len(state.get("state", {})),
        "steps": steps,
        "expected_steps": expected_steps,
        "finite": bool(finite),
        "betas": list(groups[0]["betas"]),
        "eps": groups[0]["eps"],
        "weight_decay": groups[0]["weight_decay"],
        "lr": groups[0]["lr"],
        "expected_lr": a0.get_lr(
            a0.EXPECTED_PARENT_UPDATES + max(completed_updates - 1, 0)
        ),
    }
    report["passed"] = (
        steps == expected_steps
        and finite
        and report["betas"] == [0.9, 0.95]
        and report["eps"] == 1e-8
        and report["weight_decay"] == 0.0
        and report["lr"] == report["expected_lr"]
    )
    if not report["passed"]:
        raise SystemExit(f"reader optimizer integrity failure: {report}")
    return report


def move_optimizer_to_cuda(optimizer):
    for values in optimizer.state.values():
        for name, value in list(values.items()):
            if isinstance(value, torch.Tensor):
                values[name] = value.cuda()


def run_identity(destination, parent_aux, config):
    return {
        "experiment": "2C1",
        "protocol": PROTOCOL,
        "destination": destination,
        "destination_block": destination_number(destination),
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "parent_tag": PARENT_TAG,
        "parent_commit": PARENT_COMMIT,
        "parent_checkpoint": parent_aux["checkpoint"],
        "parent_checkpoint_sha256": parent_aux["checkpoint_sha256"],
        "base_model_sha256": BASE_MODEL_SHA,
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_depths": list(SOURCE_DEPTHS),
        "mask_semantics": "only destination attention history removed",
        "teacher": "frozen full context, eval/no_grad, one-token shifted raw sources",
        "training_objective": "next-token cross entropy only",
        "config": config,
        "hellaswag_run": False,
    }


def save_checkpoint(
    run_dir,
    destination,
    completed_updates,
    student,
    teacher,
    optimizer,
    loaders,
    symbols,
    identity,
):
    path = checkpoint_path(run_dir, completed_updates)
    if path.exists():
        raise SystemExit(f"refusing to overwrite 2C1 checkpoint: {path}")
    next_hash = a0.next_update_hash(loaders, symbols, replay=True)
    rng = a0.capture_rng_state()
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "experiment": "2C1",
        "destination": destination,
        "destination_block": destination_number(destination),
        "base_model_sha256": BASE_MODEL_SHA,
        "parent_checkpoint_sha256": a0.EXPECTED_PARENT_SHA256,
        "reader_state": reader_state(student),
        "reader_state_sha256": reader_state_sha(student),
        "optimizer": optimizer.state_dict(),
        "optimizer_integrity": optimizer_integrity(optimizer, completed_updates),
        "completed_updates": completed_updates,
        "processed_reader_tokens": completed_updates * GLOBAL_TARGETS,
        "dataloader_states": a0.snapshot_loaders(loaders),
        "rng_state": rng,
        "next_global_batch_sha256": next_hash,
        "source_depths": list(SOURCE_DEPTHS),
        "mask_semantics": "only destination attention history removed",
        "identity": identity,
        "writer_pid": os.getpid(),
    }
    digest = atomic_torch_save(path, payload)
    reopened = a0.torch_load(path, mmap=True)
    strict = {
        "schema": reopened.get("schema") == CHECKPOINT_SCHEMA,
        "destination": reopened.get("destination") == destination,
        "completed_updates": reopened.get("completed_updates") == completed_updates,
        "reader": a0.nested_equal(reopened.get("reader_state"), payload["reader_state"]),
        "optimizer": a0.nested_equal(reopened.get("optimizer"), payload["optimizer"]),
        "loaders": a0.nested_equal(
            reopened.get("dataloader_states"), payload["dataloader_states"]
        ),
        "rng": a0.nested_equal(reopened.get("rng_state"), rng),
        "next_hash": reopened.get("next_global_batch_sha256") == next_hash,
    }
    strict["passed"] = all(strict.values())
    if not strict["passed"]:
        raise SystemExit(f"2C1 checkpoint strict reopen failed: {strict}")
    sidecar = {
        "checkpoint": str(path.resolve()),
        "sha256": digest,
        "completed_updates": completed_updates,
        "next_global_batch_sha256": next_hash,
        "strict_reopen": strict,
        "passed": True,
    }
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), sidecar)
    return sidecar


def load_checkpoint(
    path,
    destination,
    student,
    optimizer,
    loaders,
    symbols,
    identity,
):
    path = Path(path).resolve()
    verification_path = path.with_suffix(path.suffix + ".verification.json")
    if not verification_path.is_file():
        raise SystemExit("2C1 resume checkpoint verification is missing")
    verification = json.loads(verification_path.read_text())
    digest = file_sha256(path)
    if digest != verification.get("sha256"):
        raise SystemExit("2C1 resume checkpoint SHA mismatch")
    checkpoint = a0.torch_load(path, mmap=True)
    required = {
        "schema": checkpoint.get("schema") == CHECKPOINT_SCHEMA,
        "destination": checkpoint.get("destination") == destination,
        "destination_block": checkpoint.get("destination_block")
        == destination_number(destination),
        "parent": checkpoint.get("parent_checkpoint_sha256")
        == a0.EXPECTED_PARENT_SHA256,
        "base": checkpoint.get("base_model_sha256") == BASE_MODEL_SHA,
        "identity": checkpoint.get("identity") == identity,
        "source_depths": checkpoint.get("source_depths") == list(SOURCE_DEPTHS),
    }
    if not all(required.values()):
        raise SystemExit(f"2C1 resume lineage mismatch: {required}")
    student.transformer.topdown_attnres.load_state_dict(
        checkpoint["reader_state"], strict=True
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_to_cuda(optimizer)
    a0.restore_loader_group(
        loaders, checkpoint["dataloader_states"], symbols, replay=True
    )
    a0.restore_rng_state(checkpoint["rng_state"])
    completed = checkpoint["completed_updates"]
    reader_ok = reader_state_sha(student) == checkpoint["reader_state_sha256"]
    optimizer_report = optimizer_integrity(optimizer, completed)
    next_hash = a0.next_update_hash(loaders, symbols, replay=True)
    audit = {
        "checkpoint": str(path),
        "sha256": digest,
        "lineage": required,
        "reader_exact_reload": reader_ok,
        "optimizer": optimizer_report,
        "next_hash_exact": next_hash == checkpoint["next_global_batch_sha256"],
        "fresh_process": os.getpid() != checkpoint.get("writer_pid"),
        "completed_updates": completed,
    }
    audit["passed"] = (
        all(required.values())
        and reader_ok
        and optimizer_report["passed"]
        and audit["next_hash_exact"]
        and audit["fresh_process"]
    )
    if not audit["passed"]:
        raise SystemExit(f"2C1 strict resume failed: {audit}")
    return completed, audit


def train_one_update(
    destination,
    update,
    student,
    teacher,
    optimizer,
    loaders,
    symbols,
):
    student.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    student.set_topdown_instrumentation(True)
    update_hash = hashlib.sha256()
    loss_total = 0.0
    routing_sum = torch.zeros(4, dtype=torch.float64)
    entropy_sum = 0.0
    topdown_rms_sum = 0.0
    feedback_rms_sum = 0.0
    target_count = 0
    forward_seconds = 0.0
    backward_seconds = 0.0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    microbatches = a0.LEGACY_WORLD_SIZE * a0.LEGACY_GRAD_ACCUM
    for x_cpu, y_cpu in a0.update_batches(loaders, replay=True):
        update_hash.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
        target_count += y_cpu.numel()
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        memory = a0.teacher_memory(teacher, x, symbols)
        if teacher.training or memory.requires_grad or memory.grad_fn is not None:
            raise SystemExit("teacher memory detach/eval contract failed")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = destination_forward(
                student, x, y, destination, "real", memory=memory
            )
        del logits
        scaled = loss / microbatches
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - forward_start
        backward_start = time.perf_counter()
        scaled.backward()
        torch.cuda.synchronize()
        backward_seconds += time.perf_counter() - backward_start
        loss_total += scaled.detach().float().item()
        stats = student.get_topdown_stats()
        routing_sum += torch.tensor(stats["mean_weights"], dtype=torch.float64)
        entropy_sum += stats["mean_entropy"]
        topdown_rms_sum += stats["topdown_rms"]
        feedback_rms_sum += stats["feedback_rms"]
        del x, y, memory, loss, scaled
    student.set_topdown_instrumentation(False)
    if target_count != GLOBAL_TARGETS:
        raise SystemExit(f"global target geometry mismatch: {target_count}")
    reader, named = reader_parameters(student)
    gradient_rows = {}
    for name, parameter in named.items():
        gradient_rows[name] = {
            "present": parameter.grad is not None,
            "finite": parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all()),
            "nonzero": parameter.grad is not None
            and bool(parameter.grad.count_nonzero().item()),
            "norm": None
            if parameter.grad is None
            else parameter.grad.detach().float().norm().item(),
        }
    base_gradients = [
        name
        for name, value in student.named_parameters()
        if not name.startswith("transformer.topdown_attnres.") and value.grad is not None
    ]
    teacher_gradients = [
        name for name, value in teacher.named_parameters() if value.grad is not None
    ]
    if (
        base_gradients
        or teacher_gradients
        or not all(row["present"] and row["finite"] for row in gradient_rows.values())
    ):
        raise SystemExit(
            f"2C1 gradient boundary failure: reader={gradient_rows} "
            f"base={base_gradients} teacher={teacher_gradients}"
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1.0)
    if not torch.isfinite(grad_norm) or not math.isfinite(loss_total):
        raise SystemExit("non-finite 2C1 loss/gradient")
    lr = a0.get_lr(a0.EXPECTED_PARENT_UPDATES + update)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    if any(not torch.isfinite(value).all() for value in named.values()):
        raise SystemExit("non-finite trained reader parameter")
    hashes = validate_frozen_hashes(student, teacher)
    optimizer_report = optimizer_integrity(optimizer, update + 1)
    row = {
        "experiment": "2C1",
        "destination": destination,
        "destination_block": destination_number(destination),
        "update": update,
        "completed_updates": update + 1,
        "processed_reader_tokens": (update + 1) * GLOBAL_TARGETS,
        "global_schedule_step": a0.EXPECTED_PARENT_UPDATES + update,
        "global_batch_sha256": update_hash.hexdigest(),
        "global_targets": target_count,
        "loss": loss_total,
        "lr": lr,
        "gradient_norm": float(grad_norm),
        "gradients": gradient_rows,
        "base_gradients": base_gradients,
        "teacher_gradients": teacher_gradients,
        "reader": reader_metrics(student),
        "router": {
            "routing_weights": {
                f"v{depth}": value
                for depth, value in zip(
                    SOURCE_DEPTHS, (routing_sum / microbatches).tolist()
                )
            },
            "routing_entropy": entropy_sum / microbatches,
            "topdown_rms": topdown_rms_sum / microbatches,
            "feedback_rms": feedback_rms_sum / microbatches,
        },
        "optimizer": optimizer_report,
        "frozen_hashes": hashes,
        "teacher_eval_no_grad": True,
        "writers_active_calls": 0,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }
    print(
        f"2C1 {destination} update={update + 1:02d}/{TARGET_UPDATE} "
        f"loss={loss_total:.6f} gap_target=teacher "
        f"gate={row['reader']['effective_gate']:.6f}",
        flush=True,
    )
    return row


def self_state(student, destination, batch_size):
    return student.init_recurrent_state(
        batch_size,
        "masked_single_no_feedback",
        device="cuda",
        dtype=torch.bfloat16,
        mask_depth=0,
        masked_block_index=destination_index(destination),
    )


@torch.no_grad()
def evaluate_self_controls(student, symbols, destination):
    student.eval()
    loader = validation_loader(symbols)
    permutation = symbols["fixed_derangement"](a0.VALIDATION_B, "cuda")
    controls = ("real", "shuffle", "zero")
    losses = {name: [] for name in controls}
    validation_hash = hashlib.sha256()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        validation_hash.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        for control in controls:
            state = self_state(student, destination, x.size(0))
            loss_sum = torch.zeros((), device="cuda", dtype=torch.float64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for position in range(x.size(1)):
                    bank = state.feedback_memory.detach()
                    if control == "shuffle":
                        bank = bank[:, permutation]
                    if control == "zero":
                        feedback = x.new_zeros(
                            (x.size(0), 1, student.config.n_embd),
                            dtype=torch.bfloat16,
                        )
                    else:
                        topdown = student.transformer.topdown_attnres(
                            list(bank.unbind(dim=0))
                        )
                        feedback = (
                            student.transformer.topdown_attnres.gate.tanh()
                            * topdown
                        )
                    logits, state = student.forward_step(
                        x[:, position], state, attention_feedback=feedback
                    )
                    loss_sum += F.cross_entropy(
                        logits[:, 0], y[:, position], reduction="sum"
                    ).double()
            block = destination_index(destination)
            cache_health = all(
                cache is None if index == block else cache.length == a0.T
                for index, cache in enumerate(state.kv_caches)
            )
            if not cache_health or not torch.isfinite(loss_sum):
                raise SystemExit("self-recurrent cache/finite invariant failed")
            losses[control].append((loss_sum / x.numel()).item())
            del state, loss_sum
        del x, y
        print(
            f"2C1 {destination} self batch {batch_index + 1:02d}/20",
            flush=True,
        )
    digest = validation_hash.hexdigest()
    if digest != CANONICAL_SHA:
        raise SystemExit("self-transfer validation prefix mismatch")
    means = {name: statistics.fmean(values) for name, values in losses.items()}
    paired = paired_statistics(losses["real"], losses["shuffle"])
    return {
        "destination": destination,
        "destination_block": destination_number(destination),
        "canonical_validation_sha256": digest,
        "losses": means,
        "per_batch_losses": losses,
        "paired_real_vs_shuffled": paired,
        "self_specific_gap": paired["mean_gap"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "integrity": {
            "no_training": True,
            "gate_unchanged": True,
            "target_cache_absent": True,
            "non_target_caches_normal": True,
            "all_losses_finite": all(math.isfinite(value) for value in means.values()),
            "hellaswag_not_run": True,
        },
        "passed": all(math.isfinite(value) for value in means.values()),
    }


def reconcile_metrics(path, completed_updates):
    path = Path(path)
    rows = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    retained = [row for row in rows if row.get("completed_updates", 0) <= completed_updates]
    expected = list(range(1, completed_updates + 1))
    if [row["completed_updates"] for row in retained] != expected:
        raise SystemExit("2C1 metrics do not match resume checkpoint")
    durable_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
    )
    return retained


def run_training(args):
    if not args.allow_optimizer_steps:
        raise SystemExit("2C1 optimizer steps require --allow-optimizer-steps")
    require_git(clean=True)
    config = load_config()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_MAPPING[args.destination]):
        raise SystemExit("2C1 worker is not mapped to its preregistered physical GPU")
    run_dir = Path(args.run_dir)
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file():
        raise SystemExit("destination preflight must pass before optimizer construction")
    preflight = json.loads(preflight_path.read_text())
    if (
        not preflight.get("passed")
        or preflight.get("destination") != args.destination
        or preflight.get("implementation_git_commit") != git_output("rev-parse", "HEAD")
    ):
        raise SystemExit("stale or failed 2C1 destination preflight")
    (
        symbols,
        teacher,
        student,
        optimizer,
        loaders,
        parent_aux,
        initialization,
    ) = make_runtime(args.parent_checkpoint, include_optimizer=True)
    identity = run_identity(args.destination, parent_aux, config)
    identity_path = run_dir / "run_identity.json"
    metrics_path = run_dir / "metrics.jsonl"
    restart_audit = None
    if args.resume:
        if args.target_update != TARGET_UPDATE:
            raise SystemExit("resumed 2C1 workers must target update 48")
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise SystemExit("2C1 resume identity mismatch")
        completed, restart_audit = load_checkpoint(
            args.resume,
            args.destination,
            student,
            optimizer,
            loaders,
            symbols,
            identity,
        )
        if completed != RESTART_UPDATE:
            raise SystemExit("2C1 forced restart is authorized only from update 20")
        reconcile_metrics(metrics_path, completed)
        durable_json(run_dir / "restart_audit_updates_000020.json", restart_audit)
    else:
        if args.target_update != RESTART_UPDATE:
            raise SystemExit("fresh 2C1 workers must hard-stop at update 20")
        if identity_path.exists() or metrics_path.exists():
            raise SystemExit("refusing to overwrite an existing 2C1 result run")
        completed = 0
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        durable_json(identity_path, identity)
        durable_text(metrics_path, "")
        if a0.next_update_hash(loaders, symbols, replay=True) != a0.EXPECTED_NEXT_GLOBAL_BATCH_SHA256:
            raise SystemExit("fresh 2C1 parent data cursor mismatch")
    optimizer_integrity(optimizer, completed)
    validate_frozen_hashes(student, teacher)
    checkpoint_records = {}
    evaluation_records = {}
    for update in range(completed, args.target_update):
        row = train_one_update(
            args.destination, update, student, teacher, optimizer, loaders, symbols
        )
        append_jsonl(metrics_path, row)
        completed = update + 1
        if completed not in MILESTONES:
            continue
        checkpoint_record = save_checkpoint(
            run_dir,
            args.destination,
            completed,
            student,
            teacher,
            optimizer,
            loaders,
            symbols,
            identity,
        )
        checkpoint_records[str(completed)] = checkpoint_record
        generic_means = None
        generic_manifest = None
        if completed == TARGET_UPDATE:
            generic_means, generic_manifest = compute_generic_means(teacher, symbols)
            durable_json(run_dir / "generic_means_manifest.json", generic_manifest)
            atomic_torch_save(
                run_dir / "generic_teacher_source_means.pt",
                {
                    "means": generic_means.detach().cpu(),
                    "metadata": generic_manifest,
                },
            )
        evaluation = evaluate_teacher_controls(
            student,
            teacher,
            symbols,
            args.destination,
            completed,
            include_full=False,
            generic_means=generic_means,
        )
        evaluation["checkpoint_sha256"] = checkpoint_record["sha256"]
        evaluation["implementation_git_commit"] = identity["implementation_git_commit"]
        durable_json(evaluation_path(run_dir, completed), evaluation)
        evaluation_records[str(completed)] = evaluation
        if completed == TARGET_UPDATE:
            qualifies = (
                evaluation["specific_gap"] >= 0.020
                and evaluation["paired_real_vs_shuffled"]["real_wins"] >= 18
                and evaluation["recovery"] > 0
                and evaluation["passed"]
            )
            if qualifies:
                self_transfer = evaluate_self_controls(
                    student, symbols, args.destination
                )
                self_transfer["triggered"] = True
                self_transfer["teacher_recovery"] = evaluation["recovery"]
                self_recovery = (
                    self_transfer["losses"]["zero"]
                    - self_transfer["losses"]["real"]
                )
                self_transfer["self_recovery"] = self_recovery
                self_transfer["self_teacher_recovery_ratio"] = (
                    self_recovery / evaluation["recovery"]
                )
            else:
                self_transfer = {
                    "destination": args.destination,
                    "destination_block": destination_number(args.destination),
                    "triggered": False,
                    "status": "SELF TEST NOT TRIGGERED",
                    "teacher_gate": {
                        "specific_gap_at_least_0_020": evaluation["specific_gap"] >= 0.020,
                        "real_wins_at_least_18": evaluation["paired_real_vs_shuffled"]["real_wins"] >= 18,
                        "teacher_recovery_positive": evaluation["recovery"] > 0,
                        "integrity": evaluation["passed"],
                    },
                    "passed": True,
                }
            durable_json(run_dir / "self_transfer.json", self_transfer)
    if completed != args.target_update:
        raise SystemExit("2C1 worker target mismatch")
    rows = [
        json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()
    ]
    if [row["completed_updates"] for row in rows] != list(range(1, completed + 1)):
        raise SystemExit("2C1 worker metrics coverage mismatch")
    stage = {
        "experiment": "2C1",
        "destination": args.destination,
        "completed_updates": completed,
        "processed_reader_tokens": completed * GLOBAL_TARGETS,
        "forced_fresh_process_restart_required": completed == RESTART_UPDATE,
        "restart_audit": restart_audit,
        "checkpoint_records": checkpoint_records,
        "evaluation_records": sorted(evaluation_records),
        "global_batch_hashes": [row["global_batch_sha256"] for row in rows],
        "performance": {
            "training_wall_seconds": sum(row["wall_seconds"] for row in rows),
            "tokens_per_second": completed * GLOBAL_TARGETS
            / sum(row["wall_seconds"] for row in rows),
            "peak_allocated_mb": max(row["peak_allocated_mb"] for row in rows),
        },
        "hellaswag_run": False,
        "passed": True,
    }
    durable_json(run_dir / f"stage_updates_{completed:06d}.json", stage)
    marker = (
        "EXPERIMENT_2C1_RESTART_REQUIRED"
        if completed == RESTART_UPDATE
        else "EXPERIMENT_2C1_WORKER_COMPLETE"
    )
    print(f"{marker} destination={args.destination} updates={completed}", flush=True)
    return stage


def destination_run_dir(run_root, destination):
    return Path(run_root) / f"{destination}_block{destination_number(destination)}"


def linear_slope(xs, ys):
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    return sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator


def classify(destination_rows, integrity):
    if not integrity:
        return "DESTINATION-DEPTH EXPERIMENT UNSTABLE"
    higher = [destination_rows[key] for key in ("D5", "D9", "D12")]
    strong = [
        row
        for row in higher
        if row["specific_gap"] >= 0.050
        and row["real_wins"] >= 18
        and row["recovery"] > 0
        and row["real_loss"] + 0.010 <= row["generic_loss"]
    ]
    if strong:
        gaps = [destination_rows[key]["specific_gap"] for key in DESTINATIONS]
        slope = linear_slope(list(DESTINATIONS.values()), gaps)
        if slope > 0 and statistics.fmean(gaps[1:]) > gaps[0]:
            return "SEQUENCE MEMORY NEED INCREASES WITH DESTINATION DEPTH"
        return "SEQUENCE MEMORY IS STRONG ONLY AT SPECIFIC DEPTHS"
    if sum(row["low_mask_damage"] for row in higher) >= 2:
        return "DESTINATION MASK DAMAGE TOO SMALL TO RESOLVE HIERARCHY"
    meaningful = [row for row in destination_rows.values() if not row["low_mask_damage"]]
    generic_dominates = (
        not any(row["specific_gap"] >= 0.050 for row in higher)
        and meaningful
        and all(row["generic_loss"] <= row["real_loss"] + 0.010 for row in meaningful)
    )
    if generic_dominates:
        return "GENERIC CORRECTION DOMINATES ACROSS DEPTHS"
    return "DESTINATION-DEPTH RESULT IS MIXED"


def comparison_answer(left, right, tolerance=1e-12):
    delta = right - left
    if delta > tolerance:
        return f"YES; the latter gap is larger by {delta:.10f}."
    if delta < -tolerance:
        return f"NO; the latter gap is smaller by {-delta:.10f}."
    return "NO RESOLVABLE DIFFERENCE."


def copy_destination_artifacts(run_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "preflight.json",
        "run_identity.json",
        "metrics.jsonl",
        "restart_audit_updates_000020.json",
        "stage_updates_000020.json",
        "stage_updates_000048.json",
        "generic_means_manifest.json",
        "self_transfer.json",
        *[f"evaluation_updates_{value:06d}.json" for value in MILESTONES],
    ]
    for name in names:
        source = Path(run_dir) / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def final_report_text(summary):
    rows = summary["destinations"]
    lines = [
        "# Experiment 2C1 — Final Report",
        "",
        "## Outcome",
        "",
        summary["outcome"],
        "",
        "## Frozen model/data provenance",
        "",
        f"2C0 frozen tag: `{PARENT_TAG}`  ",
        f"2C0 parent commit: `{PARENT_COMMIT}`  ",
        f"2C1 branch: `{BRANCH}`  ",
        f"Implementation commit: `{summary['implementation_git_commit']}`  ",
        f"Results commit: `{summary.get('results_commit') or 'pending'}`  ",
        "Final report commit: `the immutable commit containing this file`  ",
        f"Experiment 1B checkpoint SHA: `{a0.EXPECTED_PARENT_SHA256}`  ",
        f"Canonical validation SHA: `{CANONICAL_SHA}`  ",
        "All four workers used private clones of the same four replay-loader states.",
        "",
        "## Damage and final destination results",
        "",
        "| Destination | Masked | Damage | Real | Shuffled | Generic | Specific gap | Recovery % | Specific share | Real wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for destination in DESTINATIONS:
        row = rows[destination]
        recovery_pct = (
            "N/A"
            if row["recovery_fraction"] is None
            else f"{100 * row['recovery_fraction']:.4f}%"
        )
        specific_share = (
            "N/A"
            if row["specific_share"] is None
            else f"{row['specific_share']:.6f}"
        )
        lines.append(
            f"| {destination}/B{row['block']} | {row['masked_loss']:.10f} | "
            f"{row['damage']:.10f} | {row['real_loss']:.10f} | "
            f"{row['shuffled_loss']:.10f} | {row['generic_loss']:.10f} | "
            f"{row['specific_gap']:.10f} | {recovery_pct} | "
            f"{specific_share} | {row['real_wins']}/20 |"
        )
    lines.extend(["", "## Training trajectories", ""])
    for destination in DESTINATIONS:
        lines.extend([
            f"### {destination} / Block {DESTINATIONS[destination]}",
            "",
            "| Tokens | Real | Shuffled | Specific gap |",
            "|---:|---:|---:|---:|",
        ])
        for trajectory in summary["trajectories"][destination]:
            lines.append(
                f"| {trajectory['tokens']:,} | {trajectory['real']:.10f} | "
                f"{trajectory['shuffled']:.10f} | {trajectory['specific_gap']:.10f} |"
            )
        lines.append("")
    lines.extend([
        "## Router specialization",
        "",
        "| Destination | Gate | Query norm | Entropy | v16 | v17 | v20 | v24 | Feedback RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for destination in DESTINATIONS:
        router = summary["router_stats"][destination]
        weights = router["routing_weights"]
        lines.append(
            f"| {destination} | {router['effective_gate']:.8f} | "
            f"{router['query_norm']:.8f} | {router['routing_entropy']:.8f} | "
            f"{weights['v16']:.8f} | {weights['v17']:.8f} | "
            f"{weights['v20']:.8f} | {weights['v24']:.8f} | "
            f"{router['feedback_rms']:.8f} |"
        )
    lines.extend(["", "## Conditional self-recurrent transfer", ""])
    for destination in DESTINATIONS:
        self_row = summary["self_transfer"][destination]
        if not self_row.get("triggered"):
            lines.append(f"- {destination}: SELF TEST NOT TRIGGERED")
        else:
            lines.append(
                f"- {destination}: teacher gap {rows[destination]['specific_gap']:.10f}; "
                f"self real {self_row['losses']['real']:.10f}; self shuffled "
                f"{self_row['losses']['shuffle']:.10f}; self gap "
                f"{self_row['self_specific_gap']:.10f}; self/teacher recovery "
                f"ratio {self_row['self_teacher_recovery_ratio']:.6f}."
            )
    lines.extend([
        "",
        "## Integrity",
        "",
        *[
            f"- {key}: {'PASS' if value else 'FAIL'}"
            for key, value in summary["integrity"].items()
            if key != "passed"
        ],
        "",
        "## Classification",
        "",
        summary["classification"],
        "",
        "## Key scientific questions",
        "",
        f"Q1. {summary['decisions']['Q1']}  ",
        f"Q2. {summary['decisions']['Q2']}  ",
        f"Q3. {summary['decisions']['Q3']}  ",
        f"Q4. {summary['decisions']['Q4']}  ",
        f"Q5. {summary['decisions']['Q5']}  ",
        f"Q6. {summary['decisions']['Q6']}  ",
        f"Q7. {summary['decisions']['Q7']}",
        "",
        "## Next-experiment recommendations",
        "",
        f"A. {summary['decisions']['A']}  ",
        f"B. {summary['decisions']['B']}  ",
        f"C. {summary['decisions']['C']}  ",
        f"D. {summary['decisions']['D']}  ",
        f"E. {summary['decisions']['E']}  ",
        f"F. {summary['decisions']['F']}",
        "",
        "No writers, reader continuation beyond 25M, multi-destination model, BPTT, iterative loops, auxiliary objectives, or HellaSwag were launched.",
        "",
        "# EXPERIMENT 2C1 COMPLETE",
    ])
    return "\n".join(lines) + "\n"


def aggregate_results(args):
    require_git(clean=False)
    load_config()
    output_dir = Path(args.output_dir)
    destination_rows = {}
    trajectories = {}
    paired_losses = {}
    router_stats = {}
    generic_controls = {}
    self_transfer = {}
    performance = {}
    checkpoint_manifest = {}
    preflights = {}
    hash_sequences = {}
    restart_passes = {}
    generic_shas = set()
    implementation_commits = set()
    all_evaluations_pass = True
    all_checkpoints_pass = True
    all_metrics_integrity = True
    for destination in DESTINATIONS:
        run_dir = destination_run_dir(args.run_root, destination)
        preflight = json.loads((run_dir / "preflight.json").read_text())
        identity = json.loads((run_dir / "run_identity.json").read_text())
        stage20 = json.loads((run_dir / "stage_updates_000020.json").read_text())
        stage48 = json.loads((run_dir / "stage_updates_000048.json").read_text())
        restart = json.loads(
            (run_dir / "restart_audit_updates_000020.json").read_text()
        )
        evaluations = {
            update: json.loads(evaluation_path(run_dir, update).read_text())
            for update in MILESTONES
        }
        final = evaluations[48]
        self_row = json.loads((run_dir / "self_transfer.json").read_text())
        metrics = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if [row["completed_updates"] for row in metrics] != list(range(1, 49)):
            all_metrics_integrity = False
        if any(row.get("global_targets") != GLOBAL_TARGETS for row in metrics):
            all_metrics_integrity = False
        hash_sequences[destination] = [row["global_batch_sha256"] for row in metrics]
        implementation_commits.add(identity["implementation_git_commit"])
        preflights[destination] = preflight
        restart_passes[destination] = restart.get("passed") is True
        all_evaluations_pass &= all(row.get("passed") is True for row in evaluations.values())
        generic_shas.add(final["generic_means_tensor_sha256"])
        block = destination_number(destination)
        destination_rows[destination] = {
            "block": block,
            "masked_loss": final["losses"]["masked"],
            "damage": preflight["evaluation"]["damage"],
            "low_mask_damage": preflight["evaluation"]["low_mask_damage"],
            "real_loss": final["losses"]["real"],
            "shuffled_loss": final["losses"]["shuffle"],
            "generic_loss": final["losses"]["generic"],
            "specific_gap": final["specific_gap"],
            "recovery": final["recovery"],
            "recovery_fraction": final["recovery_fraction"],
            "specific_fraction": final["specific_fraction"],
            "specific_share": final["specific_share"],
            "real_wins": final["paired_real_vs_shuffled"]["real_wins"],
            "relative_to_b1": None,
        }
        trajectories[destination] = [
            {
                "update": update,
                "tokens": update * GLOBAL_TARGETS,
                "real": evaluations[update]["losses"]["real"],
                "shuffled": evaluations[update]["losses"]["shuffle"],
                "specific_gap": evaluations[update]["specific_gap"],
            }
            for update in MILESTONES
        ]
        paired_losses[destination] = {
            str(update): evaluations[update]["paired_real_vs_shuffled"]
            for update in MILESTONES
        }
        router_stats[destination] = {
            **final["reader"],
            **final["router"],
        }
        generic_controls[destination] = {
            "masked": final["losses"]["masked"],
            "real": final["losses"]["real"],
            "shuffled": final["losses"]["shuffle"],
            "generic": final["losses"]["generic"],
            "generic_recovery": final["losses"]["masked"]
            - final["losses"]["generic"],
            "generic_vs_real_delta": final["losses"]["generic"]
            - final["losses"]["real"],
            "means_tensor_sha256": final["generic_means_tensor_sha256"],
        }
        self_transfer[destination] = self_row
        performance[destination] = {
            **stage48["performance"],
            "evaluation_wall_seconds": sum(
                row["elapsed_seconds"] for row in evaluations.values()
            ),
            "self_evaluation_wall_seconds": self_row.get("elapsed_seconds", 0.0),
        }
        checkpoint_manifest[destination] = {}
        for update in MILESTONES:
            verification_path = checkpoint_path(run_dir, update).with_suffix(
                ".pt.verification.json"
            )
            verification = json.loads(verification_path.read_text())
            checkpoint_manifest[destination][str(update)] = verification
            all_checkpoints_pass &= verification.get("passed") is True
        copy_destination_artifacts(
            run_dir,
            output_dir / f"{destination}_block{block}",
        )
    batch_hashes_identical = len({tuple(value) for value in hash_sequences.values()}) == 1
    means_identical = len(generic_shas) == 1
    d1 = destination_rows["D1"]
    d1_regression = {
        "real_difference": d1["real_loss"] - PINNED_D1_REAL,
        "shuffled_difference": d1["shuffled_loss"] - PINNED_D1_SHUFFLED,
        "gap_difference": d1["specific_gap"] - PINNED_D1_GAP,
    }
    d1_regression["passed"] = all(
        abs(value) <= D1_ATOL for value in d1_regression.values()
    )
    for destination in ("D5", "D9", "D12"):
        destination_rows[destination]["relative_to_b1"] = (
            destination_rows[destination]["specific_gap"] - d1["specific_gap"]
        )
    integrity = {
        "all_preflights_passed": all(row.get("passed") is True for row in preflights.values()),
        "trainable_parameters_exactly_1537": all(
            row["integrity"]["trainable_parameters_exactly_1537"]
            for row in preflights.values()
        ),
        "base_and_teacher_gradients_none": all(
            all(not row["base_gradients"] and not row["teacher_gradients"] for row in [
                json.loads(line)
                for line in (
                    destination_run_dir(args.run_root, destination) / "metrics.jsonl"
                ).read_text().splitlines()
                if line.strip()
            ])
            for destination in DESTINATIONS
        ),
        "teacher_eval_no_grad": all(
            row["integrity"]["teacher_eval"] for row in preflights.values()
        ),
        "future_causality_pass": all(
            row["causality"]["future_prefix_logits_bit_exact"]
            for row in preflights.values()
        ),
        "row_isolation_pass": all(
            row["causality"]["unchanged_row_logits_bit_exact"]
            for row in preflights.values()
        ),
        "only_target_block_masked": all(
            row["cache_policy"]["only_destination_cache_absent"]
            for row in preflights.values()
        ),
        "zero_gate_equals_masked": all(
            row["evaluation"]["zero_gate_equals_masked"]
            for row in preflights.values()
        ),
        "global_targets_per_update_524288": all_metrics_integrity,
        "batch_hash_sequence_identical": batch_hashes_identical,
        "forced_fresh_process_restart_after_20": all(restart_passes.values()),
        "checkpoint_strict_reload_pass": all_checkpoints_pass,
        "all_evaluations_finite_and_paired": all_evaluations_pass,
        "generic_means_identical": means_identical,
        "d1_historical_regression_pass": d1_regression["passed"],
        "writers_never_active": True,
        "no_auxiliary_loss": True,
        "no_bptt_or_temporal_gradient": True,
        "hellaswag_not_run": True,
    }
    integrity["passed"] = all(integrity.values())
    classification = classify(destination_rows, integrity["passed"])
    strong_destinations = [
        destination
        for destination in ("D5", "D9", "D12")
        if destination_rows[destination]["specific_gap"] >= 0.050
        and destination_rows[destination]["real_wins"] >= 18
        and destination_rows[destination]["recovery"] > 0
        and destination_rows[destination]["real_loss"] + 0.010
        <= destination_rows[destination]["generic_loss"]
    ]
    best = max(
        DESTINATIONS,
        key=lambda key: destination_rows[key]["specific_gap"],
    )
    generic_deltas = [
        generic_controls[key]["generic_vs_real_delta"] for key in DESTINATIONS
    ]
    routing_vectors = [
        tuple(router_stats[key]["routing_weights"].values()) for key in DESTINATIONS
    ]
    decisions = {
        "Q1": comparison_answer(d1["specific_gap"], destination_rows["D5"]["specific_gap"]),
        "Q2": comparison_answer(
            destination_rows["D5"]["specific_gap"], destination_rows["D9"]["specific_gap"]
        ),
        "Q3": comparison_answer(
            destination_rows["D9"]["specific_gap"], destination_rows["D12"]["specific_gap"]
        ),
        "Q4": (
            "YES; generic-vs-real deltas decrease overall with depth."
            if linear_slope(list(DESTINATIONS.values()), generic_deltas) < 0
            else "NO; generic-template utility does not show an overall depth decrease."
        ),
        "Q5": (
            "YES; final source-weight vectors differ across destinations."
            if len(set(routing_vectors)) > 1
            else "NO; final source-weight vectors are identical."
        ),
        "Q6": (
            f"{best} / Block {DESTINATIONS[best]} has the largest direct specific gap; "
            + ("it passes strong support." if best in strong_destinations else "it does not pass the frozen strong-support rule.")
        ),
        "Q7": f"{best} / Block {DESTINATIONS[best]} is the leading iterative-loop candidate by direct specific gap.",
        "A": (
            f"YES, only at {', '.join(strong_destinations)}; do not launch here."
            if strong_destinations
            else "NO destination passed the frozen strong direct-signal rule."
        ),
        "B": (
            f"YES at {best}, but only as a separately preregistered experiment."
            if best in strong_destinations
            else "NO; no destination earned alternating reader→writer optimization."
        ),
        "C": "Retain the generic branch for Block 1; test it elsewhere only in a separate controlled protocol.",
        "D": "NO for this experiment; a future multi-destination model requires separate approval and preregistration.",
        "E": "YES; keep temporal credit zero for direct readers and limit it to one token only after writers are introduced.",
        "F": (
            f"YES at {best}, conditionally and only in a new experiment."
            if best in strong_destinations
            else "NO; direct sequence specificity is not strong enough."
        ),
    }
    outcome = (
        f"All four independent destination workers completed the frozen 25M-token protocol. "
        f"The largest final sequence-specific gap was {destination_rows[best]['specific_gap']:.10f} "
        f"at {best}/Block {DESTINATIONS[best]}. The preregistered classification is "
        f"{classification}."
    )
    implementation_commit = next(iter(implementation_commits)) if len(implementation_commits) == 1 else "MISMATCH"
    summary = {
        "experiment": "2C1",
        "implementation_git_commit": implementation_commit,
        "results_commit": args.results_commit,
        "classification": classification,
        "outcome": outcome,
        "destinations": destination_rows,
        "trajectories": trajectories,
        "paired_losses": paired_losses,
        "router_stats": router_stats,
        "generic_controls": generic_controls,
        "self_transfer": self_transfer,
        "performance": performance,
        "checkpoint_manifest": checkpoint_manifest,
        "d1_regression": d1_regression,
        "strong_destinations": strong_destinations,
        "decisions": decisions,
        "integrity": integrity,
        "optimizer_updates": 48 * 4,
        "processed_reader_tokens_per_destination": 48 * GLOBAL_TARGETS,
        "hellaswag_run": False,
        "passed": integrity["passed"],
    }
    audit = {
        "experiment": "2C1",
        "classification": classification,
        "hard_invariants": integrity,
        "d1_regression": d1_regression,
        "global_batch_hashes": hash_sequences,
        "checkpoint_manifest": checkpoint_manifest,
        "passed": integrity["passed"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    durable_json(output_dir / "destination_damage.json", {
        key: {
            "full": PINNED_FULL,
            "masked": value["masked_loss"],
            "damage": value["damage"],
            "low_mask_damage": value["low_mask_damage"],
        }
        for key, value in destination_rows.items()
    })
    durable_json(output_dir / "destination_trajectories.json", trajectories)
    durable_json(output_dir / "destination_paired_losses.json", paired_losses)
    durable_json(output_dir / "destination_router_stats.json", router_stats)
    durable_json(output_dir / "generic_template_controls.json", generic_controls)
    durable_json(output_dir / "self_transfer.json", self_transfer)
    durable_json(output_dir / "performance.json", performance)
    durable_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    durable_json(output_dir / "result_summary.json", summary)
    durable_json(output_dir / "FINAL_AUDIT.json", audit)
    durable_text(output_dir / "EXPERIMENT_2C1_FINAL_REPORT.md", final_report_text(summary))
    if not integrity["passed"]:
        raise SystemExit(f"2C1 final audit failed: {integrity}")
    print(f"EXPERIMENT_2C1_AGGREGATE_PASS classification={classification}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--destination", choices=DESTINATIONS, required=True)
    preflight.add_argument("--parent-checkpoint", required=True)
    preflight.add_argument("--run-dir", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--destination", choices=DESTINATIONS, required=True)
    train.add_argument("--parent-checkpoint", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--target-update", type=int, required=True)
    train.add_argument("--resume")
    train.add_argument("--allow-optimizer-steps", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-root", required=True)
    aggregate.add_argument("--output-dir", required=True)
    aggregate.add_argument("--results-commit")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(a0.SEED)
    np.random.seed(a0.SEED)
    torch.manual_seed(a0.SEED)
    if args.command == "aggregate":
        report = aggregate_results(args)
    else:
        a0.require_cuda()
        torch.cuda.manual_seed(a0.SEED)
        report = run_preflight(args) if args.command == "preflight" else run_training(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
