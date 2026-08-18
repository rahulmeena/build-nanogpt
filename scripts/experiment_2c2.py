#!/usr/bin/env python3
"""Experiment 2C2 cumulative low-KV matched-feedback workers and audit."""

import argparse
import copy
import gc
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
import experiment_2c1 as c1  # noqa: E402


BRANCH = "experiment-2c2-cumulative-low-kv-matched-feedback"
PARENT_TAG = "experiment-2c1-destination-depth-final"
PARENT_COMMIT = "e4a5eec76181db0581d486e0f5724f196c22db64"
PARENT_IMPLEMENTATION = "cbf847f8ad43d59f38cd9cf43008562b3c64fb13"
PARENT_RESULTS = "4328d3ed6cdffa4d5bbed96ba58e3c06302333a1"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2c2_cumulative_low_kv_matched_feedback.json"
PROTOCOL = "exp2c2_cumulative_low_kv_matched_feedback_v1"
CHECKPOINT_SCHEMA = "exp2c2_cumulative_matched_readers_v1"
CONFIGURATIONS = {
    "C1": (0,),
    "C2": (0, 1),
    "C3": (0, 1, 2),
    "C4": (0, 1, 2, 3),
}
RUN_NAMES = {
    "C1": "C1_B1",
    "C2": "C2_B1_B2",
    "C3": "C3_B1_B2_B3",
    "C4": "C4_B1_B2_B3_B4",
}
GPU_MAPPING = {"C1": 0, "C2": 1, "C3": 2, "C4": 3}
MILESTONES = (10, 20, 29, 48)
TARGET_UPDATE = 48
RESTART_UPDATE = 20
GLOBAL_TARGETS = 524_288
CANONICAL_SHA = a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256
BASE_MODEL_SHA = "1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd"
PINNED_FULL = 4.0786544085
SOURCE_DEPTHS = tuple(a0.SOURCE_DEPTHS)
PARAMETERS_PER_READER = 1_537
T19_975 = 2.093024054408263
C1_EXPECTED = {
    10: {"real": 5.9533051014, "shuffle": 5.9617962360, "gap": 0.0084911346},
    20: {"real": 5.9229358435, "shuffle": 5.9391546011, "gap": 0.0162187576},
    29: {"real": 5.8944202185, "shuffle": 5.9179884195, "gap": 0.0235682011},
    48: {"real": 5.8353391409, "shuffle": 5.8765912533, "gap": 0.0412521124},
}
C1_ATOL = 5e-6


git_output = c1.git_output
file_sha256 = c1.file_sha256
tensor_bytes = c1.tensor_bytes
durable_json = c1.durable_json
durable_text = c1.durable_text
append_jsonl = c1.append_jsonl
atomic_torch_save = c1.atomic_torch_save
move_optimizer_to_cuda = c1.move_optimizer_to_cuda
generic_bank = c1.generic_bank


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2C2 requires branch {BRANCH}")
    if git_output("rev-parse", f"{PARENT_TAG}^{{}}") != PARENT_COMMIT:
        raise SystemExit("frozen Experiment 2C1 tag target mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2C2 execution requires a clean worktree")


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        "protocol": PROTOCOL,
        "configurations": {
            key: [block + 1 for block in blocks]
            for key, blocks in CONFIGURATIONS.items()
        },
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
        raise SystemExit(f"2C2 config mismatch: {mismatches}")
    return config


def blocks_for(configuration):
    try:
        return CONFIGURATIONS[configuration]
    except KeyError as exc:
        raise SystemExit(f"unknown 2C2 configuration: {configuration}") from exc


def run_dir_for(run_root, configuration):
    return Path(run_root) / RUN_NAMES[configuration]


def paired_statistics(real, shuffled):
    if len(real) != 20 or len(shuffled) != 20:
        raise ValueError("canonical paired statistics require exactly 20 batches")
    gaps = [float(right) - float(left) for left, right in zip(real, shuffled)]
    if not all(math.isfinite(value) for value in [*real, *shuffled, *gaps]):
        raise ValueError("paired losses must be finite")
    sample_std = statistics.stdev(gaps)
    standard_error = sample_std / math.sqrt(len(gaps))
    mean = statistics.fmean(gaps)
    report = {
        "real_wins": sum(value > 0 for value in gaps),
        "shuffled_wins": sum(value < 0 for value in gaps),
        "ties": sum(value == 0 for value in gaps),
        "mean_gap": mean,
        "median_gap": statistics.median(gaps),
        "sample_std": sample_std,
        "minimum": min(gaps),
        "maximum": max(gaps),
        "standard_error": standard_error,
        "ci95_lower": mean - T19_975 * standard_error,
        "ci95_upper": mean + T19_975 * standard_error,
        "gaps": gaps,
    }
    return report


def model_config(symbols, destinations=()):
    return symbols["GPTConfig"](
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        residual_mode="full_attnres",
        enable_topdown_feedback=bool(destinations),
        topdown_feedback_destinations=tuple(destinations),
    )


def readers(student):
    module = student.transformer.topdown_attnres_by_destination
    expected = {str(block) for block in student.config.topdown_feedback_destinations}
    if set(module) != expected:
        raise SystemExit("reader-destination module mapping mismatch")
    return module


def reader_parameter_map(student):
    result = {}
    for destination, reader in readers(student).items():
        named = dict(reader.named_parameters())
        if set(named) != {"query", "norm.weight", "gate"}:
            raise SystemExit(f"unexpected reader parameters at B{int(destination) + 1}")
        if sum(value.numel() for value in named.values()) != PARAMETERS_PER_READER:
            raise SystemExit("per-reader parameter count mismatch")
        result[int(destination)] = named
    return result


def reader_state(student):
    return {
        key: value.detach().cpu().clone()
        for key, value in readers(student).state_dict().items()
    }


def reader_state_sha(student):
    digest = hashlib.sha256()
    for name, value in sorted(reader_state(student).items()):
        digest.update(name.encode())
        digest.update(tensor_bytes(value))
    return digest.hexdigest()


def reader_metrics(student, routing_stats=None):
    rows = {}
    for block, named in reader_parameter_map(student).items():
        reader = readers(student)[str(block)]
        row = {
            "destination_block": block + 1,
            "gate": reader.gate.detach().float().item(),
            "effective_gate": reader.gate.detach().float().tanh().item(),
            "query_norm": reader.query.detach().float().norm().item(),
            "rmsnorm_displacement": (
                reader.norm.weight.detach().float() - 1.0
            ).norm().item(),
            "parameter_count": sum(value.numel() for value in named.values()),
        }
        if routing_stats and block in routing_stats:
            row.update(routing_stats[block])
        rows[f"B{block + 1}"] = row
    return rows


def state_hash(model, exclude_readers=False):
    digest = hashlib.sha256()
    prefix = "transformer.topdown_attnres_by_destination."
    for name, value in sorted(model.state_dict().items()):
        if exclude_readers and name.startswith(prefix):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_hashes(student, teacher):
    return {
        "student_base": state_hash(student, exclude_readers=True),
        "teacher": state_hash(teacher),
    }


def validate_frozen_hashes(student, teacher):
    hashes = frozen_hashes(student, teacher)
    expected = {"student_base": BASE_MODEL_SHA, "teacher": BASE_MODEL_SHA}
    if hashes != expected:
        raise SystemExit(f"frozen model hash mismatch: {hashes}")
    return hashes


def make_runtime(parent_checkpoint, configuration, include_optimizer):
    path = Path(parent_checkpoint).resolve()
    digest = file_sha256(path)
    if digest != a0.EXPECTED_PARENT_SHA256:
        raise SystemExit(f"parent checkpoint SHA mismatch: {digest}")
    checkpoint = a0.torch_load(path, mmap=True)
    a0.validate_parent_payload(checkpoint)
    symbols = a0.support.load_training_symbols()
    torch.manual_seed(a0.SEED)
    torch.cuda.manual_seed(a0.SEED)
    blocks = blocks_for(configuration)
    student = symbols["GPT"](model_config(symbols, blocks))
    student.load_experiment1_full_attnres_state(checkpoint["model"])
    student.freeze_for_topdown_training()
    teacher = symbols["GPT"](model_config(symbols))
    teacher.load_state_dict(checkpoint["model"], strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    parent_aux = {
        "dataloader_states": checkpoint["dataloader_states"],
        "rng_states": checkpoint["rng_states"],
        "metadata": checkpoint["metadata"],
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
    }
    del checkpoint
    gc.collect()
    student.cuda().eval()
    teacher.cuda().eval()
    validate_frozen_hashes(student, teacher)
    expected_trainable = PARAMETERS_PER_READER * len(blocks)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    if trainable != expected_trainable:
        raise SystemExit(f"trainable parameter mismatch: {trainable}")
    initialization = {}
    for block, named in reader_parameter_map(student).items():
        reader = readers(student)[str(block)]
        initialization[f"B{block + 1}"] = {
            "query_exact_zero": reader.query.count_nonzero().item() == 0,
            "rmsnorm_exact_one": torch.equal(
                reader.norm.weight.detach(), torch.ones_like(reader.norm.weight)
            ),
            "gate_exact_zero": reader.gate.detach().item() == 0.0,
            "trainable_parameters": sum(value.numel() for value in named.values()),
        }
    initialization["passed"] = all(
        all(value for key, value in row.items() if key != "trainable_parameters")
        and row["trainable_parameters"] == PARAMETERS_PER_READER
        for key, row in initialization.items()
        if key != "passed"
    )
    if not initialization["passed"]:
        raise SystemExit(f"fresh reader initialization mismatch: {initialization}")
    optimizer = a0.feedback_optimizer(student) if include_optimizer else None
    loaders = a0.make_replay_loaders(
        symbols, copy.deepcopy(parent_aux["dataloader_states"])
    )
    return symbols, teacher, student, optimizer, loaders, parent_aux, initialization


def cumulative_forward(
    student,
    x,
    y=None,
    control="real",
    memory=None,
    permutation=None,
    active_blocks=None,
):
    if control == "full_context":
        return student(x, y, mode="full_context")
    if control == "masked":
        return student(x, y, mode="masked_cumulative_no_feedback")
    if control in {"real", "generic", "zero"}:
        return student(
            x,
            y,
            mode="masked_cumulative_topdown_teacher",
            feedback_sources=memory,
            feedback_gate_override=0.0 if control == "zero" else None,
            feedback_active_destination_blocks=active_blocks,
        )
    if control == "shuffle":
        return student(
            x,
            y,
            mode="masked_cumulative_shuffled_feedback",
            feedback_sources=memory,
            feedback_permutation=permutation,
            feedback_active_destination_blocks=active_blocks,
        )
    raise ValueError(control)


def validation_loader(symbols):
    return symbols["DataLoaderLite"](
        B=a0.VALIDATION_B,
        T=a0.T,
        process_rank=0,
        num_processes=1,
        split="val",
    )


@torch.no_grad()
def evaluate_teacher_controls(
    student,
    teacher,
    symbols,
    configuration,
    completed_updates,
    include_full=False,
    generic_means=None,
    extended=False,
):
    student.eval()
    teacher.eval()
    blocks = blocks_for(configuration)
    controls = ["masked", "real", "shuffle", "zero"]
    if include_full:
        controls.insert(0, "full_context")
    if generic_means is not None:
        controls.append("generic")
    activation_sets = {}
    if extended:
        for count in range(len(blocks) + 1):
            activation_sets[f"prefix_{count}"] = tuple(blocks[:count])
        for block in blocks:
            activation_sets[f"minus_B{block + 1}"] = tuple(
                value for value in blocks if value != block
            )
    losses = {name: [] for name in controls}
    activation_losses = {name: [] for name in activation_sets}
    validation_hash = hashlib.sha256()
    permutation = symbols["fixed_derangement"](a0.VALIDATION_B, "cuda")
    routing_accumulator = {
        block: {
            "weights": torch.zeros(4, dtype=torch.float64),
            "entropy": 0.0,
            "topdown_rms": 0.0,
            "feedback_rms": 0.0,
        }
        for block in blocks
    }
    zero_equals_masked = True
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    loader = validation_loader(symbols)
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        validation_hash.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        memory = a0.teacher_memory(teacher, x, symbols)
        if include_full:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = cumulative_forward(student, x, y, "full_context")
            losses["full_context"].append(loss.detach().double().item())
            del logits, loss
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, masked_loss = cumulative_forward(student, x, y, "masked")
        losses["masked"].append(masked_loss.detach().double().item())
        del logits
        student.set_topdown_instrumentation(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, real_loss = cumulative_forward(
                student, x, y, "real", memory=memory
            )
        del logits
        stats = student.get_topdown_stats()
        student.set_topdown_instrumentation(False)
        for block in blocks:
            row = stats[block]
            routing_accumulator[block]["weights"] += torch.tensor(
                row["mean_weights"], dtype=torch.float64
            )
            routing_accumulator[block]["entropy"] += row["mean_entropy"]
            routing_accumulator[block]["topdown_rms"] += row["topdown_rms"]
            routing_accumulator[block]["feedback_rms"] += row["feedback_rms"]
        losses["real"].append(real_loss.detach().double().item())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, shuffled_loss = cumulative_forward(
                student, x, y, "shuffle", memory=memory, permutation=permutation
            )
        del logits
        losses["shuffle"].append(shuffled_loss.detach().double().item())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, zero_loss = cumulative_forward(
                student, x, y, "zero", memory=memory
            )
        del logits
        losses["zero"].append(zero_loss.detach().double().item())
        zero_equals_masked &= torch.equal(masked_loss, zero_loss)
        if generic_means is not None:
            template = generic_bank(
                generic_means, x.size(0), x.size(1), memory.dtype
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, generic_loss = cumulative_forward(
                    student, x, y, "generic", memory=template
                )
            del logits
            losses["generic"].append(generic_loss.detach().double().item())
            del generic_loss, template
        if extended:
            for name, active in activation_sets.items():
                if not active:
                    value = masked_loss
                elif active == blocks:
                    value = real_loss
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, value = cumulative_forward(
                            student,
                            x,
                            y,
                            "real",
                            memory=memory,
                            active_blocks=active,
                        )
                    del logits
                activation_losses[name].append(value.detach().double().item())
        del x, y, memory, masked_loss, real_loss, shuffled_loss, zero_loss
        print(
            f"2C2 {configuration} eval@{completed_updates} batch "
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
    routing = {}
    for block, accumulator in routing_accumulator.items():
        routing[block] = {
            "routing_weights": {
                f"v{depth}": value
                for depth, value in zip(
                    SOURCE_DEPTHS,
                    (accumulator["weights"] / a0.VALIDATION_BATCHES).tolist(),
                )
            },
            "routing_entropy": accumulator["entropy"] / a0.VALIDATION_BATCHES,
            "topdown_rms": accumulator["topdown_rms"] / a0.VALIDATION_BATCHES,
            "feedback_rms": accumulator["feedback_rms"] / a0.VALIDATION_BATCHES,
        }
    result = {
        "experiment": "2C2",
        "configuration": configuration,
        "masked_blocks": [block + 1 for block in blocks],
        "completed_updates": completed_updates,
        "processed_reader_tokens": completed_updates * GLOBAL_TARGETS,
        "canonical_validation_sha256": digest,
        "losses": means,
        "per_batch_losses": losses,
        "paired_real_vs_shuffled": paired,
        "damage": damage,
        "recovery": recovery,
        "recovery_fraction": recovery / damage if damage > 0 else None,
        "specific_gap": paired["mean_gap"],
        "specific_fraction": paired["mean_gap"] / damage if damage > 0 else None,
        "specific_share": paired["mean_gap"] / recovery if recovery > 0 else None,
        "zero_gate_equals_masked": zero_equals_masked,
        "reader": reader_metrics(student, routing),
        "activation_sets": {
            name: [block + 1 for block in active]
            for name, active in activation_sets.items()
        },
        "activation_losses": {
            name: {
                "mean": statistics.fmean(values),
                "per_batch": values,
            }
            for name, values in activation_losses.items()
        },
        "generic_means_tensor_sha256": (
            None
            if generic_means is None
            else c1.tensor_sha256("generic_teacher_source_means", generic_means)
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }
    if not all(math.isfinite(value) for value in means.values()):
        raise SystemExit("non-finite canonical evaluation")
    return result


def cache_policy(student, configuration):
    blocks = blocks_for(configuration)
    state = student.init_recurrent_state(
        2,
        "masked_l1_no_feedback",
        device="cuda",
        dtype=torch.float32,
        mask_depth=len(blocks),
    )
    report = {
        "masked_blocks": [block + 1 for block in blocks],
        "only_intended_caches_absent": all(
            (cache is None) == (index in blocks)
            for index, cache in enumerate(state.kv_caches)
        ),
        "later_caches_empty_and_present": all(
            state.kv_caches[index] is not None
            and state.kv_caches[index].length == 0
            for index in range(len(blocks), student.config.n_layer)
        ),
        "fresh_memory_zero": state.feedback_memory.count_nonzero().item() == 0,
    }
    report["passed"] = all(
        value for key, value in report.items() if key != "masked_blocks"
    )
    return report


@torch.no_grad()
def causal_mapping_preflight(student, teacher, symbols, configuration):
    blocks = blocks_for(configuration)
    loader = symbols["DataLoaderLite"](
        B=2, T=32, process_rank=0, num_processes=1, split="val"
    )
    first, _ = loader.next_batch()
    first = first.cuda()
    future = first.clone()
    future[:, 16:] = (future[:, 16:] + 17) % student.config.vocab_size
    row_changed = first.clone()
    row_changed[1] = (row_changed[1] + 29) % student.config.vocab_size
    saved = {
        block: readers(student)[str(block)].gate.detach().clone()
        for block in blocks
    }
    for block in blocks:
        readers(student)[str(block)].gate.fill_(math.atanh(0.25))
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
        logits_first, _ = cumulative_forward(
            student, first, memory=memory_first
        )
        logits_future, _ = cumulative_forward(
            student, future, memory=memory_future
        )
        logits_row, _ = cumulative_forward(
            student, row_changed, memory=memory_row
        )
        masked_logits, _ = cumulative_forward(student, first, control="masked")
        zero_logits, _ = cumulative_forward(
            student, first, control="zero", memory=memory_first
        )
        perturbation = {}
        for active_block in blocks:
            calls = {block: 0 for block in blocks}
            handles = []
            for block in blocks:
                def count_call(_module, _inputs, block=block):
                    calls[block] += 1
                handles.append(readers(student)[str(block)].register_forward_pre_hook(count_call))
            active_logits, _ = cumulative_forward(
                student,
                first,
                memory=memory_first,
                active_blocks=(active_block,),
            )
            for handle in handles:
                handle.remove()
            perturbation[f"B{active_block + 1}"] = {
                "changes_logits": not torch.equal(active_logits, masked_logits),
                "only_target_reader_called": calls[active_block] == 1
                and all(calls[block] == 0 for block in blocks if block != active_block),
                "reader_calls": {f"B{block + 1}": count for block, count in calls.items()},
            }
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
            "reader_destination_perturbations": perturbation,
            "reader_destination_mapping_exact": all(
                row["changes_logits"] and row["only_target_reader_called"]
                for row in perturbation.values()
            ),
        }
        report["passed"] = all(
            value
            for key, value in report.items()
            if key != "reader_destination_perturbations"
        )
        return report
    finally:
        for block, gate in saved.items():
            readers(student)[str(block)].gate.copy_(gate)


def run_preflight(args):
    require_git(clean=True)
    config = load_config()
    expected_device = str(GPU_MAPPING[args.configuration])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_device:
        raise SystemExit(
            f"{args.configuration} requires CUDA_VISIBLE_DEVICES={expected_device}, got {visible}"
        )
    symbols, teacher, student, _, _, parent_aux, initialization = make_runtime(
        args.parent_checkpoint, args.configuration, include_optimizer=False
    )
    before = {
        **frozen_hashes(student, teacher),
        "reader": reader_state_sha(student),
    }
    cache = cache_policy(student, args.configuration)
    causality = causal_mapping_preflight(
        student, teacher, symbols, args.configuration
    )
    evaluation = evaluate_teacher_controls(
        student,
        teacher,
        symbols,
        args.configuration,
        completed_updates=0,
        include_full=True,
    )
    after = {
        **frozen_hashes(student, teacher),
        "reader": reader_state_sha(student),
    }
    trainable_names = [
        name for name, value in student.named_parameters() if value.requires_grad
    ]
    prefix = "transformer.topdown_attnres_by_destination."
    expected_count = PARAMETERS_PER_READER * len(blocks_for(args.configuration))
    integrity = {
        "initialization": initialization["passed"],
        "cache_policy": cache["passed"],
        "causality_row_mapping": causality["passed"],
        "full_context_reference": abs(
            evaluation["losses"]["full_context"] - PINNED_FULL
        ) <= C1_ATOL,
        "zero_gate_equivalence": evaluation["zero_gate_equals_masked"],
        "trainable_parameter_count_exact": sum(
            value.numel() for value in student.parameters() if value.requires_grad
        ) == expected_count,
        "trainable_names_exact": all(name.startswith(prefix) for name in trainable_names)
        and len(trainable_names) == 3 * len(blocks_for(args.configuration)),
        "base_gradients_none": all(
            value.grad is None
            for name, value in student.named_parameters()
            if not name.startswith(prefix)
        ),
        "teacher_gradients_none": all(value.grad is None for value in teacher.parameters()),
        "teacher_eval": not teacher.training,
        "frozen_hashes_unchanged": before == after,
        "all_losses_finite": all(
            math.isfinite(value) for value in evaluation["losses"].values()
        ),
        "writers_absent": "memory_writers" not in student.transformer,
        "hellaswag_not_run": True,
    }
    integrity["passed"] = all(integrity.values())
    report = {
        "experiment": "2C2",
        "stage": "preflight",
        "configuration": args.configuration,
        "masked_blocks": [block + 1 for block in blocks_for(args.configuration)],
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "config": config,
        "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "parent_checkpoint_sha256": parent_aux["checkpoint_sha256"],
        "optimizer_constructions": 0,
        "hardware": {
            "expected_physical_gpu": GPU_MAPPING[args.configuration],
            "cuda_visible_devices": visible,
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
        raise SystemExit(f"2C2 preflight failed: {report}")
    run_dir = run_dir_for(args.run_root, args.configuration)
    run_dir.mkdir(parents=True, exist_ok=True)
    durable_json(run_dir / "preflight.json", report)
    print(
        f"EXPERIMENT_2C2_PREFLIGHT_PASS configuration={args.configuration}",
        flush=True,
    )
    return report


def run_smoke(args):
    require_git(clean=True)
    load_config()
    expected_device = str(GPU_MAPPING[args.configuration])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_device:
        raise SystemExit(f"smoke GPU mapping mismatch for {args.configuration}")
    symbols, teacher, student, optimizer, _, _, initialization = make_runtime(
        args.parent_checkpoint, args.configuration, include_optimizer=True
    )
    blocks = blocks_for(args.configuration)
    loader = symbols["DataLoaderLite"](
        B=2, T=64, process_rank=0, num_processes=1, split="train"
    )
    initial_hashes = frozen_hashes(student, teacher)
    updates = []
    prefix = "transformer.topdown_attnres_by_destination."
    for update in range(3):
        student.train()
        teacher.eval()
        optimizer.zero_grad(set_to_none=True)
        x_cpu, y_cpu = loader.next_batch()
        x = x_cpu.cuda()
        y = y_cpu.cuda()
        memory = a0.teacher_memory(teacher, x, symbols)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = cumulative_forward(student, x, y, "real", memory=memory)
        loss.backward()
        gradients = {}
        for block, named in reader_parameter_map(student).items():
            gradients[f"B{block + 1}"] = {
                name: {
                    "present": parameter.grad is not None,
                    "finite": parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all()),
                    "nonzero": parameter.grad is not None
                    and bool(parameter.grad.count_nonzero().item()),
                    "norm": None
                    if parameter.grad is None
                    else parameter.grad.detach().float().norm().item(),
                }
                for name, parameter in named.items()
            }
        frozen_gradients = [
            name
            for name, parameter in student.named_parameters()
            if not name.startswith(prefix) and parameter.grad is not None
        ]
        if frozen_gradients or any(
            not row["present"] or not row["finite"]
            for reader_row in gradients.values()
            for row in reader_row.values()
        ):
            raise SystemExit("2C2 smoke gradient boundary failure")
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        updates.append(
            {
                "update": update + 1,
                "loss": loss.detach().float().item(),
                "gradients": gradients,
                "readers": reader_metrics(student),
                "frozen_gradients": frozen_gradients,
            }
        )
    gate_learning = all(
        updates[0]["gradients"][f"B{block + 1}"]["gate"]["nonzero"]
        and abs(updates[0]["readers"][f"B{block + 1}"]["gate"]) > 0
        for block in blocks
    )
    query_learning = all(
        updates[1]["gradients"][f"B{block + 1}"]["query"]["nonzero"]
        for block in blocks
    )
    norm_learning = all(
        updates[2]["gradients"][f"B{block + 1}"]["norm.weight"]["nonzero"]
        for block in blocks
    )
    report = {
        "experiment": "2C2",
        "stage": "disposable_smoke",
        "configuration": args.configuration,
        "geometry": {"B": 2, "T": 64, "updates": 3},
        "initialization": initialization,
        "updates": updates,
        "integrity": {
            "finite_losses": all(math.isfinite(row["loss"]) for row in updates),
            "all_gates_begin_learning": gate_learning,
            "query_gradients_after_gate_moves": query_learning,
            "rmsnorm_gradients_after_query_moves": norm_learning,
            "frozen_hashes_unchanged": frozen_hashes(student, teacher) == initial_hashes,
            "teacher_frozen_eval": not teacher.training
            and all(parameter.grad is None for parameter in teacher.parameters()),
            "trainable_parameter_count_exact": sum(
                parameter.numel()
                for parameter in student.parameters()
                if parameter.requires_grad
            ) == PARAMETERS_PER_READER * len(blocks),
            "mask_policy": cache_policy(student, args.configuration)["passed"],
            "writers_absent": "memory_writers" not in student.transformer,
        },
        "states_discarded": True,
    }
    report["passed"] = all(report["integrity"].values())
    if not report["passed"]:
        raise SystemExit(f"2C2 disposable smoke failed: {report}")
    run_dir = run_dir_for(args.run_root, args.configuration)
    durable_json(run_dir / "disposable_smoke.json", report)
    print(
        f"EXPERIMENT_2C2_SMOKE_PASS configuration={args.configuration}",
        flush=True,
    )
    return report


def checkpoint_path(run_dir, completed_updates):
    return Path(run_dir) / "checkpoints" / f"checkpoint_updates_{completed_updates:06d}.pt"


def evaluation_path(run_dir, completed_updates):
    return Path(run_dir) / f"evaluation_updates_{completed_updates:06d}.json"


def optimizer_integrity(optimizer, completed_updates, reader_count):
    state = optimizer.state_dict()
    groups = state.get("param_groups", [])
    if len(groups) != 1:
        raise SystemExit("2C2 requires one reader optimizer group")
    steps = sorted(
        int(values["step"].detach().cpu().item())
        for values in state.get("state", {}).values()
    )
    expected_steps = [] if completed_updates == 0 else [completed_updates] * (3 * reader_count)
    finite = all(
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or bool(torch.isfinite(value).all())
        for values in state.get("state", {}).values()
        for value in values.values()
    )
    report = {
        "state_entries": len(state.get("state", {})),
        "steps": steps,
        "expected_steps": expected_steps,
        "finite": finite,
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


def run_identity(configuration, parent_aux, config):
    blocks = blocks_for(configuration)
    return {
        "experiment": "2C2",
        "protocol": PROTOCOL,
        "configuration": configuration,
        "masked_blocks": [block + 1 for block in blocks],
        "active_destinations": [block + 1 for block in blocks],
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "parent_tag": PARENT_TAG,
        "parent_commit": PARENT_COMMIT,
        "parent_checkpoint": parent_aux["checkpoint"],
        "parent_checkpoint_sha256": parent_aux["checkpoint_sha256"],
        "base_model_sha256": BASE_MODEL_SHA,
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_depths": list(SOURCE_DEPTHS),
        "mask_semantics": "cumulative consecutive low-block historical KV removed",
        "teacher": "frozen full context, eval/no_grad, one-token shifted raw sources",
        "training_objective": "next-token cross entropy only",
        "config": config,
        "writers_active": False,
        "auxiliary_objective": False,
        "bptt": False,
        "hellaswag_run": False,
    }


def save_checkpoint(
    run_dir,
    configuration,
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
        raise SystemExit(f"refusing to overwrite 2C2 checkpoint: {path}")
    next_hash = a0.next_update_hash(loaders, symbols, replay=True)
    rng = a0.capture_rng_state()
    blocks = blocks_for(configuration)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "experiment": "2C2",
        "configuration": configuration,
        "masked_blocks": [block + 1 for block in blocks],
        "active_destinations": [block + 1 for block in blocks],
        "base_model_sha256": BASE_MODEL_SHA,
        "parent_checkpoint_sha256": a0.EXPECTED_PARENT_SHA256,
        "reader_state": reader_state(student),
        "reader_state_sha256": reader_state_sha(student),
        "optimizer": optimizer.state_dict(),
        "optimizer_integrity": optimizer_integrity(
            optimizer, completed_updates, len(blocks)
        ),
        "reader_update_count": completed_updates,
        "completed_updates": completed_updates,
        "processed_targets": completed_updates * GLOBAL_TARGETS,
        "dataloader_states": a0.snapshot_loaders(loaders),
        "rng_state": rng,
        "schedule_position": a0.EXPECTED_PARENT_UPDATES + completed_updates,
        "next_global_batch_sha256": next_hash,
        "source_depths": list(SOURCE_DEPTHS),
        "mask_semantics": "cumulative consecutive low-block historical KV removed",
        "identity": identity,
        "writer_pid": os.getpid(),
    }
    digest = atomic_torch_save(path, payload)
    reopened = a0.torch_load(path, mmap=True)
    strict = {
        "schema": reopened.get("schema") == CHECKPOINT_SCHEMA,
        "configuration": reopened.get("configuration") == configuration,
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
        raise SystemExit(f"2C2 checkpoint strict reopen failed: {strict}")
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
    configuration,
    student,
    optimizer,
    loaders,
    symbols,
    identity,
):
    path = Path(path).resolve()
    verification_path = path.with_suffix(path.suffix + ".verification.json")
    if not verification_path.is_file():
        raise SystemExit("2C2 resume checkpoint verification is missing")
    verification = json.loads(verification_path.read_text())
    digest = file_sha256(path)
    if digest != verification.get("sha256"):
        raise SystemExit("2C2 resume checkpoint SHA mismatch")
    checkpoint = a0.torch_load(path, mmap=True)
    blocks = blocks_for(configuration)
    required = {
        "schema": checkpoint.get("schema") == CHECKPOINT_SCHEMA,
        "configuration": checkpoint.get("configuration") == configuration,
        "masked_blocks": checkpoint.get("masked_blocks")
        == [block + 1 for block in blocks],
        "parent": checkpoint.get("parent_checkpoint_sha256")
        == a0.EXPECTED_PARENT_SHA256,
        "base": checkpoint.get("base_model_sha256") == BASE_MODEL_SHA,
        "identity": checkpoint.get("identity") == identity,
        "source_depths": checkpoint.get("source_depths") == list(SOURCE_DEPTHS),
    }
    if not all(required.values()):
        raise SystemExit(f"2C2 resume lineage mismatch: {required}")
    readers(student).load_state_dict(checkpoint["reader_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_to_cuda(optimizer)
    a0.restore_loader_group(
        loaders, checkpoint["dataloader_states"], symbols, replay=True
    )
    a0.restore_rng_state(checkpoint["rng_state"])
    completed = checkpoint["completed_updates"]
    audit = {
        "checkpoint": str(path),
        "sha256": digest,
        "lineage": required,
        "reader_exact_reload": reader_state_sha(student)
        == checkpoint["reader_state_sha256"],
        "optimizer": optimizer_integrity(optimizer, completed, len(blocks)),
        "next_hash_exact": a0.next_update_hash(loaders, symbols, replay=True)
        == checkpoint["next_global_batch_sha256"],
        "fresh_process": os.getpid() != checkpoint.get("writer_pid"),
        "completed_updates": completed,
    }
    audit["passed"] = all(required.values()) and all(
        audit[key]
        for key in ("reader_exact_reload", "next_hash_exact", "fresh_process")
    ) and audit["optimizer"]["passed"]
    if not audit["passed"]:
        raise SystemExit(f"2C2 strict resume failed: {audit}")
    return completed, audit


def wait_for_data_barrier(run_root, configuration, completed_update, batch_hash):
    directory = Path(run_root) / "data_barrier" / f"update_{completed_update:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    durable_json(
        directory / f"{configuration}.json",
        {
            "configuration": configuration,
            "completed_update": completed_update,
            "global_batch_sha256": batch_hash,
            "pid": os.getpid(),
        },
    )
    deadline = time.monotonic() + 1800
    paths = {key: directory / f"{key}.json" for key in CONFIGURATIONS}
    while not all(path.is_file() for path in paths.values()):
        if time.monotonic() >= deadline:
            raise SystemExit(f"timed out at data barrier update {completed_update}")
        time.sleep(0.25)
    rows = {key: json.loads(path.read_text()) for key, path in paths.items()}
    hashes = {row.get("global_batch_sha256") for row in rows.values()}
    updates = {row.get("completed_update") for row in rows.values()}
    passed = hashes == {batch_hash} and updates == {completed_update}
    report = {
        "completed_update": completed_update,
        "hashes": {
            key: row.get("global_batch_sha256") for key, row in rows.items()
        },
        "identical_before_optimizer_update": passed,
    }
    if not passed:
        raise SystemExit(f"2C2 data sequence mismatch: {report}")
    return report


def train_one_update(
    run_root,
    configuration,
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
    blocks = blocks_for(configuration)
    update_hash = hashlib.sha256()
    loss_total = 0.0
    routing = {
        block: {
            "weights": torch.zeros(4, dtype=torch.float64),
            "entropy": 0.0,
            "topdown_rms": 0.0,
            "feedback_rms": 0.0,
        }
        for block in blocks
    }
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
            _, loss = cumulative_forward(student, x, y, "real", memory=memory)
        scaled = loss / microbatches
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - forward_start
        backward_start = time.perf_counter()
        scaled.backward()
        torch.cuda.synchronize()
        backward_seconds += time.perf_counter() - backward_start
        loss_total += scaled.detach().float().item()
        stats = student.get_topdown_stats()
        for block in blocks:
            row = stats[block]
            routing[block]["weights"] += torch.tensor(
                row["mean_weights"], dtype=torch.float64
            )
            routing[block]["entropy"] += row["mean_entropy"]
            routing[block]["topdown_rms"] += row["topdown_rms"]
            routing[block]["feedback_rms"] += row["feedback_rms"]
        del x, y, memory, loss, scaled
    student.set_topdown_instrumentation(False)
    if target_count != GLOBAL_TARGETS:
        raise SystemExit(f"global target geometry mismatch: {target_count}")
    batch_hash = update_hash.hexdigest()
    barrier = wait_for_data_barrier(
        run_root, configuration, update + 1, batch_hash
    )
    parameters = reader_parameter_map(student)
    gradient_rows = {}
    all_reader_parameters = []
    for block, named in parameters.items():
        gradient_rows[f"B{block + 1}"] = {}
        for name, parameter in named.items():
            all_reader_parameters.append(parameter)
            gradient_rows[f"B{block + 1}"][name] = {
                "present": parameter.grad is not None,
                "finite": parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all()),
                "nonzero": parameter.grad is not None
                and bool(parameter.grad.count_nonzero().item()),
                "norm": None
                if parameter.grad is None
                else parameter.grad.detach().float().norm().item(),
            }
    prefix = "transformer.topdown_attnres_by_destination."
    base_gradients = [
        name
        for name, value in student.named_parameters()
        if not name.startswith(prefix) and value.grad is not None
    ]
    teacher_gradients = [
        name for name, value in teacher.named_parameters() if value.grad is not None
    ]
    if (
        base_gradients
        or teacher_gradients
        or not all(
            row["present"] and row["finite"]
            for reader_row in gradient_rows.values()
            for row in reader_row.values()
        )
    ):
        raise SystemExit(
            f"2C2 gradient boundary failure: readers={gradient_rows} "
            f"base={base_gradients} teacher={teacher_gradients}"
        )
    per_reader_gradient_norm = {
        key: math.sqrt(
            sum(
                value["norm"] ** 2
                for value in rows.values()
                if value["norm"] is not None
            )
        )
        for key, rows in gradient_rows.items()
    }
    grad_norm = torch.nn.utils.clip_grad_norm_(all_reader_parameters, 1.0)
    if not torch.isfinite(grad_norm) or not math.isfinite(loss_total):
        raise SystemExit("non-finite 2C2 loss/gradient")
    lr = a0.get_lr(a0.EXPECTED_PARENT_UPDATES + update)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    if any(not torch.isfinite(value).all() for value in all_reader_parameters):
        raise SystemExit("non-finite trained reader parameter")
    router_rows = {}
    for block, accumulator in routing.items():
        router_rows[f"B{block + 1}"] = {
            "routing_weights": {
                f"v{depth}": value
                for depth, value in zip(
                    SOURCE_DEPTHS,
                    (accumulator["weights"] / microbatches).tolist(),
                )
            },
            "routing_entropy": accumulator["entropy"] / microbatches,
            "topdown_rms": accumulator["topdown_rms"] / microbatches,
            "feedback_rms": accumulator["feedback_rms"] / microbatches,
        }
    elapsed = time.perf_counter() - started
    row = {
        "experiment": "2C2",
        "configuration": configuration,
        "masked_blocks": [block + 1 for block in blocks],
        "update": update,
        "completed_updates": update + 1,
        "processed_targets": (update + 1) * GLOBAL_TARGETS,
        "global_schedule_step": a0.EXPECTED_PARENT_UPDATES + update,
        "global_batch_sha256": batch_hash,
        "global_targets": target_count,
        "loss": loss_total,
        "lr": lr,
        "total_reader_gradient_norm": float(grad_norm),
        "per_reader_gradient_norm": per_reader_gradient_norm,
        "gradients": gradient_rows,
        "base_gradients": base_gradients,
        "teacher_gradients": teacher_gradients,
        "readers": reader_metrics(
            student,
            {block: router_rows[f"B{block + 1}"] for block in blocks},
        ),
        "optimizer": optimizer_integrity(optimizer, update + 1, len(blocks)),
        "frozen_hashes": validate_frozen_hashes(student, teacher),
        "teacher_eval_no_grad": True,
        "data_barrier": barrier,
        "writers_active_calls": 0,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "wall_seconds": elapsed,
        "targets_per_second": GLOBAL_TARGETS / elapsed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "next_batch_sha256": a0.next_update_hash(loaders, symbols, replay=True),
    }
    gates = ",".join(
        f"B{block + 1}={row['readers'][f'B{block + 1}']['effective_gate']:.5f}"
        for block in blocks
    )
    print(
        f"2C2 {configuration} update={update + 1:02d}/{TARGET_UPDATE} "
        f"loss={loss_total:.6f} gates={gates}",
        flush=True,
    )
    return row


def direct_feedback(student, memory_bank, active_blocks):
    result = {}
    for block in active_blocks:
        reader = readers(student)[str(block)]
        topdown = reader(list(memory_bank.unbind(dim=0)))
        result[block] = reader.gate.tanh() * topdown
    return result


@torch.no_grad()
def self_resume_preflight(student, symbols, configuration):
    blocks = blocks_for(configuration)
    loader = symbols["DataLoaderLite"](
        B=2, T=8, process_rank=0, num_processes=1, split="val"
    )
    x, _ = loader.next_batch()
    x = x.cuda()
    def fresh_state():
        return student.init_recurrent_state(
            2,
            "masked_l1_no_feedback",
            device="cuda",
            dtype=torch.bfloat16,
            mask_depth=len(blocks),
        )

    state = fresh_state()
    position_zero_memory_zero = state.feedback_memory.count_nonzero().item() == 0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(4):
            feedback = direct_feedback(
                student, state.feedback_memory.detach(), blocks
            )
            _, state = student.forward_step(
                x[:, position], state, attention_feedback_by_block=feedback
            )
        payload = state.state_dict()
        restored = student.load_recurrent_state(
            payload, device="cuda", dtype=torch.bfloat16
        )
        feedback_a = direct_feedback(
            student, state.feedback_memory.detach(), blocks
        )
        feedback_b = direct_feedback(
            student, restored.feedback_memory.detach(), blocks
        )
        logits_a, next_a = student.forward_step(
            x[:, 4], state, attention_feedback_by_block=feedback_a
        )
        logits_b, next_b = student.forward_step(
            x[:, 4], restored, attention_feedback_by_block=feedback_b
        )
        def recurrent_logits(tokens):
            recurrent_state = fresh_state()
            rows = []
            for position in range(tokens.size(1)):
                feedback = direct_feedback(
                    student, recurrent_state.feedback_memory.detach(), blocks
                )
                logits, recurrent_state = student.forward_step(
                    tokens[:, position],
                    recurrent_state,
                    attention_feedback_by_block=feedback,
                )
                rows.append(logits)
            return torch.cat(rows, dim=1)

        reference_logits = recurrent_logits(x)
        row_changed = x.clone()
        row_changed[1] = (row_changed[1] + 7) % student.config.vocab_size
        row_logits = recurrent_logits(row_changed)
        future_changed = x.clone()
        future_changed[:, 4:] = (
            future_changed[:, 4:] + 11
        ) % student.config.vocab_size
        future_logits = recurrent_logits(future_changed)
    report = {
        "state_schema": payload["schema"],
        "logits_bit_exact": torch.equal(logits_a, logits_b),
        "next_state_exact": a0.nested_equal(
            next_a.state_dict(), next_b.state_dict()
        ),
        "masked_caches_absent": all(
            next_a.kv_caches[block] is None for block in blocks
        ),
        "later_caches_length_exact": all(
            next_a.kv_caches[index] is not None
            and next_a.kv_caches[index].length == 5
            for index in range(len(blocks), student.config.n_layer)
        ),
        "position_zero_memory_zero": position_zero_memory_zero,
        "row_isolation_bit_exact": torch.equal(
            reference_logits[0], row_logits[0]
        ),
        "future_prefix_bit_exact": torch.equal(
            reference_logits[:, :4], future_logits[:, :4]
        ),
        "all_states_finite": torch.isfinite(next_a.feedback_memory).all().item(),
    }
    report["passed"] = all(
        value for key, value in report.items() if key not in {"state_schema", "passed"}
    )
    return report


@torch.no_grad()
def evaluate_self_controls(student, symbols, configuration):
    student.eval()
    blocks = blocks_for(configuration)
    resume = self_resume_preflight(student, symbols, configuration)
    if not resume["passed"]:
        raise SystemExit(f"self recurrent resume preflight failed: {resume}")
    loader = validation_loader(symbols)
    controls = ["real", "shuffle", "zero"]
    if len(blocks) > 1:
        controls.append("b1_only")
    losses = {name: [] for name in controls}
    permutation = symbols["fixed_derangement"](a0.VALIDATION_B, "cuda")
    validation_hash = hashlib.sha256()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        validation_hash.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        for control in controls:
            state = student.init_recurrent_state(
                x.size(0),
                "masked_l1_no_feedback",
                device="cuda",
                dtype=torch.bfloat16,
                mask_depth=len(blocks),
            )
            token_loss = 0.0
            for position in range(x.size(1)):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    active = ()
                    bank = state.feedback_memory.detach()
                    if control == "real":
                        active = blocks
                    elif control == "shuffle":
                        active = blocks
                        bank = bank[:, permutation]
                    elif control == "b1_only":
                        active = (0,)
                    feedback = direct_feedback(student, bank, active)
                    logits, state = student.forward_step(
                        x[:, position],
                        state,
                        attention_feedback_by_block=feedback,
                    )
                    token_loss += F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y[:, position].reshape(-1)
                    ).detach().double().item()
                if not all(state.kv_caches[block] is None for block in blocks):
                    raise SystemExit("self recurrent masked cache appeared")
                if not all(
                    state.kv_caches[index] is not None
                    and state.kv_caches[index].length == position + 1
                    for index in range(len(blocks), student.config.n_layer)
                ):
                    raise SystemExit("self recurrent later KV cache failure")
                if not torch.isfinite(state.feedback_memory).all():
                    raise SystemExit("non-finite self recurrent state")
            losses[control].append(token_loss / x.size(1))
        del x, y
        print(
            f"2C2 {configuration} self batch {batch_index + 1:02d}/20",
            flush=True,
        )
    digest = validation_hash.hexdigest()
    if digest != CANONICAL_SHA:
        raise SystemExit(f"self validation hash mismatch: {digest}")
    means = {name: statistics.fmean(values) for name, values in losses.items()}
    paired = paired_statistics(losses["real"], losses["shuffle"])
    result = {
        "experiment": "2C2",
        "configuration": configuration,
        "triggered": True,
        "masked_blocks": [block + 1 for block in blocks],
        "canonical_validation_sha256": digest,
        "losses": means,
        "per_batch_losses": losses,
        "paired_real_vs_shuffled": paired,
        "self_specific_gap": paired["mean_gap"],
        "resume_preflight": resume,
        "position_zero_memory_zero": True,
        "masked_blocks_have_no_history_caches": True,
        "unmasked_blocks_retain_kv": True,
        "one_token_memory_lag": True,
        "no_bptt": True,
        "optimizer_updates": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }
    return result


def reconcile_metrics(path, completed_updates):
    path = Path(path)
    rows = []
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected = list(range(1, completed_updates + 1))
    actual = [row.get("completed_updates") for row in rows]
    if actual != expected:
        raise SystemExit(f"metrics/checkpoint mismatch: {actual} != {expected}")
    return rows


def run_training(args):
    require_git(clean=True)
    config = load_config()
    configuration = args.configuration
    blocks = blocks_for(configuration)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_MAPPING[configuration]):
        raise SystemExit(f"training GPU mapping mismatch for {configuration}")
    if args.target_update not in MILESTONES:
        raise SystemExit("target update must be a frozen milestone")
    run_dir = run_dir_for(args.run_root, configuration)
    preflight_path = run_dir / "preflight.json"
    smoke_path = run_dir / "disposable_smoke.json"
    if not preflight_path.is_file() or not smoke_path.is_file():
        raise SystemExit("passing preflight and disposable smoke are required")
    preflight = json.loads(preflight_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    if (
        not preflight.get("passed")
        or not smoke.get("passed")
        or preflight.get("configuration") != configuration
        or preflight.get("implementation_git_commit") != git_output("rev-parse", "HEAD")
    ):
        raise SystemExit("stale or failed 2C2 preflight/smoke")
    (
        symbols,
        teacher,
        student,
        optimizer,
        loaders,
        parent_aux,
        initialization,
    ) = make_runtime(args.parent_checkpoint, configuration, include_optimizer=True)
    identity = run_identity(configuration, parent_aux, config)
    durable_json(run_dir / "run_identity.json", identity)
    completed = 0
    restart_audit = None
    if args.resume:
        completed, restart_audit = load_checkpoint(
            args.resume,
            configuration,
            student,
            optimizer,
            loaders,
            symbols,
            identity,
        )
        if completed != RESTART_UPDATE:
            raise SystemExit("2C2 result resume is authorized only at update 20")
        durable_json(
            run_dir / "restart_audit_updates_000020.json", restart_audit
        )
    elif args.target_update > RESTART_UPDATE:
        raise SystemExit("a fresh process resume from update 20 is mandatory")
    metrics_path = run_dir / "metrics.jsonl"
    metrics = reconcile_metrics(metrics_path, completed)
    training_started = time.perf_counter()
    for update in range(completed, args.target_update):
        row = train_one_update(
            args.run_root,
            configuration,
            update,
            student,
            teacher,
            optimizer,
            loaders,
            symbols,
        )
        append_jsonl(metrics_path, row)
        metrics.append(row)
        completed = update + 1
        if completed in MILESTONES:
            generic_means = None
            generic_manifest = None
            if completed == TARGET_UPDATE:
                generic_means, generic_manifest = c1.compute_generic_means(
                    teacher, symbols
                )
                durable_json(
                    run_dir / "generic_means_manifest.json", generic_manifest
                )
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
                configuration,
                completed,
                generic_means=generic_means,
                extended=completed == TARGET_UPDATE,
            )
            durable_json(evaluation_path(run_dir, completed), evaluation)
            save_checkpoint(
                run_dir,
                configuration,
                completed,
                student,
                teacher,
                optimizer,
                loaders,
                symbols,
                identity,
            )
    final_evaluation = None
    self_transfer = None
    if completed == TARGET_UPDATE:
        final_evaluation = json.loads(evaluation_path(run_dir, completed).read_text())
        prefix_one = final_evaluation["activation_losses"]["prefix_1"]["mean"]
        matched_gain = prefix_one - final_evaluation["losses"]["real"]
        trigger = (
            final_evaluation["recovery"] > 0
            and final_evaluation["specific_gap"] >= 0.010
            and final_evaluation["paired_real_vs_shuffled"]["real_wins"] >= 18
            and (len(blocks) == 1 or matched_gain >= 0.020)
        )
        if trigger:
            self_transfer = evaluate_self_controls(
                student, symbols, configuration
            )
            self_transfer["teacher_recovery"] = final_evaluation["recovery"]
            self_transfer["self_recovery"] = (
                final_evaluation["losses"]["masked"]
                - self_transfer["losses"]["real"]
            )
            self_transfer["self_teacher_recovery_ratio"] = (
                self_transfer["self_recovery"] / final_evaluation["recovery"]
            )
            if len(blocks) > 1:
                self_transfer["self_matched_destination_gain"] = (
                    self_transfer["losses"]["b1_only"]
                    - self_transfer["losses"]["real"]
                )
            else:
                self_transfer["self_matched_destination_gain"] = None
        else:
            self_transfer = {
                "experiment": "2C2",
                "configuration": configuration,
                "triggered": False,
                "reason": "frozen teacher-assisted self-transfer gate not met",
                "teacher_recovery": final_evaluation["recovery"],
                "teacher_specific_gap": final_evaluation["specific_gap"],
                "teacher_real_wins": final_evaluation["paired_real_vs_shuffled"]["real_wins"],
                "matched_destination_gain": matched_gain,
                "optimizer_updates": 0,
            }
        durable_json(run_dir / "self_transfer.json", self_transfer)
    hashes = validate_frozen_hashes(student, teacher)
    stage = {
        "experiment": "2C2",
        "configuration": configuration,
        "completed_updates": completed,
        "processed_targets": completed * GLOBAL_TARGETS,
        "target_update": args.target_update,
        "restart_audit": restart_audit,
        "initialization": initialization,
        "frozen_hashes": hashes,
        "reader_state_sha256": reader_state_sha(student),
        "optimizer": optimizer_integrity(optimizer, completed, len(blocks)),
        "next_global_batch_sha256": a0.next_update_hash(
            loaders, symbols, replay=True
        ),
        "performance": {
            "training_wall_seconds_this_process": time.perf_counter()
            - training_started,
            "mean_targets_per_second": statistics.fmean(
                row["targets_per_second"] for row in metrics
            ),
            "peak_allocated_mb": max(row["peak_allocated_mb"] for row in metrics),
            "peak_reserved_mb": max(row["peak_reserved_mb"] for row in metrics),
            "evaluation_wall_seconds": 0.0
            if final_evaluation is None
            else final_evaluation["elapsed_seconds"],
            "self_evaluation_wall_seconds": 0.0
            if not self_transfer or not self_transfer.get("triggered")
            else self_transfer["elapsed_seconds"],
        },
        "optimizer_updates_total": completed,
        "writers_active_calls": 0,
        "hellaswag_run": False,
    }
    durable_json(run_dir / f"stage_updates_{completed:06d}.json", stage)
    marker = (
        "EXPERIMENT_2C2_RESTART_REQUIRED"
        if completed == RESTART_UPDATE
        else "EXPERIMENT_2C2_WORKER_COMPLETE"
    )
    print(
        f"{marker} configuration={configuration} updates={completed}", flush=True
    )
    return stage


def run_finalize(args):
    """Finish evaluation-only self transfer after an intact update-48 checkpoint."""
    require_git(clean=True)
    load_config()
    configuration = args.configuration
    blocks = blocks_for(configuration)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_MAPPING[configuration]):
        raise SystemExit(f"finalize GPU mapping mismatch for {configuration}")
    run_dir = run_dir_for(args.run_root, configuration)
    final_path = evaluation_path(run_dir, TARGET_UPDATE)
    checkpoint = checkpoint_path(run_dir, TARGET_UPDATE)
    verification_path = checkpoint.with_suffix(".pt.verification.json")
    if not final_path.is_file() or not checkpoint.is_file() or not verification_path.is_file():
        raise SystemExit("finalize requires canonical update-48 evaluation and checkpoint")
    verification = json.loads(verification_path.read_text())
    if not verification.get("passed") or file_sha256(checkpoint) != verification.get("sha256"):
        raise SystemExit("update-48 checkpoint verification failed during finalize")
    symbols, teacher, student, _, _, _, _ = make_runtime(
        args.parent_checkpoint, configuration, include_optimizer=False
    )
    payload = a0.torch_load(checkpoint, mmap=True)
    required = {
        "schema": payload.get("schema") == CHECKPOINT_SCHEMA,
        "configuration": payload.get("configuration") == configuration,
        "completed_updates": payload.get("completed_updates") == TARGET_UPDATE,
        "masked_blocks": payload.get("masked_blocks")
        == [block + 1 for block in blocks],
        "parent": payload.get("parent_checkpoint_sha256")
        == a0.EXPECTED_PARENT_SHA256,
        "base": payload.get("base_model_sha256") == BASE_MODEL_SHA,
    }
    if not all(required.values()):
        raise SystemExit(f"finalize checkpoint lineage mismatch: {required}")
    readers(student).load_state_dict(payload["reader_state"], strict=True)
    if reader_state_sha(student) != payload["reader_state_sha256"]:
        raise SystemExit("finalize reader strict load mismatch")
    final_evaluation = json.loads(final_path.read_text())
    prefix_one = final_evaluation["activation_losses"]["prefix_1"]["mean"]
    matched_gain = prefix_one - final_evaluation["losses"]["real"]
    trigger = (
        final_evaluation["recovery"] > 0
        and final_evaluation["specific_gap"] >= 0.010
        and final_evaluation["paired_real_vs_shuffled"]["real_wins"] >= 18
        and (len(blocks) == 1 or matched_gain >= 0.020)
    )
    if trigger:
        self_transfer = evaluate_self_controls(student, symbols, configuration)
        self_transfer["teacher_recovery"] = final_evaluation["recovery"]
        self_transfer["self_recovery"] = (
            final_evaluation["losses"]["masked"]
            - self_transfer["losses"]["real"]
        )
        self_transfer["self_teacher_recovery_ratio"] = (
            self_transfer["self_recovery"] / final_evaluation["recovery"]
        )
        self_transfer["self_matched_destination_gain"] = (
            None
            if len(blocks) == 1
            else self_transfer["losses"]["b1_only"]
            - self_transfer["losses"]["real"]
        )
    else:
        self_transfer = {
            "experiment": "2C2",
            "configuration": configuration,
            "triggered": False,
            "reason": "frozen teacher-assisted self-transfer gate not met",
            "teacher_recovery": final_evaluation["recovery"],
            "teacher_specific_gap": final_evaluation["specific_gap"],
            "teacher_real_wins": final_evaluation["paired_real_vs_shuffled"]["real_wins"],
            "matched_destination_gain": matched_gain,
            "optimizer_updates": 0,
        }
    self_transfer["evaluation_implementation_commit"] = git_output("rev-parse", "HEAD")
    durable_json(run_dir / "self_transfer.json", self_transfer)
    metrics = reconcile_metrics(run_dir / "metrics.jsonl", TARGET_UPDATE)
    previous_stage = run_dir / "stage_updates_000048.json"
    if previous_stage.is_file():
        previous_stage.replace(run_dir / "stage_updates_000048.pre_finalize.json")
    stage = {
        "experiment": "2C2",
        "configuration": configuration,
        "completed_updates": TARGET_UPDATE,
        "processed_targets": TARGET_UPDATE * GLOBAL_TARGETS,
        "target_update": TARGET_UPDATE,
        "restart_audit": json.loads(
            (run_dir / "restart_audit_updates_000020.json").read_text()
        ),
        "frozen_hashes": validate_frozen_hashes(student, teacher),
        "reader_state_sha256": reader_state_sha(student),
        "optimizer": payload["optimizer_integrity"],
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "performance": {
            "training_wall_seconds_all_processes": sum(
                row["wall_seconds"] for row in metrics
            ),
            "mean_targets_per_second": statistics.fmean(
                row["targets_per_second"] for row in metrics
            ),
            "peak_allocated_mb": max(row["peak_allocated_mb"] for row in metrics),
            "peak_reserved_mb": max(row["peak_reserved_mb"] for row in metrics),
            "evaluation_wall_seconds": final_evaluation["elapsed_seconds"],
            "self_evaluation_wall_seconds": self_transfer.get("elapsed_seconds", 0.0),
        },
        "optimizer_updates_total": TARGET_UPDATE,
        "finalize_optimizer_updates": 0,
        "finalize_git_commit": git_output("rev-parse", "HEAD"),
        "checkpoint_lineage": required,
        "checkpoint_sha256": verification["sha256"],
        "writers_active_calls": 0,
        "hellaswag_run": False,
    }
    durable_json(run_dir / "stage_updates_000048.json", stage)
    print(
        f"EXPERIMENT_2C2_FINALIZE_COMPLETE configuration={configuration} "
        f"self_triggered={trigger}",
        flush=True,
    )
    return stage


def copy_worker_artifacts(run_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "preflight.json",
        "disposable_smoke.json",
        "run_identity.json",
        "metrics.jsonl",
        "restart_audit_updates_000020.json",
        "stage_updates_000020.json",
        "stage_updates_000048.json",
        "generic_means_manifest.json",
        "self_transfer.json",
        *[f"evaluation_updates_{update:06d}.json" for update in MILESTONES],
    ]
    for name in names:
        source = Path(run_dir) / name
        if not source.is_file():
            raise SystemExit(f"missing required worker artifact: {source}")
        shutil.copy2(source, output_dir / name)


def classify(rows, integrity):
    if not integrity:
        return "CUMULATIVE-MASK EXPERIMENT UNSTABLE"
    c2 = rows["C2"]
    deep = [rows["C3"], rows["C4"]]
    c2_strong = (
        c2["matched_destination_gain"] >= 0.020
        and c2["specific_gap"] >= 0.010
        and c2["real_wins"] >= 18
    )
    deep_strong = any(
        row["matched_destination_gain"] >= 0.020
        and row["specific_gap"] >= 0.010
        and row["real_wins"] >= 18
        and row["recovery_fraction_retention"] >= 0.50
        for row in deep
    )
    if c2_strong and deep_strong:
        return "MATCHED MULTI-DESTINATION FEEDBACK SCALES"
    if c2["matched_destination_gain"] >= 0.020 and not deep_strong:
        return "MATCHED FEEDBACK HELPS BUT DOES NOT SCALE DEEPLY"
    cumulative = [rows[key] for key in ("C2", "C3", "C4")]
    if (
        all(row["recovery"] >= 0.020 for row in cumulative)
        and all(row["specific_gap"] < 0.010 for row in cumulative)
        and all(row["generic_loss"] <= row["real_loss"] + 0.010 for row in cumulative)
    ):
        return "MATCHED FEEDBACK IMPROVES GENERIC COMPENSATION ONLY"
    if all(row["matched_destination_gain"] < 0.020 for row in cumulative):
        return "MATCHED FEEDBACK DOES NOT RESCUE CUMULATIVE MASKING"
    return "CUMULATIVE-MASK RESULT IS MIXED"


def classification_rule(classification):
    return {
        "MATCHED MULTI-DESTINATION FEEDBACK SCALES": (
            "C2 gain/gap/wins pass and at least one of C3/C4 also passes "
            "gain/gap/wins with recovery-fraction retention >= 0.50."
        ),
        "MATCHED FEEDBACK HELPS BUT DOES NOT SCALE DEEPLY": (
            "C2 matched-destination gain is >= 0.020, but neither C3 nor C4 "
            "passes the frozen deep-scaling rule."
        ),
        "MATCHED FEEDBACK IMPROVES GENERIC COMPENSATION ONLY": (
            "All deeper configurations materially recover loss, all have gap < 0.010, "
            "and generic memory is comparable to or better than real memory."
        ),
        "MATCHED FEEDBACK DOES NOT RESCUE CUMULATIVE MASKING": (
            "Matched-destination gain is < 0.020 for C2, C3, and C4."
        ),
        "CUMULATIVE-MASK RESULT IS MIXED": (
            "Integrity passes, but no other frozen classification rule fits."
        ),
        "CUMULATIVE-MASK EXPERIMENT UNSTABLE": (
            "At least one frozen integrity requirement failed."
        ),
    }[classification]


def scientific_answers(rows, progressive, router_stats, self_transfer):
    def answer_gain(configuration):
        row = rows[configuration]
        gain = row["matched_destination_gain"]
        return (
            f"YES; matched readers improve over B1-only by {gain:.10f}."
            if gain >= 0.020
            else f"NO under the frozen 0.020 threshold; gain is {gain:.10f}."
        )

    fractions = [rows[key]["recovery_fraction"] for key in CONFIGURATIONS]
    mixtures = []
    for configuration in CONFIGURATIONS:
        vectors = [
            tuple(row["routing_weights"].values())
            for row in router_stats[configuration].values()
        ]
        if len(set(vectors)) > 1:
            mixtures.append(configuration)
    b1_vectors = [
        tuple(router_stats[key]["B1"]["routing_weights"].values())
        for key in CONFIGURATIONS
    ]
    triggered = [key for key, row in self_transfer.items() if row.get("triggered")]
    return {
        "Q1": answer_gain("C2"),
        "Q2": answer_gain("C3"),
        "Q3": answer_gain("C4"),
        "Q4": "; ".join(
            f"{key}: gap {rows[key]['specific_gap']:.10f}, generic-real "
            f"{rows[key]['generic_vs_real']:.10f}"
            for key in CONFIGURATIONS
        ),
        "Q5": (
            "Recovery fractions by C1-C4 are "
            + ", ".join(f"{value:.6f}" for value in fractions)
            + "."
        ),
        "Q6": (
            f"Destination-specific routing differs within {', '.join(mixtures)}."
            if mixtures
            else "No within-configuration destination routing difference was resolved."
        ),
        "Q7": (
            "YES; final B1 routing vectors differ across C1-C4."
            if len(set(b1_vectors)) > 1
            else "NO; final B1 routing vectors are identical across C1-C4."
        ),
        "Q8": (
            "Self transfer ran for " + ", ".join(triggered) + "; see self-transfer table."
            if triggered
            else "No configuration passed the frozen self-transfer gate."
        ),
    }


def next_decisions(rows, classification, self_transfer):
    successful = [
        key
        for key in ("C2", "C3", "C4")
        if rows[key]["matched_destination_gain"] >= 0.020
        and rows[key]["specific_gap"] >= 0.010
        and rows[key]["real_wins"] >= 18
    ]
    strongest = max(
        successful or list(CONFIGURATIONS),
        key=lambda key: rows[key]["recovery_fraction"],
    )
    positive_self = [
        key
        for key, row in self_transfer.items()
        if row.get("triggered")
        and row.get("self_recovery", 0) > 0
        and row.get("self_specific_gap", 0) >= 0.010
    ]
    return {
        "A": (
            f"YES at {', '.join(successful)} under the frozen matched-feedback rule."
            if successful
            else "NO cumulative depth passed the frozen matched-feedback rule."
        ),
        "B": f"{strongest} is strongest by recovery fraction among eligible configurations.",
        "C": (
            f"A separately preregistered self-training test is supported most strongly at {positive_self[-1]}; do not launch here."
            if positive_self
            else "NO; zero-shot self transfer did not establish a self-training candidate."
        ),
        "D": (
            "Evidence may motivate a separately approved writer experiment, but 2C2 does not authorize it."
            if successful
            else "NO; matched direct-reader evidence is insufficient for writers."
        ),
        "E": "If writers are later authorized, alternate reader/writer phases rather than co-training; do not launch here.",
        "F": (
            "A separate iterative-loop protocol is supportable only for a successful self-transfer configuration; do not launch here."
            if positive_self
            else "NO; multiple recurrent iterations are not yet supported."
        ),
        "G": "YES; keep temporal credit limited to one token when writers are eventually introduced.",
    }


def final_report_text(summary):
    rows = summary["configurations"]
    lines = [
        "# Experiment 2C2 Final Report",
        "",
        f"Classification: **{summary['classification']}**",
        "",
        f"Frozen rule: {summary['classification_rule']}",
        "",
        "## Provenance",
        "",
        f"- 2C1 frozen tag: `{PARENT_TAG}`",
        f"- 2C1 parent commit: `{PARENT_COMMIT}`",
        f"- 2C2 branch: `{BRANCH}`",
        f"- Implementation commit: `{summary['implementation_commit']}`",
        "- Evaluation-only finalize commits: `"
        + "`, `".join(summary.get("evaluation_finalize_commits", []))
        + "`",
        f"- Results commit: `{summary.get('results_commit', 'recorded by the results commit containing JSON artifacts')}`",
        "- Final-report commit: `the immutable commit containing this file`",
        "- Base checkpoint SHA256: `"
        + a0.EXPECTED_PARENT_SHA256
        + "`",
        "",
        "## Main final table",
        "",
        "| Config | Masked blocks | Readers | Masked | Real | Shuffled | Generic | Specific gap | Recovery % | Specific share | Real wins |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in rows.items():
        share = "n/a" if row["specific_share"] is None else f"{row['specific_share']:.6f}"
        lines.append(
            f"| {key} | {row['masked_blocks_label']} | {row['reader_count']} | "
            f"{row['masked_loss']:.10f} | {row['real_loss']:.10f} | "
            f"{row['shuffled_loss']:.10f} | {row['generic_loss']:.10f} | "
            f"{row['specific_gap']:.10f} | {100 * row['recovery_fraction']:.6f} | "
            f"{share} | {row['real_wins']}/20 |"
        )
    lines.extend([
        "",
        "## Matched-feedback incremental table",
        "",
        "| Config | B1-only loss | All-reader real loss | Gain from matched readers | Positive batches |",
        "|---|---:|---:|---:|---:|",
    ])
    for key in ("C2", "C3", "C4"):
        row = rows[key]
        lines.append(
            f"| {key} | {row['b1_only_loss']:.10f} | {row['real_loss']:.10f} | "
            f"{row['matched_destination_gain']:.10f} | {row['matched_gain_positive_batches']}/20 |"
        )
    lines.extend([
        "",
        "## Training trajectory",
        "",
        "| Config | Tokens | Masked | Real | Shuffled | Specific gap | Recovery % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key in CONFIGURATIONS:
        for row in summary["training_trajectories"][key]:
            lines.append(
                f"| {key} | {row['tokens']} | {row['masked']:.10f} | "
                f"{row['real']:.10f} | {row['shuffled']:.10f} | "
                f"{row['specific_gap']:.10f} | {100 * row['recovery_fraction']:.6f} |"
            )
    lines.extend([
        "",
        "## Progressive reader activation",
        "",
        "| Config | Active readers | Loss | Delta from previous |",
        "|---|---|---:|---:|",
    ])
    for key in ("C2", "C3", "C4"):
        for row in summary["progressive_reader_controls"][key]:
            lines.append(
                f"| {key} | {row['active_label']} | {row['loss']:.10f} | "
                f"{row['delta_from_previous']:.10f} |"
            )
    lines.extend([
        "",
        "## Leave-one-reader-out",
        "",
        "| Config | Reader removed | Ablated loss | Delta vs all-real | Positive batches |",
        "|---|---|---:|---:|---:|",
    ])
    for key in CONFIGURATIONS:
        for row in summary["reader_ablation"][key]:
            lines.append(
                f"| {key} | {row['reader_removed']} | {row['ablated_loss']:.10f} | "
                f"{row['delta_vs_all_real']:.10f} | {row['positive_batches']}/20 |"
            )
    lines.extend([
        "",
        "## Generic-template control",
        "",
        "| Config | Masked | Generic | Shuffled | Real | Generic-real | Shuffled-real |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, row in summary["generic_controls"].items():
        lines.append(
            f"| {key} | {row['masked']:.10f} | {row['generic']:.10f} | "
            f"{row['shuffled']:.10f} | {row['real']:.10f} | "
            f"{row['generic_vs_real']:.10f} | {row['specific_gap']:.10f} |"
        )
    lines.extend([
        "",
        "## Final readers",
        "",
        "| Config | Destination | Gate | Query norm | Entropy | v16 | v17 | v20 | v24 | Feedback RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, reader_rows in summary["router_stats"].items():
        for destination, row in reader_rows.items():
            weights = row["routing_weights"]
            lines.append(
                f"| {key} | {destination} | {row['effective_gate']:.10f} | "
                f"{row['query_norm']:.10f} | {row['routing_entropy']:.10f} | "
                f"{weights['v16']:.10f} | {weights['v17']:.10f} | "
                f"{weights['v20']:.10f} | {weights['v24']:.10f} | "
                f"{row['feedback_rms']:.10f} |"
            )
    lines.extend(["", "## B1 reader evolution", ""])
    for key in CONFIGURATIONS:
        row = summary["router_stats"][key]["B1"]
        lines.append(
            f"- {key}/B1: gate {row['effective_gate']:.10f}; query norm "
            f"{row['query_norm']:.10f}; RMS displacement {row['rmsnorm_displacement']:.10f}; "
            f"routing {row['routing_weights']}; feedback RMS {row['feedback_rms']:.10f}."
        )
    lines.append(
        f"- Pairwise B1 query cosines: {summary['b1_query_cosines']}"
    )
    lines.extend(["", "## Conditional self-recurrent transfer", ""])
    for key, row in summary["self_transfer"].items():
        if not row.get("triggered"):
            lines.append(f"- {key}: SELF TEST NOT TRIGGERED")
        else:
            lines.append(
                f"- {key}: teacher real {rows[key]['real_loss']:.10f}, teacher shuffled "
                f"{rows[key]['shuffled_loss']:.10f}, teacher gap {rows[key]['specific_gap']:.10f}, "
                f"teacher recovery {rows[key]['recovery']:.10f}; self real "
                f"{row['losses']['real']:.10f}, self shuffled {row['losses']['shuffle']:.10f}, "
                f"self gap {row['self_specific_gap']:.10f}, self recovery "
                f"{row['self_recovery']:.10f}, self/teacher recovery ratio "
                f"{row['self_teacher_recovery_ratio']:.6f}, self matched gain "
                f"{row.get('self_matched_destination_gain')}."
            )
    lines.extend(["", "## Scientific questions", ""])
    for key, value in summary["scientific_answers"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Next-experiment decisions", ""])
    for key, value in summary["next_decisions"].items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## Integrity and stopping",
        "",
        f"All frozen audit checks: **{'PASS' if summary['integrity_passed'] else 'FAIL'}**.",
        "",
        "- C1 optimizer updates: 48",
        "- C2 optimizer updates: 48",
        "- C3 optimizer updates: 48",
        "- C4 optimizer updates: 48",
        "- No writers, auxiliary objective, BPTT, reader continuation, iterative loops, additional masks, HellaSwag, or follow-on optimization were run.",
        "",
        "# EXPERIMENT 2C2 COMPLETE",
    ])
    return "\n".join(lines)


def aggregate_results(args):
    require_git(clean=True)
    config = load_config()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_rows = {}
    trajectories = {}
    paired_losses = {}
    progressive = {}
    ablations = {}
    generic_controls = {}
    router_stats = {}
    self_transfer = {}
    performance = {}
    checkpoint_manifest = {}
    preflights = {}
    smokes = {}
    restarts = {}
    stages = {}
    hash_sequences = {}
    metrics_by_configuration = {}
    canonical_evaluation_hashes = {}
    generic_shas = set()
    b1_queries = {}
    implementation_commits = set()
    for configuration, blocks in CONFIGURATIONS.items():
        run_dir = run_dir_for(args.run_root, configuration)
        preflight = json.loads((run_dir / "preflight.json").read_text())
        smoke = json.loads((run_dir / "disposable_smoke.json").read_text())
        restart = json.loads(
            (run_dir / "restart_audit_updates_000020.json").read_text()
        )
        stage = json.loads((run_dir / "stage_updates_000048.json").read_text())
        final = json.loads((run_dir / "evaluation_updates_000048.json").read_text())
        self_row = json.loads((run_dir / "self_transfer.json").read_text())
        evaluations = {
            update: json.loads(evaluation_path(run_dir, update).read_text())
            for update in MILESTONES
        }
        metrics = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            if line
        ]
        if len(metrics) != TARGET_UPDATE:
            raise SystemExit(f"{configuration} does not have exactly 48 metrics")
        metrics_by_configuration[configuration] = metrics
        canonical_evaluation_hashes[configuration] = {
            str(update): evaluations[update]["canonical_validation_sha256"]
            for update in MILESTONES
        }
        preflights[configuration] = preflight
        smokes[configuration] = smoke
        restarts[configuration] = restart
        stages[configuration] = stage
        implementation_commits.add(preflight["implementation_git_commit"])
        hash_sequences[configuration] = [row["global_batch_sha256"] for row in metrics]
        generic_shas.add(final["generic_means_tensor_sha256"])
        masked = final["losses"]["masked"]
        real = final["losses"]["real"]
        shuffled = final["losses"]["shuffle"]
        generic = final["losses"]["generic"]
        b1_only = final["activation_losses"]["prefix_1"]["mean"]
        matched_gain = b1_only - real
        matched_batch_deltas = [
            left - right
            for left, right in zip(
                final["activation_losses"]["prefix_1"]["per_batch"],
                final["per_batch_losses"]["real"],
            )
        ]
        row = {
            "masked_blocks": [block + 1 for block in blocks],
            "masked_blocks_label": "-".join(f"B{block + 1}" for block in blocks),
            "reader_count": len(blocks),
            "trainable_parameters": PARAMETERS_PER_READER * len(blocks),
            "masked_loss": masked,
            "real_loss": real,
            "shuffled_loss": shuffled,
            "generic_loss": generic,
            "damage": final["damage"],
            "recovery": final["recovery"],
            "recovery_fraction": final["recovery_fraction"],
            "specific_gap": final["specific_gap"],
            "specific_fraction": final["specific_fraction"],
            "specific_share": final["specific_share"],
            "real_wins": final["paired_real_vs_shuffled"]["real_wins"],
            "b1_only_loss": b1_only,
            "matched_destination_gain": matched_gain,
            "matched_gain_positive_batches": sum(
                value > 0 for value in matched_batch_deltas
            ),
            "generic_vs_real": generic - real,
        }
        configuration_rows[configuration] = row
        trajectories[configuration] = [
            {
                "updates": update,
                "tokens": update * GLOBAL_TARGETS,
                "masked": evaluations[update]["losses"]["masked"],
                "real": evaluations[update]["losses"]["real"],
                "shuffled": evaluations[update]["losses"]["shuffle"],
                "specific_gap": evaluations[update]["specific_gap"],
                "recovery_fraction": evaluations[update]["recovery_fraction"],
                "real_wins": evaluations[update]["paired_real_vs_shuffled"]["real_wins"],
            }
            for update in MILESTONES
        ]
        paired_losses[configuration] = {
            "real": final["per_batch_losses"]["real"],
            "shuffled": final["per_batch_losses"]["shuffle"],
            "paired": final["paired_real_vs_shuffled"],
            "b1_only": final["activation_losses"]["prefix_1"]["per_batch"],
            "matched_gain_deltas": matched_batch_deltas,
        }
        progressive[configuration] = []
        previous = None
        for count in range(len(blocks) + 1):
            value = final["activation_losses"][f"prefix_{count}"]["mean"]
            active = blocks[:count]
            progressive[configuration].append(
                {
                    "active_readers": [block + 1 for block in active],
                    "active_label": "none"
                    if not active
                    else "+".join(f"B{block + 1}" for block in active),
                    "loss": value,
                    "delta_from_previous": 0.0 if previous is None else previous - value,
                }
            )
            previous = value
        ablations[configuration] = []
        for block in blocks:
            activation = final["activation_losses"][f"minus_B{block + 1}"]
            deltas = [
                left - right
                for left, right in zip(
                    activation["per_batch"], final["per_batch_losses"]["real"]
                )
            ]
            ablations[configuration].append(
                {
                    "reader_removed": f"B{block + 1}",
                    "ablated_loss": activation["mean"],
                    "delta_vs_all_real": activation["mean"] - real,
                    "positive_batches": sum(value > 0 for value in deltas),
                    "per_batch_deltas": deltas,
                }
            )
        generic_controls[configuration] = {
            "masked": masked,
            "generic": generic,
            "shuffled": shuffled,
            "real": real,
            "generic_vs_real": generic - real,
            "specific_gap": shuffled - real,
            "means_tensor_sha256": final["generic_means_tensor_sha256"],
        }
        router_stats[configuration] = final["reader"]
        self_transfer[configuration] = self_row
        performance[configuration] = stage["performance"]
        checkpoint_manifest[configuration] = {}
        for update in MILESTONES:
            verification_path = checkpoint_path(run_dir, update).with_suffix(
                ".pt.verification.json"
            )
            verification = json.loads(verification_path.read_text())
            verification["sha256_reverified"] = (
                file_sha256(checkpoint_path(run_dir, update))
                == verification["sha256"]
            )
            checkpoint_manifest[configuration][str(update)] = verification
        checkpoint = a0.torch_load(checkpoint_path(run_dir, 48), mmap=True)
        b1_queries[configuration] = checkpoint["reader_state"]["0.query"].float()
        copy_worker_artifacts(
            run_dir, output_dir / RUN_NAMES[configuration]
        )
    c1_recovery_fraction = configuration_rows["C1"]["recovery_fraction"]
    c1_gap = configuration_rows["C1"]["specific_gap"]
    for configuration in ("C2", "C3", "C4"):
        configuration_rows[configuration]["recovery_fraction_retention"] = (
            configuration_rows[configuration]["recovery_fraction"]
            / c1_recovery_fraction
        )
        configuration_rows[configuration]["specific_gap_retention"] = (
            configuration_rows[configuration]["specific_gap"] / c1_gap
        )
    configuration_rows["C1"]["recovery_fraction_retention"] = 1.0
    configuration_rows["C1"]["specific_gap_retention"] = 1.0
    b1_cosines = {}
    keys = list(CONFIGURATIONS)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            b1_cosines[f"{left}-{right}"] = F.cosine_similarity(
                b1_queries[left], b1_queries[right], dim=0
            ).item()
    c1_regression = {}
    for row in trajectories["C1"]:
        expected = C1_EXPECTED[row["updates"]]
        checks = {
            "real": abs(row["real"] - expected["real"]) <= C1_ATOL,
            "shuffle": abs(row["shuffled"] - expected["shuffle"]) <= C1_ATOL,
            "gap": abs(row["specific_gap"] - expected["gap"]) <= C1_ATOL,
        }
        c1_regression[str(row["updates"])] = {
            "observed": row,
            "expected": expected,
            "checks": checks,
            "passed": all(checks.values()),
        }
    sequence_values = list(hash_sequences.values())
    integrity = {
        "2c1_frozen_tag_exact": git_output("rev-parse", f"{PARENT_TAG}^{{}}")
        == PARENT_COMMIT,
        "base_checkpoint_sha_exact": all(
            row["parent_checkpoint_sha256"] == a0.EXPECTED_PARENT_SHA256
            for row in preflights.values()
        ),
        "canonical_validation_hash_exact": all(
            row["evaluation"]["canonical_validation_sha256"] == CANONICAL_SHA
            for row in preflights.values()
        ) and all(
            digest == CANONICAL_SHA
            for configuration in canonical_evaluation_hashes.values()
            for digest in configuration.values()
        ),
        "c1_historical_trajectory_regression": all(
            row["passed"] for row in c1_regression.values()
        ),
        "trainable_parameter_counts_exact": all(
            row["integrity"]["trainable_parameter_count_exact"]
            for row in preflights.values()
        ),
        "teacher_frozen": all(
            row["integrity"]["teacher_eval"]
            and row["integrity"]["teacher_gradients_none"]
            for row in preflights.values()
        ),
        "base_frozen": all(
            row["frozen_hashes"]
            == {"student_base": BASE_MODEL_SHA, "teacher": BASE_MODEL_SHA}
            for row in stages.values()
        ),
        "only_intended_low_blocks_masked": all(
            row["cache_policy"]["only_intended_caches_absent"]
            for row in preflights.values()
        ),
        "all_later_blocks_retain_kv": all(
            row["cache_policy"]["later_caches_empty_and_present"]
            for row in preflights.values()
        ),
        "reader_destination_mapping_exact": all(
            row["causality"]["reader_destination_mapping_exact"]
            for row in preflights.values()
        ),
        "zero_gate_equals_cumulative_mask": all(
            row["evaluation"]["zero_gate_equals_masked"]
            for row in preflights.values()
        ),
        "future_causality": all(
            row["causality"]["future_prefix_logits_bit_exact"]
            and row["causality"]["teacher_memory_prefix_bit_exact"]
            for row in preflights.values()
        ),
        "row_isolation": all(
            row["causality"]["unchanged_row_logits_bit_exact"]
            for row in preflights.values()
        ),
        "identical_training_batch_sequence": all(
            value == sequence_values[0] for value in sequence_values[1:]
        ),
        "all_optimizer_updates_exactly_48": all(
            row["optimizer_updates_total"] == 48 for row in stages.values()
        ),
        "finalize_added_zero_optimizer_updates": all(
            row.get("finalize_optimizer_updates") == 0 for row in stages.values()
        ),
        "forced_fresh_process_restart": all(
            row["fresh_process"] for row in restarts.values()
        ),
        "checkpoint_strict_reload": all(
            row["passed"] for row in restarts.values()
        ) and all(
            verification["strict_reopen"]["passed"]
            and verification["sha256_reverified"]
            for configuration in checkpoint_manifest.values()
            for verification in configuration.values()
        ),
        "all_losses_and_gradients_finite": all(
            row["passed"] for row in smokes.values()
        ) and all(
            math.isfinite(row["loss"])
            and math.isfinite(row["total_reader_gradient_norm"])
            and all(
                gradient["present"] and gradient["finite"]
                for reader in row["gradients"].values()
                for gradient in reader.values()
            )
            for configuration in metrics_by_configuration.values()
            for row in configuration
        ),
        "generic_calibration_disjoint_and_identical": len(generic_shas) == 1
        and all(
            json.loads(
                (run_dir_for(args.run_root, key) / "generic_means_manifest.json").read_text()
            )["calibration_batch_indices"] == [20, 21, 22, 23]
            and json.loads(
                (run_dir_for(args.run_root, key) / "generic_means_manifest.json").read_text()
            )["calibration_aggregate_sha256"] == c1.CALIBRATION_SHA
            for key in CONFIGURATIONS
        ),
        "writers_never_active": all(
            row["writers_active_calls"] == 0 for row in stages.values()
        ),
        "no_auxiliary_objective": True,
        "no_bptt": True,
        "hellaswag_not_run": all(not row["hellaswag_run"] for row in stages.values()),
        "self_state_resume_equivalence": all(
            not row.get("triggered")
            or row.get("resume_preflight", {}).get("passed") is True
            for row in self_transfer.values()
        ),
        "all_preflights_passed": all(row["passed"] for row in preflights.values()),
        "all_smokes_passed_and_discarded": all(
            row["passed"] and row["states_discarded"] for row in smokes.values()
        ),
        "single_implementation_commit": len(implementation_commits) == 1,
    }
    integrity_passed = all(integrity.values())
    classification = classify(configuration_rows, integrity_passed)
    answers = scientific_answers(
        configuration_rows, progressive, router_stats, self_transfer
    )
    decisions = next_decisions(
        configuration_rows, classification, self_transfer
    )
    implementation_commit = next(iter(implementation_commits))
    summary = {
        "experiment": "2C2",
        "classification": classification,
        "classification_rule": classification_rule(classification),
        "integrity_passed": integrity_passed,
        "implementation_commit": implementation_commit,
        "evaluation_finalize_commits": sorted(
            {
                row["finalize_git_commit"]
                for row in stages.values()
                if row.get("finalize_git_commit")
            }
        ),
        "2c1_frozen_tag": PARENT_TAG,
        "2c1_parent_commit": PARENT_COMMIT,
        "2c2_branch": BRANCH,
        "base_checkpoint_sha256": a0.EXPECTED_PARENT_SHA256,
        "configurations": configuration_rows,
        "training_trajectories": trajectories,
        "progressive_reader_controls": progressive,
        "reader_ablation": ablations,
        "generic_controls": generic_controls,
        "router_stats": router_stats,
        "b1_query_cosines": b1_cosines,
        "self_transfer": self_transfer,
        "c1_regression": c1_regression,
        "scientific_answers": answers,
        "next_decisions": decisions,
        "optimizer_updates": {key: 48 for key in CONFIGURATIONS},
        "processed_targets_per_configuration": TARGET_UPDATE * GLOBAL_TARGETS,
        "follow_ons_launched": [],
        "config": config,
    }
    manifest = {
        "protocol": PROTOCOL,
        "configurations": {
            key: {
                "masked_blocks": [block + 1 for block in blocks],
                "feedback_destinations": [block + 1 for block in blocks],
                "gpu": GPU_MAPPING[key],
                "run_directory": RUN_NAMES[key],
                "trainable_parameters": PARAMETERS_PER_READER * len(blocks),
            }
            for key, blocks in CONFIGURATIONS.items()
        },
        "source_depths": list(SOURCE_DEPTHS),
        "same_source_bank_for_all_readers": True,
        "no_parameter_sharing": True,
        "no_ddp": True,
    }
    audit = {
        "experiment": "2C2",
        "classification": classification,
        "checks": integrity,
        "c1_regression": c1_regression,
        "passed": integrity_passed,
    }
    durable_json(output_dir / "result_summary.json", summary)
    durable_json(output_dir / "configuration_manifest.json", manifest)
    durable_json(output_dir / "training_trajectories.json", trajectories)
    durable_json(output_dir / "paired_losses.json", paired_losses)
    durable_json(output_dir / "progressive_reader_controls.json", progressive)
    durable_json(output_dir / "reader_ablation.json", ablations)
    durable_json(output_dir / "generic_controls.json", generic_controls)
    durable_json(output_dir / "router_stats.json", {
        "readers": router_stats,
        "b1_query_cosines": b1_cosines,
    })
    durable_json(output_dir / "self_transfer.json", self_transfer)
    durable_json(output_dir / "performance.json", performance)
    durable_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    durable_json(output_dir / "FINAL_AUDIT.json", audit)
    durable_text(output_dir / "EXPERIMENT_2C2_FINAL_REPORT.md", final_report_text(summary))
    if not integrity_passed:
        raise SystemExit(f"2C2 final audit failed: {integrity}")
    return summary


def render_final_report(args):
    require_git(clean=False)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "result_summary.json"
    audit_path = output_dir / "FINAL_AUDIT.json"
    if not summary_path.is_file() or not audit_path.is_file():
        raise SystemExit("final report rendering requires committed result summary and audit")
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed") or summary.get("integrity_passed") is not True:
        raise SystemExit("refusing to render a passing report from a failed audit")
    if git_output("rev-parse", f"{args.results_commit}^{{commit}}") != args.results_commit:
        raise SystemExit("results commit is not an exact local commit")
    summary["results_commit"] = args.results_commit
    report = final_report_text(summary)
    durable_text(output_dir / "EXPERIMENT_2C2_FINAL_REPORT.md", report)
    return {
        "experiment": "2C2",
        "results_commit": args.results_commit,
        "report": str((output_dir / "EXPERIMENT_2C2_FINAL_REPORT.md").resolve()),
        "ends_with_required_marker": report.endswith("# EXPERIMENT 2C2 COMPLETE"),
    }


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    preflight.add_argument("--parent-checkpoint", required=True)
    preflight.add_argument("--run-root", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    smoke.add_argument("--parent-checkpoint", required=True)
    smoke.add_argument("--run-root", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    train.add_argument("--parent-checkpoint", required=True)
    train.add_argument("--run-root", required=True)
    train.add_argument("--target-update", type=int, required=True)
    train.add_argument("--resume")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    finalize.add_argument("--parent-checkpoint", required=True)
    finalize.add_argument("--run-root", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-root", required=True)
    aggregate.add_argument("--output-dir", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--output-dir", required=True)
    report.add_argument("--results-commit", required=True)
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
    elif args.command == "report":
        report = render_final_report(args)
    else:
        a0.require_cuda()
        torch.cuda.manual_seed(a0.SEED)
        if args.command == "preflight":
            report = run_preflight(args)
        elif args.command == "smoke":
            report = run_smoke(args)
        elif args.command == "train":
            report = run_training(args)
        else:
            report = run_finalize(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
