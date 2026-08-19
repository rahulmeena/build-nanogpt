#!/usr/bin/env python3
"""Experiment 2C4R: path-consistent graded-KV-window self-recurrence rerun."""

import argparse
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
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2a0 as a0  # noqa: E402
import experiment_2c2 as c2  # noqa: E402
import experiment_2c3 as c3  # noqa: E402


BRANCH = "experiment-2c4r-graded-window-path-consistent-rerun"
PARENT_TAG = "experiment-2c4-graded-window-diagnostic-unstable-final"
PARENT_COMMIT = "f1f765c946e218ea3f3f54972133c7d29dceefb5"
PARENT_RESULTS = "c2d1a73b8a6a43ad93ea2ba7dc036595cdae9459"
SOURCE_2C3_TAG = "experiment-2c3-cumulative-reader-scaling-100m-final"
SOURCE_2C3_COMMIT = "8b1af7e14d1547417e799ac02fe0d513b0755f6e"
SOURCE_2C3_IMPLEMENTATION = "792dc701f29b449c611b0a524ef4277a5f982403"
SOURCE_2C3_RESULTS = "fc0b4acda5d252cdbe7fc5d3781edffba7520586"
PROTOCOL = "exp2c4r_graded_window_path_consistent_zeroopt_v1"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2c4r_graded_window_path_consistent.json"
SOURCE_RESULTS = REPO_ROOT / "results" / "experiment_2c3_cumulative_reader_scaling_100m"
SOURCE_CHECKPOINT_MANIFEST = SOURCE_RESULTS / "checkpoint_manifest.json"
FAILED_2C4_RESULTS = REPO_ROOT / "results" / "experiment_2c4_graded_window_self_rescue"
OUTPUT_NAME = "experiment_2c4r_graded_window_path_consistent"
ORACLE_PATH = REPO_ROOT / "results" / OUTPUT_NAME / "path_consistent_oracle.json"
SOURCE_RUN_ROOT = Path("/workspace/exp2c3_run_792dc70")
SOURCE_CONFIGURATION = "C4"
SOURCE_UPDATE = 191
SOURCE_TARGETS = 100_139_008
SOURCE_SHA = "fce81b995543c42821abd080f615bcb5d2f755f113345988aa24d07b265b0447"
BASE_SHA = "6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91"
CANONICAL_SHA = "3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb"
SOURCE_DEPTHS = (16, 17, 20, 24)
SOURCE_LABELS = ("v16", "v17", "v20", "v24")
RECEIVER_LABELS = ("B1", "B2", "B3", "B4")
SCHEDULES = {
    "S0": (1, 1, 1, 1) + (1024,) * 8,
    "S1": (1, 2, 4, 8) + (1024,) * 8,
    "S2": (1, 4, 16, 64) + (1024,) * 8,
    "S3": (1, 8, 64, 256) + (1024,) * 8,
}
RUN_NAMES = {
    "S0": "S0_1_1_1_1",
    "S1": "S1_1_2_4_8",
    "S2": "S2_1_4_16_64",
    "S3": "S3_1_8_64_256",
}
GPU_MAPPING = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}
CONTROLS = (
    "no_feedback",
    "teacher_real",
    "teacher_shuffled",
    "self_B1_only",
    "self_real",
    "self_shuffled",
)
POSITION_BINS = (
    ("1-16", 1, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513-1023", 513, 1023),
)
REGRESSION_ATOL = c2.C1_ATOL
SHORT_GATE_BATCHES = 2


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def file_sha256(path):
    return c2.file_sha256(Path(path))


def durable_json(path, payload):
    c2.durable_json(Path(path), payload)


def durable_text(path, text):
    c2.durable_text(Path(path), text)


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2C4R requires branch {BRANCH}")
    if git_output("rev-parse", f"{PARENT_TAG}^{{}}") != PARENT_COMMIT:
        raise SystemExit("frozen Experiment 2C4 tag target mismatch")
    if git_output("rev-parse", f"{SOURCE_2C3_TAG}^{{}}") != SOURCE_2C3_COMMIT:
        raise SystemExit("frozen Experiment 2C3 tag target mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2C4R execution requires a clean worktree")


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        "protocol": PROTOCOL,
        "2c4_final_tag": PARENT_TAG,
        "2c4_final_commit": PARENT_COMMIT,
        "2c4_results_commit": PARENT_RESULTS,
        "2c3_final_tag": SOURCE_2C3_TAG,
        "2c3_final_commit": SOURCE_2C3_COMMIT,
        "2c3_implementation_commit": SOURCE_2C3_IMPLEMENTATION,
        "2c3_results_commit": SOURCE_2C3_RESULTS,
        "source_configuration": SOURCE_CONFIGURATION,
        "source_reader_updates": SOURCE_UPDATE,
        "source_reader_targets": SOURCE_TARGETS,
        "source_checkpoint_sha256": SOURCE_SHA,
        "base_checkpoint_sha256": BASE_SHA,
        "canonical_validation_sha256": CANONICAL_SHA,
        "source_depths": list(SOURCE_DEPTHS),
        "reader_destinations": [1, 2, 3, 4],
        "schedules": {key: list(value) for key, value in SCHEDULES.items()},
        "gpu_mapping": GPU_MAPPING,
        "controls": list(CONTROLS),
        "validation_batches": 20,
        "validation_batch_size": 64,
        "validation_sequence_length": 1024,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "grad_scaler_objects": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_targets": 0,
        "execution_path": "incremental_decoder_plus_rolling_kv_plus_recurrent_state",
        "s0_regression_absolute_tolerance": REGRESSION_ATOL,
        "s0_short_gate_batches": SHORT_GATE_BATCHES,
        "writers": "forbidden",
        "inner_loops": "forbidden",
        "bptt": "forbidden",
        "hellaswag": "forbidden",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"2C4R preregistration mismatch: {mismatches}")
    return config


def expected_oracle_payload():
    source_files = (
        "canonical_losses.json",
        "result_summary.json",
        "paired_self_vs_no_feedback.json",
        "paired_real_vs_shuffled.json",
        "paired_b1_vs_all_readers.json",
        "teacher_controls.json",
    )
    summary = json.loads((FAILED_2C4_RESULTS / "result_summary.json").read_text())
    row = summary["rows"]["S0"]
    aggregate = {
        "no_feedback": row["losses"]["no_feedback"],
        "teacher_real": row["losses"]["teacher_real"],
        "teacher_shuffled": row["losses"]["teacher_shuffled"],
        "teacher_gap": row["teacher_specific_gap"],
        "teacher_recovery": row["teacher_recovery"],
        "self_real": row["losses"]["self_real"],
        "self_shuffled": row["losses"]["self_shuffled"],
        "self_gap": row["self_specific_gap"],
        "self_recovery": row["self_recovery"],
        "self_B1_only": row["losses"]["self_B1_only"],
        "self_matched_gain": row["self_matched_gain"],
    }
    per_batch = {
        control: list(row["per_batch_losses"][control]) for control in CONTROLS
    }
    return {
        "experiment": "2C4R",
        "protocol": PROTOCOL,
        "execution_path": "incremental_decoder_plus_rolling_kv_plus_recurrent_state",
        "source_experiment": "2C4",
        "source_final_tag": PARENT_TAG,
        "source_final_commit": PARENT_COMMIT,
        "source_results_commit": PARENT_RESULTS,
        "source_result_directory": str(FAILED_2C4_RESULTS.relative_to(REPO_ROOT)),
        "source_artifact_sha256": {
            name: file_sha256(FAILED_2C4_RESULTS / name) for name in source_files
        },
        "absolute_tolerance": REGRESSION_ATOL,
        "windows": list(SCHEDULES["S0"]),
        "expected_aggregate": aggregate,
        "expected_per_batch": per_batch,
        "expected_first_two_batches": {
            control: values[:SHORT_GATE_BATCHES]
            for control, values in per_batch.items()
        },
        "expected_runtime": row["runtime"],
    }


def write_oracle(args):
    require_git(clean=False)
    load_config()
    payload = expected_oracle_payload()
    output = Path(args.output).resolve()
    if output != ORACLE_PATH.resolve():
        raise SystemExit(f"2C4R oracle must be written to {ORACLE_PATH}")
    output.parent.mkdir(parents=True, exist_ok=True)
    durable_json(output, payload)
    print(f"EXPERIMENT_2C4R_ORACLE_WRITTEN path={output}", flush=True)
    return payload


def load_oracle():
    if not ORACLE_PATH.is_file():
        raise SystemExit("path-consistent oracle missing; write and commit it before evaluation")
    observed = json.loads(ORACLE_PATH.read_text())
    expected = expected_oracle_payload()
    if observed != expected:
        raise SystemExit("path-consistent oracle does not exactly match frozen 2C4 artifacts")
    return observed


def schedule_dir(run_root, schedule):
    return Path(run_root) / RUN_NAMES[schedule]


def source_manifest_row():
    manifest = json.loads(SOURCE_CHECKPOINT_MANIFEST.read_text())
    row = manifest[SOURCE_CONFIGURATION][str(SOURCE_UPDATE)]
    required = {
        "completed_updates": row.get("completed_updates") == SOURCE_UPDATE,
        "processed_targets": row.get("processed_targets") == SOURCE_TARGETS,
        "sha256": row.get("sha256") == SOURCE_SHA,
        "passed": row.get("passed") is True,
        "sha256_reverified": row.get("sha256_reverified") is True,
        "strict_reopen": row.get("strict_reopen", {}).get("passed") is True,
    }
    if not all(required.values()):
        raise SystemExit(f"canonical C4@100M manifest mismatch: {required}")
    return row, required


def runtime_settings():
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "autocast": "torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
        "canonical_dtype": "torch.bfloat16",
        "python_seed": a0.SEED,
        "numpy_seed": a0.SEED,
        "torch_cpu_seed": a0.SEED,
        "torch_cuda_seed": a0.SEED,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def load_runtime(parent_checkpoint):
    row, manifest_checks = source_manifest_row()
    source_path = Path(row["checkpoint"]).resolve()
    if not source_path.is_file():
        raise SystemExit(f"canonical C4@100M checkpoint missing: {source_path}")
    if file_sha256(source_path) != SOURCE_SHA:
        raise SystemExit("canonical C4@100M checkpoint SHA mismatch")
    args = SimpleNamespace(
        run_root=str(SOURCE_RUN_ROOT), parent_checkpoint=str(Path(parent_checkpoint).resolve())
    )
    symbols, teacher, student, payload, verification, lineage = c3.load_final_reader(
        args, SOURCE_CONFIGURATION
    )
    required = {
        **manifest_checks,
        **{f"load_{key}": value for key, value in lineage.items()},
        "configuration": payload.get("configuration") == SOURCE_CONFIGURATION,
        "destinations": payload.get("active_destinations") == [1, 2, 3, 4],
        "source_depths": payload.get("source_depths") == list(SOURCE_DEPTHS),
        "reader_updates": payload.get("reader_update_count") == SOURCE_UPDATE,
        "reader_targets": payload.get("processed_targets") == SOURCE_TARGETS,
        "checkpoint_sha": verification.get("sha256") == SOURCE_SHA,
        "base_checkpoint_sha": file_sha256(parent_checkpoint) == BASE_SHA,
        "reader_sha": c2.reader_state_sha(student) == payload["reader_state_sha256"],
        "writers_absent": "memory_writers" not in student.transformer,
    }
    if not all(required.values()):
        raise SystemExit(f"2C4R source strict-load mismatch: {required}")
    for model in (student, teacher):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in student.parameters()):
        raise SystemExit("2C4R requires every student parameter frozen")
    return symbols, teacher, student, payload, {
        "checkpoint": str(source_path),
        "checkpoint_sha256": SOURCE_SHA,
        "configuration": SOURCE_CONFIGURATION,
        "completed_updates": SOURCE_UPDATE,
        "processed_targets": SOURCE_TARGETS,
        "reader_state_sha256": payload["reader_state_sha256"],
        "base_checkpoint": str(Path(parent_checkpoint).resolve()),
        "base_checkpoint_sha256": BASE_SHA,
        "strict_load": required,
    }


def direct_feedback(student, memory_bank, active_blocks):
    result = {}
    readers = c2.readers(student)
    for block in active_blocks:
        reader = readers[str(block)]
        topdown = reader(list(memory_bank.unbind(dim=0)))
        result[block] = reader.gate.tanh() * topdown
    return result


def new_state(student, batch_size, windows, dtype, mode="masked_l1_no_feedback"):
    return student.init_recurrent_state(
        batch_size,
        mode,
        device="cuda",
        dtype=dtype,
        mask_depth=4 if mode != "full_context" else 0,
        attention_windows=windows,
    )


def expected_cache_lengths(position, windows):
    return [min(position, window - 1) for window in windows]


def cache_lengths(state):
    return [0 if cache is None else cache.length for cache in state.kv_caches]


def cache_capacities(state):
    return [0 if cache is None else cache.key.size(2) for cache in state.kv_caches]


def assert_cache_state(state, windows):
    expected_lengths = expected_cache_lengths(state.position, windows)
    for block, (cache, window, expected_length) in enumerate(
        zip(state.kv_caches, windows, expected_lengths)
    ):
        if window == 1:
            if cache is not None:
                raise SystemExit(f"B{block + 1} window-1 hidden KV cache detected")
        elif (
            cache is None
            or cache.key.size(2) != window - 1
            or cache.value.size(2) != window - 1
            or cache.length != expected_length
        ):
            raise SystemExit(f"B{block + 1} rolling KV audit failed")


def recurrent_smoke_logits(student, tokens, windows, control, teacher_sources=None):
    state = new_state(student, tokens.size(0), windows, tokens.dtype if tokens.is_floating_point() else torch.float32)
    permutation = torch.arange(tokens.size(0), device=tokens.device).roll(1)
    rows = []
    for position in range(tokens.size(1)):
        active = ()
        bank = state.feedback_memory.detach()
        if control == "teacher_real":
            active = (0, 1, 2, 3)
            bank = teacher_sources[:, :, position : position + 1]
        elif control == "teacher_shuffled":
            active = (0, 1, 2, 3)
            bank = teacher_sources[:, permutation, position : position + 1]
        elif control == "self_real":
            active = (0, 1, 2, 3)
        elif control == "self_shuffled":
            active = (0, 1, 2, 3)
            bank = bank[:, permutation]
        elif control == "self_B1_only":
            active = (0,)
        feedback = direct_feedback(student, bank, active)
        logits, state = student.forward_step(
            tokens[:, position], state, attention_feedback_by_block=feedback
        )
        rows.append(logits)
        assert_cache_state(state, windows)
    return torch.cat(rows, dim=1), state


@torch.no_grad()
def short_schedule_tests(student, teacher, symbols, schedule):
    windows = SCHEDULES[schedule]
    reports = {}
    for dtype_name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        reports[dtype_name] = {}
        for length in (8, 16, 64):
            loader = symbols["DataLoaderLite"](
                B=2, T=length, process_rank=0, num_processes=1, split="val"
            )
            tokens, _ = loader.next_batch()
            tokens = tokens.cuda()
            state = new_state(student, 2, windows, dtype)
            finite_logits = True
            finite_memory = True
            exact_lengths = True
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if dtype == torch.bfloat16
                else torch.autocast(device_type="cuda", enabled=False)
            )
            with context:
                for position in range(length):
                    logits, state = student.forward_step(tokens[:, position], state)
                    finite_logits &= bool(torch.isfinite(logits).all())
                    finite_memory &= bool(torch.isfinite(state.feedback_memory).all())
                    exact_lengths &= cache_lengths(state) == expected_cache_lengths(
                        position + 1, windows
                    )
            reports[dtype_name][str(length)] = {
                "all_logits_finite": finite_logits,
                "all_memory_finite": finite_memory,
                "cache_lengths_exact": exact_lengths,
                "absolute_position_exact": state.position == length,
                "capacities": cache_capacities(state),
                "passed": finite_logits
                and finite_memory
                and exact_lengths
                and state.position == length,
            }
    return reports


@torch.no_grad()
def serialization_test(student, symbols, schedule):
    windows = SCHEDULES[schedule]
    loader = symbols["DataLoaderLite"](
        B=2, T=8, process_rank=0, num_processes=1, split="val"
    )
    tokens, _ = loader.next_batch()
    tokens = tokens.cuda()
    state = new_state(student, 2, windows, torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(4):
            feedback = direct_feedback(student, state.feedback_memory.detach(), (0, 1, 2, 3))
            _, state = student.forward_step(
                tokens[:, position], state, attention_feedback_by_block=feedback
            )
        payload = state.state_dict()
        restored = student.load_recurrent_state(
            payload, device="cuda", dtype=torch.bfloat16
        )
        feedback_a = direct_feedback(student, state.feedback_memory.detach(), (0, 1, 2, 3))
        feedback_b = direct_feedback(student, restored.feedback_memory.detach(), (0, 1, 2, 3))
        logits_a, next_a = student.forward_step(
            tokens[:, 4], state, attention_feedback_by_block=feedback_a
        )
        logits_b, next_b = student.forward_step(
            tokens[:, 4], restored, attention_feedback_by_block=feedback_b
        )
    fresh = new_state(student, 2, windows, torch.bfloat16)
    report = {
        "schema": payload.get("schema"),
        "windows_exact": payload.get("attention_windows") == list(windows),
        "resume_logits_bit_exact": torch.equal(logits_a, logits_b),
        "resume_state_bit_exact": a0.nested_equal(next_a.state_dict(), next_b.state_dict()),
        "fresh_position_zero": fresh.position == 0,
        "fresh_memory_zero": fresh.feedback_memory.count_nonzero().item() == 0,
        "fresh_caches_empty": cache_lengths(fresh) == [0] * 12,
    }
    report["passed"] = all(value for key, value in report.items() if key not in {"schema", "passed"})
    return report


@torch.no_grad()
def full_window_equivalence(student, symbols):
    rows = {}
    full_windows = (1024,) * 12
    for dtype_name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        rows[dtype_name] = {}
        for length in (8, 16, 64):
            loader = symbols["DataLoaderLite"](
                B=2, T=length, process_rank=0, num_processes=1, split="val"
            )
            tokens, _ = loader.next_batch()
            tokens = tokens.cuda()
            old = student.init_recurrent_state(
                2, "full_context", device="cuda", dtype=dtype, mask_depth=0
            )
            rolling = new_state(
                student, 2, full_windows, dtype, mode="full_context"
            )
            old_rows = []
            rolling_rows = []
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if dtype == torch.bfloat16
                else torch.autocast(device_type="cuda", enabled=False)
            )
            with context:
                for position in range(length):
                    old_logits, old = student.forward_step(tokens[:, position], old)
                    rolling_logits, rolling = student.forward_step(
                        tokens[:, position], rolling
                    )
                    old_rows.append(old_logits)
                    rolling_rows.append(rolling_logits)
            old_logits = torch.cat(old_rows, dim=1)
            rolling_logits = torch.cat(rolling_rows, dim=1)
            maximum = (old_logits.float() - rolling_logits.float()).abs().max().item()
            rows[dtype_name][str(length)] = {
                "bit_exact": torch.equal(old_logits, rolling_logits),
                "maximum_absolute_logit_difference": maximum,
                "passed": torch.equal(old_logits, rolling_logits),
            }
    return rows


@torch.no_grad()
def causality_and_row_tests(student, teacher, symbols, schedule):
    windows = SCHEDULES[schedule]
    loader = symbols["DataLoaderLite"](
        B=4, T=16, process_rank=0, num_processes=1, split="val"
    )
    tokens, _ = loader.next_batch()
    tokens = tokens.cuda()
    future = tokens.clone()
    future[:, 8:] = (future[:, 8:] + 19) % student.config.vocab_size
    row = tokens.clone()
    row[1] = (row[1] + 23) % student.config.vocab_size

    def teacher_bank(value):
        diagnostics = teacher.capture_full_context_diagnostics(value)
        return symbols["shift_teacher_sources"](diagnostics["sources"])

    results = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        teacher_tokens = teacher_bank(tokens)
        teacher_future = teacher_bank(future)
        for control in CONTROLS:
            reference, _ = recurrent_smoke_logits(
                student,
                tokens,
                windows,
                control,
                teacher_sources=teacher_tokens,
            )
            changed, _ = recurrent_smoke_logits(
                student,
                future,
                windows,
                control,
                teacher_sources=teacher_future,
            )
            results[control] = torch.equal(reference[:, :8], changed[:, :8])
        reference, _ = recurrent_smoke_logits(
            student, tokens, windows, "self_real", teacher_sources=teacher_tokens
        )
        row_logits, _ = recurrent_smoke_logits(
            student, row, windows, "self_real", teacher_sources=teacher_tokens
        )
        donor_reference = new_state(student, 4, windows, torch.bfloat16)
        for position in range(4):
            feedback = direct_feedback(
                student, donor_reference.feedback_memory.detach(), (0, 1, 2, 3)
            )
            _, donor_reference = student.forward_step(
                tokens[:, position],
                donor_reference,
                attention_feedback_by_block=feedback,
            )
        donor_perturbed = student.load_recurrent_state(
            donor_reference.state_dict(), device="cuda", dtype=torch.bfloat16
        )
        donor_perturbed.feedback_memory[:, 1].add_(0.25)
        permutation = torch.arange(4, device="cuda").roll(1)
        reference_feedback = direct_feedback(
            student, donor_reference.feedback_memory[:, permutation], (0, 1, 2, 3)
        )
        perturbed_feedback = direct_feedback(
            student, donor_perturbed.feedback_memory[:, permutation], (0, 1, 2, 3)
        )
        shuffled_reference, _ = student.forward_step(
            tokens[:, 4],
            donor_reference,
            attention_feedback_by_block=reference_feedback,
        )
        shuffled_perturbed, _ = student.forward_step(
            tokens[:, 4],
            donor_perturbed,
            attention_feedback_by_block=perturbed_feedback,
        )
    donor_receivers = (permutation == 1).nonzero().flatten().tolist()
    shuffled_isolation = all(
        index in donor_receivers
        or torch.equal(shuffled_reference[index], shuffled_perturbed[index])
        for index in range(4)
    ) and all(
        not torch.equal(shuffled_reference[index], shuffled_perturbed[index])
        for index in donor_receivers
    )
    report = {
        "future_prefix_by_control": results,
        "future_causality_all_controls": all(results.values()),
        "real_row_isolation_bit_exact": torch.equal(reference[0], row_logits[0]),
        "shuffled_only_explicit_donor_dependence": shuffled_isolation,
        "fixed_derangement": permutation.cpu().tolist(),
    }
    report["passed"] = (
        report["future_causality_all_controls"]
        and report["real_row_isolation_bit_exact"]
        and report["shuffled_only_explicit_donor_dependence"]
    )
    return report


def run_preflight(args):
    require_git(clean=True)
    config = load_config()
    oracle = load_oracle()
    schedule = args.schedule
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(GPU_MAPPING[schedule]):
        raise SystemExit(f"{schedule} preflight requires CUDA_VISIBLE_DEVICES={GPU_MAPPING[schedule]}")
    symbols, teacher, student, payload, source = load_runtime(args.parent_checkpoint)
    before = {
        "student": c2.state_hash(student),
        "teacher": c2.state_hash(teacher),
        "reader": c2.reader_state_sha(student),
    }
    short = short_schedule_tests(student, teacher, symbols, schedule)
    serialization = serialization_test(student, symbols, schedule)
    equivalence = full_window_equivalence(student, symbols)
    causality = causality_and_row_tests(student, teacher, symbols, schedule)
    loader = symbols["DataLoaderLite"](
        B=2, T=16, process_rank=0, num_processes=1, split="val"
    )
    tokens, _ = loader.next_batch()
    tokens = tokens.cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        old_sources = teacher.capture_residual_sources(tokens, SOURCE_DEPTHS)
        diagnostics = teacher.capture_full_context_diagnostics(tokens)
    diagnostic_source_exact = torch.equal(old_sources, diagnostics["sources"])
    after = {
        "student": c2.state_hash(student),
        "teacher": c2.state_hash(teacher),
        "reader": c2.reader_state_sha(student),
    }
    checks = {
        "short_fp32_bf16": all(
            row["passed"] for dtype_rows in short.values() for row in dtype_rows.values()
        ),
        "serialization_resume": serialization["passed"],
        "full_window_equivalence": all(
            row["passed"] for dtype_rows in equivalence.values() for row in dtype_rows.values()
        ),
        "future_causality_row_isolation": causality["passed"],
        "diagnostic_source_capture_exact": diagnostic_source_exact,
        "model_hashes_unchanged": before == after,
        "path_consistent_oracle_exact": oracle == expected_oracle_payload(),
        "optimizer_objects_zero": True,
        "scheduler_objects_zero": True,
        "grad_scaler_objects_zero": True,
        "all_parameters_frozen": not any(
            parameter.requires_grad for parameter in student.parameters()
        ),
        "teacher_eval_no_grad": not teacher.training
        and not any(parameter.requires_grad for parameter in teacher.parameters()),
        "writers_absent": "memory_writers" not in student.transformer,
    }
    report = {
        "experiment": "2C4R",
        "stage": "preflight",
        "schedule": schedule,
        "windows": list(SCHEDULES[schedule]),
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "config": config,
        "path_consistent_oracle_sha256": file_sha256(ORACLE_PATH),
        "source": source,
        "runtime": runtime_settings(),
        "hardware": {
            "physical_gpu": GPU_MAPPING[schedule],
            "CUDA_VISIBLE_DEVICES": visible,
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "short_tests": short,
        "serialization": serialization,
        "full_window_equivalence": equivalence,
        "causality_and_row_isolation": causality,
        "hashes_before": before,
        "hashes_after": after,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not report["passed"]:
        raise SystemExit(f"2C4R {schedule} preflight failed: {report}")
    directory = schedule_dir(args.run_root, schedule)
    directory.mkdir(parents=True, exist_ok=True)
    durable_json(directory / "preflight.json", report)
    print(f"EXPERIMENT_2C4R_PREFLIGHT_PASS schedule={schedule}", flush=True)
    return report


def new_drift_accumulator(labels):
    return {
        label: {
            bin_name: {
                "cosine_sum": 0.0,
                "rows": 0,
                "difference_square_sum": 0.0,
                "student_square_sum": 0.0,
                "teacher_square_sum": 0.0,
                "elements": 0,
            }
            for bin_name, _, _ in POSITION_BINS
        }
        for label in labels
    }


def position_bin(position):
    for name, start, end in POSITION_BINS:
        if start <= position <= end:
            return name
    return None


def update_drift(accumulator, labels, student_states, teacher_states, position):
    bin_name = position_bin(position)
    if bin_name is None:
        return
    student_states = student_states.squeeze(2).float()
    teacher_states = teacher_states.float()
    for index, label in enumerate(labels):
        student = student_states[index]
        teacher = teacher_states[index]
        row = accumulator[label][bin_name]
        row["cosine_sum"] = row["cosine_sum"] + F.cosine_similarity(
            student, teacher, dim=-1, eps=1e-8
        ).double().sum()
        row["rows"] += student.size(0)
        difference = student - teacher
        row["difference_square_sum"] = (
            row["difference_square_sum"] + difference.double().square().sum()
        )
        row["student_square_sum"] = (
            row["student_square_sum"] + student.double().square().sum()
        )
        row["teacher_square_sum"] = (
            row["teacher_square_sum"] + teacher.double().square().sum()
        )
        row["elements"] += student.numel()


def finalize_drift(accumulator, schedule, dimension_name):
    rows = []
    total = {
        "cosine_sum": 0.0,
        "rows": 0,
        "difference_square_sum": 0.0,
        "student_square_sum": 0.0,
        "teacher_square_sum": 0.0,
        "elements": 0,
    }
    for label, bins in accumulator.items():
        for bin_name, raw in bins.items():
            numeric = {
                key: value.item() if isinstance(value, torch.Tensor) else value
                for key, value in raw.items()
            }
            cosine = numeric["cosine_sum"] / numeric["rows"]
            rms = math.sqrt(numeric["difference_square_sum"] / numeric["elements"])
            student_rms = math.sqrt(numeric["student_square_sum"] / numeric["elements"])
            teacher_rms = math.sqrt(numeric["teacher_square_sum"] / numeric["elements"])
            rows.append(
                {
                    "schedule": schedule,
                    dimension_name: label,
                    "position_bin": bin_name,
                    "teacher_self_cosine": cosine,
                    "rms_difference": rms,
                    "student_teacher_rms_ratio": student_rms / teacher_rms,
                    "rows": numeric["rows"],
                    "elements": numeric["elements"],
                    "cosine_sum": numeric["cosine_sum"],
                    "difference_square_sum": numeric["difference_square_sum"],
                    "student_square_sum": numeric["student_square_sum"],
                    "teacher_square_sum": numeric["teacher_square_sum"],
                }
            )
            for key in total:
                total[key] += numeric[key]
    aggregate = {
        "mean_cosine": total["cosine_sum"] / total["rows"],
        "mean_rms_difference": math.sqrt(
            total["difference_square_sum"] / total["elements"]
        ),
        "mean_norm_ratio": math.sqrt(
            total["student_square_sum"] / total["elements"]
        )
        / math.sqrt(total["teacher_square_sum"] / total["elements"]),
        "rows": total["rows"],
        "elements": total["elements"],
    }
    return rows, aggregate


def aggregate_drift_by_label(rows, dimension_name):
    grouped = {}
    for row in rows:
        key = (row["schedule"], row[dimension_name])
        raw = grouped.setdefault(
            key,
            {
                "cosine_sum": 0.0,
                "rows": 0,
                "difference_square_sum": 0.0,
                "student_square_sum": 0.0,
                "teacher_square_sum": 0.0,
                "elements": 0,
            },
        )
        for field in raw:
            raw[field] += row[field]
    output = []
    for (schedule, label), raw in sorted(grouped.items()):
        output.append(
            {
                "schedule": schedule,
                dimension_name: label,
                "mean_teacher_student_cosine": raw["cosine_sum"] / raw["rows"],
                "rms_difference": math.sqrt(
                    raw["difference_square_sum"] / raw["elements"]
                ),
                "norm_ratio": math.sqrt(
                    raw["student_square_sum"] / raw["elements"]
                )
                / math.sqrt(raw["teacher_square_sum"] / raw["elements"]),
                "rows": raw["rows"],
                "elements": raw["elements"],
            }
        )
    return output


def paired(left, right, left_label, right_label):
    report = c2.paired_statistics(left, right)
    return {
        "left_label": left_label,
        "right_label": right_label,
        f"{left_label}_wins": report["real_wins"],
        f"{right_label}_wins": report["shuffled_wins"],
        "ties": report["ties"],
        "mean_right_minus_left": report["mean_gap"],
        "median_right_minus_left": report["median_gap"],
        "sample_std": report["sample_std"],
        "minimum": report["minimum"],
        "maximum": report["maximum"],
        "standard_error": report["standard_error"],
        "ci95_lower": report["ci95_lower"],
        "ci95_upper": report["ci95_upper"],
        "differences": report["gaps"],
    }


@torch.no_grad()
def evaluate_schedule(args):
    require_git(clean=True)
    load_config()
    oracle = load_oracle()
    schedule = args.schedule
    gate_only = args.command == "gate-s0"
    if gate_only and schedule != "S0":
        raise SystemExit("the 2C4R two-batch gate is defined only for S0")
    batch_count = SHORT_GATE_BATCHES if gate_only else a0.VALIDATION_BATCHES
    windows = SCHEDULES[schedule]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(GPU_MAPPING[schedule]):
        raise SystemExit(f"{schedule} evaluation requires CUDA_VISIBLE_DEVICES={GPU_MAPPING[schedule]}")
    directory = schedule_dir(args.run_root, schedule)
    preflight_path = directory / "preflight.json"
    if not preflight_path.is_file() or not json.loads(preflight_path.read_text()).get("passed"):
        raise SystemExit(f"{schedule} requires a passing preflight")
    gate_path = Path(args.run_root) / "s0_two_batch_gate.json"
    if not gate_only:
        if not gate_path.is_file():
            raise SystemExit("canonical 2C4R evaluation requires the saved two-batch S0 gate")
        gate = json.loads(gate_path.read_text())
        if not gate.get("passed") or gate.get("oracle_sha256") != file_sha256(ORACLE_PATH):
            raise SystemExit("canonical 2C4R evaluation requires the exact passing S0 gate")
    symbols, teacher, student, payload, source = load_runtime(args.parent_checkpoint)
    before = {
        "student": c2.state_hash(student),
        "teacher": c2.state_hash(teacher),
        "reader": c2.reader_state_sha(student),
    }
    student.eval()
    teacher.eval()
    loader = c2.validation_loader(symbols)
    permutation = symbols["fixed_derangement"](a0.VALIDATION_B, "cuda")
    losses = {control: [] for control in CONTROLS}
    validation_hash = hashlib.sha256()
    source_drift = new_drift_accumulator(SOURCE_LABELS)
    receiver_drift = new_drift_accumulator(RECEIVER_LABELS)
    max_cache_lengths = [0] * 12
    control_seconds = {control: 0.0 for control in CONTROLS}
    process_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    process_started_unix = time.time()
    for batch_index in range(batch_count):
        x_cpu, y_cpu = loader.next_batch()
        validation_hash.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            teacher_diagnostics = teacher.capture_full_context_diagnostics(
                x, SOURCE_DEPTHS, (0, 1, 2, 3)
            )
        shifted_teacher = symbols["shift_teacher_sources"](
            teacher_diagnostics["sources"]
        )
        if (
            teacher.training
            or shifted_teacher.requires_grad
            or shifted_teacher.grad_fn is not None
        ):
            raise SystemExit("teacher diagnostic memory must be detached eval/no_grad")
        for control in CONTROLS:
            state = new_state(student, x.size(0), windows, torch.bfloat16)
            token_loss = 0.0
            torch.cuda.synchronize()
            started = time.perf_counter()
            for position in range(x.size(1)):
                active = ()
                bank = state.feedback_memory.detach()
                if control == "teacher_real":
                    active = (0, 1, 2, 3)
                    bank = shifted_teacher[:, :, position : position + 1]
                elif control == "teacher_shuffled":
                    active = (0, 1, 2, 3)
                    bank = shifted_teacher[:, permutation, position : position + 1]
                elif control == "self_B1_only":
                    active = (0,)
                elif control == "self_real":
                    active = (0, 1, 2, 3)
                elif control == "self_shuffled":
                    active = (0, 1, 2, 3)
                    bank = bank[:, permutation]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feedback = direct_feedback(student, bank, active)
                    if control == "self_real":
                        logits, state, diagnostics = student.forward_step(
                            x[:, position],
                            state,
                            attention_feedback_by_block=feedback,
                            return_diagnostics=True,
                        )
                        update_drift(
                            source_drift,
                            SOURCE_LABELS,
                            diagnostics["source_memory"],
                            teacher_diagnostics["sources"][:, :, position],
                            position,
                        )
                        update_drift(
                            receiver_drift,
                            RECEIVER_LABELS,
                            diagnostics["receiver_states"],
                            teacher_diagnostics["receivers"][:, :, position],
                            position,
                        )
                    else:
                        logits, state = student.forward_step(
                            x[:, position],
                            state,
                            attention_feedback_by_block=feedback,
                        )
                    token_loss += F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y[:, position].reshape(-1),
                    ).detach().double().item()
                assert_cache_state(state, windows)
                current_lengths = cache_lengths(state)
                max_cache_lengths = [
                    max(old, current)
                    for old, current in zip(max_cache_lengths, current_lengths)
                ]
            if (
                not torch.isfinite(state.feedback_memory).all()
                or not all(
                    cache is None
                    or (
                        torch.isfinite(cache.key[:, :, : cache.length]).all()
                        and torch.isfinite(cache.value[:, :, : cache.length]).all()
                    )
                    for cache in state.kv_caches
                )
            ):
                raise SystemExit("non-finite recurrent memory/KV detected")
            torch.cuda.synchronize()
            control_seconds[control] += time.perf_counter() - started
            losses[control].append(token_loss / x.size(1))
        del x, y, teacher_diagnostics, shifted_teacher
        print(
            f"2C4R {schedule} {'gate' if gate_only else 'canonical'} batch "
            f"{batch_index + 1:02d}/{batch_count}",
            flush=True,
        )
    digest = validation_hash.hexdigest()
    if not gate_only and digest != CANONICAL_SHA:
        raise SystemExit(f"canonical validation hash mismatch: {digest}")
    means = {name: statistics.fmean(values) for name, values in losses.items()}
    if not all(math.isfinite(value) for value in means.values()):
        raise SystemExit("non-finite 2C4R canonical loss")
    source_rows, source_aggregate = finalize_drift(
        source_drift, schedule, "source"
    )
    receiver_rows, receiver_aggregate = finalize_drift(
        receiver_drift, schedule, "receiver"
    )
    teacher_recovery = means["no_feedback"] - means["teacher_real"]
    self_recovery = means["no_feedback"] - means["self_real"]
    teacher_gap = means["teacher_shuffled"] - means["teacher_real"]
    self_gap = means["self_shuffled"] - means["self_real"]
    matched_gain = means["self_B1_only"] - means["self_real"]
    if gate_only:
        paired_self_no_feedback = None
        paired_real_shuffle = None
        paired_all_b1 = None
    else:
        paired_self_no_feedback = paired(
            losses["self_real"], losses["no_feedback"], "self_real", "no_feedback"
        )
        paired_real_shuffle = paired(
            losses["self_real"], losses["self_shuffled"], "self_real", "self_shuffled"
        )
        paired_all_b1 = paired(
            losses["self_real"], losses["self_B1_only"], "all_readers", "B1_only"
        )
    after = {
        "student": c2.state_hash(student),
        "teacher": c2.state_hash(teacher),
        "reader": c2.reader_state_sha(student),
    }
    expected_max = [window - 1 for window in windows]
    process_seconds = time.perf_counter() - process_started
    result = {
        "experiment": "2C4R",
        "schedule": schedule,
        "windows": list(windows),
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "execution_path": oracle["execution_path"],
        "path_consistent_oracle_sha256": file_sha256(ORACLE_PATH),
        "evaluated_batches": batch_count,
        "source": source,
        "canonical_validation_sha256": digest,
        "losses": means,
        "per_batch_losses": losses,
        "teacher_recovery": teacher_recovery,
        "teacher_specific_gap": teacher_gap,
        "self_recovery": self_recovery,
        "self_specific_gap": self_gap,
        "self_matched_gain": matched_gain,
        "self_teacher_recovery_ratio": (
            self_recovery / teacher_recovery if teacher_recovery > 0 else None
        ),
        "paired_self_vs_no_feedback": paired_self_no_feedback,
        "paired_real_vs_shuffled": paired_real_shuffle,
        "paired_all_readers_vs_B1_only": paired_all_b1,
        "source_drift": source_rows,
        "source_drift_aggregate": source_aggregate,
        "receiver_drift": receiver_rows,
        "receiver_drift_aggregate": receiver_aggregate,
        "cache_audit": {
            "physical_capacities": expected_max,
            "maximum_actual_historical_lengths": max_cache_lengths,
            "capacities_exact": cache_capacities(
                new_state(student, 1, windows, torch.bfloat16)
            )
            == expected_max,
            "maxima_exact": max_cache_lengths == expected_max,
            "no_hidden_full_kv_in_truncated_layers": all(
                expected_max[index] < 1023 for index in range(4)
            ),
            "low_block_window_sum": sum(windows[:4]),
            "low_block_ratio_vs_4x1024": sum(windows[:4]) / 4096,
        },
        "performance": {
            "no_feedback_wall_seconds": control_seconds["no_feedback"],
            "teacher_evaluation_wall_seconds": control_seconds["teacher_real"]
            + control_seconds["teacher_shuffled"],
            "self_evaluation_wall_seconds": control_seconds["self_B1_only"]
            + control_seconds["self_real"]
            + control_seconds["self_shuffled"],
            "control_wall_seconds": control_seconds,
            "process_wall_seconds": process_seconds,
            "process_started_unix": process_started_unix,
            "process_finished_unix": time.time(),
            "evaluated_tokens": len(CONTROLS)
            * batch_count
            * a0.VALIDATION_B
            * a0.T,
            "tokens_per_second": (
                len(CONTROLS)
                * batch_count
                * a0.VALIDATION_B
                * a0.T
                / process_seconds
            ),
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        },
        "runtime": runtime_settings(),
        "hashes_before": before,
        "hashes_after": after,
        "integrity": {
            "source_checkpoint_exact": source["checkpoint_sha256"] == SOURCE_SHA,
            "base_checkpoint_exact": source["base_checkpoint_sha256"] == BASE_SHA,
            "canonical_validation_exact": gate_only or digest == CANONICAL_SHA,
            "parameters_bit_identical": before == after,
            "reader_bit_identical": before["reader"] == after["reader"],
            "optimizer_objects_zero": True,
            "scheduler_objects_zero": True,
            "grad_scaler_objects_zero": True,
            "backward_calls_zero": True,
            "optimizer_steps_zero": True,
            "parameter_updates_zero": True,
            "training_targets_zero": True,
            "cache_capacities_exact": cache_capacities(
                new_state(student, 1, windows, torch.bfloat16)
            )
            == expected_max,
            "cache_maxima_exact": max_cache_lengths == expected_max,
            "all_losses_finite": all(math.isfinite(value) for value in means.values()),
            "all_recurrent_memories_finite": True,
            "all_KV_finite": True,
            "all_parameters_frozen": not any(
                parameter.requires_grad for parameter in student.parameters()
            ),
            "teacher_eval_no_grad": not teacher.training
            and not any(parameter.requires_grad for parameter in teacher.parameters()),
            "writers_inactive": "memory_writers" not in student.transformer,
            "source_bank_unchanged": tuple(payload.get("source_depths", ()))
            == SOURCE_DEPTHS,
            "reader_destinations_unchanged": payload.get("active_destinations")
            == [1, 2, 3, 4],
            "B5_B12_windows_1024": windows[4:] == (1024,) * 8,
            "no_inner_loops": True,
            "no_auxiliary_objective": True,
            "no_bptt": True,
            "hellaswag_not_run": True,
        },
    }
    if gate_only:
        expected_first_two = oracle["expected_first_two_batches"]
        value_checks = {
            control: [
                abs(observed - expected) <= REGRESSION_ATOL
                for observed, expected in zip(losses[control], expected_first_two[control])
            ]
            for control in CONTROLS
        }
        gate_checks = {
            "schedule_S0": schedule == "S0",
            "exact_batch_count": batch_count == SHORT_GATE_BATCHES,
            "all_six_controls_present": set(losses) == set(CONTROLS),
            "all_per_batch_values_within_tolerance": all(
                all(control_checks) for control_checks in value_checks.values()
            ),
            "runtime_exact": result["runtime"] == oracle["expected_runtime"],
            "model_and_reader_hashes_unchanged": before == after,
            "worker_integrity": all(result["integrity"].values()),
        }
        gate_report = {
            "experiment": "2C4R",
            "stage": "two_batch_S0_gate",
            "execution_path": oracle["execution_path"],
            "implementation_git_commit": result["implementation_git_commit"],
            "oracle_sha256": file_sha256(ORACLE_PATH),
            "absolute_tolerance": REGRESSION_ATOL,
            "expected": expected_first_two,
            "observed": losses,
            "deltas": {
                control: [
                    observed - expected
                    for observed, expected in zip(losses[control], expected_first_two[control])
                ]
                for control in CONTROLS
            },
            "value_checks": value_checks,
            "runtime": result["runtime"],
            "expected_runtime": oracle["expected_runtime"],
            "hashes_before": before,
            "hashes_after": after,
            "validation_prefix_sha256": digest,
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
            "debug_result": result,
        }
        durable_json(gate_path, gate_report)
        if not gate_report["passed"]:
            raise SystemExit(f"2C4R two-batch S0 gate failed: {gate_checks}")
        print("EXPERIMENT_2C4R_TWO_BATCH_S0_GATE_PASS", flush=True)
        return gate_report
    result["integrity"]["passed"] = all(result["integrity"].values())
    if not result["integrity"]["passed"]:
        raise SystemExit(f"2C4R {schedule} integrity failure: {result['integrity']}")
    durable_json(directory / "schedule_result.json", result)
    print(
        f"EXPERIMENT_2C4R_CANONICAL_COMPLETE schedule={schedule} "
        f"self_recovery={self_recovery:.10f} self_gap={self_gap:.10f}",
        flush=True,
    )
    return result


def s0_regression(row):
    oracle = load_oracle()
    expected_values = oracle["expected_aggregate"]
    observed = {
        "no_feedback": row["losses"]["no_feedback"],
        "teacher_real": row["losses"]["teacher_real"],
        "teacher_shuffled": row["losses"]["teacher_shuffled"],
        "teacher_gap": row["teacher_specific_gap"],
        "self_real": row["losses"]["self_real"],
        "self_shuffled": row["losses"]["self_shuffled"],
        "self_gap": row["self_specific_gap"],
        "self_recovery": row["self_recovery"],
        "self_B1_only": row["losses"]["self_B1_only"],
        "self_matched_gain": row["self_matched_gain"],
    }
    checks = {
        key: abs(observed[key] - expected) <= REGRESSION_ATOL
        for key, expected in expected_values.items()
    }
    return {
        "execution_path": oracle["execution_path"],
        "oracle_sha256": file_sha256(ORACLE_PATH),
        "absolute_tolerance": REGRESSION_ATOL,
        "expected": expected_values,
        "observed": observed,
        "deltas": {key: observed[key] - value for key, value in expected_values.items()},
        "checks": checks,
        "passed": all(checks.values()),
    }


def classify(rows, regression, integrity):
    if not regression["passed"] or not integrity:
        return "GRADED-WINDOW RERUN UNSTABLE", "Path-consistent S0 regression or another frozen integrity check failed."
    for key in ("S1", "S2", "S3"):
        row = rows[key]
        if (
            row["self_recovery"] >= 0.020
            and row["self_specific_gap"] >= 0.010
            and row["paired_real_vs_shuffled"]["self_real_wins"] >= 18
            and row["paired_self_vs_no_feedback"]["self_real_wins"] >= 18
            and row["teacher_recovery"] > 0
        ):
            return (
                "GRADED WINDOWS RESCUE ALIGNED SELF RECURRENCE",
                f"{key} passes every frozen strong aligned-rescue threshold.",
            )
    utility_qualifiers = [
        key
        for key in ("S1", "S2", "S3")
        if rows[key]["self_recovery"] >= 0.020
        and rows[key]["paired_self_vs_no_feedback"]["self_real_wins"] >= 15
        and rows[key]["teacher_recovery"] > 0
    ]
    if utility_qualifiers:
        best_utility = max(utility_qualifiers, key=lambda key: rows[key]["self_recovery"])
        best_row = rows[best_utility]
        if (
            best_row["self_specific_gap"] <= 0
            or best_row["paired_real_vs_shuffled"]["self_real_wins"] < 10
        ):
            return (
                "GRADED WINDOWS RESCUE RECURRENT UTILITY BUT NOT ALIGNED MEMORY",
                f"{best_utility} passes the frozen recurrent-utility thresholds, but correct sequence identity is not beneficial.",
            )
    for key in ("S1", "S2", "S3"):
        row = rows[key]
        if (
            row["self_recovery"] > 0
            and row["self_specific_gap"] > 0
            and row["paired_self_vs_no_feedback"]["self_real_wins"] >= 15
            and row["paired_real_vs_shuffled"]["self_real_wins"] >= 15
        ):
            return (
                "GRADED WINDOWS PARTIALLY RESCUE ALIGNED SELF RECURRENCE",
                f"{key} passes every frozen partial aligned-rescue threshold but no schedule passes a stronger rule.",
            )
    if (
        max(rows[key]["window_only_gain"] for key in ("S1", "S2", "S3"))
        >= 0.050
        and all(rows[key]["self_recovery"] <= 0 for key in ("S1", "S2", "S3"))
    ):
        return (
            "GRADED WINDOWS IMPROVE LOCAL CONTEXT BUT NOT RECURRENCE",
            "At least one graded schedule gains >=0.050 from local KV, while self recovery is non-positive for S1-S3.",
        )
    if all(rows[key]["self_recovery"] <= 0 for key in ("S1", "S2", "S3")):
        return (
            "SELF RECURRENCE REMAINS HARMFUL UNDER GRADED WINDOWS",
            "Self recovery is non-positive for S1-S3 and the baseline-only threshold is not met.",
        )
    return "GRADED-WINDOW RERUN RESULT IS MIXED", "All integrity checks pass, but no other frozen classification rule fits."


def monotonic(values):
    return {
        "nondecreasing": all(a <= b for a, b in zip(values, values[1:])),
        "nonincreasing": all(a >= b for a, b in zip(values, values[1:])),
        "values": values,
    }


def scientific_answers(rows, best_recovery, best_tradeoff):
    positive = [key for key in ("S1", "S2", "S3") if rows[key]["self_recovery"] > 0]
    aligned = [key for key in ("S1", "S2", "S3") if rows[key]["self_specific_gap"] > 0]
    matched = [key for key in SCHEDULES if rows[key]["self_matched_gain"] > 0]
    teacher_specific = {
        key: sum(
            shuffled > real
            for real, shuffled in zip(
                rows[key]["per_batch_losses"]["teacher_real"],
                rows[key]["per_batch_losses"]["teacher_shuffled"],
            )
        )
        for key in SCHEDULES
    }
    source_cosines = [rows[key]["source_drift_aggregate"]["mean_cosine"] for key in SCHEDULES]
    source_rms = [rows[key]["source_drift_aggregate"]["mean_rms_difference"] for key in SCHEDULES]
    drift_improves = all(a <= b for a, b in zip(source_cosines, source_cosines[1:])) and all(
        a >= b for a, b in zip(source_rms, source_rms[1:])
    )
    local_gains = {key: rows[key]["window_only_gain"] for key in ("S1", "S2", "S3")}
    return {
        "Q1": (
            f"YES: {', '.join(positive)} have positive same-window self recovery."
            if positive
            else "NO: no graded schedule makes self recurrence beneficial relative to its own no-feedback baseline."
        ),
        "Q2": (
            f"Correct self sequence identity is beneficial for {', '.join(aligned)}."
            if aligned
            else "Correct self sequence identity does not become beneficial for S1-S3."
        ),
        "Q3": f"{best_recovery} has the highest raw same-window self recovery.",
        "Q4": f"{best_tradeoff} has the best self-recovery per low-block KV-window budget among S1-S3.",
        "Q5": (
            "Ordinary local-KV restoration contributes window-only gains of "
            + ", ".join(f"{key}={value:+.10f}" for key, value in local_gains.items())
            + "; these gains are not credited to recurrence."
        ),
        "Q6": (
            f"The extra B2-B4 readers improve B1-only self recurrence in {', '.join(matched)}."
            if matched
            else "The B2-B4 matched readers do not improve any schedule over B1-only self recurrence."
        ),
        "Q7": "Teacher feedback remains sequence-specific: teacher shuffled-real gaps and paired aligned wins are "
        + ", ".join(
            f"{key}={rows[key]['teacher_specific_gap']:+.10f} ({teacher_specific[key]}/20)"
            for key in SCHEDULES
        )
        + ".",
        "Q8": (
            "YES: source cosine rises and source RMS difference falls monotonically from S0 through S3."
            if drift_improves
            else "NO: teacher/student source similarity does not improve monotonically across all four schedules."
        ),
        "Q9": "Reduced source-state drift is reported alongside recovery; it correlates with the window intervention but is not treated as a causal objective.",
        "Q10": (
            f"Alignment is restored for {', '.join(aligned)}."
            if aligned
            else "Reduced drift does not restore correct self-sequence alignment; any positive recovery is generic recurrent utility rather than aligned memory."
        ),
        "Q11": f"After separating local-window and recurrent gains, {best_recovery} is the best raw recurrence schedule and {best_tradeoff} is the best budget tradeoff; S2/S3 are not credited for their much larger local-KV gains.",
        "Q12": "The rerun supports a material contribution from the abrupt B1-B4 receptive-field cliff only to the extent that widening windows improves same-window recurrence and representation compatibility; it does not by itself prove that geometry is the sole autonomous-loop bottleneck.",
    }


def next_decisions(rows, classification, best_recovery):
    rescued = classification in {
        "GRADED WINDOWS RESCUE ALIGNED SELF RECURRENCE",
        "GRADED WINDOWS RESCUE RECURRENT UTILITY BUT NOT ALIGNED MEMORY",
        "GRADED WINDOWS PARTIALLY RESCUE ALIGNED SELF RECURRENCE",
    }
    aligned_rescued = classification in {
        "GRADED WINDOWS RESCUE ALIGNED SELF RECURRENCE",
        "GRADED WINDOWS PARTIALLY RESCUE ALIGNED SELF RECURRENCE",
    }
    best = rows[best_recovery]
    return {
        "A": (
            "YES, as a separately preregistered experiment; the graded diagnostic supports testing a full monotonic pyramid."
            if rescued
            else "NOT YET; first establish useful same-window recurrence under a graded lower geometry."
        ),
        "B": f"Use {best_recovery} as the seed geometry if a full-pyramid experiment is separately approved.",
        "C": (
            f"YES, under {best_recovery}'s frozen geometry, as a separate experiment."
            if best["teacher_recovery"] > 0 and rescued
            else "DEFER until teacher compatibility and positive same-window recurrence are both established."
        ),
        "D": "YES; incremental layer-by-layer reader training is the more interpretable design if reader adaptation is approved.",
        "E": "DEFER; all-strictly-higher source expansion would confound the path-consistent geometry result and needs its own protocol.",
        "F": (
            "YES; adapt the existing self readers before changing the source bank because recurrence helps but aligned identity remains negative."
            if rescued and not aligned_rescued
            else "NO PRIORITY CHANGE; use the aligned-recurrence result and teacher controls to design the next preregistration."
        ),
        "G": "NO; one-step recurrence must first be stable, aligned, and reproduced with any geometry-specific reader adaptation.",
        "H": "YES; writers remain deferred until direct-reader one-step recurrence is established.",
        "I": "YES; keep B1 at W=1 in the first full-pyramid experiment unless a separate protocol explicitly tests opening B1.",
    }


def invalidated_run_comparison(rows):
    old_summary = json.loads((FAILED_2C4_RESULTS / "result_summary.json").read_text())
    comparison = {}
    for schedule in SCHEDULES:
        old_row = old_summary["rows"][schedule]
        new_row = rows[schedule]
        conditions = {}
        for control in CONTROLS:
            old_value = old_row["losses"][control]
            new_value = new_row["losses"][control]
            conditions[control] = {
                "old_invalidated_2c4": old_value,
                "new_2c4r": new_value,
                "absolute_delta": abs(new_value - old_value),
            }
        metrics = {}
        for metric in (
            "teacher_recovery",
            "teacher_specific_gap",
            "self_recovery",
            "self_specific_gap",
            "self_matched_gain",
        ):
            old_value = old_row[metric]
            new_value = new_row[metric]
            metrics[metric] = {
                "old_invalidated_2c4": old_value,
                "new_2c4r": new_value,
                "absolute_delta": abs(new_value - old_value),
            }
        comparison[schedule] = {"conditions": conditions, "metrics": metrics}
    return {
        "label": "Reproducibility comparison to invalidated debug run",
        "scientific_baseline": False,
        "old_experiment": "2C4",
        "old_final_commit": PARENT_COMMIT,
        "new_experiment": "2C4R",
        "schedules": comparison,
    }


def aggregate_results(args):
    require_git(clean=True)
    load_config()
    oracle = load_oracle()
    output = Path(args.output_dir)
    if output.resolve() != ORACLE_PATH.parent.resolve():
        raise SystemExit(f"2C4R output directory must be {ORACLE_PATH.parent}")
    output.mkdir(parents=True, exist_ok=True)
    gate_path = Path(args.run_root) / "s0_two_batch_gate.json"
    if not gate_path.is_file():
        raise SystemExit("missing 2C4R two-batch S0 gate artifact")
    gate = json.loads(gate_path.read_text())
    if not gate.get("passed") or gate.get("oracle_sha256") != file_sha256(ORACLE_PATH):
        raise SystemExit("2C4R two-batch S0 gate is not an exact pass")
    rows = {}
    preflights = {}
    for key in SCHEDULES:
        directory = schedule_dir(args.run_root, key)
        preflight_path = directory / "preflight.json"
        result_path = directory / "schedule_result.json"
        if not preflight_path.is_file() or not result_path.is_file():
            raise SystemExit(f"missing 2C4R worker artifacts for {key}")
        preflights[key] = json.loads(preflight_path.read_text())
        rows[key] = json.loads(result_path.read_text())
        destination = output / RUN_NAMES[key]
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preflight_path, destination / "preflight.json")
        shutil.copy2(result_path, destination / "schedule_result.json")
    shutil.copy2(gate_path, output / "s0_two_batch_gate.json")
    baseline = rows["S0"]["losses"]["no_feedback"]
    for key, row in rows.items():
        row["window_only_gain"] = baseline - row["losses"]["no_feedback"]
    regression = s0_regression(rows["S0"])
    preflight_checks = all(row.get("passed") for row in preflights.values())
    worker_checks = all(row["integrity"]["passed"] for row in rows.values())
    integrity = preflight_checks and worker_checks and regression["passed"]
    classification, classification_rule = classify(rows, regression, integrity)
    source_drift = [entry for key in SCHEDULES for entry in rows[key]["source_drift"]]
    receiver_drift = [entry for key in SCHEDULES for entry in rows[key]["receiver_drift"]]
    receiver_by_schedule_block = aggregate_drift_by_label(receiver_drift, "receiver")
    for key in SCHEDULES:
        source_aggregate = rows[key]["source_drift_aggregate"]
        source_aggregate["delta_mean_cosine_vs_S0"] = (
            source_aggregate["mean_cosine"]
            - rows["S0"]["source_drift_aggregate"]["mean_cosine"]
        )
        source_aggregate["delta_mean_rms_vs_S0"] = (
            source_aggregate["mean_rms_difference"]
            - rows["S0"]["source_drift_aggregate"]["mean_rms_difference"]
        )
    best_recovery = max(SCHEDULES, key=lambda key: rows[key]["self_recovery"])
    best_tradeoff = max(
        ("S1", "S2", "S3"),
        key=lambda key: rows[key]["self_recovery"] / sum(SCHEDULES[key][:4]),
    )
    answers = scientific_answers(rows, best_recovery, best_tradeoff)
    decisions = next_decisions(rows, classification, best_recovery)
    paired_self = {
        key: {
            "no_feedback": rows[key]["per_batch_losses"]["no_feedback"],
            "self_real": rows[key]["per_batch_losses"]["self_real"],
            "statistics": rows[key]["paired_self_vs_no_feedback"],
        }
        for key in SCHEDULES
    }
    paired_shuffle = {
        key: {
            "self_real": rows[key]["per_batch_losses"]["self_real"],
            "self_shuffled": rows[key]["per_batch_losses"]["self_shuffled"],
            "statistics": rows[key]["paired_real_vs_shuffled"],
        }
        for key in SCHEDULES
    }
    paired_b1 = {
        key: {
            "all_readers": rows[key]["per_batch_losses"]["self_real"],
            "B1_only": rows[key]["per_batch_losses"]["self_B1_only"],
            "statistics": rows[key]["paired_all_readers_vs_B1_only"],
        }
        for key in SCHEDULES
    }
    implementation_commit = rows["S0"]["implementation_git_commit"]
    oracle_commit = git_output(
        "log", "-1", "--format=%H", "--", str(ORACLE_PATH.relative_to(REPO_ROOT))
    )
    oracle_precedes_implementation = subprocess.run(
        ["git", "merge-base", "--is-ancestor", oracle_commit, implementation_commit],
        cwd=REPO_ROOT,
        check=False,
    ).returncode == 0
    checks = {
        "2c4_final_commit_exact": git_output("rev-parse", f"{PARENT_TAG}^{{}}") == PARENT_COMMIT,
        "single_2c4r_implementation_commit_exact": all(
            row["implementation_git_commit"] == implementation_commit
            and preflights[key]["implementation_git_commit"] == implementation_commit
            for key, row in rows.items()
        ),
        "2c3_C4_100M_checkpoint_SHA_exact": all(
            row["source"]["checkpoint_sha256"] == SOURCE_SHA for row in rows.values()
        ),
        "base_checkpoint_SHA_exact": all(
            row["source"]["base_checkpoint_sha256"] == BASE_SHA for row in rows.values()
        ),
        "canonical_validation_SHA_exact": all(
            row["canonical_validation_sha256"] == CANONICAL_SHA for row in rows.values()
        ),
        "path_consistent_oracle_written_before_result_run": oracle_precedes_implementation
        and oracle == expected_oracle_payload(),
        "two_batch_S0_preflight_regression_exact": gate["passed"]
        and all(all(values) for values in gate["value_checks"].values()),
        "full_S0_no_feedback_incremental_regression": regression["checks"]["no_feedback"],
        "full_S0_teacher_real_incremental_regression": regression["checks"]["teacher_real"],
        "full_S0_teacher_shuffled_incremental_regression": regression["checks"]["teacher_shuffled"],
        "full_S0_self_real_incremental_regression": regression["checks"]["self_real"],
        "full_S0_self_shuffled_incremental_regression": regression["checks"]["self_shuffled"],
        "full_S0_self_B1_only_incremental_regression": regression["checks"]["self_B1_only"],
        "deterministic_runtime_exact": all(
            row["runtime"] == oracle["expected_runtime"] for row in rows.values()
        ),
        "optimizer_objects_zero": all(row["integrity"]["optimizer_objects_zero"] for row in rows.values()),
        "scheduler_objects_zero": all(row["integrity"]["scheduler_objects_zero"] for row in rows.values()),
        "GradScaler_objects_zero": all(row["integrity"]["grad_scaler_objects_zero"] for row in rows.values()),
        "backward_calls_zero": all(row["integrity"]["backward_calls_zero"] for row in rows.values()),
        "optimizer_steps_zero": all(row["integrity"]["optimizer_steps_zero"] for row in rows.values()),
        "parameter_updates_zero": all(row["integrity"]["parameter_updates_zero"] for row in rows.values()),
        "training_targets_zero": all(row["integrity"]["training_targets_zero"] for row in rows.values()),
        "reader_tensors_unchanged": all(
            row["hashes_before"]["reader"] == row["hashes_after"]["reader"]
            == row["source"]["reader_state_sha256"]
            for row in rows.values()
        ),
        "base_tensors_unchanged": all(
            row["hashes_before"]["student"] == row["hashes_after"]["student"]
            and row["hashes_before"]["teacher"] == row["hashes_after"]["teacher"]
            for row in rows.values()
        ),
        "window_schedules_exact": all(tuple(rows[key]["windows"]) == SCHEDULES[key] for key in SCHEDULES),
        "B1_remains_W1": all(rows[key]["windows"][0] == 1 for key in SCHEDULES),
        "B5_B12_remain_W1024": all(row["integrity"]["B5_B12_windows_1024"] for row in rows.values()),
        "physical_cache_lengths_exact": all(
            row["cache_audit"]["capacities_exact"] and row["cache_audit"]["maxima_exact"]
            for row in rows.values()
        ),
        "no_hidden_full_KV_in_truncated_layers": all(
            row["cache_audit"]["no_hidden_full_kv_in_truncated_layers"] for row in rows.values()
        ),
        "absolute_positions_unchanged": all(
            all(
                length_row["absolute_position_exact"]
                for dtype_rows in preflights[key]["short_tests"].values()
                for length_row in dtype_rows.values()
            )
            for key in SCHEDULES
        ),
        "future_causality_pass": all(
            preflights[key]["causality_and_row_isolation"]["future_causality_all_controls"]
            for key in SCHEDULES
        ),
        "row_isolation_pass": all(
            preflights[key]["causality_and_row_isolation"]["passed"] for key in SCHEDULES
        ),
        "fresh_reset_pass": all(
            preflights[key]["serialization"]["fresh_position_zero"]
            and preflights[key]["serialization"]["fresh_memory_zero"]
            and preflights[key]["serialization"]["fresh_caches_empty"]
            for key in SCHEDULES
        ),
        "serialization_resume_pass": all(
            preflights[key]["serialization"]["passed"] for key in SCHEDULES
        ),
        "source_bank_unchanged": all(row["integrity"]["source_bank_unchanged"] for row in rows.values()),
        "writers_absent": all(row["integrity"]["writers_inactive"] for row in rows.values()),
        "inner_loops_absent": all(row["integrity"]["no_inner_loops"] for row in rows.values()),
        "BPTT_absent": all(row["integrity"]["no_bptt"] for row in rows.values()),
        "auxiliary_objectives_absent": all(row["integrity"]["no_auxiliary_objective"] for row in rows.values()),
        "HellaSwag_absent": all(row["integrity"]["hellaswag_not_run"] for row in rows.values()),
        "all_losses_finite": all(row["integrity"]["all_losses_finite"] for row in rows.values()),
        "all_recurrent_memories_finite": all(
            row["integrity"]["all_recurrent_memories_finite"] for row in rows.values()
        ),
        "all_KV_finite": all(row["integrity"]["all_KV_finite"] for row in rows.values()),
    }
    audit = {
        "experiment": "2C4R",
        "classification": classification,
        "classification_rule": classification_rule,
        "S0_regression": regression,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "passed": all(checks.values()),
    }
    if not audit["passed"]:
        classification = "GRADED-WINDOW RERUN UNSTABLE"
        classification_rule = "At least one frozen path-consistent regression or integrity check failed."
        audit["classification"] = classification
        audit["classification_rule"] = classification_rule
    summary = {
        "experiment": "2C4R",
        "protocol": PROTOCOL,
        "execution_path": oracle["execution_path"],
        "implementation_git_commit": implementation_commit,
        "2c4_parent_tag": PARENT_TAG,
        "2c4_parent_commit": PARENT_COMMIT,
        "2c4_results_commit": PARENT_RESULTS,
        "2c3_source_tag": SOURCE_2C3_TAG,
        "2c3_source_commit": SOURCE_2C3_COMMIT,
        "classification": classification,
        "classification_rule": classification_rule,
        "rows": rows,
        "receiver_by_schedule_block": receiver_by_schedule_block,
        "S0_regression": regression,
        "best_schedule_by_recurrent_recovery": best_recovery,
        "best_schedule_by_KV_recovery_tradeoff": best_tradeoff,
        "monotonicity": {
            "self_recovery": monotonic([rows[key]["self_recovery"] for key in SCHEDULES]),
            "self_specific_gap": monotonic([rows[key]["self_specific_gap"] for key in SCHEDULES]),
            "source_cosine": monotonic([rows[key]["source_drift_aggregate"]["mean_cosine"] for key in SCHEDULES]),
            "source_RMS_difference": monotonic([rows[key]["source_drift_aggregate"]["mean_rms_difference"] for key in SCHEDULES]),
        },
        "scientific_answers": answers,
        "next_experiment_decisions": decisions,
        "optimizer_objects": 0,
        "scheduler_objects": 0,
        "grad_scaler_objects": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_targets": 0,
    }
    source_output = rows["S0"]["source"]
    window_schedules = {
        key: {
            "windows": list(value),
            "physical_historical_capacities": [window - 1 for window in value],
            "low_block_window_sum": sum(value[:4]),
            "low_block_ratio_vs_4x1024": sum(value[:4]) / 4096,
        }
        for key, value in SCHEDULES.items()
    }
    canonical_losses = {
        key: {
            "means": rows[key]["losses"],
            "per_batch": rows[key]["per_batch_losses"],
            "teacher_recovery": rows[key]["teacher_recovery"],
            "teacher_specific_gap": rows[key]["teacher_specific_gap"],
            "self_recovery": rows[key]["self_recovery"],
            "self_specific_gap": rows[key]["self_specific_gap"],
            "self_matched_gain": rows[key]["self_matched_gain"],
            "self_teacher_recovery_ratio": rows[key]["self_teacher_recovery_ratio"],
            "window_only_gain": rows[key]["window_only_gain"],
        }
        for key in SCHEDULES
    }
    teacher_controls = {
        key: {
            "no_feedback": rows[key]["losses"]["no_feedback"],
            "teacher_real": rows[key]["losses"]["teacher_real"],
            "teacher_shuffled": rows[key]["losses"]["teacher_shuffled"],
            "teacher_recovery": rows[key]["teacher_recovery"],
            "teacher_specific_gap": rows[key]["teacher_specific_gap"],
        }
        for key in SCHEDULES
    }
    cache_audit = {key: rows[key]["cache_audit"] for key in SCHEDULES}
    performance = {
        "schedules": {key: rows[key]["performance"] for key in SCHEDULES},
        "total_four_gpu_elapsed_wall_seconds": max(
            rows[key]["performance"]["process_finished_unix"] for key in SCHEDULES
        )
        - min(rows[key]["performance"]["process_started_unix"] for key in SCHEDULES),
    }
    old_comparison = invalidated_run_comparison(rows)
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "source_checkpoint_manifest.json", source_output)
    durable_json(output / "window_schedules.json", window_schedules)
    durable_json(output / "canonical_losses.json", canonical_losses)
    durable_json(output / "paired_self_vs_no_feedback.json", paired_self)
    durable_json(output / "paired_real_vs_shuffled.json", paired_shuffle)
    durable_json(output / "paired_b1_vs_all_readers.json", paired_b1)
    durable_json(output / "teacher_controls.json", teacher_controls)
    durable_json(
        output / "source_drift.json",
        {
            "rows": source_drift,
            "aggregate": {key: rows[key]["source_drift_aggregate"] for key in SCHEDULES},
        },
    )
    durable_json(
        output / "receiver_drift.json",
        {
            "rows": receiver_drift,
            "aggregate": {key: rows[key]["receiver_drift_aggregate"] for key in SCHEDULES},
            "by_receiver": receiver_by_schedule_block,
        },
    )
    durable_json(output / "cache_audit.json", cache_audit)
    durable_json(output / "old_invalidated_run_comparison.json", old_comparison)
    durable_json(output / "performance.json", performance)
    durable_json(output / "FINAL_AUDIT.json", audit)
    print(f"EXPERIMENT_2C4R_AGGREGATE_COMPLETE classification={classification}", flush=True)
    return summary


def final_report_text(summary, audit, results_commit):
    rows = summary["rows"]
    invalidated_comparison = invalidated_run_comparison(rows)
    recurrence_rescued = summary["classification"] in {
        "GRADED WINDOWS RESCUE ALIGNED SELF RECURRENCE",
        "GRADED WINDOWS RESCUE RECURRENT UTILITY BUT NOT ALIGNED MEMORY",
        "GRADED WINDOWS PARTIALLY RESCUE ALIGNED SELF RECURRENCE",
    }
    aligned_rescued = summary["classification"] in {
        "GRADED WINDOWS RESCUE ALIGNED SELF RECURRENCE",
        "GRADED WINDOWS PARTIALLY RESCUE ALIGNED SELF RECURRENCE",
    }
    lines = [
        "# Experiment 2C4R — Path-Consistent Graded-KV-Window Self-Recurrence Rerun",
        "",
        "## Opening result",
        "",
        f"Path-consistent S0 regression: **{'PASS' if summary['S0_regression']['passed'] else 'FAIL'}** at absolute tolerance `{summary['S0_regression']['absolute_tolerance']}`.",
        "",
        "| Schedule | B1-B4 windows | No feedback | Teacher real | Teacher shuffled | Teacher gap | Self real | Self recovery | Self shuffled-real gap | Real-vs-shuffled wins | Extra-reader gain | Mean source cosine | Mean source RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in SCHEDULES:
        row = rows[key]
        lines.append(
            f"| {key} | {'/'.join(map(str, row['windows'][:4]))} | "
            f"{row['losses']['no_feedback']:.10f} | {row['losses']['teacher_real']:.10f} | "
            f"{row['losses']['teacher_shuffled']:.10f} | {row['teacher_specific_gap']:+.10f} | "
            f"{row['losses']['self_real']:.10f} | {row['self_recovery']:+.10f} | "
            f"{row['self_specific_gap']:+.10f} | {row['paired_real_vs_shuffled']['self_real_wins']}/20 | "
            f"{row['self_matched_gain']:+.10f} | {row['source_drift_aggregate']['mean_cosine']:.8f} | "
            f"{row['source_drift_aggregate']['mean_rms_difference']:.8f} |"
        )
    lines.extend(
        [
            "",
            f"Best schedule by recurrent recovery: **{summary['best_schedule_by_recurrent_recovery']}**.",
            f"Best schedule by KV/recovery tradeoff: **{summary['best_schedule_by_KV_recovery_tradeoff']}**.",
            f"Recurrence itself rescued: **{'YES' if recurrence_rescued else 'NO'}**.",
            f"Aligned sequence memory rescued: **{'YES' if aligned_rescued else 'NO'}**.",
            f"Classification: **{summary['classification']}**.",
            f"Frozen rule: {summary['classification_rule']}",
            "",
            "Local-window gain, teacher-feedback gain, self-feedback gain, aligned-sequence gain, and extra B2-B4 reader gain are reported as separate quantities throughout. Absolute loss improvement from a larger window is not treated as recurrent rescue.",
            "",
            "## Primary result table",
            "",
            "| Schedule | Windows | No-feedback | Teacher real | Teacher shuffled | Teacher gap | Self real | Self shuffled | Self gap | Self recovery | Window-only gain | Self B1-only | Self matched gain | Self/teacher recovery | Self-vs-no-feedback wins | Real-vs-shuffled wins |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in SCHEDULES:
        row = rows[key]
        ratio = row["self_teacher_recovery_ratio"]
        lines.append(
            f"| {key} | {'/'.join(map(str, row['windows'][:4]))} | {row['losses']['no_feedback']:.10f} | "
            f"{row['losses']['teacher_real']:.10f} | {row['losses']['teacher_shuffled']:.10f} | "
            f"{row['teacher_specific_gap']:+.10f} | {row['losses']['self_real']:.10f} | "
            f"{row['losses']['self_shuffled']:.10f} | {row['self_specific_gap']:+.10f} | "
            f"{row['self_recovery']:+.10f} | {row['window_only_gain']:+.10f} | "
            f"{row['losses']['self_B1_only']:.10f} | {row['self_matched_gain']:+.10f} | "
            f"{('n/a' if ratio is None else f'{ratio:+.6f}')} | "
            f"{row['paired_self_vs_no_feedback']['self_real_wins']}/20 | "
            f"{row['paired_real_vs_shuffled']['self_real_wins']}/20 |"
        )
    lines.extend(["", "## Critical decomposition", ""])
    for key in SCHEDULES:
        row = rows[key]
        lines.extend(
            [
                f"### {key}",
                "",
                f"- Local-window benefit: {row['window_only_gain']:+.10f}",
                f"- Teacher-feedback benefit: {row['teacher_recovery']:+.10f}",
                f"- Self-feedback benefit: {row['self_recovery']:+.10f}",
                f"- Aligned self-sequence benefit: {row['self_specific_gap']:+.10f}",
                f"- Extra B2-B4 reader benefit: {row['self_matched_gain']:+.10f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Aggregate teacher↔self drift",
            "",
            "| Schedule | Mean source cosine | Mean source RMS diff | Δ cosine vs S0 | Δ RMS vs S0 | Mean source norm ratio | Self recovery | Self gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in SCHEDULES:
        row = rows[key]
        drift = row["source_drift_aggregate"]
        lines.append(
            f"| {key} | {drift['mean_cosine']:.8f} | {drift['mean_rms_difference']:.8f} | "
            f"{drift['delta_mean_cosine_vs_S0']:+.8f} | {drift['delta_mean_rms_vs_S0']:+.8f} | "
            f"{drift['mean_norm_ratio']:.8f} | {row['self_recovery']:+.10f} | {row['self_specific_gap']:+.10f} |"
        )
    lines.extend(
        [
            "",
            "The full 4 schedules × 4 sources × 7 bins source table is in `source_drift.json`; the receiver table is in `receiver_drift.json`. Teacher similarity is interpreted only as a diagnostic correlated with transfer, not as an intrinsic objective.",
            "",
            "## Receiver-state drift",
            "",
            "| Schedule | Receiver | Mean teacher/student cosine | RMS difference | Norm ratio |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for drift in summary["receiver_by_schedule_block"]:
        lines.append(
            f"| {drift['schedule']} | {drift['receiver']} | "
            f"{drift['mean_teacher_student_cosine']:.8f} | {drift['rms_difference']:.8f} | "
            f"{drift['norm_ratio']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Cache budget and physical storage",
            "",
            "| Schedule | B1 | B2 | B3 | B4 | Sum windows | Ratio vs 4×1024 | Max actual historical KV lengths B1-B4 | B5-B12 max |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for key in SCHEDULES:
        row = rows[key]
        cache = row["cache_audit"]
        lines.append(
            f"| {key} | {' | '.join(map(str, row['windows'][:4]))} | {cache['low_block_window_sum']} | "
            f"{cache['low_block_ratio_vs_4x1024']:.6f} | "
            f"{'/'.join(map(str, cache['maximum_actual_historical_lengths'][:4]))} | 1023 |"
        )
    lines.extend(
        [
            "",
            "These are low-block window budgets, not claims about exact end-to-end memory savings; B5-B12 and fixed model memory remain.",
            "",
            "## Paired controls and monotonicity",
            "",
            "| Schedule | Self-vs-no-feedback wins | Mean recovery |",
            "|---|---:|---:|",
        ]
    )
    for key in SCHEDULES:
        row = rows[key]
        lines.append(
            f"| {key} | {row['paired_self_vs_no_feedback']['self_real_wins']}/20 | "
            f"{row['self_recovery']:+.10f} |"
        )
    lines.extend(
        [
            "",
            "| Schedule | Real-vs-shuffled wins | Specific gap |",
            "|---|---:|---:|",
        ]
    )
    for key in SCHEDULES:
        row = rows[key]
        lines.append(
            f"| {key} | {row['paired_real_vs_shuffled']['self_real_wins']}/20 | "
            f"{row['self_specific_gap']:+.10f} |"
        )
    lines.extend(["", "B1-only versus all-reader paired results:", ""])
    for key in SCHEDULES:
        row = rows[key]
        lines.append(
            f"- {key}: all readers win {row['paired_all_readers_vs_B1_only']['all_readers_wins']}/20 "
            f"with mean gain {row['self_matched_gain']:+.10f}."
        )
    lines.append("")
    for metric, row in summary["monotonicity"].items():
        lines.append(
            f"- {metric}: nondecreasing={row['nondecreasing']}, nonincreasing={row['nonincreasing']}."
        )
    lines.extend(
        [
            "",
            "## Reproducibility comparison to invalidated debug run",
            "",
            "This diagnostic comparison was produced only after the 2C4R classification was frozen. The old 2C4 values are not scientific baselines and did not influence classification.",
            "",
            "| Schedule | Condition | Old invalid debug value | New valid rerun value | Absolute delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for schedule in SCHEDULES:
        for condition in CONTROLS:
            comparison = invalidated_comparison["schedules"][schedule]["conditions"][condition]
            lines.append(
                f"| {schedule} | {condition} | {comparison['old_invalidated_2c4']:.10f} | "
                f"{comparison['new_2c4r']:.10f} | {comparison['absolute_delta']:.10f} |"
            )
    lines.extend(["", "## Scientific questions", ""])
    for index in range(1, 13):
        key = f"Q{index}"
        lines.extend([f"### {key}", "", summary["scientific_answers"][key], ""])
    lines.extend(["## Next-experiment decisions", ""])
    for key in "ABCDEFGHI":
        lines.extend([f"### Decision {key}", "", summary["next_experiment_decisions"][key], ""])
    lines.extend(
        [
            "## Integrity and provenance",
            "",
            f"- 2C4 frozen tag: `{PARENT_TAG}`",
            f"- 2C4 parent commit: `{PARENT_COMMIT}`",
            f"- 2C3 frozen source tag: `{SOURCE_2C3_TAG}`",
            f"- 2C3 frozen source commit: `{SOURCE_2C3_COMMIT}`",
            f"- 2C4R implementation commit: `{summary['implementation_git_commit']}`",
            f"- 2C4R results commit: `{results_commit}`",
            f"- C4@100M checkpoint SHA-256: `{SOURCE_SHA}`",
            f"- Base checkpoint SHA-256: `{BASE_SHA}`",
            f"- Canonical validation SHA-256: `{CANONICAL_SHA}`",
            f"- Final audit: {'PASS' if audit['passed'] else 'FAIL'}; failed checks: {audit['failed_checks']}",
            "- Optimizer objects, scheduler objects, GradScaler objects, backward calls, optimizer steps, parameter updates, and training targets were all exactly zero.",
            "- No writers, inner loops, source-bank expansion, reader adaptation, auxiliary loss, BPTT, B5-B12 window changes, or HellaSwag evaluation ran.",
            "",
            "Even where drift and recurrence co-vary, the result is mechanistic evidence about the receptive-field cliff, not proof that window geometry is the only cause.",
            "",
            f"# EXPERIMENT 2C4R COMPLETE",
        ]
    )
    return "\n".join(lines)


def render_report(args):
    require_git(clean=True)
    output = Path(args.output_dir)
    summary = json.loads((output / "result_summary.json").read_text())
    audit = json.loads((output / "FINAL_AUDIT.json").read_text())
    report = final_report_text(summary, audit, args.results_commit)
    if not report.endswith("# EXPERIMENT 2C4R COMPLETE"):
        raise SystemExit("2C4R final report marker missing")
    durable_text(output / "EXPERIMENT_2C4R_FINAL_REPORT.md", report)
    return {"report": str(output), "results_commit": args.results_commit}


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    oracle_parser = subparsers.add_parser("write-oracle")
    oracle_parser.add_argument("--output", required=True)
    for command in ("preflight", "gate-s0", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--schedule", choices=SCHEDULES, required=True)
        sub.add_argument("--parent-checkpoint", required=True)
        sub.add_argument("--run-root", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-root", required=True)
    aggregate.add_argument("--output-dir", required=True)
    render = subparsers.add_parser("render-report")
    render.add_argument("--output-dir", required=True)
    render.add_argument("--results-commit", required=True)
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(a0.SEED)
    np.random.seed(a0.SEED)
    torch.manual_seed(a0.SEED)
    if args.command in {"write-oracle", "aggregate", "render-report"}:
        if args.command == "write-oracle":
            result = write_oracle(args)
        elif args.command == "aggregate":
            result = aggregate_results(args)
        else:
            result = render_report(args)
    else:
        a0.require_cuda()
        torch.cuda.manual_seed(a0.SEED)
        result = run_preflight(args) if args.command == "preflight" else evaluate_schedule(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
