#!/usr/bin/env python3
"""Experiment 2C3: strict 25M-to-100M cumulative-reader continuation."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
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
import experiment_2c2 as c2  # noqa: E402


BRANCH = "experiment-2c3-cumulative-reader-scaling-100m"
PARENT_TAG = "experiment-2c2-cumulative-low-kv-final"
PARENT_COMMIT = "5853308bc172150b05bafb32222f2461230adac5"
PARENT_IMPLEMENTATION = "48c4bf3d7c327484b2ca0037b3d1a175aa0f6df5"
PARENT_RESULTS = "404ab486a47f891030789496819a2717fd4a5491"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2c3_cumulative_reader_scaling_100m.json"
SOURCE_RESULTS = (
    REPO_ROOT / "results" / "experiment_2c2_cumulative_low_kv_matched_feedback"
)
SOURCE_CHECKPOINT_MANIFEST = SOURCE_RESULTS / "checkpoint_manifest.json"
PROTOCOL = "exp2c3_cumulative_reader_scaling_100m_v1"
CHECKPOINT_SCHEMA = "exp2c3_cumulative_reader_continuation_v1"
CONFIGURATIONS = c2.CONFIGURATIONS
RUN_NAMES = c2.RUN_NAMES
GPU_MAPPING = c2.GPU_MAPPING
SOURCE_UPDATE = 48
MILESTONES = (96, 144, 191)
RESTART_UPDATE = 144
FINAL_UPDATE = 191
GLOBAL_TARGETS = c2.GLOBAL_TARGETS
SOURCE_TARGETS = SOURCE_UPDATE * GLOBAL_TARGETS
FINAL_TARGETS = FINAL_UPDATE * GLOBAL_TARGETS
NEW_UPDATES = FINAL_UPDATE - SOURCE_UPDATE
NEW_TARGETS = NEW_UPDATES * GLOBAL_TARGETS
CANONICAL_SHA = c2.CANONICAL_SHA
BASE_MODEL_SHA = c2.BASE_MODEL_SHA
SOURCE_DEPTHS = c2.SOURCE_DEPTHS
PARAMETERS_PER_READER = c2.PARAMETERS_PER_READER
PINNED_FULL = c2.PINNED_FULL
C1_ATOL = c2.C1_ATOL
C1_EXPECTED = {
    96: {"real": 5.7143192530, "shuffle": 5.7953289270, "gap": 0.0810096741},
    191: {"real": 5.5957053900, "shuffle": 5.7372799873, "gap": 0.1415745974},
}

git_output = c2.git_output
file_sha256 = c2.file_sha256
durable_json = c2.durable_json
durable_text = c2.durable_text
append_jsonl = c2.append_jsonl
atomic_torch_save = c2.atomic_torch_save
move_optimizer_to_cuda = c2.move_optimizer_to_cuda
generic_bank = c2.generic_bank


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2C3 requires branch {BRANCH}")
    if git_output("rev-parse", f"{PARENT_TAG}^{{}}") != PARENT_COMMIT:
        raise SystemExit("frozen Experiment 2C2 tag target mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2C3 execution requires a clean worktree")


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        "protocol": PROTOCOL,
        "configurations": {
            key: [block + 1 for block in blocks]
            for key, blocks in CONFIGURATIONS.items()
        },
        "gpu_mapping": GPU_MAPPING,
        "2c2_final_commit": PARENT_COMMIT,
        "2c2_final_tag": PARENT_TAG,
        "2c2_starting_updates": SOURCE_UPDATE,
        "2c2_starting_targets": SOURCE_TARGETS,
        "base_checkpoint_sha256": a0.EXPECTED_PARENT_SHA256,
        "source_depths": list(SOURCE_DEPTHS),
        "global_targets_per_update": GLOBAL_TARGETS,
        "milestone_updates": list(MILESTONES),
        "forced_process_restart_after_update": RESTART_UPDATE,
        "new_updates_per_configuration": NEW_UPDATES,
        "new_targets_per_configuration": NEW_TARGETS,
        "final_total_updates": FINAL_UPDATE,
        "final_total_targets": FINAL_TARGETS,
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
        raise SystemExit(f"2C3 config mismatch: {mismatches}")
    return config


def blocks_for(configuration):
    return c2.blocks_for(configuration)


def run_dir_for(run_root, configuration):
    return Path(run_root) / RUN_NAMES[configuration]


def source_manifest():
    manifest = json.loads(SOURCE_CHECKPOINT_MANIFEST.read_text())
    rows = {}
    for configuration in CONFIGURATIONS:
        row = manifest[configuration][str(SOURCE_UPDATE)]
        required = {
            "passed": row.get("passed") is True,
            "completed_updates": row.get("completed_updates") == SOURCE_UPDATE,
            "strict_reopen": row.get("strict_reopen", {}).get("passed") is True,
            "sha256_reverified": row.get("sha256_reverified") is True,
        }
        if not all(required.values()):
            raise SystemExit(
                f"canonical 2C2 source manifest failed for {configuration}: {required}"
            )
        rows[configuration] = row
    return rows


def source_path(configuration):
    return Path(source_manifest()[configuration]["checkpoint"]).resolve()


def checkpoint_path(run_dir, completed_updates):
    return Path(run_dir) / "checkpoints" / f"checkpoint_updates_{completed_updates:06d}.pt"


def evaluation_path(run_dir, completed_updates):
    return Path(run_dir) / f"evaluation_updates_{completed_updates:06d}.json"


def run_identity(configuration, parent_aux, config, source_row):
    blocks = blocks_for(configuration)
    return {
        "experiment": "2C3",
        "protocol": PROTOCOL,
        "configuration": configuration,
        "masked_blocks": [block + 1 for block in blocks],
        "active_destinations": [block + 1 for block in blocks],
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "branch": BRANCH,
        "2c2_parent_tag": PARENT_TAG,
        "2c2_parent_commit": PARENT_COMMIT,
        "2c2_implementation_commit": PARENT_IMPLEMENTATION,
        "2c2_results_commit": PARENT_RESULTS,
        "source_checkpoint": source_row["checkpoint"],
        "source_checkpoint_sha256": source_row["sha256"],
        "source_completed_updates": SOURCE_UPDATE,
        "source_processed_targets": SOURCE_TARGETS,
        "base_checkpoint": parent_aux["checkpoint"],
        "base_checkpoint_sha256": parent_aux["checkpoint_sha256"],
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
        "iterative_loops": False,
        "hellaswag_run": False,
    }


def make_base_runtime(parent_checkpoint, configuration, include_optimizer):
    return c2.make_runtime(
        parent_checkpoint, configuration, include_optimizer=include_optimizer
    )


def load_source_checkpoint(
    configuration, student, optimizer, loaders, symbols, expected_row
):
    path = Path(expected_row["checkpoint"]).resolve()
    if not path.is_file():
        raise SystemExit(f"canonical 2C2 checkpoint is missing: {path}")
    digest = file_sha256(path)
    payload = a0.torch_load(path, mmap=True)
    blocks = blocks_for(configuration)
    source_identity = payload.get("identity", {})
    required = {
        "sha256": digest == expected_row["sha256"],
        "schema": payload.get("schema") == c2.CHECKPOINT_SCHEMA,
        "experiment": payload.get("experiment") == "2C2",
        "configuration": payload.get("configuration") == configuration,
        "masked_blocks": payload.get("masked_blocks")
        == [block + 1 for block in blocks],
        "active_destinations": payload.get("active_destinations")
        == [block + 1 for block in blocks],
        "completed_updates": payload.get("completed_updates") == SOURCE_UPDATE,
        "processed_targets": payload.get("processed_targets") == SOURCE_TARGETS,
        "reader_updates": payload.get("reader_update_count") == SOURCE_UPDATE,
        "schedule": payload.get("schedule_position")
        == a0.EXPECTED_PARENT_UPDATES + SOURCE_UPDATE,
        "parent": payload.get("parent_checkpoint_sha256")
        == a0.EXPECTED_PARENT_SHA256,
        "base": payload.get("base_model_sha256") == BASE_MODEL_SHA,
        "source_depths": payload.get("source_depths") == list(SOURCE_DEPTHS),
        "2c2_commit": source_identity.get("implementation_git_commit")
        == PARENT_IMPLEMENTATION,
        "2c2_branch": source_identity.get("branch")
        == "experiment-2c2-cumulative-low-kv-matched-feedback",
        "optimizer_record": payload.get("optimizer_integrity", {}).get("passed")
        is True,
        "source_sidecar": expected_row.get("strict_reopen", {}).get("passed") is True,
    }
    if not all(required.values()):
        raise SystemExit(f"2C3 source lineage mismatch: {required}")
    c2.readers(student).load_state_dict(payload["reader_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_to_cuda(optimizer)
    a0.restore_loader_group(
        loaders, payload["dataloader_states"], symbols, replay=True
    )
    a0.restore_rng_state(payload["rng_state"])
    loader_exact = a0.nested_equal(
        a0.snapshot_loaders(loaders), payload["dataloader_states"]
    )
    rng_exact = a0.nested_equal(a0.capture_rng_state(), payload["rng_state"])
    audit = {
        "checkpoint": str(path),
        "sha256": digest,
        "lineage": required,
        "reader_exact_reload": c2.reader_state_sha(student)
        == payload["reader_state_sha256"],
        "optimizer": c2.optimizer_integrity(
            optimizer, SOURCE_UPDATE, len(blocks)
        ),
        "all_adam_moments_restored": a0.nested_equal(
            optimizer.state_dict(), payload["optimizer"]
        ),
        "all_loader_states_restored": loader_exact,
        "all_rng_states_restored": rng_exact,
        "next_hash_exact": a0.next_update_hash(loaders, symbols, replay=True)
        == payload["next_global_batch_sha256"],
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "fresh_process": os.getpid() != payload.get("writer_pid"),
        "source_writer_pid": payload.get("writer_pid"),
        "resume_pid": os.getpid(),
        "completed_updates": SOURCE_UPDATE,
        "processed_targets": SOURCE_TARGETS,
    }
    audit["passed"] = (
        all(required.values())
        and audit["reader_exact_reload"]
        and audit["optimizer"]["passed"]
        and audit["all_adam_moments_restored"]
        and audit["all_loader_states_restored"]
        and audit["all_rng_states_restored"]
        and audit["next_hash_exact"]
        and audit["fresh_process"]
    )
    if not audit["passed"]:
        raise SystemExit(f"strict 2C2-to-2C3 resume failed: {audit}")
    return payload, audit


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
        raise SystemExit(f"refusing to overwrite 2C3 checkpoint: {path}")
    next_hash = a0.next_update_hash(loaders, symbols, replay=True)
    rng = a0.capture_rng_state()
    blocks = blocks_for(configuration)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "experiment": "2C3",
        "configuration": configuration,
        "masked_blocks": [block + 1 for block in blocks],
        "active_destinations": [block + 1 for block in blocks],
        "base_model_sha256": BASE_MODEL_SHA,
        "base_checkpoint_sha256": a0.EXPECTED_PARENT_SHA256,
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "source_checkpoint": identity["source_checkpoint"],
        "reader_state": c2.reader_state(student),
        "reader_state_sha256": c2.reader_state_sha(student),
        "optimizer": optimizer.state_dict(),
        "optimizer_integrity": c2.optimizer_integrity(
            optimizer, completed_updates, len(blocks)
        ),
        "reader_update_count": completed_updates,
        "completed_updates": completed_updates,
        "processed_targets": completed_updates * GLOBAL_TARGETS,
        "2c3_starting_updates": SOURCE_UPDATE,
        "2c3_starting_targets": SOURCE_TARGETS,
        "2c3_additional_updates": completed_updates - SOURCE_UPDATE,
        "2c3_additional_targets": (completed_updates - SOURCE_UPDATE)
        * GLOBAL_TARGETS,
        "dataloader_states": a0.snapshot_loaders(loaders),
        "rng_state": rng,
        "schedule_position": a0.EXPECTED_PARENT_UPDATES + completed_updates,
        "next_global_batch_sha256": next_hash,
        "source_depths": list(SOURCE_DEPTHS),
        "teacher_identity": {
            "base_model_sha256": BASE_MODEL_SHA,
            "base_checkpoint_sha256": a0.EXPECTED_PARENT_SHA256,
            "frozen": True,
            "eval_no_grad": True,
        },
        "mask_semantics": "cumulative consecutive low-block historical KV removed",
        "identity": identity,
        "git_lineage": {
            "2c2_tag": PARENT_TAG,
            "2c2_commit": PARENT_COMMIT,
            "2c3_branch": BRANCH,
            "implementation_commit": identity["implementation_git_commit"],
        },
        "writer_pid": os.getpid(),
    }
    digest = atomic_torch_save(path, payload)
    reopened = a0.torch_load(path, mmap=True)
    strict = {
        "schema": reopened.get("schema") == CHECKPOINT_SCHEMA,
        "configuration": reopened.get("configuration") == configuration,
        "completed_updates": reopened.get("completed_updates")
        == completed_updates,
        "processed_targets": reopened.get("processed_targets")
        == completed_updates * GLOBAL_TARGETS,
        "lineage": reopened.get("identity") == identity,
        "reader": a0.nested_equal(
            reopened.get("reader_state"), payload["reader_state"]
        ),
        "optimizer": a0.nested_equal(
            reopened.get("optimizer"), payload["optimizer"]
        ),
        "loaders": a0.nested_equal(
            reopened.get("dataloader_states"), payload["dataloader_states"]
        ),
        "rng": a0.nested_equal(reopened.get("rng_state"), rng),
        "next_hash": reopened.get("next_global_batch_sha256") == next_hash,
    }
    strict["passed"] = all(strict.values())
    if not strict["passed"]:
        raise SystemExit(f"2C3 checkpoint strict reopen failed: {strict}")
    sidecar = {
        "checkpoint": str(path.resolve()),
        "sha256": digest,
        "completed_updates": completed_updates,
        "processed_targets": completed_updates * GLOBAL_TARGETS,
        "2c3_additional_updates": completed_updates - SOURCE_UPDATE,
        "2c3_additional_targets": (completed_updates - SOURCE_UPDATE)
        * GLOBAL_TARGETS,
        "next_global_batch_sha256": next_hash,
        "writer_pid": os.getpid(),
        "strict_reopen": strict,
        "passed": True,
    }
    durable_text(
        path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n"
    )
    durable_json(path.with_suffix(path.suffix + ".verification.json"), sidecar)
    return sidecar


def load_checkpoint(
    path, configuration, student, optimizer, loaders, symbols, identity
):
    path = Path(path).resolve()
    verification_path = path.with_suffix(path.suffix + ".verification.json")
    if not verification_path.is_file():
        raise SystemExit("2C3 resume checkpoint verification is missing")
    verification = json.loads(verification_path.read_text())
    digest = file_sha256(path)
    payload = a0.torch_load(path, mmap=True)
    blocks = blocks_for(configuration)
    required = {
        "sha256": digest == verification.get("sha256"),
        "sidecar": verification.get("passed") is True,
        "schema": payload.get("schema") == CHECKPOINT_SCHEMA,
        "configuration": payload.get("configuration") == configuration,
        "masked_blocks": payload.get("masked_blocks")
        == [block + 1 for block in blocks],
        "identity": payload.get("identity") == identity,
        "completed_updates": payload.get("completed_updates") == RESTART_UPDATE,
        "processed_targets": payload.get("processed_targets")
        == RESTART_UPDATE * GLOBAL_TARGETS,
        "source_checkpoint": payload.get("source_checkpoint_sha256")
        == identity["source_checkpoint_sha256"],
        "base": payload.get("base_model_sha256") == BASE_MODEL_SHA,
        "source_depths": payload.get("source_depths") == list(SOURCE_DEPTHS),
    }
    if not all(required.values()):
        raise SystemExit(f"2C3 M75 resume lineage mismatch: {required}")
    c2.readers(student).load_state_dict(payload["reader_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_to_cuda(optimizer)
    a0.restore_loader_group(
        loaders, payload["dataloader_states"], symbols, replay=True
    )
    a0.restore_rng_state(payload["rng_state"])
    audit = {
        "checkpoint": str(path),
        "sha256": digest,
        "lineage": required,
        "reader_exact_reload": c2.reader_state_sha(student)
        == payload["reader_state_sha256"],
        "optimizer": c2.optimizer_integrity(
            optimizer, RESTART_UPDATE, len(blocks)
        ),
        "all_adam_moments_restored": a0.nested_equal(
            optimizer.state_dict(), payload["optimizer"]
        ),
        "all_loader_states_restored": a0.nested_equal(
            a0.snapshot_loaders(loaders), payload["dataloader_states"]
        ),
        "all_rng_states_restored": a0.nested_equal(
            a0.capture_rng_state(), payload["rng_state"]
        ),
        "next_hash_exact": a0.next_update_hash(loaders, symbols, replay=True)
        == payload["next_global_batch_sha256"],
        "fresh_process": os.getpid() != payload.get("writer_pid"),
        "checkpoint_writer_pid": payload.get("writer_pid"),
        "resume_pid": os.getpid(),
        "completed_updates": RESTART_UPDATE,
    }
    audit["passed"] = (
        all(required.values())
        and audit["reader_exact_reload"]
        and audit["optimizer"]["passed"]
        and audit["all_adam_moments_restored"]
        and audit["all_loader_states_restored"]
        and audit["all_rng_states_restored"]
        and audit["next_hash_exact"]
        and audit["fresh_process"]
    )
    if not audit["passed"]:
        raise SystemExit(f"strict M75 restart failed: {audit}")
    return payload, audit


def load_generic_means(configuration):
    path = source_path(configuration).parent.parent / "generic_teacher_source_means.pt"
    payload = a0.torch_load(path, mmap=True)
    means = payload["means"].cuda()
    metadata = payload["metadata"]
    required = {
        "calibration_batches": metadata.get("calibration_batch_indices")
        == [20, 21, 22, 23],
        "calibration_sha": metadata.get("calibration_aggregate_sha256")
        == c1.CALIBRATION_SHA,
        "shape": tuple(means.shape) == (4, 768),
        "finite": bool(torch.isfinite(means).all()),
    }
    if not all(required.values()):
        raise SystemExit(f"frozen generic means mismatch: {required}")
    return means, {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "tensor_sha256": c1.tensor_sha256(
            "generic_teacher_source_means", means
        ),
        "metadata": metadata,
        "checks": required,
    }


def cumulative_forward(
    student,
    x,
    y=None,
    control="real",
    memory=None,
    permutation=None,
    active_blocks=None,
    permutation_by_destination=None,
):
    if control == "masked":
        return student(x, y, mode="masked_cumulative_no_feedback")
    if control in {"real", "generic"}:
        return student(
            x,
            y,
            mode="masked_cumulative_topdown_teacher",
            feedback_sources=memory,
            feedback_active_destination_blocks=active_blocks,
            feedback_permutation_by_destination=permutation_by_destination,
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


def paired_report(left, right, left_label, right_label):
    base = c2.paired_statistics(left, right)
    base.update(
        {
            "left_label": left_label,
            "right_label": right_label,
            f"{left_label}_wins": base.pop("real_wins"),
            f"{right_label}_wins": base.pop("shuffled_wins"),
            "mean_right_minus_left": base.pop("mean_gap"),
        }
    )
    return base


@torch.no_grad()
def evaluate_controls(
    student, teacher, symbols, configuration, completed_updates, final=False
):
    student.eval()
    teacher.eval()
    blocks = blocks_for(configuration)
    generic_means = generic_manifest = None
    if final:
        generic_means, generic_manifest = load_generic_means(configuration)
    controls = ["masked", "real", "shuffle", "b1_only"]
    if final:
        controls.append("generic")
    losses = {name: [] for name in controls}
    activation_sets = {}
    leave_one_out = {}
    reader_alignment = {}
    if final:
        activation_sets = {
            f"prefix_{count}": tuple(blocks[:count])
            for count in range(len(blocks) + 1)
        }
        leave_one_out = {
            f"minus_B{block + 1}": tuple(
                value for value in blocks if value != block
            )
            for block in blocks
        }
        reader_alignment = {f"shuffle_B{block + 1}": block for block in blocks}
    activation_losses = {name: [] for name in activation_sets}
    leaveout_losses = {name: [] for name in leave_one_out}
    alignment_losses = {name: [] for name in reader_alignment}
    routing_accumulator = {
        block: {
            "weights": torch.zeros(4, dtype=torch.float64),
            "entropy": 0.0,
            "topdown_rms": 0.0,
            "feedback_rms": 0.0,
        }
        for block in blocks
    }
    validation_hash = hashlib.sha256()
    permutation = symbols["fixed_derangement"](a0.VALIDATION_B, "cuda")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    loader = c2.validation_loader(symbols)
    for batch_index in range(a0.VALIDATION_BATCHES):
        x_cpu, y_cpu = loader.next_batch()
        validation_hash.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)
        memory = a0.teacher_memory(teacher, x, symbols)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, masked_loss = cumulative_forward(student, x, y, "masked")
        losses["masked"].append(masked_loss.detach().double().item())
        student.set_topdown_instrumentation(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, real_loss = cumulative_forward(
                student, x, y, "real", memory=memory
            )
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
            _, shuffled_loss = cumulative_forward(
                student,
                x,
                y,
                "shuffle",
                memory=memory,
                permutation=permutation,
            )
        losses["shuffle"].append(shuffled_loss.detach().double().item())
        if len(blocks) == 1:
            b1_loss = real_loss
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, b1_loss = cumulative_forward(
                    student,
                    x,
                    y,
                    "real",
                    memory=memory,
                    active_blocks=(blocks[0],),
                )
        losses["b1_only"].append(b1_loss.detach().double().item())
        if final:
            template = generic_bank(
                generic_means, x.size(0), x.size(1), memory.dtype
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, generic_loss = cumulative_forward(
                    student, x, y, "generic", memory=template
                )
            losses["generic"].append(generic_loss.detach().double().item())
            for name, active in activation_sets.items():
                if not active:
                    value = masked_loss
                elif active == blocks:
                    value = real_loss
                elif active == (blocks[0],):
                    value = b1_loss
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, value = cumulative_forward(
                            student,
                            x,
                            y,
                            "real",
                            memory=memory,
                            active_blocks=active,
                        )
                activation_losses[name].append(value.detach().double().item())
            for name, active in leave_one_out.items():
                if active in activation_sets.values():
                    prefix_name = next(
                        key for key, value in activation_sets.items() if value == active
                    )
                    value = (
                        masked_loss
                        if not active
                        else real_loss
                        if active == blocks
                        else b1_loss
                        if active == (blocks[0],)
                        else None
                    )
                    if value is None:
                        values = activation_losses[prefix_name]
                        if len(values) == batch_index + 1:
                            leaveout_losses[name].append(values[-1])
                            continue
                else:
                    value = None
                if value is None:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, value = cumulative_forward(
                            student,
                            x,
                            y,
                            "real",
                            memory=memory,
                            active_blocks=active,
                        )
                leaveout_losses[name].append(value.detach().double().item())
            for name, block in reader_alignment.items():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, value = cumulative_forward(
                        student,
                        x,
                        y,
                        "real",
                        memory=memory,
                        permutation_by_destination={block: permutation},
                    )
                alignment_losses[name].append(value.detach().double().item())
            del template, generic_loss
        del x, y, memory, masked_loss, real_loss, shuffled_loss, b1_loss
        print(
            f"2C3 {configuration} eval@{completed_updates} batch "
            f"{batch_index + 1:02d}/{a0.VALIDATION_BATCHES}",
            flush=True,
        )
    digest = validation_hash.hexdigest()
    if digest != CANONICAL_SHA:
        raise SystemExit(f"canonical validation hash mismatch: {digest}")
    means = {name: statistics.fmean(values) for name, values in losses.items()}
    damage = means["masked"] - PINNED_FULL
    recovery = means["masked"] - means["real"]
    matched_gain = means["b1_only"] - means["real"]
    paired_sequence = paired_report(
        losses["real"], losses["shuffle"], "all_real", "all_shuffled"
    )
    paired_matched = paired_report(
        losses["real"], losses["b1_only"], "all_real", "b1_only"
    )
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
            "feedback_rms": accumulator["feedback_rms"]
            / a0.VALIDATION_BATCHES,
        }
    result = {
        "experiment": "2C3",
        "configuration": configuration,
        "masked_blocks": [block + 1 for block in blocks],
        "completed_updates": completed_updates,
        "processed_reader_targets": completed_updates * GLOBAL_TARGETS,
        "2c3_additional_updates": completed_updates - SOURCE_UPDATE,
        "2c3_additional_targets": (completed_updates - SOURCE_UPDATE)
        * GLOBAL_TARGETS,
        "canonical_validation_sha256": digest,
        "losses": means,
        "per_batch_losses": losses,
        "paired_real_vs_shuffled": paired_sequence,
        "paired_all_real_vs_b1_only": paired_matched,
        "damage": damage,
        "recovery": recovery,
        "recovery_fraction": recovery / damage if damage > 0 else None,
        "specific_gap": means["shuffle"] - means["real"],
        "specific_share": (means["shuffle"] - means["real"]) / recovery
        if recovery > 0
        else None,
        "matched_gain": matched_gain,
        "matched_share": matched_gain / recovery if recovery > 0 else None,
        "reader": c2.reader_metrics(student, routing),
        "activation_sets": {
            name: [block + 1 for block in active]
            for name, active in activation_sets.items()
        },
        "activation_losses": {
            name: {"mean": statistics.fmean(values), "per_batch": values}
            for name, values in activation_losses.items()
        },
        "leave_one_out": {
            name: {
                "active_readers": [block + 1 for block in leave_one_out[name]],
                "mean": statistics.fmean(values),
                "per_batch": values,
                "delta_vs_all_real": statistics.fmean(values) - means["real"],
                "positive_batches": sum(
                    value > real
                    for value, real in zip(values, losses["real"])
                ),
            }
            for name, values in leaveout_losses.items()
        },
        "reader_alignment": {
            name: {
                "reader_shuffled": reader_alignment[name] + 1,
                "mean": statistics.fmean(values),
                "per_batch": values,
                "alignment_value": statistics.fmean(values) - means["real"],
                "positive_batches": sum(
                    value > real
                    for value, real in zip(values, losses["real"])
                ),
            }
            for name, values in alignment_losses.items()
        },
        "generic_means": generic_manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }
    if not all(math.isfinite(value) for value in means.values()):
        raise SystemExit("non-finite canonical evaluation")
    return result


def c1_regression(evaluation, update):
    expected = C1_EXPECTED[update]
    observed = {
        "real": evaluation["losses"]["real"],
        "shuffle": evaluation["losses"]["shuffle"],
        "gap": evaluation["specific_gap"],
    }
    checks = {
        key: abs(observed[key] - expected[key]) <= C1_ATOL for key in expected
    }
    return {
        "update": update,
        "processed_targets": update * GLOBAL_TARGETS,
        "observed": observed,
        "expected": expected,
        "absolute_tolerance": C1_ATOL,
        "deltas": {key: observed[key] - expected[key] for key in expected},
        "checks": checks,
        "passed": all(checks.values()),
    }


def wait_for_m50_gate(run_root):
    path = Path(run_root) / "control" / "m50_c1_regression_gate.json"
    deadline = time.monotonic() + 3600
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise SystemExit("timed out waiting for C1 M50 regression gate")
        time.sleep(0.25)
    gate = json.loads(path.read_text())
    if gate.get("passed") is not True:
        raise SystemExit(f"C1 M50 regression gate failed: {gate}")
    return gate


def reconcile_metrics(path, completed_updates):
    path = Path(path)
    rows = []
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected = list(range(SOURCE_UPDATE + 1, completed_updates + 1))
    actual = [row.get("completed_updates") for row in rows]
    if actual != expected:
        raise SystemExit(f"2C3 metrics/checkpoint mismatch: {actual} != {expected}")
    return rows


def run_preflight(args):
    require_git(clean=True)
    config = load_config()
    configuration = args.configuration
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_MAPPING[configuration]):
        raise SystemExit(f"preflight GPU mapping mismatch for {configuration}")
    row = source_manifest()[configuration]
    (
        symbols,
        teacher,
        student,
        optimizer,
        loaders,
        parent_aux,
        _,
    ) = make_base_runtime(args.parent_checkpoint, configuration, True)
    _, resume = load_source_checkpoint(
        configuration, student, optimizer, loaders, symbols, row
    )
    identity = run_identity(configuration, parent_aux, config, row)
    cache = c2.cache_policy(student, configuration)
    causality = c2.causal_mapping_preflight(
        student, teacher, symbols, configuration
    )
    trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
    report = {
        "experiment": "2C3",
        "configuration": configuration,
        "implementation_git_commit": identity["implementation_git_commit"],
        "identity": identity,
        "source_resume": resume,
        "frozen_hashes": c2.validate_frozen_hashes(student, teacher),
        "cache_policy": cache,
        "causality": causality,
        "trainable_parameters": trainable,
        "trainable_parameter_count_exact": trainable
        == PARAMETERS_PER_READER * len(blocks_for(configuration)),
        "teacher_eval": not teacher.training,
        "teacher_gradients_none": all(
            parameter.grad is None for parameter in teacher.parameters()
        ),
        "writers_absent": not hasattr(student.transformer, "memory_writers"),
        "optimizer_updates_added": 0,
        "backward_calls": 0,
    }
    report["passed"] = (
        resume["passed"]
        and cache["passed"]
        and causality["passed"]
        and report["trainable_parameter_count_exact"]
        and report["teacher_eval"]
        and report["teacher_gradients_none"]
        and report["writers_absent"]
    )
    run_dir = run_dir_for(args.run_root, configuration)
    run_dir.mkdir(parents=True, exist_ok=True)
    durable_json(run_dir / "preflight.json", report)
    print(
        f"EXPERIMENT_2C3_PREFLIGHT_COMPLETE configuration={configuration} "
        f"passed={report['passed']}",
        flush=True,
    )
    return report


def run_training(args):
    require_git(clean=True)
    config = load_config()
    configuration = args.configuration
    blocks = blocks_for(configuration)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_MAPPING[configuration]):
        raise SystemExit(f"training GPU mapping mismatch for {configuration}")
    if args.target_update not in {RESTART_UPDATE, FINAL_UPDATE}:
        raise SystemExit("2C3 target must be M75 or M100")
    run_dir = run_dir_for(args.run_root, configuration)
    preflight = json.loads((run_dir / "preflight.json").read_text())
    if (
        preflight.get("passed") is not True
        or preflight.get("implementation_git_commit") != git_output("rev-parse", "HEAD")
    ):
        raise SystemExit("passing current-implementation preflight is required")
    source_row = source_manifest()[configuration]
    (
        symbols,
        teacher,
        student,
        optimizer,
        loaders,
        parent_aux,
        _,
    ) = make_base_runtime(args.parent_checkpoint, configuration, True)
    identity = run_identity(configuration, parent_aux, config, source_row)
    durable_json(run_dir / "run_identity.json", identity)
    restart_audit = None
    if args.resume:
        _, restart_audit = load_checkpoint(
            args.resume,
            configuration,
            student,
            optimizer,
            loaders,
            symbols,
            identity,
        )
        completed = RESTART_UPDATE
        durable_json(
            run_dir / "restart_audit_updates_000144.json", restart_audit
        )
        source_audit = json.loads((run_dir / "source_resume_audit.json").read_text())
    else:
        if args.target_update != RESTART_UPDATE:
            raise SystemExit("M100 continuation requires the mandatory M75 resume")
        _, source_audit = load_source_checkpoint(
            configuration, student, optimizer, loaders, symbols, source_row
        )
        durable_json(run_dir / "source_resume_audit.json", source_audit)
        completed = SOURCE_UPDATE
    metrics_path = run_dir / "metrics.jsonl"
    metrics = reconcile_metrics(metrics_path, completed)
    process_started = time.perf_counter()
    for update in range(completed, args.target_update):
        row = c2.train_one_update(
            args.run_root,
            configuration,
            update,
            student,
            teacher,
            optimizer,
            loaders,
            symbols,
        )
        row.update(
            {
                "experiment": "2C3",
                "starting_total_updates": SOURCE_UPDATE,
                "starting_reader_targets": SOURCE_TARGETS,
                "2c3_additional_updates": update + 1 - SOURCE_UPDATE,
                "2c3_additional_targets": (update + 1 - SOURCE_UPDATE)
                * GLOBAL_TARGETS,
                "reader_lineage_targets": (update + 1) * GLOBAL_TARGETS,
            }
        )
        append_jsonl(metrics_path, row)
        metrics.append(row)
        completed = update + 1
        if completed in MILESTONES:
            evaluation = evaluate_controls(
                student,
                teacher,
                symbols,
                configuration,
                completed,
                final=completed == FINAL_UPDATE,
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
            if completed == 96:
                if configuration == "C1":
                    gate = c1_regression(evaluation, completed)
                    gate["configuration"] = configuration
                    gate["stopping_rule"] = (
                        "continue all configurations only if C1 real, shuffled, and gap "
                        "match historical M50 within 5e-6"
                    )
                    gate_path = (
                        Path(args.run_root)
                        / "control"
                        / "m50_c1_regression_gate.json"
                    )
                    gate_path.parent.mkdir(parents=True, exist_ok=True)
                    durable_json(gate_path, gate)
                wait_for_m50_gate(args.run_root)
    hashes = c2.validate_frozen_hashes(student, teacher)
    stage = {
        "experiment": "2C3",
        "configuration": configuration,
        "completed_updates": completed,
        "processed_targets": completed * GLOBAL_TARGETS,
        "2c3_additional_updates": completed - SOURCE_UPDATE,
        "2c3_additional_targets": (completed - SOURCE_UPDATE) * GLOBAL_TARGETS,
        "target_update": args.target_update,
        "source_resume_audit": source_audit,
        "restart_audit": restart_audit,
        "frozen_hashes": hashes,
        "reader_state_sha256": c2.reader_state_sha(student),
        "optimizer": c2.optimizer_integrity(optimizer, completed, len(blocks)),
        "next_global_batch_sha256": a0.next_update_hash(
            loaders, symbols, replay=True
        ),
        "writer_pid": os.getpid(),
        "performance": {
            "process_wall_seconds": time.perf_counter() - process_started,
            "mean_targets_per_second": statistics.fmean(
                row["targets_per_second"] for row in metrics
            ),
            "peak_allocated_mb": max(row["peak_allocated_mb"] for row in metrics),
            "peak_reserved_mb": max(row["peak_reserved_mb"] for row in metrics),
        },
        "optimizer_updates_total": completed,
        "writers_active_calls": 0,
        "auxiliary_objective": False,
        "bptt": False,
        "iterative_loops": False,
        "hellaswag_run": False,
    }
    durable_json(run_dir / f"stage_updates_{completed:06d}.json", stage)
    marker = (
        "EXPERIMENT_2C3_M75_RESTART_REQUIRED"
        if completed == RESTART_UPDATE
        else "EXPERIMENT_2C3_M100_TRAINING_COMPLETE"
    )
    print(
        f"{marker} configuration={configuration} updates={completed}", flush=True
    )
    return stage


def load_final_reader(args, configuration):
    run_dir = run_dir_for(args.run_root, configuration)
    path = checkpoint_path(run_dir, FINAL_UPDATE)
    verification = json.loads(
        path.with_suffix(path.suffix + ".verification.json").read_text()
    )
    payload = a0.torch_load(path, mmap=True)
    required = {
        "sha256": file_sha256(path) == verification.get("sha256"),
        "sidecar": verification.get("passed") is True,
        "schema": payload.get("schema") == CHECKPOINT_SCHEMA,
        "configuration": payload.get("configuration") == configuration,
        "completed_updates": payload.get("completed_updates") == FINAL_UPDATE,
        "processed_targets": payload.get("processed_targets") == FINAL_TARGETS,
    }
    if not all(required.values()):
        raise SystemExit(f"final checkpoint lineage mismatch: {required}")
    symbols, teacher, student, optimizer, _, _, _ = make_base_runtime(
        args.parent_checkpoint, configuration, False
    )
    if optimizer is not None:
        raise SystemExit("self/finalize process must have zero optimizer objects")
    c2.readers(student).load_state_dict(payload["reader_state"], strict=True)
    reader_exact = c2.reader_state_sha(student) == payload["reader_state_sha256"]
    if not reader_exact:
        raise SystemExit("final reader strict reload mismatch")
    return symbols, teacher, student, payload, verification, required


def run_finalize(args):
    require_git(clean=True)
    load_config()
    configuration = args.configuration
    blocks = blocks_for(configuration)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_MAPPING[configuration]):
        raise SystemExit(f"finalize GPU mapping mismatch for {configuration}")
    run_dir = run_dir_for(args.run_root, configuration)
    final_evaluation = json.loads(evaluation_path(run_dir, FINAL_UPDATE).read_text())
    symbols, teacher, student, payload, verification, required = load_final_reader(
        args, configuration
    )
    before_sha = c2.reader_state_sha(student)
    if configuration == "C1":
        regression = c1_regression(final_evaluation, FINAL_UPDATE)
        trigger = regression["passed"] and final_evaluation["specific_gap"] > 0
        matched_gate = None
    else:
        regression = None
        matched_gate = {
            "matched_gain": final_evaluation["matched_gain"] >= 0.020,
            "specific_gap": final_evaluation["specific_gap"] >= 0.030,
            "real_wins": final_evaluation["paired_real_vs_shuffled"][
                "all_real_wins"
            ]
            >= 18,
            "recovery": final_evaluation["recovery"] > 0,
        }
        trigger = all(matched_gate.values())
    if trigger:
        self_transfer = c2.evaluate_self_controls(
            student, symbols, configuration
        )
        self_transfer["experiment"] = "2C3"
        self_transfer["optimizer_objects"] = 0
        self_transfer["backward_calls"] = 0
        self_transfer["parameter_updates"] = 0
        if configuration == "C1":
            self_transfer["losses"]["b1_only"] = self_transfer["losses"]["real"]
        self_transfer["teacher_recovery"] = final_evaluation["recovery"]
        self_transfer["self_recovery"] = (
            final_evaluation["losses"]["masked"]
            - self_transfer["losses"]["real"]
        )
        self_transfer["self_teacher_recovery_ratio"] = (
            self_transfer["self_recovery"] / final_evaluation["recovery"]
        )
        self_transfer["self_matched_gain"] = (
            self_transfer["losses"]["b1_only"]
            - self_transfer["losses"]["real"]
        )
    else:
        self_transfer = {
            "experiment": "2C3",
            "configuration": configuration,
            "triggered": False,
            "reason": "frozen M100 self-transfer gate not met",
            "c1_regression": regression,
            "matched_gate": matched_gate,
            "teacher_recovery": final_evaluation["recovery"],
            "teacher_specific_gap": final_evaluation["specific_gap"],
            "teacher_real_wins": final_evaluation["paired_real_vs_shuffled"][
                "all_real_wins"
            ],
            "matched_gain": final_evaluation["matched_gain"],
            "optimizer_objects": 0,
            "backward_calls": 0,
            "parameter_updates": 0,
        }
    self_transfer["reader_state_bit_identical"] = (
        c2.reader_state_sha(student) == before_sha
    )
    self_transfer["c1_regression"] = regression
    self_transfer["matched_gate"] = matched_gate
    durable_json(run_dir / "self_transfer.json", self_transfer)
    stage = json.loads((run_dir / "stage_updates_000191.json").read_text())
    stage["final_checkpoint_reload"] = {
        "lineage": required,
        "reader_exact": c2.reader_state_sha(student)
        == payload["reader_state_sha256"],
        "sha256": verification["sha256"],
        "fresh_process": os.getpid() != payload.get("writer_pid"),
        "checkpoint_writer_pid": payload.get("writer_pid"),
        "finalize_pid": os.getpid(),
        "passed": all(required.values())
        and c2.reader_state_sha(student) == payload["reader_state_sha256"]
        and os.getpid() != payload.get("writer_pid"),
    }
    stage["self_evaluation"] = {
        "triggered": trigger,
        "optimizer_objects": 0,
        "backward_calls": 0,
        "parameter_updates": 0,
        "reader_state_bit_identical": self_transfer[
            "reader_state_bit_identical"
        ],
        "elapsed_seconds": self_transfer.get("elapsed_seconds", 0.0),
    }
    stage["finalize_pid"] = os.getpid()
    durable_json(run_dir / "stage_updates_000191.json", stage)
    print(
        f"EXPERIMENT_2C3_FINALIZE_COMPLETE configuration={configuration} "
        f"self_triggered={trigger}",
        flush=True,
    )
    return stage


def milestone_label(update):
    return {48: "25M", 96: "50M", 144: "75M", 191: "100M"}[update]


def source_evaluation(configuration, summary, paired):
    row = summary["configurations"][configuration]
    sequence_paired = paired_report(
        paired[configuration]["real"],
        paired[configuration]["shuffled"],
        "all_real",
        "all_shuffled",
    )
    matched_paired = paired_report(
        paired[configuration]["real"],
        paired[configuration]["b1_only"],
        "all_real",
        "b1_only",
    )
    return {
        "experiment": "2C2",
        "configuration": configuration,
        "completed_updates": SOURCE_UPDATE,
        "processed_reader_targets": SOURCE_TARGETS,
        "losses": {
            "masked": row["masked_loss"],
            "real": row["real_loss"],
            "shuffle": row["shuffled_loss"],
            "b1_only": row["b1_only_loss"],
            "generic": row["generic_loss"],
        },
        "per_batch_losses": {
            "real": paired[configuration]["real"],
            "shuffle": paired[configuration]["shuffled"],
            "b1_only": paired[configuration]["b1_only"],
        },
        "paired_real_vs_shuffled": sequence_paired,
        "paired_all_real_vs_b1_only": matched_paired,
        "damage": row["damage"],
        "recovery": row["recovery"],
        "recovery_fraction": row["recovery_fraction"],
        "specific_gap": row["specific_gap"],
        "matched_gain": row["matched_destination_gain"],
        "matched_share": (
            row["matched_destination_gain"] / row["recovery"]
            if row["recovery"] > 0
            else None
        ),
        "reader": summary["router_stats"][configuration],
    }


def cosine_matrix(query_by_configuration):
    result = {}
    keys = list(CONFIGURATIONS)
    for left_index, left in enumerate(keys):
        result[left] = {}
        for right in keys:
            result[left][right] = F.cosine_similarity(
                query_by_configuration[left], query_by_configuration[right], dim=0
            ).item()
    return result


def classify(rows, integrity):
    if not integrity:
        return "CONTINUATION EXPERIMENT UNSTABLE"
    deep = [rows[key] for key in ("C3", "C4")]
    if any(
        row["matched_gain_100m"] >= 0.030
        and row["matched_gain_growth"] >= 0.015
        and row["matched_share_100m"] >= 0.10
        and row["recovery_fraction_100m"] >= 0.08
        and row["specific_gap_100m"] >= 0.050
        and row["matched_all_real_wins"] >= 18
        for row in deep
    ):
        return "MATCHED MULTI-DESTINATION FEEDBACK MATURES AND SCALES"
    if any(
        rows[key]["matched_gain_100m"] >= 0.020
        and rows[key]["matched_gain_100m"] > rows[key]["matched_gain_25m"]
        and rows[key]["matched_all_real_wins"] >= 18
        for key in ("C2", "C3", "C4")
    ):
        return "MATCHED FEEDBACK MATURES PARTIALLY"
    if all(rows[key]["matched_gain_100m"] < 0.020 for key in ("C2", "C3", "C4")):
        if any(
            rows[key]["specific_gap_100m"] >= 0.080
            and rows[key]["generic_vs_real_100m"] >= 0.050
            for key in ("C3", "C4")
        ):
            return "SEQUENCE SIGNAL MATURES BUT B1 REMAINS THE DOMINANT GATEWAY"
        return "MATCHED FEEDBACK REMAINS WEAK AFTER 100M"
    return "CONTINUATION RESULT IS MIXED"


def classification_rule(classification):
    return {
        "MATCHED MULTI-DESTINATION FEEDBACK MATURES AND SCALES": (
            "C1 regressions and integrity pass; C3 or C4 passes the frozen 100M "
            "gain, growth, share, recovery, gap, and 18/20 paired-win thresholds."
        ),
        "MATCHED FEEDBACK MATURES PARTIALLY": (
            "The strong rule fails, but C2, C3, or C4 reaches matched gain >=0.020, "
            "grows beyond 25M, and wins at least 18/20 paired batches."
        ),
        "SEQUENCE SIGNAL MATURES BUT B1 REMAINS THE DOMINANT GATEWAY": (
            "All C2-C4 matched gains remain <0.020, while C3 or C4 reaches both "
            "specific gap >=0.080 and generic-real >=0.050."
        ),
        "MATCHED FEEDBACK REMAINS WEAK AFTER 100M": (
            "All C2-C4 matched gains remain <0.020 and neither C3 nor C4 passes "
            "the frozen sequence-signal-maturation rule."
        ),
        "CONTINUATION RESULT IS MIXED": (
            "All integrity checks pass, but no other frozen classification fits."
        ),
        "CONTINUATION EXPERIMENT UNSTABLE": (
            "At least one frozen continuation integrity requirement failed."
        ),
    }[classification]


def copy_worker_artifacts(run_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "preflight.json",
        "source_resume_audit.json",
        "run_identity.json",
        "metrics.jsonl",
        "restart_audit_updates_000144.json",
        "stage_updates_000144.json",
        "stage_updates_000191.json",
        "self_transfer.json",
        *[f"evaluation_updates_{update:06d}.json" for update in MILESTONES],
    ]
    for name in names:
        source = Path(run_dir) / name
        if not source.is_file():
            raise SystemExit(f"missing required 2C3 worker artifact: {source}")
        shutil.copy2(source, output_dir / name)


def scientific_answers(rows, alignment, routing, self_transfer):
    def maturity(key):
        growth = rows[key]["matched_gain_growth"]
        return (
            f"100M matched gain is {rows[key]['matched_gain_100m']:.10f}, a "
            f"{growth:+.10f} change from 25M."
        )

    aligned = []
    for configuration in ("C2", "C3", "C4"):
        for row in alignment[configuration]:
            if row["destination"] > 1 and row["alignment_value"] > 0:
                aligned.append(
                    f"{configuration}/B{row['destination']} ({row['alignment_value']:.10f})"
                )
    triggered = [key for key, row in self_transfer.items() if row.get("triggered")]
    return {
        "Q1": maturity("C2"),
        "Q2": maturity("C3"),
        "Q3": maturity("C4"),
        "Q4": (
            f"C3 and C4 matched shares are {rows['C3']['matched_share_100m']:.6f} "
            f"and {rows['C4']['matched_share_100m']:.6f}; the complementary share "
            "is attributable to the B1-only pathway under this intervention."
        ),
        "Q5": (
            f"C3/C4 gap growth is {rows['C3']['specific_gap_growth']:+.10f} and "
            f"{rows['C4']['specific_gap_growth']:+.10f}."
        ),
        "Q6": (
            "100M generic-real differences: "
            + ", ".join(
                f"{key} {rows[key]['generic_vs_real_100m']:+.10f}"
                for key in CONFIGURATIONS
            )
            + "."
        ),
        "Q7": (
            "Positive individual alignment values: " + ", ".join(aligned) + "."
            if aligned
            else "No B2/B3/B4 reader had positive individual alignment value."
        ),
        "Q8": (
            f"At 100M, C3/B1 v17={routing['C3']['B1']['routing_weights']['v17']:.6f} "
            f"and C4/B1 v17={routing['C4']['B1']['routing_weights']['v17']:.6f}."
        ),
        "Q9": (
            "100M recovery fractions C1-C4 are "
            + ", ".join(
                f"{rows[key]['recovery_fraction_100m']:.6f}"
                for key in CONFIGURATIONS
            )
            + "."
        ),
        "Q10": (
            "Self recurrence triggered for " + ", ".join(triggered) + "."
            if triggered
            else "No configuration triggered zero-shot self recurrence."
        ),
    }


def next_decisions(rows, classification, alignment, self_transfer):
    matured = [
        key
        for key in ("C2", "C3", "C4")
        if rows[key]["matched_gain_100m"] >= 0.020
        and rows[key]["matched_all_real_wins"] >= 18
    ]
    self_positive = [
        key
        for key, row in self_transfer.items()
        if row.get("triggered") and row.get("self_recovery", 0) > 0
    ]
    later_alignment = any(
        row["destination"] > 1 and row["alignment_value"] > 0
        for key in ("C2", "C3", "C4")
        for row in alignment[key]
    )
    b1_dominant = classification in {
        "SEQUENCE SIGNAL MATURES BUT B1 REMAINS THE DOMINANT GATEWAY",
        "MATCHED FEEDBACK REMAINS WEAK AFTER 100M",
    }
    return {
        "A": (
            "No; at least one extra-reader configuration matured under the frozen rule."
            if matured
            else "Yes under this architecture and 100M schedule; extra-reader gains remained below 0.020."
        ),
        "B": (
            "Candidates are " + ", ".join(self_positive) + "; no training is launched here."
            if self_positive
            else "No multi-reader configuration established positive self-recurrent recovery."
        ),
        "C": (
            "Later-reader alignment is detectable, but writers require a separate preregistered experiment."
            if later_alignment
            else "No; direct-reader evidence remains insufficient for writers."
        ),
        "D": "If writers are later authorized, alternate frozen reader/writer phases; do not co-train in 2C3.",
        "E": "No iterative loop is authorized; any such test requires separate preregistration.",
        "F": (
            "Yes; the evidence favors B1 as the primary recurrent gateway."
            if b1_dominant
            else "Not exclusively; at least one additional destination materially contributes."
        ),
        "G": (
            "Yes; weak independent later-reader gains are consistent with needing transformed earlier recurrent computation, but this remains a hypothesis."
            if b1_dominant
            else "The current result does not require transformed inter-layer state, though it remains testable separately."
        ),
    }


def aggregate_results(args):
    require_git(clean=True)
    config = load_config()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((SOURCE_RESULTS / "result_summary.json").read_text())
    source_paired = json.loads((SOURCE_RESULTS / "paired_losses.json").read_text())
    source_rows = source_manifest()
    evaluations = {}
    metrics = {}
    preflights = {}
    source_audits = {}
    restart_audits = {}
    stages = {}
    self_transfer = {}
    checkpoint_manifest = {}
    source_checkpoint_output = {}
    performance = {}
    query_vectors = {update: {} for update in (48, 96, 144, 191)}
    for configuration in CONFIGURATIONS:
        run_dir = run_dir_for(args.run_root, configuration)
        evaluations[configuration] = {
            48: source_evaluation(
                configuration, source_summary, source_paired
            ),
            **{
                update: json.loads(evaluation_path(run_dir, update).read_text())
                for update in MILESTONES
            },
        }
        metrics[configuration] = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            if line
        ]
        preflights[configuration] = json.loads((run_dir / "preflight.json").read_text())
        source_audits[configuration] = json.loads(
            (run_dir / "source_resume_audit.json").read_text()
        )
        restart_audits[configuration] = json.loads(
            (run_dir / "restart_audit_updates_000144.json").read_text()
        )
        stages[configuration] = json.loads(
            (run_dir / "stage_updates_000191.json").read_text()
        )
        self_transfer[configuration] = json.loads(
            (run_dir / "self_transfer.json").read_text()
        )
        if len(metrics[configuration]) != NEW_UPDATES:
            raise SystemExit(f"{configuration} does not have exactly 143 new updates")
        checkpoint_manifest[configuration] = {}
        for update in MILESTONES:
            path = checkpoint_path(run_dir, update)
            verification = json.loads(
                path.with_suffix(path.suffix + ".verification.json").read_text()
            )
            verification["sha256_reverified"] = (
                file_sha256(path) == verification["sha256"]
            )
            checkpoint_manifest[configuration][str(update)] = verification
        source_checkpoint_output[configuration] = {
            **source_rows[configuration],
            "sha256_reverified_during_2c3": file_sha256(
                source_path(configuration)
            )
            == source_rows[configuration]["sha256"],
            "immutable": True,
        }
        for update, path in {
            48: source_path(configuration),
            96: checkpoint_path(run_dir, 96),
            144: checkpoint_path(run_dir, 144),
            191: checkpoint_path(run_dir, 191),
        }.items():
            payload = a0.torch_load(path, mmap=True)
            query_vectors[update][configuration] = payload["reader_state"][
                "0.query"
            ].float()
        ranges = {
            "25_to_50m": (49, 96),
            "50_to_75m": (97, 144),
            "75_to_100m": (145, 191),
        }
        performance[configuration] = {
            f"{name}_training_wall_seconds": sum(
                row["wall_seconds"]
                for row in metrics[configuration]
                if lower <= row["completed_updates"] <= upper
            )
            for name, (lower, upper) in ranges.items()
        }
        performance[configuration].update(
            {
                "total_2c3_update_wall_seconds": sum(
                    row["wall_seconds"] for row in metrics[configuration]
                ),
                "evaluation_wall_seconds": sum(
                    evaluations[configuration][update]["elapsed_seconds"]
                    for update in MILESTONES
                ),
                "self_evaluation_wall_seconds": self_transfer[configuration].get(
                    "elapsed_seconds", 0.0
                ),
                "mean_targets_per_second": statistics.fmean(
                    row["targets_per_second"] for row in metrics[configuration]
                ),
                "peak_allocated_mb": max(
                    row["peak_allocated_mb"] for row in metrics[configuration]
                ),
                "peak_reserved_mb": max(
                    row["peak_reserved_mb"] for row in metrics[configuration]
                ),
            }
        )
        copy_worker_artifacts(run_dir, output_dir / RUN_NAMES[configuration])
    controls = {
        configuration: {
            milestone_label(update): evaluations[configuration][update]
            for update in (48, 96, 144, 191)
        }
        for configuration in CONFIGURATIONS
    }
    trajectories = {}
    matched_gain_trajectory = {}
    paired_losses = {}
    routing_trajectory = {}
    rows = {}
    progressive = {}
    ablations = {}
    alignment = {}
    generic_controls = {}
    final_routing = {}
    for configuration in CONFIGURATIONS:
        trajectories[configuration] = []
        paired_losses[configuration] = {}
        routing_trajectory[configuration] = {}
        for update in (48, 96, 144, 191):
            evaluation = evaluations[configuration][update]
            losses = evaluation["losses"]
            matched_gain = losses["b1_only"] - losses["real"]
            recovery = losses["masked"] - losses["real"]
            trajectories[configuration].append(
                {
                    "milestone": milestone_label(update),
                    "completed_updates": update,
                    "reader_targets": update * GLOBAL_TARGETS,
                    "real_loss": losses["real"],
                    "specific_gap": losses["shuffle"] - losses["real"],
                    "recovery_fraction": evaluation["recovery_fraction"],
                    "matched_gain": matched_gain,
                    "matched_share": matched_gain / recovery if recovery > 0 else None,
                }
            )
            paired_losses[configuration][milestone_label(update)] = {
                "real": evaluation["per_batch_losses"]["real"],
                "shuffled": evaluation["per_batch_losses"]["shuffle"],
                "b1_only": evaluation["per_batch_losses"]["b1_only"],
                "sequence": evaluation.get("paired_real_vs_shuffled"),
                "matched": evaluation.get("paired_all_real_vs_b1_only"),
            }
            routing_trajectory[configuration][milestone_label(update)] = evaluation[
                "reader"
            ]
        matched_gain_trajectory[configuration] = [
            {
                "milestone": row["milestone"],
                "matched_gain": row["matched_gain"],
                "matched_share": row["matched_share"],
            }
            for row in trajectories[configuration]
        ]
        final = evaluations[configuration][191]
        source = evaluations[configuration][48]
        rows[configuration] = {
            "masked_blocks": [block + 1 for block in blocks_for(configuration)],
            "masked_loss_100m": final["losses"]["masked"],
            "real_loss_25m": source["losses"]["real"],
            "real_loss_100m": final["losses"]["real"],
            "shuffled_loss_100m": final["losses"]["shuffle"],
            "generic_loss_100m": final["losses"]["generic"],
            "specific_gap_25m": source["specific_gap"],
            "specific_gap_100m": final["specific_gap"],
            "specific_gap_growth": final["specific_gap"] - source["specific_gap"],
            "generic_vs_real_100m": final["losses"]["generic"]
            - final["losses"]["real"],
            "recovery_100m": final["recovery"],
            "recovery_fraction_100m": final["recovery_fraction"],
            "b1_only_loss_100m": final["losses"]["b1_only"],
            "matched_gain_25m": source["matched_gain"],
            "matched_gain_100m": final["matched_gain"],
            "matched_gain_growth": final["matched_gain"] - source["matched_gain"],
            "matched_gain_ratio": (
                final["matched_gain"] / source["matched_gain"]
                if abs(source["matched_gain"]) > 1e-12
                else None
            ),
            "matched_share_100m": final["matched_share"],
            "sequence_real_wins": final["paired_real_vs_shuffled"][
                "all_real_wins"
            ],
            "matched_all_real_wins": final["paired_all_real_vs_b1_only"][
                "all_real_wins"
            ],
            "matched_b1_only_wins": final["paired_all_real_vs_b1_only"][
                "b1_only_wins"
            ],
        }
        progressive[configuration] = [
            {
                "active_readers": final["activation_sets"][name],
                "loss": value["mean"],
                "incremental_gain": 0.0
                if index == 0
                else previous - value["mean"],
            }
            for index, (name, value), previous in (
                (
                    index,
                    item,
                    0.0
                    if index == 0
                    else list(final["activation_losses"].values())[index - 1]["mean"],
                )
                for index, item in enumerate(final["activation_losses"].items())
            )
        ]
        ablations[configuration] = [
            {
                "reader_removed": int(name.split("B")[1]),
                "loss": value["mean"],
                "delta_vs_all_real": value["delta_vs_all_real"],
                "positive_batches": value["positive_batches"],
            }
            for name, value in final["leave_one_out"].items()
        ]
        alignment[configuration] = [
            {
                "destination": value["reader_shuffled"],
                "all_real": final["losses"]["real"],
                "this_reader_shuffled": value["mean"],
                "alignment_value": value["alignment_value"],
                "positive_batches": value["positive_batches"],
            }
            for value in final["reader_alignment"].values()
        ]
        generic_controls[configuration] = {
            "masked": final["losses"]["masked"],
            "generic": final["losses"]["generic"],
            "shuffled": final["losses"]["shuffle"],
            "real": final["losses"]["real"],
            "generic_minus_real": final["losses"]["generic"]
            - final["losses"]["real"],
            "shuffled_minus_real": final["specific_gap"],
            "frozen_means": final["generic_means"],
        }
        final_routing[configuration] = final["reader"]
    query_cosines = {
        milestone_label(update): cosine_matrix(query_vectors[update])
        for update in (48, 96, 144, 191)
    }
    c1_regressions = {
        "50M": c1_regression(evaluations["C1"][96], 96),
        "100M": c1_regression(evaluations["C1"][191], 191),
    }
    hash_sequences = {
        key: [row["global_batch_sha256"] for row in value]
        for key, value in metrics.items()
    }
    sequences = list(hash_sequences.values())
    implementation_commits = {
        row["implementation_git_commit"] for row in preflights.values()
    }
    integrity = {
        "2c2_frozen_commit_exact": git_output("rev-parse", f"{PARENT_TAG}^{{}}")
        == PARENT_COMMIT,
        "2c2_final_checkpoint_shas_exact": all(
            row["sha256_reverified_during_2c3"]
            for row in source_checkpoint_output.values()
        ),
        "source_checkpoints_update_48_exact": all(
            row["completed_updates"] == SOURCE_UPDATE
            for row in source_audits.values()
        ),
        "base_checkpoint_sha_exact": all(
            row["identity"]["base_checkpoint_sha256"]
            == a0.EXPECTED_PARENT_SHA256
            for row in preflights.values()
        ),
        "canonical_validation_hash_exact": all(
            evaluation["canonical_validation_sha256"] == CANONICAL_SHA
            for configuration in evaluations.values()
            for update, evaluation in configuration.items()
            if update != 48
        ),
        "fresh_process_2c2_to_2c3_resume": all(
            row["fresh_process"] for row in source_audits.values()
        ),
        "all_adam_moments_restored": all(
            row["all_adam_moments_restored"] for row in source_audits.values()
        )
        and all(row["all_adam_moments_restored"] for row in restart_audits.values()),
        "all_adam_steps_restored": all(
            row["optimizer"]["passed"] for row in source_audits.values()
        )
        and all(row["optimizer"]["passed"] for row in restart_audits.values()),
        "all_loader_states_restored": all(
            row["all_loader_states_restored"] for row in source_audits.values()
        )
        and all(row["all_loader_states_restored"] for row in restart_audits.values()),
        "all_rng_states_restored": all(
            row["all_rng_states_restored"] for row in source_audits.values()
        )
        and all(row["all_rng_states_restored"] for row in restart_audits.values()),
        "identical_c1_c4_batch_streams": all(
            sequence == sequences[0] for sequence in sequences[1:]
        ),
        "c1_50m_historical_regression": c1_regressions["50M"]["passed"],
        "c1_100m_historical_regression": c1_regressions["100M"]["passed"],
        "trainable_parameter_counts_exact": all(
            row["trainable_parameter_count_exact"] for row in preflights.values()
        ),
        "base_frozen": all(
            row["frozen_hashes"]["student_base"] == BASE_MODEL_SHA
            for row in stages.values()
        ),
        "teacher_frozen": all(
            row["frozen_hashes"]["teacher"] == BASE_MODEL_SHA
            for row in stages.values()
        ),
        "only_intended_blocks_masked": all(
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
        "future_causality": all(
            row["causality"]["future_prefix_logits_bit_exact"]
            and row["causality"]["teacher_memory_prefix_bit_exact"]
            for row in preflights.values()
        ),
        "row_isolation": all(
            row["causality"]["unchanged_row_logits_bit_exact"]
            for row in preflights.values()
        ),
        "all_gradients_finite": all(
            gradient["present"] and gradient["finite"]
            for rows in metrics.values()
            for row in rows
            for reader in row["gradients"].values()
            for gradient in reader.values()
        ),
        "all_reader_gradients_nonzero": all(
            gradient["nonzero"]
            for rows in metrics.values()
            for row in rows
            for reader in row["gradients"].values()
            for gradient in reader.values()
        ),
        "all_losses_finite": all(
            math.isfinite(row["loss"])
            for rows in metrics.values()
            for row in rows
        )
        and all(
            math.isfinite(loss)
            for configuration in evaluations.values()
            for update, evaluation in configuration.items()
            if update != 48
            for loss in evaluation["losses"].values()
        ),
        "forced_m75_fresh_process_restart": all(
            row["fresh_process"] for row in restart_audits.values()
        ),
        "m75_checkpoint_strict_reload": all(
            row["passed"] for row in restart_audits.values()
        ),
        "final_checkpoints_strict_reload": all(
            row["final_checkpoint_reload"]["passed"] for row in stages.values()
        ),
        "exactly_191_total_reader_updates_per_config": all(
            row["optimizer_updates_total"] == FINAL_UPDATE
            for row in stages.values()
        ),
        "exactly_100139008_total_reader_targets_per_config": all(
            row["processed_targets"] == FINAL_TARGETS for row in stages.values()
        ),
        "exactly_143_new_2c3_updates_per_config": all(
            len(rows) == NEW_UPDATES for rows in metrics.values()
        ),
        "exactly_74973184_new_2c3_targets_per_config": all(
            row["2c3_additional_targets"] == NEW_TARGETS for row in stages.values()
        ),
        "writers_never_active": all(
            row["writers_active_calls"] == 0 for row in stages.values()
        ),
        "no_auxiliary_objective": all(
            row["auxiliary_objective"] is False for row in stages.values()
        ),
        "no_bptt": all(row["bptt"] is False for row in stages.values()),
        "no_iterative_loops": all(
            row["iterative_loops"] is False for row in stages.values()
        ),
        "no_additional_mask_depths": set(CONFIGURATIONS) == {"C1", "C2", "C3", "C4"},
        "hellaswag_not_run": all(
            row["hellaswag_run"] is False for row in stages.values()
        ),
        "self_evaluation_zero_optimizer": all(
            row.get("optimizer_objects") == 0
            and row.get("backward_calls") == 0
            and row.get("parameter_updates") == 0
            and row.get("reader_state_bit_identical") is True
            for row in self_transfer.values()
        ),
        "single_implementation_commit": len(implementation_commits) == 1,
    }
    integrity_passed = all(integrity.values())
    classification = classify(rows, integrity_passed)
    answers = scientific_answers(rows, alignment, final_routing, self_transfer)
    decisions = next_decisions(rows, classification, alignment, self_transfer)
    performance["four_gpu_elapsed_wall_seconds"] = max(
        value["total_2c3_update_wall_seconds"]
        + value["evaluation_wall_seconds"]
        + value["self_evaluation_wall_seconds"]
        for key, value in performance.items()
        if key in CONFIGURATIONS
    )
    summary = {
        "experiment": "2C3",
        "classification": classification,
        "classification_rule": classification_rule(classification),
        "integrity_passed": integrity_passed,
        "integrity_checks": integrity,
        "2c2_frozen_tag": PARENT_TAG,
        "2c2_parent_commit": PARENT_COMMIT,
        "2c3_branch": BRANCH,
        "implementation_commit": next(iter(implementation_commits)),
        "base_checkpoint_sha256": a0.EXPECTED_PARENT_SHA256,
        "configurations": rows,
        "training_trajectory": trajectories,
        "matched_gain_trajectory": matched_gain_trajectory,
        "progressive_activation": progressive,
        "reader_ablation": ablations,
        "reader_alignment_ablation": alignment,
        "generic_controls": generic_controls,
        "reader_routing": {
            "trajectory": routing_trajectory,
            "final": final_routing,
        },
        "b1_query_cosines": query_cosines,
        "self_transfer": self_transfer,
        "c1_regression": c1_regressions,
        "scientific_answers": answers,
        "next_experiment_decisions": decisions,
        "starting_updates_per_configuration": SOURCE_UPDATE,
        "new_updates_per_configuration": NEW_UPDATES,
        "final_updates_per_configuration": FINAL_UPDATE,
        "starting_targets_per_configuration": SOURCE_TARGETS,
        "new_targets_per_configuration": NEW_TARGETS,
        "final_targets_per_configuration": FINAL_TARGETS,
        "new_optimizer_updates_across_configurations": NEW_UPDATES * 4,
        "new_training_targets_across_configurations": NEW_TARGETS * 4,
        "follow_ons_launched": [],
        "config": config,
    }
    audit = {
        "experiment": "2C3",
        "classification": classification,
        "classification_rule": classification_rule(classification),
        "checks": integrity,
        "c1_regression": c1_regressions,
        "passed": integrity_passed,
    }
    durable_json(output_dir / "FINAL_AUDIT.json", audit)
    durable_json(output_dir / "result_summary.json", summary)
    durable_json(output_dir / "source_checkpoint_manifest.json", source_checkpoint_output)
    durable_json(output_dir / "training_trajectory.json", trajectories)
    durable_json(output_dir / "milestone_controls.json", controls)
    durable_json(output_dir / "paired_losses.json", paired_losses)
    durable_json(output_dir / "matched_gain_trajectory.json", matched_gain_trajectory)
    durable_json(output_dir / "progressive_activation.json", progressive)
    durable_json(output_dir / "reader_ablation.json", ablations)
    durable_json(output_dir / "reader_alignment_ablation.json", alignment)
    durable_json(output_dir / "generic_controls.json", generic_controls)
    durable_json(
        output_dir / "reader_routing.json",
        {"trajectory": routing_trajectory, "final": final_routing},
    )
    durable_json(output_dir / "b1_query_cosines.json", query_cosines)
    durable_json(output_dir / "self_transfer.json", self_transfer)
    durable_json(output_dir / "performance.json", performance)
    durable_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    if not integrity_passed:
        raise SystemExit(f"2C3 final audit failed: {integrity}")
    print(classification, flush=True)
    return summary


def final_report_text(summary, results_commit):
    rows = summary["configurations"]
    lines = [
        "# Experiment 2C3 Final Report",
        "",
        "## Opening summary",
        "",
        f"Classification: **{summary['classification']}**",
        "",
        f"Frozen rule: {summary['classification_rule']}",
        "",
        "100M real losses: "
        + ", ".join(f"{key}={rows[key]['real_loss_100m']:.10f}" for key in CONFIGURATIONS)
        + ".",
        "",
        "100M shuffled-real gaps: "
        + ", ".join(f"{key}={rows[key]['specific_gap_100m']:.10f}" for key in CONFIGURATIONS)
        + ".",
        "",
        "Matched gains (25M → 100M): "
        + ", ".join(
            f"{key}={rows[key]['matched_gain_25m']:.10f}→{rows[key]['matched_gain_100m']:.10f}"
            for key in ("C2", "C3", "C4")
        )
        + ".",
        "",
        "100M recovery fractions: "
        + ", ".join(f"{key}={rows[key]['recovery_fraction_100m']:.6f}" for key in CONFIGURATIONS)
        + ".",
        "",
    ]
    b1_dominant = all(rows[key]["matched_gain_100m"] < 0.020 for key in ("C2", "C3", "C4"))
    aligned = [
        f"{key}/B{row['destination']}"
        for key in ("C2", "C3", "C4")
        for row in summary["reader_alignment_ablation"][key]
        if row["destination"] > 1 and row["alignment_value"] > 0
    ]
    routing = summary["reader_routing"]["final"]
    self_triggered = [key for key, row in summary["self_transfer"].items() if row.get("triggered")]
    lines.extend(
        [
            f"B1 remained dominant under the frozen 0.020 matched-gain criterion: **{'YES' if b1_dominant else 'NO'}**.",
            "",
            "Later readers with positive individual alignment value: "
            + (", ".join(aligned) if aligned else "none")
            + ".",
            "",
            f"C3/C4 B1 v17 routing at 100M: {routing['C3']['B1']['routing_weights']['v17']:.6f} / {routing['C4']['B1']['routing_weights']['v17']:.6f}.",
            "",
            "Multi-reader self recurrence triggered: "
            + (", ".join(key for key in self_triggered if key != "C1") or "none")
            + ".",
            "",
            "## Provenance",
            "",
            f"- 2C2 frozen tag: `{PARENT_TAG}`",
            f"- 2C2 parent commit: `{PARENT_COMMIT}`",
            f"- 2C3 branch: `{BRANCH}`",
            f"- Implementation commit: `{summary['implementation_commit']}`",
            f"- Results commit: `{results_commit}`",
            "- Final-report commit: the immutable commit containing this file",
            f"- Base checkpoint SHA256: `{a0.EXPECTED_PARENT_SHA256}`",
            "",
            "## Final main table",
            "",
            "| Config | Masked | 25M Real | 100M Real | 100M Shuffled | 100M Generic | 100M Gap | Recovery % | B1-only | Matched gain | Matched share | Real wins |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, row in rows.items():
        lines.append(
            f"| {key} | {row['masked_loss_100m']:.10f} | {row['real_loss_25m']:.10f} | "
            f"{row['real_loss_100m']:.10f} | {row['shuffled_loss_100m']:.10f} | "
            f"{row['generic_loss_100m']:.10f} | {row['specific_gap_100m']:.10f} | "
            f"{100 * row['recovery_fraction_100m']:.6f} | {row['b1_only_loss_100m']:.10f} | "
            f"{row['matched_gain_100m']:.10f} | {row['matched_share_100m']:.6f} | "
            f"{row['sequence_real_wins']}/20 |"
        )
    lines.extend(["", "## Maturation", "", "| Config | Metric | 25M | 50M | 75M | 100M |", "|---|---|---:|---:|---:|---:|"])
    for key in CONFIGURATIONS:
        trajectory = summary["training_trajectory"][key]
        for metric, label in (
            ("real_loss", "Real loss"),
            ("specific_gap", "Specific gap"),
            ("recovery_fraction", "Recovery fraction"),
            ("matched_gain", "Matched gain"),
            ("matched_share", "Matched share"),
        ):
            lines.append(
                f"| {key} | {label} | "
                + " | ".join(f"{row[metric]:.10f}" for row in trajectory)
                + " |"
            )
    lines.extend(["", "## M100 matched-gain pairs", "", "| Config | B1-only | All-real | Matched gain | All-real wins | B1-only wins |", "|---|---:|---:|---:|---:|---:|"])
    for key in ("C2", "C3", "C4"):
        row = rows[key]
        lines.append(
            f"| {key} | {row['b1_only_loss_100m']:.10f} | {row['real_loss_100m']:.10f} | "
            f"{row['matched_gain_100m']:.10f} | {row['matched_all_real_wins']}/20 | "
            f"{row['matched_b1_only_wins']}/20 |"
        )
    lines.extend(["", "## Progressive activation", "", "| Config | Active readers | Loss | Incremental gain |", "|---|---|---:|---:|"])
    for key in ("C2", "C3", "C4"):
        for row in summary["progressive_activation"][key]:
            label = "+".join(f"B{value}" for value in row["active_readers"]) or "none"
            lines.append(f"| {key} | {label} | {row['loss']:.10f} | {row['incremental_gain']:.10f} |")
    lines.extend(["", "## Leave-one-reader-out", "", "| Config | Reader removed | Loss | Delta vs all-real | Positive batches |", "|---|---|---:|---:|---:|"])
    for key in CONFIGURATIONS:
        for row in summary["reader_ablation"][key]:
            lines.append(f"| {key} | B{row['reader_removed']} | {row['loss']:.10f} | {row['delta_vs_all_real']:.10f} | {row['positive_batches']}/20 |")
    lines.extend(["", "## Per-reader sequence alignment", "", "| Config | Reader shuffled | Loss | Delta vs all-real | Positive batches |", "|---|---|---:|---:|---:|"])
    for key in CONFIGURATIONS:
        for row in summary["reader_alignment_ablation"][key]:
            lines.append(f"| {key} | B{row['destination']} | {row['this_reader_shuffled']:.10f} | {row['alignment_value']:.10f} | {row['positive_batches']}/20 |")
    lines.extend(["", "## Final reader routing", "", "| Config | Destination | Gate | Query norm | RMS displacement | Entropy | v16 | v17 | v20 | v24 | Feedback RMS |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for key, readers in routing.items():
        for destination, row in readers.items():
            weights = row["routing_weights"]
            lines.append(
                f"| {key} | {destination} | {row['effective_gate']:.10f} | {row['query_norm']:.10f} | "
                f"{row['rmsnorm_displacement']:.10f} | {row['routing_entropy']:.10f} | "
                f"{weights['v16']:.10f} | {weights['v17']:.10f} | {weights['v20']:.10f} | "
                f"{weights['v24']:.10f} | {row['feedback_rms']:.10f} |"
            )
    lines.extend(["", "## B1 evolution", "", "| Config | Milestone | Gate | Query norm | v16 | v17 | v20 | v24 | Entropy | Feedback RMS |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for key in CONFIGURATIONS:
        for milestone, readers in summary["reader_routing"]["trajectory"][key].items():
            row = readers["B1"]
            weights = row["routing_weights"]
            lines.append(
                f"| {key} | {milestone} | {row['effective_gate']:.10f} | {row['query_norm']:.10f} | "
                f"{weights['v16']:.10f} | {weights['v17']:.10f} | {weights['v20']:.10f} | "
                f"{weights['v24']:.10f} | {row['routing_entropy']:.10f} | {row['feedback_rms']:.10f} |"
            )
    lines.extend(["", "## B1 query cosine matrices", ""])
    for milestone, matrix in summary["b1_query_cosines"].items():
        lines.extend([f"### {milestone}", "", "| | C1 | C2 | C3 | C4 |", "|---|---:|---:|---:|---:|"])
        for left in CONFIGURATIONS:
            lines.append("| " + left + " | " + " | ".join(f"{matrix[left][right]:.10f}" for right in CONFIGURATIONS) + " |")
        lines.append("")
    lines.extend(["## Generic comparison", "", "| Config | Masked | Generic | Shuffled | Real | Generic-real | Shuffled-real |", "|---|---:|---:|---:|---:|---:|---:|"])
    for key, row in summary["generic_controls"].items():
        lines.append(f"| {key} | {row['masked']:.10f} | {row['generic']:.10f} | {row['shuffled']:.10f} | {row['real']:.10f} | {row['generic_minus_real']:.10f} | {row['shuffled_minus_real']:.10f} |")
    lines.extend(["", "## Conditional self recurrence", "", "| Config | Status | Teacher real | Teacher shuffled | Teacher gap | Teacher recovery | Self real | Self shuffled | Self gap | Self recovery | Self/teacher | Self B1-only | Self all-real | Self matched gain |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for key, self_row in summary["self_transfer"].items():
        if not self_row.get("triggered"):
            lines.append(f"| {key} | SELF TEST NOT TRIGGERED | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            row = rows[key]
            lines.append(
                f"| {key} | TRIGGERED | {row['real_loss_100m']:.10f} | {row['shuffled_loss_100m']:.10f} | "
                f"{row['specific_gap_100m']:.10f} | {row['recovery_100m']:.10f} | "
                f"{self_row['losses']['real']:.10f} | {self_row['losses']['shuffle']:.10f} | "
                f"{self_row['self_specific_gap']:.10f} | {self_row['self_recovery']:.10f} | "
                f"{self_row['self_teacher_recovery_ratio']:.6f} | {self_row['losses']['b1_only']:.10f} | "
                f"{self_row['losses']['real']:.10f} | {self_row['self_matched_gain']:.10f} |"
            )
    lines.extend(["", "## Scientific questions", ""])
    for key, answer in summary["scientific_answers"].items():
        lines.append(f"- {key}: {answer}")
    lines.extend(["", "## Next-experiment decisions", ""])
    for key, answer in summary["next_experiment_decisions"].items():
        lines.append(f"- {key}: {answer}")
    lines.extend(["", "## Integrity and stopping", "", f"All frozen audit checks: **{'PASS' if summary['integrity_passed'] else 'FAIL'}**.", "", "| Audit check | Result |", "|---|---|"])
    for key, passed in summary["integrity_checks"].items():
        lines.append(f"| {key} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            f"- Starting updates/config: {SOURCE_UPDATE}",
            f"- New 2C3 updates/config: {NEW_UPDATES}",
            f"- Final updates/config: {FINAL_UPDATE}",
            f"- New optimizer updates across four configurations: {NEW_UPDATES * 4}",
            f"- New training targets across four configurations: {NEW_TARGETS * 4}",
            "- No writers, self training, BPTT, iterative loops, extra masks, HellaSwag, or follow-on optimization ran.",
            "",
            "# EXPERIMENT 2C3 COMPLETE",
        ]
    )
    return "\n".join(lines)


def render_final_report(args):
    require_git(clean=True)
    output_dir = Path(args.output_dir)
    summary = json.loads((output_dir / "result_summary.json").read_text())
    audit = json.loads((output_dir / "FINAL_AUDIT.json").read_text())
    if summary.get("integrity_passed") is not True or audit.get("passed") is not True:
        raise SystemExit("refusing to render passing report from failed audit")
    if git_output("rev-parse", f"{args.results_commit}^{{commit}}") != args.results_commit:
        raise SystemExit("results commit is not an exact local commit")
    report = final_report_text(summary, args.results_commit)
    if not report.endswith("# EXPERIMENT 2C3 COMPLETE"):
        raise SystemExit("required final report marker is missing")
    durable_text(output_dir / "EXPERIMENT_2C3_FINAL_REPORT.md", report)
    return {"report": str(output_dir), "results_commit": args.results_commit}


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    preflight.add_argument("--parent-checkpoint", required=True)
    preflight.add_argument("--run-root", required=True)
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
    render = subparsers.add_parser("render-report")
    render.add_argument("--output-dir", required=True)
    render.add_argument("--results-commit", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        run_preflight(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "finalize":
        run_finalize(args)
    elif args.command == "aggregate":
        aggregate_results(args)
    elif args.command == "render-report":
        render_final_report(args)


if __name__ == "__main__":
    main()
