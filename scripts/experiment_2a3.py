#!/usr/bin/env python3
"""Exact single-A100 continuation of Experiment 2A2 from 100M to 250M tokens."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2a0 as a0  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs" / "exp2a3_250m.json"
BRANCH = "codex/experiment-2a3-250m"
START_UPDATE = 191
TARGET_UPDATE = 477
MILESTONES = (286, 381, 477)
EXPECTED_CONTINUATION_SHA256 = (
    "6c206a89422470061d7997764fbd9a5708be3d9043f8fab930dd4b800bd5cb95"
)
EXPECTED_CONTINUATION_NEXT_SHA256 = (
    "9f39510b105f068966ef6c052edc015d695827c422da37495fa7c244b965af0b"
)
EXPECTED_CONTINUATION_STATE = {
    "completed_updates": START_UPDATE,
    "processed_student_tokens": START_UPDATE * a0.GLOBAL_BATCH_TOKENS,
}
EXPECTED_FROZEN_BASE_SHA256 = (
    "1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd"
)
EXPECTED_MILESTONE_NEXT_SHA256 = {
    286: "94f21a6b52b3e14bddfd0221076172d2c04a9067dac6ca6e2e9ecfdaaed99ded",
    381: "73dc271a2f06e5f841a8207a3d0243d09ad16b28106b39351381f76fc08d8af2",
    477: "95081c5f68b7d05d6e39b68043f2714657c21ca05cc317549063ba9a4f9f6986",
}
EXPECTED_REPLAY_SEQUENCE_SHA256 = (
    "a8cfc9d2898191bd792df43339af944b38d0580d5a7804c2a3169c4ce19d57b8"
)
PINNED_FULL = 4.078654408454895
PINNED_MASKED = 5.973674488067627
PINNED_START_REAL = 5.5957053899765015
PINNED_START_SHUFFLED = 5.737279987335205
EXPECTED_HELLASWAG_VAL_SHA256 = (
    "0aa3b88843990f3f10a97b9575c94d7b71fb2205240ba04ae4884d9e9c992588"
)
T19_975 = 2.093024054408263
CLASSIFICATION_RULE = (
    "Let q100,q150,q200,q250 be shuffled-minus-real recovery. REVERSING if "
    "q250<=0 or q250-q200<0; SATURATING if q250>0, q250-q200>=0, and the "
    "last gain is <=25% of max(q150-q100,q200-q150,1e-12); STILL ACCELERATING "
    "if q250>q200>q150>q100 and successive gains strictly increase; otherwise "
    "STRENGTHENING."
)
CONTINUE_TO_500M_RULE = (
    "YES only if q250>0, real beats shuffled on at least 15 of 20 batches, "
    "q250>=q200, total recovery remains positive, all controls and invariants pass, "
    "and HellaSwag real feedback is not more than 1.0 percentage point below both "
    "masked/no-feedback and gate-zero; otherwise NO."
)
BEGIN_SELF_RECURRENT_RULE = (
    "YES only if all invariants pass, q250>0, q250>=q200, real wins at least "
    "15/20 paired batches, sequence-specific recovery reaches at least 10% of "
    "the original damage, every final leave-one-source-out delta is positive, and "
    "real-feedback HellaSwag is at least masked/no-feedback; otherwise NO."
)
PINNED_PYTHON = Path("/workspace/venvs/exp1b/bin/python")


def validate_config(config):
    expected = {
        "protocol": "exp2a3_teacher_topdown_250m_v1",
        "seed": a0.SEED,
        "start_completed_updates": START_UPDATE,
        "start_processed_student_tokens": START_UPDATE * a0.GLOBAL_BATCH_TOKENS,
        "optimizer_updates": TARGET_UPDATE,
        "additional_optimizer_updates": TARGET_UPDATE - START_UPDATE,
        "legacy_world_size": a0.LEGACY_WORLD_SIZE,
        "legacy_micro_batch_sequences_per_rank": a0.LEGACY_B,
        "sequence_length": a0.T,
        "legacy_gradient_accumulation": a0.LEGACY_GRAD_ACCUM,
        "sequential_microbatches_per_update": a0.LEGACY_WORLD_SIZE * a0.LEGACY_GRAD_ACCUM,
        "global_batch_tokens": a0.GLOBAL_BATCH_TOKENS,
        "processed_student_tokens": TARGET_UPDATE * a0.GLOBAL_BATCH_TOKENS,
        "source_depths": list(a0.SOURCE_DEPTHS),
        "mode": "masked_l1_topdown_teacher",
        "validation_batches": a0.VALIDATION_BATCHES,
        "validation_batch_sequences": a0.VALIDATION_B,
        "evaluation_updates": list(MILESTONES),
        "checkpoint_updates": list(MILESTONES),
        "frozen_reference_full_context": PINNED_FULL,
        "frozen_reference_masked_l1_no_feedback": PINNED_MASKED,
        "optimizer": "AdamW betas=(0.9,0.95) eps=1e-8 no weight decay",
        "lr_schedule": "continue original 10B schedule; next global schedule step is 1145",
        "data_start": (
            "restore exact four-rank replay states from verified Experiment 2A2 "
            "update-191 checkpoint"
        ),
        "intermediate_controls": ["real_feedback", "shuffled_feedback"],
        "final_controls": [
            "full_context",
            "masked_l1_no_feedback",
            "real_feedback",
            "shuffled_feedback",
            "zero_feedback",
        ],
        "final_source_ablations": list(a0.SOURCE_DEPTHS),
        "hellaswag_examples": 10042,
        "hellaswag_split": "val",
        "hellaswag_val_sha256": EXPECTED_HELLASWAG_VAL_SHA256,
        "hellaswag_controls": [
            "full_context",
            "masked_l1_no_feedback",
            "real_feedback",
            "zero_feedback",
        ],
        "hellaswag_shuffled_feedback": "skipped: candidate-row shuffling would violate alternative isolation",
        "historical_standard_500m_hellaswag": 0.2557259510057757,
        "historical_full_attnres_500m_hellaswag": 0.252141007767377,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if set(config) != set(expected):
        mismatches["fields"] = (sorted(config), sorted(expected))
    if mismatches:
        raise SystemExit(f"Experiment 2A3 config mismatch: {mismatches}")
    return config


def paired_statistics(real, shuffled):
    if not real or len(real) != len(shuffled):
        raise ValueError("paired vectors must be nonempty and equal length")
    values = [float(s) - float(r) for r, s in zip(real, shuffled)]
    if not all(math.isfinite(value) for value in values + list(real) + list(shuffled)):
        raise ValueError("paired vectors must be finite")
    n = len(values)
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if n > 1 else 0.0
    standard_error = sample_std / math.sqrt(n)
    critical = T19_975 if n == 20 else 1.96
    return {
        "n": n,
        "differences": values,
        "mean": mean,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "tie_count": sum(value == 0 for value in values),
        "sample_standard_deviation": sample_std,
        "standard_error": standard_error,
        "ci95_lower": mean - critical * standard_error,
        "ci95_upper": mean + critical * standard_error,
        "ci95_note": "descriptive paired t interval; fixed batches are not assumed IID",
    }


def durable_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        orphaned = temporary.with_name(
            temporary.name + f".orphaned.{int(time.time())}.{os.getpid()}"
        )
        os.replace(temporary, orphaned)
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


def durable_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        orphaned = temporary.with_name(
            temporary.name + f".orphaned.{int(time.time())}.{os.getpid()}"
        )
        os.replace(temporary, orphaned)
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def durable_append_jsonl(path, payload):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def reconcile_metrics(path, completed_updates, start_update=START_UPDATE):
    path = Path(path)
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(path.read_text().splitlines()) - 1:
                raise SystemExit("only a malformed terminal metrics row is recoverable")
    prefix = [row for row in rows if row.get("update", -1) < completed_updates]
    expected = list(range(start_update, completed_updates))
    updates = [row.get("update") for row in prefix]
    if updates != expected:
        raise SystemExit(f"2A3 metrics history mismatch: {updates} != {expected}")
    trailing = rows[len(prefix):]
    if any(row.get("update", -1) < completed_updates for row in trailing):
        raise SystemExit("2A3 metrics contain duplicate/out-of-order committed rows")
    with path.open("w") as handle:
        for row in prefix:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "rows_before": len(rows),
        "rows_retained": len(prefix),
        "rows_truncated": len(rows) - len(prefix),
        "start_update": start_update,
        "completed_updates": completed_updates,
    }


def optimizer_steps(optimizer_or_state):
    state = (
        optimizer_or_state.state_dict()
        if hasattr(optimizer_or_state, "state_dict")
        else optimizer_or_state
    )
    return [int(values["step"].item()) for values in state["state"].values()]


def model_state_sha256(model, include_topdown):
    """Hash model tensors, including zero-dimensional scalar parameters."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        is_topdown = name.startswith("transformer.topdown_attnres.")
        if is_topdown != include_topdown:
            continue
        tensor = value.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(
            tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        )
    return digest.hexdigest()


def assert_optimizer_state(optimizer, completed_updates):
    report = a0.optimizer_state_report(optimizer.state_dict(), completed_updates)
    steps = optimizer_steps(optimizer)
    if steps != [completed_updates] * 3:
        raise SystemExit(f"Adam step mismatch: {steps} != {completed_updates}")
    return {"steps": steps, **report}


def source_metadata_path():
    return REPO_ROOT / "results" / "experiment_2a2_100m" / "metadata.json"


def source_summary_path():
    return REPO_ROOT / "results" / "experiment_2a2_100m" / "result_summary.json"


def source_evaluation_path():
    return (
        REPO_ROOT
        / "results"
        / "experiment_2a2_100m"
        / "evaluations"
        / "evaluation_updates_000191.json"
    )


def load_source_metadata():
    metadata_path = source_metadata_path()
    summary_path = source_summary_path()
    if not metadata_path.is_file() or not summary_path.is_file():
        raise SystemExit("committed Experiment 2A2 100M lineage artifacts are missing")
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads(summary_path.read_text())
    if metadata.get("git_commit") != "57b21510c311bbb96c8423bb70cb74d974d2f00b":
        raise SystemExit("source 100M metadata commit mismatch")
    if summary.get("completed_updates") != START_UPDATE:
        raise SystemExit("source 100M summary mismatch")
    embedded = metadata.get("source_file_sha256", {})
    live = {
        name: a0.file_sha256(REPO_ROOT / name)
        for name in embedded
    }
    mismatches = {
        name: (digest, embedded.get(name))
        for name, digest in live.items()
        if embedded.get(name) != digest
    }
    if mismatches:
        raise SystemExit(f"live Experiment 2A2 source lineage mismatch: {mismatches}")
    return metadata, summary


def require_clean_tree():
    if a0.git_output("status", "--porcelain"):
        raise SystemExit("2A3 optimizer runs require a clean committed worktree")
    if a0.git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"2A3 requires branch {BRANCH}")


def require_pinned_runtime():
    actual = Path(sys.executable).resolve()
    if not PINNED_PYTHON.exists():
        raise SystemExit(f"pinned 2A3 Python is missing: {PINNED_PYTHON}")
    if actual != PINNED_PYTHON.resolve():
        raise SystemExit(
            f"2A3 requires pinned Python {PINNED_PYTHON.resolve()}, got {actual}"
        )
    return str(actual)


def source_hashes():
    files = (
        "train_gpt2.py",
        "hellaswag.py",
        "scripts/experiment_2a0.py",
        "scripts/experiment_2a1.py",
        "scripts/experiment_2a2.py",
        "scripts/experiment_2a3.py",
        "scripts/plot_experiment_2a2.py",
        "scripts/plot_experiment_2a3.py",
        "configs/exp2a0_5m.json",
        "configs/exp2a1_25m.json",
        "configs/exp2a2_100m.json",
        "configs/exp2a3_250m.json",
    )
    return {name: a0.file_sha256(REPO_ROOT / name) for name in files}


def continuation_metadata(config, parent_aux, continuation_path):
    source_metadata, _ = load_source_metadata()
    return {
        "experiment": "Experiment 2A3",
        "kind": "exact update-191 to update-477 continuation",
        "git_commit": a0.git_output("rev-parse", "HEAD"),
        "git_status": a0.git_output("status", "--short", "--branch"),
        "parent_commit": a0.PARENT_COMMIT,
        "parent_checkpoint": parent_aux["checkpoint"],
        "parent_checkpoint_sha256": parent_aux["checkpoint_sha256"],
        "continuation_checkpoint": str(Path(continuation_path).resolve()),
        "continuation_checkpoint_sha256": EXPECTED_CONTINUATION_SHA256,
        "continuation_training_state": EXPECTED_CONTINUATION_STATE,
        "continuation_next_global_batch_sha256": EXPECTED_CONTINUATION_NEXT_SHA256,
        "continuation_metadata_sha256": hashlib.sha256(
            json.dumps(source_metadata, sort_keys=True).encode()
        ).hexdigest(),
        "lineage_artifact_sha256": {
            "results/experiment_2a2_100m/config.json": a0.file_sha256(
                REPO_ROOT / "results" / "experiment_2a2_100m" / "config.json"
            ),
            "results/experiment_2a2_100m/metadata.json": a0.file_sha256(
                source_metadata_path()
            ),
            "results/experiment_2a2_100m/result_summary.json": a0.file_sha256(
                source_summary_path()
            ),
            "results/experiment_2a2_100m/evaluations/evaluation_updates_000191.json": a0.file_sha256(
                source_evaluation_path()
            ),
        },
        "source_depths": list(a0.SOURCE_DEPTHS),
        "destination": "Block 1 Attention input",
        "teacher": "immutable Experiment 1B parent; frozen eval/no_grad/detached",
        "student_base": "frozen",
        "trainable_parameters": 1537,
        "trajectory_classification_rule": CLASSIFICATION_RULE,
        "continue_to_500m_rule": CONTINUE_TO_500M_RULE,
        "begin_self_generated_recurrence_rule": BEGIN_SELF_RECURRENT_RULE,
        "config": config,
        "source_file_sha256": source_hashes(),
        "dataset": a0.dataset_manifest_report(verify_shards=False),
        "determinism": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "python_executable": str(Path(sys.executable).resolve()),
    }


def make_runtime(parent_checkpoint, device):
    symbols, teacher, student, parent_aux = a0.load_models(
        parent_checkpoint, device, include_teacher=True
    )
    optimizer = a0.feedback_optimizer(student)
    loaders = a0.make_replay_loaders(symbols, parent_aux["dataloader_states"])
    return symbols, teacher, student, optimizer, loaders, parent_aux


def validate_source_checkpoint(path):
    path = Path(path).resolve()
    if a0.file_sha256(path) != EXPECTED_CONTINUATION_SHA256:
        raise SystemExit("Experiment 2A2 update-191 continuation SHA mismatch")
    integrity = a0.verify_checkpoint_sidecars(path)
    checkpoint = a0.torch_load(path, mmap=True)
    if checkpoint.get("schema") != a0.CHECKPOINT_SCHEMA:
        raise SystemExit("continuation schema mismatch")
    if checkpoint.get("parent_checkpoint_sha256") != a0.EXPECTED_PARENT_SHA256:
        raise SystemExit("continuation root-parent mismatch")
    if checkpoint.get("training_state") != EXPECTED_CONTINUATION_STATE:
        raise SystemExit("continuation state mismatch")
    if checkpoint.get("next_global_batch_sha256") != EXPECTED_CONTINUATION_NEXT_SHA256:
        raise SystemExit("continuation next-batch mismatch")
    if len(checkpoint.get("dataloader_states", [])) != a0.LEGACY_WORLD_SIZE:
        raise SystemExit("continuation must contain four replay loader states")
    if len(checkpoint.get("optimizer", {}).get("state", {})) != 3:
        raise SystemExit("continuation must contain three optimizer states")
    if [int(v["step"].item()) for v in checkpoint["optimizer"]["state"].values()] != [START_UPDATE] * 3:
        raise SystemExit("continuation Adam step mismatch")
    a0.optimizer_state_report(checkpoint["optimizer"], START_UPDATE)
    return integrity, checkpoint


def restore_source_checkpoint(
    path, student, optimizer, loaders, symbols, parent_aux, source_metadata
):
    integrity, checkpoint = validate_source_checkpoint(path)
    if checkpoint.get("metadata") != source_metadata:
        raise SystemExit("continuation source metadata mismatch")
    student.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    a0.restore_loader_group(loaders, checkpoint["dataloader_states"], symbols, True)
    a0.restore_rng_state(checkpoint["rng_state"])
    next_hash = a0.next_update_hash(loaders, symbols, True)
    audit = {
        "integrity": integrity,
        "model_exact_reload": a0.nested_equal(student.state_dict(), checkpoint["model"]),
        "optimizer_exact_reload": a0.nested_equal(
            optimizer.state_dict(), checkpoint["optimizer"]
        ),
        "loader_exact_reload": a0.nested_equal(
            a0.snapshot_loaders(loaders), checkpoint["dataloader_states"]
        ),
        "rng_exact_reload": a0.nested_equal(
            a0.capture_rng_state(), checkpoint["rng_state"]
        ),
        "next_global_batch_sha256": next_hash,
        "optimizer": assert_optimizer_state(optimizer, START_UPDATE),
    }
    audit["passed"] = (
        all(
            audit[key]
            for key in (
                "model_exact_reload",
                "optimizer_exact_reload",
                "loader_exact_reload",
                "rng_exact_reload",
            )
        )
        and next_hash == EXPECTED_CONTINUATION_NEXT_SHA256
    )
    if not audit["passed"]:
        raise SystemExit(f"continuation source restore failed: {audit}")
    del checkpoint
    return EXPECTED_CONTINUATION_STATE.copy(), audit


def derive_expected_hashes(symbols, loader_states):
    probe = a0.make_loaders_from_states(symbols, loader_states, True)
    result = {}
    for update in range(START_UPDATE, TARGET_UPDATE + 1):
        digest = hashlib.sha256()
        for x, y in a0.update_batches(probe, True):
            digest.update(bytes.fromhex(a0.batch_payload_hash(x, y)))
        result[update] = digest.hexdigest()
    del probe
    sequence_digest = hashlib.sha256(
        "".join(result[update] for update in range(START_UPDATE, TARGET_UPDATE + 1)).encode()
    ).hexdigest()
    if sequence_digest != EXPECTED_REPLAY_SEQUENCE_SHA256:
        raise SystemExit(
            f"replay oracle sequence mismatch: {sequence_digest} != "
            f"{EXPECTED_REPLAY_SEQUENCE_SHA256}"
        )
    if result[START_UPDATE] != EXPECTED_CONTINUATION_NEXT_SHA256:
        raise SystemExit("replay oracle first continuation hash mismatch")
    for milestone, expected in EXPECTED_MILESTONE_NEXT_SHA256.items():
        if result[milestone] != expected:
            raise SystemExit(f"milestone next-hash mismatch at {milestone}")
    return result


def assert_runtime_contract(student, teacher, optimizer, completed_updates):
    contract = a0.smoke_model_contract(student, teacher)
    if sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad) != 1537:
        raise SystemExit("Experiment 2A3 trainable-parameter count changed")
    nonfinite = [
        name
        for name, parameter in student.named_parameters()
        if parameter.requires_grad and not torch.isfinite(parameter).all()
    ]
    if nonfinite:
        raise SystemExit(f"non-finite feedback parameters: {nonfinite}")
    optimizer_report = assert_optimizer_state(optimizer, completed_updates)
    return {"model_contract": contract, "optimizer": optimizer_report, "passed": True}


def assert_frozen_hashes(student, teacher, expected_base, expected_teacher):
    base = model_state_sha256(student, include_topdown=False)
    teacher_hash = model_state_sha256(teacher, include_topdown=False)
    if base != expected_base:
        raise SystemExit(f"frozen student base changed: {base} != {expected_base}")
    if teacher_hash != expected_teacher:
        raise SystemExit(f"frozen teacher changed: {teacher_hash} != {expected_teacher}")
    return {"student_base_sha256": base, "teacher_sha256": teacher_hash, "passed": True}


def snapshot_training_boundary(student, teacher, optimizer, loaders):
    router = student.transformer.topdown_attnres
    return {
        "student_base_sha256": model_state_sha256(student, include_topdown=False),
        "student_topdown_sha256": model_state_sha256(student, include_topdown=True),
        "teacher_sha256": model_state_sha256(teacher, include_topdown=False),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "loaders": copy.deepcopy(a0.snapshot_loaders(loaders)),
        "rng": copy.deepcopy(a0.capture_rng_state()),
        "student_training": student.training,
        "teacher_training": teacher.training,
        "router_instrumentation": router.instrumentation_enabled,
        "router_source_mask": router.masked_source,
    }


def assert_training_boundary_unchanged(before, student, teacher, optimizer, loaders):
    checks = {
        "student_base": before["student_base_sha256"]
        == model_state_sha256(student, include_topdown=False),
        "student_topdown": before["student_topdown_sha256"]
        == model_state_sha256(student, include_topdown=True),
        "teacher": before["teacher_sha256"]
        == model_state_sha256(teacher, include_topdown=False),
        "optimizer": a0.nested_equal(before["optimizer"], optimizer.state_dict()),
        "loaders": a0.nested_equal(before["loaders"], a0.snapshot_loaders(loaders)),
        "rng": a0.nested_equal(before["rng"], a0.capture_rng_state()),
        "student_mode": before["student_training"] == student.training,
        "teacher_mode": before["teacher_training"] == teacher.training,
        "router_instrumentation": before["router_instrumentation"]
        == student.transformer.topdown_attnres.instrumentation_enabled,
        "router_source_mask": before["router_source_mask"]
        == student.transformer.topdown_attnres.masked_source,
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"evaluation changed training state: {checks}")
    return checks


def train_one_update(
    teacher,
    student,
    optimizer,
    loaders,
    symbols,
    update,
    expected_batch_sha256,
    metrics_path,
):
    """Run one cumulative Experiment-2 update, guarding data before mutation."""
    if not START_UPDATE <= update < TARGET_UPDATE:
        raise SystemExit(f"invalid 2A3 update index: {update}")
    preview = a0.next_update_hash(loaders, symbols, replay=True)
    if preview != expected_batch_sha256:
        raise SystemExit(
            f"pre-step replay hash mismatch at update {update}: "
            f"{preview} != {expected_batch_sha256}"
        )
    student.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    student.set_topdown_instrumentation(True)
    loss_total = 0.0
    routing_sums = torch.zeros(len(a0.SOURCE_DEPTHS), dtype=torch.float64)
    entropy_sum = 0.0
    update_hash = hashlib.sha256()
    microbatches = a0.LEGACY_WORLD_SIZE * a0.LEGACY_GRAD_ACCUM
    forward_seconds = 0.0
    backward_seconds = 0.0
    wall_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for x_cpu, y_cpu in a0.update_batches(loaders, replay=True):
            update_hash.update(bytes.fromhex(a0.batch_payload_hash(x_cpu, y_cpu)))
            x = x_cpu.to("cuda", non_blocking=True)
            y = y_cpu.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            memory = a0.teacher_memory(teacher, x, symbols)
            if teacher.training or memory.requires_grad or memory.grad_fn is not None:
                raise SystemExit(f"teacher detach contract failed at update {update}")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = student(
                    x,
                    y,
                    mode="masked_l1_topdown_teacher",
                    feedback_sources=memory,
                )
            del logits
            scaled_loss = loss / microbatches
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            scaled_loss.backward()
            torch.cuda.synchronize()
            backward_seconds += time.perf_counter() - backward_start
            loss_total += scaled_loss.detach().float().item()
            stats = student.get_topdown_stats()
            if stats is None:
                raise SystemExit(f"missing routing instrumentation at update {update}")
            routing_sums += torch.tensor(stats["mean_weights"], dtype=torch.float64)
            entropy_sum += float(stats["mean_entropy"])
            del x, y, memory, loss, scaled_loss
    finally:
        student.set_topdown_instrumentation(False)

    actual_hash = update_hash.hexdigest()
    if actual_hash != expected_batch_sha256:
        raise SystemExit(
            f"consumed replay hash mismatch before optimizer step at update {update}: "
            f"{actual_hash} != {expected_batch_sha256}"
        )
    gradients = a0.gradient_report(student, teacher)
    for name in ("gate", "query", "rmsnorm"):
        if not gradients[name]["present"] or not gradients[name]["finite"]:
            raise SystemExit(f"invalid {name} gradient at update {update}: {gradients}")
    if gradients["base_tensors_with_grad"] or gradients["teacher_tensors_with_grad"]:
        raise SystemExit(f"freeze boundary violated at update {update}: {gradients}")
    if not math.isfinite(loss_total):
        raise SystemExit(f"non-finite loss at update {update}")
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in trainable) != 1537:
        raise SystemExit("trainable parameter count changed before optimizer step")
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    if not torch.isfinite(grad_norm):
        raise SystemExit(f"non-finite gradient norm at update {update}")
    schedule_step = a0.EXPECTED_PARENT_UPDATES + update
    lr = a0.get_lr(schedule_step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    optimizer_report = assert_optimizer_state(optimizer, update + 1)
    nonfinite_trainable = [
        name
        for name, parameter in student.named_parameters()
        if parameter.requires_grad and not torch.isfinite(parameter).all()
    ]
    if nonfinite_trainable:
        raise SystemExit(
            f"non-finite feedback parameters after update {update}: {nonfinite_trainable}"
        )
    router = student.transformer.topdown_attnres
    row = {
        "kind": "train",
        "update": update,
        "completed_updates": update + 1,
        "processed_student_tokens": (update + 1) * a0.GLOBAL_BATCH_TOKENS,
        "global_schedule_step": schedule_step,
        "lr": lr,
        "loss": loss_total,
        "grad_norm": float(grad_norm),
        "gradients": gradients,
        "optimizer": optimizer_report,
        "gate": router.gate.detach().float().item(),
        "gate_coefficient": router.gate.detach().float().tanh().item(),
        "query_norm": router.query.detach().float().norm().item(),
        "rmsnorm_displacement": (
            router.norm.weight.detach().float() - 1
        ).norm().item(),
        "routing_weights": {
            f"v{depth}": value
            for depth, value in zip(
                a0.SOURCE_DEPTHS, (routing_sums / microbatches).tolist()
            )
        },
        "routing_entropy": entropy_sum / microbatches,
        "teacher_eval_no_grad": True,
        "trainable_parameters_finite": True,
        "global_batch_sha256": actual_hash,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "wall_seconds": time.perf_counter() - wall_start,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }
    finite_scalars = (
        "loss",
        "grad_norm",
        "lr",
        "gate",
        "gate_coefficient",
        "query_norm",
        "rmsnorm_displacement",
        "routing_entropy",
    )
    if not all(math.isfinite(float(row[name])) for name in finite_scalars):
        raise SystemExit(f"non-finite training row at update {update}: {row}")
    if not math.isclose(
        sum(row["routing_weights"].values()), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise SystemExit(f"routing simplex failed at update {update}")
    durable_append_jsonl(metrics_path, row)
    print(
        f"update {update + 1:02d}/{TARGET_UPDATE} loss={loss_total:.6f} "
        f"gate={row['gate']:.6g} qnorm={row['query_norm']:.6g}",
        flush=True,
    )
    return row


@torch.no_grad()
def evaluate_milestone(student, teacher, symbols, device, completed_updates, full=False):
    if completed_updates not in MILESTONES:
        raise SystemExit(f"invalid evaluation milestone: {completed_updates}")
    if full != (completed_updates == TARGET_UPDATE):
        raise SystemExit("only update 477 may use the full control matrix")
    loader = symbols["DataLoaderLite"](
        B=a0.VALIDATION_B,
        T=a0.T,
        process_rank=0,
        num_processes=1,
        split="val",
    )
    permutation = symbols["fixed_derangement"](a0.VALIDATION_B, device)
    expected_indices = torch.arange(a0.VALIDATION_B, device=device)
    if (
        torch.any(permutation == expected_indices)
        or not torch.equal(torch.sort(permutation).values, expected_indices)
    ):
        raise SystemExit("validation shuffle is not a fixed-point-free permutation")
    names = ["real_feedback", "shuffled_feedback"]
    if full:
        names = [
            "full_context",
            "masked_l1_no_feedback",
            "real_feedback",
            "shuffled_feedback",
            "zero_feedback",
        ] + [f"mask_v{depth}" for depth in a0.SOURCE_DEPTHS]
    batch_losses = {name: [] for name in names}
    batch_payload_sha256 = []
    routing_weights = torch.zeros(len(a0.SOURCE_DEPTHS), dtype=torch.float64)
    routing_entropy = 0.0
    validation_hash = hashlib.sha256()
    zero_equals_masked = True
    prior_student_training = student.training
    prior_teacher_training = teacher.training
    router = student.transformer.topdown_attnres
    prior_instrumentation = router.instrumentation_enabled
    prior_source_mask = router.masked_source
    if prior_source_mask is not None:
        raise SystemExit("milestone evaluation requires no active source mask")
    student.eval()
    teacher.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for batch_index in range(a0.VALIDATION_BATCHES):
            x_cpu, y_cpu = loader.next_batch()
            payload_digest = a0.batch_payload_hash(x_cpu, y_cpu)
            batch_payload_sha256.append(payload_digest)
            validation_hash.update(bytes.fromhex(payload_digest))
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            if full:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = student(x, y, mode="full_context")
                del logits
                batch_losses["full_context"].append(loss.detach().double().item())
                del loss
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, masked_loss = student(x, y, mode="masked_l1_no_feedback")
                del logits
                masked_value = masked_loss.detach().double().item()
                batch_losses["masked_l1_no_feedback"].append(masked_value)
            memory = a0.teacher_memory(teacher, x, symbols)
            if memory.requires_grad or memory.grad_fn is not None or teacher.training:
                raise SystemExit("teacher evaluation memory violated detach/eval contract")
            student.set_topdown_instrumentation(True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, real_loss = student(
                    x,
                    y,
                    mode="masked_l1_topdown_teacher",
                    feedback_sources=memory,
                )
            del logits
            stats = student.get_topdown_stats()
            student.set_topdown_instrumentation(False)
            if stats is None or stats.get("source_depths") != list(a0.SOURCE_DEPTHS):
                raise SystemExit("invalid milestone routing instrumentation")
            routing_weights += torch.tensor(stats["mean_weights"], dtype=torch.float64)
            routing_entropy += float(stats["mean_entropy"])
            batch_losses["real_feedback"].append(real_loss.detach().double().item())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, shuffled_loss = student(
                    x,
                    y,
                    mode="masked_l1_shuffled_feedback",
                    feedback_sources=memory,
                    feedback_permutation=permutation,
                )
            del logits
            batch_losses["shuffled_feedback"].append(
                shuffled_loss.detach().double().item()
            )
            if full:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, zero_loss = student(
                        x,
                        y,
                        mode="masked_l1_topdown_teacher",
                        feedback_sources=memory,
                        feedback_gate_override=0.0,
                    )
                del logits
                zero_value = zero_loss.detach().double().item()
                batch_losses["zero_feedback"].append(zero_value)
                zero_equals_masked &= torch.equal(zero_loss, masked_loss)
                for depth in a0.SOURCE_DEPTHS:
                    student.set_topdown_source_mask(depth)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, ablated_loss = student(
                            x,
                            y,
                            mode="masked_l1_topdown_teacher",
                            feedback_sources=memory,
                        )
                    del logits
                    batch_losses[f"mask_v{depth}"].append(
                        ablated_loss.detach().double().item()
                    )
                    del ablated_loss
                student.set_topdown_source_mask(None)
                del zero_loss, masked_loss
            del x, y, memory, real_loss, shuffled_loss
            print(
                f"milestone {completed_updates} validation "
                f"{batch_index + 1:02d}/{a0.VALIDATION_BATCHES}",
                flush=True,
            )
    finally:
        student.set_topdown_source_mask(prior_source_mask)
        student.set_topdown_instrumentation(prior_instrumentation)
        student.train(prior_student_training)
        teacher.train(prior_teacher_training)

    validation_digest = validation_hash.hexdigest()
    if validation_digest != a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256:
        raise SystemExit(f"canonical validation prefix mismatch: {validation_digest}")
    if any(
        len(values) != a0.VALIDATION_BATCHES
        or not all(math.isfinite(value) for value in values)
        for values in batch_losses.values()
    ):
        raise SystemExit("milestone batch losses are incomplete or non-finite")
    losses = {
        name: sum(values) / a0.VALIDATION_BATCHES
        for name, values in batch_losses.items()
    }
    if full:
        if losses["full_context"] != PINNED_FULL:
            raise SystemExit(
                f"full-context reference drifted: {losses['full_context']} != {PINNED_FULL}"
            )
        if losses["masked_l1_no_feedback"] != PINNED_MASKED:
            raise SystemExit(
                "masked reference drifted: "
                f"{losses['masked_l1_no_feedback']} != {PINNED_MASKED}"
            )
        if not zero_equals_masked:
            raise SystemExit("forced-zero feedback did not equal masked control batchwise")
        full_loss = losses["full_context"]
        masked_loss = losses["masked_l1_no_feedback"]
    else:
        full_loss = PINNED_FULL
        masked_loss = PINNED_MASKED
        losses = {
            "full_context_reference": PINNED_FULL,
            "masked_l1_no_feedback_reference": PINNED_MASKED,
            **losses,
        }
    damage = masked_loss - full_loss
    total_recovery = masked_loss - losses["real_feedback"]
    sequence_specific = losses["shuffled_feedback"] - losses["real_feedback"]
    paired = paired_statistics(
        batch_losses["real_feedback"], batch_losses["shuffled_feedback"]
    )
    paired.update(
        {
            "mean_real_minus_shuffled": -paired["mean"],
            "median_real_minus_shuffled": -paired["median"],
            "real_beats_shuffled_count": paired["positive_count"],
            "shuffled_beats_real_count": paired["negative_count"],
        }
    )
    mean_weights = routing_weights / a0.VALIDATION_BATCHES
    entropy = routing_entropy / a0.VALIDATION_BATCHES
    if not math.isclose(float(mean_weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit("milestone routing weights do not sum to one")
    router = student.transformer.topdown_attnres
    routing_scalars = {
        "gate": router.gate.detach().float().item(),
        "gate_coefficient": router.gate.detach().float().tanh().item(),
        "query_norm": router.query.detach().float().norm().item(),
        "rmsnorm_displacement": (
            router.norm.weight.detach().float() - 1
        ).norm().item(),
        "mean_tokenwise_entropy": entropy,
        "normalized_entropy": entropy / math.log(len(a0.SOURCE_DEPTHS)),
        "effective_source_count": math.exp(entropy),
    }
    if not all(math.isfinite(float(value)) for value in routing_scalars.values()):
        raise SystemExit(f"non-finite milestone router statistics: {routing_scalars}")
    if not all(math.isfinite(float(value)) for value in mean_weights.tolist()):
        raise SystemExit("non-finite milestone routing weights")
    result = {
        "completed_updates": completed_updates,
        "processed_student_tokens": completed_updates * a0.GLOBAL_BATCH_TOKENS,
        "full_control_matrix": full,
        "validation_batches": a0.VALIDATION_BATCHES,
        "validation_B": a0.VALIDATION_B,
        "validation_T": a0.T,
        "validation_global_batches_sha256": validation_digest,
        "frozen_references_reused": not full,
        "losses": losses,
        "batch_losses": batch_losses,
        "batch_payload_sha256": batch_payload_sha256,
        "damage": damage,
        "total_recovery": total_recovery,
        "total_recovery_fraction": total_recovery / damage,
        "sequence_specific_recovery": sequence_specific,
        "sequence_specific_recovery_fraction": sequence_specific / damage,
        "sequence_specific_share_of_total_recovery": (
            sequence_specific / total_recovery if total_recovery > 0 else None
        ),
        "paired_shuffled_minus_real": paired,
        "requested_real_minus_shuffled": {
            "mean": paired["mean_real_minus_shuffled"],
            "median": paired["median_real_minus_shuffled"],
            "real_feedback_beats_shuffled_batches": paired[
                "real_beats_shuffled_count"
            ],
            "shuffled_feedback_beats_real_batches": paired[
                "shuffled_beats_real_count"
            ],
            "ties": paired["tie_count"],
        },
        "routing": {
            **routing_scalars,
            "mean_weights": {
                f"v{depth}": value
                for depth, value in zip(a0.SOURCE_DEPTHS, mean_weights.tolist())
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }
    if full:
        result["zero_equals_masked_each_batch"] = zero_equals_masked
        result["source_ablation"] = {
            f"v{depth}": {
                "validation_loss": losses[f"mask_v{depth}"],
                "delta_vs_real_feedback": (
                    losses[f"mask_v{depth}"] - losses["real_feedback"]
                ),
                "paired_ablation_minus_real": paired_statistics(
                    batch_losses["real_feedback"], batch_losses[f"mask_v{depth}"]
                ),
            }
            for depth in a0.SOURCE_DEPTHS
        }
    return result


def validate_evaluation_artifact(
    artifact,
    completed_updates,
    checkpoint_sha256,
    metadata_sha256,
    full,
    expected_source_hashes=None,
):
    expected = {
        "completed_updates": completed_updates,
        "processed_student_tokens": completed_updates * a0.GLOBAL_BATCH_TOKENS,
        "checkpoint_sha256": checkpoint_sha256,
        "metadata_sha256": metadata_sha256,
        "full_control_matrix": full,
        "validation_global_batches_sha256": a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256,
    }
    mismatches = {
        key: (artifact.get(key), value)
        for key, value in expected.items()
        if artifact.get(key) != value
    }
    vectors = artifact.get("batch_losses", {})
    real = vectors.get("real_feedback", [])
    shuffled = vectors.get("shuffled_feedback", [])
    payload_hashes = artifact.get("batch_payload_sha256", [])
    required_vectors = {"real_feedback", "shuffled_feedback"}
    if full:
        required_vectors.update(
            {
                "full_context",
                "masked_l1_no_feedback",
                "zero_feedback",
                *(f"mask_v{depth}" for depth in a0.SOURCE_DEPTHS),
            }
        )
    vectors_valid = all(
        len(vectors.get(name, [])) == a0.VALIDATION_BATCHES
        and all(math.isfinite(float(value)) for value in vectors[name])
        for name in required_vectors
    )
    recomputed_pair = None
    if len(real) == 20 and len(shuffled) == 20:
        recomputed_pair = paired_statistics(real, shuffled)
    stored_pair = artifact.get("paired_shuffled_minus_real", {})
    pair_valid = recomputed_pair is not None and all(
        stored_pair.get(key) == recomputed_pair[key]
        for key in (
            "n",
            "differences",
            "mean",
            "median",
            "minimum",
            "maximum",
            "positive_count",
            "negative_count",
            "tie_count",
            "sample_standard_deviation",
            "standard_error",
            "ci95_lower",
            "ci95_upper",
            "ci95_note",
        )
    )
    requested = artifact.get("requested_real_minus_shuffled", {})
    requested_valid = recomputed_pair is not None and requested == {
        "mean": -recomputed_pair["mean"],
        "median": -recomputed_pair["median"],
        "real_feedback_beats_shuffled_batches": recomputed_pair["positive_count"],
        "shuffled_feedback_beats_real_batches": recomputed_pair["negative_count"],
        "ties": recomputed_pair["tie_count"],
    }
    losses = artifact.get("losses", {})
    means_valid = vectors_valid and all(
        losses.get(name) == sum(vectors.get(name, [])) / a0.VALIDATION_BATCHES
        for name in required_vectors
    )
    if full:
        full_value = losses.get("full_context")
        masked_value = losses.get("masked_l1_no_feedback")
    else:
        full_value = losses.get("full_context_reference")
        masked_value = losses.get("masked_l1_no_feedback_reference")
    real_value = losses.get("real_feedback")
    shuffled_value = losses.get("shuffled_feedback")
    aggregate_values = (full_value, masked_value, real_value, shuffled_value)
    aggregate_finite = all(
        value is not None and math.isfinite(float(value)) for value in aggregate_values
    )
    arithmetic_valid = False
    if aggregate_finite:
        damage = masked_value - full_value
        recovery = masked_value - real_value
        specific = shuffled_value - real_value
        arithmetic_valid = (
            damage > 0
            and artifact.get("damage") == damage
            and artifact.get("total_recovery") == recovery
            and artifact.get("total_recovery_fraction") == recovery / damage
            and artifact.get("sequence_specific_recovery") == specific
            and artifact.get("sequence_specific_recovery_fraction")
            == specific / damage
            and artifact.get("sequence_specific_share_of_total_recovery")
            == (specific / recovery if recovery > 0 else None)
        )
    routing = artifact.get("routing", {})
    weights = routing.get("mean_weights", {})
    routing_scalars = [
        routing.get("gate"),
        routing.get("gate_coefficient"),
        routing.get("query_norm"),
        routing.get("rmsnorm_displacement"),
        routing.get("mean_tokenwise_entropy"),
        routing.get("normalized_entropy"),
        routing.get("effective_source_count"),
    ]
    routing_valid = (
        set(weights) == {f"v{depth}" for depth in a0.SOURCE_DEPTHS}
        and all(value is not None and math.isfinite(float(value)) for value in routing_scalars)
        and all(math.isfinite(float(value)) for value in weights.values())
        and all(0.0 <= float(value) <= 1.0 for value in weights.values())
        and math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6)
        and routing["mean_tokenwise_entropy"] >= 0
        and routing["mean_tokenwise_entropy"] <= math.log(len(a0.SOURCE_DEPTHS)) + 1e-6
        and routing["normalized_entropy"]
        == routing["mean_tokenwise_entropy"] / math.log(len(a0.SOURCE_DEPTHS))
        and routing["effective_source_count"]
        == math.exp(routing["mean_tokenwise_entropy"])
    )
    final_valid = vectors_valid
    if full:
        final_valid = final_valid and (
            full_value == PINNED_FULL
            and masked_value == PINNED_MASKED
            and artifact.get("zero_equals_masked_each_batch") is True
            and vectors["zero_feedback"] == vectors["masked_l1_no_feedback"]
            and set(artifact.get("source_ablation", {}))
            == {f"v{depth}" for depth in a0.SOURCE_DEPTHS}
        )
        if final_valid:
            for depth in a0.SOURCE_DEPTHS:
                name = f"v{depth}"
                vector_name = f"mask_v{depth}"
                stored = artifact["source_ablation"][name]
                recomputed = paired_statistics(
                    vectors["real_feedback"], vectors[vector_name]
                )
                final_valid &= (
                    stored.get("validation_loss") == losses[vector_name]
                    and stored.get("delta_vs_real_feedback")
                    == losses[vector_name] - real_value
                    and all(
                        stored.get("paired_ablation_minus_real", {}).get(key)
                        == recomputed[key]
                        for key in (
                            "n",
                            "differences",
                            "mean",
                            "median",
                            "minimum",
                            "maximum",
                            "positive_count",
                            "negative_count",
                            "tie_count",
                            "sample_standard_deviation",
                            "standard_error",
                            "ci95_lower",
                            "ci95_upper",
                            "ci95_note",
                        )
                    )
                )
    source_milestone_valid = True
    if completed_updates == START_UPDATE:
        source_milestone_valid = (
            real_value == PINNED_START_REAL
            and shuffled_value == PINNED_START_SHUFFLED
        )
    source_hashes_valid = (
        expected_source_hashes is None
        or artifact.get("source_file_sha256") == expected_source_hashes
    )
    isolation = artifact.get("state_isolation", {})
    required_isolation_keys = {
        "student_base",
        "student_topdown",
        "teacher",
        "optimizer",
        "loaders",
        "rng",
        "student_mode",
        "teacher_mode",
        "router_instrumentation",
        "router_source_mask",
        "passed",
    }
    isolation_valid = set(isolation) == required_isolation_keys and isolation.get("passed") is True and all(
        value is True for key, value in isolation.items() if key != "passed"
    )
    reference_valid = full or (
        full_value == PINNED_FULL and masked_value == PINNED_MASKED
    )
    if (
        mismatches
        or not vectors_valid
        or not pair_valid
        or not requested_valid
        or not means_valid
        or not arithmetic_valid
        or not routing_valid
        or not final_valid
        or not source_milestone_valid
        or not source_hashes_valid
        or not isolation_valid
        or not reference_valid
        or len(payload_hashes) != 20
        or hashlib.sha256(
            b"".join(bytes.fromhex(value) for value in payload_hashes)
        ).hexdigest()
        != a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256
    ):
        raise SystemExit(
            f"milestone evaluation artifact mismatch: {mismatches}, "
            f"paired_lengths={(len(real), len(shuffled))}"
        )
    return artifact


def load_source_evaluation():
    """Validate and reuse the committed 100M evaluation without rerunning it."""
    path = source_evaluation_path()
    if not path.is_file():
        raise SystemExit("committed Experiment 2A2 100M evaluation is missing")
    source_metadata, _ = load_source_metadata()
    artifact = json.loads(path.read_text())
    return validate_evaluation_artifact(
        artifact,
        START_UPDATE,
        EXPECTED_CONTINUATION_SHA256,
        hashlib.sha256(
            json.dumps(source_metadata, sort_keys=True).encode()
        ).hexdigest(),
        True,
        source_metadata["source_file_sha256"],
    )


def bound_milestone_evaluation(
    path,
    student,
    teacher,
    optimizer,
    loaders,
    symbols,
    device,
    completed_updates,
    checkpoint_sha256,
    metadata,
):
    path = Path(path)
    quarantine_incomplete(path)
    metadata_sha256 = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    full = completed_updates == TARGET_UPDATE
    if path.exists():
        artifact = json.loads(path.read_text())
        return validate_evaluation_artifact(
            artifact,
            completed_updates,
            checkpoint_sha256,
            metadata_sha256,
            full,
            metadata["source_file_sha256"],
        )
    before = snapshot_training_boundary(student, teacher, optimizer, loaders)
    result = evaluate_milestone(
        student, teacher, symbols, device, completed_updates, full=full
    )
    isolation = assert_training_boundary_unchanged(
        before, student, teacher, optimizer, loaders
    )
    result.update(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "metadata_sha256": metadata_sha256,
            "source_file_sha256": metadata["source_file_sha256"],
            "state_isolation": isolation,
        }
    )
    validate_evaluation_artifact(
        result,
        completed_updates,
        checkpoint_sha256,
        metadata_sha256,
        full,
        metadata["source_file_sha256"],
    )
    durable_write_json(path, result)
    return result


@torch.no_grad()
def end_to_end_future_invariance_test(
    student, teacher, symbols, device, suffix_start=512
):
    """Verify real-feedback student logits before a changed suffix are exact."""
    loader = symbols["DataLoaderLite"](
        B=2, T=a0.T, process_rank=0, num_processes=1, split="val"
    )
    first, _ = loader.next_batch()
    second = first.clone()
    second[:, suffix_start:] = (second[:, suffix_start:] + 1) % 50257
    prior_student_training = student.training
    prior_teacher_training = teacher.training
    student.eval()
    teacher.eval()
    rows = []
    try:
        for tokens in (first, second):
            tokens = tokens.to(device)
            memory = a0.teacher_memory(teacher, tokens, symbols)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = student(
                    tokens,
                    mode="masked_l1_topdown_teacher",
                    feedback_sources=memory,
                )
            rows.append(logits[:, :suffix_start].detach().cpu())
            del tokens, memory, logits
    finally:
        student.train(prior_student_training)
        teacher.train(prior_teacher_training)
    passed = torch.equal(rows[0], rows[1])
    report = {
        "B": 2,
        "T": a0.T,
        "changed_suffix_starts_at": suffix_start,
        "compared_logit_positions": [0, suffix_start - 1],
        "prefix_logits_bit_exact": passed,
        "maximum_absolute_difference": (rows[0].float() - rows[1].float()).abs().max().item(),
        "passed": passed,
    }
    if not passed:
        raise SystemExit(f"end-to-end future-token invariance failed: {report}")
    return report


def hellaswag_dataset_report(download=False):
    path = REPO_ROOT / "hellaswag" / "hellaswag_val.jsonl"
    if download and not path.is_file():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
            path,
        )
    if not path.is_file():
        raise SystemExit(
            "full HellaSwag validation data is missing; run the optimizer-free preflight"
        )
    digest = a0.file_sha256(path)
    with path.open() as handle:
        examples = sum(1 for line in handle if line.strip())
    report = {
        "path": str(path.resolve()),
        "sha256": digest,
        "examples": examples,
        "split": "val",
        "passed": digest == EXPECTED_HELLASWAG_VAL_SHA256 and examples == 10042,
    }
    if not report["passed"]:
        raise SystemExit(f"HellaSwag validation dataset mismatch: {report}")
    return report


def iterate_hellaswag_examples():
    path = Path(hellaswag_dataset_report(download=False)["path"])
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def render_hellaswag_example(example):
    """Exact local copy of the upstream completion-scoring tensorization."""
    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    context = encoder.encode(example["ctx"])
    token_rows = []
    mask_rows = []
    for ending in example["endings"]:
        ending_tokens = encoder.encode(" " + ending)
        token_rows.append(context + ending_tokens)
        mask_rows.append([0] * len(context) + [1] * len(ending_tokens))
    width = max(len(row) for row in token_rows)
    tokens = torch.zeros((4, width), dtype=torch.long)
    mask = torch.zeros((4, width), dtype=torch.long)
    for index, (token_row, mask_row) in enumerate(zip(token_rows, mask_rows)):
        tokens[index, : len(token_row)] = torch.tensor(token_row)
        mask[index, : len(mask_row)] = torch.tensor(mask_row)
    return tokens, mask, int(example["label"])


def hellaswag_candidate_scores(tokens, mask, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1),
        reduction="none",
    ).view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    counts = shift_mask.sum(dim=1)
    if torch.any(counts <= 0):
        raise SystemExit("HellaSwag candidate has an empty completion mask")
    return (losses * shift_mask).sum(dim=1) / counts


@torch.no_grad()
def hellaswag_candidate_isolation_test(student, teacher, symbols, device):
    """Prove candidates/examples are independent and feedback remains detached."""
    prior_rng = copy.deepcopy(a0.capture_rng_state())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260814)
    first = torch.randint(0, 50257, (4, 96), generator=generator)
    changed = first.clone()
    changed[1:] = torch.randint(0, 50257, (3, 96), generator=generator)
    other = torch.randint(0, 50257, (4, 96), generator=generator)
    prior_student_training = student.training
    prior_teacher_training = teacher.training
    student.eval()
    teacher.eval()

    def real_forward(tokens_cpu):
        tokens = tokens_cpu.to(device)
        memory = a0.teacher_memory(teacher, tokens, symbols)
        if memory.requires_grad or memory.grad_fn is not None:
            raise SystemExit("HellaSwag teacher memory is not detached")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = student(
                tokens,
                mode="masked_l1_topdown_teacher",
                feedback_sources=memory,
            )
        result = (memory[:, 0].detach().cpu(), logits[0].detach().cpu())
        del tokens, memory, logits
        return result

    try:
        first_memory, first_logits = real_forward(first)
        changed_memory, changed_logits = real_forward(changed)
        real_forward(other)
        repeated_memory, repeated_logits = real_forward(first)
        example = next(iterate_hellaswag_examples())
        example_tokens_cpu, example_mask_cpu, _ = render_hellaswag_example(example)
        example_tokens = example_tokens_cpu.to(device)
        example_mask = example_mask_cpu.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            upstream_full_logits, _ = teacher(example_tokens)
            upstream_masked_logits, _ = student(
                example_tokens, mode="masked_l1_no_feedback"
            )
        upstream_memory = a0.teacher_memory(teacher, example_tokens, symbols)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            upstream_real_logits, _ = student(
                example_tokens,
                mode="masked_l1_topdown_teacher",
                feedback_sources=upstream_memory,
            )
            upstream_zero_logits, _ = student(
                example_tokens,
                mode="masked_l1_topdown_teacher",
                feedback_sources=upstream_memory,
                feedback_gate_override=0.0,
            )
        upstream_scores = [
            hellaswag_candidate_scores(example_tokens, example_mask, logits)
            for logits in (
                upstream_full_logits,
                upstream_masked_logits,
                upstream_real_logits,
                upstream_zero_logits,
            )
        ]
        upstream_finite = all(torch.isfinite(scores).all() for scores in upstream_scores)
        upstream_zero_equals_masked = torch.equal(upstream_scores[1], upstream_scores[3])
        del (
            example_tokens,
            example_mask,
            upstream_memory,
            upstream_full_logits,
            upstream_masked_logits,
            upstream_real_logits,
            upstream_zero_logits,
            upstream_scores,
        )
    finally:
        student.train(prior_student_training)
        teacher.train(prior_teacher_training)
        a0.restore_rng_state(prior_rng)
    report = {
        "candidate_zero_teacher_memory_bit_exact_when_other_candidates_change": torch.equal(
            first_memory, changed_memory
        ),
        "candidate_zero_logits_bit_exact_when_other_candidates_change": torch.equal(
            first_logits, changed_logits
        ),
        "example_reset_teacher_memory_bit_exact": torch.equal(
            first_memory, repeated_memory
        ),
        "example_reset_logits_bit_exact": torch.equal(first_logits, repeated_logits),
        "position_zero_memory_exactly_zero": torch.count_nonzero(
            first_memory[:, 0]
        ).item()
        == 0,
        "upstream_example_scores_finite": upstream_finite,
        "upstream_example_zero_equals_masked": upstream_zero_equals_masked,
        "shuffled_feedback_evaluated": False,
        "shuffled_feedback_skip_reason": (
            "candidate-row shuffling would exchange teacher memory between the four "
            "answer alternatives and violate strict candidate isolation"
        ),
    }
    report["passed"] = all(
        value is True
        for key, value in report.items()
        if key not in {"shuffled_feedback_evaluated", "shuffled_feedback_skip_reason"}
    )
    if not report["passed"]:
        raise SystemExit(f"HellaSwag candidate/example isolation failed: {report}")
    return report


@torch.no_grad()
def evaluate_hellaswag(student, teacher, symbols, device):
    dataset = hellaswag_dataset_report(download=False)
    modes = (
        "full_context",
        "masked_l1_no_feedback",
        "real_feedback",
        "zero_feedback",
    )
    correct = {mode: 0 for mode in modes}
    predictions = {mode: [] for mode in modes}
    labels = []
    zero_equals_masked = True
    prior_student_training = student.training
    prior_teacher_training = teacher.training
    prior_rng = copy.deepcopy(a0.capture_rng_state())
    student.eval()
    teacher.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for index, example in enumerate(iterate_hellaswag_examples()):
            tokens_cpu, mask_cpu, label = render_hellaswag_example(example)
            if tokens_cpu.shape[0] != 4 or tokens_cpu.shape[1] > 1024:
                raise SystemExit(
                    f"invalid HellaSwag candidate tensor at example {index}: "
                    f"{tuple(tokens_cpu.shape)}"
                )
            tokens = tokens_cpu.to(device)
            mask = mask_cpu.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                full_logits, _ = teacher(tokens)
            full_scores = hellaswag_candidate_scores(tokens, mask, full_logits)
            del full_logits
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masked_logits, _ = student(
                    tokens, mode="masked_l1_no_feedback"
                )
            masked_scores = hellaswag_candidate_scores(tokens, mask, masked_logits)
            del masked_logits
            memory = a0.teacher_memory(teacher, tokens, symbols)
            if memory.requires_grad or memory.grad_fn is not None or teacher.training:
                raise SystemExit("HellaSwag teacher memory violated eval/detach contract")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                real_logits, _ = student(
                    tokens,
                    mode="masked_l1_topdown_teacher",
                    feedback_sources=memory,
                )
            real_scores = hellaswag_candidate_scores(tokens, mask, real_logits)
            del real_logits
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                zero_logits, _ = student(
                    tokens,
                    mode="masked_l1_topdown_teacher",
                    feedback_sources=memory,
                    feedback_gate_override=0.0,
                )
            zero_scores = hellaswag_candidate_scores(tokens, mask, zero_logits)
            del zero_logits, memory
            zero_equals_masked &= torch.equal(zero_scores, masked_scores)
            scores = {
                "full_context": full_scores,
                "masked_l1_no_feedback": masked_scores,
                "real_feedback": real_scores,
                "zero_feedback": zero_scores,
            }
            labels.append(int(label))
            for mode, values in scores.items():
                if not torch.isfinite(values).all():
                    raise SystemExit(
                        f"non-finite HellaSwag scores at example {index}, mode {mode}"
                    )
                prediction = int(values.argmin().item())
                predictions[mode].append(prediction)
                correct[mode] += int(prediction == label)
            del tokens, mask, full_scores, masked_scores, real_scores, zero_scores
            if (index + 1) % 100 == 0 or index + 1 == 10042:
                print(f"HellaSwag {index + 1:05d}/10042", flush=True)
    finally:
        student.train(prior_student_training)
        teacher.train(prior_teacher_training)
        a0.restore_rng_state(prior_rng)
    if len(labels) != 10042 or not zero_equals_masked:
        raise SystemExit(
            f"HellaSwag completeness/zero equivalence failed: "
            f"examples={len(labels)}, zero_equals_masked={zero_equals_masked}"
        )
    if correct["full_context"] != 2532:
        raise SystemExit(
            f"full-context HellaSwag anchor drifted: {correct['full_context']} != 2532"
        )
    return {
        "dataset": dataset,
        "examples": len(labels),
        "labels": labels,
        "predictions": predictions,
        "correct": correct,
        "accuracy": {mode: correct[mode] / len(labels) for mode in modes},
        "zero_equals_masked_each_example": zero_equals_masked,
        "shuffled_feedback": {
            "evaluated": False,
            "reason": (
                "candidate-row shuffling would exchange memory between answer "
                "alternatives and contaminate HellaSwag scoring"
            ),
        },
        "historical_references": {
            "standard_gpt2_500m": {
                "correct": 2568,
                "examples": 10042,
                "accuracy": 2568 / 10042,
            },
            "full_attnres_500m": {
                "correct": 2532,
                "examples": 10042,
                "accuracy": 2532 / 10042,
            },
        },
        "equal_token_pretraining_comparison": False,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "passed": True,
    }


def hellaswag_path(run_dir):
    return Path(run_dir) / "evaluations" / f"hellaswag_updates_{TARGET_UPDATE:06d}.json"


def validate_hellaswag_artifact(
    artifact, checkpoint_sha256, metadata_sha256, source_file_sha256
):
    modes = {
        "full_context",
        "masked_l1_no_feedback",
        "real_feedback",
        "zero_feedback",
    }
    labels = artifact.get("labels", [])
    predictions = artifact.get("predictions", {})
    correct = artifact.get("correct", {})
    accuracy = artifact.get("accuracy", {})
    candidate = artifact.get("candidate_isolation", {})
    candidate_required = {
        "candidate_zero_teacher_memory_bit_exact_when_other_candidates_change",
        "candidate_zero_logits_bit_exact_when_other_candidates_change",
        "example_reset_teacher_memory_bit_exact",
        "example_reset_logits_bit_exact",
        "position_zero_memory_exactly_zero",
        "upstream_example_scores_finite",
        "upstream_example_zero_equals_masked",
        "shuffled_feedback_evaluated",
        "shuffled_feedback_skip_reason",
        "passed",
    }
    candidate_valid = (
        set(candidate) == candidate_required
        and candidate.get("passed") is True
        and candidate.get("shuffled_feedback_evaluated") is False
        and all(
            candidate.get(key) is True
            for key in candidate_required
            if key
            not in {
                "passed",
                "shuffled_feedback_evaluated",
                "shuffled_feedback_skip_reason",
            }
        )
    )
    valid = (
        artifact.get("passed") is True
        and artifact.get("completed_updates") == TARGET_UPDATE
        and artifact.get("processed_student_tokens")
        == TARGET_UPDATE * a0.GLOBAL_BATCH_TOKENS
        and artifact.get("checkpoint_sha256") == checkpoint_sha256
        and artifact.get("metadata_sha256") == metadata_sha256
        and artifact.get("source_file_sha256") == source_file_sha256
        and artifact.get("examples") == 10042
        and artifact.get("dataset", {}).get("sha256")
        == EXPECTED_HELLASWAG_VAL_SHA256
        and len(labels) == 10042
        and set(predictions) == modes
        and set(correct) == modes
        and set(accuracy) == modes
        and all(len(predictions[mode]) == 10042 for mode in modes)
        and all(
            correct[mode]
            == sum(int(pred == label) for pred, label in zip(predictions[mode], labels))
            and accuracy[mode] == correct[mode] / 10042
            for mode in modes
        )
        and correct.get("full_context") == 2532
        and predictions.get("zero_feedback")
        == predictions.get("masked_l1_no_feedback")
        and artifact.get("zero_equals_masked_each_example") is True
        and artifact.get("shuffled_feedback", {}).get("evaluated") is False
        and candidate_valid
        and validate_isolation_record(artifact.get("state_isolation", {}))
    )
    if not valid:
        raise SystemExit("HellaSwag artifact failed strict validation")
    return artifact


def bound_hellaswag_evaluation(
    path,
    student,
    teacher,
    optimizer,
    loaders,
    symbols,
    device,
    checkpoint_sha256,
    metadata,
):
    path = Path(path)
    quarantine_incomplete(path)
    metadata_sha256 = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    if path.exists():
        return validate_hellaswag_artifact(
            json.loads(path.read_text()),
            checkpoint_sha256,
            metadata_sha256,
            metadata["source_file_sha256"],
        )
    before = snapshot_training_boundary(student, teacher, optimizer, loaders)
    isolation_test = hellaswag_candidate_isolation_test(
        student, teacher, symbols, device
    )
    result = evaluate_hellaswag(student, teacher, symbols, device)
    state_isolation = assert_training_boundary_unchanged(
        before, student, teacher, optimizer, loaders
    )
    result.update(
        {
            "completed_updates": TARGET_UPDATE,
            "processed_student_tokens": TARGET_UPDATE * a0.GLOBAL_BATCH_TOKENS,
            "checkpoint_sha256": checkpoint_sha256,
            "metadata_sha256": metadata_sha256,
            "source_file_sha256": metadata["source_file_sha256"],
            "candidate_isolation": isolation_test,
            "state_isolation": state_isolation,
        }
    )
    validate_hellaswag_artifact(
        result,
        checkpoint_sha256,
        metadata_sha256,
        metadata["source_file_sha256"],
    )
    durable_write_json(path, result)
    return result


def classify_trajectory(rows):
    if len(rows) != 4:
        raise ValueError("trajectory classification requires exactly four milestones")
    completed = [row.get("completed_updates") for row in rows]
    if completed != [START_UPDATE, *MILESTONES]:
        raise ValueError(
            "trajectory milestones must be ordered exactly as updates 191,286,381,477"
        )
    values = []
    for row in rows:
        value = row.get("sequence_specific_recovery")
        if value is None and "evaluation" in row:
            value = row["evaluation"].get("sequence_specific_recovery")
        if value is None or not math.isfinite(float(value)):
            raise ValueError("trajectory row lacks finite sequence-specific recovery")
        values.append(float(value))
    q100, q150, q200, q250 = values
    gain_1 = q150 - q100
    gain_2 = q200 - q150
    gain_3 = q250 - q200
    if q250 <= 0 or gain_3 < 0:
        label = "MEMORY SIGNAL REVERSING"
    elif gain_3 <= 0.25 * max(gain_1, gain_2, 1e-12):
        label = "MEMORY SIGNAL SATURATING"
    elif q250 > q200 > q150 > q100 and gain_3 > gain_2 > gain_1 > 0:
        label = "MEMORY SIGNAL STILL ACCELERATING"
    else:
        label = "MEMORY SIGNAL STRENGTHENING"
    return {
        "label": label,
        "values": values,
        "successive_gains": [gain_1, gain_2, gain_3],
        "rule": CLASSIFICATION_RULE,
    }


def checkpoint_path(run_dir, completed_updates):
    return Path(run_dir) / "checkpoints" / f"checkpoint_updates_{completed_updates:06d}.pt"


def evaluation_path(run_dir, completed_updates):
    return Path(run_dir) / "evaluations" / f"evaluation_updates_{completed_updates:06d}.json"


def quarantine_incomplete(path):
    incomplete = Path(path).with_name(Path(path).name + ".incomplete")
    if not incomplete.exists():
        return None
    quarantine = incomplete.with_name(
        incomplete.name + f".orphaned.{int(time.time())}.{os.getpid()}"
    )
    os.replace(incomplete, quarantine)
    return str(quarantine.resolve())


def ensure_checkpoint_sidecars(
    path, symbols, parent_aux, metadata, completed_updates
):
    """Recover only derived sidecars after strict checkpoint revalidation."""
    path = Path(path).resolve()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    verification_path = path.with_suffix(path.suffix + ".verification.json")
    if checksum_path.is_file() and verification_path.is_file():
        return a0.verify_checkpoint_sidecars(path)
    digest = a0.file_sha256(path)
    if checksum_path.is_file():
        fields = checksum_path.read_text().strip().split()
        if len(fields) != 2 or fields[0] != digest or fields[1] != path.name:
            raise SystemExit("surviving checkpoint checksum sidecar is invalid")
    if verification_path.is_file():
        surviving = json.loads(verification_path.read_text())
        if (
            surviving.get("sha256") != digest
            or surviving.get("checkpoint") != str(path)
            or surviving.get("verification", {}).get("passed") is not True
        ):
            raise SystemExit("surviving checkpoint verification sidecar is invalid")
    checkpoint = a0.torch_load(path, mmap=True)
    expected_fields = {
        "schema",
        "model",
        "optimizer",
        "training_state",
        "dataloader_states",
        "rng_state",
        "metadata",
        "parent_checkpoint_sha256",
        "next_global_batch_sha256",
    }
    expected_state = {
        "completed_updates": completed_updates,
        "processed_student_tokens": completed_updates * a0.GLOBAL_BATCH_TOKENS,
    }
    expected_next = EXPECTED_MILESTONE_NEXT_SHA256[completed_updates]
    if (
        set(checkpoint) != expected_fields
        or checkpoint.get("schema") != a0.CHECKPOINT_SCHEMA
        or checkpoint.get("parent_checkpoint_sha256")
        != parent_aux["checkpoint_sha256"]
        or checkpoint.get("metadata") != metadata
        or checkpoint.get("training_state") != expected_state
        or checkpoint.get("next_global_batch_sha256") != expected_next
        or len(checkpoint.get("dataloader_states", [])) != a0.LEGACY_WORLD_SIZE
        or len(checkpoint.get("optimizer", {}).get("state", {})) != 3
        or optimizer_steps(checkpoint["optimizer"]) != [completed_updates] * 3
    ):
        raise SystemExit(f"orphaned checkpoint failed strict recovery audit: {path}")
    a0.optimizer_state_report(checkpoint["optimizer"], completed_updates)
    probe = a0.make_loaders_from_states(
        symbols, checkpoint["dataloader_states"], replay=True
    )
    restored_next = a0.next_update_hash(probe, symbols, replay=True)
    del probe
    if restored_next != expected_next:
        raise SystemExit("orphaned checkpoint serialized-loader replay mismatch")
    verification = {
        "checkpoint": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "verification": {
            "schema": checkpoint["schema"],
            "completed_updates": completed_updates,
            "processed_student_tokens": expected_state["processed_student_tokens"],
            "loader_states": len(checkpoint["dataloader_states"]),
            "next_global_batch_sha256": expected_next,
            "optimizer": a0.optimizer_state_report(
                checkpoint["optimizer"], completed_updates
            ),
            "strict_orphan_recovery": True,
            "passed": True,
        },
    }
    del checkpoint
    quarantine_incomplete(verification_path)
    quarantine_incomplete(checksum_path)
    durable_write_json(verification_path, verification)
    durable_write_text(
        checksum_path,
        f"{digest}  {path.name}\n",
    )
    return a0.verify_checkpoint_sidecars(path)


def validate_training_rows(rows, completed_updates, expected_hashes):
    expected_updates = list(range(START_UPDATE, completed_updates))
    if [row.get("update") for row in rows] != expected_updates:
        raise SystemExit("2A3 training rows are missing, duplicated, or out of order")
    for row in rows:
        update = row["update"]
        expected = {
            "completed_updates": update + 1,
            "processed_student_tokens": (update + 1) * a0.GLOBAL_BATCH_TOKENS,
            "global_schedule_step": a0.EXPECTED_PARENT_UPDATES + update,
            "lr": a0.get_lr(a0.EXPECTED_PARENT_UPDATES + update),
            "global_batch_sha256": expected_hashes[update],
        }
        mismatches = {
            key: (row.get(key), value)
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches:
            raise SystemExit(f"training row {update} mismatch: {mismatches}")
        if row.get("optimizer", {}).get("steps") != [update + 1] * 3:
            raise SystemExit(f"training row {update} optimizer-step mismatch")
        finite_names = (
            "loss",
            "grad_norm",
            "gate",
            "gate_coefficient",
            "query_norm",
            "rmsnorm_displacement",
            "routing_entropy",
            "forward_seconds",
            "backward_seconds",
            "wall_seconds",
            "peak_allocated_mb",
            "peak_reserved_mb",
        )
        if not all(
            row.get(name) is not None and math.isfinite(float(row[name]))
            for name in finite_names
        ):
            raise SystemExit(f"training row {update} has a non-finite scalar")
        weights = row.get("routing_weights", {})
        if (
            set(weights) != {f"v{depth}" for depth in a0.SOURCE_DEPTHS}
            or not all(0.0 <= float(value) <= 1.0 for value in weights.values())
            or not math.isclose(
                sum(float(value) for value in weights.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not 0.0 <= float(row["routing_entropy"]) <= math.log(4)
            or row.get("optimizer", {}).get("nonfinite_tensors")
            or row.get("teacher_eval_no_grad") is not True
            or row.get("trainable_parameters_finite") is not True
        ):
            raise SystemExit(f"training row {update} routing/runtime contract mismatch")
        gradients = row.get("gradients", {})
        if (
            gradients.get("base_tensors_with_grad")
            or gradients.get("teacher_tensors_with_grad")
            or not all(
                gradients.get(name, {}).get("present")
                and gradients[name].get("finite")
                for name in ("gate", "query", "rmsnorm")
            )
        ):
            raise SystemExit(f"training row {update} gradient contract mismatch")
    return {"updates": expected_updates, "passed": True}


def prepare_fresh_run(run_dir, config, metadata):
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite run directory: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "evaluations").mkdir()
    durable_write_json(run_dir / "config.json", config)
    durable_write_json(run_dir / "metadata.json", metadata)
    durable_write_text(run_dir / "metrics.jsonl", "")
    return run_dir


def verify_run_identity(run_dir, config, metadata):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SystemExit("resume run directory does not exist")
    if json.loads((run_dir / "config.json").read_text()) != config:
        raise SystemExit("resume config mismatch")
    if json.loads((run_dir / "metadata.json").read_text()) != metadata:
        raise SystemExit("resume metadata/source mismatch")
    return run_dir


def load_new_checkpoint(
    path,
    student,
    optimizer,
    loaders,
    symbols,
    parent_aux,
    metadata,
):
    preview = a0.torch_load(path, mmap=True)
    preview_completed = preview.get("training_state", {}).get("completed_updates")
    del preview
    if preview_completed not in MILESTONES:
        raise SystemExit(
            f"2A3 resume checkpoint is not a milestone: {preview_completed}"
        )
    ensure_checkpoint_sidecars(
        path, symbols, parent_aux, metadata, preview_completed
    )
    state, audit = a0.load_exp2_resume(
        path,
        student,
        optimizer,
        loaders,
        symbols,
        True,
        parent_aux,
        metadata,
    )
    completed = state["completed_updates"]
    if completed not in MILESTONES:
        raise SystemExit(f"2A3 resume checkpoint is not a milestone: {completed}")
    expected_path = checkpoint_path(Path(path).parents[1], completed).resolve()
    if Path(path).resolve() != expected_path:
        raise SystemExit(f"resume checkpoint filename/state mismatch: {path}")
    expected_next = EXPECTED_MILESTONE_NEXT_SHA256[completed]
    if audit["next_global_batch_sha256"] != expected_next:
        raise SystemExit("resume checkpoint replay-oracle mismatch")
    assert_optimizer_state(optimizer, completed)
    return state, audit


def save_or_verify_checkpoint(
    run_dir,
    completed_updates,
    student,
    optimizer,
    loaders,
    symbols,
    parent_aux,
    metadata,
):
    path = checkpoint_path(run_dir, completed_updates)
    state = {
        "completed_updates": completed_updates,
        "processed_student_tokens": completed_updates * a0.GLOBAL_BATCH_TOKENS,
    }
    if path.exists():
        integrity = ensure_checkpoint_sidecars(
            path, symbols, parent_aux, metadata, completed_updates
        )
        checkpoint = a0.torch_load(path, mmap=True)
        if (
            checkpoint.get("training_state") != state
            or checkpoint.get("metadata") != metadata
            or checkpoint.get("next_global_batch_sha256")
            != EXPECTED_MILESTONE_NEXT_SHA256[completed_updates]
            or not a0.nested_equal(checkpoint.get("model"), student.state_dict())
            or not a0.nested_equal(checkpoint.get("optimizer"), optimizer.state_dict())
            or not a0.nested_equal(
                checkpoint.get("dataloader_states"), a0.snapshot_loaders(loaders)
            )
            or not a0.nested_equal(checkpoint.get("rng_state"), a0.capture_rng_state())
        ):
            raise SystemExit(f"existing milestone checkpoint does not match live state: {path}")
        del checkpoint
        return integrity
    quarantine_incomplete(path)
    sidecar = a0.save_exp2a0_checkpoint(
        path,
        student,
        optimizer,
        loaders,
        symbols,
        True,
        state,
        parent_aux,
        metadata,
    )
    if sidecar["verification"]["next_global_batch_sha256"] != EXPECTED_MILESTONE_NEXT_SHA256[completed_updates]:
        raise SystemExit("published milestone next-batch hash mismatch")
    # Rewrite the derived sidecars durably after the parent helper publishes them.
    durable_write_json(
        path.with_suffix(path.suffix + ".verification.json"), sidecar
    )
    durable_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{sidecar['sha256']}  {path.name}\n",
    )
    return a0.verify_checkpoint_sidecars(path)


def fresh_restart_at_milestone(
    path,
    old_student,
    old_teacher,
    old_optimizer,
    old_loaders,
    symbols,
    parent_checkpoint,
    device,
    metadata,
):
    old_boundary = snapshot_training_boundary(
        old_student,
        old_teacher,
        old_optimizer,
        old_loaders,
    )
    symbols2, teacher2, student2, optimizer2, loaders2, parent_aux2 = make_runtime(
        parent_checkpoint, device
    )
    state, audit = load_new_checkpoint(
        path,
        student2,
        optimizer2,
        loaders2,
        symbols2,
        parent_aux2,
        metadata,
    )
    checks = {
        "student_base": old_boundary["student_base_sha256"]
        == model_state_sha256(student2, include_topdown=False),
        "student_topdown": old_boundary["student_topdown_sha256"]
        == model_state_sha256(student2, include_topdown=True),
        "teacher": old_boundary["teacher_sha256"]
        == model_state_sha256(teacher2, include_topdown=False),
        "optimizer": a0.nested_equal(
            old_boundary["optimizer"], optimizer2.state_dict()
        ),
        "loaders": a0.nested_equal(
            old_boundary["loaders"], a0.snapshot_loaders(loaders2)
        ),
        "rng": a0.nested_equal(old_boundary["rng"], a0.capture_rng_state()),
        "load_audit": audit,
    }
    checks["passed"] = all(
        checks[key]
        for key in (
            "student_base",
            "student_topdown",
            "teacher",
            "optimizer",
            "loaders",
            "rng",
        )
    ) and audit["passed"]
    if not checks["passed"]:
        raise SystemExit(f"fresh milestone restart mismatch: {checks}")
    return symbols2, teacher2, student2, optimizer2, loaders2, parent_aux2, state, checks


def validate_restart_record(record):
    required = {
        "student_base",
        "student_topdown",
        "teacher",
        "optimizer",
        "loaders",
        "rng",
        "load_audit",
        "passed",
    }
    load_audit = record.get("load_audit", {})
    completed = load_audit.get("completed_updates")
    return (
        set(record) == required
        and all(record[key] is True for key in required - {"load_audit", "passed"})
        and completed in MILESTONES[:-1]
        and validate_resume_load_audit(
            load_audit, completed, EXPECTED_MILESTONE_NEXT_SHA256[completed]
        )
        and record["passed"] is True
    )


def validate_isolation_record(isolation):
    required = {
        "student_base",
        "student_topdown",
        "teacher",
        "optimizer",
        "loaders",
        "rng",
        "student_mode",
        "teacher_mode",
        "router_instrumentation",
        "router_source_mask",
        "passed",
    }
    return set(isolation) == required and all(
        value is True for value in isolation.values()
    )


def validate_resume_load_audit(audit, completed_updates=None, expected_next=None):
    required = {
        "integrity",
        "completed_updates",
        "processed_student_tokens",
        "next_global_batch_sha256",
        "model_strict_reload",
        "optimizer_exact_reload",
        "loader_exact_reload",
        "rng_exact_reload",
        "passed",
    }
    if set(audit) != required or not all(
        audit.get(key) is True
        for key in (
            "model_strict_reload",
            "optimizer_exact_reload",
            "loader_exact_reload",
            "rng_exact_reload",
            "passed",
        )
    ):
        return False
    completed = audit.get("completed_updates")
    return (
        isinstance(completed, int)
        and (completed_updates is None or completed == completed_updates)
        and audit.get("processed_student_tokens")
        == completed * a0.GLOBAL_BATCH_TOKENS
        and (
            expected_next is None
            or audit.get("next_global_batch_sha256") == expected_next
        )
        and audit.get("integrity", {}).get("sha256") is not None
    )


def validate_source_restore_audit(audit):
    required = {
        "integrity",
        "model_exact_reload",
        "optimizer_exact_reload",
        "loader_exact_reload",
        "rng_exact_reload",
        "next_global_batch_sha256",
        "optimizer",
        "passed",
    }
    return (
        set(audit) == required
        and all(
            audit.get(key) is True
            for key in (
                "model_exact_reload",
                "optimizer_exact_reload",
                "loader_exact_reload",
                "rng_exact_reload",
                "passed",
            )
        )
        and audit.get("integrity", {}).get("sha256")
        == EXPECTED_CONTINUATION_SHA256
        and audit.get("next_global_batch_sha256")
        == EXPECTED_CONTINUATION_NEXT_SHA256
        and audit.get("optimizer", {}).get("steps") == [START_UPDATE] * 3
        and audit.get("optimizer", {}).get("state_entries") == 3
        and not audit.get("optimizer", {}).get("nonfinite_tensors")
    )


def validate_causality_record(record):
    required = {"teacher_memory", "end_to_end", "state_isolation", "passed"}
    memory = record.get("teacher_memory", {})
    end_to_end = record.get("end_to_end", {})
    isolation = record.get("state_isolation", {})
    return (
        set(record) == required
        and record.get("passed") is True
        and memory.get("passed") is True
        and memory.get("position_zero_memory_exactly_zero") is True
        and set(memory.get("per_source_memory_bit_exact", {}))
        == {f"v{depth}" for depth in a0.SOURCE_DEPTHS}
        and all(memory["per_source_memory_bit_exact"].values())
        and end_to_end.get("passed") is True
        and end_to_end.get("prefix_logits_bit_exact") is True
        and end_to_end.get("maximum_absolute_difference") == 0.0
        and validate_isolation_record(isolation)
    )


def run_preflight(args, device):
    """Read-only validation of the exact continuation before optimizer access."""
    require_clean_tree()
    config = validate_config(json.loads(CONFIG_PATH.read_text()))
    dataset = a0.dataset_manifest_report(verify_shards=True)
    hellaswag_dataset = hellaswag_dataset_report(download=True)
    source_metadata, source_summary = load_source_metadata()
    symbols, teacher, student, optimizer, loaders, parent_aux = make_runtime(
        args.parent_checkpoint, device
    )
    state, restore_audit = restore_source_checkpoint(
        args.continuation_checkpoint,
        student,
        optimizer,
        loaders,
        symbols,
        parent_aux,
        source_metadata,
    )
    contract = assert_runtime_contract(student, teacher, optimizer, START_UPDATE)
    frozen = assert_frozen_hashes(
        student, teacher, EXPECTED_FROZEN_BASE_SHA256, EXPECTED_FROZEN_BASE_SHA256
    )
    source_checkpoint = a0.torch_load(args.continuation_checkpoint, mmap=True)
    expected_hashes = derive_expected_hashes(
        symbols, source_checkpoint["dataloader_states"]
    )
    del source_checkpoint
    before = snapshot_training_boundary(student, teacher, optimizer, loaders)
    memory_causality = a0.production_causality_test(teacher, symbols, device)
    if not memory_causality["passed"]:
        raise SystemExit(f"teacher-memory causality failed: {memory_causality}")
    end_to_end_causality = end_to_end_future_invariance_test(
        student, teacher, symbols, device
    )
    isolation = assert_training_boundary_unchanged(
        before, student, teacher, optimizer, loaders
    )
    source_evaluation = load_source_evaluation()
    hellaswag_before = snapshot_training_boundary(
        student, teacher, optimizer, loaders
    )
    hellaswag_isolation = hellaswag_candidate_isolation_test(
        student, teacher, symbols, device
    )
    hellaswag_state_isolation = assert_training_boundary_unchanged(
        hellaswag_before, student, teacher, optimizer, loaders
    )
    report = {
        "kind": "Experiment 2A3 optimizer-free continuation preflight",
        "git_commit": a0.git_output("rev-parse", "HEAD"),
        "git_branch": a0.git_output("branch", "--show-current"),
        "config": config,
        "dataset": dataset,
        "hellaswag_dataset": hellaswag_dataset,
        "hellaswag_candidate_isolation": hellaswag_isolation,
        "hellaswag_state_isolation": hellaswag_state_isolation,
        "source_summary_sha256": a0.file_sha256(source_summary_path()),
        "source_checkpoint_sha256": EXPECTED_CONTINUATION_SHA256,
        "source_training_state": state,
        "source_restore": restore_audit,
        "runtime_contract": contract,
        "frozen_state": frozen,
        "teacher_memory_causality": memory_causality,
        "end_to_end_causality": end_to_end_causality,
        "causality_state_isolation": isolation,
        "canonical_update191_evaluation": source_evaluation,
        "canonical_update191_evaluation_rerun": False,
        "expected_global_batch_sha256": {
            str(update): expected_hashes[update]
            for update in range(START_UPDATE, TARGET_UPDATE + 1)
        },
        "first_new_schedule_step": a0.EXPECTED_PARENT_UPDATES + START_UPDATE,
        "first_new_lr": a0.get_lr(a0.EXPECTED_PARENT_UPDATES + START_UPDATE),
        "optimizer_steps_executed": 0,
        "passed": True,
    }
    durable_write_json(args.out, report)
    return report


def validate_or_write_dataset_artifact(run_dir, dataset):
    path = Path(run_dir) / "dataset_verification.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != dataset or not stored.get("shards_verified"):
            raise SystemExit("stored/live dataset verification mismatch")
    else:
        durable_write_json(path, dataset)


def validate_or_write_replay_oracle(run_dir, expected_hashes):
    payload = {
        "start_update": START_UPDATE,
        "target_update": TARGET_UPDATE,
        "training_global_batch_sha256": {
            str(update): expected_hashes[update]
            for update in range(START_UPDATE, TARGET_UPDATE)
        },
        "next_after_target_sha256": expected_hashes[TARGET_UPDATE],
    }
    path = Path(run_dir) / "replay_oracle.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise SystemExit("stored/live replay oracle mismatch")
    else:
        durable_write_json(path, payload)
    return payload


def read_metrics(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def validate_existing_evaluation(
    path, completed_updates, checkpoint_sha256, metadata
):
    if not Path(path).is_file():
        raise SystemExit(f"required milestone evaluation is missing: {path}")
    artifact = json.loads(Path(path).read_text())
    return validate_evaluation_artifact(
        artifact,
        completed_updates,
        checkpoint_sha256,
        hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
        completed_updates == TARGET_UPDATE,
        metadata["source_file_sha256"],
    )


def validate_prior_chain(run_dir, completed, metadata, symbols, parent_aux):
    run_dir = Path(run_dir)
    load_source_evaluation()
    source_audit_path = run_dir / "source_continuation_audit.json"
    if not source_audit_path.is_file():
        raise SystemExit("resume source-continuation audit is missing")
    source_audit = json.loads(source_audit_path.read_text())
    if (
        set(source_audit) != {"integrity", "restore", "passed"}
        or source_audit.get("passed") is not True
        or source_audit.get("integrity", {}).get("sha256")
        != EXPECTED_CONTINUATION_SHA256
        or not validate_source_restore_audit(source_audit.get("restore", {}))
    ):
        raise SystemExit("resume source-continuation audit is invalid")
    source_causality_path = run_dir / f"causality_updates_{START_UPDATE:06d}.json"
    if not source_causality_path.is_file() or not validate_causality_record(
        json.loads(source_causality_path.read_text())
    ):
        raise SystemExit("resume update-191 causality audit is missing or invalid")
    checkpoint_records = {}
    for milestone in (value for value in MILESTONES if value < completed):
        path = checkpoint_path(run_dir, milestone)
        if not path.is_file():
            raise SystemExit(f"resume prior checkpoint {milestone} is missing")
        integrity = ensure_checkpoint_sidecars(
            path, symbols, parent_aux, metadata, milestone
        )
        checkpoint_records[str(milestone)] = integrity
        validate_existing_evaluation(
            evaluation_path(run_dir, milestone),
            milestone,
            integrity["sha256"],
            metadata,
        )
        causality_path = run_dir / f"causality_updates_{milestone:06d}.json"
        restart_path = run_dir / f"restart_audit_updates_{milestone:06d}.json"
        if (
            not causality_path.is_file()
            or not validate_causality_record(json.loads(causality_path.read_text()))
            or not restart_path.is_file()
            or not validate_restart_record(json.loads(restart_path.read_text()))
        ):
            raise SystemExit(f"resume prior milestone {milestone} chain is invalid")
    return checkpoint_records


def final_summary(
    run_dir, metadata, checkpoint_records, restart_records, causality, hellaswag_result
):
    run_dir = Path(run_dir)
    rows = read_metrics(run_dir / "metrics.jsonl")
    replay_oracle = json.loads((run_dir / "replay_oracle.json").read_text())
    expected_hashes = {
        int(update): digest
        for update, digest in replay_oracle["training_global_batch_sha256"].items()
    }
    row_audit = validate_training_rows(rows, TARGET_UPDATE, expected_hashes)
    source_evaluation = load_source_evaluation()
    evaluations = {START_UPDATE: source_evaluation}
    for milestone in MILESTONES:
        checkpoint_integrity = a0.verify_checkpoint_sidecars(
            checkpoint_path(run_dir, milestone)
        )
        if checkpoint_integrity["sha256"] != checkpoint_records[str(milestone)]["sha256"]:
            raise SystemExit(f"checkpoint manifest mismatch at update {milestone}")
        evaluations[milestone] = validate_existing_evaluation(
            evaluation_path(run_dir, milestone),
            milestone,
            checkpoint_integrity["sha256"],
            metadata,
        )
    trajectory_rows = []
    for milestone in (START_UPDATE, *MILESTONES):
        evaluation = evaluations[milestone]
        trajectory_rows.append(
            {
                "completed_updates": milestone,
                "tokens": milestone * a0.GLOBAL_BATCH_TOKENS,
                "real_feedback_loss": evaluation["losses"]["real_feedback"],
                "shuffled_feedback_loss": evaluation["losses"]["shuffled_feedback"],
                "real_minus_shuffled": evaluation["requested_real_minus_shuffled"]["mean"],
                "shuffled_minus_real": evaluation["sequence_specific_recovery"],
                "sequence_specific_recovery": evaluation[
                    "sequence_specific_recovery"
                ],
                "total_recovery_fraction": evaluation["total_recovery_fraction"],
                "sequence_specific_recovery_fraction": evaluation[
                    "sequence_specific_recovery_fraction"
                ],
                "sequence_specific_share_of_total_recovery": evaluation[
                    "sequence_specific_share_of_total_recovery"
                ],
                "paired": evaluation["paired_shuffled_minus_real"],
                "routing": evaluation["routing"],
            }
        )
    classification = classify_trajectory(trajectory_rows)
    final = evaluations[TARGET_UPDATE]
    _, source_summary = load_source_metadata()
    historical = copy.deepcopy(source_summary.get("trajectory", []))
    if [row.get("completed_updates") for row in historical] != [10, 20, 29, 48, 96, 191]:
        raise SystemExit("audited Experiment 2A2 historical trajectory is invalid")
    for source_row, current_row in zip(historical[-1:], trajectory_rows[:1]):
        exact_fields = (
            "real_feedback_loss",
            "shuffled_feedback_loss",
            "sequence_specific_recovery",
            "total_recovery_fraction",
            "sequence_specific_recovery_fraction",
        )
        if any(source_row.get(key) != current_row.get(key) for key in exact_fields):
            raise SystemExit("update-191 continuity evaluation disagrees with audited history")
    for row in historical:
        total_recovery = row["total_recovery_fraction"] * (PINNED_MASKED - PINNED_FULL)
        row["sequence_specific_share_of_total_recovery"] = (
            row["sequence_specific_recovery"] / total_recovery
        )
    full_trajectory = historical + trajectory_rows[1:]
    q100, q150, q200, q250 = classification["values"]
    hellaswag_accuracy = hellaswag_result["accuracy"]
    continue_to_500m = (
        q250 > 0
        and final["paired_shuffled_minus_real"]["positive_count"] >= 15
        and q250 >= q200
        and final["total_recovery"] > 0
        and hellaswag_accuracy["real_feedback"]
        >= hellaswag_accuracy["masked_l1_no_feedback"] - 0.01
        and hellaswag_accuracy["real_feedback"]
        >= hellaswag_accuracy["zero_feedback"] - 0.01
    )
    begin_self_recurrent = (
        q250 > 0
        and q250 >= q200
        and final["paired_shuffled_minus_real"]["positive_count"] >= 15
        and final["sequence_specific_recovery_fraction"] >= 0.10
        and all(
            item["delta_vs_real_feedback"] > 0
            for item in final["source_ablation"].values()
        )
        and hellaswag_accuracy["real_feedback"]
        >= hellaswag_accuracy["masked_l1_no_feedback"]
    )
    interval_training = {}
    for lower, upper in zip((START_UPDATE, *MILESTONES[:-1]), MILESTONES):
        interval_rows = [row for row in rows if lower <= row["update"] < upper]
        if len(interval_rows) != upper - lower:
            raise SystemExit(f"training interval {lower}:{upper} is incomplete")
        losses = [row["loss"] for row in interval_rows]
        interval_training[f"updates_{lower + 1}_through_{upper}"] = {
            "updates": len(losses),
            "mean_loss": statistics.fmean(losses),
            "last_loss": losses[-1],
            "minimum_loss": min(losses),
            "maximum_loss": max(losses),
        }
    summary = {
        "experiment": "Experiment 2A3",
        "resume": {
            "starting_checkpoint": metadata["continuation_checkpoint"],
            "starting_sha256": metadata["continuation_checkpoint_sha256"],
            "starting_updates": START_UPDATE,
            "starting_tokens": START_UPDATE * a0.GLOBAL_BATCH_TOKENS,
            "last_consumed_schedule_step": (
                a0.EXPECTED_PARENT_UPDATES + START_UPDATE - 1
            ),
            "next_schedule_step": a0.EXPECTED_PARENT_UPDATES + START_UPDATE,
            "learning_rate_at_restart": a0.get_lr(
                a0.EXPECTED_PARENT_UPDATES + START_UPDATE
            ),
            "verification": json.loads(
                (run_dir / "source_continuation_audit.json").read_text()
            ),
        },
        "completed_updates": TARGET_UPDATE,
        "processed_student_tokens": TARGET_UPDATE * a0.GLOBAL_BATCH_TOKENS,
        "additional_updates": TARGET_UPDATE - START_UPDATE,
        "additional_student_tokens": (TARGET_UPDATE - START_UPDATE)
        * a0.GLOBAL_BATCH_TOKENS,
        "trajectory": full_trajectory,
        "continuation_trajectory": trajectory_rows,
        "trajectory_classification": classification,
        "continue_to_500m": {
            "answer": "YES" if continue_to_500m else "NO",
            "rule": CONTINUE_TO_500M_RULE,
            "passed": continue_to_500m,
        },
        "begin_self_generated_recurrence": {
            "answer": "YES" if begin_self_recurrent else "NO",
            "rule": BEGIN_SELF_RECURRENT_RULE,
            "passed": begin_self_recurrent,
        },
        "final_controls": final["losses"],
        "final_damage": final["damage"],
        "final_total_recovery": final["total_recovery"],
        "final_total_recovery_fraction": final["total_recovery_fraction"],
        "final_sequence_specific_recovery": final["sequence_specific_recovery"],
        "final_sequence_specific_recovery_fraction": final[
            "sequence_specific_recovery_fraction"
        ],
        "final_sequence_specific_share_of_total_recovery": final[
            "sequence_specific_share_of_total_recovery"
        ],
        "final_router": final["routing"],
        "final_source_ablation": final["source_ablation"],
        "source_ablation_comparison_100m_to_250m": {
            "100m": source_evaluation["source_ablation"],
            "250m": final["source_ablation"],
            "ranking_100m": sorted(
                source_evaluation["source_ablation"],
                key=lambda name: source_evaluation["source_ablation"][name][
                    "delta_vs_real_feedback"
                ],
                reverse=True,
            ),
            "ranking_250m": sorted(
                final["source_ablation"],
                key=lambda name: final["source_ablation"][name][
                    "delta_vs_real_feedback"
                ],
                reverse=True,
            ),
            "note": "renormalized leave-one-source-out effects; not additive contributions",
        },
        "hellaswag": hellaswag_result,
        "checkpoint_records": checkpoint_records,
        "restart_records": restart_records,
        "causality": causality,
        "training_row_audit": row_audit,
        "training_intervals": interval_training,
        "resources": {
            "peak_allocated_mb": max(row["peak_allocated_mb"] for row in rows),
            "peak_reserved_mb": max(row["peak_reserved_mb"] for row in rows),
            "training_update_wall_seconds": sum(row["wall_seconds"] for row in rows),
            "training_forward_seconds": sum(row["forward_seconds"] for row in rows),
            "training_backward_seconds": sum(row["backward_seconds"] for row in rows),
            "evaluation_wall_seconds": sum(
                evaluations[milestone]["elapsed_seconds"] for milestone in MILESTONES
            ),
            "hellaswag_wall_seconds": hellaswag_result["elapsed_seconds"],
        },
        "terminal_line": "# EXPERIMENT 2A3 250M COMPLETE",
        "no_follow_on_launched": True,
        "passed": True,
    }
    durable_write_json(run_dir / "result_summary.json", summary)
    return summary


def run_continuation(args, device):
    invocation_started = time.perf_counter()
    if not args.allow_optimizer_steps:
        raise SystemExit(
            "optimizer steps are locked; explicit --allow-optimizer-steps is required"
        )
    require_clean_tree()
    config = validate_config(json.loads(CONFIG_PATH.read_text()))
    dataset = a0.dataset_manifest_report(verify_shards=True)
    hellaswag_dataset_report(download=False)
    source_metadata, _ = load_source_metadata()
    symbols, teacher, student, optimizer, loaders, parent_aux = make_runtime(
        args.parent_checkpoint, device
    )
    metadata = continuation_metadata(
        config, parent_aux, args.continuation_checkpoint
    )
    source_integrity, source_checkpoint = validate_source_checkpoint(
        args.continuation_checkpoint
    )
    expected_hashes = derive_expected_hashes(
        symbols, source_checkpoint["dataloader_states"]
    )
    del source_checkpoint

    run_dir = Path(args.run_dir)
    checkpoint_records = {}
    restart_records = {}
    causality_records = {}
    if args.resume:
        verify_run_identity(run_dir, config, metadata)
        resume_path = Path(args.resume).resolve()
        resume_payload = a0.torch_load(resume_path, mmap=True)
        resume_completed = resume_payload.get("training_state", {}).get(
            "completed_updates"
        )
        del resume_payload
        if resume_completed not in MILESTONES or resume_path != checkpoint_path(
            run_dir, resume_completed
        ).resolve():
            raise SystemExit("resume checkpoint is not bound to this run directory")
        checkpoint_records.update(
            validate_prior_chain(
                run_dir, resume_completed, metadata, symbols, parent_aux
            )
        )
        validate_or_write_dataset_artifact(run_dir, dataset)
        validate_or_write_replay_oracle(run_dir, expected_hashes)
        state, resume_audit = load_new_checkpoint(
            args.resume,
            student,
            optimizer,
            loaders,
            symbols,
            parent_aux,
            metadata,
        )
        completed = state["completed_updates"]
        reconciliation = reconcile_metrics(
            run_dir / "metrics.jsonl", completed, start_update=START_UPDATE
        )
        durable_write_json(
            run_dir / f"resume_audit_updates_{completed:06d}.json",
            {"load": resume_audit, "metrics": reconciliation, "passed": True},
        )
        if completed in MILESTONES[:-1]:
            assert_frozen_hashes(
                student,
                teacher,
                EXPECTED_FROZEN_BASE_SHA256,
                EXPECTED_FROZEN_BASE_SHA256,
            )
            resume_restart = {
                "student_base": resume_audit["model_strict_reload"],
                "student_topdown": resume_audit["model_strict_reload"],
                "teacher": True,
                "optimizer": resume_audit["optimizer_exact_reload"],
                "loaders": resume_audit["loader_exact_reload"],
                "rng": resume_audit["rng_exact_reload"],
                "load_audit": resume_audit,
                "passed": resume_audit["passed"],
            }
            if not validate_restart_record(resume_restart):
                raise SystemExit("resume boundary did not satisfy restart contract")
            restart_records[str(completed)] = resume_restart
            durable_write_json(
                run_dir / f"restart_audit_updates_{completed:06d}.json",
                resume_restart,
            )
        for milestone in MILESTONES:
            path = checkpoint_path(run_dir, milestone)
            if path.is_file():
                checkpoint_records[str(milestone)] = a0.verify_checkpoint_sidecars(path)
        current_integrity = a0.verify_checkpoint_sidecars(args.resume)
        bound_milestone_evaluation(
            evaluation_path(run_dir, completed),
            student,
            teacher,
            optimizer,
            loaders,
            symbols,
            device,
            completed,
            current_integrity["sha256"],
            metadata,
        )
    else:
        run_dir = prepare_fresh_run(run_dir, config, metadata)
        validate_or_write_dataset_artifact(run_dir, dataset)
        validate_or_write_replay_oracle(run_dir, expected_hashes)
        state, source_restore = restore_source_checkpoint(
            args.continuation_checkpoint,
            student,
            optimizer,
            loaders,
            symbols,
            parent_aux,
            source_metadata,
        )
        completed = START_UPDATE
        durable_write_json(
            run_dir / "source_continuation_audit.json",
            {
                "integrity": source_integrity,
                "restore": source_restore,
                "passed": True,
            },
        )
        source_evaluation = load_source_evaluation()
        durable_write_json(
            run_dir / "source_evaluation_reference.json",
            {
                "path": str(source_evaluation_path().resolve()),
                "sha256": a0.file_sha256(source_evaluation_path()),
                "checkpoint_sha256": source_evaluation["checkpoint_sha256"],
                "completed_updates": START_UPDATE,
                "rerun": False,
                "passed": True,
            },
        )

    assert_runtime_contract(student, teacher, optimizer, completed)
    expected_teacher_hash = model_state_sha256(teacher, include_topdown=False)
    assert_frozen_hashes(
        student,
        teacher,
        EXPECTED_FROZEN_BASE_SHA256,
        EXPECTED_FROZEN_BASE_SHA256,
    )
    next_hash = a0.next_update_hash(loaders, symbols, True)
    if next_hash != expected_hashes[completed]:
        raise SystemExit(f"continuation boundary batch mismatch at update {completed}")
    boundary_before = snapshot_training_boundary(
        student, teacher, optimizer, loaders
    )
    boundary_memory_causality = a0.production_causality_test(
        teacher, symbols, device
    )
    boundary_end_to_end = end_to_end_future_invariance_test(
        student, teacher, symbols, device
    )
    boundary_isolation = assert_training_boundary_unchanged(
        boundary_before, student, teacher, optimizer, loaders
    )
    if not boundary_memory_causality["passed"] or not boundary_end_to_end["passed"]:
        raise SystemExit(f"causality failed before continuation at update {completed}")
    causality_records[str(completed)] = {
        "teacher_memory": boundary_memory_causality,
        "end_to_end": boundary_end_to_end,
        "state_isolation": boundary_isolation,
        "passed": True,
    }
    durable_write_json(
        run_dir / f"causality_updates_{completed:06d}.json",
        causality_records[str(completed)],
    )

    milestone_sequence = [value for value in MILESTONES if value >= completed]
    for milestone in milestone_sequence:
        if completed < milestone:
            for update in range(completed, milestone):
                train_one_update(
                    teacher,
                    student,
                    optimizer,
                    loaders,
                    symbols,
                    update,
                    expected_hashes[update],
                    run_dir / "metrics.jsonl",
                )
                assert_frozen_hashes(
                    student,
                    teacher,
                    EXPECTED_FROZEN_BASE_SHA256,
                    expected_teacher_hash,
                )
            completed = milestone
        assert_runtime_contract(student, teacher, optimizer, completed)
        checkpoint_record = save_or_verify_checkpoint(
            run_dir,
            completed,
            student,
            optimizer,
            loaders,
            symbols,
            parent_aux,
            metadata,
        )
        checkpoint_records[str(completed)] = checkpoint_record
        checkpoint_sha256 = checkpoint_record["sha256"]
        causality_before = snapshot_training_boundary(
            student, teacher, optimizer, loaders
        )
        memory_causality = a0.production_causality_test(teacher, symbols, device)
        end_to_end = end_to_end_future_invariance_test(
            student, teacher, symbols, device
        )
        causality_isolation = assert_training_boundary_unchanged(
            causality_before, student, teacher, optimizer, loaders
        )
        if not memory_causality["passed"] or not end_to_end["passed"]:
            raise SystemExit(f"causality failed at update {completed}")
        causality_records[str(completed)] = {
            "teacher_memory": memory_causality,
            "end_to_end": end_to_end,
            "state_isolation": causality_isolation,
            "passed": True,
        }
        durable_write_json(
            run_dir / f"causality_updates_{completed:06d}.json",
            causality_records[str(completed)],
        )
        bound_milestone_evaluation(
            evaluation_path(run_dir, completed),
            student,
            teacher,
            optimizer,
            loaders,
            symbols,
            device,
            completed,
            checkpoint_sha256,
            metadata,
        )
        if completed != TARGET_UPDATE:
            (
                symbols_new,
                teacher_new,
                student_new,
                optimizer_new,
                loaders_new,
                parent_aux_new,
                state,
                restart_audit,
            ) = fresh_restart_at_milestone(
                checkpoint_path(run_dir, completed),
                student,
                teacher,
                optimizer,
                loaders,
                symbols,
                args.parent_checkpoint,
                device,
                metadata,
            )
            restart_records[str(completed)] = restart_audit
            durable_write_json(
                run_dir / f"restart_audit_updates_{completed:06d}.json",
                restart_audit,
            )
            del student, teacher, optimizer, loaders, symbols, parent_aux
            symbols, teacher, student, optimizer, loaders, parent_aux = (
                symbols_new,
                teacher_new,
                student_new,
                optimizer_new,
                loaders_new,
                parent_aux_new,
            )
            gc.collect()
            torch.cuda.empty_cache()
            completed = state["completed_updates"]

    if completed != TARGET_UPDATE:
        raise SystemExit(f"terminal update mismatch: {completed} != {TARGET_UPDATE}")
    final_checkpoint_sha256 = checkpoint_records[str(TARGET_UPDATE)]["sha256"]
    hellaswag_result = bound_hellaswag_evaluation(
        hellaswag_path(run_dir),
        student,
        teacher,
        optimizer,
        loaders,
        symbols,
        device,
        final_checkpoint_sha256,
        metadata,
    )
    rows = read_metrics(run_dir / "metrics.jsonl")
    validate_training_rows(rows, TARGET_UPDATE, expected_hashes)
    # Preserve restart/causality artifacts from any earlier invocation.
    for milestone in MILESTONES[:-1]:
        restart_path = run_dir / f"restart_audit_updates_{milestone:06d}.json"
        if restart_path.is_file() and str(milestone) not in restart_records:
            restart_records[str(milestone)] = json.loads(restart_path.read_text())
    for milestone in (START_UPDATE, *MILESTONES):
        causality_path = run_dir / f"causality_updates_{milestone:06d}.json"
        if causality_path.is_file() and str(milestone) not in causality_records:
            causality_records[str(milestone)] = json.loads(causality_path.read_text())
    if set(restart_records) != {"286", "381"} or not all(
        validate_restart_record(record) for record in restart_records.values()
    ):
        raise SystemExit(f"milestone restart audit is incomplete: {restart_records}")
    if set(causality_records) != {"191", "286", "381", "477"} or not all(
        validate_causality_record(record) for record in causality_records.values()
    ):
        raise SystemExit(f"milestone causality audit is incomplete: {causality_records}")
    summary = final_summary(
        run_dir,
        metadata,
        checkpoint_records,
        restart_records,
        causality_records,
        hellaswag_result,
    )
    summary["resources"]["completion_invocation_wall_seconds"] = (
        time.perf_counter() - invocation_started
    )
    summary["resources"]["peak_allocated_mb_all_evaluations_and_training"] = max(
        [summary["resources"]["peak_allocated_mb"]]
        + [
            json.loads(evaluation_path(run_dir, milestone).read_text()).get(
                "peak_allocated_mb", 0.0
            )
            for milestone in MILESTONES
        ]
        + [hellaswag_result.get("peak_allocated_mb", 0.0)]
    )
    summary["resources"]["peak_reserved_mb_all_evaluations_and_training"] = max(
        [summary["resources"]["peak_reserved_mb"]]
        + [
            json.loads(evaluation_path(run_dir, milestone).read_text()).get(
                "peak_reserved_mb", 0.0
            )
            for milestone in MILESTONES
        ]
        + [hellaswag_result.get("peak_reserved_mb", 0.0)]
    )
    durable_write_json(run_dir / "result_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--parent-checkpoint", required=True)
    preflight.add_argument("--continuation-checkpoint", required=True)
    preflight.add_argument("--out", required=True)
    continuation = subparsers.add_parser("continue-250m")
    continuation.add_argument("--parent-checkpoint", required=True)
    continuation.add_argument("--continuation-checkpoint", required=True)
    continuation.add_argument("--run-dir", required=True)
    continuation.add_argument("--resume")
    continuation.add_argument("--allow-optimizer-steps", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    require_pinned_runtime()
    device = a0.require_cuda()
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(a0.SEED)
    np.random.seed(a0.SEED)
    torch.manual_seed(a0.SEED)
    torch.cuda.manual_seed(a0.SEED)
    if args.command == "preflight":
        report = run_preflight(args, device)
    else:
        report = run_continuation(args, device)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
