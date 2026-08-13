#!/usr/bin/env python3
"""Exact single-A100 continuation of Experiment 2A0 from 5M to 25M tokens."""

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


CONFIG_PATH = REPO_ROOT / "configs" / "exp2a1_25m.json"
BRANCH = "experiment-2a1-25m-continuation"
START_UPDATE = 10
TARGET_UPDATE = 48
MILESTONES = (20, 29, 48)
EXPECTED_CONTINUATION_SHA256 = (
    "cf68b9765072e2403c16e935ba02e92f826d48600953f904e11f2bd4d266638e"
)
EXPECTED_CONTINUATION_NEXT_SHA256 = (
    "01cc1b4fe5b9e3e40047f7a686fa683be3bfaaa0f9bbd2fde1752541bd8e0a6a"
)
EXPECTED_CONTINUATION_STATE = {
    "completed_updates": START_UPDATE,
    "processed_student_tokens": START_UPDATE * a0.GLOBAL_BATCH_TOKENS,
}
EXPECTED_FROZEN_BASE_SHA256 = (
    "1bff02fed4110735e5d495cb76670dc05f0b2004371c5ce7df9396ec725095fd"
)
EXPECTED_MILESTONE_NEXT_SHA256 = {
    20: "921abc217182d1f7596f26ac421e0ba317b0c9b8b510a3baa03bf26c604d4471",
    29: "51c1a47728a9293c62481fdd1e5b4f8fe92a5eb5a98494e3a886de29dfa86674",
    48: "1c3290f72d60d356d636e57017bbc5f2cb2ec470af7860d9d95d5c95116d24a5",
}
EXPECTED_TRAINING_HASHES = (
    "01cc1b4fe5b9e3e40047f7a686fa683be3bfaaa0f9bbd2fde1752541bd8e0a6a",
    "f32f9713f981648be7c34336f55519ae1a0d4d2306c279076657703e42cd6790",
    "ea58474bc65e45012ab2540d7331870558f2e1a5f477f28a784ba30501d4daf0",
    "faa3f14cef81dcc4b353efb98241d6622b84d0989bfda7b1f7359af024e2e7db",
    "12874aaf7a4087a3ba7ecd3f82f844f6a28ffe603eb9b7123220cd8712c26c29",
    "67ae9ec62b31966043379f8580e3eb1b84e84a046601ae02e293170693032fcc",
    "278fe745a0dca1ff7e54c7b012929e79c868e2e739edd105e74798809e1cca8f",
    "01f83f664658d13ed1df4fe096e87b1e51189279fb7d6aa6bb56c1c4ddd1e9c2",
    "c5e282add7e9874417aaa04909fd3b4b90be0c3f16c688217b864b71848bfb8e",
    "2cd187df0ee2984705c0f47c2b67cbf9e63846db8ccc0e34a525e61e15604bba",
    "921abc217182d1f7596f26ac421e0ba317b0c9b8b510a3baa03bf26c604d4471",
    "22026663f31f700dcffb403771ac4776e43336944a4f0950a4ad5529686dacd3",
    "749bfa57d092690ef2af2b16658ded5c2bde5a6fbd615f7a7c518fde8d4f12b6",
    "12e2be89f1cc84220b3b91932fe48599fc6e84818c20ca4094d2b62efa38bc97",
    "aa7a995a92ef7f7a4600022f86014676c18d1447ac11224ac066aa2a02404618",
    "07c74bb1d7ab73ef3d74754d5d8d73fed9c60a1336c4bfe78d06bd0c3d161cdf",
    "a9101a740c653502d2db2b3c137c31427846da1fce74968c1b14660db32db050",
    "44439d1af97b4b238799fc7967ebd80336a9749be19907349614a95f0a4b0738",
    "6209a0303cc1a5a6fc1100ac3664f0ecdec86d1a1eac8bfb532b25164fa3348c",
    "51c1a47728a9293c62481fdd1e5b4f8fe92a5eb5a98494e3a886de29dfa86674",
    "be078fdfaaf003572c9b4c8fc1dbe8545f72b1eb13cfa18da60383b80edfde99",
    "d638662befe9b603611a8045b9b7b082a73d3893570919661399c7e26158798f",
    "758684ea3ce8043918171b0bb938d42b6851f4574ceb77559625f238bdf2bc1b",
    "8d11d06c89b1a79987a0f5ea23124d26707aeecd5525d994a623287d80c8a90c",
    "27361e0a8571fff0f2b2230905fc2c1074731390674e6ae72df511c6bfcc597d",
    "132e5153970301048918a328b75411481c6784490abcf9ed2299502533f76c2e",
    "7870ff7767bde30988c20d6d65f1610aa8dcd88c7c1d1a9d0d70c20cf844d842",
    "414bd2dc3eb6b5cc12638a7f018e7e4b96d1a14b3e84fea47f2572d9eed14eb9",
    "d2e36519c3bda897292ee0301a8f9b007e6dfc87e30c73bc390dde729c1d9b74",
    "42ab6edb525e1589155dea9ba1417041cf793a68945ec7cb84531f8331ef786a",
    "9ee1f17b9a18bb9a2529efec47b8c496aefe3fd1b0e0d1a4840540217a4ce492",
    "3870edef4b558633e9d6ebe174ef60230b699013119962def468bc9f5cb409cf",
    "1e1597d784eaba1c4e0d9a1b0d3b297e94646bd487e874da748f4feb2d44a149",
    "3642f263729e9325babec3781ff90df9d92c711c148da83d53c726206efb5305",
    "ecc590a4a24e74b822dc5e6ecf54fa193bdad11e8e4c83c6db292dac4ed06efe",
    "d2cfb10e5c4ef5b0550124910e425ea923ab97b3a9a400bfdee6d287d4303d44",
    "ea92439a4ee5186e2107e8d11eeb541a4222b5434b5c5cb5dba36f8ff4c6f12c",
    "5c5acc9efdb3af6c97e46f75e471f4866e29c3bb7cf4657fa67f468d17c9dc97",
)
EXPECTED_HASH_BY_UPDATE = {
    update: digest for update, digest in enumerate(EXPECTED_TRAINING_HASHES, START_UPDATE)
}
PINNED_FULL = 4.078654408454895
PINNED_MASKED = 5.973674488067627
PINNED_UPDATE10_REAL = 5.953305101394653
PINNED_UPDATE10_SHUFFLED = 5.961796236038208
T19_975 = 2.093024054408263
CLASSIFICATION_RULE = (
    "Let s be shuffled-minus-real recovery at updates 10,20,29,48. "
    "DISAPPEARING if final<=0 or final<half the positive update-10 value; "
    "STABLE if the full range is <=25% of max(abs(mean(s)),1e-12); "
    "SATURATING if final is positive and either below the prior maximum or the "
    "29-to-48 gain is <=25% of the positive 10-to-29 gain; otherwise STRENGTHENING."
)
PINNED_PYTHON = Path("/workspace/venvs/exp1b/bin/python")


def validate_config(config):
    expected = {
        "protocol": "exp2a1_teacher_topdown_25m_v1",
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
        "lr_schedule": "continue original 10B schedule; next global schedule step is 964",
        "data_start": (
            "restore exact four-rank replay states from verified Experiment 2A0 "
            "update-10 checkpoint"
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
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if set(config) != set(expected):
        mismatches["fields"] = (sorted(config), sorted(expected))
    if mismatches:
        raise SystemExit(f"Experiment 2A1 config mismatch: {mismatches}")
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
        raise SystemExit(f"2A1 metrics history mismatch: {updates} != {expected}")
    trailing = rows[len(prefix):]
    if any(row.get("update", -1) < completed_updates for row in trailing):
        raise SystemExit("2A1 metrics contain duplicate/out-of-order committed rows")
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
    return REPO_ROOT / "results" / "experiment_2a0_5m" / "metadata.json"


def source_summary_path():
    return REPO_ROOT / "results" / "experiment_2a0_5m" / "result_summary.json"


def load_source_metadata():
    metadata_path = source_metadata_path()
    summary_path = source_summary_path()
    if not metadata_path.is_file() or not summary_path.is_file():
        raise SystemExit("committed Experiment 2A0 5M lineage artifacts are missing")
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads(summary_path.read_text())
    if metadata.get("git_commit") != "2eaa26f3a3c1d32c5172a522a2fa96bed4a3b70f":
        raise SystemExit("source 5M metadata commit mismatch")
    if summary.get("completed_updates") != START_UPDATE:
        raise SystemExit("source 5M summary mismatch")
    embedded = metadata.get("source_file_sha256", {})
    live = {
        "train_gpt2.py": a0.file_sha256(REPO_ROOT / "train_gpt2.py"),
        "scripts/experiment_2a0.py": a0.file_sha256(
            REPO_ROOT / "scripts" / "experiment_2a0.py"
        ),
        "configs/exp2a0_5m.json": a0.file_sha256(
            REPO_ROOT / "configs" / "exp2a0_5m.json"
        ),
    }
    mismatches = {
        name: (digest, embedded.get(name))
        for name, digest in live.items()
        if embedded.get(name) != digest
    }
    if mismatches:
        raise SystemExit(f"live Experiment 2A0 source lineage mismatch: {mismatches}")
    return metadata, summary


def require_clean_tree():
    if a0.git_output("status", "--porcelain"):
        raise SystemExit("2A1 optimizer runs require a clean committed worktree")
    if a0.git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"2A1 requires branch {BRANCH}")


def require_pinned_runtime():
    actual = Path(sys.executable).resolve()
    if not PINNED_PYTHON.exists():
        raise SystemExit(f"pinned 2A1 Python is missing: {PINNED_PYTHON}")
    if actual != PINNED_PYTHON.resolve():
        raise SystemExit(
            f"2A1 requires pinned Python {PINNED_PYTHON.resolve()}, got {actual}"
        )
    return str(actual)


def source_hashes():
    files = (
        "train_gpt2.py",
        "scripts/experiment_2a0.py",
        "scripts/experiment_2a1.py",
        "configs/exp2a0_5m.json",
        "configs/exp2a1_25m.json",
    )
    return {name: a0.file_sha256(REPO_ROOT / name) for name in files}


def continuation_metadata(config, parent_aux, continuation_path):
    source_metadata, _ = load_source_metadata()
    return {
        "experiment": "Experiment 2A1",
        "kind": "exact update-10 to update-48 continuation",
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
            "results/experiment_2a0_5m/config.json": a0.file_sha256(
                REPO_ROOT / "results" / "experiment_2a0_5m" / "config.json"
            ),
            "results/experiment_2a0_5m/metadata.json": a0.file_sha256(
                source_metadata_path()
            ),
            "results/experiment_2a0_5m/result_summary.json": a0.file_sha256(
                source_summary_path()
            ),
        },
        "source_depths": list(a0.SOURCE_DEPTHS),
        "destination": "Block 1 Attention input",
        "teacher": "immutable Experiment 1B parent; frozen eval/no_grad/detached",
        "student_base": "frozen",
        "trainable_parameters": 1537,
        "trajectory_classification_rule": CLASSIFICATION_RULE,
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
        raise SystemExit("Experiment 2A0 update-10 continuation SHA mismatch")
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
    if [int(v["step"].item()) for v in checkpoint["optimizer"]["state"].values()] != [10] * 3:
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
    for update, expected in EXPECTED_HASH_BY_UPDATE.items():
        if result[update] != expected:
            raise SystemExit(f"replay oracle mismatch at update {update}")
    for milestone, expected in EXPECTED_MILESTONE_NEXT_SHA256.items():
        if result[milestone] != expected:
            raise SystemExit(f"milestone next-hash mismatch at {milestone}")
    return result


def assert_runtime_contract(student, teacher, optimizer, completed_updates):
    contract = a0.smoke_model_contract(student, teacher)
    if sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad) != 1537:
        raise SystemExit("Experiment 2A1 trainable-parameter count changed")
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
        raise SystemExit(f"invalid 2A1 update index: {update}")
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
    if completed_updates not in (START_UPDATE,) + MILESTONES:
        raise SystemExit(f"invalid evaluation milestone: {completed_updates}")
    if full != (completed_updates == TARGET_UPDATE):
        raise SystemExit("only update 48 may use the full control matrix")
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
    if completed_updates == START_UPDATE:
        if losses["real_feedback"] != PINNED_UPDATE10_REAL:
            raise SystemExit(
                f"update-10 real loss mismatch: {losses['real_feedback']} "
                f"!= {PINNED_UPDATE10_REAL}"
            )
        if losses["shuffled_feedback"] != PINNED_UPDATE10_SHUFFLED:
            raise SystemExit(
                f"update-10 shuffled loss mismatch: {losses['shuffled_feedback']} "
                f"!= {PINNED_UPDATE10_SHUFFLED}"
            )
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
    update10_valid = True
    if completed_updates == START_UPDATE:
        update10_valid = (
            real_value == PINNED_UPDATE10_REAL
            and shuffled_value == PINNED_UPDATE10_SHUFFLED
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
        or not update10_valid
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


def classify_trajectory(rows):
    if len(rows) != 4:
        raise ValueError("trajectory classification requires exactly four milestones")
    completed = [row.get("completed_updates") for row in rows]
    if completed != [START_UPDATE, *MILESTONES]:
        raise ValueError(
            "trajectory milestones must be ordered exactly as updates 10,20,29,48"
        )
    values = []
    for row in rows:
        value = row.get("sequence_specific_recovery")
        if value is None and "evaluation" in row:
            value = row["evaluation"].get("sequence_specific_recovery")
        if value is None or not math.isfinite(float(value)):
            raise ValueError("trajectory row lacks finite sequence-specific recovery")
        values.append(float(value))
    initial, _, penultimate, final = values
    scale = max(abs(statistics.fmean(values)), 1e-12)
    if final <= 0 or (initial > 0 and final < 0.5 * initial):
        label = "MEMORY SIGNAL DISAPPEARING"
    elif max(values) - min(values) <= 0.25 * scale:
        label = "MEMORY SIGNAL STABLE"
    else:
        early_gain = penultimate - initial
        late_gain = final - penultimate
        if final < max(values[:-1]) or (
            early_gain > 0 and late_gain <= 0.25 * early_gain
        ):
            label = "MEMORY SIGNAL SATURATING"
        else:
            label = "MEMORY SIGNAL STRENGTHENING"
    return {"label": label, "values": values, "rule": CLASSIFICATION_RULE}


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


def validate_training_rows(rows, completed_updates):
    expected_updates = list(range(START_UPDATE, completed_updates))
    if [row.get("update") for row in rows] != expected_updates:
        raise SystemExit("2A1 training rows are missing, duplicated, or out of order")
    for row in rows:
        update = row["update"]
        expected = {
            "completed_updates": update + 1,
            "processed_student_tokens": (update + 1) * a0.GLOBAL_BATCH_TOKENS,
            "global_schedule_step": a0.EXPECTED_PARENT_UPDATES + update,
            "lr": a0.get_lr(a0.EXPECTED_PARENT_UPDATES + update),
            "global_batch_sha256": EXPECTED_HASH_BY_UPDATE[update],
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
            f"2A1 resume checkpoint is not a milestone: {preview_completed}"
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
        raise SystemExit(f"2A1 resume checkpoint is not a milestone: {completed}")
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
    evaluation_before = snapshot_training_boundary(
        student, teacher, optimizer, loaders
    )
    source_evaluation = evaluate_milestone(
        student, teacher, symbols, device, START_UPDATE, full=False
    )
    evaluation_isolation = assert_training_boundary_unchanged(
        evaluation_before, student, teacher, optimizer, loaders
    )
    report = {
        "kind": "Experiment 2A1 optimizer-free continuation preflight",
        "git_commit": a0.git_output("rev-parse", "HEAD"),
        "git_branch": a0.git_output("branch", "--show-current"),
        "config": config,
        "dataset": dataset,
        "source_summary_sha256": a0.file_sha256(source_summary_path()),
        "source_checkpoint_sha256": EXPECTED_CONTINUATION_SHA256,
        "source_training_state": state,
        "source_restore": restore_audit,
        "runtime_contract": contract,
        "frozen_state": frozen,
        "teacher_memory_causality": memory_causality,
        "end_to_end_causality": end_to_end_causality,
        "causality_state_isolation": isolation,
        "canonical_update10_evaluation": source_evaluation,
        "canonical_update10_evaluation_state_isolation": evaluation_isolation,
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
    validate_existing_evaluation(
        evaluation_path(run_dir, START_UPDATE),
        START_UPDATE,
        EXPECTED_CONTINUATION_SHA256,
        metadata,
    )
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
    causality10_path = run_dir / f"causality_updates_{START_UPDATE:06d}.json"
    if not causality10_path.is_file() or not validate_causality_record(
        json.loads(causality10_path.read_text())
    ):
        raise SystemExit("resume update-10 causality audit is missing or invalid")
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


def final_summary(run_dir, metadata, checkpoint_records, restart_records, causality):
    run_dir = Path(run_dir)
    rows = read_metrics(run_dir / "metrics.jsonl")
    row_audit = validate_training_rows(rows, TARGET_UPDATE)
    source_evaluation = validate_existing_evaluation(
        evaluation_path(run_dir, START_UPDATE),
        START_UPDATE,
        EXPECTED_CONTINUATION_SHA256,
        metadata,
    )
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
                "paired": evaluation["paired_shuffled_minus_real"],
                "routing": evaluation["routing"],
            }
        )
    classification = classify_trajectory(trajectory_rows)
    final = evaluations[TARGET_UPDATE]
    summary = {
        "experiment": "Experiment 2A1",
        "completed_updates": TARGET_UPDATE,
        "processed_student_tokens": TARGET_UPDATE * a0.GLOBAL_BATCH_TOKENS,
        "additional_updates": TARGET_UPDATE - START_UPDATE,
        "additional_student_tokens": (TARGET_UPDATE - START_UPDATE)
        * a0.GLOBAL_BATCH_TOKENS,
        "trajectory": trajectory_rows,
        "trajectory_classification": classification,
        "final_controls": final["losses"],
        "final_damage": final["damage"],
        "final_total_recovery": final["total_recovery"],
        "final_total_recovery_fraction": final["total_recovery_fraction"],
        "final_sequence_specific_recovery": final["sequence_specific_recovery"],
        "final_sequence_specific_recovery_fraction": final[
            "sequence_specific_recovery_fraction"
        ],
        "final_router": final["routing"],
        "final_source_ablation": final["source_ablation"],
        "checkpoint_records": checkpoint_records,
        "restart_records": restart_records,
        "causality": causality,
        "training_row_audit": row_audit,
        "resources": {
            "peak_allocated_mb": max(row["peak_allocated_mb"] for row in rows),
            "peak_reserved_mb": max(row["peak_reserved_mb"] for row in rows),
            "training_update_wall_seconds": sum(row["wall_seconds"] for row in rows),
            "training_forward_seconds": sum(row["forward_seconds"] for row in rows),
            "training_backward_seconds": sum(row["backward_seconds"] for row in rows),
            "evaluation_wall_seconds": sum(
                evaluation["elapsed_seconds"] for evaluation in evaluations.values()
            ),
        },
        "terminal_line": "EXPERIMENT 2A1 25M COMPLETE",
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
        bound_milestone_evaluation(
            evaluation_path(run_dir, START_UPDATE),
            student,
            teacher,
            optimizer,
            loaders,
            symbols,
            device,
            START_UPDATE,
            EXPECTED_CONTINUATION_SHA256,
            metadata,
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
    rows = read_metrics(run_dir / "metrics.jsonl")
    validate_training_rows(rows, TARGET_UPDATE)
    # Preserve restart/causality artifacts from any earlier invocation.
    for milestone in MILESTONES[:-1]:
        restart_path = run_dir / f"restart_audit_updates_{milestone:06d}.json"
        if restart_path.is_file() and str(milestone) not in restart_records:
            restart_records[str(milestone)] = json.loads(restart_path.read_text())
    for milestone in (START_UPDATE, *MILESTONES):
        causality_path = run_dir / f"causality_updates_{milestone:06d}.json"
        if causality_path.is_file() and str(milestone) not in causality_records:
            causality_records[str(milestone)] = json.loads(causality_path.read_text())
    if set(restart_records) != {"20", "29"} or not all(
        validate_restart_record(record) for record in restart_records.values()
    ):
        raise SystemExit(f"milestone restart audit is incomplete: {restart_records}")
    if set(causality_records) != {"10", "20", "29", "48"} or not all(
        validate_causality_record(record) for record in causality_records.values()
    ):
        raise SystemExit(f"milestone causality audit is incomplete: {causality_records}")
    summary = final_summary(
        run_dir, metadata, checkpoint_records, restart_records, causality_records
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
            for milestone in (START_UPDATE, *MILESTONES)
        ]
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
    continuation = subparsers.add_parser("continue-25m")
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
