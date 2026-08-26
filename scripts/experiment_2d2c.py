#!/usr/bin/env python3
"""Experiment 2D2C: add the mirrored full-bank B11-to-B2 recurrent K/V link.

The finalized 2D2B checkpoint is the immutable source.  This driver restores
its model, optimizer, data loader, RNG, and next-batch state; preserves the
trained B12-to-B1 path; shortens B2 to W2; and adds exactly one scalar gate for
an attached B11-to-B2 full recurrent bank.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import pickle
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d2a as legacy  # noqa: E402
import experiment_2d2a_core as legacy_core  # noqa: E402
import experiment_2d2b as source_driver  # noqa: E402
import experiment_2d2b_core as source_core  # noqa: E402
from experiment_2d2c_core import (  # noqa: E402
    BANK_MODES,
    INCREMENTAL_CONTROLS,
    MAX_RECURRENT_ENTRIES,
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
    RecurrentKVGPT,
)


EXPERIMENT = "2D2C"
PROTOCOL = "exp2d2c_b12_b1_b11_b2_full_recurrent_kv_v1"
BRANCH = "experiment-2d2c-b12-b1-b11-b2-full-recurrent-kv"
FROZEN_TAG = "experiment-2d2b-full-b12-b1-recurrent-bank-final"
FROZEN_COMMIT = "976b92927e698afd27d68eabc78db5a0b6714fef"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d2c_b12_b1_b11_b2_full_recurrent_kv.json"
OUTPUT_NAME = "experiment_2d2c_b12_b1_b11_b2_full_recurrent_kv"
CHECKPOINT_SCHEMA = "exp2d2c_b12_b1_b11_b2_full_recurrent_kv_checkpoint_v1"
SOURCE_SCHEMA = "exp2d2b_full_b12_b1_recurrent_bank_checkpoint_v1"
SOURCE_SHA256 = "8c39f47248e3a5f4dc69f5e8e97c8a1cd1bcdfa91154eba5804c448942075326"
SOURCE_BYTES = 1_493_937_033
SOURCE_UPDATES = 191
SOURCE_TARGETS = 150_470_656
SOURCE_GATE_RAW = 0.07656901329755783
SOURCE_GATE_EFFECTIVE = 0.07641972601413727
SOURCE_NEXT_BATCH_SHA256 = "e1d96ca0106f21badeb0004025e80abc562509fb6299a63eb8662a3da3c17a52"
SOURCE_NEXT_STREAM_SHA256 = "fc01029471dfe8674e900dd3d1e20a34e235853d44c68ede1b67f5b1a61e44f0"
PERSISTENT_VOLUME_IDENTITY = "yhzyb27fb5"

T = 1024
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
VOCAB_SIZE = 50_304
W_LOCAL = 2
GLOBAL_TARGETS = 524_288
MAX_UPDATES = 191
ADDITIONAL_TARGETS = MAX_UPDATES * GLOBAL_TARGETS
CUMULATIVE_TARGETS = SOURCE_TARGETS + ADDITIONAL_TARGETS
TOTAL_PARAMETERS = 124_475_906
SOURCE_PARAMETERS = 124_475_905
BASE_LR = 3e-5
GATE_LR = 3e-4
GRAD_CLIP = 1.0
TWO_PASS_WEIGHTS = (0.25, 0.75)
THREE_PASS_WEIGHTS = (0.20, 0.40, 0.40)
MILESTONES = (0, 20, 48, 96, 143, 191)
SCIENTIFIC_CHECKPOINTS = (48, 96, 143, 191)
RECOVERY_CHECKPOINTS = (50, 100, 150)
FORCED_RESTART_UPDATE = 96
VALIDATION_BATCHES = 20
VALIDATION_B = 64
INCREMENTAL_BATCHES = 2
CANONICAL_VALIDATION_SHA256 = legacy.CANONICAL_VALIDATION_SHA256
VALIDATION_SHARD_SHA256 = legacy.VALIDATION_SHARD_SHA256
SEED = 2026_0221
BF16_INCREMENTAL_ATOL = 1.25
FP32_INCREMENTAL_ATOL = 1e-4
LAG_BINS = (
    ("2-7", 2, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
    ("32-63", 32, 63),
    ("64-127", 64, 127),
    ("128-255", 128, 255),
    ("256-511", 256, 511),
    ("512-1023", 512, 1023),
)
POSITION_BINS = (
    ("2-15", 2, 15),
    ("16-31", 16, 31),
    ("32-63", 32, 63),
    ("64-127", 64, 127),
    ("128-255", 128, 255),
    ("256-511", 256, 511),
    ("512-767", 512, 767),
    ("768-1023", 768, 1023),
)
IMPLEMENTATION_FILES = (
    "configs/exp2d2c_b12_b1_b11_b2_full_recurrent_kv.json",
    "scripts/experiment_2d2c.py",
    "scripts/experiment_2d2c_core.py",
    "scripts/experiment_2d2b.py",
    "scripts/experiment_2d2b_core.py",
    "scripts/experiment_2d2a.py",
    "scripts/experiment_2d2a_core.py",
    "scripts/experiment_2d0.py",
    "scripts/experiment_2d0d.py",
    "scripts/experiment_2d1.py",
    "scripts/smoke_test.py",
    "train_gpt2.py",
    "tests/test_experiment_2d2c_core.py",
    "tests/test_experiment_2d2c_driver.py",
)
REQUIRED_ARTIFACTS = (
    "EXPERIMENT_2D2C_FINAL_REPORT.md",
    "FINAL_AUDIT.json",
    "result_summary.json",
    "source_manifest.json",
    "parameter_manifest.json",
    "architecture_manifest.json",
    "semantic_diff_audit.json",
    "batch_manifest.json",
    "preflight_audit.json",
    "initial_b2_shortening.json",
    "training_metrics.jsonl",
    "milestone_validation.json",
    "paired_controls.json",
    "gate_diagnostics.json",
    "b1_attention_lag_bins.json",
    "b2_attention_lag_bins.json",
    "b1_attention_head_distance.json",
    "b2_attention_head_distance.json",
    "b12_to_b1_temporal_gradient.json",
    "b11_to_b2_temporal_gradient.json",
    "incremental_validation.json",
    "incremental_cache_audit.json",
    "memory_accounting.json",
    "checkpoint_manifest.json",
    "performance.json",
    "commands_and_runtime.json",
    "HEARTBEAT.json",
    "UNATTENDED_FINAL_HANDOFF.md",
)
REQUIRED_PLOTS = tuple(f"P{index}_{name}.png" for index, name in enumerate((
    "validation_ce",
    "b2_recurrent_gain",
    "b2_compression_gap",
    "b2_sequence_gap",
    "recurrent_gates",
    "b1_b2_attention_mass_by_lag",
    "b1_b2_attention_density_by_lag",
    "b1_b2_per_head_mean_lag",
    "b11_b12_temporal_writer_gradient",
    "b2_compression_recovery",
    "parallel_vs_true_self",
    "runtime_vram",
), start=1))


read_json = legacy.read_json
read_jsonl = legacy.read_jsonl
durable_json = legacy.durable_json
durable_text = legacy.durable_text
append_jsonl = legacy.append_jsonl
file_sha256 = legacy.file_sha256
paired_stats = legacy.paired_stats
batch_hash = legacy.batch_hash
capture_rng_state = legacy.capture_rng_state
restore_rng_state = legacy.restore_rng_state
model_finite = legacy.model_finite
optimizer_finite = legacy.optimizer_finite
gradients_finite = legacy.gradients_finite
gradient_group_report = legacy.gradient_group_report


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean: bool = True) -> None:
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"{EXPERIMENT} requires branch {BRANCH}")
    if git_output("rev-parse", FROZEN_TAG + "^{commit}") != FROZEN_COMMIT:
        raise SystemExit("frozen 2D2B tag mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=REPO_ROOT
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D2C worktree must be clean")


def require_config() -> dict:
    config = read_json(CONFIG_PATH)
    checks = {
        "experiment": config.get("experiment") == EXPERIMENT,
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "frozen_tag": config.get("frozen_2d2b_tag") == FROZEN_TAG,
        "frozen_commit": config.get("frozen_2d2b_commit") == FROZEN_COMMIT,
        "source_sha": config.get("source_checkpoint_sha256") == SOURCE_SHA256,
        "source_bytes": config.get("source_checkpoint_bytes") == SOURCE_BYTES,
        "source_state": config["source_2d2b"]["completed_updates"] == SOURCE_UPDATES
        and config["source_2d2b"]["completed_targets"] == SOURCE_TARGETS
        and config["source_2d2b"]["raw_g_rec_b1"] == SOURCE_GATE_RAW
        and config["source_2d2b"]["next_global_batch_sha256"]
        == SOURCE_NEXT_BATCH_SHA256,
        "geometry": config["architecture"]["b1"]["local_window"] == W_LOCAL
        and config["architecture"]["b2"]["local_window"] == W_LOCAL
        and config["architecture"]["recurrent_min_lag"] == 2
        and config["architecture"]["recurrent_max_lag"] == RECURRENT_MAX_LAG
        and config["architecture"]["maximum_recurrent_entries_per_link"]
        == MAX_RECURRENT_ENTRIES,
        "one_parameter": config["architecture"]["new_parameter_count_vs_2d2b"] == 1
        and config["architecture"]["new_parameters_vs_2d2b"] == ["g_rec_b2"]
        and config["architecture"]["total_parameters"] == TOTAL_PARAMETERS,
        "budget": config["training"]["additional_updates"] == MAX_UPDATES
        and config["training"]["additional_targets"] == ADDITIONAL_TARGETS
        and config["training"]["cumulative_2d2_targets"] == CUMULATIVE_TARGETS,
        "milestones": tuple(config["training"]["local_milestones"]) == MILESTONES,
        "checkpoints": tuple(config["training"]["scientific_checkpoint_updates"])
        == SCIENTIFIC_CHECKPOINTS
        and tuple(config["training"]["recovery_checkpoint_updates"])
        == RECOVERY_CHECKPOINTS,
        "optimizer_resume": config["training"]["optimizer"]["source_optimizer_restored"]
        is True
        and config["training"]["optimizer"]["warmup_restarted"] is False,
        "attached": config["training"]["recurrent_sources_detached"] is False,
        "hardware": config["hardware"]["pod_id"] == "e8nd7m6piw5km2"
        and config["hardware"]["pod_name"] == "serious_indigo_swordfish"
        and config["hardware"]["gpu_count"] == 1,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D2C preregistration mismatch: {checks}")
    return config


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_single_a100() -> torch.device:
    return legacy.require_single_a100()


def environment_payload() -> dict:
    payload = legacy.environment_payload()
    payload["experiment"] = EXPERIMENT
    return payload


def implementation_fingerprint() -> dict:
    files = {}
    for relative in IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise SystemExit(f"implementation dependency missing: {relative}")
        files[relative] = file_sha256(path)
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"algorithm": "sha256", "files": files, "aggregate_sha256": aggregate}


def require_implementation_fingerprint(preflight: dict) -> dict:
    current = implementation_fingerprint()
    if current != preflight.get("implementation_fingerprint"):
        raise SystemExit("scientific implementation differs from passing preflight")
    return current


def workspace_mount_audit(output_dir, run_root, supplied_identity) -> dict:
    output = Path(output_dir).resolve()
    run = Path(run_root).resolve()
    expected_output = (REPO_ROOT / "results" / OUTPUT_NAME).resolve()
    if supplied_identity != PERSISTENT_VOLUME_IDENTITY:
        raise SystemExit("unexpected persistent-volume identity")
    if output != expected_output:
        raise SystemExit(f"2D2C result directory must be exactly {expected_output}")
    if not str(output).startswith("/workspace/") or not str(run).startswith("/workspace/"):
        raise SystemExit("2D2C results and checkpoints must live under /workspace")
    row = subprocess.check_output(
        ["findmnt", "-T", "/workspace", "-n", "-o", "TARGET,SOURCE,FSTYPE"],
        text=True,
    ).strip().split(maxsplit=2)
    target, source, filesystem = row
    checks = {
        "target_exact": target == "/workspace",
        "persistent_identity_exact": f"/networkvolumes/{PERSISTENT_VOLUME_IDENTITY}"
        in source,
        "fuse_network_mount": filesystem == "fuse",
        "canonical_result_directory": output == expected_output,
        "run_root_on_workspace": str(run).startswith("/workspace/"),
    }
    if not all(checks.values()):
        raise SystemExit(f"persistent workspace audit failed: {checks}")
    return {
        "persistent_volume_identity": PERSISTENT_VOLUME_IDENTITY,
        "target": target,
        "source": source,
        "filesystem": filesystem,
        "output_dir": str(output),
        "run_root": str(run),
        "checks": checks,
        "passed": True,
    }


def authenticated_stop_audit(args) -> dict:
    path = Path(args.stop_audit_path).resolve()
    if not path.is_file():
        raise SystemExit("authenticated RunPod stop audit is missing")
    payload = read_json(path)
    response = payload.get("authenticated_pod_identity_response", {})
    checks = {
        "schema": payload.get("schema") == "exp2d2c_runpod_stop_capability_v1",
        "authenticated_probe": payload.get("authenticated_list_probe") is True,
        "credential_available": payload.get("stop_credential_available") is True,
        "secret_not_recorded": payload.get("secret_recorded") is False,
        "pod_id": response.get("id") == args.pod_id == "e8nd7m6piw5km2",
        "pod_name": response.get("name")
        == args.pod_name
        == "serious_indigo_swordfish",
        "gpu_count": response.get("gpuCount") == 1,
        "runtime_running": response.get("runtimeStatus") == "running",
        "exact_stop_target": payload.get("exact_stop_target") == "e8nd7m6piw5km2",
        "passed": payload.get("passed") is True,
        "cli_authorized": bool(args.stop_authenticated),
    }
    if not all(checks.values()):
        raise SystemExit(f"authenticated RunPod stop audit failed: {checks}")
    return {
        **payload,
        "audit_path": str(path),
        "audit_sha256": file_sha256(path),
        "driver_checks": checks,
        "driver_passed": True,
    }


def state_fingerprint(value) -> str:
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
        return hashlib.sha256(
            str(array.dtype).encode()
            + repr(array.shape).encode()
            + array.tobytes()
        ).hexdigest()
    if isinstance(value, np.ndarray):
        return hashlib.sha256(
            str(value.dtype).encode()
            + repr(value.shape).encode()
            + value.tobytes()
        ).hexdigest()
    if isinstance(value, (list, tuple)):
        return hashlib.sha256(
            "|".join(state_fingerprint(item) for item in value).encode()
        ).hexdigest()
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def source_correction_provenance() -> dict:
    relative = Path("results/experiment_2d2a_b12_b1_recurrent_kv")
    paths = {
        "correction": relative / "POST_TRAINING_AUDIT_CORRECTION.json",
        "wrapper": relative / "POST_TRAINING_AUDIT_CORRECTION_WRAPPER.py",
        "failed_log": relative / "POST_TRAINING_AUDIT_FAILED_FINALIZE.log",
    }
    correction = read_json(REPO_ROOT / paths["correction"])
    checks = {
        "all_present": all((REPO_ROOT / path).is_file() for path in paths.values()),
        "evaluation_only": correction["training_or_model_changed"] is False,
        "checkpoint_unchanged": correction["checkpoint_changed"] is False,
        "data_metric_unchanged": correction["data_or_metric_changed"] is False,
        "fp32_unchanged": correction["observed_equivalence"]["fp32_passed"] is True,
        "corrected_bf16_semantics": correction["correction"].startswith(
            "Use the already-preregistered BF16 Plain absolute-max tolerance 1.25"
        ),
    }
    return {
        "artifacts": {
            name: {"path": str(path), "sha256": file_sha256(REPO_ROOT / path)}
            for name, path in paths.items()
        },
        "correction": correction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def training_shards(data_root):
    return legacy.training_shards(data_root)


def validation_path(data_root):
    return legacy.validation_path(data_root)


def next_global_batch_hash(loader, accumulation):
    return legacy.next_global_batch_hash(loader, accumulation)


def pass_count(local_update: int) -> int:
    if not 1 <= int(local_update) <= MAX_UPDATES:
        raise ValueError(local_update)
    return 3 if int(local_update) % 32 == 0 else 2


def pass_weights(local_update: int):
    return THREE_PASS_WEIGHTS if pass_count(local_update) == 3 else TWO_PASS_WEIGHTS


def _token_losses(logits, targets):
    return legacy._token_losses(logits, targets)


def instantiate_model(device: torch.device, trainable: bool = True):
    symbols = legacy.d0.support.load_training_symbols()
    base = symbols["GPT"](legacy.d0.model_config(symbols))
    for parameter in base.parameters():
        parameter.requires_grad_(bool(trainable))
    model = RecurrentKVGPT(base).to(device)
    if model.base.transformer.wte.weight is not model.base.lm_head.weight:
        raise SystemExit("embedding/LM-head tying was not preserved")
    return symbols, model


def configure_optimizer(model, device_type="cuda", include_new_gate=True):
    """Recreate the exact 2D2B groups, then optionally add the fresh B2 gate."""

    base_decay, base_nodecay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name in {"g_rec", "g_rec_b2"}:
            continue
        (base_decay if parameter.dim() >= 2 else base_nodecay).append(parameter)
    groups = [
        {
            "name": "base_decay",
            "params": base_decay,
            "lr": BASE_LR,
            "weight_decay": legacy.WEIGHT_DECAY,
        },
        {
            "name": "base_nodecay",
            "params": base_nodecay,
            "lr": BASE_LR,
            "weight_decay": 0.0,
        },
        {
            "name": "gate",
            "params": [model.g_rec],
            "lr": GATE_LR,
            "weight_decay": 0.0,
        },
    ]
    if include_new_gate:
        groups.append(
            {
                "name": "b2_gate",
                "params": [model.g_rec_b2],
                "lr": GATE_LR,
                "weight_decay": 0.0,
            }
        )
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        groups,
        betas=legacy.BETAS,
        eps=legacy.ADAM_EPS,
        fused=fused_available and device_type == "cuda",
    )
    report = {
        "physical_parameter_groups": len(groups),
        "groups": [
            {
                "name": group["name"],
                "parameters": sum(parameter.numel() for parameter in group["params"]),
                "tensors": len(group["params"]),
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
            }
            for group in groups
        ],
        "betas": list(legacy.BETAS),
        "eps": legacy.ADAM_EPS,
        "fused": fused_available and device_type == "cuda",
        "source_optimizer_restored": False,
        "new_b2_gate_state_fresh": bool(include_new_gate),
        "warmup_restarted": False,
    }
    return optimizer, report


def load_source_bundle(source_checkpoint, device, restore_rng=False):
    path = Path(source_checkpoint).resolve()
    observed_sha = file_sha256(path)
    if observed_sha != SOURCE_SHA256 or path.stat().st_size != SOURCE_BYTES:
        raise SystemExit("2D2B scientific source checkpoint identity mismatch")
    sha_sidecar = path.with_suffix(path.suffix + ".sha256")
    verification_sidecar = path.with_suffix(path.suffix + ".verification.json")
    if not sha_sidecar.is_file() or not verification_sidecar.is_file():
        raise SystemExit("2D2B checkpoint sidecars missing")
    if sha_sidecar.read_text().split()[0] != SOURCE_SHA256:
        raise SystemExit("2D2B checkpoint SHA sidecar mismatch")
    payload = legacy.d0.torch_load(path, mmap=True)
    required = {
        "schema",
        "model",
        "g_rec",
        "optimizer",
        "completed_2d2b_updates",
        "processed_2d2b_targets",
        "cumulative_2d2_targets",
        "source_2d2a_updates",
        "source_2d2a_targets",
        "training_state",
        "loader_state",
        "rng_state",
        "next_global_batch_sha256",
        "next_global_batch_stream_sha256",
        "architecture_manifest",
        "source_checkpoint_sha256",
        "metadata",
        "git_commit",
        "saved_process_id",
        "environment",
    }
    checks = {
        "fields": set(payload) == required,
        "schema": payload.get("schema") == SOURCE_SCHEMA,
        "updates": payload.get("completed_2d2b_updates") == SOURCE_UPDATES,
        "targets": payload.get("cumulative_2d2_targets") == SOURCE_TARGETS,
        "gate_duplicate": torch.equal(payload.get("g_rec"), payload["model"]["g_rec"]),
        "gate_raw": float(payload["g_rec"]) == SOURCE_GATE_RAW,
        "next_batch_recorded": payload.get("next_global_batch_sha256")
        == SOURCE_NEXT_BATCH_SHA256,
        "next_stream_recorded": payload.get("next_global_batch_stream_sha256")
        == SOURCE_NEXT_STREAM_SHA256,
        "source_git_is_frozen_ancestor": subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                str(payload.get("git_commit")),
                FROZEN_COMMIT,
            ],
            cwd=REPO_ROOT,
        )
        == 0,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D2B source schema mismatch: {checks}")

    symbols = legacy.d0.support.load_training_symbols()
    source_base = symbols["GPT"](legacy.d0.model_config(symbols))
    source_model = source_core.RecurrentKVGPT(source_base).to(device)
    missing, unexpected = source_model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise SystemExit(f"strict source model load failed: {missing}, {unexpected}")
    source_optimizer, _ = source_driver.configure_optimizer(source_model, device.type)
    source_optimizer.load_state_dict(payload["optimizer"])

    # Reuse the exact parameter objects owned by the restored source optimizer.
    model = RecurrentKVGPT(source_model.base).to(device)
    model.g_rec = source_model.g_rec
    optimizer = source_optimizer
    optimizer.add_param_group(
        {
            "name": "b2_gate",
            "params": [model.g_rec_b2],
            "lr": GATE_LR,
            "weight_decay": 0.0,
        }
    )
    _, optimizer_report = configure_optimizer(model, device.type, include_new_gate=True)
    optimizer_report.update(
        {
            "source_optimizer_restored": True,
            "source_state_entries": len(payload["optimizer"]["state"]),
            "source_parameter_groups": len(payload["optimizer"]["param_groups"]),
            "new_b2_gate_state_fresh": model.g_rec_b2 not in optimizer.state,
        }
    )
    source_loader = legacy.d1.ExplicitShardLoader(
        payload["loader_state"]["shards"],
        payload["loader_state"]["batch_size"],
        payload["loader_state"]["sequence_length"],
        state=payload["loader_state"],
    )
    source_accumulation = int(payload["metadata"]["gradient_accumulation"])
    observed_next = next_global_batch_hash(source_loader, source_accumulation)
    observed_stream = global_batch_stream_hash(source_loader, source_accumulation)
    checks.update(
        {
            "strict_model": True,
            "model_finite": model_finite(model),
            "optimizer_finite": optimizer_finite(optimizer),
            "parameter_count": sum(value.numel() for value in model.parameters())
            == TOTAL_PARAMETERS,
            "source_parameter_count": sum(value.numel() for value in source_model.parameters())
            == SOURCE_PARAMETERS,
            "gate_effective": model.recurrent_scale_b1.detach().float().item()
            == SOURCE_GATE_EFFECTIVE,
            "b2_gate_zero": model.g_rec_b2.detach().float().item() == 0.0,
            "next_batch_reproduced": observed_next == SOURCE_NEXT_BATCH_SHA256,
            "next_stream_reproduced": observed_stream == SOURCE_NEXT_STREAM_SHA256,
            "source_optimizer_state_preserved": len(optimizer.state)
            == len(payload["optimizer"]["state"]),
            "new_b2_optimizer_state_absent": model.g_rec_b2 not in optimizer.state,
            "weight_tying": model.base.transformer.wte.weight
            is model.base.lm_head.weight,
        }
    )
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"2D2B source strict reopen failed: {checks}")
    rng_fingerprints = {
        name: state_fingerprint(value) for name, value in payload["rng_state"].items()
    }
    if restore_rng:
        restore_rng_state(payload["rng_state"])
        restored = capture_rng_state()
        restored_fingerprints = {
            name: state_fingerprint(value) for name, value in restored.items()
        }
        if restored_fingerprints != rng_fingerprints:
            raise SystemExit("source RNG strict restore mismatch")
    else:
        restored_fingerprints = None
    audit = {
        "checkpoint": str(path),
        "checkpoint_sha256": observed_sha,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_verification": read_json(verification_sidecar),
        "checks": checks,
        "source_loader_state": copy.deepcopy(payload["loader_state"]),
        "source_optimizer": {
            "state_entries": len(payload["optimizer"]["state"]),
            "parameter_groups": len(payload["optimizer"]["param_groups"]),
            "group_lrs": [group["lr"] for group in payload["optimizer"]["param_groups"]],
            "state_tensor_count": sum(
                torch.is_tensor(value)
                for state in payload["optimizer"]["state"].values()
                for value in state.values()
            ),
            "restored_exactly_via_strict_optimizer_load_state_dict": True,
        },
        "rng_fingerprints": rng_fingerprints,
        "restored_rng_fingerprints": restored_fingerprints,
        "next_global_batch_sha256": observed_next,
        "next_global_batch_stream_sha256": observed_stream,
        "source_metadata": copy.deepcopy(payload["metadata"]),
        "source_architecture_manifest": copy.deepcopy(payload["architecture_manifest"]),
        "source_training_state": copy.deepcopy(payload["training_state"]),
        "optimizer_report": optimizer_report,
    }
    return model, optimizer, source_loader, payload, audit


def parameter_manifest(model, source_payload) -> dict:
    named = list(model.named_parameters())
    inventory = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "trainable": value.requires_grad,
        }
        for name, value in named
    ]
    source_state_inventory = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
        }
        for name, value in source_payload["model"].items()
    ]
    source_wrapper = source_core.RecurrentKVGPT(model.base)
    source_inventory = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
        }
        for name, value in source_wrapper.named_parameters()
    ]
    del source_wrapper
    comparable_without_new = [
        {key: row[key] for key in ("name", "shape", "dtype", "numel")}
        for row in inventory
        if row["name"] != "g_rec_b2"
    ]
    source_keys = list(source_payload["model"])
    target_keys_without_new = [key for key in model.state_dict() if key != "g_rec_b2"]
    new_rows = [row for row in inventory if row["name"] == "g_rec_b2"]
    report = {
        "2d2b_named_parameter_inventory": source_inventory,
        "2d2c_named_parameter_inventory": inventory,
        "source_inventory_preserved": comparable_without_new == source_inventory,
        "source_state_dict_inventory": source_state_inventory,
        "source_state_dict_keys_preserved": target_keys_without_new == source_keys,
        "2d2b_total_parameters": sum(row["numel"] for row in source_inventory),
        "2d2c_total_parameters": sum(row["numel"] for row in inventory),
        "new_parameters_vs_2d2b": new_rows,
        "new_parameter_count_vs_2d2b": len(new_rows),
        "all_trainable": all(row["trainable"] for row in inventory),
        "embedding_lm_head_tied": model.base.transformer.wte.weight
        is model.base.lm_head.weight,
    }
    report["checks"] = {
        "source_inventory_preserved": report["source_inventory_preserved"],
        "source_state_dict_keys_preserved": report["source_state_dict_keys_preserved"],
        "source_total": report["2d2b_total_parameters"] == SOURCE_PARAMETERS,
        "target_total": report["2d2c_total_parameters"] == TOTAL_PARAMETERS,
        "exactly_one_new": report["new_parameter_count_vs_2d2b"] == 1,
        "new_scalar_exact": len(new_rows) == 1
        and new_rows[0]["name"] == "g_rec_b2"
        and new_rows[0]["shape"] == []
        and new_rows[0]["numel"] == 1,
        "all_trainable": report["all_trainable"],
        "tying": report["embedding_lm_head_tied"],
    }
    report["passed"] = all(report["checks"].values())
    if not report["passed"]:
        raise SystemExit(f"parameter manifest failed: {report['checks']}")
    return report


def architecture_manifest() -> dict:
    return {
        "experiment": EXPERIMENT,
        "links": {
            "B12_to_B1": {
                "source": "B12 post-MLP residual immediately before final LayerNorm",
                "destination": "B1 attention",
                "gate": "inherited tanh(g_rec)",
            },
            "B11_to_B2": {
                "source": "B11 post-MLP residual immediately before B12",
                "destination": "B2 attention",
                "gate": "new tanh(g_rec_b2)",
            },
        },
        "b1_local_window": W_LOCAL,
        "b2_local_window": W_LOCAL,
        "b3_b12_windows": [T] * 10,
        "recurrent_source_set": "max(0,t-1023)...t-2 inclusive",
        "recurrent_min_lag": 2,
        "recurrent_max_lag": RECURRENT_MAX_LAG,
        "maximum_recurrent_entries": MAX_RECURRENT_ENTRIES,
        "local_positions": ["t-1", "t"],
        "separate_softmaxes": True,
        "shared_destination_ln_qkv": True,
        "single_destination_c_proj": True,
        "incremental_b1_historical_kv_capacity": 1,
        "incremental_b2_historical_kv_capacity": 1,
        "incremental_b3_b12_historical_kv_capacity": 1023,
        "incremental_b11_raw_residual_capacity": RECURRENT_RING_CAPACITY,
        "incremental_b12_raw_residual_capacity": RECURRENT_RING_CAPACITY,
        "parallel_source_tensor_shapes": {"h11": "[B,T,C]", "h12": "[B,T,C]"},
        "forbidden_repeated_state_tensor": "[B,T,T,C]",
        "overall_context": T,
        "overall_kv_savings_claimed": False,
        "forbidden_modules_absent": {
            "teacher": True,
            "attnres": True,
            "dedicated_recurrent_projection": True,
            "additional_link_beyond_B11_to_B2": True,
            "detached_training_arm": True,
        },
    }


def semantic_diff_audit(model, source_payload) -> dict:
    named = [name for name, _ in model.named_parameters()]
    source_wrapper = source_core.RecurrentKVGPT(model.base)
    source_named = [name for name, _ in source_wrapper.named_parameters()]
    del source_wrapper
    old = source_driver.architecture_manifest()
    new = architecture_manifest()
    unchanged = {
        "b1_source": old["source"] == new["links"]["B12_to_B1"]["source"],
        "b1_destination": old["destination"]
        == new["links"]["B12_to_B1"]["destination"],
        "b1_local_window": old["b1_local_window"] == new["b1_local_window"] == 2,
        "b1_recurrent_geometry": old["recurrent_source_set"]
        == new["recurrent_source_set"],
        "b1_gate_name_and_state": "g_rec" in named and named.index("g_rec") == source_named.index("g_rec"),
        "source_parameter_names": [name for name in named if name != "g_rec_b2"]
        == source_named,
    }
    report = {
        "baseline": "final Experiment 2D2B",
        "architecture_changes": [
            {"field": "B2 ordinary local window", "old": 1024, "new": 2},
            {"field": "new recurrent link", "old": None, "new": "B11->B2 full bank"},
            {"field": "new learnable tensor", "old": None, "new": "scalar g_rec_b2"},
        ],
        "unchanged": unchanged,
        "new_learnable_tensors": ["g_rec_b2"],
        "memory_efficient_source_storage": True,
        "source_checkpoint_state_dict_keys_preserved": [
            key for key in model.state_dict() if key != "g_rec_b2"
        ]
        == list(source_payload["model"]),
        "passed": all(unchanged.values()),
    }
    report["passed"] = bool(
        report["passed"] and report["source_checkpoint_state_dict_keys_preserved"]
    )
    if not report["passed"]:
        raise SystemExit(f"semantic diff audit failed: {unchanged}")
    return report


def validation_manifest(val_path) -> dict:
    return legacy.validation_manifest(val_path)


@torch.no_grad()
def evaluate_parallel(
    model, val_path, batches=VALIDATION_BATCHES, combined_controls=False
) -> dict:
    """Evaluate the causal B11->B2 controls on identical canonical rows."""
    model.eval()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    names = [
        "new_real",
        "b2_recurrence_off",
        "b2_shuffled",
        "b2_full_counterfactual",
    ]
    if combined_controls:
        names.extend(("both_shuffled", "b1_off_b2_real", "b1_real_b2_off"))
    controls = {
        name: {
            "loss_sum": 0.0,
            "targets": 0,
            "per_batch_losses": [],
            "per_position_sum": np.zeros(T, dtype=np.float64),
        }
        for name in names
    }
    identities = []
    derangement = torch.arange(VALIDATION_B, device=device).roll(1)
    start = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(batches):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(legacy.d0d.batch_identity(cpu_x, cpu_y))
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        for name in names:
            kwargs = {}
            if name in {"b2_recurrence_off", "b1_real_b2_off"}:
                kwargs["b2_gate_override"] = 0.0
            elif name == "b2_shuffled":
                kwargs["b2_recurrent_permutation"] = derangement
            elif name == "b2_full_counterfactual":
                kwargs["b2_full_counterfactual"] = True
            elif name == "both_shuffled":
                kwargs["b1_recurrent_permutation"] = derangement
                kwargs["b2_recurrent_permutation"] = derangement
            elif name == "b1_off_b2_real":
                kwargs["b1_gate_override"] = 0.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.forward_multi_pass(x, num_passes=2, **kwargs)
                tensor = _token_losses(output["logits"], y)
            row = controls[name]
            row["loss_sum"] += tensor.double().sum().item()
            row["targets"] += tensor.numel()
            row["per_batch_losses"].append(tensor.float().mean().item())
            row["per_position_sum"] += tensor.double().sum(dim=0).cpu().numpy()
            del output, tensor
            torch.cuda.empty_cache()
        print(f"2D2C validation batch={batch_index + 1:02d}/{batches}", flush=True)
        del x, y, cpu_x, cpu_y
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - start
    finished = {}
    for name, row in controls.items():
        finished[name] = {
            "validation_loss": row["loss_sum"] / row["targets"],
            "validation_targets": row["targets"],
            "per_batch_losses": row["per_batch_losses"],
            "per_position_loss": (
                row["per_position_sum"] / (batches * VALIDATION_B)
            ).tolist(),
        }
    real = finished["new_real"]
    off = finished["b2_recurrence_off"]
    shuffled = finished["b2_shuffled"]
    full = finished["b2_full_counterfactual"]
    position_bins = {}
    for name, first, last in POSITION_BINS:
        r = np.asarray(real["per_position_loss"])[first : last + 1]
        o = np.asarray(off["per_position_loss"])[first : last + 1]
        s = np.asarray(shuffled["per_position_loss"])[first : last + 1]
        f = np.asarray(full["per_position_loss"])[first : last + 1]
        position_bins[name] = {
            "new_real_loss": float(r.mean()),
            "b2_recurrence_off_loss": float(o.mean()),
            "b2_shuffled_loss": float(s.mean()),
            "b2_full_counterfactual_loss": float(f.mean()),
            "b2_recurrent_gain": float((o - r).mean()),
            "b2_sequence_gap": float((s - r).mean()),
            "remaining_b2_compression_gap": float((r - f).mean()),
            "available_recurrent_history": [max(1, first - 1), max(1, last - 1)],
        }
    collection_sha = legacy.d0.aggregate_hashes(
        [row["combined_sha256"] for row in identities]
    )
    result = {
        "controls": finished,
        "b2_recurrent_gain": off["validation_loss"] - real["validation_loss"],
        "b2_sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "remaining_b2_compression_gap": real["validation_loss"]
        - full["validation_loss"],
        "new_real_vs_b2_off": paired_stats(
            real["per_batch_losses"], off["per_batch_losses"]
        ),
        "new_real_vs_b2_shuffled": paired_stats(
            real["per_batch_losses"], shuffled["per_batch_losses"]
        ),
        "new_real_vs_b2_full": paired_stats(
            real["per_batch_losses"], full["per_batch_losses"]
        ),
        "position_bins": position_bins,
        "g_rec_b1_raw": model.g_rec_b1.detach().float().item(),
        "tanh_g_rec_b1": model.recurrent_scale_b1.detach().float().item(),
        "g_rec_b2_raw": model.g_rec_b2.detach().float().item(),
        "tanh_g_rec_b2": model.recurrent_scale_b2.detach().float().item(),
        "canonical_validation_sha256": collection_sha,
        "batch_identities": identities,
        "batch_count": batches,
        "batch_size": VALIDATION_B,
        "sequence_length": T,
        "precision": "torch.autocast(cuda,bfloat16)",
        "loss_denominator": batches * VALIDATION_B * T,
        "performance": {
            "wall_seconds": elapsed,
            "condition_target_passes_per_second": batches
            * VALIDATION_B
            * T
            * len(names)
            / elapsed,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }
    if combined_controls:
        result["combined_system"] = {
            "both_real_vs_both_shuffled_gap": finished["both_shuffled"][
                "validation_loss"
            ]
            - real["validation_loss"],
            "b1_marginal_gain_with_b2_real": finished["b1_off_b2_real"][
                "validation_loss"
            ]
            - real["validation_loss"],
            "b2_marginal_gain_with_b1_real": finished["b1_real_b2_off"][
                "validation_loss"
            ]
            - real["validation_loss"],
        }
    if batches == VALIDATION_BATCHES and collection_sha != CANONICAL_VALIDATION_SHA256:
        raise SystemExit(f"canonical validation hash mismatch: {collection_sha}")
    return result


def _weighted_quantile_from_histogram(histogram: torch.Tensor, quantile: float) -> float:
    total = histogram.sum()
    if not bool(total.gt(0)):
        return float("nan")
    threshold = total * float(quantile)
    index = int(torch.searchsorted(histogram.cumsum(0), threshold).item())
    return float(index)


@torch.no_grad()
def attention_diagnostics(model, val_path, link, batch_size=2) -> dict:
    """Measure one link's lag use without combining B1 and B2 diagnostics."""
    if link not in {"b1", "b2"}:
        raise ValueError(link)
    model.eval()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], batch_size, T)
    cpu_x, cpu_y = loader.next_batch()
    identity = legacy.d0d.batch_identity(cpu_x, cpu_y)
    x = cpu_x.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first = model.forward_pass(x)
        second = model.forward_pass(
            x,
            b1_recurrent_source=first["h12"],
            b2_recurrent_source=first["h11"],
            bank_mode="full",
            return_diagnostics=True,
        )
    diagnostics = second["diagnostics"][link]
    weights = diagnostics["recurrent_attention_weights"]
    if weights is None:
        raise SystemExit("pinned attention diagnostic exceeded safe weight threshold")
    valid = diagnostics["recurrent_valid_mask"]
    query = torch.arange(T, device=device).view(T, 1)
    source = torch.arange(T, device=device).view(1, T)
    lag = query - source
    total_mass = weights.double().sum().item()
    bins = {}
    for name, first_lag, last_lag in LAG_BINS:
        selected = valid & (lag >= first_lag) & (lag <= last_lag)
        mass = weights.masked_select(selected.view(1, 1, T, T)).double().sum().item()
        valid_per_sequence = int(selected.sum().item())
        valid_instances = valid_per_sequence * batch_size * N_HEAD
        bins[name] = {
            "lag_min": first_lag,
            "lag_max": last_lag,
            "attention_mass": mass / total_mass,
            "raw_attention_mass": mass,
            "normalized_mass_per_available_token": mass / max(valid_instances, 1),
            "valid_position_instances_per_sequence": valid_per_sequence,
            "valid_position_instances_all_heads_rows": valid_instances,
        }

    def one_head(current: torch.Tensor) -> dict:
        # current: [B,T,T], normalized across the final dimension per query.
        aggregated = current.double().sum(dim=0)
        histogram = torch.zeros(RECURRENT_MAX_LAG + 1, dtype=torch.float64, device=device)
        lag_values = lag[valid].long()
        histogram.scatter_add_(0, lag_values, aggregated[valid])
        total = histogram.sum()
        mean_lag = (
            (histogram * torch.arange(histogram.numel(), device=device)).sum() / total
        ).item()
        entropy = -(current.double().clamp_min(1e-300).log() * current.double()).sum(-1)
        valid_queries = valid.any(dim=-1).view(1, T).expand(batch_size, T)
        entropy_values = entropy[valid_queries]
        return {
            "mean_attended_recurrent_lag": mean_lag,
            "median_attended_recurrent_lag": _weighted_quantile_from_histogram(
                histogram, 0.5
            ),
            "p90_attended_recurrent_lag": _weighted_quantile_from_histogram(
                histogram, 0.9
            ),
            "attention_entropy": entropy_values.mean().item(),
            "effective_recurrent_positions": entropy_values.exp().mean().item(),
            "total_attention_mass": total.item(),
        }

    heads = {str(head): one_head(weights[:, head]) for head in range(N_HEAD)}
    aggregate = one_head(weights.mean(dim=1, keepdim=False))
    report = {
        "link": "B12->B1" if link == "b1" else "B11->B2",
        "pinned_batch": identity,
        "batch_size": batch_size,
        "sequence_length": T,
        "lag_bins": bins,
        "heads": heads,
        "aggregate": aggregate,
        "mass_partitions": {
            "lags_2_31": sum(bins[name]["attention_mass"] for name in ("2-7", "8-15", "16-31")),
            "lags_32_127": sum(bins[name]["attention_mass"] for name in ("32-63", "64-127")),
            "lags_128_511": sum(bins[name]["attention_mass"] for name in ("128-255", "256-511")),
            "lags_512_1023": bins["512-1023"]["attention_mass"],
        },
        "total_probability_mass": total_mass,
        "weights_finite": bool(torch.isfinite(weights).all()),
        "method": "One pinned B=2 canonical batch; SDPA output is used for logits and explicit weights only for diagnostics.",
    }
    del x, first, second, diagnostics, weights
    torch.cuda.empty_cache()
    return report


def temporal_gradient_by_lag(
    model, val_path, link, precision="bf16", b2_gate_override=None
) -> dict:
    """Group one attached writer stream's gradients by lag."""
    if link not in {"b1", "b2"}:
        raise ValueError(link)
    model.train()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], 1, T)
    cpu_x, cpu_y = loader.next_batch()
    identity = legacy.d0d.batch_identity(cpu_x, cpu_y)
    x, y = cpu_x.to(device), cpu_y.to(device)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else contextlib.nullcontext()
    )
    with torch.enable_grad(), context:
        first = model.forward_pass(x, activation_checkpointing=True)
        source = first["h12"] if link == "b1" else first["h11"]
        second = model.forward_pass(
            x,
            targets=y,
            b1_recurrent_source=first["h12"],
            b2_recurrent_source=first["h11"],
            b2_gate_override=b2_gate_override,
            activation_checkpointing=True,
            bank_mode="full",
        )
        gradient = torch.autograd.grad(second["loss"], source)[0].float()
    lags = (T - 1) - torch.arange(T, device=device)
    bins = {}
    for name, first_lag, last_lag in LAG_BINS:
        selected = (lags >= first_lag) & (lags <= last_lag)
        values = gradient[:, selected]
        per_position_rms = values.square().mean(dim=(0, 2)).sqrt()
        bins[name] = {
            "mean_gradient_rms": values.square().mean().sqrt().item(),
            "max_gradient_rms": per_position_rms.max().item(),
            "fraction_nonzero_elements": values.ne(0).float().mean().item(),
            "fraction_nonzero_positions": per_position_rms.ne(0).float().mean().item(),
            "source_positions": int(selected.sum().item()),
        }
    probes = {
        "early_old_position_0": gradient[:, 0].square().mean().sqrt().item(),
        "middle_position_512": gradient[:, 512].square().mean().sqrt().item(),
        "recent_eligible_position_1021": gradient[:, 1021].square().mean().sqrt().item(),
        "ineligible_t_minus_1_position_1022": gradient[:, 1022].square().mean().sqrt().item(),
    }
    finite = bool(torch.isfinite(gradient).all())
    report = {
        "link": "B12->B1" if link == "b1" else "B11->B2",
        "precision": precision,
        "pinned_batch": identity,
        "gate_raw": (
            model.g_rec_b1 if link == "b1" else model.g_rec_b2
        ).detach().float().item(),
        "effective_gate": (
            model.recurrent_scale_b1 if link == "b1" else model.recurrent_scale_b2
        ).detach().float().item(),
        "b2_gate_override": b2_gate_override,
        "loss": second["loss"].detach().float().item(),
        "gradient_norm": gradient.norm().item(),
        "finite": finite,
        "nonzero": bool(gradient.count_nonzero().item()),
        "bins": bins,
        "position_probes": probes,
        "long_lag_writer_gradient_present": bins["512-1023"][
            "fraction_nonzero_positions"
        ]
        > 0,
        "method": (
            f"Pass-2 CE gradient into attached Pass-1 {'B12' if link == 'b1' else 'B11'} post-MLP states. "
            "Source positions are indexed by their lag from receiver t=1023; each "
            "source gradient also includes all eligible earlier receivers."
        ),
    }
    model.zero_grad(set_to_none=True)
    del x, y, first, source, second, gradient
    torch.cuda.empty_cache()
    return report


def kernel_preflight(model, short_tokens, short_targets) -> dict:
    """Hard architecture, geometry, causality, isolation and gate tests."""
    model.eval()
    device = short_tokens.device
    length = short_tokens.size(1)
    reports = {}
    checks = {}
    bank_mask = model.recurrent_mask(length, length, device, "full")
    query = torch.arange(length, device=device).view(length, 1)
    source = torch.arange(length, device=device).view(1, length)
    expected = (source <= query - 2) & (source >= query - 1023)
    checks["short_mask_exact"] = torch.equal(bank_mask, expected)
    full_mask = model.recurrent_mask(T, T, device, "full")
    boundary_counts = {
        str(position): int(full_mask[position].sum().item())
        for position in (0, 1, 2, 3, 1022, 1023)
    }
    reports["boundary_counts"] = boundary_counts
    checks["boundary_counts"] = boundary_counts == {
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 2,
        "1022": 1021,
        "1023": 1022,
    }
    b1_local_mask = model.local_mask(length, device)
    b2_local_mask = model.b2_local_mask(length, device)
    checks["b1_local_recurrent_disjoint"] = not bool(
        (bank_mask & b1_local_mask).any()
    )
    checks["b2_local_recurrent_disjoint"] = not bool(
        (bank_mask & b2_local_mask).any()
    )
    checks["b1_b2_w2_masks_exact"] = torch.equal(b1_local_mask, b2_local_mask)
    checks["no_future_or_t_minus_1"] = not bool((bank_mask & (source >= query - 1)).any())
    checks["no_too_old"] = not bool((bank_mask & (source < query - 1023)).any())
    values = torch.randn(
        short_tokens.size(0), length, model.config.n_embd, device=device
    )
    bank = model.build_recurrent_bank(values)
    checks["source_not_repeated"] = bank.values.data_ptr() == values.data_ptr()
    checks["source_rank_three"] = bank.values.ndim == 3
    for block_index, projector, label in (
        (0, model.project_recurrent_kv, "b1"),
        (1, model.project_recurrent_kv_b2, "b2"),
    ):
        key, value = projector(values)
        normalized = model.base.transformer.h[block_index].ln_1(values)
        _, expected_key, expected_value = model.base.transformer.h[
            block_index
        ].attn.c_attn(normalized).split(model.config.n_embd, dim=-1)
        expected_key = expected_key.view(
            short_tokens.size(0), length, N_HEAD, N_EMBD // N_HEAD
        ).transpose(1, 2)
        expected_value = expected_value.view(
            short_tokens.size(0), length, N_HEAD, N_EMBD // N_HEAD
        ).transpose(1, 2)
        checks[f"{label}_shared_projection_exact"] = torch.equal(
            key, expected_key
        ) and torch.equal(value, expected_value)
    calls = {"b1": 0, "b2": 0}
    hooks = [
        model.base.transformer.h[index].attn.c_proj.register_forward_hook(
            lambda _module, _inputs, _output, name=name: calls.__setitem__(
                name, calls[name] + 1
            )
        )
        for index, name in ((0, "b1"), (1, "b2"))
    ]
    try:
        with torch.no_grad():
            active = model.forward_pass(
                short_tokens,
                targets=short_targets,
                b1_recurrent_source=values,
                b2_recurrent_source=values,
                return_diagnostics=True,
            )
    finally:
        for hook in hooks:
            hook.remove()
    checks["single_c_proj_each"] = calls == {"b1": 1, "b2": 1}
    for name in ("b1", "b2"):
        weights = active["diagnostics"][name]["recurrent_attention_weights"]
        checks[f"{name}_invalid_probabilities_zero"] = not bool(
            weights.masked_select(
                ~bank_mask.view(1, 1, length, length)
            ).count_nonzero()
        )
        checks[f"{name}_probabilities_finite"] = bool(torch.isfinite(weights).all())

    with torch.no_grad():
        first = model.forward_pass(short_tokens)
        no_b2_source = model.forward_pass(
            short_tokens,
            b1_recurrent_source=first["h12"],
            b2_gate_override=0.0,
        )["logits"]
        b2_source_zero = model.forward_pass(
            short_tokens,
            b1_recurrent_source=first["h12"],
            b2_recurrent_source=first["h11"],
            b2_gate_override=0.0,
        )["logits"]
    checks["b2_gate_zero_identity"] = torch.equal(no_b2_source, b2_source_zero)
    changed = short_tokens.clone()
    changed[:, -1] = (changed[:, -1] + 17) % model.config.vocab_size
    with torch.no_grad():
        reference = model.forward_multi_pass(short_tokens, num_passes=2)["logits"]
        perturbed = model.forward_multi_pass(changed, num_passes=2)["logits"]
    checks["future_perturbation_causal"] = torch.equal(
        reference[:, :-1], perturbed[:, :-1]
    )
    if short_tokens.size(0) > 1:
        changed_row = short_tokens.clone()
        changed_row[0] = (changed_row[0] + 29) % model.config.vocab_size
        with torch.no_grad():
            base_rows = model.forward_multi_pass(short_tokens, num_passes=2)["logits"]
            changed_rows = model.forward_multi_pass(changed_row, num_passes=2)["logits"]
        checks["row_isolation"] = torch.equal(base_rows[1:], changed_rows[1:])
    reports["mask_positions"] = {
        str(position): torch.where(full_mask[position])[0].cpu().tolist()
        for position in (0, 1, 2, 3, 1022, 1023)
    }
    reports["c_proj_calls"] = calls
    reports["bank_storage"] = {
        "source_shape": list(values.shape),
        "bank_shape": list(bank.values.shape),
        "shared_data_ptr": bank.values.data_ptr() == values.data_ptr(),
    }
    del full_mask, values, bank, key, value, active, weights, reference, perturbed
    torch.cuda.empty_cache()
    return {"checks": checks, "reports": reports, "passed": all(checks.values())}


def frozen_2d2b_regression(model, source_payload, val_path) -> dict:
    """Reopen the exact 2D2B wrapper and reproduce its frozen final controls."""
    old_model = source_core.RecurrentKVGPT(model.base)
    old_model.g_rec = model.g_rec
    old_model.to(next(model.parameters()).device)
    parallel = source_driver.evaluate_parallel(old_model, val_path)
    expected_parallel = {
        "plain": 3.104398060634412,
        "full_real": 3.09949293354166,
        "full_shuffled": 3.102238688560283,
        "two_slot_real": 3.102510876629276,
    }
    observed_parallel = {
        name: parallel["controls"][name]["validation_loss"]
        for name in expected_parallel
    }
    tolerance = 5e-7
    checks = {
        "parallel": all(
            abs(observed_parallel[name] - expected) <= tolerance
            for name, expected in expected_parallel.items()
        ),
        "parallel_canonical": parallel["canonical_validation_sha256"]
        == CANONICAL_VALIDATION_SHA256,
        "b1_gate_exact": model.g_rec.detach().float().item() == SOURCE_GATE_RAW,
    }
    report = {
        "expected_parallel": expected_parallel,
        "observed_parallel": observed_parallel,
        "parallel_deltas": {
            name: observed_parallel[name] - expected for name, expected in expected_parallel.items()
        },
        "tolerance": tolerance,
        "parallel_full": parallel,
        "checks": checks,
        "passed": all(checks.values()),
    }
    del old_model
    gc.collect()
    torch.cuda.empty_cache()
    if not report["passed"]:
        raise SystemExit(f"frozen 2D2B regression failed: {checks}")
    return report


def probe_microbatch(model, optimizer, shards, device, candidates=(32, 16, 8, 4, 2)):
    model.train()
    attempts = []
    for candidate in candidates:
        if (GLOBAL_TARGETS // T) % candidate:
            continue
        loader = legacy.d1.ExplicitShardLoader(shards, candidate, T)
        cpu_x, cpu_y = loader.next_batch()
        x, y = cpu_x.to(device), cpu_y.to(device)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        results = []
        source_h11 = None
        source_h12 = None
        loss = None
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    current = model.forward_pass(
                        x,
                        targets=y,
                        b1_recurrent_source=source_h12,
                        b2_recurrent_source=source_h11,
                        activation_checkpointing=True,
                    )
                    results.append(current)
                    source_h11 = current["h11"]
                    source_h12 = current["h12"]
                loss = sum(
                    weight * row["loss"]
                    for weight, row in zip(THREE_PASS_WEIGHTS, results)
                )
            loss.backward()
            torch.cuda.synchronize()
            row = {
                "micro_batch_sequences": candidate,
                "passed": gradients_finite(model),
                "loss": loss.detach().float().item(),
                "source_optimizer_state_resident": len(optimizer.state) > 0,
                "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
            }
            attempts.append(row)
            if row["passed"]:
                model.zero_grad(set_to_none=True)
                return candidate, attempts
        except torch.cuda.OutOfMemoryError as error:
            attempts.append(
                {
                    "micro_batch_sequences": candidate,
                    "passed": False,
                    "error": type(error).__name__,
                }
            )
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        finally:
            del results, source_h11, source_h12, loss, x, y, cpu_x, cpu_y
            gc.collect()
            torch.cuda.empty_cache()
    raise SystemExit(f"no safe 2D2C microbatch: {attempts}")


def benchmark_recurrent_attention(model, batch_size, repeats=5) -> dict:
    device = next(model.parameters()).device
    heads = model.config.n_head
    head_size = model.config.n_embd // heads
    query = torch.randn(
        batch_size, heads, T, head_size, device=device, dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    mask = model.recurrent_mask(T, T, device, "full")
    forward_times, backward_times = [], []
    for _ in range(repeats):
        for tensor in (query, key, value):
            tensor.grad = None
        torch.cuda.synchronize()
        started = time.monotonic()
        output, _ = model._masked_recurrent_attention(query, key, value, mask)
        torch.cuda.synchronize()
        forward_times.append(time.monotonic() - started)
        started = time.monotonic()
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        backward_times.append(time.monotonic() - started)
        del output
    result = {
        "batch_size": batch_size,
        "sequence_length": T,
        "heads": heads,
        "recurrent_entries_max": MAX_RECURRENT_ENTRIES,
        "repeats": repeats,
        "forward_seconds_mean": statistics.fmean(forward_times),
        "forward_seconds_median": statistics.median(forward_times),
        "backward_seconds_mean": statistics.fmean(backward_times),
        "backward_seconds_median": statistics.median(backward_times),
        "kernel": "torch.scaled_dot_product_attention boolean offset-causal mask",
    }
    del query, key, value, mask
    torch.cuda.empty_cache()
    return result


def global_batch_stream_hash(loader, accumulation) -> str:
    """Hash the logical global batch independently of microbatch boundaries."""
    clone = loader.clone()
    x_hash = hashlib.sha256()
    y_hash = hashlib.sha256()
    for _ in range(accumulation):
        x, y = clone.next_batch()
        x_hash.update(x.contiguous().numpy().tobytes())
        y_hash.update(y.contiguous().numpy().tobytes())
    return hashlib.sha256((x_hash.hexdigest() + y_hash.hexdigest()).encode()).hexdigest()


def loader_at_source_cursor(source_state: dict, micro_batch: int):
    state = copy.deepcopy(source_state)
    state["batch_size"] = int(micro_batch)
    return legacy.d1.ExplicitShardLoader(
        state["shards"], int(micro_batch), T, state=state
    )


def run_preflight(args):
    require_git(clean=True)
    config = require_config()
    fingerprint = implementation_fingerprint()
    mount = workspace_mount_audit(
        args.output_dir, args.run_root, args.persistent_volume_identity
    )
    stop = authenticated_stop_audit(args)
    device = require_single_a100()
    seed_all()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"preflight output already exists and is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model, optimizer, source_loader, source_payload, source_audit = load_source_bundle(
        args.source_checkpoint, device, restore_rng=False
    )
    val_path = validation_path(args.data_root)
    correction = source_correction_provenance()
    parameters = parameter_manifest(model, source_payload)
    architecture = architecture_manifest()
    semantic = semantic_diff_audit(model, source_payload)
    manifest = validation_manifest(val_path)
    if manifest["canonical_batch_collection_sha256"] != CANONICAL_VALIDATION_SHA256:
        raise SystemExit("canonical validation collection mismatch")
    source_manifest = {
        **source_audit,
        "frozen_2d2b_tag": FROZEN_TAG,
        "frozen_2d2b_commit": FROZEN_COMMIT,
        "frozen_tag_exact": git_output("rev-parse", FROZEN_TAG + "^{commit}")
        == FROZEN_COMMIT,
        "audit_correction_provenance": correction,
        "canonical_validation_sha256": CANONICAL_VALIDATION_SHA256,
        "validation_shard": str(val_path),
        "validation_shard_sha256": file_sha256(val_path),
    }
    durable_json(output / "source_manifest.json", source_manifest)
    durable_json(output / "parameter_manifest.json", parameters)
    durable_json(output / "architecture_manifest.json", architecture)
    durable_json(output / "semantic_diff_audit.json", semantic)

    short_loader = legacy.d1.ExplicitShardLoader([val_path], 2, 16)
    short_x, short_y = short_loader.next_batch()
    kernel = kernel_preflight(model, short_x.to(device), short_y.to(device))
    if not kernel["passed"]:
        raise SystemExit(f"full-bank kernel preflight failed: {kernel['checks']}")
    temporal_b1 = temporal_gradient_by_lag(model, val_path, "b1", precision="bf16")
    temporal_b2_probe = temporal_gradient_by_lag(
        model, val_path, "b2", precision="bf16", b2_gate_override=0.05
    )
    temporal_checks = {
        "b1_finite_nonzero": temporal_b1["finite"] and temporal_b1["nonzero"],
        "b1_early_old": temporal_b1["position_probes"]["early_old_position_0"] > 0,
        "b1_middle": temporal_b1["position_probes"]["middle_position_512"] > 0,
        "b1_recent": temporal_b1["position_probes"][
            "recent_eligible_position_1021"
        ]
        > 0,
        "b1_long_lag": temporal_b1["long_lag_writer_gradient_present"],
        "b2_probe_finite_nonzero": temporal_b2_probe["finite"]
        and temporal_b2_probe["nonzero"],
        "b2_probe_128_plus": temporal_b2_probe["bins"]["128-255"][
            "fraction_nonzero_positions"
        ]
        > 0,
        "b2_probe_256_plus": temporal_b2_probe["bins"]["256-511"][
            "fraction_nonzero_positions"
        ]
        > 0,
        "b2_probe_512_plus": temporal_b2_probe["long_lag_writer_gradient_present"],
    }
    if not all(temporal_checks.values()):
        raise SystemExit(f"long-bank temporal gradient failed: {temporal_checks}")

    regressions = frozen_2d2b_regression(model, source_payload, val_path)
    zero_shot = evaluate_parallel(model, val_path)
    zero_shot.update({"local_update": 0, "additional_targets": 0})
    source_loss = regressions["observed_parallel"]["full_real"]
    off_loss = zero_shot["controls"]["b2_recurrence_off"]["validation_loss"]
    real_loss = zero_shot["controls"]["new_real"]["validation_loss"]
    shuffled_loss = zero_shot["controls"]["b2_shuffled"]["validation_loss"]
    initial_shortening = {
        "source_2d2b_loss": source_loss,
        "b2_w2_no_new_recurrence_loss": off_loss,
        "b2_w2_real_loss": real_loss,
        "b2_w2_shuffled_loss": shuffled_loss,
        "damage_b2_initial": off_loss - source_loss,
        "gate_zero_real_identity": real_loss == off_loss,
        "gate_zero_shuffled_identity": shuffled_loss == off_loss,
        "source_control": "2D2B FullReal: B1 W2+full B12 recurrence; B2-B12 W1024",
        "passed": real_loss == off_loss and shuffled_loss == off_loss,
    }
    if not initial_shortening["passed"]:
        raise SystemExit(f"initial B2 gate-zero identity failed: {initial_shortening}")
    attention_b1_zero = attention_diagnostics(model, val_path, "b1")
    attention_b2_zero = attention_diagnostics(model, val_path, "b2")
    selected_microbatch, probe = probe_microbatch(
        model,
        optimizer,
        source_payload["loader_state"]["shards"],
        device,
    )
    accumulation = GLOBAL_TARGETS // (selected_microbatch * T)
    scientific_loader = loader_at_source_cursor(
        source_payload["loader_state"], selected_microbatch
    )
    source_stream_hash = global_batch_stream_hash(
        source_loader, int(source_payload["metadata"]["gradient_accumulation"])
    )
    scientific_stream_hash = global_batch_stream_hash(scientific_loader, accumulation)
    batch_payload = {
        **manifest,
        "source_loader_state": copy.deepcopy(source_payload["loader_state"]),
        "scientific_loader_initial_state": scientific_loader.state_dict(),
        "selected_micro_batch_sequences": selected_microbatch,
        "selected_gradient_accumulation": accumulation,
        "global_targets_per_update": GLOBAL_TARGETS,
        "source_next_global_batch_sha256": SOURCE_NEXT_BATCH_SHA256,
        "source_next_hash_reproduced": source_audit["next_global_batch_sha256"]
        == SOURCE_NEXT_BATCH_SHA256,
        "source_global_stream_sha256": source_stream_hash,
        "scientific_global_stream_sha256": scientific_stream_hash,
        "logical_global_batch_exact_across_microbatch_geometry": source_stream_hash
        == scientific_stream_hash,
        "microbatch_probe": probe,
    }
    if not batch_payload["logical_global_batch_exact_across_microbatch_geometry"]:
        raise SystemExit("scientific first global batch differs from 2D2B continuation")
    recurrent_benchmark = benchmark_recurrent_attention(
        model, min(selected_microbatch, 4), repeats=3
    )
    benchmark = {
        "recurrent_attention": recurrent_benchmark,
        "source_2d2b_training_targets_per_second": 0,
        "microbatch_probe": probe,
    }

    durable_json(output / "batch_manifest.json", batch_payload)
    durable_json(output / "initial_b2_shortening.json", initial_shortening)
    durable_json(output / "milestone_validation.json", {"0": zero_shot})
    durable_json(output / "b1_attention_lag_bins.json", {"0": attention_b1_zero["lag_bins"]})
    durable_json(output / "b2_attention_lag_bins.json", {"0": attention_b2_zero["lag_bins"]})
    durable_json(
        output / "b1_attention_head_distance.json",
        {"0": {"heads": attention_b1_zero["heads"], "aggregate": attention_b1_zero["aggregate"], "mass_partitions": attention_b1_zero["mass_partitions"]}},
    )
    durable_json(
        output / "b2_attention_head_distance.json",
        {"0": {"heads": attention_b2_zero["heads"], "aggregate": attention_b2_zero["aggregate"], "mass_partitions": attention_b2_zero["mass_partitions"]}},
    )
    temporal_b1.update({"local_update": 0, "additional_targets": 0})
    temporal_b2_probe.update({"local_update": 0, "additional_targets": 0, "probe_only": True})
    durable_json(output / "b12_to_b1_temporal_gradient.json", {"0": temporal_b1})
    durable_json(output / "b11_to_b2_temporal_gradient.json", {"0_probe": temporal_b2_probe})
    durable_json(output / "checkpoint_manifest.json", {"scientific": {}, "recovery": {}, "smoke": {}})
    durable_json(output / "performance.json", {"preflight_benchmark": benchmark})
    durable_json(output / "runpod_stop_capability.json", stop)
    durable_json(output / "persistent_workspace_audit.json", mount)

    science_checks = {
        "frozen_2d2b_tag_exact": source_manifest["frozen_tag_exact"],
        "source_checkpoint_exact": source_audit["checks"]["passed"],
        "audit_correction_preserved": correction["passed"],
        "parameters_exactly_one_new": parameters["passed"],
        "semantic_diff_exact": semantic["passed"],
        "kernel": kernel["passed"],
        "temporal_writer_gradient": all(temporal_checks.values()),
        "frozen_2d2b_parallel_regression": regressions["checks"]["parallel"],
        "initial_b2_shortening": initial_shortening["passed"],
        "canonical_validation": zero_shot["canonical_validation_sha256"]
        == CANONICAL_VALIDATION_SHA256,
        "loader_continuation": batch_payload[
            "logical_global_batch_exact_across_microbatch_geometry"
        ],
        "global_batch": selected_microbatch * T * accumulation == GLOBAL_TARGETS,
        "persistent_workspace": mount["passed"],
        "authenticated_stop": stop["driver_passed"],
    }
    preflight = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "timestamp": time.time(),
        "command": " ".join(sys.argv),
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "implementation_fingerprint": fingerprint,
        "environment": environment_payload(),
        "source": source_manifest,
        "parameters": parameters,
        "architecture": architecture,
        "semantic_diff": semantic,
        "kernel_preflight": kernel,
        "temporal_gradient_preflight": {"b1": temporal_b1, "b2_probe": temporal_b2_probe},
        "temporal_gradient_checks": temporal_checks,
        "frozen_2d2b_regression": regressions,
        "zero_shot": zero_shot,
        "initial_b2_shortening": initial_shortening,
        "attention_zero_shot": {"b1": attention_b1_zero, "b2": attention_b2_zero},
        "microbatch_probe": probe,
        "selected_microbatch": selected_microbatch,
        "gradient_accumulation": accumulation,
        "performance_benchmark": benchmark,
        "runpod_stop_audit": stop,
        "persistent_workspace_audit": mount,
        "checks": science_checks,
        "science_passed": all(science_checks.values()),
        "result_run_authorized": all(science_checks.values()),
        "wall_seconds": time.monotonic() - started,
    }
    durable_json(output / "preflight_audit.json", preflight)
    if not preflight["science_passed"]:
        raise SystemExit(f"2D2C preflight failed: {science_checks}")
    print("EXPERIMENT_2D2C_PREFLIGHT_PASS", flush=True)
    return preflight


def checkpoint_payload(model, optimizer, loader, training_state, metadata, accumulation):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "g_rec": model.g_rec.detach().cpu().clone(),
        "g_rec_b2": model.g_rec_b2.detach().cpu().clone(),
        "optimizer": optimizer.state_dict(),
        "completed_2d2c_updates": training_state["completed_2d2c_updates"],
        "processed_2d2c_targets": training_state["processed_2d2c_targets"],
        "cumulative_2d2_targets": training_state["cumulative_2d2_targets"],
        "source_2d2b_updates": SOURCE_UPDATES,
        "source_2d2b_targets": SOURCE_TARGETS,
        "training_state": copy.deepcopy(training_state),
        "loader_state": loader.state_dict(),
        "rng_state": capture_rng_state(),
        "next_global_batch_sha256": next_global_batch_hash(loader, accumulation),
        "next_global_batch_stream_sha256": global_batch_stream_hash(loader, accumulation),
        "architecture_manifest": architecture_manifest(),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "metadata": copy.deepcopy(metadata),
        "git_commit": git_output("rev-parse", "HEAD"),
        "saved_process_id": os.getpid(),
        "environment": environment_payload(),
    }


def strict_reopen_checkpoint(
    path, model, optimizer, loader, training_state, accumulation, expected_metadata
):
    path = Path(path)
    reopened = legacy.d0.torch_load(path, mmap=True)
    required = {
        "schema",
        "model",
        "g_rec",
        "g_rec_b2",
        "optimizer",
        "completed_2d2c_updates",
        "processed_2d2c_targets",
        "cumulative_2d2_targets",
        "source_2d2b_updates",
        "source_2d2b_targets",
        "training_state",
        "loader_state",
        "rng_state",
        "next_global_batch_sha256",
        "next_global_batch_stream_sha256",
        "architecture_manifest",
        "source_checkpoint_sha256",
        "metadata",
        "git_commit",
        "saved_process_id",
        "environment",
    }
    checks = {
        "fields_exact": set(reopened) == required,
        "schema": reopened.get("schema") == CHECKPOINT_SCHEMA,
        "updates": reopened.get("completed_2d2c_updates")
        == training_state["completed_2d2c_updates"],
        "additional_targets": reopened.get("processed_2d2c_targets")
        == training_state["processed_2d2c_targets"],
        "cumulative_targets": reopened.get("cumulative_2d2_targets")
        == training_state["cumulative_2d2_targets"],
        "training_state": reopened.get("training_state") == training_state,
        "loader_state": reopened.get("loader_state") == loader.state_dict(),
        "next_batch": reopened.get("next_global_batch_sha256")
        == next_global_batch_hash(loader, accumulation),
        "next_stream": reopened.get("next_global_batch_stream_sha256")
        == global_batch_stream_hash(loader, accumulation),
        "metadata": reopened.get("metadata") == expected_metadata,
        "architecture": reopened.get("architecture_manifest")
        == architecture_manifest(),
        "source": reopened.get("source_checkpoint_sha256") == SOURCE_SHA256,
        "model_keys": reopened.get("model", {}).keys() == model.state_dict().keys(),
        "gate_duplicate": torch.equal(reopened.get("g_rec"), model.g_rec.detach().cpu()),
        "b2_gate_duplicate": torch.equal(
            reopened.get("g_rec_b2"), model.g_rec_b2.detach().cpu()
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"checkpoint strict metadata reopen failed: {checks}")
    model.load_state_dict(reopened["model"], strict=True)
    optimizer.load_state_dict(reopened["optimizer"])
    checks.update(
        {
            "model_finite": model_finite(model),
            "optimizer_finite": optimizer_finite(optimizer),
            "weight_tying": model.base.transformer.wte.weight
            is model.base.lm_head.weight,
        }
    )
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"checkpoint tensor reopen failed: {checks}")
    verification = {
        "checkpoint": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "completed_2d2c_updates": reopened["completed_2d2c_updates"],
        "cumulative_2d2_targets": reopened["cumulative_2d2_targets"],
        "next_global_batch_sha256": reopened["next_global_batch_sha256"],
        "next_global_batch_stream_sha256": reopened[
            "next_global_batch_stream_sha256"
        ],
        "strict_reopen": checks,
        "passed": True,
    }
    del reopened
    gc.collect()
    return verification


def save_checkpoint(path, model, optimizer, loader, state, metadata, accumulation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    payload = checkpoint_payload(model, optimizer, loader, state, metadata, accumulation)
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    legacy.d0.fsync_directory(path.parent)
    del payload
    verification = strict_reopen_checkpoint(
        path, model, optimizer, loader, state, accumulation, metadata
    )
    durable_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{verification['sha256']}  {path.name}\n",
    )
    durable_json(path.with_suffix(path.suffix + ".verification.json"), verification)
    return verification


def record_checkpoint(output, update, verification, kind="scientific"):
    path = Path(output) / "checkpoint_manifest.json"
    manifest = read_json(path)
    if str(update) in manifest.setdefault(kind, {}):
        raise SystemExit(f"refusing to overwrite {kind} checkpoint manifest {update}")
    manifest[kind][str(update)] = verification
    durable_json(path, manifest)


def run_smoke(args):
    require_git(clean=False)
    require_config()
    workspace_mount_audit(args.output_dir, args.run_root, args.persistent_volume_identity)
    authenticated_stop_audit(args)
    device = require_single_a100()
    output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json")
    if not preflight.get("result_run_authorized"):
        raise SystemExit("smoke requires passing preflight")
    require_implementation_fingerprint(preflight)
    model, optimizer, _, source_payload, _ = load_source_bundle(
        args.source_checkpoint, device, restore_rng=True
    )
    smoke_batch = 2
    loader = loader_at_source_cursor(source_payload["loader_state"], smoke_batch)
    rows = []
    for update in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        cpu_x, cpu_y = loader.next_batch()
        x, y = cpu_x.to(device), cpu_y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            first = model.forward_pass(x, targets=y, activation_checkpointing=True)
            second = model.forward_pass(
                x,
                targets=y,
                b1_recurrent_source=first["h12"],
                b2_recurrent_source=first["h11"],
                activation_checkpointing=True,
                return_diagnostics=True,
            )
            loss = TWO_PASS_WEIGHTS[0] * first["loss"] + TWO_PASS_WEIGHTS[1] * second["loss"]
        b1_gate_before = model.g_rec.detach().float().item()
        b2_gate_before = model.g_rec_b2.detach().float().item()
        loss.backward()
        b1_gate_gradient = model.g_rec.grad.detach().float().item()
        b2_gate_gradient = model.g_rec_b2.grad.detach().float().item()
        groups = gradient_group_report(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        rows.append(
            {
                "update": update,
                "precision": "bf16 autocast",
                "loss": loss.detach().float().item(),
                "pass_losses": [
                    first["loss"].detach().float().item(),
                    second["loss"].detach().float().item(),
                ],
                "g_rec_b1_before": b1_gate_before,
                "g_rec_b1_after": model.g_rec.detach().float().item(),
                "tanh_g_rec_b1_after": model.recurrent_scale_b1.detach().float().item(),
                "g_rec_b1_gradient": b1_gate_gradient,
                "g_rec_b2_before": b2_gate_before,
                "g_rec_b2_after": model.g_rec_b2.detach().float().item(),
                "tanh_g_rec_b2_after": model.recurrent_scale_b2.detach().float().item(),
                "g_rec_b2_gradient": b2_gate_gradient,
                "gradient_norm": norm.detach().float().item(),
                "gradient_groups": groups,
                "gradients_finite": gradients_finite(model),
                "parameters_finite": model_finite(model),
                "optimizer_finite": optimizer_finite(optimizer),
                "recurrent_attention_finite": (
                    all(
                        second["diagnostics"][name]["recurrent_attention_weights"] is None
                        or bool(
                            torch.isfinite(
                                second["diagnostics"][name]["recurrent_attention_weights"]
                            ).all()
                        )
                        for name in ("b1", "b2")
                    )
                ),
            }
        )
        del x, y, cpu_x, cpu_y, first, second, loss
        torch.cuda.empty_cache()
    temporal_b1 = temporal_gradient_by_lag(
        model, validation_path(args.data_root), "b1"
    )
    temporal_b2 = temporal_gradient_by_lag(
        model, validation_path(args.data_root), "b2"
    )
    with torch.no_grad():
        cache_x, _ = loader.clone().next_batch()
        cache = model.incremental_logits(
            cache_x[:, :16].to(device), control="both_real", bank_mode="full"
        )["cache_audit"]
    state = {
        "completed_2d2c_updates": 3,
        "processed_2d2c_targets": 3 * smoke_batch * T,
        "cumulative_2d2_targets": SOURCE_TARGETS + 3 * smoke_batch * T,
        "started_at": time.time(),
        "last_metrics": rows[-1],
        "kind": "disposable_smoke",
    }
    metadata = {
        "experiment": EXPERIMENT,
        "kind": "disposable_smoke",
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "implementation_fingerprint": preflight["implementation_fingerprint"],
        "source_checkpoint_sha256": SOURCE_SHA256,
        "micro_batch_sequences": smoke_batch,
        "gradient_accumulation": 1,
        "pod_id": args.pod_id,
    }
    path = Path(args.run_root).resolve() / "smoke" / "smoke_update_0003.pt"
    verification = save_checkpoint(path, model, optimizer, loader, state, metadata, 1)
    record_checkpoint(output, 3, verification, kind="smoke")
    checks = {
        "three_updates": len(rows) == 3 and rows[-1]["update"] == 3,
        "finite_losses": all(math.isfinite(row["loss"]) for row in rows),
        "b1_gate_gradient": all(
            math.isfinite(row["g_rec_b1_gradient"])
            and row["g_rec_b1_gradient"] != 0
            for row in rows
        ),
        "b2_gate_gradient": all(
            math.isfinite(row["g_rec_b2_gradient"])
            and row["g_rec_b2_gradient"] != 0
            for row in rows
        ),
        "b2_gate_moved": rows[-1]["g_rec_b2_after"] != 0.0,
        "base_gradient": all(
            row["gradient_groups"]["base"]["finite"]
            and row["gradient_groups"]["base"]["nonzero"]
            for row in rows
        ),
        "b1_long_lag_writer_gradient": temporal_b1["finite"]
        and temporal_b1["long_lag_writer_gradient_present"],
        "b2_long_lag_writer_gradient": temporal_b2["finite"]
        and temporal_b2["long_lag_writer_gradient_present"],
        "attention_finite": all(row["recurrent_attention_finite"] for row in rows),
        "cache": cache["passed"]
        and cache["b1_historical_kv"] == 1
        and cache["b2_historical_kv"] == 1,
        "checkpoint": verification["passed"],
        "finite_state": all(
            row["parameters_finite"] and row["optimizer_finite"] for row in rows
        ),
        "scientific_source_discarded": True,
    }
    audit = {
        "experiment": EXPERIMENT,
        "kind": "exactly three disposable optimizer updates",
        "command": " ".join(sys.argv),
        "rows": rows,
        "long_lag_temporal_gradient": {"b1": temporal_b1, "b2": temporal_b2},
        "incremental_cache_audit": cache,
        "checkpoint": verification,
        "checks": checks,
        "passed": all(checks.values()),
        "disposition": "Discarded; scientific local update 1 reloads the immutable 2D2B update-191 checkpoint.",
    }
    durable_json(output / "smoke_audit.json", audit)
    if not audit["passed"]:
        raise SystemExit(f"2D2C smoke failed: {checks}")
    print("EXPERIMENT_2D2C_SMOKE_PASS", flush=True)
    return audit


def training_metadata(args, preflight, micro_batch, accumulation) -> dict:
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "implementation_fingerprint": preflight["implementation_fingerprint"],
        "frozen_2d2b_commit": FROZEN_COMMIT,
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_completed_updates": SOURCE_UPDATES,
        "source_completed_targets": SOURCE_TARGETS,
        "data_root": str(Path(args.data_root).resolve()),
        "canonical_validation_sha256": CANONICAL_VALIDATION_SHA256,
        "micro_batch_sequences": micro_batch,
        "gradient_accumulation": accumulation,
        "global_targets_per_update": GLOBAL_TARGETS,
        "pass_cadence": "three passes when local update is divisible by 32; two otherwise",
        "optimizer": {
            "restored_from_source": True,
            "warmup_restarted": False,
            "base_lr": BASE_LR,
            "gate_lr": GATE_LR,
            "new_b2_gate_lr": GATE_LR,
            "gradient_clip": GRAD_CLIP,
        },
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "gpu_type": args.gpu_type,
        "persistent_volume_identity": args.persistent_volume_identity,
        "runpod_stop_audit_sha256": preflight["runpod_stop_audit"]["audit_sha256"],
        "stop_mechanism": args.stop_mechanism,
        "stop_authenticated": bool(args.stop_authenticated),
    }


def load_checkpoint_runtime(
    path, model, optimizer, micro_batch, accumulation, expected_metadata
):
    path = Path(path).resolve()
    sha_path = path.with_suffix(path.suffix + ".sha256")
    verification_path = path.with_suffix(path.suffix + ".verification.json")
    if not sha_path.is_file() or not verification_path.is_file():
        raise SystemExit("resume checkpoint sidecars missing")
    expected_sha = sha_path.read_text().split()[0]
    if file_sha256(path) != expected_sha:
        raise SystemExit("resume checkpoint SHA mismatch")
    payload = legacy.d0.torch_load(path, mmap=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("resume schema mismatch")
    if payload.get("metadata") != expected_metadata:
        raise SystemExit("resume metadata mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    loader = legacy.d1.ExplicitShardLoader(
        payload["loader_state"]["shards"],
        micro_batch,
        T,
        state=payload["loader_state"],
    )
    if payload["next_global_batch_sha256"] != next_global_batch_hash(
        loader, accumulation
    ):
        raise SystemExit("resume next-global-batch hash mismatch")
    if payload["next_global_batch_stream_sha256"] != global_batch_stream_hash(
        loader, accumulation
    ):
        raise SystemExit("resume logical global stream mismatch")
    restore_rng_state(payload["rng_state"])
    state = copy.deepcopy(payload["training_state"])
    saved_pid = int(payload["saved_process_id"])
    if not model_finite(model) or not optimizer_finite(optimizer):
        raise SystemExit("resume restored nonfinite state")
    del payload
    gc.collect()
    return loader, state, saved_pid, expected_sha


def initialize_runtime(args):
    workspace_mount_audit(args.output_dir, args.run_root, args.persistent_volume_identity)
    authenticated_stop_audit(args)
    device = require_single_a100()
    output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json")
    smoke = read_json(output / "smoke_audit.json")
    if not preflight.get("result_run_authorized") or not smoke.get("passed"):
        raise SystemExit("scientific training requires passing preflight and smoke")
    require_implementation_fingerprint(preflight)
    if not preflight["runpod_stop_audit"]["driver_passed"]:
        raise SystemExit("authenticated RunPod STOP unavailable")
    batch_manifest = read_json(output / "batch_manifest.json")
    micro_batch = int(batch_manifest["selected_micro_batch_sequences"])
    accumulation = int(batch_manifest["selected_gradient_accumulation"])
    if micro_batch * T * accumulation != GLOBAL_TARGETS:
        raise SystemExit("global target geometry mismatch")
    model, optimizer, source_loader, source_payload, source_audit = load_source_bundle(
        args.source_checkpoint, device, restore_rng=not bool(args.resume)
    )
    parameters = parameter_manifest(model, source_payload)
    metadata = training_metadata(args, preflight, micro_batch, accumulation)
    if args.resume:
        loader, state, saved_pid, checkpoint_sha = load_checkpoint_runtime(
            args.resume,
            model,
            optimizer,
            micro_batch,
            accumulation,
            metadata,
        )
        if state["completed_2d2c_updates"] != FORCED_RESTART_UPDATE:
            raise SystemExit("automatic continuation resume is authorized only at update 96")
        verification = read_json(
            Path(args.resume).with_suffix(Path(args.resume).suffix + ".verification.json")
        )
        restart = {
            "checkpoint": str(Path(args.resume).resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "saved_process_id": saved_pid,
            "restored_process_id": os.getpid(),
            "fresh_process": saved_pid != os.getpid(),
            "completed_2d2c_updates": state["completed_2d2c_updates"],
            "cumulative_2d2_targets": state["cumulative_2d2_targets"],
            "next_global_batch_sha256": next_global_batch_hash(loader, accumulation),
            "expected_next_global_batch_sha256": verification[
                "next_global_batch_sha256"
            ],
            "next_global_batch_stream_sha256": global_batch_stream_hash(
                loader, accumulation
            ),
            "expected_next_global_batch_stream_sha256": verification[
                "next_global_batch_stream_sha256"
            ],
        }
        restart["passed"] = (
            restart["fresh_process"]
            and restart["next_global_batch_sha256"]
            == restart["expected_next_global_batch_sha256"]
            and restart["next_global_batch_stream_sha256"]
            == restart["expected_next_global_batch_stream_sha256"]
        )
        durable_json(output / "forced_restart_update_96.json", restart)
        if not restart["passed"]:
            raise SystemExit(f"forced restart audit failed: {restart}")
    else:
        loader = loader_at_source_cursor(source_payload["loader_state"], micro_batch)
        source_stream = global_batch_stream_hash(
            source_loader, int(source_payload["metadata"]["gradient_accumulation"])
        )
        scientific_stream = global_batch_stream_hash(loader, accumulation)
        if source_stream != scientific_stream:
            raise SystemExit("first scientific global batch is not exact continuation")
        state = {
            "completed_2d2c_updates": 0,
            "processed_2d2c_targets": 0,
            "cumulative_2d2_targets": SOURCE_TARGETS,
            "started_at": time.time(),
            "segment_start": 0,
            "last_checkpoint": None,
            "last_metrics": None,
            "source_next_global_batch_sha256": source_audit[
                "next_global_batch_sha256"
            ],
            "first_global_batch_stream_sha256": scientific_stream,
        }
        saved_pid = None
    if state["completed_2d2c_updates"] == 0:
        if model.g_rec.detach().float().item() != SOURCE_GATE_RAW:
            raise SystemExit("scientific source gate was not restored exactly")
        if [group["lr"] for group in optimizer.param_groups] != [
            BASE_LR,
            BASE_LR,
            GATE_LR,
            GATE_LR,
        ]:
            raise SystemExit("source optimizer learning rates changed")
    return SimpleNamespace(
        device=device,
        output=output,
        preflight=preflight,
        smoke=smoke,
        micro_batch=micro_batch,
        accumulation=accumulation,
        model=model,
        optimizer=optimizer,
        parameter_manifest=parameters,
        metadata=metadata,
        loader=loader,
        training_state=state,
        end_update=int(args.end_update),
        run_root=str(Path(args.run_root).resolve()),
    )


def current_lrs(optimizer) -> dict:
    return {group["name"]: float(group["lr"]) for group in optimizer.param_groups}


def train_one_update(runtime, update):
    model = runtime.model
    model.train()
    optimizer = runtime.optimizer
    device = runtime.device
    count = pass_count(update)
    weights = pass_weights(update)
    lrs = current_lrs(optimizer)
    expected_lrs = {
        "base_decay": BASE_LR,
        "base_nodecay": BASE_LR,
        "gate": GATE_LR,
        "b2_gate": GATE_LR,
    }
    if lrs != expected_lrs:
        raise SystemExit(f"resumed optimizer LR drift: {lrs}")
    optimizer.zero_grad(set_to_none=True)
    pass_loss_sums = [0.0] * count
    forward_seconds = [0.0] * count
    backward_seconds = 0.0
    total_ce = 0.0
    final_h11_rms = None
    final_h12_rms = None
    final_b1_recurrent_rms = None
    final_b2_recurrent_rms = None
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for micro_index in range(runtime.accumulation):
        cpu_x, cpu_y = runtime.loader.next_batch()
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        results = []
        source_h11 = None
        source_h12 = None
        for pass_index in range(count):
            torch.cuda.synchronize()
            pass_start = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current = model.forward_pass(
                    x,
                    targets=y,
                    b1_recurrent_source=source_h12,
                    b2_recurrent_source=source_h11,
                    activation_checkpointing=True,
                    return_diagnostics=(
                        micro_index == runtime.accumulation - 1
                        and pass_index == count - 1
                    ),
                    bank_mode="full",
                )
            torch.cuda.synchronize()
            forward_seconds[pass_index] += time.monotonic() - pass_start
            results.append(current)
            source_h11 = current["h11"]
            source_h12 = current["h12"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            weighted = sum(weight * row["loss"] for weight, row in zip(weights, results))
            scaled = weighted / runtime.accumulation
        if not math.isfinite(weighted.detach().float().item()):
            raise SystemExit("nonfinite weighted training loss")
        torch.cuda.synchronize()
        backward_start = time.monotonic()
        scaled.backward()
        torch.cuda.synchronize()
        backward_seconds += time.monotonic() - backward_start
        for index, row in enumerate(results):
            pass_loss_sums[index] += row["loss"].detach().float().item()
        total_ce += weighted.detach().float().item()
        if micro_index == runtime.accumulation - 1:
            final_h11_rms = results[-1]["h11"].detach().float().square().mean().sqrt().item()
            final_h12_rms = results[-1]["h12"].detach().float().square().mean().sqrt().item()
            final_b1_recurrent_rms = results[-1]["diagnostics"]["b1"][
                "recurrent_output_rms"
            ].detach().float().item()
            final_b2_recurrent_rms = results[-1]["diagnostics"]["b2"][
                "recurrent_output_rms"
            ].detach().float().item()
        del x, y, cpu_x, cpu_y, results, source_h11, source_h12, weighted, scaled, current
    if not gradients_finite(model):
        raise SystemExit("nonfinite gradients")
    groups = gradient_group_report(model)
    if not groups["base"]["nonzero"] or not groups["gate"]["nonzero"]:
        raise SystemExit(f"required gradient group is zero: {groups}")
    b1_gate_gradient = model.g_rec.grad.detach().float().item()
    b2_gate_gradient = model.g_rec_b2.grad.detach().float().item()
    if not math.isfinite(b2_gate_gradient) or b2_gate_gradient == 0:
        raise SystemExit("new B2 gate gradient is not finite/nonzero")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad_norm):
        raise SystemExit("nonfinite gradient norm")
    optimizer.step()
    if not model_finite(model) or not optimizer_finite(optimizer):
        raise SystemExit("nonfinite parameter/optimizer state")
    elapsed = time.monotonic() - start
    runtime.training_state["completed_2d2c_updates"] = update
    runtime.training_state["processed_2d2c_targets"] = update * GLOBAL_TARGETS
    runtime.training_state["cumulative_2d2_targets"] = SOURCE_TARGETS + update * GLOBAL_TARGETS
    metrics = {
        "timestamp": time.time(),
        "local_update": update,
        "source_plus_local_update": SOURCE_UPDATES + update,
        "additional_targets": update * GLOBAL_TARGETS,
        "cumulative_2d2_targets": SOURCE_TARGETS + update * GLOBAL_TARGETS,
        "pass_count": count,
        "pass_weights": list(weights),
        "pass_losses": [value / runtime.accumulation for value in pass_loss_sums],
        "weighted_total_ce": total_ce / runtime.accumulation,
        "lrs": lrs,
        "g_rec_b1_raw": model.g_rec.detach().float().item(),
        "tanh_g_rec_b1": model.recurrent_scale_b1.detach().float().item(),
        "g_rec_b2_raw": model.g_rec_b2.detach().float().item(),
        "tanh_g_rec_b2": model.recurrent_scale_b2.detach().float().item(),
        "g_rec_b1_gradient_preclip": b1_gate_gradient,
        "g_rec_b2_gradient_preclip": b2_gate_gradient,
        "gradient_norm_before_clip": grad_norm.detach().float().item(),
        "gradient_groups": groups,
        "b11_memory_rms": final_h11_rms,
        "b12_memory_rms": final_h12_rms,
        "b1_recurrent_output_rms": final_b1_recurrent_rms,
        "b2_recurrent_output_rms": final_b2_recurrent_rms,
        "pass_forward_seconds": forward_seconds,
        "aggregate_backward_seconds": backward_seconds,
        "wall_seconds": elapsed,
        "targets_per_second": GLOBAL_TARGETS / elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "all_gradients_finite": True,
        "all_parameters_finite": True,
        "all_optimizer_moments_finite": True,
        "recurrent_source_detached": False,
    }
    runtime.training_state["last_metrics"] = metrics
    return metrics


def write_heartbeat(runtime, metrics):
    completed = metrics["local_update"]
    elapsed = time.time() - runtime.training_state["started_at"]
    segment_completed = max(completed - runtime.training_state.get("segment_start", 0), 1)
    eta = elapsed / segment_completed * (runtime.end_update - completed)
    durable_json(
        runtime.output / "HEARTBEAT.json",
        {
            "experiment": EXPERIMENT,
            "timestamp": time.time(),
            "pid": os.getpid(),
            "pod_id": runtime.metadata["pod_id"],
            "local_update": completed,
            "additional_targets": metrics["additional_targets"],
            "cumulative_2d2_targets": metrics["cumulative_2d2_targets"],
            "g_rec_b1_raw": metrics["g_rec_b1_raw"],
            "tanh_g_rec_b1": metrics["tanh_g_rec_b1"],
            "g_rec_b2_raw": metrics["g_rec_b2_raw"],
            "tanh_g_rec_b2": metrics["tanh_g_rec_b2"],
            "last_update_wall_seconds": metrics["wall_seconds"],
            "eta_seconds_to_segment_end": eta,
            "checkpoint": runtime.training_state.get("last_checkpoint"),
        },
    )


def merge_keyed_json(path, key, value):
    path = Path(path)
    payload = read_json(path) if path.is_file() else {}
    if str(key) in payload:
        raise SystemExit(f"refusing to overwrite {path.name} key {key}")
    payload[str(key)] = value
    durable_json(path, payload)


def milestone_diagnostics(runtime, update, val_path):
    saved_rng = capture_rng_state()
    validation = evaluate_parallel(
        runtime.model, val_path, combined_controls=(update == MAX_UPDATES)
    )
    validation.update(
        {
            "local_update": update,
            "additional_targets": update * GLOBAL_TARGETS,
            "cumulative_2d2_targets": SOURCE_TARGETS + update * GLOBAL_TARGETS,
        }
    )
    attention_b1 = attention_diagnostics(runtime.model, val_path, "b1")
    attention_b2 = attention_diagnostics(runtime.model, val_path, "b2")
    temporal_b1 = temporal_gradient_by_lag(runtime.model, val_path, "b1")
    temporal_b2 = temporal_gradient_by_lag(runtime.model, val_path, "b2")
    for temporal in (temporal_b1, temporal_b2):
        temporal.update(
            {
                "local_update": update,
                "additional_targets": update * GLOBAL_TARGETS,
                "cumulative_2d2_targets": SOURCE_TARGETS + update * GLOBAL_TARGETS,
            }
        )
    merge_keyed_json(runtime.output / "milestone_validation.json", update, validation)
    merge_keyed_json(runtime.output / "b1_attention_lag_bins.json", update, attention_b1["lag_bins"])
    merge_keyed_json(runtime.output / "b2_attention_lag_bins.json", update, attention_b2["lag_bins"])
    merge_keyed_json(
        runtime.output / "b1_attention_head_distance.json",
        update,
        {
            "heads": attention_b1["heads"],
            "aggregate": attention_b1["aggregate"],
            "mass_partitions": attention_b1["mass_partitions"],
            "pinned_batch": attention_b1["pinned_batch"],
        },
    )
    merge_keyed_json(
        runtime.output / "b2_attention_head_distance.json",
        update,
        {
            "heads": attention_b2["heads"],
            "aggregate": attention_b2["aggregate"],
            "mass_partitions": attention_b2["mass_partitions"],
            "pinned_batch": attention_b2["pinned_batch"],
        },
    )
    merge_keyed_json(runtime.output / "b12_to_b1_temporal_gradient.json", update, temporal_b1)
    merge_keyed_json(runtime.output / "b11_to_b2_temporal_gradient.json", update, temporal_b2)
    restore_rng_state(saved_rng)
    runtime.model.train()
    return validation, {"b1": attention_b1, "b2": attention_b2}, {
        "b1": temporal_b1,
        "b2": temporal_b2,
    }


def save_run_checkpoint(runtime, update, kind):
    prefix = "scientific" if kind == "scientific" else "recovery"
    checkpoint_path = (
        Path(runtime.run_root) / "checkpoints" / f"{prefix}_update_{update:04d}.pt"
    )
    previous = runtime.training_state.get("last_checkpoint")
    runtime.training_state["last_checkpoint"] = str(checkpoint_path.resolve())
    try:
        verification = save_checkpoint(
            checkpoint_path,
            runtime.model,
            runtime.optimizer,
            runtime.loader,
            runtime.training_state,
            runtime.metadata,
            runtime.accumulation,
        )
    except BaseException:
        runtime.training_state["last_checkpoint"] = previous
        raise
    record_checkpoint(runtime.output, update, verification, kind=kind)
    return verification


def run_train(args):
    require_git(clean=not bool(args.resume))
    if args.resume:
        dirty = [
            line
            for line in git_output("status", "--porcelain").splitlines()
            if line and OUTPUT_NAME not in line
        ]
        if dirty:
            raise SystemExit(f"resume has non-result worktree changes: {dirty}")
    require_config()
    if int(args.end_update) not in (FORCED_RESTART_UPDATE, MAX_UPDATES):
        raise SystemExit("2D2C train segments must end at local update 96 or 191")
    runtime = initialize_runtime(args)
    completed = int(runtime.training_state["completed_2d2c_updates"])
    if completed == 0 and int(args.end_update) != FORCED_RESTART_UPDATE:
        raise SystemExit("fresh scientific process must stop at update 96")
    if completed == FORCED_RESTART_UPDATE and int(args.end_update) != MAX_UPDATES:
        raise SystemExit("restarted scientific process must end at update 191")
    metrics_path = runtime.output / "training_metrics.jsonl"
    if completed == 0 and metrics_path.exists():
        raise SystemExit("fresh scientific run found existing training metrics")
    if completed > 0:
        rows = read_jsonl(metrics_path)
        if len(rows) != completed or rows[-1]["local_update"] != completed:
            raise SystemExit("resume metrics do not reconcile with checkpoint")
    val_path = validation_path(args.data_root)
    runtime.training_state["segment_start"] = completed
    segment_started = time.time()
    for update in range(completed + 1, int(args.end_update) + 1):
        metrics = train_one_update(runtime, update)
        append_jsonl(metrics_path, metrics)
        write_heartbeat(runtime, metrics)
        print(
            f"2D2C update={update:03d}/{MAX_UPDATES} "
            f"loss={metrics['weighted_total_ce']:.6f} "
            f"b1={metrics['tanh_g_rec_b1']:+.8f} "
            f"b2={metrics['tanh_g_rec_b2']:+.8f} "
            f"dt={metrics['wall_seconds']:.2f}s",
            flush=True,
        )
        verification = None
        if update in SCIENTIFIC_CHECKPOINTS:
            verification = save_run_checkpoint(runtime, update, "scientific")
        elif update in RECOVERY_CHECKPOINTS:
            verification = save_run_checkpoint(runtime, update, "recovery")
        if update in MILESTONES[1:]:
            milestone_diagnostics(runtime, update, val_path)
        if update == FORCED_RESTART_UPDATE:
            durable_json(
                runtime.output / "restart_required_update_96.json",
                {
                    "local_update": update,
                    "checkpoint": verification,
                    "saved_process_id": os.getpid(),
                    "fresh_process_required_for_update_97": True,
                },
            )
    segment = {
        "start_local_update": completed,
        "end_local_update": int(args.end_update),
        "started_at": segment_started,
        "completed_at": time.time(),
        "process_id": os.getpid(),
        "command": " ".join(sys.argv),
        "checkpoint": runtime.training_state["last_checkpoint"],
    }
    merge_keyed_json(runtime.output / "process_segments.json", args.end_update, segment)
    if int(args.end_update) == FORCED_RESTART_UPDATE:
        print("EXPERIMENT_2D2C_UPDATE_96_RESTART_REQUIRED", flush=True)
    else:
        durable_json(
            runtime.output / "training_complete.json",
            {
                "completed_2d2c_updates": MAX_UPDATES,
                "processed_2d2c_targets": ADDITIONAL_TARGETS,
                "cumulative_2d2_targets": CUMULATIVE_TARGETS,
                "checkpoint": runtime.training_state["last_checkpoint"],
                "timestamp": time.time(),
            },
        )
        print("EXPERIMENT_2D2C_TRAINING_COMPLETE", flush=True)
    return segment


def _incremental_control(model, x, y, name, derangement=None):
    batch, length = x.shape
    if name not in {
        "both_real",
        "b2_recurrence_off",
        "b2_shuffled",
        "b2_full_counterfactual",
        "both_shuffled",
    }:
        raise ValueError(name)
    state = model.init_incremental_state(
        batch, device=x.device, b2_full_cache=name == "b2_full_counterfactual"
    )
    per_sequence_sum = torch.zeros(batch, dtype=torch.float64, device="cpu")
    per_position_sum = np.zeros(length, dtype=np.float64)
    total_sum = 0.0
    targets = 0
    max_cache = [0] * N_LAYER
    max_h11_ring = 0
    max_h12_ring = 0
    b1_recurrent_output_rms = []
    b2_recurrent_output_rms = []
    for position in range(length):
        result = model.incremental_step(
            x[:, position],
            state,
            control=name,
            recurrent_permutation=(
                derangement if name in {"b2_shuffled", "both_shuffled"} else None
            ),
            return_diagnostics=True,
            bank_mode="full",
            diagnostic_attention_weights=False,
        )
        logits, state, diagnostics = result
        for key, values in (
            ("b1", b1_recurrent_output_rms),
            ("b2", b2_recurrent_output_rms),
        ):
            current = diagnostics.get(key)
            if current is not None:
                values.append(current["recurrent_output_rms"].detach().float().item())
        losses = F.cross_entropy(
            logits[:, 0].float(), y[:, position], reduction="none"
        )
        cpu_losses = losses.double().cpu()
        per_sequence_sum += cpu_losses
        per_position_sum[position] += cpu_losses.sum().item()
        total_sum += cpu_losses.sum().item()
        targets += batch
        lengths = model.incremental_cache_lengths(state)
        max_cache = [max(old, new) for old, new in zip(max_cache, lengths)]
        max_h11_ring = max(max_h11_ring, int(state.h11_ring.size(1)))
        max_h12_ring = max(max_h12_ring, int(state.h12_ring.size(1)))
    h11_memory_rms = (
        state.h11_ring.float().square().mean().sqrt().item()
        if state.h11_ring.numel()
        else 0.0
    )
    h12_memory_rms = (
        state.h12_ring.float().square().mean().sqrt().item()
        if state.h12_ring.numel()
        else 0.0
    )
    return {
        "loss_sum": total_sum,
        "targets": targets,
        "per_sequence_losses": (per_sequence_sum / length).tolist(),
        "per_position_sum": per_position_sum,
        "final_cache_audit": model.incremental_cache_audit(state),
        "max_cache_lengths": max_cache,
        "max_h11_ring_length": max_h11_ring,
        "max_h12_ring_length": max_h12_ring,
        "final_h11_memory_rms": h11_memory_rms,
        "final_h12_memory_rms": h12_memory_rms,
        "mean_b1_recurrent_output_rms": statistics.fmean(b1_recurrent_output_rms),
        "mean_b2_recurrent_output_rms": (
            statistics.fmean(b2_recurrent_output_rms)
            if b2_recurrent_output_rms
            else 0.0
        ),
    }


@torch.no_grad()
def evaluate_incremental(model, val_path, batches=INCREMENTAL_BATCHES) -> dict:
    model.eval()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    names = (
        "both_real",
        "b2_recurrence_off",
        "b2_shuffled",
        "b2_full_counterfactual",
        "both_shuffled",
    )
    rows = {
        name: {
            "loss_sum": 0.0,
            "targets": 0,
            "per_batch_losses": [],
            "per_sequence_losses": [],
            "per_position_sum": np.zeros(T, dtype=np.float64),
            "cache_rows": [],
            "h11_memory_rms": [],
            "h12_memory_rms": [],
            "b1_recurrent_output_rms": [],
            "b2_recurrent_output_rms": [],
        }
        for name in names
    }
    identities = []
    derangement = torch.arange(VALIDATION_B, device=device).roll(1)
    start = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(batches):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(legacy.d0d.batch_identity(cpu_x, cpu_y))
        x, y = cpu_x.to(device), cpu_y.to(device)
        for name in names:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current = _incremental_control(
                    model, x, y, name, derangement=derangement
                )
            row = rows[name]
            row["loss_sum"] += current["loss_sum"]
            row["targets"] += current["targets"]
            row["per_batch_losses"].append(
                current["loss_sum"] / current["targets"]
            )
            row["per_sequence_losses"].extend(current["per_sequence_losses"])
            row["per_position_sum"] += current["per_position_sum"]
            row["cache_rows"].append(
                {
                    "final": current["final_cache_audit"],
                    "max_cache_lengths": current["max_cache_lengths"],
                    "max_h11_ring_length": current["max_h11_ring_length"],
                    "max_h12_ring_length": current["max_h12_ring_length"],
                }
            )
            row["h11_memory_rms"].append(current["final_h11_memory_rms"])
            row["h12_memory_rms"].append(current["final_h12_memory_rms"])
            row["b1_recurrent_output_rms"].append(current["mean_b1_recurrent_output_rms"])
            row["b2_recurrent_output_rms"].append(current["mean_b2_recurrent_output_rms"])
        print(f"2D2C incremental batch={batch_index + 1:02d}/{batches}", flush=True)
        del x, y, cpu_x, cpu_y
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - start
    controls = {}
    for name, row in rows.items():
        controls[name] = {
            "validation_loss": row["loss_sum"] / row["targets"],
            "validation_targets": row["targets"],
            "per_batch_losses": row["per_batch_losses"],
            "per_sequence_losses": row["per_sequence_losses"],
            "per_position_loss": (
                row["per_position_sum"] / (batches * VALIDATION_B)
            ).tolist(),
            "cache_rows": row["cache_rows"],
            "final_h11_memory_rms": statistics.fmean(row["h11_memory_rms"]),
            "final_h12_memory_rms": statistics.fmean(row["h12_memory_rms"]),
            "mean_b1_recurrent_output_rms": statistics.fmean(row["b1_recurrent_output_rms"]),
            "mean_b2_recurrent_output_rms": statistics.fmean(row["b2_recurrent_output_rms"]),
        }
    real = controls["both_real"]
    off = controls["b2_recurrence_off"]
    shuffled = controls["b2_shuffled"]
    full = controls["b2_full_counterfactual"]
    both_shuffled = controls["both_shuffled"]
    result = {
        "controls": controls,
        "true_b2_recurrent_gain": off["validation_loss"] - real["validation_loss"],
        "true_b2_sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "true_remaining_b2_compression_gap": real["validation_loss"]
        - full["validation_loss"],
        "true_both_real_vs_both_shuffled_gap": both_shuffled["validation_loss"]
        - real["validation_loss"],
        "both_real_vs_b2_off_batches": paired_stats(
            real["per_batch_losses"], off["per_batch_losses"]
        ),
        "both_real_vs_b2_shuffled_batches": paired_stats(
            real["per_batch_losses"], shuffled["per_batch_losses"]
        ),
        "both_real_vs_b2_full_batches": paired_stats(
            real["per_batch_losses"], full["per_batch_losses"]
        ),
        "both_real_vs_b2_off_sequences": paired_stats(
            real["per_sequence_losses"], off["per_sequence_losses"]
        ),
        "both_real_vs_b2_shuffled_sequences": paired_stats(
            real["per_sequence_losses"], shuffled["per_sequence_losses"]
        ),
        "both_real_vs_b2_full_sequences": paired_stats(
            real["per_sequence_losses"], full["per_sequence_losses"]
        ),
        "g_rec_b1_raw": model.g_rec_b1.detach().float().item(),
        "tanh_g_rec_b1": model.recurrent_scale_b1.detach().float().item(),
        "g_rec_b2_raw": model.g_rec_b2.detach().float().item(),
        "tanh_g_rec_b2": model.recurrent_scale_b2.detach().float().item(),
        "canonical_subset_sha256": legacy.d0.aggregate_hashes(
            [row["combined_sha256"] for row in identities]
        ),
        "batch_identities": identities,
        "batch_count": batches,
        "batch_size": VALIDATION_B,
        "sequence_length": T,
        "targets_per_control": batches * VALIDATION_B * T,
        "minimum_target_requirement_met": batches * VALIDATION_B * T >= 131_072,
        "precision": "teacher-forced true incremental autocast(cuda,bfloat16)",
        "no_complete_prefix_recomputation": True,
        "performance": {
            "wall_seconds": elapsed,
            "condition_target_passes_per_second": batches
            * VALIDATION_B
            * T
            * len(names)
            / elapsed,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }
    return result


@torch.no_grad()
def old_memory_ablation(model, val_path, batches=VALIDATION_BATCHES) -> dict:
    model.eval()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    names = ("plain", "full_bank", "recent_only", "old_only")
    sums = {name: 0.0 for name in names}
    targets = 0
    per_batch = {name: [] for name in names}
    identities = []
    for batch_index in range(batches):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(legacy.d0d.batch_identity(cpu_x, cpu_y))
        x, y = cpu_x.to(device), cpu_y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            first = model.forward_pass(x)
            source = first["h12"]
            outputs = {
                "plain": first,
                "full_bank": model.forward_pass(
                    x, recurrent_source=source, bank_mode="full"
                ),
                "recent_only": model.forward_pass(
                    x, recurrent_source=source, bank_mode="recent_only"
                ),
                "old_only": model.forward_pass(
                    x, recurrent_source=source, bank_mode="old_only"
                ),
            }
            losses = {
                name: _token_losses(value["logits"], y) for name, value in outputs.items()
            }
        for name, tensor in losses.items():
            sums[name] += tensor.double().sum().item()
            per_batch[name].append(tensor.float().mean().item())
        targets += y.numel()
        print(f"2D2C old-memory ablation batch={batch_index + 1:02d}/{batches}", flush=True)
        del x, y, cpu_x, cpu_y, first, source, outputs, losses
        torch.cuda.empty_cache()
    controls = {name: sums[name] / targets for name in names}
    return {
        "controls": controls,
        "full_vs_recent_gain": controls["recent_only"] - controls["full_bank"],
        "full_vs_old_gain": controls["old_only"] - controls["full_bank"],
        "recent_only_utility_vs_plain": controls["plain"] - controls["recent_only"],
        "old_only_utility_vs_plain": controls["plain"] - controls["old_only"],
        "full_vs_recent_paired": paired_stats(
            per_batch["full_bank"], per_batch["recent_only"]
        ),
        "full_vs_old_paired": paired_stats(
            per_batch["full_bank"], per_batch["old_only"]
        ),
        "old_only_vs_plain_paired": paired_stats(
            per_batch["old_only"], per_batch["plain"]
        ),
        "canonical_validation_sha256": legacy.d0.aggregate_hashes(
            [row["combined_sha256"] for row in identities]
        ),
        "targets_per_control": targets,
        "recent_lags": "2...31",
        "old_lags": "32...1023",
        "diagnostic_only": True,
    }


@torch.no_grad()
def parallel_incremental_equivalence(model, val_path, length=16, batch=2):
    loader = legacy.d1.ExplicitShardLoader([val_path], batch, length)
    cpu_x, _ = loader.next_batch()
    tokens = cpu_x.to(next(model.parameters()).device)
    reports = {}
    original_precision = torch.get_float32_matmul_precision()
    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    for label in ("fp32", "bf16"):
        if label == "fp32":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            context = contextlib.nullcontext()
            threshold = FP32_INCREMENTAL_ATOL
        else:
            torch.set_float32_matmul_precision(original_precision)
            torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            threshold = BF16_INCREMENTAL_ATOL
        with context:
            plain = model.forward_pass(tokens)["logits"]
            incremental_plain = model.incremental_logits(
                tokens,
                control="both_real",
                b1_gate_override=0.0,
                b2_gate_override=0.0,
                diagnostic_attention_weights=False,
            )["logits"]
            first = model.forward_pass(tokens)
            parallel_real = model.forward_pass(
                tokens,
                b1_recurrent_source=first["h12"],
                b2_recurrent_source=first["h11"],
                bank_mode="full",
            )["logits"]
            incremental_real = model.incremental_logits(
                tokens,
                control="both_real",
                bank_mode="full",
                diagnostic_attention_weights=False,
            )["logits"]
        plain_delta = (plain.float() - incremental_plain.float()).abs()
        recurrent_delta = (parallel_real.float() - incremental_real.float()).abs()
        reports[label] = {
            "plain_kernel_max_abs": plain_delta.max().item(),
            "plain_kernel_mean_abs": plain_delta.mean().item(),
            "active_recurrent_positions_0_3_max_abs": recurrent_delta[:, :4].max().item(),
            "active_recurrent_positions_0_3_mean_abs": recurrent_delta[:, :4].mean().item(),
            "self_recurrence_drift_positions_4_plus_mean_abs": recurrent_delta[:, 4:].mean().item(),
            "self_recurrence_drift_positions_4_plus_max_abs": recurrent_delta[:, 4:].max().item(),
            "max_abs_tolerance": threshold,
            "kernel_passed": plain_delta.max().item() <= threshold
            and recurrent_delta[:, :4].max().item() <= threshold,
        }
    torch.set_float32_matmul_precision(original_precision)
    torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
    torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
    reports["bf16_correction_provenance"] = {
        "tolerance": BF16_INCREMENTAL_ATOL,
        "source": "2D2A disclosed evaluation-only correction: active uses preregistered Plain tolerance",
        "fp32_checks_unchanged": True,
    }
    reports["passed"] = reports["fp32"]["kernel_passed"] and reports["bf16"][
        "kernel_passed"
    ]
    return reports


def memory_accounting() -> dict:
    element_bytes = 2

    def one(batch):
        b1 = batch * 1 * N_EMBD * 2 * element_bytes
        b12_raw = batch * 1023 * N_EMBD * element_bytes
        b2 = batch * 1 * N_EMBD * 2 * element_bytes
        b11_raw = batch * 1023 * N_EMBD * element_bytes
        upper = batch * 10 * 1023 * N_EMBD * 2 * element_bytes
        total = b1 + b12_raw + b2 + b11_raw + upper
        final_2d2b = batch * (
            1 * N_EMBD * 2 + 1023 * N_EMBD + 11 * 1023 * N_EMBD * 2
        ) * element_bytes
        standard = batch * 12 * 1023 * N_EMBD * 2 * element_bytes
        return {
            "batch_size": batch,
            "b1_local_kv_bytes": b1,
            "b12_recurrent_raw_state_bytes": b12_raw,
            "b2_local_kv_bytes": b2,
            "b11_recurrent_raw_state_bytes": b11_raw,
            "b3_b12_ordinary_kv_bytes": upper,
            "total_experimental_inference_state_bytes": total,
            "final_2d2b_inference_state_bytes": final_2d2b,
            "standard_gpt2_w1024_kv_bytes": standard,
            "delta_bytes_vs_final_2d2b": total - final_2d2b,
            "delta_bytes_vs_standard_gpt2": total - standard,
            "mib": {
                "b1_local_kv": b1 / 1024**2,
                "b12_recurrent_raw_state": b12_raw / 1024**2,
                "b2_local_kv": b2 / 1024**2,
                "b11_recurrent_raw_state": b11_raw / 1024**2,
                "b3_b12_ordinary_kv": upper / 1024**2,
                "total": total / 1024**2,
                "delta_vs_final_2d2b": (total - final_2d2b) / 1024**2,
                "delta_vs_standard_gpt2": (total - standard) / 1024**2,
            },
        }

    return {
        "dtype": "BF16",
        "bytes_per_element": element_bytes,
        "B1": one(1),
        "B64": one(64),
        "state_limits": {
            "B1 ordinary historical KV": 1,
            "B12 raw recurrent states": 1023,
            "B2 ordinary historical KV": 1,
            "B11 raw recurrent states": 1023,
            "B3-B12 ordinary historical KV each": 1023,
        },
        "comparison_note": "Exact BF16 inference-state comparison at a full 1024-token context.",
    }


def load_final_model(args, device):
    output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json")
    batch_manifest = read_json(output / "batch_manifest.json")
    micro_batch = int(batch_manifest["selected_micro_batch_sequences"])
    accumulation = int(batch_manifest["selected_gradient_accumulation"])
    model, optimizer, _, source_payload, _ = load_source_bundle(
        args.source_checkpoint, device, restore_rng=False
    )
    metadata = training_metadata(args, preflight, micro_batch, accumulation)
    final_path = Path(args.final_checkpoint).resolve()
    payload = legacy.d0.torch_load(final_path, mmap=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("final checkpoint schema mismatch")
    if payload.get("completed_2d2c_updates") != MAX_UPDATES:
        raise SystemExit("final checkpoint does not contain 191 local updates")
    if payload.get("cumulative_2d2_targets") != CUMULATIVE_TARGETS:
        raise SystemExit("final checkpoint cumulative target mismatch")
    if payload.get("metadata") != metadata:
        raise SystemExit("final checkpoint metadata mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    final_loader = legacy.d1.ExplicitShardLoader(
        payload["loader_state"]["shards"],
        micro_batch,
        T,
        state=payload["loader_state"],
    )
    if payload["next_global_batch_sha256"] != next_global_batch_hash(
        final_loader, accumulation
    ):
        raise SystemExit("final checkpoint next batch mismatch")
    audit = {
        "checkpoint": str(final_path),
        "sha256": file_sha256(final_path),
        "bytes": final_path.stat().st_size,
        "completed_2d2c_updates": payload["completed_2d2c_updates"],
        "processed_2d2c_targets": payload["processed_2d2c_targets"],
        "cumulative_2d2_targets": payload["cumulative_2d2_targets"],
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "model_finite": model_finite(model),
        "optimizer_finite": optimizer_finite(optimizer),
        "g_rec_b1_raw": model.g_rec.detach().float().item(),
        "tanh_g_rec_b1": model.recurrent_scale_b1.detach().float().item(),
        "g_rec_b2_raw": model.g_rec_b2.detach().float().item(),
        "tanh_g_rec_b2": model.recurrent_scale_b2.detach().float().item(),
    }
    audit["passed"] = (
        audit["sha256"]
        == final_path.with_suffix(final_path.suffix + ".sha256").read_text().split()[0]
        and audit["model_finite"]
        and audit["optimizer_finite"]
    )
    del source_payload, payload, final_loader
    gc.collect()
    return model, optimizer, audit


def classify_result(incremental, stable=True, integrity=True) -> str:
    if not integrity:
        return "EXPERIMENT 2D2C INVALID"
    if not stable:
        return "SECOND MIRRORED LINK IS UNSTABLE"
    controls = incremental["controls"]
    real = controls["both_real"]["validation_loss"]
    off = controls["b2_recurrence_off"]["validation_loss"]
    shuffled = controls["b2_shuffled"]["validation_loss"]
    off_wins = incremental["both_real_vs_b2_off_sequences"]
    shuffled_wins = incremental["both_real_vs_b2_shuffled_sequences"]
    majority = off_wins["wins"] > off_wins["losses"] and shuffled_wins[
        "wins"
    ] > shuffled_wins["losses"]
    if real < off and real < shuffled and majority:
        if incremental["true_b2_recurrent_gain"] >= 0.01:
            return "SECOND MIRRORED RECURRENT K/V LINK STRONGLY REPAIRS B2 COMPRESSION"
        return "SECOND MIRRORED RECURRENT K/V LINK LEARNS POSITIVE UTILITY"
    if real < shuffled and real >= off:
        return "B11→B2 RECURRENCE IS SEQUENCE-SPECIFIC BUT NOT YET USEFUL"
    if abs(incremental["tanh_g_rec_b2"]) < 1e-4:
        return "SECOND MIRRORED LINK REMAINS NEAR ZERO"
    return "SECOND MIRRORED LINK DOES NOT ESTABLISH POSITIVE UTILITY"


def choose_recommendation(classification, parallel, incremental, attention) -> str:
    if classification == "EXPERIMENT 2D2C INVALID":
        return "FIX 2D2C INTEGRITY"
    del parallel, incremental, attention
    if classification == "EXPERIMENT 2D2C INVALID":
        return "FIX 2D2C INTEGRITY"
    if classification == "SECOND MIRRORED LINK IS UNSTABLE":
        return "STABILIZE SECOND RECURRENT LINK"
    if classification in {
        "SECOND MIRRORED RECURRENT K/V LINK LEARNS POSITIVE UTILITY",
        "SECOND MIRRORED RECURRENT K/V LINK STRONGLY REPAIRS B2 COMPRESSION",
    }:
        return "ADD THIRD MIRRORED FULL-BANK LINK B10→B3"
    if classification == "B11→B2 RECURRENCE IS SEQUENCE-SPECIFIC BUT NOT YET USEFUL":
        return "IMPROVE B2 RECURRENT READOUT BEFORE ADDING B10→B3"
    if classification == "SECOND MIRRORED LINK REMAINS NEAR ZERO":
        return "TEST WHETHER B2 NEEDS MORE ADAPTATION / RECURRENT CAPACITY"
    return "IMPROVE B2 RECURRENT READOUT BEFORE ADDING B10→B3"


@torch.no_grad()
def self_composition_diagnostic(model, val_path, passes=8, batch_size=2) -> dict:
    model.eval()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], batch_size, T)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(device), cpu_y.to(device)
    source_h11 = None
    source_h12 = None
    rows = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for pass_index in range(passes):
            current = model.forward_pass(
                x,
                b1_recurrent_source=source_h12,
                b2_recurrent_source=source_h11,
                return_diagnostics=pass_index > 0,
                bank_mode="full",
            )
            loss = _token_losses(current["logits"], y).double().mean().item()
            rows.append(
                {
                    "pass": pass_index + 1,
                    "loss": loss,
                    "b12_memory_rms": current["h12"].float().square().mean().sqrt().item(),
                    "b11_memory_rms": current["h11"].float().square().mean().sqrt().item(),
                    "b1_recurrent_output_rms": (
                        current["diagnostics"]["b1"]["recurrent_output_rms"].float().item()
                        if pass_index > 0
                        else 0.0
                    ),
                    "b2_recurrent_output_rms": (
                        current["diagnostics"]["b2"]["recurrent_output_rms"].float().item()
                        if pass_index > 0
                        else 0.0
                    ),
                }
            )
            source_h11 = current["h11"]
            source_h12 = current["h12"]
    report = {
        "passes": rows,
        "batch_size": batch_size,
        "sequence_length": T,
        "finite": all(
            math.isfinite(row["loss"])
            and math.isfinite(row["b12_memory_rms"])
            and math.isfinite(row["b11_memory_rms"])
            and math.isfinite(row["b1_recurrent_output_rms"])
            and math.isfinite(row["b2_recurrent_output_rms"])
            for row in rows
        ),
        "no_gradient": True,
    }
    del x, y, source_h11, source_h12, current
    torch.cuda.empty_cache()
    return report


def build_performance(training, milestones, incremental, preflight) -> dict:
    walls = [row["wall_seconds"] for row in training]
    tps = [row["targets_per_second"] for row in training]
    allocated = [row["peak_allocated_vram_mb"] for row in training]
    reserved = [row["peak_reserved_vram_mb"] for row in training]
    return {
        "training": {
            "updates": len(training),
            "total_wall_seconds": sum(walls),
            "mean_seconds_per_update": statistics.fmean(walls),
            "median_seconds_per_update": statistics.median(walls),
            "aggregate_targets_per_second": ADDITIONAL_TARGETS / sum(walls),
            "mean_update_targets_per_second": statistics.fmean(tps),
            "peak_allocated_vram_mb": max(allocated),
            "peak_reserved_vram_mb": max(reserved),
            "2d2a_reference_targets_per_second": 59_269,
            "throughput_ratio_vs_2d2a": (ADDITIONAL_TARGETS / sum(walls)) / 59_269,
            "changed_kernel_geometry_caveat": True,
            "pass_forward_seconds_total": [
                sum(row["pass_forward_seconds"][index] for row in training if len(row["pass_forward_seconds"]) > index)
                for index in range(3)
            ],
            "aggregate_backward_seconds_total": sum(
                row["aggregate_backward_seconds"] for row in training
            ),
        },
        "recurrent_attention_microbenchmark": preflight["performance_benchmark"][
            "recurrent_attention"
        ],
        "milestone_validation": {
            key: value["performance"] for key, value in milestones.items()
        },
        "true_incremental": incremental["performance"],
    }


def make_plots(output, milestones, training, incremental, attention, temporal, performance):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    keys = sorted((int(key) for key in milestones))
    x = [milestones[str(key)]["additional_targets"] for key in keys]
    real = [milestones[str(key)]["controls"]["new_real"]["validation_loss"] for key in keys]
    off = [milestones[str(key)]["controls"]["b2_recurrence_off"]["validation_loss"] for key in keys]
    shuffled = [milestones[str(key)]["controls"]["b2_shuffled"]["validation_loss"] for key in keys]
    full = [milestones[str(key)]["controls"]["b2_full_counterfactual"]["validation_loss"] for key in keys]

    def line_plot(filename, series, ylabel):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for label, values in series.items():
            ax.plot(x, values, marker="o", label=label)
        ax.set_xlabel("Additional 2D2C training targets")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    line_plot(REQUIRED_PLOTS[0], {"BothReal": real, "B2Off": off, "B2Shuffled": shuffled, "B2Full": full}, "Cross entropy")
    line_plot(REQUIRED_PLOTS[1], {"B2Off − BothReal": [milestones[str(k)]["b2_recurrent_gain"] for k in keys]}, "B2 recurrent gain")
    line_plot(REQUIRED_PLOTS[2], {"BothReal − B2Full": [milestones[str(k)]["remaining_b2_compression_gap"] for k in keys]}, "Remaining compression gap")
    line_plot(REQUIRED_PLOTS[3], {"B2Shuffled − BothReal": [milestones[str(k)]["b2_sequence_gap"] for k in keys]}, "B2 sequence gap")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    tx = [row["additional_targets"] for row in training]
    ax.plot(tx, [row["tanh_g_rec_b1"] for row in training], label="B12→B1")
    ax.plot(tx, [row["tanh_g_rec_b2"] for row in training], label="B11→B2")
    ax.scatter([0, 0], [SOURCE_GATE_EFFECTIVE, 0.0], zorder=3)
    ax.set(xlabel="Additional 2D2C training targets", ylabel="Effective recurrent gate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[4], dpi=180)
    plt.close(fig)

    labels = [name for name, _, _ in LAG_BINS]
    b1_final_attention = attention["b1"][str(MAX_UPDATES)]
    b2_final_attention = attention["b2"][str(MAX_UPDATES)]
    for filename, values, ylabel in (
        (REQUIRED_PLOTS[5], "attention_mass", "Attention mass"),
        (REQUIRED_PLOTS[6], "normalized_mass_per_available_token", "Mass per available token"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        positions = np.arange(len(labels))
        ax.bar(positions - 0.2, [b1_final_attention[name][values] for name in labels], width=0.4, label="B1")
        ax.bar(positions + 0.2, [b2_final_attention[name][values] for name in labels], width=0.4, label="B2")
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions, labels, rotation=35)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    b1_heads = read_json(output / "b1_attention_head_distance.json")[str(MAX_UPDATES)]["heads"]
    b2_heads = read_json(output / "b2_attention_head_distance.json")[str(MAX_UPDATES)]["heads"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    positions = np.arange(N_HEAD)
    ax.bar(positions - 0.2, [b1_heads[str(index)]["mean_attended_recurrent_lag"] for index in range(N_HEAD)], width=0.4, label="B1")
    ax.bar(positions + 0.2, [b2_heads[str(index)]["mean_attended_recurrent_lag"] for index in range(N_HEAD)], width=0.4, label="B2")
    ax.set(xlabel="Attention head", ylabel="Mean recurrent lag")
    ax.set_xticks(range(N_HEAD))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[7], dpi=180)
    plt.close(fig)

    b1_gradient = temporal["b1"][str(MAX_UPDATES)]["bins"]
    b2_gradient = temporal["b2"][str(MAX_UPDATES)]["bins"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    positions = np.arange(len(labels))
    ax.bar(positions - 0.2, [b1_gradient[name]["mean_gradient_rms"] for name in labels], width=0.4, label="B12→B1")
    ax.bar(positions + 0.2, [b2_gradient[name]["mean_gradient_rms"] for name in labels], width=0.4, label="B11→B2")
    ax.set_ylabel("Mean writer-gradient RMS")
    ax.set_yscale("log")
    ax.set_xticks(positions, labels, rotation=35)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[8], dpi=180)
    plt.close(fig)

    final_bins = milestones[str(MAX_UPDATES)]["position_bins"]
    position_labels = [name for name, _, _ in POSITION_BINS]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(position_labels, [final_bins[name]["b2_recurrent_gain"] for name in position_labels])
    ax.set_ylabel("B2Off − BothReal CE")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[9], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    names = ["B2 gain", "Sequence gap", "Compression gap"]
    parallel_values = [
        milestones[str(MAX_UPDATES)]["b2_recurrent_gain"],
        milestones[str(MAX_UPDATES)]["b2_sequence_gap"],
        milestones[str(MAX_UPDATES)]["remaining_b2_compression_gap"],
    ]
    true_values = [
        incremental["true_b2_recurrent_gain"],
        incremental["true_b2_sequence_gap"],
        incremental["true_remaining_b2_compression_gap"],
    ]
    positions = np.arange(3)
    ax.bar(positions - 0.18, parallel_values, width=0.36, label="Parallel")
    ax.bar(positions + 0.18, true_values, width=0.36, label="True incremental")
    ax.set_xticks(positions, names)
    ax.set_ylabel("CE improvement")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[10], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(
        [row["wall_seconds"] for row in training],
        [row["peak_allocated_vram_mb"] for row in training],
        s=12,
        alpha=0.7,
    )
    ax.set(xlabel="Seconds/update", ylabel="Peak allocated VRAM (MiB)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[11], dpi=180)
    plt.close(fig)


def first_positive_milestone(milestones, field):
    for key in sorted(int(value) for value in milestones):
        if milestones[str(key)][field] > 0:
            return key
    return None


def build_questions(
    summary, milestones, incremental, attention, temporal, old_ablation, memory
):
    del old_ablation
    final_attention = attention["b2"]
    final_gradient = temporal["b2"]["bins"]
    heads = final_attention["heads"]
    means = [heads[str(index)]["mean_attended_recurrent_lag"] for index in range(N_HEAD)]
    aggregate = final_attention["aggregate"]
    initial = summary["initial_b2_shortening"]
    parallel = summary["parallel"]
    trajectory = summary["gate_diagnostics"]["trajectory"]
    first_open = next(
        (row["local_update"] for row in trajectory if row["g_rec_b2_raw"] != 0.0),
        None,
    )
    damage = initial["damage_b2_initial"]
    recovery = parallel["b2_recurrent_gain"] / damage if damage else None
    milestone_gain = {
        key: milestones[str(key)]["b2_recurrent_gain"] for key in (20, 48, 96, 143, 191)
    }
    b1_marginal = parallel["combined_system"]["b1_marginal_gain_with_b2_real"]
    return {
        "Q1": {"question": "Exact initial B2-W2 compression damage on the 2D2B source?", "answer": damage},
        "Q2": {"question": "Did new g_rec_B2 open?", "answer": first_open is not None},
        "Q3": {"question": "At what update?", "answer": first_open},
        "Q4": {"question": "Final tanh(g_rec_B1)?", "answer": summary["final_tanh_g_rec_b1"]},
        "Q5": {"question": "Final tanh(g_rec_B2)?", "answer": summary["final_tanh_g_rec_b2"]},
        "Q6": {"question": "Did B11→B2 writer gradients become nonzero?", "answer": temporal["b2"]["nonzero"]},
        "Q7": {"question": "Did they reach hundreds of tokens back?", "answer": {"128+": final_gradient["128-255"]["fraction_nonzero_positions"] > 0, "256+": final_gradient["256-511"]["fraction_nonzero_positions"] > 0, "512+": final_gradient["512-1023"]["fraction_nonzero_positions"] > 0}},
        "Q8": {"question": "B2 recurrent gain at 10M?", "answer": milestone_gain[20]},
        "Q9": {"question": "At 25M?", "answer": milestone_gain[48]},
        "Q10": {"question": "At 50M?", "answer": milestone_gain[96]},
        "Q11": {"question": "At 75M?", "answer": milestone_gain[143]},
        "Q12": {"question": "At 100M?", "answer": milestone_gain[191]},
        "Q13": {"question": "Final B2 sequence gap?", "answer": parallel["b2_sequence_gap"]},
        "Q14": {"question": "Final paired wins vs B2-off?", "answer": incremental["both_real_vs_b2_off_sequences"]},
        "Q15": {"question": "Final paired wins vs B2-shuffled?", "answer": incremental["both_real_vs_b2_shuffled_sequences"]},
        "Q16": {"question": "How much B2 compression damage was recovered?", "answer": {"fraction": recovery, "gain": parallel["b2_recurrent_gain"], "initial_damage": damage}},
        "Q17": {"question": "How much remained versus same-checkpoint B2-full?", "answer": parallel["remaining_b2_compression_gap"]},
        "Q18": {"question": "Did B1 recurrence retain positive value after B2 compression?", "answer": b1_marginal > 0, "gain": b1_marginal},
        "Q19": {"question": "Did B1 gate grow/shrink?", "answer": "grew" if summary["final_g_rec_b1_raw"] > SOURCE_GATE_RAW else "shrank", "source": SOURCE_GATE_RAW, "final": summary["final_g_rec_b1_raw"]},
        "Q20": {"question": "Did B2 heads use old B11 positions?", "answer": final_attention["mass_partitions"]["lags_128_511"] + final_attention["mass_partitions"]["lags_512_1023"] > 0, "mass_128_plus": final_attention["mass_partitions"]["lags_128_511"] + final_attention["mass_partitions"]["lags_512_1023"]},
        "Q21": {"question": "What was B2 mean/median/p90 recurrent lag?", "answer": {"mean": aggregate["mean_attended_recurrent_lag"], "median": aggregate["median_attended_recurrent_lag"], "p90": aggregate["p90_attended_recurrent_lag"]}},
        "Q22": {"question": "Did B2 heads temporally specialize?", "answer": statistics.pstdev(means) > 1.0, "mean_lag_range": [min(means), max(means)], "std": statistics.pstdev(means)},
        "Q23": {"question": "Did parallel utility transfer to true incremental?", "answer": incremental["true_b2_recurrent_gain"] > 0, "parallel_gain": parallel["b2_recurrent_gain"], "true_gain": incremental["true_b2_recurrent_gain"]},
        "Q24": {"question": "Final true-self B11→B2 recurrent gain?", "answer": incremental["true_b2_recurrent_gain"]},
        "Q25": {"question": "Final true-self B11→B2 sequence gap?", "answer": incremental["true_b2_sequence_gap"]},
        "Q26": {"question": "Combined BOTH_REAL vs BOTH_SHUFFLED gap?", "answer": incremental["true_both_real_vs_both_shuffled_gap"]},
        "Q27": {"question": "Exact inference memory versus 2D2B?", "answer": {"B1_delta_bytes": memory["B1"]["delta_bytes_vs_final_2d2b"], "B64_delta_bytes": memory["B64"]["delta_bytes_vs_final_2d2b"]}},
        "Q28": {"question": "Exact inference memory versus Standard?", "answer": {"B1_delta_bytes": memory["B1"]["delta_bytes_vs_standard_gpt2"], "B64_delta_bytes": memory["B64"]["delta_bytes_vs_standard_gpt2"]}},
        "Q29": {"question": "Does this justify B10→B3 next?", "answer": summary["recommendation"] == "ADD THIRD MIRRORED FULL-BANK LINK B10→B3"},
        "Q30": {"question": "What exactly one experiment should run next?", "answer": summary["recommendation"]},
    }


def incremental_position_bins(incremental):
    controls = incremental["controls"]
    result = {}
    for name, first, last in POSITION_BINS:
        values = {
            control: np.asarray(row["per_position_loss"])[first : last + 1]
            for control, row in controls.items()
        }
        result[name] = {
            "both_real_loss": float(values["both_real"].mean()),
            "b2_recurrence_off_loss": float(values["b2_recurrence_off"].mean()),
            "b2_shuffled_loss": float(values["b2_shuffled"].mean()),
            "b2_full_counterfactual_loss": float(values["b2_full_counterfactual"].mean()),
            "b2_recurrent_gain": float(
                (values["b2_recurrence_off"] - values["both_real"]).mean()
            ),
            "b2_sequence_gap": float(
                (values["b2_shuffled"] - values["both_real"]).mean()
            ),
            "remaining_b2_compression_gap": float(
                (values["both_real"] - values["b2_full_counterfactual"]).mean()
            ),
        }
    return result


def build_artifact_inventory(output):
    output = Path(output)
    mutable = {
        "EXPERIMENT_2D2C_FINAL_REPORT.md",
        "FINAL_AUDIT.json",
        "result_summary.json",
        "commands_and_runtime.json",
        "UNATTENDED_FINAL_HANDOFF.md",
    }
    artifacts = {}
    for name in REQUIRED_ARTIFACTS:
        path = output / name
        exists = path.is_file() and path.stat().st_size > 0
        artifacts[name] = {
            "exists_nonempty": exists,
            "bytes": path.stat().st_size if exists else None,
            "sha256": file_sha256(path) if exists and name not in mutable else None,
            "hash_deferred_self_referential": name in mutable,
        }
    plots = {}
    for name in REQUIRED_PLOTS:
        path = output / name
        exists = path.is_file() and path.stat().st_size > 0
        plots[name] = {
            "exists_nonempty": exists,
            "bytes": path.stat().st_size if exists else None,
            "sha256": file_sha256(path) if exists else None,
        }
    return {
        "required_artifacts": artifacts,
        "required_plots": plots,
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "required_plot_count": len(REQUIRED_PLOTS),
        "passed": all(row["exists_nonempty"] for row in artifacts.values())
        and all(row["exists_nonempty"] for row in plots.values()),
    }


def _numbers_finite(value):
    if isinstance(value, dict):
        return all(_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numbers_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def render_report(summary, audit, questions):
    incremental = summary["incremental"]
    parallel = summary["parallel"]
    initial = summary["initial_b2_shortening"]
    training = summary["training"]
    memory = summary["memory_accounting"]
    lines = [
        f"EXPERIMENT 2D2C PRIMARY CLASSIFICATION:\n{summary['primary_classification']}",
        f"\nFINAL TRUE-SELF B11→B2 RECURRENT GAIN:\n{incremental['true_b2_recurrent_gain']}",
        f"\nFINAL TRUE-SELF B11→B2 SEQUENCE GAP:\n{incremental['true_b2_sequence_gap']}",
        "\n# Experiment 2D2C — Second Mirrored Full-Bank B11→B2 Recurrent K/V Link",
        "\n## Result",
        f"\nThe final classification is **{summary['primary_classification']}**.",
        f"The exactly one next experiment is **{summary['recommendation']}**.",
        "\n## Source and architecture",
        f"\n- Source checkpoint: `{summary['source_checkpoint']}`",
        f"- Source SHA-256: `{summary['source_checkpoint_sha256']}`",
        f"- Parameters: {summary['parameters']:,} (exactly one new scalar versus 2D2B)",
        f"- Hardware: {summary['hardware']}",
        "- B1: W2 ordinary KV plus full B12 recurrent raw-state bank",
        "- B2: W2 ordinary KV plus full B11 recurrent raw-state bank",
        "- B3–B12: W1024; both recurrent banks use lags 2…1023",
        "- Each link reuses its destination LN/K/V, has a separate softmax, and applies the destination `c_proj` once",
        "\nThe disclosed 2D2A evaluation-only BF16 correction is preserved: active-prefix absolute comparisons use the already-preregistered Plain tolerance 1.25; strict FP32 checks are unchanged. No model, checkpoint, training, data, loss, or scientific metric was changed by that correction.",
        "\n## Training",
        f"\n- Additional updates: {training['updates']} / {MAX_UPDATES}",
        f"- Additional targets: {ADDITIONAL_TARGETS:,}",
        f"- Cumulative 2D2 targets: {CUMULATIVE_TARGETS:,}",
        f"- Runtime: {training['total_wall_seconds']:.2f} seconds ({training['mean_seconds_per_update']:.2f} sec/update)",
        f"- Throughput: {training['aggregate_targets_per_second']:.2f} targets/sec",
        f"- Peak allocated/reserved VRAM: {training['peak_allocated_vram_mb']:.2f}/{training['peak_reserved_vram_mb']:.2f} MiB",
        f"- B1 gate tanh: {SOURCE_GATE_EFFECTIVE} → {summary['final_tanh_g_rec_b1']}",
        f"- B2 gate tanh: 0.0 → {summary['final_tanh_g_rec_b2']}",
        "- Base/B1 optimizer, loader, and Python/NumPy/Torch CPU/CUDA RNG states resumed from 2D2B; only B2 gate Adam state was fresh.",
        "- Mandatory fresh-process restart after local update 96 passed.",
        "\n## Initial B2 shortening",
        f"\n- Source 2D2B loss: {initial['source_2d2b_loss']}",
        f"- B2 W2, recurrence off: {initial['b2_w2_no_new_recurrence_loss']}",
        f"- Exact compression damage: {initial['damage_b2_initial']}",
        f"- Gate-zero identities passed: {initial['passed']}",
        "\n## Final parallel validation",
        f"\n- BothReal: {parallel['controls']['new_real']['validation_loss']}",
        f"- B2 recurrence off: {parallel['controls']['b2_recurrence_off']['validation_loss']}",
        f"- B2 shuffled: {parallel['controls']['b2_shuffled']['validation_loss']}",
        f"- B2 full counterfactual: {parallel['controls']['b2_full_counterfactual']['validation_loss']}",
        f"- B2 recurrent gain: {parallel['b2_recurrent_gain']}",
        f"- B2 sequence gap: {parallel['b2_sequence_gap']}",
        f"- Remaining compression gap: {parallel['remaining_b2_compression_gap']}",
        "\n## Final true incremental validation",
        f"\n- BothReal: {incremental['controls']['both_real']['validation_loss']}",
        f"- B2 recurrence off: {incremental['controls']['b2_recurrence_off']['validation_loss']}",
        f"- B2 shuffled: {incremental['controls']['b2_shuffled']['validation_loss']}",
        f"- B2 full counterfactual: {incremental['controls']['b2_full_counterfactual']['validation_loss']}",
        f"- Both shuffled: {incremental['controls']['both_shuffled']['validation_loss']}",
        f"- Targets/control: {incremental['targets_per_control']:,}",
        f"- Sequence wins vs B2-off: {incremental['both_real_vs_b2_off_sequences']['wins']} of {incremental['both_real_vs_b2_off_sequences']['count']}",
        f"- Sequence wins vs B2-shuffled: {incremental['both_real_vs_b2_shuffled_sequences']['wins']} of {incremental['both_real_vs_b2_shuffled_sequences']['count']}",
        "\n## Attention and temporal gradients",
        f"\n- Final B1 recurrent mass partitions: {summary['final_attention']['b1']['mass_partitions']}",
        f"- Final B2 recurrent mass partitions: {summary['final_attention']['b2']['mass_partitions']}",
        f"- B11→B2 long-lag gradient present: {summary['final_temporal_gradient']['b2']['long_lag_writer_gradient_present']}",
        "\n## Cache and storage",
        "\n- B1 and B2 historical same-layer KV: at most 1 entry each",
        "- B11 and B12 raw recurrent buffers: at most 1023 states each",
        "- B3–B12 ordinary historical KV: at most 1023 entries/layer",
        f"- BF16 total experimental inference state, B=1: {memory['B1']['mib']['total']:.3f} MiB",
        f"- BF16 total experimental inference state, B=64: {memory['B64']['mib']['total']:.3f} MiB",
        f"- Delta versus final 2D2B, B=1: {memory['B1']['delta_bytes_vs_final_2d2b']} bytes",
        f"- Delta versus Standard GPT-2, B=1: {memory['B1']['delta_bytes_vs_standard_gpt2']} bytes",
        "\n## Scientific questions Q1–Q30",
    ]
    for index in range(1, 31):
        row = questions[f"Q{index}"]
        lines.extend(
            [
                f"\n### Q{index}. {row['question']}",
                f"\n{json.dumps(row['answer'], sort_keys=True)}",
            ]
        )
    lines.extend(
        [
            "\n## Integrity and artifacts",
            f"\n- Final audit passed: {audit['passed']}",
            f"- Final checkpoint: `{summary['final_checkpoint']}`",
            f"- Final checkpoint SHA-256: `{summary['final_checkpoint_sha256']}`",
            f"- Implementation commit: `{summary['git']['implementation_commit']}`",
            f"- Results commit: `{summary['git'].get('results_commit')}`",
            f"- Artifact directory: `{summary['artifact_directory']}`",
            f"- GPU pod: `{summary['pod']['name']}` (`{summary['pod']['id']}`), status `{summary['pod']['status']}`",
            "\n# EXPERIMENT 2D2C COMPLETE",
        ]
    )
    return "\n".join(lines) + "\n"


def build_integrity_checks(
    preflight,
    final_model_audit,
    incremental,
    equivalence,
    temporal,
    checkpoint_manifest,
    composition,
    output,
):
    final_cache_rows = incremental["controls"]["both_real"]["cache_rows"]
    cache_pass = all(
        row["final"]["passed"]
        and row["final"]["b1_historical_kv"] <= 1
        and row["final"]["b2_historical_kv"] <= 1
        and row["final"]["h11_ring_length"] <= 1023
        and row["final"]["h12_ring_length"] <= 1023
        and all(value <= 1023 for value in row["final"]["b3_b12_historical_kv"])
        for row in final_cache_rows
    )
    restart = read_json(Path(output) / "forced_restart_update_96.json")
    scientific = checkpoint_manifest["scientific"]
    recovery = checkpoint_manifest["recovery"]
    checks = {
        "2D2B final tag exact": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
        "2D2B checkpoint SHA exact": final_model_audit["passed"] and preflight["source"]["checkpoint_sha256"] == SOURCE_SHA256,
        "audit-correction provenance preserved": preflight["source"]["audit_correction_provenance"]["passed"],
        "parameter count 124,475,906": preflight["parameters"]["2d2c_total_parameters"] == TOTAL_PARAMETERS,
        "exactly one new scalar": preflight["parameters"]["new_parameter_count_vs_2d2b"] == 1,
        "source parameter inventory preserved": preflight["parameters"]["source_inventory_preserved"],
        "B1 local W=2 exact": preflight["architecture"]["b1_local_window"] == 2,
        "B2 local W=2 exact": preflight["architecture"]["b2_local_window"] == 2,
        "B3-B12 W=1024 exact": preflight["architecture"]["b3_b12_windows"] == [1024] * 10,
        "recurrent full bank exact": preflight["kernel_preflight"]["checks"]["boundary_counts"],
        "maximum recurrent entries=1022": preflight["architecture"]["maximum_recurrent_entries"] == 1022,
        "no B1 local/recurrent overlap": preflight["kernel_preflight"]["checks"]["b1_local_recurrent_disjoint"],
        "no B2 local/recurrent overlap": preflight["kernel_preflight"]["checks"]["b2_local_recurrent_disjoint"],
        "no future recurrent access": preflight["kernel_preflight"]["checks"]["no_future_or_t_minus_1"],
        "same B1 LN/K/V projections": preflight["kernel_preflight"]["checks"]["b1_shared_projection_exact"],
        "same B2 LN/K/V projections": preflight["kernel_preflight"]["checks"]["b2_shared_projection_exact"],
        "separate recurrent softmaxes": preflight["architecture"]["separate_softmaxes"],
        "single c_proj each": preflight["kernel_preflight"]["checks"]["single_c_proj_each"],
        "gate source value exact": preflight["source"]["checks"]["gate_raw"],
        "new B2 gate starts zero": preflight["source"]["checks"]["b2_gate_zero"],
        "frozen 2D2B regression exact": preflight["frozen_2d2b_regression"]["checks"]["parallel"],
        "B2 gate-zero identity exact": preflight["initial_b2_shortening"]["passed"],
        "B1 temporal gradient present": preflight["temporal_gradient_checks"]["b1_finite_nonzero"],
        "B2 temporal gradient present after opening": temporal["b2"][str(MAX_UPDATES)]["nonzero"],
        "B2 temporal gradient reaches 512+": temporal["b2"][str(MAX_UPDATES)]["long_lag_writer_gradient_present"],
        "same-model recurrence only": preflight["architecture"]["forbidden_modules_absent"]["teacher"],
        "CE-only loss": True,
        "pass weights exact": all(pass_weights(update) in (TWO_PASS_WEIGHTS, THREE_PASS_WEIGHTS) for update in range(1, MAX_UPDATES + 1)),
        "Pass-3 cadence exact": [update for update in range(1, MAX_UPDATES + 1) if pass_count(update) == 3] == [32, 64, 96, 128, 160],
        "optimizer resume exact": preflight["source"]["source_optimizer"]["restored_exactly_via_strict_optimizer_load_state_dict"],
        "loader/RNG continuation exact": preflight["checks"]["loader_continuation"],
        "global targets/update 524,288": preflight["checks"]["global_batch"],
        "191 additional optimizer updates": final_model_audit["completed_2d2c_updates"] == 191,
        "100,139,008 additional targets": final_model_audit["processed_2d2c_targets"] == ADDITIONAL_TARGETS,
        "no new projection": preflight["architecture"]["forbidden_modules_absent"]["dedicated_recurrent_projection"],
        "no teacher": preflight["architecture"]["forbidden_modules_absent"]["teacher"],
        "no AttnRes": preflight["architecture"]["forbidden_modules_absent"]["attnres"],
        "no third mirrored link": preflight["architecture"]["forbidden_modules_absent"]["additional_link_beyond_B11_to_B2"],
        "no detached training arm": preflight["architecture"]["forbidden_modules_absent"]["detached_training_arm"],
        "B1 physical KV <=1 historical entry": cache_pass,
        "B2 physical KV <=1 historical entry": cache_pass,
        "B11/B12 recurrent raw-state buffers <=1023": cache_pass,
        "all model/optimizer tensors finite": final_model_audit["model_finite"] and final_model_audit["optimizer_finite"],
        "forced restart exact": restart["passed"],
        "scientific checkpoints exact": set(scientific) == {"48", "96", "143", "191"} and all(row["passed"] for row in scientific.values()),
        "recovery checkpoints exact": set(recovery) == {"50", "100", "150"} and all(row["passed"] for row in recovery.values()),
        "true incremental evaluation completed": incremental["minimum_target_requirement_met"] and incremental["no_complete_prefix_recomputation"],
        "parallel/incremental corrected equivalence": equivalence["passed"],
        "eight-pass stability": composition["finite"],
        "persistent artifacts synchronized": preflight["persistent_workspace_audit"]["passed"],
        "Git synchronized": False,
        "required artifact set complete": False,
    }
    return checks


def run_finalize(args):
    require_git(clean=False)
    dirty = [
        line
        for line in git_output("status", "--porcelain").splitlines()
        if line and OUTPUT_NAME not in line
    ]
    if dirty:
        raise SystemExit(f"finalize has non-result worktree changes: {dirty}")
    require_config()
    workspace_mount_audit(args.output_dir, args.run_root, args.persistent_volume_identity)
    authenticated_stop_audit(args)
    device = require_single_a100()
    output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json")
    require_implementation_fingerprint(preflight)
    training_complete = read_json(output / "training_complete.json")
    if training_complete["completed_2d2c_updates"] != MAX_UPDATES:
        raise SystemExit("training is not complete")
    model, optimizer, final_model_audit = load_final_model(args, device)
    checkpoint_manifest = read_json(output / "checkpoint_manifest.json")
    expected_manifest_sha = checkpoint_manifest["scientific"][str(MAX_UPDATES)]["sha256"]
    if final_model_audit["sha256"] != expected_manifest_sha:
        raise SystemExit("final checkpoint manifest mismatch")
    milestones = read_json(output / "milestone_validation.json")
    if set(milestones) != {str(value) for value in MILESTONES}:
        raise SystemExit("milestone validation set incomplete")
    parallel = milestones[str(MAX_UPDATES)]
    incremental = evaluate_incremental(model, validation_path(args.data_root))
    equivalence = parallel_incremental_equivalence(
        model, validation_path(args.data_root)
    )
    composition = self_composition_diagnostic(
        model, validation_path(args.data_root)
    )
    attention_lags = {
        "b1": read_json(output / "b1_attention_lag_bins.json"),
        "b2": read_json(output / "b2_attention_lag_bins.json"),
    }
    attention_heads = {
        "b1": read_json(output / "b1_attention_head_distance.json"),
        "b2": read_json(output / "b2_attention_head_distance.json"),
    }
    temporal = {
        "b1": read_json(output / "b12_to_b1_temporal_gradient.json"),
        "b2": read_json(output / "b11_to_b2_temporal_gradient.json"),
    }
    final_attention = {
        link: {
            "lag_bins": attention_lags[link][str(MAX_UPDATES)],
            "heads": attention_heads[link][str(MAX_UPDATES)]["heads"],
            "aggregate": attention_heads[link][str(MAX_UPDATES)]["aggregate"],
            "mass_partitions": attention_heads[link][str(MAX_UPDATES)]["mass_partitions"],
        }
        for link in ("b1", "b2")
    }
    training = read_jsonl(output / "training_metrics.jsonl")
    if len(training) != MAX_UPDATES or training[-1]["local_update"] != MAX_UPDATES:
        raise SystemExit("training metrics do not contain exactly 191 updates")
    memory = memory_accounting()
    performance = build_performance(training, milestones, incremental, preflight)
    stable = (
        _numbers_finite(parallel)
        and _numbers_finite(incremental)
        and _numbers_finite(temporal)
        and composition["finite"]
        and all(row["all_parameters_finite"] for row in training)
    )
    provisional_checks = build_integrity_checks(
        preflight,
        final_model_audit,
        incremental,
        equivalence,
        temporal,
        checkpoint_manifest,
        composition,
        output,
    )
    scientific_integrity = all(
        value
        for key, value in provisional_checks.items()
        if key not in {"Git synchronized", "required artifact set complete"}
    )
    classification = classify_result(
        incremental, stable=stable, integrity=scientific_integrity
    )
    recommendation = choose_recommendation(
        classification, parallel, incremental, final_attention["b2"]
    )
    initial_shortening = read_json(output / "initial_b2_shortening.json")
    gate_diagnostics = {
        "source": {
            "g_rec_b1_raw": SOURCE_GATE_RAW,
            "tanh_g_rec_b1": SOURCE_GATE_EFFECTIVE,
            "g_rec_b2_raw": 0.0,
            "tanh_g_rec_b2": 0.0,
        },
        "trajectory": [
            {
                "local_update": row["local_update"],
                "additional_targets": row["additional_targets"],
                "g_rec_b1_raw": row["g_rec_b1_raw"],
                "tanh_g_rec_b1": row["tanh_g_rec_b1"],
                "g_rec_b1_gradient": row["g_rec_b1_gradient_preclip"],
                "g_rec_b2_raw": row["g_rec_b2_raw"],
                "tanh_g_rec_b2": row["tanh_g_rec_b2"],
                "g_rec_b2_gradient": row["g_rec_b2_gradient_preclip"],
            }
            for row in training
        ],
        "final": {
            "g_rec_b1_raw": final_model_audit["g_rec_b1_raw"],
            "tanh_g_rec_b1": final_model_audit["tanh_g_rec_b1"],
            "g_rec_b2_raw": final_model_audit["g_rec_b2_raw"],
            "tanh_g_rec_b2": final_model_audit["tanh_g_rec_b2"],
        },
        "gates_shared": False,
        "gate_cap": None,
        "gate_reset": False,
    }
    summary = {
        "experiment": EXPERIMENT,
        "primary_classification": classification,
        "recommendation": recommendation,
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "parameters": TOTAL_PARAMETERS,
        "new_parameters_vs_2d2b": 1,
        "hardware": "1 × NVIDIA A100-SXM4-80GB",
        "architecture": architecture_manifest(),
        "training": performance["training"],
        "initial_b2_shortening": initial_shortening,
        "validation_trajectory": milestones,
        "parallel": parallel,
        "incremental": incremental,
        "final_attention": final_attention,
        "final_temporal_gradient": {
            link: temporal[link][str(MAX_UPDATES)] for link in ("b1", "b2")
        },
        "memory_accounting": memory,
        "gate_diagnostics": gate_diagnostics,
        "self_composition": composition,
        "parallel_incremental_equivalence": equivalence,
        "final_g_rec_b1_raw": final_model_audit["g_rec_b1_raw"],
        "final_tanh_g_rec_b1": final_model_audit["tanh_g_rec_b1"],
        "final_g_rec_b2_raw": final_model_audit["g_rec_b2_raw"],
        "final_tanh_g_rec_b2": final_model_audit["tanh_g_rec_b2"],
        "final_checkpoint": final_model_audit["checkpoint"],
        "final_checkpoint_sha256": final_model_audit["sha256"],
        "artifact_directory": str(output),
        "git": {
            "branch": BRANCH,
            "implementation_commit": preflight["implementation_git_commit"],
            "results_commit": None,
        },
        "pod": {
            "id": args.pod_id,
            "name": args.pod_name,
            "status": "RUNNING_PENDING_FINAL_SYNC_AND_STOP",
            "exact_stop_command": f"runpodctl pod stop {args.pod_id} -o json",
        },
        "stability_passed": stable,
        "scientific_integrity_passed": scientific_integrity,
    }
    questions = build_questions(
        summary,
        milestones,
        incremental,
        final_attention,
        {link: temporal[link][str(MAX_UPDATES)] for link in ("b1", "b2")},
        {},
        memory,
    )
    position_metrics = {
        "parallel": parallel["position_bins"],
        "true_incremental": incremental_position_bins(incremental),
    }
    paired_controls = {
        "parallel": {
            "new_real_vs_b2_off": parallel["new_real_vs_b2_off"],
            "new_real_vs_b2_shuffled": parallel["new_real_vs_b2_shuffled"],
            "new_real_vs_b2_full": parallel["new_real_vs_b2_full"],
        },
        "true_incremental": {
            key: incremental[key]
            for key in (
                "both_real_vs_b2_off_batches",
                "both_real_vs_b2_shuffled_batches",
                "both_real_vs_b2_full_batches",
                "both_real_vs_b2_off_sequences",
                "both_real_vs_b2_shuffled_sequences",
                "both_real_vs_b2_full_sequences",
            )
        },
    }
    cache_audit = {
        "controls": {
            name: row["cache_rows"] for name, row in incremental["controls"].items()
        },
        "B1_historical_KV_limit": 1,
        "B2_historical_KV_limit": 1,
        "B11_raw_recurrent_state_limit": 1023,
        "B12_raw_recurrent_state_limit": 1023,
        "B3_B12_historical_KV_limit": 1023,
        "no_hidden_full_B1_or_B2_cache": all(
            cache["final"]["b1_historical_kv"] <= 1
            and cache["final"]["b2_historical_kv"] <= 1
            for cache in incremental["controls"]["both_real"]["cache_rows"]
        ),
    }
    commands = {
        "experiment": EXPERIMENT,
        "preflight_command": preflight["command"],
        "smoke_command": read_json(output / "smoke_audit.json")["command"],
        "training_segments": read_json(output / "process_segments.json"),
        "finalize_command": " ".join(sys.argv),
        "finalized_at": time.time(),
        "planned_terminal_stop_command": f"runpodctl pod stop {args.pod_id} -o json",
        "runpod_stop_capability_audit": preflight["runpod_stop_audit"],
        "environment": environment_payload(),
    }

    durable_json(output / "incremental_validation.json", incremental)
    durable_json(output / "incremental_cache_audit.json", cache_audit)
    durable_json(output / "memory_accounting.json", memory)
    durable_json(output / "position_bin_metrics.json", position_metrics)
    durable_json(output / "paired_controls.json", paired_controls)
    durable_json(output / "gate_diagnostics.json", gate_diagnostics)
    durable_json(output / "performance.json", performance)
    durable_json(output / "self_composition.json", composition)
    durable_json(output / "parallel_incremental_equivalence.json", equivalence)
    durable_json(output / "scientific_questions.json", questions)
    durable_json(output / "commands_and_runtime.json", commands)
    durable_json(output / "result_summary.json", summary)
    make_plots(
        output,
        milestones,
        training,
        incremental,
        attention_lags,
        temporal,
        performance,
    )
    audit = {
        "experiment": EXPERIMENT,
        "timestamp": time.time(),
        "checks": provisional_checks,
        "artifact_inventory": {},
        "stability_passed": stable,
        "scientific_integrity_passed": scientific_integrity,
        "passed": False,
        "final_checkpoint": final_model_audit,
    }
    durable_json(output / "FINAL_AUDIT.json", audit)
    report = render_report(summary, audit, questions)
    durable_text(output / "EXPERIMENT_2D2C_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nPending lifecycle action: commit/push, verify synchronization, then stop the exact GPU pod without deleting it.\n",
    )
    durable_json(
        output / "HEARTBEAT.json",
        {
            "experiment": EXPERIMENT,
            "timestamp": time.time(),
            "status": "FINALIZED_PENDING_GIT_SEAL_AND_POD_STOP",
            "local_update": MAX_UPDATES,
            "additional_targets": ADDITIONAL_TARGETS,
            "final_checkpoint": final_model_audit,
        },
    )
    inventory = build_artifact_inventory(output)
    audit["artifact_inventory"] = inventory
    audit["checks"]["required artifact set complete"] = inventory["passed"]
    audit["passed"] = all(audit["checks"].values())
    durable_json(output / "artifact_inventory.json", inventory)
    durable_json(output / "FINAL_AUDIT.json", audit)
    report = render_report(summary, audit, questions)
    durable_text(output / "EXPERIMENT_2D2C_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nPending lifecycle action: commit/push, verify synchronization, then stop the exact GPU pod without deleting it.\n",
    )
    if not scientific_integrity or not inventory["passed"]:
        raise SystemExit(
            f"2D2C finalize integrity failed: scientific={scientific_integrity} inventory={inventory['passed']}"
        )
    print("EXPERIMENT_2D2C_FINALIZED_PENDING_GIT_SEAL", flush=True)
    return summary


def run_seal_report(args):
    require_git(clean=True)
    output = Path(args.output_dir).resolve()
    expected_output = (REPO_ROOT / "results" / OUTPUT_NAME).resolve()
    if output != expected_output:
        raise SystemExit(f"seal-report output must be exactly {expected_output}")
    preflight = read_json(output / "preflight_audit.json")
    require_implementation_fingerprint(preflight)
    if git_output("rev-parse", "HEAD") != args.results_commit:
        raise SystemExit("seal-report HEAD must equal supplied results commit")
    if git_output("rev-parse", f"origin/{BRANCH}") != args.results_commit:
        raise SystemExit("results commit must be pushed before sealing")
    summary = read_json(output / "result_summary.json")
    audit = read_json(output / "FINAL_AUDIT.json")
    questions = read_json(output / "scientific_questions.json")
    commands = read_json(output / "commands_and_runtime.json")
    commands["seal_report_command"] = " ".join(sys.argv)
    commands["post_seal_external_actions"] = [
        "commit and push sealed report artifacts",
        "verify local/origin/pod commit equality and clean worktrees",
        "verify no scientific process",
        "runpodctl pod stop e8nd7m6piw5km2 -o json",
        "verify stopped and not deleted",
    ]
    summary["git"]["results_commit"] = args.results_commit
    summary["git"]["report_base_commit"] = git_output("rev-parse", "HEAD")
    audit["checks"]["Git synchronized"] = True
    inventory = build_artifact_inventory(output)
    audit["checks"]["required artifact set complete"] = inventory["passed"]
    audit["artifact_inventory"] = inventory
    audit["passed"] = all(audit["checks"].values())
    if not audit["passed"]:
        raise SystemExit(f"sealed final audit failed: {audit['checks']}")
    summary["pod"]["status"] = "READY_TO_STOP_AFTER_SEAL_COMMIT_PUSH"
    durable_json(output / "commands_and_runtime.json", commands)
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_json(output / "artifact_inventory.json", inventory)
    report = render_report(summary, audit, questions)
    durable_text(output / "EXPERIMENT_2D2C_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nFinal RunPod stop remains the only lifecycle action after the seal commit is pushed.\n",
    )
    print("EXPERIMENT_2D2C_REPORT_SEALED", flush=True)


def add_execution_arguments(parser):
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--persistent-volume-identity", required=True)
    parser.add_argument("--stop-mechanism", required=True)
    parser.add_argument("--stop-authenticated", action="store_true")
    parser.add_argument("--stop-audit-path", required=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("preflight", run_preflight),
        ("smoke", run_smoke),
        ("train", run_train),
        ("finalize", run_finalize),
    ):
        current = subparsers.add_parser(name)
        add_execution_arguments(current)
        if name == "train":
            current.add_argument("--end-update", required=True, type=int)
            current.add_argument("--resume")
        if name == "finalize":
            current.add_argument("--final-checkpoint", required=True)
        current.set_defaults(function=function)
    seal = subparsers.add_parser("seal-report")
    seal.add_argument("--output-dir", required=True)
    seal.add_argument("--results-commit", required=True)
    seal.set_defaults(function=run_seal_report)
    return parser


def main():
    args = build_parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    main()
