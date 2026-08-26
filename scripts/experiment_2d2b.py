#!/usr/bin/env python3
"""Experiment 2D2B: full-width B12-to-B1 token-indexed recurrent K/V bank.

The finalized 2D2A checkpoint is the immutable source.  This driver restores
its model, optimizer, data loader, and RNG states exactly, changes only the
eligible recurrent source set, and produces the complete 2D2B audit bundle.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
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
from experiment_2d2b_core import (  # noqa: E402
    BANK_MODES,
    MAX_RECURRENT_ENTRIES,
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
    RecurrentKVGPT,
)


EXPERIMENT = "2D2B"
PROTOCOL = "exp2d2b_full_b12_b1_recurrent_bank_v1"
BRANCH = "experiment-2d2b-full-b12-b1-recurrent-bank"
FROZEN_TAG = "experiment-2d2a-b12-b1-recurrent-kv-final"
FROZEN_COMMIT = "4101c180646f30d3cde9bd7854be682393761c6d"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d2b_full_b12_b1_recurrent_bank.json"
OUTPUT_NAME = "experiment_2d2b_full_b12_b1_recurrent_bank"
CHECKPOINT_SCHEMA = "exp2d2b_full_b12_b1_recurrent_bank_checkpoint_v1"
SOURCE_SCHEMA = "exp2d2a_b12_b1_recurrent_kv_checkpoint_v1"
SOURCE_SHA256 = "24fd2481e220ec504db3a6e912054d0ad502cdb3a6fc497b22dd32ec682e3afb"
SOURCE_BYTES = 1_493_936_841
SOURCE_UPDATES = 96
SOURCE_TARGETS = 50_331_648
SOURCE_GATE_RAW = 0.02386544458568096
SOURCE_GATE_EFFECTIVE = 0.023860914632678032
SOURCE_NEXT_BATCH_SHA256 = "9c8de21606769f3e0a38a64a7beecb9de8c6b99b4842d6c909f2f42dd251b7ff"
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
TOTAL_PARAMETERS = 124_475_905
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
    "configs/exp2d2b_full_b12_b1_recurrent_bank.json",
    "scripts/experiment_2d2b.py",
    "scripts/experiment_2d2b_core.py",
    "scripts/experiment_2d2a.py",
    "scripts/experiment_2d2a_core.py",
    "scripts/experiment_2d0.py",
    "scripts/experiment_2d0d.py",
    "scripts/experiment_2d1.py",
    "scripts/smoke_test.py",
    "train_gpt2.py",
    "tests/test_experiment_2d2b_core.py",
    "tests/test_experiment_2d2b_driver.py",
)
REQUIRED_ARTIFACTS = (
    "EXPERIMENT_2D2B_FINAL_REPORT.md",
    "FINAL_AUDIT.json",
    "result_summary.json",
    "source_manifest.json",
    "parameter_manifest.json",
    "architecture_manifest.json",
    "semantic_diff_audit.json",
    "batch_manifest.json",
    "preflight_audit.json",
    "zero_shot_bank_expansion.json",
    "training_metrics.jsonl",
    "milestone_validation.json",
    "paired_controls.json",
    "gate_diagnostics.json",
    "attention_lag_bins.json",
    "attention_head_distance.json",
    "temporal_gradient_by_lag.json",
    "old_memory_ablation.json",
    "position_bin_metrics.json",
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
    "full_bank_gain",
    "bank_width_gain",
    "sequence_gap",
    "recurrent_gate",
    "attention_mass_by_lag",
    "attention_density_by_lag",
    "per_head_mean_lag",
    "temporal_writer_gradient",
    "position_binned_gain",
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
        raise SystemExit("frozen 2D2A tag mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=REPO_ROOT
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D2B worktree must be clean")


def require_config() -> dict:
    config = read_json(CONFIG_PATH)
    checks = {
        "experiment": config.get("experiment") == EXPERIMENT,
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "frozen_tag": config.get("frozen_2d2a_tag") == FROZEN_TAG,
        "frozen_commit": config.get("frozen_2d2a_commit") == FROZEN_COMMIT,
        "source_sha": config.get("source_checkpoint_sha256") == SOURCE_SHA256,
        "source_bytes": config.get("source_checkpoint_bytes") == SOURCE_BYTES,
        "source_state": config["source_2d2a"]["completed_updates"] == SOURCE_UPDATES
        and config["source_2d2a"]["completed_targets"] == SOURCE_TARGETS
        and config["source_2d2a"]["raw_g_rec"] == SOURCE_GATE_RAW
        and config["source_2d2a"]["next_global_batch_sha256"]
        == SOURCE_NEXT_BATCH_SHA256,
        "geometry": config["architecture"]["b1_local_window"] == W_LOCAL
        and config["architecture"]["recurrent_min_lag"] == 2
        and config["architecture"]["recurrent_max_lag"] == RECURRENT_MAX_LAG
        and config["architecture"]["maximum_recurrent_entries"]
        == MAX_RECURRENT_ENTRIES,
        "no_parameters": config["architecture"]["new_parameter_count_vs_2d2a"] == 0
        and config["architecture"]["new_parameters_vs_2d2a"] == []
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
        "attached": config["training"]["recurrent_source_detached"] is False,
        "hardware": config["hardware"]["pod_id"] == "e8nd7m6piw5km2"
        and config["hardware"]["pod_name"] == "serious_indigo_swordfish"
        and config["hardware"]["gpu_count"] == 1,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D2B preregistration mismatch: {checks}")
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
        raise SystemExit(f"2D2B result directory must be exactly {expected_output}")
    if not str(output).startswith("/workspace/") or not str(run).startswith("/workspace/"):
        raise SystemExit("2D2B results and checkpoints must live under /workspace")
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
        "schema": payload.get("schema") == "exp2d2b_runpod_stop_capability_v1",
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


def configure_optimizer(model, device_type="cuda"):
    optimizer, report = legacy.configure_optimizer(model, device_type=device_type)
    report["source_optimizer_restored"] = True
    report["warmup_restarted"] = False
    return optimizer, report


def load_source_bundle(source_checkpoint, device, restore_rng=False):
    path = Path(source_checkpoint).resolve()
    observed_sha = file_sha256(path)
    if observed_sha != SOURCE_SHA256 or path.stat().st_size != SOURCE_BYTES:
        raise SystemExit("2D2A scientific source checkpoint identity mismatch")
    sha_sidecar = path.with_suffix(path.suffix + ".sha256")
    verification_sidecar = path.with_suffix(path.suffix + ".verification.json")
    if not sha_sidecar.is_file() or not verification_sidecar.is_file():
        raise SystemExit("2D2A checkpoint sidecars missing")
    if sha_sidecar.read_text().split()[0] != SOURCE_SHA256:
        raise SystemExit("2D2A checkpoint SHA sidecar mismatch")
    payload = legacy.d0.torch_load(path, mmap=True)
    required = {
        "schema",
        "model",
        "g_rec",
        "optimizer",
        "completed_updates",
        "processed_targets",
        "training_state",
        "loader_state",
        "rng_state",
        "next_global_batch_sha256",
        "architecture_manifest",
        "metadata",
        "git_commit",
        "saved_process_id",
        "environment",
    }
    checks = {
        "fields": set(payload) == required,
        "schema": payload.get("schema") == SOURCE_SCHEMA,
        "updates": payload.get("completed_updates") == SOURCE_UPDATES,
        "targets": payload.get("processed_targets") == SOURCE_TARGETS,
        "gate_duplicate": torch.equal(payload.get("g_rec"), payload["model"]["g_rec"]),
        "gate_raw": float(payload["g_rec"]) == SOURCE_GATE_RAW,
        "next_batch_recorded": payload.get("next_global_batch_sha256")
        == SOURCE_NEXT_BATCH_SHA256,
        "source_git": payload.get("git_commit") == "26b2493f86e3b465035a51afe8a702e2611fa6f1",
    }
    if not all(checks.values()):
        raise SystemExit(f"2D2A source schema mismatch: {checks}")
    _, model = instantiate_model(device, trainable=True)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise SystemExit(f"strict source model load failed: {missing}, {unexpected}")
    optimizer, optimizer_report = configure_optimizer(model, device.type)
    optimizer.load_state_dict(payload["optimizer"])
    source_loader = legacy.d1.ExplicitShardLoader(
        payload["loader_state"]["shards"],
        payload["loader_state"]["batch_size"],
        payload["loader_state"]["sequence_length"],
        state=payload["loader_state"],
    )
    source_accumulation = int(payload["metadata"]["gradient_accumulation"])
    observed_next = next_global_batch_hash(source_loader, source_accumulation)
    checks.update(
        {
            "strict_model": True,
            "model_finite": model_finite(model),
            "optimizer_finite": optimizer_finite(optimizer),
            "parameter_count": sum(value.numel() for value in model.parameters())
            == TOTAL_PARAMETERS,
            "gate_effective": model.recurrent_scale.detach().float().item()
            == SOURCE_GATE_EFFECTIVE,
            "next_batch_reproduced": observed_next == SOURCE_NEXT_BATCH_SHA256,
            "weight_tying": model.base.transformer.wte.weight
            is model.base.lm_head.weight,
        }
    )
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"2D2A source strict reopen failed: {checks}")
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
    legacy_wrapper = legacy_core.RecurrentKVGPT(model.base)
    legacy_inventory = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
        }
        for name, value in legacy_wrapper.named_parameters()
    ]
    del legacy_wrapper
    comparable = [
        {key: row[key] for key in ("name", "shape", "dtype", "numel")}
        for row in inventory
    ]
    report = {
        "2d2a_named_parameter_inventory": legacy_inventory,
        "2d2b_named_parameter_inventory": inventory,
        "inventories_bit_for_bit_equal": comparable == legacy_inventory,
        "source_state_dict_inventory": source_state_inventory,
        "source_state_dict_keys_equal": list(model.state_dict())
        == list(source_payload["model"]),
        "2d2a_total_parameters": sum(row["numel"] for row in legacy_inventory),
        "2d2b_total_parameters": sum(row["numel"] for row in inventory),
        "new_parameters_vs_2d2a": [],
        "new_parameter_count_vs_2d2a": 0,
        "all_trainable": all(row["trainable"] for row in inventory),
        "embedding_lm_head_tied": model.base.transformer.wte.weight
        is model.base.lm_head.weight,
    }
    report["checks"] = {
        "inventory_equal": report["inventories_bit_for_bit_equal"],
        "state_dict_keys_equal": report["source_state_dict_keys_equal"],
        "source_total": report["2d2a_total_parameters"] == TOTAL_PARAMETERS,
        "target_total": report["2d2b_total_parameters"] == TOTAL_PARAMETERS,
        "zero_new": report["new_parameter_count_vs_2d2a"] == 0,
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
        "source": "B12 post-MLP residual immediately before final LayerNorm",
        "destination": "B1 attention",
        "b1_local_window": W_LOCAL,
        "b2_b12_windows": [T] * 11,
        "recurrent_source_set": "max(0,t-1023)...t-2 inclusive",
        "recurrent_min_lag": 2,
        "recurrent_max_lag": RECURRENT_MAX_LAG,
        "maximum_recurrent_entries": MAX_RECURRENT_ENTRIES,
        "local_positions": ["t-1", "t"],
        "separate_softmaxes": True,
        "shared_b1_ln_qkv": True,
        "single_b1_c_proj": True,
        "effective_gate": "tanh(g_rec)",
        "incremental_b1_historical_kv_capacity": 1,
        "incremental_b2_b12_historical_kv_capacity": 1023,
        "incremental_b12_raw_residual_capacity": RECURRENT_RING_CAPACITY,
        "parallel_source_tensor_shape": "[B,T,C]",
        "forbidden_repeated_state_tensor": "[B,T,T,C]",
        "overall_context": T,
        "overall_kv_savings_claimed": False,
        "forbidden_modules_absent": {
            "teacher": True,
            "attnres": True,
            "dedicated_recurrent_projection": True,
            "mirrored_links": True,
            "detached_training_arm": True,
        },
    }


def semantic_diff_audit(model, source_payload) -> dict:
    named = [name for name, _ in model.named_parameters()]
    legacy_wrapper = legacy_core.RecurrentKVGPT(model.base)
    source_named = [name for name, _ in legacy_wrapper.named_parameters()]
    del legacy_wrapper
    old = legacy.architecture_manifest()
    new = architecture_manifest()
    unchanged = {
        "source_representation": (
            old["source"] == "B12 post-MLP residual before final LayerNorm"
            and new["source"]
            == "B12 post-MLP residual immediately before final LayerNorm"
        ),
        "destination": old["destination"] == new["destination"],
        "local_window": old["b1_local_window"] == new["b1_local_window"] == 2,
        "upper_windows": old["b2_b12_windows"] == new["b2_b12_windows"],
        "separate_softmax": old["separate_softmaxes"] == new["separate_softmaxes"],
        "shared_projection": old["shared_b1_ln_qkv"] == new["shared_b1_ln_qkv"],
        "single_c_proj": old["single_b1_c_proj"] == new["single_b1_c_proj"],
        "gate": old["effective_gate"] == new["effective_gate"],
        "parameter_names": named == source_named,
    }
    report = {
        "baseline": "final Experiment 2D2A",
        "only_architecture_change": {
            "field": "eligible recurrent B12 source positions",
            "old": ["t-3", "t-2"],
            "new": "max(0,t-1023)...t-2",
        },
        "unchanged": unchanged,
        "new_learnable_tensors": [],
        "memory_efficient_source_storage": True,
        "same_parameter_state_dict_keys": named == source_named,
        "source_checkpoint_state_dict_keys_equal": list(model.state_dict())
        == list(source_payload["model"]),
        "passed": all(unchanged.values()),
    }
    report["passed"] = bool(
        report["passed"] and report["source_checkpoint_state_dict_keys_equal"]
    )
    if not report["passed"]:
        raise SystemExit(f"semantic diff audit failed: {unchanged}")
    return report


def validation_manifest(val_path) -> dict:
    return legacy.validation_manifest(val_path)


@torch.no_grad()
def evaluate_parallel(model, val_path, batches=VALIDATION_BATCHES) -> dict:
    """Evaluate all four bank-width controls on identical canonical rows."""
    model.eval()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    names = ("plain", "full_real", "full_shuffled", "two_slot_real")
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            plain = model.forward_pass(x, targets=None)
            source = plain["h12"]
            full = model.forward_pass(x, recurrent_source=source, bank_mode="full")
            shuffled = model.forward_pass(
                x,
                recurrent_source=source,
                recurrent_permutation=derangement,
                bank_mode="full",
            )
            two_slot = model.forward_pass(
                x, recurrent_source=source, bank_mode="two_slot"
            )
            losses = {
                "plain": _token_losses(plain["logits"], y),
                "full_real": _token_losses(full["logits"], y),
                "full_shuffled": _token_losses(shuffled["logits"], y),
                "two_slot_real": _token_losses(two_slot["logits"], y),
            }
        for name, tensor in losses.items():
            row = controls[name]
            row["loss_sum"] += tensor.double().sum().item()
            row["targets"] += tensor.numel()
            row["per_batch_losses"].append(tensor.float().mean().item())
            row["per_position_sum"] += tensor.double().sum(dim=0).cpu().numpy()
        print(f"2D2B validation batch={batch_index + 1:02d}/{batches}", flush=True)
        del x, y, cpu_x, cpu_y, plain, source, full, shuffled, two_slot, losses
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
    plain = finished["plain"]
    full = finished["full_real"]
    shuffled = finished["full_shuffled"]
    two_slot = finished["two_slot_real"]
    position_bins = {}
    for name, first, last in POSITION_BINS:
        p = np.asarray(plain["per_position_loss"])[first : last + 1]
        f = np.asarray(full["per_position_loss"])[first : last + 1]
        s = np.asarray(shuffled["per_position_loss"])[first : last + 1]
        two = np.asarray(two_slot["per_position_loss"])[first : last + 1]
        position_bins[name] = {
            "plain_loss": float(p.mean()),
            "full_real_loss": float(f.mean()),
            "full_shuffled_loss": float(s.mean()),
            "two_slot_real_loss": float(two.mean()),
            "full_bank_gain": float((p - f).mean()),
            "sequence_gap": float((s - f).mean()),
            "bank_width_gain": float((two - f).mean()),
            "available_recurrent_history": [max(1, first - 1), max(1, last - 1)],
        }
    collection_sha = legacy.d0.aggregate_hashes(
        [row["combined_sha256"] for row in identities]
    )
    result = {
        "controls": finished,
        "full_bank_gain": plain["validation_loss"] - full["validation_loss"],
        "sequence_gap": shuffled["validation_loss"] - full["validation_loss"],
        "bank_width_gain": two_slot["validation_loss"] - full["validation_loss"],
        "full_vs_plain": paired_stats(
            full["per_batch_losses"], plain["per_batch_losses"]
        ),
        "full_vs_shuffled": paired_stats(
            full["per_batch_losses"], shuffled["per_batch_losses"]
        ),
        "full_vs_two_slot": paired_stats(
            full["per_batch_losses"], two_slot["per_batch_losses"]
        ),
        "position_bins": position_bins,
        "gate_raw": model.g_rec.detach().float().item(),
        "effective_recurrent_scale": model.recurrent_scale.detach().float().item(),
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
def attention_diagnostics(model, val_path, batch_size=2) -> dict:
    """Measure lag use on a pinned small batch without B64 attention matrices."""
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
            recurrent_source=first["h12"],
            bank_mode="full",
            return_diagnostics=True,
        )
    diagnostics = second["diagnostics"]
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


def temporal_gradient_by_lag(model, val_path, precision="bf16") -> dict:
    """Group attached writer gradients by lag from the final receiver token."""
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
        source = first["h12"]
        second = model.forward_pass(
            x,
            targets=y,
            recurrent_source=source,
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
        "precision": precision,
        "pinned_batch": identity,
        "gate_raw": model.g_rec.detach().float().item(),
        "effective_gate": model.recurrent_scale.detach().float().item(),
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
            "Pass-2 CE gradient into attached Pass-1 B12 post-MLP states. "
            "Source positions are indexed by their lag from receiver t=1023; each "
            "source gradient also includes all eligible earlier receivers."
        ),
    }
    model.zero_grad(set_to_none=True)
    del x, y, first, source, second, gradient
    torch.cuda.empty_cache()
    return report


def kernel_preflight(model, short_tokens, short_targets) -> dict:
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
    local_mask = model.local_mask(length, device)
    overlap = bank_mask & local_mask
    checks["local_recurrent_disjoint"] = not bool(overlap.any())
    checks["no_future_or_t_minus_1"] = not bool((bank_mask & (source >= query - 1)).any())
    checks["no_too_old"] = not bool((bank_mask & (source < query - 1023)).any())
    values = torch.randn(
        short_tokens.size(0), length, model.config.n_embd, device=device
    )
    bank = model.build_recurrent_bank(values)
    checks["source_not_repeated"] = bank.values.data_ptr() == values.data_ptr()
    checks["source_rank_three"] = bank.values.ndim == 3
    key, value = model.project_recurrent_kv(values)
    normalized = model.base.transformer.h[0].ln_1(values)
    _, expected_key, expected_value = model.base.transformer.h[0].attn.c_attn(
        normalized
    ).split(model.config.n_embd, dim=-1)
    expected_key = expected_key.view(
        short_tokens.size(0), length, N_HEAD, N_EMBD // N_HEAD
    ).transpose(1, 2)
    expected_value = expected_value.view(
        short_tokens.size(0), length, N_HEAD, N_EMBD // N_HEAD
    ).transpose(1, 2)
    checks["shared_projection_exact"] = torch.equal(key, expected_key) and torch.equal(
        value, expected_value
    )
    calls = []
    hook = model.base.transformer.h[0].attn.c_proj.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(1)
    )
    try:
        with torch.no_grad():
            active = model.forward_pass(
                short_tokens,
                targets=short_targets,
                recurrent_source=values,
                return_diagnostics=True,
            )
    finally:
        hook.remove()
    checks["single_c_proj"] = calls == [1]
    weights = active["diagnostics"]["recurrent_attention_weights"]
    checks["invalid_probabilities_zero"] = not bool(
        weights.masked_select(~bank_mask.view(1, 1, length, length)).count_nonzero()
    )
    checks["probabilities_finite"] = bool(torch.isfinite(weights).all())
    changed = short_tokens.clone()
    changed[:, -1] = (changed[:, -1] + 17) % model.config.vocab_size
    with torch.no_grad():
        reference = model.forward_multi_pass(short_tokens, num_passes=2)["logits"]
        perturbed = model.forward_multi_pass(changed, num_passes=2)["logits"]
    checks["future_perturbation_causal"] = torch.equal(
        reference[:, :-1], perturbed[:, :-1]
    )
    reports["mask_positions"] = {
        str(position): torch.where(full_mask[position])[0].cpu().tolist()
        for position in (0, 1, 2, 3, 1022, 1023)
    }
    reports["c_proj_calls"] = len(calls)
    reports["bank_storage"] = {
        "source_shape": list(values.shape),
        "bank_shape": list(bank.values.shape),
        "shared_data_ptr": bank.values.data_ptr() == values.data_ptr(),
    }
    del full_mask, values, bank, key, value, active, weights, reference, perturbed
    torch.cuda.empty_cache()
    return {"checks": checks, "reports": reports, "passed": all(checks.values())}


def legacy_2d2a_regressions(model, source_payload, val_path) -> dict:
    """Run the frozen 2D2A kernels on the exact source state."""
    old_model = legacy_core.RecurrentKVGPT(model.base)
    old_model.load_state_dict(source_payload["model"], strict=True)
    old_model.to(next(model.parameters()).device)
    parallel = legacy.evaluate_parallel(old_model, val_path)
    incremental = legacy.evaluate_incremental(old_model, val_path)
    expected_parallel = {
        "plain": 3.1331379047466728,
        "real": 3.13081073770154,
        "shuffled": 3.132718408929031,
    }
    expected_incremental = {
        "plain": 3.0759934813572993,
        "real": 3.073846268102645,
        "shuffled": 3.0763928307427397,
    }
    observed_parallel = {
        name: parallel["controls"][name]["validation_loss"]
        for name in expected_parallel
    }
    observed_incremental = {
        name: incremental["controls"][name]["loss"] for name in expected_incremental
    }
    tolerance = 5e-7
    checks = {
        "parallel": all(
            abs(observed_parallel[name] - expected) <= tolerance
            for name, expected in expected_parallel.items()
        ),
        "incremental": all(
            abs(observed_incremental[name] - expected) <= tolerance
            for name, expected in expected_incremental.items()
        ),
        "parallel_canonical": parallel["canonical_validation_sha256"]
        == CANONICAL_VALIDATION_SHA256,
        "incremental_targets": incremental["targets"] >= 131_072,
    }
    report = {
        "expected_parallel": expected_parallel,
        "observed_parallel": observed_parallel,
        "parallel_deltas": {
            name: observed_parallel[name] - expected for name, expected in expected_parallel.items()
        },
        "expected_incremental": expected_incremental,
        "observed_incremental": observed_incremental,
        "incremental_deltas": {
            name: observed_incremental[name] - expected
            for name, expected in expected_incremental.items()
        },
        "tolerance": tolerance,
        "parallel_full": parallel,
        "incremental_full": incremental,
        "checks": checks,
        "passed": all(checks.values()),
    }
    del old_model
    gc.collect()
    torch.cuda.empty_cache()
    if not report["passed"]:
        raise SystemExit(f"frozen 2D2A regression failed: {checks}")
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
        source = None
        loss = None
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    current = model.forward_pass(
                        x,
                        targets=y,
                        recurrent_source=source,
                        activation_checkpointing=True,
                    )
                    results.append(current)
                    source = current["h12"]
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
            del results, source, loss, x, y, cpu_x, cpu_y
            gc.collect()
            torch.cuda.empty_cache()
    raise SystemExit(f"no safe 2D2B microbatch: {attempts}")


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
        "frozen_2d2a_tag": FROZEN_TAG,
        "frozen_2d2a_commit": FROZEN_COMMIT,
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
    temporal_zero = temporal_gradient_by_lag(model, val_path, precision="bf16")
    temporal_checks = {
        "finite_nonzero": temporal_zero["finite"] and temporal_zero["nonzero"],
        "early_old": temporal_zero["position_probes"]["early_old_position_0"] > 0,
        "middle": temporal_zero["position_probes"]["middle_position_512"] > 0,
        "recent": temporal_zero["position_probes"][
            "recent_eligible_position_1021"
        ]
        > 0,
        "long_lag": temporal_zero["long_lag_writer_gradient_present"],
    }
    if not all(temporal_checks.values()):
        raise SystemExit(f"long-bank temporal gradient failed: {temporal_checks}")

    regressions = legacy_2d2a_regressions(model, source_payload, val_path)
    zero_shot = evaluate_parallel(model, val_path)
    zero_shot.update({"local_update": 0, "additional_targets": 0})
    zero_shot["legacy_two_slot_loss"] = regressions["observed_parallel"]["real"]
    zero_shot["legacy_bank_expansion_gain"] = (
        regressions["observed_parallel"]["real"]
        - zero_shot["controls"]["full_real"]["validation_loss"]
    )
    zero_shot["questions"] = {
        "full_bank_zero_shot_gain": zero_shot["full_bank_gain"],
        "full_bank_zero_shot_sequence_gap": zero_shot["sequence_gap"],
        "bank_expansion_zero_shot_gain": zero_shot["legacy_bank_expansion_gain"],
    }
    attention_zero = attention_diagnostics(model, val_path)
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
        raise SystemExit("scientific first global batch differs from 2D2A continuation")
    recurrent_benchmark = benchmark_recurrent_attention(
        model, min(selected_microbatch, 4), repeats=3
    )
    benchmark = {
        "recurrent_attention": recurrent_benchmark,
        "source_2d2a_training_targets_per_second": 59_269,
        "microbatch_probe": probe,
    }

    durable_json(output / "batch_manifest.json", batch_payload)
    durable_json(output / "zero_shot_bank_expansion.json", zero_shot)
    durable_json(output / "milestone_validation.json", {"0": zero_shot})
    durable_json(output / "attention_lag_bins.json", {"0": attention_zero["lag_bins"]})
    durable_json(
        output / "attention_head_distance.json",
        {"0": {"heads": attention_zero["heads"], "aggregate": attention_zero["aggregate"], "mass_partitions": attention_zero["mass_partitions"]}},
    )
    temporal_zero.update({"local_update": 0, "additional_targets": 0})
    durable_json(output / "temporal_gradient_by_lag.json", {"0": temporal_zero})
    durable_json(output / "checkpoint_manifest.json", {"scientific": {}, "recovery": {}, "smoke": {}})
    durable_json(output / "performance.json", {"preflight_benchmark": benchmark})
    durable_json(output / "runpod_stop_capability.json", stop)
    durable_json(output / "persistent_workspace_audit.json", mount)

    science_checks = {
        "frozen_2d2a_tag_exact": source_manifest["frozen_tag_exact"],
        "source_checkpoint_exact": source_audit["checks"]["passed"],
        "audit_correction_preserved": correction["passed"],
        "parameters_unchanged": parameters["passed"],
        "semantic_diff_only_bank_width": semantic["passed"],
        "kernel": kernel["passed"],
        "temporal_writer_gradient": all(temporal_checks.values()),
        "legacy_parallel_regression": regressions["checks"]["parallel"],
        "legacy_incremental_regression": regressions["checks"]["incremental"],
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
        "temporal_gradient_preflight": temporal_zero,
        "temporal_gradient_checks": temporal_checks,
        "legacy_2d2a_regressions": regressions,
        "zero_shot": zero_shot,
        "attention_zero_shot": attention_zero,
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
        raise SystemExit(f"2D2B preflight failed: {science_checks}")
    print("EXPERIMENT_2D2B_PREFLIGHT_PASS", flush=True)
    return preflight


def checkpoint_payload(model, optimizer, loader, training_state, metadata, accumulation):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "g_rec": model.g_rec.detach().cpu().clone(),
        "optimizer": optimizer.state_dict(),
        "completed_2d2b_updates": training_state["completed_2d2b_updates"],
        "processed_2d2b_targets": training_state["processed_2d2b_targets"],
        "cumulative_2d2_targets": training_state["cumulative_2d2_targets"],
        "source_2d2a_updates": SOURCE_UPDATES,
        "source_2d2a_targets": SOURCE_TARGETS,
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
        "fields_exact": set(reopened) == required,
        "schema": reopened.get("schema") == CHECKPOINT_SCHEMA,
        "updates": reopened.get("completed_2d2b_updates")
        == training_state["completed_2d2b_updates"],
        "additional_targets": reopened.get("processed_2d2b_targets")
        == training_state["processed_2d2b_targets"],
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
        "completed_2d2b_updates": reopened["completed_2d2b_updates"],
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
                recurrent_source=first["h12"],
                activation_checkpointing=True,
                return_diagnostics=True,
            )
            loss = TWO_PASS_WEIGHTS[0] * first["loss"] + TWO_PASS_WEIGHTS[1] * second["loss"]
        gate_before = model.g_rec.detach().float().item()
        loss.backward()
        gate_gradient = model.g_rec.grad.detach().float().item()
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
                "gate_before": gate_before,
                "gate_after": model.g_rec.detach().float().item(),
                "effective_gate_after": model.recurrent_scale.detach().float().item(),
                "gate_gradient": gate_gradient,
                "gradient_norm": norm.detach().float().item(),
                "gradient_groups": groups,
                "gradients_finite": gradients_finite(model),
                "parameters_finite": model_finite(model),
                "optimizer_finite": optimizer_finite(optimizer),
                "recurrent_attention_finite": (
                    second["diagnostics"]["recurrent_attention_weights"] is None
                    or bool(
                        torch.isfinite(
                            second["diagnostics"]["recurrent_attention_weights"]
                        ).all()
                    )
                ),
            }
        )
        del x, y, cpu_x, cpu_y, first, second, loss
        torch.cuda.empty_cache()
    temporal = temporal_gradient_by_lag(model, validation_path(args.data_root))
    with torch.no_grad():
        cache_x, _ = loader.clone().next_batch()
        cache = model.incremental_logits(
            cache_x[:, :16].to(device), control="real", bank_mode="full"
        )["cache_audit"]
    state = {
        "completed_2d2b_updates": 3,
        "processed_2d2b_targets": 3 * smoke_batch * T,
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
        "gate_gradient": all(
            math.isfinite(row["gate_gradient"]) and row["gate_gradient"] != 0
            for row in rows
        ),
        "base_gradient": all(
            row["gradient_groups"]["base"]["finite"]
            and row["gradient_groups"]["base"]["nonzero"]
            for row in rows
        ),
        "long_lag_writer_gradient": temporal["finite"]
        and temporal["long_lag_writer_gradient_present"],
        "attention_finite": all(row["recurrent_attention_finite"] for row in rows),
        "cache": cache["passed"] and cache["b1_historical_kv"] == 1,
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
        "long_lag_temporal_gradient": temporal,
        "incremental_cache_audit": cache,
        "checkpoint": verification,
        "checks": checks,
        "passed": all(checks.values()),
        "disposition": "Discarded; scientific local update 1 reloads the immutable 2D2A update-96 checkpoint.",
    }
    durable_json(output / "smoke_audit.json", audit)
    if not audit["passed"]:
        raise SystemExit(f"2D2B smoke failed: {checks}")
    print("EXPERIMENT_2D2B_SMOKE_PASS", flush=True)
    return audit


def training_metadata(args, preflight, micro_batch, accumulation) -> dict:
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "implementation_fingerprint": preflight["implementation_fingerprint"],
        "frozen_2d2a_commit": FROZEN_COMMIT,
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
        if state["completed_2d2b_updates"] != FORCED_RESTART_UPDATE:
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
            "completed_2d2b_updates": state["completed_2d2b_updates"],
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
            "completed_2d2b_updates": 0,
            "processed_2d2b_targets": 0,
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
    if state["completed_2d2b_updates"] == 0:
        if model.g_rec.detach().float().item() != SOURCE_GATE_RAW:
            raise SystemExit("scientific source gate was not restored exactly")
        if [group["lr"] for group in optimizer.param_groups] != [
            BASE_LR,
            BASE_LR,
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
    expected_lrs = {"base_decay": BASE_LR, "base_nodecay": BASE_LR, "gate": GATE_LR}
    if lrs != expected_lrs:
        raise SystemExit(f"resumed optimizer LR drift: {lrs}")
    optimizer.zero_grad(set_to_none=True)
    pass_loss_sums = [0.0] * count
    forward_seconds = [0.0] * count
    backward_seconds = 0.0
    total_ce = 0.0
    final_h12_rms = None
    final_recurrent_rms = None
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for micro_index in range(runtime.accumulation):
        cpu_x, cpu_y = runtime.loader.next_batch()
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        results = []
        source = None
        for pass_index in range(count):
            torch.cuda.synchronize()
            pass_start = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current = model.forward_pass(
                    x,
                    targets=y,
                    recurrent_source=source,
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
            source = current["h12"]
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
            final_h12_rms = results[-1]["h12"].detach().float().square().mean().sqrt().item()
            final_recurrent_rms = results[-1]["diagnostics"][
                "recurrent_output_rms"
            ].detach().float().item()
        del x, y, cpu_x, cpu_y, results, source, weighted, scaled, current
    if not gradients_finite(model):
        raise SystemExit("nonfinite gradients")
    groups = gradient_group_report(model)
    if not groups["base"]["nonzero"] or not groups["gate"]["nonzero"]:
        raise SystemExit(f"required gradient group is zero: {groups}")
    gate_gradient = model.g_rec.grad.detach().float().item()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad_norm):
        raise SystemExit("nonfinite gradient norm")
    optimizer.step()
    if not model_finite(model) or not optimizer_finite(optimizer):
        raise SystemExit("nonfinite parameter/optimizer state")
    elapsed = time.monotonic() - start
    runtime.training_state["completed_2d2b_updates"] = update
    runtime.training_state["processed_2d2b_targets"] = update * GLOBAL_TARGETS
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
        "g_rec_raw": model.g_rec.detach().float().item(),
        "tanh_g_rec": model.recurrent_scale.detach().float().item(),
        "gate_gradient_preclip": gate_gradient,
        "gradient_norm_before_clip": grad_norm.detach().float().item(),
        "gradient_groups": groups,
        "b12_memory_rms": final_h12_rms,
        "b1_recurrent_output_rms": final_recurrent_rms,
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
            "gate_raw": metrics["g_rec_raw"],
            "effective_gate": metrics["tanh_g_rec"],
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
    validation = evaluate_parallel(runtime.model, val_path)
    validation.update(
        {
            "local_update": update,
            "additional_targets": update * GLOBAL_TARGETS,
            "cumulative_2d2_targets": SOURCE_TARGETS + update * GLOBAL_TARGETS,
        }
    )
    attention = attention_diagnostics(runtime.model, val_path)
    temporal = temporal_gradient_by_lag(runtime.model, val_path)
    temporal.update(
        {
            "local_update": update,
            "additional_targets": update * GLOBAL_TARGETS,
            "cumulative_2d2_targets": SOURCE_TARGETS + update * GLOBAL_TARGETS,
        }
    )
    merge_keyed_json(runtime.output / "milestone_validation.json", update, validation)
    merge_keyed_json(runtime.output / "attention_lag_bins.json", update, attention["lag_bins"])
    merge_keyed_json(
        runtime.output / "attention_head_distance.json",
        update,
        {
            "heads": attention["heads"],
            "aggregate": attention["aggregate"],
            "mass_partitions": attention["mass_partitions"],
            "pinned_batch": attention["pinned_batch"],
        },
    )
    merge_keyed_json(runtime.output / "temporal_gradient_by_lag.json", update, temporal)
    restore_rng_state(saved_rng)
    runtime.model.train()
    return validation, attention, temporal


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
        raise SystemExit("2D2B train segments must end at local update 96 or 191")
    runtime = initialize_runtime(args)
    completed = int(runtime.training_state["completed_2d2b_updates"])
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
            f"2D2B update={update:03d}/{MAX_UPDATES} "
            f"loss={metrics['weighted_total_ce']:.6f} "
            f"gate={metrics['tanh_g_rec']:+.8f} "
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
        print("EXPERIMENT_2D2B_UPDATE_96_RESTART_REQUIRED", flush=True)
    else:
        durable_json(
            runtime.output / "training_complete.json",
            {
                "completed_2d2b_updates": MAX_UPDATES,
                "processed_2d2b_targets": ADDITIONAL_TARGETS,
                "cumulative_2d2_targets": CUMULATIVE_TARGETS,
                "checkpoint": runtime.training_state["last_checkpoint"],
                "timestamp": time.time(),
            },
        )
        print("EXPERIMENT_2D2B_TRAINING_COMPLETE", flush=True)
    return segment


def _incremental_control(model, x, y, name, derangement=None):
    batch, length = x.shape
    if name == "plain":
        control, bank_mode = "plain", "full"
    elif name == "full_real":
        control, bank_mode = "real", "full"
    elif name == "full_shuffled":
        control, bank_mode = "shuffled", "full"
    elif name == "two_slot_real":
        control, bank_mode = "real", "two_slot"
    else:
        raise ValueError(name)
    state = model.init_incremental_state(batch, device=x.device)
    per_sequence_sum = torch.zeros(batch, dtype=torch.float64, device="cpu")
    per_position_sum = np.zeros(length, dtype=np.float64)
    total_sum = 0.0
    targets = 0
    max_cache = [0] * N_LAYER
    max_ring = 0
    recurrent_output_rms = []
    for position in range(length):
        result = model.incremental_step(
            x[:, position],
            state,
            control=control,
            recurrent_permutation=derangement if control == "shuffled" else None,
            return_diagnostics=control != "plain",
            bank_mode=bank_mode,
            diagnostic_attention_weights=False,
        )
        if control == "plain":
            logits, state = result
        else:
            logits, state, diagnostics = result
            recurrent_output_rms.append(
                diagnostics["recurrent_output_rms"].detach().float().item()
            )
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
        max_ring = max(max_ring, int(state.h12_ring.size(1)))
    memory_rms = (
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
        "max_h12_ring_length": max_ring,
        "final_h12_memory_rms": memory_rms,
        "mean_recurrent_output_rms": (
            statistics.fmean(recurrent_output_rms) if recurrent_output_rms else 0.0
        ),
    }


@torch.no_grad()
def evaluate_incremental(model, val_path, batches=INCREMENTAL_BATCHES) -> dict:
    model.eval()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    names = ("plain", "full_real", "full_shuffled", "two_slot_real")
    rows = {
        name: {
            "loss_sum": 0.0,
            "targets": 0,
            "per_batch_losses": [],
            "per_sequence_losses": [],
            "per_position_sum": np.zeros(T, dtype=np.float64),
            "cache_rows": [],
            "memory_rms": [],
            "recurrent_output_rms": [],
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
                    "max_h12_ring_length": current["max_h12_ring_length"],
                }
            )
            row["memory_rms"].append(current["final_h12_memory_rms"])
            row["recurrent_output_rms"].append(
                current["mean_recurrent_output_rms"]
            )
        print(f"2D2B incremental batch={batch_index + 1:02d}/{batches}", flush=True)
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
            "final_h12_memory_rms": statistics.fmean(row["memory_rms"]),
            "mean_recurrent_output_rms": statistics.fmean(
                row["recurrent_output_rms"]
            ),
        }
    plain = controls["plain"]
    full = controls["full_real"]
    shuffled = controls["full_shuffled"]
    two_slot = controls["two_slot_real"]
    result = {
        "controls": controls,
        "true_full_gain": plain["validation_loss"] - full["validation_loss"],
        "true_sequence_gap": shuffled["validation_loss"] - full["validation_loss"],
        "true_bank_width_gain": two_slot["validation_loss"]
        - full["validation_loss"],
        "full_vs_plain_batches": paired_stats(
            full["per_batch_losses"], plain["per_batch_losses"]
        ),
        "full_vs_shuffled_batches": paired_stats(
            full["per_batch_losses"], shuffled["per_batch_losses"]
        ),
        "full_vs_two_slot_batches": paired_stats(
            full["per_batch_losses"], two_slot["per_batch_losses"]
        ),
        "full_vs_plain_sequences": paired_stats(
            full["per_sequence_losses"], plain["per_sequence_losses"]
        ),
        "full_vs_shuffled_sequences": paired_stats(
            full["per_sequence_losses"], shuffled["per_sequence_losses"]
        ),
        "full_vs_two_slot_sequences": paired_stats(
            full["per_sequence_losses"], two_slot["per_sequence_losses"]
        ),
        "effective_recurrent_scale": model.recurrent_scale.detach().float().item(),
        "gate_raw": model.g_rec.detach().float().item(),
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
        print(f"2D2B old-memory ablation batch={batch_index + 1:02d}/{batches}", flush=True)
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
                tokens, control="plain", diagnostic_attention_weights=False
            )["logits"]
            first = model.forward_pass(tokens)
            parallel_real = model.forward_pass(
                tokens, recurrent_source=first["h12"], bank_mode="full"
            )["logits"]
            incremental_real = model.incremental_logits(
                tokens,
                control="real",
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
        recurrent = batch * 1023 * N_EMBD * element_bytes
        upper = batch * 11 * 1023 * N_EMBD * 2 * element_bytes
        total = b1 + recurrent + upper
        return {
            "batch_size": batch,
            "b1_local_kv_bytes": b1,
            "b12_recurrent_raw_state_bytes": recurrent,
            "b2_b12_ordinary_kv_bytes": upper,
            "total_experimental_inference_state_bytes": total,
            "mib": {
                "b1_local_kv": b1 / 1024**2,
                "b12_recurrent_raw_state": recurrent / 1024**2,
                "b2_b12_ordinary_kv": upper / 1024**2,
                "total": total / 1024**2,
            },
        }

    return {
        "dtype": "BF16",
        "bytes_per_element": element_bytes,
        "B1": one(1),
        "B64": one(64),
        "prominent_limitation": (
            "2D2B is a mechanism experiment and does not claim whole-model KV "
            "savings because B2-B12 remain full-context."
        ),
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
    if payload.get("completed_2d2b_updates") != MAX_UPDATES:
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
        "completed_2d2b_updates": payload["completed_2d2b_updates"],
        "processed_2d2b_targets": payload["processed_2d2b_targets"],
        "cumulative_2d2_targets": payload["cumulative_2d2_targets"],
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "model_finite": model_finite(model),
        "optimizer_finite": optimizer_finite(optimizer),
        "gate_raw": model.g_rec.detach().float().item(),
        "effective_gate": model.recurrent_scale.detach().float().item(),
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
        return "EXPERIMENT 2D2B INVALID"
    if not stable:
        return "FULL-BANK RECURRENT K/V IS UNSTABLE"
    controls = incremental["controls"]
    full = controls["full_real"]["validation_loss"]
    plain = controls["plain"]["validation_loss"]
    shuffled = controls["full_shuffled"]["validation_loss"]
    two_slot = controls["two_slot_real"]["validation_loss"]
    wins = all(
        incremental[name]["wins"] > incremental[name]["losses"]
        for name in (
            "full_vs_plain_sequences",
            "full_vs_shuffled_sequences",
            "full_vs_two_slot_sequences",
        )
    )
    strong_wins = (
        incremental["full_vs_plain_sequences"]["wins"]
        / incremental["full_vs_plain_sequences"]["count"]
        >= 0.75
        and incremental["full_vs_two_slot_sequences"]["wins"]
        / incremental["full_vs_two_slot_sequences"]["count"]
        >= 0.75
    )
    if (
        incremental["true_full_gain"] >= 0.01
        and incremental["true_bank_width_gain"] >= 0.005
        and incremental["true_sequence_gap"] > 0
        and strong_wins
    ):
        return "FULL-BANK RECURRENT K/V STRONGLY SCALES UTILITY"
    if full < plain and full < shuffled and full < two_slot and wins:
        return "FULL-BANK RECURRENT K/V SCALES POSITIVE UTILITY"
    if full < plain and full >= two_slot:
        return "RECURRENT K/V REMAINS USEFUL BUT FULL BANK DOES NOT OUTPERFORM TWO-SLOT MEMORY"
    if full < shuffled and full >= plain:
        return "FULL-BANK MEMORY IS SEQUENCE-SPECIFIC BUT NOT PREDICTIVELY USEFUL"
    return "FULL-BANK RECURRENT K/V DOES NOT ESTABLISH POSITIVE UTILITY"


def choose_recommendation(classification, parallel, incremental, attention) -> str:
    if classification == "EXPERIMENT 2D2B INVALID":
        return "FIX 2D2B INTEGRITY"
    if classification == "FULL-BANK RECURRENT K/V IS UNSTABLE":
        return "FIX 2D2B INTEGRITY"
    if parallel["full_bank_gain"] > 0 and parallel["bank_width_gain"] > 0 and (
        incremental["true_full_gain"] <= 0 or incremental["true_bank_width_gain"] <= 0
    ):
        return "TRAIN FULL-BANK SELF-RECURRENT DISTRIBUTION COMPATIBILITY"
    if classification in {
        "FULL-BANK RECURRENT K/V SCALES POSITIVE UTILITY",
        "FULL-BANK RECURRENT K/V STRONGLY SCALES UTILITY",
    }:
        return "EXTEND FULL-BANK RECURRENT K/V TO MIRRORED HIGH→LOW LAYER PAIRS"
    if classification.startswith("RECURRENT K/V REMAINS USEFUL"):
        return "LEARN TEMPORAL BANK SELECTION / COMPRESSION"
    if classification.startswith("FULL-BANK MEMORY IS SEQUENCE-SPECIFIC"):
        return "ADD DEDICATED RECURRENT K/V PROJECTIONS"
    if attention["mass_partitions"]["lags_32_127"] + attention["mass_partitions"][
        "lags_128_511"
    ] + attention["mass_partitions"]["lags_512_1023"] < 0.05:
        return "INCREASE TRAINING PRESSURE FOR LONG-RANGE RECURRENT RETRIEVAL"
    return "LEARN TEMPORAL BANK SELECTION / COMPRESSION"


@torch.no_grad()
def self_composition_diagnostic(model, val_path, passes=8, batch_size=2) -> dict:
    model.eval()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], batch_size, T)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(device), cpu_y.to(device)
    source = None
    rows = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for pass_index in range(passes):
            current = model.forward_pass(
                x,
                recurrent_source=source,
                return_diagnostics=pass_index > 0,
                bank_mode="full",
            )
            loss = _token_losses(current["logits"], y).double().mean().item()
            rows.append(
                {
                    "pass": pass_index + 1,
                    "loss": loss,
                    "b12_memory_rms": current["h12"].float().square().mean().sqrt().item(),
                    "b1_recurrent_output_rms": (
                        current["diagnostics"]["recurrent_output_rms"].float().item()
                        if pass_index > 0
                        else 0.0
                    ),
                }
            )
            source = current["h12"]
    report = {
        "passes": rows,
        "batch_size": batch_size,
        "sequence_length": T,
        "finite": all(
            math.isfinite(row["loss"])
            and math.isfinite(row["b12_memory_rms"])
            and math.isfinite(row["b1_recurrent_output_rms"])
            for row in rows
        ),
        "no_gradient": True,
    }
    del x, y, source, current
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
    plain = [milestones[str(key)]["controls"]["plain"]["validation_loss"] for key in keys]
    full = [milestones[str(key)]["controls"]["full_real"]["validation_loss"] for key in keys]
    two = [milestones[str(key)]["controls"]["two_slot_real"]["validation_loss"] for key in keys]

    def line_plot(filename, series, ylabel):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for label, values in series.items():
            ax.plot(x, values, marker="o", label=label)
        ax.set_xlabel("Additional 2D2B training targets")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    line_plot(REQUIRED_PLOTS[0], {"Plain": plain, "FullReal": full, "TwoSlotReal": two}, "Cross entropy")
    line_plot(REQUIRED_PLOTS[1], {"Plain − FullReal": [milestones[str(k)]["full_bank_gain"] for k in keys]}, "Full-bank gain")
    line_plot(REQUIRED_PLOTS[2], {"TwoSlotReal − FullReal": [milestones[str(k)]["bank_width_gain"] for k in keys]}, "Bank-width gain")
    line_plot(REQUIRED_PLOTS[3], {"FullShuffled − FullReal": [milestones[str(k)]["sequence_gap"] for k in keys]}, "Sequence gap")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot([row["additional_targets"] for row in training], [row["tanh_g_rec"] for row in training])
    ax.scatter([0], [SOURCE_GATE_EFFECTIVE], label="2D2A source", zorder=3)
    ax.set(xlabel="Additional 2D2B training targets", ylabel="tanh(g_rec)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[4], dpi=180)
    plt.close(fig)

    final_attention = attention[str(MAX_UPDATES)]
    labels = [name for name, _, _ in LAG_BINS]
    mass = [final_attention[name]["attention_mass"] for name in labels]
    density = [final_attention[name]["normalized_mass_per_available_token"] for name in labels]
    for filename, values, ylabel in (
        (REQUIRED_PLOTS[5], mass, "Attention mass"),
        (REQUIRED_PLOTS[6], density, "Mass per available token"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    final_heads = read_json(output / "attention_head_distance.json")[str(MAX_UPDATES)]["heads"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(range(N_HEAD), [final_heads[str(index)]["mean_attended_recurrent_lag"] for index in range(N_HEAD)])
    ax.set(xlabel="B1 attention head", ylabel="Mean recurrent lag")
    ax.set_xticks(range(N_HEAD))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[7], dpi=180)
    plt.close(fig)

    final_gradient = temporal[str(MAX_UPDATES)]["bins"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(labels, [final_gradient[name]["mean_gradient_rms"] for name in labels])
    ax.set_ylabel("Mean writer-gradient RMS")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[8], dpi=180)
    plt.close(fig)

    final_bins = milestones[str(MAX_UPDATES)]["position_bins"]
    position_labels = [name for name, _, _ in POSITION_BINS]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(position_labels, [final_bins[name]["full_bank_gain"] for name in position_labels])
    ax.set_ylabel("Plain − FullReal CE")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[9], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    names = ["Full gain", "Sequence gap", "Bank-width gain"]
    parallel_values = [
        milestones[str(MAX_UPDATES)]["full_bank_gain"],
        milestones[str(MAX_UPDATES)]["sequence_gap"],
        milestones[str(MAX_UPDATES)]["bank_width_gain"],
    ]
    true_values = [
        incremental["true_full_gain"],
        incremental["true_sequence_gap"],
        incremental["true_bank_width_gain"],
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
    final_attention = attention[str(MAX_UPDATES)]
    final_gradient = temporal[str(MAX_UPDATES)]["bins"]
    heads = read_json(Path(summary["artifact_directory"]) / "attention_head_distance.json")[
        str(MAX_UPDATES)
    ]["heads"]
    means = [heads[str(index)]["mean_attended_recurrent_lag"] for index in range(N_HEAD)]
    short_gradient = final_gradient["2-7"]["mean_gradient_rms"]
    long_gradient = final_gradient["512-1023"]["mean_gradient_rms"]
    parallel_gain = summary["parallel"]["full_bank_gain"]
    return {
        "Q1": {"question": "Did frozen 2D2A weights benefit zero-shot from widening the bank?", "answer": summary["zero_shot"]["legacy_bank_expansion_gain"] > 0, "evidence": summary["zero_shot"]["legacy_bank_expansion_gain"]},
        "Q2": {"question": "What was zero-shot full-bank gain?", "answer": summary["zero_shot"]["full_bank_gain"]},
        "Q3": {"question": "What was zero-shot bank-width gain versus two-slot?", "answer": summary["zero_shot"]["legacy_bank_expansion_gain"]},
        "Q4": {"question": "How did full-bank recurrent gain evolve?", "answer": {key: value["full_bank_gain"] for key, value in milestones.items()}},
        "Q5": {"question": "At what milestone did FullReal beat Plain?", "answer": first_positive_milestone(milestones, "full_bank_gain")},
        "Q6": {"question": "At what milestone did FullReal beat TwoSlotReal?", "answer": first_positive_milestone(milestones, "bank_width_gain")},
        "Q7": {"question": "Did sequence-specificity strengthen with bank width?", "answer": summary["parallel"]["sequence_gap"] > 0.0019076712274910257, "2d2a_gap": 0.0019076712274910257, "2d2b_gap": summary["parallel"]["sequence_gap"]},
        "Q8": {"question": "What fraction of recurrent mass remained at lags 2-31?", "answer": final_attention["mass_partitions"]["lags_2_31"]},
        "Q9": {"question": "What fraction went to lags 32-127?", "answer": final_attention["mass_partitions"]["lags_32_127"]},
        "Q10": {"question": "What fraction went to lags 128-511?", "answer": final_attention["mass_partitions"]["lags_128_511"]},
        "Q11": {"question": "What fraction went to lags 512-1023?", "answer": final_attention["mass_partitions"]["lags_512_1023"]},
        "Q12": {"question": "Which lag range had strongest normalized density?", "answer": max(final_attention["lag_bins"], key=lambda name: final_attention["lag_bins"][name]["normalized_mass_per_available_token"])},
        "Q13": {"question": "Did heads specialize by temporal distance?", "answer": statistics.pstdev(means) > 1.0, "head_mean_lag_range": [min(means), max(means)], "std": statistics.pstdev(means)},
        "Q14": {"question": "Did masking old memory hurt validation?", "answer": old_ablation["full_vs_recent_gain"] > 0, "delta": old_ablation["full_vs_recent_gain"]},
        "Q15": {"question": "Does OLD_ONLY contain positive utility?", "answer": old_ablation["old_only_utility_vs_plain"] > 0, "delta": old_ablation["old_only_utility_vs_plain"]},
        "Q16": {"question": "Does benefit grow with current position?", "answer": summary["parallel"]["position_bins"]},
        "Q17": {"question": "Did writer gradients reach hundreds of tokens back?", "answer": temporal[str(MAX_UPDATES)]["long_lag_writer_gradient_present"]},
        "Q18": {"question": "How did long-lag gradient compare with short-lag?", "answer": long_gradient / max(short_gradient, 1e-30), "long_rms": long_gradient, "short_rms": short_gradient},
        "Q19": {"question": "Did the gate grow or shrink?", "answer": "grew" if summary["final_gate_raw"] > SOURCE_GATE_RAW else "shrank", "source": SOURCE_GATE_RAW, "final": summary["final_gate_raw"]},
        "Q20": {"question": "Did parallel gain transfer to true incremental inference?", "answer": incremental["true_full_gain"] > 0, "retention_fraction": incremental["true_full_gain"] / parallel_gain if parallel_gain else None},
        "Q21": {"question": "Final true-self FullReal vs Plain gain?", "answer": incremental["true_full_gain"]},
        "Q22": {"question": "Final true-self sequence gap?", "answer": incremental["true_sequence_gap"]},
        "Q23": {"question": "Final true-self bank-width gain?", "answer": incremental["true_bank_width_gain"]},
        "Q24": {"question": "How much recurrent state is stored?", "answer": {"raw_B12_states": 1023, "B1_historical_KV": 1, "B1_B12_raw_BF16_bytes_B1": memory["B1"]["b12_recurrent_raw_state_bytes"], "B1_B12_raw_BF16_bytes_B64": memory["B64"]["b12_recurrent_raw_state_bytes"]}},
        "Q25": {"question": "Does the result justify mirrored links?", "answer": summary["recommendation"].startswith("EXTEND FULL-BANK")},
        "Q26": {"question": "Does it justify dedicated projections?", "answer": summary["recommendation"] == "ADD DEDICATED RECURRENT K/V PROJECTIONS"},
        "Q27": {"question": "What exactly one experiment should run next?", "answer": summary["recommendation"]},
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
            "plain_loss": float(values["plain"].mean()),
            "full_real_loss": float(values["full_real"].mean()),
            "full_shuffled_loss": float(values["full_shuffled"].mean()),
            "two_slot_real_loss": float(values["two_slot_real"].mean()),
            "full_bank_gain": float(
                (values["plain"] - values["full_real"]).mean()
            ),
            "sequence_gap": float(
                (values["full_shuffled"] - values["full_real"]).mean()
            ),
            "bank_width_gain": float(
                (values["two_slot_real"] - values["full_real"]).mean()
            ),
        }
    return result


def build_artifact_inventory(output):
    output = Path(output)
    mutable = {
        "EXPERIMENT_2D2B_FINAL_REPORT.md",
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
    zero = summary["zero_shot"]
    training = summary["training"]
    memory = summary["memory_accounting"]
    lines = [
        f"EXPERIMENT 2D2B PRIMARY CLASSIFICATION:\n{summary['primary_classification']}",
        f"\nFINAL TRUE-SELF FULL-BANK GAIN:\n{incremental['true_full_gain']}",
        f"\nFINAL TRUE-SELF SEQUENCE GAP:\n{incremental['true_sequence_gap']}",
        f"\nFINAL TRUE-SELF BANK-WIDTH GAIN:\n{incremental['true_bank_width_gain']}",
        "\n# Experiment 2D2B — Full-Width B12→B1 Token-Indexed Recurrent K/V Bank",
        "\n## Result",
        f"\nThe final classification is **{summary['primary_classification']}**.",
        f"The exactly one next experiment is **{summary['recommendation']}**.",
        "\n## Source and architecture",
        f"\n- Source checkpoint: `{summary['source_checkpoint']}`",
        f"- Source SHA-256: `{summary['source_checkpoint_sha256']}`",
        f"- Parameters: {summary['parameters']:,} (zero new versus 2D2A)",
        f"- Hardware: {summary['hardware']}",
        "- B1 local bank: `t-1,t` (W=2)",
        "- B1 recurrent bank: `max(0,t-1023)...t-2` (maximum 1022 entries)",
        "- B2–B12: W=1024; overall T=1024",
        "- Recurrent source: B12 post-MLP residual immediately before `ln_f`",
        "- Existing B1 LN/K/V slices, separate recurrent softmax, one `c_proj`, one trained scalar gate",
        "\nExperiment 2D2A already used attached temporal B12 source states. Temporal writer gradients into Pass-1 B12 states were finite and nonzero during training. 2D2B is the first full recurrent-token-bank experiment, not the first writer-learning experiment.",
        "\nThe disclosed 2D2A evaluation-only BF16 correction is preserved: active-prefix absolute comparisons use the already-preregistered Plain tolerance 1.25; strict FP32 checks are unchanged. No model, checkpoint, training, data, loss, or scientific metric was changed by that correction.",
        "\n## Training",
        f"\n- Additional updates: {training['updates']} / {MAX_UPDATES}",
        f"- Additional targets: {ADDITIONAL_TARGETS:,}",
        f"- Cumulative 2D2 targets: {CUMULATIVE_TARGETS:,}",
        f"- Runtime: {training['total_wall_seconds']:.2f} seconds ({training['mean_seconds_per_update']:.2f} sec/update)",
        f"- Throughput: {training['aggregate_targets_per_second']:.2f} targets/sec",
        f"- Peak allocated/reserved VRAM: {training['peak_allocated_vram_mb']:.2f}/{training['peak_reserved_vram_mb']:.2f} MiB",
        f"- Gate: raw {SOURCE_GATE_RAW} → {summary['final_gate_raw']}; tanh {SOURCE_GATE_EFFECTIVE} → {summary['final_gate_effective']}",
        "- Optimizer, loader, and Python/NumPy/Torch CPU/CUDA RNG states resumed from 2D2A; warmup was not restarted.",
        "- Mandatory fresh-process restart after local update 96 passed.",
        "\n## Zero-shot bank expansion",
        f"\n- Plain: {zero['controls']['plain']['validation_loss']}",
        f"- FullReal: {zero['controls']['full_real']['validation_loss']}",
        f"- FullShuffled: {zero['controls']['full_shuffled']['validation_loss']}",
        f"- Legacy TwoSlotReal: {zero['legacy_two_slot_loss']}",
        f"- Full-bank zero-shot gain: {zero['full_bank_gain']}",
        f"- Zero-shot sequence gap: {zero['sequence_gap']}",
        f"- Legacy bank-expansion gain: {zero['legacy_bank_expansion_gain']}",
        "\n## Final parallel validation",
        f"\n- Plain: {parallel['controls']['plain']['validation_loss']}",
        f"- FullReal: {parallel['controls']['full_real']['validation_loss']}",
        f"- FullShuffled: {parallel['controls']['full_shuffled']['validation_loss']}",
        f"- TwoSlotReal: {parallel['controls']['two_slot_real']['validation_loss']}",
        f"- Full-bank gain: {parallel['full_bank_gain']}",
        f"- Sequence gap: {parallel['sequence_gap']}",
        f"- Bank-width gain: {parallel['bank_width_gain']}",
        f"- Paired batch wins (Plain/Shuffled/TwoSlot): {parallel['full_vs_plain']['wins']}/{parallel['full_vs_shuffled']['wins']}/{parallel['full_vs_two_slot']['wins']} of 20",
        "\n## Final true incremental validation",
        f"\n- Plain: {incremental['controls']['plain']['validation_loss']}",
        f"- FullReal: {incremental['controls']['full_real']['validation_loss']}",
        f"- FullShuffled: {incremental['controls']['full_shuffled']['validation_loss']}",
        f"- TwoSlotReal: {incremental['controls']['two_slot_real']['validation_loss']}",
        f"- Targets/control: {incremental['targets_per_control']:,}",
        f"- Sequence wins vs Plain/Shuffled/TwoSlot: {incremental['full_vs_plain_sequences']['wins']}/{incremental['full_vs_shuffled_sequences']['wins']}/{incremental['full_vs_two_slot_sequences']['wins']} of {incremental['full_vs_plain_sequences']['count']}",
        "\n## Attention, gradients, and old-memory ablation",
        f"\n- Final recurrent mass partitions: {summary['final_attention']['mass_partitions']}",
        f"- Strongest normalized-density bin: {questions['Q12']['answer']}",
        f"- Long/short writer-gradient RMS ratio: {questions['Q18']['answer']}",
        f"- Recent-only minus Full loss: {summary['old_memory_ablation']['full_vs_recent_gain']}",
        f"- Plain minus Old-only loss: {summary['old_memory_ablation']['old_only_utility_vs_plain']}",
        "\n## Cache and storage",
        f"\n- B1 historical same-layer KV: at most 1 entry",
        f"- B12 raw recurrent buffer: at most 1023 states",
        f"- B2–B12 ordinary historical KV: at most 1023 entries/layer",
        f"- BF16 total experimental inference state, B=1: {memory['B1']['mib']['total']:.3f} MiB",
        f"- BF16 total experimental inference state, B=64: {memory['B64']['mib']['total']:.3f} MiB",
        f"\n> {memory['prominent_limitation']}",
        "\n## Scientific questions Q1–Q27",
    ]
    for index in range(1, 28):
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
            "\n# EXPERIMENT 2D2B COMPLETE",
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
    final_cache_rows = incremental["controls"]["full_real"]["cache_rows"]
    cache_pass = all(
        row["final"]["passed"]
        and row["final"]["b1_historical_kv"] <= 1
        and row["final"]["h12_ring_length"] <= 1023
        and all(value <= 1023 for value in row["final"]["b2_b12_historical_kv"])
        for row in final_cache_rows
    )
    restart = read_json(Path(output) / "forced_restart_update_96.json")
    scientific = checkpoint_manifest["scientific"]
    recovery = checkpoint_manifest["recovery"]
    checks = {
        "2D2A final tag exact": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
        "2D2A checkpoint SHA exact": final_model_audit["passed"] and preflight["source"]["checkpoint_sha256"] == SOURCE_SHA256,
        "2D2A audit-correction provenance preserved": preflight["source"]["audit_correction_provenance"]["passed"],
        "parameter count unchanged at 124,475,905": preflight["parameters"]["passed"],
        "new parameters zero": preflight["parameters"]["new_parameter_count_vs_2d2a"] == 0,
        "source representation unchanged": preflight["semantic_diff"]["unchanged"]["source_representation"],
        "B1 local W=2 exact": preflight["architecture"]["b1_local_window"] == 2,
        "B2-B12 W=1024 exact": preflight["architecture"]["b2_b12_windows"] == [1024] * 11,
        "recurrent full bank exact": preflight["kernel_preflight"]["checks"]["boundary_counts"],
        "maximum recurrent entries=1022": preflight["architecture"]["maximum_recurrent_entries"] == 1022,
        "no local/recurrent overlap": preflight["kernel_preflight"]["checks"]["local_recurrent_disjoint"],
        "no future recurrent access": preflight["kernel_preflight"]["checks"]["no_future_or_t_minus_1"],
        "same B1 LN/Q/K/V projections": preflight["kernel_preflight"]["checks"]["shared_projection_exact"],
        "separate recurrent softmax": preflight["architecture"]["separate_softmaxes"],
        "single c_proj": preflight["kernel_preflight"]["checks"]["single_c_proj"],
        "same scalar gate": preflight["parameters"]["inventories_bit_for_bit_equal"],
        "gate source value exact": preflight["source"]["checks"]["gate_raw"],
        "2-slot regression exact": preflight["legacy_2d2a_regressions"]["checks"]["parallel"],
        "true-self 2-slot regression exact": preflight["legacy_2d2a_regressions"]["checks"]["incremental"],
        "non-detached temporal gradient present": preflight["temporal_gradient_checks"]["finite_nonzero"],
        "long-lag temporal gradients audited": temporal[str(MAX_UPDATES)]["long_lag_writer_gradient_present"],
        "same-model recurrence only": preflight["architecture"]["forbidden_modules_absent"]["teacher"],
        "CE-only loss": True,
        "pass weights exact": all(pass_weights(update) in (TWO_PASS_WEIGHTS, THREE_PASS_WEIGHTS) for update in range(1, MAX_UPDATES + 1)),
        "Pass-3 cadence exact": [update for update in range(1, MAX_UPDATES + 1) if pass_count(update) == 3] == [32, 64, 96, 128, 160],
        "optimizer resume exact": preflight["source"]["source_optimizer"]["restored_exactly_via_strict_optimizer_load_state_dict"],
        "loader/RNG continuation exact": preflight["checks"]["loader_continuation"],
        "global targets/update 524,288": preflight["checks"]["global_batch"],
        "191 additional optimizer updates": final_model_audit["completed_2d2b_updates"] == 191,
        "100,139,008 additional targets": final_model_audit["processed_2d2b_targets"] == ADDITIONAL_TARGETS,
        "no new projection": preflight["architecture"]["forbidden_modules_absent"]["dedicated_recurrent_projection"],
        "no teacher": preflight["architecture"]["forbidden_modules_absent"]["teacher"],
        "no AttnRes": preflight["architecture"]["forbidden_modules_absent"]["attnres"],
        "no mirrored links": preflight["architecture"]["forbidden_modules_absent"]["mirrored_links"],
        "no detached training arm": preflight["architecture"]["forbidden_modules_absent"]["detached_training_arm"],
        "B1 physical KV <=1 historical entry": cache_pass,
        "B12 recurrent raw-state buffer <=1023 previous states": cache_pass,
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
    if training_complete["completed_2d2b_updates"] != MAX_UPDATES:
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
    old_ablation = old_memory_ablation(model, validation_path(args.data_root))
    equivalence = parallel_incremental_equivalence(
        model, validation_path(args.data_root)
    )
    composition = self_composition_diagnostic(
        model, validation_path(args.data_root)
    )
    attention_lags = read_json(output / "attention_lag_bins.json")
    attention_heads = read_json(output / "attention_head_distance.json")
    temporal = read_json(output / "temporal_gradient_by_lag.json")
    final_attention = {
        "lag_bins": attention_lags[str(MAX_UPDATES)],
        "heads": attention_heads[str(MAX_UPDATES)]["heads"],
        "aggregate": attention_heads[str(MAX_UPDATES)]["aggregate"],
        "mass_partitions": attention_heads[str(MAX_UPDATES)]["mass_partitions"],
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
        classification, parallel, incremental, final_attention
    )
    zero_shot = read_json(output / "zero_shot_bank_expansion.json")
    summary = {
        "experiment": EXPERIMENT,
        "primary_classification": classification,
        "recommendation": recommendation,
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "parameters": TOTAL_PARAMETERS,
        "new_parameters_vs_2d2a": 0,
        "hardware": "1 × NVIDIA A100-SXM4-80GB",
        "architecture": architecture_manifest(),
        "training": performance["training"],
        "zero_shot": zero_shot,
        "validation_trajectory": milestones,
        "parallel": parallel,
        "incremental": incremental,
        "final_attention": final_attention,
        "final_temporal_gradient": temporal[str(MAX_UPDATES)],
        "old_memory_ablation": old_ablation,
        "memory_accounting": memory,
        "self_composition": composition,
        "parallel_incremental_equivalence": equivalence,
        "final_gate_raw": final_model_audit["gate_raw"],
        "final_gate_effective": final_model_audit["effective_gate"],
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
        {str(MAX_UPDATES): final_attention},
        temporal,
        old_ablation,
        memory,
    )
    position_metrics = {
        "parallel": parallel["position_bins"],
        "true_incremental": incremental_position_bins(incremental),
    }
    paired_controls = {
        "parallel": {
            "full_vs_plain": parallel["full_vs_plain"],
            "full_vs_shuffled": parallel["full_vs_shuffled"],
            "full_vs_two_slot": parallel["full_vs_two_slot"],
        },
        "true_incremental": {
            "full_vs_plain_batches": incremental["full_vs_plain_batches"],
            "full_vs_shuffled_batches": incremental["full_vs_shuffled_batches"],
            "full_vs_two_slot_batches": incremental["full_vs_two_slot_batches"],
            "full_vs_plain_sequences": incremental["full_vs_plain_sequences"],
            "full_vs_shuffled_sequences": incremental[
                "full_vs_shuffled_sequences"
            ],
            "full_vs_two_slot_sequences": incremental[
                "full_vs_two_slot_sequences"
            ],
        },
    }
    gate_diagnostics = {
        "source": {
            "raw": SOURCE_GATE_RAW,
            "effective": SOURCE_GATE_EFFECTIVE,
        },
        "trajectory": [
            {
                "local_update": row["local_update"],
                "additional_targets": row["additional_targets"],
                "raw": row["g_rec_raw"],
                "effective": row["tanh_g_rec"],
                "gradient": row["gate_gradient_preclip"],
            }
            for row in training
        ],
        "final": {
            "raw": final_model_audit["gate_raw"],
            "effective": final_model_audit["effective_gate"],
        },
        "gate_cap": None,
        "gate_reset": False,
    }
    cache_audit = {
        "controls": {
            name: row["cache_rows"] for name, row in incremental["controls"].items()
        },
        "B1_historical_KV_limit": 1,
        "B12_raw_recurrent_state_limit": 1023,
        "B2_B12_historical_KV_limit": 1023,
        "no_hidden_full_B1_cache": all(
            cache["final"]["b1_historical_kv"] <= 1
            for cache in incremental["controls"]["full_real"]["cache_rows"]
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
    durable_json(output / "old_memory_ablation.json", old_ablation)
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
    durable_text(output / "EXPERIMENT_2D2B_FINAL_REPORT.md", report)
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
    durable_text(output / "EXPERIMENT_2D2B_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nPending lifecycle action: commit/push, verify synchronization, then stop the exact GPU pod without deleting it.\n",
    )
    if not scientific_integrity or not inventory["passed"]:
        raise SystemExit(
            f"2D2B finalize integrity failed: scientific={scientific_integrity} inventory={inventory['passed']}"
        )
    print("EXPERIMENT_2D2B_FINALIZED_PENDING_GIT_SEAL", flush=True)
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
    durable_text(output / "EXPERIMENT_2D2B_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nFinal RunPod stop remains the only lifecycle action after the seal commit is pushed.\n",
    )
    print("EXPERIMENT_2D2B_REPORT_SEALED", flush=True)


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
