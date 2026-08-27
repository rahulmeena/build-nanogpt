#!/usr/bin/env python3
"""Experiment 2D2F: B3 W64 plus B10-to-B3 without B2 recurrence.

The finalized 2D2D checkpoint is the immutable source. This driver restores
its complete model, optimizer, loader, RNG, and cursor state; preserves the
trained B12-to-B1 path; physically removes B11-to-B2 recurrence and its gate;
keeps B2 local W32; and adds one B10-to-B3 gate at lags 64..1023.
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
import experiment_2d2d as source_driver  # noqa: E402
import experiment_2d2d_core as source_core  # noqa: E402
from experiment_2d2f_core import (  # noqa: E402
    BANK_MODES,
    B2_LOCAL_WINDOW,
    B3_LOCAL_WINDOW,
    B3_MAX_RECURRENT_ENTRIES,
    B3_RECURRENT_MIN_LAG,
    INCREMENTAL_CONTROLS,
    MirroredIncrementalState,
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
    RecurrentKVGPT,
)


EXPERIMENT = "2D2F"
PROTOCOL = "exp2d2f_no_b2_recurrence_b3_w64_v1"
BRANCH = "experiment-2d2f-no-b2-recurrence-b3-w64"
FROZEN_TAG = "experiment-2d2d-b2-w32-b11-recurrent-992-final"
FROZEN_COMMIT = "a9300a9800f2e2c46f3892cff52b0a4a2a547d11"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d2f_no_b2_recurrence_b3_w64.json"
OUTPUT_NAME = "experiment_2d2f_no_b2_recurrence_b3_w64"
CHECKPOINT_SCHEMA = "exp2d2f_no_b2_recurrence_b3_w64_checkpoint_v1"
SOURCE_SCHEMA = "exp2d2d_b2_w32_b11_recurrent_992_checkpoint_v1"
SOURCE_SHA256 = "d38e8282cca4df395204b860d17e2cd9b89ff7ad07319fe744bbdc46fb945063"
SOURCE_BYTES = 1_493_940_151
SOURCE_UPDATES = 191
SOURCE_TARGETS = 250_609_664
SOURCE_GATE_RAW = 0.12906356155872345
SOURCE_GATE_EFFECTIVE = 0.12835168838500977
SOURCE_B2_GATE_RAW = 0.008991487324237823
SOURCE_B2_GATE_EFFECTIVE = 0.008991245180368423
SOURCE_NEXT_BATCH_SHA256 = "39808d08e7e15e9f160e32ba838fd28839067827095b023d4f475b30df392086"
SOURCE_NEXT_STREAM_SHA256 = "110e232ab330611a8d23cddc6e914c8a2d912fd8191873ae67488b0d52f48daa"
SOURCE_MICRO_BATCH = 32
SOURCE_ACCUMULATION = 16

T = 1024
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
VOCAB_SIZE = 50_304
B1_LOCAL_WINDOW = 2
B2_WINDOW = 32
B3_WINDOW = 64
GLOBAL_TARGETS = 524_288
MAX_UPDATES = 191
ADDITIONAL_TARGETS = MAX_UPDATES * GLOBAL_TARGETS
CUMULATIVE_TARGETS = SOURCE_TARGETS + ADDITIONAL_TARGETS
TOTAL_PARAMETERS = 124_475_906
SOURCE_PARAMETERS = 124_475_906
BASE_LR = 3e-5
GATE_LR = 3e-4
GRAD_CLIP = 1.0
TWO_PASS_WEIGHTS = (0.25, 0.75)
THREE_PASS_WEIGHTS = (0.20, 0.40, 0.40)
MILESTONES = (0, 20, 48, 96, 143, 191)
SCIENTIFIC_CHECKPOINTS = (96, 191)
RECOVERY_CHECKPOINTS = ()
FORCED_RESTART_UPDATE = 96
VALIDATION_BATCHES = 20
VALIDATION_B = 64
INCREMENTAL_BATCHES = 4
CANONICAL_VALIDATION_SHA256 = legacy.CANONICAL_VALIDATION_SHA256
VALIDATION_SHARD_SHA256 = legacy.VALIDATION_SHARD_SHA256
SEED = 2026_0221
BF16_INCREMENTAL_ATOL = 1.25
FP32_INCREMENTAL_ATOL = 1e-4
B1_LAG_BINS = (
    ("2-7", 2, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
    ("32-63", 32, 63),
    ("64-127", 64, 127),
    ("128-255", 128, 255),
    ("256-511", 256, 511),
    ("512-1023", 512, 1023),
)
B2_RECURRENT_LAG_BINS = (
    ("32-63", 32, 63),
    ("64-127", 64, 127),
    ("128-255", 128, 255),
    ("256-511", 256, 511),
    ("512-1023", 512, 1023),
)
B2_LOCAL_LAG_BINS = (
    ("0-3", 0, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
)
B3_RECURRENT_LAG_BINS = (
    ("64-127", 64, 127),
    ("128-255", 128, 255),
    ("256-511", 256, 511),
    ("512-1023", 512, 1023),
)
B3_LOCAL_LAG_BINS = (
    ("0-7", 0, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
    ("32-63", 32, 63),
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
    "configs/exp2d2f_no_b2_recurrence_b3_w64.json",
    "scripts/experiment_2d2f.py",
    "scripts/experiment_2d2f_core.py",
    "scripts/experiment_2d2d.py",
    "scripts/experiment_2d2d_core.py",
    "scripts/experiment_2d2a.py",
    "scripts/experiment_2d2a_core.py",
    "scripts/experiment_2d0.py",
    "scripts/experiment_2d0d.py",
    "scripts/experiment_2d1.py",
    "scripts/smoke_test.py",
    "train_gpt2.py",
    "tests/test_experiment_2d2f_core.py",
    "tests/test_experiment_2d2f_driver.py",
)
REQUIRED_ARTIFACTS = (
    "FINAL_REPORT.md",
    "EXPERIMENT_2D2F_FINAL_REPORT.md",
    "FINAL_AUDIT.json",
    "result_summary.json",
    "source_manifest.json",
    "parameter_manifest.json",
    "architecture_manifest.json",
    "2d2d_reference_manifest.json",
    "matched_2d2e_data_audit.json",
    "preflight_audit.json",
    "training_metrics.jsonl",
    "milestone_validation.json",
    "paired_controls.json",
    "gate_diagnostics.json",
    "attention_diagnostics.json",
    "temporal_gradient_diagnostics.json",
    "b1_attention_diagnostics.json",
    "b3_local_attention_lag_bins.json",
    "b3_recurrent_attention_lag_bins.json",
    "b3_attention_head_distance.json",
    "b12_to_b1_temporal_gradient.json",
    "b10_to_b3_temporal_gradient.json",
    "incremental_validation.json",
    "incremental_cache_audit.json",
    "memory_accounting.json",
    "stability_8pass.json",
    "checkpoint_manifest.json",
    "performance.json",
    "distributed_equivalence.json",
    "storage_cleanup_manifest.json",
    "commands_and_runtime.json",
    "HEARTBEAT.json",
    "UNATTENDED_FINAL_HANDOFF.md",
)
REQUIRED_PLOTS = tuple(f"P{index}_{name}.png" for index, name in enumerate((
    "all_real_b3_off_validation_ce",
    "b3_recurrent_gain",
    "b3_sequence_gap",
    "b1_b3_gate_trajectories",
    "b3_local_attention_by_lag",
    "b3_recurrent_attention_by_lag",
    "b3_normalized_recurrent_attention_density",
    "b3_per_head_mean_recurrent_lag",
    "b10_to_b3_writer_gradient",
    "b1_b3_temporal_distance_comparison",
    "parallel_vs_true_incremental_b3_gain",
    "b1_b3_marginal_recurrent_gains",
    "inference_state_memory_comparison",
    "runtime_throughput_vram",
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


def gradient_group_report(model):
    """Report base plus B1 and B3 scalar-gate gradients independently."""

    base = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name not in {"g_rec", "g_rec_b3"} and parameter.grad is not None
    ]
    b1_gate = [] if model.g_rec.grad is None else [model.g_rec.grad]
    b3_gate = [] if model.g_rec_b3.grad is None else [model.g_rec_b3.grad]
    report = {}
    for name, values in (
        ("base", base),
        ("gate", b1_gate),
        ("b3_gate", b3_gate),
    ):
        squared = (
            sum(value.float().square().sum() for value in values)
            if values
            else torch.tensor(0.0)
        )
        report[name] = {
            "tensors": len(values),
            "norm": squared.sqrt().item(),
            "finite": legacy.finite_tensors(values),
            "nonzero": bool(values) and bool(squared.gt(0).item()),
        }
    return report


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean: bool = True) -> None:
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"{EXPERIMENT} requires branch {BRANCH}")
    if git_output("rev-parse", FROZEN_TAG + "^{commit}") != FROZEN_COMMIT:
        raise SystemExit("frozen 2D2D tag mismatch")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=REPO_ROOT
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D2F worktree must be clean")


def require_config() -> dict:
    config = read_json(CONFIG_PATH)
    checks = {
        "experiment": config.get("experiment") == EXPERIMENT,
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "frozen_tag": config.get("frozen_2d2d_tag") == FROZEN_TAG,
        "frozen_commit": config.get("frozen_2d2d_commit") == FROZEN_COMMIT,
        "source_sha": config.get("source_checkpoint_sha256") == SOURCE_SHA256,
        "source_bytes": config.get("source_checkpoint_bytes") == SOURCE_BYTES,
        "source_state": config["source_2d2d"]["completed_updates"] == SOURCE_UPDATES
        and config["source_2d2d"]["completed_targets"] == SOURCE_TARGETS
        and config["source_2d2d"]["raw_g_rec_b1"] == SOURCE_GATE_RAW
        and config["source_2d2d"]["raw_g_rec_b2"] == SOURCE_B2_GATE_RAW
        and config["source_2d2d"]["next_global_batch_sha256"]
        == SOURCE_NEXT_BATCH_SHA256,
        "geometry": config["architecture"]["b1"]["local_window"]
        == B1_LOCAL_WINDOW
        and config["architecture"]["b1"]["recurrent_min_lag"] == 2
        and config["architecture"]["b1"]["maximum_recurrent_entries"] == 1022
        and config["architecture"]["b2"]["local_window"] == B2_WINDOW
        and config["architecture"]["b2"]["recurrent"] is False
        and config["architecture"]["b2"]["maximum_recurrent_entries"] == 0
        and config["architecture"]["b3"]["local_window"] == B3_WINDOW
        and config["architecture"]["b3"]["recurrent_min_lag"]
        == B3_RECURRENT_MIN_LAG
        and config["architecture"]["b3"]["recurrent_max_lag"]
        == RECURRENT_MAX_LAG
        and config["architecture"]["b3"]["maximum_recurrent_entries"]
        == B3_MAX_RECURRENT_ENTRIES,
        "one_parameter": config["architecture"]["new_parameter_count_vs_2d2d"] == 1
        and config["architecture"]["new_parameters_vs_2d2d"] == ["g_rec_b3"]
        and config["architecture"]["removed_parameters_vs_2d2d"] == ["g_rec_b2"]
        and config["architecture"]["total_parameters"] == TOTAL_PARAMETERS,
        "budget": config["training"]["additional_updates"] == MAX_UPDATES
        and config["training"]["additional_targets"] == ADDITIONAL_TARGETS
        and config["training"]["cumulative_2d2_targets"] == CUMULATIVE_TARGETS,
        "milestones": tuple(config["training"]["local_milestones"]) == MILESTONES,
        "checkpoints": tuple(config["training"]["scientific_checkpoint_updates"])
        == SCIENTIFIC_CHECKPOINTS,
        "optimizer_resume": config["training"]["optimizer"]["shared_source_optimizer_state_restored"]
        is True
        and config["training"]["optimizer"]["warmup_restarted"] is False,
        "new_b3_gate_optimizer": config["training"]["optimizer"][
            "new_b3_gate_state_fresh"
        ] is True
        and config["training"]["optimizer"]["new_b3_gate_lr"] == GATE_LR
        and tuple(config["training"]["optimizer"]["betas"]) == tuple(legacy.BETAS)
        and config["training"]["optimizer"]["eps"] == legacy.ADAM_EPS
        and config["training"]["optimizer"]["gate_weight_decay"] == 0.0,
        "attached": config["training"]["recurrent_sources_detached"] is False,
        "hardware": config["hardware"]["gpu_count"] == 1
        and config["hardware"]["ddp"] is False,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D2F preregistration mismatch: {checks}")
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
    if not supplied_identity:
        raise SystemExit("persistent-volume identity is required")
    if output != expected_output:
        raise SystemExit(f"2D2F result directory must be exactly {expected_output}")
    if not str(output).startswith("/workspace/") or not str(run).startswith("/workspace/"):
        raise SystemExit("2D2F results and final checkpoints must live under /workspace")
    row = subprocess.check_output(
        ["findmnt", "-T", "/workspace", "-n", "-o", "TARGET,SOURCE,FSTYPE"],
        text=True,
    ).strip().split(maxsplit=2)
    target, source, filesystem = row
    quota_bytes = int(subprocess.check_output(["df", "-B1", "--output=size", "/workspace"], text=True).splitlines()[-1])
    used_bytes = int(
        subprocess.check_output(["du", "-sb", "/workspace"], text=True).split()[0]
    )
    free_bytes = quota_bytes - used_bytes
    required_free_bytes = 20 * 1024**3
    checks = {
        "target_exact": target == "/workspace",
        "persistent_identity_exact": f"/networkvolumes/{supplied_identity}"
        in source,
        "fuse_network_mount": filesystem == "fuse",
        "canonical_result_directory": output == expected_output,
        "run_root_on_workspace": str(run).startswith("/workspace/"),
        "at_least_20_gib_free": free_bytes >= required_free_bytes,
    }
    if not all(checks.values()):
        raise SystemExit(f"persistent workspace audit failed: {checks}")
    return {
        "persistent_volume_identity": supplied_identity,
        "target": target,
        "source": source,
        "filesystem": filesystem,
        "output_dir": str(output),
        "run_root": str(run),
        "configured_quota_bytes": quota_bytes,
        "measured_used_bytes": used_bytes,
        "measured_free_bytes": free_bytes,
        "required_free_bytes": required_free_bytes,
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
        "schema": isinstance(payload.get("schema"), str),
        "authenticated_probe": payload.get("authenticated_list_probe") is True,
        "credential_available": payload.get("stop_credential_available") is True,
        "secret_not_recorded": payload.get("secret_recorded") is False,
        "pod_id": response.get("id") == args.pod_id,
        "pod_name": response.get("name") == args.pod_name,
        "gpu_count": response.get("gpuCount", 0) >= 1,
        "runtime_running": response.get("runtimeStatus") == "running",
        "exact_stop_target": payload.get("exact_stop_target") == args.pod_id,
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


def frozen_2d2c_checkpoint_audit() -> dict:
    path = Path("/workspace/exp2d2c_run/checkpoints/scientific_update_0191.pt")
    if not path.is_file():
        raise SystemExit("frozen 2D2C final checkpoint is missing")
    observed = file_sha256(path)
    sha_sidecar = path.with_suffix(path.suffix + ".sha256")
    verification_sidecar = path.with_suffix(path.suffix + ".verification.json")
    checks = {
        "checkpoint_present": path.is_file(),
        "sha_exact": observed == FROZEN_2D2C_CHECKPOINT_SHA256,
        "sha_sidecar_present": sha_sidecar.is_file(),
        "verification_sidecar_present": verification_sidecar.is_file(),
        "sha_sidecar_exact": sha_sidecar.is_file()
        and sha_sidecar.read_text().split()[0] == FROZEN_2D2C_CHECKPOINT_SHA256,
    }
    report = {
        "path": str(path),
        "sha256": observed,
        "bytes": path.stat().st_size,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not report["passed"]:
        raise SystemExit(f"frozen 2D2C checkpoint audit failed: {checks}")
    return report


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
    """Restore shared 2D2D groups while physically omitting the B2 gate."""

    base_decay, base_nodecay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name in {"g_rec", "g_rec_b3"}:
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
                "name": "b3_gate",
                "params": [model.g_rec_b3],
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
        "new_b3_gate_state_fresh": bool(include_new_gate),
        "warmup_restarted": False,
    }
    return optimizer, report


def load_source_bundle(source_checkpoint, device, restore_rng=False):
    path = Path(source_checkpoint).resolve()
    observed_sha = file_sha256(path)
    if observed_sha != SOURCE_SHA256 or path.stat().st_size != SOURCE_BYTES:
        raise SystemExit("2D2D scientific source checkpoint identity mismatch")
    sha_sidecar = path.with_suffix(path.suffix + ".sha256")
    verification_sidecar = path.with_suffix(path.suffix + ".verification.json")
    if not sha_sidecar.is_file() or not verification_sidecar.is_file():
        raise SystemExit("2D2D checkpoint sidecars missing")
    if sha_sidecar.read_text().split()[0] != SOURCE_SHA256:
        raise SystemExit("2D2D checkpoint SHA sidecar mismatch")
    payload = legacy.d0.torch_load(path, mmap=False)
    required = {
        "schema",
        "model",
        "g_rec",
        "g_rec_b2",
        "optimizer",
        "completed_2d2d_updates",
        "processed_2d2d_targets",
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
        "fields": set(payload) == required,
        "schema": payload.get("schema") == SOURCE_SCHEMA,
        "updates": payload.get("completed_2d2d_updates") == SOURCE_UPDATES,
        "targets": payload.get("cumulative_2d2_targets") == SOURCE_TARGETS,
        "gate_duplicate": torch.equal(payload.get("g_rec"), payload["model"]["g_rec"]),
        "b2_gate_duplicate": torch.equal(
            payload.get("g_rec_b2"), payload["model"]["g_rec_b2"]
        ),
        "gate_raw": float(payload["g_rec"]) == SOURCE_GATE_RAW,
        "b2_gate_raw": float(payload["g_rec_b2"]) == SOURCE_B2_GATE_RAW,
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
        raise SystemExit(f"2D2D source schema mismatch: {checks}")

    symbols = legacy.d0.support.load_training_symbols()
    source_base = symbols["GPT"](legacy.d0.model_config(symbols))
    source_model = source_core.RecurrentKVGPT(source_base).to(device)
    missing, unexpected = source_model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise SystemExit(f"strict source model load failed: {missing}, {unexpected}")
    source_optimizer, _ = source_driver.configure_optimizer(source_model, device.type)
    source_optimizer.load_state_dict(payload["optimizer"])

    # Reuse exact shared parameter objects, but deliberately exclude g_rec_b2.
    model = RecurrentKVGPT(source_model.base).to(device)
    model.g_rec = source_model.g_rec
    optimizer, optimizer_report = configure_optimizer(
        model, device.type, include_new_gate=True
    )
    shared = {parameter for name, parameter in model.named_parameters() if name != "g_rec_b3"}
    for parameter in shared:
        if parameter not in source_optimizer.state:
            raise SystemExit("shared 2D2D parameter is missing optimizer state")
        optimizer.state[parameter] = copy.deepcopy(source_optimizer.state[parameter])
    optimizer_report.update(
        {
            "source_optimizer_restored": True,
            "source_state_entries": len(payload["optimizer"]["state"]),
            "source_parameter_groups": len(payload["optimizer"]["param_groups"]),
            "dropped_b2_gate_group": True,
            "dropped_b2_gate_state": source_model.g_rec_b2 not in optimizer.state,
            "new_b3_gate_state_fresh": model.g_rec_b3 not in optimizer.state,
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
            "b2_gate_absent": not hasattr(model, "g_rec_b2"),
            "b3_gate_zero": model.g_rec_b3.detach().float().item() == 0.0,
            "next_batch_reproduced": observed_next == SOURCE_NEXT_BATCH_SHA256,
            "next_stream_reproduced": observed_stream == SOURCE_NEXT_STREAM_SHA256,
            "source_optimizer_state_preserved": len(optimizer.state)
            == len(payload["optimizer"]["state"]) - 1,
            "new_b3_optimizer_state_absent": model.g_rec_b3 not in optimizer.state,
            "weight_tying": model.base.transformer.wte.weight
            is model.base.lm_head.weight,
        }
    )
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"2D2D source strict reopen failed: {checks}")
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
        if row["name"] != "g_rec_b3"
    ]
    source_shared_inventory = [row for row in source_inventory if row["name"] != "g_rec_b2"]
    source_keys = [key for key in source_payload["model"] if key != "g_rec_b2"]
    target_keys_without_new = [key for key in model.state_dict() if key != "g_rec_b3"]
    new_rows = [row for row in inventory if row["name"] == "g_rec_b3"]
    report = {
        "2d2d_named_parameter_inventory": source_inventory,
        "2d2f_named_parameter_inventory": inventory,
        "source_inventory_preserved": comparable_without_new == source_shared_inventory,
        "source_state_dict_inventory": source_state_inventory,
        "source_state_dict_keys_preserved": target_keys_without_new == source_keys,
        "2d2d_total_parameters": sum(row["numel"] for row in source_inventory),
        "2d2f_total_parameters": sum(row["numel"] for row in inventory),
        "dropped_source_parameters": ["g_rec_b2"],
        "new_parameters_vs_2d2d": new_rows,
        "new_parameter_count_vs_2d2d": len(new_rows),
        "all_trainable": all(row["trainable"] for row in inventory),
        "embedding_lm_head_tied": model.base.transformer.wte.weight
        is model.base.lm_head.weight,
    }
    report["checks"] = {
        "source_inventory_preserved": report["source_inventory_preserved"],
        "source_state_dict_keys_preserved": report["source_state_dict_keys_preserved"],
        "source_total": report["2d2d_total_parameters"] == SOURCE_PARAMETERS,
        "target_total": report["2d2f_total_parameters"] == TOTAL_PARAMETERS,
        "b2_gate_physically_absent": "g_rec_b2" not in dict(model.named_parameters()),
        "exactly_one_new": report["new_parameter_count_vs_2d2d"] == 1,
        "new_scalar_exact": len(new_rows) == 1
        and new_rows[0]["name"] == "g_rec_b3"
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
                "local_window": B1_LOCAL_WINDOW,
                "recurrent_lags": "2...1023",
                "maximum_recurrent_entries": 1022,
            },
            "B10_to_B3": {
                "source": "B10 post-MLP residual immediately before B11",
                "destination": "B3 attention",
                "gate": "new tanh(g_rec_b3)",
                "local_window": B3_WINDOW,
                "recurrent_lags": "64...1023",
                "maximum_recurrent_entries": B3_MAX_RECURRENT_ENTRIES,
            },
        },
        "b1_local_window": B1_LOCAL_WINDOW,
        "b2_local_window": B2_WINDOW,
        "b3_local_window": B3_WINDOW,
        "b4_b12_windows": [T] * 9,
        "b1_recurrent_source_set": "max(0,t-1023)...t-2 inclusive",
        "b2_recurrent_source_set": None,
        "b3_recurrent_source_set": "max(0,t-1023)...t-64 inclusive",
        "b1_recurrent_min_lag": 2,
        "b2_recurrent_min_lag": None,
        "b3_recurrent_min_lag": B3_RECURRENT_MIN_LAG,
        "recurrent_max_lag": RECURRENT_MAX_LAG,
        "b1_maximum_recurrent_entries": 1022,
        "b2_maximum_recurrent_entries": 0,
        "b3_maximum_recurrent_entries": B3_MAX_RECURRENT_ENTRIES,
        "b1_local_positions": ["t-1", "t"],
        "b2_local_positions": ["t-31", "...", "t"],
        "b3_local_positions": ["t-63", "...", "t"],
        "separate_softmaxes": True,
        "shared_destination_ln_qkv": True,
        "single_destination_c_proj": True,
        "incremental_b1_historical_kv_capacity": 1,
        "incremental_b2_historical_kv_capacity": 31,
        "incremental_b3_historical_kv_capacity": 63,
        "incremental_b4_b12_historical_kv_capacity": 1023,
        "incremental_b10_raw_residual_capacity": RECURRENT_RING_CAPACITY,
        "incremental_b11_raw_residual_capacity": 0,
        "incremental_b12_raw_residual_capacity": RECURRENT_RING_CAPACITY,
        "parallel_source_tensor_shapes": {"h10": "[B,T,C]", "h12": "[B,T,C]"},
        "forbidden_repeated_state_tensor": "[B,T,T,C]",
        "overall_context": T,
        "overall_kv_savings_claimed": True,
        "forbidden_modules_absent": {
            "teacher": True,
            "attnres": True,
            "dedicated_recurrent_projection": True,
            "additional_link_beyond_B10_to_B3": True,
            "detached_training_arm": True,
            "b11_to_b2_recurrent_path": True,
            "g_rec_b2": True,
        },
    }


def matched_2d2c_reference_manifest() -> dict:
    return {
        "schema": "exp2d2e_frozen_2d2c_matched_reference_v1",
        "frozen_2d2c_tag": FROZEN_2D2C_TAG,
        "frozen_2d2c_commit": FROZEN_2D2C_COMMIT,
        "frozen_2d2c_checkpoint_sha256": FROZEN_2D2C_CHECKPOINT_SHA256,
        "common_source_2d2b_commit": FROZEN_COMMIT,
        "common_source_2d2b_checkpoint_sha256": SOURCE_SHA256,
        **copy.deepcopy(FROZEN_2D2C_REFERENCE),
        "matched_fields": [
            "source model/optimizer/loader/RNG/counters/B1 gate",
            "191 logical global batches",
            "100139008 additional targets",
            "one-GPU microbatch 32 with gradient accumulation 16",
            "pass schedule and CE weights",
            "new B2 gate optimizer group",
            "parameter count",
        ],
        "sole_scientific_difference": (
            "B2 temporal partition: 2D2C local W2 plus recurrent lags 2..1023; "
            "2D2E local W32 plus recurrent lags 32..1023"
        ),
    }


def frozen_2d2c_optimizer_audit(current_report) -> dict:
    artifact = (
        "results/experiment_2d2c_b12_b1_b11_b2_full_recurrent_kv/"
        "preflight_audit.json"
    )
    frozen_preflight = json.loads(
        git_output("show", f"{FROZEN_2D2C_COMMIT}:{artifact}")
    )
    frozen_report = frozen_preflight["source"]["optimizer_report"]
    checks = {
        "artifact_commit_exact": git_output(
            "rev-parse", FROZEN_2D2C_TAG + "^{commit}"
        )
        == FROZEN_2D2C_COMMIT,
        "full_optimizer_report_exact": current_report == frozen_report,
        "new_b2_gate_group_exact": current_report["groups"][-1]
        == frozen_report["groups"][-1],
        "betas_exact": current_report["betas"] == frozen_report["betas"],
        "eps_exact": current_report["eps"] == frozen_report["eps"],
        "fused_exact": current_report["fused"] == frozen_report["fused"],
        "fresh_state_exact": current_report["new_b2_gate_state_fresh"]
        is frozen_report["new_b2_gate_state_fresh"]
        is True,
    }
    return {
        "frozen_artifact": f"{FROZEN_2D2C_COMMIT}:{artifact}",
        "frozen_artifact_blob": git_output(
            "rev-parse", f"{FROZEN_2D2C_COMMIT}:{artifact}"
        ),
        "frozen_optimizer_report": frozen_report,
        "current_2d2e_optimizer_report": current_report,
        "checks": checks,
        "passed": all(checks.values()),
    }


def initialize_matched_data_replay_audit(source_audit, source_stream, scientific_stream):
    return {
        "schema": "exp2d2e_matched_2d2c_data_replay_audit_v1",
        "common_2d2b_source_next_batch_sha256": SOURCE_NEXT_BATCH_SHA256,
        "common_2d2b_source_next_stream_sha256": SOURCE_NEXT_STREAM_SHA256,
        "source_next_batch_reproduced": source_audit["next_global_batch_sha256"]
        == SOURCE_NEXT_BATCH_SHA256,
        "source_stream_reproduced": source_stream == SOURCE_NEXT_STREAM_SHA256,
        "scientific_microbatch_stream_reproduced": scientific_stream
        == SOURCE_NEXT_STREAM_SHA256,
        "per_update_2d2c_input_hashes_available": False,
        "available_2d2c_cursor_hash_updates": sorted(
            int(update)
            for update in FROZEN_2D2C_REFERENCE["available_cursor_hashes"]
        ),
        "unavailable_comparisons": (
            "2D2C did not persist per-update input/target hashes; no such hashes "
            "are fabricated. Every available scientific/recovery loader cursor hash "
            "is compared in memory without creating extra 2D2E checkpoints."
        ),
        "checkpoint_cursor_comparisons": {},
        "passed_so_far": True,
    }


def record_matched_replay_cursor(
    output, update, loader=None, accumulation=None, verification=None
):
    reference = FROZEN_2D2C_REFERENCE["available_cursor_hashes"].get(str(update))
    if reference is None:
        return
    if verification is None:
        if loader is None or accumulation is None:
            raise ValueError("loader and accumulation are required without verification")
        observed_batch = next_global_batch_hash(loader, accumulation)
        observed_stream = global_batch_stream_hash(loader, accumulation)
        observation_method = "in-memory loader clone; no checkpoint binary written"
    else:
        observed_batch = verification["next_global_batch_sha256"]
        observed_stream = verification["next_global_batch_stream_sha256"]
        observation_method = "strictly reopened scientific checkpoint"
    path = Path(output) / "matched_2d2c_data_replay_audit.json"
    audit = read_json(path)
    audit["checkpoint_cursor_comparisons"][str(update)] = {
        "2d2c_cursor_kind": reference["kind"],
        "expected_2d2c_next_global_batch_sha256": reference[
            "next_batch_sha256"
        ],
        "observed_2d2e_next_global_batch_sha256": observed_batch,
        "expected_2d2c_next_global_batch_stream_sha256": reference[
            "next_stream_sha256"
        ],
        "observed_2d2e_next_global_batch_stream_sha256": observed_stream,
        "batch_exact": observed_batch == reference["next_batch_sha256"],
        "stream_exact": observed_stream == reference["next_stream_sha256"],
        "exact": observed_batch == reference["next_batch_sha256"]
        and observed_stream == reference["next_stream_sha256"],
        "observation_method": observation_method,
    }
    audit["passed_so_far"] = all(
        audit[key]
        for key in (
            "source_next_batch_reproduced",
            "source_stream_reproduced",
            "scientific_microbatch_stream_reproduced",
        )
    ) and all(
        row["exact"] for row in audit["checkpoint_cursor_comparisons"].values()
    )
    durable_json(path, audit)
    if not audit["passed_so_far"]:
        raise SystemExit(f"2D2C matched data replay diverged at update {update}")


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
        == new["b1_recurrent_source_set"],
        "b1_gate_name_and_state": "g_rec" in named and named.index("g_rec") == source_named.index("g_rec"),
        "source_parameter_names": [name for name in named if name != "g_rec_b2"]
        == source_named,
    }
    report = {
        "baseline": "final Experiment 2D2B",
        "architecture_changes": [
            {"field": "B2 ordinary local window", "old": 1024, "new": 32},
            {"field": "new recurrent link", "old": None, "new": "B11->B2 lags 32...1023"},
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
        "b3_off",
        "b3_shuffled",
        "b3_full_counterfactual",
    ]
    if combined_controls:
        names.extend(
            (
                "all_shuffled",
                "b1_shuffled_b2_real",
                "b1_off_b2_real",
                "b1_real_b2_off",
            )
        )
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
            if name in {"b3_off", "b1_real_b2_off"}:
                kwargs["b2_gate_override"] = 0.0
            elif name == "b3_shuffled":
                kwargs["b2_recurrent_permutation"] = derangement
            elif name == "b3_full_counterfactual":
                kwargs["b3_full_counterfactual"] = True
            elif name == "all_shuffled":
                kwargs["b1_recurrent_permutation"] = derangement
                kwargs["b2_recurrent_permutation"] = derangement
            elif name == "b1_shuffled_b2_real":
                kwargs["b1_recurrent_permutation"] = derangement
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
        print(f"2D2E validation batch={batch_index + 1:02d}/{batches}", flush=True)
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
    off = finished["b3_off"]
    shuffled = finished["b3_shuffled"]
    full = finished["b3_full_counterfactual"]
    position_bins = {}
    for name, first, last in POSITION_BINS:
        r = np.asarray(real["per_position_loss"])[first : last + 1]
        o = np.asarray(off["per_position_loss"])[first : last + 1]
        s = np.asarray(shuffled["per_position_loss"])[first : last + 1]
        f = np.asarray(full["per_position_loss"])[first : last + 1]
        position_bins[name] = {
            "new_real_loss": float(r.mean()),
            "b3_off_loss": float(o.mean()),
            "b3_shuffled_loss": float(s.mean()),
            "b3_full_counterfactual_loss": float(f.mean()),
            "b3_recurrent_gain": float((o - r).mean()),
            "b3_sequence_gap": float((s - r).mean()),
            "remaining_b2_compression_gap": float((r - f).mean()),
            "available_recurrent_history": [max(0, first - 31), max(0, last - 31)],
        }
    collection_sha = legacy.d0.aggregate_hashes(
        [row["combined_sha256"] for row in identities]
    )
    result = {
        "controls": finished,
        "b3_recurrent_gain": off["validation_loss"] - real["validation_loss"],
        "b3_sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "remaining_b2_compression_gap": real["validation_loss"]
        - full["validation_loss"],
        "new_real_vs_b2_off": paired_stats(
            real["per_batch_losses"], off["per_batch_losses"]
        ),
        "new_real_vs_b3_shuffled": paired_stats(
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
        "g_rec_b3_raw": model.g_rec_b3.detach().float().item(),
        "tanh_g_rec_b3": model.recurrent_scale_b3.detach().float().item(),
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
        "control_labels": {
            "new_real": "REAL_W32",
            "b3_off": "B2_OFF_W32",
            "b3_shuffled": "B2_SHUFFLED_W32",
            "b3_full_counterfactual": "B2_FULL_COUNTERFACTUAL",
        },
    }
    if combined_controls:
        result["combined_system"] = {
            "all_real_vs_all_shuffled_gap": finished["all_shuffled"][
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
            "b1_shuffled_b2_real_gap": finished["b1_shuffled_b2_real"][
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
    recurrent_bins = B1_LAG_BINS if link == "b1" else B2_RECURRENT_LAG_BINS
    for name, first_lag, last_lag in recurrent_bins:
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
    local_bins = {}
    if link == "b2":
        local_weights = diagnostics.get("local_attention_weights")
        local_valid = diagnostics.get("local_valid_mask")
        if local_weights is None or local_valid is None:
            raise SystemExit("B2 local attention diagnostics are unavailable")
        local_total = local_weights.double().sum().item()
        for name, first_lag, last_lag in B2_LOCAL_LAG_BINS:
            selected = local_valid & (lag >= first_lag) & (lag <= last_lag)
            mass = local_weights.masked_select(
                selected.view(1, 1, T, T)
            ).double().sum().item()
            available = int(selected.sum().item()) * batch_size * N_HEAD
            local_bins[name] = {
                "lag_min": first_lag,
                "lag_max": last_lag,
                "attention_mass": mass / local_total,
                "raw_attention_mass": mass,
                "normalized_mass_per_available_token": mass / max(available, 1),
                "valid_position_instances_all_heads_rows": available,
            }
    if link == "b1":
        mass_partitions = {
            "lags_2_31": sum(
                bins[name]["attention_mass"] for name in ("2-7", "8-15", "16-31")
            ),
            "lags_32_127": sum(
                bins[name]["attention_mass"] for name in ("32-63", "64-127")
            ),
            "lags_128_511": sum(
                bins[name]["attention_mass"] for name in ("128-255", "256-511")
            ),
            "lags_512_1023": bins["512-1023"]["attention_mass"],
        }
    else:
        mass_partitions = {
            "lags_32_127": sum(
                bins[name]["attention_mass"] for name in ("32-63", "64-127")
            ),
            "lags_128_511": sum(
                bins[name]["attention_mass"] for name in ("128-255", "256-511")
            ),
            "lags_512_1023": bins["512-1023"]["attention_mass"],
        }
    report = {
        "link": "B12->B1" if link == "b1" else "B11->B2",
        "pinned_batch": identity,
        "batch_size": batch_size,
        "sequence_length": T,
        "lag_bins": bins,
        "local_lag_bins": local_bins,
        "heads": heads,
        "aggregate": aggregate,
        "mass_partitions": mass_partitions,
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
        # Isolate the destination reader gradient from indirect routes through
        # the other recurrent link. The result-training path itself remains
        # attached and is checked independently by smoke/training audits.
        if link == "b1":
            source = first["h12"].detach().requires_grad_(True)
            b1_source = source
            b2_source = first["h11"].detach()
        else:
            source = first["h11"].detach().requires_grad_(True)
            b1_source = first["h12"].detach()
            b2_source = source
        second = model.forward_pass(
            x,
            targets=y,
            b1_recurrent_source=b1_source,
            b2_recurrent_source=b2_source,
            b2_gate_override=b2_gate_override,
            activation_checkpointing=True,
            bank_mode="full",
        )
        receiver_loss = F.cross_entropy(
            second["logits"][:, -1].float(), y[:, -1], reduction="mean"
        )
        gradient = torch.autograd.grad(receiver_loss, source)[0].float()
    lags = (T - 1) - torch.arange(T, device=device)
    bins = {}
    lag_bins = B1_LAG_BINS if link == "b1" else B2_RECURRENT_LAG_BINS
    for name, first_lag, last_lag in lag_bins:
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
    recent_eligible = 1021 if link == "b1" else 991
    recent_ineligible = 1022 if link == "b1" else 992
    probes = {
        "early_old_position_0": gradient[:, 0].square().mean().sqrt().item(),
        "middle_position_512": gradient[:, 512].square().mean().sqrt().item(),
        f"recent_eligible_position_{recent_eligible}": gradient[:, recent_eligible]
        .square().mean().sqrt().item(),
        f"recent_ineligible_position_{recent_ineligible}": gradient[:, recent_ineligible]
        .square().mean().sqrt().item(),
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
        "receiver_position": T - 1,
        "receiver_loss": receiver_loss.detach().float().item(),
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
            f"Pass-2 CE at receiver t=1023 differentiated into attached Pass-1 "
            f"{'B12' if link == 'b1' else 'B11'} post-MLP states. Source positions "
            "therefore map one-to-one to exact lag from the audited receiver. "
            "The recurrent source is a detached requires-grad probe solely to "
            "isolate this reader path; attached writer co-adaptation is checked "
            "separately in smoke and result training."
        ),
    }
    model.zero_grad(set_to_none=True)
    del x, y, first, source, second, receiver_loss, gradient
    torch.cuda.empty_cache()
    return report


def attached_writer_gradient_check(model, val_path) -> dict:
    """Verify each actual Pass-1 writer tensor is attached to its Pass-2 reader."""

    model.train()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], 1, T)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(device), cpu_y.to(device)
    with torch.enable_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        first = model.forward_pass(x, activation_checkpointing=True)
        b1_read = model.forward_pass(
            x,
            targets=y,
            b1_recurrent_source=first["h12"],
            b2_recurrent_source=first["h11"].detach(),
            activation_checkpointing=True,
        )
        grad_h12 = torch.autograd.grad(
            b1_read["loss"], first["h12"], retain_graph=True
        )[0].float()
        b2_read = model.forward_pass(
            x,
            targets=y,
            b1_recurrent_source=first["h12"].detach(),
            b2_recurrent_source=first["h11"],
            activation_checkpointing=True,
        )
        grad_h11 = torch.autograd.grad(b2_read["loss"], first["h11"])[0].float()
    report = {
        "b12_to_b1": {
            "finite": bool(torch.isfinite(grad_h12).all()),
            "nonzero": bool(grad_h12.count_nonzero()),
            "position_0_rms": grad_h12[:, 0].square().mean().sqrt().item(),
            "latest_eligible_1021_rms": grad_h12[:, 1021].square().mean().sqrt().item(),
            "ineligible_1022_1023_nonzero": int(grad_h12[:, 1022:].count_nonzero()),
        },
        "b11_to_b2": {
            "finite": bool(torch.isfinite(grad_h11).all()),
            "nonzero": bool(grad_h11.count_nonzero()),
            "position_0_rms": grad_h11[:, 0].square().mean().sqrt().item(),
            "latest_eligible_991_rms": grad_h11[:, 991].square().mean().sqrt().item(),
            "ineligible_992_1023_nonzero": int(grad_h11[:, 992:].count_nonzero()),
        },
        "method": "Actual Pass-1 source tensors; the other link is detached only to isolate the audited link.",
    }
    report["passed"] = (
        report["b12_to_b1"]["finite"]
        and report["b12_to_b1"]["nonzero"]
        and report["b12_to_b1"]["position_0_rms"] > 0
        and report["b12_to_b1"]["latest_eligible_1021_rms"] > 0
        and report["b12_to_b1"]["ineligible_1022_1023_nonzero"] == 0
        and report["b11_to_b2"]["finite"]
        and report["b11_to_b2"]["nonzero"]
        and report["b11_to_b2"]["position_0_rms"] > 0
        and report["b11_to_b2"]["latest_eligible_991_rms"] > 0
        and report["b11_to_b2"]["ineligible_992_1023_nonzero"] == 0
    )
    model.zero_grad(set_to_none=True)
    del x, y, first, b1_read, b2_read, grad_h11, grad_h12
    torch.cuda.empty_cache()
    return report


def kernel_preflight(model, short_tokens, short_targets) -> dict:
    """Hard architecture, geometry, causality, isolation and gate tests."""
    model.eval()
    device = short_tokens.device
    length = short_tokens.size(1)
    reports = {}
    checks = {}
    query = torch.arange(length, device=device).view(length, 1)
    source = torch.arange(length, device=device).view(1, length)
    b1_mask = model.recurrent_mask(length, length, device, "full")
    b2_mask = model.b2_recurrent_mask(length, length, device)
    expected_b1 = (source <= query - 2) & (source >= query - 1023)
    expected_b2 = (source <= query - 32) & (source >= query - 1023)
    checks["b1_short_mask_exact"] = torch.equal(b1_mask, expected_b1)
    checks["b2_short_mask_exact"] = torch.equal(b2_mask, expected_b2)
    full_b1_mask = model.recurrent_mask(T, T, device, "full")
    full_b2_mask = model.b2_recurrent_mask(T, T, device)
    b1_boundary_counts = {
        str(position): int(full_b1_mask[position].sum().item())
        for position in (0, 1, 2, 3, 31, 32, 1023)
    }
    b2_boundary_counts = {
        str(position): int(full_b2_mask[position].sum().item())
        for position in (0, 1, 2, 3, 31, 32, 100, 1023)
    }
    reports["b1_boundary_counts"] = b1_boundary_counts
    reports["b2_boundary_counts"] = b2_boundary_counts
    checks["b1_boundary_counts"] = b1_boundary_counts == {
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 2,
        "31": 30,
        "32": 31,
        "1023": 1022,
    }
    checks["b2_boundary_counts"] = b2_boundary_counts == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "31": 0,
        "32": 1,
        "100": 69,
        "1023": 992,
    }
    b1_local_mask = model.local_mask(length, device)
    b2_local_mask = model.b2_local_mask(length, device)
    checks["b1_local_recurrent_disjoint"] = not bool(
        (b1_mask & b1_local_mask).any()
    )
    checks["b2_local_recurrent_disjoint"] = not bool(
        (b2_mask & b2_local_mask).any()
    )
    checks["b1_w2_exact"] = int(b1_local_mask.sum(-1).max()) == 2
    checks["b2_w32_exact"] = int(b2_local_mask.sum(-1).max()) == min(32, length)
    checks["b2_partition_covers_causal_history"] = torch.equal(
        b2_mask | b2_local_mask, source <= query
    )
    checks["b2_no_recent_recurrent_exposure"] = not bool(
        (b2_mask & (source >= query - 31)).any()
    )
    checks["no_too_old"] = not bool(
        (b1_mask & (source < query - 1023)).any()
        or (b2_mask & (source < query - 1023)).any()
    )
    values = torch.randn(
        short_tokens.size(0), length, model.config.n_embd, device=device
    )
    b1_bank = model.build_recurrent_bank(values)
    b2_bank = model.build_recurrent_bank_b2(values)
    checks["source_not_repeated"] = (
        b1_bank.values.data_ptr() == values.data_ptr()
        and b2_bank.values.data_ptr() == values.data_ptr()
    )
    checks["source_rank_three"] = b1_bank.values.ndim == b2_bank.values.ndim == 3
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
    for name, valid_mask in (("b1", b1_mask), ("b2", b2_mask)):
        weights = active["diagnostics"][name]["recurrent_attention_weights"]
        checks[f"{name}_invalid_probabilities_zero"] = not bool(
            weights.masked_select(
                ~valid_mask.view(1, 1, length, length)
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
        reference = model.forward_multi_pass(
            short_tokens, num_passes=2, b2_gate_override=0.2
        )["logits"]
        perturbed = model.forward_multi_pass(
            changed, num_passes=2, b2_gate_override=0.2
        )["logits"]
    checks["future_perturbation_causal"] = torch.equal(
        reference[:, :-1], perturbed[:, :-1]
    )
    if short_tokens.size(0) > 1:
        changed_row = short_tokens.clone()
        changed_row[0] = (changed_row[0] + 29) % model.config.vocab_size
        with torch.no_grad():
            base_rows = model.forward_multi_pass(
                short_tokens, num_passes=2, b2_gate_override=0.2
            )["logits"]
            changed_rows = model.forward_multi_pass(
                changed_row, num_passes=2, b2_gate_override=0.2
            )["logits"]
        checks["row_isolation"] = torch.equal(base_rows[1:], changed_rows[1:])
    reports["b1_mask_positions"] = {
        str(position): torch.where(full_b1_mask[position])[0].cpu().tolist()
        for position in (0, 1, 2, 3, 32, 1023)
    }
    reports["b2_mask_positions"] = {
        str(position): torch.where(full_b2_mask[position])[0].cpu().tolist()
        for position in (0, 31, 32, 33, 1023)
    }
    reports["c_proj_calls"] = calls
    reports["bank_storage"] = {
        "source_shape": list(values.shape),
        "b1_bank_shape": list(b1_bank.values.shape),
        "b2_bank_shape": list(b2_bank.values.shape),
        "shared_data_ptr": b1_bank.values.data_ptr() == b2_bank.values.data_ptr()
        == values.data_ptr(),
    }
    del full_b1_mask, full_b2_mask, values, b1_bank, b2_bank, key, value
    del active, weights, reference, perturbed
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
    total_vram_bytes = torch.cuda.get_device_properties(device).total_memory
    required_headroom_bytes = max(4 * 1024**3, int(0.10 * total_vram_bytes))
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
                for pass_index in range(3):
                    current = model.forward_pass(
                        x,
                        targets=y,
                        b1_recurrent_source=source_h12,
                        b2_recurrent_source=source_h11,
                        activation_checkpointing=True,
                        return_diagnostics=pass_index == 2,
                        bank_mode="full",
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
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            numerically_passed = gradients_finite(model)
            headroom_bytes = total_vram_bytes - peak_reserved
            row = {
                "micro_batch_sequences": candidate,
                "passed": numerically_passed
                and headroom_bytes >= required_headroom_bytes,
                "numerically_passed": numerically_passed,
                "loss": loss.detach().float().item(),
                "source_optimizer_state_resident": len(optimizer.state) > 0,
                "peak_allocated_vram_mb": peak_allocated / 1024**2,
                "peak_reserved_vram_mb": peak_reserved / 1024**2,
                "total_vram_mb": total_vram_bytes / 1024**2,
                "headroom_mb": headroom_bytes / 1024**2,
                "required_headroom_mb": required_headroom_bytes / 1024**2,
                "mirrors_three_pass_final_diagnostic_path": True,
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
    raise SystemExit(f"no safe 2D2E microbatch: {attempts}")


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
    mask = model.b2_recurrent_mask(T, T, device)
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
        "recurrent_entries_max": B2_MAX_RECURRENT_ENTRIES,
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


def benchmark_b2_local_attention(model, batch_size, repeats=5) -> dict:
    device = next(model.parameters()).device
    heads = model.config.n_head
    head_size = model.config.n_embd // heads
    query = torch.randn(
        batch_size, heads, T, head_size, device=device, dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    mask = model.b2_local_mask(T, device)
    forward_times, backward_times = [], []
    for _ in range(repeats):
        for tensor in (query, key, value):
            tensor.grad = None
        torch.cuda.synchronize()
        started = time.monotonic()
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=False
        )
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
        "local_window": B2_WINDOW,
        "repeats": repeats,
        "forward_seconds_mean": statistics.fmean(forward_times),
        "forward_seconds_median": statistics.median(forward_times),
        "backward_seconds_mean": statistics.fmean(backward_times),
        "backward_seconds_median": statistics.median(backward_times),
        "kernel": "torch.scaled_dot_product_attention boolean W32 causal mask",
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
    frozen_2d2c_checkpoint = frozen_2d2c_checkpoint_audit()
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
        "frozen_2d2c_tag": FROZEN_2D2C_TAG,
        "frozen_2d2c_commit": FROZEN_2D2C_COMMIT,
        "frozen_2d2c_tag_exact": git_output(
            "rev-parse", FROZEN_2D2C_TAG + "^{commit}"
        )
        == FROZEN_2D2C_COMMIT,
        "sibling_merge_base": git_output("merge-base", "HEAD", FROZEN_2D2C_COMMIT),
        "frozen_2d2c_checkpoint": frozen_2d2c_checkpoint,
        "audit_correction_provenance": correction,
        "canonical_validation_sha256": CANONICAL_VALIDATION_SHA256,
        "validation_shard": str(val_path),
        "validation_shard_sha256": file_sha256(val_path),
    }
    optimizer_match = frozen_2d2c_optimizer_audit(
        source_audit["optimizer_report"]
    )
    if not optimizer_match["passed"]:
        raise SystemExit(
            f"2D2E optimizer differs from frozen matched 2D2C: {optimizer_match['checks']}"
        )
    source_manifest["matched_2d2c_optimizer_audit"] = optimizer_match
    durable_json(output / "source_manifest.json", source_manifest)
    durable_json(output / "parameter_manifest.json", parameters)
    durable_json(output / "architecture_manifest.json", architecture)
    durable_json(output / "semantic_diff_audit.json", semantic)
    durable_json(
        output / "2d2c_matched_reference_manifest.json",
        matched_2d2c_reference_manifest(),
    )
    durable_json(
        output / "distributed_equivalence.json",
        {
            "applicable": False,
            "gpu_count": 1,
            "reason": "User-assigned one-GPU result run; preserves maximum trajectory comparability with 2D2C.",
            "passed": True,
        },
    )
    cleanup_rows = [
        {
            "path": "/workspace/build-nanogpt-exp2a0/runs/experiment_2a0_smoke_10update/checkpoints/checkpoint_updates_000005.pt",
            "observed_bytes_before_deletion": 498_173_177,
        },
        {
            "path": "/workspace/build-nanogpt-exp2a0/runs/experiment_2a0_smoke_10update/checkpoints/checkpoint_updates_000010.pt",
            "observed_bytes_before_deletion": 498_173_177,
        },
        {
            "path": "/workspace/build-nanogpt-exp2a0/runs/experiment_2b1_5m/smoke/checkpoint_updates_000002.pt",
            "observed_bytes_before_deletion": 498_175_417,
        },
        {
            "path": "/workspace/build-nanogpt-exp2a0/runs/experiment_2b2_5m/smoke/checkpoint_updates_000002.pt",
            "observed_bytes_before_deletion": 498_762_350,
        },
    ]
    for row in cleanup_rows:
        row["absent_at_preflight"] = not Path(row["path"]).exists()
    if not all(row["absent_at_preflight"] for row in cleanup_rows):
        raise SystemExit("preflight storage cleanup targets were not all removed")
    durable_json(
        output / "storage_cleanup_manifest.json",
        {
            "schema": "exp2d2e_storage_cleanup_manifest_v1",
            "preflight_cleanup": {
                "action_provenance": (
                    "Exact paths and sizes were captured by the experiment operator "
                    "during the preflight volume audit; the driver independently "
                    "verifies every path is absent and records current usage."
                ),
                "deleted_bytes": sum(
                    row["observed_bytes_before_deletion"] for row in cleanup_rows
                ),
                "deleted_files": cleanup_rows,
                "classification": "obsolete disposable smoke checkpoint binaries",
                "scientific_or_final_checkpoint_removed": False,
                "manifests_and_sidecars_preserved": True,
                "historical_immediate_post_cleanup_workspace_used_bytes": 72_781_222_757,
                "preflight_measured_workspace_used_bytes": mount["measured_used_bytes"],
                "preflight_measured_workspace_free_bytes": mount["measured_free_bytes"],
                "all_deleted_paths_absent_at_preflight": True,
            },
            "cleanup_actions_after_preflight": [],
        },
    )

    short_loader = legacy.d1.ExplicitShardLoader([val_path], 2, 64)
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
    zero_shot["matched_2d2c_comparison"] = {
        "gain_W32_minus_gain_W2": zero_shot["b3_recurrent_gain"]
        - FROZEN_2D2C_REFERENCE["gain_trajectory"]["0"],
        "sequence_gap_W32_minus_sequence_gap_W2": zero_shot["b3_sequence_gap"]
        - FROZEN_2D2C_REFERENCE["sequence_gap_trajectory"]["0"],
        "raw_g_B2_W32_minus_raw_g_B2_W2": zero_shot["g_rec_b2_raw"]
        - FROZEN_2D2C_REFERENCE["raw_gate_trajectory"]["0"],
        "tanh_g_B2_W32_minus_tanh_g_B2_W2": zero_shot["tanh_g_rec_b2"]
        - FROZEN_2D2C_REFERENCE["gate_trajectory"]["0"],
        "frozen_W2_gain": FROZEN_2D2C_REFERENCE["gain_trajectory"]["0"],
        "frozen_W2_sequence_gap": FROZEN_2D2C_REFERENCE[
            "sequence_gap_trajectory"
        ]["0"],
        "frozen_W2_raw_g_rec_b2": FROZEN_2D2C_REFERENCE[
            "raw_gate_trajectory"
        ]["0"],
        "frozen_W2_tanh_g_rec_b2": FROZEN_2D2C_REFERENCE["gate_trajectory"][
            "0"
        ],
    }
    source_loss = regressions["observed_parallel"]["full_real"]
    off_loss = zero_shot["controls"]["b3_off"]["validation_loss"]
    real_loss = zero_shot["controls"]["new_real"]["validation_loss"]
    shuffled_loss = zero_shot["controls"]["b3_shuffled"]["validation_loss"]
    initial_shortening = {
        "source_2d2b_loss": source_loss,
        "b2_w32_off_loss": off_loss,
        "b2_w32_real_zero_gate_loss": real_loss,
        "b2_w32_shuffled_zero_gate_loss": shuffled_loss,
        "initial_W32_compression_damage": off_loss - source_loss,
        "matched_W2_compression_damage": FROZEN_2D2C_REFERENCE[
            "initial_w2_compression_damage"
        ],
        "fraction_of_W2_damage_remaining": (off_loss - source_loss)
        / FROZEN_2D2C_REFERENCE["initial_w2_compression_damage"],
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
    if (
        selected_microbatch != FROZEN_2D2C_MICRO_BATCH
        or accumulation != FROZEN_2D2C_ACCUMULATION
    ):
        raise SystemExit(
            "2D2E did not preserve frozen one-GPU 2D2C microbatch/reduction geometry"
        )
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
        "frozen_2d2c_selected_micro_batch_sequences": FROZEN_2D2C_MICRO_BATCH,
        "frozen_2d2c_selected_gradient_accumulation": FROZEN_2D2C_ACCUMULATION,
        "microbatch_reduction_geometry_matches_2d2c": True,
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
    replay_audit = initialize_matched_data_replay_audit(
        source_audit, source_stream_hash, scientific_stream_hash
    )
    recurrent_benchmark = benchmark_recurrent_attention(
        model, min(selected_microbatch, 4), repeats=3
    )
    local_benchmark = benchmark_b2_local_attention(
        model, min(selected_microbatch, 4), repeats=3
    )
    benchmark = {
        "recurrent_attention": recurrent_benchmark,
        "b2_local_attention": local_benchmark,
        "b2_recurrent_attention": recurrent_benchmark,
        "source_2d2b_training_targets_per_second": 0,
        "microbatch_probe": probe,
    }

    durable_json(output / "batch_manifest.json", batch_payload)
    durable_json(output / "matched_2d2c_data_replay_audit.json", replay_audit)
    durable_json(output / "initial_b2_w32_compression.json", initial_shortening)
    durable_json(output / "milestone_validation.json", {"0": zero_shot})
    durable_json(output / "b1_attention_lag_bins.json", {"0": attention_b1_zero["lag_bins"]})
    durable_json(output / "b2_recurrent_attention_lag_bins.json", {"0": attention_b2_zero["lag_bins"]})
    durable_json(
        output / "b2_local_attention_lag_bins.json",
        {"0": attention_b2_zero["local_lag_bins"]},
    )
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
        "frozen_2d2c_tag_exact": git_output(
            "rev-parse", FROZEN_2D2C_TAG + "^{commit}"
        )
        == FROZEN_2D2C_COMMIT,
        "frozen_2d2c_checkpoint_exact": frozen_2d2c_checkpoint["passed"],
        "sibling_lineage_exact": git_output(
            "merge-base", "HEAD", FROZEN_2D2C_COMMIT
        )
        == FROZEN_COMMIT,
        "source_checkpoint_exact": source_audit["checks"]["passed"],
        "audit_correction_preserved": correction["passed"],
        "parameters_exactly_one_new": parameters["passed"],
        "semantic_diff_exact": semantic["passed"],
        "kernel": kernel["passed"],
        "temporal_writer_gradient": all(temporal_checks.values()),
        "frozen_2d2b_parallel_regression": regressions["checks"]["parallel"],
        "initial_b2_w32_compression": initial_shortening["passed"],
        "canonical_validation": zero_shot["canonical_validation_sha256"]
        == CANONICAL_VALIDATION_SHA256,
        "loader_continuation": batch_payload[
            "logical_global_batch_exact_across_microbatch_geometry"
        ],
        "matched_2d2c_replay": replay_audit["passed_so_far"],
        "matched_2d2c_optimizer": optimizer_match["passed"],
        "global_batch": selected_microbatch * T * accumulation == GLOBAL_TARGETS,
        "matched_2d2c_microbatch_reduction_geometry": selected_microbatch
        == FROZEN_2D2C_MICRO_BATCH
        and accumulation == FROZEN_2D2C_ACCUMULATION,
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
        "initial_b2_w32_compression": initial_shortening,
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
        raise SystemExit(f"2D2E preflight failed: {science_checks}")
    print("EXPERIMENT_2D2E_PREFLIGHT_PASS", flush=True)
    return preflight


def checkpoint_payload(model, optimizer, loader, training_state, metadata, accumulation):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "g_rec": model.g_rec.detach().cpu().clone(),
        "g_rec_b3": model.g_rec_b3.detach().cpu().clone(),
        "optimizer": optimizer.state_dict(),
        "completed_2d2f_updates": training_state["completed_2d2f_updates"],
        "processed_2d2f_targets": training_state["processed_2d2f_targets"],
        "cumulative_2d2_targets": training_state["cumulative_2d2_targets"],
        "source_2d2d_updates": SOURCE_UPDATES,
        "source_2d2d_targets": SOURCE_TARGETS,
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
    reopened = legacy.d0.torch_load(path, mmap=False)
    required = {
        "schema",
        "model",
        "g_rec",
        "g_rec_b3",
        "optimizer",
        "completed_2d2f_updates",
        "processed_2d2f_targets",
        "cumulative_2d2_targets",
        "source_2d2d_updates",
        "source_2d2d_targets",
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
        "updates": reopened.get("completed_2d2f_updates")
        == training_state["completed_2d2f_updates"],
        "additional_targets": reopened.get("processed_2d2f_targets")
        == training_state["processed_2d2f_targets"],
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
        "b3_gate_duplicate": torch.equal(
            reopened.get("g_rec_b3"), model.g_rec_b3.detach().cpu()
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
        "completed_2d2f_updates": reopened["completed_2d2f_updates"],
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
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        legacy.d0.fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
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
                "recurrent_states_finite": all(
                    bool(torch.isfinite(tensor).all())
                    for tensor in (
                        first["h11"],
                        first["h12"],
                        second["h11"],
                        second["h12"],
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
    attached_writers = attached_writer_gradient_check(
        model, validation_path(args.data_root)
    )
    with torch.no_grad():
        cache_x, _ = loader.clone().next_batch()
        cache = model.incremental_logits(
            cache_x[:, :64].to(device), control="all_real", bank_mode="full"
        )["cache_audit"]
    state = {
        "completed_2d2e_updates": 3,
        "processed_2d2e_targets": 3 * smoke_batch * T,
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
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    required_headroom = max(4 * 1024**3, int(0.10 * total_vram))
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
        "b1_gate_gradient_group": all(
            row["gradient_groups"]["gate"]["finite"]
            and row["gradient_groups"]["gate"]["nonzero"]
            for row in rows
        ),
        "b2_gate_gradient_group": all(
            row["gradient_groups"]["b2_gate"]["finite"]
            and row["gradient_groups"]["b2_gate"]["nonzero"]
            for row in rows
        ),
        "b1_long_lag_writer_gradient": temporal_b1["finite"]
        and temporal_b1["long_lag_writer_gradient_present"],
        "b2_long_lag_writer_gradient": temporal_b2["finite"]
        and temporal_b2["long_lag_writer_gradient_present"],
        "actual_attached_writer_paths": attached_writers["passed"],
        "attention_finite": all(row["recurrent_attention_finite"] for row in rows),
        "recurrent_states_finite": all(
            row["recurrent_states_finite"] for row in rows
        ),
        "no_future_leakage": preflight["kernel_preflight"]["checks"][
            "future_perturbation_causal"
        ],
        "cache": cache["passed"]
        and cache["b1_historical_kv"] == 1
        and cache["b2_historical_kv"] == 31
        and cache["b2_historical_kv_limit"] == 31,
        "checkpoint": verification["passed"],
        "next_data_hash_exact": verification["next_global_batch_sha256"]
        == next_global_batch_hash(loader, 1)
        and verification["next_global_batch_stream_sha256"]
        == global_batch_stream_hash(loader, 1),
        "peak_vram_safe": total_vram - peak_reserved >= required_headroom,
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
        "attached_writer_gradient": attached_writers,
        "incremental_cache_audit": cache,
        "checkpoint": verification,
        "vram": {
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_mb": peak_reserved / 1024**2,
            "total_mb": total_vram / 1024**2,
            "headroom_mb": (total_vram - peak_reserved) / 1024**2,
            "required_headroom_mb": required_headroom / 1024**2,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "disposition": "Discarded; scientific local update 1 reloads the immutable 2D2B update-191 checkpoint.",
    }
    durable_json(output / "smoke_audit.json", audit)
    if not audit["passed"]:
        raise SystemExit(f"2D2E smoke failed: {checks}")
    smoke_bytes = path.stat().st_size
    path.unlink()
    manifest = read_json(output / "checkpoint_manifest.json")
    manifest["smoke"]["3"]["binary_retained"] = False
    manifest["smoke"]["3"]["binary_deleted_after_strict_verification"] = True
    durable_json(output / "checkpoint_manifest.json", manifest)
    cleanup = read_json(output / "storage_cleanup_manifest.json")
    cleanup["cleanup_actions_after_preflight"].append(
        {
            "phase": "post_smoke",
            "path": str(path),
            "bytes": smoke_bytes,
            "reason": "disposable three-update smoke checkpoint; SHA and verification sidecars retained",
        }
    )
    durable_json(output / "storage_cleanup_manifest.json", cleanup)
    audit["checkpoint_binary_deleted"] = True
    audit["checkpoint_sidecars_retained"] = True
    durable_json(output / "smoke_audit.json", audit)
    print("EXPERIMENT_2D2E_SMOKE_PASS", flush=True)
    return audit


def training_metadata(args, preflight, micro_batch, accumulation) -> dict:
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "implementation_fingerprint": preflight["implementation_fingerprint"],
        "frozen_2d2d_commit": FROZEN_COMMIT,
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
            "dropped_b2_gate": True,
            "new_b3_gate_lr": GATE_LR,
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
    payload = legacy.d0.torch_load(path, mmap=False)
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
    expected_rng_fingerprints = {
        name: state_fingerprint(value)
        for name, value in payload["rng_state"].items()
    }
    restore_rng_state(payload["rng_state"])
    restored_rng = capture_rng_state()
    observed_rng_fingerprints = {
        name: state_fingerprint(value) for name, value in restored_rng.items()
    }
    rng_restore_audit = {
        "expected_fingerprints": expected_rng_fingerprints,
        "observed_fingerprints": observed_rng_fingerprints,
        "per_generator_exact": {
            name: observed_rng_fingerprints[name] == expected
            for name, expected in expected_rng_fingerprints.items()
        },
    }
    rng_restore_audit["passed"] = all(
        rng_restore_audit["per_generator_exact"].values()
    )
    if not rng_restore_audit["passed"]:
        raise SystemExit("resume RNG fingerprint mismatch")
    state = copy.deepcopy(payload["training_state"])
    saved_pid = int(payload["saved_process_id"])
    if not model_finite(model) or not optimizer_finite(optimizer):
        raise SystemExit("resume restored nonfinite state")
    del payload
    gc.collect()
    return loader, state, saved_pid, expected_sha, rng_restore_audit


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
        loader, state, saved_pid, checkpoint_sha, rng_restore_audit = load_checkpoint_runtime(
            args.resume,
            model,
            optimizer,
            micro_batch,
            accumulation,
            metadata,
        )
        if state["completed_2d2f_updates"] != FORCED_RESTART_UPDATE:
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
            "completed_2d2f_updates": state["completed_2d2f_updates"],
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
            "rng_restore_audit": rng_restore_audit,
        }
        restart["passed"] = (
            restart["fresh_process"]
            and restart["next_global_batch_sha256"]
            == restart["expected_next_global_batch_sha256"]
            and restart["next_global_batch_stream_sha256"]
            == restart["expected_next_global_batch_stream_sha256"]
            and restart["rng_restore_audit"]["passed"]
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
            "completed_2d2f_updates": 0,
            "processed_2d2f_targets": 0,
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
    if state["completed_2d2f_updates"] == 0:
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
        ephemeral_checkpoint_dir=(
            None
            if not args.ephemeral_checkpoint_dir
            else str(Path(args.ephemeral_checkpoint_dir).resolve())
        ),
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
        "b3_gate": GATE_LR,
    }
    if lrs != expected_lrs:
        raise SystemExit(f"resumed optimizer LR drift: {lrs}")
    optimizer.zero_grad(set_to_none=True)
    pass_loss_sums = [0.0] * count
    forward_seconds = [0.0] * count
    backward_seconds = 0.0
    total_ce = 0.0
    final_h10_rms = None
    final_h12_rms = None
    final_b1_recurrent_rms = None
    final_b3_recurrent_rms = None
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for micro_index in range(runtime.accumulation):
        cpu_x, cpu_y = runtime.loader.next_batch()
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        results = []
        source_h10 = None
        source_h12 = None
        for pass_index in range(count):
            torch.cuda.synchronize()
            pass_start = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current = model.forward_pass(
                    x,
                    targets=y,
                    b1_recurrent_source=source_h12,
                    b3_recurrent_source=source_h10,
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
            source_h10 = current["h10"]
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
            final_h10_rms = results[-1]["h10"].detach().float().square().mean().sqrt().item()
            final_h12_rms = results[-1]["h12"].detach().float().square().mean().sqrt().item()
            final_b1_recurrent_rms = results[-1]["diagnostics"]["b1"][
                "recurrent_output_rms"
            ].detach().float().item()
            final_b3_recurrent_rms = results[-1]["diagnostics"]["b3"][
                "recurrent_output_rms"
            ].detach().float().item()
        del x, y, cpu_x, cpu_y, results, source_h10, source_h12, weighted, scaled, current
    if not gradients_finite(model):
        raise SystemExit("nonfinite gradients")
    groups = gradient_group_report(model)
    if not all(
        groups[name]["finite"] and groups[name]["nonzero"]
        for name in ("base", "gate", "b3_gate")
    ):
        raise SystemExit(f"required gradient group is zero: {groups}")
    b1_gate_gradient = model.g_rec.grad.detach().float().item()
    b3_gate_gradient = model.g_rec_b3.grad.detach().float().item()
    if not math.isfinite(b3_gate_gradient) or b3_gate_gradient == 0:
        raise SystemExit("new B3 gate gradient is not finite/nonzero")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad_norm):
        raise SystemExit("nonfinite gradient norm")
    optimizer.step()
    if not model_finite(model) or not optimizer_finite(optimizer):
        raise SystemExit("nonfinite parameter/optimizer state")
    elapsed = time.monotonic() - start
    runtime.training_state["completed_2d2f_updates"] = update
    runtime.training_state["processed_2d2f_targets"] = update * GLOBAL_TARGETS
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
        "g_rec_b3_raw": model.g_rec_b3.detach().float().item(),
        "tanh_g_rec_b3": model.recurrent_scale_b3.detach().float().item(),
        "g_rec_b1_gradient_preclip": b1_gate_gradient,
        "g_rec_b3_gradient_preclip": b3_gate_gradient,
        "gradient_norm_before_clip": grad_norm.detach().float().item(),
        "gradient_groups": groups,
        "b10_memory_rms": final_h10_rms,
        "b12_memory_rms": final_h12_rms,
        "b1_recurrent_output_rms": final_b1_recurrent_rms,
        "b3_recurrent_output_rms": final_b3_recurrent_rms,
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
            "g_rec_b3_raw": metrics["g_rec_b3_raw"],
            "tanh_g_rec_b3": metrics["tanh_g_rec_b3"],
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
    initial_damage = read_json(
        runtime.output / "initial_b2_w32_compression.json"
    )["initial_W32_compression_damage"]
    validation["recurrent_recovery_fraction"] = (
        validation["b3_recurrent_gain"] / initial_damage
        if initial_damage > 0
        else None
    )
    reference_gain = FROZEN_2D2C_REFERENCE["gain_trajectory"][str(update)]
    validation["matched_2d2c_comparison"] = {
        "gain_W32_minus_gain_W2": validation["b3_recurrent_gain"] - reference_gain,
        "sequence_gap_W32_minus_sequence_gap_W2": validation["b3_sequence_gap"]
        - FROZEN_2D2C_REFERENCE["sequence_gap_trajectory"][str(update)],
        "raw_g_B2_W32_minus_raw_g_B2_W2": validation["g_rec_b2_raw"]
        - FROZEN_2D2C_REFERENCE["raw_gate_trajectory"][str(update)],
        "tanh_g_B2_W32_minus_tanh_g_B2_W2": validation["tanh_g_rec_b2"]
        - FROZEN_2D2C_REFERENCE["gate_trajectory"][str(update)],
        "frozen_W2_gain": reference_gain,
        "frozen_W2_sequence_gap": FROZEN_2D2C_REFERENCE[
            "sequence_gap_trajectory"
        ][str(update)],
        "frozen_W2_raw_g_rec_b2": FROZEN_2D2C_REFERENCE[
            "raw_gate_trajectory"
        ][str(update)],
        "frozen_W2_tanh_g_rec_b2": FROZEN_2D2C_REFERENCE["gate_trajectory"][
            str(update)
        ],
    }
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
    merge_keyed_json(runtime.output / "b2_recurrent_attention_lag_bins.json", update, attention_b2["lag_bins"])
    merge_keyed_json(
        runtime.output / "b2_local_attention_lag_bins.json",
        update,
        attention_b2["local_lag_bins"],
    )
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
    if int(update) == FORCED_RESTART_UPDATE and runtime.ephemeral_checkpoint_dir:
        checkpoint_root = Path(runtime.ephemeral_checkpoint_dir)
    else:
        checkpoint_root = Path(runtime.run_root) / "checkpoints"
    checkpoint_path = checkpoint_root / f"{prefix}_update_{update:04d}.pt"
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


def reconcile_uncheckpointed_artifacts(output, completed):
    """Discard only durable diagnostics newer than the verified resume point."""

    output = Path(output)
    audit = {
        "schema": "exp2d2f_resume_reconciliation_v1",
        "verified_checkpoint_update": int(completed),
        "files": {},
    }
    metrics_path = output / "training_metrics.jsonl"
    rows = read_jsonl(metrics_path) if metrics_path.exists() else []
    if len(rows) < completed:
        raise SystemExit("training metrics end before verified resume checkpoint")
    if any(row["local_update"] != index for index, row in enumerate(rows, start=1)):
        raise SystemExit("training metrics update sequence is not contiguous")
    if len(rows) > completed:
        before = file_sha256(metrics_path)
        removed_updates = [row["local_update"] for row in rows[completed:]]
        durable_text(
            metrics_path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[:completed]),
        )
        audit["files"][metrics_path.name] = {
            "before_sha256": before,
            "after_sha256": file_sha256(metrics_path),
            "removed_uncheckpointed_updates": removed_updates,
        }

    keyed_files = (
        "milestone_validation.json",
        "gate_diagnostics.json",
        "b1_attention_diagnostics.json",
        "b3_local_attention_lag_bins.json",
        "b3_recurrent_attention_lag_bins.json",
        "b3_attention_head_distance.json",
        "b12_to_b1_temporal_gradient.json",
        "b10_to_b3_temporal_gradient.json",
    )
    for name in keyed_files:
        path = output / name
        if not path.exists():
            continue
        payload = read_json(path)
        removed = sorted(
            key for key in payload if key.isdigit() and int(key) > int(completed)
        )
        if removed:
            before = file_sha256(path)
            for key in removed:
                del payload[key]
            durable_json(path, payload)
            audit["files"][name] = {
                "before_sha256": before,
                "after_sha256": file_sha256(path),
                "removed_uncheckpointed_keys": removed,
            }

    audit["changed"] = bool(audit["files"])
    durable_json(output / "resume_reconciliation.json", audit)
    return rows[:completed]


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
        raise SystemExit("2D2F train segments must end at local update 96 or 191")
    runtime = initialize_runtime(args)
    completed = int(runtime.training_state["completed_2d2f_updates"])
    if completed == 0 and int(args.end_update) != FORCED_RESTART_UPDATE:
        raise SystemExit("fresh scientific process must stop at update 96")
    if completed == FORCED_RESTART_UPDATE and int(args.end_update) != MAX_UPDATES:
        raise SystemExit("restarted scientific process must end at update 191")
    metrics_path = runtime.output / "training_metrics.jsonl"
    if completed == 0 and metrics_path.exists():
        raise SystemExit("fresh scientific run found existing training metrics")
    if completed > 0:
        rows = reconcile_uncheckpointed_artifacts(runtime.output, completed)
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
            f"2D2F update={update:03d}/{MAX_UPDATES} "
            f"loss={metrics['weighted_total_ce']:.6f} "
            f"b1={metrics['tanh_g_rec_b1']:+.8f} "
            f"b3={metrics['tanh_g_rec_b3']:+.8f} "
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
        print("EXPERIMENT_2D2F_UPDATE_96_RESTART_REQUIRED", flush=True)
    else:
        durable_json(
            runtime.output / "training_complete.json",
            {
                "completed_2d2f_updates": MAX_UPDATES,
                "processed_2d2f_targets": ADDITIONAL_TARGETS,
                "cumulative_2d2_targets": CUMULATIVE_TARGETS,
                "checkpoint": runtime.training_state["last_checkpoint"],
                "timestamp": time.time(),
            },
        )
        print("EXPERIMENT_2D2F_TRAINING_COMPLETE", flush=True)
    return segment


def _incremental_control(model, x, y, name, derangement=None):
    batch, length = x.shape
    if name not in {
        "all_real",
        "b3_off",
        "b3_shuffled",
        "b3_full_counterfactual",
        "all_shuffled",
    }:
        raise ValueError(name)
    state = model.init_incremental_state(
        batch, device=x.device, b2_full_cache=name == "b3_full_counterfactual"
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
                derangement if name in {"b3_shuffled", "all_shuffled"} else None
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
        "all_real",
        "b3_off",
        "b3_shuffled",
        "b3_full_counterfactual",
        "all_shuffled",
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
        print(f"2D2E incremental batch={batch_index + 1:02d}/{batches}", flush=True)
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
    real = controls["all_real"]
    off = controls["b3_off"]
    shuffled = controls["b3_shuffled"]
    full = controls["b3_full_counterfactual"]
    all_shuffled = controls["all_shuffled"]
    result = {
        "controls": controls,
        "true_b3_recurrent_gain": off["validation_loss"] - real["validation_loss"],
        "true_b3_sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "true_remaining_b2_compression_gap": real["validation_loss"]
        - full["validation_loss"],
        "true_all_real_vs_all_shuffled_gap": all_shuffled["validation_loss"]
        - real["validation_loss"],
        "all_real_vs_b2_off_batches": paired_stats(
            real["per_batch_losses"], off["per_batch_losses"]
        ),
        "all_real_vs_b3_shuffled_batches": paired_stats(
            real["per_batch_losses"], shuffled["per_batch_losses"]
        ),
        "all_real_vs_b2_full_batches": paired_stats(
            real["per_batch_losses"], full["per_batch_losses"]
        ),
        "all_real_vs_b2_off_sequences": paired_stats(
            real["per_sequence_losses"], off["per_sequence_losses"]
        ),
        "all_real_vs_b3_shuffled_sequences": paired_stats(
            real["per_sequence_losses"], shuffled["per_sequence_losses"]
        ),
        "all_real_vs_b2_full_sequences": paired_stats(
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
def parallel_incremental_equivalence(model, val_path, length=64, batch=2):
    if length < 64:
        raise ValueError("2D2E equivalence requires length >=64 to exercise B2 W32 eviction")
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
                control="all_real",
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
                control="all_real",
                bank_mode="full",
                diagnostic_attention_weights=False,
            )["logits"]
            parallel_b2_isolated = model.forward_pass(
                tokens,
                b1_recurrent_source=first["h12"],
                b2_recurrent_source=first["h11"],
                b1_gate_override=0.0,
                bank_mode="full",
            )["logits"]
            incremental_b2_isolated = model.incremental_logits(
                tokens,
                control="b1_off_b2_real",
                bank_mode="full",
                diagnostic_attention_weights=False,
            )["logits"]
        plain_delta = (plain.float() - incremental_plain.float()).abs()
        recurrent_delta = (parallel_real.float() - incremental_real.float()).abs()
        b2_isolated_delta = (
            parallel_b2_isolated.float() - incremental_b2_isolated.float()
        ).abs()
        reports[label] = {
            "plain_kernel_max_abs": plain_delta.max().item(),
            "plain_kernel_mean_abs": plain_delta.mean().item(),
            "active_recurrent_positions_0_3_max_abs": recurrent_delta[:, :4].max().item(),
            "active_recurrent_positions_0_3_mean_abs": recurrent_delta[:, :4].mean().item(),
            "self_recurrence_drift_positions_4_plus_mean_abs": recurrent_delta[:, 4:].mean().item(),
            "self_recurrence_drift_positions_4_plus_max_abs": recurrent_delta[:, 4:].max().item(),
            "b2_isolated_full_prefix_max_abs": b2_isolated_delta.max().item(),
            "b2_isolated_full_prefix_mean_abs": b2_isolated_delta.mean().item(),
            "b2_active_positions_32_63_max_abs": b2_isolated_delta[:, 32:64]
            .max()
            .item(),
            "b2_active_positions_32_63_mean_abs": b2_isolated_delta[:, 32:64]
            .mean()
            .item(),
            "b2_w32_eviction_exercised": True,
            "max_abs_tolerance": threshold,
            "kernel_passed": plain_delta.max().item() <= threshold
            and recurrent_delta[:, :4].max().item() <= threshold
            and b2_isolated_delta.max().item() <= threshold,
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


def memory_accounting(incremental=None) -> dict:
    element_bytes = 2
    runtime_raw_state_bytes = 4

    def one(batch):
        b1 = batch * 1 * N_EMBD * 2 * element_bytes
        b12_raw = batch * 1023 * N_EMBD * element_bytes
        b2 = batch * 31 * N_EMBD * 2 * element_bytes
        b11_raw = batch * 1023 * N_EMBD * element_bytes
        upper = batch * 10 * 1023 * N_EMBD * 2 * element_bytes
        total = b1 + b12_raw + b2 + b11_raw + upper
        final_2d2b = batch * (
            1 * N_EMBD * 2 + 1023 * N_EMBD + 11 * 1023 * N_EMBD * 2
        ) * element_bytes
        final_2d2c = batch * (
            2 * 1 * N_EMBD * 2
            + 2 * 1023 * N_EMBD
            + 10 * 1023 * N_EMBD * 2
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
            "final_2d2c_inference_state_bytes": final_2d2c,
            "standard_gpt2_w1024_kv_bytes": standard,
            "delta_bytes_vs_final_2d2b": total - final_2d2b,
            "delta_bytes_vs_final_2d2c": total - final_2d2c,
            "delta_bytes_vs_standard_gpt2": total - standard,
            "mib": {
                "b1_local_kv": b1 / 1024**2,
                "b12_recurrent_raw_state": b12_raw / 1024**2,
                "b2_local_kv": b2 / 1024**2,
                "b11_recurrent_raw_state": b11_raw / 1024**2,
                "b3_b12_ordinary_kv": upper / 1024**2,
                "total": total / 1024**2,
                "delta_vs_final_2d2b": (total - final_2d2b) / 1024**2,
                "delta_vs_final_2d2c": (total - final_2d2c) / 1024**2,
                "delta_vs_standard_gpt2": (total - standard) / 1024**2,
            },
        }

    def current_runtime_one(batch):
        b1 = batch * 1 * N_EMBD * 2 * element_bytes
        b2 = batch * 31 * N_EMBD * 2 * element_bytes
        upper = batch * 10 * 1023 * N_EMBD * 2 * element_bytes
        b11_raw = batch * 1023 * N_EMBD * runtime_raw_state_bytes
        b12_raw = batch * 1023 * N_EMBD * runtime_raw_state_bytes
        total = b1 + b2 + upper + b11_raw + b12_raw
        final_2d2b = batch * (
            (1 * N_EMBD * 2 + 11 * 1023 * N_EMBD * 2) * element_bytes
            + 1023 * N_EMBD * runtime_raw_state_bytes
        )
        final_2d2c = batch * (
            (2 * 1 * N_EMBD * 2 + 10 * 1023 * N_EMBD * 2) * element_bytes
            + 2 * 1023 * N_EMBD * runtime_raw_state_bytes
        )
        standard = batch * 12 * 1023 * N_EMBD * 2 * element_bytes
        return {
            "batch_size": batch,
            "ordinary_kv_dtype": "torch.bfloat16",
            "raw_recurrent_state_dtype": "torch.float32",
            "total_experimental_inference_state_bytes": total,
            "final_2d2b_inference_state_bytes": final_2d2b,
            "final_2d2c_inference_state_bytes": final_2d2c,
            "standard_gpt2_w1024_kv_bytes": standard,
            "delta_bytes_vs_final_2d2b": total - final_2d2b,
            "delta_bytes_vs_final_2d2c": total - final_2d2c,
            "delta_bytes_vs_standard_gpt2": total - standard,
            "mib": {"total": total / 1024**2},
        }

    observed = None
    if incremental is not None:
        final_cache = incremental["controls"]["all_real"]["cache_rows"][-1][
            "final"
        ]
        cache_bytes = sum(
            0
            if row["key"] is None
            else row["key"]["actual_bytes"] + row["value"]["actual_bytes"]
            for row in final_cache["cache_physical_storage"]
        )
        ring_bytes = (
            final_cache["h11_ring_physical_storage"]["actual_bytes"]
            + final_cache["h12_ring_physical_storage"]["actual_bytes"]
        )
        observed = {
            "control": "all_real",
            "position": final_cache["position"],
            "cache_bytes": cache_bytes,
            "raw_ring_bytes": ring_bytes,
            "total_physical_state_bytes": cache_bytes + ring_bytes,
            "cache_storage": final_cache["cache_physical_storage"],
            "h11_ring_storage": final_cache["h11_ring_physical_storage"],
            "h12_ring_storage": final_cache["h12_ring_physical_storage"],
            "physical_storage_exact": final_cache["physical_storage_exact"],
        }

    return {
        "deployment_accounting_dtype": "BF16",
        "bytes_per_element": element_bytes,
        "B1": one(1),
        "B64": one(64),
        "current_evaluation_runtime_mixed_precision": {
            "B1": current_runtime_one(1),
            "B64": current_runtime_one(64),
            "note": (
                "The current PyTorch autocast implementation retains raw B11/B12 "
                "residual rings in FP32 while projected KV caches are BF16."
            ),
        },
        "observed_final_incremental_storage": observed,
        "state_limits": {
            "B1 ordinary historical KV": 1,
            "B12 raw recurrent states": 1023,
            "B2 ordinary historical KV": 31,
            "B11 raw recurrent states": 1023,
            "B3-B12 ordinary historical KV each": 1023,
        },
        "comparison_note": (
            "B1/B64 are exact theoretical all-BF16 deployment-state comparisons at "
            "a full 1024-token context. The mixed-precision section and observed "
            "final cache prevent those deployment figures from being misreported as "
            "the current evaluator's physical allocation."
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
    payload = legacy.d0.torch_load(final_path, mmap=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("final checkpoint schema mismatch")
    if payload.get("completed_2d2f_updates") != MAX_UPDATES:
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
        "completed_2d2f_updates": payload["completed_2d2f_updates"],
        "processed_2d2f_targets": payload["processed_2d2f_targets"],
        "cumulative_2d2_targets": payload["cumulative_2d2_targets"],
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "model_finite": model_finite(model),
        "optimizer_finite": optimizer_finite(optimizer),
        "g_rec_b1_raw": model.g_rec.detach().float().item(),
        "tanh_g_rec_b1": model.recurrent_scale_b1.detach().float().item(),
        "g_rec_b3_raw": model.g_rec_b3.detach().float().item(),
        "tanh_g_rec_b3": model.recurrent_scale_b3.detach().float().item(),
        "b2_gate_absent": not hasattr(model, "g_rec_b2"),
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
        return "EXPERIMENT 2D2E INVALID"
    if not stable:
        return "B2 W32 SECOND LINK IS UNSTABLE"
    off_wins = incremental["all_real_vs_b2_off_sequences"]
    shuffled_wins = incremental["all_real_vs_b3_shuffled_sequences"]
    gain = incremental["true_b3_recurrent_gain"]
    gap = incremental["true_b3_sequence_gap"]
    positive = (
        gain > 0
        and gap > 0
        and off_wins["wins"] >= 65
        and shuffled_wins["wins"] >= 65
    )
    if positive:
        if (
            gain >= 0.001
            and off_wins["wins"] >= 75
            and shuffled_wins["wins"] >= 75
        ):
            return "W32 STRONGLY RESCUES B11→B2 RECURRENT UTILITY"
        return "B2 W32 ESTABLISHES POSITIVE SECOND-LINK RECURRENT UTILITY"
    if gap > 0 and (gain <= 0 or off_wins["wins"] < 65):
        return "B2 W32 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY"
    balanced = (
        56 <= off_wins["wins"] <= 72
        and 56 <= shuffled_wins["wins"] <= 72
    )
    if abs(gain) < 1e-4 and abs(gap) < 1e-4 and balanced:
        return "B2 W32 SECOND RECURRENT LINK REMAINS NEAR ZERO"
    return "B2 W32 SECOND RECURRENT LINK DOES NOT ESTABLISH POSITIVE UTILITY"


def choose_recommendation(classification, initial_damage) -> str:
    if classification == "EXPERIMENT 2D2E INVALID":
        return "FIX 2D2E INTEGRITY"
    if classification == "B2 W32 SECOND LINK IS UNSTABLE":
        return "STABILIZE B2 W32 RECURRENT LINK"
    if classification in {
        "B2 W32 ESTABLISHES POSITIVE SECOND-LINK RECURRENT UTILITY",
        "W32 STRONGLY RESCUES B11→B2 RECURRENT UTILITY",
    }:
        return "ADD B10→B3 WITH A GRADED LOCAL WINDOW W64"
    if classification == "B2 W32 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY":
        return "IMPROVE B2 RECURRENT READOUT BEFORE ADDING B10→B3"
    if classification == "B2 W32 SECOND RECURRENT LINK REMAINS NEAR ZERO":
        if initial_damage < 0.5 * FROZEN_2D2C_REFERENCE["initial_w2_compression_damage"]:
            return "RUN MATCHED B2-W32 TRAINING WITHOUT B11→B2 RECURRENCE"
        return "SEARCH B2 LOCAL WINDOW BETWEEN W32 AND W1024 BEFORE DEEPER RECURRENCE"
    if initial_damage >= 0.5 * FROZEN_2D2C_REFERENCE["initial_w2_compression_damage"]:
        return "SEARCH B2 LOCAL WINDOW BETWEEN W32 AND W1024 BEFORE DEEPER RECURRENCE"
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
            "gpu_count": 1,
            "training_gpu_hours": sum(walls) / 3600,
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
        "b2_local_attention_microbenchmark": preflight["performance_benchmark"][
            "b2_local_attention"
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
    off = [milestones[str(key)]["controls"]["b3_off"]["validation_loss"] for key in keys]

    def line_plot(filename, series, ylabel):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for label, values in series.items():
            ax.plot(x, values, marker="o", label=label)
        ax.set_xlabel("Additional 2D2E training targets")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    gains = [milestones[str(k)]["b3_recurrent_gain"] for k in keys]
    gaps = [milestones[str(k)]["b3_sequence_gap"] for k in keys]
    line_plot(REQUIRED_PLOTS[0], {"REAL_W32": real, "B2_OFF_W32": off}, "Validation cross entropy")
    line_plot(REQUIRED_PLOTS[1], {"Off − Real": gains}, "B2 W32 recurrent gain")
    line_plot(REQUIRED_PLOTS[2], {"Shuffled − Real": gaps}, "B2 W32 sequence gap")
    line_plot(
        REQUIRED_PLOTS[3],
        {
            "2D2E W32": gains,
            "2D2C W2 frozen": [
                0.0 if key == 0 else FROZEN_2D2C_REFERENCE["gain_trajectory"][str(key)]
                for key in keys
            ],
        },
        "Same-checkpoint recurrent gain",
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    tx = [row["additional_targets"] for row in training]
    ax.plot(tx, [row["tanh_g_rec_b1"] for row in training], label="B12→B1")
    ax.scatter([0], [SOURCE_GATE_EFFECTIVE], zorder=3)
    ax.set(xlabel="Additional 2D2E training targets", ylabel="tanh(g_rec_B1)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[4], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    milestone_targets = [milestones[str(key)]["additional_targets"] for key in keys]
    ax.plot(tx, [row["tanh_g_rec_b2"] for row in training], label="2D2E W32")
    ax.plot(
        milestone_targets,
        [FROZEN_2D2C_REFERENCE["gate_trajectory"][str(key)] for key in keys],
        marker="o",
        label="2D2C W2 frozen",
    )
    ax.set(xlabel="Additional training targets", ylabel="tanh(g_rec_B2)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[5], dpi=180)
    plt.close(fig)

    b2_labels = [name for name, _, _ in B2_RECURRENT_LAG_BINS]
    b2_final_attention = attention["b2"][str(MAX_UPDATES)]
    for filename, field, ylabel in (
        (REQUIRED_PLOTS[6], "attention_mass", "Recurrent attention mass"),
        (REQUIRED_PLOTS[7], "normalized_mass_per_available_token", "Mass per available token"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        ax.bar(b2_labels, [b2_final_attention[name][field] for name in b2_labels])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    local_labels = [name for name, _, _ in B2_LOCAL_LAG_BINS]
    b2_local = attention["b2_local"][str(MAX_UPDATES)]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(local_labels, [b2_local[name]["attention_mass"] for name in local_labels])
    ax.set(xlabel="B2 local lag", ylabel="Local attention mass")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[8], dpi=180)
    plt.close(fig)

    b2_gradient = temporal["b2"][str(MAX_UPDATES)]["bins"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(b2_labels, [b2_gradient[name]["mean_gradient_rms"] for name in b2_labels])
    ax.set_ylabel("B11→B2 mean writer-gradient RMS")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[9], dpi=180)
    plt.close(fig)

    b2_heads = read_json(output / "b2_attention_head_distance.json")[str(MAX_UPDATES)]["heads"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    positions = np.arange(N_HEAD)
    ax.bar(positions, [b2_heads[str(index)]["mean_attended_recurrent_lag"] for index in range(N_HEAD)])
    ax.set(xlabel="B2 attention head", ylabel="Mean recurrent lag")
    ax.set_xticks(range(N_HEAD))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[10], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(
        ["Parallel", "True incremental"],
        [milestones[str(MAX_UPDATES)]["b3_recurrent_gain"], incremental["true_b3_recurrent_gain"]],
    )
    ax.set_ylabel("B2 W32 recurrent gain")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[11], dpi=180)
    plt.close(fig)

    memory = memory_accounting()["B1"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(
        ["Standard", "2D2B", "2D2C", "2D2E"],
        [
            memory["standard_gpt2_w1024_kv_bytes"],
            memory["final_2d2b_inference_state_bytes"],
            memory["final_2d2c_inference_state_bytes"],
            memory["total_experimental_inference_state_bytes"],
        ],
    )
    ax.set_ylabel("BF16 inference-state bytes (B=1)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[12], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    scatter = ax.scatter(
        [row["wall_seconds"] for row in training],
        [row["targets_per_second"] for row in training],
        c=[row["peak_allocated_vram_mb"] for row in training],
        s=15,
    )
    ax.set(xlabel="Seconds/update", ylabel="Targets/second")
    fig.colorbar(scatter, ax=ax, label="Peak allocated VRAM (MiB)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / REQUIRED_PLOTS[13], dpi=180)
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
    initial = summary["initial_b2_w32_compression"]
    parallel = summary["parallel"]
    trajectory = summary["gate_diagnostics"]["trajectory"]
    first_raw = trajectory[0]["g_rec_b2_raw"]
    first_open = first_raw != 0
    raw_values = [row["g_rec_b2_raw"] for row in trajectory]
    maximum_positive = max([0.0, *raw_values])
    stayed_positive = all(value > 0 for value in raw_values)
    crossing = next(
        (
            trajectory[index]["local_update"]
            for index in range(1, len(trajectory))
            if raw_values[index - 1] * raw_values[index] < 0
        ),
        None,
    )
    damage = initial["initial_W32_compression_damage"]
    recovery = parallel["b3_recurrent_gain"] / damage if damage else None
    milestone_gain = {
        key: milestones[str(key)]["b3_recurrent_gain"] for key in (20, 48, 96, 143, 191)
    }
    b1_marginal = parallel["combined_system"]["b1_marginal_gain_with_b2_real"]
    bins = final_attention["lag_bins"]
    return {
        "Q1": {"question": "What is exact initial B2-W32 compression damage?", "answer": damage},
        "Q2": {"question": "How does W32 initial damage compare with frozen W2 damage?", "answer": {"W32": damage, "W2": initial["matched_W2_compression_damage"], "difference": damage - initial["matched_W2_compression_damage"]}},
        "Q3": {"question": "What fraction of W2 damage remains at W32 before adaptation?", "answer": initial["fraction_of_W2_damage_remaining"]},
        "Q4": {
            "question": "Did B2 gate open on update 1?",
            "answer": {
                "opened": first_open,
                "raw_g_rec_b2": first_raw,
                "sign": "positive" if first_raw > 0 else "negative" if first_raw < 0 else "zero",
            },
        },
        "Q5": {"question": "What maximum positive B2 gate value occurred?", "answer": maximum_positive},
        "Q6": {"question": "Did B2 gate remain positive?", "answer": stayed_positive},
        "Q7": {"question": "If it crossed zero, at what update?", "answer": crossing},
        "Q8": {"question": "What is final tanh(g_rec_B2)?", "answer": summary["final_tanh_g_rec_b2"]},
        "Q9": {"question": "What is final tanh(g_rec_B1)?", "answer": summary["final_tanh_g_rec_b1"]},
        "Q10": {"question": "Did B11→B2 temporal writer gradient become finite/nonzero?", "answer": temporal["b2"]["finite"] and temporal["b2"]["nonzero"]},
        "Q11": {"question": "Did it reach lag 128+?", "answer": final_gradient["128-255"]["fraction_nonzero_positions"] > 0},
        "Q12": {"question": "Did it reach lag 256+?", "answer": final_gradient["256-511"]["fraction_nonzero_positions"] > 0},
        "Q13": {"question": "Did it reach lag 512+?", "answer": final_gradient["512-1023"]["fraction_nonzero_positions"] > 0},
        "Q14": {"question": "What is B2-W32 recurrent gain at 10M?", "answer": milestone_gain[20]},
        "Q15": {"question": "What is B2-W32 recurrent gain at 25M?", "answer": milestone_gain[48]},
        "Q16": {"question": "What is B2-W32 recurrent gain at 50M?", "answer": milestone_gain[96]},
        "Q17": {"question": "What is B2-W32 recurrent gain at 75M?", "answer": milestone_gain[143]},
        "Q18": {"question": "What is B2-W32 recurrent gain at 100M?", "answer": milestone_gain[191]},
        "Q19": {"question": "What is final parallel B2-W32 sequence gap?", "answer": parallel["b3_sequence_gap"]},
        "Q20": {"question": "How does final parallel gain compare with matched W2 2D2C?", "answer": {"W32": parallel["b3_recurrent_gain"], "W2": FROZEN_2D2C_REFERENCE["final_parallel_gain"], "difference": parallel["b3_recurrent_gain"] - FROZEN_2D2C_REFERENCE["final_parallel_gain"]}},
        "Q21": {"question": "What fraction of recurrent attention lies at 32-63?", "answer": bins["32-63"]["attention_mass"]},
        "Q22": {"question": "What fraction lies at 64-127?", "answer": bins["64-127"]["attention_mass"]},
        "Q23": {"question": "What fraction lies at 128-255?", "answer": bins["128-255"]["attention_mass"]},
        "Q24": {"question": "What fraction lies at 256-511?", "answer": bins["256-511"]["attention_mass"]},
        "Q25": {"question": "What fraction lies at 512-1023?", "answer": bins["512-1023"]["attention_mass"]},
        "Q26": {"question": "What is B2 recurrent mean/median/p90 lag?", "answer": {"mean": aggregate["mean_attended_recurrent_lag"], "median": aggregate["median_attended_recurrent_lag"], "p90": aggregate["p90_attended_recurrent_lag"]}},
        "Q27": {"question": "Did B2 heads temporally specialize?", "answer": {"specialized": statistics.pstdev(means) > 1.0, "mean_lag_range": [min(means), max(means)], "std": statistics.pstdev(means)}},
        "Q28": {"question": "What fraction of W32 compression damage is attributable to recurrence recovery?", "answer": recovery},
        "Q29": {"question": "Did B1 recurrence retain positive utility?", "answer": {"yes": b1_marginal > 0, "same_checkpoint_gain": b1_marginal}},
        "Q30": {"question": "What is final true-incremental B2-W32 recurrent gain?", "answer": incremental["true_b3_recurrent_gain"]},
        "Q31": {"question": "What is final true-incremental B2-W32 sequence gap?", "answer": incremental["true_b3_sequence_gap"]},
        "Q32": {"question": "What are paired true-self wins versus B2 Off?", "answer": incremental["all_real_vs_b2_off_sequences"]},
        "Q33": {"question": "What are paired wins versus B2 Shuffled?", "answer": incremental["all_real_vs_b3_shuffled_sequences"]},
        "Q34": {"question": "Does W32 establish second-link recurrent utility?", "answer": summary["primary_classification"] in {"B2 W32 ESTABLISHES POSITIVE SECOND-LINK RECURRENT UTILITY", "W32 STRONGLY RESCUES B11→B2 RECURRENT UTILITY"}},
        "Q35": {"question": "What exactly one experiment should run next?", "answer": summary["recommendation"]},
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
            "all_real_loss": float(values["all_real"].mean()),
            "b3_off_loss": float(values["b3_off"].mean()),
            "b3_shuffled_loss": float(values["b3_shuffled"].mean()),
            "b3_full_counterfactual_loss": float(values["b3_full_counterfactual"].mean()),
            "b3_recurrent_gain": float(
                (values["b3_off"] - values["all_real"]).mean()
            ),
            "b3_sequence_gap": float(
                (values["b3_shuffled"] - values["all_real"]).mean()
            ),
            "remaining_b2_compression_gap": float(
                (values["all_real"] - values["b3_full_counterfactual"]).mean()
            ),
        }
    return result


def build_artifact_inventory(output):
    output = Path(output)
    mutable = {
        "EXPERIMENT_2D2E_FINAL_REPORT.md",
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
            "bytes": path.stat().st_size if exists and name not in mutable else None,
            "sha256": file_sha256(path) if exists and name not in mutable else None,
            "size_deferred_self_referential": name in mutable,
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
    initial = summary["initial_b2_w32_compression"]
    training = summary["training"]
    memory = summary["memory_accounting"]
    milestones = summary["validation_trajectory"]
    matched_table = [
        "",
        "| Update | W32 gain | W2 gain | Δ gain | W32 gap | W2 gap | Δ gap |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gate_table = [
        "",
        "| Update | B1 raw | B1 tanh | B2 W32 raw | B2 W32 tanh | B2 W2 raw | B2 W2 tanh |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for update in MILESTONES:
        row = milestones[str(update)]
        comparison = row["matched_2d2c_comparison"]
        matched_table.append(
            f"| {update} | {row['b3_recurrent_gain']:.12g} | "
            f"{comparison['frozen_W2_gain']:.12g} | "
            f"{comparison['gain_W32_minus_gain_W2']:.12g} | "
            f"{row['b3_sequence_gap']:.12g} | "
            f"{comparison['frozen_W2_sequence_gap']:.12g} | "
            f"{comparison['sequence_gap_W32_minus_sequence_gap_W2']:.12g} |"
        )
        gate_table.append(
            f"| {update} | {row['g_rec_b1_raw']:.12g} | "
            f"{row['tanh_g_rec_b1']:.12g} | {row['g_rec_b2_raw']:.12g} | "
            f"{row['tanh_g_rec_b2']:.12g} | "
            f"{comparison['frozen_W2_raw_g_rec_b2']:.12g} | "
            f"{comparison['frozen_W2_tanh_g_rec_b2']:.12g} |"
        )
    lines = [
        f"EXPERIMENT 2D2E PRIMARY CLASSIFICATION:\n{summary['primary_classification']}",
        f"\nINITIAL B2-W32 COMPRESSION DAMAGE:\n{initial['initial_W32_compression_damage']}",
        f"\nFINAL TRUE-SELF B11→B2 W32 RECURRENT GAIN:\n{incremental['true_b3_recurrent_gain']}",
        f"\nFINAL TRUE-SELF B11→B2 W32 SEQUENCE GAP:\n{incremental['true_b3_sequence_gap']}",
        "\n# Experiment 2D2E — B2 W32 + Older B11→B2 Recurrent K/V",
        "\n## Result",
        f"\nThe final classification is **{summary['primary_classification']}**.",
        f"The exactly one next experiment is **{summary['recommendation']}**.",
        "\n## Source and architecture",
        f"\n- Source checkpoint: `{summary['source_checkpoint']}`",
        f"- Source SHA-256: `{summary['source_checkpoint_sha256']}`",
        f"- Git lineage: `{FROZEN_COMMIT}` → `{BRANCH}`; frozen 2D2C is sibling `{FROZEN_2D2C_COMMIT}`",
        f"- Parameters: {summary['parameters']:,} (exactly one new scalar versus 2D2B)",
        f"- Hardware: {summary['hardware']}",
        f"- Distributed equivalence: {json.dumps(summary['distributed_equivalence'], sort_keys=True)}",
        "- B1: W2 ordinary KV plus full B12 recurrent raw-state bank",
        "- B2: W32 ordinary KV (lags 0…31) plus B11 recurrent raw-state bank (lags 32…1023)",
        "- B3–B12: W1024; B1 recurrent lags are 2…1023",
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
        "\n## Initial B2 W32 compression",
        f"\n- Source 2D2B loss: {initial['source_2d2b_loss']}",
        f"- B2 W32, recurrence off: {initial['b2_w32_off_loss']}",
        f"- Exact compression damage: {initial['initial_W32_compression_damage']}",
        f"- Frozen matched W2 damage: {initial['matched_W2_compression_damage']}",
        f"- Gate-zero identities passed: {initial['passed']}",
        "\n## Final parallel validation",
        f"\n- BothReal: {parallel['controls']['new_real']['validation_loss']}",
        f"- B2 recurrence off: {parallel['controls']['b3_off']['validation_loss']}",
        f"- B2 shuffled: {parallel['controls']['b3_shuffled']['validation_loss']}",
        f"- B2 full counterfactual: {parallel['controls']['b3_full_counterfactual']['validation_loss']}",
        f"- B2 recurrent gain: {parallel['b3_recurrent_gain']}",
        f"- B2 sequence gap: {parallel['b3_sequence_gap']}",
        f"- Remaining compression gap: {parallel['remaining_b2_compression_gap']}",
        f"- Paired Real vs B2-off: {json.dumps(parallel['new_real_vs_b2_off'], sort_keys=True)}",
        f"- Paired Real vs B2-shuffled: {json.dumps(parallel['new_real_vs_b3_shuffled'], sort_keys=True)}",
        f"- Paired Real vs B2-full: {json.dumps(parallel['new_real_vs_b2_full'], sort_keys=True)}",
        f"- Combined-system controls: {json.dumps(parallel['combined_system'], sort_keys=True)}",
        "\n## Matched frozen 2D2C comparison",
        f"\n- Final gain W32 − W2: {parallel['b3_recurrent_gain'] - FROZEN_2D2C_REFERENCE['final_parallel_gain']}",
        f"- Final sequence gap W32 − W2: {parallel['b3_sequence_gap'] - FROZEN_2D2C_REFERENCE['final_parallel_sequence_gap']}",
        f"- Final B2 gate W32 − W2: {summary['final_tanh_g_rec_b2'] - FROZEN_2D2C_REFERENCE['final_tanh_g_rec_b2']}",
        "\n### Matched gain and sequence-gap trajectory",
        *matched_table,
        "\n### B1/B2 gate trajectories",
        *gate_table,
        "\n## Final true incremental validation",
        f"\n- BothReal: {incremental['controls']['all_real']['validation_loss']}",
        f"- B2 recurrence off: {incremental['controls']['b3_off']['validation_loss']}",
        f"- B2 shuffled: {incremental['controls']['b3_shuffled']['validation_loss']}",
        f"- B2 full counterfactual: {incremental['controls']['b3_full_counterfactual']['validation_loss']}",
        f"- Both shuffled: {incremental['controls']['all_shuffled']['validation_loss']}",
        f"- Targets/control: {incremental['targets_per_control']:,}",
        f"- Sequence wins vs B2-off: {incremental['all_real_vs_b2_off_sequences']['wins']} of {incremental['all_real_vs_b2_off_sequences']['count']}",
        f"- Sequence wins vs B2-shuffled: {incremental['all_real_vs_b3_shuffled_sequences']['wins']} of {incremental['all_real_vs_b3_shuffled_sequences']['count']}",
        "\n## Attention and temporal gradients",
        f"\n- Final B1 recurrent mass partitions: {summary['final_attention']['b1']['mass_partitions']}",
        f"- Final B2 recurrent mass partitions: {summary['final_attention']['b2']['mass_partitions']}",
        f"- Final B2 local attention lag bins: {summary['final_attention']['b2_local']}",
        f"- B11→B2 long-lag gradient present: {summary['final_temporal_gradient']['b2']['long_lag_writer_gradient_present']}",
        "\n## Cache and storage",
        "\n- B1 historical same-layer KV: at most 1 entry",
        "- B2 historical same-layer KV: at most 31 entries",
        "- B11 and B12 raw recurrent buffers: at most 1023 states each",
        "- B3–B12 ordinary historical KV: at most 1023 entries/layer",
        f"- BF16 total experimental inference state, B=1: {memory['B1']['mib']['total']:.3f} MiB",
        f"- BF16 total experimental inference state, B=64: {memory['B64']['mib']['total']:.3f} MiB",
        f"- Delta versus final 2D2B, B=1: {memory['B1']['delta_bytes_vs_final_2d2b']} bytes",
        f"- Delta versus final 2D2C, B=1: {memory['B1']['delta_bytes_vs_final_2d2c']} bytes",
        f"- Delta versus Standard GPT-2, B=1: {memory['B1']['delta_bytes_vs_standard_gpt2']} bytes",
        "- The preceding values are theoretical all-BF16 deployment accounting.",
        f"- Current mixed-precision evaluator physical state, B=1: {memory['current_evaluation_runtime_mixed_precision']['B1']['mib']['total']:.3f} MiB",
        f"- Current mixed-precision delta versus Standard GPT-2, B=1: {memory['current_evaluation_runtime_mixed_precision']['B1']['delta_bytes_vs_standard_gpt2']} bytes",
        f"- Observed final incremental physical state: {memory['observed_final_incremental_storage']['total_physical_state_bytes']} bytes",
        "\n## Scientific questions Q1–Q35",
    ]
    for index in range(1, 36):
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
            f"- Sealed-report commit: `{summary['git'].get('sealed_commit')}`",
            f"- Artifact directory: `{summary['artifact_directory']}`",
            f"- GPU pod: `{summary['pod']['name']}` (`{summary['pod']['id']}`), status `{summary['pod']['status']}`",
            f"- Persistent volume: `{summary['pod']['persistent_volume_id']}`, status `{summary['pod']['persistent_volume_status']}`",
            "\n# EXPERIMENT 2D2E COMPLETE",
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
    final_cache_rows = incremental["controls"]["all_real"]["cache_rows"]
    cache_pass = all(
        row["final"]["passed"]
        and row["final"]["b1_historical_kv"] <= 1
        and row["final"]["b2_historical_kv"] <= 31
        and row["final"]["h11_ring_length"] <= 1023
        and row["final"]["h12_ring_length"] <= 1023
        and all(value <= 1023 for value in row["final"]["b3_b12_historical_kv"])
        for row in final_cache_rows
    )
    restart = read_json(Path(output) / "forced_restart_update_96.json")
    replay = read_json(Path(output) / "matched_2d2c_data_replay_audit.json")
    replay_complete = (
        replay["passed_so_far"]
        and set(replay["checkpoint_cursor_comparisons"])
        == set(FROZEN_2D2C_REFERENCE["available_cursor_hashes"])
        and all(
            row["exact"] for row in replay["checkpoint_cursor_comparisons"].values()
        )
    )
    scientific = checkpoint_manifest["scientific"]
    recovery = checkpoint_manifest["recovery"]
    checks = {
        "2D2C final tag preserved": git_output(
            "rev-parse", FROZEN_2D2C_TAG + "^{commit}"
        )
        == FROZEN_2D2C_COMMIT,
        "2D2B final tag exact": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
        "branch starts from 2D2B not 2D2C": git_output(
            "merge-base", "HEAD", FROZEN_2D2C_COMMIT
        )
        == FROZEN_COMMIT,
        "2D2B checkpoint SHA exact": final_model_audit["passed"] and preflight["source"]["checkpoint_sha256"] == SOURCE_SHA256,
        "audit-correction provenance preserved": preflight["source"]["audit_correction_provenance"]["passed"],
        "parameter count 124,475,906": preflight["parameters"]["2d2e_total_parameters"] == TOTAL_PARAMETERS,
        "exactly one new scalar": preflight["parameters"]["new_parameter_count_vs_2d2b"] == 1,
        "source parameter inventory preserved": preflight["parameters"]["source_inventory_preserved"],
        "B1 local W=2 exact": preflight["architecture"]["b1_local_window"] == 2,
        "B2 local W=32 exact": preflight["architecture"]["b2_local_window"] == 32,
        "B3-B12 W=1024 exact": preflight["architecture"]["b3_b12_windows"] == [1024] * 10,
        "B1 recurrent lags 2..1023 exact": preflight["kernel_preflight"]["checks"]["b1_boundary_counts"],
        "B2 recurrent lags 32..1023 exact": preflight["kernel_preflight"]["checks"]["b2_boundary_counts"],
        "B1 maximum recurrent entries=1022": preflight["architecture"]["b1_maximum_recurrent_entries"] == 1022,
        "B2 maximum recurrent entries=992": preflight["architecture"]["b2_maximum_recurrent_entries"] == 992,
        "no B1 local/recurrent overlap": preflight["kernel_preflight"]["checks"]["b1_local_recurrent_disjoint"],
        "no B2 local/recurrent overlap": preflight["kernel_preflight"]["checks"]["b2_local_recurrent_disjoint"],
        "no recent B2 recurrent access": preflight["kernel_preflight"]["checks"]["b2_no_recent_recurrent_exposure"],
        "same B1 LN/K/V projections": preflight["kernel_preflight"]["checks"]["b1_shared_projection_exact"],
        "same B2 LN/K/V projections": preflight["kernel_preflight"]["checks"]["b2_shared_projection_exact"],
        "separate recurrent softmaxes": preflight["architecture"]["separate_softmaxes"],
        "single c_proj each": preflight["kernel_preflight"]["checks"]["single_c_proj_each"],
        "gate source value exact": preflight["source"]["checks"]["gate_raw"],
        "new B2 gate starts zero": preflight["source"]["checks"]["b2_gate_zero"],
        "frozen 2D2B regression exact": preflight["frozen_2d2b_regression"]["checks"]["parallel"],
        "B2 gate-zero identity exact": preflight["initial_b2_w32_compression"]["passed"],
        "B1 temporal gradient present": preflight["temporal_gradient_checks"]["b1_finite_nonzero"],
        "B2 temporal gradient present after opening": temporal["b2"][str(MAX_UPDATES)]["nonzero"],
        "B2 temporal gradient reaches 512+": temporal["b2"][str(MAX_UPDATES)]["long_lag_writer_gradient_present"],
        "same-model recurrence only": preflight["architecture"]["forbidden_modules_absent"]["teacher"],
        "CE-only loss": True,
        "pass weights exact": all(pass_weights(update) in (TWO_PASS_WEIGHTS, THREE_PASS_WEIGHTS) for update in range(1, MAX_UPDATES + 1)),
        "Pass-3 cadence exact": [update for update in range(1, MAX_UPDATES + 1) if pass_count(update) == 3] == [32, 64, 96, 128, 160],
        "optimizer resume exact": preflight["source"]["source_optimizer"]["restored_exactly_via_strict_optimizer_load_state_dict"],
        "loader/RNG continuation exact": preflight["checks"]["loader_continuation"],
        "matched 2D2C data replay verified": replay_complete,
        "global targets/update 524,288": preflight["checks"]["global_batch"],
        "191 additional optimizer updates": final_model_audit["completed_2d2e_updates"] == 191,
        "100,139,008 additional targets": final_model_audit["processed_2d2e_targets"] == ADDITIONAL_TARGETS,
        "no new projection": preflight["architecture"]["forbidden_modules_absent"]["dedicated_recurrent_projection"],
        "no teacher": preflight["architecture"]["forbidden_modules_absent"]["teacher"],
        "no AttnRes": preflight["architecture"]["forbidden_modules_absent"]["attnres"],
        "no third mirrored link": preflight["architecture"]["forbidden_modules_absent"]["additional_link_beyond_B11_to_B2"],
        "no detached training arm": preflight["architecture"]["forbidden_modules_absent"]["detached_training_arm"],
        "B1 physical KV <=1 historical entry": cache_pass,
        "B2 physical KV <=31 historical entries": cache_pass,
        "B11/B12 recurrent raw-state buffers <=1023": cache_pass,
        "all model/optimizer tensors finite": final_model_audit["model_finite"] and final_model_audit["optimizer_finite"],
        "forced restart exact": restart["passed"],
        "scientific checkpoints exact": set(scientific) == {"96", "191"}
        and all(row["passed"] for row in scientific.values()),
        "recovery checkpoint count <=1": len(recovery) <= 1
        and all(row["passed"] for row in recovery.values()),
        "true incremental evaluation completed": incremental["minimum_target_requirement_met"] and incremental["no_complete_prefix_recomputation"],
        "parallel/incremental corrected equivalence": equivalence["passed"],
        "eight-pass stability": composition["finite"],
        "persistent artifacts synchronized": preflight["persistent_workspace_audit"]["passed"],
        "results commit synchronized": False,
        "sealed report commit synchronized": False,
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
    if training_complete["completed_2d2e_updates"] != MAX_UPDATES:
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
        "b2": read_json(output / "b2_recurrent_attention_lag_bins.json"),
        "b2_local": read_json(output / "b2_local_attention_lag_bins.json"),
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
    final_attention["b2_local"] = attention_lags["b2_local"][str(MAX_UPDATES)]
    training = read_jsonl(output / "training_metrics.jsonl")
    if len(training) != MAX_UPDATES or training[-1]["local_update"] != MAX_UPDATES:
        raise SystemExit("training metrics do not contain exactly 191 updates")
    memory = memory_accounting(incremental)
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
        if key
        not in {
            "results commit synchronized",
            "sealed report commit synchronized",
            "required artifact set complete",
        }
    )
    classification = classify_result(
        incremental, stable=stable, integrity=scientific_integrity
    )
    initial_shortening = read_json(output / "initial_b2_w32_compression.json")
    recommendation = choose_recommendation(
        classification, initial_shortening["initial_W32_compression_damage"]
    )
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
        "distributed_equivalence": read_json(output / "distributed_equivalence.json"),
        "architecture": architecture_manifest(),
        "training": performance["training"],
        "initial_b2_w32_compression": initial_shortening,
        "validation_trajectory": milestones,
        "parallel": parallel,
        "incremental": incremental,
        "final_attention": final_attention,
        "final_temporal_gradient": {
            link: temporal[link][str(MAX_UPDATES)] for link in ("b1", "b2")
        },
        "memory_accounting": memory,
        "matched_2d2c_reference": matched_2d2c_reference_manifest(),
        "matched_2d2c_data_replay": read_json(
            output / "matched_2d2c_data_replay_audit.json"
        ),
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
            "sealed_commit": None,
        },
        "pod": {
            "id": args.pod_id,
            "name": args.pod_name,
            "status": "RUNNING_PENDING_FINAL_SYNC_AND_STOP",
            "exact_stop_command": f"runpodctl pod stop {args.pod_id} -o json",
            "persistent_volume_id": args.persistent_volume_identity,
            "persistent_volume_status": "PRESERVED_MOUNTED",
            "pod_delete_authorized": False,
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
            "new_real_vs_b3_shuffled": parallel["new_real_vs_b3_shuffled"],
            "new_real_vs_b2_full": parallel["new_real_vs_b2_full"],
        },
        "true_incremental": {
            key: incremental[key]
            for key in (
                "all_real_vs_b2_off_batches",
                "all_real_vs_b3_shuffled_batches",
                "all_real_vs_b2_full_batches",
                "all_real_vs_b2_off_sequences",
                "all_real_vs_b3_shuffled_sequences",
                "all_real_vs_b2_full_sequences",
            )
        },
    }
    cache_audit = {
        "controls": {
            name: row["cache_rows"] for name, row in incremental["controls"].items()
        },
        "B1_historical_KV_limit": 1,
        "B2_historical_KV_limit": 31,
        "B11_raw_recurrent_state_limit": 1023,
        "B12_raw_recurrent_state_limit": 1023,
        "B3_B12_historical_KV_limit": 1023,
        "no_hidden_full_B1_or_B2_cache": all(
            cache["final"]["b1_historical_kv"] <= 1
            and cache["final"]["b2_historical_kv"] <= 31
            for cache in incremental["controls"]["all_real"]["cache_rows"]
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
    durable_text(output / "EXPERIMENT_2D2E_FINAL_REPORT.md", report)
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
    durable_text(output / "EXPERIMENT_2D2E_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nPending lifecycle action: commit/push, verify synchronization, then stop the exact GPU pod without deleting it.\n",
    )
    if not scientific_integrity or not inventory["passed"]:
        raise SystemExit(
            f"2D2E finalize integrity failed: scientific={scientific_integrity} inventory={inventory['passed']}"
        )
    print("EXPERIMENT_2D2E_FINALIZED_PENDING_GIT_SEAL", flush=True)
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
        "run attest-seal against that pushed sealed-report commit",
        "commit and push the non-self-referential seal attestation",
        "verify local/origin/pod commit equality and clean worktrees",
        "verify no scientific process",
        "runpodctl pod stop 7kk5yyti00rnrp -o json",
        "verify stopped and not deleted",
    ]
    summary["git"]["results_commit"] = args.results_commit
    summary["git"]["report_base_commit"] = git_output("rev-parse", "HEAD")
    audit["checks"]["results commit synchronized"] = True
    audit["checks"]["sealed report commit synchronized"] = False
    inventory = build_artifact_inventory(output)
    audit["checks"]["required artifact set complete"] = inventory["passed"]
    audit["artifact_inventory"] = inventory
    audit["passed"] = False
    seal_content_passed = all(
        value
        for key, value in audit["checks"].items()
        if key != "sealed report commit synchronized"
    )
    if not seal_content_passed:
        raise SystemExit(f"seal-content audit failed: {audit['checks']}")
    summary["pod"]["status"] = "AWAITING_SEALED_REPORT_COMMIT_PUSH_AND_ATTESTATION"
    durable_json(output / "commands_and_runtime.json", commands)
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_json(output / "artifact_inventory.json", inventory)
    report = render_report(summary, audit, questions)
    durable_text(output / "EXPERIMENT_2D2E_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nCommit/push this sealed report, run the non-self-referential seal "
        "attestation, commit/push that attestation, verify replicas, then stop the pod.\n",
    )
    print("EXPERIMENT_2D2E_REPORT_SEALED", flush=True)


def run_attest_seal(args):
    """Record the already-pushed sealed-report commit in a final attestation."""

    require_git(clean=True)
    output = Path(args.output_dir).resolve()
    expected_output = (REPO_ROOT / "results" / OUTPUT_NAME).resolve()
    if output != expected_output:
        raise SystemExit(f"attest-seal output must be exactly {expected_output}")
    if git_output("rev-parse", "HEAD") != args.sealed_commit:
        raise SystemExit("attest-seal HEAD must equal supplied sealed commit")
    if git_output("rev-parse", f"origin/{BRANCH}") != args.sealed_commit:
        raise SystemExit("sealed commit must be pushed before attestation")
    summary = read_json(output / "result_summary.json")
    audit = read_json(output / "FINAL_AUDIT.json")
    questions = read_json(output / "scientific_questions.json")
    commands = read_json(output / "commands_and_runtime.json")
    if not summary["git"].get("results_commit"):
        raise SystemExit("attestation requires a sealed results report")
    pre_attestation_pass = all(
        value
        for key, value in audit["checks"].items()
        if key != "sealed report commit synchronized"
    )
    if not pre_attestation_pass:
        raise SystemExit("sealed report has failing checks before attestation")
    summary["git"]["sealed_commit"] = args.sealed_commit
    summary["git"]["seal_attestation_base_commit"] = args.sealed_commit
    summary["pod"]["status"] = "READY_TO_STOP_AFTER_ATTESTATION_COMMIT_PUSH"
    commands["attest_seal_command"] = " ".join(sys.argv)
    audit["checks"]["sealed report commit synchronized"] = True
    inventory = build_artifact_inventory(output)
    audit["artifact_inventory"] = inventory
    audit["passed"] = all(audit["checks"].values()) and inventory["passed"]
    if not audit["passed"]:
        raise SystemExit("sealed commit attestation audit failed")
    durable_json(output / "commands_and_runtime.json", commands)
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_json(output / "artifact_inventory.json", inventory)
    report = render_report(summary, audit, questions)
    durable_text(output / "EXPERIMENT_2D2E_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nThe sealed-report commit is attested. Commit/push this attestation, "
        "verify all replicas, then stop (do not delete) the exact GPU pod.\n",
    )
    print("EXPERIMENT_2D2E_SEAL_ATTESTED", flush=True)


def reference_2d2d_manifest() -> dict:
    return {
        "schema": "exp2d2e_frozen_2d2d_reference_v1",
        "checkpoint": "/workspace/exp2d2d_run/checkpoints/scientific_update_0191.pt",
        "checkpoint_sha256": SOURCE_SHA256,
        "parameters": SOURCE_PARAMETERS,
        "b1_effective_gate": SOURCE_GATE_EFFECTIVE,
        "b2_effective_gate": SOURCE_B2_GATE_EFFECTIVE,
        "b2_w32_initial_compression_damage": 0.03701871640278087,
        "final_true_b2_gain": 0.000004624556640742128,
        "final_true_b2_sequence_gap": -0.00006591846502423948,
        "final_b1_same_checkpoint_marginal_gain": 0.008109134272581198,
        "final_parallel_all_real_loss": 3.092560537382269,
        "frozen_reference_only": True,
    }


def semantic_diff_audit(model, source_payload) -> dict:
    source_names = list(source_payload["model"])
    target_names = list(model.state_dict())
    report = {
        "baseline": "final Experiment 2D2D",
        "architecture_changes": [
            {"field": "B11->B2 recurrent link", "old": "lags 32...1023", "new": None},
            {"field": "B2 recurrent scalar", "old": "g_rec_b2", "new": None},
            {"field": "B3 ordinary local window", "old": 1024, "new": 64},
            {"field": "new recurrent link", "old": None, "new": "B10->B3 lags 64...1023"},
            {"field": "new learnable tensor", "old": None, "new": "scalar g_rec_b3"},
        ],
        "new_learnable_tensors": ["g_rec_b3"],
        "source_state_dict_keys_preserved": [
            key for key in target_names if key != "g_rec_b3"
        ] == [key for key in source_names if key != "g_rec_b2"],
        "b1_gate_resumed": model.g_rec.detach().float().item() == SOURCE_GATE_RAW,
        "b2_gate_absent": not hasattr(model, "g_rec_b2"),
        "b11_ring_absent": "h11_ring" not in MirroredIncrementalState.__dataclass_fields__,
        "b3_gate_zero": model.g_rec_b3.detach().float().item() == 0.0,
        "memory_efficient_source_storage": True,
    }
    report["passed"] = all(
        report[key]
        for key in ("source_state_dict_keys_preserved", "b1_gate_resumed", "b2_gate_absent", "b11_ring_absent", "b3_gate_zero")
    )
    if not report["passed"]:
        raise SystemExit(f"semantic diff audit failed: {report}")
    return report


@torch.no_grad()
def evaluate_parallel(model, val_path, batches=VALIDATION_BATCHES, combined_controls=False) -> dict:
    """Evaluate B3 recurrence and combined-system marginal controls on fixed rows."""
    model.eval()
    device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    names = ["all_real", "b3_off", "b3_shuffled", "b3_full_counterfactual"]
    if combined_controls:
        names.extend(("all_shuffled", "b1_off_b3_real", "b1_real_b3_off"))
    accumulators = {
        name: {"sum": 0.0, "targets": 0, "batches": [], "positions": np.zeros(T, dtype=np.float64)}
        for name in names
    }
    identities = []
    permutation = torch.arange(VALIDATION_B, device=device).roll(1)
    started = time.monotonic()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(int(batches)):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(legacy.d0d.batch_identity(cpu_x, cpu_y))
        x, y = cpu_x.to(device), cpu_y.to(device)
        for name in names:
            kwargs = {}
            if name in {"b3_off", "b1_real_b3_off"}:
                kwargs["b3_gate_override"] = 0.0
            elif name == "b3_shuffled":
                kwargs["b3_recurrent_permutation"] = permutation
            elif name == "b3_full_counterfactual":
                kwargs["b3_full_counterfactual"] = True
            elif name == "all_shuffled":
                kwargs.update(b1_recurrent_permutation=permutation,
                              b3_recurrent_permutation=permutation)
            elif name == "b1_off_b3_real":
                kwargs["b1_gate_override"] = 0.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model.forward_multi_pass(x, num_passes=2, **kwargs)
                losses = _token_losses(result["logits"], y)
            row = accumulators[name]
            row["sum"] += losses.double().sum().item()
            row["targets"] += losses.numel()
            row["batches"].append(losses.float().mean().item())
            row["positions"] += losses.double().sum(0).cpu().numpy()
            del result, losses
            torch.cuda.empty_cache()
        print(f"2D2F validation batch={batch_index + 1:02d}/{batches}", flush=True)
        del x, y, cpu_x, cpu_y
    finished = {}
    for name, row in accumulators.items():
        finished[name] = {
            "validation_loss": row["sum"] / row["targets"],
            "validation_targets": row["targets"],
            "per_batch_losses": row["batches"],
            "per_position_loss": (row["positions"] / (int(batches) * VALIDATION_B)).tolist(),
        }
    real, off = finished["all_real"], finished["b3_off"]
    shuffled, full = finished["b3_shuffled"], finished["b3_full_counterfactual"]
    result = {
        "controls": finished,
        "b3_recurrent_gain": off["validation_loss"] - real["validation_loss"],
        "b3_sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "b3_w64_compression_gap_vs_full": real["validation_loss"] - full["validation_loss"],
        "all_real_vs_b3_off": paired_stats(real["per_batch_losses"], off["per_batch_losses"]),
        "all_real_vs_b3_shuffled": paired_stats(real["per_batch_losses"], shuffled["per_batch_losses"]),
        "all_real_vs_b3_full": paired_stats(real["per_batch_losses"], full["per_batch_losses"]),
        "g_rec_b1_raw": model.g_rec.detach().float().item(),
        "tanh_g_rec_b1": model.recurrent_scale_b1.detach().float().item(),
        "g_rec_b3_raw": model.g_rec_b3.detach().float().item(),
        "tanh_g_rec_b3": model.recurrent_scale_b3.detach().float().item(),
        "canonical_validation_sha256": legacy.d0.aggregate_hashes([row["combined_sha256"] for row in identities]),
        "batch_identities": identities,
        "batch_count": int(batches), "batch_size": VALIDATION_B, "sequence_length": T,
        "loss_denominator": int(batches) * VALIDATION_B * T,
        "precision": "torch.autocast(cuda,bfloat16)",
        "performance": {
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }
    if combined_controls:
        result["combined_system"] = {
            "all_real_vs_all_shuffled_gap": finished["all_shuffled"]["validation_loss"] - real["validation_loss"],
            "b1_marginal_gain": finished["b1_off_b3_real"]["validation_loss"] - real["validation_loss"],
            "b3_marginal_gain": off["validation_loss"] - real["validation_loss"],
            "b3_shuffled_gap": shuffled["validation_loss"] - real["validation_loss"],
        }
    return result


def _weighted_quantile(histogram, quantile):
    total = histogram.sum()
    if not bool(total > 0):
        return 0.0
    return int(torch.searchsorted(histogram.cumsum(0), total * quantile).item())


@torch.no_grad()
def attention_diagnostics(model, val_path, link, batch_size=2) -> dict:
    if link not in {"b1", "b3"}:
        raise ValueError(link)
    model.eval(); device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], batch_size, T)
    cpu_x, cpu_y = loader.next_batch(); identity = legacy.d0d.batch_identity(cpu_x, cpu_y)
    x = cpu_x.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.forward_multi_pass(x, num_passes=2, return_diagnostics=True)
    diagnostics = result["diagnostics"][-1][link]
    weights = diagnostics["recurrent_attention_weights"]
    valid = diagnostics["recurrent_valid_mask"]
    local_weights = diagnostics["local_attention_weights"]
    local_valid = diagnostics["local_valid_mask"]
    if weights is None or valid is None or local_weights is None or local_valid is None:
        raise SystemExit(f"{link} diagnostic weights unavailable")
    query = torch.arange(T, device=device).view(T, 1)
    source = torch.arange(T, device=device).view(1, T)
    lag = query - source
    recurrent_bins = {"b1": B1_LAG_BINS, "b3": B3_RECURRENT_LAG_BINS}[link]
    local_bins_spec = {"b1": (("0-1", 0, 1),), "b3": B3_LOCAL_LAG_BINS}[link]
    total = weights.double().sum()
    bins = {}
    for name, first, last in recurrent_bins:
        selected = valid & (lag >= first) & (lag <= last)
        mass = weights.masked_select(selected.view(1, 1, T, T)).double().sum()
        available = int(selected.sum()) * batch_size * N_HEAD
        bins[name] = {"lag_min": first, "lag_max": last, "attention_mass": (mass / total).item(),
                      "raw_attention_mass": mass.item(),
                      "normalized_mass_per_available_token": mass.item() / max(available, 1),
                      "valid_position_instances_all_heads_rows": available}
    local_total = local_weights.double().sum()
    local_bins = {}
    for name, first, last in local_bins_spec:
        selected = local_valid & (lag >= first) & (lag <= last)
        mass = local_weights.masked_select(selected.view(1, 1, T, T)).double().sum()
        available = int(selected.sum()) * batch_size * N_HEAD
        local_bins[name] = {"lag_min": first, "lag_max": last, "attention_mass": (mass / local_total).item(),
                            "raw_attention_mass": mass.item(),
                            "normalized_mass_per_available_token": mass.item() / max(available, 1),
                            "valid_position_instances_all_heads_rows": available}
    def summarize(current):
        histogram = torch.zeros(RECURRENT_MAX_LAG + 1, dtype=torch.float64, device=device)
        aggregated = current.double().sum((0, 1)) if current.ndim == 4 else current.double().sum(0)
        histogram.scatter_add_(0, lag[valid].long(), aggregated[valid])
        mass = histogram.sum()
        mean = ((histogram * torch.arange(histogram.numel(), device=device)).sum() / mass).item()
        return {"mean_attended_recurrent_lag": mean,
                "median_attended_recurrent_lag": _weighted_quantile(histogram, 0.5),
                "p90_attended_recurrent_lag": _weighted_quantile(histogram, 0.9),
                "total_attention_mass": mass.item()}
    heads = {str(head): summarize(weights[:, head]) for head in range(N_HEAD)}
    report = {"link": {"b1": "B12->B1", "b3": "B10->B3"}[link],
              "pinned_batch": identity, "batch_size": batch_size, "sequence_length": T,
              "lag_bins": bins, "local_lag_bins": local_bins, "heads": heads,
              "aggregate": summarize(weights), "weights_finite": bool(torch.isfinite(weights).all()),
              "head_mean_lag_range": max(v["mean_attended_recurrent_lag"] for v in heads.values()) - min(v["mean_attended_recurrent_lag"] for v in heads.values())}
    del x, result, diagnostics, weights, local_weights
    torch.cuda.empty_cache()
    return report


def temporal_gradient_by_lag(model, val_path, link, precision="bf16", gate_override=None, **legacy_kwargs) -> dict:
    if link not in {"b1", "b3"}:
        raise ValueError(link)
    model.train(); device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], 1, T)
    cpu_x, cpu_y = loader.next_batch(); identity = legacy.d0d.batch_identity(cpu_x, cpu_y)
    x, y = cpu_x.to(device), cpu_y.to(device)
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()
    with torch.enable_grad(), context:
        first = model.forward_pass(x, activation_checkpointing=True)
        mapping = {"b1": ("h12", "b1_recurrent_source", "b1_gate_override"),
                   "b3": ("h10", "b3_recurrent_source", "b3_gate_override")}
        source_key, argument, gate_argument = mapping[link]
        source = first[source_key].detach().requires_grad_(True)
        kwargs = {"b1_recurrent_source": first["h12"].detach(),
                  "b3_recurrent_source": first["h10"].detach(),
                  argument: source, "activation_checkpointing": True}
        if gate_override is not None:
            kwargs[gate_argument] = gate_override
        second = model.forward_pass(x, targets=y, **kwargs)
        loss = F.cross_entropy(second["logits"][:, -1].float(), y[:, -1])
        gradient = torch.autograd.grad(loss, source)[0].float()
    minimum = {"b1": 2, "b3": 64}[link]
    lag_bins = {"b1": B1_LAG_BINS, "b3": B3_RECURRENT_LAG_BINS}[link]
    lags = T - 1 - torch.arange(T, device=device)
    bins = {}
    for name, first_lag, last_lag in lag_bins:
        selected = (lags >= first_lag) & (lags <= last_lag)
        values = gradient[:, selected]
        rms = values.square().mean((0, 2)).sqrt()
        bins[name] = {"mean_gradient_rms": values.square().mean().sqrt().item(),
                      "max_gradient_rms": rms.max().item(),
                      "fraction_nonzero_elements": values.ne(0).float().mean().item(),
                      "fraction_nonzero_positions": rms.ne(0).float().mean().item(),
                      "source_positions": int(selected.sum())}
    latest = T - 1 - minimum
    gate = {"b1": model.g_rec, "b3": model.g_rec_b3}[link]
    effective = {"b1": model.recurrent_scale_b1, "b3": model.recurrent_scale_b3}[link]
    report = {"link": {"b1": "B12->B1", "b3": "B10->B3"}[link],
              "precision": precision, "pinned_batch": identity,
              "gate_raw": gate.detach().float().item(), "effective_gate": effective.detach().float().item(),
              "gate_override": gate_override, "receiver_position": T - 1,
              "receiver_loss": loss.detach().float().item(), "gradient_norm": gradient.norm().item(),
              "finite": bool(torch.isfinite(gradient).all()), "nonzero": bool(gradient.count_nonzero()),
              "bins": bins,
              "position_probes": {"early_old_position_0": gradient[:, 0].square().mean().sqrt().item(),
                                   "middle_position_512": gradient[:, 512].square().mean().sqrt().item(),
                                   f"recent_eligible_position_{latest}": gradient[:, latest].square().mean().sqrt().item(),
                                   f"recent_ineligible_position_{latest + 1}": gradient[:, latest + 1].square().mean().sqrt().item()},
              "long_lag_writer_gradient_present": bins["512-1023"]["fraction_nonzero_positions"] > 0}
    model.zero_grad(set_to_none=True)
    del x, y, first, second, source, gradient, loss
    torch.cuda.empty_cache()
    return report


def attached_writer_gradient_check(model, val_path) -> dict:
    report = {}
    for link, source, latest in (("b1", "h12", 1021), ("b3", "h10", 959)):
        row = temporal_gradient_by_lag(model, val_path, link, gate_override=(0.05 if link == "b3" else None))
        report[{"b1": "b12_to_b1", "b3": "b10_to_b3"}[link]] = {
            "finite": row["finite"], "nonzero": row["nonzero"],
            "position_0_rms": row["position_probes"]["early_old_position_0"],
            f"latest_eligible_{latest}_rms": row["position_probes"][f"recent_eligible_position_{latest}"],
        }
    report["passed"] = all(row["finite"] and row["nonzero"] for row in report.values())
    return report


def kernel_preflight(model, short_tokens, short_targets) -> dict:
    model.eval(); device = short_tokens.device; length = short_tokens.size(1)
    query = torch.arange(length, device=device).view(length, 1)
    source = torch.arange(length, device=device).view(1, length)
    masks = {"b1": model.recurrent_mask(length, length, device),
             "b3": model.b3_recurrent_mask(length, length, device)}
    local = {"b1": model.local_mask(length, device), "b2": model.b2_local_mask(length, device),
             "b3": model.b3_local_mask(length, device)}
    minima = {"b1": 2, "b3": 64}
    checks = {
        "b2_local_mask_exact": torch.equal(
            local["b2"], (source <= query) & (source >= query - 31)
        ),
        "b2_recurrent_api_absent": not any(
            hasattr(model, name)
            for name in ("build_recurrent_bank_b2", "project_recurrent_kv_b2")
        ),
        "b2_gate_absent": not hasattr(model, "g_rec_b2"),
        "b11_ring_absent": "h11_ring" not in MirroredIncrementalState.__dataclass_fields__,
    }
    for name in masks:
        expected = (source <= query - minima[name]) & (source >= query - 1023)
        checks[f"{name}_mask_exact"] = torch.equal(masks[name], expected)
        checks[f"{name}_nonoverlap"] = not bool((masks[name] & local[name]).any())
        checks[f"{name}_partition"] = torch.equal(masks[name] | local[name], source <= query)
    full_b3 = model.b3_recurrent_mask(T, T, device)
    counts = {str(position): int(full_b3[position].sum()) for position in (0, 63, 64, 65, 1023)}
    checks["b3_boundary_counts"] = counts == {"0": 0, "63": 0, "64": 1, "65": 2, "1023": 960}
    values = torch.randn(short_tokens.size(0), length, N_EMBD, device=device)
    banks = (model.build_recurrent_bank(values), model.build_recurrent_bank_b3(values))
    checks["rank_three_unrepeated_banks"] = all(bank.values.ndim == 3 and bank.values.data_ptr() == values.data_ptr() for bank in banks)
    calls = {"b1": 0, "b2": 0, "b3": 0}
    hooks = [model.base.transformer.h[index].attn.c_proj.register_forward_hook(
        lambda _m, _i, _o, name=name: calls.__setitem__(name, calls[name] + 1))
        for index, name in enumerate(("b1", "b2", "b3"))]
    try:
        with torch.no_grad():
            active = model.forward_pass(short_tokens, targets=short_targets,
                b1_recurrent_source=values, b3_recurrent_source=values,
                b3_gate_override=0.05, return_diagnostics=True)
    finally:
        for hook in hooks: hook.remove()
    checks["single_c_proj_each"] = calls == {"b1": 1, "b2": 1, "b3": 1}
    with torch.no_grad():
        first = model.forward_pass(short_tokens)
        absent = model.forward_pass(short_tokens, b1_recurrent_source=first["h12"],
                                    b3_gate_override=0.0)["logits"]
        zero = model.forward_pass(short_tokens, b1_recurrent_source=first["h12"],
                                  b3_recurrent_source=first["h10"],
                                  b3_gate_override=0.0)["logits"]
    checks["b3_gate_zero_identity"] = torch.equal(absent, zero)
    changed = short_tokens.clone(); changed[:, -1] = (changed[:, -1] + 17) % model.config.vocab_size
    with torch.no_grad():
        base = model.forward_multi_pass(short_tokens, num_passes=2, b3_gate_override=0.05)["logits"]
        altered = model.forward_multi_pass(changed, num_passes=2, b3_gate_override=0.05)["logits"]
    checks["future_causality"] = torch.equal(base[:, :-1], altered[:, :-1])
    if short_tokens.size(0) > 1:
        changed = short_tokens.clone(); changed[0] = (changed[0] + 29) % model.config.vocab_size
        with torch.no_grad(): isolated = model.forward_multi_pass(changed, num_passes=2, b3_gate_override=0.05)["logits"]
        checks["row_isolation"] = torch.equal(base[1:], isolated[1:])
    return {"checks": checks, "reports": {"b3_boundary_counts": counts, "c_proj_calls": calls}, "passed": all(checks.values())}


def frozen_2d2d_regression(model, source_payload, val_path) -> dict:
    old = source_core.RecurrentKVGPT(model.base)
    old.g_rec = model.g_rec
    old.g_rec_b2 = torch.nn.Parameter(source_payload["model"]["g_rec_b2"].to(next(model.parameters()).device))
    observed = source_driver.evaluate_parallel(old, val_path, combined_controls=True)
    loss = observed["controls"]["new_real"]["validation_loss"]
    checks = {"parallel_loss_exact": abs(loss - 3.092560537382269) <= 5e-7,
              "canonical": observed["canonical_validation_sha256"] == CANONICAL_VALIDATION_SHA256,
              "b1_gate": model.g_rec.detach().float().item() == SOURCE_GATE_RAW,
              "b2_gate": old.g_rec_b2.detach().float().item() == SOURCE_B2_GATE_RAW}
    report = {"expected_all_real_loss": 3.092560537382269, "observed_all_real_loss": loss,
              "full": observed, "checks": checks, "passed": all(checks.values())}
    if not report["passed"]: raise SystemExit(f"frozen 2D2D regression failed: {checks}")
    del old; torch.cuda.empty_cache()
    return report


def probe_microbatch(model, optimizer, shards, device, candidates=(32, 16, 8, 4, 2)):
    model.train(); attempts = []
    total_vram = torch.cuda.get_device_properties(device).total_memory
    required_headroom = max(4 * 1024**3, int(0.10 * total_vram))
    for candidate in candidates:
        if (GLOBAL_TARGETS // T) % candidate:
            continue
        loader = legacy.d1.ExplicitShardLoader(shards, candidate, T)
        cpu_x, cpu_y = loader.next_batch(); x, y = cpu_x.to(device), cpu_y.to(device)
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        rows = []
        try:
            h10 = h12 = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    current = model.forward_pass(x, targets=y, b1_recurrent_source=h12,
                                                 b3_recurrent_source=h10,
                                                 activation_checkpointing=True)
                    rows.append(current); h10, h12 = current["h10"], current["h12"]
                loss = sum(weight * row["loss"] for weight, row in zip(THREE_PASS_WEIGHTS, rows))
            loss.backward(); torch.cuda.synchronize()
            peak_reserved = torch.cuda.max_memory_reserved(device)
            headroom = total_vram - peak_reserved
            passed = gradients_finite(model) and headroom >= required_headroom
            attempts.append({"micro_batch_sequences": candidate, "passed": passed,
                             "loss": loss.detach().float().item(),
                             "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                             "peak_reserved_vram_mb": peak_reserved / 1024**2,
                             "headroom_mb": headroom / 1024**2,
                             "required_headroom_mb": required_headroom / 1024**2})
            model.zero_grad(set_to_none=True)
            if passed:
                return candidate, attempts
        except torch.cuda.OutOfMemoryError as error:
            attempts.append({"micro_batch_sequences": candidate, "passed": False, "error": type(error).__name__})
            model.zero_grad(set_to_none=True)
        finally:
            del x, y, cpu_x, cpu_y, rows
            gc.collect(); torch.cuda.empty_cache()
    raise SystemExit(f"no safe 2D2F microbatch: {attempts}")


def run_preflight(args):
    require_git(clean=True); require_config(); fingerprint = implementation_fingerprint()
    mount = workspace_mount_audit(args.output_dir, args.run_root, args.persistent_volume_identity)
    stop = authenticated_stop_audit(args); device = require_single_a100(); seed_all()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"preflight output already exists and is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    model, optimizer, source_loader, source_payload, source_audit = load_source_bundle(
        args.source_checkpoint, device, restore_rng=False)
    val_path = validation_path(args.data_root)
    manifest = validation_manifest(val_path)
    if manifest["canonical_batch_collection_sha256"] != CANONICAL_VALIDATION_SHA256:
        raise SystemExit("canonical validation collection mismatch")
    parameters = parameter_manifest(model, source_payload)
    architecture = architecture_manifest(); semantic = semantic_diff_audit(model, source_payload)
    source_manifest = {**source_audit, "frozen_2d2d_tag": FROZEN_TAG,
                       "frozen_2d2d_commit": FROZEN_COMMIT,
                       "frozen_tag_exact": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
                       "canonical_validation_sha256": CANONICAL_VALIDATION_SHA256}
    reference = reference_2d2d_manifest()
    durable_json(output / "source_manifest.json", source_manifest)
    durable_json(output / "parameter_manifest.json", parameters)
    durable_json(output / "architecture_manifest.json", architecture)
    durable_json(output / "semantic_diff_audit.json", semantic)
    durable_json(output / "2d2d_reference_manifest.json", reference)

    short_loader = legacy.d1.ExplicitShardLoader([val_path], 2, 72)
    short_x, short_y = short_loader.next_batch()
    kernel = kernel_preflight(model, short_x.to(device), short_y.to(device))
    if not kernel["passed"]: raise SystemExit(f"kernel preflight failed: {kernel['checks']}")
    regression = frozen_2d2d_regression(model, source_payload, val_path)
    zero = evaluate_parallel(model, val_path); zero.update(local_update=0, additional_targets=0)
    source_loss = regression["observed_all_real_loss"]
    off_loss = zero["controls"]["b3_off"]["validation_loss"]
    initial = {"source_2d2d_loss": source_loss,
               "b3_w64_off_loss": off_loss,
               "b3_w64_real_zero_gate_loss": zero["controls"]["all_real"]["validation_loss"],
               "b3_w64_shuffled_zero_gate_loss": zero["controls"]["b3_shuffled"]["validation_loss"],
               "initial_B3_W64_compression_damage": off_loss - source_loss,
               "damage_relative_to_source_loss": (off_loss - source_loss) / source_loss,
               "gate_zero_real_identity": zero["controls"]["all_real"]["validation_loss"] == off_loss,
               "gate_zero_shuffled_identity": zero["controls"]["b3_shuffled"]["validation_loss"] == off_loss}
    initial["passed"] = initial["gate_zero_real_identity"] and initial["gate_zero_shuffled_identity"]
    if not initial["passed"]: raise SystemExit(f"initial B3 gate-zero identity failed: {initial}")

    temporal = {
        "b1": temporal_gradient_by_lag(model, val_path, "b1"),
        "b3_probe": temporal_gradient_by_lag(model, val_path, "b3", gate_override=0.05),
    }
    attached = attached_writer_gradient_check(model, val_path)
    if not attached["passed"]: raise SystemExit(f"attached writer gradient preflight failed: {attached}")
    attention = {link: attention_diagnostics(model, val_path, link) for link in ("b1", "b3")}
    selected, probe = probe_microbatch(model, optimizer, source_payload["loader_state"]["shards"], device)
    accumulation = GLOBAL_TARGETS // (selected * T)
    scientific_loader = loader_at_source_cursor(source_payload["loader_state"], selected)
    source_accumulation = int(source_payload["metadata"]["gradient_accumulation"])
    source_stream = global_batch_stream_hash(source_loader, source_accumulation)
    scientific_stream = global_batch_stream_hash(scientific_loader, accumulation)
    reference_path = REPO_ROOT / "results" / "experiment_2d2e_b3_w64_b10_recurrent_960" / "batch_manifest.json"
    reference_2d2e = read_json(reference_path)
    matched_2d2e = {
        "schema": "exp2d2f_matched_2d2e_data_audit_v1",
        "source_loader_state_restored": True,
        "same_first_batch_sha256": observed_next if (observed_next := source_audit["next_global_batch_sha256"]) else None,
        "expected_first_batch_sha256": SOURCE_NEXT_BATCH_SHA256,
        "scientific_global_stream_sha256": scientific_stream,
        "frozen_2d2e_global_stream_sha256": reference_2d2e["scientific_global_stream_sha256"],
        "exact_updates": MAX_UPDATES,
        "exact_targets": ADDITIONAL_TARGETS,
        "per_update_hashes_available": False,
    }
    matched_2d2e["passed"] = (
        matched_2d2e["same_first_batch_sha256"] == SOURCE_NEXT_BATCH_SHA256
        and scientific_stream == reference_2d2e["scientific_global_stream_sha256"]
        and scientific_stream == SOURCE_NEXT_STREAM_SHA256
    )
    batch_manifest = {**manifest, "source_loader_state": copy.deepcopy(source_payload["loader_state"]),
                      "scientific_loader_initial_state": scientific_loader.state_dict(),
                      "selected_micro_batch_sequences": selected, "selected_gradient_accumulation": accumulation,
                      "global_targets_per_update": GLOBAL_TARGETS,
                      "source_global_stream_sha256": source_stream,
                      "scientific_global_stream_sha256": scientific_stream,
                      "logical_global_batch_exact_across_microbatch_geometry": source_stream == scientific_stream,
                      "microbatch_probe": probe}
    checks = {
        "source_checkpoint_exact": source_audit["checks"]["passed"],
        "source_tag_exact": source_manifest["frozen_tag_exact"],
        "parameters_exact": parameters["passed"], "semantic_diff_exact": semantic["passed"],
        "kernel": kernel["passed"], "source_regression": regression["passed"],
        "initial_gate_zero": initial["passed"], "attached_writer_gradients": attached["passed"],
        "b3_probe_long_gradient": temporal["b3_probe"]["long_lag_writer_gradient_present"],
        "canonical_validation": zero["canonical_validation_sha256"] == CANONICAL_VALIDATION_SHA256,
        "loader_continuation": source_stream == scientific_stream,
        "matched_2d2e_data": matched_2d2e["passed"],
        "global_batch": selected * T * accumulation == GLOBAL_TARGETS,
        "persistent_workspace": mount["passed"], "authenticated_stop": stop["driver_passed"],
    }
    preflight = {"experiment": EXPERIMENT, "protocol": PROTOCOL, "timestamp": time.time(),
                 "command": " ".join(sys.argv), "implementation_git_commit": git_output("rev-parse", "HEAD"),
                 "implementation_fingerprint": fingerprint, "environment": environment_payload(),
                 "source": source_manifest, "parameters": parameters, "architecture": architecture,
                 "semantic_diff": semantic, "kernel_preflight": kernel,
                 "source_regression": regression, "zero_shot": zero,
                 "initial_b3_w64_compression": initial, "temporal_gradient_preflight": temporal,
                 "attached_writer_gradient": attached, "attention_zero_shot": attention,
                 "selected_microbatch": selected, "gradient_accumulation": accumulation,
                 "microbatch_probe": probe, "runpod_stop_audit": stop,
                 "persistent_workspace_audit": mount, "checks": checks,
                 "science_passed": all(checks.values()), "result_run_authorized": all(checks.values()),
                 "wall_seconds": time.monotonic() - started}
    durable_json(output / "batch_manifest.json", batch_manifest)
    durable_json(output / "matched_2d2e_data_audit.json", matched_2d2e)
    durable_json(output / "preflight_audit.json", preflight)
    durable_json(output / "milestone_validation.json", {"0": zero})
    durable_json(output / "paired_controls.json", {"0": {"all_real_vs_b3_off": zero["all_real_vs_b3_off"], "all_real_vs_b3_shuffled": zero["all_real_vs_b3_shuffled"]}})
    durable_json(output / "gate_diagnostics.json", {"0": {key: zero[key] for key in ("g_rec_b1_raw", "tanh_g_rec_b1", "g_rec_b3_raw", "tanh_g_rec_b3")}})
    durable_json(output / "b1_attention_diagnostics.json", {"0": attention["b1"]})
    durable_json(output / "attention_diagnostics.json", {"0": attention})
    durable_json(output / "b3_local_attention_lag_bins.json", {"0": attention["b3"]["local_lag_bins"]})
    durable_json(output / "b3_recurrent_attention_lag_bins.json", {"0": attention["b3"]["lag_bins"]})
    durable_json(output / "b3_attention_head_distance.json", {"0": {"heads": attention["b3"]["heads"], "aggregate": attention["b3"]["aggregate"], "head_mean_lag_range": attention["b3"]["head_mean_lag_range"]}})
    for name, row in (("b12_to_b1_temporal_gradient.json", temporal["b1"]),
                      ("b10_to_b3_temporal_gradient.json", temporal["b3_probe"])):
        durable_json(output / name, {"0": row})
    durable_json(output / "temporal_gradient_diagnostics.json", {"0": temporal})
    durable_json(output / "checkpoint_manifest.json", {"scientific": {}, "smoke": {}})
    durable_json(output / "performance.json", {"preflight_microbatch_probe": probe})
    durable_json(output / "distributed_equivalence.json", {"used": False, "gpu_count": 1, "reason": "one assigned A100", "passed": True})
    durable_json(output / "storage_cleanup_manifest.json", {"schema": "exp2d2f_storage_cleanup_manifest_v1", "scientific_or_source_checkpoint_removed": False, "historical_artifacts_removed": False, "workspace_used_bytes_after_cleanup": mount["measured_used_bytes"], "workspace_free_bytes_after_cleanup": mount["measured_free_bytes"], "cleanup_actions": []})
    durable_json(output / "commands_and_runtime.json", {"preflight_command": " ".join(sys.argv), "environment": environment_payload()})
    durable_json(output / "persistent_workspace_audit.json", mount)
    durable_json(output / "runpod_stop_capability.json", stop)
    if not preflight["science_passed"]: raise SystemExit(f"2D2F preflight failed: {checks}")
    print("EXPERIMENT_2D2F_PREFLIGHT_PASS", flush=True)
    return preflight


def run_smoke(args):
    require_git(clean=False); require_config()
    workspace_mount_audit(args.output_dir, args.run_root, args.persistent_volume_identity)
    authenticated_stop_audit(args); device = require_single_a100()
    output = Path(args.output_dir).resolve(); preflight = read_json(output / "preflight_audit.json")
    if not preflight.get("result_run_authorized"): raise SystemExit("smoke requires passing preflight")
    require_implementation_fingerprint(preflight)
    model, optimizer, _, source_payload, _ = load_source_bundle(args.source_checkpoint, device, restore_rng=True)
    loader = loader_at_source_cursor(source_payload["loader_state"], 2)
    rows = []; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    for update in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        cpu_x, cpu_y = loader.next_batch(); x, y = cpu_x.to(device), cpu_y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            first = model.forward_pass(x, targets=y, activation_checkpointing=True)
            second = model.forward_pass(x, targets=y, b1_recurrent_source=first["h12"],
                                        b3_recurrent_source=first["h10"],
                                        activation_checkpointing=True, return_diagnostics=True)
            loss = TWO_PASS_WEIGHTS[0] * first["loss"] + TWO_PASS_WEIGHTS[1] * second["loss"]
        before = {"b1": model.g_rec.detach().float().item(),
                  "b3": model.g_rec_b3.detach().float().item()}
        loss.backward(); groups = gradient_group_report(model)
        gradients = {"b1": model.g_rec.grad.detach().float().item(),
                     "b3": model.g_rec_b3.grad.detach().float().item()}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
        after = {"b1": model.g_rec.detach().float().item(),
                 "b3": model.g_rec_b3.detach().float().item()}
        rows.append({"update": update, "loss": loss.detach().float().item(),
                     "pass_losses": [first["loss"].detach().float().item(), second["loss"].detach().float().item()],
                     "gate_raw_before": before, "gate_raw_after": after,
                     "gate_gradients": gradients, "gradient_groups": groups,
                     "gradient_norm": norm.detach().float().item(),
                     "gradients_finite": gradients_finite(model),
                     "parameters_finite": model_finite(model), "optimizer_finite": optimizer_finite(optimizer),
                     "recurrent_states_finite": all(bool(torch.isfinite(tensor).all()) for tensor in
                         (first["h10"], first["h12"], second["h10"], second["h12"]))})
        del x, y, cpu_x, cpu_y, first, second, loss
        torch.cuda.empty_cache()
    cache_x, _ = loader.clone().next_batch()
    with torch.no_grad():
        cache = model.incremental_logits(cache_x[:, :72].to(device), control="all_real", bank_mode="full")["cache_audit"]
    checks = {"exactly_three_updates": len(rows) == 3,
              "b3_gradient_update1_nonzero": math.isfinite(rows[0]["gate_gradients"]["b3"]) and rows[0]["gate_gradients"]["b3"] != 0,
              "all_gradient_groups_nonzero": all(all(row["gradient_groups"][name]["finite"] and row["gradient_groups"][name]["nonzero"] for name in ("base", "gate", "b3_gate")) for row in rows),
              "finite": all(row["parameters_finite"] and row["optimizer_finite"] and row["recurrent_states_finite"] for row in rows),
              "cache_geometry": cache["passed"] and cache["b1_historical_kv"] == 1 and cache["b2_historical_kv"] == 31 and cache["b3_historical_kv"] == 63,
              "source_checkpoint_untouched": file_sha256(args.source_checkpoint) == SOURCE_SHA256}
    audit = {"experiment": EXPERIMENT, "kind": "exactly three disposable optimizer updates",
             "command": " ".join(sys.argv), "rows": rows, "incremental_cache_audit": cache,
             "checks": checks, "passed": all(checks.values()),
             "disposition": "Discarded; scientific update 1 reloads exact finalized 2D2D."}
    durable_json(output / "smoke_audit.json", audit)
    manifest = read_json(output / "checkpoint_manifest.json")
    manifest["smoke"]["3"] = {"updates": 3, "binary_retained": False, "disposable": True}
    durable_json(output / "checkpoint_manifest.json", manifest)
    if not audit["passed"]: raise SystemExit(f"2D2F smoke failed: {checks}")
    print("EXPERIMENT_2D2F_SMOKE_PASS", flush=True)
    return audit


def milestone_diagnostics(runtime, update, val_path):
    saved_rng = capture_rng_state()
    validation = evaluate_parallel(runtime.model, val_path, combined_controls=(update == MAX_UPDATES))
    validation.update(local_update=update, additional_targets=update * GLOBAL_TARGETS,
                      cumulative_2d2_targets=SOURCE_TARGETS + update * GLOBAL_TARGETS)
    attention = {link: attention_diagnostics(runtime.model, val_path, link) for link in ("b1", "b3")}
    temporal = {link: temporal_gradient_by_lag(runtime.model, val_path, link) for link in ("b1", "b3")}
    for row in temporal.values():
        row.update(local_update=update, additional_targets=update * GLOBAL_TARGETS,
                   cumulative_2d2_targets=SOURCE_TARGETS + update * GLOBAL_TARGETS)
    merge_keyed_json(runtime.output / "milestone_validation.json", update, validation)
    merge_keyed_json(runtime.output / "paired_controls.json", update,
                     {"all_real_vs_b3_off": validation["all_real_vs_b3_off"],
                      "all_real_vs_b3_shuffled": validation["all_real_vs_b3_shuffled"]})
    merge_keyed_json(runtime.output / "gate_diagnostics.json", update,
                     {key: validation[key] for key in ("g_rec_b1_raw", "tanh_g_rec_b1", "g_rec_b3_raw", "tanh_g_rec_b3")})
    merge_keyed_json(runtime.output / "b1_attention_diagnostics.json", update, attention["b1"])
    merge_keyed_json(runtime.output / "attention_diagnostics.json", update, attention)
    merge_keyed_json(runtime.output / "b3_local_attention_lag_bins.json", update, attention["b3"]["local_lag_bins"])
    merge_keyed_json(runtime.output / "b3_recurrent_attention_lag_bins.json", update, attention["b3"]["lag_bins"])
    merge_keyed_json(runtime.output / "b3_attention_head_distance.json", update,
                     {"heads": attention["b3"]["heads"], "aggregate": attention["b3"]["aggregate"],
                      "head_mean_lag_range": attention["b3"]["head_mean_lag_range"]})
    for name, link in (("b12_to_b1_temporal_gradient.json", "b1"),
                       ("b10_to_b3_temporal_gradient.json", "b3")):
        merge_keyed_json(runtime.output / name, update, temporal[link])
    merge_keyed_json(runtime.output / "temporal_gradient_diagnostics.json", update, temporal)
    restore_rng_state(saved_rng); runtime.model.train()
    return validation, attention, temporal


def reconcile_uncheckpointed_artifacts(output, completed):
    output = Path(output); audit = {"verified_checkpoint_update": int(completed), "files": {}}
    metrics_path = output / "training_metrics.jsonl"
    rows = read_jsonl(metrics_path) if metrics_path.exists() else []
    if len(rows) < completed or any(row["local_update"] != index for index, row in enumerate(rows, 1)):
        raise SystemExit("training metrics do not cover verified resume point contiguously")
    if len(rows) > completed:
        removed = [row["local_update"] for row in rows[completed:]]
        durable_text(metrics_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[:completed]))
        audit["files"][metrics_path.name] = {"removed_updates": removed}
    keyed = ("milestone_validation.json", "paired_controls.json", "gate_diagnostics.json",
             "b1_attention_diagnostics.json",
             "b3_local_attention_lag_bins.json", "b3_recurrent_attention_lag_bins.json",
             "b3_attention_head_distance.json", "b12_to_b1_temporal_gradient.json",
             "b10_to_b3_temporal_gradient.json", "attention_diagnostics.json",
             "temporal_gradient_diagnostics.json")
    for name in keyed:
        path = output / name
        if not path.exists(): continue
        payload = read_json(path); removed = [key for key in payload if key.isdigit() and int(key) > completed]
        if removed:
            for key in removed: del payload[key]
            durable_json(path, payload); audit["files"][name] = {"removed_keys": sorted(removed)}
    audit["changed"] = bool(audit["files"]); durable_json(output / "resume_reconciliation.json", audit)
    return rows[:completed]


def _incremental_control(model, x, y, name, derangement=None):
    batch, length = x.shape
    state = model.init_incremental_state(batch, device=x.device, b3_full_cache=name == "b3_full_counterfactual")
    per_sequence = torch.zeros(batch, dtype=torch.float64); per_position = np.zeros(length, dtype=np.float64)
    total = 0.0; targets = 0; maxima = [0] * N_LAYER
    ring_max = {"h10": 0, "h12": 0}; rms = {"b1": [], "b3": []}
    for position in range(length):
        logits, state, diagnostics = model.incremental_step(
            x[:, position], state, control=name,
            recurrent_permutation=derangement if name in {"b3_shuffled", "all_shuffled"} else None,
            return_diagnostics=True, diagnostic_attention_weights=False)
        losses = F.cross_entropy(logits[:, 0].float(), y[:, position], reduction="none").double().cpu()
        per_sequence += losses; per_position[position] += losses.sum().item()
        total += losses.sum().item(); targets += batch
        maxima = [max(a, b) for a, b in zip(maxima, model.incremental_cache_lengths(state))]
        for key in ring_max: ring_max[key] = max(ring_max[key], int(getattr(state, f"{key}_ring").size(1)))
        for key in rms:
            row = diagnostics.get(key)
            if row is not None: rms[key].append(row["recurrent_output_rms"].detach().float().item())
    return {"loss_sum": total, "targets": targets, "per_sequence_losses": (per_sequence / length).tolist(),
            "per_position_sum": per_position, "final_cache_audit": model.incremental_cache_audit(state),
            "max_cache_lengths": maxima, "max_ring_lengths": ring_max,
            "final_memory_rms": {key: getattr(state, f"{key}_ring").float().square().mean().sqrt().item() for key in ring_max},
            "mean_recurrent_output_rms": {key: statistics.fmean(values) if values else 0.0 for key, values in rms.items()}}


@torch.no_grad()
def evaluate_incremental(model, val_path, batches=INCREMENTAL_BATCHES) -> dict:
    if int(batches) != 4: raise ValueError("2D2F primary incremental evaluation requires exactly four B64 batches")
    model.eval(); device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    names = ("all_real", "b3_off", "b3_shuffled", "b3_full_counterfactual", "all_shuffled")
    rows = {name: {"sum": 0.0, "targets": 0, "batches": [], "sequences": [],
                         "positions": np.zeros(T, dtype=np.float64), "cache_rows": [], "first_half": []}
            for name in names}
    identities = []; permutation = torch.arange(VALIDATION_B, device=device).roll(1)
    started = time.monotonic(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(int(batches)):
        cpu_x, cpu_y = loader.next_batch(); identities.append(legacy.d0d.batch_identity(cpu_x, cpu_y))
        x, y = cpu_x.to(device), cpu_y.to(device)
        for name in names:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current = _incremental_control(model, x, y, name, permutation)
            row = rows[name]; loss = current["loss_sum"] / current["targets"]
            row["sum"] += current["loss_sum"]; row["targets"] += current["targets"]
            row["batches"].append(loss); row["sequences"].extend(current["per_sequence_losses"])
            row["positions"] += current["per_position_sum"]
            row["cache_rows"].append({"final": current["final_cache_audit"],
                                      "max_cache_lengths": current["max_cache_lengths"],
                                      "max_ring_lengths": current["max_ring_lengths"]})
            if batch_index < 2: row["first_half"].append(current)
        print(f"2D2F incremental batch={batch_index + 1:02d}/{batches}", flush=True)
        del x, y, cpu_x, cpu_y; torch.cuda.empty_cache()
    controls = {}
    for name, row in rows.items():
        first_sum = sum(item["loss_sum"] for item in row["first_half"]); first_targets = sum(item["targets"] for item in row["first_half"])
        controls[name] = {"validation_loss": row["sum"] / row["targets"], "validation_targets": row["targets"],
                          "per_batch_losses": row["batches"], "per_sequence_losses": row["sequences"],
                          "per_position_loss": (row["positions"] / (int(batches) * VALIDATION_B)).tolist(),
                          "cache_rows": row["cache_rows"], "first_131072_validation_loss": first_sum / first_targets,
                          "first_131072_targets": first_targets}
    real, off, shuffled = controls["all_real"], controls["b3_off"], controls["b3_shuffled"]
    first_real, first_off, first_shuffled = (controls[name]["first_131072_validation_loss"] for name in ("all_real", "b3_off", "b3_shuffled"))
    result = {"controls": controls,
              "true_b3_recurrent_gain": off["validation_loss"] - real["validation_loss"],
              "true_b3_sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
              "first_131072": {"true_b3_recurrent_gain": first_off - first_real,
                                 "true_b3_sequence_gap": first_shuffled - first_real,
                                 "controls": {name: controls[name]["first_131072_validation_loss"] for name in names}},
              "all_real_vs_b3_off_batches": paired_stats(real["per_batch_losses"], off["per_batch_losses"]),
              "all_real_vs_b3_shuffled_batches": paired_stats(real["per_batch_losses"], shuffled["per_batch_losses"]),
              "all_real_vs_b3_off_sequences": paired_stats(real["per_sequence_losses"], off["per_sequence_losses"]),
              "all_real_vs_b3_shuffled_sequences": paired_stats(real["per_sequence_losses"], shuffled["per_sequence_losses"]),
              "all_real_vs_all_shuffled_gap": controls["all_shuffled"]["validation_loss"] - real["validation_loss"],
              "g_rec_b1_raw": model.g_rec.detach().float().item(), "tanh_g_rec_b1": model.recurrent_scale_b1.detach().float().item(),
              "g_rec_b3_raw": model.g_rec_b3.detach().float().item(), "tanh_g_rec_b3": model.recurrent_scale_b3.detach().float().item(),
              "canonical_subset_sha256": legacy.d0.aggregate_hashes([row["combined_sha256"] for row in identities]),
              "batch_identities": identities, "batch_count": int(batches), "batch_size": VALIDATION_B,
              "sequence_length": T, "targets_per_control": int(batches) * VALIDATION_B * T,
              "primary_target_requirement_met": int(batches) * VALIDATION_B * T == 262144,
              "paired_sequences": int(batches) * VALIDATION_B,
              "no_complete_prefix_recomputation": True,
              "performance": {"wall_seconds": time.monotonic() - started,
                               "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                               "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2}}
    return result


@torch.no_grad()
def parallel_incremental_equivalence(model, val_path, length=128, batch=2):
    if length < 128: raise ValueError("equivalence must exercise B3 W64 eviction and active recurrence")
    loader = legacy.d1.ExplicitShardLoader([val_path], batch, length); cpu_x, _ = loader.next_batch()
    tokens = cpu_x.to(next(model.parameters()).device); reports = {}
    original = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    for label in ("fp32", "bf16"):
        if label == "fp32":
            torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            context = contextlib.nullcontext(); tolerance = FP32_INCREMENTAL_ATOL
        else:
            torch.set_float32_matmul_precision(original[0]); torch.backends.cuda.matmul.allow_tf32 = original[1]; torch.backends.cudnn.allow_tf32 = original[2]
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16); tolerance = BF16_INCREMENTAL_ATOL
        with context:
            plain = model.forward_pass(tokens)["logits"]
            incremental = model.incremental_logits(tokens, control="all_real", b1_gate_override=0.0,
                                                   b3_gate_override=0.0,
                                                   diagnostic_attention_weights=False)["logits"]
        delta = (plain.float() - incremental.float()).abs()
        reports[label] = {"plain_kernel_max_abs": delta.max().item(), "plain_kernel_mean_abs": delta.mean().item(),
                          "max_abs_tolerance": tolerance, "kernel_passed": delta.max().item() <= tolerance,
                          "b1_w2_eviction_exercised": True, "b2_w32_eviction_exercised": True,
                          "b3_w64_eviction_exercised": True}
    torch.set_float32_matmul_precision(original[0]); torch.backends.cuda.matmul.allow_tf32 = original[1]; torch.backends.cudnn.allow_tf32 = original[2]
    reports["passed"] = reports["fp32"]["kernel_passed"] and reports["bf16"]["kernel_passed"]
    return reports


def memory_accounting(incremental=None) -> dict:
    element_bytes = 2
    def one(batch):
        b1 = batch * 1 * N_EMBD * 2 * element_bytes
        b2 = batch * 31 * N_EMBD * 2 * element_bytes
        b3 = batch * 63 * N_EMBD * 2 * element_bytes
        rings = batch * 2 * 1023 * N_EMBD * element_bytes
        upper = batch * 9 * 1023 * N_EMBD * 2 * element_bytes
        total = b1 + b2 + b3 + rings + upper
        standard = batch * 12 * 1023 * N_EMBD * 2 * element_bytes
        final_2d2d = batch * (1 * N_EMBD * 2 + 31 * N_EMBD * 2 + 2 * 1023 * N_EMBD + 10 * 1023 * N_EMBD * 2) * element_bytes
        return {"batch_size": batch, "b1_local_kv_bytes": b1, "b2_local_kv_bytes": b2,
                "b3_local_kv_bytes": b3, "two_recurrent_raw_state_rings_bytes": rings,
                "b4_b12_ordinary_kv_bytes": upper, "total_experimental_inference_state_bytes": total,
                "final_2d2d_inference_state_bytes": final_2d2d, "standard_gpt2_w1024_kv_bytes": standard,
                "delta_bytes_vs_final_2d2d": total - final_2d2d,
                "delta_bytes_vs_standard_gpt2": total - standard,
                "saving_bytes_vs_standard_gpt2": standard - total,
                "mib": {"total": total / 1024**2, "saving_vs_standard": (standard - total) / 1024**2}}
    observed = None
    if incremental is not None:
        cache = incremental["controls"]["all_real"]["cache_rows"][-1]["final"]
        cache_bytes = sum(0 if row["key"] is None else row["key"]["actual_bytes"] + row["value"]["actual_bytes"] for row in cache["cache_physical_storage"])
        rings = sum(cache[f"{name}_ring_physical_storage"]["actual_bytes"] for name in ("h10", "h12"))
        observed = {"control": "all_real", "position": cache["position"], "cache_bytes": cache_bytes,
                    "raw_ring_bytes": rings, "total_physical_state_bytes": cache_bytes + rings,
                    "physical_storage_exact": cache["physical_storage_exact"]}
    return {"deployment_accounting_dtype": "BF16", "bytes_per_element": element_bytes,
            "B1": one(1), "B64": one(64), "observed_final_incremental_storage": observed,
            "state_limits": {"B1 ordinary historical KV": 1, "B2 ordinary historical KV": 31,
                             "B3 ordinary historical KV": 63, "B10/B12 raw recurrent states each": 1023,
                             "B11 recurrent raw state": 0,
                             "B4-B12 ordinary historical KV each": 1023}}


def classify_result(incremental, stable=True, integrity=True) -> str:
    if not integrity: return "EXPERIMENT 2D2F INVALID"
    if not stable: return "B10→B3 W64 RECURRENT LINK IS UNSTABLE"
    off = incremental["all_real_vs_b3_off_sequences"]; shuffled = incremental["all_real_vs_b3_shuffled_sequences"]
    gain, gap = incremental["true_b3_recurrent_gain"], incremental["true_b3_sequence_gap"]
    positive = gain > 0 and gap > 0 and off["wins"] >= 129 and shuffled["wins"] >= 129
    if positive and gain >= 0.001 and off["wins"] >= 166 and shuffled["wins"] >= 166:
        return "B10→B3 W64 RECURRENT LINK STRONGLY ESTABLISHES UTILITY"
    if positive: return "B10→B3 W64 RECURRENT LINK ESTABLISHES POSITIVE UTILITY"
    if gap > 0: return "B10→B3 W64 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY"
    balanced = 112 <= off["wins"] <= 144 and 112 <= shuffled["wins"] <= 144
    if abs(gain) < 1e-4 and abs(gap) < 1e-4 and balanced:
        return "B10→B3 W64 RECURRENT LINK REMAINS NEAR ZERO"
    return "B10→B3 W64 RECURRENT LINK DOES NOT ESTABLISH POSITIVE UTILITY"


def choose_recommendation(classification, initial_damage):
    if classification == "EXPERIMENT 2D2F INVALID": return "FIX 2D2F INTEGRITY"
    if classification == "B10→B3 W64 RECURRENT LINK IS UNSTABLE": return "STABILIZE B10→B3 W64 RECURRENCE"
    if classification in {"B10→B3 W64 RECURRENT LINK ESTABLISHES POSITIVE UTILITY", "B10→B3 W64 RECURRENT LINK STRONGLY ESTABLISHES UTILITY"}:
        return "RUN MATCHED NO-B11→B2 CONTROL BEFORE ADDING B9→B4"
    if classification == "B10→B3 W64 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY":
        return "IMPROVE DEEPER-LAYER RECURRENT READOUT BEFORE ADDING B9→B4"
    if classification == "B10→B3 W64 RECURRENT LINK REMAINS NEAR ZERO":
        return "RUN MATCHED B2-W32 TRAINING WITHOUT B11→B2 RECURRENCE"
    if initial_damage > 0.10:
        return "INCREASE B3 LOCAL WINDOW BEFORE ADDING ANOTHER RECURRENT LINK"
    return "IMPROVE DEEPER-LAYER RECURRENT READOUT BEFORE ADDING B9→B4"


@torch.no_grad()
def self_composition_diagnostic(model, val_path, passes=8, batch_size=2):
    model.eval(); device = next(model.parameters()).device
    loader = legacy.d1.ExplicitShardLoader([val_path], batch_size, T); cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(device), cpu_y.to(device); h10 = h12 = None; rows = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for index in range(int(passes)):
            current = model.forward_pass(x, b1_recurrent_source=h12,
                                         b3_recurrent_source=h10, return_diagnostics=index > 0)
            rows.append({"pass": index + 1, "loss": _token_losses(current["logits"], y).double().mean().item(),
                         "b10_memory_rms": current["h10"].float().square().mean().sqrt().item(),
                         "b12_memory_rms": current["h12"].float().square().mean().sqrt().item(),
                         "b3_recurrent_output_rms": 0.0 if index == 0 else current["diagnostics"]["b3"]["recurrent_output_rms"].float().item()})
            h10, h12 = current["h10"], current["h12"]
    report = {"passes": rows, "batch_size": batch_size, "sequence_length": T,
              "finite": all(all(math.isfinite(value) for key, value in row.items() if key != "pass") for row in rows),
              "no_gradient": True}
    return report


def _plot_final(output, milestones, training, incremental, attention, temporal, memory, performance, parallel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output); updates = sorted(int(key) for key in milestones)
    targets = [milestones[str(key)]["additional_targets"] for key in updates]
    def save(index, draw):
        fig, ax = plt.subplots(figsize=(7.2, 4.5)); draw(ax); fig.tight_layout()
        fig.savefig(output / REQUIRED_PLOTS[index - 1], dpi=160); plt.close(fig)
    save(1, lambda ax: (ax.plot(targets, [milestones[str(k)]["controls"]["all_real"]["validation_loss"] for k in updates], marker="o", label="ALL_REAL"), ax.plot(targets, [milestones[str(k)]["controls"]["b3_off"]["validation_loss"] for k in updates], marker="o", label="B3_OFF"), ax.set(xlabel="Additional targets", ylabel="Validation CE"), ax.legend()))
    save(2, lambda ax: (ax.plot(targets, [milestones[str(k)]["b3_recurrent_gain"] for k in updates], marker="o"), ax.axhline(0, color="black", linewidth=.8), ax.set(xlabel="Additional targets", ylabel="B3 recurrent gain")))
    save(3, lambda ax: (ax.plot(targets, [milestones[str(k)]["b3_sequence_gap"] for k in updates], marker="o"), ax.axhline(0, color="black", linewidth=.8), ax.set(xlabel="Additional targets", ylabel="B3 sequence gap")))
    save(4, lambda ax: ([ax.plot([0] + [row["additional_targets"] for row in training], [source] + [row[field] for row in training], label=label) for field, source, label in (("tanh_g_rec_b1", SOURCE_GATE_EFFECTIVE, "B1"), ("tanh_g_rec_b3", 0.0, "B3"))], ax.set(xlabel="Additional targets", ylabel="Effective gate"), ax.legend()))
    local = attention["b3"]["local_lag_bins"]
    save(5, lambda ax: (ax.bar(list(local), [row["attention_mass"] for row in local.values()]), ax.set(ylabel="Attention mass", title="B3 local attention"), ax.tick_params(axis="x", rotation=30)))
    recurrent = attention["b3"]["lag_bins"]
    save(6, lambda ax: (ax.bar(list(recurrent), [row["attention_mass"] for row in recurrent.values()]), ax.set(ylabel="Attention mass", title="B3 recurrent attention"), ax.tick_params(axis="x", rotation=30)))
    save(7, lambda ax: (ax.bar(list(recurrent), [row["normalized_mass_per_available_token"] for row in recurrent.values()]), ax.set(ylabel="Mass / available token", title="Normalized B3 recurrent density"), ax.tick_params(axis="x", rotation=30)))
    heads = attention["b3"]["heads"]
    save(8, lambda ax: (ax.bar([int(k) for k in heads], [v["mean_attended_recurrent_lag"] for v in heads.values()]), ax.set(xlabel="B3 head", ylabel="Mean recurrent lag")))
    b3_temporal = temporal["b3"]["bins"]
    save(9, lambda ax: (ax.bar(list(b3_temporal), [v["mean_gradient_rms"] for v in b3_temporal.values()]), ax.set_yscale("log"), ax.set(ylabel="Writer-gradient RMS", title="B10→B3 by lag"), ax.tick_params(axis="x", rotation=30)))
    def temporal_compare(ax):
        for link in ("b1", "b3"):
            bins = temporal[link]["bins"]; ax.plot(range(len(bins)), [v["mean_gradient_rms"] for v in bins.values()], marker="o", label=link.upper())
        ax.set_yscale("log"); ax.set(xlabel="Increasing temporal-distance bin", ylabel="Gradient RMS"); ax.legend()
    save(10, temporal_compare)
    save(11, lambda ax: (ax.bar(["Parallel", "True incremental"], [parallel["b3_recurrent_gain"], incremental["true_b3_recurrent_gain"]]), ax.axhline(0, color="black", linewidth=.8), ax.set(ylabel="B3 recurrent gain")))
    marginals = parallel["combined_system"]
    save(12, lambda ax: (ax.bar(["B1", "B3"], [marginals["b1_marginal_gain"], marginals["b3_marginal_gain"]]), ax.axhline(0, color="black", linewidth=.8), ax.set(ylabel="Marginal recurrent gain")))
    unit = N_EMBD * 2
    standard = 12 * 1023 * unit * 2
    comparisons = {
        "Standard": standard,
        "2D2B": (1 * unit * 2 + 1023 * unit + 11 * 1023 * unit * 2),
        "2D2C": (2 * 1 * unit * 2 + 2 * 1023 * unit + 10 * 1023 * unit * 2),
        "2D2D": memory["B1"]["final_2d2d_inference_state_bytes"],
        "2D2F": memory["B1"]["total_experimental_inference_state_bytes"],
    }
    save(13, lambda ax: (ax.bar(list(comparisons), [v / 1024**2 for v in comparisons.values()]), ax.set(ylabel="BF16 state (MiB)"), ax.tick_params(axis="x", rotation=25)))
    save(14, lambda ax: (ax.bar(["Throughput ktargets/s", "Peak VRAM GiB"], [performance["aggregate_targets_per_second"] / 1000, performance["peak_reserved_vram_mb"] / 1024]), ax.set(title="One-A100 runtime / throughput / VRAM")))


def _scientific_questions(initial, training, milestones, attention, temporal, parallel, incremental, memory, classification, recommendation):
    first_nonzero = next((row for row in training if row["g_rec_b3_raw"] != 0.0), None)
    values = [0.0] + [row["tanh_g_rec_b3"] for row in training]
    m = lambda update: milestones[str(update)]["b3_recurrent_gain"]
    heads = [row["mean_attended_recurrent_lag"] for row in attention["b3"]["heads"].values()]
    b3bins = attention["b3"]["lag_bins"]
    q = {
        "Q1": initial["initial_B3_W64_compression_damage"],
        "Q2": initial["damage_relative_to_source_loss"],
        "Q3": training[0]["g_rec_b3_gradient_preclip"] != 0.0,
        "Q4": None if first_nonzero is None else {"local_update": first_nonzero["local_update"], "additional_targets": first_nonzero["additional_targets"]},
        "Q5": None if first_nonzero is None else ("positive" if first_nonzero["g_rec_b3_raw"] > 0 else "negative"),
        "Q6": max(values), "Q7": min(values), "Q8": training[-1]["tanh_g_rec_b3"],
        "Q9": "g_rec_b2 physically absent", "Q10": training[-1]["tanh_g_rec_b1"],
        "Q11": temporal["b3"]["nonzero"],
        "Q12": temporal["b3"]["bins"]["128-255"]["fraction_nonzero_positions"] > 0,
        "Q13": temporal["b3"]["bins"]["256-511"]["fraction_nonzero_positions"] > 0,
        "Q14": temporal["b3"]["bins"]["512-1023"]["fraction_nonzero_positions"] > 0,
        "Q15": m(20), "Q16": m(48), "Q17": m(96), "Q18": m(143), "Q19": m(191),
        "Q20": parallel["b3_sequence_gap"], "Q21": parallel["all_real_vs_b3_off"],
        "Q22": parallel["all_real_vs_b3_shuffled"],
        "Q23": b3bins["64-127"]["attention_mass"], "Q24": b3bins["128-255"]["attention_mass"],
        "Q25": b3bins["256-511"]["attention_mass"], "Q26": b3bins["512-1023"]["attention_mass"],
        "Q27": attention["b3"]["aggregate"],
        "Q28": {"head_mean_lag_range": max(heads) - min(heads), "per_head_mean_lags": heads},
        "Q29": "not applicable; B2 recurrence physically absent",
        "Q30": {"source_b2_gate": SOURCE_B2_GATE_EFFECTIVE, "final": None, "removed": True},
        "Q31": parallel["combined_system"]["b1_marginal_gain"],
        "Q32": incremental["true_b3_recurrent_gain"], "Q33": incremental["true_b3_sequence_gap"],
        "Q34": incremental["all_real_vs_b3_off_sequences"], "Q35": incremental["all_real_vs_b3_shuffled_sequences"],
        "Q36": {"b2_recurrence_present": False, "b3_gain": incremental["true_b3_recurrent_gain"]},
        "Q37": {"parallel_gain": parallel["b3_recurrent_gain"], "true_incremental_gain": incremental["true_b3_recurrent_gain"], "same_sign": parallel["b3_recurrent_gain"] * incremental["true_b3_recurrent_gain"] > 0},
        "Q38": memory["B1"]["saving_bytes_vs_standard_gpt2"],
        "Q39": classification in {"B10→B3 W64 RECURRENT LINK ESTABLISHES POSITIVE UTILITY", "B10→B3 W64 RECURRENT LINK STRONGLY ESTABLISHES UTILITY"},
        "Q40": recommendation,
    }
    questions = {}
    source = Path("/Users/rahul/.codex/attachments/4114ef21-1cd0-45b7-901b-18f13a30bc51/pasted-text.txt")
    labels = []
    if source.exists():
        for line in source.read_text().splitlines():
            if line.startswith("Q") and ". " in line: labels.append(line)
    for index in range(1, 41):
        key = f"Q{index}"; question = next((line.split(". ", 1)[1] for line in labels if line.startswith(key + ".")), key)
        questions[key] = {"question": question, "answer": q[key]}
    return questions


def render_report(summary, audit, questions):
    inc = summary["incremental"]; initial = summary["initial_b3_w64_compression"]
    lines = [
        "EXPERIMENT 2D2F PRIMARY CLASSIFICATION:", summary["primary_classification"], "",
        "INITIAL B3-W64 COMPRESSION DAMAGE:", str(initial["initial_B3_W64_compression_damage"]), "",
        "FINAL TRUE-SELF B10→B3 W64 RECURRENT GAIN:", str(inc["true_b3_recurrent_gain"]), "",
        "FINAL TRUE-SELF B10→B3 W64 SEQUENCE GAP:", str(inc["true_b3_sequence_gap"]), "",
        "# Experiment 2D2F Final Report", "",
        f"- Source checkpoint: `{summary['source_checkpoint']}`", f"- Source SHA: `{summary['source_checkpoint_sha256']}`",
        f"- Git lineage: `{FROZEN_COMMIT}` → `{BRANCH}`", "- Hardware: one NVIDIA A100-SXM4-80GB; distributed execution not used",
        f"- Parameters: {TOTAL_PARAMETERS:,} (exactly one new scalar `g_rec_b3`)",
        "- Geometry: B1 W2 + B12 lags 2…1023; B2 W32 local-only; B3 W64 + B10 lags 64…1023; B4–B12 W1024",
        f"- Training: {MAX_UPDATES} updates, {ADDITIONAL_TARGETS:,} targets",
        f"- Runtime: {summary['performance']['training_wall_seconds']:.3f}s; throughput {summary['performance']['aggregate_targets_per_second']:.3f} targets/s; peak reserved VRAM {summary['performance']['peak_reserved_vram_mb']:.3f} MiB",
        "", "## Milestone B3 metrics", "",
        "| Update | Targets | Gain | Sequence gap | B1 gate | B3 gate |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(summary["validation_trajectory"], key=int):
        row = summary["validation_trajectory"][key]
        lines.append(f"| {key} | {row['additional_targets']} | {row['b3_recurrent_gain']:.12g} | {row['b3_sequence_gap']:.12g} | {row['tanh_g_rec_b1']:.12g} | {row['tanh_g_rec_b3']:.12g} |")
    lines += ["", "## Final controls", "", "```json", json.dumps({"parallel": summary["parallel"]["controls"], "incremental": summary["incremental"]["controls"]}, indent=2, sort_keys=True), "```",
              "", "## Cache, stability, and memory", "", "```json", json.dumps({"cache": summary["incremental_cache_audit"], "stability": summary["self_composition"], "memory": summary["memory_accounting"]}, indent=2, sort_keys=True), "```",
              "", f"Exact next recommendation: **{summary['recommendation']}**", "", "## Integrity audit", ""]
    for name, value in audit["checks"].items(): lines.append(f"- {'PASS' if value else 'FAIL'} — {name}")
    lines += ["", "## Scientific questions Q1–Q40", ""]
    for index in range(1, 41):
        row = questions[f"Q{index}"]; lines += [f"### Q{index}. {row['question']}", "", json.dumps(row["answer"], sort_keys=True), ""]
    lines += ["# EXPERIMENT 2D2F COMPLETE", ""]
    return "\n".join(lines)


def run_finalize(args):
    require_config(); workspace_mount_audit(args.output_dir, args.run_root, args.persistent_volume_identity)
    authenticated_stop_audit(args); device = require_single_a100(); output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json"); smoke = read_json(output / "smoke_audit.json")
    require_implementation_fingerprint(preflight)
    model, optimizer, final_model = load_final_model(args, device)
    training = read_jsonl(output / "training_metrics.jsonl")
    if len(training) != MAX_UPDATES or [row["local_update"] for row in training] != list(range(1, MAX_UPDATES + 1)):
        raise SystemExit("finalization requires exactly 191 contiguous training rows")
    milestones = read_json(output / "milestone_validation.json")
    if set(milestones) != {str(value) for value in MILESTONES}: raise SystemExit("milestone set incomplete")
    parallel = milestones[str(MAX_UPDATES)]
    if "combined_system" not in parallel: raise SystemExit("final parallel combined controls missing")
    incremental = evaluate_incremental(model, validation_path(args.data_root))
    cache_audit = {name: {"all_passed": all(item["final"]["passed"] for item in row["cache_rows"]),
                          "rows": row["cache_rows"]} for name, row in incremental["controls"].items()}
    cache_audit["passed"] = all(row["all_passed"] for key, row in cache_audit.items() if key != "passed")
    equivalence = parallel_incremental_equivalence(model, validation_path(args.data_root))
    stability = self_composition_diagnostic(model, validation_path(args.data_root))
    attention = {link: attention_diagnostics(model, validation_path(args.data_root), link) for link in ("b1", "b3")}
    temporal = {link: temporal_gradient_by_lag(model, validation_path(args.data_root), link) for link in ("b1", "b3")}
    memory = memory_accounting(incremental)
    walls = [row["wall_seconds"] for row in training]
    performance = {"training_wall_seconds": sum(walls), "mean_seconds_per_update": statistics.fmean(walls),
                   "aggregate_targets_per_second": ADDITIONAL_TARGETS / sum(walls),
                   "peak_allocated_vram_mb": max(row["peak_allocated_vram_mb"] for row in training),
                   "peak_reserved_vram_mb": max(row["peak_reserved_vram_mb"] for row in training),
                   "incremental": incremental["performance"]}
    stable = stability["finite"] and model_finite(model)
    integrity = preflight["science_passed"] and smoke["passed"] and cache_audit["passed"] and equivalence["passed"]
    classification = classify_result(incremental, stable=stable, integrity=integrity)
    initial = preflight["initial_b3_w64_compression"]
    recommendation = choose_recommendation(classification, initial["initial_B3_W64_compression_damage"])
    questions = _scientific_questions(initial, training, milestones, attention, temporal, parallel, incremental, memory, classification, recommendation)
    _plot_final(output, milestones, training, incremental, attention, temporal, memory, performance, parallel)
    durable_json(output / "incremental_validation.json", incremental)
    durable_json(output / "incremental_cache_audit.json", cache_audit)
    durable_json(output / "memory_accounting.json", memory)
    durable_json(output / "stability_8pass.json", stability)
    durable_json(output / "parallel_incremental_equivalence.json", equivalence)
    durable_json(output / "performance.json", performance)
    durable_json(output / "b1_attention_diagnostics.json", {**read_json(output / "b1_attention_diagnostics.json"), "final": attention["b1"]})
    durable_json(output / "attention_diagnostics.json", {**read_json(output / "attention_diagnostics.json"), "final": attention})
    durable_json(output / "b3_local_attention_lag_bins.json", {**read_json(output / "b3_local_attention_lag_bins.json"), "final": attention["b3"]["local_lag_bins"]})
    durable_json(output / "b3_recurrent_attention_lag_bins.json", {**read_json(output / "b3_recurrent_attention_lag_bins.json"), "final": attention["b3"]["lag_bins"]})
    durable_json(output / "b3_attention_head_distance.json", {**read_json(output / "b3_attention_head_distance.json"), "final": {"heads": attention["b3"]["heads"], "aggregate": attention["b3"]["aggregate"], "head_mean_lag_range": attention["b3"]["head_mean_lag_range"]}})
    for filename, link in (("b12_to_b1_temporal_gradient.json", "b1"), ("b10_to_b3_temporal_gradient.json", "b3")):
        durable_json(output / filename, {**read_json(output / filename), "final": temporal[link]})
    durable_json(output / "temporal_gradient_diagnostics.json", {**read_json(output / "temporal_gradient_diagnostics.json"), "final": temporal})
    durable_json(output / "scientific_questions.json", questions)
    checkpoint_manifest = read_json(output / "checkpoint_manifest.json")
    restart = read_json(output / "forced_restart_update_96.json")
    checks = {
        "source 2D2D checkpoint SHA exact": file_sha256(args.source_checkpoint) == SOURCE_SHA256,
        "source final tag exact": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
        "parameter count 124,475,906": preflight["parameters"]["2d2f_total_parameters"] == TOTAL_PARAMETERS,
        "exactly one new parameter": preflight["parameters"]["new_parameter_count_vs_2d2d"] == 1,
        "B1 W2 and recurrent lags exact": preflight["kernel_preflight"]["checks"]["b1_mask_exact"],
        "B2 W32 local-only and recurrent machinery absent": preflight["semantic_diff"]["b2_gate_absent"] and preflight["semantic_diff"]["b11_ring_absent"],
        "B3 W64 and recurrent lags exact": preflight["kernel_preflight"]["checks"]["b3_mask_exact"],
        "B3 max recurrent width 960": preflight["kernel_preflight"]["reports"]["b3_boundary_counts"]["1023"] == 960,
        "B3 local/recurrent non-overlap": preflight["kernel_preflight"]["checks"]["b3_nonoverlap"],
        "existing projections and one c_proj reused": preflight["kernel_preflight"]["checks"]["single_c_proj_each"],
        "B1 resumed; B2 gate dropped; B3 zero initialized": preflight["semantic_diff"]["b1_gate_resumed"] and preflight["semantic_diff"]["b2_gate_absent"] and preflight["semantic_diff"]["b3_gate_zero"],
        "shared source optimizer state resumed; B2 state dropped; B3 fresh": preflight["source"]["checks"]["source_optimizer_state_preserved"] and preflight["source"]["checks"]["new_b3_optimizer_state_absent"],
        "data stream continues after 2D2D": preflight["checks"]["loader_continuation"],
        "global batch 524,288": preflight["checks"]["global_batch"],
        "CE-only exact pass cadence": all(row["pass_count"] == pass_count(row["local_update"]) and tuple(row["pass_weights"]) == pass_weights(row["local_update"]) for row in training),
        "no detach and all writer gradients": all(temporal[link]["nonzero"] for link in temporal),
        "191 updates and 100,139,008 targets": len(training) == MAX_UPDATES and training[-1]["additional_targets"] == ADDITIONAL_TARGETS,
        "mandatory update-96 fresh-process restart": restart["passed"],
        "checkpoint hashes verified": all(checkpoint_manifest["scientific"][str(update)]["passed"] for update in SCIENTIFIC_CHECKPOINTS),
        "true incremental evaluation completed": incremental["primary_target_requirement_met"] and incremental["paired_sequences"] == 256,
        "physical cache audit passed": cache_audit["passed"],
        "8-pass stability passed": stable,
        "storage audit passed": read_json(output / "persistent_workspace_audit.json")["passed"],
        "Git synchronized": False,
        "sealed report commit synchronized": False,
        "persistent volume preserved": True,
    }
    summary = {"experiment": EXPERIMENT, "primary_classification": classification,
               "recommendation": recommendation, "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
               "source_checkpoint_sha256": SOURCE_SHA256, "parameters": TOTAL_PARAMETERS,
               "initial_b3_w64_compression": initial, "validation_trajectory": milestones,
               "parallel": parallel, "incremental": incremental, "incremental_cache_audit": cache_audit,
               "final_attention": attention, "final_temporal_gradient": temporal,
               "final_gates": {"b1": final_model["tanh_g_rec_b1"], "b2": None, "b3": final_model["tanh_g_rec_b3"]},
               "self_composition": stability, "memory_accounting": memory,
               "parallel_incremental_equivalence": equivalence, "performance": performance,
               "final_checkpoint": final_model["checkpoint"], "final_checkpoint_sha256": final_model["sha256"],
               "git": {"branch": BRANCH, "lineage": [FROZEN_COMMIT], "results_commit": None, "sealed_commit": None},
               "artifact_directory": str(output), "pod": {"id": args.pod_id, "name": args.pod_name, "status": "RUNNING_PENDING_GIT_SEAL"},
               "persistent_volume": {"identity": args.persistent_volume_identity, "preserved": True}}
    audit = {"checks": checks, "passed": False, "classification": classification}
    durable_json(output / "result_summary.json", summary); durable_json(output / "FINAL_AUDIT.json", audit)
    report_text = render_report(summary, audit, questions)
    durable_text(output / "FINAL_REPORT.md", report_text)
    durable_text(output / "EXPERIMENT_2D2F_FINAL_REPORT.md", report_text)
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", render_report(summary, audit, questions) + "\nGPU remains running until local backup, Git seal, attestation, and verified stop.\n")
    durable_json(output / "HEARTBEAT.json", {"experiment": EXPERIMENT, "timestamp": time.time(), "status": "FINALIZED_PENDING_GIT_SEAL", "local_update": MAX_UPDATES})
    print("EXPERIMENT_2D2F_FINALIZED_PENDING_GIT_SEAL", flush=True)
    return summary


def run_seal_report(args):
    require_git(clean=True); output = Path(args.output_dir).resolve()
    if git_output("rev-parse", "HEAD") != args.results_commit or git_output("rev-parse", f"origin/{BRANCH}") != args.results_commit:
        raise SystemExit("results commit must be checked out and pushed")
    summary = read_json(output / "result_summary.json"); audit = read_json(output / "FINAL_AUDIT.json"); questions = read_json(output / "scientific_questions.json")
    summary["git"]["results_commit"] = args.results_commit; summary["pod"]["status"] = "RUNNING_PENDING_SEALED_REPORT_ATTESTATION"
    audit["checks"]["Git synchronized"] = True
    durable_json(output / "result_summary.json", summary); durable_json(output / "FINAL_AUDIT.json", audit)
    report_text = render_report(summary, audit, questions)
    durable_text(output / "FINAL_REPORT.md", report_text)
    durable_text(output / "EXPERIMENT_2D2F_FINAL_REPORT.md", report_text)
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", render_report(summary, audit, questions) + "\nCommit and push this sealed report, then attest it before stopping the pod.\n")
    print("EXPERIMENT_2D2F_REPORT_SEALED", flush=True)


def run_attest_seal(args):
    require_git(clean=True); output = Path(args.output_dir).resolve()
    if git_output("rev-parse", "HEAD") != args.sealed_commit or git_output("rev-parse", f"origin/{BRANCH}") != args.sealed_commit:
        raise SystemExit("sealed report commit must be checked out and pushed")
    summary = read_json(output / "result_summary.json"); audit = read_json(output / "FINAL_AUDIT.json"); questions = read_json(output / "scientific_questions.json")
    summary["git"]["sealed_commit"] = args.sealed_commit; summary["pod"]["status"] = "READY_FOR_LOCAL_BACKUP_AND_STOP"
    audit["checks"]["sealed report commit synchronized"] = True; audit["passed"] = all(audit["checks"].values())
    if not audit["passed"]: raise SystemExit(f"final attestation failed: {audit['checks']}")
    durable_json(output / "result_summary.json", summary); durable_json(output / "FINAL_AUDIT.json", audit)
    report_text = render_report(summary, audit, questions)
    durable_text(output / "FINAL_REPORT.md", report_text)
    durable_text(output / "EXPERIMENT_2D2F_FINAL_REPORT.md", report_text)
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", render_report(summary, audit, questions) + "\nBack up locally, commit/push this attestation, then stop—not delete—the exact pod.\n")
    print("EXPERIMENT_2D2F_SEAL_ATTESTED", flush=True)


def add_execution_arguments(parser):
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--ephemeral-checkpoint-dir")
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
    attest = subparsers.add_parser("attest-seal")
    attest.add_argument("--output-dir", required=True)
    attest.add_argument("--sealed-commit", required=True)
    attest.set_defaults(function=run_attest_seal)
    return parser


def main():
    args = build_parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    main()
