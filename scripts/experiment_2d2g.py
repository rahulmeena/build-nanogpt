#!/usr/bin/env python3
"""Experiment 2D2G: matched-age full-B2 control followed by B10->B3.

Stage A continues the exact final 2D2B model for 191 matched updates.  Stage B
then keeps B2 at W1024, introduces B3 W64 plus B10 recurrent memory, and runs
the next 191 matched updates.  Each stage is an independent scientific
checkpoint chain with a mandatory fresh-process boundary after update 96.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import gc
import hashlib
import inspect
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0 as d0  # noqa: E402
import experiment_2d1 as d1  # noqa: E402
import experiment_2d2a as utility  # noqa: E402
import experiment_2d2b as source_driver  # noqa: E402
from experiment_2d2g_core import (  # noqa: E402
    B1_LOCAL_WINDOW,
    B2_LOCAL_WINDOW,
    B3_LOCAL_WINDOW,
    B3_MAX_RECURRENT_ENTRIES,
    B3_RECURRENT_MIN_LAG,
    INCREMENTAL_CONTROLS,
    RECURRENT_MAX_LAG,
    RECURRENT_RING_CAPACITY,
    StageARecurrentKVGPT,
    StageBRecurrentKVGPT,
)


EXPERIMENT = "2D2G"
PROTOCOL = "master_parallel_2d2g_b2_full_b3_w64_v1"
BRANCH = "experiment-2d2g-b2-full-b3-w64"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d2g_b2_full_b3_w64.json"
OUTPUT_NAME = "experiment_2d2g_b2_full_b3_w64"
SOURCE_SHA256 = "8c39f47248e3a5f4dc69f5e8e97c8a1cd1bcdfa91154eba5804c448942075326"
SOURCE_COMMIT = "976b92927e698afd27d68eabc78db5a0b6714fef"
SOURCE_SCHEMA = "exp2d2b_full_b12_b1_recurrent_bank_checkpoint_v1"
STAGE_A_SCHEMA = "exp2d2g_stage_a_matched_age_checkpoint_v1"
STAGE_B_SCHEMA = "exp2d2g_stage_b_b3_w64_checkpoint_v1"

T = 1024
N_LAYER = 12
N_EMBD = 768
GLOBAL_TARGETS = 524_288
UPDATES_PER_STAGE = 191
TARGETS_PER_STAGE = 100_139_008
STAGE_A_PARAMETERS = 124_475_905
STAGE_B_PARAMETERS = 124_475_906
BASE_LR = 3e-5
GATE_LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
TWO_PASS_WEIGHTS = (0.25, 0.75)
THREE_PASS_WEIGHTS = (0.20, 0.40, 0.40)
MILESTONES = (0, 20, 48, 96, 143, 191)
VALIDATION_BATCHES = 20
VALIDATION_B = 64
INCREMENTAL_BATCHES = 4
SMOKE_UPDATES = 3
SEED = 2026_0221
VALIDATION_SHARD_SHA256 = "8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f"

EPHEMERAL_ROOT = Path("/tmp/parallel_2d2_ephemeral")
PERSISTENT_ROOT = Path("/workspace")
CHECKPOINT_PERSIST_LOCK = (
    PERSISTENT_ROOT / "parallel_2d2_master" / "locks" / "checkpoint_persist.lock"
)

SOURCE_NEXT_BATCH = "e1d96ca0106f21badeb0004025e80abc562509fb6299a63eb8662a3da3c17a52"
SOURCE_NEXT_STREAM = "fc01029471dfe8674e900dd3d1e20a34e235853d44c68ede1b67f5b1a61e44f0"
STAGE_A_UPDATE96_BATCH = "313c1cd86d1924914e395df1bddf98fcb9ab351592c014b3b1357ecbbab33c97"
STAGE_A_UPDATE96_STREAM = "6a870df86881ba3def08dbbe7bbb5554b959780d788d509c578558b52dd19f31"
STAGE_A_FINAL_BATCH = "39808d08e7e15e9f160e32ba838fd28839067827095b023d4f475b30df392086"
STAGE_A_FINAL_STREAM = "110e232ab330611a8d23cddc6e914c8a2d912fd8191873ae67488b0d52f48daa"
STAGE_B_UPDATE96_BATCH = "1fce24bac89d50b68868e689d246ea08e784d738db29bb68c5166d0339347109"
STAGE_B_UPDATE96_STREAM = "8dfb91e0d6022f092a67045540ce5bb9e90d9554a4d8610afa123251ff7f8703"
STAGE_B_FINAL_BATCH = "91fa2cae4e6e52cfddd2b470175ec704f0548b447f02861917ec548736fe18e7"
STAGE_B_FINAL_STREAM = "4da6fed71755e523030a2d8e9e7cc96d19691a8c9b3ac8c490426bafe3d44e82"

B3_RECURRENT_LAG_BINS = (
    ("64-127", 64, 127),
    ("128-255", 128, 255),
    ("256-511", 256, 511),
    ("512-1023", 512, 1023),
)
IMPLEMENTATION_FILES = (
    "configs/exp2d2g_b2_full_b3_w64.json",
    "scripts/experiment_2d2g.py",
    "scripts/experiment_2d2g_core.py",
    "tests/test_experiment_2d2g_core.py",
    "tests/test_experiment_2d2g_driver.py",
)

REQUIRED_TRAINING_ARTIFACTS = (
    "FINAL_REPORT.md",
    "FINAL_AUDIT.json",
    "result_summary.json",
    "source_manifest.json",
    "architecture_manifest.json",
    "parameter_manifest.json",
    "training_metrics.jsonl",
    "milestone_validation.json",
    "paired_controls.json",
    "gate_diagnostics.json",
    "attention_diagnostics.json",
    "temporal_gradient_diagnostics.json",
    "incremental_validation.json",
    "incremental_cache_audit.json",
    "memory_accounting.json",
    "stability_8pass.json",
    "performance.json",
    "checkpoint_manifest.json",
    "commands_and_runtime.json",
    "storage_cleanup_manifest.json",
    "HEARTBEAT.json",
    "UNATTENDED_FINAL_HANDOFF.md",
)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(array.tobytes())
    return digest.hexdigest()


def aggregate_hashes(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(str(value)))
    return digest.hexdigest()


def read_json(path: Path | str):
    with open(path) as handle:
        return json.load(handle)


def durable_json(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def durable_text(path: Path | str, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def append_jsonl(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def is_relative_to(path: Path | str, parent: Path | str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def checkpoint_sidecar_audit(path: Path | str, expected_sha: str | None = None) -> dict:
    path = Path(path).resolve()
    sha_path = path.with_suffix(path.suffix + ".sha256")
    verification_path = path.with_suffix(path.suffix + ".verification.json")
    observed_sha = file_sha256(path)
    recorded_sha = sha_path.read_text().split()[0] if sha_path.is_file() else None
    verification = read_json(verification_path) if verification_path.is_file() else {}
    checks = {
        "sha_sidecar_present": sha_path.is_file(),
        "verification_sidecar_present": verification_path.is_file(),
        "sha_sidecar_matches": recorded_sha == observed_sha,
        "expected_sha_matches": expected_sha is None or observed_sha == expected_sha,
        "verification_passed": verification.get("passed") is True,
    }
    return {
        "checkpoint": str(path),
        "sha256": observed_sha,
        "sha_sidecar": str(sha_path),
        "verification_sidecar": str(verification_path),
        "checks": checks,
        "passed": all(checks.values()),
    }


def append_command_runtime(output: Path | str, row: dict) -> None:
    path = Path(output) / "commands_and_runtime.json"
    payload = read_json(path) if path.exists() else {"commands": []}
    payload.setdefault("commands", []).append(row)
    durable_json(path, payload)


def record_cleanup(output: Path | str, actions: list[dict]) -> None:
    path = Path(output) / "storage_cleanup_manifest.json"
    payload = read_json(path) if path.exists() else {
        "scientific_source_removed": False,
        "cleanup_actions": [],
    }
    payload.setdefault("cleanup_actions", []).extend(actions)
    payload["scientific_source_removed"] = False
    durable_json(path, payload)


def required_artifact_inventory(output: Path | str) -> dict:
    output = Path(output).resolve()
    rows = {}
    for name in REQUIRED_TRAINING_ARTIFACTS:
        path = output / name
        rows[name] = {
            "path": str(path),
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }
    return {
        "required": list(REQUIRED_TRAINING_ARTIFACTS),
        "artifacts": rows,
        "passed": all(row["present"] and row["bytes"] > 0 for row in rows.values()),
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean: bool = True) -> None:
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"2D2G requires branch {BRANCH}")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], cwd=REPO_ROOT
    ):
        raise SystemExit("frozen 2D2B commit is not an ancestor")
    if clean and git_output("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked 2D2G implementation files must be clean")


def implementation_fingerprint() -> dict:
    files = {name: file_sha256(REPO_ROOT / name) for name in IMPLEMENTATION_FILES}
    return {"files": files, "aggregate_sha256": aggregate_hashes(files[name] for name in sorted(files))}


def require_fingerprint(output_dir) -> None:
    preflight = read_json(Path(output_dir) / "preflight_audit.json")
    if preflight.get("implementation_fingerprint") != implementation_fingerprint():
        raise SystemExit("2D2G implementation changed after preflight")


def require_config() -> dict:
    config = read_json(CONFIG_PATH)
    checks = {
        "experiment": config.get("experiment") == EXPERIMENT,
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "source": config["source_2d2b"]["sha256"] == SOURCE_SHA256,
        "stage_a_parameters": config["stage_a"]["architecture"]["parameters"] == STAGE_A_PARAMETERS,
        "stage_b_parameters": config["stage_b"]["architecture"]["parameters"] == STAGE_B_PARAMETERS,
        "budgets": config["stage_a"]["updates"] == UPDATES_PER_STAGE
        and config["stage_b"]["updates"] == UPDATES_PER_STAGE
        and config["stage_a"]["targets"] == TARGETS_PER_STAGE
        and config["stage_b"]["targets"] == TARGETS_PER_STAGE,
        "no_b11": config["stage_b"]["architecture"]["b11_to_b2_recurrence"] is False
        and config["stage_b"]["architecture"]["b11_raw_ring_present"] is False,
        "geometry": config["stage_b"]["architecture"]["b2_local_window"] == 1024
        and config["stage_b"]["architecture"]["b3_local_window"] == 64
        and config["stage_b"]["architecture"]["b3_max_recurrent_entries"] == 960,
        "checkpoint_policy": config["checkpoint_policy"]["stage_b_smoke"]
        == "ephemeral_disposable_strict_reopen_then_delete"
        and config["checkpoint_policy"]["stage_a_update_96"] == "ephemeral"
        and config["checkpoint_policy"]["stage_a_final"]
        == "ephemeral_until_stage_b_sealed"
        and config["checkpoint_policy"]["stage_b_update_96"] == "ephemeral"
        and config["checkpoint_policy"]["stage_b_final"]
        == "persistent_after_checkpoint_lock"
        and config["checkpoint_policy"]["serialize_persistent_final_with"]
        == str(CHECKPOINT_PERSIST_LOCK),
    }
    if not all(checks.values()):
        raise SystemExit(f"2D2G config mismatch: {checks}")
    return config


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def require_assigned_a100() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("worker must see exactly one CUDA device via CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    if "A100" not in torch.cuda.get_device_name(device):
        raise SystemExit("assigned device is not an A100")
    if torch.cuda.get_device_properties(device).total_memory < 75 * 1024**3:
        raise SystemExit("assigned A100 is not an 80GB-class device")
    return device


def pass_count(local_update: int) -> int:
    return 3 if int(local_update) % 32 == 0 else 2


def pass_weights(local_update: int):
    return THREE_PASS_WEIGHTS if pass_count(local_update) == 3 else TWO_PASS_WEIGHTS


def batch_identity(x: torch.Tensor, y: torch.Tensor) -> dict:
    return {
        "input_sha256": tensor_sha256(x),
        "target_sha256": tensor_sha256(y),
        "combined_sha256": tensor_sha256(x, y),
    }


def next_global_batch_hash(loader, accumulation: int) -> str:
    clone = loader.clone()
    identities = []
    for _ in range(int(accumulation)):
        x, y = clone.next_batch()
        identities.append(d0.batch_payload_hash(x, y))
    return aggregate_hashes(identities)


def global_batch_stream_hash(loader, accumulation: int) -> str:
    clone = loader.clone()
    x_hash = hashlib.sha256()
    y_hash = hashlib.sha256()
    for _ in range(int(accumulation)):
        x, y = clone.next_batch()
        x_hash.update(x.contiguous().numpy().tobytes())
        y_hash.update(y.contiguous().numpy().tobytes())
    return hashlib.sha256((x_hash.hexdigest() + y_hash.hexdigest()).encode()).hexdigest()


def validation_path(data_root: Path | str) -> Path:
    rows = sorted(Path(data_root).glob("*val*.npy"))
    if len(rows) != 1:
        raise SystemExit(f"expected one validation shard, found {len(rows)}")
    path = rows[0].resolve()
    if file_sha256(path) != VALIDATION_SHARD_SHA256:
        raise SystemExit("canonical validation shard SHA mismatch")
    return path


def training_shards(data_root: Path | str):
    rows = sorted(Path(data_root).glob("*train*.npy"))
    if not rows:
        raise SystemExit("no training shards")
    return rows


def instantiate_base(device: torch.device):
    symbols = d0.support.load_training_symbols()
    base = symbols["GPT"](d0.model_config(symbols))
    return symbols, base.to(device)


def configure_stage_a_optimizer(model, device_type="cuda"):
    base_decay, base_nodecay = [], []
    for name, parameter in model.named_parameters():
        if name == "g_rec":
            continue
        (base_decay if parameter.dim() >= 2 else base_nodecay).append(parameter)
    groups = [
        {"name": "base_decay", "params": base_decay, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
        {"name": "base_nodecay", "params": base_nodecay, "lr": BASE_LR, "weight_decay": 0.0},
        {"name": "gate", "params": [model.g_rec], "lr": GATE_LR, "weight_decay": 0.0},
    ]
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
    return torch.optim.AdamW(groups, betas=BETAS, eps=ADAM_EPS, fused=fused)


def configure_stage_b_optimizer(model, device_type="cuda"):
    base_decay, base_nodecay = [], []
    for name, parameter in model.named_parameters():
        if name in {"g_rec", "g_rec_b3"}:
            continue
        (base_decay if parameter.dim() >= 2 else base_nodecay).append(parameter)
    groups = [
        {"name": "base_decay", "params": base_decay, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
        {"name": "base_nodecay", "params": base_nodecay, "lr": BASE_LR, "weight_decay": 0.0},
        {"name": "gate", "params": [model.g_rec], "lr": GATE_LR, "weight_decay": 0.0},
        {"name": "b3_gate", "params": [model.g_rec_b3], "lr": GATE_LR, "weight_decay": 0.0},
    ]
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
    return torch.optim.AdamW(groups, betas=BETAS, eps=ADAM_EPS, fused=fused)


def optimizer_finite(optimizer) -> bool:
    return all(
        not torch.is_tensor(value) or bool(torch.isfinite(value).all())
        for state in optimizer.state.values()
        for value in state.values()
    )


def model_finite(model) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())


def load_2d2b_source(path: Path | str, device: torch.device, restore_rng: bool = False):
    path = Path(path).resolve()
    sidecars = checkpoint_sidecar_audit(path, SOURCE_SHA256)
    if not sidecars["passed"]:
        raise SystemExit(f"2D2B source checkpoint/sidecar mismatch: {sidecars}")
    payload = d0.torch_load(path, mmap=False)
    if payload.get("schema") != SOURCE_SCHEMA:
        raise SystemExit(f"2D2B source schema mismatch: {payload.get('schema')}")
    _, base = instantiate_base(device)
    model = StageARecurrentKVGPT(base)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = configure_stage_a_optimizer(model, device.type)
    optimizer.load_state_dict(payload["optimizer"])
    loader = d1.ExplicitShardLoader(
        payload["loader_state"]["shards"],
        payload["loader_state"]["batch_size"],
        payload["loader_state"]["sequence_length"],
        state=payload["loader_state"],
    )
    accumulation = int(payload["metadata"]["gradient_accumulation"])
    observed_batch = next_global_batch_hash(loader, accumulation)
    observed_stream = global_batch_stream_hash(loader, accumulation)
    checks = {
        "parameters": sum(p.numel() for p in model.parameters()) == STAGE_A_PARAMETERS,
        "next_batch": observed_batch == SOURCE_NEXT_BATCH,
        "next_stream": observed_stream == SOURCE_NEXT_STREAM,
        "model_finite": model_finite(model),
        "optimizer_finite": optimizer_finite(optimizer),
        "weight_tying": model.base.transformer.wte.weight is model.base.lm_head.weight,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D2B strict reopen failed: {checks}")
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return model, optimizer, loader, payload, {
        "checkpoint": str(path),
        "sha256": SOURCE_SHA256,
        "sidecar_audit": sidecars,
        "checks": checks,
        "next_global_batch_sha256": observed_batch,
        "next_global_batch_stream_sha256": observed_stream,
    }


def transplant_stage_a_to_b(stage_a_model, stage_a_optimizer, device: torch.device):
    model = StageBRecurrentKVGPT(stage_a_model.base).to(device)
    model.g_rec = stage_a_model.g_rec
    optimizer = configure_stage_b_optimizer(model, device.type)
    shared = set(model.parameters()) - {model.g_rec_b3}
    source_parameters = {p for group in stage_a_optimizer.param_groups for p in group["params"]}
    if shared != source_parameters:
        raise SystemExit("Stage A/B shared optimizer parameter identity mismatch")
    for parameter in shared:
        if parameter in stage_a_optimizer.state:
            optimizer.state[parameter] = copy.deepcopy(stage_a_optimizer.state[parameter])
    source_groups = {group["name"]: group for group in stage_a_optimizer.param_groups}
    for group in optimizer.param_groups:
        if group["name"] == "b3_gate":
            continue
        source = source_groups[group["name"]]
        for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "capturable", "differentiable"):
            if key in source:
                group[key] = source[key]

    def state_value_equal(left, right):
        if torch.is_tensor(left) or torch.is_tensor(right):
            return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
        return left == right

    optimizer_state_exact = True
    for parameter in shared:
        left = stage_a_optimizer.state.get(parameter, {})
        right = optimizer.state.get(parameter, {})
        if set(left) != set(right) or any(
            not state_value_equal(left[key], right[key]) for key in left
        ):
            optimizer_state_exact = False
            break
    optimizer_groups_exact = True
    for name, source in source_groups.items():
        destination = next(group for group in optimizer.param_groups if group["name"] == name)
        if set(source["params"]) != set(destination["params"]):
            optimizer_groups_exact = False
            break
        for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "capturable", "differentiable"):
            if key in source and source[key] != destination.get(key):
                optimizer_groups_exact = False
                break
    checks = {
        "parameters": sum(p.numel() for p in model.parameters()) == STAGE_B_PARAMETERS,
        "no_b2_gate": not hasattr(model, "g_rec_b2"),
        "new_b3_zero": model.g_rec_b3.detach().float().item() == 0.0,
        "new_b3_state_absent": model.g_rec_b3 not in optimizer.state,
        "source_state_entries_preserved": len(optimizer.state) == len(stage_a_optimizer.state),
        "optimizer_state_exact": optimizer_state_exact,
        "optimizer_groups_exact": optimizer_groups_exact,
        "weight_tying": model.base.transformer.wte.weight is model.base.lm_head.weight,
    }
    if not all(checks.values()):
        raise SystemExit(f"Stage A to B transplant failed: {checks}")
    return model, optimizer, checks


def stage_architecture(stage: str) -> dict:
    if stage == "a":
        return {
            "stage": "2D2G-A",
            "b1": {"local_window": 2, "source": "B12", "recurrent_lags": [2, 1023]},
            "b2": {"local_window": 1024, "recurrent": False},
            "b3_b12_local_window": 1024,
            "parameters": STAGE_A_PARAMETERS,
        }
    return {
        "stage": "2D2G-B",
        "b1": {"local_window": 2, "source": "B12", "recurrent_lags": [2, 1023]},
        "b2": {"local_window": 1024, "recurrent": False, "b11_ring": False},
        "b3": {
            "local_window": B3_LOCAL_WINDOW,
            "source": "B10 post-MLP residual before B11",
            "recurrent_lags": [B3_RECURRENT_MIN_LAG, RECURRENT_MAX_LAG],
            "maximum_recurrent_entries": B3_MAX_RECURRENT_ENTRIES,
            "gate": "tanh(g_rec_b3)",
        },
        "b4_b12_local_window": 1024,
        "parameters": STAGE_B_PARAMETERS,
    }


def checkpoint_payload(stage, model, optimizer, loader, completed, accumulation, metadata):
    next_batch = next_global_batch_hash(loader, accumulation)
    next_stream = global_batch_stream_hash(loader, accumulation)
    return {
        "schema": STAGE_A_SCHEMA if stage == "a" else STAGE_B_SCHEMA,
        "stage": stage,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loader_state": loader.state_dict(),
        "rng_state": capture_rng_state(),
        "completed_local_updates": int(completed),
        "processed_stage_targets": int(completed) * GLOBAL_TARGETS,
        "targets_per_update": GLOBAL_TARGETS,
        "gradient_accumulation": int(accumulation),
        "next_global_batch_sha256": next_batch,
        "next_global_batch_stream_sha256": next_stream,
        "architecture_manifest": stage_architecture(stage),
        "metadata": metadata,
        "git_commit": git_output("rev-parse", "HEAD"),
        "saved_process_id": os.getpid(),
        "saved_at": time.time(),
    }


def expected_cursor(stage: str, completed: int):
    table = {
        ("a", 96): (STAGE_A_UPDATE96_BATCH, STAGE_A_UPDATE96_STREAM),
        ("a", 191): (STAGE_A_FINAL_BATCH, STAGE_A_FINAL_STREAM),
        ("b", 96): (STAGE_B_UPDATE96_BATCH, STAGE_B_UPDATE96_STREAM),
        ("b", 191): (STAGE_B_FINAL_BATCH, STAGE_B_FINAL_STREAM),
    }
    return table.get((stage, int(completed)))


def save_checkpoint(path, stage, model, optimizer, loader, completed, accumulation, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        stage, model, optimizer, loader, completed, accumulation, metadata
    )
    expected = expected_cursor(stage, completed)
    if expected and (
        payload["next_global_batch_sha256"], payload["next_global_batch_stream_sha256"]
    ) != expected:
        raise SystemExit(
            f"Stage {stage.upper()} update {completed} matched-data cursor mismatch"
        )
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    sha = file_sha256(path)
    strict = strict_reopen_checkpoint(path, stage, completed, metadata)
    if not strict["passed"]:
        raise SystemExit(f"strict checkpoint reopen failed: {strict}")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), strict)
    return {
        "checkpoint": str(path.resolve()),
        "sha256": sha,
        "bytes": path.stat().st_size,
        "next_global_batch_sha256": payload["next_global_batch_sha256"],
        "next_global_batch_stream_sha256": payload["next_global_batch_stream_sha256"],
        "strict_reopen": strict,
    }


def strict_reopen_checkpoint(path, stage, completed, metadata):
    payload = d0.torch_load(Path(path), mmap=False)
    expected_schema = STAGE_A_SCHEMA if stage == "a" else STAGE_B_SCHEMA
    model_keys = list(payload.get("model", {}))
    reference = next(iter(payload["model"].values()))
    device = reference.device
    _, base = instantiate_base(device)
    reopened_model = (
        StageARecurrentKVGPT(base)
        if stage == "a"
        else StageBRecurrentKVGPT(base)
    )
    reopened_model.load_state_dict(payload["model"], strict=True)
    reopened_optimizer = (
        configure_stage_a_optimizer(reopened_model, device.type)
        if stage == "a"
        else configure_stage_b_optimizer(reopened_model, device.type)
    )
    reopened_optimizer.load_state_dict(payload["optimizer"])
    reopened_loader = d1.ExplicitShardLoader(
        payload["loader_state"]["shards"],
        payload["loader_state"]["batch_size"],
        payload["loader_state"]["sequence_length"],
        state=payload["loader_state"],
    )
    checks = {
        "schema": payload.get("schema") == expected_schema,
        "stage": payload.get("stage") == stage,
        "updates": payload.get("completed_local_updates") == int(completed),
        "targets": payload.get("processed_stage_targets") == int(completed) * GLOBAL_TARGETS,
        "metadata": payload.get("metadata") == metadata,
        "architecture": payload.get("architecture_manifest") == stage_architecture(stage),
        "model_keys": ("g_rec_b3" in model_keys) == (stage == "b"),
        "no_b2_gate": "g_rec_b2" not in model_keys,
        "rng": set(payload.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "strict_model_load": True,
        "strict_optimizer_load": True,
        "model_finite": model_finite(reopened_model),
        "optimizer_finite": optimizer_finite(reopened_optimizer),
        "loader_next_batch": next_global_batch_hash(
            reopened_loader, payload["gradient_accumulation"]
        ) == payload["next_global_batch_sha256"],
        "loader_next_stream": global_batch_stream_hash(
            reopened_loader, payload["gradient_accumulation"]
        ) == payload["next_global_batch_stream_sha256"],
    }
    expected = expected_cursor(stage, completed)
    if expected:
        checks["matched_cursor"] = (
            payload.get("next_global_batch_sha256"),
            payload.get("next_global_batch_stream_sha256"),
        ) == expected
    checks["passed"] = all(checks.values())
    del reopened_model, reopened_optimizer, reopened_loader, base, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checks


def loaded_stage_checkpoint_audit(
    path: Path | str,
    stage: str,
    model,
    optimizer,
    loader,
    payload: dict,
) -> dict:
    expected_schema = STAGE_A_SCHEMA if stage == "a" else STAGE_B_SCHEMA
    completed = int(payload.get("completed_local_updates", -1))
    expected = expected_cursor(stage, completed)
    model_keys = set(payload.get("model", {}))
    checks = {
        "sidecars": checkpoint_sidecar_audit(path)["passed"],
        "schema": payload.get("schema") == expected_schema,
        "stage": payload.get("stage") == stage,
        "targets": payload.get("processed_stage_targets") == completed * GLOBAL_TARGETS,
        "targets_per_update": payload.get("targets_per_update") == GLOBAL_TARGETS,
        "architecture": payload.get("architecture_manifest") == stage_architecture(stage),
        "model_keys": ("g_rec_b3" in model_keys) == (stage == "b"),
        "no_b2_gate": "g_rec_b2" not in model_keys and not hasattr(model, "g_rec_b2"),
        "parameters": sum(parameter.numel() for parameter in model.parameters())
        == (STAGE_A_PARAMETERS if stage == "a" else STAGE_B_PARAMETERS),
        "model_finite": model_finite(model),
        "optimizer_finite": optimizer_finite(optimizer),
        "next_batch_reproduced": next_global_batch_hash(
            loader, payload["gradient_accumulation"]
        )
        == payload.get("next_global_batch_sha256"),
        "next_stream_reproduced": global_batch_stream_hash(
            loader, payload["gradient_accumulation"]
        )
        == payload.get("next_global_batch_stream_sha256"),
        "rng_complete": set(payload.get("rng_state", {}))
        == {"python", "numpy", "torch_cpu", "torch_cuda"},
    }
    if expected is not None:
        checks["preregistered_cursor"] = (
            payload.get("next_global_batch_sha256"),
            payload.get("next_global_batch_stream_sha256"),
        ) == expected
    return {"checks": checks, "passed": all(checks.values())}


def load_stage_a_checkpoint(path, device, restore_rng=False):
    path = Path(path).resolve()
    sidecars = checkpoint_sidecar_audit(path)
    if not sidecars["passed"]:
        raise SystemExit(f"Stage A checkpoint sidecar mismatch: {sidecars}")
    payload = d0.torch_load(path, mmap=False)
    if payload.get("schema") != STAGE_A_SCHEMA:
        raise SystemExit("not a 2D2G-A checkpoint")
    _, base = instantiate_base(device)
    model = StageARecurrentKVGPT(base)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = configure_stage_a_optimizer(model, device.type)
    optimizer.load_state_dict(payload["optimizer"])
    loader = d1.ExplicitShardLoader(
        payload["loader_state"]["shards"],
        payload["loader_state"]["batch_size"],
        payload["loader_state"]["sequence_length"],
        state=payload["loader_state"],
    )
    audit = loaded_stage_checkpoint_audit(path, "a", model, optimizer, loader, payload)
    if not audit["passed"]:
        raise SystemExit(f"Stage A strict load failed: {audit}")
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return model, optimizer, loader, payload


def load_stage_b_checkpoint(path, device, restore_rng=False):
    path = Path(path).resolve()
    sidecars = checkpoint_sidecar_audit(path)
    if not sidecars["passed"]:
        raise SystemExit(f"Stage B checkpoint sidecar mismatch: {sidecars}")
    payload = d0.torch_load(path, mmap=False)
    if payload.get("schema") != STAGE_B_SCHEMA:
        raise SystemExit("not a 2D2G-B checkpoint")
    _, base = instantiate_base(device)
    model = StageBRecurrentKVGPT(base)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = configure_stage_b_optimizer(model, device.type)
    optimizer.load_state_dict(payload["optimizer"])
    loader = d1.ExplicitShardLoader(
        payload["loader_state"]["shards"],
        payload["loader_state"]["batch_size"],
        payload["loader_state"]["sequence_length"],
        state=payload["loader_state"],
    )
    audit = loaded_stage_checkpoint_audit(path, "b", model, optimizer, loader, payload)
    if not audit["passed"]:
        raise SystemExit(f"Stage B strict load failed: {audit}")
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return model, optimizer, loader, payload


def persist_checkpoint(local_path, persistent_dir, lock_path) -> dict:
    local_path = Path(local_path).resolve()
    persistent_dir = Path(persistent_dir).resolve()
    persistent_dir.mkdir(parents=True, exist_ok=True)
    lock_path = Path(lock_path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    destination = persistent_dir / local_path.name

    local_sidecars = checkpoint_sidecar_audit(local_path)
    if not local_sidecars["passed"]:
        raise SystemExit(f"local final checkpoint is not strictly verified: {local_sidecars}")
    local_sha = local_sidecars["sha256"]
    persistent_sha = None
    reused_existing = False
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if destination.exists():
                persistent_sha = file_sha256(destination)
                if persistent_sha != local_sha:
                    raise SystemExit(
                        "refusing to overwrite a different persistent scientific checkpoint"
                    )
                reused_existing = True
            else:
                temporary = destination.with_suffix(
                    destination.suffix + f".tmp.{os.getpid()}"
                )
                shutil.copy2(local_path, temporary)
                temporary_sha = file_sha256(temporary)
                if temporary_sha != local_sha:
                    temporary.unlink(missing_ok=True)
                    raise SystemExit("temporary persistent checkpoint copy SHA mismatch")
                os.replace(temporary, destination)
                persistent_sha = file_sha256(destination)
            for suffix in (".sha256", ".verification.json"):
                source_sidecar = local_path.with_suffix(local_path.suffix + suffix)
                destination_sidecar = destination.with_suffix(destination.suffix + suffix)
                if destination_sidecar.exists() and file_sha256(
                    destination_sidecar
                ) != file_sha256(source_sidecar):
                    raise SystemExit(
                        f"refusing to overwrite different persistent sidecar {destination_sidecar}"
                    )
                if not destination_sidecar.exists():
                    temporary_sidecar = destination_sidecar.with_suffix(
                        destination_sidecar.suffix + f".tmp.{os.getpid()}"
                    )
                    shutil.copy2(source_sidecar, temporary_sidecar)
                    os.replace(temporary_sidecar, destination_sidecar)
            persistent_sidecars = checkpoint_sidecar_audit(destination, local_sha)
            if persistent_sha != local_sha or not persistent_sidecars["passed"]:
                raise SystemExit("persistent checkpoint copy/sidecar verification failed")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    report = {
        "local": str(local_path),
        "persistent": str(destination),
        "local_sha256": local_sha,
        "persistent_sha256": persistent_sha,
        "bytes": destination.stat().st_size,
        "lock": str(lock_path),
        "local_sidecar_audit": local_sidecars,
        "persistent_sidecar_audit": persistent_sidecars,
        "persistent_sha_verified_while_lock_held": True,
        "reused_existing_exact_checkpoint": reused_existing,
        "passed": local_sha == persistent_sha and persistent_sidecars["passed"],
    }
    if not report["passed"]:
        raise SystemExit("persistent checkpoint copy SHA mismatch")
    return report


def validate_final_persistence_paths(local_path, persistent_dir, lock_path) -> dict:
    local_path = Path(local_path).resolve()
    persistent_dir = Path(persistent_dir).resolve()
    lock_path = Path(lock_path).resolve()
    checks = {
        "local_checkpoint_is_ephemeral": is_relative_to(local_path, EPHEMERAL_ROOT),
        "persistent_directory_is_workspace": is_relative_to(
            persistent_dir, PERSISTENT_ROOT
        ),
        "persistent_directory_not_ephemeral": not is_relative_to(
            persistent_dir, EPHEMERAL_ROOT
        ),
        "shared_lock_exact": lock_path == CHECKPOINT_PERSIST_LOCK.resolve(),
    }
    return {
        "local_checkpoint": str(local_path),
        "persistent_directory": str(persistent_dir),
        "lock_path": str(lock_path),
        "checks": checks,
        "passed": all(checks.values()),
    }


def require_ephemeral_checkpoint_dir(path: Path | str) -> Path:
    path = Path(path).resolve()
    if not is_relative_to(path, EPHEMERAL_ROOT):
        raise SystemExit(
            f"disposable/update checkpoints must be under node-local {EPHEMERAL_ROOT}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def training_metadata(args, stage, accumulation):
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "stage": stage,
        "branch": BRANCH,
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "assigned_physical_cuda_device": 1,
        "targets_per_update": GLOBAL_TARGETS,
        "sequence_length": T,
        "gradient_accumulation": int(accumulation),
        "two_pass_weights": list(TWO_PASS_WEIGHTS),
        "three_pass_weights": list(THREE_PASS_WEIGHTS),
        "three_pass_every": 32,
        "fresh_process_restart_update": 96,
    }


def choose_microbatch(source_loader_state, device, stage="a"):
    # The finalized sources used B32/A16.  Keep it when it fits; callers may
    # explicitly lower this via --micro-batch after a disposable preflight.
    del device, stage
    return int(source_loader_state["batch_size"])


def loader_at_cursor(loader_state, micro_batch):
    translated = copy.deepcopy(loader_state)
    translated["batch_size"] = int(micro_batch)
    return d1.ExplicitShardLoader(
        translated["shards"], int(micro_batch), T, state=translated
    )


def gradient_groups(model, stage):
    excluded = {"g_rec"} if stage == "a" else {"g_rec", "g_rec_b3"}
    base = [p.grad for n, p in model.named_parameters() if n not in excluded and p.grad is not None]
    groups = {"base": base, "gate": [] if model.g_rec.grad is None else [model.g_rec.grad]}
    if stage == "b":
        groups["b3_gate"] = [] if model.g_rec_b3.grad is None else [model.g_rec_b3.grad]
    report = {}
    for name, values in groups.items():
        squared = sum(v.float().square().sum() for v in values) if values else torch.tensor(0.0)
        report[name] = {
            "tensors": len(values),
            "norm": squared.sqrt().item(),
            "finite": all(bool(torch.isfinite(v).all()) for v in values),
            "nonzero": bool(values) and bool(squared.gt(0).item()),
        }
    return report


def train_update(model, optimizer, loader, accumulation, stage, update, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = pass_count(update)
    weights = pass_weights(update)
    totals = [0.0] * count
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(accumulation):
        cpu_x, cpu_y = loader.next_batch()
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model.forward_multi_pass(
                x,
                targets=y,
                num_passes=count,
                activation_checkpointing=True,
                return_diagnostics=False,
            )
            loss = result["loss"] / accumulation
        loss.backward()
        for index, value in enumerate(result["pass_losses"]):
            totals[index] += value.detach().float().item()
        del x, y, cpu_x, cpu_y, result, loss
    groups = gradient_groups(model, stage)
    if not all(row["finite"] and row["nonzero"] for row in groups.values()):
        raise SystemExit(f"required gradient group missing: {groups}")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad_norm):
        raise SystemExit("nonfinite gradient norm")
    optimizer.step()
    if not model_finite(model) or not optimizer_finite(optimizer):
        raise SystemExit("nonfinite model/optimizer after update")
    elapsed = time.monotonic() - started
    row = {
        "stage": stage,
        "local_update": int(update),
        "processed_stage_targets": int(update) * GLOBAL_TARGETS,
        "pass_count": count,
        "pass_weights": list(weights),
        "pass_losses": [value / accumulation for value in totals],
        "gradient_groups": groups,
        "gradient_norm_before_clip": grad_norm.detach().float().item(),
        "g_rec_b1_raw": model.g_rec.detach().float().item(),
        "tanh_g_rec_b1": model.g_rec.detach().float().tanh().item(),
        "wall_seconds": elapsed,
        "targets_per_second": GLOBAL_TARGETS / elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    if stage == "b":
        row.update(
            g_rec_b3_raw=model.g_rec_b3.detach().float().item(),
            tanh_g_rec_b3=model.recurrent_scale_b3.detach().float().item(),
        )
    return row


def require_segment_boundary(start_update: int, end_update: int, resumed_pid=None) -> None:
    if not start_update < end_update <= UPDATES_PER_STAGE:
        raise SystemExit("invalid training segment")
    if start_update < 96 < end_update:
        raise SystemExit("a training process may not cross mandatory update 96")
    if start_update == 96 and resumed_pid == os.getpid():
        raise SystemExit("update-96 continuation must use a fresh process")
    if start_update not in {0, 96}:
        raise SystemExit("result-bearing segments may start only at 0 or 96")


def write_heartbeat(output, stage, update, metrics, checkpoint=None):
    durable_json(
        Path(output) / "HEARTBEAT.json",
        {
            "experiment": EXPERIMENT,
            "stage": stage,
            "phase": "training",
            "local_update": int(update),
            "targets": int(update) * GLOBAL_TARGETS,
            "last_loss": metrics["pass_losses"][-1],
            "g_rec_b1": metrics["tanh_g_rec_b1"],
            "g_rec_b3": metrics.get("tanh_g_rec_b3"),
            "pid": os.getpid(),
            "checkpoint": checkpoint,
            "timestamp": time.time(),
        },
    )


def initialize_stage_a(args, device):
    if args.resume:
        model, optimizer, loader, payload = load_stage_a_checkpoint(
            args.resume, device, restore_rng=True
        )
        completed = int(payload["completed_local_updates"])
        accumulation = int(payload["gradient_accumulation"])
        metadata = payload["metadata"]
        saved_pid = payload["saved_process_id"]
    else:
        if not args.source_checkpoint:
            raise SystemExit("new Stage A segment requires --source-checkpoint")
        model, optimizer, source_loader, payload, _ = load_2d2b_source(
            args.source_checkpoint, device, restore_rng=False
        )
        micro_batch = int(args.micro_batch or choose_microbatch(payload["loader_state"], device, "a"))
        if GLOBAL_TARGETS % (micro_batch * T):
            raise SystemExit("microbatch does not divide logical global batch")
        accumulation = GLOBAL_TARGETS // (micro_batch * T)
        loader = loader_at_cursor(payload["loader_state"], micro_batch)
        if next_global_batch_hash(loader, accumulation) != SOURCE_NEXT_BATCH:
            raise SystemExit("Stage A translated loader changed first logical batch")
        restore_rng_state(payload["rng_state"])
        completed = 0
        metadata = training_metadata(args, "a", accumulation)
        metadata["immutable_source_checkpoint_sha256"] = SOURCE_SHA256
        saved_pid = None
        del source_loader
    return model, optimizer, loader, completed, accumulation, metadata, saved_pid


def initialize_stage_b(args, device):
    if args.resume:
        model, optimizer, loader, payload = load_stage_b_checkpoint(
            args.resume, device, restore_rng=True
        )
        completed = int(payload["completed_local_updates"])
        accumulation = int(payload["gradient_accumulation"])
        metadata = payload["metadata"]
        saved_pid = payload["saved_process_id"]
    else:
        if not args.stage_a_checkpoint:
            raise SystemExit("new Stage B segment requires --stage-a-checkpoint")
        stage_a, stage_a_optimizer, stage_a_loader, payload = load_stage_a_checkpoint(
            args.stage_a_checkpoint, device, restore_rng=False
        )
        if int(payload["completed_local_updates"]) != UPDATES_PER_STAGE:
            raise SystemExit("Stage B requires exact final Stage A checkpoint")
        if payload["next_global_batch_sha256"] != STAGE_A_FINAL_BATCH:
            raise SystemExit("Stage A final is not at matched Stage B cursor")
        model, optimizer, transplant = transplant_stage_a_to_b(
            stage_a, stage_a_optimizer, device
        )
        micro_batch = int(args.micro_batch or payload["loader_state"]["batch_size"])
        if GLOBAL_TARGETS % (micro_batch * T):
            raise SystemExit("microbatch does not divide logical global batch")
        accumulation = GLOBAL_TARGETS // (micro_batch * T)
        loader = loader_at_cursor(payload["loader_state"], micro_batch)
        if next_global_batch_hash(loader, accumulation) != STAGE_A_FINAL_BATCH:
            raise SystemExit("Stage B translated loader changed first logical batch")
        restore_rng_state(payload["rng_state"])
        completed = 0
        metadata = training_metadata(args, "b", accumulation)
        metadata["immutable_stage_a_checkpoint_sha256"] = file_sha256(
            args.stage_a_checkpoint
        )
        metadata["stage_a_completed_local_updates"] = int(
            payload["completed_local_updates"]
        )
        metadata["stage_a_next_global_batch_sha256"] = payload[
            "next_global_batch_sha256"
        ]
        metadata["stage_a_next_global_batch_stream_sha256"] = payload[
            "next_global_batch_stream_sha256"
        ]
        metadata["optimizer_transplant_checks"] = transplant
        saved_pid = None
        del stage_a, stage_a_optimizer, stage_a_loader
    return model, optimizer, loader, completed, accumulation, metadata, saved_pid


def run_train_stage(args, stage):
    require_git(clean=False)
    require_config()
    device = require_assigned_a100()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_fingerprint(output)
    preflight = read_json(output / "preflight_audit.json")
    if not preflight.get("authorized"):
        raise SystemExit("result training requires passing preflight")
    if stage == "b":
        smoke = read_json(output / "smoke_audit.json")
        if not smoke.get("passed"):
            raise SystemExit("Stage B training requires passing disposable smoke")
    checkpoint_dir = require_ephemeral_checkpoint_dir(args.checkpoint_dir)
    if stage == "a":
        runtime = initialize_stage_a(args, device)
    else:
        runtime = initialize_stage_b(args, device)
    model, optimizer, loader, completed, accumulation, metadata, saved_pid = runtime
    if stage == "b":
        smoke = read_json(output / "smoke_audit.json")
        stage_a_sha = metadata.get("immutable_stage_a_checkpoint_sha256")
        if smoke.get("immutable_stage_a_sha256") != stage_a_sha:
            raise SystemExit("Stage B smoke and scientific Stage A checkpoint differ")
        transition_checks = {
            "stage_a_update_191": metadata.get("stage_a_completed_local_updates")
            == UPDATES_PER_STAGE,
            "stage_a_batch_cursor": metadata.get("stage_a_next_global_batch_sha256")
            == STAGE_A_FINAL_BATCH,
            "stage_a_stream_cursor": metadata.get(
                "stage_a_next_global_batch_stream_sha256"
            )
            == STAGE_A_FINAL_STREAM,
            "optimizer_transplant": all(
                metadata.get("optimizer_transplant_checks", {}).values()
            ),
            "smoke_same_stage_a_sha": smoke.get("immutable_stage_a_sha256")
            == stage_a_sha,
        }
        transition = {
            "immutable_stage_a_checkpoint_sha256": stage_a_sha,
            "checks": transition_checks,
            "passed": all(transition_checks.values()),
        }
        durable_json(output / "stage_b_transition_audit.json", transition)
        if not transition["passed"]:
            raise SystemExit(f"Stage B transition audit failed: {transition}")
    require_segment_boundary(completed, int(args.end_update), saved_pid)
    if completed == 96:
        restart = {
                "checkpoint_process_id": saved_pid,
                "resumed_process_id": os.getpid(),
                "fresh_process": saved_pid != os.getpid(),
                "next_global_batch_sha256": next_global_batch_hash(loader, accumulation),
                "expected_next_global_batch_sha256": expected_cursor(stage, 96)[0],
                "next_global_batch_stream_sha256": global_batch_stream_hash(
                    loader, accumulation
                ),
                "expected_next_global_batch_stream_sha256": expected_cursor(stage, 96)[1],
                "passed": saved_pid != os.getpid()
                and next_global_batch_hash(loader, accumulation)
                == expected_cursor(stage, 96)[0]
                and global_batch_stream_hash(loader, accumulation)
                == expected_cursor(stage, 96)[1],
            }
        durable_json(
            output / f"stage_{stage}_forced_restart_update_96.json",
            restart,
        )
        if not restart["passed"]:
            raise SystemExit(f"Stage {stage.upper()} forced-restart audit failed")
    metrics_path = output / f"stage_{stage}_training_metrics.jsonl"
    if completed == 0:
        durable_text(metrics_path, "")
        if stage == "b":
            durable_text(output / "training_metrics.jsonl", "")
    else:
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        if len(rows) != completed or rows[-1]["local_update"] != completed:
            raise SystemExit("training metrics do not match resume checkpoint")
        if stage == "b":
            generic = [
                json.loads(line)
                for line in (output / "training_metrics.jsonl").read_text().splitlines()
                if line
            ]
            if len(generic) != completed or generic[-1]["local_update"] != completed:
                raise SystemExit("generic training metrics do not match Stage B resume")
    started = time.monotonic()
    for update in range(completed + 1, int(args.end_update) + 1):
        row = train_update(model, optimizer, loader, accumulation, stage, update, device)
        append_jsonl(metrics_path, row)
        if stage == "b":
            append_jsonl(output / "training_metrics.jsonl", row)
        if stage == "b" and update in MILESTONES:
            saved_rng = capture_rng_state()
            validation = evaluate_parallel(model, validation_path(args.data_root))
            validation.update(
                local_update=update,
                processed_stage_targets=update * GLOBAL_TARGETS,
            )
            merge_keyed(output / "milestone_validation.json", update, validation)
            merge_keyed(
                output / "gate_diagnostics.json",
                update,
                {
                    "g_rec_b1_raw": validation["g_rec_b1_raw"],
                    "tanh_g_rec_b1": validation["tanh_g_rec_b1"],
                    "g_rec_b3_raw": validation["g_rec_b3_raw"],
                    "tanh_g_rec_b3": validation["tanh_g_rec_b3"],
                },
            )
            restore_rng_state(saved_rng)
            model.train()
        write_heartbeat(output, stage, update, row)
        print(
            f"2D2G-{stage.upper()} update={update:03d} "
            f"loss={row['pass_losses'][-1]:.7f} "
            f"targets={update * GLOBAL_TARGETS:,}",
            flush=True,
        )
    final_update = int(args.end_update)
    checkpoint = checkpoint_dir / f"stage_{stage}_scientific_update_{final_update:04d}.pt"
    verification = save_checkpoint(
        checkpoint,
        stage,
        model,
        optimizer,
        loader,
        final_update,
        accumulation,
        metadata,
    )
    manifest_path = output / "checkpoint_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"stage_a": {}, "stage_b": {}}
    manifest[f"stage_{stage}"][str(final_update)] = verification
    durable_json(manifest_path, manifest)
    match_path = output / f"stage_{stage}_data_match.json"
    match = read_json(match_path) if match_path.exists() else {}
    expected = expected_cursor(stage, final_update)
    match[f"update_{final_update}"] = {
        "observed_next_global_batch_sha256": verification["next_global_batch_sha256"],
        "observed_next_global_batch_stream_sha256": verification["next_global_batch_stream_sha256"],
        "expected": list(expected),
        "exact": (
            verification["next_global_batch_sha256"],
            verification["next_global_batch_stream_sha256"],
        ) == expected,
    }
    if stage == "b":
        match["pending_stage_a"] = False
    if final_update == UPDATES_PER_STAGE:
        match["passed"] = match[f"update_{final_update}"]["exact"]
    durable_json(match_path, match)
    write_heartbeat(output, stage, final_update, row, verification["checkpoint"])
    runtime_path = output / "commands_and_runtime.json"
    commands = read_json(runtime_path) if runtime_path.exists() else {"commands": []}
    commands["commands"].append(
        {
            "command": " ".join(sys.argv),
            "stage": stage,
            "start_update": completed + 1,
            "end_update": final_update,
            "pid": os.getpid(),
            "wall_seconds": time.monotonic() - started,
        }
    )
    durable_json(runtime_path, commands)
    if final_update == 96:
        durable_json(
            output / f"stage_{stage}_restart_required_update_96.json",
            {
                "checkpoint": verification,
                "saved_process_id": os.getpid(),
                "status": "MANDATORY_FRESH_PROCESS_REQUIRED",
            },
        )
    print(f"EXPERIMENT_2D2G_STAGE_{stage.upper()}_SEGMENT_COMPLETE", flush=True)
    return verification


def run_train_a(args):
    return run_train_stage(args, "a")


def run_train_b(args):
    return run_train_stage(args, "b")


def run_preflight(args):
    require_git(clean=True)
    config = require_config()
    device = require_assigned_a100()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_model, source_optimizer, source_loader, payload, source = load_2d2b_source(
        args.source_checkpoint, device, restore_rng=False
    )
    stage_b, stage_b_optimizer, transplant = transplant_stage_a_to_b(
        source_model, source_optimizer, device
    )
    short = torch.randint(0, 128, (2, 72), device=device)
    with torch.no_grad():
        first = stage_b.forward_pass(short)
        zero = stage_b.forward_pass(
            short,
            b1_recurrent_source=first["h12"],
            b3_recurrent_source=first["h10"],
            b3_gate_override=0.0,
        )["logits"]
        absent = stage_b.forward_pass(
            short, b1_recurrent_source=first["h12"], b3_gate_override=0.0
        )["logits"]
        cache = stage_b.incremental_logits(
            short,
            control="real",
            b1_gate_override=0.0,
            b3_gate_override=0.0,
        )
    causality = b3_causality_audit(stage_b, short.size(1), device)
    val = validation_path(args.data_root)
    data = {
        "training_shards": [str(path.resolve()) for path in training_shards(args.data_root)],
        "validation_shard": str(val),
        "validation_sha256": file_sha256(val),
    }
    checks = {
        "source": all(source["checks"].values()),
        "source_sidecars": source["sidecar_audit"]["passed"],
        "stage_a_parameters": sum(p.numel() for p in source_model.parameters()) == STAGE_A_PARAMETERS,
        "stage_b_parameters": sum(p.numel() for p in stage_b.parameters()) == STAGE_B_PARAMETERS,
        "transplant": all(transplant.values()),
        "b3_gate_zero_identity": torch.equal(zero, absent),
        "no_b11_gate": not hasattr(stage_b, "g_rec_b2"),
        "no_b11_ring": not hasattr(cache["state"], "h11_ring"),
        "cache": cache["cache_audit"]["passed"],
        "causality": causality["passed"],
        "data": bool(data["training_shards"]),
    }
    audit = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "command": " ".join(sys.argv),
        "source": source,
        "stage_a_architecture": stage_architecture("a"),
        "stage_b_architecture": stage_architecture("b"),
        "transplant": transplant,
        "causality": causality,
        "data": data,
        "config": config,
        "implementation_fingerprint": implementation_fingerprint(),
        "checks": checks,
        "passed": all(checks.values()),
        "authorized": all(checks.values()),
    }
    durable_json(output / "preflight_audit.json", audit)
    durable_json(output / "source_manifest.json", source)
    durable_json(output / "architecture_manifest.json", {
        "stage_a": stage_architecture("a"), "stage_b": stage_architecture("b")
    })
    durable_json(output / "parameter_manifest.json", {
        "stage_a_parameters": STAGE_A_PARAMETERS,
        "stage_b_parameters": STAGE_B_PARAMETERS,
        "new_stage_b_parameters": ["g_rec_b3"],
        "no_g_rec_b2": True,
        "passed": checks["stage_a_parameters"] and checks["stage_b_parameters"],
    })
    durable_json(output / "stage_a_data_match.json", {
        "reference": "2D2D", "source_batch": SOURCE_NEXT_BATCH,
        "source_stream": SOURCE_NEXT_STREAM,
        "passed": source["checks"]["next_batch"] and source["checks"]["next_stream"]
    })
    durable_json(output / "stage_b_data_match.json", {
        "reference": "2D2E", "expected_start_batch": STAGE_A_FINAL_BATCH,
        "expected_start_stream": STAGE_A_FINAL_STREAM, "pending_stage_a": True
    })
    durable_json(output / "storage_cleanup_manifest.json", {
        "scientific_source_removed": False, "cleanup_actions": []
    })
    durable_json(output / "checkpoint_manifest.json", {"stage_a": {}, "stage_b": {}})
    durable_json(output / "commands_and_runtime.json", {"commands": [{"command": " ".join(sys.argv), "kind": "preflight"}]})
    if not audit["passed"]:
        raise SystemExit(f"2D2G preflight failed: {checks}")
    print("EXPERIMENT_2D2G_PREFLIGHT_PASS", flush=True)
    return audit


def b3_causality_audit(model, length: int, device: torch.device) -> dict:
    length = int(length)
    recurrent = model.b3_recurrent_mask(length, length, device)
    local = model.b3_local_mask(length, device)
    query = torch.arange(length, device=device).view(length, 1)
    source = torch.arange(length, device=device).view(1, length)
    lag = query - source
    expected_recurrent = (lag >= B3_RECURRENT_MIN_LAG) & (
        lag <= RECURRENT_MAX_LAG
    )
    expected_local = (lag >= 0) & (lag < B3_LOCAL_WINDOW)
    union_expected = (lag >= 0) & (lag <= RECURRENT_MAX_LAG)
    selected_lags = lag[recurrent]
    checks = {
        "recurrent_mask_exact": torch.equal(recurrent, expected_recurrent),
        "local_mask_exact": torch.equal(local, expected_local),
        "no_native_local_overlap": not bool((recurrent & local).any()),
        "no_future_access": not bool((recurrent & source.gt(query)).any()),
        "minimum_lag_64": bool(selected_lags.numel())
        and int(selected_lags.min().item()) == B3_RECURRENT_MIN_LAG,
        "maximum_lag_at_most_1023": not bool(
            (selected_lags > RECURRENT_MAX_LAG).any()
        ),
        "full_nonoverlapping_temporal_coverage": torch.equal(
            recurrent | local, union_expected
        ),
    }
    return {
        "length": length,
        "eligibility": "max(0,t-1023) <= j <= t-64",
        "selected_entries": int(recurrent.sum().item()),
        "checks": checks,
        "passed": all(checks.values()),
    }


@torch.no_grad()
def future_leakage_audit(model, tokens: torch.Tensor) -> dict:
    model.eval()
    reference = model.forward_multi_pass(tokens, num_passes=2)["logits"]
    future = tokens.clone()
    future[0, -1] = (future[0, -1] + 1) % int(model.config.vocab_size)
    future_result = model.forward_multi_pass(future, num_passes=2)["logits"]
    other_row = tokens.clone()
    other_row[1] = (other_row[1] + 3) % int(model.config.vocab_size)
    other_row_result = model.forward_multi_pass(other_row, num_passes=2)["logits"]
    checks = {
        "future_token_cannot_change_past_logits": torch.equal(
            reference[0, :-1], future_result[0, :-1]
        ),
        "other_row_cannot_change_reference_row": torch.equal(
            reference[0], other_row_result[0]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def b3_writer_gradient_audit(model, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.train()
    first = model.forward_pass(x, targets=y)
    source = first["h10"]
    second = model.forward_pass(
        x,
        targets=y,
        b1_recurrent_source=first["h12"].detach(),
        b3_recurrent_source=source,
    )
    gradient = torch.autograd.grad(second["loss"], source, retain_graph=False)[0]
    gradient = gradient.detach().float()
    length = gradient.size(1)
    eligible_end = max(0, length - B3_RECURRENT_MIN_LAG)
    eligible = gradient[:, :eligible_end]
    forbidden = gradient[:, eligible_end:]
    by_lag = {}
    for name, low, high in B3_RECURRENT_LAG_BINS:
        start = max(0, length - high - 1)
        end = max(0, length - low)
        values = gradient[:, start:end]
        rms = values.square().mean(-1).sqrt() if values.numel() else gradient.new_zeros((0,))
        by_lag[name] = {
            "source_position_interval": [start, end],
            "mean_gradient_rms": rms.mean().item() if rms.numel() else 0.0,
            "max_gradient_rms": rms.max().item() if rms.numel() else 0.0,
            "fraction_nonzero": (rms > 0).float().mean().item() if rms.numel() else 0.0,
            "finite": bool(torch.isfinite(rms).all()),
        }
    gate = model.recurrent_scale_b3.detach().float().item()
    checks = {
        "gate_open": gate != 0.0,
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "eligible_writer_gradient_nonzero": bool(eligible.count_nonzero())
        if eligible.numel()
        else False,
        "ineligible_last_64_exact_zero": not bool(forbidden.count_nonzero()),
        "all_lag_bins_nonzero": all(
            row["fraction_nonzero"] > 0 for row in by_lag.values()
        ),
    }
    return {
        "source": "B10 post-MLP residual before B11",
        "b1_source_detached_for_b3_path_isolation": True,
        "actual_tanh_g_rec_b3": gate,
        "eligible_source_positions": [0, eligible_end],
        "ineligible_source_positions": [eligible_end, length],
        "lag_bins": by_lag,
        "checks": checks,
        "attached": checks["eligible_writer_gradient_nonzero"],
        "passed": all(checks.values()),
    }


def run_smoke_b(args):
    started = time.monotonic()
    require_git(clean=False)
    require_config()
    device = require_assigned_a100()
    output = Path(args.output_dir).resolve()
    require_fingerprint(output)
    checkpoint_dir = require_ephemeral_checkpoint_dir(args.checkpoint_dir)
    stage_a, stage_a_optimizer, _, payload = load_stage_a_checkpoint(
        args.stage_a_checkpoint, device, restore_rng=False
    )
    if payload["completed_local_updates"] != 191:
        raise SystemExit("smoke requires final Stage A")
    model, optimizer, transplant = transplant_stage_a_to_b(
        stage_a, stage_a_optimizer, device
    )
    source_sha_before = file_sha256(args.stage_a_checkpoint)
    saved_rng = capture_rng_state()
    milestone_zero = evaluate_parallel(model, validation_path(args.data_root))
    milestone_zero.update(local_update=0, processed_stage_targets=0)
    merge_keyed(output / "milestone_validation.json", 0, milestone_zero)
    merge_keyed(
        output / "gate_diagnostics.json",
        0,
        {
            "g_rec_b1_raw": milestone_zero["g_rec_b1_raw"],
            "tanh_g_rec_b1": milestone_zero["tanh_g_rec_b1"],
            "g_rec_b3_raw": milestone_zero["g_rec_b3_raw"],
            "tanh_g_rec_b3": milestone_zero["tanh_g_rec_b3"],
        },
    )
    restore_rng_state(saved_rng)
    smoke_micro_batch = int(args.micro_batch or payload["loader_state"]["batch_size"])
    if GLOBAL_TARGETS % (smoke_micro_batch * T):
        raise SystemExit("smoke microbatch does not divide logical global batch")
    smoke_accumulation = GLOBAL_TARGETS // (smoke_micro_batch * T)
    loader = loader_at_cursor(payload["loader_state"], smoke_micro_batch)
    restore_rng_state(payload["rng_state"])
    rows = []
    for update in range(1, SMOKE_UPDATES + 1):
        before = model.g_rec_b3.detach().float().item()
        row = train_update(
            model, optimizer, loader, smoke_accumulation, "b", update, device
        )
        after = model.g_rec_b3.detach().float().item()
        rows.append({**row, "g_rec_b3_before": before, "g_rec_b3_after": after})
    cache_x, cache_y = loader.clone().next_batch()
    cache = model.incremental_logits(cache_x[:, :72].to(device), control="real")["cache_audit"]

    audit_x = cache_x[:2].to(device)
    audit_y = cache_y[:2].to(device)
    causality = b3_causality_audit(model, audit_x.size(1), device)
    leakage = future_leakage_audit(model, audit_x[:, :72])
    writer = b3_writer_gradient_audit(model, audit_x, audit_y)

    smoke_checkpoint = checkpoint_dir / (
        f"stage_b_disposable_smoke_update_{SMOKE_UPDATES:04d}_pid_{os.getpid()}.pt"
    )
    smoke_metadata = training_metadata(args, "b", smoke_accumulation)
    smoke_metadata.update(
        {
            "disposable_smoke": True,
            "immutable_stage_a_sha256": source_sha_before,
        }
    )
    checkpoint_verification = save_checkpoint(
        smoke_checkpoint,
        "b",
        model,
        optimizer,
        loader,
        SMOKE_UPDATES,
        smoke_accumulation,
        smoke_metadata,
    )
    reopened_model, reopened_optimizer, reopened_loader, reopened_payload = (
        load_stage_b_checkpoint(smoke_checkpoint, device, restore_rng=True)
    )
    reload_checks = {
        "strict_reopen": checkpoint_verification["strict_reopen"]["passed"],
        "completed_updates": reopened_payload["completed_local_updates"] == SMOKE_UPDATES,
        "next_batch": next_global_batch_hash(reopened_loader, smoke_accumulation)
        == checkpoint_verification["next_global_batch_sha256"],
        "next_stream": global_batch_stream_hash(reopened_loader, smoke_accumulation)
        == checkpoint_verification["next_global_batch_stream_sha256"],
        "model_finite": model_finite(reopened_model),
        "optimizer_finite": optimizer_finite(reopened_optimizer),
        "gate_exact": torch.equal(reopened_model.g_rec_b3, model.g_rec_b3),
    }
    checkpoint_reload = {
        "checkpoint": checkpoint_verification,
        "checks": reload_checks,
        "passed": all(reload_checks.values()),
    }
    del reopened_model, reopened_optimizer, reopened_loader, reopened_payload
    gc.collect()
    torch.cuda.empty_cache()

    disposable_paths = [
        smoke_checkpoint,
        smoke_checkpoint.with_suffix(smoke_checkpoint.suffix + ".sha256"),
        smoke_checkpoint.with_suffix(smoke_checkpoint.suffix + ".verification.json"),
    ]
    cleanup_actions = []
    for path in disposable_paths:
        existed = path.is_file()
        size = path.stat().st_size if existed else 0
        if existed:
            path.unlink()
        cleanup_actions.append(
            {
                "path": str(path),
                "bytes": size,
                "kind": "disposable_smoke_checkpoint_or_sidecar",
                "removed": existed and not path.exists(),
            }
        )
    record_cleanup(output, cleanup_actions)

    clean_stage_a, clean_stage_a_optimizer, clean_loader, clean_payload = (
        load_stage_a_checkpoint(args.stage_a_checkpoint, device, restore_rng=False)
    )
    clean_stage_b, clean_stage_b_optimizer, clean_transplant = transplant_stage_a_to_b(
        clean_stage_a, clean_stage_a_optimizer, device
    )
    clean_reload_checks = {
        "immutable_stage_a_sha_unchanged": file_sha256(args.stage_a_checkpoint)
        == source_sha_before,
        "stage_a_final_update": clean_payload["completed_local_updates"]
        == UPDATES_PER_STAGE,
        "stage_a_final_batch": next_global_batch_hash(
            clean_loader, clean_payload["gradient_accumulation"]
        )
        == STAGE_A_FINAL_BATCH,
        "stage_a_final_stream": global_batch_stream_hash(
            clean_loader, clean_payload["gradient_accumulation"]
        )
        == STAGE_A_FINAL_STREAM,
        "fresh_b3_gate_zero": clean_stage_b.g_rec_b3.detach().float().item() == 0.0,
        "fresh_b3_optimizer_state_absent": clean_stage_b.g_rec_b3
        not in clean_stage_b_optimizer.state,
        "transplant_exact": all(clean_transplant.values()),
    }
    clean_reload = {"checks": clean_reload_checks, "passed": all(clean_reload_checks.values())}
    checks = {
        "three_updates": len(rows) == SMOKE_UPDATES,
        "finite": all(math.isfinite(row["pass_losses"][-1]) for row in rows),
        "finite_gradients": all(
            all(group["finite"] for group in row["gradient_groups"].values())
            for row in rows
        ),
        "gate_changed": rows[0]["g_rec_b3_after"] != rows[0]["g_rec_b3_before"],
        "b3_gradient": rows[0]["gradient_groups"]["b3_gate"]["nonzero"],
        "writer_gradient_after_gate_open": writer["passed"],
        "causality": causality["passed"],
        "no_future_leakage": leakage["passed"],
        "cache": cache["passed"] and cache["has_b11_ring"] is False,
        "source_untouched": file_sha256(args.stage_a_checkpoint) == source_sha_before,
        "transplant": all(transplant.values()),
        "checkpoint_reload": checkpoint_reload["passed"],
        "disposable_checkpoint_removed": all(
            row["removed"] for row in cleanup_actions
        ),
        "exact_source_reloaded_for_science": clean_reload["passed"],
    }
    audit = {
        "kind": "three disposable Stage B updates",
        "immutable_stage_a_checkpoint": str(Path(args.stage_a_checkpoint).resolve()),
        "immutable_stage_a_sha256": source_sha_before,
        "micro_batch": smoke_micro_batch,
        "gradient_accumulation": smoke_accumulation,
        "targets_per_disposable_update": GLOBAL_TARGETS,
        "rows": rows,
        "incremental_cache_audit": cache,
        "causality_audit": causality,
        "future_leakage_audit": leakage,
        "post_open_writer_gradient_audit": writer,
        "disposable_checkpoint_reload_audit": checkpoint_reload,
        "science_source_reload_audit": clean_reload,
        "cleanup_actions": cleanup_actions,
        "checks": checks,
        "passed": all(checks.values()),
        "disposition": "discarded; scientific Stage B reloads exact Stage A final",
    }
    durable_json(output / "causality_audit.json", {
        "mask": causality,
        "future_leakage": leakage,
        "passed": causality["passed"] and leakage["passed"],
    })
    durable_json(output / "smoke_audit.json", audit)
    append_command_runtime(
        output,
        {
            "command": " ".join(sys.argv),
            "kind": "disposable_stage_b_smoke",
            "pid": os.getpid(),
            "wall_seconds": time.monotonic() - started,
        },
    )
    del clean_stage_a, clean_stage_a_optimizer, clean_loader, clean_payload
    del clean_stage_b, clean_stage_b_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if not audit["passed"]:
        raise SystemExit(f"2D2G smoke failed: {checks}")
    print("EXPERIMENT_2D2G_STAGE_B_SMOKE_PASS", flush=True)
    return audit


def paired_stats(left, right) -> dict:
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    return {
        "count": len(differences),
        "differences_left_minus_right": differences,
        "mean": statistics.fmean(differences),
        "median": statistics.median(differences),
        "sample_std": statistics.stdev(differences) if len(differences) > 1 else 0.0,
        "wins": sum(value < 0 for value in differences),
        "losses": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
    }


@torch.no_grad()
def evaluate_parallel(model, val_path, batches=VALIDATION_BATCHES) -> dict:
    model.eval()
    device = next(model.parameters()).device
    loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    names = INCREMENTAL_CONTROLS
    rows = {name: {"sum": 0.0, "targets": 0, "batches": []} for name in names}
    identities = []
    permutation = torch.arange(VALIDATION_B, device=device).roll(1)
    for _ in range(int(batches)):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(batch_identity(cpu_x, cpu_y))
        x, y = cpu_x.to(device), cpu_y.to(device)
        for name in names:
            kwargs = {}
            if name == "b3_off":
                kwargs["b3_gate_override"] = 0.0
            elif name == "b3_shuffled":
                kwargs["b3_recurrent_permutation"] = permutation
            elif name == "b3_full_counterfactual":
                kwargs["b3_full_counterfactual"] = True
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model.forward_multi_pass(x, targets=y, num_passes=2, **kwargs)
            loss = result["loss"].detach().float().item()
            rows[name]["sum"] += loss * y.numel()
            rows[name]["targets"] += y.numel()
            rows[name]["batches"].append(loss)
        del x, y, cpu_x, cpu_y
    controls = {
        name: {
            "validation_loss": row["sum"] / row["targets"],
            "validation_targets": row["targets"],
            "per_batch_losses": row["batches"],
        }
        for name, row in rows.items()
    }
    real = controls["real"]
    off = controls["b3_off"]
    shuffled = controls["b3_shuffled"]
    return {
        "controls": controls,
        "b3_gain": off["validation_loss"] - real["validation_loss"],
        "b3_sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "real_vs_off_batches": paired_stats(real["per_batch_losses"], off["per_batch_losses"]),
        "real_vs_shuffled_batches": paired_stats(real["per_batch_losses"], shuffled["per_batch_losses"]),
        "batch_identities": identities,
        "subset_sha256": aggregate_hashes(row["combined_sha256"] for row in identities),
        "g_rec_b1_raw": model.g_rec.detach().float().item(),
        "tanh_g_rec_b1": model.recurrent_scale_b1.detach().float().item(),
        "g_rec_b3_raw": model.g_rec_b3.detach().float().item(),
        "tanh_g_rec_b3": model.recurrent_scale_b3.detach().float().item(),
    }


def merge_keyed(path, key, value):
    path = Path(path)
    payload = read_json(path) if path.exists() else {}
    payload[str(key)] = value
    durable_json(path, payload)


def milestone_key_audit(payload: dict) -> dict:
    expected = {str(update) for update in MILESTONES}
    observed = set(payload)
    return {
        "expected_keys": sorted(expected, key=int),
        "observed_keys": sorted(observed),
        "missing_keys": sorted(expected - observed),
        "unexpected_keys": sorted(observed - expected),
        "passed": observed == expected,
    }


def incremental_control(model, x, y, name, permutation=None):
    state = model.init_incremental_state(
        x.size(0), device=x.device, b3_full_cache=name == "b3_full_counterfactual"
    )
    per_sequence = torch.zeros(x.size(0), dtype=torch.float64)
    total = 0.0
    cache_rows = []
    for position in range(x.size(1)):
        logits, state, diagnostics = model.incremental_step(
            x[:, position],
            state,
            control=name,
            recurrent_permutation=permutation if name == "b3_shuffled" else None,
            return_diagnostics=True,
        )
        losses = F.cross_entropy(
            logits[:, 0].float(), y[:, position], reduction="none"
        ).double().cpu()
        per_sequence += losses
        total += losses.sum().item()
        if position in {0, 63, 64, 1023}:
            cache_rows.append(diagnostics["cache_audit"])
    return {
        "loss_sum": total,
        "targets": x.numel(),
        "per_sequence_losses": (per_sequence / x.size(1)).tolist(),
        "final_cache_audit": model.incremental_cache_audit(state),
        "cache_rows": cache_rows,
    }


@torch.no_grad()
def evaluate_incremental(model, val_path, batches=INCREMENTAL_BATCHES) -> dict:
    if int(batches) != 4:
        raise ValueError("primary 2D2G incremental evaluation requires 4 B64 batches")
    model.eval()
    device = next(model.parameters()).device
    loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    rows = {
        name: {"sum": 0.0, "targets": 0, "batches": [], "sequences": [], "cache_rows": []}
        for name in INCREMENTAL_CONTROLS
    }
    identities = []
    permutation = torch.arange(VALIDATION_B, device=device).roll(1)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(int(batches)):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(batch_identity(cpu_x, cpu_y))
        x, y = cpu_x.to(device), cpu_y.to(device)
        for name in INCREMENTAL_CONTROLS:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current = incremental_control(model, x, y, name, permutation)
            row = rows[name]
            row["sum"] += current["loss_sum"]
            row["targets"] += current["targets"]
            row["batches"].append(current["loss_sum"] / current["targets"])
            row["sequences"].extend(current["per_sequence_losses"])
            row["cache_rows"].append(current["final_cache_audit"])
        print(f"2D2G-B incremental batch {batch_index + 1}/4", flush=True)
        del x, y, cpu_x, cpu_y
        torch.cuda.empty_cache()
    controls = {
        name: {
            "validation_loss": row["sum"] / row["targets"],
            "validation_targets": row["targets"],
            "per_batch_losses": row["batches"],
            "per_sequence_losses": row["sequences"],
            "cache_rows": row["cache_rows"],
        }
        for name, row in rows.items()
    }
    real, off, shuffled = (
        controls["real"], controls["b3_off"], controls["b3_shuffled"]
    )
    return {
        "controls": controls,
        "true_b3_recurrent_gain": off["validation_loss"] - real["validation_loss"],
        "true_b3_sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "real_vs_off_sequences": paired_stats(real["per_sequence_losses"], off["per_sequence_losses"]),
        "real_vs_shuffled_sequences": paired_stats(real["per_sequence_losses"], shuffled["per_sequence_losses"]),
        "batch_identities": identities,
        "subset_sha256": aggregate_hashes(row["combined_sha256"] for row in identities),
        "targets_per_control": int(batches) * VALIDATION_B * T,
        "paired_sequences": int(batches) * VALIDATION_B,
        "no_complete_prefix_recomputation": True,
        "performance": {
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }


def attention_diagnostics(model, val_path) -> dict:
    model.eval()
    device = next(model.parameters()).device
    loader = d1.ExplicitShardLoader([val_path], 1, T)
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first = model.forward_pass(x, targets=y)
        second = model.forward_pass(
            x,
            targets=y,
            b1_recurrent_source=first["h12"],
            b3_recurrent_source=first["h10"],
            return_diagnostics=True,
        )
    diag = second["diagnostics"]["b3"]
    weights = diag["recurrent_attention_weights"].detach().float().cpu()
    mask = diag["recurrent_valid_mask"].cpu()
    length = weights.size(2)
    lag_mass = {name: 0.0 for name, _, _ in B3_RECURRENT_LAG_BINS}
    lag_count = {name: 0 for name, _, _ in B3_RECURRENT_LAG_BINS}
    weighted_lags = []
    head_numerator = torch.zeros(weights.size(1))
    head_denominator = torch.zeros(weights.size(1))
    entropy_sum = 0.0
    row_count = 0
    for query in range(length):
        sources = torch.where(mask[query])[0]
        if sources.numel() == 0:
            continue
        current = weights[0, :, query, sources]
        lags = query - sources
        mean_head = current.mean(0)
        for name, low, high in B3_RECURRENT_LAG_BINS:
            selected = (lags >= low) & (lags <= high)
            lag_mass[name] += mean_head[selected].sum().item()
            lag_count[name] += int(selected.sum())
        for lag, mass in zip(lags.tolist(), mean_head.tolist()):
            weighted_lags.append((int(lag), float(mass)))
        head_numerator += (current * lags.float()).sum(1)
        head_denominator += current.sum(1)
        entropy_sum += (-(current.clamp_min(1e-30).log() * current).sum(1)).mean().item()
        row_count += 1
    total_mass = sum(mass for _, mass in weighted_lags)
    ordered = sorted(weighted_lags)
    def quantile(q):
        threshold = total_mass * q
        cumulative = 0.0
        for lag, mass in ordered:
            cumulative += mass
            if cumulative >= threshold:
                return lag
        return ordered[-1][0]
    head_means = (head_numerator / head_denominator.clamp_min(1e-30)).tolist()
    return {
        "lag_bins": {
            name: {
                "raw_attention_mass": lag_mass[name] / max(row_count, 1),
                "normalized_mass_per_token": lag_mass[name] / max(lag_count[name], 1),
            }
            for name, _, _ in B3_RECURRENT_LAG_BINS
        },
        "entropy": entropy_sum / max(row_count, 1),
        "effective_positions": math.exp(entropy_sum / max(row_count, 1)),
        "mean_lag": sum(lag * mass for lag, mass in weighted_lags) / total_mass,
        "median_lag": quantile(0.5),
        "p90_lag": quantile(0.9),
        "per_head_mean_lags": head_means,
        "per_head_mean_lag_range": max(head_means) - min(head_means),
    }


def temporal_gradient_diagnostics(model, val_path) -> dict:
    device = next(model.parameters()).device
    loader = d1.ExplicitShardLoader([val_path], 2, T)
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    return b3_writer_gradient_audit(model, x, y)


@torch.no_grad()
def stability_8pass(model, val_path) -> dict:
    model.eval()
    device = next(model.parameters()).device
    loader = d1.ExplicitShardLoader([val_path], 2, T)
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    source_h10 = source_h12 = None
    rows = []
    for pass_index in range(1, 9):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            row = model.forward_pass(
                x,
                targets=y,
                b1_recurrent_source=source_h12,
                b3_recurrent_source=source_h10,
                return_diagnostics=True,
            )
        diag = row["diagnostics"]["b3"]
        rows.append({
            "pass": pass_index,
            "loss": row["loss"].detach().float().item(),
            "h10_rms": row["h10"].float().square().mean().sqrt().item(),
            "h12_rms": row["h12"].float().square().mean().sqrt().item(),
            "b3_recurrent_output_rms": diag["recurrent_output_rms"].item(),
            "finite": bool(torch.isfinite(row["logits"]).all()),
        })
        source_h10, source_h12 = row["h10"], row["h12"]
    return {"passes": rows, "passed": all(row["finite"] for row in rows)}


def memory_accounting() -> dict:
    def one(batch):
        bf16 = 2
        b1_kv = batch * 1 * N_EMBD * 2 * bf16
        b2_kv = batch * 1023 * N_EMBD * 2 * bf16
        b3_kv = batch * 63 * N_EMBD * 2 * bf16
        upper_kv = batch * 9 * 1023 * N_EMBD * 2 * bf16
        rings = batch * 2 * 1023 * N_EMBD * bf16
        total = b1_kv + b2_kv + b3_kv + upper_kv + rings
        standard = batch * 12 * 1023 * N_EMBD * 2 * bf16
        return {
            "b1_local_kv_bytes": b1_kv,
            "b2_full_kv_bytes": b2_kv,
            "b3_local_kv_bytes": b3_kv,
            "b4_b12_full_kv_bytes": upper_kv,
            "h10_h12_raw_ring_bytes": rings,
            "b11_ring_bytes": 0,
            "total_inference_state_bytes": total,
            "standard_gpt2_bytes": standard,
            "saving_bytes_vs_standard": standard - total,
        }
    return {"dtype": "BF16", "B1": one(1), "B64": one(64)}


def classify_result(incremental, stable=True, integrity=True):
    if not integrity:
        return "INVALID"
    if not stable:
        return "INVALID"
    gain = incremental["true_b3_recurrent_gain"]
    gap = incremental["true_b3_sequence_gap"]
    off = incremental["real_vs_off_sequences"]["wins"]
    shuffled = incremental["real_vs_shuffled_sequences"]["wins"]
    if gain > 0 and gap > 0 and off >= 166 and shuffled >= 166 and gain >= 0.001:
        return "STRONG POSITIVE"
    if gain > 0 and gap > 0 and off >= 129 and shuffled >= 129:
        return "POSITIVE UTILITY ESTABLISHED"
    if gap > 0:
        return "SEQUENCE-SPECIFIC BUT NOT ESTABLISHED"
    if abs(gain) < 0.0001 and abs(gap) < 0.0001:
        return "NEAR ZERO"
    if gain < 0:
        return "HARMFUL"
    return "NEAR ZERO"


def render_report(summary, audit) -> str:
    incremental = summary["incremental"]
    lines = [
        "# Experiment 2D2G Final Report",
        "",
        f"Primary classification: **{summary['primary_classification']}**",
        "",
        "## Architecture",
        "",
        "Stage A continued exact 2D2B for 191 matched updates. Stage B kept B2 W1024 with no B11 recurrence and added B3 W64 plus B10 recurrence at lags 64–1023.",
        "",
        "## True incremental result",
        "",
        f"- Real: `{incremental['controls']['real']['validation_loss']}`",
        f"- B3 off: `{incremental['controls']['b3_off']['validation_loss']}`",
        f"- B3 shuffled: `{incremental['controls']['b3_shuffled']['validation_loss']}`",
        f"- Gain: `{incremental['true_b3_recurrent_gain']}`",
        f"- Sequence gap: `{incremental['true_b3_sequence_gap']}`",
        f"- Wins vs off: `{incremental['real_vs_off_sequences']['wins']}/256`",
        f"- Wins vs shuffled: `{incremental['real_vs_shuffled_sequences']['wins']}/256`",
        "",
        "## Integrity",
        "",
        f"Final audit passed: `{audit['passed']}`.",
        "Stage B contains no B11 recurrent gate or raw-state ring.",
        "",
    ]
    return "\n".join(lines)


def run_finalize(args):
    started = time.monotonic()
    require_git(clean=False)
    require_config()
    device = require_assigned_a100()
    output = Path(args.output_dir).resolve()
    require_fingerprint(output)
    persisted = read_json(output / "persistent_final_checkpoint.json")
    if not persisted.get("passed") or persisted.get("local_sha256") != file_sha256(
        args.stage_b_checkpoint
    ):
        raise SystemExit("finalize requires SHA-verified persistent final checkpoint")
    persistent_checkpoint = Path(persisted["persistent"]).resolve()
    persistent_sidecars = checkpoint_sidecar_audit(
        persistent_checkpoint, persisted["local_sha256"]
    )
    if not persistent_sidecars["passed"]:
        raise SystemExit("finalize persistent checkpoint re-verification failed")
    model, optimizer, _, payload = load_stage_b_checkpoint(
        args.stage_b_checkpoint, device, restore_rng=False
    )
    if int(payload["completed_local_updates"]) != 191:
        raise SystemExit("finalize requires Stage B update 191")
    if file_sha256(args.stage_b_checkpoint) != Path(args.stage_b_checkpoint).with_suffix(
        Path(args.stage_b_checkpoint).suffix + ".sha256"
    ).read_text().split()[0]:
        raise SystemExit("final checkpoint sidecar mismatch")
    val = validation_path(args.data_root)
    parallel = evaluate_parallel(model, val)
    incremental = evaluate_incremental(model, val)
    attention = attention_diagnostics(model, val)
    temporal = temporal_gradient_diagnostics(model, val)
    stability = stability_8pass(model, val)
    memory = memory_accounting()
    cache = incremental["controls"]["real"]["cache_rows"][-1]
    stage_a_match = read_json(output / "stage_a_data_match.json")
    stage_b_match = read_json(output / "stage_b_data_match.json")
    smoke = read_json(output / "smoke_audit.json")
    causality = read_json(output / "causality_audit.json")
    transition = read_json(output / "stage_b_transition_audit.json")
    restart_a = read_json(output / "stage_a_forced_restart_update_96.json")
    restart_b = read_json(output / "stage_b_forced_restart_update_96.json")
    checkpoint_manifest = read_json(output / "checkpoint_manifest.json")
    milestones = read_json(output / "milestone_validation.json")
    training_milestone_191_present = "191" in milestones
    legacy_191_final_removed = milestones.pop("191_final", None) is not None
    milestones["191"] = parallel
    milestone_keys = milestone_key_audit(milestones)
    durable_json(output / "milestone_validation.json", milestones)
    checks = {
        "final_checkpoint_update_191": payload["completed_local_updates"] == 191,
        "final_cursor_matched_2d2e": (
            payload["next_global_batch_sha256"], payload["next_global_batch_stream_sha256"]
        ) == (STAGE_B_FINAL_BATCH, STAGE_B_FINAL_STREAM),
        "parameters": sum(p.numel() for p in model.parameters()) == STAGE_B_PARAMETERS,
        "no_b2_gate": not hasattr(model, "g_rec_b2"),
        "no_b11_ring": cache["has_b11_ring"] is False,
        "incremental_targets": incremental["targets_per_control"] == 262_144,
        "paired_sequences": incremental["paired_sequences"] == 256,
        "cache_audit": cache["passed"],
        "stability": stability["passed"],
        "writer_gradient": temporal["passed"],
        "causality": causality["passed"],
        "disposable_smoke": smoke["passed"],
        "stage_a_data_match": stage_a_match.get("passed") is True,
        "stage_b_data_match": stage_b_match.get("passed") is True,
        "stage_b_transition": transition["passed"],
        "stage_a_fresh_process_update_96": restart_a["passed"],
        "stage_b_fresh_process_update_96": restart_b["passed"],
        "training_milestone_191_present": training_milestone_191_present,
        "exact_milestone_set": milestone_keys["passed"],
        "stage_a_checkpoints": all(
            key in checkpoint_manifest["stage_a"] for key in ("96", "191")
        ),
        "stage_b_checkpoints": all(
            key in checkpoint_manifest["stage_b"]
            for key in ("96", "191", "191_persistent")
        ),
        "persistent_checkpoint_reverified": persistent_sidecars["passed"],
        "persistent_sha_matches_local": persisted["persistent_sha256"]
        == file_sha256(args.stage_b_checkpoint),
        "model_finite": model_finite(model),
        "optimizer_finite": optimizer_finite(optimizer),
    }
    scientific_integrity = all(checks.values())
    classification = classify_result(
        incremental, stability["passed"], scientific_integrity
    )
    audit = {"checks": checks, "passed": all(checks.values()), "classification": classification}
    audit["milestone_reconciliation"] = {
        "final_evaluation_overwrote_update_191": True,
        "legacy_191_final_removed": legacy_191_final_removed,
        "key_audit": milestone_keys,
    }
    summary = {
        "experiment": EXPERIMENT,
        "primary_classification": classification,
        "source_2d2b_checkpoint": read_json(output / "source_manifest.json")[
            "checkpoint"
        ],
        "local_final_checkpoint": str(Path(args.stage_b_checkpoint).resolve()),
        "final_checkpoint": str(persistent_checkpoint),
        "final_checkpoint_sha256": file_sha256(args.stage_b_checkpoint),
        "parameters": STAGE_B_PARAMETERS,
        "architecture": stage_architecture("b"),
        "parallel": parallel,
        "incremental": incremental,
        "attention": attention,
        "temporal_gradient": temporal,
        "stability_8pass": stability,
        "memory_accounting": memory,
        "final_gates": {
            "b1": model.recurrent_scale_b1.detach().float().item(),
            "b3": model.recurrent_scale_b3.detach().float().item(),
        },
        "git": {"branch": BRANCH, "implementation_commit": git_output("rev-parse", "HEAD")},
    }
    durable_json(output / "paired_controls.json", {
        "parallel": {
            "real_vs_off": parallel["real_vs_off_batches"],
            "real_vs_shuffled": parallel["real_vs_shuffled_batches"],
        },
        "incremental": {
            "real_vs_off": incremental["real_vs_off_sequences"],
            "real_vs_shuffled": incremental["real_vs_shuffled_sequences"],
        },
    })
    gate_trajectory = read_json(output / "gate_diagnostics.json")
    gate_trajectory["final"] = summary["final_gates"]
    durable_json(output / "gate_diagnostics.json", gate_trajectory)
    durable_json(output / "attention_diagnostics.json", attention)
    durable_json(output / "temporal_gradient_diagnostics.json", temporal)
    durable_json(output / "incremental_validation.json", incremental)
    durable_json(output / "incremental_cache_audit.json", cache)
    durable_json(output / "memory_accounting.json", memory)
    durable_json(output / "stability_8pass.json", stability)
    command_runtime = read_json(output / "commands_and_runtime.json")
    training_wall_seconds = sum(
        float(row.get("wall_seconds", 0.0))
        for row in command_runtime.get("commands", [])
        if row.get("kind") != "preflight"
    )
    finalize_wall_seconds = time.monotonic() - started
    durable_json(output / "performance.json", {
        "incremental": incremental["performance"],
        "stage_a_metrics": str(output / "stage_a_training_metrics.jsonl"),
        "stage_b_metrics": str(output / "stage_b_training_metrics.jsonl"),
        "recorded_pre_finalize_wall_seconds": training_wall_seconds,
        "finalization_wall_seconds": finalize_wall_seconds,
        "recorded_lane_gpu_hours": (training_wall_seconds + finalize_wall_seconds) / 3600.0,
    })
    durable_json(output / "result_summary.json", summary)
    append_command_runtime(
        output,
        {
            "command": " ".join(sys.argv),
            "kind": "finalize",
            "pid": os.getpid(),
            "wall_seconds": finalize_wall_seconds,
        },
    )
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_text(output / "FINAL_REPORT.md", render_report(summary, audit))
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", render_report(summary, audit))
    durable_json(output / "HEARTBEAT.json", {
        "experiment": EXPERIMENT, "phase": "FINALIZED_PENDING_GIT",
        "timestamp": time.time(), "pid": os.getpid()
    })
    inventory = required_artifact_inventory(output)
    checks["required_artifacts"] = inventory["passed"]
    final_integrity = all(checks.values())
    classification = classify_result(
        incremental, stability["passed"], final_integrity
    )
    summary["primary_classification"] = classification
    audit["classification"] = classification
    audit["checks"] = checks
    audit["passed"] = final_integrity
    durable_json(output / "artifact_inventory.json", inventory)
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_text(output / "FINAL_REPORT.md", render_report(summary, audit))
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", render_report(summary, audit))
    if not audit["passed"]:
        raise SystemExit(f"2D2G final audit failed: {checks}")
    print("EXPERIMENT_2D2G_FINALIZED_PENDING_GIT", flush=True)
    return summary


def run_persist_final(args):
    started = time.monotonic()
    paths = validate_final_persistence_paths(
        args.local_checkpoint, args.persistent_dir, args.lock_path
    )
    if not paths["passed"]:
        raise SystemExit(f"invalid local-to-persistent checkpoint paths: {paths}")
    report = persist_checkpoint(args.local_checkpoint, args.persistent_dir, args.lock_path)
    report["path_audit"] = paths
    output = Path(args.output_dir).resolve()
    durable_json(output / "persistent_final_checkpoint.json", report)
    manifest = read_json(output / "checkpoint_manifest.json")
    manifest["stage_b"]["191_persistent"] = report
    durable_json(output / "checkpoint_manifest.json", manifest)
    append_command_runtime(
        output,
        {
            "command": " ".join(sys.argv),
            "kind": "persist_final_checkpoint",
            "pid": os.getpid(),
            "wall_seconds": time.monotonic() - started,
            "lock": str(CHECKPOINT_PERSIST_LOCK),
        },
    )
    print("EXPERIMENT_2D2G_FINAL_CHECKPOINT_PERSISTED", flush=True)
    return report


def add_common(parser):
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--pod-name", required=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    add_common(preflight)
    preflight.add_argument("--source-checkpoint", required=True)
    preflight.add_argument("--data-root", required=True)
    preflight.set_defaults(function=run_preflight)

    smoke = sub.add_parser("smoke-b")
    add_common(smoke)
    smoke.add_argument("--stage-a-checkpoint", required=True)
    smoke.add_argument("--data-root", required=True)
    smoke.add_argument("--checkpoint-dir", required=True)
    smoke.add_argument("--micro-batch", type=int)
    smoke.set_defaults(function=run_smoke_b)

    train_a = sub.add_parser("train-a")
    add_common(train_a)
    train_a.add_argument("--source-checkpoint")
    train_a.add_argument("--checkpoint-dir", required=True)
    train_a.add_argument("--end-update", required=True, type=int)
    train_a.add_argument("--resume")
    train_a.add_argument("--micro-batch", type=int)
    train_a.add_argument("--data-root", required=True)
    train_a.set_defaults(function=run_train_a)

    train_b = sub.add_parser("train-b")
    add_common(train_b)
    train_b.add_argument("--stage-a-checkpoint")
    train_b.add_argument("--checkpoint-dir", required=True)
    train_b.add_argument("--end-update", required=True, type=int)
    train_b.add_argument("--resume")
    train_b.add_argument("--micro-batch", type=int)
    train_b.add_argument("--data-root", required=True)
    train_b.set_defaults(function=run_train_b)

    finalize = sub.add_parser("finalize")
    add_common(finalize)
    finalize.add_argument("--stage-b-checkpoint", required=True)
    finalize.add_argument("--data-root", required=True)
    finalize.set_defaults(function=run_finalize)

    persist = sub.add_parser("persist-final")
    persist.add_argument("--output-dir", required=True)
    persist.add_argument("--local-checkpoint", required=True)
    persist.add_argument("--persistent-dir", required=True)
    persist.add_argument("--lock-path", required=True)
    persist.set_defaults(function=run_persist_final)
    return parser


def main():
    args = build_parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    main()
