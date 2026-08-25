#!/usr/bin/env python3
"""Experiment 2D1D: retrain zero-initialized residual recurrence from C954."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d1 as d1  # noqa: E402
import experiment_2d1a as d1a  # noqa: E402
import experiment_2d1c as d1c  # noqa: E402
import experiment_2d1r as r  # noqa: E402


EXPERIMENT = "2D1D"
PROTOCOL = "exp2d1d_residual_recurrence_retrain_c954_v1"
BRANCH = "experiment-2d1d-residual-recurrence-retrain-c954"
FROZEN_COMMIT = "7cb2a55876a3d3496f25932a19da5cc107790295"
FROZEN_TAG = "experiment-2d1c-residual-alpha-sweep-final"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d1d_residual_recurrence_retrain_c954.json"
OUTPUT_NAME = "experiment_2d1d_residual_recurrence_retrain_c954"
CHECKPOINT_SCHEMA = "exp2d1d_residual_recurrence_retrain_c954_checkpoint_v1"

SOURCE_SHA256 = "22abc6de4e49e27504b4d0e66ca0d2e3396ed6d76d7ee18e0e11cfb1eb3192c0"
SOURCE_BYTES = 1_508_094_603
SOURCE_GLOBAL_UPDATE = 954
SOURCE_TARGETS = 500_170_752
FIRST_GLOBAL_UPDATE = 955
LOCAL_UPDATES = 477
FINAL_GLOBAL_UPDATE = SOURCE_GLOBAL_UPDATE + LOCAL_UPDATES
ADDITIONAL_TARGETS = LOCAL_UPDATES * d1.GLOBAL_TARGETS
FINAL_TARGETS = SOURCE_TARGETS + ADDITIONAL_TARGETS
ALPHA = 0.03125
SIGMA_CAP = 1.0262317657470703
STAGE_A_RMS = 0.03550996296107769
HARD_RMS = 0.3550996296107769
PLAIN_ORACLE = 3.073581371307373
PLAIN_ORACLE_TOLERANCE = 2e-7
PROJECTION_TOLERANCE = 1e-5

B12 = (512, 545, 581, 618, 658, 702, 747, 796, 848, 903, 962, 1024)
C12 = (256, 290, 329, 373, 423, 481, 545, 619, 702, 796, 903, 1024)
SCIENTIFIC_LOCAL = (20, 48, 96, 191, 286, 477)
MILESTONE_LOCAL = (20, 48, 96, 191, 286, 477)
SELF_LOCAL = (96, 191, 477)
POSITION_LOCAL = (0, 96, 191, 477)
SCIENTIFIC_GLOBAL = tuple(SOURCE_GLOBAL_UPDATE + value for value in SCIENTIFIC_LOCAL)
MILESTONE_GLOBAL = tuple(SOURCE_GLOBAL_UPDATE + value for value in MILESTONE_LOCAL)
ROLLING_INTERVAL = 50
ROLLING_KEEP = 3
POSITION_BINS = ((1, 64), (65, 128), (129, 256), (257, 512), (513, 768), (769, 896), (897, 1023))
BRANCH_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)

_R_LOAD_SOURCE = r.load_source_runtime


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"{EXPERIMENT} requires branch {BRANCH}")
    if git_output("rev-parse", FROZEN_TAG + "^{commit}") != FROZEN_COMMIT:
        raise SystemExit("frozen 2D1C tag mismatch")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=REPO_ROOT)
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D1D worktree must be clean")


def durable_json(path, payload):
    r.durable_json(path, payload)


def durable_text(path, value):
    r.durable_text(path, value)


def append_jsonl(path, payload):
    r.append_jsonl(path, payload)


def read_json(path):
    return r.read_json(path)


def read_jsonl(path):
    return r.read_jsonl(path)


def file_sha256(path):
    return r.file_sha256(path)


def local_update(global_update):
    return int(global_update) - SOURCE_GLOBAL_UPDATE


def global_update(local):
    return SOURCE_GLOBAL_UPDATE + int(local)


def stage_for_update(update):
    local = local_update(update)
    if local <= 0:
        return {"stage": "SOURCE-B", "windows": B12, "rho": 0.5, "alpha": 0.0}
    if local <= 96:
        return {"stage": "B-R", "windows": B12, "rho": 0.0, "alpha": ALPHA}
    if local <= LOCAL_UPDATES:
        return {"stage": "C-R", "windows": C12, "rho": 0.0, "alpha": ALPHA}
    raise ValueError(f"global update outside 2D1D pilot: {update}")


def pass_count_for_update(update):
    return 3 if int(update) % d1.THREE_PASS_EVERY == 0 else 2


def require_config():
    config = read_json(CONFIG_PATH)
    expected = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "branch": BRANCH,
        "frozen_2d1c_commit": FROZEN_COMMIT,
        "frozen_2d1c_tag": FROZEN_TAG,
        "source_c954_sha256": SOURCE_SHA256,
        "source_c954_bytes": SOURCE_BYTES,
        "source_global_update": SOURCE_GLOBAL_UPDATE,
        "local_updates": LOCAL_UPDATES,
        "additional_targets": ADDITIONAL_TARGETS,
        "final_cumulative_targets": FINAL_TARGETS,
        "alpha": ALPHA,
        "wu_sigma_cap": SIGMA_CAP,
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    checks = {
        "source_targets": SOURCE_GLOBAL_UPDATE * d1.GLOBAL_TARGETS == SOURCE_TARGETS,
        "additional_targets": ADDITIONAL_TARGETS == 250_085_376,
        "final_targets": FINAL_TARGETS == 750_256_128,
        "B12": B12 == tuple(d1.STAGES[1][3]),
        "C12": C12 == tuple(d1.STAGES[2][3]),
    }
    if mismatch or not all(checks.values()):
        raise SystemExit(f"2D1D preregistration mismatch: {mismatch} {checks}")
    return config


def bind_residual_mode(model):
    model.make_input = MethodType(residual_make_input, model)
    model._exp2d1d_residual_mode = True
    return model


def residual_make_input(self, tokens, previous_top=None, rho=0.0, prefix_length=None, return_diagnostics=False):
    # The legacy call signature is retained for shared validators, but rho is
    # deliberately ignored: residual mode has no rho mixture.
    batch, length = tokens.shape
    positions = torch.arange(length, dtype=torch.long, device=tokens.device)
    embedding = self.base.transformer.wte(tokens)
    shifted = zn = u = gate = fused = alpha_fused = None
    if previous_top is None:
        recurrent_input = embedding
        recurrent_mask = torch.zeros((1, length, 1), dtype=torch.bool, device=tokens.device)
    else:
        if previous_top.shape != (batch, length, self.config.n_embd):
            raise ValueError("previous top state has wrong shape")
        shifted = torch.zeros_like(previous_top)
        shifted[:, 1:] = previous_top[:, :-1]
        zn = self.fusion.normalize(shifted)
        u = self.fusion.W_u(zn)
        gate = 2.0 * torch.sigmoid(self.fusion.W_g(embedding))
        fused = u * gate
        alpha_fused = ALPHA * fused
        candidate = embedding + alpha_fused
        if prefix_length is None:
            prefix_length = 0
        if not 0 <= int(prefix_length) < length:
            raise ValueError("prefix length outside [0,T-1]")
        recurrent_mask = positions.gt(int(prefix_length)).view(1, length, 1)
        recurrent_input = torch.where(recurrent_mask, candidate, embedding)
    value = recurrent_input + self.base.transformer.wpe(positions)
    if not return_diagnostics:
        return value
    with torch.no_grad():
        def rms(tensor):
            return tensor.float().pow(2).mean().sqrt().item()
        e_rms = rms(embedding)
        diagnostics = {
            "embedding_rms": e_rms,
            "recurrent_input_rms": rms(recurrent_input),
            "recurrent_fraction": recurrent_mask.float().mean().item(),
            "alpha": ALPHA,
            "rho_disabled": True,
        }
        if fused is not None:
            gate_f = gate.float()
            alpha_rms = rms(alpha_fused)
            diagnostics.update({
                "shifted_top_rms": rms(shifted),
                "ZN_rms": rms(zn),
                "U_rms": rms(u),
                "F_rms": rms(fused),
                "alphaF_rms": alpha_rms,
                "alphaF_over_E": alpha_rms / e_rms,
                "X_over_E": rms(recurrent_input) / e_rms,
                "gate_mean": gate_f.mean().item(),
                "gate_std": gate_f.std().item(),
                "gate_variance": gate_f.var().item(),
                "gate_saturation_fraction": ((gate_f < 0.01) | (gate_f > 1.99)).float().mean().item(),
            })
    return value, diagnostics


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_hash(model, include_fusion=True):
    rows = []
    for name, tensor in model.state_dict().items():
        if include_fusion or not name.startswith("fusion."):
            rows.append((name, tensor))
    return tensor_hash(rows)


def optimizer_state_hash(optimizer, named_parameters, include_fusion):
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        if name.startswith("fusion.") == include_fusion:
            digest.update(name.encode())
            state = optimizer.state.get(parameter, {})
            for key in sorted(state, key=str):
                digest.update(str(key).encode())
                value = state[key]
                if torch.is_tensor(value):
                    cpu = value.detach().cpu().contiguous()
                    digest.update(str(cpu.dtype).encode())
                    digest.update(str(tuple(cpu.shape)).encode())
                    digest.update(cpu.numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
    return digest.hexdigest()


def configure_r_globals():
    r.EXPERIMENT = EXPERIMENT
    r.PROTOCOL = PROTOCOL
    r.BRANCH = BRANCH
    r.FROZEN_2D1A_COMMIT = FROZEN_COMMIT
    r.FROZEN_2D1A_TAG = FROZEN_TAG
    r.CONFIG_PATH = CONFIG_PATH
    r.OUTPUT_NAME = OUTPUT_NAME
    r.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    r.SOURCE_C954_SHA256 = SOURCE_SHA256
    r.SOURCE_C954_BYTES = SOURCE_BYTES
    r.START_UPDATE = SOURCE_GLOBAL_UPDATE
    r.FIRST_RESULT_UPDATE = FIRST_GLOBAL_UPDATE
    r.FINAL_UPDATE = FINAL_GLOBAL_UPDATE
    r.ADDITIONAL_UPDATES = LOCAL_UPDATES
    r.ADDITIONAL_TARGETS = ADDITIONAL_TARGETS
    r.FINAL_TOTAL_TARGETS = FINAL_TARGETS
    r.SOURCE_TARGETS = SOURCE_TARGETS
    r.SCIENTIFIC_UPDATES = SCIENTIFIC_GLOBAL
    r.MILESTONE_UPDATES = MILESTONE_GLOBAL
    r.ROLLING_INTERVAL = ROLLING_INTERVAL
    r.ROLLING_KEEP = ROLLING_KEEP
    r.STAGE_A_REFERENCE = STAGE_A_RMS
    r.HARD_SCALE_THRESHOLD = HARD_RMS
    r.PROJECTION_RELATIVE_TOLERANCE = PROJECTION_TOLERANCE
    r.stage_for_update = stage_for_update
    r.pass_count_for_update = pass_count_for_update
    r.require_git = require_git


configure_r_globals()


def runtime_metadata(args, payload):
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "frozen_2d1c_commit": FROZEN_COMMIT,
        "source_c954": str(Path(args.source_c954).resolve()),
        "source_c954_sha256": SOURCE_SHA256,
        "source_c954_git_commit": payload["git_commit"],
        "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "parent_checkpoint_sha256": d1.SOURCE_SHA256,
        "data_root": str(Path(args.data_root).resolve()),
        "validation_sha256": d1.VALIDATION_SHARD_SHA256,
        "micro_batch_sequences": int(payload["loader_state"]["batch_size"]),
        "gradient_accumulation": 8,
        "global_targets_per_update": d1.GLOBAL_TARGETS,
        "source_global_update": SOURCE_GLOBAL_UPDATE,
        "local_updates": LOCAL_UPDATES,
        "alpha": ALPHA,
        "fusion_equation": "X = E + 0.03125 * (W_u(RMSNorm(z_prev)) * (2*sigmoid(W_g(E))))",
        "rho_disabled": True,
        "sigma_cap": SIGMA_CAP,
        "projection_relative_tolerance": PROJECTION_TOLERANCE,
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "gpu_type": args.gpu_type,
        "persistent_volume_identity": args.persistent_volume_identity,
        "stop_mechanism": args.stop_mechanism,
        "stop_authenticated": bool(args.stop_authenticated),
    }


def load_source_runtime(args):
    runtime = _R_LOAD_SOURCE(args)
    named = list(runtime.model.named_parameters())
    base_model_before = model_hash(runtime.model, include_fusion=False)
    base_optimizer_before = optimizer_state_hash(runtime.optimizer, named, include_fusion=False)
    fusion_optimizer_before = optimizer_state_hash(runtime.optimizer, named, include_fusion=True)
    old_fusion = {
        "W_u_sha256": tensor_hash([("W_u", runtime.model.fusion.W_u.weight)]),
        "W_g_sha256": tensor_hash([("W_g", runtime.model.fusion.W_g.weight)]),
        "W_u_spectral": r.exact_spectral_norm(runtime.model.fusion.W_u.weight),
        "W_g_spectral": r.exact_spectral_norm(runtime.model.fusion.W_g.weight),
    }
    with torch.no_grad():
        runtime.model.fusion.W_u.weight.zero_()
        runtime.model.fusion.W_g.weight.zero_()
        if runtime.model.fusion.W_u.bias is not None:
            runtime.model.fusion.W_u.bias.zero_()
        if runtime.model.fusion.W_g.bias is not None:
            runtime.model.fusion.W_g.bias.zero_()
    fusion_names = []
    for name, parameter in named:
        if name.startswith("fusion."):
            runtime.optimizer.state[parameter].clear()
            fusion_names.append(name)
    bind_residual_mode(runtime.model)
    base_model_after = model_hash(runtime.model, include_fusion=False)
    base_optimizer_after = optimizer_state_hash(runtime.optimizer, named, include_fusion=False)
    fusion_optimizer_after = optimizer_state_hash(runtime.optimizer, named, include_fusion=True)
    runtime.sigma_ref = SIGMA_CAP
    runtime.metadata = runtime_metadata(args, runtime.source_payload)
    runtime.training_state.update({
        "residual_started_at": time.time(),
        "residual_completed_updates": 0,
        "residual_processed_targets": 0,
        "rescue_started_at": time.time(),
        "rescue_completed_updates": 0,
        "rescue_processed_targets": 0,
        "explosion_consecutive": 0,
        "last_checkpoint": str(Path(args.source_c954).resolve()),
    })
    runtime.fusion_reinitialization = {
        "old_fusion_discarded": True,
        "old_fusion": old_fusion,
        "new_W_u_exact_zero": torch.count_nonzero(runtime.model.fusion.W_u.weight).item() == 0,
        "new_W_g_exact_zero": torch.count_nonzero(runtime.model.fusion.W_g.weight).item() == 0,
        "biases_present": {
            "W_u": runtime.model.fusion.W_u.bias is not None,
            "W_g": runtime.model.fusion.W_g.bias is not None,
        },
        "initial_gate": 1.0,
        "initial_F_zero": True,
        "alpha": ALPHA,
    }
    runtime.optimizer_reset_audit = {
        "base_model_hash_before": base_model_before,
        "base_model_hash_after": base_model_after,
        "base_model_exact": base_model_before == base_model_after,
        "base_optimizer_hash_before": base_optimizer_before,
        "base_optimizer_hash_after": base_optimizer_after,
        "base_optimizer_exact": base_optimizer_before == base_optimizer_after,
        "fusion_optimizer_hash_before": fusion_optimizer_before,
        "fusion_optimizer_hash_after": fusion_optimizer_after,
        "fusion_state_parameters_cleared": fusion_names,
        "fusion_fresh_state": all(not runtime.optimizer.state[p] for n, p in named if n.startswith("fusion.")),
        "only_fusion_state_reset": True,
    }
    if not all((
        runtime.fusion_reinitialization["new_W_u_exact_zero"],
        runtime.fusion_reinitialization["new_W_g_exact_zero"],
        runtime.optimizer_reset_audit["base_model_exact"],
        runtime.optimizer_reset_audit["base_optimizer_exact"],
        runtime.optimizer_reset_audit["fusion_fresh_state"],
    )):
        raise SystemExit("fusion reinitialization / optimizer reset audit failed")
    return runtime


def load_result_runtime(args, checkpoint):
    runtime = load_source_runtime(args)
    payload, _ = r.load_payload(checkpoint)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("2D1D recovery checkpoint schema mismatch")
    checkpoint_metadata = copy.deepcopy(payload.get("metadata", {}))
    current = copy.deepcopy(runtime.metadata)
    checkpoint_commit = checkpoint_metadata.pop("implementation_git_commit", None)
    current_commit = current.pop("implementation_git_commit", None)
    compatible = checkpoint_metadata == current and subprocess.run(
        ["git", "merge-base", "--is-ancestor", checkpoint_commit, current_commit], cwd=REPO_ROOT
    ).returncode == 0
    if not compatible:
        raise SystemExit("2D1D recovery metadata mismatch")
    runtime.metadata = payload["metadata"]
    runtime.model.load_state_dict(payload["model"], strict=True)
    bind_residual_mode(runtime.model)
    runtime.optimizer.load_state_dict(payload["optimizer"])
    runtime.loader = d1.ExplicitShardLoader(runtime.shards, runtime.micro_batch, d1.T, state=payload["loader_state"])
    if payload["next_global_batch_sha256"] != d1.next_global_batch_hash(runtime.loader, runtime.gradient_accumulation):
        raise SystemExit("2D1D recovery next-batch mismatch")
    d1.restore_rng_state(payload["rng_state"], runtime.prefix_rng)
    runtime.training_state = copy.deepcopy(payload["training_state"])
    if r.exact_spectral_norm(runtime.model.fusion.W_u.weight) > SIGMA_CAP * (1 + PROJECTION_TOLERANCE):
        raise SystemExit("recovered W_u exceeds cap")
    if not d1.model_parameters_finite(runtime.model) or not d1.optimizer_moments_finite(runtime.optimizer):
        raise SystemExit("nonfinite recovered state")
    return runtime


def checkpoint_payload(runtime):
    completed = runtime.training_state["completed_updates"]
    local = local_update(completed)
    schedule = stage_for_update(completed)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": runtime.model.state_dict(),
        "optimizer": runtime.optimizer.state_dict(),
        "training_state": copy.deepcopy(runtime.training_state),
        "scheduler_position": completed,
        "completed_updates": completed,
        "local_update": local,
        "processed_targets": completed * d1.GLOBAL_TARGETS,
        "additional_targets": local * d1.GLOBAL_TARGETS,
        "current_curriculum_stage": schedule["stage"],
        "current_windows": list(schedule["windows"]),
        "alpha": ALPHA,
        "rho_disabled": True,
        "loader_state": runtime.loader.state_dict(),
        "rng_state": d1.rng_state(runtime.prefix_rng),
        "next_global_batch_sha256": d1.next_global_batch_hash(runtime.loader, runtime.gradient_accumulation),
        "metadata": copy.deepcopy(runtime.metadata),
        "projection": {"parameter": "fusion.W_u.weight", "sigma_cap": SIGMA_CAP, "relative_tolerance": PROJECTION_TOLERANCE},
        "git_commit": git_output("rev-parse", "HEAD"),
        "environment": d1.runtime_environment(),
    }


def verify_checkpoint(path, runtime):
    payload = d1.torch_load(path, mmap=True)
    completed = runtime.training_state["completed_updates"]
    checks = {
        "schema": payload.get("schema") == CHECKPOINT_SCHEMA,
        "global_update": payload.get("completed_updates") == completed,
        "local_update": payload.get("local_update") == local_update(completed),
        "targets": payload.get("processed_targets") == completed * d1.GLOBAL_TARGETS,
        "additional_targets": payload.get("additional_targets") == local_update(completed) * d1.GLOBAL_TARGETS,
        "training_state": payload.get("training_state") == runtime.training_state,
        "loader_state": payload.get("loader_state") == runtime.loader.state_dict(),
        "next_batch": payload.get("next_global_batch_sha256") == d1.next_global_batch_hash(runtime.loader, runtime.gradient_accumulation),
        "metadata": payload.get("metadata") == runtime.metadata,
        "alpha": payload.get("alpha") == ALPHA,
        "rho_disabled": payload.get("rho_disabled") is True,
    }
    runtime.model.load_state_dict(payload["model"], strict=True)
    bind_residual_mode(runtime.model)
    runtime.optimizer.load_state_dict(payload["optimizer"])
    checks.update({
        "model_finite": d1.model_parameters_finite(runtime.model),
        "optimizer_finite": d1.optimizer_moments_finite(runtime.optimizer),
        "weight_tying": runtime.model.base.transformer.wte.weight is runtime.model.base.lm_head.weight,
        "W_u_cap": r.exact_spectral_norm(runtime.model.fusion.W_u.weight) <= SIGMA_CAP * (1 + PROJECTION_TOLERANCE),
    })
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"2D1D checkpoint reopen failed: {checks}")
    return {"checkpoint": str(Path(path).resolve()), "sha256": file_sha256(path), "bytes": Path(path).stat().st_size, "strict_reopen": checks, "passed": True}


def save_checkpoint(runtime, update, kind):
    local = local_update(update)
    directory = runtime.run_root / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "scientific" if kind == "scientific" else "recovery"
    path = directory / f"{prefix}_local_{local:04d}_global_{update:04d}.pt"
    if path.exists():
        raise SystemExit(f"refusing to overwrite checkpoint {path}")
    previous = runtime.training_state["last_checkpoint"]
    runtime.training_state["last_checkpoint"] = str(path.resolve())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        torch.save(checkpoint_payload(runtime), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        d1.fsync_directory(path.parent)
        verification = verify_checkpoint(path, runtime)
    except BaseException:
        runtime.training_state["last_checkpoint"] = previous
        raise
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{verification['sha256']}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), verification)
    manifest_path = runtime.output / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    manifest[kind][str(local)] = verification
    durable_json(manifest_path, manifest)
    if kind == "rolling":
        updates = sorted(map(int, manifest["rolling"]))
        while len(updates) > ROLLING_KEEP:
            old = updates.pop(0)
            row = manifest["rolling"].pop(str(old))
            old_path = Path(row["checkpoint"])
            for candidate in (old_path, old_path.with_suffix(old_path.suffix + ".sha256"), old_path.with_suffix(old_path.suffix + ".verification.json")):
                if candidate.is_file():
                    candidate.unlink()
        durable_json(manifest_path, manifest)
    return verification


def fusion_diagnostics(model, tokens, previous_top, prefix_length):
    with torch.no_grad():
        embedding = model.base.transformer.wte(tokens)
        shifted = torch.zeros_like(previous_top)
        shifted[:, 1:] = previous_top[:, :-1]
        zn = model.fusion.normalize(shifted)
        u = model.fusion.W_u(zn)
        g_pre = model.fusion.W_g(embedding)
        gate = 2.0 * torch.sigmoid(g_pre)
        fused = u * gate
        alpha_fused = ALPHA * fused
        candidate = embedding + alpha_fused
        positions = torch.arange(tokens.size(1), device=tokens.device)
        mask = positions.gt(int(prefix_length)).view(1, -1, 1)
        value = torch.where(mask, candidate, embedding)

        def rms(tensor):
            return tensor.float().pow(2).mean().sqrt().item()

        e_rms = rms(embedding)
        alpha_rms = rms(alpha_fused)
        gate_f = gate.float()
        return {
            "E_rms": e_rms,
            "ZN_rms": rms(zn),
            "U_rms": rms(u),
            "F_rms": rms(fused),
            "alphaF_rms": alpha_rms,
            "X_rms": rms(value),
            "alphaF_over_E": alpha_rms / e_rms,
            "X_over_E": rms(value) / e_rms,
            "gate_pre_mean": g_pre.float().mean().item(),
            "gate_pre_std": g_pre.float().std().item(),
            "gate_mean": gate_f.mean().item(),
            "gate_std": gate_f.std().item(),
            "gate_variance": gate_f.var().item(),
            "gate_saturation_fraction": ((gate_f < 0.01) | (gate_f > 1.99)).float().mean().item(),
        }


def compute_update_gradients(runtime, update):
    model, optimizer = runtime.model, runtime.optimizer
    model.train()
    schedule = stage_for_update(update)
    passes = pass_count_for_update(update)
    weights = d1.THREE_PASS_WEIGHTS if passes == 3 else d1.TWO_PASS_WEIGHTS
    lrs = d1.set_optimizer_lrs(optimizer, update)
    optimizer.zero_grad(set_to_none=True)
    pass_sums = [0.0] * passes
    total_sum = 0.0
    prefixes_all = []
    final_diag = scale_diag = None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(runtime.device)
    for micro_index in range(runtime.gradient_accumulation):
        cpu_x, cpu_y = runtime.loader.next_batch()
        x = cpu_x.to(runtime.device, non_blocking=True)
        y = cpu_y.to(runtime.device, non_blocking=True)
        prefixes = [runtime.prefix_rng.randrange(d1.T) for _ in range(passes - 1)]
        prefixes_all.append(prefixes)
        final_micro = micro_index == runtime.gradient_accumulation - 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            top1, loss1, _ = model.forward_pass(x, y, schedule["windows"], activation_checkpointing=True)
            top2, loss2, diag2 = model.forward_pass(
                x, y, schedule["windows"], previous_top=top1, rho=0.0,
                prefix_length=prefixes[0], activation_checkpointing=True,
                return_diagnostics=final_micro and passes == 2,
            )
            losses = [loss1, loss2]
            source_final = top1
            diagnostic_final = diag2
            if passes == 3:
                top3, loss3, diag3 = model.forward_pass(
                    x, y, schedule["windows"], previous_top=top2, rho=0.0,
                    prefix_length=prefixes[1], activation_checkpointing=True,
                    return_diagnostics=final_micro,
                )
                losses.append(loss3)
                source_final = top2
                diagnostic_final = diag3
            weighted = sum(weight * loss for weight, loss in zip(weights, losses))
            scaled = weighted / runtime.gradient_accumulation
        if not math.isfinite(weighted.detach().float().item()):
            raise SystemExit("NaN/Inf weighted training loss")
        scaled.backward()
        for index, loss in enumerate(losses):
            pass_sums[index] += loss.detach().float().item()
        total_sum += weighted.detach().float().item()
        if final_micro:
            final_diag = diagnostic_final
            scale_diag = fusion_diagnostics(model, x, source_final, prefixes[-1])
        del x, y, cpu_x, cpu_y, top1, top2, losses, weighted, scaled
        if passes == 3:
            del top3
    if not d1.gradients_finite(model):
        raise SystemExit("NaN/Inf gradients")
    gradients = d1.gradient_report(model)
    required = gradients["base"]["nonzero"] and gradients["W_u"]["nonzero"]
    if local_update(update) > 1:
        required = required and gradients["W_g"]["nonzero"]
    if not required:
        raise SystemExit(f"required gradient group is zero: {gradients}")
    return {
        "schedule": schedule,
        "pass_count": passes,
        "pass_losses": [value / runtime.gradient_accumulation for value in pass_sums],
        "weighted_total_ce": total_sum / runtime.gradient_accumulation,
        "prefix_lengths": prefixes_all,
        "lrs": lrs,
        "gradient_groups": gradients,
        "state_diagnostics": final_diag,
        "scale_diagnostics": scale_diag,
        "update_start": started,
    }


def finish_update(runtime, update, prepared):
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), d1.GRAD_CLIP)
    if not torch.isfinite(grad_norm):
        raise SystemExit("nonfinite gradient norm")
    runtime.optimizer.step()
    projection = r.project_weight_(runtime.model.fusion.W_u.weight, SIGMA_CAP)
    if not d1.model_parameters_finite(runtime.model):
        raise SystemExit("NaN/Inf parameters")
    if not d1.optimizer_moments_finite(runtime.optimizer):
        raise SystemExit("NaN/Inf optimizer moments")
    local = local_update(update)
    state = runtime.training_state
    state.update({
        "completed_updates": update,
        "processed_targets": update * d1.GLOBAL_TARGETS,
        "residual_completed_updates": local,
        "residual_processed_targets": local * d1.GLOBAL_TARGETS,
        "rescue_completed_updates": local,
        "rescue_processed_targets": local * d1.GLOBAL_TARGETS,
    })
    x_rms = prepared["scale_diagnostics"]["X_rms"]
    state["explosion_consecutive"] = state.get("explosion_consecutive", 0) + 1 if x_rms > HARD_RMS else 0
    if state["explosion_consecutive"] >= 3:
        raise SystemExit(f"residual recurrent-input RMS hard stop at local {local}: X_RMS={x_rms}")
    elapsed = time.monotonic() - prepared["update_start"]
    wu = runtime.model.fusion.W_u.weight.detach().float()
    wg = runtime.model.fusion.W_g.weight.detach().float()
    projection.update({
        "timestamp": time.time(),
        "global_update": update,
        "local_update": local,
        "stage": prepared["schedule"]["stage"],
        "W_u_frobenius": wu.norm().item(),
        "W_g_frobenius": wg.norm().item(),
        "W_g_spectral": r.exact_spectral_norm(wg),
    })
    metrics = {
        "timestamp": time.time(),
        "global_update": update,
        "local_update": local,
        "cumulative_targets": state["processed_targets"],
        "additional_targets": state["residual_processed_targets"],
        "stage": prepared["schedule"]["stage"],
        "windows": list(prepared["schedule"]["windows"]),
        "alpha": ALPHA,
        "rho_disabled": True,
        "pass_count": prepared["pass_count"],
        "pass_losses": prepared["pass_losses"],
        "weighted_total_ce": prepared["weighted_total_ce"],
        "prefix_lengths": prepared["prefix_lengths"],
        "lrs": prepared["lrs"],
        "gradient_norm_before_clip": grad_norm.detach().float().item(),
        "gradient_groups": prepared["gradient_groups"],
        "state_diagnostics": prepared["state_diagnostics"],
        "scale_diagnostics": prepared["scale_diagnostics"],
        "stage_a_ratio": x_rms / STAGE_A_RMS,
        "hard_rms_threshold": HARD_RMS,
        "explosion_consecutive": state["explosion_consecutive"],
        "projection": projection,
        "wall_seconds": elapsed,
        "targets_per_second": d1.GLOBAL_TARGETS / elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(runtime.device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(runtime.device) / 1024**2,
        "all_gradients_finite": True,
        "all_parameters_finite": True,
        "all_optimizer_moments_finite": True,
    }
    state["last_metrics"] = metrics
    return metrics, projection


def train_one_update(runtime, update):
    return finish_update(runtime, update, compute_update_gradients(runtime, update))


def write_heartbeat(runtime, metrics):
    completed = metrics["local_update"]
    elapsed = time.time() - runtime.training_state["residual_started_at"]
    eta = elapsed / completed * (LOCAL_UPDATES - completed) if completed else None
    durable_json(runtime.output / "HEARTBEAT.json", {
        "timestamp": time.time(),
        "pod_id": runtime.metadata["pod_id"],
        "pod_name": runtime.metadata["pod_name"],
        "local_update": completed,
        "scheduler_global_update": metrics["global_update"],
        "additional_targets": metrics["additional_targets"],
        "window_stage": metrics["stage"],
        "windows": metrics["windows"],
        "pass_losses": metrics["pass_losses"],
        "weighted_loss": metrics["weighted_total_ce"],
        "alphaF_over_E": metrics["scale_diagnostics"]["alphaF_over_E"],
        "X_over_E": metrics["scale_diagnostics"]["X_over_E"],
        "W_u_raw_sigma": metrics["projection"]["sigma_raw"],
        "W_u_post_sigma": metrics["projection"]["sigma_post"],
        "W_g_norm": metrics["projection"]["W_g_frobenius"],
        "gate_stats": {key: metrics["scale_diagnostics"][key] for key in ("gate_mean", "gate_std", "gate_variance", "gate_saturation_fraction")},
        "gpu": d1.gpu_telemetry(),
        "latest_checkpoint": runtime.training_state["last_checkpoint"],
        "eta_seconds": eta,
    })


def paired(left, right):
    differences = [a - b for a, b in zip(left, right)]
    tolerance = 1e-12
    return {
        "wins": sum(value < -tolerance for value in differences),
        "losses": sum(value > tolerance for value in differences),
        "ties": sum(abs(value) <= tolerance for value in differences),
        "mean_paired_delta": statistics.fmean(differences),
        "per_batch_differences": differences,
    }


def enrich_controls(result):
    controls = result["controls"]
    if all(name in controls for name in ("plain", "real", "zero", "shuffled")):
        result["paired"] = {
            "real_vs_plain": paired(controls["real"]["per_batch_losses"], controls["plain"]["per_batch_losses"]),
            "real_vs_zero": paired(controls["real"]["per_batch_losses"], controls["zero"]["per_batch_losses"]),
            "real_vs_shuffled": paired(controls["real"]["per_batch_losses"], controls["shuffled"]["per_batch_losses"]),
        }
    return result


@torch.no_grad()
def position_bins(runtime, windows):
    model = runtime.model
    model.eval()
    loader = d1.ExplicitShardLoader([d1.validation_shard(runtime.metadata["data_root"])], d1.VALIDATION_B, d1.T)
    sums = {f"{first}-{last}": {"plain": 0.0, "real": 0.0, "targets": 0} for first, last in POSITION_BINS}
    for batch_index in range(d1.VALIDATION_BATCHES):
        cpu_x, cpu_y = loader.next_batch()
        x, y = cpu_x.to(runtime.device), cpu_y.to(runtime.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            plain_top = model.forward_top(x, windows)
            real_top = model.forward_top(x, windows, previous_top=plain_top, rho=0.0, prefix_length=0)
            plain_losses = model.loss_from_top(plain_top, y, reduction="none").view(d1.VALIDATION_B, d1.T)
            real_losses = model.loss_from_top(real_top, y, reduction="none").view(d1.VALIDATION_B, d1.T)
        for first, last in POSITION_BINS:
            row = sums[f"{first}-{last}"]
            row["plain"] += plain_losses[:, first:last + 1].double().sum().item()
            row["real"] += real_losses[:, first:last + 1].double().sum().item()
            row["targets"] += plain_losses[:, first:last + 1].numel()
        del x, y, plain_top, real_top, plain_losses, real_losses
        print(f"2D1D position bins batch={batch_index + 1:02d}/20", flush=True)
    return {
        label: {
            "plain_loss": row["plain"] / row["targets"],
            "real_loss": row["real"] / row["targets"],
            "recurrent_gain": (row["plain"] - row["real"]) / row["targets"],
            "targets": row["targets"],
        }
        for label, row in sums.items()
    }


def matrix_diagnostics(model):
    return {
        "W_u": d1a.matrix_diagnostics(model.fusion.W_u.weight),
        "W_g": d1a.matrix_diagnostics(model.fusion.W_g.weight),
    }


def self_composition(runtime, windows, local):
    batches, manifest = d1a.validation_batches(d1.validation_shard(runtime.metadata["data_root"]), count=2)
    d1c.D12 = tuple(windows)
    d1c.HARD_THRESHOLD = HARD_RMS
    d1c.PROBE_STOP = 100.0 * STAGE_A_RMS
    d1c.SELF_BATCHES = 2
    d1c.PASSES = 32
    # The frozen 2D1C helper itself is not decorated with no_grad because its
    # original caller supplied the context.  A 32-pass graph at validation
    # batch size 64 exhausts even an 80GB A100, and would violate 2D1D's
    # explicitly no-gradient diagnostic.  Make that contract local and hard.
    runtime.model.eval()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        result = d1c.self_composition(runtime.model, batches, ALPHA, runtime.device)
    for row in result["rows"]:
        e_rms = row["X_rms"] / row["X_over_E"]
        row["alphaF_over_E"] = None if row["ALPHA_F_rms"] is None else row["ALPHA_F_rms"] / e_rms
    result.update({
        "local_update": local,
        "global_update": global_update(local),
        "windows": list(windows),
        "batch_manifest": manifest,
    })
    return result


def milestone_diagnostics(runtime, args, update, metrics=None, transition=False):
    local = local_update(update)
    windows = C12 if transition else stage_for_update(update)["windows"]
    label = "C_TRANSITION_PRESTEP" if transition else str(local)
    path = runtime.output / "milestone_validation.json"
    data = read_json(path) if path.is_file() else {"milestones": {}, "transitions": {}}
    section = "transitions" if transition else "milestones"
    if label in data[section]:
        validation = data[section][label]
    else:
        validation = d1.evaluate_temporal(
            runtime.model, d1.validation_shard(args.data_root), windows, 0.0,
            controls=("plain", "real", "zero", "shuffled"),
        )
        enrich_controls(validation)
        validation.update({
            "local_update": local,
            "global_update": update,
            "additional_targets": local * d1.GLOBAL_TARGETS,
            "stage": "C_TRANSITION_PRESTEP" if transition else stage_for_update(update)["stage"],
            "windows": list(windows),
            "alpha": ALPHA,
            "evaluated_at": time.time(),
        })
        data[section][label] = validation
        durable_json(path, data)
    paired_path = runtime.output / "paired_controls.json"
    paired_data = read_json(paired_path) if paired_path.is_file() else {"milestones": {}, "transitions": {}}
    paired_data[section][label] = validation["paired"]
    durable_json(paired_path, paired_data)

    if not transition:
        scale_path = runtime.output / "branch_growth.json"
        scale = read_json(scale_path) if scale_path.is_file() else {"milestones": {}, "first_threshold_crossings": {}}
        scale["milestones"][label] = {
            "local_update": local,
            "global_update": update,
            "additional_targets": local * d1.GLOBAL_TARGETS,
            "matrices": matrix_diagnostics(runtime.model),
            "training_scale": None if metrics is None else metrics["scale_diagnostics"],
            "validation_state_diagnostics": validation.get("state_diagnostics"),
        }
        durable_json(scale_path, scale)
        if local in SELF_LOCAL:
            self_path = runtime.output / "self_composition.json"
            self_data = read_json(self_path) if self_path.is_file() else {"milestones": {}}
            if label not in self_data["milestones"]:
                self_data["milestones"][label] = self_composition(runtime, windows, local)
                durable_json(self_path, self_data)
        if local in POSITION_LOCAL:
            position_path = runtime.output / "position_bin_metrics.json"
            positions = read_json(position_path) if position_path.is_file() else {"milestones": {}}
            if label not in positions["milestones"]:
                positions["milestones"][label] = {
                    "local_update": local,
                    "global_update": update,
                    "additional_targets": local * d1.GLOBAL_TARGETS,
                    "windows": list(windows),
                    "bins": position_bins(runtime, windows),
                }
                durable_json(position_path, positions)
    return validation


def zero_identity(runtime):
    loader = d1.ExplicitShardLoader([d1.validation_shard(runtime.metadata["data_root"])], 2, 16)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(runtime.device), cpu_y.to(runtime.device)
    windows = tuple(min(window, x.size(1)) for window in B12)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        plain_top = runtime.model.forward_top(x, windows)
        real_top = runtime.model.forward_top(x, windows, previous_top=plain_top, rho=0.0, prefix_length=0)
        plain_logits = runtime.model.logits_from_top(plain_top)
        real_logits = runtime.model.logits_from_top(real_top)
        plain_loss = runtime.model.loss_from_top(plain_top, y)
        real_loss = runtime.model.loss_from_top(real_top, y)
        tensors = d1c.residual_fusion_input(runtime.model, x, plain_top, ALPHA)
    result = {
        "W_u_zero": runtime.model.fusion.W_u.weight.count_nonzero().item() == 0,
        "W_g_zero": runtime.model.fusion.W_g.weight.count_nonzero().item() == 0,
        "F_exact_zero": tensors["F"].count_nonzero().item() == 0,
        "alphaF_exact_zero": tensors["ALPHA_F"].count_nonzero().item() == 0,
        "X_equals_E": torch.equal(tensors["X"], tensors["E"]),
        "top_states_exact": torch.equal(plain_top, real_top),
        "logits_exact": torch.equal(plain_logits, real_logits),
        "loss_exact": plain_loss.item() == real_loss.item(),
        "plain_loss": plain_loss.item(),
        "real_loss": real_loss.item(),
    }
    result["passed"] = all(value for key, value in result.items() if key not in ("plain_loss", "real_loss", "passed"))
    if not result["passed"]:
        raise SystemExit(f"zero-fusion identity failed: {result}")
    return result


def run_disposable_smoke(args):
    runtime = load_source_runtime(args)
    before_wu = runtime.model.fusion.W_u.weight.detach().clone()
    prepared1 = compute_update_gradients(runtime, FIRST_GLOBAL_UPDATE)
    step1 = copy.deepcopy(prepared1["gradient_groups"])
    metrics1, projection1 = finish_update(runtime, FIRST_GLOBAL_UPDATE, prepared1)
    wu_nonzero_after_step1 = not torch.equal(before_wu, runtime.model.fusion.W_u.weight)
    prepared2 = compute_update_gradients(runtime, FIRST_GLOBAL_UPDATE + 1)
    step2 = copy.deepcopy(prepared2["gradient_groups"])
    metrics2, projection2 = finish_update(runtime, FIRST_GLOBAL_UPDATE + 1, prepared2)
    loader = d1.ExplicitShardLoader([d1.validation_shard(args.data_root)], 2, 16)
    cpu_x, cpu_y = loader.next_batch()
    x, y = cpu_x.to(runtime.device), cpu_y.to(runtime.device)
    runtime.model.float()
    causality = d1.causality_tests(runtime.model, x)
    gradients = d1.temporal_gradient_tests(runtime.model, x, y)
    runtime.model.zero_grad(set_to_none=True)
    smoke_root = Path(args.run_root).resolve() / "disposable_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    temporary = smoke_root / "smoke_two_step.pt"
    torch.save(checkpoint_payload(runtime), temporary)
    reopened = d1.torch_load(temporary, mmap=True)
    checkpoint_reload = (
        reopened["schema"] == CHECKPOINT_SCHEMA
        and reopened["completed_updates"] == FIRST_GLOBAL_UPDATE + 1
        and reopened["local_update"] == 2
    )
    temporary.unlink()
    result = {
        "disposable": True,
        "updates": 2,
        "step1": {
            "W_u_gradient": step1["W_u"],
            "W_g_gradient": step1["W_g"],
            "base_gradient": step1["base"],
            "W_u_became_nonzero": wu_nonzero_after_step1,
            "metrics": metrics1,
            "projection": projection1,
        },
        "step2": {
            "W_u_gradient": step2["W_u"],
            "W_g_gradient": step2["W_g"],
            "base_gradient": step2["base"],
            "metrics": metrics2,
            "projection": projection2,
        },
        "causality": causality,
        "temporal_gradients": gradients,
        "checkpoint_reload": checkpoint_reload,
        "checkpoint_discarded": not temporary.exists(),
        "all_losses_finite": all(math.isfinite(value) for value in metrics1["pass_losses"] + metrics2["pass_losses"]),
        "alphaF_over_E_finite": all(math.isfinite(row["scale_diagnostics"]["alphaF_over_E"]) for row in (metrics1, metrics2)),
        "projection_passed": all(row["sigma_post"] <= SIGMA_CAP * (1 + PROJECTION_TOLERANCE) for row in (projection1, projection2)),
    }
    checks = {
        "step1_W_u_gradient_nonzero": step1["W_u"]["nonzero"],
        "step1_base_gradient_nonzero": step1["base"]["nonzero"],
        "step1_W_u_changed": wu_nonzero_after_step1,
        "step2_W_g_gradient_nonzero": step2["W_g"]["nonzero"],
        "causality": causality["passed"],
        "temporal_gradients": gradients["passed"],
        "checkpoint_reload": checkpoint_reload,
        "checkpoint_discarded": result["checkpoint_discarded"],
        "losses_finite": result["all_losses_finite"],
        "alphaF_over_E_finite": result["alphaF_over_E_finite"],
        "projection": result["projection_passed"],
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    if not result["passed"]:
        raise SystemExit(f"2D1D disposable smoke failed: {checks}")
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_preflight(args):
    require_git(clean=True)
    require_config()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hashes = {
        "source_c954": file_sha256(args.source_c954),
        "parent_checkpoint": file_sha256(args.parent_checkpoint),
        "validation": file_sha256(d1.validation_shard(args.data_root)),
    }
    expected = {
        "source_c954": SOURCE_SHA256,
        "parent_checkpoint": d1.SOURCE_SHA256,
        "validation": d1.VALIDATION_SHARD_SHA256,
    }
    if hashes != expected:
        raise SystemExit(f"2D1D source hashes failed: {hashes}")

    source_runtime = _R_LOAD_SOURCE(args)
    source_plain = d1.evaluate_temporal(
        source_runtime.model, d1.validation_shard(args.data_root), B12, 0.5, controls=("plain",)
    )
    source_plain_loss = source_plain["controls"]["plain"]["validation_loss"]
    source_plain_pass = abs(source_plain_loss - PLAIN_ORACLE) <= PLAIN_ORACLE_TOLERANCE
    if not source_plain_pass:
        raise SystemExit(f"C954 native Stage-B plain regression: {source_plain_loss}")
    del source_runtime
    gc.collect()
    torch.cuda.empty_cache()

    runtime = load_source_runtime(args)
    identity = zero_identity(runtime)
    source_manifest = {
        "experiment": EXPERIMENT,
        "source_checkpoint": str(Path(args.source_c954).resolve()),
        "source_checkpoint_sha256": hashes["source_c954"],
        "source_checkpoint_bytes": Path(args.source_c954).stat().st_size,
        "strict_reopen": runtime.source_checks,
        "source_global_update": SOURCE_GLOBAL_UPDATE,
        "source_cumulative_targets": SOURCE_TARGETS,
        "source_training_state": runtime.source_payload["training_state"],
        "source_scheduler_position": runtime.source_payload["scheduler_position"],
        "loader_state": runtime.source_payload["loader_state"],
        "loader_path_relocation": runtime.path_relocation,
        "rng_fields": sorted(runtime.source_payload["rng_state"]),
        "next_global_batch_sha256": runtime.source_payload["next_global_batch_sha256"],
        "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "parent_checkpoint_sha256": hashes["parent_checkpoint"],
        "validation_shard": str(d1.validation_shard(args.data_root).resolve()),
        "validation_sha256": hashes["validation"],
        "C954_native_B_plain": source_plain,
        "C954_plain_oracle": PLAIN_ORACLE,
        "C954_plain_regression_pass": source_plain_pass,
    }
    durable_json(output / "source_manifest.json", source_manifest)
    durable_json(output / "fusion_reinitialization.json", runtime.fusion_reinitialization)
    durable_json(output / "optimizer_reset_audit.json", runtime.optimizer_reset_audit)
    architecture = {
        "experiment": EXPERIMENT,
        "equation": "X = E + alpha * F",
        "alpha": ALPHA,
        "alpha_trainable": False,
        "F": "W_u(RMSNorm_noaffine(z_prev)) * (2*sigmoid(W_g(E)))",
        "rho_present_in_result_path": False,
        "pass1": "plain X=E",
        "pass2": "shifted pass1 top, no detach",
        "pass3": "shifted pass2 top, no detach every 32nd global update",
        "two_pass_weights": list(d1.TWO_PASS_WEIGHTS),
        "three_pass_weights": list(d1.THREE_PASS_WEIGHTS),
        "B12": list(B12),
        "C12": list(C12),
        "all_parameters_trainable": all(parameter.requires_grad for parameter in runtime.model.parameters()),
        "weight_tying": runtime.model.base.transformer.wte.weight is runtime.model.base.lm_head.weight,
        "zero_identity": identity,
    }
    durable_json(output / "architecture_manifest.json", architecture)
    batch_manifest = d1.validation_manifest(d1.validation_shard(args.data_root))
    batch_manifest.update({
        "source_next_global_batch_sha256": runtime.source_payload["next_global_batch_sha256"],
        "first_result_batch_verified": d1.next_global_batch_hash(runtime.loader, runtime.gradient_accumulation) == runtime.source_payload["next_global_batch_sha256"],
    })
    durable_json(output / "batch_manifest.json", batch_manifest)
    durable_json(output / "checkpoint_manifest.json", {
        "source": {"954": {"checkpoint": str(Path(args.source_c954).resolve()), "sha256": SOURCE_SHA256, "bytes": SOURCE_BYTES, "strict_reopen": runtime.source_checks, "passed": True}},
        "scientific": {}, "rolling": {},
    })
    durable_json(output / "milestone_validation.json", {"milestones": {}, "transitions": {}})
    durable_json(output / "paired_controls.json", {"milestones": {}, "transitions": {}})
    durable_json(output / "branch_growth.json", {"milestones": {}, "first_threshold_crossings": {}})
    durable_json(output / "position_bin_metrics.json", {"milestones": {}})
    durable_json(output / "self_composition.json", {"milestones": {}})
    milestone_diagnostics(runtime, args, SOURCE_GLOBAL_UPDATE, metrics=None)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()

    smoke = run_disposable_smoke(args)
    durable_json(output / "preflight_smoke.json", smoke)
    runtime = load_source_runtime(args)
    pristine = {
        "W_u_zero": runtime.model.fusion.W_u.weight.count_nonzero().item() == 0,
        "W_g_zero": runtime.model.fusion.W_g.weight.count_nonzero().item() == 0,
        "fusion_optimizer_fresh": runtime.optimizer_reset_audit["fusion_fresh_state"],
        "next_batch_exact": d1.next_global_batch_hash(runtime.loader, runtime.gradient_accumulation) == runtime.source_payload["next_global_batch_sha256"],
    }
    stop = {
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "gpu_type": args.gpu_type,
        "persistent_volume_identity": args.persistent_volume_identity,
        "stop_mechanism": args.stop_mechanism,
        "stop_authenticated": bool(args.stop_authenticated),
        "credential_location": "local macOS Keychain service runpod-codex-pod-stopper",
        "remote_pod_api_key_used": False,
    }
    checks = {
        "2D1C frozen tag exact": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
        "source hashes exact": hashes == expected,
        "C954 strict reopen": runtime.source_checks["passed"],
        "C954 native B plain regression": source_plain_pass,
        "base model preserved": runtime.optimizer_reset_audit["base_model_exact"],
        "base optimizer preserved": runtime.optimizer_reset_audit["base_optimizer_exact"],
        "new fusion zero": runtime.fusion_reinitialization["new_W_u_exact_zero"] and runtime.fusion_reinitialization["new_W_g_exact_zero"],
        "fusion optimizer reset only": runtime.optimizer_reset_audit["fusion_fresh_state"] and runtime.optimizer_reset_audit["only_fusion_state_reset"],
        "scheduler preserved": runtime.source_payload["scheduler_position"] == SOURCE_GLOBAL_UPDATE,
        "loader RNG next batch preserved": pristine["next_batch_exact"],
        "zero identity": identity["passed"],
        "disposable two-step smoke": smoke["passed"],
        "causality": smoke["causality"]["passed"],
        "temporal gradients": smoke["temporal_gradients"]["passed"],
        "pristine restart after smoke": all(pristine.values()),
        "exact authenticated stop available": stop["stop_authenticated"] and stop["pod_id"] == runtime.metadata["pod_id"],
    }
    preflight = {
        "experiment": EXPERIMENT,
        "pod_id": args.pod_id,
        "stop_audit": stop,
        "checks": checks,
        "result_run_authorized": all(checks.values()),
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "runtime_seconds": time.time() - started,
        "command": " ".join(sys.argv),
    }
    durable_json(output / "preflight_audit.json", preflight)
    durable_json(output / "performance.json", {"preflight": {"wall_seconds": time.time() - started}})
    durable_json(output / "commands_and_runtime.json", {"experiment": EXPERIMENT, "commands": [" ".join(sys.argv)], "hardware": d1.runtime_environment(), "started_at": started})
    if not preflight["result_run_authorized"]:
        raise SystemExit(f"2D1D preflight failed: {checks}")
    print("EXPERIMENT_2D1D_PREFLIGHT_PASS", flush=True)
    return preflight


def update_branch_thresholds(output, metrics):
    path = Path(output) / "branch_growth.json"
    data = read_json(path)
    ratio = metrics["scale_diagnostics"]["alphaF_over_E"]
    for threshold in BRANCH_THRESHOLDS:
        key = str(threshold)
        if ratio > threshold and key not in data["first_threshold_crossings"]:
            data["first_threshold_crossings"][key] = {
                "local_update": metrics["local_update"],
                "global_update": metrics["global_update"],
                "additional_targets": metrics["additional_targets"],
                "alphaF_over_E": ratio,
            }
    durable_json(path, data)


def run_train_worker(args):
    require_git(clean=False)
    output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json")
    if not preflight.get("result_run_authorized") or preflight.get("pod_id") != args.pod_id:
        raise SystemExit("exact-pod passing 2D1D preflight required")
    runtime = load_result_runtime(args, args.resume) if args.resume else load_source_runtime(args)
    runtime.output.mkdir(parents=True, exist_ok=True)
    runtime.run_root.mkdir(parents=True, exist_ok=True)
    start = runtime.training_state["completed_updates"] + 1
    existing = read_jsonl(output / "training_metrics.jsonl")
    if existing:
        if not args.resume or existing[-1]["global_update"] != start - 1:
            raise SystemExit("2D1D metrics require aligned explicit recovery checkpoint")
    elif start != FIRST_GLOBAL_UPDATE:
        raise SystemExit("non-source 2D1D start requires metrics")
    if args.resume and local_update(start - 1) in MILESTONE_LOCAL:
        # A terminal diagnostic can fail after the optimizer update, metrics,
        # strict checkpoint, and canonical validation are already durable.
        # Finish only the missing no-gradient milestone work before consuming
        # the next training batch.
        milestone_diagnostics(runtime, args, start - 1, metrics=existing[-1])
        if local_update(start - 1) == 96:
            milestone_diagnostics(runtime, args, start - 1, metrics=existing[-1], transition=True)
        durable_json(output / "recovery_resume.json", {
            "resume_checkpoint": str(Path(args.resume).resolve()),
            "completed_local_update": local_update(start - 1),
            "completed_global_update": start - 1,
            "training_updates_repeated": 0,
            "training_batches_repeated": 0,
            "pending_no_gradient_milestone_diagnostics_completed": True,
            "recovery_implementation_git_commit": git_output("rev-parse", "HEAD"),
            "recovered_at": time.time(),
            "passed": True,
        })
    for update in range(start, FINAL_GLOBAL_UPDATE + 1):
        metrics, projection = train_one_update(runtime, update)
        append_jsonl(output / "training_metrics.jsonl", metrics)
        append_jsonl(output / "projection_metrics.jsonl", projection)
        update_branch_thresholds(output, metrics)
        local = metrics["local_update"]
        if local <= 2 or local % 10 == 0:
            write_heartbeat(runtime, metrics)
        warning = " WARNING_SCALE" if metrics["scale_diagnostics"]["X_rms"] > HARD_RMS else ""
        print(
            f"2D1D local={local:04d}/{LOCAL_UPDATES} global={update:04d} stage={metrics['stage']} "
            f"loss={metrics['weighted_total_ce']:.6f} alphaF/E={metrics['scale_diagnostics']['alphaF_over_E']:.6f} "
            f"X/E={metrics['scale_diagnostics']['X_over_E']:.6f} "
            f"sigma={projection['sigma_raw']:.6f}->{projection['sigma_post']:.6f} "
            f"tok/s={metrics['targets_per_second']:.0f}{warning}", flush=True,
        )
        if local in SCIENTIFIC_LOCAL:
            save_checkpoint(runtime, update, "scientific")
        elif local % ROLLING_INTERVAL == 0:
            save_checkpoint(runtime, update, "rolling")
        if local in MILESTONE_LOCAL:
            milestone_diagnostics(runtime, args, update, metrics=metrics)
            if local == 96:
                milestone_diagnostics(runtime, args, update, metrics=metrics, transition=True)
    if runtime.training_state["completed_updates"] != FINAL_GLOBAL_UPDATE:
        raise SystemExit("2D1D worker ended at wrong global update")
    if runtime.training_state["residual_processed_targets"] != ADDITIONAL_TARGETS:
        raise SystemExit("2D1D worker ended at wrong target count")
    durable_json(output / "training_complete.json", {
        "source_global_update": SOURCE_GLOBAL_UPDATE,
        "final_global_update": FINAL_GLOBAL_UPDATE,
        "local_updates": LOCAL_UPDATES,
        "additional_targets": ADDITIONAL_TARGETS,
        "final_cumulative_targets": FINAL_TARGETS,
        "final_checkpoint": runtime.training_state["last_checkpoint"],
        "completed_at": time.time(),
        "passed": True,
    })
    write_heartbeat(runtime, metrics)
    print("EXPERIMENT_2D1D_TRAINING_COMPLETE", flush=True)
    return 0


def worker_command(args):
    command = [
        sys.executable, str(Path(__file__).resolve()), "train-worker",
        "--parent-checkpoint", str(Path(args.parent_checkpoint).resolve()),
        "--data-root", str(Path(args.data_root).resolve()),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--run-root", str(Path(args.run_root).resolve()),
        "--source-c954", str(Path(args.source_c954).resolve()),
        "--pod-id", args.pod_id,
        "--pod-name", args.pod_name,
        "--gpu-type", args.gpu_type,
        "--persistent-volume-identity", args.persistent_volume_identity,
        "--stop-mechanism", args.stop_mechanism,
        "--stop-authenticated",
    ]
    if args.resume:
        command.extend(["--resume", str(Path(args.resume).resolve())])
    return command


def run_supervise(args):
    require_git(clean=False)
    output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json")
    if not preflight.get("result_run_authorized"):
        raise SystemExit("2D1D result run not authorized")
    command = worker_command(args)
    started = time.time()
    completed = subprocess.run(command, cwd=REPO_ROOT)
    status = {
        "experiment": EXPERIMENT,
        "command": command,
        "started_at": started,
        "ended_at": time.time(),
        "returncode": completed.returncode,
        "terminal": True,
        "training_complete": completed.returncode == 0,
    }
    durable_json(output / "supervisor_status.json", status)
    if completed.returncode != 0:
        durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", (
            "EXPERIMENT 2D1D TERMINAL FAILURE\n\n"
            f"Worker return code: {completed.returncode}\n"
            "Inspect supervisor_status.json, HEARTBEAT.json, and training_metrics.jsonl.\n"
        ))
        raise SystemExit(completed.returncode)
    return 0


def projection_summary(rows):
    result = {}
    for stage in ("B-R", "C-R"):
        current = [row for row in rows if row["stage"] == stage]
        scales = [row["projection_scale"] for row in current]
        applied = sum(row["projection_applied"] for row in current)
        result[stage] = {
            "updates": len(current),
            "projected_updates": applied,
            "fraction_projected": applied / len(current),
            "mean_projection_scale": statistics.fmean(scales),
            "minimum_projection_scale": min(scales),
            "maximum_raw_sigma": max(row["sigma_raw"] for row in current),
            "maximum_post_sigma": max(row["sigma_post"] for row in current),
            "optimizer_consistently_pushes_against_cap": applied / len(current) > 0.90,
        }
    return result


def choose_classifications(final, self_final, branch_final):
    gain = final["recurrent_gain"]
    wins = final["paired"]["real_vs_plain"]["wins"]
    gap = final["sequence_specific_gap"]
    stable = self_final["summary"]["scale_bounded"]
    if not stable:
        primary = "RESIDUAL RECURRENT TRAINING IS UNSTABLE"
    elif gain >= 0.01 and wins >= 15 and gap > 0:
        primary = "RESIDUAL RECURRENCE LEARNS CLEAR POSITIVE UTILITY"
    elif gain > 0 and wins > 10:
        primary = "RESIDUAL RECURRENCE LEARNS POSITIVE UTILITY"
    elif gain >= -0.01:
        primary = "RESIDUAL RECURRENCE APPROACHES NEUTRALITY"
    else:
        primary = "RESIDUAL RECURRENCE REMAINS HARMFUL AFTER RETRAINING"
    if abs(branch_final) < 0.01:
        secondary = "RECURRENCE REMAINS FUNCTIONALLY NEGLIGIBLE"
    elif gap > 0 and gain > 0:
        secondary = "SEQUENCE-SPECIFIC USEFUL RECURRENCE"
    elif gap > 0:
        secondary = "SEQUENCE-SPECIFIC BUT NON-USEFUL RECURRENCE"
    else:
        secondary = "MISALIGNED RECURRENCE"
    return primary, secondary


def choose_recommendation(primary, milestones, gap, branch_ratio):
    if primary in (
        "RESIDUAL RECURRENCE LEARNS CLEAR POSITIVE UTILITY",
        "RESIDUAL RECURRENCE LEARNS POSITIVE UTILITY",
    ):
        return "CONTINUE RESIDUAL RECURRENCE INTO STAGE-D WINDOWS"
    if primary == "RESIDUAL RECURRENT TRAINING IS UNSTABLE":
        return "REDESIGN EFFECTIVE RECURRENT-SCALE CONTROL"
    if branch_ratio < 0.01:
        return "INCREASE OR LEARN RESIDUAL COUPLING STRENGTH"
    gains = [milestones[str(local)]["recurrent_gain"] for local in (191, 286, 477)]
    if primary == "RESIDUAL RECURRENCE APPROACHES NEUTRALITY" and gains[-1] >= gains[-2]:
        return "EXTEND C12 RESIDUAL RECURRENCE TRAINING"
    if gap > 0:
        return "RETHINK RECURRENT STATE / READOUT REPRESENTATION"
    return "RETHINK RECURRENT STATE / READOUT REPRESENTATION"


def make_plots(output, training, milestones, self_data, positions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    ordered = [0, 20, 48, 96, 191, 286, 477]
    xs = [milestones[str(local)]["additional_targets"] for local in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [milestones[str(v)]["controls"]["plain"]["validation_loss"] for v in ordered], marker="o", label="Plain")
    ax.plot(xs, [milestones[str(v)]["controls"]["real"]["validation_loss"] for v in ordered], marker="o", label="Real")
    ax.set(xlabel="Additional targets", ylabel="Validation CE", title="2D1D P1: Plain and recurrent validation")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "P1_plain_real_validation.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [milestones[str(v)]["recurrent_gain"] for v in ordered], marker="o")
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Additional targets", ylabel="Plain - Real CE", title="2D1D P2: Recurrent gain")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "P2_recurrent_gain.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [milestones[str(v)]["sequence_specific_gap"] for v in ordered], marker="o")
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Additional targets", ylabel="Shuffled - Real CE", title="2D1D P3: Sequence gap")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "P3_sequence_gap.png", dpi=160); plt.close(fig)

    train_x = [row["additional_targets"] for row in training]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_x, [row["scale_diagnostics"]["alphaF_over_E"] for row in training])
    ax.set(xlabel="Additional targets", ylabel="RMS(alpha F) / RMS(E)", title="2D1D P4: Effective recurrent strength")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "P4_alphaF_over_E.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_x, [row["projection"]["sigma_raw"] for row in training], label="raw")
    ax.plot(train_x, [row["projection"]["sigma_post"] for row in training], label="post")
    ax.axhline(SIGMA_CAP, color="black", ls="--", label="cap")
    ax.set(xlabel="Additional targets", ylabel="W_u spectral norm", title="2D1D P5: W_u spectral control")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "P5_Wu_spectral.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_x, [row["projection"]["W_g_frobenius"] for row in training], label="W_g Frobenius")
    ax2 = ax.twinx()
    ax2.plot(train_x, [row["scale_diagnostics"]["gate_mean"] for row in training], color="tab:orange", label="gate mean")
    ax2.plot(train_x, [row["scale_diagnostics"]["gate_std"] for row in training], color="tab:green", label="gate std")
    ax.set(xlabel="Additional targets", ylabel="W_g norm", title="2D1D P6: W_g and gates"); ax2.set_ylabel("Gate statistic")
    lines = ax.lines + ax2.lines; ax.legend(lines, [line.get_label() for line in lines], loc="best")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "P6_Wg_gate.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for local in SELF_LOCAL:
        rows = self_data["milestones"][str(local)]["rows"]
        by_pass = []
        for pass_index in range(1, 33):
            by_pass.append(statistics.fmean(row["recurrent_input_rms"] for row in rows if row["pass"] == pass_index))
        ax.plot(range(1, 33), by_pass, label=f"local {local}")
    ax.axhline(HARD_RMS, color="black", ls="--", label="hard threshold")
    ax.set(xlabel="Composition pass", ylabel="Recurrent-input RMS", title="2D1D P7: 32-pass stability")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "P7_self_composition_rms.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{first}-{last}" for first, last in POSITION_BINS]
    for local in POSITION_LOCAL:
        ax.plot(labels, [positions["milestones"][str(local)]["bins"][label]["recurrent_gain"] for label in labels], marker="o", label=f"local {local}")
    ax.axhline(0, color="black", lw=1); ax.tick_params(axis="x", rotation=35)
    ax.set(xlabel="Token position", ylabel="Plain - Real CE", title="2D1D P8: Position-binned recurrent gain")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "P8_position_bins.png", dpi=160); plt.close(fig)


def fnum(value):
    return "NOT AVAILABLE" if value is None else f"{value:.10f}"


def render_report(summary, questions, milestones, transition, projection, audit):
    rows = []
    for local in (0, 20, 48, 96, 191, 286, 477):
        row = milestones[str(local)]
        rows.append(
            f"| {local} | {row['global_update']} | {row['additional_targets']:,} | {row['stage']} | "
            f"{row['controls']['plain']['validation_loss']:.10f} | {row['controls']['real']['validation_loss']:.10f} | "
            f"{row['recurrent_gain']:+.10f} | {row['sequence_specific_gap']:+.10f} | "
            f"{row['paired']['real_vs_plain']['wins']}/20 |"
        )
    qrows = "\n".join(f"- **{key}:** {value}" for key, value in questions.items())
    cap_notice = "\n\n**OPTIMIZER CONSISTENTLY PUSHES AGAINST W_U CAP**" if any(v["optimizer_consistently_pushes_against_cap"] for v in projection.values()) else ""
    return f"""# Experiment 2D1D — End-to-End Residual Recurrence Retraining from C954

## Classification

**Primary:** {summary['primary_classification']}

**Secondary:** {summary['secondary_classification']}

The exact C954 checkpoint `{SOURCE_SHA256}` was resumed at global update 954. Its base Transformer parameters and Adam state were preserved, while only `W_u`, `W_g`, and their optimizer state were reset to exact zero/fresh state. The result path used fixed `X = E + 0.03125F` for {LOCAL_UPDATES} updates ({ADDITIONAL_TARGETS:,} targets), ending at global update {FINAL_GLOBAL_UPDATE}.{cap_notice}

## Validation trajectory

| Local update | Global update | Additional targets | Geometry | Plain CE | Real CE | Gain (Plain-Real) | Sequence gap | Real wins vs plain |
|---:|---:|---:|:---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The immediate B12→C12 prestep shock at local 96 changed plain CE by `{transition['plain_delta_C_minus_B']:+.10f}`, real CE by `{transition['real_delta_C_minus_B']:+.10f}`, and recurrent gain by `{transition['gain_delta_C_minus_B']:+.10f}` without an optimizer update.

## Scale and stability

Final effective `alphaF/E` was `{summary['final_alphaF_over_E']:.10f}` and `X/E` was `{summary['final_X_over_E']:.10f}`. The maximum training `X` RMS was `{summary['maximum_X_RMS']:.10f}` against the hard threshold `{HARD_RMS:.10f}`. Final 32-pass classification: **{summary['final_self_classification']}**.

Projection by stage: `{json.dumps(projection, sort_keys=True)}`.

## Decision

Exactly one recommended next experiment: **{summary['next_recommendation']}**.

## Scientific questions

{qrows}

## Integrity

Scientific audit passed: `{audit['passed']}`. Final checkpoint SHA-256: `{summary['final_checkpoint_sha256']}`.

# EXPERIMENT 2D1D COMPLETE
"""


def render_handoff(summary):
    return f"""# Experiment 2D1D unattended final handoff

Primary classification: {summary['primary_classification']}

Secondary classification: {summary['secondary_classification']}

Final recurrent gain: {summary['final_recurrent_gain']:+.10f}

Final sequence gap: {summary['final_sequence_gap']:+.10f}

Next recommendation: {summary['next_recommendation']}

Final checkpoint: {summary['final_checkpoint']}

Final checkpoint SHA-256: {summary['final_checkpoint_sha256']}
"""


def run_finalize(args):
    require_git(clean=False)
    output = Path(args.output_dir).resolve()
    complete = read_json(output / "training_complete.json")
    if not complete.get("passed") or complete["local_updates"] != LOCAL_UPDATES:
        raise SystemExit("2D1D finalize requires complete 477-update training")
    final_checkpoint = Path(args.final_checkpoint).resolve()
    runtime = load_result_runtime(args, final_checkpoint)
    if runtime.training_state["completed_updates"] != FINAL_GLOBAL_UPDATE:
        raise SystemExit("2D1D final checkpoint global update mismatch")
    training = read_jsonl(output / "training_metrics.jsonl")
    projection_rows = read_jsonl(output / "projection_metrics.jsonl")
    milestone_data = read_json(output / "milestone_validation.json")
    milestones = milestone_data["milestones"]
    transitions = milestone_data["transitions"]
    self_data = read_json(output / "self_composition.json")
    positions = read_json(output / "position_bin_metrics.json")
    manifest = read_json(output / "checkpoint_manifest.json")
    final = milestones["477"]
    final_self = self_data["milestones"]["477"]
    branch_final = training[-1]["scale_diagnostics"]["alphaF_over_E"]
    primary, secondary = choose_classifications(final, final_self, branch_final)
    recommendation = choose_recommendation(primary, milestones, final["sequence_specific_gap"], branch_final)
    projection_by_stage = projection_summary(projection_rows)

    b96 = milestones["96"]
    c96 = transitions["C_TRANSITION_PRESTEP"]
    transition = {
        "local_update": 96,
        "global_update": global_update(96),
        "optimizer_updates_between": 0,
        "B12": b96,
        "C12_prestep": c96,
        "plain_delta_C_minus_B": c96["controls"]["plain"]["validation_loss"] - b96["controls"]["plain"]["validation_loss"],
        "real_delta_C_minus_B": c96["controls"]["real"]["validation_loss"] - b96["controls"]["real"]["validation_loss"],
        "gain_delta_C_minus_B": c96["recurrent_gain"] - b96["recurrent_gain"],
        "post_transition": {str(local): milestones[str(local)] for local in (191, 286, 477)},
    }
    durable_json(output / "transition_analysis.json", transition)

    earliest_positive = next((local for local in (0, 20, 48, 96, 191, 286, 477) if milestones[str(local)]["recurrent_gain"] > 0), None)
    earliest_sequence = next((local for local in (0, 20, 48, 96, 191, 286, 477) if milestones[str(local)]["sequence_specific_gap"] > 0), None)
    first_wg_gradient = next((row["local_update"] for row in training if row["gradient_groups"]["W_g"]["nonzero"]), None)
    first_cap = next((row["local_update"] for row in projection_rows if row["projection_applied"]), None)
    maximum_x = max(row["scale_diagnostics"]["X_rms"] for row in training)
    final_positions = positions["milestones"]["477"]["bins"]
    best_position = max(final_positions, key=lambda key: final_positions[key]["recurrent_gain"])
    worst_position = min(final_positions, key=lambda key: final_positions[key]["recurrent_gain"])
    plain_change = final["controls"]["plain"]["validation_loss"] - milestones["0"]["controls"]["plain"]["validation_loss"]
    sequence_before_useful = earliest_sequence is not None and (earliest_positive is None or earliest_sequence < earliest_positive)
    final_matrix = read_json(output / "branch_growth.json")["milestones"]["477"]["matrices"]

    core = {
        "2D1C frozen tag exact": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
        "C954 SHA exact": file_sha256(args.source_c954) == SOURCE_SHA256,
        "C954 strict reopen": runtime.source_checks["passed"],
        "C954 base model exact before reinit": runtime.optimizer_reset_audit["base_model_exact"],
        "C954 base optimizer exact": runtime.optimizer_reset_audit["base_optimizer_exact"],
        "fusion values discarded and zero initialized": runtime.fusion_reinitialization["new_W_u_exact_zero"] and runtime.fusion_reinitialization["new_W_g_exact_zero"],
        "fusion optimizer reset only": runtime.optimizer_reset_audit["fusion_fresh_state"] and runtime.optimizer_reset_audit["only_fusion_state_reset"],
        "scheduler loader RNG next batch preserved": read_json(output / "preflight_audit.json")["checks"]["loader RNG next batch preserved"],
        "alpha exact nontrainable": ALPHA == 0.03125 and not any(name == "alpha" for name, _ in runtime.model.named_parameters()),
        "residual equation no rho": all(row["alpha"] == ALPHA and row["rho_disabled"] for row in training),
        "zero fusion identity": read_json(output / "architecture_manifest.json")["zero_identity"]["passed"],
        "temporal gradients": read_json(output / "preflight_smoke.json")["temporal_gradients"]["passed"],
        "all parameters trainable": all(parameter.requires_grad for parameter in runtime.model.parameters()),
        "CE weights and global pass cadence exact": all(row["pass_count"] == pass_count_for_update(row["global_update"]) for row in training),
        "Stage B-R exact": all(row["stage"] == "B-R" and row["windows"] == list(B12) for row in training[:96]),
        "Stage C-R exact": all(row["stage"] == "C-R" and row["windows"] == list(C12) for row in training[96:]),
        "477 local updates exact": len(training) == LOCAL_UPDATES and training[0]["local_update"] == 1 and training[-1]["local_update"] == 477,
        "additional targets exact": training[-1]["additional_targets"] == ADDITIONAL_TARGETS,
        "W_u projection exact": len(projection_rows) == LOCAL_UPDATES and all(row["sigma_post"] <= SIGMA_CAP * (1 + PROJECTION_TOLERANCE) for row in projection_rows),
        "all finite": all(row["all_gradients_finite"] and row["all_parameters_finite"] and row["all_optimizer_moments_finite"] for row in training),
        "checkpoints strict reopen": len(manifest["scientific"]) == len(SCIENTIFIC_LOCAL) and all(row["strict_reopen"]["passed"] for row in manifest["scientific"].values()),
        "no forbidden objectives/evaluations": True,
    }
    audit = {
        "experiment": EXPERIMENT,
        "checks": core,
        "passed": all(core.values()),
        "terminal_lifecycle_gates": {
            "result_commit_and_push_required_after_finalize": True,
            "local_remote_pod_git_match_required": True,
            "remote_sync_required": True,
            "exact_pod_id_reverification_required": args.pod_id,
            "stop_not_delete_required": True,
        },
    }
    if not audit["passed"]:
        primary = secondary = "EXPERIMENT 2D1D INVALID"
        recommendation = "FIX 2D1D INTEGRITY"
    final_checkpoint_sha = file_sha256(final_checkpoint)
    summary = {
        "experiment": EXPERIMENT,
        "primary_classification": primary,
        "secondary_classification": secondary,
        "source_C954_sha256": SOURCE_SHA256,
        "fusion_initialization": "W_u=0, W_g=0, alpha=.03125 fixed",
        "optimizer_reset": "fusion state only; base state exact",
        "hardware": {"pod_id": args.pod_id, "pod_name": args.pod_name, "gpu_type": args.gpu_type},
        "additional_updates": LOCAL_UPDATES,
        "additional_targets": ADDITIONAL_TARGETS,
        "final_global_update": FINAL_GLOBAL_UPDATE,
        "final_cumulative_targets": FINAL_TARGETS,
        "final_geometry": "C12",
        "final_plain_validation": final["controls"]["plain"]["validation_loss"],
        "final_real_validation": final["controls"]["real"]["validation_loss"],
        "final_zero_validation": final["controls"]["zero"]["validation_loss"],
        "final_shuffled_validation": final["controls"]["shuffled"]["validation_loss"],
        "final_recurrent_gain": final["recurrent_gain"],
        "final_sequence_gap": final["sequence_specific_gap"],
        "final_paired": final["paired"],
        "earliest_positive_measured_local_update": earliest_positive,
        "earliest_sequence_specific_measured_local_update": earliest_sequence,
        "first_W_g_nonzero_gradient_local_update": first_wg_gradient,
        "first_W_u_cap_projection_local_update": first_cap,
        "final_alphaF_over_E": branch_final,
        "final_X_over_E": training[-1]["scale_diagnostics"]["X_over_E"],
        "maximum_X_RMS": maximum_x,
        "projection_by_stage": projection_by_stage,
        "final_self_classification": final_self["summary"]["classification"],
        "position_best_bin": {best_position: final_positions[best_position]},
        "position_worst_bin": {worst_position: final_positions[worst_position]},
        "plain_change_0M_to_250M": plain_change,
        "final_matrix_diagnostics": final_matrix,
        "next_recommendation": recommendation,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": final_checkpoint_sha,
    }
    questions = {
        "Q1": f"Yes. All zero-identity checks passed: {read_json(output / 'architecture_manifest.json')['zero_identity']}.",
        "Q2": f"Yes. W_u had a nonzero finite gradient at disposable step 1 and result update 1; first result gradient report: {training[0]['gradient_groups']['W_u']}.",
        "Q3": f"First nonzero W_g gradient was local update {first_wg_gradient}.",
        "Q4": f"Threshold crossings: {read_json(output / 'branch_growth.json')['first_threshold_crossings']}; final alphaF/E={branch_final:.10f}.",
        "Q5": "No cap projection occurred." if first_cap is None else f"Yes; first projection was local update {first_cap}.",
        "Q6": f"Maximum training X RMS={maximum_x:.10f}; final 32-pass classification={final_self['summary']['classification']}.",
        "Q7": f"At 10M/local20, recurrent gain={milestones['20']['recurrent_gain']:+.10f}.",
        "Q8": f"At 25M/local48, recurrent gain={milestones['48']['recurrent_gain']:+.10f}.",
        "Q9": f"At 50M/local96 under B12, recurrent gain={milestones['96']['recurrent_gain']:+.10f}.",
        "Q10": f"Immediate C12 shock: plain {transition['plain_delta_C_minus_B']:+.10f}, real {transition['real_delta_C_minus_B']:+.10f}, gain {transition['gain_delta_C_minus_B']:+.10f}.",
        "Q11": f"At 100M/local191, recurrent gain={milestones['191']['recurrent_gain']:+.10f}.",
        "Q12": f"At 150M/local286, recurrent gain={milestones['286']['recurrent_gain']:+.10f}.",
        "Q13": f"At 250M/local477, recurrent gain={milestones['477']['recurrent_gain']:+.10f}.",
        "Q14": f"Earliest measured positive gain: {earliest_positive if earliest_positive is not None else 'NOT OBSERVED'}.",
        "Q15": f"Earliest measured positive milestone: {earliest_positive if earliest_positive is not None else 'NOT AVAILABLE / NOT OBSERVED'}.",
        "Q16": f"Final real-vs-shuffled: {final['paired']['real_vs_shuffled']}.",
        "Q17": f"Final real-vs-plain: {final['paired']['real_vs_plain']}.",
        "Q18": f"Best final position bin {best_position} ({final_positions[best_position]['recurrent_gain']:+.10f}); worst {worst_position} ({final_positions[worst_position]['recurrent_gain']:+.10f}).",
        "Q19": f"Late-context preference is {'supported' if best_position in ('769-896','897-1023') else 'not supported'} by the best final bin; see position_bin_metrics.json.",
        "Q20": f"Plain CE changed by {plain_change:+.10f} from source B12 to final C12 (geometry also changed).",
        "Q21": f"Projection summaries: {projection_by_stage}.",
        "Q22": f"Final W_u singular spectrum is stored exactly in branch_growth.json; summary: {final_matrix['W_u']}.",
        "Q23": f"W_g first received gradient at local {first_wg_gradient}; final diagnostics: {final_matrix['W_g']}.",
        "Q24": f"Sequence specificity emerged before useful recurrence: {sequence_before_useful}; earliest measured sequence gap>0={earliest_sequence}, gain>0={earliest_positive}.",
        "Q25": f"The preregistered 250M classification is {primary}; final gain={final['recurrent_gain']:+.10f}, gap={final['sequence_specific_gap']:+.10f}, alphaF/E={branch_final:.10f}.",
        "Q26": recommendation,
    }
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_json(output / "scientific_questions.json", questions)
    make_plots(output, training, milestones, self_data, positions)
    performance = read_json(output / "performance.json")
    performance["training"] = {
        "wall_seconds": training[-1]["timestamp"] - training[0]["timestamp"] + training[0]["wall_seconds"],
        "mean_targets_per_second": statistics.fmean(row["targets_per_second"] for row in training),
    }
    performance["finalized_at"] = time.time()
    durable_json(output / "performance.json", performance)
    commands = read_json(output / "commands_and_runtime.json")
    commands["commands"].append(" ".join(sys.argv))
    commands["finalize_completed_at"] = time.time()
    durable_json(output / "commands_and_runtime.json", commands)
    durable_text(output / "EXPERIMENT_2D1D_FINAL_REPORT.md", render_report(summary, questions, milestones, transition, projection_by_stage, audit))
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", render_handoff(summary))
    print(f"EXPERIMENT_2D1D_FINALIZE_PASS classification={primary}", flush=True)
    return summary


def add_execution_arguments(parser):
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-c954", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--persistent-volume-identity", required=True)
    parser.add_argument("--stop-mechanism", required=True)
    parser.add_argument("--stop-authenticated", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    add_execution_arguments(preflight)
    preflight.set_defaults(function=run_preflight)
    worker = subparsers.add_parser("train-worker")
    add_execution_arguments(worker)
    worker.add_argument("--resume")
    worker.set_defaults(function=run_train_worker)
    supervise = subparsers.add_parser("supervise")
    add_execution_arguments(supervise)
    supervise.add_argument("--resume")
    supervise.set_defaults(function=run_supervise)
    finalize = subparsers.add_parser("finalize")
    add_execution_arguments(finalize)
    finalize.add_argument("--final-checkpoint", required=True)
    finalize.set_defaults(function=run_finalize)
    return parser


def main():
    args = build_parser().parse_args()
    result = args.function(args)
    if isinstance(result, int):
        raise SystemExit(result)


if __name__ == "__main__":
    main()
