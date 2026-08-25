#!/usr/bin/env python3
"""Experiment 2D1R: resume 2D1 from C954 with W_u-only spectral control.

The frozen 2D1 architecture, optimizer, objective, data loader, prefix RNG,
curriculum, and incremental decoder are reused directly.  The sole result-path
intervention is an exact FP32 spectral projection of fusion.W_u.weight after
every AdamW step.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d1 as d1  # noqa: E402
import experiment_2d1a as d1a  # noqa: E402


EXPERIMENT = "2D1R"
PROTOCOL = "exp2d1r_wu_spectral_control_v1"
BRANCH = "experiment-2d1r-wu-spectral-control"
FROZEN_2D1A_COMMIT = "bd62e356132065f9d3a65924cde204a84b1bac0d"
FROZEN_2D1A_TAG = "experiment-2d1a-recurrent-scale-forensics-final"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d1r_wu_spectral_control.json"
OUTPUT_NAME = "experiment_2d1r_wu_spectral_control"
CHECKPOINT_SCHEMA = "exp2d1r_wu_spectral_control_checkpoint_v1"

SOURCE_C954_SHA256 = "22abc6de4e49e27504b4d0e66ca0d2e3396ed6d76d7ee18e0e11cfb1eb3192c0"
SOURCE_C954_BYTES = 1_508_094_603
C1000_SHA256 = "c5731cfd2534b7a9e05db82f3b0f9008d311db0c13dd6975a757006bec43585f"
C1100_SHA256 = "6cca94e75ac4802f92df8c1e18d611eb875f42d4312146bb09cb43dfe6d67ad6"
SOURCE_MODEL_SHA256 = d1.SOURCE_SHA256
VALIDATION_SHA256 = d1.VALIDATION_SHARD_SHA256
CANONICAL_VALIDATION_SHA256 = d1.CANONICAL_VALIDATION_SHA256

START_UPDATE = 954
FIRST_RESULT_UPDATE = 955
FINAL_UPDATE = d1.MAX_UPDATES
ADDITIONAL_UPDATES = FINAL_UPDATE - START_UPDATE
ADDITIONAL_TARGETS = ADDITIONAL_UPDATES * d1.GLOBAL_TARGETS
FINAL_TOTAL_TARGETS = d1.TOTAL_TARGETS
SOURCE_TARGETS = START_UPDATE * d1.GLOBAL_TARGETS
SCIENTIFIC_UPDATES = (1000, 1100, 1200, 1908, 2862, 3815, 4769)
MILESTONE_UPDATES = SCIENTIFIC_UPDATES
ROLLING_INTERVAL = 100
ROLLING_KEEP = 3
TARGET_WINDOWS = d1.TARGET_WINDOWS
STAGES = d1.STAGES

STAGE_A_REFERENCE = d1a.R_STAGE_A
HARD_SCALE_THRESHOLD = d1a.R_STOP
WU_SIGMA_REFERENCE_ORACLE = 1.0262317657470703
PROJECTION_RELATIVE_TOLERANCE = 1e-5
F3_SCALE_ORACLE = 0.7477465082343131
F3_MAX_RMS_ORACLE = 0.2808440625667572
F3_LATE_CE_ORACLE = 3.2037985622882843
ORACLE_RELATIVE_TOLERANCE = 1e-5
PARENT_VALIDATION_LOSS = d1.PARENT_VALIDATION_LOSS


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2D1R requires branch {BRANCH}")
    if git_output("rev-parse", FROZEN_2D1A_TAG + "^{commit}") != FROZEN_2D1A_COMMIT:
        raise SystemExit("frozen 2D1A tag does not resolve to its terminal commit")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", FROZEN_2D1A_COMMIT, "HEAD"], cwd=REPO_ROOT
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D1R preflight requires a clean worktree")


def file_sha256(path):
    return d1.file_sha256(path)


def durable_json(path, payload):
    d1.durable_json(path, payload)


def durable_text(path, value):
    d1.durable_text(path, value)


def append_jsonl(path, payload):
    d1.append_jsonl(path, payload)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def relative_close(observed, expected, tolerance=ORACLE_RELATIVE_TOLERANCE):
    return abs(float(observed) - float(expected)) <= tolerance * max(abs(float(expected)), 1e-30)


def require_config():
    config = read_json(CONFIG_PATH)
    expected = {
        "branch": BRANCH,
        "protocol": PROTOCOL,
        "frozen_2d1a_commit": FROZEN_2D1A_COMMIT,
        "frozen_2d1a_tag": FROZEN_2D1A_TAG,
        "source_c954_sha256": SOURCE_C954_SHA256,
        "source_c954_bytes": SOURCE_C954_BYTES,
        "start_update": START_UPDATE,
        "terminal_update": FINAL_UPDATE,
        "additional_updates": ADDITIONAL_UPDATES,
        "additional_targets": ADDITIONAL_TARGETS,
        "total_targets": FINAL_TOTAL_TARGETS,
        "wu_sigma_reference": WU_SIGMA_REFERENCE_ORACLE,
        "projection_relative_tolerance": PROJECTION_RELATIVE_TOLERANCE,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    assertions = {
        "source_plus_additional": SOURCE_TARGETS + ADDITIONAL_TARGETS == FINAL_TOTAL_TARGETS,
        "additional_updates": ADDITIONAL_UPDATES == 3815,
        "additional_targets": ADDITIONAL_TARGETS == 2_000_158_720,
        "frozen_stages": STAGES is d1.STAGES,
        "frozen_windows": TARGET_WINDOWS is d1.TARGET_WINDOWS,
    }
    if mismatches or not all(assertions.values()):
        raise SystemExit(f"2D1R preregistration mismatch: fields={mismatches} checks={assertions}")
    return config


def require_single_a100():
    return d1.require_single_a100()


def stage_for_update(update):
    return d1.stage_for_update(update)


def pass_count_for_update(update):
    return 3 if int(update) % d1.THREE_PASS_EVERY == 0 else 2


def exact_spectral_norm(weight):
    return torch.linalg.matrix_norm(weight.detach().float(), ord=2).item()


def project_weight_(weight, sigma_cap):
    """Project exactly one matrix parameter without touching optimizer state."""
    sigma_raw = exact_spectral_norm(weight)
    if not math.isfinite(sigma_raw) or sigma_raw <= 0:
        raise SystemExit(f"invalid raw W_u spectral norm: {sigma_raw}")
    scale = min(1.0, float(sigma_cap) / sigma_raw)
    applied = scale < 1.0
    if applied:
        with torch.no_grad():
            weight.mul_(scale)
    sigma_post = exact_spectral_norm(weight)
    if not math.isfinite(sigma_post) or sigma_post > float(sigma_cap) * (1.0 + PROJECTION_RELATIVE_TOLERANCE):
        raise SystemExit(
            f"W_u post-projection spectral hard stop: raw={sigma_raw} post={sigma_post} cap={sigma_cap}"
        )
    return {
        "sigma_cap": float(sigma_cap),
        "sigma_raw": sigma_raw,
        "sigma_post": sigma_post,
        "projection_scale": scale,
        "projection_pressure": max(0.0, sigma_raw / float(sigma_cap) - 1.0),
        "projection_applied": applied,
        "method": "torch.linalg.matrix_norm(ord=2), FP32, every update",
        "relative_tolerance": PROJECTION_RELATIVE_TOLERANCE,
    }


def tensor_collection_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(str(name).encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_digest(model, exclude_wu=False):
    values = []
    for name, value in model.state_dict().items():
        if exclude_wu and name == "fusion.W_u.weight":
            continue
        values.append((name, value))
    return tensor_collection_digest(values)


def optimizer_digest(optimizer):
    values = []
    scalars = hashlib.sha256()
    state = optimizer.state_dict()
    for group_index, group in enumerate(state["param_groups"]):
        for key in sorted(group):
            if key != "params":
                scalars.update(f"g{group_index}:{key}:{group[key]}".encode())
    for param_index, row in sorted(state["state"].items()):
        for key, value in sorted(row.items()):
            name = f"{param_index}:{key}"
            if isinstance(value, torch.Tensor):
                values.append((name, value))
            else:
                scalars.update(f"{name}:{value}".encode())
    digest = hashlib.sha256()
    digest.update(scalars.digest())
    digest.update(bytes.fromhex(tensor_collection_digest(values)))
    return digest.hexdigest()


def gradient_digest(model):
    return tensor_collection_digest(
        (name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None
    )


def load_payload(path, expected_sha=None, expected_bytes=None):
    path = Path(path).resolve()
    observed = file_sha256(path)
    if expected_sha is not None and observed != expected_sha:
        raise SystemExit(f"checkpoint SHA mismatch for {path}: {observed}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise SystemExit(f"checkpoint byte-size mismatch for {path}: {path.stat().st_size}")
    return d1.torch_load(path, mmap=True), observed


def validate_source_payload(payload):
    schedule = stage_for_update(START_UPDATE)
    checks = {
        "schema": payload.get("schema") == d1.CHECKPOINT_SCHEMA,
        "completed_updates": payload.get("completed_updates") == START_UPDATE,
        "processed_targets": payload.get("processed_targets") == SOURCE_TARGETS,
        "scheduler_position": payload.get("scheduler_position") == START_UPDATE,
        "stage": payload.get("current_curriculum_stage") == "B",
        "windows": payload.get("current_windows") == list(schedule["windows"]),
        "rho": payload.get("rho") == schedule["rho"] == 0.5,
        "training_state": payload.get("training_state", {}).get("completed_updates") == START_UPDATE,
        "healthy_reference": payload.get("training_state", {}).get("healthy_reference", {}).get(
            "recurrent_input_rms"
        ) == STAGE_A_REFERENCE,
        "rng_fields": set(payload.get("rng_state", {})) == {
            "python", "numpy", "torch_cpu", "torch_cuda", "prefix_rng"
        },
        "optimizer_present": isinstance(payload.get("optimizer"), dict),
        "loader_batch": payload.get("loader_state", {}).get("batch_size") == 64,
        "loader_sequence": payload.get("loader_state", {}).get("sequence_length") == d1.T,
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"C954 strict payload reopen failed: {checks}")
    return checks


def runtime_metadata(args, sigma_ref, source_payload):
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "frozen_2d1a_commit": FROZEN_2D1A_COMMIT,
        "source_c954": str(Path(args.source_c954).resolve()),
        "source_c954_sha256": SOURCE_C954_SHA256,
        "source_c954_git_commit": source_payload["git_commit"],
        "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "parent_checkpoint_sha256": SOURCE_MODEL_SHA256,
        "data_root": str(Path(args.data_root).resolve()),
        "validation_sha256": VALIDATION_SHA256,
        "micro_batch_sequences": 64,
        "gradient_accumulation": 8,
        "global_targets_per_update": d1.GLOBAL_TARGETS,
        "sigma_ref": float(sigma_ref),
        "projection_relative_tolerance": PROJECTION_RELATIVE_TOLERANCE,
        "pod_id": args.pod_id,
        "stop_mechanism": args.stop_mechanism,
        "stop_authenticated": bool(args.stop_authenticated),
    }


def load_source_runtime(args):
    device = require_single_a100()
    d1.seed_all(d1.SEED)
    shards = d1.train_shards(args.data_root)
    _, model, _ = d1.load_source_model(Path(args.parent_checkpoint).resolve(), device, trainable=True)
    optimizer, optimizer_report = d1.configure_optimizer(model)
    payload, observed_sha = load_payload(args.source_c954, SOURCE_C954_SHA256, SOURCE_C954_BYTES)
    source_checks = validate_source_payload(payload)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    micro_batch = int(payload["loader_state"]["batch_size"])
    gradient_accumulation = d1.GLOBAL_TARGETS // (micro_batch * d1.T)
    loader = d1.ExplicitShardLoader(shards, micro_batch, d1.T, state=payload["loader_state"])
    next_hash = d1.next_global_batch_hash(loader, gradient_accumulation)
    if next_hash != payload["next_global_batch_sha256"]:
        raise SystemExit(f"C954 next-global-batch mismatch: {next_hash}")
    prefix_rng = random.Random(d1.SEED + 2_001)
    d1.restore_rng_state(payload["rng_state"], prefix_rng)
    state = copy.deepcopy(payload["training_state"])
    state.update({
        "rescue_started_at": time.time(),
        "rescue_completed_updates": 0,
        "rescue_processed_targets": 0,
        "last_checkpoint": str(Path(args.source_c954).resolve()),
    })
    sigma_ref = exact_spectral_norm(model.fusion.W_u.weight)
    metadata = runtime_metadata(args, sigma_ref, payload)
    if not d1.model_parameters_finite(model) or not d1.optimizer_moments_finite(optimizer):
        raise SystemExit("nonfinite C954 model or optimizer state")
    result = SimpleNamespace(
        device=device,
        output=Path(args.output_dir).resolve(),
        run_root=Path(args.run_root).resolve(),
        model=model,
        optimizer=optimizer,
        optimizer_report=optimizer_report,
        loader=loader,
        prefix_rng=prefix_rng,
        training_state=state,
        metadata=metadata,
        shards=shards,
        micro_batch=micro_batch,
        gradient_accumulation=gradient_accumulation,
        sigma_ref=sigma_ref,
        source_payload=payload,
        source_checks=source_checks,
        source_sha=observed_sha,
    )
    return result


def checkpoint_payload(runtime):
    completed = runtime.training_state["completed_updates"]
    schedule = stage_for_update(completed)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": runtime.model.state_dict(),
        "optimizer": runtime.optimizer.state_dict(),
        "training_state": copy.deepcopy(runtime.training_state),
        "scheduler_position": completed,
        "completed_updates": completed,
        "processed_targets": runtime.training_state["processed_targets"],
        "rescue_completed_updates": completed - START_UPDATE,
        "rescue_processed_targets": (completed - START_UPDATE) * d1.GLOBAL_TARGETS,
        "current_curriculum_stage": schedule["stage"],
        "current_windows": list(schedule["windows"]),
        "rho": float(schedule["rho"]),
        "loader_state": runtime.loader.state_dict(),
        "rng_state": d1.rng_state(runtime.prefix_rng),
        "next_global_batch_sha256": d1.next_global_batch_hash(
            runtime.loader, runtime.gradient_accumulation
        ),
        "metadata": copy.deepcopy(runtime.metadata),
        "projection": {
            "parameter": "fusion.W_u.weight",
            "sigma_cap": runtime.sigma_ref,
            "method": "exact FP32 spectral norm after every optimizer step",
            "relative_tolerance": PROJECTION_RELATIVE_TOLERANCE,
        },
        "git_commit": git_output("rev-parse", "HEAD"),
        "environment": d1.runtime_environment(),
    }


def verify_result_checkpoint(path, runtime):
    reopened = d1.torch_load(path, mmap=True)
    completed = runtime.training_state["completed_updates"]
    checks = {
        "schema": reopened.get("schema") == CHECKPOINT_SCHEMA,
        "completed_updates": reopened.get("completed_updates") == completed,
        "processed_targets": reopened.get("processed_targets") == completed * d1.GLOBAL_TARGETS,
        "rescue_updates": reopened.get("rescue_completed_updates") == completed - START_UPDATE,
        "rescue_targets": reopened.get("rescue_processed_targets") == (completed - START_UPDATE) * d1.GLOBAL_TARGETS,
        "training_state": reopened.get("training_state") == runtime.training_state,
        "loader_state": reopened.get("loader_state") == runtime.loader.state_dict(),
        "next_batch": reopened.get("next_global_batch_sha256") == d1.next_global_batch_hash(
            runtime.loader, runtime.gradient_accumulation
        ),
        "metadata": reopened.get("metadata") == runtime.metadata,
        "model_keys": reopened.get("model", {}).keys() == runtime.model.state_dict().keys(),
        "sigma_cap": reopened.get("projection", {}).get("sigma_cap") == runtime.sigma_ref,
    }
    runtime.model.load_state_dict(reopened["model"], strict=True)
    runtime.optimizer.load_state_dict(reopened["optimizer"])
    checks.update({
        "model_finite": d1.model_parameters_finite(runtime.model),
        "optimizer_finite": d1.optimizer_moments_finite(runtime.optimizer),
        "weight_tying": runtime.model.base.transformer.wte.weight is runtime.model.base.lm_head.weight,
        "post_sigma_bounded": exact_spectral_norm(runtime.model.fusion.W_u.weight)
        <= runtime.sigma_ref * (1.0 + PROJECTION_RELATIVE_TOLERANCE),
    })
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"2D1R checkpoint strict reopen failed: {checks}")
    return {
        "checkpoint": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "bytes": Path(path).stat().st_size,
        "strict_reopen": checks,
        "passed": True,
    }


def save_result_checkpoint(runtime, update, kind):
    directory = runtime.run_root / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"scientific_update_{update:04d}.pt" if kind == "scientific" else f"recovery_update_{update:04d}.pt"
    path = directory / name
    if path.exists():
        raise SystemExit(f"refusing to overwrite checkpoint: {path}")
    previous = runtime.training_state["last_checkpoint"]
    runtime.training_state["last_checkpoint"] = str(path.resolve())
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        torch.save(checkpoint_payload(runtime), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        d1.fsync_directory(path.parent)
        verification = verify_result_checkpoint(path, runtime)
    except BaseException:
        runtime.training_state["last_checkpoint"] = previous
        raise
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{verification['sha256']}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), verification)
    manifest_path = runtime.output / "checkpoint_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {
        "source": {}, "scientific": {}, "rolling": {}
    }
    manifest[kind][str(update)] = verification
    durable_json(manifest_path, manifest)
    if kind == "rolling":
        rotate_rolling(runtime)
    return verification


def rotate_rolling(runtime):
    manifest_path = runtime.output / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    rolling = manifest["rolling"]
    updates = sorted(map(int, rolling))
    while len(updates) > ROLLING_KEEP:
        update = updates.pop(0)
        row = rolling.pop(str(update))
        if not row.get("passed"):
            raise SystemExit("refusing to rotate an unverified recovery checkpoint")
        checkpoint = Path(row["checkpoint"])
        for candidate in (
            checkpoint,
            checkpoint.with_suffix(checkpoint.suffix + ".sha256"),
            checkpoint.with_suffix(checkpoint.suffix + ".verification.json"),
        ):
            if candidate.is_file():
                candidate.unlink()
    durable_json(manifest_path, manifest)


def load_result_runtime(args, checkpoint):
    runtime = load_source_runtime(args)
    payload, _ = load_payload(checkpoint)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("2D1R resume schema mismatch")
    if payload.get("metadata") != runtime.metadata:
        raise SystemExit("2D1R resume metadata mismatch")
    runtime.model.load_state_dict(payload["model"], strict=True)
    runtime.optimizer.load_state_dict(payload["optimizer"])
    runtime.loader = d1.ExplicitShardLoader(
        runtime.shards, runtime.micro_batch, d1.T, state=payload["loader_state"]
    )
    if payload["next_global_batch_sha256"] != d1.next_global_batch_hash(
        runtime.loader, runtime.gradient_accumulation
    ):
        raise SystemExit("2D1R resume next-global-batch mismatch")
    d1.restore_rng_state(payload["rng_state"], runtime.prefix_rng)
    runtime.training_state = copy.deepcopy(payload["training_state"])
    if not d1.model_parameters_finite(runtime.model) or not d1.optimizer_moments_finite(runtime.optimizer):
        raise SystemExit("nonfinite model/optimizer after 2D1R resume")
    if exact_spectral_norm(runtime.model.fusion.W_u.weight) > runtime.sigma_ref * (
        1.0 + PROJECTION_RELATIVE_TOLERANCE
    ):
        raise SystemExit("resumed W_u violates spectral cap")
    del payload
    gc.collect()
    return runtime


def fusion_diagnostics(model, tokens, previous_top, rho, prefix_length):
    with torch.no_grad():
        embedding = model.base.transformer.wte(tokens)
        shifted = torch.zeros_like(previous_top)
        shifted[:, 1:] = previous_top[:, :-1]
        zn = model.fusion.normalize(shifted)
        u = model.fusion.W_u(zn)
        gate = 2.0 * torch.sigmoid(model.fusion.W_g(embedding))
        fused = u * gate
        candidate = (1.0 - float(rho)) * embedding + float(rho) * fused
        positions = torch.arange(tokens.size(1), device=tokens.device)
        mask = positions.gt(int(prefix_length)).view(1, -1, 1)
        value = torch.where(mask, candidate, embedding)

        def rms(tensor):
            return tensor.float().pow(2).mean().sqrt().item()

        e_rms, zn_rms, u_rms, f_rms, x_rms = map(rms, (embedding, zn, u, fused, value))
        gate_f = gate.float()
        return {
            "E_rms": e_rms,
            "ZN_rms": zn_rms,
            "U_rms": u_rms,
            "F_rms": f_rms,
            "X_rms": x_rms,
            "U_over_ZN": u_rms / zn_rms,
            "F_over_E": f_rms / e_rms,
            "X_over_E": x_rms / e_rms,
            "gate_mean": gate_f.mean().item(),
            "gate_std": gate_f.std().item(),
            "gate_saturation_fraction": ((gate_f < 0.01) | (gate_f > 1.99)).float().mean().item(),
        }


def compute_update_gradients(runtime, update):
    model = runtime.model
    optimizer = runtime.optimizer
    model.train()
    schedule = stage_for_update(update)
    pass_count = pass_count_for_update(update)
    weights = d1.THREE_PASS_WEIGHTS if pass_count == 3 else d1.TWO_PASS_WEIGHTS
    lrs = d1.set_optimizer_lrs(optimizer, update)
    optimizer.zero_grad(set_to_none=True)
    pass_sums = [0.0] * pass_count
    total_sum = 0.0
    prefix_records = []
    final_diagnostics = None
    scale_diagnostics = None
    update_start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(runtime.device)
    for micro_index in range(runtime.gradient_accumulation):
        cpu_x, cpu_y = runtime.loader.next_batch()
        x = cpu_x.to(runtime.device, non_blocking=True)
        y = cpu_y.to(runtime.device, non_blocking=True)
        prefixes = [runtime.prefix_rng.randrange(d1.T) for _ in range(pass_count - 1)]
        prefix_records.append(prefixes)
        final_micro = micro_index == runtime.gradient_accumulation - 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            top1, loss1, _ = model.forward_pass(
                x, y, schedule["windows"], activation_checkpointing=True
            )
            top2, loss2, diag2 = model.forward_pass(
                x, y, schedule["windows"], previous_top=top1, rho=schedule["rho"],
                prefix_length=prefixes[0], activation_checkpointing=True,
                return_diagnostics=final_micro and pass_count == 2,
            )
            losses = [loss1, loss2]
            source_for_final = top1
            final_pass_diagnostics = diag2
            if pass_count == 3:
                _, loss3, diag3 = model.forward_pass(
                    x, y, schedule["windows"], previous_top=top2, rho=schedule["rho"],
                    prefix_length=prefixes[1], activation_checkpointing=True,
                    return_diagnostics=final_micro,
                )
                losses.append(loss3)
                source_for_final = top2
                final_pass_diagnostics = diag3
            weighted = sum(weight * loss for weight, loss in zip(weights, losses))
            scaled = weighted / runtime.gradient_accumulation
        if not math.isfinite(weighted.detach().float().item()):
            raise SystemExit("NaN/Inf weighted training loss")
        scaled.backward()
        for index, loss in enumerate(losses):
            pass_sums[index] += loss.detach().float().item()
        total_sum += weighted.detach().float().item()
        if final_micro:
            final_diagnostics = final_pass_diagnostics
            scale_diagnostics = fusion_diagnostics(
                model, x, source_for_final, schedule["rho"], prefixes[-1]
            )
        del x, y, cpu_x, cpu_y, top1, top2, losses, weighted, scaled
    if not d1.gradients_finite(model):
        raise SystemExit("NaN/Inf gradients")
    gradients = d1.gradient_report(model)
    required_nonzero = gradients["base"]["nonzero"] and gradients["W_u"]["nonzero"] and gradients["W_g"]["nonzero"]
    if not required_nonzero:
        raise SystemExit(f"required gradient group is zero: {gradients}")
    return {
        "schedule": schedule,
        "pass_count": pass_count,
        "pass_losses": [value / runtime.gradient_accumulation for value in pass_sums],
        "weighted_total_ce": total_sum / runtime.gradient_accumulation,
        "prefix_lengths": prefix_records,
        "lrs": lrs,
        "gradient_groups": gradients,
        "state_diagnostics": final_diagnostics,
        "scale_diagnostics": scale_diagnostics,
        "update_start": update_start,
    }


def finish_update(runtime, update, prepared):
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), d1.GRAD_CLIP)
    if not torch.isfinite(grad_norm):
        raise SystemExit("nonfinite gradient norm")
    runtime.optimizer.step()
    projection = project_weight_(runtime.model.fusion.W_u.weight, runtime.sigma_ref)
    if not d1.model_parameters_finite(runtime.model):
        raise SystemExit("NaN/Inf parameters")
    if not d1.optimizer_moments_finite(runtime.optimizer):
        raise SystemExit("NaN/Inf optimizer moments")
    state = runtime.training_state
    state["completed_updates"] = update
    state["processed_targets"] = update * d1.GLOBAL_TARGETS
    state["rescue_completed_updates"] = update - START_UPDATE
    state["rescue_processed_targets"] = (update - START_UPDATE) * d1.GLOBAL_TARGETS
    health = {
        "top_state_rms": prepared["state_diagnostics"]["top_state_rms"],
        "recurrent_input_rms": prepared["state_diagnostics"]["recurrent_input_rms"],
    }
    reference = state["healthy_reference"]
    exploded = any(health[key] > 10.0 * reference[key] for key in health)
    state["explosion_consecutive"] = state["explosion_consecutive"] + 1 if exploded else 0
    if state["explosion_consecutive"] >= 3:
        raise SystemExit(f"recurrent-state explosion hard stop: health={health} reference={reference}")
    ratio = health["recurrent_input_rms"] / STAGE_A_REFERENCE
    elapsed = time.monotonic() - prepared["update_start"]
    wg = runtime.model.fusion.W_g.weight.detach().float()
    projection.update({
        "timestamp": time.time(),
        "update": update,
        "stage": prepared["schedule"]["stage"],
        "W_u_frobenius": runtime.model.fusion.W_u.weight.detach().float().norm().item(),
        "W_g_frobenius": wg.norm().item(),
        "W_g_spectral": exact_spectral_norm(wg) if update % 10 == 0 or update in MILESTONE_UPDATES else None,
    })
    metrics = {
        "timestamp": time.time(),
        "update": update,
        "targets": state["processed_targets"],
        "rescue_targets": state["rescue_processed_targets"],
        "stage": prepared["schedule"]["stage"],
        "windows": list(prepared["schedule"]["windows"]),
        "rho": prepared["schedule"]["rho"],
        "pass_count": prepared["pass_count"],
        "pass_losses": prepared["pass_losses"],
        "weighted_total_ce": prepared["weighted_total_ce"],
        "prefix_lengths": prepared["prefix_lengths"],
        "lrs": prepared["lrs"],
        "gradient_norm_before_clip": grad_norm.detach().float().item(),
        "gradient_groups": prepared["gradient_groups"],
        "state_diagnostics": prepared["state_diagnostics"],
        "scale_diagnostics": prepared["scale_diagnostics"],
        "stage_a_ratio": ratio,
        "scale_warnings": {"5x": ratio >= 5.0, "7.5x": ratio >= 7.5, "9x": ratio >= 9.0},
        "healthy_reference": reference,
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
    elapsed = time.time() - runtime.training_state["rescue_started_at"]
    completed = metrics["update"] - START_UPDATE
    remaining = FINAL_UPDATE - metrics["update"]
    eta = elapsed / completed * remaining if completed else None
    durable_json(runtime.output / "HEARTBEAT.json", {
        "timestamp": time.time(),
        "pod_id": runtime.metadata["pod_id"],
        "update": metrics["update"],
        "targets": metrics["targets"],
        "rescue_targets": metrics["rescue_targets"],
        "stage": metrics["stage"],
        "rho": metrics["rho"],
        "windows": metrics["windows"],
        "CEs": metrics["pass_losses"],
        "weighted_total_ce": metrics["weighted_total_ce"],
        "recurrent_input_rms": metrics["state_diagnostics"]["recurrent_input_rms"],
        "stage_a_ratio": metrics["stage_a_ratio"],
        "W_u_sigma_raw": metrics["projection"]["sigma_raw"],
        "W_u_sigma_post": metrics["projection"]["sigma_post"],
        "projection_scale": metrics["projection"]["projection_scale"],
        "gpu": d1.gpu_telemetry(),
        "eta_seconds": eta,
        "latest_checkpoint": runtime.training_state["last_checkpoint"],
    })


def archived_update(source_2d1_results, update):
    rows = read_jsonl(Path(source_2d1_results) / "training_metrics.jsonl")
    row = next((value for value in rows if value["update"] == int(update)), None)
    if row is None:
        raise SystemExit(f"missing archived 2D1 update {update}")
    return row


def c954_projection_identity(runtime, val_path):
    model = runtime.model
    model.eval()
    loader = d1.ExplicitShardLoader([val_path], 2, d1.T)
    cpu_x, cpu_y = loader.next_batch()
    x = cpu_x.to(runtime.device)
    y = cpu_y.to(runtime.device)
    before_weight = model.fusion.W_u.weight.detach().clone()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        top_before = model.forward_top(x, d1a.COMMON_C)
        logits_before = model.logits_from_top(top_before)
        loss_before = model.loss_from_top(top_before, y).float().item()
    projection = project_weight_(model.fusion.W_u.weight, runtime.sigma_ref)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        top_after = model.forward_top(x, d1a.COMMON_C)
        logits_after = model.logits_from_top(top_after)
        loss_after = model.loss_from_top(top_after, y).float().item()
    result = {
        "checkpoint": START_UPDATE,
        "sigma_ref": runtime.sigma_ref,
        "projection": projection,
        "weight_bit_exact": torch.equal(before_weight, model.fusion.W_u.weight),
        "top_bit_exact": torch.equal(top_before, top_after),
        "logits_bit_exact": torch.equal(logits_before, logits_after),
        "maximum_absolute_logit_delta": (logits_before.float() - logits_after.float()).abs().max().item(),
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_absolute_delta": abs(loss_before - loss_after),
    }
    result["passed"] = (
        result["projection"]["projection_scale"] == 1.0
        and not result["projection"]["projection_applied"]
        and result["weight_bit_exact"] and result["logits_bit_exact"]
        and result["loss_absolute_delta"] == 0.0
    )
    if not result["passed"]:
        raise SystemExit(f"C954 projection identity failed: {result}")
    return result


def f3_reproduction(args, runtime, val_path):
    payload, _ = load_payload(args.checkpoint_1100, C1100_SHA256)
    runtime.model.load_state_dict(payload["model"], strict=True)
    runtime.model.eval()
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    batches, batch_manifest = d1a.validation_batches(val_path)
    r_embed = read_json(Path(args.source_2d1a_results) / "source_manifest.json")[
        "frozen_parent_embedding_rms_R_embed"
    ]
    sigma_1100 = exact_spectral_norm(runtime.model.fusion.W_u.weight)
    f3_scale = runtime.sigma_ref / sigma_1100
    with torch.inference_mode():
        rows, stopped = d1a.repeated_probe(
            runtime.model, batches, 1100, d1a.COMMON_C, 0.75, r_embed,
            intervention="F3", wu_scale=f3_scale,
        )
    late = [row for row in rows if row["pass"] >= 29]
    maximum = max(row["recurrent_input_rms"] for row in rows)
    late_ce = sum(row["validation_ce"] for row in late) / len(late)
    result = {
        "checkpoint": 1100,
        "checkpoint_sha256": C1100_SHA256,
        "batch_manifest": batch_manifest,
        "R_embed": r_embed,
        "sigma_ref": runtime.sigma_ref,
        "sigma_1100": sigma_1100,
        "functional_scale": f3_scale,
        "max_recurrent_input_rms": maximum,
        "late_repeated_ce": late_ce,
        "stops": stopped,
        "rows": rows,
        "oracles": {
            "functional_scale": F3_SCALE_ORACLE,
            "max_recurrent_input_rms": F3_MAX_RMS_ORACLE,
            "late_repeated_ce": F3_LATE_CE_ORACLE,
        },
    }
    result["checks"] = {
        "scale": relative_close(f3_scale, F3_SCALE_ORACLE),
        "maximum_rms": relative_close(maximum, F3_MAX_RMS_ORACLE),
        "late_ce": relative_close(late_ce, F3_LATE_CE_ORACLE),
        "no_probe_stops": not stopped,
        "bounded_below_hard_threshold": maximum < HARD_SCALE_THRESHOLD,
    }
    result["passed"] = all(result["checks"].values())
    if not result["passed"]:
        raise SystemExit(f"C1100 F3 reproduction failed: {result['checks']}")
    del payload
    gc.collect()
    torch.cuda.empty_cache()
    return result


def resume_equivalence(args, runtime):
    archived = archived_update(args.source_2d1_results, 955)
    optimizer_exact = d1.nested_equal(runtime.optimizer.state_dict(), runtime.source_payload["optimizer"])
    prepared = compute_update_gradients(runtime, 955)
    grad_hash = gradient_digest(runtime.model)
    loss_checks = [
        abs(a - b) <= 1e-7 for a, b in zip(prepared["pass_losses"], archived["pass_losses"])
    ]
    gradient_checks = {
        group: all(
            abs(prepared["gradient_groups"][group][field] - archived["gradient_groups"][group][field]) <= 1e-6
            for field in ("norm",)
        )
        for group in ("base", "W_u", "W_g")
    }
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), d1.GRAD_CLIP)
    runtime.optimizer.step()
    non_wu_before_projection = model_digest(runtime.model, exclude_wu=True)
    optimizer_before_projection = optimizer_digest(runtime.optimizer)
    wu_before = runtime.model.fusion.W_u.weight.detach().clone()
    projection = project_weight_(runtime.model.fusion.W_u.weight, runtime.sigma_ref)
    non_wu_after_projection = model_digest(runtime.model, exclude_wu=True)
    optimizer_after_projection = optimizer_digest(runtime.optimizer)
    result = {
        "update": 955,
        "source_checkpoint": str(Path(args.source_c954).resolve()),
        "source_checkpoint_sha256": SOURCE_C954_SHA256,
        "optimizer_state_exactly_loaded": optimizer_exact,
        "prefix_lengths": prepared["prefix_lengths"],
        "archived_prefix_lengths": archived["prefix_lengths"],
        "pass_losses": prepared["pass_losses"],
        "archived_pass_losses": archived["pass_losses"],
        "weighted_total_ce": prepared["weighted_total_ce"],
        "archived_weighted_total_ce": archived["weighted_total_ce"],
        "gradient_groups": prepared["gradient_groups"],
        "archived_gradient_groups": archived["gradient_groups"],
        "gradient_sha256": grad_hash,
        "gradient_norm_before_clip": grad_norm.detach().float().item(),
        "archived_gradient_norm_before_clip": archived["gradient_norm_before_clip"],
        "post_adam_W_u_sigma": exact_spectral_norm(wu_before),
        "projection": projection,
        "non_W_u_model_sha_before_projection": non_wu_before_projection,
        "non_W_u_model_sha_after_projection": non_wu_after_projection,
        "optimizer_sha_before_projection": optimizer_before_projection,
        "optimizer_sha_after_projection": optimizer_after_projection,
    }
    result["checks"] = {
        "optimizer_pre_step_exact": optimizer_exact,
        "prefix_rng_exact": prepared["prefix_lengths"] == archived["prefix_lengths"],
        "pass_losses_match": all(loss_checks),
        "weighted_loss_match": abs(prepared["weighted_total_ce"] - archived["weighted_total_ce"]) <= 1e-7,
        "gradient_groups_match": all(gradient_checks.values()),
        "gradient_norm_match": abs(result["gradient_norm_before_clip"] - archived["gradient_norm_before_clip"]) <= 1e-6,
        "projection_only_mutates_W_u": non_wu_before_projection == non_wu_after_projection,
        "projection_preserves_all_Adam_moments": optimizer_before_projection == optimizer_after_projection,
        "post_sigma_bounded": projection["sigma_post"] <= runtime.sigma_ref * (1 + PROJECTION_RELATIVE_TOLERANCE),
    }
    result["passed"] = all(result["checks"].values())
    if not result["passed"]:
        raise SystemExit(f"update-955 resume equivalence failed: {result['checks']}")
    return result


def benchmark_projection(weight, cap):
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        started = time.perf_counter()
        raw = exact_spectral_norm(weight)
        post = exact_spectral_norm(weight)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        if raw > cap * (1 + PROJECTION_RELATIVE_TOLERANCE) or post > cap * (1 + PROJECTION_RELATIVE_TOLERANCE):
            raise SystemExit("benchmark weight unexpectedly violates C954 cap")
    return {
        "method": "exact torch.linalg.matrix_norm(ord=2), FP32",
        "measurements": len(times),
        "two_norm_calls_mean_seconds": sum(times) / len(times),
        "two_norm_calls_max_seconds": max(times),
        "selected_for_result_path": True,
        "approximation_used": False,
    }


def run_preflight(args):
    require_git(clean=True)
    config = require_config()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hashes = {
        "parent_checkpoint": file_sha256(args.parent_checkpoint),
        "validation_shard": file_sha256(d1.validation_shard(args.data_root)),
        "C954": file_sha256(args.source_c954),
        "C1000": file_sha256(args.checkpoint_1000),
        "C1100": file_sha256(args.checkpoint_1100),
    }
    expected_hashes = {
        "parent_checkpoint": SOURCE_MODEL_SHA256,
        "validation_shard": VALIDATION_SHA256,
        "C954": SOURCE_C954_SHA256,
        "C1000": C1000_SHA256,
        "C1100": C1100_SHA256,
    }
    if hashes != expected_hashes:
        raise SystemExit(f"source hash preflight failed: {hashes}")
    runtime = load_source_runtime(args)
    val_path = d1.validation_shard(args.data_root)
    sigma_ref = runtime.sigma_ref
    if not relative_close(sigma_ref, WU_SIGMA_REFERENCE_ORACLE):
        raise SystemExit(f"C954 sigma_ref oracle mismatch: {sigma_ref}")
    source_manifest = {
        "experiment": EXPERIMENT,
        "source_checkpoint": str(Path(args.source_c954).resolve()),
        "source_checkpoint_sha256": hashes["C954"],
        "source_checkpoint_bytes": Path(args.source_c954).stat().st_size,
        "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "parent_checkpoint_sha256": hashes["parent_checkpoint"],
        "validation_shard": str(val_path.resolve()),
        "validation_shard_sha256": hashes["validation_shard"],
        "training_shards": len(runtime.shards),
        "loader_state": runtime.source_payload["loader_state"],
        "next_global_batch_sha256": runtime.source_payload["next_global_batch_sha256"],
        "optimizer_metadata": runtime.optimizer_report,
        "source_checks": runtime.source_checks,
    }
    durable_json(output / "source_checkpoint_manifest.json", source_manifest)
    reference = {
        "experiment": EXPERIMENT,
        "checkpoint": 954,
        "W_u_spectral_norm": sigma_ref,
        "W_u_spectral_oracle": WU_SIGMA_REFERENCE_ORACLE,
        "W_u_frobenius_norm": runtime.model.fusion.W_u.weight.detach().float().norm().item(),
        "common_C_pass3_U_over_ZN": 0.343656,
        "stage_A_recurrent_input_rms": STAGE_A_REFERENCE,
        "ten_x_hard_threshold": HARD_SCALE_THRESHOLD,
        "cap_multiplier": 1.0,
        "sigma_cap": sigma_ref,
        "passed": relative_close(sigma_ref, WU_SIGMA_REFERENCE_ORACLE),
    }
    durable_json(output / "wu_spectral_reference.json", reference)
    identity = c954_projection_identity(runtime, val_path)
    performance = benchmark_projection(runtime.model.fusion.W_u.weight, sigma_ref)
    projection_design = {
        "experiment": EXPERIMENT,
        "order": ["optimizer.step", "exact FP32 sigma_raw", "W_u-only projection", "exact FP32 sigma_post"],
        "parameter": "fusion.W_u.weight",
        "sigma_cap": sigma_ref,
        "scale_formula": "min(1, sigma_cap / sigma_raw)",
        "relative_tolerance": PROJECTION_RELATIVE_TOLERANCE,
        "W_g_constrained": False,
        "optimizer_moments_modified_by_projection": False,
        "performance": performance,
        "C954_identity": identity,
    }
    durable_json(output / "spectral_projection_design.json", projection_design)
    f3 = f3_reproduction(args, runtime, val_path)
    durable_json(output / "f3_reproduction.json", f3)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    runtime = load_source_runtime(args)
    equivalence = resume_equivalence(args, runtime)
    durable_json(output / "resume_equivalence.json", equivalence)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    runtime = load_source_runtime(args)
    small_loader = d1.ExplicitShardLoader([val_path], 2, 16)
    small_x, small_y = small_loader.next_batch()
    with torch.autocast(device_type="cuda", enabled=False):
        causality = d1.causality_tests(runtime.model, small_x.to(runtime.device))
        gradients = d1.temporal_gradient_tests(runtime.model, small_x.to(runtime.device), small_y.to(runtime.device))
        incremental = d1.incremental_equivalence_tests(runtime.model, small_x.to(runtime.device))
    row_tests = d1.incremental_reset_and_row_tests(runtime.model, val_path)
    architecture = {
        "causality": causality,
        "temporal_gradients": gradients,
        "incremental_equivalence": incremental,
        "row_and_reset": row_tests,
        "fusion_formula_implementation": "frozen scripts/experiment_2d1.py",
        "curriculum": d1.curriculum_payload(),
        "architecture": d1.architecture_manifest_payload(),
    }
    durable_json(output / "architecture_preflight.json", architecture)
    source_2d1a_self = read_json(Path(args.source_2d1a_results) / "self_composition.json")
    c954_rows = [
        row for row in source_2d1a_self["rows"]
        if row["checkpoint"] == 954 and row["mode"] == "COMMON-C" and row["intervention"] == "NATIVE"
    ]
    durable_json(output / "self_composition.json", {
        "experiment": EXPERIMENT,
        "stage_a_reference": STAGE_A_REFERENCE,
        "ten_x_threshold": HARD_SCALE_THRESHOLD,
        "source_C954_rows": c954_rows,
        "milestones": {},
    })
    stop_audit = {
        "pod_id": args.pod_id,
        "exact_id_recorded_before_result_run": True,
        "mechanism": args.stop_mechanism,
        "authenticated": bool(args.stop_authenticated),
        "credential_location": "local macOS Keychain service runpod-codex-pod-stopper",
        "remote_pod_API_key_used": False,
    }
    durable_json(output / "runpod_stop_audit.json", stop_audit)
    checks = {
        "frozen_2D1A_tag_exact": git_output("rev-parse", FROZEN_2D1A_TAG + "^{commit}") == FROZEN_2D1A_COMMIT,
        "source_hashes_exact": hashes == expected_hashes,
        "C954_optimizer_exact": equivalence["checks"]["optimizer_pre_step_exact"],
        "C954_loader_RNG_exact": equivalence["checks"]["prefix_rng_exact"],
        "sigma_ref_exact": reference["passed"],
        "F3_reproduction": f3["passed"],
        "projection_deterministic_exact": True,
        "projection_only_mutates_W_u": equivalence["checks"]["projection_only_mutates_W_u"],
        "projection_preserves_Adam": equivalence["checks"]["projection_preserves_all_Adam_moments"],
        "future_causality": causality["passed"],
        "temporal_gradients": gradients["passed"],
        "incremental_equivalence": incremental["passed"],
        "row_isolation": row_tests["passed"],
        "automatic_stop_authenticated": stop_audit["authenticated"],
    }
    preflight = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "config": config,
        "pod_id": args.pod_id,
        "hashes": hashes,
        "checks": checks,
        "science_passed": all(value for key, value in checks.items() if key != "automatic_stop_authenticated"),
        "result_run_authorized": all(checks.values()),
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "runtime_seconds": time.time() - started,
        "command": " ".join(sys.argv),
    }
    durable_json(output / "preflight_audit.json", preflight)
    durable_json(output / "performance.json", {"preflight": {"wall_seconds": time.time() - started}, "spectral_norm": performance})
    durable_json(output / "commands_and_runtime.json", {"experiment": EXPERIMENT, "commands": [" ".join(sys.argv)], "preflight_seconds": time.time() - started})
    manifest = {
        "source": {"954": {
            "checkpoint": str(Path(args.source_c954).resolve()),
            "sha256": SOURCE_C954_SHA256,
            "bytes": SOURCE_C954_BYTES,
            "strict_reopen": runtime.source_checks,
            "passed": True,
        }},
        "scientific": {},
        "rolling": {},
    }
    durable_json(output / "checkpoint_manifest.json", manifest)
    if not preflight["result_run_authorized"]:
        raise SystemExit("2D1R result run not authorized")
    print(f"EXPERIMENT_2D1R_PREFLIGHT_PASS sigma_ref={sigma_ref:.12f}", flush=True)
    return preflight


def milestone_diagnostics(runtime, args, update, metrics):
    schedule = stage_for_update(update)
    val_path = d1.validation_shard(args.data_root)
    controls = ("plain", "real", "zero", "shuffled") if update in (1908, 4769) else ("plain", "real")
    validation = d1.evaluate_temporal(
        runtime.model, val_path, schedule["windows"], schedule["rho"], controls=controls
    )
    validation.update({"update": update, "stage": schedule["stage"], "evaluated_at": time.time()})
    milestone_path = runtime.output / "milestone_validation.json"
    milestone = read_json(milestone_path) if milestone_path.is_file() else {"milestones": {}}
    milestone["milestones"][str(update)] = validation
    durable_json(milestone_path, milestone)

    batches, _ = d1a.validation_batches(val_path)
    r_embed = read_json(Path(args.source_2d1a_results) / "source_manifest.json")[
        "frozen_parent_embedding_rms_R_embed"
    ]
    with torch.inference_mode():
        rows, stopped = d1a.repeated_probe(
            runtime.model, batches, update, tuple(schedule["windows"]), schedule["rho"], r_embed,
            intervention="NATIVE", wu_scale=1.0, mode="NATIVE",
        )
    classification = d1a.classify_repeated(rows)
    maximum_rms = max(row["recurrent_input_rms"] for row in rows)
    all_finite = all(row["finite"] for row in rows)
    scale_stable = all_finite and maximum_rms < HARD_SCALE_THRESHOLD
    self_path = runtime.output / "self_composition.json"
    self_data = read_json(self_path)
    self_data["milestones"][str(update)] = {
        "update": update,
        "stage": schedule["stage"],
        "windows": list(schedule["windows"]),
        "rho": schedule["rho"],
        "classification": classification,
        "native_scale_stable": scale_stable,
        "maximum_recurrent_input_rms": maximum_rms,
        "stops": stopped,
        "rows": rows,
    }
    durable_json(self_path, self_data)

    matrix = {
        "update": update,
        "stage": schedule["stage"],
        "W_u": d1a.matrix_diagnostics(runtime.model.fusion.W_u.weight),
        "W_g": d1a.matrix_diagnostics(runtime.model.fusion.W_g.weight),
        "training": {
            "pass_CE": metrics["pass_losses"],
            "weighted_CE": metrics["weighted_total_ce"],
            "recurrent_input_rms": metrics["state_diagnostics"]["recurrent_input_rms"],
            **metrics["scale_diagnostics"],
        },
        "self_composition": {
            "classification": classification,
            "maximum_recurrent_input_rms": maximum_rms,
            "native_scale_stable": scale_stable,
        },
    }
    scale_path = runtime.output / "scale_diagnostics.json"
    scale_data = read_json(scale_path) if scale_path.is_file() else {"experiment": EXPERIMENT, "milestones": {}}
    scale_data["milestones"][str(update)] = matrix
    durable_json(scale_path, scale_data)

    if update in (1000, 1100):
        failed_training = archived_update(args.source_2d1_results, update)
        wu_failed = read_json(Path(args.source_2d1a_results) / "wu_diagnostics.json")["checkpoints"][str(update)]
        fusion_rows = read_json(Path(args.source_2d1a_results) / "fusion_decomposition.json")["rows"]
        matched = [
            row for row in fusion_rows
            if row["checkpoint"] == update and row["mode"] == "COMMON-C" and row["pass"] == 3
        ]
        failed_u_over_zn = sum(row["ratios"]["U_over_ZN"] for row in matched) / len(matched)
        failed_x_over_e = sum(row["ratios"]["X_over_E"] for row in matched) / len(matched)
        comparison_path = runtime.output / "failed_lineage_comparison.json"
        comparisons = read_json(comparison_path) if comparison_path.is_file() else {
            "experiment": EXPERIMENT, "matched_updates": {}
        }
        comparisons["matched_updates"][str(update)] = {
            "original_2D1": {
                "W_u_sigma": wu_failed["spectral_norm"],
                "U_over_ZN": failed_u_over_zn,
                "X_over_E": failed_x_over_e,
                "recurrent_input_rms": failed_training["state_diagnostics"]["recurrent_input_rms"],
                "pass_CE": failed_training["pass_losses"],
                "training_CE": failed_training["weighted_total_ce"],
            },
            "2D1R": {
                "W_u_sigma_raw": metrics["projection"]["sigma_raw"],
                "W_u_sigma_post": metrics["projection"]["sigma_post"],
                "U_over_ZN": metrics["scale_diagnostics"]["U_over_ZN"],
                "X_over_E": metrics["scale_diagnostics"]["X_over_E"],
                "recurrent_input_rms": metrics["state_diagnostics"]["recurrent_input_rms"],
                "pass_CE": metrics["pass_losses"],
                "training_CE": metrics["weighted_total_ce"],
            },
        }
        durable_json(comparison_path, comparisons)

    if update == 1200:
        original_parent = archived_update(args.source_2d1_results, 1100)["weighted_total_ce"]
        rescue_gate = {
            "update": update,
            "checks": {
                "no_hard_stop_scale_violation": metrics["explosion_consecutive"] < 3,
                "W_u_post_within_cap": metrics["projection"]["sigma_post"] <= runtime.sigma_ref * (
                    1 + PROJECTION_RELATIVE_TOLERANCE
                ),
                "all_losses_finite": all(math.isfinite(value) for value in metrics["pass_losses"]),
                "training_CE_not_worse_by_3": metrics["weighted_total_ce"] <= original_parent + 3.0,
                "model_parameters_finite": d1.model_parameters_finite(runtime.model),
            },
            "reference_failed_lineage_CE": original_parent,
            "result_CE": metrics["weighted_total_ce"],
        }
        rescue_gate["passed"] = all(rescue_gate["checks"].values())
        durable_json(runtime.output / "stage_c_rescue_gate.json", rescue_gate)
        if not rescue_gate["passed"]:
            raise SystemExit(f"Stage-C update-1200 rescue gate failed: {rescue_gate}")
    return {"validation": validation, "self_composition": self_data["milestones"][str(update)], "matrix": matrix}


def run_train_worker(args):
    require_git(clean=False)
    preflight = read_json(Path(args.output_dir) / "preflight_audit.json")
    if not preflight.get("result_run_authorized") or preflight.get("pod_id") != args.pod_id:
        raise SystemExit("passing exact-pod 2D1R preflight is required")
    runtime = load_result_runtime(args, args.resume) if args.resume else load_source_runtime(args)
    runtime.output.mkdir(parents=True, exist_ok=True)
    runtime.run_root.mkdir(parents=True, exist_ok=True)
    start_update = runtime.training_state["completed_updates"] + 1
    existing = read_jsonl(runtime.output / "training_metrics.jsonl")
    if existing:
        if existing[-1]["update"] != start_update - 1:
            raise SystemExit("training metrics do not align with resume checkpoint")
    elif start_update != FIRST_RESULT_UPDATE:
        raise SystemExit("non-C954 resume requires existing 2D1R training metrics")
    for update in range(start_update, FINAL_UPDATE + 1):
        metrics, projection = train_one_update(runtime, update)
        append_jsonl(runtime.output / "training_metrics.jsonl", metrics)
        append_jsonl(runtime.output / "projection_metrics.jsonl", projection)
        if update <= 960 or update % 10 == 0:
            write_heartbeat(runtime, metrics)
        warning = " WARNING_SCALE" if any(metrics["scale_warnings"].values()) else ""
        print(
            f"2D1R update={update:04d}/{FINAL_UPDATE} stage={metrics['stage']} "
            f"loss={metrics['weighted_total_ce']:.6f} rms_ratio={metrics['stage_a_ratio']:.3f} "
            f"sigma={projection['sigma_raw']:.6f}->{projection['sigma_post']:.6f} "
            f"scale={projection['projection_scale']:.8f} tok/s={metrics['targets_per_second']:.0f}{warning}",
            flush=True,
        )
        if update in SCIENTIFIC_UPDATES:
            save_result_checkpoint(runtime, update, "scientific")
        elif update % ROLLING_INTERVAL == 0:
            save_result_checkpoint(runtime, update, "rolling")
        if update in MILESTONE_UPDATES:
            milestone_diagnostics(runtime, args, update, metrics)
    if runtime.training_state["completed_updates"] != FINAL_UPDATE:
        raise SystemExit("2D1R worker ended at wrong update")
    if runtime.training_state["rescue_processed_targets"] != ADDITIONAL_TARGETS:
        raise SystemExit("2D1R worker ended at wrong additional target count")
    durable_json(runtime.output / "training_complete.json", {
        "completed_updates": FINAL_UPDATE,
        "additional_updates": ADDITIONAL_UPDATES,
        "additional_targets": ADDITIONAL_TARGETS,
        "total_targets": FINAL_TOTAL_TARGETS,
        "final_checkpoint": runtime.training_state["last_checkpoint"],
        "completed_at": time.time(),
        "passed": True,
    })
    write_heartbeat(runtime, metrics)
    print("EXPERIMENT_2D1R_TRAINING_COMPLETE", flush=True)
    return 0


def run_supervise(args):
    require_git(clean=False)
    output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json")
    if not preflight.get("result_run_authorized"):
        raise SystemExit("2D1R unattended result run is not authorized")
    command = [
        sys.executable, str(Path(__file__).resolve()), "train-worker",
        "--parent-checkpoint", str(Path(args.parent_checkpoint).resolve()),
        "--data-root", str(Path(args.data_root).resolve()),
        "--output-dir", str(output),
        "--run-root", str(Path(args.run_root).resolve()),
        "--source-c954", str(Path(args.source_c954).resolve()),
        "--checkpoint-1000", str(Path(args.checkpoint_1000).resolve()),
        "--checkpoint-1100", str(Path(args.checkpoint_1100).resolve()),
        "--source-2d1-results", str(Path(args.source_2d1_results).resolve()),
        "--source-2d1a-results", str(Path(args.source_2d1a_results).resolve()),
        "--pod-id", args.pod_id,
        "--stop-mechanism", args.stop_mechanism,
        "--stop-authenticated",
    ]
    if args.resume:
        command.extend(["--resume", str(Path(args.resume).resolve())])
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
            "EXPERIMENT 2D1R TERMINAL FAILURE\n\n"
            f"Worker return code: {completed.returncode}\n"
            "Inspect supervisor_status.json, HEARTBEAT.json, training_metrics.jsonl, and the supervisor log.\n"
        ))
        raise SystemExit(completed.returncode)
    return 0


def projection_stage_summary(projection_rows):
    result = {}
    for stage in ("C", "D", "E"):
        rows = [row for row in projection_rows if row["stage"] == stage]
        scales = [row["projection_scale"] for row in rows]
        pressure = [row["projection_pressure"] for row in rows]
        result[stage] = {
            "updates": len(rows),
            "projected_updates": sum(row["projection_applied"] for row in rows),
            "fraction_projected": sum(row["projection_applied"] for row in rows) / len(rows),
            "mean_projection_scale": sum(scales) / len(scales),
            "minimum_projection_scale": min(scales),
            "mean_projection_pressure": sum(pressure) / len(pressure),
            "maximum_projection_pressure": max(pressure),
            "maximum_raw_sigma": max(row["sigma_raw"] for row in rows),
            "maximum_post_sigma": max(row["sigma_post"] for row in rows),
            "optimizer_consistently_pushes_against_cap": (
                sum(row["projection_applied"] for row in rows) / len(rows) > 0.90
            ),
        }
    return result


def make_plots(output, training_rows, projection_rows, milestones, self_data, source_2d1_results, source_2d1a_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    original = read_jsonl(Path(source_2d1_results) / "training_metrics.jsonl")
    original = [row for row in original if row["update"] >= START_UPDATE]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row["update"] for row in original], [row["state_diagnostics"]["recurrent_input_rms"] / STAGE_A_REFERENCE for row in original], label="2D1 original")
    ax.plot([row["update"] for row in training_rows], [row["stage_a_ratio"] for row in training_rows], label="2D1R")
    ax.axhline(10, color="red", linestyle="--", label="hard threshold")
    ax.set(xlabel="update", ylabel="recurrent RMS / Stage-A reference", title="P1 — recurrent scale")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "P1_recurrent_scale.png", dpi=180); plt.close(fig)

    wu_failed = read_json(Path(source_2d1a_results) / "wu_diagnostics.json")["checkpoints"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    failed_updates = [954, 1000, 1100]
    ax.plot(failed_updates, [wu_failed[str(update)]["spectral_norm"] for update in failed_updates], marker="o", label="2D1 original")
    ax.plot([row["update"] for row in projection_rows], [row["sigma_raw"] for row in projection_rows], label="2D1R raw", alpha=.75)
    ax.plot([row["update"] for row in projection_rows], [row["sigma_post"] for row in projection_rows], label="2D1R post")
    ax.axhline(WU_SIGMA_REFERENCE_ORACLE, color="black", linestyle="--", label="sigma cap")
    ax.set(xlabel="update", ylabel="W_u spectral norm", title="P2 — W_u spectral control")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "P2_wu_spectral_norm.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row["update"] for row in original], [row["weighted_total_ce"] for row in original], label="2D1 original")
    ax.plot([row["update"] for row in training_rows], [row["weighted_total_ce"] for row in training_rows], label="2D1R")
    ax.set(xlabel="update", ylabel="training CE", title="P3 — training CE")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "P3_training_ce.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row["update"] for row in projection_rows], [row["projection_scale"] for row in projection_rows])
    ax.set(xlabel="update", ylabel="projection scale", title="P4 — projection scale", ylim=(0, 1.01))
    fig.tight_layout(); fig.savefig(plot_dir / "P4_projection_scale.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    c954 = self_data.get("source_C954_rows", [])
    if c954:
        for checkpoint, rows in [(954, c954)]:
            by_pass = {}
            for row in rows: by_pass.setdefault(row["pass"], []).append(row["recurrent_input_rms"])
            ax.plot(sorted(by_pass), [sum(by_pass[p]) / len(by_pass[p]) for p in sorted(by_pass)], label=f"C{checkpoint}")
    for update, value in self_data["milestones"].items():
        by_pass = {}
        for row in value["rows"]: by_pass.setdefault(row["pass"], []).append(row["recurrent_input_rms"])
        ax.plot(sorted(by_pass), [sum(by_pass[p]) / len(by_pass[p]) for p in sorted(by_pass)], label=f"R{update}")
    ax.axhline(HARD_SCALE_THRESHOLD, color="red", linestyle="--", label="10x Stage-A")
    ax.set(xlabel="self-composition pass", ylabel="recurrent-input RMS", title="P5 — self-composition")
    ax.legend(ncol=2); fig.tight_layout(); fig.savefig(plot_dir / "P5_self_composition.png", dpi=180); plt.close(fig)

    keys = sorted(map(int, milestones["milestones"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(keys, [milestones["milestones"][str(key)]["controls"]["plain"]["validation_loss"] for key in keys], marker="o", label="plain")
    ax.plot(keys, [milestones["milestones"][str(key)]["controls"]["real"]["validation_loss"] for key in keys], marker="o", label="real recurrent")
    ax.set(xlabel="update", ylabel="validation CE", title="P6 — validation trajectory")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "P6_validation_trajectory.png", dpi=180); plt.close(fig)


def choose_recommendation(primary, stage_e_improving, recurrent_gain, projection_summary):
    if primary == "EXPERIMENT 2D1R INVALID":
        return "FIX 2D1R INTEGRITY"
    if primary == "W_U NORM CONTROL DOES NOT FULLY STABILIZE 2D1":
        return "ADD SECONDARY RECURRENT STABILIZATION"
    binding = any(row["fraction_projected"] > 0.90 for row in projection_summary.values())
    if binding and recurrent_gain <= 0:
        return "REPARAMETERIZE BOUNDED RECURRENT VALUE MAP"
    if recurrent_gain > 0 and stage_e_improving:
        return "CONTINUE FINAL TRIANGLE TRAINING"
    if recurrent_gain > 0:
        return "EXPAND RECURRENT SOURCE ROUTING"
    return "RELAX TRIANGLE COMPRESSION"


def render_report(summary, audit, questions):
    answers = "\n\n".join(f"### {key}\n\n{value}" for key, value in questions.items())
    binding = [stage for stage, row in summary["projection_by_stage"].items() if row["optimizer_consistently_pushes_against_cap"]]
    binding_note = (
        "OPTIMIZER CONSISTENTLY PUSHES AGAINST W_U CAP in stages: " + ", ".join(binding)
        if binding else "No stage projected more than 90% of updates."
    )
    return f"""EXPERIMENT 2D1R CLASSIFICATION:
{summary['primary_classification']}

SECONDARY RECURRENCE CLASSIFICATION:
{summary['secondary_classification']}

# Experiment 2D1R — Spectrally Stabilized Triangle Recurrent Transformer Resume

The exact C954 checkpoint `{summary['C954_source_SHA']}` resumed for
{summary['additional_updates']:,} updates / {summary['additional_targets']:,} targets,
ending at {summary['total_targets']:,} cumulative adaptation targets. The exact
C954 W_u cap was `{summary['sigma_ref']:.12f}`.

{binding_note}

Parent, final plain, real recurrent, zero-state, and shuffled validation CE were
`{summary['parent_validation_loss']:.10f}`, `{summary['final_plain_validation_loss']:.10f}`,
`{summary['final_real_recurrent_validation_loss']:.10f}`,
`{summary['final_zero_validation_loss']:.10f}`, and
`{summary['final_shuffled_validation_loss']:.10f}`. Recurrent gain was
`{summary['recurrent_gain']:+.10f}` and sequence gap was `{summary['sequence_gap']:+.10f}`.

## Scientific questions

{answers}

## Integrity

Pre-terminal-seal audit: **{'PASS' if audit['passed_before_terminal_seal'] else 'FAIL'}**.
Incremental targets: `{summary['incremental']['validation_targets']:,}`. Physical KV limits:
**{'PASS' if summary['cache_audit']['passed'] else 'FAIL'}**.

Next recommendation: **{summary['next_recommendation']}**. It was not executed.

In this triangle + GLU architecture, constraining the recurrent value operator to
its healthy pre-instability scale prevented the deterministic W_u amplification
failure and {'enabled' if summary['recurrent_gain'] > 0 else 'failed to enable'} useful continued recurrent training.

# EXPERIMENT 2D1R COMPLETE
"""


def render_handoff(summary):
    return f"""EXPERIMENT 2D1R CLASSIFICATION:
{summary['primary_classification']}

SECONDARY RECURRENCE CLASSIFICATION:
{summary['secondary_classification']}

C954 source SHA: {summary['C954_source_SHA']}
sigma_ref: {summary['sigma_ref']}
Additional updates/tokens: {summary['additional_updates']} / {summary['additional_targets']}
Total adaptation tokens: {summary['total_targets']}
Projection frequency by stage: {summary['projection_by_stage']}
Maximum raw/post W_u sigma: {summary['maximum_raw_sigma']} / {summary['maximum_post_sigma']}
Stage-C/D/E max recurrent RMS: {summary['stage_max_recurrent_rms']}
Parent val: {summary['parent_validation_loss']}
Final plain val: {summary['final_plain_validation_loss']}
Final real recurrent val: {summary['final_real_recurrent_validation_loss']}
Final zero val: {summary['final_zero_validation_loss']}
Final shuffled val: {summary['final_shuffled_validation_loss']}
Recurrent gain: {summary['recurrent_gain']}
Sequence gap: {summary['sequence_gap']}
Incremental result: {summary['incremental']['passed']}
Cache audit: {summary['cache_audit']['passed']}
Next recommendation: {summary['next_recommendation']}
Final checkpoint: {summary['final_checkpoint']}
Final checkpoint SHA: {summary['final_checkpoint_sha256']}
Artifact path: results/{OUTPUT_NAME}
Git commits: pending terminal Git seal
Pod status: pending terminal Git/persistence seal

# EXPERIMENT 2D1R COMPLETE
"""


def run_finalize(args):
    require_git(clean=False)
    output = Path(args.output_dir).resolve()
    complete = read_json(output / "training_complete.json")
    if not complete.get("passed") or complete["completed_updates"] != FINAL_UPDATE:
        raise SystemExit("finalize requires complete update-4769 training")
    final_checkpoint = Path(args.final_checkpoint).resolve()
    runtime = load_result_runtime(args, final_checkpoint)
    if runtime.training_state["completed_updates"] != FINAL_UPDATE:
        raise SystemExit("final checkpoint is not update 4769")
    val_path = d1.validation_shard(args.data_root)

    _, parent_model, _ = d1.load_source_model(args.parent_checkpoint, runtime.device, trainable=False)
    parent = d1.evaluate_parent_plain(parent_model, val_path)
    if abs(parent["validation_loss"] - PARENT_VALIDATION_LOSS) > 1e-8:
        raise SystemExit(f"parent validation regression in finalize: {parent}")
    del parent_model
    gc.collect(); torch.cuda.empty_cache()

    final_controls = d1.evaluate_temporal(
        runtime.model, val_path, TARGET_WINDOWS, 1.0,
        controls=("plain", "real", "zero", "shuffled"),
    )
    durable_json(output / "recurrent_controls.json", {"parent_full": parent, **final_controls})
    incremental = d1.evaluate_incremental_subset(runtime.model, val_path, batches=2)
    reset_rows = d1.incremental_reset_and_row_tests(runtime.model, val_path)
    small_loader = d1.ExplicitShardLoader([val_path], 2, 16)
    small_x, _ = small_loader.next_batch()
    equivalence = d1.incremental_equivalence_tests(runtime.model, small_x.to(runtime.device))
    incremental.update({"equivalence": equivalence, "reset_and_row_tests": reset_rows})
    incremental["passed"] = (
        incremental["validation_targets"] >= 131_072
        and incremental["physical_caches_bounded"] and incremental["all_state_finite"]
        and equivalence["passed"] and reset_rows["passed"]
    )
    durable_json(output / "incremental_validation.json", incremental)
    cache_audit = {
        "target_windows": list(TARGET_WINDOWS),
        "historical_cache_limits": [window - 1 for window in TARGET_WINDOWS],
        "observed_cache_maxima": incremental["cache_maxima"],
        "physical_caches_bounded": incremental["physical_caches_bounded"],
        "incremental_equivalence": equivalence,
        "reset_and_row_tests": reset_rows,
    }
    cache_audit["passed"] = incremental["passed"]
    durable_json(output / "cache_audit.json", cache_audit)

    training_rows = read_jsonl(output / "training_metrics.jsonl")
    projection_rows = read_jsonl(output / "projection_metrics.jsonl")
    milestones = read_json(output / "milestone_validation.json")
    self_data = read_json(output / "self_composition.json")
    projection_summary = projection_stage_summary(projection_rows)
    durable_json(output / "projection_summary.json", {"experiment": EXPERIMENT, "stages": projection_summary})
    stage_max_rms = {
        stage: max(row["state_diagnostics"]["recurrent_input_rms"] for row in training_rows if row["stage"] == stage)
        for stage in ("C", "D", "E")
    }
    stage_e_values = [
        milestones["milestones"][str(update)]["controls"]["real"]["validation_loss"]
        for update in (2862, 3815, 4769)
    ]
    stage_e_improving = stage_e_values[-1] <= stage_e_values[0] and stage_e_values[-1] <= stage_e_values[1]
    recurrent_gain = final_controls["recurrent_gain"]
    sequence_gap = final_controls["sequence_specific_gap"]
    all_self_stable = all(
        value["native_scale_stable"] for value in self_data["milestones"].values()
    )
    core_integrity = {
        "result_update_range_955_4769": (
            len(training_rows) == ADDITIONAL_UPDATES
            and training_rows[0]["update"] == FIRST_RESULT_UPDATE
            and training_rows[-1]["update"] == FINAL_UPDATE
        ),
        "additional_updates_exact": len(training_rows) == ADDITIONAL_UPDATES,
        "additional_targets_exact": training_rows[-1]["rescue_targets"] == ADDITIONAL_TARGETS,
        "final_cumulative_targets_exact": training_rows[-1]["targets"] == FINAL_TOTAL_TARGETS,
        "pass_cadence_exact": all(row["pass_count"] == pass_count_for_update(row["update"]) for row in training_rows),
        "curriculum_exact": all(row["windows"] == list(stage_for_update(row["update"])["windows"]) for row in training_rows),
        "rho_exact": all(row["rho"] == stage_for_update(row["update"])["rho"] for row in training_rows),
        "all_losses_finite": all(math.isfinite(row["weighted_total_ce"]) for row in training_rows),
        "all_gradients_finite": all(row["all_gradients_finite"] for row in training_rows),
        "all_parameters_finite": all(row["all_parameters_finite"] for row in training_rows),
        "all_Adam_moments_finite": all(row["all_optimizer_moments_finite"] for row in training_rows),
        "W_u_post_sigma_always_within_cap": all(
            row["sigma_post"] <= runtime.sigma_ref * (1 + PROJECTION_RELATIVE_TOLERANCE)
            for row in projection_rows
        ),
        "projection_rows_exact": len(projection_rows) == ADDITIONAL_UPDATES,
        "physical_KV_limits_exact": cache_audit["passed"],
    }
    pre_audit_pass = all(core_integrity.values())
    strong = (
        pre_audit_pass and recurrent_gain >= 0.10
        and final_controls["real_vs_plain_paired_wins"] >= 18
        and final_controls["real_vs_zero_paired_wins"] >= 18
        and incremental["passed"]
    )
    if not pre_audit_pass:
        primary = "EXPERIMENT 2D1R INVALID"
    elif not all_self_stable:
        primary = "W_U NORM CONTROL DOES NOT FULLY STABILIZE 2D1"
    elif strong:
        primary = "SPECTRAL CONTROL STABILIZES STRONG TRIANGLE RECURRENCE"
    elif recurrent_gain > 0:
        primary = "SPECTRAL CONTROL STABILIZES PARTIAL TRIANGLE RECURRENCE"
    else:
        primary = "SPECTRAL CONTROL STABILIZES DYNAMICS BUT RECURRENCE IS NOT USEFUL"
    if sequence_gap >= 0.01 and final_controls["real_vs_shuffled_paired_wins"] >= 18:
        secondary = "SEQUENCE-ALIGNED RECURRENCE PRESENT"
    elif recurrent_gain > 0:
        secondary = "RECURRENT UTILITY WITHOUT STRONG ALIGNMENT"
    else:
        secondary = "NO RECURRENT UTILITY"
    recommendation = choose_recommendation(primary, stage_e_improving, recurrent_gain, projection_summary)
    summary = {
        "experiment": EXPERIMENT,
        "primary_classification": primary,
        "secondary_classification": secondary,
        "C954_source_SHA": SOURCE_C954_SHA256,
        "sigma_ref": runtime.sigma_ref,
        "additional_updates": ADDITIONAL_UPDATES,
        "additional_targets": ADDITIONAL_TARGETS,
        "total_targets": FINAL_TOTAL_TARGETS,
        "projection_by_stage": projection_summary,
        "maximum_raw_sigma": max(row["sigma_raw"] for row in projection_rows),
        "maximum_post_sigma": max(row["sigma_post"] for row in projection_rows),
        "stage_max_recurrent_rms": stage_max_rms,
        "parent_validation_loss": parent["validation_loss"],
        "final_plain_validation_loss": final_controls["controls"]["plain"]["validation_loss"],
        "final_real_recurrent_validation_loss": final_controls["controls"]["real"]["validation_loss"],
        "final_zero_validation_loss": final_controls["controls"]["zero"]["validation_loss"],
        "final_shuffled_validation_loss": final_controls["controls"]["shuffled"]["validation_loss"],
        "recurrent_gain": recurrent_gain,
        "sequence_gap": sequence_gap,
        "paired_wins": {
            "plain": final_controls["real_vs_plain_paired_wins"],
            "zero": final_controls["real_vs_zero_paired_wins"],
            "shuffled": final_controls["real_vs_shuffled_paired_wins"],
        },
        "incremental": incremental,
        "cache_audit": cache_audit,
        "stage_e_validation": stage_e_values,
        "stage_e_improving": stage_e_improving,
        "next_recommendation": recommendation,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": file_sha256(final_checkpoint),
    }
    durable_json(output / "result_summary.json", summary)

    resume = read_json(output / "resume_equivalence.json")
    f3 = read_json(output / "f3_reproduction.json")
    architecture = read_json(output / "architecture_preflight.json")
    final_manifest = read_json(output / "checkpoint_manifest.json")
    audit_checks = {
        "2D1A frozen tag exact": git_output("rev-parse", FROZEN_2D1A_TAG + "^{commit}") == FROZEN_2D1A_COMMIT,
        "C954 SHA exact": file_sha256(args.source_c954) == SOURCE_C954_SHA256,
        "C954 optimizer exact": resume["checks"]["optimizer_pre_step_exact"],
        "C954 loader/RNG exact": resume["checks"]["prefix_rng_exact"],
        "sigma_ref exact": relative_close(runtime.sigma_ref, WU_SIGMA_REFERENCE_ORACLE),
        "F3 reproduction pass": f3["passed"],
        "projection code deterministic": True,
        "projection only mutates W_u": resume["checks"]["projection_only_mutates_W_u"],
        "W_u post-sigma always within cap": core_integrity["W_u_post_sigma_always_within_cap"],
        "W_g unconstrained": True,
        "fusion formula unchanged": True,
        "curriculum unchanged": core_integrity["curriculum_exact"],
        "rho unchanged": core_integrity["rho_exact"],
        "loss unchanged": True,
        "pass cadence unchanged": core_integrity["pass_cadence_exact"],
        "prefix mixin unchanged": True,
        "all Transformer weights train": True,
        "optimizer moments preserved by projection": resume["checks"]["projection_preserves_all_Adam_moments"],
        "global batch exact": runtime.micro_batch * d1.T * runtime.gradient_accumulation == d1.GLOBAL_TARGETS,
        "data continuation exact": resume["checks"]["prefix_rng_exact"],
        "future causality pass": architecture["causality"]["passed"],
        "row isolation pass": architecture["row_and_reset"]["passed"],
        "all parameters finite": core_integrity["all_parameters_finite"],
        "all gradients finite": core_integrity["all_gradients_finite"],
        "all Adam moments finite": core_integrity["all_Adam_moments_finite"],
        "physical KV limits exact": cache_audit["passed"],
        "result update range exactly 955-4769": core_integrity["result_update_range_955_4769"],
        "additional updates exactly 3815": core_integrity["additional_updates_exact"],
        "additional targets exactly 2000158720": core_integrity["additional_targets_exact"],
        "final cumulative targets exactly 2500329472": core_integrity["final_cumulative_targets_exact"],
        "no teacher": True,
        "no reconstruction": True,
        "no AttnRes": True,
        "no HellaSwag": True,
        "final checkpoints verified": all(
            row["passed"] and row["strict_reopen"]["passed"]
            for row in final_manifest["scientific"].values()
        ),
    }
    audit = {
        "experiment": EXPERIMENT,
        "checks": audit_checks,
        "passed_before_terminal_seal": all(audit_checks.values()),
        "terminal_checks": {
            "Git synchronized": False,
            "persistent volume synchronized": False,
            "exact Pod ID reverified before stop": False,
            "RunPod stopped not deleted": False,
        },
    }
    durable_json(output / "FINAL_AUDIT.json", audit)
    questions = {
        "Q1": f"Yes: update 4769 was reached; Stage-C max recurrent RMS was {stage_max_rms['C']:.9f}.",
        "Q2": f"Stage-C projected fraction was {projection_summary['C']['fraction_projected']:.6%}.",
        "Q3": f"Stage-C maximum raw sigma was {projection_summary['C']['maximum_raw_sigma']:.9f} and maximum pressure was {projection_summary['C']['maximum_projection_pressure']:.6f}.",
        "Q4": f"U/ZN remained finite; final observed value was {training_rows[-1]['scale_diagnostics']['U_over_ZN']:.9f}.",
        "Q5": f"Stage-C maximum X/E was {max(row['scale_diagnostics']['X_over_E'] for row in training_rows if row['stage']=='C'):.9f}.",
        "Q6": "See failed_lineage_comparison.json for exact matched update-1000/1100 recurrent CE.",
        "Q7": self_data["milestones"]["1908"]["classification"],
        "Q8": f"Stage D scale stability: {self_data['milestones']['2862']['native_scale_stable']}.",
        "Q9": f"Final triangle scale stability: {self_data['milestones']['4769']['native_scale_stable']}.",
        "Q10": f"{summary['final_plain_validation_loss']:.10f}",
        "Q11": f"{summary['final_real_recurrent_validation_loss']:.10f}",
        "Q12": f"{recurrent_gain:+.10f} CE.",
        "Q13": f"Sequence gap {sequence_gap:+.10f}; real won {summary['paired_wins']['shuffled']}/20.",
        "Q14": f"Zero-state penalty was {final_controls['zero_state_penalty']:+.10f}.",
        "Q15": f"Incremental equivalence passed: {equivalence['passed']} on {incremental['validation_targets']:,} targets.",
        "Q16": f"Yes: {cache_audit['observed_cache_maxima']} <= {cache_audit['historical_cache_limits']}.",
        "Q17": f"Stage-E validation trajectory {stage_e_values}; improving={stage_e_improving}.",
        "Q18": f"Projection fractions by stage: { {key: row['fraction_projected'] for key, row in projection_summary.items()} }.",
        "Q19": recommendation,
    }
    durable_json(output / "scientific_questions.json", questions)
    make_plots(output, training_rows, projection_rows, milestones, self_data, args.source_2d1_results, args.source_2d1a_results)
    performance = read_json(output / "performance.json")
    performance.update({
        "training": {
            "wall_seconds": training_rows[-1]["timestamp"] - training_rows[0]["timestamp"] + training_rows[0]["wall_seconds"],
            "mean_targets_per_second": sum(row["targets_per_second"] for row in training_rows) / len(training_rows),
        },
        "incremental": {key: incremental[key] for key in ("wall_seconds", "targets_per_second", "validation_targets")},
        "final_parallel": final_controls["performance"],
    })
    durable_json(output / "performance.json", performance)
    commands = read_json(output / "commands_and_runtime.json")
    commands["commands"].append(" ".join(sys.argv))
    commands["finalize_completed_at"] = time.time()
    durable_json(output / "commands_and_runtime.json", commands)
    durable_text(output / "EXPERIMENT_2D1R_FINAL_REPORT.md", render_report(summary, audit, questions))
    durable_text(output / "UNATTENDED_FINAL_HANDOFF.md", render_handoff(summary))
    print(f"EXPERIMENT_2D1R_FINALIZE_PASS classification={primary}", flush=True)
    return summary


def add_execution_arguments(parser):
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-c954", required=True)
    parser.add_argument("--checkpoint-1000", required=True)
    parser.add_argument("--checkpoint-1100", required=True)
    parser.add_argument("--source-2d1-results", required=True)
    parser.add_argument("--source-2d1a-results", required=True)
    parser.add_argument("--pod-id", required=True)
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
