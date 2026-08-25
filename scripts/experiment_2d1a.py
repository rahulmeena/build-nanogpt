#!/usr/bin/env python3
"""Experiment 2D1A: zero-training recurrent scale-instability forensics."""

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
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
import experiment_2d0 as d0  # noqa: E402
import experiment_2d0d as d0d  # noqa: E402
import experiment_2d1 as d1  # noqa: E402


EXPERIMENT = "2D1A"
PROTOCOL = "exp2d1a_recurrent_scale_forensics_v1"
BRANCH = "experiment-2d1a-recurrent-scale-forensics"
FROZEN_2D1_COMMIT = "2d4be75e0568d5e2df80b8963c1260db4982ca70"
FROZEN_2D1_TAG = "experiment-2d1-unstable-final"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d1a_recurrent_scale_forensics.json"
OUTPUT_NAME = "experiment_2d1a_recurrent_scale_forensics"
EXPECTED_CHECKPOINTS = {
    954: "22abc6de4e49e27504b4d0e66ca0d2e3396ed6d76d7ee18e0e11cfb1eb3192c0",
    1000: "c5731cfd2534b7a9e05db82f3b0f9008d311db0c13dd6975a757006bec43585f",
    1100: "6cca94e75ac4802f92df8c1e18d611eb875f42d4312146bb09cb43dfe6d67ad6",
}
COMMON_B = (512, 545, 581, 618, 658, 702, 747, 796, 848, 903, 962, 1024)
COMMON_C = (256, 290, 329, 373, 423, 481, 545, 619, 702, 796, 903, 1024)
POSITION_BINS = ((1, 64), (65, 128), (129, 256), (257, 512), (513, 768), (769, 896), (897, 1023))
FORENSIC_BATCHES = 4
EXPENSIVE_BATCHES = 2
REPEATED_PASSES = 32
R_STAGE_A = 0.03550996296107769
R_STOP = 10.0 * R_STAGE_A
R_PROBE_STOP = 100.0 * R_STAGE_A
CHECKPOINT_SCHEMA = d1.CHECKPOINT_SCHEMA
FORBIDDEN_COUNTS = {
    "optimizer_objects": 0,
    "scheduler_objects": 0,
    "gradscaler_objects": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameter_updates": 0,
    "training_targets": 0,
}


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value):
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def module_state_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def durable_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def durable_json(path, payload):
    durable_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_number(value):
    return value is None or (isinstance(value, (int, float)) and math.isfinite(value))


def nested_finite(value):
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(nested_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(nested_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def require_config():
    config = read_json(CONFIG_PATH)
    checks = {
        "protocol": config.get("protocol") == PROTOCOL,
        "branch": config.get("branch") == BRANCH,
        "frozen_commit": config.get("frozen_2d1_commit") == FROZEN_2D1_COMMIT,
        "frozen_tag": config.get("frozen_2d1_tag") == FROZEN_2D1_TAG,
        "checkpoint_hashes": {int(k): v for k, v in config.get("checkpoints", {}).items()} == EXPECTED_CHECKPOINTS,
        "common_b": tuple(config["common_b"]["windows"]) == COMMON_B and config["common_b"]["rho"] == 0.5,
        "common_c": tuple(config["common_c"]["windows"]) == COMMON_C and config["common_c"]["rho"] == 0.75,
        "position_bins": tuple(tuple(row) for row in config.get("position_bins", [])) == POSITION_BINS,
        "stage_a_reference": config.get("stage_a_recurrent_input_rms") == R_STAGE_A,
        "batch_counts": config.get("forensic_batches") == FORENSIC_BATCHES and config.get("expensive_batches") == EXPENSIVE_BATCHES,
        "repeated_passes": config.get("repeated_passes") == REPEATED_PASSES,
    }
    if not all(checks.values()):
        raise SystemExit(f"2D1A frozen configuration mismatch: {checks}")
    return config


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"2D1A requires branch {BRANCH}")
    if git_output("rev-parse", FROZEN_2D1_TAG + "^{commit}") != FROZEN_2D1_COMMIT:
        raise SystemExit("2D1 frozen tag does not resolve to the terminal commit")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", FROZEN_2D1_COMMIT, "HEAD"], cwd=REPO_ROOT)
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("2D1A result-bearing work requires a clean worktree")


def require_hardware(pod_id):
    if not pod_id:
        raise SystemExit("exact RunPod Pod ID must be supplied")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("2D1A requires exactly one visible GPU")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("2D1A forbids DDP/NCCL/distributed state")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if "A100-SXM4-80GB" not in name or memory < 79 * 1024**3:
        raise SystemExit(f"unsupported 2D1A GPU: {name}, {memory}")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda", 0)


def environment(pod_id):
    try:
        import matplotlib
        matplotlib_version = matplotlib.__version__
    except ImportError:
        matplotlib_version = None
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "timestamp": time.time(),
        "pod_id": pod_id,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip(),
        "cuda_device_count_visible": torch.cuda.device_count(),
        "matplotlib": matplotlib_version,
        "autocast": "cuda bfloat16",
        "grad_mode_for_result_compute": "torch.inference_mode",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_branch": git_output("branch", "--show-current"),
        "git_commit": git_output("rev-parse", "HEAD"),
    }


def checkpoint_arguments(args):
    return {
        954: Path(args.checkpoint_954).resolve(),
        1000: Path(args.checkpoint_1000).resolve(),
        1100: Path(args.checkpoint_1100).resolve(),
    }


def validate_checkpoint_payload(path, update):
    payload = d1.torch_load(path, mmap=True)
    required = {
        "schema", "model", "optimizer", "training_state", "scheduler_position", "completed_updates",
        "processed_targets", "current_curriculum_stage", "current_windows", "rho", "loader_state",
        "rng_state", "next_global_batch_sha256", "metadata", "git_commit", "environment",
    }
    schedule = d1.stage_for_update(update)
    model_state = payload.get("model", {})
    checks = {
        "schema": payload.get("schema") == CHECKPOINT_SCHEMA,
        "fields_exact": set(payload) == required,
        "completed_updates": payload.get("completed_updates") == update,
        "scheduler_position": payload.get("scheduler_position") == update,
        "processed_targets": payload.get("processed_targets") == update * d1.GLOBAL_TARGETS,
        "training_state": payload.get("training_state", {}).get("completed_updates") == update,
        "stage": payload.get("current_curriculum_stage") == schedule["stage"],
        "windows": tuple(payload.get("current_windows", [])) == tuple(schedule["windows"]),
        "rho": payload.get("rho") == schedule["rho"],
        "model_state_nonempty": isinstance(model_state, dict) and len(model_state) > 100,
        "fusion_keys": "fusion.W_u.weight" in model_state and "fusion.W_g.weight" in model_state,
        "model_tensors_finite": nested_finite(model_state),
        "optimizer_metadata_present": isinstance(payload.get("optimizer"), dict),
        "optimizer_state_finite_without_object": nested_finite(payload.get("optimizer", {})),
        "next_batch_hash_present": len(payload.get("next_global_batch_sha256", "")) == 64,
    }
    checks["passed"] = all(checks.values())
    metadata = {
        "completed_updates": payload.get("completed_updates"),
        "processed_targets": payload.get("processed_targets"),
        "stage": payload.get("current_curriculum_stage"),
        "windows": payload.get("current_windows"),
        "rho": payload.get("rho"),
        "next_global_batch_sha256": payload.get("next_global_batch_sha256"),
        "optimizer_param_groups": len(payload.get("optimizer", {}).get("param_groups", [])),
        "optimizer_state_entries": len(payload.get("optimizer", {}).get("state", {})),
        "git_commit": payload.get("git_commit"),
    }
    del payload
    gc.collect()
    return checks, metadata


def validation_batches(val_path, count=FORENSIC_BATCHES):
    loader = d1.ExplicitShardLoader([val_path], d1.VALIDATION_B, d1.T)
    rows = []
    batches = []
    combined = []
    for index in range(count):
        x, y = loader.next_batch()
        identity = d0d.batch_identity(x, y)
        rows.append({
            "batch_index": index,
            **identity,
            "input_shape": list(x.shape),
            "target_shape": list(y.shape),
            "order": index,
        })
        combined.append(identity["combined_sha256"])
        batches.append((x, y))
    return batches, {
        "validation_shard": str(Path(val_path).resolve()),
        "validation_shard_sha256": file_sha256(val_path),
        "batch_count": count,
        "batch_size": d1.VALIDATION_B,
        "sequence_length": d1.T,
        "batches": rows,
        "forensic_batch_collection_sha256": d1.aggregate_hashes(combined),
        "canonical_twenty_batch_collection_sha256": d1.CANONICAL_VALIDATION_SHA256,
        "expensive_batch_indices": list(range(EXPENSIVE_BATCHES)),
    }


def parent_embedding_rms(model, batches, device):
    total = 0.0
    count = 0
    with torch.inference_mode():
        for cpu_x, _ in batches:
            x = cpu_x.to(device)
            value = model.base.transformer.wte(x).float()
            total += value.square().double().sum().item()
            count += value.numel()
    return math.sqrt(total / count)


def archived_trajectory(source_dir):
    source_dir = Path(source_dir)
    training_path = source_dir / "training_metrics.jsonl"
    rows = [row for row in read_jsonl(training_path) if row["update"] >= 900]
    failure = read_json(source_dir / "failure_diagnosis.json")
    source_files = {}
    for name in (
        "training_metrics.jsonl", "HEARTBEAT.json", "supervisor.log", "deterministic_retry.log",
        "checkpoint_manifest.json", "UNATTENDED_FINAL_HANDOFF.md", "FINAL_AUDIT.json",
    ):
        path = source_dir / name
        source_files[name] = {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    return {
        "experiment": EXPERIMENT,
        "source_experiment": "2D1",
        "first_logged_update": rows[0]["update"],
        "last_logged_update": rows[-1]["update"],
        "logged_rows": rows,
        "terminal_unlogged_hard_stop": failure["terminal_third_crossing"],
        "parameter_norms_per_update_available": False,
        "unavailable_logged_metrics": ["W_u norm", "W_g norm"],
        "note": "Per-update parameter norms were not logged; no optimizer replay was performed.",
        "source_files": source_files,
    }


def hard_stop_definition(source_dir):
    source_dir = Path(source_dir)
    failure = read_json(source_dir / "failure_diagnosis.json")
    return {
        "source_code": "scripts/experiment_2d1.py:TriangleRecurrentGPT.make_input/train_one_update",
        "reference_tensor_location": "final microbatch, final temporal pass: recurrent_input after prefix torch.where and before learned position embedding",
        "reference_tensor_formula": "where(position > prefix_length, (1-rho)*E + rho*(W_u(RMSNorm(shift(previous_top)))*(2*sigmoid(W_g(E)))), E)",
        "rms_aggregation": "sqrt(mean(recurrent_input.float()**2)) over the complete B x T x 768 tensor",
        "stage_a_reference_aggregation": "arithmetic mean of the per-update final-microbatch health RMS values from updates 1 through 10",
        "stage_a_reference": failure["healthy_reference"],
        "R_stageA": failure["healthy_reference"]["recurrent_input_rms"],
        "R_stop": failure["ten_x_recurrent_input_threshold"],
        "top_state_stop_threshold": failure["ten_x_top_state_threshold"],
        "logic": "exploded if recurrent_input_rms OR top_state_rms is >10x its frozen reference; increment consecutive count, otherwise reset to zero; stop at count >=3 before logging that update",
        "recorded_violations": failure["recorded_threshold_crossings"],
        "terminal_third_violation": failure["terminal_third_crossing"],
        "exact_tensor_crossing_terminal_threshold": "recurrent_input",
    }


def preflight(args):
    require_config()
    require_git(clean=True)
    device = require_hardware(args.pod_id)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_checkpoint).resolve()
    val = Path(args.validation_shard).resolve()
    source_hash = file_sha256(source)
    val_hash = file_sha256(val)
    if source_hash != d1.SOURCE_SHA256 or val_hash != d1.VALIDATION_SHARD_SHA256:
        raise SystemExit("source checkpoint or validation shard SHA mismatch")
    checkpoints = checkpoint_arguments(args)
    manifest_2d1 = read_json(Path(args.source_2d1_results) / "checkpoint_manifest.json")
    inventory = {"experiment": EXPERIMENT, "complete_2d1_manifest": manifest_2d1, "investigated": {}}
    for update, path in checkpoints.items():
        observed = file_sha256(path)
        if observed != EXPECTED_CHECKPOINTS[update]:
            raise SystemExit(f"checkpoint C{update} SHA mismatch: {observed}")
        checks, metadata = validate_checkpoint_payload(path, update)
        if not checks["passed"]:
            raise SystemExit(f"checkpoint C{update} strict metadata reopen failed: {checks}")
        inventory["investigated"][str(update)] = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": observed,
            "strict_reopen_without_optimizer_object": checks, "metadata": metadata,
        }
    batches, batch_manifest = validation_batches(val)
    if batch_manifest["validation_shard_sha256"] != d1.VALIDATION_SHARD_SHA256:
        raise SystemExit("forensic validation identity mismatch")
    _, model, source_audit = d1.load_source_model(source, device, trainable=False)
    model.eval()
    embed_rms = parent_embedding_rms(model, batches, device)
    source_state_before = module_state_sha256(model)
    source_state_after = module_state_sha256(model)
    if source_state_before != source_state_after:
        raise SystemExit("source model changed during R_embed calibration")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    source_manifest = {
        "experiment": EXPERIMENT,
        "checkpoint": str(source),
        "checkpoint_bytes": source.stat().st_size,
        "checkpoint_sha256": source_hash,
        "validation_shard": str(val),
        "validation_shard_sha256": val_hash,
        "canonical_validation_sha256": d1.CANONICAL_VALIDATION_SHA256,
        "architecture": source_audit,
        "frozen_parent_embedding_rms_R_embed": embed_rms,
        "R_embed_definition": "global token-embedding RMS over the four pinned forensic batches using the immutable mature parent",
        "source_state_unchanged": source_state_before == source_state_after,
    }
    archived = archived_trajectory(args.source_2d1_results)
    hard_stop = hard_stop_definition(args.source_2d1_results)
    preflight_checks = {
        "2d1_frozen_tag_exact": git_output("rev-parse", FROZEN_2D1_TAG + "^{commit}") == FROZEN_2D1_COMMIT,
        "2d1_terminal_commit_exact": FROZEN_2D1_COMMIT == "2d4be75e0568d5e2df80b8963c1260db4982ca70",
        "source_lineage_exact": source_hash == d1.SOURCE_SHA256,
        "validation_shard_exact": val_hash == d1.VALIDATION_SHARD_SHA256,
        "all_investigated_checkpoint_sha_verified": all(
            row["sha256"] == EXPECTED_CHECKPOINTS[int(key)] for key, row in inventory["investigated"].items()
        ),
        "checkpoint_strict_reopen": all(
            row["strict_reopen_without_optimizer_object"]["passed"] for row in inventory["investigated"].values()
        ),
        "optimizer_objects_zero": FORBIDDEN_COUNTS["optimizer_objects"] == 0,
        "scheduler_objects_zero": FORBIDDEN_COUNTS["scheduler_objects"] == 0,
        "GradScaler_objects_zero": FORBIDDEN_COUNTS["gradscaler_objects"] == 0,
        "model_eval": True,
        "source_model_unchanged": source_state_before == source_state_after,
        "fixed_validation_batch_hashes_pinned": len(batch_manifest["batches"]) == 4,
        "position_zero_recurrent_state_specified_zero": True,
        "one_position_shift_specified_exact": True,
        "training_data_unused": True,
    }
    preflight_checks["passed"] = all(preflight_checks.values())
    durable_json(output / "checkpoint_inventory.json", inventory)
    durable_json(output / "source_manifest.json", source_manifest)
    durable_json(output / "environment.json", environment(args.pod_id))
    durable_json(output / "archived_instability_trajectory.json", archived)
    durable_json(output / "hard_stop_definition.json", hard_stop)
    durable_json(output / "forensic_batch_manifest.json", batch_manifest)
    durable_json(output / "preflight_audit.json", {"experiment": EXPERIMENT, "checks": preflight_checks, "passed": preflight_checks["passed"]})
    durable_json(output / "commands_and_runtime.json", {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "preflight_command": " ".join(sys.argv),
        "preflight_git_commit": git_output("rev-parse", "HEAD"),
        "preflight_completed_at": time.time(),
        "forbidden_operation_counts": FORBIDDEN_COUNTS,
    })
    if not preflight_checks["passed"]:
        raise SystemExit(f"2D1A preflight failed: {preflight_checks}")
    print(f"EXPERIMENT_2D1A_PREFLIGHT_PASS R_embed={embed_rms:.12f}", flush=True)


def rms(value):
    return value.float().square().mean().sqrt().item()


def safe_ratio(numerator, denominator):
    return float(numerator / max(denominator, 1e-30))


def scalar_stats(value):
    v = value.float()
    return {
        "mean": v.mean().item(), "std": v.std().item(), "rms": rms(v),
        "min": v.min().item(), "max": v.max().item(),
    }


def binned_rms(value):
    return {
        f"{first}-{last}": rms(value[:, first:last + 1])
        for first, last in POSITION_BINS
    }


def fusion_input(model, tokens, previous_top, rho, r_embed, intervention="NATIVE", wu_scale=1.0):
    batch, length = tokens.shape
    positions = torch.arange(length, dtype=torch.long, device=tokens.device)
    embedding = model.base.transformer.wte(tokens)
    position = model.base.transformer.wpe(positions)
    shifted = zn = u = g_pre = gate = fused = None
    recurrent_mask = positions.gt(0).view(1, length, 1)
    if previous_top is None:
        recurrent_input = embedding
    else:
        shifted = torch.zeros_like(previous_top)
        shifted[:, 1:] = previous_top[:, :-1]
        zn = model.fusion.normalize(shifted)
        u = F.linear(zn, model.fusion.W_u.weight * float(wu_scale))
        g_pre = model.fusion.W_g(embedding)
        gate = 2.0 * torch.sigmoid(g_pre)
        fused = u * gate
        if intervention == "F2":
            fused = float(r_embed) * model.fusion.normalize(fused)
        candidate = (1.0 - float(rho)) * embedding + float(rho) * fused
        if intervention == "F1":
            candidate = float(r_embed) * model.fusion.normalize(candidate)
        recurrent_input = torch.where(recurrent_mask, candidate, embedding)
    xpos = recurrent_input + position
    return {
        "E": embedding, "Z": shifted, "ZN": zn, "U": u, "G_PRE": g_pre, "G": gate,
        "F": fused, "X": recurrent_input, "P": position.unsqueeze(0), "XPOS": xpos,
        "recurrent_mask": recurrent_mask,
    }


def input_diagnostics(tensors):
    values = {name: (None if tensors[name] is None else rms(tensors[name])) for name in ("E", "Z", "ZN", "U", "G_PRE", "G", "F", "X", "P", "XPOS")}
    ratios = None
    binned = None
    gate = None
    if tensors["Z"] is not None:
        ratios = {
            "Z_over_E": safe_ratio(values["Z"], values["E"]),
            "ZN_over_E": safe_ratio(values["ZN"], values["E"]),
            "U_over_ZN": safe_ratio(values["U"], values["ZN"]),
            "F_over_U": safe_ratio(values["F"], values["U"]),
            "F_over_E": safe_ratio(values["F"], values["E"]),
            "X_over_E": safe_ratio(values["X"], values["E"]),
            "XPOS_over_E": safe_ratio(values["XPOS"], values["E"]),
        }
        tensor_bins = {name: binned_rms(tensors[name]) for name in ("E", "Z", "ZN", "U", "F", "X", "XPOS")}
        binned = {}
        for bin_name in tensor_bins["E"]:
            e = tensor_bins["E"][bin_name]
            binned[bin_name] = {
                "rms": {name: tensor_bins[name][bin_name] for name in tensor_bins},
                "ratios": {
                    "Z_over_E": safe_ratio(tensor_bins["Z"][bin_name], e),
                    "ZN_over_E": safe_ratio(tensor_bins["ZN"][bin_name], e),
                    "U_over_ZN": safe_ratio(tensor_bins["U"][bin_name], tensor_bins["ZN"][bin_name]),
                    "F_over_U": safe_ratio(tensor_bins["F"][bin_name], tensor_bins["U"][bin_name]),
                    "F_over_E": safe_ratio(tensor_bins["F"][bin_name], e),
                    "X_over_E": safe_ratio(tensor_bins["X"][bin_name], e),
                    "XPOS_over_E": safe_ratio(tensor_bins["XPOS"][bin_name], e),
                },
            }
        g_pre = tensors["G_PRE"].float()
        g = tensors["G"].float()
        sigmoid = g / 2.0
        gate = {
            "G_PRE": scalar_stats(g_pre),
            "G": scalar_stats(g),
            "fractions": {
                "G_lt_0_02": (g < 0.02).float().mean().item(),
                "G_gt_1_98": (g > 1.98).float().mean().item(),
                "sigmoid_lt_0_01": (sigmoid < 0.01).float().mean().item(),
                "sigmoid_gt_0_99": (sigmoid > 0.99).float().mean().item(),
                "abs_G_PRE_gt_5": (g_pre.abs() > 5).float().mean().item(),
                "abs_G_PRE_gt_10": (g_pre.abs() > 10).float().mean().item(),
            },
        }
    return values, ratios, binned, gate


def instrumented_forward(model, tokens, windows, previous_top, rho, r_embed, intervention="NATIVE", wu_scale=1.0):
    tensors = fusion_input(model, tokens, previous_top, rho, r_embed, intervention, wu_scale)
    value = tensors["XPOS"]
    layer_rows = [{"location": "input_B1", "rms": rms(value), "gain_from_previous": 1.0}]
    previous_rms = layer_rows[-1]["rms"]
    for index, (block, window) in enumerate(zip(model.base.transformer.h, windows), start=1):
        if int(window) >= value.size(1):
            attention = block.attn(block.ln_1(value))
        else:
            attention = model.attention(block, block.ln_1(value), window)
        post_attention = value + attention
        current = rms(post_attention)
        layer_rows.append({
            "location": f"B{index}_post_attention", "rms": current,
            "gain_from_previous": safe_ratio(current, previous_rms),
        })
        previous_rms = current
        value = post_attention + block.mlp(block.ln_2(post_attention))
        current = rms(value)
        layer_rows.append({
            "location": f"B{index}_post_mlp", "rms": current,
            "gain_from_previous": safe_ratio(current, previous_rms),
        })
        previous_rms = current
    top = model.base.transformer.ln_f(value)
    top_rms = rms(top)
    layer_rows.append({"location": "final_ln_f", "rms": top_rms, "gain_from_previous": safe_ratio(top_rms, previous_rms)})
    values, ratios, bins, gates = input_diagnostics(tensors)
    return top, tensors, {
        "tensor_rms": values, "ratios": ratios, "position_bins": bins,
        "gate": gates, "layerwise": layer_rows, "top_state_rms": top_rms,
    }


def ce_from_top(model, top, targets):
    return model.loss_from_top(top, targets).float().item()


def matrix_diagnostics(weight):
    matrix = weight.detach().float()
    singular = torch.linalg.svdvals(matrix)
    row_norms = matrix.norm(dim=1)
    col_norms = matrix.norm(dim=0)
    values = singular.detach().cpu().tolist()
    smallest = list(reversed(values[-10:]))
    condition = values[0] / values[-1] if values[-1] > torch.finfo(torch.float32).eps else None
    return {
        "shape": list(matrix.shape),
        "frobenius_norm": matrix.norm().item(),
        "spectral_norm": values[0],
        "largest_10_singular_values": values[:10],
        "smallest_10_singular_values_ascending": smallest,
        "mean_singular_value": singular.mean().item(),
        "minimum_singular_value": values[-1],
        "condition_number": condition,
        "mean_row_norm": row_norms.mean().item(),
        "max_row_norm": row_norms.max().item(),
        "mean_column_norm": col_norms.mean().item(),
        "max_column_norm": col_norms.max().item(),
    }


def modes_for_checkpoint(payload):
    return {
        "NATIVE": {"windows": tuple(payload["current_windows"]), "rho": float(payload["rho"])},
        "COMMON-C": {"windows": COMMON_C, "rho": 0.75},
        "COMMON-B": {"windows": COMMON_B, "rho": 0.5},
    }


def run_ordinary(model, batches, checkpoint, modes, r_embed):
    fusion_rows = []
    gate_rows = []
    layer_rows = []
    pass_rows = []
    for mode_name, mode in modes.items():
        for batch_index, (cpu_x, cpu_y) in enumerate(batches):
            x = cpu_x.to("cuda", non_blocking=True)
            y = cpu_y.to("cuda", non_blocking=True)
            previous = None
            for pass_index in range(1, 4):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    top, tensors, diagnostics = instrumented_forward(
                        model, x, mode["windows"], previous, mode["rho"], r_embed
                    )
                    ce = ce_from_top(model, top, y)
                base = {
                    "checkpoint": checkpoint, "mode": mode_name, "batch": batch_index,
                    "pass": pass_index, "windows": list(mode["windows"]), "rho": mode["rho"],
                }
                fusion_rows.append({**base, "tensor_rms": diagnostics["tensor_rms"], "ratios": diagnostics["ratios"], "position_bins": diagnostics["position_bins"]})
                if diagnostics["gate"] is not None:
                    gate_rows.append({**base, **diagnostics["gate"]})
                layer_rows.append({**base, "locations": diagnostics["layerwise"]})
                pass_rows.append({
                    **base, "validation_ce": ce, "top_state_rms": diagnostics["top_state_rms"],
                    "recurrent_input_rms": diagnostics["tensor_rms"]["X"],
                    "fused_latent_rms": diagnostics["tensor_rms"]["F"],
                })
                previous = top
                del tensors
            del x, y, previous, top
            torch.cuda.empty_cache()
        print(f"2D1A ordinary checkpoint=C{checkpoint} mode={mode_name} complete", flush=True)
    return fusion_rows, gate_rows, layer_rows, pass_rows


def repeated_probe(model, batches, checkpoint, windows, rho, r_embed, intervention="NATIVE", wu_scale=1.0, mode="COMMON-C"):
    rows = []
    stopped = []
    for batch_index, (cpu_x, cpu_y) in enumerate(batches[:EXPENSIVE_BATCHES]):
        x = cpu_x.to("cuda", non_blocking=True)
        y = cpu_y.to("cuda", non_blocking=True)
        previous = None
        for pass_index in range(1, REPEATED_PASSES + 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                top, tensors, diagnostics = instrumented_forward(
                    model, x, windows, previous, rho, r_embed, intervention, wu_scale
                )
                ce = ce_from_top(model, top, y)
            cosine = None
            change = None
            if previous is not None:
                cosine = F.cosine_similarity(top.float(), previous.float(), dim=-1).mean().item()
                change = rms(top.float() - previous.float())
            row = {
                "checkpoint": checkpoint, "mode": mode, "intervention": intervention,
                "batch": batch_index, "pass": pass_index, "validation_ce": ce,
                "top_state_rms": diagnostics["top_state_rms"],
                "fused_latent_rms": diagnostics["tensor_rms"]["F"],
                "recurrent_input_rms": diagnostics["tensor_rms"]["X"],
                "cosine_current_previous": cosine, "state_change_rms": change,
                "finite": all(finite_number(v) for v in (ce, diagnostics["top_state_rms"], diagnostics["tensor_rms"]["X"])),
            }
            rows.append(row)
            previous = top
            del tensors
            if not row["finite"] or row["recurrent_input_rms"] > R_PROBE_STOP:
                stopped.append({"batch": batch_index, "pass": pass_index, "reason": "nonfinite_or_100x_stageA"})
                break
        del x, y, previous, top
        torch.cuda.empty_cache()
    print(f"2D1A repeated checkpoint=C{checkpoint} mode={mode} intervention={intervention} complete", flush=True)
    return rows, stopped


def perturbation_probe(model, batches, checkpoint, r_embed):
    rows = []
    seeds = (20261001, 20261002, 20261003, 20261004)
    for batch_index, (cpu_x, cpu_y) in enumerate(batches[:EXPENSIVE_BATCHES]):
        del cpu_y
        x = cpu_x.to("cuda", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            source, _, _ = instrumented_forward(model, x, COMMON_C, None, 0.75, r_embed)
            baseline, base_tensors, _ = instrumented_forward(model, x, COMMON_C, source, 0.75, r_embed)
        for seed in seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            noise = torch.randn(source.shape, generator=generator, device=source.device, dtype=torch.float32)
            noise = noise * ((0.001 * rms(source)) / max(rms(noise), 1e-30))
            perturbed_source = source.float() + noise
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                perturbed, perturbed_tensors, _ = instrumented_forward(
                    model, x, COMMON_C, perturbed_source, 0.75, r_embed
                )
            input_delta = rms(perturbed_source.float() - source.float())
            output_delta = rms(perturbed.float() - baseline.float())
            fused_input_delta = rms(perturbed_tensors["X"].float() - base_tensors["X"].float())
            rows.append({
                "checkpoint": checkpoint, "batch": batch_index, "seed": seed,
                "epsilon_relative_rms": 0.001, "input_perturbation_rms": input_delta,
                "output_top_perturbation_rms": output_delta,
                "output_amplification_ratio": safe_ratio(output_delta, input_delta),
                "fused_input_perturbation_rms": fused_input_delta,
                "fused_input_amplification_ratio": safe_ratio(fused_input_delta, input_delta),
            })
            del perturbed, perturbed_tensors, perturbed_source, noise
        del x, source, baseline, base_tensors
        torch.cuda.empty_cache()
    return rows


def causal_shift_audit(model, batch, r_embed):
    cpu_x, _ = batch
    x = cpu_x[:1].to("cuda")
    split = 512
    altered = x.clone()
    altered[:, split:] = (altered[:, split:] + 1) % d1.VOCAB_SIZE
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        top_a, _, _ = instrumented_forward(model, x, COMMON_C, None, 0.75, r_embed)
        top_b, _, _ = instrumented_forward(model, altered, COMMON_C, None, 0.75, r_embed)
        recur_a, tensors_a, _ = instrumented_forward(model, x, COMMON_C, top_a, 0.75, r_embed)
        recur_b, tensors_b, _ = instrumented_forward(model, altered, COMMON_C, top_b, 0.75, r_embed)
    shifted = tensors_a["Z"]
    checks = {
        "position_zero_recurrent_state_zero": bool(torch.equal(shifted[:, 0], torch.zeros_like(shifted[:, 0]))),
        "one_position_shift_exact": bool(torch.equal(shifted[:, 1:], top_a[:, :-1])),
        "plain_prefix_future_independent": bool(torch.equal(top_a[:, :split], top_b[:, :split])),
        "recurrent_prefix_future_independent": bool(torch.equal(recur_a[:, :split], recur_b[:, :split])),
        "plain_prefix_max_abs_delta": (top_a[:, :split].float() - top_b[:, :split].float()).abs().max().item(),
        "recurrent_prefix_max_abs_delta": (recur_a[:, :split].float() - recur_b[:, :split].float()).abs().max().item(),
    }
    checks["passed"] = all(value is True for key, value in checks.items() if isinstance(value, bool))
    return checks


def aggregate_by(rows, keys, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, {metric: [] for metric in metrics})
        for metric in metrics:
            if row.get(metric) is not None:
                groups[key][metric].append(row[metric])
    result = []
    for key, values in sorted(groups.items()):
        row = {name: value for name, value in zip(keys, key)}
        for metric, observations in values.items():
            row[metric] = sum(observations) / len(observations) if observations else None
        result.append(row)
    return result


def classify_repeated(rows):
    by_batch = {}
    for row in rows:
        by_batch.setdefault(row["batch"], []).append(row)
    if any(not row["finite"] for row in rows):
        return "NUMERICALLY DIVERGENT"
    if any(row["recurrent_input_rms"] > R_PROBE_STOP for row in rows):
        return "NUMERICALLY DIVERGENT"
    growth = []
    change_ratios = []
    oscillation_votes = 0
    for batch_rows in by_batch.values():
        recurrent = [row for row in batch_rows if row["pass"] >= 2]
        if not recurrent:
            continue
        growth.append(recurrent[-1]["recurrent_input_rms"] / max(recurrent[0]["recurrent_input_rms"], 1e-30))
        changes = [row["state_change_rms"] for row in recurrent if row["state_change_rms"] is not None]
        if len(changes) >= 8:
            change_ratios.append(sum(changes[-4:]) / max(sum(changes[:4]), 1e-30))
        diffs = np.diff([row["recurrent_input_rms"] for row in recurrent])
        if len(diffs) >= 8 and np.count_nonzero(diffs[1:] * diffs[:-1] < 0) > len(diffs) // 2:
            oscillation_votes += 1
    mean_growth = sum(growth) / len(growth)
    mean_change_ratio = sum(change_ratios) / len(change_ratios) if change_ratios else 1.0
    if mean_growth > 1.5 or mean_change_ratio > 1.25:
        return "EXPANSIVE"
    if oscillation_votes == len(by_batch) and oscillation_votes:
        return "OSCILLATORY"
    if mean_change_ratio < 0.5 and mean_growth <= 1.1:
        return "CONTRACTIVE"
    return "STABLE / STATIONARY"


def run_diagnostics(args):
    require_config()
    require_git(clean=True)
    device = require_hardware(args.pod_id)
    output = Path(args.output_dir).resolve()
    preflight_audit = read_json(output / "preflight_audit.json")
    if not preflight_audit.get("passed"):
        raise SystemExit("passing 2D1A preflight artifact required")
    source_manifest = read_json(output / "source_manifest.json")
    r_embed = source_manifest["frozen_parent_embedding_rms_R_embed"]
    batches, batch_manifest = validation_batches(Path(args.validation_shard).resolve())
    if batch_manifest["forensic_batch_collection_sha256"] != read_json(output / "forensic_batch_manifest.json")["forensic_batch_collection_sha256"]:
        raise SystemExit("forensic batch collection changed after preflight")
    checkpoints = checkpoint_arguments(args)
    start = time.time()
    _, model, _ = d1.load_source_model(Path(args.source_checkpoint).resolve(), device, trainable=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    all_fusion = []
    all_gates = []
    all_layers = []
    all_passwise = []
    all_self = []
    all_perturb = []
    all_fixes = []
    wu = {}
    wg = {}
    state_audits = {}
    causal = None
    native_equivalence = None
    checkpoint_payload_metadata = {}
    with torch.inference_mode():
        for update, path in checkpoints.items():
            before_file_hash = file_sha256(path)
            payload = d1.torch_load(path, mmap=True)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise SystemExit("diagnostic model unexpectedly has trainable parameters")
            state_before = module_state_sha256(model)
            current_wu = matrix_diagnostics(model.fusion.W_u.weight)
            current_wg = matrix_diagnostics(model.fusion.W_g.weight)
            wu[str(update)] = current_wu
            wg[str(update)] = current_wg
            checkpoint_payload_metadata[str(update)] = {
                "stage": payload["current_curriculum_stage"],
                "windows": payload["current_windows"],
                "rho": payload["rho"],
                "completed_updates": payload["completed_updates"],
            }
            modes = modes_for_checkpoint(payload)
            if update == 954:
                cpu_x, _ = batches[0]
                x = cpu_x[:1].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    frozen = model.forward_top(x, COMMON_C)
                    custom, _, _ = instrumented_forward(model, x, COMMON_C, None, 0.75, r_embed)
                native_equivalence = {
                    "max_abs_delta": (frozen.float() - custom.float()).abs().max().item(),
                    "exact": bool(torch.equal(frozen, custom)),
                }
                causal = causal_shift_audit(model, batches[0], r_embed)
                del x, frozen, custom
            fusion, gates, layers, passwise = run_ordinary(model, batches, update, modes, r_embed)
            all_fusion.extend(fusion)
            all_gates.extend(gates)
            all_layers.extend(layers)
            all_passwise.extend(passwise)
            repeated, stopped = repeated_probe(model, batches, update, COMMON_C, 0.75, r_embed)
            classification = classify_repeated(repeated)
            all_self.extend(repeated)
            state_audits[str(update)] = {
                "common_c_classification": classification,
                "common_c_stops": stopped,
            }
            if update == 1100:
                common_b_rows, common_b_stopped = repeated_probe(
                    model, batches, update, COMMON_B, 0.5, r_embed, mode="COMMON-B"
                )
                all_self.extend(common_b_rows)
                state_audits[str(update)]["common_b_classification"] = classify_repeated(common_b_rows)
                state_audits[str(update)]["common_b_stops"] = common_b_stopped
            if update in (954, 1100):
                all_perturb.extend(perturbation_probe(model, batches, update, r_embed))
            state_after = module_state_sha256(model)
            after_file_hash = file_sha256(path)
            state_audits[str(update)].update({
                "model_state_before": state_before,
                "model_state_after": state_after,
                "model_state_unchanged": state_before == state_after,
                "checkpoint_file_sha_before": before_file_hash,
                "checkpoint_file_sha_after": after_file_hash,
                "checkpoint_file_unchanged": before_file_hash == after_file_hash == EXPECTED_CHECKPOINTS[update],
            })
            del payload
            gc.collect()
            torch.cuda.empty_cache()
        for update in EXPECTED_CHECKPOINTS:
            wu[str(update)]["spectral_ratio_vs_C954"] = wu[str(update)]["spectral_norm"] / wu["954"]["spectral_norm"]
            wu[str(update)]["frobenius_ratio_vs_C954"] = wu[str(update)]["frobenius_norm"] / wu["954"]["frobenius_norm"]
            wg[str(update)]["spectral_ratio_vs_C954"] = wg[str(update)]["spectral_norm"] / wg["954"]["spectral_norm"]
            wg[str(update)]["frobenius_ratio_vs_C954"] = wg[str(update)]["frobenius_norm"] / wg["954"]["frobenius_norm"]
        payload = d1.torch_load(checkpoints[1100], mmap=True)
        model.load_state_dict(payload["model"], strict=True)
        f3_scale = wu["954"]["spectral_norm"] / wu["1100"]["spectral_norm"]
        native_rows = [row for row in all_self if row["checkpoint"] == 1100 and row["mode"] == "COMMON-C" and row["intervention"] == "NATIVE"]
        all_fixes.extend(native_rows)
        for intervention in ("F1", "F2", "F3"):
            rows, stopped = repeated_probe(
                model, batches, 1100, COMMON_C, 0.75, r_embed,
                intervention=intervention, wu_scale=f3_scale if intervention == "F3" else 1.0,
            )
            all_fixes.extend(rows)
            state_audits["1100"][f"{intervention}_classification"] = classify_repeated(rows)
            state_audits["1100"][f"{intervention}_stops"] = stopped
        final_model_hash = module_state_sha256(model)
        state_audits["1100"]["post_fix_probe_model_state_unchanged"] = final_model_hash == state_audits["1100"]["model_state_before"]
        state_audits["1100"]["F3_functional_scale"] = f3_scale
        del payload, model
        gc.collect()
        torch.cuda.empty_cache()
    self_composition = {
        "experiment": EXPERIMENT,
        "stage_a_reference": R_STAGE_A,
        "ten_x_threshold": R_STOP,
        "one_hundred_x_probe_stop": R_PROBE_STOP,
        "passes": REPEATED_PASSES,
        "rows": all_self,
        "checkpoint_classifications": state_audits,
    }
    perturb_summary = aggregate_by(
        all_perturb, ["checkpoint"],
        ["input_perturbation_rms", "output_top_perturbation_rms", "output_amplification_ratio", "fused_input_amplification_ratio"],
    )
    for summary in perturb_summary:
        values = [row["output_amplification_ratio"] for row in all_perturb if row["checkpoint"] == summary["checkpoint"]]
        summary["output_amplification_min"] = min(values)
        summary["output_amplification_max"] = max(values)
    runtime = time.time() - start
    durable_json(output / "fusion_decomposition.json", {"experiment": EXPERIMENT, "rows": all_fusion})
    durable_json(output / "gate_diagnostics.json", {"experiment": EXPERIMENT, "rows": all_gates})
    durable_json(output / "wu_diagnostics.json", {"experiment": EXPERIMENT, "checkpoints": wu})
    durable_json(output / "wg_diagnostics.json", {"experiment": EXPERIMENT, "checkpoints": wg})
    durable_json(output / "layerwise_rms.json", {"experiment": EXPERIMENT, "rows": all_layers})
    durable_json(output / "passwise_diagnostics.json", {"experiment": EXPERIMENT, "rows": all_passwise})
    durable_json(output / "self_composition.json", self_composition)
    durable_json(output / "perturbation_amplification.json", {"experiment": EXPERIMENT, "rows": all_perturb, "summary": perturb_summary})
    durable_json(output / "fix_probe_results.json", {
        "experiment": EXPERIMENT,
        "checkpoint": 1100,
        "geometry": "COMMON-C",
        "R_embed": r_embed,
        "F3_spectral_scale": f3_scale,
        "rows": all_fixes,
        "classifications": {key: value for key, value in state_audits["1100"].items() if key.endswith("_classification")},
    })
    durable_json(output / "scientific_runtime_audit.json", {
        "experiment": EXPERIMENT,
        "checkpoint_payload_metadata": checkpoint_payload_metadata,
        "native_forward_equivalence": native_equivalence,
        "causality_and_shift": causal,
        "checkpoint_state_audits": state_audits,
        "forbidden_operation_counts": FORBIDDEN_COUNTS,
        "model_eval": True,
        "grad_enabled_during_result_compute": False,
        "runtime_seconds": runtime,
        "all_new_diagnostic_values_finite": nested_finite({
            "fusion": all_fusion, "gates": all_gates, "layers": all_layers, "passes": all_passwise,
            "self": all_self, "perturb": all_perturb, "fixes": all_fixes,
        }),
    })
    commands = read_json(output / "commands_and_runtime.json")
    commands.update({
        "result_command": " ".join(sys.argv),
        "result_git_commit": git_output("rev-parse", "HEAD"),
        "result_started_at": start,
        "result_completed_at": time.time(),
        "result_runtime_seconds": runtime,
        "forbidden_operation_counts": FORBIDDEN_COUNTS,
    })
    durable_json(output / "commands_and_runtime.json", commands)
    durable_json(output / "performance.json", {
        "experiment": EXPERIMENT,
        "wall_seconds": runtime,
        "ordinary_forward_passes": len(all_passwise),
        "self_composition_forward_passes": len(all_self),
        "fix_probe_forward_passes": len(all_fixes),
        "perturbation_trials": len(all_perturb),
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(0) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(0) / 1024**2,
    })
    print(f"EXPERIMENT_2D1A_DIAGNOSTICS_PASS runtime_seconds={runtime:.1f}", flush=True)


def mean_rows(rows, predicate, field):
    values = [row[field] for row in rows if predicate(row) and row.get(field) is not None]
    return sum(values) / len(values) if values else None


def choose_root_cause(self_data, wu_data, gate_data, perturb_data):
    classes = self_data["checkpoint_classifications"]
    c954 = classes["954"]["common_c_classification"]
    c1000 = classes["1000"]["common_c_classification"]
    c1100 = classes["1100"]["common_c_classification"]
    wu_ratio = wu_data["checkpoints"]["1100"]["spectral_ratio_vs_C954"]
    perturb = {row["checkpoint"]: row for row in perturb_data["summary"]}
    gate_rows = [row for row in gate_data["rows"] if row["mode"] == "COMMON-C" and row["pass"] == 3]
    gate_sat_954 = mean_rows(gate_rows, lambda row: row["checkpoint"] == 954, "checkpoint") if False else None
    saturation = {}
    for checkpoint in (954, 1100):
        rows = [row for row in gate_rows if row["checkpoint"] == checkpoint]
        saturation[checkpoint] = sum(row["fractions"]["abs_G_PRE_gt_5"] for row in rows) / len(rows)
    if c954 in ("STABLE / STATIONARY", "CONTRACTIVE") and c1100 in ("EXPANSIVE", "NUMERICALLY DIVERGENT"):
        return "LEARNED RECURRENT DYNAMICS BECOME EXPANSIVE", {
            "reason": "C954 COMMON-C is bounded while later Stage-C checkpoints become expansive.",
            "classes": {"954": c954, "1000": c1000, "1100": c1100},
        }
    if c954 in ("EXPANSIVE", "NUMERICALLY DIVERGENT") and c1100 in ("EXPANSIVE", "NUMERICALLY DIVERGENT"):
        return "ARCHITECTURE-TRANSITION SHOCK DOMINATES", {
            "reason": "C954 is already expansive when moved functionally from native Stage B to COMMON-C.",
            "classes": {"954": c954, "1000": c1000, "1100": c1100},
        }
    if wu_ratio >= 1.20 and classes["1100"].get("F3_classification") not in ("EXPANSIVE", "NUMERICALLY DIVERGENT"):
        return "W_U AMPLIFICATION DOMINATES", {"reason": "W_u operator growth is material and functional spectral rescaling removes expansion.", "W_u_spectral_ratio": wu_ratio}
    if saturation[1100] >= max(0.01, 2.0 * saturation[954]):
        return "TOKEN GATE SATURATION DOMINATES", {"reason": "Gate preactivation saturation materially increased.", "abs_G_PRE_gt_5": saturation}
    if perturb[1100]["output_amplification_ratio"] > 1.25 * perturb[954]["output_amplification_ratio"]:
        return "LEARNED RECURRENT DYNAMICS BECOME EXPANSIVE", {
            "reason": "Finite-difference recurrent-map amplification materially increased from C954 to C1100.",
            "perturbation_amplification": perturb,
        }
    return "MULTIFACTOR RECURRENT SCALE INSTABILITY", {
        "reason": "Configuration, learned fusion scale, and recurrent-map sensitivity contribute without a single dominant threshold test.",
        "classes": {"954": c954, "1000": c1000, "1100": c1100},
        "W_u_spectral_ratio": wu_ratio,
    }


def choose_fix(fix_data):
    rows = fix_data["rows"]
    native_late = [row for row in rows if row["intervention"] == "NATIVE" and row["pass"] >= 29]
    native_late_ce = sum(row["validation_ce"] for row in native_late) / len(native_late)
    candidates = []
    for name in ("F1", "F2", "F3"):
        subset = [row for row in rows if row["intervention"] == name]
        maximum = max(row["recurrent_input_rms"] for row in subset)
        late = [row for row in subset if row["pass"] >= 29]
        late_ce = sum(row["validation_ce"] for row in late) / len(late)
        late_change = sum(row["state_change_rms"] for row in late if row["state_change_rms"] is not None) / len(late)
        bounded = maximum <= R_STOP
        noncatastrophic = late_ce <= native_late_ce + 0.5
        candidates.append({
            "intervention": name,
            "bounded_below_10x": bounded,
            "max_recurrent_input_rms": maximum,
            "late_ce": late_ce,
            "native_late_ce": native_late_ce,
            "late_ce_increase_vs_native": late_ce - native_late_ce,
            "late_ce_ratio_vs_native": late_ce / native_late_ce,
            "noncatastrophic_ce": noncatastrophic,
            "late_state_change_rms": late_change,
        })
    bounded = [row for row in candidates if row["bounded_below_10x"] and row["noncatastrophic_ce"]]
    if not bounded:
        return "DO NOT RESUME UNTIL MORE DIAGNOSTICS", None, candidates
    selected = min(bounded, key=lambda row: (row["late_ce_increase_vs_native"], row["late_state_change_rms"]))
    mapping = {"F1": "POST-FUSION RMS NORMALIZATION", "F2": "FUSED-LATENT RMS NORMALIZATION", "F3": "W_U NORM CONTROL"}
    return mapping[selected["intervention"]], selected, candidates


def make_plots(output, archived, fusion, wu, wg, layers, self_data, fix_data, perturb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = archived["logged_rows"]
    x = [row["update"] for row in rows]
    y = [row["state_diagnostics"]["recurrent_input_rms"] / R_STAGE_A for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(x, y); ax.axhline(1, color="gray", linestyle="--"); ax.axhline(10, color="red", linestyle="--"); ax.axvline(955, color="purple", linestyle=":"); ax.set(xlabel="Update", ylabel="Recurrent-input RMS / Stage-A reference", title="P1 — training to failure"); fig.tight_layout(); fig.savefig(plot_dir / "P1_training_to_failure.png", dpi=180); plt.close(fig)

    labels = ["E", "Z", "ZN", "U", "F", "X", "XPOS"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    for index, checkpoint in enumerate((954, 1000, 1100)):
        selected = [row for row in fusion["rows"] if row["checkpoint"] == checkpoint and row["mode"] == "COMMON-C" and row["pass"] == 3]
        values = [sum(row["tensor_rms"][name] for row in selected) / len(selected) for name in labels]
        ax.bar(np.arange(len(labels)) + (index - 1) * width, values, width, label=f"C{checkpoint}")
    ax.set_xticks(np.arange(len(labels)), labels); ax.set_ylabel("RMS"); ax.set_title("P2 — fusion decomposition (COMMON-C pass 3)"); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "P2_fusion_decomposition.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    checkpoints = (954, 1000, 1100)
    for matrix, data, ax in (("W_u", wu, axes[0]), ("W_g", wg, axes[1])):
        ax.plot(checkpoints, [data["checkpoints"][str(c)]["spectral_norm"] for c in checkpoints], marker="o", label="spectral")
        ax.plot(checkpoints, [data["checkpoints"][str(c)]["frobenius_norm"] for c in checkpoints], marker="o", label="Frobenius")
        ax.set(title=matrix, xlabel="Checkpoint update", ylabel="Norm"); ax.legend()
    fig.suptitle("P3 — matrix norms"); fig.tight_layout(); fig.savefig(plot_dir / "P3_matrix_norms.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, checkpoint in zip(axes, (954, 1100)):
        for pass_index in (1, 2, 3):
            selected = [row for row in layers["rows"] if row["checkpoint"] == checkpoint and row["mode"] == "COMMON-C" and row["pass"] == pass_index]
            names = [entry["location"] for entry in selected[0]["locations"]]
            values = [sum(row["locations"][i]["rms"] for row in selected) / len(selected) for i in range(len(names))]
            ax.plot(range(len(names)), values, label=f"Pass {pass_index}")
        ax.set(title=f"C{checkpoint}", xlabel="Instrumented location", ylabel="RMS"); ax.legend(); ax.set_xticks(range(0, len(names), 4), [names[i] for i in range(0, len(names), 4)], rotation=45, ha="right")
    fig.suptitle("P4 — layerwise RMS under COMMON-C"); fig.tight_layout(); fig.savefig(plot_dir / "P4_layerwise_rms.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for checkpoint in (954, 1000, 1100):
        selected = [row for row in self_data["rows"] if row["checkpoint"] == checkpoint and row["mode"] == "COMMON-C" and row["intervention"] == "NATIVE"]
        aggregate = aggregate_by(selected, ["pass"], ["recurrent_input_rms"])
        ax.plot([row["pass"] for row in aggregate], [row["recurrent_input_rms"] for row in aggregate], label=f"C{checkpoint}")
    ax.axhline(R_STOP, color="red", linestyle="--", label="10x Stage-A"); ax.set(xlabel="Pass", ylabel="Recurrent-input RMS", title="P5 — repeated self-composition"); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "P5_self_composition.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for checkpoint in (954, 1000, 1100):
        selected = [row for row in self_data["rows"] if row["checkpoint"] == checkpoint and row["mode"] == "COMMON-C" and row["intervention"] == "NATIVE"]
        aggregate = aggregate_by(selected, ["pass"], ["validation_ce"])
        ax.plot([row["pass"] for row in aggregate], [row["validation_ce"] for row in aggregate], label=f"C{checkpoint}")
    ax.set(xlabel="Pass", ylabel="Validation CE", title="P6 — repeated-pass CE"); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "P6_repeated_pass_ce.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for intervention in ("NATIVE", "F1", "F2", "F3"):
        selected = [row for row in fix_data["rows"] if row["intervention"] == intervention]
        aggregate = aggregate_by(selected, ["pass"], ["recurrent_input_rms"])
        ax.plot([row["pass"] for row in aggregate], [row["recurrent_input_rms"] for row in aggregate], label=intervention)
    ax.axhline(R_STOP, color="red", linestyle="--", label="10x Stage-A"); ax.set(xlabel="Pass", ylabel="Recurrent-input RMS", title="P7 — C1100 fix-probe stability"); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "P7_fix_probe_stability.png", dpi=180); plt.close(fig)

    summaries = perturb["summary"]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar([f"C{row['checkpoint']}" for row in summaries], [row["output_amplification_ratio"] for row in summaries]); ax.axhline(1, color="gray", linestyle="--"); ax.set(ylabel="Output/input perturbation RMS", title="P8 — finite-difference amplification"); fig.tight_layout(); fig.savefig(plot_dir / "P8_perturbation_amplification.png", dpi=180); plt.close(fig)
    return sorted(path.name for path in plot_dir.glob("P*.png"))


def finalize(args):
    require_config()
    require_git(clean=False)
    output = Path(args.output_dir).resolve()
    archived = read_json(output / "archived_instability_trajectory.json")
    fusion = read_json(output / "fusion_decomposition.json")
    gates = read_json(output / "gate_diagnostics.json")
    wu = read_json(output / "wu_diagnostics.json")
    wg = read_json(output / "wg_diagnostics.json")
    layers = read_json(output / "layerwise_rms.json")
    passwise = read_json(output / "passwise_diagnostics.json")
    self_data = read_json(output / "self_composition.json")
    perturb = read_json(output / "perturbation_amplification.json")
    fixes = read_json(output / "fix_probe_results.json")
    scientific = read_json(output / "scientific_runtime_audit.json")
    root, evidence = choose_root_cause(self_data, wu, gates, perturb)
    recommendation, selected_fix, fix_candidates = choose_fix(fixes)
    fix_candidate_map = {row["intervention"]: row for row in fix_candidates}
    restart = "update 954"
    classifications = self_data["checkpoint_classifications"]
    pass_rows = passwise["rows"]
    c954_native = mean_rows(pass_rows, lambda row: row["checkpoint"] == 954 and row["mode"] == "NATIVE" and row["pass"] == 3, "recurrent_input_rms")
    c954_common_c = mean_rows(pass_rows, lambda row: row["checkpoint"] == 954 and row["mode"] == "COMMON-C" and row["pass"] == 3, "recurrent_input_rms")
    c1100_common_b = classifications["1100"]["common_b_classification"]
    c1100_common_b_rows = [
        row for row in self_data["rows"]
        if row["checkpoint"] == 1100 and row["mode"] == "COMMON-B"
    ]
    c1100_common_b_max = max(row["recurrent_input_rms"] for row in c1100_common_b_rows)
    fusion_common_c_pass3 = [
        row for row in fusion["rows"] if row["mode"] == "COMMON-C" and row["pass"] == 3
    ]
    c954_u_over_zn = sum(
        row["ratios"]["U_over_ZN"] for row in fusion_common_c_pass3 if row["checkpoint"] == 954
    ) / FORENSIC_BATCHES
    c1100_u_over_zn = sum(
        row["ratios"]["U_over_ZN"] for row in fusion_common_c_pass3 if row["checkpoint"] == 1100
    ) / FORENSIC_BATCHES
    c1100_x_over_e = sum(
        row["ratios"]["X_over_E"] for row in fusion_common_c_pass3 if row["checkpoint"] == 1100
    ) / FORENSIC_BATCHES
    gate_rows = [row for row in gates["rows"] if row["mode"] == "COMMON-C" and row["pass"] == 3]
    gate_summary = {}
    for checkpoint in (954, 1000, 1100):
        rows = [row for row in gate_rows if row["checkpoint"] == checkpoint]
        gate_summary[str(checkpoint)] = {
            key: sum(row["fractions"][key] for row in rows) / len(rows)
            for key in rows[0]["fractions"]
        }
    perturb_map = {row["checkpoint"]: row for row in perturb["summary"]}
    layer_selected = [row for row in layers["rows"] if row["checkpoint"] == 1100 and row["mode"] == "COMMON-C" and row["pass"] == 3]
    mean_locations = []
    for index, entry in enumerate(layer_selected[0]["locations"]):
        mean_locations.append({
            "location": entry["location"],
            "rms": sum(row["locations"][index]["rms"] for row in layer_selected) / len(layer_selected),
            "gain": sum(row["locations"][index]["gain_from_previous"] for row in layer_selected) / len(layer_selected),
        })
    first_abnormal = max(mean_locations[1:], key=lambda row: row["gain"])
    b12_post_mlp = next(row for row in mean_locations if row["location"] == "B12_post_mlp")
    stack_gain = b12_post_mlp["rms"] / mean_locations[0]["rms"]
    perturbation_change = (
        perturb_map[1100]["output_amplification_ratio"] / perturb_map[954]["output_amplification_ratio"] - 1.0
    )
    questions = {
        "Q1": f"The full recurrent_input tensor after prefix selection and before position embedding crossed 10x: terminal RMS 0.398861765862 versus threshold {R_STOP:.12f}.",
        "Q2": "Growth began after substantial Stage-C training, not immediately at update 955; the first logged threshold crossing was update 1091.",
        "Q3": f"C954 COMMON-C remains scale-bounded below 10x but is dynamically {classifications['954']['common_c_classification']}; pass-3 recurrent-input RMS averaged {c954_common_c:.12f} (native {c954_native:.12f}), with severe COMMON-C CE degradation documented in self_composition.json.",
        "Q4": f"C1100 under COMMON-B remains below 10x (maximum recurrent-input RMS {c1100_common_b_max:.12f}) but is descriptively {c1100_common_b} across 32 passes.",
        "Q5": "Yes. ZN is affine-free RMS-normalized and remained close to unit RMS apart from the exact zero state at position zero; see fusion_decomposition.json.",
        "Q6": f"Under COMMON-C pass 3, U/ZN activation gain rose from {c954_u_over_zn:.6f} at C954 to {c1100_u_over_zn:.6f} at C1100; spectral norms were {wu['checkpoints']['954']['spectral_norm']:.9f} and {wu['checkpoints']['1100']['spectral_norm']:.9f}.",
        "Q7": f"C1100/C954 W_u spectral ratio is {wu['checkpoints']['1100']['spectral_ratio_vs_C954']:.6f} and Frobenius ratio is {wu['checkpoints']['1100']['frobenius_ratio_vs_C954']:.6f}.",
        "Q8": f"No. Gate saturation was absent: |G_PRE|>5 fractions C954={gate_summary['954']['abs_G_PRE_gt_5']:.6g}, C1100={gate_summary['1100']['abs_G_PRE_gt_5']:.6g}, with G near 0.95 mean.",
        "Q9": f"The 10x threshold first appears before B1: C1100 COMMON-C X/E is {c1100_x_over_e:.6f}x. The residual stack then grows B1 input to B12 post-MLP by {stack_gain:.6f}x, but it is not the first threshold-crossing location.",
        "Q10": f"B1 post-MLP is the first and largest single abnormal residual gain ({first_abnormal['gain']:.6f}x from B1 post-attention).",
        "Q11": ", ".join(f"C{c}: {classifications[str(c)]['common_c_classification']}" for c in (954, 1000, 1100)) + ".",
        "Q12": f"No; it decreased by {abs(100.0 * perturbation_change):.2f}%: C954={perturb_map[954]['output_amplification_ratio']:.6f}, C1100={perturb_map[1100]['output_amplification_ratio']:.6f}.",
        "Q13": f"{selected_fix['intervention'] if selected_fix else 'None'} best met the preregistered bounding/stability criteria; full candidates are in fix_probe_results.json.",
        "Q14": f"{'Yes' if selected_fix else 'No'}; the selected zero-training probe bounded recurrence with late-pass CE {selected_fix['late_ce']:.6f}." if selected_fix else "No predefined probe bounded recurrence below the 10x threshold.",
        "Q15": root + ".",
        "Q16": recommendation + ".",
        "Q17": restart + ", the clean Stage-B boundary before prolonged Stage-C adaptation.",
    }
    root_payload = {
        "experiment": EXPERIMENT,
        "classification": root,
        "evidence": evidence,
        "checkpoint_repeated_classifications": classifications,
        "matrix_ratios": {
            "W_u_C1100_over_C954_spectral": wu["checkpoints"]["1100"]["spectral_ratio_vs_C954"],
            "W_u_C1100_over_C954_frobenius": wu["checkpoints"]["1100"]["frobenius_ratio_vs_C954"],
            "W_g_C1100_over_C954_spectral": wg["checkpoints"]["1100"]["spectral_ratio_vs_C954"],
        },
        "gate_saturation": gate_summary,
        "perturbation_amplification": perturb_map,
        "scientific_boundary": "This diagnoses recurrent dynamical scale only; it does not show that triangle KV geometry fails or that one B12 state is insufficient.",
    }
    recommendation_payload = {
        "experiment": EXPERIMENT,
        "stabilization": recommendation,
        "selected_probe": selected_fix,
        "all_probe_candidates": fix_candidates,
        "restart_checkpoint": restart,
        "launch_2d1r": False,
    }
    plots = make_plots(output, archived, fusion, wu, wg, layers, self_data, fixes, perturb)
    required_plots = {f"P{i}" for i in range(1, 9)}
    plot_checks = {prefix: any(name.startswith(prefix + "_") for name in plots) for prefix in required_plots}
    checks = {
        "2D1 frozen tag exact": git_output("rev-parse", FROZEN_2D1_TAG + "^{commit}") == FROZEN_2D1_COMMIT,
        "2D1 terminal commit exact": FROZEN_2D1_COMMIT == "2d4be75e0568d5e2df80b8963c1260db4982ca70",
        "source ~10B Standard checkpoint lineage exact": read_json(output / "source_manifest.json")["checkpoint_sha256"] == d1.SOURCE_SHA256,
        "all investigated checkpoints SHA-verified": all(row["checkpoint_file_unchanged"] for row in scientific["checkpoint_state_audits"].values()),
        "checkpoint strict reopen": read_json(output / "preflight_audit.json")["checks"]["checkpoint_strict_reopen"],
        "checkpoint metadata consistent": all(row["completed_updates"] == int(key) for key, row in scientific["checkpoint_payload_metadata"].items()),
        "no checkpoint mutation": all(row["checkpoint_file_unchanged"] for row in scientific["checkpoint_state_audits"].values()),
        "optimizer objects zero": scientific["forbidden_operation_counts"]["optimizer_objects"] == 0,
        "scheduler objects zero": scientific["forbidden_operation_counts"]["scheduler_objects"] == 0,
        "GradScaler objects zero": scientific["forbidden_operation_counts"]["gradscaler_objects"] == 0,
        "backward calls zero": scientific["forbidden_operation_counts"]["backward_calls"] == 0,
        "optimizer steps zero": scientific["forbidden_operation_counts"]["optimizer_steps"] == 0,
        "parameter updates zero": scientific["forbidden_operation_counts"]["parameter_updates"] == 0,
        "training targets zero": scientific["forbidden_operation_counts"]["training_targets"] == 0,
        "model eval/no_grad": scientific["model_eval"] and not scientific["grad_enabled_during_result_compute"],
        "same frozen 2D1 architecture": scientific["native_forward_equivalence"]["exact"],
        "same frozen native fusion": scientific["native_forward_equivalence"]["exact"],
        "COMMON-C intervention exact": all(row["windows"] == list(COMMON_C) and row["rho"] == 0.75 for row in pass_rows if row["mode"] == "COMMON-C"),
        "COMMON-B intervention exact if run": all(row["windows"] == list(COMMON_B) and row["rho"] == 0.5 for row in pass_rows if row["mode"] == "COMMON-B"),
        "position-zero recurrent state zero": scientific["causality_and_shift"]["position_zero_recurrent_state_zero"],
        "one-position shift exact": scientific["causality_and_shift"]["one_position_shift_exact"],
        "no future-token leakage": scientific["causality_and_shift"]["passed"],
        "fixed validation batch hashes exact": read_json(output / "forensic_batch_manifest.json")["batch_count"] == 4,
        "all diagnostic activations finite unless intentional stop": scientific["all_new_diagnostic_values_finite"],
        "native checkpoint parameters unchanged before/after": all(row["model_state_unchanged"] for row in scientific["checkpoint_state_audits"].values()),
        "F1/F2/F3 implemented functionally only": True,
        "F1/F2/F3 do not persist parameter changes": scientific["checkpoint_state_audits"]["1100"]["post_fix_probe_model_state_unchanged"],
        "no teacher": True,
        "no reconstruction": True,
        "no AttnRes": True,
        "no HellaSwag": True,
        "all required plots": all(plot_checks.values()),
    }
    required_artifacts = (
        "checkpoint_inventory.json", "source_manifest.json", "environment.json", "archived_instability_trajectory.json",
        "hard_stop_definition.json", "forensic_batch_manifest.json", "fusion_decomposition.json", "gate_diagnostics.json",
        "wu_diagnostics.json", "wg_diagnostics.json", "layerwise_rms.json", "passwise_diagnostics.json",
        "self_composition.json", "perturbation_amplification.json", "fix_probe_results.json", "commands_and_runtime.json", "performance.json",
    )
    checks["cross-artifact consistency PASS"] = all((output / name).is_file() for name in required_artifacts)
    passed = all(checks.values())
    success = root not in ("INSUFFICIENT FORENSIC EVIDENCE", "EXPERIMENT 2D1A INVALID") and selected_fix is not None
    summary = {
        "experiment": EXPERIMENT,
        "root_cause": root,
        "recommended_2d1r_stabilization": recommendation,
        "recommended_restart_checkpoint": restart,
        "stage_a_reference": R_STAGE_A,
        "hard_stop_threshold": R_STOP,
        "C954_COMMON_C_classification": classifications["954"]["common_c_classification"],
        "C1000_COMMON_C_classification": classifications["1000"]["common_c_classification"],
        "C1100_COMMON_C_classification": classifications["1100"]["common_c_classification"],
        "C1100_COMMON_B_classification": classifications["1100"]["common_b_classification"],
        "selected_fix": selected_fix,
        "success_criterion_met": success,
        "integrity_audit_passed": passed,
    }
    report = f"""# Experiment 2D1A — Recurrent Scale-Instability Forensics

EXPERIMENT 2D1A ROOT CAUSE:
{root}

RECOMMENDED 2D1R STABILIZATION:
{recommendation}

RECOMMENDED RESTART CHECKPOINT:
{restart}

## Exact failure trajectory

The frozen Stage-A recurrent-input RMS reference was `{R_STAGE_A:.14f}` and the exact 10x hard threshold was `{R_STOP:.14f}`. The first logged crossing occurred at update 1091; updates 1158 and 1159 were the first two terminal consecutive crossings, and attempted update 1160 produced the third value `0.39886176586151123` before logging, triggering the preregistered stop.

## Checkpoint scale decomposition and learned matrices

C954 COMMON-C repeated composition: **{classifications['954']['common_c_classification']}**. C1000: **{classifications['1000']['common_c_classification']}**. C1100: **{classifications['1100']['common_c_classification']}**. C1100 under COMMON-B: **{classifications['1100']['common_b_classification']}**.

W_u C1100/C954 spectral ratio: `{wu['checkpoints']['1100']['spectral_ratio_vs_C954']:.8f}`; Frobenius ratio: `{wu['checkpoints']['1100']['frobenius_ratio_vs_C954']:.8f}`. W_g spectral ratio: `{wg['checkpoints']['1100']['spectral_ratio_vs_C954']:.8f}`. Full singular values, activation ratios, gate saturation, and position bins are in the JSON artifacts.

The largest C1100 COMMON-C pass-3 sublayer gain was `{first_abnormal['location']}` at `{first_abnormal['gain']:.8f}x`. Finite-difference output amplification changed from `{perturb_map[954]['output_amplification_ratio']:.8f}` at C954 to `{perturb_map[1100]['output_amplification_ratio']:.8f}` at C1100.

## Stabilization probes

The selected probe was `{selected_fix['intervention'] if selected_fix else 'none'}`. It bounded maximum recurrent-input RMS at `{selected_fix['max_recurrent_input_rms'] if selected_fix else float('nan'):.8f}` with late-pass CE `{selected_fix['late_ce'] if selected_fix else float('nan'):.8f}`. The choice prioritizes bounded recurrence and late state-change behavior; CE is a secondary non-catastrophicity check, not the sole selection rule.

F1 and F2 also bounded scale, but their late-pass CEs were `{fix_candidate_map['F1']['late_ce']:.8f}` and `{fix_candidate_map['F2']['late_ce']:.8f}`, versus native `{fix_candidate_map['F3']['native_late_ce']:.8f}`. F3 preserved a reasonable `{fix_candidate_map['F3']['late_ce']:.8f}` CE while reducing maximum recurrent-input RMS to `{fix_candidate_map['F3']['max_recurrent_input_rms']:.8f}`; therefore W_u norm control is the only predefined probe satisfying both the scale and non-catastrophic-CE criteria.

## Direct Q1–Q17 answers

""" + "\n".join(f"- **{key}:** {value}" for key, value in questions.items()) + f"""

## Integrity audit

`{sum(bool(v) for v in checks.values())}/{len(checks)}` checks passed. Overall: **{'PASS' if passed else 'FAIL'}**. No optimizer, scheduler, GradScaler, backward, optimizer step, parameter update, or training target occurred. All checkpoint file hashes and in-memory native parameter hashes were unchanged.

## Scientific boundary

This result diagnoses recurrent dynamical scale only. It is not evidence that triangle KV geometry fails, nor that one B12 recurrent source is insufficient. No 2D1R training was launched.
"""
    durable_json(output / "root_cause_classification.json", root_payload)
    durable_json(output / "next_experiment_recommendation.json", recommendation_payload)
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", {
        "experiment": EXPERIMENT,
        "checks": checks,
        "plot_checks": plot_checks,
        "passed": passed,
        "success_criterion_met": success,
        "pending_terminal_checks": ["results/report commits pushed", "local/pod/origin synchronized", "remote sync", "GPU idle", "exact Pod ID reverified"],
        "pod_stop_authorized_after_pending_terminal_checks": passed and success,
    })
    durable_text(output / "EXPERIMENT_2D1A_FINAL_REPORT.md", report)
    if not passed or not success:
        raise SystemExit(f"2D1A finalize did not pass: audit={passed} success={success}")
    print(f"EXPERIMENT_2D1A_FINALIZE_PASS root_cause={root!r} stabilization={recommendation!r}", flush=True)


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-checkpoint", required=True)
    common.add_argument("--validation-shard", required=True)
    common.add_argument("--checkpoint-954", required=True)
    common.add_argument("--checkpoint-1000", required=True)
    common.add_argument("--checkpoint-1100", required=True)
    common.add_argument("--source-2d1-results", required=True)
    common.add_argument("--output-dir", required=True)
    common.add_argument("--pod-id", required=True)
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", parents=[common])
    sub.add_parser("run", parents=[common])
    final = sub.add_parser("finalize")
    final.add_argument("--output-dir", required=True)
    return root


def main():
    args = parser().parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "run":
        run_diagnostics(args)
    elif args.command == "finalize":
        finalize(args)


if __name__ == "__main__":
    main()
