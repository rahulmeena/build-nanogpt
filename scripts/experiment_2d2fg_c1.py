#!/usr/bin/env python3
"""Frozen large true-incremental 2D2F versus 2D2G confirmation.

This module intentionally has no training, optimizer-construction, backward,
or scheduler path.  It strictly reopens two immutable checkpoints and runs
six functional inference controls on one shared, preregistered validation set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as torch_f


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0 as d0  # noqa: E402
import experiment_2d1 as d1  # noqa: E402
from experiment_2d2f_core import RecurrentKVGPT as FRecurrentKVGPT  # noqa: E402
from experiment_2d2g_core import StageBRecurrentKVGPT as GRecurrentKVGPT  # noqa: E402


EXPERIMENT = "2D2FG-C1"
BRANCH = "experiment-2d2fg-c1-frozen-large-head-to-head"
CONFIG = REPO_ROOT / "configs/exp2d2fg_c1_frozen_large_true_incremental_head_to_head.json"
OUTPUT_NAME = "experiment_2d2fg_c1_frozen_large_true_incremental_head_to_head"
F_SHA256 = "a58dd647e7aa70c22b5c1cd49cc708d85576e95c0a0c30fc93b1f0c02eae0ea6"
G_SHA256 = "36ec3fa28741fa8ce999c43e309baa175112b795c079575d2a89525f134c3da0"
F_SCHEMA = "exp2d2f_no_b2_recurrence_b3_w64_checkpoint_v1"
G_SCHEMA = "exp2d2g_stage_b_b3_w64_checkpoint_v1"
PARAMETERS = 124_475_906
F_GATE_B1 = 0.1745906472
F_GATE_B3 = 0.017236143350601196
G_GATE_B1 = 0.1679071039
G_GATE_B3 = 0.011108865961432457
VALIDATION_SHA256 = "8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f"
OLD_SUBSET_SHA256 = "8befbf790b3e522747cd39da306ec124464bf8dde1604caf64f299efa7e36216"
C1_SUBSET_SHA256 = "8c0a01e707a5d3928cf856e3d89d086321cd27b42962dd4dec6181ff30a48fcc"
T = 1024
BATCH = 64
SKIP_BATCHES = 20
BATCHES = 16
SEQUENCES = BATCH * BATCHES
TARGETS = SEQUENCES * T
OLD_START_TOKEN_OFFSET = 0
C1_START_TOKEN_OFFSET = 4 * BATCH * T + 1
# 2D2E-C1 consumed the inclusive raw interval [262145, 1310721].
# Begin one token later so the optional historical subset is also disjoint.
NEW_START_TOKEN_OFFSET = C1_START_TOKEN_OFFSET + TARGETS + 1
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_828
REGRESSION_ATOL = 1e-6
F_STATE_BYTES = 31_718_400
G_STATE_BYTES = 34_765_824
POSITION_BINS = (
    ("1-31", 0, 31),
    ("32-63", 31, 63),
    ("64-127", 63, 127),
    ("128-255", 127, 255),
    ("256-511", 255, 511),
    ("512-767", 511, 767),
    ("768-1023", 767, 1023),
)
AUDIT_POSITIONS = {1, 31, 32, 63, 64, 128, 512, 1024}
REQUIRED_FILES = (
    "EXPERIMENT_2D2FG_C1_FINAL_REPORT.md",
    "FINAL_AUDIT.json",
    "result_summary.json",
    "checkpoint_manifest.json",
    "architecture_manifest.json",
    "validation_subset_manifest.json",
    "disjointness_audit.json",
    "per_sequence_losses.json",
    "f_real.json",
    "f_b3_off.json",
    "f_b3_shuffled.json",
    "g_real.json",
    "g_b3_off.json",
    "g_b3_shuffled.json",
    "paired_f_vs_g.json",
    "recurrent_gain_comparison.json",
    "sequence_gap_comparison.json",
    "bootstrap_results.json",
    "position_bin_comparison.json",
    "memory_accounting.json",
    "performance.json",
    "commands_and_runtime.json",
    "HEARTBEAT.json",
    "UNATTENDED_FINAL_HANDOFF.md",
    "P1_per_sequence_f_minus_g.png",
    "P2_bootstrap_f_minus_g.png",
    "P3_recurrence_gains.png",
    "P4_sequence_gaps.png",
    "P5_position_bin_f_minus_g.png",
    "P6_ce_vs_state_bytes.png",
)
IMPLEMENTATION_FILES = (
    "configs/exp2d2fg_c1_frozen_large_true_incremental_head_to_head.json",
    "scripts/experiment_2d2fg_c1.py",
    "scripts/experiment_2d2f_core.py",
    "scripts/experiment_2d2g_core.py",
    "tests/test_experiment_2d2fg_c1.py",
)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value)
    return digest.hexdigest()


def aggregate_hashes(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def atomic_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path | str, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def implementation_fingerprint() -> dict:
    rows = {name: file_sha256(REPO_ROOT / name) for name in IMPLEMENTATION_FILES}
    return {
        "files": rows,
        "aggregate_sha256": hashlib.sha256(canonical_json(rows).encode()).hexdigest(),
    }


def require_git(clean: bool = True) -> None:
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"requires branch {BRANCH}")
    if clean and git_output("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("tracked implementation files are not clean")


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(canonical_json(list(value.shape)).encode())
        digest.update(tensor_bytes(value))
    return digest.hexdigest()


def model_finite(model) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())


def require_single_a100() -> tuple[torch.device, dict]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("exactly one visible CUDA device is required")
    if any(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise SystemExit("DDP/NCCL rank environment is forbidden")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if "A100-SXM4-80GB" not in properties.name or properties.total_memory < 79 * 1024**3:
        raise SystemExit(f"wrong GPU: {properties.name}, {properties.total_memory}")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().split(", ")
    return device, {
        "gpu_model": properties.name,
        "gpu_uuid": query[1],
        "driver": query[2],
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "visible_device_count": torch.cuda.device_count(),
        "total_memory_bytes": properties.total_memory,
        "ddp_initialized": torch.distributed.is_available() and torch.distributed.is_initialized(),
    }


def instantiate_base(device: torch.device):
    symbols = d0.support.load_training_symbols()
    base = symbols["GPT"](d0.model_config(symbols)).to(device)
    return base


def expected_architecture(label: str, artifact: dict) -> dict:
    return artifact if label == "F" else artifact["stage_b"]


def load_frozen_model(
    label: str,
    checkpoint: Path,
    architecture_artifact: Path,
    checkpoint_manifest_artifact: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    expected_sha = F_SHA256 if label == "F" else G_SHA256
    expected_schema = F_SCHEMA if label == "F" else G_SCHEMA
    expected_b1 = F_GATE_B1 if label == "F" else G_GATE_B1
    expected_b3 = F_GATE_B3 if label == "F" else G_GATE_B3
    observed_sha = file_sha256(checkpoint)
    sha_sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    verification_sidecar = checkpoint.with_suffix(checkpoint.suffix + ".verification.json")
    if observed_sha != expected_sha:
        raise SystemExit(f"{label} checkpoint SHA mismatch")
    if not sha_sidecar.is_file() or sha_sidecar.read_text().split()[0] != expected_sha:
        raise SystemExit(f"{label} checkpoint SHA sidecar mismatch")
    if not verification_sidecar.is_file():
        raise SystemExit(f"{label} checkpoint verification sidecar missing")
    payload = torch_load(checkpoint)
    architecture = read_json(architecture_artifact)
    manifest = read_json(checkpoint_manifest_artifact)
    model = (FRecurrentKVGPT if label == "F" else GRecurrentKVGPT)(instantiate_base(device)).to(device)
    model.load_state_dict(payload["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    payload_digest = state_dict_sha256(payload["model"])
    model_digest = state_dict_sha256(model.state_dict())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    architecture_expected = expected_architecture(label, architecture)
    checks = {
        "checkpoint_sha_exact": observed_sha == expected_sha,
        "sidecars_present_and_exact": sha_sidecar.is_file()
        and verification_sidecar.is_file()
        and read_json(verification_sidecar).get("passed") is True,
        "schema_exact": payload.get("schema") == expected_schema,
        "completed_updates_exact": (
            payload.get("completed_2d2f_updates") == 191
            if label == "F"
            else payload.get("completed_local_updates") == 191 and payload.get("stage") == "b"
        ),
        "architecture_checkpoint_matches_artifact": payload.get("architecture_manifest") == architecture_expected,
        "checkpoint_manifest_binds_sha": expected_sha in canonical_json(manifest),
        "strict_model_load": True,
        "payload_model_digest_matches_loaded_model": payload_digest == model_digest,
        "parameter_count_exact": parameter_count == PARAMETERS,
        "finite_tensors": model_finite(model),
        "weight_tying": model.base.transformer.wte.weight is model.base.lm_head.weight,
        "g_rec_present": hasattr(model, "g_rec"),
        "g_rec_b3_present": hasattr(model, "g_rec_b3"),
        "g_rec_b2_absent": not hasattr(model, "g_rec_b2") and "g_rec_b2" not in payload["model"],
        "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "b1_gate_expected": abs(model.g_rec.detach().float().tanh().item() - expected_b1) <= 2e-8,
        "b3_gate_expected": abs(model.g_rec_b3.detach().float().tanh().item() - expected_b3) <= 2e-8,
        "optimizer_not_instantiated": True,
    }
    if not all(checks.values()):
        raise SystemExit(f"{label} frozen-model integrity failed: {checks}")
    audit = {
        "label": label,
        "path": str(checkpoint.resolve()),
        "sha256": observed_sha,
        "bytes": checkpoint.stat().st_size,
        "schema": payload.get("schema"),
        "parameter_count": parameter_count,
        "g_rec_b1_raw": model.g_rec.detach().float().item(),
        "tanh_g_rec_b1": model.g_rec.detach().float().tanh().item(),
        "g_rec_b3_raw": model.g_rec_b3.detach().float().item(),
        "tanh_g_rec_b3": model.g_rec_b3.detach().float().tanh().item(),
        "model_state_sha256": model_digest,
        "architecture": architecture_expected,
        "architecture_artifact": {
            "path": str(architecture_artifact.resolve()),
            "sha256": file_sha256(architecture_artifact),
        },
        "checkpoint_manifest_artifact": {
            "path": str(checkpoint_manifest_artifact.resolve()),
            "sha256": file_sha256(checkpoint_manifest_artifact),
        },
        "checkpoint_verification_sidecar_sha256": file_sha256(verification_sidecar),
        "checks": checks,
        "passed": True,
    }
    del payload
    gc.collect()
    return model.eval(), audit


def batch_identity(x: torch.Tensor, y: torch.Tensor) -> dict:
    x_bytes = x.contiguous().numpy().tobytes()
    y_bytes = y.contiguous().numpy().tobytes()
    return {
        "input_sha256": bytes_sha256(x_bytes),
        "target_sha256": bytes_sha256(y_bytes),
        "combined_sha256": bytes_sha256(x_bytes, y_bytes),
    }


def sequence_identity(x: torch.Tensor, y: torch.Tensor, batch_index: int, row: int) -> dict:
    x_bytes = x[row].contiguous().numpy().tobytes()
    y_bytes = y[row].contiguous().numpy().tobytes()
    return {
        "batch": batch_index,
        "row": row,
        "input_sha256": bytes_sha256(x_bytes),
        "target_sha256": bytes_sha256(y_bytes),
        "combined_sha256": bytes_sha256(x_bytes, y_bytes),
    }


def collect_subset(val_path: Path, start_token_offset: int, batches: int, include_sequences: bool) -> dict:
    state = {
        "shards": [str(val_path.resolve())],
        "batch_size": BATCH,
        "sequence_length": T,
        "current_shard": 0,
        "current_position": int(start_token_offset),
    }
    loader = d1.ExplicitShardLoader([val_path], BATCH, T, state=state)
    batch_rows = []
    sequence_rows = []
    for batch_index in range(int(batches)):
        x, y = loader.next_batch()
        batch_rows.append(batch_identity(x, y))
        if include_sequences:
            sequence_rows.extend(sequence_identity(x, y, batch_index, row) for row in range(BATCH))
    start = int(start_token_offset)
    end = start + int(batches) * BATCH * T
    return {
        "start_token_offset": start,
        "batch_count": int(batches),
        "batch_size": BATCH,
        "sequence_length": T,
        "sequence_count": int(batches) * BATCH,
        "targets_per_control": int(batches) * BATCH * T,
        "raw_token_interval_inclusive": [start, end],
        "batch_identities": batch_rows,
        "sequence_identities": sequence_rows,
        "subset_sha256": aggregate_hashes(row["combined_sha256"] for row in batch_rows),
        "end_cursor": loader.state_dict(),
    }


def build_subset_and_disjointness(val_path: Path) -> tuple[dict, dict]:
    old = collect_subset(val_path, OLD_START_TOKEN_OFFSET, 4, True)
    c1 = collect_subset(val_path, C1_START_TOKEN_OFFSET, 16, True)
    new = collect_subset(val_path, NEW_START_TOKEN_OFFSET, BATCHES, True)
    old_sequences = {row["combined_sha256"] for row in old["sequence_identities"]}
    c1_sequences = {row["combined_sha256"] for row in c1["sequence_identities"]}
    new_sequences = {row["combined_sha256"] for row in new["sequence_identities"]}
    old_batches = {row["combined_sha256"] for row in old["batch_identities"]}
    c1_batches = {row["combined_sha256"] for row in c1["batch_identities"]}
    new_batches = {row["combined_sha256"] for row in new["batch_identities"]}
    checks = {
        "old_subset_hash_reproduced": old["subset_sha256"] == OLD_SUBSET_SHA256,
        "c1_subset_hash_reproduced": c1["subset_sha256"] == C1_SUBSET_SHA256,
        "new_sequence_count_exact": len(new_sequences) == SEQUENCES,
        "new_targets_exact": new["targets_per_control"] == TARGETS,
        "no_sequence_overlap_with_old": not (new_sequences & old_sequences),
        "no_batch_overlap_with_old": not (new_batches & old_batches),
        "no_sequence_overlap_with_c1": not (new_sequences & c1_sequences),
        "no_batch_overlap_with_c1": not (new_batches & c1_batches),
        "raw_intervals_disjoint": new["raw_token_interval_inclusive"][0]
        > c1["raw_token_interval_inclusive"][1],
    }
    if not all(checks.values()):
        raise SystemExit(f"fresh-subset disjointness failed: {checks}")
    manifest = {
        "schema": "2d2fg_c1_validation_subset_v1",
        "validation_shard": str(val_path.resolve()),
        "validation_shard_sha256": file_sha256(val_path),
        "selection": new,
        "passed": True,
    }
    audit = {
        "schema": "2d2fg_c1_disjointness_v1",
        "mandatory_prior_subset": old,
        "optional_2d2e_c1_subset": c1,
        "new_subset": {
            "subset_sha256": new["subset_sha256"],
            "raw_token_interval_inclusive": new["raw_token_interval_inclusive"],
            "sequence_count": new["sequence_count"],
            "batch_count": new["batch_count"],
        },
        "intersection_counts": {
            "new_vs_old_sequences": len(new_sequences & old_sequences),
            "new_vs_old_batches": len(new_batches & old_batches),
            "new_vs_c1_sequences": len(new_sequences & c1_sequences),
            "new_vs_c1_batches": len(new_batches & c1_batches),
        },
        "checks": checks,
        "passed": True,
    }
    return manifest, audit


def control_name(label: str, condition: str) -> str:
    if label == "F":
        return {"real": "all_real", "off": "b3_off", "shuffled": "b3_shuffled"}[condition]
    return {"real": "real", "off": "b3_off", "shuffled": "b3_shuffled"}[condition]


def cache_limits(label: str) -> dict:
    return {
        "b1": 1,
        "b2": 31 if label == "F" else 1023,
        "b3": 63,
        "b4_b12": 1023,
        "h10": 1023,
        "h12": 1023,
        "h11": 0,
    }


def validate_cache_audit(label: str, audit: dict) -> bool:
    limits = cache_limits(label)
    lengths = audit.get("cache_lengths")
    if not isinstance(lengths, list) or len(lengths) != 12:
        return False
    expected = [limits["b1"], limits["b2"], limits["b3"]] + [limits["b4_b12"]] * 9
    if any(int(value) > cap for value, cap in zip(lengths, expected)):
        return False
    if int(audit.get("h10_ring_length", -1)) > limits["h10"]:
        return False
    if int(audit.get("h12_ring_length", -1)) > limits["h12"]:
        return False
    no_h11 = (
        audit.get("b11_recurrent_ring_present", False) is False
        and audit.get("has_b11_ring", False) is False
        and "h11_ring" not in audit
        and "h11_ring_length" not in audit
    )
    return (
        no_h11
        and audit.get("passed") is True
        and audit.get("physical_storage_exact") is True
    )


@torch.inference_mode()
def run_control(model, label: str, condition: str, x: torch.Tensor, y: torch.Tensor, permutation) -> dict:
    device = model.base.transformer.wte.weight.device
    name = control_name(label, condition)
    state = model.init_incremental_state(BATCH, device=device, b3_full_cache=False)
    per_sequence = torch.zeros(BATCH, dtype=torch.float64)
    per_position_sum = np.zeros(T, dtype=np.float64)
    total = 0.0
    cache_rows = []
    for position in range(T):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, state, _ = model.incremental_step(
                x[:, position],
                state,
                control=name,
                recurrent_permutation=permutation if condition == "shuffled" else None,
                return_diagnostics=False,
                diagnostic_attention_weights=False,
            )
        losses = torch_f.cross_entropy(
            logits[:, 0].float(), y[:, position], reduction="none"
        ).double().cpu()
        per_sequence += losses
        per_position_sum[position] += losses.sum().item()
        total += losses.sum().item()
        one_based = position + 1
        if one_based in AUDIT_POSITIONS:
            audit = model.incremental_cache_audit(state)
            cache_rows.append({"position": one_based, "audit": audit, "passed": validate_cache_audit(label, audit)})
    final_audit = model.incremental_cache_audit(state)
    if not validate_cache_audit(label, final_audit) or not all(row["passed"] for row in cache_rows):
        raise SystemExit(f"{label} {condition} cache audit failed")
    return {
        "loss_sum": total,
        "targets": x.numel(),
        "per_sequence_losses": (per_sequence / T).tolist(),
        "per_position_sum": per_position_sum.tolist(),
        "cache_rows": cache_rows,
        "final_cache_audit": final_audit,
        "true_incremental": True,
        "complete_prefix_recomputation": False,
    }


def update_heartbeat(output: Path, phase: str, **extra) -> None:
    atomic_json(
        output / "HEARTBEAT.json",
        {
            "schema": "2d2fg_c1_heartbeat_v1",
            "experiment": EXPERIMENT,
            "phase": phase,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            **extra,
        },
    )


def selected_loader(val_path: Path):
    state = {
        "shards": [str(val_path.resolve())],
        "batch_size": BATCH,
        "sequence_length": T,
        "current_shard": 0,
        "current_position": NEW_START_TOKEN_OFFSET,
    }
    return d1.ExplicitShardLoader([val_path], BATCH, T, state=state)


def archived_loader(val_path: Path):
    return d1.ExplicitShardLoader([val_path], BATCH, T)


def condition_from_raw(raw: dict) -> dict:
    targets = sum(row["targets"] for row in raw["batches"])
    loss_sum = sum(row["loss_sum"] for row in raw["batches"])
    per_position = np.sum(
        np.asarray([row["per_position_sum"] for row in raw["batches"]], dtype=np.float64), axis=0
    ) / (len(raw["batches"]) * BATCH)
    per_sequence = [value for row in raw["batches"] for value in row["per_sequence_losses"]]
    return {
        "validation_loss": loss_sum / targets,
        "validation_targets": targets,
        "per_sequence_losses": per_sequence,
        "per_position_loss": per_position.tolist(),
        "per_batch_losses": [row["loss_sum"] / row["targets"] for row in raw["batches"]],
        "cache_rows": [row["cache_rows"] for row in raw["batches"]],
        "final_cache_audits": [row["final_cache_audit"] for row in raw["batches"]],
        "true_incremental": True,
        "complete_prefix_recomputation": False,
        "performance": raw["performance"],
    }


@torch.inference_mode()
def run_model_evaluation(
    model,
    label: str,
    val_path: Path,
    expected_subset: dict,
    archived_result: dict,
    output: Path,
) -> tuple[dict, dict, str, dict]:
    device = model.base.transformer.wte.weight.device
    permutation = torch.arange(BATCH, device=device).roll(1)
    if bool(torch.any(permutation == torch.arange(BATCH, device=device))):
        raise SystemExit("shuffle permutation has a fixed point")
    regression_loader = archived_loader(val_path)
    regression_x, regression_y = regression_loader.next_batch()
    regression_identity = batch_identity(regression_x, regression_y)
    regression_x = regression_x.to(device)
    regression_y = regression_y.to(device)
    regression = {}
    for condition in ("real", "off", "shuffled"):
        current = run_control(model, label, condition, regression_x, regression_y, permutation)
        archived_key = control_name(label, condition)
        archived = archived_result["controls"][archived_key]
        expected_sequences = archived["per_sequence_losses"][:BATCH]
        maximum = max(
            abs(left - right)
            for left, right in zip(current["per_sequence_losses"], expected_sequences)
        )
        observed_loss = current["loss_sum"] / current["targets"]
        expected_loss = archived["per_batch_losses"][0]
        regression[condition] = {
            "observed_loss": observed_loss,
            "archived_loss": expected_loss,
            "absolute_loss_difference": abs(observed_loss - expected_loss),
            "maximum_per_sequence_absolute_difference": maximum,
            "tolerance": REGRESSION_ATOL,
            "passed": abs(observed_loss - expected_loss) <= REGRESSION_ATOL and maximum <= REGRESSION_ATOL,
        }
    del regression_x, regression_y
    torch.cuda.empty_cache()
    if not all(row["passed"] for row in regression.values()):
        raise SystemExit(f"{label} archived-subset regression failed: {regression}")
    loader = selected_loader(val_path)
    raw = {condition: {"batches": []} for condition in ("real", "off", "shuffled")}
    identities = []
    model_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(BATCHES):
        cpu_x, cpu_y = loader.next_batch()
        identity = batch_identity(cpu_x, cpu_y)
        expected = expected_subset["selection"]["batch_identities"][batch_index]
        if identity != expected:
            raise SystemExit(f"{label} batch {batch_index} differs from preregistered subset")
        identities.append(identity)
        x, y = cpu_x.to(device), cpu_y.to(device)
        for condition in ("real", "off", "shuffled"):
            started = time.monotonic()
            current = run_control(model, label, condition, x, y, permutation)
            current["wall_seconds"] = time.monotonic() - started
            raw[condition]["batches"].append(current)
            update_heartbeat(
                output,
                "evaluating",
                model=label,
                completed_batches=batch_index,
                current_batch=batch_index + 1,
                completed_condition=condition,
                total_batches=BATCHES,
            )
            torch.cuda.empty_cache()
        del x, y, cpu_x, cpu_y
        torch.cuda.empty_cache()
    model_wall = time.monotonic() - model_started
    for condition in raw:
        condition_wall = sum(row["wall_seconds"] for row in raw[condition]["batches"])
        raw[condition]["performance"] = {
            "wall_seconds": condition_wall,
            "targets_per_second": TARGETS / condition_wall,
        }
    result = {condition: condition_from_raw(raw[condition]) for condition in raw}
    performance = {
        "wall_seconds": model_wall,
        "all_conditions_targets_per_second": (3 * TARGETS) / model_wall,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "condition_performance": {condition: result[condition]["performance"] for condition in result},
    }
    subset_sha = aggregate_hashes(row["combined_sha256"] for row in identities)
    if subset_sha != expected_subset["selection"]["subset_sha256"]:
        raise SystemExit(f"{label} evaluated subset aggregate differs")
    return result, {
        "archived_prefix_identity": regression_identity,
        "conditions": regression,
        "passed": True,
    }, subset_sha, performance


def paired_stats(values) -> dict:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows),
        "median": statistics.median(rows),
        "sample_standard_deviation": statistics.stdev(rows),
        "positive": sum(value > 0 for value in rows),
        "negative": sum(value < 0 for value in rows),
        "ties": sum(value == 0 for value in rows),
    }


def bootstrap(values, seed: int, store_distribution: bool = False) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 500
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        stop = min(start + chunk, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, array.size, size=(stop - start, array.size), endpoint=False)
        distribution[start:stop] = array[indices].mean(axis=1)
    result = {
        "seed": int(seed),
        "resamples": BOOTSTRAP_RESAMPLES,
        "sampling_unit": "sequence",
        "statistic": "paired mean",
        "percentile_method": "numpy linear percentile",
        "confidence_level": 0.95,
        "bootstrap_mean": float(distribution.mean()),
        "lower": float(np.percentile(distribution, 2.5)),
        "upper": float(np.percentile(distribution, 97.5)),
    }
    if store_distribution:
        result["distribution"] = distribution.tolist()
    return result


def recurrence_classification(gain: float, gap: float, gain_ci: dict, gap_ci: dict) -> str:
    if gain_ci["lower"] > 0 and gap_ci["lower"] > 0:
        return "STRONGLY CONFIRMED"
    if gain > 0 and gap > 0:
        return "DIRECTIONALLY CONFIRMED"
    return "NOT CONFIRMED"


def absolute_classification(delta: float, interval: dict) -> str:
    if delta > 0 and interval["lower"] > 0:
        return "2D2G ABSOLUTE CE ADVANTAGE STRONGLY CONFIRMED"
    if delta > 0:
        return "2D2G ABSOLUTE CE ADVANTAGE DIRECTIONALLY CONFIRMED"
    return "ABSOLUTE CE ADVANTAGE NOT CONFIRMED"


def position_bin_rows(f_real: dict, g_real: dict) -> dict:
    f = np.asarray(f_real["per_position_loss"], dtype=np.float64)
    g = np.asarray(g_real["per_position_loss"], dtype=np.float64)
    rows = {}
    for label, start, stop in POSITION_BINS:
        f_ce = float(f[start:stop].mean())
        g_ce = float(g[start:stop].mean())
        rows[label] = {
            "position_start_inclusive": start + 1,
            "position_end_inclusive": stop,
            "f_ce": f_ce,
            "g_ce": g_ce,
            "f_minus_g": f_ce - g_ce,
        }
    return rows


def theoretical_memory() -> dict:
    saving = G_STATE_BYTES - F_STATE_BYTES
    return {
        "accounting_dtype": "BF16",
        "bytes_per_element": 2,
        "2D2F": {
            "total_inference_state_bytes_B1": F_STATE_BYTES,
            "b1_historical_kv": 1,
            "b2_historical_kv": 31,
            "b3_historical_kv": 63,
            "b10_raw_ring": 1023,
            "b12_raw_ring": 1023,
            "b11_raw_ring": 0,
        },
        "2D2G": {
            "total_inference_state_bytes_B1": G_STATE_BYTES,
            "b1_historical_kv": 1,
            "b2_historical_kv": 1023,
            "b3_historical_kv": 63,
            "b10_raw_ring": 1023,
            "b12_raw_ring": 1023,
            "b11_raw_ring": 0,
        },
        "f_saving_bytes": saving,
        "f_saving_mib": saving / 1024**2,
        "f_saving_percent_of_g": 100.0 * saving / G_STATE_BYTES,
        "passed": saving == 3_047_424,
    }


def build_questions(summary: dict) -> dict:
    paired = summary["paired_f_vs_g"]
    boot = summary["bootstrap"]
    f = summary["2D2F"]
    g = summary["2D2G"]
    memory = summary["memory"]
    return {
        "Q1": f["real_ce"],
        "Q2": g["real_ce"],
        "Q3": summary["f_minus_g_ce"],
        "Q4": {"historical": 0.0036352843635349963, "new": summary["f_minus_g_ce"], "difference": summary["f_minus_g_ce"] - 0.0036352843635349963},
        "Q5": paired["f_wins"],
        "Q6": paired["g_wins"],
        "Q7": paired["paired_stats"]["median"],
        "Q8": {key: boot["f_minus_g"][key] for key in ("lower", "upper")},
        "Q9": summary["absolute_quality_classification"],
        "Q10": f["b3_gain"],
        "Q11": f["b3_sequence_gap"],
        "Q12": f["gain_paired"]["positive"],
        "Q13": f["gap_paired"]["positive"],
        "Q14": {"gain": {key: boot["f_gain"][key] for key in ("lower", "upper")}, "gap": {key: boot["f_gap"][key] for key in ("lower", "upper")}},
        "Q15": g["b3_gain"],
        "Q16": g["b3_sequence_gap"],
        "Q17": g["gain_paired"]["positive"],
        "Q18": g["gap_paired"]["positive"],
        "Q19": {"gain": {key: boot["g_gain"][key] for key in ("lower", "upper")}, "gap": {key: boot["g_gap"][key] for key in ("lower", "upper")}},
        "Q20": summary["delta_recurrent_gain"],
        "Q21": {key: boot["recurrent_gain_difference"][key] for key in ("lower", "upper")},
        "Q22": summary["stronger_sequence_specificity"],
        "Q23": {"bytes": memory["f_saving_bytes"], "mib": memory["f_saving_mib"], "percent": memory["f_saving_percent_of_g"]},
        "Q24": summary["canonical_architecture"] == "2D2F",
        "Q25": summary["recommended_next_training_experiment"],
    }


def make_plots(output: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = summary["2D2F"]
    g = summary["2D2G"]
    differences = np.asarray(summary["paired_f_vs_g"]["per_sequence_f_minus_g"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(differences, bins=50)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Per-sequence CE: F_REAL - G_REAL", ylabel="Sequences")
    fig.tight_layout(); fig.savefig(output / REQUIRED_FILES[-6], dpi=160); plt.close(fig)

    distribution = summary["bootstrap"]["f_minus_g"]["distribution"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(distribution, bins=60)
    ax.axvline(summary["bootstrap"]["f_minus_g"]["lower"], color="red")
    ax.axvline(summary["bootstrap"]["f_minus_g"]["upper"], color="red")
    ax.set(xlabel="Bootstrap mean F_REAL - G_REAL", ylabel="Resamples")
    fig.tight_layout(); fig.savefig(output / REQUIRED_FILES[-5], dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["2D2F", "2D2G"], [f["b3_gain"], g["b3_gain"]])
    ax.axhline(0, color="black", linewidth=0.8); ax.set(ylabel="OFF - REAL CE")
    fig.tight_layout(); fig.savefig(output / REQUIRED_FILES[-4], dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["2D2F", "2D2G"], [f["b3_sequence_gap"], g["b3_sequence_gap"]])
    ax.axhline(0, color="black", linewidth=0.8); ax.set(ylabel="SHUFFLED - REAL CE")
    fig.tight_layout(); fig.savefig(output / REQUIRED_FILES[-3], dpi=160); plt.close(fig)

    bins = summary["position_bins"]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(list(bins), [row["f_minus_g"] for row in bins.values()])
    ax.axhline(0, color="black", linewidth=0.8); ax.set(ylabel="F_REAL - G_REAL CE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(output / REQUIRED_FILES[-2], dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter([F_STATE_BYTES, G_STATE_BYTES], [f["real_ce"], g["real_ce"]])
    ax.annotate("2D2F", (F_STATE_BYTES, f["real_ce"])); ax.annotate("2D2G", (G_STATE_BYTES, g["real_ce"]))
    ax.set(xlabel="BF16 inference-state bytes", ylabel="REAL CE")
    fig.tight_layout(); fig.savefig(output / REQUIRED_FILES[-1], dpi=160); plt.close(fig)


def report_text(summary: dict, audit: dict, git_results_commit: str | None = None) -> str:
    f = summary["2D2F"]
    g = summary["2D2G"]
    boot = summary["bootstrap"]
    memory = summary["memory"]
    git_value = git_results_commit or "PENDING RESULTS COMMIT"
    lines = [
        "EXPERIMENT 2D2FG-C1 COMPLETE",
        "",
        "ABSOLUTE QUALITY CLASSIFICATION:",
        summary["absolute_quality_classification"],
        "",
        "2D2F REAL CE:",
        repr(f["real_ce"]),
        "",
        "2D2G REAL CE:",
        repr(g["real_ce"]),
        "",
        "F_MINUS_G CE:",
        repr(summary["f_minus_g_ce"]),
        "",
        "2D2F B3 RECURRENCE CONFIRMATION:",
        f["recurrence_confirmation"],
        "",
        "2D2G B3 RECURRENCE CONFIRMATION:",
        g["recurrence_confirmation"],
        "",
        "CANONICAL ARCHITECTURE UNDER PREREGISTERED CE/STATE POLICY:",
        summary["canonical_architecture"],
        "",
        "## Frozen provenance",
        "",
        f"- 2D2F checkpoint SHA-256: `{summary['checkpoints']['F']['sha256']}`",
        f"- 2D2G checkpoint SHA-256: `{summary['checkpoints']['G']['sha256']}`",
        f"- Validation subset SHA-256: `{summary['validation_subset_sha256']}`",
        f"- Sequences: `{SEQUENCES}`; targets per control: `{TARGETS}`.",
        f"- Mandatory and optional disjointness passed: `{summary['disjointness_passed']}`.",
        "",
        "## Absolute CE and paired sequence result",
        "",
        f"- F_REAL: `{f['real_ce']}`",
        f"- G_REAL: `{g['real_ce']}`",
        f"- F minus G: `{summary['f_minus_g_ce']}`",
        f"- Mean/median/std: `{summary['paired_f_vs_g']['paired_stats']}`",
        f"- F wins / G wins / ties: `{summary['paired_f_vs_g']['f_wins']}` / `{summary['paired_f_vs_g']['g_wins']}` / `{summary['paired_f_vs_g']['ties']}`",
        f"- Bootstrap 95% CI: `[{boot['f_minus_g']['lower']}, {boot['f_minus_g']['upper']}]` (20,000 paired sequence resamples; seed 20260828).",
        "",
        "## B3 recurrence",
        "",
        f"- 2D2F gain OFF-REAL: `{f['b3_gain']}`; CI `[{boot['f_gain']['lower']}, {boot['f_gain']['upper']}]`; paired `{f['gain_paired']}`.",
        f"- 2D2F gap SHUFFLED-REAL: `{f['b3_sequence_gap']}`; CI `[{boot['f_gap']['lower']}, {boot['f_gap']['upper']}]`; paired `{f['gap_paired']}`.",
        f"- 2D2G gain OFF-REAL: `{g['b3_gain']}`; CI `[{boot['g_gain']['lower']}, {boot['g_gain']['upper']}]`; paired `{g['gain_paired']}`.",
        f"- 2D2G gap SHUFFLED-REAL: `{g['b3_sequence_gap']}`; CI `[{boot['g_gap']['lower']}, {boot['g_gap']['upper']}]`; paired `{g['gap_paired']}`.",
        f"- F gain minus G gain: `{summary['delta_recurrent_gain']}`; CI `[{boot['recurrent_gain_difference']['lower']}, {boot['recurrent_gain_difference']['upper']}]`.",
        f"- F gap minus G gap: `{summary['delta_sequence_gap']}`; CI `[{boot['sequence_gap_difference']['lower']}, {boot['sequence_gap_difference']['upper']}]`.",
        "",
        "## Position-bin comparison",
        "",
        "| Positions | F CE | G CE | F-G |",
        "|---|---:|---:|---:|",
    ]
    for label, row in summary["position_bins"].items():
        lines.append(f"| {label} | {row['f_ce']:.12g} | {row['g_ce']:.12g} | {row['f_minus_g']:.12g} |")
    lines += [
        "",
        "## Memory and preregistered policy",
        "",
        f"- F state: `{F_STATE_BYTES}` bytes; G state: `{G_STATE_BYTES}` bytes.",
        f"- F saves `{memory['f_saving_bytes']}` bytes (`{memory['f_saving_mib']}` MiB; `{memory['f_saving_percent_of_g']}`%).",
        f"- F CE cost: `{summary['f_minus_g_ce']}`; CE cost per MiB saved: `{summary['ce_cost_per_mib_saved']}`.",
        f"- The preregistered <=0.005 CE/state policy selects **{summary['canonical_architecture']}**.",
        "",
        "## Scientific questions Q1-Q25",
        "",
    ]
    for index in range(1, 26):
        lines += [f"### Q{index}", "", f"`{json.dumps(summary['questions'][f'Q{index}'], sort_keys=True)}`", ""]
    lines += [
        "## Exactly one recommended next training experiment",
        "",
        summary["recommended_next_training_experiment"],
        "",
        "Do not execute it as part of this confirmation.",
        "",
        "## Integrity, Git, and runtime",
        "",
        f"- Zero mutation counters: `{summary['mutation_counters']}`",
        f"- Cache and physical-state audit passed: `{audit['checks']['all_cache_audits_passed']}`",
        f"- Implementation regression passed: `{audit['checks']['both_archived_prefix_regressions_passed']}`",
        f"- Results commit: `{git_value}`",
        f"- Artifact path: `{summary['artifact_path']}`",
        f"- Pod `{summary['pod']['id']}` is `{summary['pod']['status']}` pending terminal publication/backup stop.",
        f"- Persistent volume `{summary['pod']['persistent_volume_id']}` is retained.",
        "",
        "# EXPERIMENT 2D2FG-C1 COMPLETE",
        "",
    ]
    return "\n".join(lines)


def run_preflight(args) -> None:
    require_git(clean=True)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    update_heartbeat(output, "preflight")
    device, hardware = require_single_a100()
    if file_sha256(args.validation_shard) != VALIDATION_SHA256:
        raise SystemExit("validation shard SHA mismatch")
    stop = read_json(args.stop_capability)
    stop_checks = {
        "authenticated": stop.get("authenticated") is True,
        "pod_id_exact": stop.get("pod_id") == "rvgztsr0azrwyo",
        "pod_name_exact": stop.get("pod_name") == "happy_apricot_stork",
        "gpu_count_exact": stop.get("gpu_count") == 1,
        "volume_exact": stop.get("volume_id") == "yhzyb27fb5",
        "exact_stop_command": stop.get("exact_stop_command") == "runpodctl pod stop rvgztsr0azrwyo -o json",
        "passed": stop.get("passed") is True,
    }
    if not all(stop_checks.values()):
        raise SystemExit(f"authenticated stop capability failed: {stop_checks}")
    subset, disjointness = build_subset_and_disjointness(Path(args.validation_shard))
    atomic_json(output / "validation_subset_manifest.json", subset)
    atomic_json(output / "disjointness_audit.json", disjointness)
    models = {}
    for label, checkpoint, architecture, checkpoint_manifest in (
        ("F", args.f_checkpoint, args.f_architecture, args.f_checkpoint_manifest),
        ("G", args.g_checkpoint, args.g_architecture, args.g_checkpoint_manifest),
    ):
        model, audit = load_frozen_model(
            label,
            Path(checkpoint),
            Path(architecture),
            Path(checkpoint_manifest),
            device,
        )
        models[label] = audit
        del model
        gc.collect(); torch.cuda.empty_cache()
    storage = subprocess.check_output(["df", "-Pk", "/workspace"], text=True).splitlines()[-1].split()
    preflight = {
        "schema": "2d2fg_c1_preflight_v1",
        "experiment": EXPERIMENT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "branch": git_output("branch", "--show-current"),
            "implementation_commit": git_output("rev-parse", "HEAD"),
            "implementation_fingerprint": implementation_fingerprint(),
        },
        "hardware": hardware,
        "storage": {
            "filesystem_1k_blocks": int(storage[1]),
            "used_1k_blocks": int(storage[2]),
            "available_1k_blocks": int(storage[3]),
            "mount": storage[-1],
        },
        "stop_capability": {"audit": stop, "checks": stop_checks, "passed": True},
        "checkpoints": models,
        "validation_subset_sha256": subset["selection"]["subset_sha256"],
        "disjointness": disjointness,
        "mutation_counters": {"optimizer_steps": 0, "backward_calls": 0, "parameter_updates": 0, "scheduler_steps": 0, "training_targets": 0},
        "passed": True,
    }
    atomic_json(output / "preflight_audit.json", preflight)
    atomic_json(output / "checkpoint_manifest.json", {"F": models["F"], "G": models["G"], "passed": True})
    atomic_json(output / "architecture_manifest.json", {"F": models["F"]["architecture"], "G": models["G"]["architecture"], "passed": True})
    update_heartbeat(output, "preflight_complete", passed=True)
    print(output / "preflight_audit.json")


def require_preflight(output: Path) -> dict:
    preflight = read_json(output / "preflight_audit.json")
    if preflight.get("passed") is not True:
        raise SystemExit("passing preflight is required")
    if preflight["git"]["implementation_commit"] != git_output("rev-parse", "HEAD"):
        raise SystemExit("implementation commit changed after preflight")
    if preflight["git"]["implementation_fingerprint"] != implementation_fingerprint():
        raise SystemExit("implementation fingerprint changed after preflight")
    return preflight


def run_evaluate(args) -> None:
    require_git(clean=True)
    output = Path(args.output_dir).resolve()
    preflight = require_preflight(output)
    device, hardware = require_single_a100()
    subset = read_json(output / "validation_subset_manifest.json")
    disjointness = read_json(output / "disjointness_audit.json")
    if disjointness.get("passed") is not True:
        raise SystemExit("disjointness audit must pass")
    started = time.monotonic()
    update_heartbeat(output, "loading_F")
    f_model, f_checkpoint = load_frozen_model(
        "F", Path(args.f_checkpoint), Path(args.f_architecture), Path(args.f_checkpoint_manifest), device
    )
    f_initial = state_dict_sha256(f_model.state_dict())
    f_result, f_regression, f_subset, f_performance = run_model_evaluation(
        f_model, "F", Path(args.validation_shard), subset, read_json(args.f_incremental_archive), output
    )
    f_final = state_dict_sha256(f_model.state_dict())
    if f_initial != f_final:
        raise SystemExit("2D2F model state changed during evaluation")
    del f_model
    gc.collect(); torch.cuda.empty_cache()
    update_heartbeat(output, "loading_G")
    g_model, g_checkpoint = load_frozen_model(
        "G", Path(args.g_checkpoint), Path(args.g_architecture), Path(args.g_checkpoint_manifest), device
    )
    g_initial = state_dict_sha256(g_model.state_dict())
    g_result, g_regression, g_subset, g_performance = run_model_evaluation(
        g_model, "G", Path(args.validation_shard), subset, read_json(args.g_incremental_archive), output
    )
    g_final = state_dict_sha256(g_model.state_dict())
    if g_initial != g_final:
        raise SystemExit("2D2G model state changed during evaluation")
    del g_model
    gc.collect(); torch.cuda.empty_cache()
    if f_subset != g_subset or f_subset != subset["selection"]["subset_sha256"]:
        raise SystemExit("F and G did not evaluate the exact same subset")
    f_real = np.asarray(f_result["real"]["per_sequence_losses"])
    f_off = np.asarray(f_result["off"]["per_sequence_losses"])
    f_shuffled = np.asarray(f_result["shuffled"]["per_sequence_losses"])
    g_real = np.asarray(g_result["real"]["per_sequence_losses"])
    g_off = np.asarray(g_result["off"]["per_sequence_losses"])
    g_shuffled = np.asarray(g_result["shuffled"]["per_sequence_losses"])
    vectors = {
        "f_minus_g": f_real - g_real,
        "f_gain": f_off - f_real,
        "f_gap": f_shuffled - f_real,
        "g_gain": g_off - g_real,
        "g_gap": g_shuffled - g_real,
    }
    vectors["recurrent_gain_difference"] = vectors["f_gain"] - vectors["g_gain"]
    vectors["sequence_gap_difference"] = vectors["f_gap"] - vectors["g_gap"]
    bootstrap_rows = {}
    for offset, name in enumerate(
        ("f_minus_g", "f_gain", "f_gap", "g_gain", "g_gap", "recurrent_gain_difference", "sequence_gap_difference")
    ):
        bootstrap_rows[name] = bootstrap(
            vectors[name], BOOTSTRAP_SEED + offset, store_distribution=name == "f_minus_g"
        )
    f_real_ce = f_result["real"]["validation_loss"]
    f_off_ce = f_result["off"]["validation_loss"]
    f_shuffled_ce = f_result["shuffled"]["validation_loss"]
    g_real_ce = g_result["real"]["validation_loss"]
    g_off_ce = g_result["off"]["validation_loss"]
    g_shuffled_ce = g_result["shuffled"]["validation_loss"]
    f_gain = f_off_ce - f_real_ce
    f_gap = f_shuffled_ce - f_real_ce
    g_gain = g_off_ce - g_real_ce
    g_gap = g_shuffled_ce - g_real_ce
    delta = f_real_ce - g_real_ce
    memory = theoretical_memory()
    canonical = "2D2F" if delta <= 0.005 and F_STATE_BYTES < G_STATE_BYTES else "2D2G"
    recommendation = (
        "2D2J: add B4 W128 plus B9→B4 to the clean frozen 2D2F architecture; retain B1 W2+B12→B1, B2 W32 with no recurrence, and B3 W64+B10→B3."
        if canonical == "2D2F"
        else "2D2J-G: add B4 W128 plus B9→B4 to the frozen 2D2G architecture; make no other architecture change."
    )
    f_gain_stats = paired_stats(vectors["f_gain"])
    f_gap_stats = paired_stats(vectors["f_gap"])
    g_gain_stats = paired_stats(vectors["g_gain"])
    g_gap_stats = paired_stats(vectors["g_gap"])
    paired_fg = paired_stats(vectors["f_minus_g"])
    position_bins = position_bin_rows(f_result["real"], g_result["real"])
    mutation = {"optimizer_steps": 0, "backward_calls": 0, "parameter_updates": 0, "scheduler_steps": 0, "training_targets": 0}
    summary = {
        "schema": "2d2fg_c1_result_summary_v1",
        "experiment": EXPERIMENT,
        "absolute_quality_classification": absolute_classification(delta, bootstrap_rows["f_minus_g"]),
        "canonical_architecture": canonical,
        "2D2F": {
            "real_ce": f_real_ce,
            "off_ce": f_off_ce,
            "shuffled_ce": f_shuffled_ce,
            "b3_gain": f_gain,
            "b3_sequence_gap": f_gap,
            "gain_paired": f_gain_stats,
            "gap_paired": f_gap_stats,
            "recurrence_confirmation": recurrence_classification(f_gain, f_gap, bootstrap_rows["f_gain"], bootstrap_rows["f_gap"]),
        },
        "2D2G": {
            "real_ce": g_real_ce,
            "off_ce": g_off_ce,
            "shuffled_ce": g_shuffled_ce,
            "b3_gain": g_gain,
            "b3_sequence_gap": g_gap,
            "gain_paired": g_gain_stats,
            "gap_paired": g_gap_stats,
            "recurrence_confirmation": recurrence_classification(g_gain, g_gap, bootstrap_rows["g_gain"], bootstrap_rows["g_gap"]),
        },
        "f_minus_g_ce": delta,
        "paired_f_vs_g": {
            "paired_stats": paired_fg,
            "f_wins": paired_fg["negative"],
            "g_wins": paired_fg["positive"],
            "ties": paired_fg["ties"],
            "per_sequence_f_minus_g": vectors["f_minus_g"].tolist(),
        },
        "delta_recurrent_gain": f_gain - g_gain,
        "delta_sequence_gap": f_gap - g_gap,
        "stronger_sequence_specificity": "2D2F" if f_gap > g_gap else "2D2G",
        "bootstrap": bootstrap_rows,
        "position_bins": position_bins,
        "memory": memory,
        "ce_cost_per_mib_saved": delta / memory["f_saving_mib"],
        "checkpoints": {"F": f_checkpoint, "G": g_checkpoint},
        "validation_subset_sha256": f_subset,
        "disjointness_passed": True,
        "archived_prefix_regression": {"F": f_regression, "G": g_regression},
        "mutation_counters": mutation,
        "recommended_next_training_experiment": recommendation,
        "performance": {
            "total_wall_seconds": time.monotonic() - started,
            "hardware": hardware,
            "F": f_performance,
            "G": g_performance,
        },
        "pod": {
            "id": "rvgztsr0azrwyo",
            "name": "happy_apricot_stork",
            "status": "RUNNING",
            "persistent_volume_id": "yhzyb27fb5",
            "persistent_volume_retained": True,
        },
        "artifact_path": str(output),
        "git": {"implementation_commit": preflight["git"]["implementation_commit"], "results_commit": None},
    }
    summary["questions"] = build_questions(summary)
    per_sequence = {
        "sequence_identities": subset["selection"]["sequence_identities"],
        "F_REAL": f_result["real"]["per_sequence_losses"],
        "F_B3_OFF": f_result["off"]["per_sequence_losses"],
        "F_B3_SHUFFLED": f_result["shuffled"]["per_sequence_losses"],
        "G_REAL": g_result["real"]["per_sequence_losses"],
        "G_B3_OFF": g_result["off"]["per_sequence_losses"],
        "G_B3_SHUFFLED": g_result["shuffled"]["per_sequence_losses"],
    }
    atomic_json(output / "per_sequence_losses.json", per_sequence)
    for filename, row in (
        ("f_real.json", f_result["real"]),
        ("f_b3_off.json", f_result["off"]),
        ("f_b3_shuffled.json", f_result["shuffled"]),
        ("g_real.json", g_result["real"]),
        ("g_b3_off.json", g_result["off"]),
        ("g_b3_shuffled.json", g_result["shuffled"]),
    ):
        atomic_json(output / filename, row)
    atomic_json(output / "paired_f_vs_g.json", summary["paired_f_vs_g"])
    atomic_json(output / "recurrent_gain_comparison.json", {"F": {"effect": f_gain, "paired": f_gain_stats}, "G": {"effect": g_gain, "paired": g_gain_stats}, "F_minus_G": f_gain - g_gain, "bootstrap": bootstrap_rows["recurrent_gain_difference"]})
    atomic_json(output / "sequence_gap_comparison.json", {"F": {"effect": f_gap, "paired": f_gap_stats}, "G": {"effect": g_gap, "paired": g_gap_stats}, "F_minus_G": f_gap - g_gap, "bootstrap": bootstrap_rows["sequence_gap_difference"]})
    atomic_json(output / "bootstrap_results.json", bootstrap_rows)
    atomic_json(output / "position_bin_comparison.json", position_bins)
    atomic_json(output / "memory_accounting.json", memory)
    atomic_json(output / "performance.json", summary["performance"])
    atomic_json(output / "result_summary.json", summary)
    commands = {
        "schema": "2d2fg_c1_commands_runtime_v1",
        "argv": sys.argv,
        "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "mutation_counters": mutation,
        "result_bearing_mode": "torch.inference_mode true incremental only",
        "no_optimizer_constructed": True,
        "no_backward_called": True,
        "started_utc": preflight["created_utc"],
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": summary["performance"]["total_wall_seconds"],
    }
    atomic_json(output / "commands_and_runtime.json", commands)
    audit_checks = {
        "F_checkpoint_integrity": f_checkpoint["passed"],
        "G_checkpoint_integrity": g_checkpoint["passed"],
        "same_validation_subset_all_conditions": f_subset == g_subset == subset["selection"]["subset_sha256"],
        "fresh_subset_disjoint": disjointness["passed"],
        "targets_per_control_exact": all(row["validation_targets"] == TARGETS for row in (*f_result.values(), *g_result.values())),
        "paired_sequences_complete": all(len(row["per_sequence_losses"]) == SEQUENCES for row in (*f_result.values(), *g_result.values())),
        "all_true_incremental": all(row["true_incremental"] and not row["complete_prefix_recomputation"] for row in (*f_result.values(), *g_result.values())),
        "all_cache_audits_passed": all(all(all(item["passed"] for item in batch) for batch in row["cache_rows"]) for row in (*f_result.values(), *g_result.values())),
        "F_no_model_change": f_initial == f_final,
        "G_no_model_change": g_initial == g_final,
        "zero_mutation_counters": all(value == 0 for value in mutation.values()),
        "both_archived_prefix_regressions_passed": f_regression["passed"] and g_regression["passed"],
        "bootstrap_reproducible_configuration": all(row["resamples"] == BOOTSTRAP_RESAMPLES for row in bootstrap_rows.values()),
        "memory_accounting_exact": memory["passed"],
        "one_recommended_training_experiment": recommendation.count("2D2J") == 1,
    }
    audit = {
        "schema": "2d2fg_c1_final_audit_v1",
        "experiment": EXPERIMENT,
        "classification": summary["absolute_quality_classification"],
        "checks": audit_checks,
        "passed": all(audit_checks.values()),
    }
    if not audit["passed"]:
        raise SystemExit(f"terminal scientific audit failed: {audit_checks}")
    atomic_json(output / "FINAL_AUDIT.json", audit)
    make_plots(output, summary)
    atomic_text(output / "EXPERIMENT_2D2FG_C1_FINAL_REPORT.md", report_text(summary, audit))
    atomic_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        "# 2D2FG-C1 unattended handoff\n\n"
        f"Scientific evaluation and final audit passed. Classification: **{summary['absolute_quality_classification']}**.\n\n"
        "Commit and push the result artifacts, mirror them locally, verify origin, then stop the exact pod without deleting it or the persistent volume.\n",
    )
    missing = [name for name in REQUIRED_FILES if not (output / name).is_file() or (output / name).stat().st_size == 0]
    if missing:
        raise SystemExit(f"required artifacts missing: {missing}")
    update_heartbeat(output, "scientific_evaluation_complete", passed=True)
    print(output / "EXPERIMENT_2D2FG_C1_FINAL_REPORT.md")


def run_seal(args) -> None:
    require_git(clean=False)
    output = Path(args.output_dir).resolve()
    summary = read_json(output / "result_summary.json")
    audit = read_json(output / "FINAL_AUDIT.json")
    commit = args.results_commit
    if subprocess.call(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO_ROOT):
        raise SystemExit("results commit is not an ancestor of current HEAD")
    summary["git"]["results_commit"] = commit
    audit["results_commit"] = commit
    atomic_json(output / "result_summary.json", summary)
    atomic_json(output / "FINAL_AUDIT.json", audit)
    atomic_text(output / "EXPERIMENT_2D2FG_C1_FINAL_REPORT.md", report_text(summary, audit, commit))
    inventory = {}
    for name in REQUIRED_FILES:
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"required artifact absent at seal: {name}")
        inventory[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    atomic_json(output / "artifact_inventory.json", {"files": inventory, "passed": True})
    update_heartbeat(output, "results_sealed", results_commit=commit, passed=True)
    print(output / "artifact_inventory.json")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", required=True)
    common.add_argument("--validation-shard", required=True)
    common.add_argument("--f-checkpoint", required=True)
    common.add_argument("--g-checkpoint", required=True)
    common.add_argument("--f-architecture", required=True)
    common.add_argument("--g-architecture", required=True)
    common.add_argument("--f-checkpoint-manifest", required=True)
    common.add_argument("--g-checkpoint-manifest", required=True)
    preflight = sub.add_parser("preflight", parents=[common])
    preflight.add_argument("--stop-capability", required=True)
    evaluate = sub.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--f-incremental-archive", required=True)
    evaluate.add_argument("--g-incremental-archive", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--output-dir", required=True)
    seal.add_argument("--results-commit", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "preflight":
        run_preflight(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "seal":
        run_seal(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
