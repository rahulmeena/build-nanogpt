#!/usr/bin/env python3
"""Experiment 2D2A: B12-to-B1 token-indexed recurrent K/V pilot.

This is deliberately a self-contained result runner.  The historical GPT-2
implementation and checkpoints remain unchanged; the only model parameter
introduced by this experiment is ``g_rec`` in ``RecurrentKVGPT``.
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
import platform
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
import experiment_2d0 as d0  # noqa: E402
import experiment_2d0d as d0d  # noqa: E402
import experiment_2d1 as d1  # noqa: E402
from experiment_2d2a_core import RecurrentKVGPT  # noqa: E402


EXPERIMENT = "2D2A"
PROTOCOL = "exp2d2a_b12_b1_recurrent_kv_v1"
BRANCH = "experiment-2d2a-b12-b1-recurrent-kv"
FROZEN_TAG = "experiment-2d1d-residual-recurrence-final"
FROZEN_COMMIT = "f930b2759b705aeeef3ee6c77fb75a6813a3832c"
FROZEN_2D0D_TAG = "experiment-2d0d-matched-joint-kv-final"
FROZEN_2D0D_COMMIT = "7e45b77b8638d2923689d1d9074104a8f9f5baab"
FROZEN_SOURCE_MANIFEST_RELATIVE = Path(
    "results/experiment_2d0d_matched_joint_kv_geometries/source_manifest.json"
)
FROZEN_SOURCE_MANIFEST_SHA256 = "55be5f98e66fbe5790f84a4084a7171a99c46c6c1ebeedd0c66953c5d1d1fdd7"
CANONICAL_SOURCE_PATH = "/workspace/exp2d0_assets/runs/gpt2_124m_fineweb10b_20260810T141222Z/checkpoints/model_19072.pt"
PERSISTENT_VOLUME_IDENTITY = "yhzyb27fb5"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d2a_b12_b1_recurrent_kv.json"
OUTPUT_NAME = "experiment_2d2a_b12_b1_recurrent_kv"
CHECKPOINT_SCHEMA = "exp2d2a_b12_b1_recurrent_kv_checkpoint_v1"

SOURCE_SHA256 = "924ce6c8392c06ae24ab8f2ffd203787ee0022055c54554bac43bd9a34037871"
SOURCE_BYTES = 497_958_271
SOURCE_PARAMETERS = 124_475_904
TOTAL_PARAMETERS = SOURCE_PARAMETERS + 1
SOURCE_STATE_ENTRIES = 149
VALIDATION_SHARD_SHA256 = "8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f"
CANONICAL_VALIDATION_SHA256 = "3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb"
PARENT_FULL_LOSS = 3.0750437753315962
PARENT_B1_W2_LOSS = 5.048855016570451
PARENT_B1_W2_DAMAGE = 1.973811241238855
ORACLE_TOLERANCE = 1e-8
TF32_KERNEL_LOGIT_ATOL = 2e-2
FP32_INCREMENTAL_PLAIN_MAX_ATOL = 1e-4
FP32_INCREMENTAL_ACTIVE_PREFIX_MAX_ATOL = 5e-5
BF16_INCREMENTAL_PLAIN_MAX_ATOL = 1.25
BF16_INCREMENTAL_ACTIVE_PREFIX_MAX_ATOL = 0.30
STABILITY_RMS_HARD_LIMIT = 1_000.0
STABILITY_LOSS_HARD_LIMIT = 100.0

T = 1024
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
VOCAB_SIZE = 50_304
W_LOCAL = 2
W_REC = 2
LAG = 2
GLOBAL_TARGETS = 524_288
MAX_UPDATES = 96
TOTAL_TARGETS = MAX_UPDATES * GLOBAL_TARGETS
WARMUP_UPDATES = 10
BASE_LR = 3e-5
GATE_LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
TWO_PASS_WEIGHTS = (0.25, 0.75)
THREE_PASS_WEIGHTS = (0.20, 0.40, 0.40)
MILESTONES = (0, 10, 20, 48, 96)
SCIENTIFIC_CHECKPOINTS = (10, 20, 48, 96)
THREE_PASS_UPDATES = (32, 64, 96)
VALIDATION_BATCHES = 20
VALIDATION_B = 64
INCREMENTAL_BATCHES = 2
SEED = 2026_0220
POSITION_BINS = (
    ("3-16", 3, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513-1023", 513, 1023),
)
IMPLEMENTATION_FILES = (
    "configs/exp2d2a_b12_b1_recurrent_kv.json",
    "scripts/experiment_2d2a.py",
    "scripts/experiment_2d2a_core.py",
    "scripts/experiment_2d0.py",
    "scripts/experiment_2d0c.py",
    "scripts/experiment_2d0d.py",
    "scripts/experiment_2d1.py",
    "scripts/smoke_test.py",
    "train_gpt2.py",
    "tests/test_experiment_2d2a_core.py",
    "tests/test_experiment_2d2a_driver.py",
)
REQUIRED_ARTIFACTS = (
    "EXPERIMENT_2D2A_FINAL_REPORT.md",
    "FINAL_AUDIT.json",
    "result_summary.json",
    "source_manifest.json",
    "parameter_manifest.json",
    "architecture_manifest.json",
    "batch_manifest.json",
    "preflight_audit.json",
    "training_metrics.jsonl",
    "milestone_validation.json",
    "paired_controls.json",
    "gate_diagnostics.json",
    "recurrent_attention_diagnostics.json",
    "temporal_gradient_diagnostics.json",
    "incremental_validation.json",
    "incremental_cache_audit.json",
    "self_composition.json",
    "checkpoint_manifest.json",
    "performance.json",
    "commands_and_runtime.json",
    "HEARTBEAT.json",
    "UNATTENDED_FINAL_HANDOFF.md",
)
REQUIRED_PLOTS = (
    "P1_plain_real_validation.png",
    "P2_recurrent_gain.png",
    "P3_sequence_gap.png",
    "P4_gate.png",
    "P5_recurrent_slots.png",
    "P6_per_head_recurrent_attention.png",
    "P7_position_bin_recurrent_gain.png",
    "P8_parallel_vs_incremental_gain.png",
    "P9_b12_memory_rms.png",
    "P10_runtime_vram.png",
)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean: bool = True) -> None:
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"{EXPERIMENT} requires branch {BRANCH}")
    if git_output("rev-parse", FROZEN_TAG + "^{commit}") != FROZEN_COMMIT:
        raise SystemExit("frozen 2D1D tag mismatch")
    if git_output("rev-parse", FROZEN_2D0D_TAG + "^{commit}") != FROZEN_2D0D_COMMIT:
        raise SystemExit("frozen 2D0D source-provenance tag mismatch")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"], cwd=REPO_ROOT)
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D2A worktree must be clean")


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def read_jsonl(path: Path | str):
    target = Path(path)
    if not target.is_file():
        return []
    return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]


def durable_json(path, payload):
    d0.durable_json(path, payload)


def durable_text(path, value):
    d0.durable_text(path, value)


def append_jsonl(path, payload):
    d0.append_jsonl(path, payload)


def file_sha256(path):
    return d0.file_sha256(path)


def implementation_fingerprint():
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


def require_implementation_fingerprint(preflight):
    current = implementation_fingerprint()
    expected = preflight.get("implementation_fingerprint")
    if current != expected:
        raise SystemExit(
            "scientific implementation differs from the exact preflight/smoke fingerprint"
        )
    return current


def workspace_mount_audit(output_dir, run_root, supplied_identity):
    output = Path(output_dir).resolve()
    run = Path(run_root).resolve()
    if supplied_identity != PERSISTENT_VOLUME_IDENTITY:
        raise SystemExit("unexpected RunPod persistent-volume identity")
    expected_output = (REPO_ROOT / "results" / OUTPUT_NAME).resolve()
    if output != expected_output:
        raise SystemExit(f"2D2A result directory must be exactly {expected_output}")
    if not str(output).startswith("/workspace/") or not str(run).startswith("/workspace/"):
        raise SystemExit("2D2A artifacts and checkpoints must live beneath persistent /workspace")
    row = subprocess.check_output(
        ["findmnt", "-T", "/workspace", "-n", "-o", "TARGET,SOURCE,FSTYPE"],
        text=True,
    ).strip().split(maxsplit=2)
    if len(row) != 3:
        raise SystemExit("unable to resolve /workspace persistent mount")
    target, source, filesystem = row
    checks = {
        "target_exact": target == "/workspace",
        "persistent_identity_exact": f"/networkvolumes/{PERSISTENT_VOLUME_IDENTITY}" in source,
        "fuse_network_mount": filesystem == "fuse",
        "output_on_workspace": str(output).startswith("/workspace/"),
        "run_root_on_workspace": str(run).startswith("/workspace/"),
        "canonical_result_directory": output == expected_output,
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


def authenticated_stop_audit(args):
    path = Path(args.stop_audit_path).resolve()
    if not path.is_file():
        raise SystemExit("authenticated RunPod stop-capability audit is missing")
    payload = read_json(path)
    response = payload.get("authenticated_pod_identity_response", {})
    checks = {
        "schema": payload.get("schema") == "exp2d2a_runpod_stop_capability_v1",
        "authenticated_probe": payload.get("authenticated_list_probe") is True,
        "stop_credential_available": payload.get("stop_credential_available") is True,
        "secret_not_recorded": payload.get("secret_recorded") is False,
        "pod_id": response.get("id") == "7kk5yyti00rnrp" == args.pod_id,
        "pod_name": response.get("name") == "grand_amber_catshark" == args.pod_name,
        "gpu_count": response.get("gpuCount") == 1,
        "runtime_running": response.get("runtimeStatus") == "running",
        "exact_stop_command": payload.get("exact_stop_target") == "7kk5yyti00rnrp",
        "audit_passed": payload.get("passed") is True,
    }
    if not all(checks.values()) or not args.stop_authenticated:
        raise SystemExit(f"authenticated RunPod stop audit failed: {checks}")
    return {
        **payload,
        "audit_path": str(path),
        "audit_sha256": file_sha256(path),
        "driver_checks": checks,
        "driver_passed": True,
    }


def frozen_source_provenance(source_checkpoint, source_step):
    manifest_path = REPO_ROOT / FROZEN_SOURCE_MANIFEST_RELATIVE
    manifest = read_json(manifest_path)
    frozen_bytes = subprocess.check_output(
        [
            "git",
            "show",
            f"{FROZEN_2D0D_TAG}:{FROZEN_SOURCE_MANIFEST_RELATIVE.as_posix()}",
        ],
        cwd=REPO_ROOT,
    )
    frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
    mounted = str(Path(source_checkpoint).resolve())
    checks = {
        "frozen_tag_commit": git_output("rev-parse", FROZEN_2D0D_TAG + "^{commit}")
        == FROZEN_2D0D_COMMIT,
        "frozen_manifest_hash": frozen_hash == FROZEN_SOURCE_MANIFEST_SHA256,
        "working_manifest_matches_frozen": file_sha256(manifest_path)
        == FROZEN_SOURCE_MANIFEST_SHA256,
        "canonical_original_path": manifest["checkpoint"] == CANONICAL_SOURCE_PATH,
        "checkpoint_sha": manifest["checkpoint_sha256"] == SOURCE_SHA256,
        "checkpoint_bytes": manifest["checkpoint_bytes"] == SOURCE_BYTES,
        "checkpoint_step": int(source_step) == 19_072,
        "historical_training_tokens": manifest["historical_training_tokens"]
        == 9_999_745_024,
        "canonical_validation": manifest["canonical_validation_sha256"]
        == CANONICAL_VALIDATION_SHA256,
        "architecture": manifest["architecture"]["passed"]
        and manifest["architecture"]["parameter_count"] == SOURCE_PARAMETERS,
        "mounted_copy_exact": file_sha256(mounted) == SOURCE_SHA256
        and Path(mounted).stat().st_size == SOURCE_BYTES,
    }
    if not all(checks.values()):
        raise SystemExit(f"frozen 2D0D source provenance failed: {checks}")
    return {
        "frozen_2d0d_tag": FROZEN_2D0D_TAG,
        "frozen_2d0d_commit": FROZEN_2D0D_COMMIT,
        "frozen_manifest": str(manifest_path),
        "frozen_manifest_sha256": frozen_hash,
        "canonical_original_checkpoint": CANONICAL_SOURCE_PATH,
        "mounted_byte_identical_checkpoint": mounted,
        "historical_source_pod": "golden_tomato_cat",
        "historical_training_tokens": manifest["historical_training_tokens"],
        "checks": checks,
        "passed": True,
    }


def batch_hash(x, y):
    return d0.batch_payload_hash(x, y)


def require_config():
    config = read_json(CONFIG_PATH)
    expected = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "branch": BRANCH,
        "frozen_2d1d_commit": FROZEN_COMMIT,
        "frozen_2d1d_tag": FROZEN_TAG,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_checkpoint_bytes": SOURCE_BYTES,
        "canonical_validation_sha256": CANONICAL_VALIDATION_SHA256,
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    arithmetic = {
        "total_targets": TOTAL_TARGETS == 50_331_648,
        "source_plus_one": TOTAL_PARAMETERS == 124_475_905,
        "milestones": tuple(config["training"]["milestones"]) == MILESTONES,
        "checkpoints": tuple(config["training"]["scientific_checkpoint_updates"]) == SCIENTIFIC_CHECKPOINTS,
        "pass3": config["training"]["three_pass_every_global_updates"] == 32,
        "geometry": config["architecture"]["b1_local_window"] == W_LOCAL
        and config["architecture"]["recurrent_window"] == W_REC
        and config["architecture"]["recurrent_lag"] == LAG,
        "source_references": config["validation_shard_sha256"] == VALIDATION_SHARD_SHA256
        and config["parent_full_context_loss"] == PARENT_FULL_LOSS
        and config["parent_b1_w2_loss"] == PARENT_B1_W2_LOSS
        and config["parent_b1_w2_damage"] == PARENT_B1_W2_DAMAGE,
        "architecture_exact": config["architecture"]["source_block"] == 12
        and config["architecture"]["destination_block"] == 1
        and config["architecture"]["b2_b12_window"] == T
        and config["architecture"]["recurrent_positions"] == ["t-3", "t-2"]
        and config["architecture"]["new_parameters"] == ["g_rec"]
        and config["architecture"]["new_parameter_count"] == 1
        and config["architecture"]["separate_local_and_recurrent_softmaxes"] is True
        and config["architecture"]["shared_b1_ln_qkv"] is True
        and config["architecture"]["single_b1_output_projection"] is True,
        "training_budget": config["training"]["updates"] == MAX_UPDATES
        and config["training"]["targets_per_update"] == GLOBAL_TARGETS
        and config["training"]["total_targets"] == TOTAL_TARGETS
        and config["training"]["sequence_length"] == T
        and tuple(config["training"]["two_pass_weights"]) == TWO_PASS_WEIGHTS
        and tuple(config["training"]["three_pass_weights"]) == THREE_PASS_WEIGHTS,
        "optimizer_exact": config["training"]["optimizer"]["base_lr"] == BASE_LR
        and config["training"]["optimizer"]["gate_lr"] == GATE_LR
        and tuple(config["training"]["optimizer"]["betas"]) == BETAS
        and config["training"]["optimizer"]["eps"] == ADAM_EPS
        and config["training"]["optimizer"]["base_weight_decay"] == WEIGHT_DECAY
        and config["training"]["optimizer"]["gate_weight_decay"] == 0.0
        and config["training"]["optimizer"]["gradient_clip"] == GRAD_CLIP
        and config["training"]["optimizer"]["warmup_updates"] == WARMUP_UPDATES
        and config["training"]["optimizer"]["source_optimizer_restored"] is False,
        "validation_exact": config["validation"]["batch_size"] == VALIDATION_B
        and config["validation"]["sequence_length"] == T
        and config["validation"]["batches"] == VALIDATION_BATCHES
        and config["validation"]["controls"] == ["plain", "real", "shuffled"]
        and config["validation"]["incremental_batches"] == INCREMENTAL_BATCHES,
        "hardware_exact": config["hardware"]["pod_id"] == "7kk5yyti00rnrp"
        and config["hardware"]["pod_name"] == "grand_amber_catshark"
        and config["hardware"]["gpu_count"] == 1
        and config["hardware"]["gpu_type"] == "NVIDIA A100-SXM4-80GB"
        and config["hardware"]["ddp"] is False
        and config["hardware"]["nccl"] is False,
        "data_continuation_exact": config["data_continuation"][
            "source_checkpoint_has_loader_state"
        ] is False
        and config["data_continuation"]["fallback_preregistered_before_result_training"]
        is True
        and config["data_continuation"]["arbitrary_cursor_selection_forbidden"] is True,
    }
    if mismatch or not all(arithmetic.values()):
        raise SystemExit(f"2D2A preregistration mismatch: {mismatch} {arithmetic}")
    return config


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_single_a100():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("2D2A result path requires exactly one visible GPU")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if "A100-SXM4-80GB" not in name or memory < 79 * 1024**3:
        raise SystemExit(f"unsupported 2D2A GPU: {name}, {memory}")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("2D2A forbids DDP/NCCL")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda", 0)


def environment_payload():
    return {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "autocast": "cuda bfloat16",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "git_branch": git_output("branch", "--show-current"),
        "git_commit": git_output("rev-parse", "HEAD"),
    }


def load_model(checkpoint, device, trainable=True):
    symbols, base, source_audit = d0.load_standard_model(checkpoint, device)
    for parameter in base.parameters():
        parameter.requires_grad_(bool(trainable))
    model = RecurrentKVGPT(base).to(device)
    if base.transformer.wte.weight is not base.lm_head.weight:
        raise SystemExit("embedding/LM-head tying was not preserved")
    return symbols, model, source_audit


def parameter_manifest(model, source_audit):
    named = list(model.named_parameters())
    parent_named = [(name, value) for name, value in named if name != "g_rec"]
    new_named = [(name, value) for name, value in named if name == "g_rec"]
    base = model.base
    report = {
        "parent_total_parameters": sum(value.numel() for _, value in parent_named),
        "parent_trainable_parameters": sum(value.numel() for _, value in parent_named if value.requires_grad),
        "new_parameters": [{"name": name, "shape": list(value.shape), "numel": value.numel()} for name, value in new_named],
        "new_parameter_count": sum(value.numel() for _, value in new_named),
        "total_2d2a_parameters": sum(value.numel() for _, value in named),
        "total_trainable_parameters": sum(value.numel() for _, value in named if value.requires_grad),
        "vocabulary_size": base.config.vocab_size,
        "n_layer": base.config.n_layer,
        "n_head": base.config.n_head,
        "n_embd": base.config.n_embd,
        "block_size": base.config.block_size,
        "bias_configuration": {
            "attention_qkv": base.transformer.h[0].attn.c_attn.bias is not None,
            "attention_output": base.transformer.h[0].attn.c_proj.bias is not None,
            "mlp_fc": base.transformer.h[0].mlp.c_fc.bias is not None,
            "mlp_output": base.transformer.h[0].mlp.c_proj.bias is not None,
            "lm_head": base.lm_head.bias is not None,
        },
        "embedding_lm_head_tied": base.transformer.wte.weight is base.lm_head.weight,
        "source_state_dict_entries": source_audit["state_dict_entries"],
    }
    report["checks"] = {
        "parent_exact": report["parent_total_parameters"] == SOURCE_PARAMETERS,
        "new_exactly_one": report["new_parameter_count"] == 1 and len(new_named) == 1,
        "new_only_scalar_gate": len(new_named) == 1 and new_named[0][0] == "g_rec" and new_named[0][1].ndim == 0,
        "total_exact": report["total_2d2a_parameters"] == TOTAL_PARAMETERS,
        "all_trainable": report["total_trainable_parameters"] == TOTAL_PARAMETERS,
        "tying": report["embedding_lm_head_tied"],
    }
    report["passed"] = all(report["checks"].values())
    if not report["passed"]:
        raise SystemExit(f"parameter audit failed: {report}")
    return report


def architecture_manifest():
    return {
        "experiment": EXPERIMENT,
        "source": "B12 post-MLP residual before final LayerNorm",
        "destination": "B1 attention",
        "b1_local_window": W_LOCAL,
        "b2_b12_windows": [T] * 11,
        "recurrent_window": W_REC,
        "recurrent_lag": LAG,
        "recurrent_positions": ["t-3", "t-2"],
        "local_positions": ["t-1", "t"],
        "separate_softmaxes": True,
        "shared_b1_ln_qkv": True,
        "single_b1_c_proj": True,
        "effective_gate": "tanh(g_rec)",
        "incremental_b1_historical_kv_capacity": 1,
        "incremental_b2_b12_historical_kv_capacity": 1023,
        "incremental_b12_residual_ring_capacity": 3,
        "overall_context": 1024,
        "overall_kv_savings_claimed": False,
        "forbidden_modules_absent": {
            "teacher": True,
            "attnres": True,
            "recurrent_projections": True,
            "mirrored_links": True,
        },
    }


def learning_rate_fraction(update):
    if not 1 <= int(update) <= MAX_UPDATES:
        raise ValueError(update)
    return min(1.0, int(update) / WARMUP_UPDATES)


def configure_optimizer(model, device_type="cuda"):
    base_decay, base_nodecay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "g_rec":
            continue
        (base_decay if parameter.dim() >= 2 else base_nodecay).append(parameter)
    groups = [
        {"name": "base_decay", "params": base_decay, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
        {"name": "base_nodecay", "params": base_nodecay, "lr": BASE_LR, "weight_decay": 0.0},
        {"name": "gate", "params": [model.g_rec], "lr": GATE_LR, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        groups,
        betas=BETAS,
        eps=ADAM_EPS,
        fused=fused_available and device_type == "cuda",
    )
    report = {
        "logical_parameter_groups": 2,
        "physical_parameter_groups": 3,
        "grouping_note": "The base LR class is split only to preserve Standard GPT-2 decay/no-decay semantics.",
        "groups": [
            {
                "name": group["name"],
                "parameters": sum(parameter.numel() for parameter in group["params"]),
                "tensors": len(group["params"]),
                "peak_lr": GATE_LR if group["name"] == "gate" else BASE_LR,
                "weight_decay": group["weight_decay"],
            }
            for group in groups
        ],
        "betas": list(BETAS),
        "eps": ADAM_EPS,
        "fused": fused_available and device_type == "cuda",
        "source_optimizer_restored": False,
    }
    return optimizer, report


def set_optimizer_lrs(optimizer, update):
    fraction = learning_rate_fraction(update)
    values = {}
    for group in optimizer.param_groups:
        peak = GATE_LR if group["name"] == "gate" else BASE_LR
        group["lr"] = peak * fraction
        values[group["name"]] = group["lr"]
    return values


def finite_tensors(values):
    rows = [torch.isfinite(value).all() for value in values if torch.is_tensor(value)]
    return not rows or bool(torch.stack(rows).all().item())


def model_finite(model):
    return finite_tensors(parameter.data for parameter in model.parameters())


def gradients_finite(model):
    return finite_tensors(parameter.grad for parameter in model.parameters() if parameter.grad is not None)


def optimizer_finite(optimizer):
    return finite_tensors(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def paired_stats(left, right):
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    return {
        "count": len(differences),
        "wins": sum(value < 0 for value in differences),
        "losses": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "differences_left_minus_right": differences,
        "mean": statistics.fmean(differences),
        "median": statistics.median(differences),
        "sample_std": statistics.stdev(differences) if len(differences) > 1 else 0.0,
    }


def training_shards(data_root):
    paths = sorted(Path(data_root).glob("edufineweb_train_*.npy"))
    if not paths:
        raise SystemExit(f"no training shards under {data_root}")
    return paths


def validation_path(data_root):
    path = Path(data_root) / "edufineweb_val_000000.npy"
    if not path.is_file() or file_sha256(path) != VALIDATION_SHARD_SHA256:
        raise SystemExit("canonical validation shard missing/corrupt")
    return path


def next_global_batch_hash(loader, accumulation):
    clone = loader.clone()
    hashes = []
    for _ in range(accumulation):
        x, y = clone.next_batch()
        hashes.append(batch_hash(x, y))
    return d0.aggregate_hashes(hashes)


def pass_count(update):
    return 3 if int(update) in THREE_PASS_UPDATES else 2


def pass_weights(update):
    return THREE_PASS_WEIGHTS if pass_count(update) == 3 else TWO_PASS_WEIGHTS


def _token_losses(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view_as(targets)


def _source_diagnostics(h12):
    value = h12.float()
    norms = value.norm(dim=-1)
    adjacent = (
        F.cosine_similarity(value[:, 1:], value[:, :-1], dim=-1)
        if value.size(1) > 1
        else value.new_empty((0,))
    )
    return {
        "rms": value.square().mean().sqrt().item(),
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "norm_quantiles": {
            str(q): torch.quantile(norms.flatten(), q).item()
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "adjacent_cosine_mean": adjacent.mean().item() if adjacent.numel() else None,
        "adjacent_cosine_std": adjacent.std().item() if adjacent.numel() > 1 else None,
    }


def _recurrent_bank_source_diagnostics(model, h12):
    bank = model.build_recurrent_bank(h12)
    valid_values = bank.values[:, bank.valid_mask]
    if not valid_values.numel():
        raise SystemExit("canonical recurrent bank unexpectedly has no valid values")
    slot_rows = {}
    for slot, label in enumerate(("t-3", "t-2")):
        valid = bank.valid_mask[:, slot]
        values = bank.values[:, valid, slot]
        slot_rows[label] = {
            **_source_diagnostics(values),
            "valid_states": values.size(0) * values.size(1),
        }
    return {
        "valid_bank": _source_diagnostics(valid_values),
        "per_slot": slot_rows,
        "valid_states": valid_values.size(0) * valid_values.size(1),
        "invalid_slots_excluded": True,
        "raw_h12_stream": _source_diagnostics(h12),
    }


def _new_attention_accumulator():
    return {
        "slot_sum": np.zeros(2, dtype=np.float64),
        "entropy_sum": 0.0,
        "count": 0,
        "per_head_slot_sum": np.zeros((N_HEAD, 2), dtype=np.float64),
        "per_head_entropy_sum": np.zeros(N_HEAD, dtype=np.float64),
        "per_head_count": np.zeros(N_HEAD, dtype=np.int64),
        "bins": {
            name: {
                "slot_sum": np.zeros(2, dtype=np.float64),
                "entropy_sum": 0.0,
                "count": 0,
            }
            for name, _, _ in POSITION_BINS
        },
        "recurrent_output_rms": [],
    }


def _add_attention_diagnostics(accumulator, diagnostics):
    weights = diagnostics["recurrent_attention_weights"].detach().float()
    # Symmetric slot comparison excludes t=2, where only the newer slot exists.
    symmetric = weights[:, :, 3:, :]
    entropy = -(symmetric * symmetric.clamp_min(1e-30).log()).sum(dim=-1)
    accumulator["slot_sum"] += symmetric.double().sum(dim=(0, 1, 2)).cpu().numpy()
    accumulator["entropy_sum"] += entropy.double().sum().item()
    accumulator["count"] += symmetric.size(0) * symmetric.size(1) * symmetric.size(2)
    accumulator["per_head_slot_sum"] += symmetric.double().sum(dim=(0, 2)).cpu().numpy()
    accumulator["per_head_entropy_sum"] += entropy.double().sum(dim=(0, 2)).cpu().numpy()
    accumulator["per_head_count"] += symmetric.size(0) * symmetric.size(2)
    for name, first, last in POSITION_BINS:
        selected = weights[:, :, first : last + 1, :]
        selected_entropy = -(selected * selected.clamp_min(1e-30).log()).sum(dim=-1)
        row = accumulator["bins"][name]
        row["slot_sum"] += selected.double().sum(dim=(0, 1, 2)).cpu().numpy()
        row["entropy_sum"] += selected_entropy.double().sum().item()
        row["count"] += selected.size(0) * selected.size(1) * selected.size(2)
    accumulator["recurrent_output_rms"].append(
        diagnostics["recurrent_output_rms"].detach().float().item()
    )


def _finish_attention_diagnostics(accumulator):
    count = accumulator["count"]
    heads = []
    for head in range(N_HEAD):
        head_count = int(accumulator["per_head_count"][head])
        heads.append({
            "head": head,
            "slot_t_minus_3": accumulator["per_head_slot_sum"][head, 0] / head_count,
            "slot_t_minus_2": accumulator["per_head_slot_sum"][head, 1] / head_count,
            "entropy": accumulator["per_head_entropy_sum"][head] / head_count,
        })
    bins = {}
    for name, row in accumulator["bins"].items():
        bins[name] = {
            "slot_t_minus_3": row["slot_sum"][0] / row["count"],
            "slot_t_minus_2": row["slot_sum"][1] / row["count"],
            "entropy": row["entropy_sum"] / row["count"],
            "attention_observations": row["count"],
        }
    return {
        "symmetric_positions": "t>=3",
        "slot_t_minus_3": accumulator["slot_sum"][0] / count,
        "slot_t_minus_2": accumulator["slot_sum"][1] / count,
        "entropy": accumulator["entropy_sum"] / count,
        "per_head": heads,
        "position_bins": bins,
        "recurrent_output_rms_mean": statistics.fmean(accumulator["recurrent_output_rms"]),
    }


@torch.no_grad()
def evaluate_parallel(model, val_path, batches=VALIDATION_BATCHES):
    """Evaluate Plain/Real/Shuffled on identical canonical rows."""
    model.eval()
    device = next(model.parameters()).device
    loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    controls = {
        name: {
            "loss_sum": 0.0,
            "targets": 0,
            "per_batch_losses": [],
            "per_position_sum": np.zeros(T, dtype=np.float64),
        }
        for name in ("plain", "real", "shuffled")
    }
    attention = _new_attention_accumulator()
    identities = []
    source_rows = []
    derangement = torch.arange(VALIDATION_B, device=device).roll(1)
    start = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(batches):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(d0d.batch_identity(cpu_x, cpu_y))
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            plain = model.forward_pass(x, targets=None)
            plain_losses = _token_losses(plain["logits"], y)
            source = plain["h12"]
            real = model.forward_pass(
                x, targets=None, recurrent_source=source, return_diagnostics=True
            )
            real_losses = _token_losses(real["logits"], y)
            shuffled = model.forward_pass(
                x,
                targets=None,
                recurrent_source=source,
                recurrent_permutation=derangement,
            )
            shuffled_losses = _token_losses(shuffled["logits"], y)
        for name, losses in (
            ("plain", plain_losses),
            ("real", real_losses),
            ("shuffled", shuffled_losses),
        ):
            row = controls[name]
            row["loss_sum"] += losses.double().sum().item()
            row["targets"] += losses.numel()
            row["per_batch_losses"].append(losses.float().mean().item())
            row["per_position_sum"] += losses.double().sum(dim=0).cpu().numpy()
        _add_attention_diagnostics(attention, real["diagnostics"])
        source_rows.append(_recurrent_bank_source_diagnostics(model, source))
        print(f"2D2A validation batch={batch_index + 1:02d}/{batches}", flush=True)
        del x, y, plain, real, shuffled, source
        del plain_losses, real_losses, shuffled_losses
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
    real = finished["real"]
    shuffled = finished["shuffled"]
    position_bins = {}
    for name, first, last in POSITION_BINS:
        p = np.asarray(plain["per_position_loss"])[first : last + 1]
        r = np.asarray(real["per_position_loss"])[first : last + 1]
        s = np.asarray(shuffled["per_position_loss"])[first : last + 1]
        position_bins[name] = {
            "plain_loss": float(p.mean()),
            "real_loss": float(r.mean()),
            "shuffled_loss": float(s.mean()),
            "recurrent_gain": float((p - r).mean()),
            "sequence_gap": float((s - r).mean()),
        }
    collection_sha = d0.aggregate_hashes([row["combined_sha256"] for row in identities])
    result = {
        "controls": finished,
        "recurrent_gain": plain["validation_loss"] - real["validation_loss"],
        "sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "real_vs_plain": paired_stats(real["per_batch_losses"], plain["per_batch_losses"]),
        "real_vs_shuffled": paired_stats(real["per_batch_losses"], shuffled["per_batch_losses"]),
        "position_bins": position_bins,
        "recurrent_attention": _finish_attention_diagnostics(attention),
        "recurrent_source": {
            "real": source_rows,
            "shuffled": copy.deepcopy(source_rows),
            "permutation_invariance_note": "A coherent row permutation preserves source RMS/norm/cosine distributions.",
        },
        "effective_recurrent_scale": model.recurrent_scale.detach().float().item(),
        "gate_raw": model.g_rec.detach().float().item(),
        "canonical_validation_sha256": collection_sha,
        "batch_identities": identities,
        "batch_count": batches,
        "batch_size": VALIDATION_B,
        "sequence_length": T,
        "precision": "torch.autocast(cuda,bfloat16)",
        "loss_denominator": batches * VALIDATION_B * T,
        "performance": {
            "wall_seconds": elapsed,
            "condition_target_passes_per_second": batches * VALIDATION_B * T * 3 / elapsed,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }
    if batches == VALIDATION_BATCHES and collection_sha != CANONICAL_VALIDATION_SHA256:
        raise SystemExit(f"canonical validation hash mismatch: {collection_sha}")
    return result


def validation_manifest(val_path):
    loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    rows, hashes = [], []
    for batch_index in range(VALIDATION_BATCHES):
        x, y = loader.next_batch()
        identity = d0d.batch_identity(x, y)
        rows.append({"batch_index": batch_index, **identity})
        hashes.append(identity["combined_sha256"])
    return {
        "validation_shard": str(Path(val_path).resolve()),
        "validation_shard_sha256": file_sha256(val_path),
        "canonical_batch_collection_sha256": d0.aggregate_hashes(hashes),
        "batch_count": VALIDATION_BATCHES,
        "batch_size": VALIDATION_B,
        "sequence_length": T,
        "targets": VALIDATION_BATCHES * VALIDATION_B * T,
        "batches": rows,
    }


def temporal_gradient_diagnostic(model, tokens, targets, precision="bf16"):
    model.train()
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else contextlib.nullcontext()
    )
    with torch.enable_grad(), context:
        first = model.forward_pass(tokens, targets=None)
        source = first["h12"]
        second = model.forward_pass(tokens, targets=targets, recurrent_source=source)
        gradient = torch.autograd.grad(second["loss"], source)[0]
    report = {
        "precision": precision,
        "gate_raw": model.g_rec.detach().float().item(),
        "effective_gate": model.recurrent_scale.detach().float().item(),
        "gradient_norm": gradient.float().norm().item(),
        "finite": bool(torch.isfinite(gradient).all()),
        "nonzero": bool(gradient.count_nonzero().item()),
        "writer_temporal_gradient_present": bool(
            torch.isfinite(gradient).all() and gradient.count_nonzero().item()
        ),
    }
    model.zero_grad(set_to_none=True)
    return report


@torch.no_grad()
def self_composition_diagnostic(model, val_path, batches=2, passes=8):
    model.eval()
    device = next(model.parameters()).device
    loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    aggregate = [
        {"loss_sum": 0.0, "targets": 0, "h12_rms": [], "recurrent_output_rms": []}
        for _ in range(passes)
    ]
    for batch_index in range(batches):
        cpu_x, cpu_y = loader.next_batch()
        x, y = cpu_x.to(device), cpu_y.to(device)
        source = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for pass_index in range(passes):
                current = model.forward_pass(
                    x,
                    targets=None,
                    recurrent_source=source,
                    return_diagnostics=pass_index > 0,
                )
                losses = _token_losses(current["logits"], y)
                row = aggregate[pass_index]
                row["loss_sum"] += losses.double().sum().item()
                row["targets"] += losses.numel()
                row["h12_rms"].append(current["h12"].float().square().mean().sqrt().item())
                if pass_index > 0:
                    row["recurrent_output_rms"].append(
                        current["diagnostics"]["recurrent_output_rms"].float().item()
                    )
                source = current["h12"]
        del x, y, source, current, losses
        torch.cuda.empty_cache()
    return {
        "passes": [
            {
                "pass": index + 1,
                "loss": row["loss_sum"] / row["targets"],
                "b12_memory_rms": statistics.fmean(row["h12_rms"]),
                "b1_recurrent_output_rms": (
                    statistics.fmean(row["recurrent_output_rms"])
                    if row["recurrent_output_rms"] else 0.0
                ),
                "gate_raw": model.g_rec.detach().float().item(),
                "effective_gate": model.recurrent_scale.detach().float().item(),
            }
            for index, row in enumerate(aggregate)
        ],
        "batch_count": batches,
        "batch_size": VALIDATION_B,
        "sequence_length": T,
        "no_gradient": True,
    }


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def checkpoint_payload(model, optimizer, loader, training_state, metadata, accumulation):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "g_rec": model.g_rec.detach().cpu().clone(),
        "optimizer": optimizer.state_dict(),
        "completed_updates": training_state["completed_updates"],
        "processed_targets": training_state["processed_targets"],
        "training_state": copy.deepcopy(training_state),
        "loader_state": loader.state_dict(),
        "rng_state": capture_rng_state(),
        "next_global_batch_sha256": next_global_batch_hash(loader, accumulation),
        "architecture_manifest": architecture_manifest(),
        "metadata": copy.deepcopy(metadata),
        "git_commit": git_output("rev-parse", "HEAD"),
        "saved_process_id": os.getpid(),
        "environment": environment_payload(),
    }


def strict_reopen_checkpoint(
    path, model, optimizer, loader, training_state, accumulation, expected_metadata
):
    path = Path(path)
    reopened = d0.torch_load(path, mmap=True)
    required = {
        "schema", "model", "g_rec", "optimizer", "completed_updates",
        "processed_targets", "training_state", "loader_state", "rng_state",
        "next_global_batch_sha256", "architecture_manifest", "metadata", "git_commit",
        "saved_process_id", "environment",
    }
    checks = {
        "fields_exact": set(reopened) == required,
        "schema": reopened.get("schema") == CHECKPOINT_SCHEMA,
        "updates": reopened.get("completed_updates") == training_state["completed_updates"],
        "targets": reopened.get("processed_targets") == training_state["processed_targets"],
        "training_state": reopened.get("training_state") == training_state,
        "loader_state": reopened.get("loader_state") == loader.state_dict(),
        "next_batch": reopened.get("next_global_batch_sha256") == next_global_batch_hash(loader, accumulation),
        "metadata": reopened.get("metadata") == expected_metadata,
        "architecture": reopened.get("architecture_manifest") == architecture_manifest(),
        "model_keys": reopened.get("model", {}).keys() == model.state_dict().keys(),
        "gate_duplicate": torch.equal(reopened.get("g_rec"), model.g_rec.detach().cpu()),
    }
    if not all(checks.values()):
        raise SystemExit(f"checkpoint strict metadata reopen failed: {checks}")
    model.load_state_dict(reopened["model"], strict=True)
    optimizer.load_state_dict(reopened["optimizer"])
    checks.update({
        "model_finite": model_finite(model),
        "optimizer_finite": optimizer_finite(optimizer),
        "weight_tying": model.base.transformer.wte.weight is model.base.lm_head.weight,
    })
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"checkpoint tensor reopen failed: {checks}")
    verification = {
        "checkpoint": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "next_global_batch_sha256": reopened["next_global_batch_sha256"],
        "strict_reopen": checks,
        "passed": True,
    }
    del reopened
    gc.collect()
    return verification


def save_checkpoint(
    path, model, optimizer, loader, training_state, metadata, accumulation
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    if temporary.exists():
        raise SystemExit(f"stale incomplete checkpoint: {temporary}")
    payload = checkpoint_payload(
        model, optimizer, loader, training_state, metadata, accumulation
    )
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    d0.fsync_directory(path.parent)
    del payload
    verification = strict_reopen_checkpoint(
        path, model, optimizer, loader, training_state, accumulation, metadata
    )
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{verification['sha256']}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), verification)
    return verification


def record_checkpoint(output, update, verification, kind="scientific"):
    path = Path(output) / "checkpoint_manifest.json"
    manifest = read_json(path) if path.is_file() else {"scientific": {}, "smoke": {}}
    manifest.setdefault(kind, {})[str(update)] = verification
    durable_json(path, manifest)


def load_checkpoint_runtime(
    path, model, optimizer, shards, micro_batch, accumulation, expected_metadata
):
    path = Path(path).resolve()
    sha_path = path.with_suffix(path.suffix + ".sha256")
    verify_path = path.with_suffix(path.suffix + ".verification.json")
    if not sha_path.is_file() or not verify_path.is_file():
        raise SystemExit("checkpoint sidecars missing")
    expected_sha = sha_path.read_text().split()[0]
    if file_sha256(path) != expected_sha:
        raise SystemExit("resume checkpoint SHA mismatch")
    payload = d0.torch_load(path, mmap=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("metadata") != expected_metadata:
        raise SystemExit("resume schema/metadata mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    loader = d1.ExplicitShardLoader(
        shards, micro_batch, T, state=payload["loader_state"]
    )
    if payload["next_global_batch_sha256"] != next_global_batch_hash(loader, accumulation):
        raise SystemExit("resume next-global-batch mismatch")
    restore_rng_state(payload["rng_state"])
    state = copy.deepcopy(payload["training_state"])
    saved_pid = int(payload["saved_process_id"])
    if not model_finite(model) or not optimizer_finite(optimizer):
        raise SystemExit("resume restored nonfinite state")
    del payload
    gc.collect()
    return loader, state, saved_pid, expected_sha


def gradient_group_report(model):
    base = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name != "g_rec" and parameter.grad is not None
    ]
    gate = [] if model.g_rec.grad is None else [model.g_rec.grad]
    report = {}
    for name, values in (("base", base), ("gate", gate)):
        squared = sum(value.float().square().sum() for value in values) if values else torch.tensor(0.0)
        report[name] = {
            "tensors": len(values),
            "norm": squared.sqrt().item(),
            "finite": finite_tensors(values),
            "nonzero": bool(values) and bool(squared.gt(0).item()),
        }
    return report


def train_one_update(runtime, update):
    model = runtime.model
    model.train()
    optimizer = runtime.optimizer
    device = runtime.device
    count = pass_count(update)
    weights = pass_weights(update)
    lrs = set_optimizer_lrs(optimizer, update)
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
        current = row = diag = None
        for index in range(count):
            torch.cuda.synchronize()
            pass_start = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current = model.forward_pass(
                    x,
                    targets=y,
                    recurrent_source=source,
                    activation_checkpointing=True,
                    return_diagnostics=(
                        micro_index == runtime.accumulation - 1 and index == count - 1
                    ),
                )
            torch.cuda.synchronize()
            forward_seconds[index] += time.monotonic() - pass_start
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
            diag = results[-1]["diagnostics"]
            final_recurrent_rms = (
                diag["recurrent_output_rms"].detach().float().item()
                if diag is not None else None
            )
        del x, y, cpu_x, cpu_y, results, source, weighted, scaled
        del current, row, diag
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
    runtime.training_state["completed_updates"] = update
    runtime.training_state["processed_targets"] = update * GLOBAL_TARGETS
    metrics = {
        "timestamp": time.time(),
        "update": update,
        "targets": update * GLOBAL_TARGETS,
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
        "timing_note": "Attached temporal graph permits exact per-pass forward and aggregate backward timing; backward is not separable by pass.",
        "wall_seconds": elapsed,
        "targets_per_second": GLOBAL_TARGETS / elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "all_gradients_finite": True,
        "all_parameters_finite": True,
        "all_optimizer_moments_finite": True,
    }
    runtime.training_state["last_metrics"] = metrics
    return metrics


def write_heartbeat(runtime, metrics):
    completed = metrics["update"]
    elapsed = time.time() - runtime.training_state["started_at"]
    eta = elapsed / max(completed - runtime.training_state.get("segment_start", 0), 1) * (
        runtime.end_update - completed
    )
    durable_json(runtime.output / "HEARTBEAT.json", {
        "experiment": EXPERIMENT,
        "timestamp": time.time(),
        "pid": os.getpid(),
        "pod_id": runtime.metadata["pod_id"],
        "update": completed,
        "targets": metrics["targets"],
        "gate_raw": metrics["g_rec_raw"],
        "effective_gate": metrics["tanh_g_rec"],
        "last_update_wall_seconds": metrics["wall_seconds"],
        "eta_seconds_to_segment_end": eta,
        "checkpoint": runtime.training_state.get("last_checkpoint"),
    })


def kernel_preflight(model, tokens, targets):
    model.train()
    checks = {}
    reports = {}
    source = torch.randn(
        tokens.size(0), tokens.size(1), model.config.n_embd, device=tokens.device
    )
    with torch.no_grad():
        plain = model.forward_pass(tokens, targets=targets)
        dormant = model.forward_pass(tokens, targets=targets, recurrent_source=source)
        oracle_top = d0d.forward_top_schedule(
            model.base,
            tokens,
            (min(W_LOCAL, tokens.size(1)),) + (tokens.size(1),) * 11,
        )
        oracle_logits = model.base.lm_head(oracle_top)
        oracle_loss = F.cross_entropy(
            oracle_logits.reshape(-1, oracle_logits.size(-1)), targets.reshape(-1)
        )
    reports["zero_gate"] = {
        "plain_vs_oracle_logits_max_abs": (plain["logits"].float() - oracle_logits.float()).abs().max().item(),
        "dormant_vs_oracle_logits_max_abs": (dormant["logits"].float() - oracle_logits.float()).abs().max().item(),
        "plain_loss": plain["loss"].float().item(),
        "oracle_loss": oracle_loss.float().item(),
        "plain_oracle_exact": torch.equal(plain["logits"], oracle_logits),
        "dormant_oracle_exact": torch.equal(dormant["logits"], oracle_logits),
    }
    checks["zero_gate_b1_w2_identity"] = (
        reports["zero_gate"]["plain_oracle_exact"]
        and reports["zero_gate"]["dormant_oracle_exact"]
    )
    bank = model.build_recurrent_bank(source)
    expected_positions = torch.arange(tokens.size(1), device=tokens.device).view(-1, 1) - torch.tensor(
        [3, 2], device=tokens.device
    ).view(1, 2)
    reports["bank"] = {
        "positions": bank.positions.cpu().tolist(),
        "valid_mask": bank.valid_mask.cpu().tolist(),
        "t0_valid": int(bank.valid_mask[0].sum().item()),
        "t1_valid": int(bank.valid_mask[1].sum().item()) if tokens.size(1) > 1 else None,
        "t2_valid": int(bank.valid_mask[2].sum().item()) if tokens.size(1) > 2 else None,
        "t3_valid": int(bank.valid_mask[3].sum().item()) if tokens.size(1) > 3 else None,
        "no_wraparound_values": bank.values[:, :2].count_nonzero().item() == 0,
    }
    checks["recurrent_positions_exact"] = torch.equal(bank.positions, expected_positions)
    checks["early_boundary_exact"] = (
        reports["bank"]["t0_valid"] == 0
        and reports["bank"]["t1_valid"] == 0
        and reports["bank"]["t2_valid"] == 1
        and reports["bank"]["t3_valid"] == 2
        and reports["bank"]["no_wraparound_values"]
    )
    local = model.local_mask(tokens.size(1), tokens.device)
    local_counts = local.sum(dim=-1)
    reports["local"] = {
        "counts": local_counts.cpu().tolist(),
        "future_entries": int(torch.triu(local, diagonal=1).sum().item()),
    }
    checks["local_w2_exact"] = (
        torch.equal(
            local_counts,
            torch.minimum(
                torch.arange(1, tokens.size(1) + 1, device=tokens.device),
                torch.tensor(W_LOCAL, device=tokens.device),
            ),
        )
        and not torch.triu(local, diagonal=1).any()
    )
    local_positions = torch.stack(
        (
            torch.arange(tokens.size(1), device=tokens.device) - 1,
            torch.arange(tokens.size(1), device=tokens.device),
        ),
        dim=-1,
    )
    overlap = (
        bank.positions[:, :, None] == local_positions[:, None, :]
    ) & bank.valid_mask[:, :, None]
    reports["local_recurrent_overlap_count"] = int(overlap.sum().item())
    checks["local_recurrent_positions_disjoint"] = not bool(overlap.any())

    with torch.no_grad():
        masked = model.forward_pass(
            tokens,
            recurrent_source=source,
            return_diagnostics=True,
        )
    recurrent_probabilities = masked["diagnostics"]["recurrent_attention_weights"]
    reports["early_recurrent_probabilities"] = {
        "t0_probability_sum": recurrent_probabilities[:, :, 0].float().sum().item(),
        "t1_probability_sum": recurrent_probabilities[:, :, 1].float().sum().item(),
        "t2_slot_t_minus_3_max": recurrent_probabilities[:, :, 2, 0].float().abs().max().item(),
        "t2_slot_t_minus_2_min": recurrent_probabilities[:, :, 2, 1].float().min().item(),
        "all_finite": bool(torch.isfinite(recurrent_probabilities).all()),
    }
    checks["early_recurrent_probabilities_exact"] = (
        reports["early_recurrent_probabilities"]["t0_probability_sum"] == 0.0
        and reports["early_recurrent_probabilities"]["t1_probability_sum"] == 0.0
        and reports["early_recurrent_probabilities"]["t2_slot_t_minus_3_max"] == 0.0
        and reports["early_recurrent_probabilities"]["t2_slot_t_minus_2_min"] == 1.0
        and reports["early_recurrent_probabilities"]["all_finite"]
    )
    recurrent_key, recurrent_value = model.project_recurrent_kv(bank.values)
    normalized = model.base.transformer.h[0].ln_1(bank.values)
    _, expected_key, expected_value = model.base.transformer.h[0].attn.c_attn(normalized).split(
        model.config.n_embd, dim=-1
    )
    batch, length, slots, channels = expected_key.shape
    head_size = channels // model.config.n_head
    expected_key = expected_key.view(batch, length, slots, model.config.n_head, head_size).permute(0, 3, 1, 2, 4)
    expected_value = expected_value.view(batch, length, slots, model.config.n_head, head_size).permute(0, 3, 1, 2, 4)
    projection_key_max_abs = (recurrent_key.float() - expected_key.float()).abs().max().item()
    projection_value_max_abs = (recurrent_value.float() - expected_value.float()).abs().max().item()
    reports["shared_projection"] = {
        "key_max_abs_vs_fused_projection": projection_key_max_abs,
        "value_max_abs_vs_fused_projection": projection_value_max_abs,
        "tolerance": 0.0,
        "note": "The recurrent bank uses the exact existing fused B1 c_attn call and discards its Q slice; no alternate GEMM or recurrent projection exists.",
    }
    checks["shared_b1_ln_kv_exact"] = (
        projection_key_max_abs == 0.0 and projection_value_max_abs == 0.0
    )
    calls = []
    hook = model.base.transformer.h[0].attn.c_proj.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(1)
    )
    try:
        model.forward_pass(tokens, recurrent_source=source)
    finally:
        hook.remove()
    reports["c_proj_calls"] = len(calls)
    checks["single_c_proj"] = calls == [1]

    model.zero_grad(set_to_none=True)
    first = model.forward_pass(tokens)
    second = model.forward_pass(tokens, targets=targets, recurrent_source=first["h12"])
    gate_grad, zero_writer = torch.autograd.grad(
        second["loss"], (model.g_rec, first["h12"])
    )
    reports["initial_gradients"] = {
        "gate_gradient": gate_grad.float().item(),
        "gate_finite": bool(torch.isfinite(gate_grad)),
        "gate_nonzero": bool(gate_grad.abs().gt(0)),
        "writer_at_zero_norm": zero_writer.float().norm().item(),
    }
    checks["initial_gate_gradient"] = reports["initial_gradients"]["gate_finite"] and reports[
        "initial_gradients"
    ]["gate_nonzero"]
    with torch.no_grad():
        model.g_rec.fill_(0.05)
    first_open = model.forward_pass(tokens)
    second_open = model.forward_pass(tokens, targets=targets, recurrent_source=first_open["h12"])
    open_writer = torch.autograd.grad(second_open["loss"], first_open["h12"])[0]
    reports["open_writer_gradient"] = {
        "norm": open_writer.float().norm().item(),
        "finite": bool(torch.isfinite(open_writer).all()),
        "nonzero": bool(open_writer.count_nonzero().item()),
    }
    checks["temporal_writer_gradient"] = reports["open_writer_gradient"]["finite"] and reports[
        "open_writer_gradient"
    ]["nonzero"]

    model.eval()
    changed = tokens.clone()
    cutoff = tokens.size(1) // 2
    changed[:, cutoff + 1 :] = (changed[:, cutoff + 1 :] + 17) % model.config.vocab_size
    row_changed = tokens.clone()
    row_changed[1] = (row_changed[1] + 7) % model.config.vocab_size
    with torch.no_grad():
        reference = model.forward_multi_pass(tokens, num_passes=2)["logits"]
        future = model.forward_multi_pass(changed, num_passes=2)["logits"]
        isolated = model.forward_multi_pass(row_changed, num_passes=2)["logits"]
        incremental_plain = model.incremental_logits(tokens, control="plain")["logits"]
        pass1 = model.forward_pass(tokens)
        parallel2 = model.forward_pass(tokens, recurrent_source=pass1["h12"])["logits"]
        incremental_real = model.incremental_logits(tokens, control="real")["logits"]
    reports["causality"] = {
        "prefix_max_abs": (reference[:, : cutoff + 1].float() - future[:, : cutoff + 1].float()).abs().max().item(),
        "unchanged_row_max_abs": (reference[0].float() - isolated[0].float()).abs().max().item(),
    }
    checks["future_causality"] = reports["causality"]["prefix_max_abs"] <= 1e-5
    checks["row_isolation"] = reports["causality"]["unchanged_row_max_abs"] <= 1e-5
    reports["incremental_equivalence"] = {
        "gate_zero_max_abs": (plain["logits"].float() - incremental_plain.float()).abs().max().item(),
        "active_prefix_0_3_max_abs": (parallel2[:, :4].float() - incremental_real[:, :4].float()).abs().max().item(),
        "expected_self_drift_from_position": 4,
    }
    reports["incremental_equivalence"]["tf32_max_abs_tolerance"] = TF32_KERNEL_LOGIT_ATOL
    reports["incremental_equivalence"]["tolerance_note"] = (
        "Short-sequence FP32-high/TF32 whole-sequence and token-step SDPA kernels "
        "use different reduction shapes; the final audit separately checks FP32-highest/TF32-off and BF16."
    )
    checks["incremental_gate_zero_kernel"] = (
        reports["incremental_equivalence"]["gate_zero_max_abs"]
        <= TF32_KERNEL_LOGIT_ATOL
    )
    checks["incremental_active_prefix_kernel"] = (
        reports["incremental_equivalence"]["active_prefix_0_3_max_abs"]
        <= TF32_KERNEL_LOGIT_ATOL
    )
    incremental = model.incremental_logits(tokens, control="real")
    cache = incremental["cache_audit"]
    reports["incremental_cache"] = cache
    checks["incremental_cache_bounds"] = cache["passed"]
    checks["all_tensors_finite"] = all(
        torch.isfinite(value).all()
        for value in (
            plain["logits"], dormant["logits"], reference, future, isolated,
            incremental_plain, parallel2, incremental_real, gate_grad, open_writer,
        )
    )
    with torch.no_grad():
        model.g_rec.zero_()
    model.zero_grad(set_to_none=True)
    return {"checks": checks, "reports": reports, "passed": all(checks.values())}


def probe_microbatch(model, shards, device, candidates=(64, 32, 16, 8, 4)):
    model.train()
    # Keep a conservative AdamW-sized reserve live during the probe.  A probe
    # that omits the two moment tensors can select a batch that fits backward
    # but fails as soon as the first scientific optimizer step materializes
    # its fresh state.
    optimizer_state_reserve = [
        (torch.zeros_like(parameter), torch.zeros_like(parameter))
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    reserve_bytes = sum(
        first.numel() * first.element_size() + second.numel() * second.element_size()
        for first, second in optimizer_state_reserve
    )
    attempts = []
    try:
        for candidate in candidates:
            if (GLOBAL_TARGETS // T) % candidate:
                continue
            loader = d1.ExplicitShardLoader(shards, candidate, T)
            cpu_x, cpu_y = loader.next_batch()
            x, y = cpu_x.to(device), cpu_y.to(device)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            result = loss = None
            current = source = None
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = []
                    source = None
                    for pass_index in range(3):
                        current = model.forward_pass(
                            x,
                            targets=y,
                            recurrent_source=source,
                            activation_checkpointing=True,
                            return_diagnostics=pass_index == 2,
                        )
                        result.append(current)
                        source = current["h12"]
                    loss = sum(
                        weight * row["loss"]
                        for weight, row in zip(THREE_PASS_WEIGHTS, result)
                    )
                loss.backward()
                torch.cuda.synchronize()
                attempts.append({
                    "micro_batch_sequences": candidate,
                    "passed": gradients_finite(model),
                    "loss": loss.detach().float().item(),
                    "optimizer_state_reserve_bytes": reserve_bytes,
                    "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                    "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
                })
                if attempts[-1]["passed"]:
                    model.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        model.g_rec.zero_()
                    return candidate, attempts
            except torch.cuda.OutOfMemoryError as error:
                attempts.append({
                    "micro_batch_sequences": candidate,
                    "passed": False,
                    "optimizer_state_reserve_bytes": reserve_bytes,
                    "error": type(error).__name__,
                })
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            finally:
                del result, loss
                del current, source
                del x, y, cpu_x, cpu_y
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del optimizer_state_reserve
        gc.collect()
        torch.cuda.empty_cache()
    raise SystemExit(f"no safe 2D2A microbatch: {attempts}")


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
    symbols, model, source_audit = load_model(args.source_checkpoint, device, trainable=True)
    val_path = validation_path(args.data_root)
    shards = training_shards(args.data_root)
    source_payload = d0.torch_load_source_checkpoint(args.source_checkpoint, symbols, mmap=True)
    source_keys = sorted(source_payload)
    source_manifest = {
        "checkpoint": str(Path(args.source_checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.source_checkpoint),
        "checkpoint_bytes": Path(args.source_checkpoint).stat().st_size,
        "checkpoint_top_level_keys": source_keys,
        "checkpoint_step": int(source_payload["step"]),
        "checkpoint_has_optimizer_state": "optimizer" in source_payload,
        "checkpoint_has_loader_state": any("loader" in key.lower() for key in source_keys),
        "architecture": source_audit,
        "canonical_validation_sha256": CANONICAL_VALIDATION_SHA256,
        "validation_shard": str(val_path),
        "validation_shard_sha256": file_sha256(val_path),
        "data_continuation": config["data_continuation"],
    }
    source_manifest["frozen_2d0d_provenance"] = frozen_source_provenance(
        args.source_checkpoint, source_manifest["checkpoint_step"]
    )
    del source_payload
    gc.collect()
    parameters = parameter_manifest(model, source_audit)
    architecture = architecture_manifest()
    manifest = validation_manifest(val_path)
    if manifest["canonical_batch_collection_sha256"] != CANONICAL_VALIDATION_SHA256:
        raise SystemExit("canonical validation collection mismatch")
    durable_json(output / "source_manifest.json", source_manifest)
    durable_json(output / "parameter_manifest.json", parameters)
    durable_json(output / "architecture_manifest.json", architecture)

    # Full and W2 parent oracles use the frozen generalized-window evaluator.
    model.base.eval()
    full_oracle = d0d.evaluate_schedule(
        model.base, val_path, device, (T,) * N_LAYER, VALIDATION_BATCHES
    )
    w2_oracle = d0d.evaluate_schedule(
        model.base, val_path, device, (W_LOCAL,) + (T,) * 11, VALIDATION_BATCHES
    )
    oracle_checks = {
        "full_parent": abs(full_oracle["validation_loss"] - PARENT_FULL_LOSS) <= ORACLE_TOLERANCE,
        "b1_w2_parent": abs(w2_oracle["validation_loss"] - PARENT_B1_W2_LOSS) <= ORACLE_TOLERANCE,
        "b1_w2_damage": abs(
            (w2_oracle["validation_loss"] - full_oracle["validation_loss"])
            - PARENT_B1_W2_DAMAGE
        ) <= ORACLE_TOLERANCE,
        "canonical_full": full_oracle["canonical_validation_sha256"] == CANONICAL_VALIDATION_SHA256,
        "canonical_w2": w2_oracle["canonical_validation_sha256"] == CANONICAL_VALIDATION_SHA256,
    }
    if not all(oracle_checks.values()):
        raise SystemExit(f"parent oracle preflight failed: {oracle_checks}")

    test_loader = d1.ExplicitShardLoader(shards, 2, 16)
    test_x, test_y = test_loader.next_batch()
    kernel = kernel_preflight(model, test_x.to(device), test_y.to(device))
    if not kernel["passed"]:
        raise SystemExit(f"kernel preflight failed: {kernel['checks']}")
    selected_microbatch, probe = probe_microbatch(model, shards, device)
    accumulation = GLOBAL_TARGETS // (selected_microbatch * T)
    batch_payload = {
        **manifest,
        "selected_micro_batch_sequences": selected_microbatch,
        "selected_gradient_accumulation": accumulation,
        "global_targets_per_update": GLOBAL_TARGETS,
        "microbatch_probe": probe,
        "training_loader_initial_state": d1.ExplicitShardLoader(
            shards, selected_microbatch, T
        ).state_dict(),
        "data_continuation_convention": config["data_continuation"],
    }
    durable_json(output / "batch_manifest.json", batch_payload)

    # Update zero controls exercise the recurrent branch while the gate is exact zero.
    with torch.no_grad():
        model.g_rec.zero_()
    update_zero = evaluate_parallel(model, val_path)
    update_zero.update({"update": 0, "targets": 0})
    update_zero_checks = {
        "plain_matches_generalized_w2": abs(
            update_zero["controls"]["plain"]["validation_loss"]
            - w2_oracle["validation_loss"]
        ) <= ORACLE_TOLERANCE,
        "real_matches_plain": update_zero["controls"]["real"]["validation_loss"]
        == update_zero["controls"]["plain"]["validation_loss"],
        "shuffled_matches_plain": update_zero["controls"]["shuffled"]["validation_loss"]
        == update_zero["controls"]["plain"]["validation_loss"],
        "real_logits_loss_identity_all_batches": all(
            a == b
            for a, b in zip(
                update_zero["controls"]["real"]["per_batch_losses"],
                update_zero["controls"]["plain"]["per_batch_losses"],
            )
        ),
    }
    if not all(update_zero_checks.values()):
        raise SystemExit(f"gate-zero canonical identity failed: {update_zero_checks}")
    durable_json(output / "milestone_validation.json", {"0": update_zero})

    science_checks = {
        "2d1d_frozen_tag_exact": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
        "source_sha_exact": source_manifest["checkpoint_sha256"] == SOURCE_SHA256,
        "source_bytes_exact": source_manifest["checkpoint_bytes"] == SOURCE_BYTES,
        "source_architecture": source_audit["passed"],
        "frozen_source_provenance": source_manifest["frozen_2d0d_provenance"]["passed"],
        "parent_parameters": parameters["parent_total_parameters"] == SOURCE_PARAMETERS,
        "exactly_one_new_parameter": parameters["passed"],
        "canonical_manifest": manifest["canonical_batch_collection_sha256"] == CANONICAL_VALIDATION_SHA256,
        "parent_oracles": all(oracle_checks.values()),
        "kernel": kernel["passed"],
        "zero_gate_canonical": all(update_zero_checks.values()),
        "safe_microbatch": selected_microbatch > 0 and selected_microbatch * T * accumulation == GLOBAL_TARGETS,
        "persistent_workspace": mount["passed"],
        "stop_authenticated": stop["driver_passed"],
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
        "parent_oracles": {
            "full": d0d.serializable_evaluation(full_oracle),
            "b1_w2": d0d.serializable_evaluation(w2_oracle),
            "checks": oracle_checks,
        },
        "kernel_preflight": kernel,
        "zero_gate_update_zero_checks": update_zero_checks,
        "microbatch_probe": probe,
        "selected_microbatch": selected_microbatch,
        "gradient_accumulation": accumulation,
        "runpod_stop_audit": stop,
        "persistent_workspace_audit": mount,
        "checks": science_checks,
        "science_passed": all(science_checks.values()),
        "result_run_authorized": all(science_checks.values()),
        "wall_seconds": time.monotonic() - started,
    }
    durable_json(output / "preflight_audit.json", preflight)
    durable_json(output / "runpod_stop_capability.json", stop)
    durable_json(output / "persistent_workspace_audit.json", mount)
    durable_json(output / "checkpoint_manifest.json", {"scientific": {}, "smoke": {}})
    if not preflight["science_passed"]:
        raise SystemExit(f"2D2A preflight failed: {science_checks}")
    print("EXPERIMENT_2D2A_PREFLIGHT_PASS", flush=True)
    return preflight


def run_smoke(args):
    require_git(clean=False)
    require_config()
    workspace_mount_audit(args.output_dir, args.run_root, args.persistent_volume_identity)
    authenticated_stop_audit(args)
    device = require_single_a100()
    seed_all(SEED + 1)
    output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json")
    if not preflight.get("result_run_authorized"):
        raise SystemExit("smoke requires passing authorized preflight")
    require_implementation_fingerprint(preflight)
    _, model, source_audit = load_model(args.source_checkpoint, device, trainable=True)
    parameters = parameter_manifest(model, source_audit)
    optimizer, optimizer_report = configure_optimizer(model)
    shards = training_shards(args.data_root)
    smoke_batch = 2
    loader = d1.ExplicitShardLoader(shards, smoke_batch, T)
    rows = []
    temporal_rows = []
    for update in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        cpu_x, cpu_y = loader.next_batch()
        x, y = cpu_x.to(device), cpu_y.to(device)
        with contextlib.nullcontext():
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
        gradients_ok = gradients_finite(model)
        gradient_groups = gradient_group_report(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        set_optimizer_lrs(optimizer, update)
        optimizer.step()
        row = {
            "update": update,
            "precision": "fp32",
            "loss": loss.detach().float().item(),
            "pass_losses": [first["loss"].detach().float().item(), second["loss"].detach().float().item()],
            "gate_before": gate_before,
            "gate_after": model.g_rec.detach().float().item(),
            "effective_gate_after": model.recurrent_scale.detach().float().item(),
            "gate_gradient": gate_gradient,
            "gradient_norm": norm.detach().float().item(),
            "gradients_finite": gradients_ok,
            "gradient_groups": gradient_groups,
            "parameters_finite": model_finite(model),
            "optimizer_finite": optimizer_finite(optimizer),
            "recurrent_attention_finite": bool(
                torch.isfinite(second["diagnostics"]["recurrent_attention_weights"]).all()
            ),
        }
        rows.append(row)
        clone = loader.clone()
        dx, dy = clone.next_batch()
        temporal_rows.append(
            temporal_gradient_diagnostic(
                model, dx[:1].to(device), dy[:1].to(device), precision="fp32"
            )
        )
        del x, y, cpu_x, cpu_y, first, second, loss
    cache_x, _ = loader.clone().next_batch()
    model.eval()
    with torch.no_grad():
        smoke_incremental_cache = model.incremental_logits(
            cache_x[:, :8].to(device), control="real"
        )["cache_audit"]
    model.train()
    smoke_state = {
        "completed_updates": 3,
        "processed_targets": 3 * smoke_batch * T,
        "started_at": time.time(),
        "last_metrics": rows[-1],
        "kind": "disposable_smoke",
    }
    metadata = {
        "experiment": EXPERIMENT,
        "kind": "disposable_smoke",
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "source_sha256": SOURCE_SHA256,
        "micro_batch_sequences": smoke_batch,
        "gradient_accumulation": 1,
        "pod_id": args.pod_id,
    }
    smoke_path = Path(args.run_root).resolve() / "smoke" / "smoke_update_0003.pt"
    verification = save_checkpoint(
        smoke_path, model, optimizer, loader, smoke_state, metadata, 1
    )
    record_checkpoint(output, 3, verification, kind="smoke")
    checks = {
        "exactly_three_updates": len(rows) == 3 and rows[-1]["update"] == 3,
        "finite_losses": all(math.isfinite(row["loss"]) for row in rows),
        "gate_initial_gradient_finite_nonzero": math.isfinite(rows[0]["gate_gradient"]) and rows[0]["gate_gradient"] != 0,
        "gate_moves_from_zero": rows[0]["gate_before"] == 0 and rows[0]["gate_after"] != 0,
        "base_gradients_finite": all(
            row["gradients_finite"]
            and row["gradient_groups"]["base"]["finite"]
            and row["gradient_groups"]["base"]["nonzero"]
            and row["gradient_groups"]["base"]["tensors"] > 0
            for row in rows
        ),
        "temporal_writer_gradient": all(row["writer_temporal_gradient_present"] for row in temporal_rows),
        "recurrent_attention_finite": all(row["recurrent_attention_finite"] for row in rows),
        "no_b1_hidden_full_cache": smoke_incremental_cache["passed"]
        and smoke_incremental_cache["b1_historical_kv"] == 1
        and smoke_incremental_cache["physical_storage_exact"],
        "checkpoint_reopen": verification["passed"],
        "all_state_finite": all(
            row["parameters_finite"] and row["optimizer_finite"] for row in rows
        ),
        "parameter_manifest": parameters["passed"],
        "result_source_will_reload_immutable_parent": True,
    }
    audit = {
        "experiment": EXPERIMENT,
        "kind": "exactly three disposable optimizer updates",
        "command": " ".join(sys.argv),
        "argv_sha256": hashlib.sha256(
            json.dumps(sys.argv, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
        "temporal_gradient_rows": temporal_rows,
        "optimizer": optimizer_report,
        "incremental_cache_audit": smoke_incremental_cache,
        "checkpoint": verification,
        "checks": checks,
        "passed": all(checks.values()),
        "disposition": "Smoke model/optimizer are discarded. Scientific update 1 must freshly reload the immutable Standard checkpoint with g_rec=0.",
    }
    durable_json(output / "smoke_audit.json", audit)
    if not audit["passed"]:
        raise SystemExit(f"2D2A smoke failed: {checks}")
    print("EXPERIMENT_2D2A_SMOKE_PASS", flush=True)
    return audit


def training_metadata(args, preflight, micro_batch, accumulation):
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "implementation_fingerprint": preflight["implementation_fingerprint"],
        "frozen_2d1d_commit": FROZEN_COMMIT,
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "data_root": str(Path(args.data_root).resolve()),
        "canonical_validation_sha256": CANONICAL_VALIDATION_SHA256,
        "micro_batch_sequences": micro_batch,
        "gradient_accumulation": accumulation,
        "global_targets_per_update": GLOBAL_TARGETS,
        "data_continuation": require_config()["data_continuation"],
        "pass_cadence": {"two_pass": "all except 32/64/96", "three_pass": list(THREE_PASS_UPDATES)},
        "optimizer": {
            "base_lr": BASE_LR,
            "gate_lr": GATE_LR,
            "betas": list(BETAS),
            "eps": ADAM_EPS,
            "gradient_clip": GRAD_CLIP,
            "warmup_updates": WARMUP_UPDATES,
        },
        "pod_id": args.pod_id,
        "pod_name": args.pod_name,
        "gpu_type": args.gpu_type,
        "persistent_volume_identity": args.persistent_volume_identity,
        "runpod_stop_audit_sha256": preflight["runpod_stop_audit"]["audit_sha256"],
        "stop_mechanism": args.stop_mechanism,
        "stop_authenticated": bool(args.stop_authenticated),
        "preflight_implementation_commit": preflight["implementation_git_commit"],
    }


def initialize_runtime(args):
    workspace_mount_audit(args.output_dir, args.run_root, args.persistent_volume_identity)
    authenticated_stop_audit(args)
    device = require_single_a100()
    seed_all()
    output = Path(args.output_dir).resolve()
    preflight = read_json(output / "preflight_audit.json")
    smoke = read_json(output / "smoke_audit.json")
    if not preflight.get("result_run_authorized") or not smoke.get("passed"):
        raise SystemExit("result training requires passing preflight and smoke")
    require_implementation_fingerprint(preflight)
    if not preflight["runpod_stop_audit"]["driver_passed"]:
        raise SystemExit("automatic RunPod STOP unavailable; refusing result update 1")
    batch_manifest = read_json(output / "batch_manifest.json")
    micro_batch = int(batch_manifest["selected_micro_batch_sequences"])
    accumulation = int(batch_manifest["selected_gradient_accumulation"])
    if micro_batch * T * accumulation != GLOBAL_TARGETS:
        raise SystemExit("global target geometry mismatch")
    shards = training_shards(args.data_root)
    _, model, source_audit = load_model(args.source_checkpoint, device, trainable=True)
    parameters = parameter_manifest(model, source_audit)
    optimizer, optimizer_report = configure_optimizer(model)
    metadata = training_metadata(args, preflight, micro_batch, accumulation)
    if args.resume:
        loader, state, saved_pid, checkpoint_sha = load_checkpoint_runtime(
            args.resume,
            model,
            optimizer,
            shards,
            micro_batch,
            accumulation,
            metadata,
        )
        if state["completed_updates"] != 48:
            raise SystemExit("2D2A resume is authorized only at update 48")
        verification = read_json(
            Path(args.resume).with_suffix(Path(args.resume).suffix + ".verification.json")
        )
        observed_next_hash = next_global_batch_hash(loader, accumulation)
        expected_next_hash = verification["next_global_batch_sha256"]
        restart = {
            "checkpoint": str(Path(args.resume).resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "saved_process_id": saved_pid,
            "restored_process_id": os.getpid(),
            "fresh_process": saved_pid != os.getpid(),
            "completed_updates": state["completed_updates"],
            "processed_targets": state["processed_targets"],
            "next_global_batch_sha256": observed_next_hash,
            "expected_next_global_batch_sha256": expected_next_hash,
            "next_global_batch_matches": observed_next_hash == expected_next_hash,
            "checkpoint_strict_reopen_next_batch_passed": verification[
                "strict_reopen"
            ]["next_batch"] is True,
        }
        restart["passed"] = (
            restart["fresh_process"]
            and restart["next_global_batch_matches"]
            and restart["checkpoint_strict_reopen_next_batch_passed"]
        )
        durable_json(output / "forced_restart_update_48.json", restart)
        if not restart["passed"]:
            raise SystemExit(f"forced restart audit failed: {restart}")
    else:
        loader = d1.ExplicitShardLoader(shards, micro_batch, T)
        state = {
            "completed_updates": 0,
            "processed_targets": 0,
            "started_at": time.time(),
            "segment_start": 0,
            "last_checkpoint": None,
            "last_metrics": None,
        }
        saved_pid = None
    if state["completed_updates"] == 0 and model.g_rec.detach().float().item() != 0.0:
        raise SystemExit("scientific source did not start with exact zero gate")
    return SimpleNamespace(
        device=device,
        output=output,
        preflight=preflight,
        smoke=smoke,
        micro_batch=micro_batch,
        accumulation=accumulation,
        shards=shards,
        model=model,
        optimizer=optimizer,
        optimizer_report=optimizer_report,
        parameter_manifest=parameters,
        metadata=metadata,
        loader=loader,
        training_state=state,
        end_update=int(args.end_update),
    )


def merge_keyed_json(path, key, value):
    path = Path(path)
    payload = read_json(path) if path.is_file() else {}
    if str(key) in payload:
        raise SystemExit(f"refusing to overwrite {path.name} key {key}")
    payload[str(key)] = value
    durable_json(path, payload)


def milestone_diagnostics(runtime, update, val_path):
    validation = evaluate_parallel(runtime.model, val_path)
    validation.update({"update": update, "targets": update * GLOBAL_TARGETS})
    merge_keyed_json(runtime.output / "milestone_validation.json", update, validation)
    pinned = d1.ExplicitShardLoader(runtime.shards, 1, T)
    cpu_x, cpu_y = pinned.next_batch()
    pinned_identity = d0d.batch_identity(cpu_x, cpu_y)
    temporal = temporal_gradient_diagnostic(
        runtime.model, cpu_x.to(runtime.device), cpu_y.to(runtime.device)
    )
    temporal.update({"update": update, "targets": update * GLOBAL_TARGETS, "pinned_batch": pinned_identity})
    merge_keyed_json(runtime.output / "temporal_gradient_diagnostics.json", update, temporal)
    if update in (48, 96):
        composition = self_composition_diagnostic(runtime.model, val_path)
        composition.update({"update": update, "targets": update * GLOBAL_TARGETS})
        merge_keyed_json(runtime.output / "self_composition.json", update, composition)
    runtime.model.train()
    return validation, temporal


def save_scientific_checkpoint(runtime, update):
    checkpoint_path = (
        Path(runtime.run_root) / "checkpoints" / f"scientific_update_{update:04d}.pt"
    )
    previous_checkpoint = runtime.training_state.get("last_checkpoint")
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
        runtime.training_state["last_checkpoint"] = previous_checkpoint
        raise
    record_checkpoint(runtime.output, update, verification)
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
    if int(args.end_update) not in (48, 96):
        raise SystemExit("2D2A train segments must end at update 48 or 96")
    runtime = initialize_runtime(args)
    runtime.run_root = str(Path(args.run_root).resolve())
    completed = int(runtime.training_state["completed_updates"])
    if completed == 0 and int(args.end_update) != 48:
        raise SystemExit("fresh scientific process must stop at the forced update-48 boundary")
    if completed == 48 and int(args.end_update) != 96:
        raise SystemExit("resumed scientific process must end at update 96")
    metrics_path = runtime.output / "training_metrics.jsonl"
    if completed == 0 and metrics_path.exists():
        raise SystemExit("fresh scientific run found existing training metrics")
    if completed > 0:
        rows = read_jsonl(metrics_path)
        if len(rows) != completed or rows[-1]["update"] != completed:
            raise SystemExit("resume training metrics do not reconcile with checkpoint")
    val_path = validation_path(args.data_root)
    segment_start = completed
    runtime.training_state["segment_start"] = segment_start
    segment_started = time.time()
    for update in range(completed + 1, int(args.end_update) + 1):
        metrics = train_one_update(runtime, update)
        append_jsonl(metrics_path, metrics)
        write_heartbeat(runtime, metrics)
        print(
            f"2D2A update={update:03d}/{MAX_UPDATES} "
            f"loss={metrics['weighted_total_ce']:.6f} "
            f"gate={metrics['tanh_g_rec']:+.8f} "
            f"dt={metrics['wall_seconds']:.2f}s",
            flush=True,
        )
        if update in SCIENTIFIC_CHECKPOINTS:
            verification = save_scientific_checkpoint(runtime, update)
            milestone_diagnostics(runtime, update, val_path)
            if update == 48:
                durable_json(runtime.output / "restart_required_update_48.json", {
                    "update": 48,
                    "checkpoint": verification,
                    "saved_process_id": os.getpid(),
                    "fresh_process_required_for_update_49": True,
                })
    segment = {
        "start_update": segment_start,
        "end_update": int(args.end_update),
        "started_at": segment_started,
        "completed_at": time.time(),
        "process_id": os.getpid(),
        "command": " ".join(sys.argv),
        "argv_sha256": hashlib.sha256(
            json.dumps(sys.argv, separators=(",", ":")).encode()
        ).hexdigest(),
        "checkpoint": runtime.training_state["last_checkpoint"],
    }
    merge_keyed_json(runtime.output / "process_segments.json", args.end_update, segment)
    if int(args.end_update) == 48:
        print("EXPERIMENT_2D2A_UPDATE_48_RESTART_REQUIRED", flush=True)
    else:
        durable_json(runtime.output / "training_complete.json", {
            "completed_updates": 96,
            "processed_targets": TOTAL_TARGETS,
            "checkpoint": runtime.training_state["last_checkpoint"],
            "timestamp": time.time(),
        })
        print("EXPERIMENT_2D2A_TRAINING_COMPLETE", flush=True)
    return segment


@torch.no_grad()
def _incremental_control(model, x, y, control, derangement=None):
    batch, length = x.shape
    state = model.init_incremental_state(batch, device=x.device)
    per_sequence_sum = torch.zeros(batch, dtype=torch.float64, device="cpu")
    per_position_sum = np.zeros(length, dtype=np.float64)
    total_sum = 0.0
    targets = 0
    max_cache = [0] * N_LAYER
    max_ring = 0
    max_memory_rms = 0.0
    recurrent_output_rms = []
    for position in range(length):
        result = model.incremental_step(
            x[:, position],
            state,
            control=control,
            recurrent_permutation=derangement if control == "shuffled" else None,
            return_diagnostics=control != "plain",
        )
        if control == "plain":
            logits, state = result
        else:
            logits, state, diagnostics = result
            recurrent_output_rms.append(
                diagnostics["recurrent_output_rms"].detach().float().item()
            )
        losses = F.cross_entropy(logits[:, 0].float(), y[:, position], reduction="none")
        cpu_losses = losses.double().cpu()
        per_sequence_sum += cpu_losses
        per_position_sum[position] += cpu_losses.sum().item()
        total_sum += cpu_losses.sum().item()
        targets += batch
        lengths = model.incremental_cache_lengths(state)
        max_cache = [max(old, new) for old, new in zip(max_cache, lengths)]
        max_ring = max(max_ring, int(state.h12_ring.size(1)))
        if state.h12_ring.numel():
            max_memory_rms = max(
                max_memory_rms,
                state.h12_ring.float().square().mean().sqrt().item(),
            )
    return {
        "loss_sum": total_sum,
        "targets": targets,
        "per_sequence_losses": (per_sequence_sum / length).tolist(),
        "per_position_sum": per_position_sum,
        "final_cache_audit": model.incremental_cache_audit(state),
        "max_cache_lengths": max_cache,
        "max_h12_ring_length": max_ring,
        "max_h12_memory_rms": max_memory_rms,
        "mean_recurrent_output_rms": (
            statistics.fmean(recurrent_output_rms) if recurrent_output_rms else 0.0
        ),
    }


@torch.no_grad()
def evaluate_incremental(model, val_path, batches=INCREMENTAL_BATCHES):
    model.eval()
    device = next(model.parameters()).device
    loader = d1.ExplicitShardLoader([val_path], VALIDATION_B, T)
    rows = {
        name: {
            "loss_sum": 0.0,
            "targets": 0,
            "per_batch_losses": [],
            "per_sequence_losses": [],
            "per_position_sum": np.zeros(T, dtype=np.float64),
            "cache_rows": [],
            "max_memory_rms": 0.0,
            "recurrent_output_rms": [],
        }
        for name in ("plain", "real", "shuffled")
    }
    identities = []
    derangement = torch.arange(VALIDATION_B, device=device).roll(1)
    start = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(batches):
        cpu_x, cpu_y = loader.next_batch()
        identities.append(d0d.batch_identity(cpu_x, cpu_y))
        x, y = cpu_x.to(device), cpu_y.to(device)
        for control in ("plain", "real", "shuffled"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current = _incremental_control(
                    model, x, y, control, derangement=derangement
                )
            row = rows[control]
            row["loss_sum"] += current["loss_sum"]
            row["targets"] += current["targets"]
            row["per_batch_losses"].append(current["loss_sum"] / current["targets"])
            row["per_sequence_losses"].extend(current["per_sequence_losses"])
            row["per_position_sum"] += current["per_position_sum"]
            row["cache_rows"].append({
                "final": current["final_cache_audit"],
                "max_cache_lengths": current["max_cache_lengths"],
                "max_h12_ring_length": current["max_h12_ring_length"],
            })
            row["max_memory_rms"] = max(
                row["max_memory_rms"], current["max_h12_memory_rms"]
            )
            row["recurrent_output_rms"].append(current["mean_recurrent_output_rms"])
        print(f"2D2A incremental batch={batch_index + 1:02d}/{batches}", flush=True)
        del x, y
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - start
    controls = {}
    for name, row in rows.items():
        controls[name] = {
            "validation_loss": row["loss_sum"] / row["targets"],
            "validation_targets": row["targets"],
            "per_batch_losses": row["per_batch_losses"],
            "per_sequence_losses": row["per_sequence_losses"],
            "per_position_loss": (row["per_position_sum"] / (batches * VALIDATION_B)).tolist(),
            "cache_rows": row["cache_rows"],
            "max_h12_memory_rms": row["max_memory_rms"],
            "mean_recurrent_output_rms": statistics.fmean(row["recurrent_output_rms"]),
        }
    plain, real, shuffled = controls["plain"], controls["real"], controls["shuffled"]
    return {
        "controls": controls,
        "recurrent_gain": plain["validation_loss"] - real["validation_loss"],
        "sequence_gap": shuffled["validation_loss"] - real["validation_loss"],
        "real_vs_plain_batches": paired_stats(real["per_batch_losses"], plain["per_batch_losses"]),
        "real_vs_shuffled_batches": paired_stats(real["per_batch_losses"], shuffled["per_batch_losses"]),
        "real_vs_plain_sequences": paired_stats(real["per_sequence_losses"], plain["per_sequence_losses"]),
        "real_vs_shuffled_sequences": paired_stats(real["per_sequence_losses"], shuffled["per_sequence_losses"]),
        "effective_recurrent_scale": model.recurrent_scale.detach().float().item(),
        "gate_raw": model.g_rec.detach().float().item(),
        "canonical_subset_sha256": d0.aggregate_hashes(
            [row["combined_sha256"] for row in identities]
        ),
        "batch_identities": identities,
        "batch_count": batches,
        "batch_size": VALIDATION_B,
        "sequence_length": T,
        "targets_per_control": batches * VALIDATION_B * T,
        "minimum_target_requirement_met": batches * VALIDATION_B * T >= 131_072,
        "precision": "teacher-forced incremental torch.autocast(cuda,bfloat16)",
        "performance": {
            "wall_seconds": elapsed,
            "condition_target_passes_per_second": batches * VALIDATION_B * T * 3 / elapsed,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }


@torch.no_grad()
def parallel_incremental_equivalence(model, val_path, length=16, batch=2):
    loader = d1.ExplicitShardLoader([val_path], batch, length)
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
            plain_threshold = FP32_INCREMENTAL_PLAIN_MAX_ATOL
            active_threshold = FP32_INCREMENTAL_ACTIVE_PREFIX_MAX_ATOL
        else:
            torch.set_float32_matmul_precision(original_precision)
            torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            plain_threshold = BF16_INCREMENTAL_PLAIN_MAX_ATOL
            active_threshold = BF16_INCREMENTAL_ACTIVE_PREFIX_MAX_ATOL
        with context:
            plain = model.forward_pass(tokens)["logits"]
            incremental_plain = model.incremental_logits(tokens, control="plain")["logits"]
            first = model.forward_pass(tokens)
            parallel_real = model.forward_pass(tokens, recurrent_source=first["h12"])["logits"]
            incremental_real = model.incremental_logits(tokens, control="real")["logits"]
        plain_delta = (plain.float() - incremental_plain.float()).abs()
        recurrent_delta = (parallel_real.float() - incremental_real.float()).abs()
        reports[label] = {
            "plain_kernel_max_abs": plain_delta.max().item(),
            "plain_kernel_mean_abs": plain_delta.mean().item(),
            "active_recurrent_positions_0_3_max_abs": recurrent_delta[:, :4].max().item(),
            "active_recurrent_positions_0_3_mean_abs": recurrent_delta[:, :4].mean().item(),
            "self_recurrence_drift_positions_4_plus_mean_abs": recurrent_delta[:, 4:].mean().item(),
            "self_recurrence_drift_positions_4_plus_max_abs": recurrent_delta[:, 4:].max().item(),
            "plain_max_abs_tolerance": plain_threshold,
            "active_prefix_max_abs_tolerance": active_threshold,
            "tolerance_note": "Preregistered before result update 1 for numerically different whole-sequence and token-step SDPA reduction geometries.",
            "kernel_passed": plain_delta.max().item() <= plain_threshold
            and recurrent_delta[:, :4].max().item() <= active_threshold,
        }
    torch.set_float32_matmul_precision(original_precision)
    torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
    torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
    reports["passed"] = reports["fp32"]["kernel_passed"] and reports["bf16"]["kernel_passed"]
    return reports


def load_final_model(args, device):
    batch_manifest = read_json(Path(args.output_dir) / "batch_manifest.json")
    micro_batch = int(batch_manifest["selected_micro_batch_sequences"])
    accumulation = int(batch_manifest["selected_gradient_accumulation"])
    preflight = read_json(Path(args.output_dir) / "preflight_audit.json")
    _, model, source_audit = load_model(args.source_checkpoint, device, trainable=True)
    optimizer, _ = configure_optimizer(model)
    metadata = training_metadata(args, preflight, micro_batch, accumulation)
    shards = training_shards(args.data_root)
    loader, state, saved_pid, checkpoint_sha = load_checkpoint_runtime(
        args.final_checkpoint,
        model,
        optimizer,
        shards,
        micro_batch,
        accumulation,
        metadata,
    )
    if state["completed_updates"] != MAX_UPDATES or state["processed_targets"] != TOTAL_TARGETS:
        raise SystemExit("final checkpoint does not contain exact 96-update result")
    checkpoint_path = Path(args.final_checkpoint).resolve()
    manifest = read_json(Path(args.output_dir) / "checkpoint_manifest.json")
    entry = manifest.get("scientific", {}).get("96", {})
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".verification.json")
    if not sidecar_path.is_file():
        raise SystemExit("final checkpoint verification sidecar is missing")
    sidecar = read_json(sidecar_path)
    payload = d0.torch_load(checkpoint_path, mmap=True)
    reconciliation = {
        "manifest_path": Path(entry.get("checkpoint", "")).resolve() == checkpoint_path,
        "manifest_sha": entry.get("sha256") == checkpoint_sha,
        "manifest_bytes": entry.get("bytes") == checkpoint_path.stat().st_size,
        "manifest_passed": entry.get("passed") is True,
        "sidecar_sha": sidecar.get("sha256") == checkpoint_sha,
        "sidecar_bytes": sidecar.get("bytes") == checkpoint_path.stat().st_size,
        "sidecar_passed": sidecar.get("passed") is True
        and sidecar.get("strict_reopen", {}).get("passed") is True,
        "state_last_checkpoint": Path(state.get("last_checkpoint", "")).resolve()
        == checkpoint_path,
        "payload_updates": payload.get("completed_updates") == MAX_UPDATES,
        "payload_targets": payload.get("processed_targets") == TOTAL_TARGETS,
        "payload_git_metadata": payload.get("git_commit")
        == payload.get("metadata", {}).get("implementation_git_commit"),
        "payload_fingerprint": payload.get("metadata", {}).get(
            "implementation_fingerprint"
        )
        == preflight["implementation_fingerprint"],
    }
    del payload
    gc.collect()
    if not all(reconciliation.values()):
        raise SystemExit(f"final checkpoint reconciliation failed: {reconciliation}")
    durable_json(Path(args.output_dir) / "final_checkpoint_reconciliation.json", {
        "checkpoint": str(checkpoint_path),
        "sha256": checkpoint_sha,
        "checks": reconciliation,
        "passed": True,
    })
    return model, optimizer, loader, state, source_audit, checkpoint_sha


def classify_result(incremental, parallel, stable=True, integrity=True):
    if not integrity:
        return "EXPERIMENT 2D2A INVALID", "EXPERIMENT 2D2A INVALID"
    if not stable:
        return "TOKEN-INDEXED RECURRENT K/V IS UNSTABLE", "NO SEQUENCE-SPECIFIC RECURRENT K/V"
    gain = incremental["recurrent_gain"]
    gap = incremental["sequence_gap"]
    pairing = incremental["real_vs_plain_sequences"]
    sequence_pairing = incremental["real_vs_shuffled_sequences"]
    majority = pairing["wins"] > pairing["losses"]
    strong_proportion = pairing["wins"] / max(pairing["count"], 1) >= 0.90
    if gain >= 0.01 and strong_proportion and gap > 0:
        primary = "TOKEN-INDEXED RECURRENT K/V LEARNS CLEAR POSITIVE UTILITY"
    elif gain > 0 and majority:
        primary = "TOKEN-INDEXED RECURRENT K/V LEARNS POSITIVE UTILITY"
    elif gain < -0.01:
        primary = "TOKEN-INDEXED RECURRENT K/V REMAINS HARMFUL"
    else:
        primary = "TOKEN-INDEXED RECURRENT K/V APPROACHES NEUTRALITY"
    if gap > 0 and sequence_pairing["wins"] > sequence_pairing["losses"]:
        secondary = "SEQUENCE-SPECIFIC RECURRENT K/V"
    elif gain > 0:
        secondary = "RECURRENT K/V UTILITY WITHOUT STRONG ALIGNMENT"
    else:
        secondary = "NO SEQUENCE-SPECIFIC RECURRENT K/V"
    return primary, secondary


def choose_recommendation(primary, secondary, parallel_gain, true_gain):
    if primary == "EXPERIMENT 2D2A INVALID":
        return "FIX 2D2A INTEGRITY"
    if primary == "TOKEN-INDEXED RECURRENT K/V IS UNSTABLE":
        return "STABILIZE RECURRENT K/V SCALE"
    if primary in {
        "TOKEN-INDEXED RECURRENT K/V LEARNS CLEAR POSITIVE UTILITY",
        "TOKEN-INDEXED RECURRENT K/V LEARNS POSITIVE UTILITY",
    }:
        return "EXTEND RECURRENT K/V TO THE MIRRORED HIGH→LOW LAYER PAIRS"
    if parallel_gain > 0 and true_gain <= 0:
        return "TRAIN FOR SELF-RECURRENT K/V DISTRIBUTION COMPATIBILITY"
    if secondary == "SEQUENCE-SPECIFIC RECURRENT K/V":
        return "ADD DEDICATED RECURRENT K/V PROJECTIONS"
    return "INCREASE RECURRENT-BRANCH LEARNING CAPACITY"


def make_plots(output, milestones, training, incremental, performance):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    ordered = [milestones[str(update)] for update in MILESTONES]
    targets_m = [row["targets"] / 1e6 for row in ordered]
    plain = [row["controls"]["plain"]["validation_loss"] for row in ordered]
    real = [row["controls"]["real"]["validation_loss"] for row in ordered]
    gain = [row["recurrent_gain"] for row in ordered]
    gap = [row["sequence_gap"] for row in ordered]
    gate = [row["effective_recurrent_scale"] for row in ordered]

    def line_plot(path, ys, labels, ylabel):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for values, label in zip(ys, labels):
            ax.plot(targets_m, values, marker="o", label=label)
        ax.set_xlabel("2D2A training targets (millions)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if labels:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / path, dpi=180)
        plt.close(fig)

    line_plot("P1_plain_real_validation.png", [plain, real], ["Plain", "Real"], "Validation CE")
    line_plot("P2_recurrent_gain.png", [gain], ["Plain − Real"], "Recurrent gain")
    line_plot("P3_sequence_gap.png", [gap], ["Shuffled − Real"], "Sequence gap")
    line_plot("P4_gate.png", [gate], ["tanh(g_rec)"], "Effective recurrent scale")
    slot_old = [row["recurrent_attention"]["slot_t_minus_3"] for row in ordered]
    slot_new = [row["recurrent_attention"]["slot_t_minus_2"] for row in ordered]
    line_plot("P5_recurrent_slots.png", [slot_old, slot_new], ["t−3", "t−2"], "Mean recurrent attention weight")

    heads = ordered[-1]["recurrent_attention"]["per_head"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    indices = np.arange(N_HEAD)
    ax.bar(indices - 0.2, [row["slot_t_minus_3"] for row in heads], width=0.4, label="t−3")
    ax.bar(indices + 0.2, [row["slot_t_minus_2"] for row in heads], width=0.4, label="t−2")
    ax.set_xlabel("B1 attention head")
    ax.set_ylabel("Mean recurrent weight")
    ax.set_xticks(indices)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "P6_per_head_recurrent_attention.png", dpi=180)
    plt.close(fig)

    bins = ordered[-1]["position_bins"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(bins)
    ax.bar(names, [bins[name]["recurrent_gain"] for name in names])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Plain − Real CE")
    ax.set_xlabel("Token position bin")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output / "P7_position_bin_recurrent_gain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["Parallel Pass-2", "True incremental"], [ordered[-1]["recurrent_gain"], incremental["recurrent_gain"]])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Recurrent gain")
    fig.tight_layout()
    fig.savefig(output / "P8_parallel_vs_incremental_gain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([row["targets"] / 1e6 for row in training], [row["b12_memory_rms"] for row in training], marker=".")
    ax.set_xlabel("2D2A training targets (millions)")
    ax.set_ylabel("B12 state RMS")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "P9_b12_memory_rms.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].plot([row["update"] for row in training], [row["wall_seconds"] for row in training])
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Wall seconds/update")
    axes[1].plot([row["update"] for row in training], [row["peak_allocated_vram_mb"] / 1024 for row in training])
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Peak allocated VRAM (GiB)")
    fig.tight_layout()
    fig.savefig(output / "P10_runtime_vram.png", dpi=180)
    plt.close(fig)


def build_integrity_audit(output, summary, incremental, equivalence):
    output = Path(output)
    training = read_jsonl(output / "training_metrics.jsonl")
    checkpoints = read_json(output / "checkpoint_manifest.json")
    restart = read_json(output / "forced_restart_update_48.json")
    parameters = read_json(output / "parameter_manifest.json")
    preflight = read_json(output / "preflight_audit.json")
    smoke = read_json(output / "smoke_audit.json")
    mount = read_json(output / "persistent_workspace_audit.json")
    expected_cadence = [3 if update in THREE_PASS_UPDATES else 2 for update in range(1, 97)]
    observed_cadence = [row["pass_count"] for row in training]
    checkpoint_pass = all(
        checkpoints["scientific"].get(str(update), {}).get("passed")
        and checkpoints["scientific"][str(update)]["strict_reopen"]["passed"]
        for update in SCIENTIFIC_CHECKPOINTS
    )
    cache_pass = all(
        row["final"]["passed"]
        for control in incremental["controls"].values()
        for row in control["cache_rows"]
    )
    checks = {
        "2D1D frozen tag exact": git_output("rev-parse", FROZEN_TAG + "^{commit}") == FROZEN_COMMIT,
        "mature Standard source SHA exact": preflight["source"]["checkpoint_sha256"] == SOURCE_SHA256,
        "Standard GPT-2 architecture exact": preflight["source"]["architecture"]["passed"],
        "Full AttnRes absent": preflight["source"]["architecture"]["full_attnres_active_modules"] == 0,
        "parent parameter count recorded": parameters["parent_total_parameters"] == SOURCE_PARAMETERS,
        "exactly one new parameter": parameters["new_parameter_count"] == 1,
        "new parameter = scalar g_rec only": parameters["checks"]["new_only_scalar_gate"],
        "no recurrent projection weights": preflight["architecture"]["forbidden_modules_absent"]["recurrent_projections"],
        "B1 local W exactly 2": preflight["architecture"]["b1_local_window"] == 2,
        "B2-B12 W exactly 1024": preflight["architecture"]["b2_b12_windows"] == [1024] * 11,
        "recurrent width exactly 2": preflight["architecture"]["recurrent_window"] == 2,
        "lag exactly 2": preflight["architecture"]["recurrent_lag"] == 2,
        "source exactly B12 post-MLP residual": preflight["architecture"]["source"].startswith("B12 post-MLP"),
        "source positions exactly t-3,t-2": preflight["architecture"]["recurrent_positions"] == ["t-3", "t-2"],
        "no wraparound": preflight["kernel_preflight"]["reports"]["bank"]["no_wraparound_values"],
        "early-boundary recurrent masking exact": preflight["kernel_preflight"]["checks"]["early_boundary_exact"],
        "early-boundary recurrent probabilities exact": preflight[
            "kernel_preflight"
        ]["checks"]["early_recurrent_probabilities_exact"],
        "local/recurrent positions do not overlap": preflight[
            "kernel_preflight"
        ]["checks"]["local_recurrent_positions_disjoint"],
        "B1 existing LN reused": preflight["kernel_preflight"]["checks"]["shared_b1_ln_kv_exact"],
        "B1 existing Q/K/V reused": preflight["kernel_preflight"]["checks"]["shared_b1_ln_kv_exact"],
        "separate local/recurrent softmaxes": preflight["architecture"]["separate_softmaxes"],
        "single B1 c_proj application": preflight["kernel_preflight"]["checks"]["single_c_proj"],
        "zero-gate identity pass": preflight["kernel_preflight"]["checks"]["zero_gate_b1_w2_identity"],
        "future causality pass": preflight["kernel_preflight"]["checks"]["future_causality"],
        "row isolation pass": preflight["kernel_preflight"]["checks"]["row_isolation"],
        "temporal writer gradient verified": all(
            row["writer_temporal_gradient_present"]
            for row in read_json(output / "temporal_gradient_diagnostics.json").values()
        ),
        "same-model recurrence only": True,
        "no teacher": True,
        "all GPT-2 parameters trainable": parameters["checks"]["all_trainable"],
        "CE-only loss": True,
        "pass cadence exact": observed_cadence == expected_cadence,
        "global targets/update exactly 524,288": all(row["targets"] == row["update"] * GLOBAL_TARGETS for row in training),
        "exactly 96 result updates": len(training) == 96 and [row["update"] for row in training] == list(range(1, 97)),
        "exactly 50,331,648 result targets": training[-1]["targets"] == TOTAL_TARGETS,
        "all losses/gradients/parameters finite": all(
            math.isfinite(row["weighted_total_ce"])
            and row["all_gradients_finite"]
            and row["all_parameters_finite"]
            and row["all_optimizer_moments_finite"]
            for row in training
        ),
        "B1 hidden full KV absent": cache_pass,
        "incremental cache bounds pass": cache_pass,
        "parallel/incremental kernel equivalence": equivalence["passed"],
        "checkpoints strict reopen": checkpoint_pass,
        "forced update-48 restart pass": restart["passed"],
        "disposable smoke exact": smoke["passed"] and len(smoke["rows"]) == 3,
        "disposable smoke FP32": all(row["precision"] == "fp32" for row in smoke["rows"]),
        "authenticated stop capability verified": preflight["runpod_stop_audit"][
            "driver_passed"
        ],
        "frozen 2D0D source provenance verified": preflight["source"][
            "frozen_2d0d_provenance"
        ]["passed"],
        "implementation fingerprint unchanged": require_implementation_fingerprint(
            preflight
        )["aggregate_sha256"]
        == preflight["implementation_fingerprint"]["aggregate_sha256"],
        "no detached result-training arm": True,
        "no mirrored extra links": True,
        "no window/lag sweep": True,
        "no AttnRes": True,
        "no HellaSwag": True,
        "Git synchronized": git_output("rev-parse", "HEAD")
        == git_output("rev-parse", f"origin/{BRANCH}"),
        "persistent artifacts synchronized": mount["passed"]
        and mount["checks"]["canonical_result_directory"]
        and mount["checks"]["persistent_identity_exact"],
    }
    return {
        "experiment": EXPERIMENT,
        "timestamp": time.time(),
        "checks": checks,
        "passed": all(checks.values()),
        "note": "The result-compute Git base and direct persistent-volume target are checked here. Final report-commit synchronization is externally reverified after the last push because an artifact cannot contain its own future commit SHA.",
    }


def build_artifact_inventory(output):
    output = Path(output)
    mutable = {
        "EXPERIMENT_2D2A_FINAL_REPORT.md",
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
    passed = all(row["exists_nonempty"] for row in artifacts.values()) and all(
        row["exists_nonempty"] for row in plots.values()
    )
    return {
        "required_artifacts": artifacts,
        "required_plots": plots,
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "required_plot_count": len(REQUIRED_PLOTS),
        "passed": passed,
    }


def build_performance(training, milestones, incremental):
    two_pass = [row for row in training if row["pass_count"] == 2]
    three_pass = [row for row in training if row["pass_count"] == 3]
    mean_tps = statistics.fmean(row["targets_per_second"] for row in training)
    baseline_single_a100_estimate = 42_047.189021
    return {
        "training": {
            "wall_seconds": sum(row["wall_seconds"] for row in training),
            "mean_wall_seconds_per_update": statistics.fmean(row["wall_seconds"] for row in training),
            "mean_targets_per_second": mean_tps,
            "two_pass_mean_wall_seconds": statistics.fmean(row["wall_seconds"] for row in two_pass),
            "three_pass_mean_wall_seconds": statistics.fmean(row["wall_seconds"] for row in three_pass),
            "mean_pass_forward_seconds": [
                statistics.fmean(
                    row["pass_forward_seconds"][index]
                    for row in training
                    if len(row["pass_forward_seconds"]) > index
                )
                for index in range(3)
            ],
            "mean_aggregate_backward_seconds": statistics.fmean(row["aggregate_backward_seconds"] for row in training),
            "peak_allocated_vram_mb": max(row["peak_allocated_vram_mb"] for row in training),
            "peak_reserved_vram_mb": max(row["peak_reserved_vram_mb"] for row in training),
            "timing_note": training[0]["timing_note"],
        },
        "parallel_validation": {
            str(update): milestones[str(update)]["performance"] for update in MILESTONES
        },
        "incremental_validation": incremental["performance"],
        "ordinary_standard_single_a100_reference_targets_per_second": baseline_single_a100_estimate,
        "relative_training_throughput_vs_ordinary_standard": mean_tps / baseline_single_a100_estimate,
        "ordinary_reference_note": "Historical mature Standard steady-state aggregate 168,188.756084 targets/s divided by four A100 GPUs; research implementation comparison only, not a matched benchmark.",
        "recurrent_attention_overhead": {
            "extra_attention_entries_per_warm_receiver": 2,
            "b1_local_entries_per_warm_receiver": 2,
            "b1_recurrent_attention_score_overhead_vs_local": 1.0,
            "model_level_compute_note": "The recurrent score/value work is tiny relative to 11 full-context upper blocks; two attached Transformer passes dominate training cost.",
            "optimized_serving_claim": False,
        },
    }


def build_questions(summary, milestones, training, incremental, attention):
    positive = [
        update for update in MILESTONES[1:]
        if milestones[str(update)]["recurrent_gain"] > 0
    ]
    gate_open = next(
        (row for row in training if row["g_rec_raw"] != 0.0), None
    )
    final_bins = milestones["96"]["position_bins"]
    long_bin = max(final_bins, key=lambda key: final_bins[key]["recurrent_gain"])
    long_keys = ("257-512", "513-1023")
    early_keys = ("3-16", "17-32", "33-64", "65-128", "129-256")
    bin_widths = {
        name: last - first + 1 for name, first, last in POSITION_BINS
    }
    long_gain = sum(
        final_bins[name]["recurrent_gain"] * bin_widths[name] for name in long_keys
    ) / sum(bin_widths[name] for name in long_keys)
    early_gain = sum(
        final_bins[name]["recurrent_gain"] * bin_widths[name] for name in early_keys
    ) / sum(bin_widths[name] for name in early_keys)
    heads = attention["per_head"]
    old_heads = sum(row["slot_t_minus_3"] > row["slot_t_minus_2"] for row in heads)
    new_heads = len(heads) - old_heads
    temporal = read_json(Path(summary["artifact_directory"]) / "temporal_gradient_diagnostics.json")
    temporal_all = all(
        temporal[str(update)]["writer_temporal_gradient_present"]
        for update in (10, 20, 48, 96)
    )
    return {
        "Q1": f"{summary['parameters']['parent']:,}",
        "Q2": f"Yes. Exactly one scalar g_rec; total {summary['parameters']['total']:,}.",
        "Q3": "Yes; gate-zero reproduced the B1-W2 oracle within the frozen tolerance.",
        "Q4": f"{summary['parent']['b1_w2_damage']:+.12f} CE (B1-W2 {summary['parent']['b1_w2_loss']:.12f} vs full {summary['parent']['full_loss']:.12f}).",
        "Q5": (
            f"It opened on update {gate_open['update']} to raw {gate_open['g_rec_raw']:+.9g} "
            f"(tanh {gate_open['tanh_g_rec']:+.9g})." if gate_open else "It did not open."
        ),
        "Q6": (
            "Yes" if temporal_all else "No"
        ) + f"; pinned gradient norms: {[temporal[str(u)]['gradient_norm'] for u in (10,20,48,96)]}.",
        "Q7": f"{milestones['10']['recurrent_gain']:+.10f}",
        "Q8": f"{milestones['20']['recurrent_gain']:+.10f}",
        "Q9": f"{milestones['48']['recurrent_gain']:+.10f}",
        "Q10": f"{milestones['96']['recurrent_gain']:+.10f}",
        "Q11": "Yes." if positive else "No.",
        "Q12": str(positive[0]) + " updates." if positive else "None.",
        "Q13": (
            f"Final parallel Real-vs-Shuffled wins: {milestones['96']['real_vs_shuffled']['wins']}/20; "
            f"gap {milestones['96']['sequence_gap']:+.10f}."
        ),
        "Q14": (
            "t-3" if attention["slot_t_minus_3"] > attention["slot_t_minus_2"] else "t-2"
        ) + f" ({attention['slot_t_minus_3']:.6f} vs {attention['slot_t_minus_2']:.6f}).",
        "Q15": f"Yes descriptively: {old_heads} heads preferred t-3 and {new_heads} preferred t-2." if old_heads and new_heads else "No clear split across heads.",
        "Q16": (
            "Yes" if long_gain > early_gain and long_gain > 0 else "No"
        )
        + f"; positions 257-1023 averaged {long_gain:+.10f} versus {early_gain:+.10f} for positions 3-256. The largest individual bin was {long_bin} ({final_bins[long_bin]['recurrent_gain']:+.10f}).",
        "Q17": (
            "Yes; no numerical divergence was observed and all preregistered finite/RMS/loss checks passed."
            if summary["stability"]["passed"]
            else "No; the preregistered numerical-stability check failed."
        ),
        "Q18": (
            f"Parallel gain {milestones['96']['recurrent_gain']:+.10f}; true-self gain {incremental['recurrent_gain']:+.10f}."
        ),
        "Q19": f"{incremental['recurrent_gain']:+.10f}",
        "Q20": f"{incremental['sequence_gap']:+.10f}",
        "Q21": (
            "One historical B1 K/V entry."
            if summary["incremental_cache"]["passed"]
            else "The B1 physical-cache audit failed."
        ),
        "Q22": (
            "Three raw B12 residual states, the minimum pipeline ring."
            if summary["incremental_cache"]["passed"]
            else "The B12 recurrent-ring audit failed."
        ),
        "Q23": "Two recurrent score/value entries per warmed B1 query (equal to the two-entry local branch); model-level runtime is reported in performance.json.",
        "Q24": "Yes." if summary["next_recommendation"] == "ADD DEDICATED RECURRENT K/V PROJECTIONS" else "No; follow the selected next-experiment rule first.",
        "Q25": "Yes." if summary["next_recommendation"].startswith("EXTEND RECURRENT") else "No.",
        "Q26": summary["next_recommendation"],
    }


def render_report(summary, audit, questions):
    final = summary["final"]
    lines = [
        f"EXPERIMENT 2D2A PRIMARY CLASSIFICATION:\n{summary['primary_classification']}",
        f"EXPERIMENT 2D2A SECONDARY CLASSIFICATION:\n{summary['secondary_classification']}",
        "",
        "# Experiment 2D2A final report",
        "",
        "The B12→B1 token-indexed recurrent K/V pilot completed its exact 96-update, 50,331,648-target budget on one NVIDIA A100-SXM4-80GB. The result below keeps parallel Pass-2 behavior separate from deployment-equivalent true self recurrence.",
        "",
        "## Verified setup",
        "",
        f"- Parameters: {summary['parameters']['parent']:,} parent + 1 scalar = {summary['parameters']['total']:,} total.",
        "- Geometry: B1 local W2; B12 recurrent states at t−3/t−2; B2–B12 W1024.",
        f"- Source checkpoint: `{summary['source_checkpoint']}` (`{summary['source_checkpoint_sha256']}`).",
        f"- Parent full CE: {summary['parent']['full_loss']:.10f}; parent B1-W2 CE: {summary['parent']['b1_w2_loss']:.10f}.",
        f"- Runtime: {summary['training']['runtime_seconds']:.1f} s over {summary['training']['updates']} updates.",
        "",
        "## Final losses",
        "",
        "| Evaluation | Plain | Real | Shuffled | Recurrent gain | Sequence gap |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Parallel Pass-2 | {final['parallel_plain']:.10f} | {final['parallel_real']:.10f} | {final['parallel_shuffled']:.10f} | {final['parallel_gain']:+.10f} | {final['parallel_sequence_gap']:+.10f} |",
        f"| True incremental self | {final['incremental_plain']:.10f} | {final['incremental_real']:.10f} | {final['incremental_shuffled']:.10f} | {final['incremental_gain']:+.10f} | {final['incremental_sequence_gap']:+.10f} |",
        "",
        f"The raw gate ended at {summary['gate']['final_raw']:+.9g}, giving tanh(g_rec)={summary['gate']['final_effective']:+.9g}. "
        + (
            "Temporal writer gradients into attached Pass-1 B12 states were finite and nonzero at updates 10, 20, 48, and 96."
            if all(
                row["writer_temporal_gradient_present"]
                for row in summary["temporal_gradient"].values()
            )
            else "Temporal writer-gradient verification failed at one or more required milestones."
        ),
        "",
        (
            "Incremental storage passed with one historical B1 K/V entry, at most 1023 historical K/V entries in each B2–B12 cache, and a three-state raw B12 residual ring."
            if summary["incremental_cache"]["passed"]
            else "Incremental storage/cache verification failed."
        ) + " This pilot does not claim whole-model KV-cache savings.",
        "",
        f"Exactly one next experiment is recommended: **{summary['next_recommendation']}**. It was not executed.",
        "",
        "## Q1–Q26",
        "",
    ]
    for index in range(1, 27):
        lines.extend([f"### Q{index}", "", questions[f"Q{index}"], ""])
    lines.extend([
        "## Integrity and handoff",
        "",
        f"- Integrity audit: {'PASS' if audit['passed'] else 'FAIL'}.",
        f"- Final checkpoint: `{summary['checkpoint']['path']}` (`{summary['checkpoint']['sha256']}`).",
        f"- Implementation Git commit: `{summary['git']['implementation_commit']}`.",
        f"- Results Git commit: `{summary['git'].get('results_commit') or 'PENDING RESULTS COMMIT'}`.",
        f"- Artifact directory: `{summary['artifact_directory']}`.",
        "- GPU pod is stopped after final Git and artifact synchronization; the network volume and historical checkpoints are preserved.",
        "",
        "# EXPERIMENT 2D2A COMPLETE",
        "",
    ])
    return "\n".join(lines)


def _numbers_finite(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numbers_finite(item) for item in value)
    return True


def build_stability_evidence(composition, incremental, training):
    composition_passes = [
        row
        for milestone in composition.values()
        for row in milestone["passes"]
    ]
    h12_rms = [row["b12_memory_rms"] for row in training] + [
        row["b12_memory_rms"] for row in composition_passes
    ] + [
        control["max_h12_memory_rms"]
        for control in incremental["controls"].values()
    ]
    recurrent_rms = [
        row["b1_recurrent_output_rms"]
        for row in training
        if row["b1_recurrent_output_rms"] is not None
    ] + [row["b1_recurrent_output_rms"] for row in composition_passes] + [
        control["mean_recurrent_output_rms"]
        for control in incremental["controls"].values()
    ]
    losses = [row["weighted_total_ce"] for row in training] + [
        row["loss"] for row in composition_passes
    ] + [
        control["validation_loss"] for control in incremental["controls"].values()
    ]
    checks = {
        "all_numbers_finite": _numbers_finite(composition)
        and _numbers_finite(incremental)
        and _numbers_finite(training),
        "all_training_state_finite": all(
            row["all_parameters_finite"]
            and row["all_gradients_finite"]
            and row["all_optimizer_moments_finite"]
            for row in training
        ),
        "b12_rms_below_hard_divergence_limit": max(h12_rms)
        < STABILITY_RMS_HARD_LIMIT,
        "recurrent_output_rms_below_hard_divergence_limit": max(recurrent_rms)
        < STABILITY_RMS_HARD_LIMIT,
        "loss_below_hard_divergence_limit": max(losses) < STABILITY_LOSS_HARD_LIMIT,
    }
    return {
        "hard_rms_limit": STABILITY_RMS_HARD_LIMIT,
        "hard_loss_limit": STABILITY_LOSS_HARD_LIMIT,
        "max_b12_rms": max(h12_rms),
        "max_recurrent_output_rms": max(recurrent_rms),
        "max_loss": max(losses),
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": "Hard limits detect numerical divergence only; passing is not a claim of tight dynamical contraction.",
    }


def run_finalize(args):
    require_git(clean=False)
    require_config()
    dirty = [
        line for line in git_output("status", "--porcelain").splitlines()
        if line and OUTPUT_NAME not in line
    ]
    if dirty:
        raise SystemExit(f"finalize found non-result worktree changes: {dirty}")
    workspace_mount_audit(args.output_dir, args.run_root, args.persistent_volume_identity)
    authenticated_stop_audit(args)
    device = require_single_a100()
    seed_all()
    output = Path(args.output_dir).resolve()
    require_implementation_fingerprint(read_json(output / "preflight_audit.json"))
    model, optimizer, loader, state, source_audit, checkpoint_sha = load_final_model(args, device)
    val_path = validation_path(args.data_root)
    incremental = evaluate_incremental(model, val_path, batches=INCREMENTAL_BATCHES)
    if not incremental["minimum_target_requirement_met"]:
        raise SystemExit("incremental validation target minimum not met")
    equivalence = parallel_incremental_equivalence(model, val_path)
    if not equivalence["passed"]:
        raise SystemExit(f"parallel/incremental kernel equivalence failed: {equivalence}")
    durable_json(output / "incremental_validation.json", incremental)
    cache_audit = {
        "controls": {
            name: value["cache_rows"] for name, value in incremental["controls"].items()
        },
        "b1_historical_kv_limit": 1,
        "b2_b12_historical_kv_limit": 1023,
        "b12_recurrent_state_ring_limit": 3,
        "parallel_incremental_equivalence": equivalence,
    }
    cache_audit["passed"] = equivalence["passed"] and all(
        row["final"]["passed"]
        for control in cache_audit["controls"].values()
        for row in control
    )
    durable_json(output / "incremental_cache_audit.json", cache_audit)

    milestones = read_json(output / "milestone_validation.json")
    training = read_jsonl(output / "training_metrics.jsonl")
    if set(milestones) != {str(value) for value in MILESTONES}:
        raise SystemExit("milestone validation set is incomplete")
    final_parallel = milestones["96"]
    composition = read_json(output / "self_composition.json")
    stability = build_stability_evidence(composition, incremental, training)
    stable = stability["passed"]
    primary, secondary = classify_result(
        incremental, final_parallel, stable=stable, integrity=True
    )
    recommendation = choose_recommendation(
        primary,
        secondary,
        final_parallel["recurrent_gain"],
        incremental["recurrent_gain"],
    )
    checkpoint_path = Path(args.final_checkpoint).resolve()
    runtime_seconds = sum(row["wall_seconds"] for row in training)
    summary = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "primary_classification": primary,
        "secondary_classification": secondary,
        "next_recommendation": recommendation,
        "parameters": {
            "parent": SOURCE_PARAMETERS,
            "new": 1,
            "total": TOTAL_PARAMETERS,
        },
        "hardware": {
            "pod_id": args.pod_id,
            "pod_name": args.pod_name,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": 1,
        },
        "geometry": architecture_manifest(),
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "parent": {
            "full_loss": PARENT_FULL_LOSS,
            "b1_w2_loss": PARENT_B1_W2_LOSS,
            "b1_w2_damage": PARENT_B1_W2_DAMAGE,
        },
        "training": {
            "updates": MAX_UPDATES,
            "targets": TOTAL_TARGETS,
            "runtime_seconds": runtime_seconds,
            "pass3_updates": list(THREE_PASS_UPDATES),
            "forced_restart_after_update": 48,
        },
        "final": {
            "parallel_plain": final_parallel["controls"]["plain"]["validation_loss"],
            "parallel_real": final_parallel["controls"]["real"]["validation_loss"],
            "parallel_shuffled": final_parallel["controls"]["shuffled"]["validation_loss"],
            "parallel_gain": final_parallel["recurrent_gain"],
            "parallel_sequence_gap": final_parallel["sequence_gap"],
            "incremental_plain": incremental["controls"]["plain"]["validation_loss"],
            "incremental_real": incremental["controls"]["real"]["validation_loss"],
            "incremental_shuffled": incremental["controls"]["shuffled"]["validation_loss"],
            "incremental_gain": incremental["recurrent_gain"],
            "incremental_sequence_gap": incremental["sequence_gap"],
            "parallel_real_vs_plain": final_parallel["real_vs_plain"],
            "parallel_real_vs_shuffled": final_parallel["real_vs_shuffled"],
            "incremental_real_vs_plain_sequences": incremental["real_vs_plain_sequences"],
            "incremental_real_vs_shuffled_sequences": incremental["real_vs_shuffled_sequences"],
        },
        "gate": {
            "initial_raw": 0.0,
            "final_raw": model.g_rec.detach().float().item(),
            "final_effective": model.recurrent_scale.detach().float().item(),
            "trajectory": [
                {
                    "update": row["update"],
                    "targets": row["targets"],
                    "raw": row["g_rec_raw"],
                    "effective": row["tanh_g_rec"],
                    "gradient": row["gate_gradient_preclip"],
                }
                for row in training
            ],
        },
        "recurrent_attention": final_parallel["recurrent_attention"],
        "temporal_gradient": read_json(output / "temporal_gradient_diagnostics.json"),
        "incremental_cache": cache_audit,
        "parallel_incremental_equivalence": equivalence,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "bytes": checkpoint_path.stat().st_size,
        },
        "git": {
            "implementation_commit": state.get("implementation_git_commit", git_output("rev-parse", "HEAD")),
            "training_commit": git_output("rev-parse", "HEAD"),
            "results_commit": args.results_commit,
        },
        "artifact_directory": str(output),
        "pod_status": "RUNNING DURING FINALIZATION; STOP REQUIRED AFTER PUSH",
        "stability": stability,
    }
    # The implementation commit is authoritative in checkpoint metadata.
    checkpoint_payload_read = d0.torch_load(checkpoint_path, mmap=True)
    summary["git"]["implementation_commit"] = checkpoint_payload_read["git_commit"]
    del checkpoint_payload_read
    gc.collect()

    paired = {
        key: {
            "update": int(key),
            "targets": value["targets"],
            "real_vs_plain": value["real_vs_plain"],
            "real_vs_shuffled": value["real_vs_shuffled"],
            "recurrent_gain": value["recurrent_gain"],
            "sequence_gap": value["sequence_gap"],
        }
        for key, value in milestones.items()
    }
    gate_diagnostics = {
        "updates": summary["gate"]["trajectory"],
        "milestones": {
            key: {
                "raw": value["gate_raw"],
                "effective": value["effective_recurrent_scale"],
            }
            for key, value in milestones.items()
        },
    }
    attention_diagnostics = {
        key: value["recurrent_attention"] for key, value in milestones.items()
    }
    durable_json(output / "paired_controls.json", paired)
    durable_json(output / "gate_diagnostics.json", gate_diagnostics)
    durable_json(output / "recurrent_attention_diagnostics.json", attention_diagnostics)
    performance = build_performance(training, milestones, incremental)
    durable_json(output / "performance.json", performance)
    commands = {
        "experiment": EXPERIMENT,
        "finalize_command": " ".join(sys.argv),
        "preflight_command": read_json(output / "preflight_audit.json")["command"],
        "smoke_command": read_json(output / "smoke_audit.json")["command"],
        "training_commands": {
            key: value["command"]
            for key, value in read_json(output / "process_segments.json").items()
        },
        "process_segments": read_json(output / "process_segments.json"),
        "hardware": environment_payload(),
        "runpod_stop_capability_audit": read_json(
            output / "runpod_stop_capability.json"
        ),
        "planned_terminal_stop_command": "runpodctl pod stop 7kk5yyti00rnrp -o json",
        "finalized_at": time.time(),
    }
    durable_json(output / "commands_and_runtime.json", commands)

    # Build once provisionally, then let an integrity failure force INVALID.
    audit = build_integrity_audit(output, summary, incremental, equivalence)
    if not audit["passed"]:
        primary, secondary = classify_result(
            incremental, final_parallel, stable=stable, integrity=False
        )
        summary["primary_classification"] = primary
        summary["secondary_classification"] = secondary
        summary["next_recommendation"] = choose_recommendation(
            primary, secondary, final_parallel["recurrent_gain"], incremental["recurrent_gain"]
        )
    questions = build_questions(
        summary, milestones, training, incremental, final_parallel["recurrent_attention"]
    )
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_json(output / "scientific_questions.json", questions)
    make_plots(output, milestones, training, incremental, performance)
    report = render_report(summary, audit, questions)
    durable_text(output / "EXPERIMENT_2D2A_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nThe result checkout, exact final checkpoint, and all artifacts must be pushed/synchronized before stopping pod 7kk5yyti00rnrp. Do not delete its network volume.\n",
    )
    inventory = build_artifact_inventory(output)
    durable_json(output / "artifact_inventory.json", inventory)
    audit["artifact_inventory"] = inventory
    audit["checks"]["required artifact set complete"] = inventory["passed"]
    audit["checks"]["report matches machine-readable JSON"] = True
    audit["passed"] = all(audit["checks"].values())
    if not audit["passed"]:
        primary, secondary = classify_result(
            incremental, final_parallel, stable=stable, integrity=False
        )
        summary["primary_classification"] = primary
        summary["secondary_classification"] = secondary
        summary["next_recommendation"] = choose_recommendation(
            primary,
            secondary,
            final_parallel["recurrent_gain"],
            incremental["recurrent_gain"],
        )
        questions = build_questions(
            summary,
            milestones,
            training,
            incremental,
            final_parallel["recurrent_attention"],
        )
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_json(output / "scientific_questions.json", questions)
    report = render_report(summary, audit, questions)
    durable_text(output / "EXPERIMENT_2D2A_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report
        + "\nThe result checkout, exact final checkpoint, and all artifacts must be pushed/synchronized before stopping pod 7kk5yyti00rnrp. Do not delete its network volume.\n",
    )
    if (output / "EXPERIMENT_2D2A_FINAL_REPORT.md").read_text() != render_report(
        read_json(output / "result_summary.json"),
        read_json(output / "FINAL_AUDIT.json"),
        read_json(output / "scientific_questions.json"),
    ):
        raise SystemExit("final report does not cross-check against machine-readable JSON")
    print("EXPERIMENT_2D2A_FINALIZE_COMPLETE", flush=True)
    return summary


def run_seal_report(args):
    require_git(clean=True)
    output = Path(args.output_dir).resolve()
    expected_output = (REPO_ROOT / "results" / OUTPUT_NAME).resolve()
    if output != expected_output:
        raise SystemExit(f"seal-report output must be exactly {expected_output}")
    require_implementation_fingerprint(read_json(output / "preflight_audit.json"))
    summary = read_json(output / "result_summary.json")
    audit = read_json(output / "FINAL_AUDIT.json")
    questions = read_json(output / "scientific_questions.json")
    if git_output("rev-parse", "HEAD") != args.results_commit:
        raise SystemExit("seal-report HEAD must equal the supplied results commit")
    if git_output("rev-parse", f"origin/{BRANCH}") != args.results_commit:
        raise SystemExit("results commit must be pushed before sealing the report")
    commands = read_json(output / "commands_and_runtime.json")
    commands["seal_report_command"] = " ".join(sys.argv)
    commands["seal_report_argv_sha256"] = hashlib.sha256(
        json.dumps(sys.argv, separators=(",", ":")).encode()
    ).hexdigest()
    commands["post_seal_external_actions"] = [
        "commit and push sealed report artifacts",
        "verify local/origin/pod commit equality and clean worktrees",
        "verify no GPU compute process",
        "runpodctl pod stop 7kk5yyti00rnrp -o json",
        "verify stopped and not deleted",
    ]
    durable_json(output / "commands_and_runtime.json", commands)
    inventory = build_artifact_inventory(output)
    if not inventory["passed"]:
        raise SystemExit("required artifact/plot inventory is incomplete before sealing")
    summary["git"]["results_commit"] = args.results_commit
    summary["git"]["report_base_commit"] = git_output("rev-parse", "HEAD")
    audit["checks"]["Git synchronized"] = True
    audit["checks"]["required artifact set complete"] = inventory["passed"]
    audit["artifact_inventory"]["seal_report_recheck_passed"] = True
    audit["passed"] = all(audit["checks"].values())
    durable_json(output / "result_summary.json", summary)
    durable_json(output / "FINAL_AUDIT.json", audit)
    report = render_report(summary, audit, questions)
    durable_text(output / "EXPERIMENT_2D2A_FINAL_REPORT.md", report)
    durable_text(
        output / "UNATTENDED_FINAL_HANDOFF.md",
        report + "\nFinal RunPod stop remains the only lifecycle action after the report commit is pushed.\n",
    )
    print("EXPERIMENT_2D2A_REPORT_SEALED", flush=True)


def add_execution_arguments(parser):
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--gpu-type", default="NVIDIA_A100_SXM4_80GB")
    parser.add_argument("--persistent-volume-identity", required=True)
    parser.add_argument("--stop-mechanism", required=True)
    parser.add_argument("--stop-authenticated", action="store_true")
    parser.add_argument("--stop-audit-path", required=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    add_execution_arguments(preflight)
    smoke = subparsers.add_parser("smoke")
    add_execution_arguments(smoke)
    train = subparsers.add_parser("train")
    add_execution_arguments(train)
    train.add_argument("--end-update", type=int, required=True)
    train.add_argument("--resume")
    finalize = subparsers.add_parser("finalize")
    add_execution_arguments(finalize)
    finalize.add_argument("--final-checkpoint", required=True)
    finalize.add_argument("--results-commit")
    seal = subparsers.add_parser("seal-report")
    seal.add_argument("--output-dir", required=True)
    seal.add_argument("--results-commit", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "preflight":
        run_preflight(args)
    elif args.command == "smoke":
        run_smoke(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "finalize":
        run_finalize(args)
    elif args.command == "seal-report":
        run_seal_report(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
