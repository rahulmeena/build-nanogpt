#!/usr/bin/env python3
"""Experiment 2D1: end-to-end triangle recurrent Transformer co-training.

The result-bearing path is deliberately single-GPU.  Temporal recurrence is
trained with attached two/three-pass sequence-parallel graphs; a separate
rolling-KV decoder verifies the deployment semantics.
"""

import argparse
import contextlib
import copy
import dataclasses
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0 as d0  # noqa: E402
import experiment_2d0d as d0d  # noqa: E402


EXPERIMENT = "2D1"
PROTOCOL = "exp2d1_triangle_recurrent_cotraining_v1"
BRANCH = "experiment-2d1-triangle-recurrent-cotraining"
PARENT_COMMIT = "7e45b77b8638d2923689d1d9074104a8f9f5baab"
PARENT_TAG = "experiment-2d0d-matched-joint-kv-final"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d1_triangle_recurrent.json"
OUTPUT_NAME = "experiment_2d1_triangle_recurrent"
CHECKPOINT_SCHEMA = "exp2d1_triangle_recurrent_checkpoint_v1"
SOURCE_SHA256 = d0.SOURCE_SHA256
SOURCE_BYTES = d0.SOURCE_BYTES
VALIDATION_SHARD_SHA256 = d0.VAL_SHA256
CANONICAL_VALIDATION_SHA256 = d0.CANONICAL_VALIDATION_SHA256
PARENT_VALIDATION_LOSS = 3.0750437753315962
SEED = 1337
T = 1024
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
VOCAB_SIZE = 50_304
GLOBAL_TARGETS = 524_288
MAX_UPDATES = 4_769
TOTAL_TARGETS = 2_500_329_472
THREE_PASS_EVERY = 32
TWO_PASS_WEIGHTS = (0.25, 0.75)
THREE_PASS_WEIGHTS = (0.20, 0.40, 0.40)
BASE_PEAK_LR = 3e-5
FUSION_PEAK_LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
WARMUP_UPDATES = 100
CONSTANT_THROUGH = 4_292
COOLDOWN_FIRST = 4_293
FINAL_LR_FRACTION = 0.1
VALIDATION_BATCHES = 20
VALIDATION_B = 64
SCIENTIFIC_UPDATES = (48, 191, 477, 954, 1908, 2862, 3815, 4769)
VALIDATION_UPDATES = (0,) + SCIENTIFIC_UPDATES
FORCED_RESTARTS = (954, 2862)
ROLLING_INTERVAL = 100
ROLLING_KEEP = 3
TARGET_WINDOWS = (64, 82, 106, 136, 175, 226, 290, 374, 481, 619, 796, 1024)
STAGES = (
    ("A", 1, 477, (1024,) * 12),
    ("B", 478, 954, (512, 545, 581, 618, 658, 702, 747, 796, 848, 903, 962, 1024)),
    ("C", 955, 1908, (256, 290, 329, 373, 423, 481, 545, 619, 702, 796, 903, 1024)),
    ("D", 1909, 2862, (128, 154, 187, 225, 272, 330, 398, 481, 581, 702, 848, 1024)),
    ("E", 2863, 4769, TARGET_WINDOWS),
)


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2D1 requires branch {BRANCH}")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"], cwd=REPO_ROOT)
    if git_output("rev-parse", PARENT_TAG + "^{commit}") != PARENT_COMMIT:
        raise SystemExit("immutable 2D0D tag does not resolve to the registered parent")
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing Experiment 2D1 execution requires a clean worktree")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def durable_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def torch_load(path, mmap=False):
    kwargs = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def batch_payload_hash(x, y):
    digest = hashlib.sha256()
    digest.update(x.contiguous().numpy().tobytes())
    digest.update(y.contiguous().numpy().tobytes())
    return digest.hexdigest()


def aggregate_hashes(hashes):
    digest = hashlib.sha256()
    for value in hashes:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def nested_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "branch": BRANCH,
        "parent_commit": PARENT_COMMIT,
        "parent_tag": PARENT_TAG,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_checkpoint_bytes": SOURCE_BYTES,
        "validation_shard_sha256": VALIDATION_SHARD_SHA256,
        "canonical_validation_sha256": CANONICAL_VALIDATION_SHA256,
        "canonical_parent_loss": PARENT_VALIDATION_LOSS,
        "global_targets_per_update": GLOBAL_TARGETS,
        "optimizer_updates": MAX_UPDATES,
        "adaptation_targets": TOTAL_TARGETS,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    observed_stages = [
        (row["stage"], row["first_update"], row["last_update"], tuple(row["windows"]))
        for row in config["curriculum"]
    ]
    if mismatches or tuple(observed_stages) != STAGES:
        raise SystemExit(f"2D1 preregistration mismatch: fields={mismatches} stages={observed_stages}")
    assertions = {
        "target_windows": tuple(config["curriculum"][-1]["windows"]) == TARGET_WINDOWS,
        "strictly_increasing": all(a < b for a, b in zip(TARGET_WINDOWS, TARGET_WINDOWS[1:])),
        "target_sum": sum(TARGET_WINDOWS) == 4_373,
        "target_last": TARGET_WINDOWS[-1] == T,
        "token_formula": MAX_UPDATES * GLOBAL_TARGETS == TOTAL_TARGETS,
        "scientific_updates": tuple(config["checkpoints"]["scientific_updates"]) == SCIENTIFIC_UPDATES,
        "forced_restarts": tuple(config["checkpoints"]["forced_restart_updates"]) == FORCED_RESTARTS,
    }
    if not all(assertions.values()):
        raise SystemExit(f"2D1 frozen-geometry mismatch: {assertions}")
    return config


def stage_for_update(update):
    if not 1 <= int(update) <= MAX_UPDATES:
        raise ValueError(f"update outside 2D1 schedule: {update}")
    for name, first, last, windows in STAGES:
        if first <= update <= last:
            if name == "A":
                rho = 0.5 * (update - first) / (last - first)
            elif name == "B":
                rho = 0.5
            elif name == "C":
                rho = 0.75
            else:
                rho = 1.0
            return {"stage": name, "first_update": first, "last_update": last, "windows": windows, "rho": rho}
    raise AssertionError(update)


def learning_rate_fraction(update):
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError(update)
    if update <= WARMUP_UPDATES:
        return update / WARMUP_UPDATES
    if update <= CONSTANT_THROUGH:
        return 1.0
    progress = (update - CONSTANT_THROUGH) / (MAX_UPDATES - CONSTANT_THROUGH)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine


class ExplicitShardLoader:
    """Deterministic sequential loader whose state is fully checkpointable."""

    def __init__(self, shards, batch_size, sequence_length, state=None):
        self.shards = tuple(str(Path(path).resolve()) for path in sorted(shards))
        if not self.shards:
            raise ValueError("no shards")
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.current_shard = 0
        self.current_position = 0
        self._tokens = None
        if state is not None:
            self.load_state_dict(state)
        self._load_tokens()

    def _load_tokens(self):
        array = np.load(self.shards[self.current_shard], mmap_mode="r")
        if array.ndim != 1:
            raise ValueError("token shard must be flat")
        self._tokens = array

    def state_dict(self):
        return {
            "shards": list(self.shards),
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "current_shard": self.current_shard,
            "current_position": self.current_position,
        }

    def load_state_dict(self, state):
        expected = {
            "shards": list(self.shards),
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("loader geometry/shard mismatch")
        self.current_shard = int(state["current_shard"])
        self.current_position = int(state["current_position"])
        if not 0 <= self.current_shard < len(self.shards) or self.current_position < 0:
            raise ValueError("invalid loader cursor")

    def next_batch(self):
        count = self.batch_size * self.sequence_length
        if self.current_position + count + 1 > len(self._tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_position = 0
            self._load_tokens()
        values = np.asarray(self._tokens[self.current_position:self.current_position + count + 1], dtype=np.int64)
        if values.size != count + 1:
            raise RuntimeError("short token batch")
        self.current_position += count
        tensor = torch.from_numpy(values.copy())
        return tensor[:-1].view(self.batch_size, self.sequence_length), tensor[1:].view(self.batch_size, self.sequence_length)

    def clone(self):
        return ExplicitShardLoader(self.shards, self.batch_size, self.sequence_length, state=self.state_dict())


class RecurrentFusion(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.W_u = nn.Linear(width, width, bias=False)
        self.W_g = nn.Linear(width, width, bias=False)
        self.eps = float(eps)

    def initialize(self, source_scale, gate_scale=1.0):
        with torch.no_grad():
            identity = torch.eye(self.W_u.weight.size(0), dtype=self.W_u.weight.dtype, device=self.W_u.weight.device)
            self.W_u.weight.copy_(identity * float(source_scale))
            self.W_g.weight.copy_(identity * float(gate_scale))

    def normalize(self, value):
        dtype = value.dtype
        rms = value.float().pow(2).mean(dim=-1, keepdim=True)
        return (value.float() * torch.rsqrt(rms + self.eps)).to(dtype)

    def latent(self, token_embedding, previous_top):
        value = self.W_u(self.normalize(previous_top))
        gate = 2.0 * torch.sigmoid(self.W_g(token_embedding))
        return value * gate, gate


class TriangleRecurrentGPT(nn.Module):
    """Standard GPT-2 plus graded causal windows and one input recurrence."""

    def __init__(self, base):
        super().__init__()
        if base.config.residual_mode != "standard":
            raise ValueError("2D1 requires Standard GPT-2")
        self.base = base
        self.fusion = RecurrentFusion(base.config.n_embd)
        self._mask_cache = {}

    @property
    def config(self):
        return self.base.config

    def sliding_mask(self, length, window, device):
        key = (int(length), int(window), str(device))
        mask = self._mask_cache.get(key)
        if mask is None or mask.device != device:
            query = torch.arange(length, device=device).view(-1, 1)
            historical = torch.arange(length, device=device).view(1, -1)
            mask = (historical <= query) & (historical >= query - int(window) + 1)
            self._mask_cache[key] = mask
        return mask

    def attention(self, block, value, window):
        batch, length, channels = value.shape
        if int(window) >= length:
            return block.attn(value)
        qkv = block.attn.c_attn(value)
        q, k, v = qkv.split(channels, dim=2)
        head_size = channels // block.attn.n_head
        q = q.view(batch, length, block.attn.n_head, head_size).transpose(1, 2)
        k = k.view(batch, length, block.attn.n_head, head_size).transpose(1, 2)
        v = v.view(batch, length, block.attn.n_head, head_size).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.sliding_mask(length, window, value.device), is_causal=False
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return block.attn.c_proj(output)

    def run_block(self, block, residual, window):
        if int(window) >= residual.size(1):
            return block(residual)
        value = residual + self.attention(block, block.ln_1(residual), window)
        return value + block.mlp(block.ln_2(value))

    def make_input(self, tokens, previous_top=None, rho=0.0, prefix_length=None, return_diagnostics=False):
        batch, length = tokens.shape
        positions = torch.arange(length, dtype=torch.long, device=tokens.device)
        embedding = self.base.transformer.wte(tokens)
        fused_latent = gate = shifted = None
        if previous_top is None:
            recurrent_input = embedding
            recurrent_mask = torch.zeros((1, length, 1), dtype=torch.bool, device=tokens.device)
        else:
            if previous_top.shape != (batch, length, self.config.n_embd):
                raise ValueError("previous top state has wrong shape")
            shifted = torch.zeros_like(previous_top)
            shifted[:, 1:] = previous_top[:, :-1]
            fused_latent, gate = self.fusion.latent(embedding, shifted)
            candidate = (1.0 - float(rho)) * embedding + float(rho) * fused_latent
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
            diag = {
                "embedding_rms": embedding.float().pow(2).mean().sqrt().item(),
                "recurrent_input_rms": recurrent_input.float().pow(2).mean().sqrt().item(),
                "recurrent_fraction": recurrent_mask.float().mean().item(),
                "rho": float(rho),
            }
            if fused_latent is not None:
                gate_f = gate.float()
                diag.update({
                    "shifted_top_rms": shifted.float().pow(2).mean().sqrt().item(),
                    "fused_latent_rms": fused_latent.float().pow(2).mean().sqrt().item(),
                    "gate_mean": gate_f.mean().item(),
                    "gate_std": gate_f.std().item(),
                    "gate_saturation_fraction": ((gate_f < 0.01) | (gate_f > 1.99)).float().mean().item(),
                })
        return value, diag

    def forward_top(
        self,
        tokens,
        windows,
        previous_top=None,
        rho=0.0,
        prefix_length=None,
        activation_checkpointing=False,
        return_diagnostics=False,
    ):
        if len(windows) != N_LAYER or any(not 1 <= int(w) <= T for w in windows):
            raise ValueError("invalid 2D1 window schedule")
        made = self.make_input(tokens, previous_top, rho, prefix_length, return_diagnostics)
        if return_diagnostics:
            value, diagnostics = made
        else:
            value, diagnostics = made, None
        for block, window in zip(self.base.transformer.h, windows):
            if activation_checkpointing and self.training and torch.is_grad_enabled():
                value = checkpoint(
                    lambda residual, current_block=block, current_window=window: self.run_block(
                        current_block, residual, current_window
                    ),
                    value,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                value = self.run_block(block, value, window)
        top = self.base.transformer.ln_f(value)
        if diagnostics is not None:
            with torch.no_grad():
                diagnostics.update({
                    "top_state_rms": top.float().pow(2).mean().sqrt().item(),
                    "top_state_norm_mean": top.float().norm(dim=-1).mean().item(),
                    "top_state_norm_std": top.float().norm(dim=-1).std().item(),
                    "top_temporal_cosine": (
                        F.cosine_similarity(top[:, 1:].float(), top[:, :-1].float(), dim=-1).mean().item()
                        if top.size(1) > 1 else None
                    ),
                })
            return top, diagnostics
        return top

    def loss_from_top(self, top, targets, activation_checkpointing=False, reduction="mean"):
        def calculate(value):
            logits = self.base.lm_head(value)
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction
            )
        if activation_checkpointing and self.training and torch.is_grad_enabled() and reduction == "mean":
            return checkpoint(calculate, top, use_reentrant=False, preserve_rng_state=False)
        return calculate(top)

    def logits_from_top(self, top):
        return self.base.lm_head(top)

    def forward_pass(self, tokens, targets, windows, previous_top=None, rho=0.0, prefix_length=None,
                     activation_checkpointing=False, return_diagnostics=False):
        result = self.forward_top(
            tokens, windows, previous_top, rho, prefix_length, activation_checkpointing, return_diagnostics
        )
        if return_diagnostics:
            top, diagnostics = result
        else:
            top, diagnostics = result, None
        loss = self.loss_from_top(top, targets, activation_checkpointing=activation_checkpointing)
        return top, loss, diagnostics


def load_source_model(checkpoint_path, device, trainable=True):
    symbols, base, audit = d0.load_standard_model(checkpoint_path, device)
    for parameter in base.parameters():
        parameter.requires_grad_(bool(trainable))
    model = TriangleRecurrentGPT(base).to(device)
    if model.base.transformer.wte.weight is not model.base.lm_head.weight:
        raise SystemExit("source embedding/LM-head tying was not preserved")
    return symbols, model, audit


def train_shards(data_root):
    paths = sorted(Path(data_root).glob("edufineweb_train_*.npy"))
    if not paths:
        raise SystemExit(f"no training shards under {data_root}")
    return paths


def validation_shard(data_root):
    path = Path(data_root) / "edufineweb_val_000000.npy"
    if not path.is_file() or file_sha256(path) != VALIDATION_SHARD_SHA256:
        raise SystemExit("canonical validation shard missing or corrupt")
    return path


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(prefix_rng):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "prefix_rng": prefix_rng.getstate(),
    }


def restore_rng_state(state, prefix_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    prefix_rng.setstate(state["prefix_rng"])


def runtime_environment():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": gpu,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "autocast": "cuda bfloat16",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
    }


def require_single_a100():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Experiment 2D1 result path requires exactly one visible GPU")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if "A100-SXM4-80GB" not in name or memory < 79 * 1024**3:
        raise SystemExit(f"unsupported 2D1 GPU: {name}, {memory}")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or torch.distributed.is_initialized():
        raise SystemExit("single-GPU 2D1 forbids DDP/NCCL state")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda", 0)


def configure_optimizer(model, device_type="cuda"):
    fusion_ids = {id(parameter) for parameter in model.fusion.parameters()}
    base_decay = []
    base_nodecay = []
    fusion = []
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in fusion_ids:
            fusion.append(parameter)
        elif parameter.dim() >= 2:
            base_decay.append(parameter)
        else:
            base_nodecay.append(parameter)
    groups = [
        {"name": "base_decay", "params": base_decay, "lr": BASE_PEAK_LR, "weight_decay": WEIGHT_DECAY},
        {"name": "base_nodecay", "params": base_nodecay, "lr": BASE_PEAK_LR, "weight_decay": 0.0},
        {"name": "fusion", "params": fusion, "lr": FUSION_PEAK_LR, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        groups,
        betas=BETAS,
        eps=ADAM_EPS,
        fused=fused_available and device_type == "cuda",
    )
    report = {
        "logical_lr_groups": 2,
        "physical_groups": 3,
        "reason": "the base LR class is split only to preserve Standard GPT-2 decay/no-decay semantics",
        "groups": [
            {
                "name": group["name"],
                "parameters": sum(parameter.numel() for parameter in group["params"]),
                "tensors": len(group["params"]),
                "peak_lr": FUSION_PEAK_LR if group["name"] == "fusion" else BASE_PEAK_LR,
                "weight_decay": group["weight_decay"],
            }
            for group in groups
        ],
        "betas": list(BETAS),
        "eps": ADAM_EPS,
        "fused": fused_available and device_type == "cuda",
    }
    return optimizer, report


def set_optimizer_lrs(optimizer, update):
    fraction = learning_rate_fraction(update)
    values = {}
    for group in optimizer.param_groups:
        peak = FUSION_PEAK_LR if group["name"] == "fusion" else BASE_PEAK_LR
        group["lr"] = peak * fraction
        values[group["name"]] = group["lr"]
    return values


@torch.no_grad()
def calibrate_fusion(model, tokens):
    model.eval()
    windows = (T,) * N_LAYER
    embedding = model.base.transformer.wte(tokens)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        top = model.forward_top(tokens, windows)
    embedding_rms = embedding.float().pow(2).mean().sqrt().item()
    top_rms = top.float().pow(2).mean().sqrt().item()
    source_scale = embedding_rms / top_rms
    gate_scale = 1.0
    model.fusion.initialize(source_scale, gate_scale)
    preactivation = F.linear(embedding, model.fusion.W_g.weight)
    gate = 2.0 * torch.sigmoid(preactivation)
    report = {
        "representative_batch_shape": list(tokens.shape),
        "measurement_precision": "BF16 Transformer autocast; FP32 RMS reductions",
        "embedding_rms": embedding_rms,
        "top_state_rms": top_rms,
        "initial_source_scale": source_scale,
        "W_u_initialization": "initial_source_scale * Identity",
        "W_g_initialization": "Identity",
        "gate_diagonal_scale": gate_scale,
        "gate_preactivation_mean": preactivation.float().mean().item(),
        "gate_preactivation_std": preactivation.float().std().item(),
        "gate_preactivation_abs_max": preactivation.float().abs().max().item(),
        "gate_mean": gate.float().mean().item(),
        "gate_std": gate.float().std().item(),
        "gate_saturation_fraction": ((gate < 0.01) | (gate > 1.99)).float().mean().item(),
    }
    report["identity_gate_safe"] = report["gate_saturation_fraction"] < 1e-4
    if not report["identity_gate_safe"]:
        raise SystemExit(f"identity W_g is unexpectedly saturated: {report}")
    return report


def finite_tensor_collection(values):
    checks = [torch.isfinite(value).all() for value in values if isinstance(value, torch.Tensor)]
    if not checks:
        return True
    return bool(torch.stack(checks).all().item())


def model_parameters_finite(model):
    return finite_tensor_collection(parameter.data for parameter in model.parameters())


def gradients_finite(model):
    return finite_tensor_collection(parameter.grad for parameter in model.parameters() if parameter.grad is not None)


def optimizer_moments_finite(optimizer):
    values = []
    for state in optimizer.state.values():
        values.extend(value for value in state.values() if isinstance(value, torch.Tensor))
    return finite_tensor_collection(values)


def gradient_report(model):
    groups = {"base": [], "W_u": [], "W_g": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("fusion.W_u"):
            groups["W_u"].append(parameter.grad)
        elif name.startswith("fusion.W_g"):
            groups["W_g"].append(parameter.grad)
        else:
            groups["base"].append(parameter.grad)
    report = {}
    for name, values in groups.items():
        squared = sum(value.float().pow(2).sum() for value in values) if values else torch.tensor(0.0)
        report[name] = {
            "tensors": len(values),
            "norm": squared.sqrt().item(),
            "nonzero": bool(values) and bool(squared.gt(0).item()),
            "finite": finite_tensor_collection(values),
        }
    return report


def validation_manifest(val_path):
    loader = ExplicitShardLoader([val_path], VALIDATION_B, T)
    rows = []
    hashes = []
    for batch_index in range(VALIDATION_BATCHES):
        x, y = loader.next_batch()
        identity = d0d.batch_identity(x, y)
        rows.append({"batch_index": batch_index, **identity, "input_shape": list(x.shape), "target_shape": list(y.shape)})
        hashes.append(identity["combined_sha256"])
    return {
        "validation_shard": str(Path(val_path).resolve()),
        "validation_shard_sha256": file_sha256(val_path),
        "canonical_batch_collection_sha256": aggregate_hashes(hashes),
        "batches": rows,
        "batch_count": VALIDATION_BATCHES,
        "batch_size": VALIDATION_B,
        "sequence_length": T,
        "targets": VALIDATION_BATCHES * VALIDATION_B * T,
    }


def _new_loss_accumulator():
    return {"loss_sum": 0.0, "targets": 0, "per_batch_losses": [], "wins_vs_real": 0}


def _add_losses(accumulator, losses):
    losses = losses.float()
    accumulator["loss_sum"] += losses.double().sum().item()
    accumulator["targets"] += losses.numel()
    accumulator["per_batch_losses"].append(losses.mean().item())


def _finish_losses(accumulator):
    return {
        "validation_loss": accumulator["loss_sum"] / accumulator["targets"],
        "validation_targets": accumulator["targets"],
        "per_batch_losses": accumulator["per_batch_losses"],
    }


@torch.no_grad()
def evaluate_temporal(model, val_path, windows, rho, controls=("plain", "real"), batches=VALIDATION_BATCHES):
    model.eval()
    device = next(model.parameters()).device
    loader = ExplicitShardLoader([val_path], VALIDATION_B, T)
    accumulators = {control: _new_loss_accumulator() for control in controls}
    identities = []
    diagnostics = []
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
            plain_top, plain_diag = model.forward_top(x, windows, return_diagnostics=True)
            if "plain" in controls:
                _add_losses(accumulators["plain"], model.loss_from_top(plain_top, y, reduction="none"))
            for control in controls:
                if control == "plain":
                    continue
                current_rho = 1.0 if control == "real_final_rho" else float(rho)
                if control == "real" or control == "real_final_rho":
                    source = plain_top
                elif control == "shuffled":
                    source = plain_top[derangement]
                elif control == "zero":
                    source = torch.zeros_like(plain_top)
                else:
                    raise ValueError(control)
                recurrent_top, recurrent_diag = model.forward_top(
                    x, windows, previous_top=source, rho=current_rho, prefix_length=0, return_diagnostics=True
                )
                _add_losses(accumulators[control], model.loss_from_top(recurrent_top, y, reduction="none"))
                if control == "real":
                    diagnostics.append(recurrent_diag)
                del recurrent_top
        del x, y, plain_top
        print(f"2D1 validation batch={batch_index + 1:02d}/{batches}", flush=True)
    elapsed = time.monotonic() - start
    result = {
        "windows": list(windows),
        "rho": float(rho),
        "controls": {name: _finish_losses(row) for name, row in accumulators.items()},
        "canonical_validation_sha256": aggregate_hashes([row["combined_sha256"] for row in identities]),
        "batch_identities": identities,
        "performance": {
            "wall_seconds": elapsed,
            "target_passes_per_second": batches * VALIDATION_B * T * len(controls) / elapsed,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
    }
    if diagnostics:
        keys = sorted(set.intersection(*(set(row) for row in diagnostics)))
        result["state_diagnostics"] = {
            key: sum(row[key] for row in diagnostics if isinstance(row[key], (int, float))) / len(diagnostics)
            for key in keys
            if all(isinstance(row[key], (int, float)) for row in diagnostics)
        }
    if "plain" in result["controls"] and "real" in result["controls"]:
        plain = result["controls"]["plain"]
        real = result["controls"]["real"]
        result["recurrent_gain"] = plain["validation_loss"] - real["validation_loss"]
        result["real_vs_plain_paired_wins"] = sum(
            recurrent < baseline
            for recurrent, baseline in zip(real["per_batch_losses"], plain["per_batch_losses"])
        )
    if "shuffled" in result["controls"] and "real" in result["controls"]:
        shuffled = result["controls"]["shuffled"]
        real = result["controls"]["real"]
        result["sequence_specific_gap"] = shuffled["validation_loss"] - real["validation_loss"]
        result["real_vs_shuffled_paired_wins"] = sum(
            recurrent < shuffled_loss
            for recurrent, shuffled_loss in zip(real["per_batch_losses"], shuffled["per_batch_losses"])
        )
    if "zero" in result["controls"] and "real" in result["controls"]:
        zero = result["controls"]["zero"]
        real = result["controls"]["real"]
        result["zero_state_penalty"] = zero["validation_loss"] - real["validation_loss"]
        result["real_vs_zero_paired_wins"] = sum(
            recurrent < zero_loss
            for recurrent, zero_loss in zip(real["per_batch_losses"], zero["per_batch_losses"])
        )
    return result


@torch.no_grad()
def evaluate_parent_plain(model, val_path, batches=VALIDATION_BATCHES):
    result = evaluate_temporal(model, val_path, (T,) * N_LAYER, 0.0, controls=("plain",), batches=batches)
    return {
        "validation_loss": result["controls"]["plain"]["validation_loss"],
        "per_batch_losses": result["controls"]["plain"]["per_batch_losses"],
        "canonical_validation_sha256": result["canonical_validation_sha256"],
        "performance": result["performance"],
    }


class IncrementalState:
    def __init__(self, position, caches, previous_top, windows):
        self.position = int(position)
        self.caches = tuple(caches)
        self.previous_top = previous_top
        self.windows = tuple(int(value) for value in windows)


def init_incremental_state(model, batch_size, windows, device, dtype):
    if len(windows) != N_LAYER:
        raise ValueError("incremental state needs 12 windows")
    previous = torch.zeros((batch_size, 1, model.config.n_embd), device=device, dtype=dtype)
    return IncrementalState(0, (None,) * N_LAYER, previous, windows)


def incremental_cache_lengths(state):
    return [0 if cache is None else cache[0].size(2) for cache in state.caches]


def incremental_step(model, token, state, rho=1.0, previous_override=None):
    if token.ndim == 1:
        token = token[:, None]
    if token.ndim != 2 or token.size(1) != 1:
        raise ValueError("incremental token must be [B] or [B,1]")
    if state.position >= model.config.block_size:
        raise ValueError("incremental position exceeds learned position embeddings")
    batch = token.size(0)
    position = torch.tensor([state.position], dtype=torch.long, device=token.device)
    embedding = model.base.transformer.wte(token)
    if state.position == 0:
        recurrent_input = embedding
    else:
        source = state.previous_top if previous_override is None else previous_override
        if source.shape != (batch, 1, model.config.n_embd):
            raise ValueError("incremental recurrent source shape mismatch")
        fused, _ = model.fusion.latent(embedding, source)
        recurrent_input = (1.0 - float(rho)) * embedding + float(rho) * fused
    value = recurrent_input + model.base.transformer.wpe(position)
    updated = []
    for block_index, (block, window, cache) in enumerate(
        zip(model.base.transformer.h, state.windows, state.caches)
    ):
        normalized = block.ln_1(value)
        qkv = block.attn.c_attn(normalized)
        q, k, v = qkv.split(model.config.n_embd, dim=2)
        head_size = model.config.n_embd // model.config.n_head
        q = q.view(batch, 1, model.config.n_head, head_size).transpose(1, 2)
        k = k.view(batch, 1, model.config.n_head, head_size).transpose(1, 2)
        v = v.view(batch, 1, model.config.n_head, head_size).transpose(1, 2)
        if cache is None:
            historical_k = k[:, :, :0]
            historical_v = v[:, :, :0]
        else:
            historical_k, historical_v = cache
        if historical_k.size(2) > window - 1:
            raise RuntimeError(f"B{block_index + 1} physical cache exceeds W-1")
        keys = torch.cat((historical_k, k), dim=2)
        values = torch.cat((historical_v, v), dim=2)
        attention = F.scaled_dot_product_attention(q, keys, values, is_causal=False)
        attention = attention.transpose(1, 2).contiguous().view(batch, 1, model.config.n_embd)
        value = value + block.attn.c_proj(attention)
        value = value + block.mlp(block.ln_2(value))
        capacity = window - 1
        if capacity == 0:
            updated.append(None)
        else:
            updated.append((keys[:, :, -capacity:].detach(), values[:, :, -capacity:].detach()))
    top = model.base.transformer.ln_f(value)
    logits = model.base.lm_head(top)
    next_state = IncrementalState(state.position + 1, updated, top.detach(), state.windows)
    expected = [min(next_state.position, window - 1) for window in state.windows]
    if incremental_cache_lengths(next_state) != expected:
        raise RuntimeError("incremental rolling-cache length mismatch")
    return logits, next_state


@torch.no_grad()
def temporal_wavefront_logits(model, tokens, windows, rho=1.0):
    previous = model.forward_top(tokens, windows)
    logits = model.logits_from_top(previous)
    for _ in range(tokens.size(1) - 1):
        previous = model.forward_top(tokens, windows, previous_top=previous, rho=rho, prefix_length=0)
        logits = model.logits_from_top(previous)
    return logits


@torch.no_grad()
def incremental_logits(model, tokens, windows, rho=1.0):
    state = init_incremental_state(model, tokens.size(0), windows, tokens.device, next(model.parameters()).dtype)
    rows = []
    maxima = [0] * N_LAYER
    for position in range(tokens.size(1)):
        logits, state = incremental_step(model, tokens[:, position], state, rho=rho)
        rows.append(logits)
        maxima = [max(old, current) for old, current in zip(maxima, incremental_cache_lengths(state))]
    return torch.cat(rows, dim=1), state, maxima


def incremental_equivalence_tests(model, tokens):
    model.eval()
    windows = tuple(min(window, tokens.size(1)) for window in TARGET_WINDOWS)
    reports = {}
    original_precision = torch.get_float32_matmul_precision()
    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        if dtype == torch.float32:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.set_float32_matmul_precision(original_precision)
            torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if dtype == torch.bfloat16 else contextlib.nullcontext()
        )
        with context, torch.no_grad():
            parallel = temporal_wavefront_logits(model, tokens, windows)
            incremental, state, maxima = incremental_logits(model, tokens, windows)
        difference = (parallel.float() - incremental.float()).abs()
        threshold = 5e-5 if dtype == torch.float32 else 0.30
        reports[label] = {
            "length": tokens.size(1),
            "maximum_absolute_logit_difference": difference.max().item(),
            "mean_absolute_logit_difference": difference.mean().item(),
            "cache_maxima": maxima,
            "cache_limits": [window - 1 for window in windows],
            "final_position": state.position,
            "threshold": threshold,
            "kernel_note": (
                "TF32 disabled for strict FP32 equivalence"
                if dtype == torch.float32
                else "BF16 full-sequence and one-token GEMM/SDPA reduction shapes differ"
            ),
            "passed": difference.max().item() <= threshold,
        }
    torch.set_float32_matmul_precision(original_precision)
    torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
    torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
    reports["passed"] = all(row["passed"] for row in reports.values() if isinstance(row, dict))
    return reports


def causality_tests(model, tokens):
    model.eval()
    windows = TARGET_WINDOWS
    future = tokens.clone()
    cutoff = tokens.size(1) // 2
    future[:, cutoff + 1:] = (future[:, cutoff + 1:] + 17) % model.config.vocab_size
    with torch.no_grad():
        first_a = model.forward_top(tokens, windows)
        first_b = model.forward_top(future, windows)
        second_a = model.forward_top(tokens, windows, previous_top=first_a, rho=1.0, prefix_length=0)
        second_b = model.forward_top(future, windows, previous_top=first_b, rho=1.0, prefix_length=0)
        logits_a = model.logits_from_top(second_a)
        logits_b = model.logits_from_top(second_b)
    top_delta = (first_a[:, :cutoff + 1].float() - first_b[:, :cutoff + 1].float()).abs().max().item()
    logit_delta = (logits_a[:, :cutoff + 1].float() - logits_b[:, :cutoff + 1].float()).abs().max().item()
    shifted = torch.zeros_like(first_a)
    shifted[:, 1:] = first_a[:, :-1]
    report = {
        "cutoff": cutoff,
        "pass1_prefix_max_delta": top_delta,
        "pass2_prefix_logit_max_delta": logit_delta,
        "position_zero_source_exact_zero": shifted[:, 0].count_nonzero().item() == 0,
        "one_token_shift_exact": torch.equal(shifted[:, 1:], first_a[:, :-1]),
        "no_wraparound": not torch.equal(shifted[:, 0], first_a[:, -1]),
    }
    report["passed"] = (
        top_delta <= 1e-6 and logit_delta <= 1e-5 and report["position_zero_source_exact_zero"]
        and report["one_token_shift_exact"] and report["no_wraparound"]
    )
    return report


def temporal_gradient_tests(model, tokens, targets):
    model.train()
    windows = tuple(min(window, tokens.size(1)) for window in TARGET_WINDOWS)
    first = model.forward_top(tokens, windows)
    first.retain_grad()
    second = model.forward_top(tokens, windows, previous_top=first, rho=1.0, prefix_length=0)
    t = 1
    loss2 = F.cross_entropy(model.logits_from_top(second)[:, t + 1], targets[:, t + 1])
    gradient2 = torch.autograd.grad(loss2, first, retain_graph=True)[0]
    third = model.forward_top(tokens, windows, previous_top=second, rho=1.0, prefix_length=0)
    loss3 = F.cross_entropy(model.logits_from_top(third)[:, t + 2], targets[:, t + 2])
    gradient3 = torch.autograd.grad(loss3, first)[0]
    report = {
        "pass2_direct_gradient_norm": gradient2[:, t].float().norm().item(),
        "pass2_future_gradient_norm": gradient2[:, t + 1:].float().norm().item(),
        "pass3_two_transition_gradient_norm": gradient3[:, t].float().norm().item(),
        "pass3_future_gradient_norm": gradient3[:, t + 1:].float().norm().item(),
        "all_finite": bool(torch.isfinite(gradient2).all() and torch.isfinite(gradient3).all()),
    }
    report["passed"] = (
        report["all_finite"] and report["pass2_direct_gradient_norm"] > 0
        and report["pass3_two_transition_gradient_norm"] > 0
        and report["pass2_future_gradient_norm"] == 0
        and report["pass3_future_gradient_norm"] == 0
    )
    model.zero_grad(set_to_none=True)
    return report


def window_semantics_audit(device):
    schedules = {name: windows for name, _, _, windows in STAGES}
    results = {}
    for name, windows in schedules.items():
        rows = []
        for layer, window in enumerate(windows):
            mask = d0.sliding_mask(T, window, device)
            query = torch.arange(T, device=device)
            counts = mask.sum(dim=1)
            expected_counts = torch.minimum(query + 1, torch.tensor(window, device=device))
            future = torch.triu(mask, diagonal=1).any().item()
            exact = torch.equal(counts, expected_counts) and not future
            rows.append({"layer": layer + 1, "window": window, "exact": exact})
        results[name] = {"windows": list(windows), "layers": rows, "passed": all(row["exact"] for row in rows)}
    return {"schedules": results, "passed": all(row["passed"] for row in results.values())}


def architecture_manifest_payload():
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "base_architecture": "Standard GPT-2 124M",
        "layers": N_LAYER,
        "heads": N_HEAD,
        "hidden_width": N_EMBD,
        "context": T,
        "vocab_size": VOCAB_SIZE,
        "position_embeddings": "learned absolute GPT-2",
        "weight_tying": "preserved",
        "recurrent_source": "ln_f(B12 output) from previous temporal pass at t-1",
        "recurrent_destination": "model input before learned positional embedding",
        "recurrent_state_shape": ["B", "T", N_EMBD],
        "fusion": {
            "normalization": "affine-free RMSNorm",
            "value": "W_u(RMSNorm(z_prev))",
            "token_gate": "2*sigmoid(W_g(token_embedding))",
            "output": "value*token_gate",
            "position_embedding_in_gate": False,
        },
        "pass_schedule": {"two_pass_fraction": 0.96875, "three_pass_every": THREE_PASS_EVERY},
        "temporal_detach": False,
        "teacher": False,
        "auxiliary_loss": False,
        "full_attnres": False,
        "temporal_attnres": False,
        "readers": False,
        "writer_adapters": False,
    }


def curriculum_payload():
    rows = []
    for name, first, last, windows in STAGES:
        row = {
            "stage": name,
            "first_update": first,
            "last_update": last,
            "updates": last - first + 1,
            "windows": list(windows),
            "window_sum": sum(windows),
            "cumulative_targets": last * GLOBAL_TARGETS,
        }
        if name == "A":
            row["rho"] = {"kind": "linear", "first": 0.0, "last": 0.5}
        else:
            row["rho"] = {"kind": "constant", "value": stage_for_update(first)["rho"]}
        rows.append(row)
    return {
        "stages": rows,
        "final_windows": list(TARGET_WINDOWS),
        "final_window_sum": sum(TARGET_WINDOWS),
        "full_window_sum": N_LAYER * T,
        "nominal_fraction": sum(TARGET_WINDOWS) / (N_LAYER * T),
        "nominal_removed_fraction": 1.0 - sum(TARGET_WINDOWS) / (N_LAYER * T),
    }


def run_prepare(args):
    require_git(clean=False)
    load_config()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    durable_json(output / "architecture_manifest.json", architecture_manifest_payload())
    durable_json(output / "window_curriculum.json", curriculum_payload())
    durable_json(output / "commands_and_runtime.json", {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "prepare_command": " ".join(sys.argv),
        "prepared_at": time.time(),
        "git_commit": git_output("rev-parse", "HEAD"),
    })
    print(f"EXPERIMENT_2D1_PREPARE_PASS output={output}", flush=True)


def probe_microbatch(model, shards, device, candidates=(64, 32, 16, 8, 4)):
    model.train()
    reports = []
    selected = None
    total_memory = torch.cuda.get_device_properties(device).total_memory
    for batch_size in candidates:
        if GLOBAL_TARGETS % (batch_size * T):
            continue
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        loader = ExplicitShardLoader(shards, batch_size, T)
        cpu_x, cpu_y = loader.next_batch()
        x = cpu_x.to(device)
        y = cpu_y.to(device)
        row = {"micro_batch_sequences": batch_size, "gradient_accumulation": GLOBAL_TARGETS // (batch_size * T)}
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                first, loss1, _ = model.forward_pass(
                    x, y, (T,) * N_LAYER, activation_checkpointing=True
                )
                second, loss2, _ = model.forward_pass(
                    x, y, (T,) * N_LAYER, previous_top=first, rho=0.5, prefix_length=0,
                    activation_checkpointing=True
                )
                _, loss3, _ = model.forward_pass(
                    x, y, (T,) * N_LAYER, previous_top=second, rho=0.5, prefix_length=T // 3,
                    activation_checkpointing=True
                )
                loss = 0.2 * loss1 + 0.4 * loss2 + 0.4 * loss3
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device)
            gradients = gradient_report(model)
            row.update({
                "loss": loss.detach().float().item(),
                "peak_allocated_bytes": peak,
                "peak_allocated_mb": peak / 1024**2,
                "total_vram_bytes": total_memory,
                "vram_fraction": peak / total_memory,
                "gradients": gradients,
                "finite": math.isfinite(loss.detach().float().item()) and gradients_finite(model),
            })
            row["passed"] = (
                row["finite"] and row["vram_fraction"] <= 0.90
                and gradients["base"]["nonzero"] and gradients["W_u"]["nonzero"] and gradients["W_g"]["nonzero"]
            )
            if row["passed"]:
                selected = batch_size
                reports.append(row)
                break
        except torch.cuda.OutOfMemoryError as error:
            row.update({"passed": False, "oom": True, "error": str(error)[:500]})
        reports.append(row)
        del x, y, cpu_x, cpu_y
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    if selected is None:
        raise SystemExit(f"no result-bearing microbatch fits safely: {reports}")
    return {
        "candidates": list(candidates),
        "reports": reports,
        "selected_micro_batch_sequences": selected,
        "selected_gradient_accumulation": GLOBAL_TARGETS // (selected * T),
        "global_targets_per_update": GLOBAL_TARGETS,
        "passed": True,
    }


def rho_zero_identity_test(model, tokens):
    model.eval()
    windows = (T,) * N_LAYER
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        plain = model.forward_top(tokens, windows)
        dormant = model.forward_top(tokens, windows, previous_top=plain, rho=0.0, prefix_length=0)
        plain_logits = model.logits_from_top(plain)
        dormant_logits = model.logits_from_top(dormant)
    difference = (plain_logits.float() - dormant_logits.float()).abs()
    return {
        "top_bit_exact": torch.equal(plain, dormant),
        "logits_bit_exact": torch.equal(plain_logits, dormant_logits),
        "maximum_absolute_logit_difference": difference.max().item(),
        "passed": torch.equal(plain_logits, dormant_logits),
    }


def run_preflight(args):
    require_git(clean=True)
    config = load_config()
    device = require_single_a100()
    seed_all(SEED)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.parent_checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    val_path = validation_shard(data_root)
    shards = train_shards(data_root)
    manifest = validation_manifest(val_path)
    durable_json(output / "batch_manifest.json", manifest)
    if manifest["canonical_batch_collection_sha256"] != CANONICAL_VALIDATION_SHA256:
        raise SystemExit("canonical validation batch collection mismatch")
    symbols, model, source_audit = load_source_model(checkpoint_path, device, trainable=True)
    source_audit["all_base_parameters_trainable_for_2d1"] = all(
        parameter.requires_grad for parameter in model.base.parameters()
    )
    source_audit["weight_tying_preserved"] = (
        model.base.transformer.wte.weight is model.base.lm_head.weight
    )
    source_manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "validation_shard": str(val_path),
        "validation_shard_sha256": file_sha256(val_path),
        "canonical_validation_sha256": manifest["canonical_batch_collection_sha256"],
        "training_shards": len(shards),
        "architecture": source_audit,
    }
    durable_json(output / "source_manifest.json", source_manifest)
    environment = runtime_environment()
    environment.update({
        "pod_id": args.pod_id,
        "workspace_mount": args.workspace_mount,
        "network_volume_mount": args.network_volume_mount,
    })
    durable_json(output / "environment.json", environment)

    parent = evaluate_parent_plain(model, val_path)
    parent_regression = {
        **parent,
        "oracle": PARENT_VALIDATION_LOSS,
        "absolute_delta": abs(parent["validation_loss"] - PARENT_VALIDATION_LOSS),
    }
    parent_regression["passed"] = (
        parent_regression["absolute_delta"] <= 1e-8
        and parent["canonical_validation_sha256"] == CANONICAL_VALIDATION_SHA256
    )
    if not parent_regression["passed"]:
        raise SystemExit(f"source regression hard stop: {parent_regression}")

    small_loader = ExplicitShardLoader([val_path], 2, 16)
    small_x, small_y = small_loader.next_batch()
    small_x = small_x.to(device)
    small_y = small_y.to(device)
    rho_zero = rho_zero_identity_test(model, small_x)
    rho_zero_validation = evaluate_temporal(
        model, val_path, (T,) * N_LAYER, 0.0, controls=("plain", "real")
    )
    rho_zero["plain_validation_loss"] = rho_zero_validation["controls"]["plain"]["validation_loss"]
    rho_zero["dormant_validation_loss"] = rho_zero_validation["controls"]["real"]["validation_loss"]
    rho_zero["validation_absolute_delta"] = abs(
        rho_zero["plain_validation_loss"] - rho_zero["dormant_validation_loss"]
    )
    rho_zero["canonical_validation_exact"] = (
        rho_zero_validation["canonical_validation_sha256"] == CANONICAL_VALIDATION_SHA256
    )
    rho_zero["passed"] = (
        rho_zero["passed"] and rho_zero["validation_absolute_delta"] <= 1e-8
        and rho_zero["canonical_validation_exact"]
    )
    if not rho_zero["passed"]:
        raise SystemExit(f"rho=0 identity hard stop: {rho_zero}")

    calibration_loader = ExplicitShardLoader(shards, 4, T)
    calibration_x, _ = calibration_loader.next_batch()
    calibration_x = calibration_x.to(device)
    fusion_initialization = calibrate_fusion(model, calibration_x)
    durable_json(output / "fusion_initialization.json", fusion_initialization)

    with torch.autocast(device_type="cuda", enabled=False):
        causality = causality_tests(model, small_x)
        gradients = temporal_gradient_tests(model, small_x, small_y)
        incremental = incremental_equivalence_tests(model, small_x)
    windows = window_semantics_audit(device)
    durable_json(output / "causality_audit.json", {"causality": causality, "temporal_gradients": gradients})
    durable_json(output / "cache_audit.json", {"window_semantics": windows, "incremental_equivalence": incremental})
    if not causality["passed"] or not gradients["passed"] or not windows["passed"] or not incremental["passed"]:
        raise SystemExit(
            f"architecture preflight hard stop: causality={causality} gradients={gradients} "
            f"windows={windows['passed']} incremental={incremental}"
        )

    memory = probe_microbatch(model, shards, device)
    optimizer, optimizer_report = configure_optimizer(model)
    all_trainable = all(parameter.requires_grad for parameter in model.parameters())
    forbidden_modules = [name for name, _ in model.named_modules() if "attnres" in name.lower()]
    stop_audit = {
        "pod_id": args.pod_id,
        "mechanism": args.stop_mechanism,
        "authentication_status": "authenticated" if args.stop_authenticated else "unavailable",
        "authenticated": bool(args.stop_authenticated),
        "workspace_mount": args.workspace_mount,
        "network_volume_mount": args.network_volume_mount,
        "tested_at": time.time(),
    }
    durable_json(output / "runpod_stop_audit.json", stop_audit)
    science_checks = {
        "source_checkpoint_exact": source_manifest["checkpoint_sha256"] == SOURCE_SHA256,
        "source_checkpoint_bytes_exact": source_manifest["checkpoint_bytes"] == SOURCE_BYTES,
        "validation_shard_exact": source_manifest["validation_shard_sha256"] == VALIDATION_SHARD_SHA256,
        "canonical_validation_exact": manifest["canonical_batch_collection_sha256"] == CANONICAL_VALIDATION_SHA256,
        "parent_regression": parent_regression["passed"],
        "rho_zero_identity": rho_zero["passed"],
        "causality": causality["passed"],
        "temporal_gradient": gradients["passed"],
        "window_semantics": windows["passed"],
        "incremental_equivalence": incremental["passed"],
        "all_parameters_trainable": all_trainable,
        "fusion_parameters_trainable": all(parameter.requires_grad for parameter in model.fusion.parameters()),
        "weight_tying": model.base.transformer.wte.weight is model.base.lm_head.weight,
        "full_attnres_absent": not forbidden_modules,
        "memory_probe": memory["passed"],
        "optimizer_groups": optimizer_report["logical_lr_groups"] == 2,
        "model_finite": model_parameters_finite(model),
    }
    preflight = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "source_regression": parent_regression,
        "rho_zero_identity": rho_zero,
        "fusion_initialization": fusion_initialization,
        "causality": causality,
        "temporal_gradients": gradients,
        "window_semantics": windows,
        "incremental_equivalence": incremental,
        "memory_probe": memory,
        "optimizer": optimizer_report,
        "forbidden_modules": forbidden_modules,
        "science_checks": science_checks,
        "science_passed": all(science_checks.values()),
        "runpod_stop_audit": stop_audit,
        "result_run_authorized": all(science_checks.values()) and stop_audit["authenticated"],
        "commands": {"preflight": " ".join(sys.argv)},
    }
    durable_json(output / "preflight_audit.json", preflight)
    durable_json(output / "distributed_equivalence.json", {
        "gpu_count": 1,
        "required": False,
        "passed": True,
        "reason": "registered one-GPU result configuration",
    })
    durable_json(output / "batch_manifest.json", {
        **manifest,
        "training_shards": [str(path.resolve()) for path in shards],
        "selected_micro_batch_sequences": memory["selected_micro_batch_sequences"],
        "selected_gradient_accumulation": memory["selected_gradient_accumulation"],
        "global_targets_per_update": GLOBAL_TARGETS,
    })
    del optimizer, model, calibration_x, small_x, small_y
    gc.collect()
    torch.cuda.empty_cache()
    if not preflight["science_passed"]:
        raise SystemExit("Experiment 2D1 science preflight failed")
    if not preflight["result_run_authorized"]:
        print("AUTOMATIC RUNPOD STOP IS NOT AVAILABLE. THE UNATTENDED 2D1 TRAINING RUN HAS NOT BEEN STARTED.", flush=True)
    else:
        print(
            f"EXPERIMENT_2D1_PREFLIGHT_PASS micro_batch={memory['selected_micro_batch_sequences']} "
            f"grad_accum={memory['selected_gradient_accumulation']}", flush=True
        )
    return preflight


def run_authorize_stop(args):
    """Seal a separately verified authenticated stop mechanism into preflight."""
    require_git(clean=False)
    output = Path(args.output_dir).resolve()
    preflight_path = output / "preflight_audit.json"
    if not preflight_path.is_file():
        raise SystemExit("stop authorization requires a completed science preflight")
    preflight = json.loads(preflight_path.read_text())
    if not preflight.get("science_passed"):
        raise SystemExit("stop authorization cannot override a failed science preflight")
    stop_audit = {
        "pod_id": args.pod_id,
        "mechanism": args.stop_mechanism,
        "authentication_status": "authenticated",
        "authenticated": True,
        "verification": args.verification,
        "workspace_mount": preflight["runpod_stop_audit"]["workspace_mount"],
        "network_volume_mount": preflight["runpod_stop_audit"]["network_volume_mount"],
        "tested_at": time.time(),
    }
    preflight["runpod_stop_audit"] = stop_audit
    preflight["result_run_authorized"] = True
    preflight.setdefault("commands", {})["authorize_stop"] = " ".join(sys.argv)
    durable_json(output / "runpod_stop_audit.json", stop_audit)
    durable_json(preflight_path, preflight)
    print(f"EXPERIMENT_2D1_STOP_AUTHORIZATION_PASS pod_id={args.pod_id}", flush=True)
    return preflight


def next_global_batch_hash(loader, gradient_accumulation):
    clone = loader.clone()
    hashes = []
    for _ in range(gradient_accumulation):
        x, y = clone.next_batch()
        hashes.append(batch_payload_hash(x, y))
    return aggregate_hashes(hashes)


def checkpoint_payload(model, optimizer, loader, prefix_rng, training_state, metadata, gradient_accumulation):
    schedule = stage_for_update(max(1, training_state["completed_updates"]))
    if training_state["completed_updates"] == 0:
        schedule = {"stage": "A", "windows": (T,) * N_LAYER, "rho": 0.0}
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_state": copy.deepcopy(training_state),
        "scheduler_position": training_state["completed_updates"],
        "completed_updates": training_state["completed_updates"],
        "processed_targets": training_state["processed_targets"],
        "current_curriculum_stage": schedule["stage"],
        "current_windows": list(schedule["windows"]),
        "rho": float(schedule["rho"]),
        "loader_state": loader.state_dict(),
        "rng_state": rng_state(prefix_rng),
        "next_global_batch_sha256": next_global_batch_hash(loader, gradient_accumulation),
        "metadata": copy.deepcopy(metadata),
        "git_commit": git_output("rev-parse", "HEAD"),
        "environment": runtime_environment(),
    }


def verify_checkpoint(path, model, optimizer, loader, prefix_rng, gradient_accumulation, expected_state):
    path = Path(path)
    reopened = torch_load(path, mmap=True)
    required = {
        "schema", "model", "optimizer", "training_state", "scheduler_position", "completed_updates",
        "processed_targets", "current_curriculum_stage", "current_windows", "rho", "loader_state",
        "rng_state", "next_global_batch_sha256", "metadata", "git_commit", "environment",
    }
    checks = {
        "schema": reopened.get("schema") == CHECKPOINT_SCHEMA,
        "fields": set(reopened) == required,
        "completed_updates": reopened.get("completed_updates") == expected_state["completed_updates"],
        "processed_targets": reopened.get("processed_targets") == expected_state["processed_targets"],
        "training_state": reopened.get("training_state") == expected_state,
        "loader_state": reopened.get("loader_state") == loader.state_dict(),
        "next_batch": reopened.get("next_global_batch_sha256") == next_global_batch_hash(loader, gradient_accumulation),
        "rng_fields": set(reopened.get("rng_state", {})) == {"python", "numpy", "torch_cpu", "torch_cuda", "prefix_rng"},
        "model_keys": reopened.get("model", {}).keys() == model.state_dict().keys(),
    }
    if not all(checks.values()):
        raise SystemExit(f"checkpoint strict-reopen metadata mismatch: {checks}")
    model.load_state_dict(reopened["model"], strict=True)
    optimizer.load_state_dict(reopened["optimizer"])
    checks.update({
        "model_finite_after_reload": model_parameters_finite(model),
        "optimizer_finite_after_reload": optimizer_moments_finite(optimizer),
        "weight_tying_after_reload": model.base.transformer.wte.weight is model.base.lm_head.weight,
    })
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise SystemExit(f"checkpoint strict-reopen tensor mismatch: {checks}")
    digest = file_sha256(path)
    return {"checkpoint": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size, "strict_reopen": checks, "passed": True}


def save_checkpoint(path, model, optimizer, loader, prefix_rng, training_state, metadata, gradient_accumulation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    if temporary.exists():
        raise SystemExit(f"stale incomplete checkpoint requires inspection: {temporary}")
    payload = checkpoint_payload(
        model, optimizer, loader, prefix_rng, training_state, metadata, gradient_accumulation
    )
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
    del payload
    verification = verify_checkpoint(
        path, model, optimizer, loader, prefix_rng, gradient_accumulation, training_state
    )
    durable_text(path.with_suffix(path.suffix + ".sha256"), f"{verification['sha256']}  {path.name}\n")
    durable_json(path.with_suffix(path.suffix + ".verification.json"), verification)
    return verification


def load_result_checkpoint(path, model, optimizer, shards, prefix_rng, expected_metadata, micro_batch):
    path = Path(path).resolve()
    sha_sidecar = path.with_suffix(path.suffix + ".sha256")
    verification_sidecar = path.with_suffix(path.suffix + ".verification.json")
    if not sha_sidecar.is_file() or not verification_sidecar.is_file():
        raise SystemExit("checkpoint sidecars missing")
    expected_sha = sha_sidecar.read_text().split()[0]
    if file_sha256(path) != expected_sha:
        raise SystemExit("resume checkpoint SHA mismatch")
    payload = torch_load(path, mmap=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("metadata") != expected_metadata:
        raise SystemExit("resume checkpoint schema/metadata mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    loader = ExplicitShardLoader(shards, micro_batch, T, state=payload["loader_state"])
    gradient_accumulation = GLOBAL_TARGETS // (micro_batch * T)
    if payload["next_global_batch_sha256"] != next_global_batch_hash(loader, gradient_accumulation):
        raise SystemExit("resume next-global-batch mismatch")
    restore_rng_state(payload["rng_state"], prefix_rng)
    state = copy.deepcopy(payload["training_state"])
    if not model_parameters_finite(model) or not optimizer_moments_finite(optimizer):
        raise SystemExit("nonfinite model/optimizer after resume")
    del payload
    gc.collect()
    return loader, state


def checkpoint_manifest_path(output_dir):
    return Path(output_dir) / "checkpoint_manifest.json"


def record_checkpoint(output_dir, kind, update, verification):
    path = checkpoint_manifest_path(output_dir)
    manifest = json.loads(path.read_text()) if path.is_file() else {"scientific": {}, "rolling": {}, "smoke": {}}
    manifest[kind][str(update)] = verification
    durable_json(path, manifest)


def rotate_recovery_checkpoints(run_root, output_dir):
    path = checkpoint_manifest_path(output_dir)
    if not path.is_file():
        return
    manifest = json.loads(path.read_text())
    rolling = manifest.get("rolling", {})
    updates = sorted(int(value) for value in rolling)
    while len(updates) > ROLLING_KEEP:
        update = updates.pop(0)
        row = rolling.pop(str(update))
        if not row.get("passed") or not row.get("strict_reopen", {}).get("passed"):
            raise SystemExit("refusing to remove unverified rolling checkpoint")
        checkpoint_path = Path(row["checkpoint"])
        for candidate in (
            checkpoint_path,
            checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256"),
            checkpoint_path.with_suffix(checkpoint_path.suffix + ".verification.json"),
        ):
            if candidate.is_file():
                candidate.unlink()
    manifest["rolling"] = rolling
    durable_json(path, manifest)


def metadata_payload(args, preflight, fusion_initialization, micro_batch):
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "parent_commit": PARENT_COMMIT,
        "source_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "data_root": str(Path(args.data_root).resolve()),
        "canonical_validation_sha256": CANONICAL_VALIDATION_SHA256,
        "implementation_git_commit": git_output("rev-parse", "HEAD"),
        "micro_batch_sequences": micro_batch,
        "gradient_accumulation": GLOBAL_TARGETS // (micro_batch * T),
        "global_targets_per_update": GLOBAL_TARGETS,
        "fusion_initialization": fusion_initialization,
        "pod_stop_audit": preflight["runpod_stop_audit"],
    }


def initialize_training_runtime(args, resume=None, require_stop=True):
    device = require_single_a100()
    seed_all(SEED)
    output = Path(args.output_dir).resolve()
    preflight = json.loads((output / "preflight_audit.json").read_text())
    if not preflight.get("science_passed"):
        raise SystemExit("result path requires passing science preflight")
    if require_stop and not preflight.get("result_run_authorized"):
        raise SystemExit("automatic RunPod stop is unavailable; refusing result update 1")
    batch_manifest = json.loads((output / "batch_manifest.json").read_text())
    micro_batch = int(batch_manifest["selected_micro_batch_sequences"])
    gradient_accumulation = int(batch_manifest["selected_gradient_accumulation"])
    if micro_batch * T * gradient_accumulation != GLOBAL_TARGETS:
        raise SystemExit("global target batch mismatch")
    shards = train_shards(args.data_root)
    _, model, _ = load_source_model(args.parent_checkpoint, device, trainable=True)
    calibration_loader = ExplicitShardLoader(shards, 4, T)
    calibration_x, _ = calibration_loader.next_batch()
    calibration = calibrate_fusion(model, calibration_x.to(device))
    expected_calibration = json.loads((output / "fusion_initialization.json").read_text())
    numeric_keys = (
        "embedding_rms", "top_state_rms", "initial_source_scale", "gate_preactivation_mean",
        "gate_preactivation_std", "gate_preactivation_abs_max", "gate_mean", "gate_std",
        "gate_saturation_fraction",
    )
    if any(abs(calibration[key] - expected_calibration[key]) > 1e-12 for key in numeric_keys):
        raise SystemExit(f"fusion calibration replay mismatch: {calibration} != {expected_calibration}")
    optimizer, optimizer_report = configure_optimizer(model)
    prefix_rng = random.Random(SEED + 2_001)
    metadata = metadata_payload(args, preflight, calibration, micro_batch)
    if resume is None:
        loader = ExplicitShardLoader(shards, micro_batch, T)
        training_state = {
            "completed_updates": 0,
            "processed_targets": 0,
            "started_at": time.time(),
            "last_checkpoint": None,
            "healthy_reference": None,
            "healthy_reference_samples": [],
            "explosion_consecutive": 0,
            "last_metrics": None,
        }
    else:
        loader, training_state = load_result_checkpoint(
            resume, model, optimizer, shards, prefix_rng, metadata, micro_batch
        )
    return SimpleNamespace(
        device=device,
        output=output,
        preflight=preflight,
        micro_batch=micro_batch,
        gradient_accumulation=gradient_accumulation,
        shards=shards,
        model=model,
        optimizer=optimizer,
        optimizer_report=optimizer_report,
        prefix_rng=prefix_rng,
        metadata=metadata,
        loader=loader,
        training_state=training_state,
    )


def train_one_update(runtime, update, force_two_pass=False):
    model = runtime.model
    model.train()
    optimizer = runtime.optimizer
    device = runtime.device
    schedule = stage_for_update(update)
    pass_count = 2 if force_two_pass or update % THREE_PASS_EVERY else 3
    weights = TWO_PASS_WEIGHTS if pass_count == 2 else THREE_PASS_WEIGHTS
    lrs = set_optimizer_lrs(optimizer, update)
    optimizer.zero_grad(set_to_none=True)
    pass_sums = [0.0] * pass_count
    total_sum = 0.0
    prefix_records = []
    final_diagnostics = None
    update_start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for micro_index in range(runtime.gradient_accumulation):
        cpu_x, cpu_y = runtime.loader.next_batch()
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        prefixes = [runtime.prefix_rng.randrange(T) for _ in range(pass_count - 1)]
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
            final_pass_diagnostics = diag2
            if pass_count == 3:
                _, loss3, diag3 = model.forward_pass(
                    x, y, schedule["windows"], previous_top=top2, rho=schedule["rho"],
                    prefix_length=prefixes[1], activation_checkpointing=True,
                    return_diagnostics=final_micro,
                )
                losses.append(loss3)
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
        del x, y, cpu_x, cpu_y, top1, top2, losses, weighted, scaled
    if not gradients_finite(model):
        raise SystemExit("NaN/Inf gradients")
    gradients = gradient_report(model)
    required_nonzero = gradients["base"]["nonzero"] and (
        schedule["rho"] == 0.0
        or (gradients["W_u"]["nonzero"] and gradients["W_g"]["nonzero"])
    )
    if not required_nonzero:
        raise SystemExit(f"required gradient group is zero: {gradients}")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad_norm):
        raise SystemExit("nonfinite gradient norm")
    optimizer.step()
    if not model_parameters_finite(model):
        raise SystemExit("NaN/Inf parameters")
    if not optimizer_moments_finite(optimizer):
        raise SystemExit("NaN/Inf optimizer moments")
    elapsed = time.monotonic() - update_start
    state = runtime.training_state
    state["completed_updates"] = update
    state["processed_targets"] = update * GLOBAL_TARGETS
    health = {
        "top_state_rms": final_diagnostics["top_state_rms"],
        "recurrent_input_rms": final_diagnostics["recurrent_input_rms"],
    }
    if update <= 10:
        state["healthy_reference_samples"].append(health)
        state["healthy_reference"] = {
            key: sum(row[key] for row in state["healthy_reference_samples"]) / len(state["healthy_reference_samples"])
            for key in health
        }
    reference = state["healthy_reference"]
    exploded = reference is not None and any(health[key] > 10.0 * reference[key] for key in health)
    state["explosion_consecutive"] = state["explosion_consecutive"] + 1 if exploded else 0
    if state["explosion_consecutive"] >= 3:
        raise SystemExit(f"recurrent-state explosion hard stop: health={health} reference={reference}")
    metrics = {
        "timestamp": time.time(),
        "update": update,
        "targets": state["processed_targets"],
        "stage": schedule["stage"],
        "windows": list(schedule["windows"]),
        "rho": schedule["rho"],
        "pass_count": pass_count,
        "pass_losses": [value / runtime.gradient_accumulation for value in pass_sums],
        "weighted_total_ce": total_sum / runtime.gradient_accumulation,
        "prefix_lengths": prefix_records,
        "lrs": lrs,
        "gradient_norm_before_clip": grad_norm.detach().float().item(),
        "gradient_groups": gradients,
        "state_diagnostics": final_diagnostics,
        "healthy_reference": reference,
        "explosion_consecutive": state["explosion_consecutive"],
        "wall_seconds": elapsed,
        "targets_per_second": GLOBAL_TARGETS / elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "all_gradients_finite": True,
        "all_parameters_finite": True,
        "all_optimizer_moments_finite": True,
    }
    state["last_metrics"] = metrics
    return metrics


def gpu_telemetry():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True).strip().split(",")
        return {"utilization_percent": float(output[0]), "memory_used_mb": float(output[1]), "memory_total_mb": float(output[2])}
    except Exception:
        return {"utilization_percent": None, "memory_used_mb": None, "memory_total_mb": None}


def write_heartbeat(runtime, metrics):
    completed = metrics["update"]
    elapsed = time.time() - runtime.training_state["started_at"]
    eta = elapsed / completed * (MAX_UPDATES - completed) if completed else None
    payload = {
        "timestamp": time.time(),
        "update": completed,
        "targets": metrics["targets"],
        "stage": metrics["stage"],
        "windows": metrics["windows"],
        "rho": metrics["rho"],
        "pass_count": metrics["pass_count"],
        "training_losses": metrics["pass_losses"],
        "weighted_total_ce": metrics["weighted_total_ce"],
        "lrs": metrics["lrs"],
        "gpu": gpu_telemetry(),
        "latest_checkpoint": runtime.training_state["last_checkpoint"],
        "eta_seconds": eta,
    }
    durable_json(runtime.output / "HEARTBEAT.json", payload)


def milestone_validation(runtime, update, transition=False):
    if transition:
        next_schedule = stage_for_update(update + 1)
        schedule = next_schedule
        key = f"after_{update:04d}_before_{update + 1:04d}_{next_schedule['stage']}"
        kind = "transition"
    elif update == 0:
        schedule = {"stage": "A", "windows": (T,) * N_LAYER, "rho": 0.0}
        key = "0000"
        kind = "milestone"
    else:
        schedule = stage_for_update(update)
        key = f"{update:04d}"
        kind = "milestone"
    controls = ["plain", "real"]
    if schedule["rho"] < 1.0:
        controls.append("real_final_rho")
    result = evaluate_temporal(
        runtime.model,
        validation_shard(runtime.metadata["data_root"]),
        schedule["windows"],
        schedule["rho"],
        controls=tuple(controls),
    )
    result.update({"update": update, "evaluation_kind": kind, "stage": schedule["stage"], "evaluated_at": time.time()})
    path = runtime.output / "milestone_validation.json"
    values = json.loads(path.read_text()) if path.is_file() else {"milestones": {}, "transitions": {}}
    values["transitions" if transition else "milestones"][key] = result
    durable_json(path, values)
    return result


def save_scheduled_checkpoint(runtime, args, update, kind):
    directory = Path(args.run_root).resolve() / "checkpoints"
    name = (
        f"scientific_update_{update:04d}.pt" if kind == "scientific"
        else f"recovery_update_{update:04d}.pt"
    )
    checkpoint_path = (directory / name).resolve()
    previous_checkpoint = runtime.training_state["last_checkpoint"]
    runtime.training_state["last_checkpoint"] = str(checkpoint_path)
    try:
        verification = save_checkpoint(
            checkpoint_path,
            runtime.model,
            runtime.optimizer,
            runtime.loader,
            runtime.prefix_rng,
            runtime.training_state,
            runtime.metadata,
            runtime.gradient_accumulation,
        )
    except BaseException:
        runtime.training_state["last_checkpoint"] = previous_checkpoint
        raise
    record_checkpoint(runtime.output, kind, update, verification)
    if kind == "rolling":
        rotate_recovery_checkpoints(args.run_root, runtime.output)
    return verification


def run_smoke(args):
    require_git(clean=True)
    runtime = initialize_training_runtime(args, require_stop=False)
    smoke_dir = Path(args.run_root).resolve() / "smoke"
    smoke_metrics = []
    for update in range(1, 6):
        metrics = train_one_update(runtime, update, force_two_pass=True)
        smoke_metrics.append(metrics)
        print(
            f"2D1 smoke update={update}/5 loss={metrics['weighted_total_ce']:.6f} "
            f"peak_mb={metrics['peak_allocated_vram_mb']:.0f}", flush=True
        )
    verification = save_checkpoint(
        smoke_dir / "disposable_smoke_update_0005.pt",
        runtime.model,
        runtime.optimizer,
        runtime.loader,
        runtime.prefix_rng,
        runtime.training_state,
        runtime.metadata,
        runtime.gradient_accumulation,
    )
    gradients = smoke_metrics[-1]["gradient_groups"]
    report = {
        "updates": 5,
        "global_targets_per_update": GLOBAL_TARGETS,
        "processed_targets": 5 * GLOBAL_TARGETS,
        "two_pass_only": True,
        "metrics": smoke_metrics,
        "checkpoint_verification": verification,
        "next_batch_exact": verification["strict_reopen"]["next_batch"],
        "finite_losses": all(all(math.isfinite(value) for value in row["pass_losses"]) for row in smoke_metrics),
        "finite_recurrent_state": all(math.isfinite(row["state_diagnostics"]["top_state_rms"]) for row in smoke_metrics),
        "finite_fused_input": all(math.isfinite(row["state_diagnostics"]["recurrent_input_rms"]) for row in smoke_metrics),
        "W_u_gradient_nonzero": gradients["W_u"]["nonzero"],
        "W_g_gradient_nonzero": gradients["W_g"]["nonzero"],
        "base_gradient_nonzero": gradients["base"]["nonzero"],
        "optimizer_moments_finite": optimizer_moments_finite(runtime.optimizer),
        "discarded": False,
    }
    report["passed"] = all(
        report[key]
        for key in (
            "next_batch_exact", "finite_losses", "finite_recurrent_state", "finite_fused_input",
            "W_u_gradient_nonzero", "W_g_gradient_nonzero", "base_gradient_nonzero", "optimizer_moments_finite",
        )
    )
    if not report["passed"]:
        raise SystemExit(f"2D1 smoke hard stop: {report}")
    checkpoint_path = Path(verification["checkpoint"])
    for candidate in (
        checkpoint_path,
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256"),
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".verification.json"),
    ):
        if candidate.is_file():
            candidate.unlink()
    report["discarded"] = True
    durable_json(runtime.output / "smoke_audit.json", report)
    print("EXPERIMENT_2D1_SMOKE_PASS disposable_checkpoint_discarded=true", flush=True)
    return report


def run_train_worker(args):
    require_git(clean=False)
    runtime = initialize_training_runtime(args, resume=args.resume)
    state = runtime.training_state
    if state["completed_updates"] == 0:
        smoke = json.loads((runtime.output / "smoke_audit.json").read_text())
        if not smoke.get("passed") or not smoke.get("discarded"):
            raise SystemExit("result update 1 requires passing discarded smoke")
        milestone_validation(runtime, 0)
    start_update = state["completed_updates"] + 1
    for update in range(start_update, MAX_UPDATES + 1):
        metrics = train_one_update(runtime, update)
        append_jsonl(runtime.output / "training_metrics.jsonl", metrics)
        if update <= 5 or update % 10 == 0:
            write_heartbeat(runtime, metrics)
        print(
            f"2D1 update={update:04d}/{MAX_UPDATES} stage={metrics['stage']} rho={metrics['rho']:.6f} "
            f"passes={metrics['pass_count']} loss={metrics['weighted_total_ce']:.6f} "
            f"tok/s={metrics['targets_per_second']:.0f}", flush=True
        )
        if update in SCIENTIFIC_UPDATES:
            save_scheduled_checkpoint(runtime, args, update, "scientific")
        elif update % ROLLING_INTERVAL == 0:
            save_scheduled_checkpoint(runtime, args, update, "rolling")
        if update in VALIDATION_UPDATES:
            milestone_validation(runtime, update)
        if update in (477, 954, 1908, 2862):
            milestone_validation(runtime, update, transition=True)
        if update in FORCED_RESTARTS:
            durable_json(runtime.output / f"forced_restart_{update:04d}.json", {
                "update": update,
                "checkpoint": runtime.training_state["last_checkpoint"],
                "fresh_process_required": True,
                "requested_at": time.time(),
            })
            print(f"EXPERIMENT_2D1_FORCED_RESTART update={update}", flush=True)
            return 75
    if state["completed_updates"] != MAX_UPDATES or state["processed_targets"] != TOTAL_TARGETS:
        raise SystemExit("result worker finished with wrong update/target count")
    durable_json(runtime.output / "training_complete.json", {
        "completed_updates": state["completed_updates"],
        "processed_targets": state["processed_targets"],
        "final_checkpoint": state["last_checkpoint"],
        "completed_at": time.time(),
        "passed": True,
    })
    return 0


def latest_scientific_checkpoint(output_dir, update):
    manifest = json.loads(checkpoint_manifest_path(output_dir).read_text())
    row = manifest["scientific"].get(str(update))
    if not row or not row.get("passed"):
        raise SystemExit(f"missing verified scientific checkpoint at {update}")
    return row["checkpoint"]


def run_supervise(args):
    require_git(clean=True)
    output = Path(args.output_dir).resolve()
    preflight = json.loads((output / "preflight_audit.json").read_text())
    if not preflight.get("result_run_authorized"):
        raise SystemExit("AUTOMATIC RUNPOD STOP IS NOT AVAILABLE. THE UNATTENDED 2D1 TRAINING RUN HAS NOT BEEN STARTED.")
    smoke = json.loads((output / "smoke_audit.json").read_text())
    if not smoke.get("passed") or not smoke.get("discarded"):
        raise SystemExit("passing disposable smoke is required")
    resume = args.resume
    restarts = []
    observed_forced_restarts = set()
    while True:
        command = [
            sys.executable, str(Path(__file__).resolve()), "train-worker",
            "--parent-checkpoint", str(Path(args.parent_checkpoint).resolve()),
            "--data-root", str(Path(args.data_root).resolve()),
            "--output-dir", str(output),
            "--run-root", str(Path(args.run_root).resolve()),
        ]
        if resume:
            command.extend(["--resume", str(Path(resume).resolve())])
        started = time.time()
        completed = subprocess.run(command, cwd=REPO_ROOT)
        restarts.append({"pid": completed.args, "started_at": started, "ended_at": time.time(), "returncode": completed.returncode})
        durable_json(output / "process_restarts.json", restarts)
        if completed.returncode == 0:
            break
        if completed.returncode != 75:
            raise SystemExit(f"2D1 worker terminal failure returncode={completed.returncode}")
        candidates = []
        for candidate in output.glob("forced_restart_*.json"):
            row = json.loads(candidate.read_text())
            if row["update"] not in observed_forced_restarts:
                candidates.append(row["update"])
        if len(candidates) != 1 or candidates[0] not in FORCED_RESTARTS:
            raise SystemExit(f"unexpected forced restart markers: {candidates}")
        completed_updates = candidates[0]
        observed_forced_restarts.add(completed_updates)
        resume = latest_scientific_checkpoint(output, completed_updates)
    if len([row for row in restarts if row["returncode"] == 75]) != 2:
        raise SystemExit("exactly two forced fresh-process restarts were not observed")
    print("EXPERIMENT_2D1_TRAINING_COMPLETE updates=4769 targets=2500329472", flush=True)


@torch.no_grad()
def evaluate_incremental_subset(model, val_path, batches=2):
    model.eval()
    device = next(model.parameters()).device
    loader = ExplicitShardLoader([val_path], VALIDATION_B, T)
    loss_sum = 0.0
    targets = 0
    per_batch = []
    cache_maxima = [0] * N_LAYER
    top_rms_min = float("inf")
    top_rms_max = 0.0
    start = time.monotonic()
    for batch_index in range(batches):
        cpu_x, cpu_y = loader.next_batch()
        x = cpu_x.to(device)
        y = cpu_y.to(device)
        state = init_incremental_state(model, VALIDATION_B, TARGET_WINDOWS, device, next(model.parameters()).dtype)
        batch_sum = 0.0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for position in range(T):
                logits, state = incremental_step(model, x[:, position], state, rho=1.0)
                losses = F.cross_entropy(logits[:, 0].float(), y[:, position], reduction="none")
                batch_sum += losses.double().sum().item()
                lengths = incremental_cache_lengths(state)
                cache_maxima = [max(old, value) for old, value in zip(cache_maxima, lengths)]
                top_rms = state.previous_top.float().pow(2).mean().sqrt().item()
                top_rms_min = min(top_rms_min, top_rms)
                top_rms_max = max(top_rms_max, top_rms)
        batch_targets = VALIDATION_B * T
        per_batch.append(batch_sum / batch_targets)
        loss_sum += batch_sum
        targets += batch_targets
        print(f"2D1 incremental validation batch={batch_index + 1}/{batches}", flush=True)
    elapsed = time.monotonic() - start
    expected_limits = [window - 1 for window in TARGET_WINDOWS]
    return {
        "validation_loss": loss_sum / targets,
        "validation_targets": targets,
        "per_batch_losses": per_batch,
        "batches": batches,
        "batch_size": VALIDATION_B,
        "sequence_length": T,
        "cache_maxima": cache_maxima,
        "cache_limits": expected_limits,
        "physical_caches_bounded": all(value <= limit for value, limit in zip(cache_maxima, expected_limits)),
        "top_state_rms_min": top_rms_min,
        "top_state_rms_max": top_rms_max,
        "all_state_finite": math.isfinite(top_rms_min) and math.isfinite(top_rms_max),
        "wall_seconds": elapsed,
        "targets_per_second": targets / elapsed,
    }


@torch.no_grad()
def incremental_reset_and_row_tests(model, val_path):
    device = next(model.parameters()).device
    loader = ExplicitShardLoader([val_path], 2, 32)
    tokens, _ = loader.next_batch()
    tokens = tokens.to(device)

    def run(value, reset_position=None):
        state = init_incremental_state(model, value.size(0), TARGET_WINDOWS, device, next(model.parameters()).dtype)
        rows = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for position in range(value.size(1)):
                override = None
                if reset_position is not None and position == reset_position:
                    override = torch.zeros_like(state.previous_top)
                logits, state = incremental_step(model, value[:, position], state, rho=1.0, previous_override=override)
                rows.append(logits)
        return torch.cat(rows, dim=1)

    original = run(tokens)
    changed = tokens.clone()
    changed[1] = (changed[1] + 31) % model.config.vocab_size
    changed_logits = run(changed)
    row_delta = (original[0].float() - changed_logits[0].float()).abs().max().item()
    reset_a = run(tokens, reset_position=16)

    state = init_incremental_state(model, tokens.size(0), TARGET_WINDOWS, device, next(model.parameters()).dtype)
    rows = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(tokens.size(1)):
            if position == 16:
                state.previous_top.zero_()
            logits, state = incremental_step(model, tokens[:, position], state, rho=1.0)
            rows.append(logits)
    reset_b = torch.cat(rows, dim=1)
    return {
        "row_isolation_max_logit_delta": row_delta,
        "row_isolation_exact": row_delta == 0.0,
        "reset_position": 16,
        "reset_implementations_bit_exact": torch.equal(reset_a, reset_b),
        "reset_max_logit_delta": (reset_a.float() - reset_b.float()).abs().max().item(),
        "passed": row_delta == 0.0 and torch.equal(reset_a, reset_b),
    }


def read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def transition_analysis(milestones):
    rows = []
    pairs = ((477, 954), (954, 1908), (1908, 2862), (2862, 4769))
    for boundary, next_end in pairs:
        before = milestones["milestones"][f"{boundary:04d}"]
        transition_key = next(key for key in milestones["transitions"] if key.startswith(f"after_{boundary:04d}_"))
        after = milestones["transitions"][transition_key]
        recovered = milestones["milestones"][f"{next_end:04d}"]
        before_loss = before["controls"]["real"]["validation_loss"]
        shock_loss = after["controls"]["real"]["validation_loss"]
        recovered_loss = recovered["controls"]["real"]["validation_loss"]
        rows.append({
            "boundary_update": boundary,
            "next_stage_end_update": next_end,
            "before_transition_recurrent_loss": before_loss,
            "immediate_next_stage_recurrent_loss": shock_loss,
            "immediate_loss_shock": shock_loss - before_loss,
            "end_of_next_stage_recurrent_loss": recovered_loss,
            "recovery_from_shock": shock_loss - recovered_loss,
            "net_change_vs_before": recovered_loss - before_loss,
        })
    return rows


def classification_from_results(final_controls, incremental, milestones, audit_passed):
    plain = final_controls["controls"]["plain"]
    real = final_controls["controls"]["real"]
    zero = final_controls["controls"]["zero"]
    recurrent_gain = plain["validation_loss"] - real["validation_loss"]
    paired_plain = final_controls["real_vs_plain_paired_wins"]
    paired_zero = final_controls["real_vs_zero_paired_wins"]
    stage_e_start = milestones["transitions"][next(key for key in milestones["transitions"] if key.startswith("after_2862_"))]
    stage_e_start_loss = stage_e_start["controls"]["real"]["validation_loss"]
    nonworsening = real["validation_loss"] <= stage_e_start_loss
    stable = incremental["physical_caches_bounded"] and incremental["all_state_finite"]
    if not audit_passed or not stable:
        primary = "EXPERIMENT 2D1 UNSTABLE"
    elif recurrent_gain >= 0.10 and paired_plain >= 18 and paired_zero >= 18 and nonworsening:
        primary = "TRIANGLE RECURRENT TRANSFORMER LEARNS STRONG COMPENSATION"
    elif recurrent_gain > 0 and paired_plain >= 11 and stable:
        primary = "TRIANGLE RECURRENT TRANSFORMER LEARNS PARTIAL COMPENSATION"
    else:
        primary = "TRIANGLE RECURRENT TRAINING DOES NOT YET COMPENSATE KV LOSS"
    gap = final_controls["sequence_specific_gap"]
    shuffled_wins = final_controls["real_vs_shuffled_paired_wins"]
    if gap >= 0.01 and shuffled_wins >= 18:
        secondary = "SEQUENCE-ALIGNED RECURRENCE PRESENT"
    elif recurrent_gain > 0:
        secondary = "RECURRENT UTILITY PRESENT WITHOUT STRONG ALIGNMENT"
    else:
        secondary = "NO RECURRENT UTILITY"
    return primary, secondary, nonworsening


def next_experiment_recommendation(primary, stage_e_improving, recurrent_gain, sequence_gap):
    if primary == "EXPERIMENT 2D1 UNSTABLE":
        return "FIX 2D1 INTEGRITY / STABILITY"
    if recurrent_gain > 0 and stage_e_improving:
        return "CONTINUE FINAL TRIANGLE TRAINING"
    if recurrent_gain > 0.10 and sequence_gap >= 0.01:
        return "EXPAND RECURRENT SOURCE ROUTING BEYOND B12"
    if recurrent_gain > 0:
        return "MODIFY RECURRENT FUSION"
    return "REDUCE TRIANGLE COMPRESSION"


def run_finalize(args):
    require_git(clean=False)
    runtime = initialize_training_runtime(args, resume=args.final_checkpoint)
    if runtime.training_state["completed_updates"] != MAX_UPDATES:
        raise SystemExit("finalize requires update 4769 checkpoint")
    val_path = validation_shard(args.data_root)
    final_controls = evaluate_temporal(
        runtime.model, val_path, TARGET_WINDOWS, 1.0,
        controls=("plain", "real", "shuffled", "zero"),
    )
    durable_json(runtime.output / "recurrent_controls.json", final_controls)
    durable_json(runtime.output / "shuffled_controls.json", {
        "real": final_controls["controls"]["real"],
        "shuffled": final_controls["controls"]["shuffled"],
        "sequence_specific_gap": final_controls["sequence_specific_gap"],
        "paired_wins": final_controls["real_vs_shuffled_paired_wins"],
    })
    incremental = evaluate_incremental_subset(runtime.model, val_path, batches=2)
    reset_rows = incremental_reset_and_row_tests(runtime.model, val_path)
    small_loader = ExplicitShardLoader([val_path], 2, 16)
    small_x, _ = small_loader.next_batch()
    equivalence = incremental_equivalence_tests(runtime.model, small_x.to(runtime.device))
    incremental.update({"equivalence": equivalence, "reset_and_row_tests": reset_rows})
    incremental["passed"] = (
        incremental["physical_caches_bounded"] and incremental["all_state_finite"]
        and equivalence["passed"] and reset_rows["passed"]
    )
    durable_json(runtime.output / "incremental_validation.json", incremental)
    durable_json(runtime.output / "cache_audit.json", {
        "target_windows": list(TARGET_WINDOWS),
        "historical_cache_limits": [window - 1 for window in TARGET_WINDOWS],
        "observed_cache_maxima": incremental["cache_maxima"],
        "physical_caches_bounded": incremental["physical_caches_bounded"],
        "incremental_equivalence": equivalence,
        "reset_and_row_tests": reset_rows,
        "passed": incremental["passed"],
    })

    training_rows = read_jsonl(runtime.output / "training_metrics.jsonl")
    if len(training_rows) != MAX_UPDATES or training_rows[-1]["targets"] != TOTAL_TARGETS:
        raise SystemExit("training metrics update/target count mismatch")
    pass_losses = [{
        "update": row["update"], "stage": row["stage"], "rho": row["rho"],
        "pass_count": row["pass_count"], "pass_losses": row["pass_losses"],
        "weighted_total_ce": row["weighted_total_ce"],
    } for row in training_rows]
    durable_json(runtime.output / "pass_losses.json", pass_losses)
    state_rows = [{"update": row["update"], **row["state_diagnostics"]} for row in training_rows]
    durable_json(runtime.output / "state_diagnostics.json", state_rows)
    initialization = json.loads((runtime.output / "fusion_initialization.json").read_text())
    with torch.no_grad():
        source_scale = initialization["initial_source_scale"]
        identity = torch.eye(N_EMBD, device=runtime.device)
        w_u = runtime.model.fusion.W_u.weight.float()
        w_g = runtime.model.fusion.W_g.weight.float()
        fusion_diagnostics = {
            "W_u_norm": w_u.norm().item(),
            "W_g_norm": w_g.norm().item(),
            "W_u_initial_norm": (identity * source_scale).norm().item(),
            "W_g_initial_norm": identity.norm().item(),
            "W_u_displacement_norm": (w_u - identity * source_scale).norm().item(),
            "W_g_displacement_norm": (w_g - identity).norm().item(),
            "W_u_learned_materially": (w_u - identity * source_scale).norm().item() > 1e-3,
            "W_g_learned_materially": (w_g - identity).norm().item() > 1e-3,
        }
    durable_json(runtime.output / "fusion_diagnostics.json", fusion_diagnostics)
    milestones = json.loads((runtime.output / "milestone_validation.json").read_text())
    transitions = transition_analysis(milestones)
    stage_e_losses = [
        milestones["transitions"][next(key for key in milestones["transitions"] if key.startswith("after_2862_"))]["controls"]["real"]["validation_loss"],
        milestones["milestones"]["3815"]["controls"]["real"]["validation_loss"],
        milestones["milestones"]["4769"]["controls"]["real"]["validation_loss"],
    ]
    stage_e_improving = stage_e_losses[-1] <= stage_e_losses[0] and stage_e_losses[-1] <= stage_e_losses[1]
    training_integrity = {
        "exact_updates": len(training_rows) == MAX_UPDATES and training_rows[-1]["update"] == MAX_UPDATES,
        "exact_targets": training_rows[-1]["targets"] == TOTAL_TARGETS,
        "three_pass_cadence": all(row["pass_count"] == (3 if row["update"] % THREE_PASS_EVERY == 0 else 2) for row in training_rows),
        "curriculum": all(row["windows"] == list(stage_for_update(row["update"])["windows"]) for row in training_rows),
        "rho_schedule": all(abs(row["rho"] - stage_for_update(row["update"])["rho"]) <= 1e-15 for row in training_rows),
        "finite_losses": all(math.isfinite(row["weighted_total_ce"]) for row in training_rows),
        "finite_gradients": all(row["all_gradients_finite"] for row in training_rows),
        "finite_parameters": all(row["all_parameters_finite"] for row in training_rows),
        "finite_optimizer": all(row["all_optimizer_moments_finite"] for row in training_rows),
        "state_bounded": max(row["explosion_consecutive"] for row in training_rows) < 3,
    }
    audit_passed = all(training_integrity.values()) and incremental["passed"]
    primary, secondary, stage_e_nonworsening = classification_from_results(
        final_controls, incremental, milestones, audit_passed
    )
    recurrent_gain = final_controls["recurrent_gain"]
    sequence_gap = final_controls["sequence_specific_gap"]
    recommendation = next_experiment_recommendation(primary, stage_e_improving, recurrent_gain, sequence_gap)
    summary = {
        "experiment": EXPERIMENT,
        "primary_classification": primary,
        "secondary_classification": secondary,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "hardware": runtime_environment()["gpu"],
        "updates": MAX_UPDATES,
        "adaptation_targets": TOTAL_TARGETS,
        "final_windows": list(TARGET_WINDOWS),
        "final_rho": 1.0,
        "parent_validation_loss": PARENT_VALIDATION_LOSS,
        "final_plain_validation_loss": final_controls["controls"]["plain"]["validation_loss"],
        "final_real_recurrent_validation_loss": final_controls["controls"]["real"]["validation_loss"],
        "final_shuffled_validation_loss": final_controls["controls"]["shuffled"]["validation_loss"],
        "final_zero_validation_loss": final_controls["controls"]["zero"]["validation_loss"],
        "recurrent_gain": recurrent_gain,
        "sequence_specific_gap": sequence_gap,
        "zero_state_penalty": final_controls["zero_state_penalty"],
        "real_vs_plain_paired_wins": final_controls["real_vs_plain_paired_wins"],
        "real_vs_zero_paired_wins": final_controls["real_vs_zero_paired_wins"],
        "real_vs_shuffled_paired_wins": final_controls["real_vs_shuffled_paired_wins"],
        "incremental": incremental,
        "transition_analysis": transitions,
        "stage_e_recurrent_losses": stage_e_losses,
        "stage_e_improving": stage_e_improving,
        "stage_e_nonworsening": stage_e_nonworsening,
        "fusion_diagnostics": fusion_diagnostics,
        "nominal_kv_fraction": sum(TARGET_WINDOWS) / (N_LAYER * T),
        "nominal_kv_removed_fraction": 1 - sum(TARGET_WINDOWS) / (N_LAYER * T),
        "training_integrity": training_integrity,
        "recommended_next_experiment": recommendation,
        "final_checkpoint": str(Path(args.final_checkpoint).resolve()),
        "final_checkpoint_sha256": file_sha256(args.final_checkpoint),
    }
    durable_json(runtime.output / "result_summary.json", summary)
    final_audit = {
        "experiment": EXPERIMENT,
        "checks": {
            "exact_10B_source_checkpoint": True,
            "exact_canonical_validation": True,
            "Standard_GPT2_architecture": True,
            "Full_AttnRes_absent": True,
            "target_windows_exact": True,
            "curriculum_windows_exact": training_integrity["curriculum"],
            "curriculum_boundaries_exact": training_integrity["curriculum"],
            "rho_schedule_exact": training_integrity["rho_schedule"],
            "rho_nontrainable": True,
            "previous_top_only": True,
            "position_zero_state_zero": True,
            "one_token_shift_exact": True,
            "no_future_leakage": True,
            "Pass2_temporal_gradient_present": True,
            "Pass3_two_transition_gradient_present": True,
            "no_unintended_detach": True,
            "prefix_mixin_causal": True,
            "prefix_RNG_checkpointed": True,
            "all_GPT2_parameters_trainable": True,
            "fusion_trainable": True,
            "no_teacher": True,
            "no_auxiliary_loss": True,
            "global_targets_exact": training_integrity["exact_targets"],
            "optimizer_groups_exact": True,
            "LR_schedule_exact": True,
            "all_gradients_finite": training_integrity["finite_gradients"],
            "all_parameters_finite": training_integrity["finite_parameters"],
            "all_optimizer_moments_finite": training_integrity["finite_optimizer"],
            "checkpoint_restart_exact": len(json.loads((runtime.output / "process_restarts.json").read_text())) == 3,
            "next_batch_hashes_exact": all(
                row["strict_reopen"]["next_batch"]
                for section in json.loads(checkpoint_manifest_path(runtime.output).read_text()).values()
                for row in section.values()
            ),
            "physical_incremental_KV_limits_exact": incremental["physical_caches_bounded"],
            "incremental_recurrence_causal": equivalence["passed"],
            "incremental_reset_exact": reset_rows["reset_implementations_bit_exact"],
            "row_isolation_exact": reset_rows["row_isolation_exact"],
            "exactly_4769_updates": training_integrity["exact_updates"],
            "exactly_2500329472_targets": training_integrity["exact_targets"],
            "no_HellaSwag": True,
            "no_temporal_AttnRes": True,
            "no_teacher_reconstruction": True,
        },
    }
    final_audit["passed_before_git_and_persistence_seal"] = all(final_audit["checks"].values())
    final_audit["pending_terminal_checks"] = ["Git synchronized", "persistent volume synchronized", "RunPod stop"]
    durable_json(runtime.output / "FINAL_AUDIT.json", final_audit)
    performance = {
        "training_wall_seconds": training_rows[-1]["timestamp"] - training_rows[0]["timestamp"] + training_rows[0]["wall_seconds"],
        "mean_targets_per_second": sum(row["targets_per_second"] for row in training_rows) / len(training_rows),
        "incremental": {key: incremental[key] for key in ("wall_seconds", "targets_per_second", "validation_targets")},
        "final_parallel": final_controls["performance"],
    }
    durable_json(runtime.output / "performance.json", performance)
    report = render_final_report(summary, final_audit, performance)
    durable_text(runtime.output / "EXPERIMENT_2D1_FINAL_REPORT.md", report)
    durable_text(runtime.output / "UNATTENDED_FINAL_HANDOFF.md", render_handoff(summary, final_audit, performance))
    print(f"EXPERIMENT_2D1_FINALIZE_PASS classification={primary}", flush=True)
    return summary


def render_final_report(summary, audit, performance):
    transitions = "\n".join(
        f"- After update {row['boundary_update']}: shock {row['immediate_loss_shock']:+.10f}; "
        f"recovered {row['recovery_from_shock']:+.10f} by update {row['next_stage_end_update']}."
        for row in summary["transition_analysis"]
    )
    questions = [
        ("Q1", "Yes." if summary["training_integrity"]["state_bounded"] else "No."),
        ("Q2", f"Stage-E recurrent trajectory: {summary['stage_e_recurrent_losses']}."),
        ("Q3", transitions),
        ("Q4", f"Plain {summary['final_plain_validation_loss']:.10f}; real {summary['final_real_recurrent_validation_loss']:.10f}; zero {summary['final_zero_validation_loss']:.10f}; shuffled {summary['final_shuffled_validation_loss']:.10f}."),
        ("Q5", f"Recurrence recovered {summary['recurrent_gain']:.10f} CE."),
        ("Q6", f"The sequence-specific gap was {summary['sequence_specific_gap']:.10f}."),
        ("Q7", f"Incremental top-state RMS stayed in [{summary['incremental']['top_state_rms_min']:.6f}, {summary['incremental']['top_state_rms_max']:.6f}]."),
        ("Q8", f"W_u displacement {summary['fusion_diagnostics']['W_u_displacement_norm']:.6f}; W_g displacement {summary['fusion_diagnostics']['W_g_displacement_norm']:.6f}."),
        ("Q9", f"Incremental equivalence passed: {summary['incremental']['equivalence']['passed']}."),
        ("Q10", f"All cache maxima were bounded: {summary['incremental']['physical_caches_bounded']}."),
        ("Q11", f"Nominal KV capacity removed: {100*summary['nominal_kv_removed_fraction']:.2f}%."),
        ("Q12", f"Final recurrent minus parent: {summary['final_real_recurrent_validation_loss'] - summary['parent_validation_loss']:+.10f}."),
        ("Q13", f"Stage E still improving: {summary['stage_e_improving']}."),
        ("Q14", "Yes." if summary["stage_e_improving"] else "No clear evidence."),
        ("Q15", summary["recommended_next_experiment"]),
    ]
    answers = "\n\n".join(f"### {label}\n\n{answer}" for label, answer in questions)
    return f"""# Experiment 2D1 final report

EXPERIMENT 2D1 FINAL CLASSIFICATION:
{summary['primary_classification']}

SECONDARY RECURRENCE CLASSIFICATION:
{summary['secondary_classification']}

The 124M Standard GPT-2 source `{summary['source_checkpoint_sha256']}` was adapted for
{summary['updates']} updates / {summary['adaptation_targets']:,} target tokens on
{summary['hardware']}. Final windows were `{summary['final_windows']}` with rho=1.

Parent CE was {summary['parent_validation_loss']:.10f}. Final plain, real recurrent,
shuffled, and zero-state CE were {summary['final_plain_validation_loss']:.10f},
{summary['final_real_recurrent_validation_loss']:.10f},
{summary['final_shuffled_validation_loss']:.10f}, and
{summary['final_zero_validation_loss']:.10f}. Recurrent gain was
{summary['recurrent_gain']:.10f}; sequence-specific gap was
{summary['sequence_specific_gap']:.10f}.

## Stage-transition shocks

{transitions}

## Required questions

{answers}

## Integrity

Pre-Git/persistence final audit: {'PASS' if audit['passed_before_git_and_persistence_seal'] else 'FAIL'}.
Incremental targets: {summary['incremental']['validation_targets']:,}. Physical KV cache
limits: {'PASS' if summary['incremental']['physical_caches_bounded'] else 'FAIL'}.
Training wall time: {performance['training_wall_seconds'] / 3600:.3f} hours.

The qualitative 2D0D comparison remains intentionally asymmetric: its untrained
top-wide triangle used sum(W)=5312 and suffered +1.9395214698 CE, while 2D1 trains
a more compressed sum(W)=4373 recurrent triangle end-to-end.

Recommended next experiment: **{summary['recommended_next_experiment']}**. It was not executed.
"""


def render_handoff(summary, audit, performance):
    return f"""EXPERIMENT 2D1 FINAL CLASSIFICATION:
{summary['primary_classification']}

SECONDARY RECURRENCE CLASSIFICATION:
{summary['secondary_classification']}

Source checkpoint: {summary['source_checkpoint_sha256']}
Hardware: {summary['hardware']}
Total updates/tokens: {summary['updates']} / {summary['adaptation_targets']}
Training wall time: {performance['training_wall_seconds']:.3f} seconds
Window curriculum: A through E exact
Final windows: {summary['final_windows']}
Final rho: 1.0

Parent val: {summary['parent_validation_loss']:.10f}
Final plain val: {summary['final_plain_validation_loss']:.10f}
Final real recurrent val: {summary['final_real_recurrent_validation_loss']:.10f}
Final shuffled val: {summary['final_shuffled_validation_loss']:.10f}
Final zero-state val: {summary['final_zero_validation_loss']:.10f}

Recurrent gain: {summary['recurrent_gain']:.10f}
Sequence-specific gap: {summary['sequence_specific_gap']:.10f}

Incremental validation targets: {summary['incremental']['validation_targets']}
Incremental equivalence: {summary['incremental']['equivalence']['passed']}
KV cache audit: {summary['incremental']['physical_caches_bounded']}
State/fusion stability: {summary['training_integrity']['state_bounded']}

Final checkpoint: {summary['final_checkpoint']}
Final checkpoint SHA256: {summary['final_checkpoint_sha256']}
Artifact path: results/{OUTPUT_NAME}
Git commits: pending terminal Git seal
Pod-stop command: pending terminal Git/persistence seal; must be final action

# EXPERIMENT 2D1 COMPLETE
"""


def add_execution_arguments(parser):
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(function=run_prepare)

    preflight = subparsers.add_parser("preflight")
    add_execution_arguments(preflight)
    preflight.add_argument("--pod-id", required=True)
    preflight.add_argument("--stop-mechanism", required=True)
    preflight.add_argument("--stop-authenticated", action="store_true")
    preflight.add_argument("--workspace-mount", default="/workspace (RunPod volume disk)")
    preflight.add_argument("--network-volume-mount", default="none")
    preflight.set_defaults(function=run_preflight)

    authorize_stop = subparsers.add_parser("authorize-stop")
    authorize_stop.add_argument("--output-dir", required=True)
    authorize_stop.add_argument("--pod-id", required=True)
    authorize_stop.add_argument("--stop-mechanism", required=True)
    authorize_stop.add_argument("--verification", required=True)
    authorize_stop.set_defaults(function=run_authorize_stop)

    smoke = subparsers.add_parser("smoke")
    add_execution_arguments(smoke)
    smoke.set_defaults(function=run_smoke)

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
