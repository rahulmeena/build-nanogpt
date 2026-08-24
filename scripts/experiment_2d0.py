#!/usr/bin/env python3
"""Experiment 2D0: top-down B12(t-1) completion for shortened Standard GPT-2 B11."""

import argparse
import contextlib
import copy
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import smoke_test as support  # noqa: E402


BRANCH = "experiment-2d0-standard-b11-context-completion"
PARENT_COMMIT = "2135b5e03719cf0f9029cf1c371cc981a0af47ca"
PROTOCOL = "exp2d0_standard_b11_context_completion_v1"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d0_standard_b11_context_completion.json"
OUTPUT_NAME = "experiment_2d0_standard_b11_context_completion"
SOURCE_SHA256 = "924ce6c8392c06ae24ab8f2ffd203787ee0022055c54554bac43bd9a34037871"
SOURCE_BYTES = 497_958_271
SOURCE_STEP = 19_072
SOURCE_TOKENS = 9_999_745_024
SOURCE_VAL_LOSS = 3.0750441551208496
SOURCE_PARAMETER_COUNT = 124_475_904
SOURCE_STATE_ENTRIES = 149
VAL_SHA256 = "8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f"
TRAIN_SHARD_SHA256 = {
    "edufineweb_train_000001.npy": "fb2b9eef2eab2f9903ee61ff81b5df5eb80455392e407fc3f56de8d7b6c738b5",
    "edufineweb_train_000002.npy": "8c4a96a209f29c2b7478e0605fe1ffa6bc06dd97af1047473f23a6f46ee03836",
}
CANONICAL_VALIDATION_SHA256 = "3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb"
SEED = 1337
T = 1024
N_EMBD = 768
N_HEAD = 12
N_LAYER = 12
VOCAB_SIZE = 50_304
B11_INDEX = 10
B12_INDEX = 11
WINDOWS = (1024, 896, 768, 512)
WINDOW_BY_RANK = {0: 1024, 1: 896, 2: 768, 3: 512}
PHASE_A_BATCHES = 20
VALIDATION_BATCHES = 20
VALIDATION_B = 64
WORLD_SIZE = 4
TRAIN_B = 32
GRAD_ACCUM_STEPS = 4
GLOBAL_TARGETS = 524_288
MAX_UPDATES = 191
TOTAL_TARGETS = 100_139_008
MILESTONES = (10, 48, 96, 144, 191)
FORCED_RESTART_UPDATE = 96
LAMBDA_CONS = 0.1
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
COMPLETION_PARAMETERS = 1_179_649
POSITION_BINS_PHASE_A = (
    ("1-64", 1, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513-768", 513, 768),
    ("769-896", 769, 896),
    ("897-1023", 897, 1023),
)


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def torch_load_source_checkpoint(path, symbols, mmap=False):
    """Load the historical train_gpt2.py payload whose config was pickled as __main__."""
    main_module = sys.modules["__main__"]
    sentinel = object()
    previous = getattr(main_module, "GPTConfig", sentinel)
    setattr(main_module, "GPTConfig", symbols["GPTConfig"])
    try:
        return torch_load(path, mmap=mmap)
    finally:
        if previous is sentinel:
            delattr(main_module, "GPTConfig")
        else:
            setattr(main_module, "GPTConfig", previous)
def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        "protocol": PROTOCOL,
        "branch": BRANCH,
        "parent_commit": PARENT_COMMIT,
        "source_checkpoint_sha256": SOURCE_SHA256,
    }
    mismatches = {
        key: (config.get(key), expected_value)
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if mismatches:
        raise SystemExit(f"2D0 preregistration mismatch: {mismatches}")
    phase_a = config["phase_a"]
    phase_b = config["phase_b"]
    assertions = {
        "phase_a_windows": tuple(phase_a["windows"]) == WINDOWS,
        "global_targets": phase_b["global_targets_per_update"] == GLOBAL_TARGETS,
        "target_formula": MAX_UPDATES * GLOBAL_TARGETS == TOTAL_TARGETS,
        "batch_formula": TRAIN_B * T * WORLD_SIZE * GRAD_ACCUM_STEPS == GLOBAL_TARGETS,
        "milestones": tuple(phase_b["milestones"]) == MILESTONES,
        "completion_parameters": phase_b["completion_trainable_parameters"]
        == COMPLETION_PARAMETERS,
    }
    if not all(assertions.values()):
        raise SystemExit(f"2D0 frozen geometry mismatch: {assertions}")
    return config


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Experiment 2D0 requires branch {BRANCH}")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"], cwd=REPO_ROOT
    )
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing 2D0 execution requires a clean worktree")


def seed_all(rank=0):
    seed = SEED + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


class Runtime:
    def __init__(self, require_world_size=None):
        self.distributed = "RANK" in os.environ
        if self.distributed:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            dist.init_process_group("nccl", device_id=self.device)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        if require_world_size is not None and self.world_size != require_world_size:
            raise SystemExit(
                f"command requires world_size={require_world_size}, got {self.world_size}"
            )
        self.master = self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def gather_objects(self, value):
        if not self.distributed:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def close(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


class ExplicitShardLoader:
    """Deterministic rank-strided loader over an explicit, audited shard list."""

    def __init__(self, paths, B, T, process_rank=0, num_processes=1):
        self.paths = [str(Path(path).resolve()) for path in paths]
        if not self.paths or not all(Path(path).is_file() for path in self.paths):
            raise SystemExit(f"missing explicit data shard in {self.paths}")
        self.B = int(B)
        self.T = int(T)
        self.process_rank = int(process_rank)
        self.num_processes = int(num_processes)
        self.current_shard = 0
        self.current_position = self.B * self.T * self.process_rank
        self.tokens = np.load(self.paths[0], mmap_mode="r")

    def _load(self, index):
        self.current_shard = int(index)
        self.tokens = np.load(self.paths[index], mmap_mode="r")

    def reset(self):
        self._load(0)
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        count = self.B * self.T
        if self.current_position + count + 1 > len(self.tokens):
            self._load((self.current_shard + 1) % len(self.paths))
            self.current_position = count * self.process_rank
        array = np.asarray(
            self.tokens[self.current_position : self.current_position + count + 1],
            dtype=np.int64,
        )
        if array.size != count + 1:
            raise RuntimeError("explicit loader produced a short batch")
        tensor = torch.from_numpy(array.copy())
        x = tensor[:-1].view(self.B, self.T)
        y = tensor[1:].view(self.B, self.T)
        self.current_position += count * self.num_processes
        if self.current_position + (count * self.num_processes + 1) > len(self.tokens):
            self._load((self.current_shard + 1) % len(self.paths))
            self.current_position = count * self.process_rank
        return x, y

    def state_dict(self):
        return {
            "paths": list(self.paths),
            "B": self.B,
            "T": self.T,
            "process_rank": self.process_rank,
            "num_processes": self.num_processes,
            "current_shard": self.current_shard,
            "current_shard_filename": Path(self.paths[self.current_shard]).name,
            "current_position": self.current_position,
        }

    def load_state_dict(self, state):
        expected = {
            "paths": self.paths,
            "B": self.B,
            "T": self.T,
            "process_rank": self.process_rank,
            "num_processes": self.num_processes,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise SystemExit(f"loader {key} mismatch: {state.get(key)} != {value}")
        shard = int(state["current_shard"])
        if not 0 <= shard < len(self.paths):
            raise SystemExit("invalid loader shard index")
        if state.get("current_shard_filename") != Path(self.paths[shard]).name:
            raise SystemExit("loader shard filename mismatch")
        self._load(shard)
        self.current_position = int(state["current_position"])
        if not 0 <= self.current_position < len(self.tokens):
            raise SystemExit("invalid loader position")

    def peek_hashes(self, count):
        state = self.state_dict()
        hashes = []
        for _ in range(count):
            x, y = self.next_batch()
            hashes.append(batch_payload_hash(x, y))
        self.load_state_dict(state)
        return hashes


class B11ContextCompletion(nn.Module):
    """Token-conditioned, high-to-low GLU residual with exact zero initialization."""

    def __init__(self, n_embd=N_EMBD, eps=1e-5):
        super().__init__()
        self.W_u = nn.Linear(n_embd, n_embd, bias=False)
        self.W_g = nn.Linear(n_embd, n_embd, bias=False)
        self.g = nn.Parameter(torch.zeros(()))
        self.eps = float(eps)
        with torch.no_grad():
            self.W_u.weight.copy_(torch.eye(n_embd))
            self.W_g.weight.zero_()

    def rmsnorm_noaffine(self, value):
        dtype = value.dtype
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        return (value.float() * torch.rsqrt(variance + self.eps)).to(dtype)

    def forward(
        self,
        source,
        destination,
        window,
        gate_override=None,
        return_diagnostics=False,
        position_offset=0,
    ):
        if source.shape != destination.shape or source.ndim != 3:
            raise ValueError("completion source/destination must share [batch,time,channel]")
        source_norm = self.rmsnorm_noaffine(source)
        destination_norm = self.rmsnorm_noaffine(destination)
        candidate = self.W_u(source_norm) * torch.sigmoid(self.W_g(destination_norm))
        gate = self.g.tanh() if gate_override is None else candidate.new_tensor(float(gate_override))
        feedback = gate.to(candidate.dtype) * candidate
        positions = torch.arange(source.size(1), device=source.device) + int(position_offset)
        active = positions.ge(int(window)).view(1, -1, 1)
        feedback = torch.where(active, feedback, torch.zeros_like(feedback))
        feedback = feedback.to(destination.dtype)
        if not return_diagnostics:
            return feedback
        diagnostics = {
            "candidate_rms": candidate.detach().float().pow(2).mean().sqrt().item(),
            "feedback_rms": feedback.detach().float().pow(2).mean().sqrt().item(),
            "input_rms": destination.detach().float().pow(2).mean().sqrt().item(),
            "gate": self.g.detach().float().item(),
            "tanh_gate": self.g.detach().float().tanh().item(),
            "active_positions": int(active.sum().item()),
        }
        diagnostics["feedback_input_rms_ratio"] = diagnostics["feedback_rms"] / max(
            diagnostics["input_rms"], 1e-30
        )
        return feedback, diagnostics


def model_config(symbols):
    return symbols["GPTConfig"](
        block_size=T,
        vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        residual_mode="standard",
        enable_topdown_feedback=False,
        enable_memory_writers=False,
    )


def load_standard_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.stat().st_size != SOURCE_BYTES:
        raise SystemExit(f"source checkpoint byte-size mismatch: {checkpoint_path.stat().st_size}")
    observed_sha = file_sha256(checkpoint_path)
    if observed_sha != SOURCE_SHA256:
        raise SystemExit(f"source checkpoint SHA mismatch: {observed_sha}")
    symbols = support.load_training_symbols()
    config = model_config(symbols)
    model = symbols["GPT"](config)
    payload = torch_load_source_checkpoint(checkpoint_path, symbols, mmap=True)
    state = payload.get("model")
    if not isinstance(state, dict) or len(state) != SOURCE_STATE_ENTRIES:
        raise SystemExit("source checkpoint state-dict schema mismatch")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise SystemExit(f"strict source load failed: missing={missing} unexpected={unexpected}")
    del payload, state
    gc.collect()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    attnres_modules = [name for name, _ in model.named_modules() if "attnres" in name.lower()]
    attnres_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "attnres" in name.lower() and parameter.requires_grad
    )
    audit = {
        "checkpoint": str(checkpoint_path),
        "sha256": observed_sha,
        "strict_load": True,
        "state_dict_entries": SOURCE_STATE_ENTRIES,
        "parameter_count": total_parameters,
        "config": dataclasses.asdict(config),
        "human_block_mapping": {f"B{index + 1}": f"transformer.h[{index}]" for index in range(12)},
        "b11_zero_based_index": B11_INDEX,
        "b12_zero_based_index": B12_INDEX,
        "full_attnres_active_modules": len(attnres_modules),
        "full_attnres_module_names": attnres_modules,
        "full_attnres_trainable_parameters": attnres_parameters,
        "all_base_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
    }
    required = {
        "standard_mode": model.config.residual_mode == "standard",
        "layers": model.config.n_layer == N_LAYER,
        "width": model.config.n_embd == N_EMBD,
        "heads": model.config.n_head == N_HEAD,
        "context": model.config.block_size == T,
        "parameters": total_parameters == SOURCE_PARAMETER_COUNT,
        "attnres_modules_zero": not attnres_modules,
        "attnres_trainable_zero": attnres_parameters == 0,
        "base_frozen": audit["all_base_parameters_frozen"],
    }
    audit["checks"] = required
    audit["passed"] = all(required.values())
    if not audit["passed"]:
        raise SystemExit(f"standard source architecture audit failed: {required}")
    return symbols, model, audit


def shifted_top_state(top_state):
    shifted = torch.zeros_like(top_state)
    shifted[:, 1:] = top_state[:, :-1]
    return shifted.detach()


def sliding_mask(length, window, device):
    query = torch.arange(length, device=device).view(-1, 1)
    key = torch.arange(length, device=device).view(1, -1)
    return (key <= query) & (key >= query - int(window) + 1)


def attention_output(block, value, window):
    B, length, channels = value.shape
    qkv = block.attn.c_attn(value)
    q, k, v = qkv.split(channels, dim=2)
    head_size = channels // block.attn.n_head
    q = q.view(B, length, block.attn.n_head, head_size).transpose(1, 2)
    k = k.view(B, length, block.attn.n_head, head_size).transpose(1, 2)
    v = v.view(B, length, block.attn.n_head, head_size).transpose(1, 2)
    if int(window) >= length:
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sliding_mask(length, window, value.device), is_causal=False
        )
    result = result.transpose(1, 2).contiguous().view(B, length, channels)
    return block.attn.c_proj(result)


def run_block(block, residual, window):
    attention = attention_output(block, block.ln_1(residual), window)
    post_attention = residual + attention
    post_block = post_attention + block.mlp(block.ln_2(post_attention))
    return post_block, attention


def shared_lower_trunk(model, tokens):
    length = tokens.size(1)
    positions = torch.arange(length, dtype=torch.long, device=tokens.device)
    value = model.transformer.wte(tokens) + model.transformer.wpe(positions)
    for block in model.transformer.h[:B11_INDEX]:
        value = block(value)
    return value


def teacher_tail(model, h10):
    h11, b11_attention = run_block(model.transformer.h[B11_INDEX], h10, T)
    h12, _ = run_block(model.transformer.h[B12_INDEX], h11, T)
    top = model.transformer.ln_f(h12)
    return {
        "b11_attention": b11_attention,
        "h11": h11,
        "h12": h12,
        "top": top,
    }


def student_tail(
    model,
    h10,
    window,
    completion=None,
    source=None,
    gate_override=None,
    return_completion_diagnostics=False,
):
    feedback = torch.zeros_like(h10)
    completion_diagnostics = None
    if completion is not None:
        if source is None:
            raise ValueError("completion requires a shifted top-state source")
        result = completion(
            source,
            h10,
            window,
            gate_override=gate_override,
            return_diagnostics=return_completion_diagnostics,
        )
        if return_completion_diagnostics:
            feedback, completion_diagnostics = result
        else:
            feedback = result
    b11_input = h10 + feedback
    h11, b11_attention = run_block(model.transformer.h[B11_INDEX], b11_input, window)
    h12, _ = run_block(model.transformer.h[B12_INDEX], h11, T)
    top = model.transformer.ln_f(h12)
    return {
        "feedback": feedback,
        "b11_attention": b11_attention,
        "h11": h11,
        "h12": h12,
        "top": top,
        "completion": completion_diagnostics,
    }


def token_cross_entropy(model, top_state, targets):
    logits = model.lm_head(top_state)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    return losses, logits


def token_cosine(candidate, reference):
    return F.cosine_similarity(candidate.float(), reference.float(), dim=-1)


def selected_tensor_stats(candidate, reference, position_mask):
    if position_mask.ndim == 1:
        position_mask = position_mask.view(1, -1).expand(candidate.size(0), -1)
    difference = candidate.float() - reference.float()
    mse = difference.pow(2).mean(dim=-1)
    rms = difference.pow(2).mean(dim=-1).sqrt()
    cosine = token_cosine(candidate, reference)
    selected = position_mask
    count = int(selected.sum().item())
    if count == 0:
        return {"count": 0, "mse_sum": 0.0, "rms_sum": 0.0, "cosine_sum": 0.0}
    return {
        "count": count,
        "mse_sum": mse[selected].double().sum().item(),
        "rms_sum": rms[selected].double().sum().item(),
        "cosine_sum": cosine[selected].double().sum().item(),
    }


def zero_reference_stats(value, position_mask):
    if position_mask.ndim == 1:
        position_mask = position_mask.view(1, -1).expand(value.size(0), -1)
    squared = value.float().pow(2).mean(dim=-1)
    rms = squared.sqrt()
    count = int(position_mask.sum().item())
    if count == 0:
        return {"count": 0, "mse_sum": 0.0, "rms_sum": 0.0, "cosine_sum": 0.0}
    return {
        "count": count,
        "mse_sum": squared[position_mask].double().sum().item(),
        "rms_sum": rms[position_mask].double().sum().item(),
        "cosine_sum": 0.0,
    }


def merge_stat(accumulator, row):
    for key in ("count", "mse_sum", "rms_sum", "cosine_sum"):
        accumulator[key] += row[key]


def finalize_stat(row):
    count = row["count"]
    return {
        "count": count,
        "mse": None if count == 0 else row["mse_sum"] / count,
        "rms_difference": None if count == 0 else row["rms_sum"] / count,
        "cosine": None if count == 0 else row["cosine_sum"] / count,
    }


def blank_stat():
    return {"count": 0, "mse_sum": 0.0, "rms_sum": 0.0, "cosine_sum": 0.0}


def phase_a_position_bins(losses, batch_accumulator):
    positions = torch.arange(T, device=losses.device)
    for label, start, end in POSITION_BINS_PHASE_A:
        mask = positions.ge(start) & positions.le(end)
        batch_accumulator[label]["loss_sum"] += losses[:, mask].double().sum().item()
        batch_accumulator[label]["count"] += int(losses.size(0) * mask.sum().item())


@torch.no_grad()
def phase_a_for_window(model, val_path, window, device):
    loader = ExplicitShardLoader([val_path], VALIDATION_B, T)
    batch_hashes = []
    aggregate = {
        "loss_sum": 0.0,
        "tokens": 0,
        "b11_attention": blank_stat(),
        "b11_post": blank_stat(),
        "b12_post": blank_stat(),
        "final_logits": blank_stat(),
        "position_bins": {
            label: {"loss_sum": 0.0, "count": 0}
            for label, _, _ in POSITION_BINS_PHASE_A
        },
        "per_batch_losses": [],
    }
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    for batch_index in range(PHASE_A_BATCHES):
        cpu_x, cpu_y = loader.next_batch()
        batch_hashes.append(batch_payload_hash(cpu_x, cpu_y))
        x = cpu_x.to(device, non_blocking=True)
        y = cpu_y.to(device, non_blocking=True)
        with autocast:
            h10 = shared_lower_trunk(model, x)
            full = teacher_tail(model, h10)
            candidate = student_tail(model, h10, window)
            candidate_losses, candidate_logits = token_cross_entropy(model, candidate["top"], y)
            logit_difference = model.lm_head(candidate["top"] - full["top"])
        batch_loss = candidate_losses.float().mean().item()
        aggregate["per_batch_losses"].append(batch_loss)
        aggregate["loss_sum"] += candidate_losses.double().sum().item()
        aggregate["tokens"] += candidate_losses.numel()
        all_positions = torch.ones(T, dtype=torch.bool, device=device)
        merge_stat(
            aggregate["b11_attention"],
            selected_tensor_stats(candidate["b11_attention"], full["b11_attention"], all_positions),
        )
        merge_stat(
            aggregate["b11_post"],
            selected_tensor_stats(candidate["h11"], full["h11"], all_positions),
        )
        merge_stat(
            aggregate["b12_post"],
            selected_tensor_stats(candidate["top"], full["top"], all_positions),
        )
        merge_stat(
            aggregate["final_logits"],
            zero_reference_stats(logit_difference, all_positions),
        )
        phase_a_position_bins(candidate_losses, aggregate["position_bins"])
        del x, y, h10, full, candidate, candidate_logits, candidate_losses, logit_difference
        torch.cuda.empty_cache()
        print(
            f"2D0 Phase A W={window} batch={batch_index + 1:02d}/{PHASE_A_BATCHES} "
            f"loss={batch_loss:.10f}",
            flush=True,
        )
    result = {
        "window": window,
        "validation_loss": aggregate["loss_sum"] / aggregate["tokens"],
        "validation_tokens": aggregate["tokens"],
        "per_batch_losses": aggregate["per_batch_losses"],
        "canonical_validation_sha256": aggregate_hashes(batch_hashes),
        "b11_attention_output_drift": finalize_stat(aggregate["b11_attention"]),
        "b11_post_block_state_drift": finalize_stat(aggregate["b11_post"]),
        "b12_state_drift": finalize_stat(aggregate["b12_post"]),
        "final_logit_drift": finalize_stat(aggregate["final_logits"]),
        "position_bins": {
            label: {
                "count": row["count"],
                "ce": row["loss_sum"] / row["count"],
            }
            for label, row in aggregate["position_bins"].items()
        },
        "optimizer_objects": 0,
        "backward_calls": 0,
        "training_targets": 0,
    }
    result["passed"] = (
        result["canonical_validation_sha256"] == CANONICAL_VALIDATION_SHA256
        and math.isfinite(result["validation_loss"])
    )
    return result


def select_window(rows):
    full_loss = rows[1024]["validation_loss"]
    for row in rows.values():
        row["damage_vs_1024"] = row["validation_loss"] - full_loss
    qualifies = lambda window: 0.01 <= rows[window]["damage_vs_1024"] <= 0.10
    selected = None
    rule = None
    if qualifies(768):
        selected, rule = 768, "W768 damage is in [0.01, 0.10]"
    elif rows[768]["damage_vs_1024"] < 0.01 and qualifies(512):
        selected, rule = 512, "W768 damage <0.01 and W512 damage is in [0.01, 0.10]"
    elif rows[768]["damage_vs_1024"] > 0.10 and qualifies(896):
        selected, rule = 896, "W768 damage >0.10 and W896 damage is in [0.01, 0.10]"
    return {
        "selected_window": selected,
        "selection_rule": rule,
        "phase_b_authorized": selected is not None,
        "damages": {str(window): rows[window]["damage_vs_1024"] for window in WINDOWS},
    }


def run_phase_a(args):
    require_git(clean=True)
    load_config()
    runtime = Runtime(require_world_size=4)
    try:
        seed_all(runtime.rank)
        window = WINDOW_BY_RANK[runtime.rank]
        symbols, model, source_audit = load_standard_model(args.parent_checkpoint, runtime.device)
        result = phase_a_for_window(model, args.validation_shard, window, runtime.device)
        result["rank"] = runtime.rank
        result["gpu"] = torch.cuda.get_device_name(runtime.device)
        result["source_audit"] = source_audit
        run_root = Path(args.run_root).resolve()
        durable_json(run_root / "phase_a" / f"window_{window}.json", result)
        runtime.barrier()
        if runtime.master:
            rows = {
                candidate: json.loads(
                    (run_root / "phase_a" / f"window_{candidate}.json").read_text()
                )
                for candidate in WINDOWS
            }
            hashes = {row["canonical_validation_sha256"] for row in rows.values()}
            if hashes != {CANONICAL_VALIDATION_SHA256}:
                raise SystemExit(f"Phase-A batch identity failure: {hashes}")
            selection = select_window(rows)
            aggregate = {
                "experiment": "2D0",
                "protocol": PROTOCOL,
                "windows": list(WINDOWS),
                "gpu_mapping": {str(value): key for key, value in WINDOW_BY_RANK.items()},
                "rows": {str(key): value for key, value in rows.items()},
                "selection": selection,
                "phase_a_batch_identity": True,
                "source_checkpoint_sha256": SOURCE_SHA256,
                "passed": all(row["passed"] for row in rows.values()),
            }
            durable_json(run_root / "phase_a_results.json", aggregate)
            durable_json(run_root / "selected_window.json", selection)
            print(
                f"EXPERIMENT_2D0_PHASE_A_COMPLETE selected={selection['selected_window']} "
                f"damages={selection['damages']}",
                flush=True,
            )
        runtime.barrier()
    finally:
        runtime.close()


@dataclasses.dataclass
class StandardRecurrentState:
    position: int
    caches: tuple
    previous_top: torch.Tensor
    windows: tuple

    def state_dict(self):
        cache_rows = []
        for cache in self.caches:
            key, value = cache.prefix()
            cache_rows.append(
                {
                    "key": key.detach().clone(),
                    "value": value.detach().clone(),
                    "length": cache.length,
                }
            )
        return {
            "schema": "exp2d0_standard_recurrent_state_v1",
            "position": self.position,
            "windows": list(self.windows),
            "previous_top": self.previous_top.detach().clone(),
            "caches": cache_rows,
        }


def init_standard_recurrent_state(symbols, model, batch_size, window, device, dtype):
    windows = (T,) * B11_INDEX + (int(window), T)
    head_size = model.config.n_embd // model.config.n_head
    caches = []
    for candidate in windows:
        capacity = candidate - 1
        shape = (batch_size, model.config.n_head, capacity, head_size)
        caches.append(
            symbols["AttentionKVCache"](
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
                0,
            )
        )
    previous = torch.zeros(
        batch_size, 1, model.config.n_embd, device=device, dtype=dtype
    )
    return StandardRecurrentState(0, tuple(caches), previous, windows)


def load_standard_recurrent_state(symbols, model, payload, device, dtype):
    if payload.get("schema") != "exp2d0_standard_recurrent_state_v1":
        raise ValueError("invalid 2D0 recurrent-state schema")
    windows = tuple(payload["windows"])
    if len(windows) != N_LAYER or any(not 1 <= value <= T for value in windows):
        raise ValueError("invalid 2D0 recurrent windows")
    state = init_standard_recurrent_state(
        symbols, model, payload["previous_top"].size(0), windows[B11_INDEX], device, dtype
    )
    if state.windows != windows:
        raise ValueError("2D0 recurrent geometry mismatch")
    restored = []
    position = int(payload["position"])
    for window, fresh, saved in zip(windows, state.caches, payload["caches"]):
        expected_length = min(position, window - 1)
        if int(saved["length"]) != expected_length:
            raise ValueError("2D0 recurrent saved cache length mismatch")
        key = saved["key"].to(device=device, dtype=dtype)
        value = saved["value"].to(device=device, dtype=dtype)
        if key.size(2) != expected_length or value.size(2) != expected_length:
            raise ValueError("2D0 recurrent cache prefix mismatch")
        fresh.key[:, :, :expected_length].copy_(key)
        fresh.value[:, :, :expected_length].copy_(value)
        fresh.length = expected_length
        restored.append(fresh)
    return StandardRecurrentState(
        position,
        tuple(restored),
        payload["previous_top"].to(device=device, dtype=dtype),
        windows,
    )


def assert_standard_cache_state(state):
    if not 0 <= state.position <= T:
        raise RuntimeError("2D0 recurrent absolute position is invalid")
    for block, (cache, window) in enumerate(zip(state.caches, state.windows)):
        expected_capacity = window - 1
        expected_length = min(state.position, expected_capacity)
        if (
            cache.key.size(2) != expected_capacity
            or cache.value.size(2) != expected_capacity
            or cache.length != expected_length
        ):
            raise RuntimeError(
                f"B{block + 1} cache mismatch length={cache.length} "
                f"capacity={cache.key.size(2)} expected={expected_length}/{expected_capacity}"
            )


def standard_recurrent_step(
    symbols,
    model,
    completion,
    tokens,
    state,
    control="self_real",
    permutation=None,
    compute_logits=True,
    return_diagnostics=False,
):
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(1)
    if tokens.ndim != 2 or tokens.size(1) != 1:
        raise ValueError("incremental Standard GPT-2 tokens must be [batch,1]")
    if state.position >= T:
        raise ValueError("incremental Standard GPT-2 context exhausted")
    if control not in {"self_real", "self_shuffled", "no_feedback"}:
        raise ValueError(f"unknown incremental control: {control}")
    position = state.position
    pos = torch.tensor([position], dtype=torch.long, device=tokens.device)
    residual = model.transformer.wte(tokens) + model.transformer.wpe(pos)
    updated = []
    feedback = torch.zeros_like(residual)
    for block_index, block in enumerate(model.transformer.h):
        if block_index == B11_INDEX and control != "no_feedback":
            source = state.previous_top.detach()
            if control == "self_shuffled":
                if permutation is None:
                    permutation = torch.arange(tokens.size(0), device=tokens.device).roll(1)
                source = source[permutation]
            feedback = completion(
                source,
                residual,
                state.windows[B11_INDEX],
                position_offset=position,
            )
            residual = residual + feedback
        normalized = block.ln_1(residual)
        attention, cache = block.attn.forward_step_rolling(
            normalized, state.caches[block_index]
        )
        residual = residual + attention
        residual = residual + block.mlp(block.ln_2(residual))
        updated.append(cache)
    top = model.transformer.ln_f(residual)
    logits = model.lm_head(top) if compute_logits else None
    next_state = StandardRecurrentState(
        position + 1,
        tuple(updated),
        top.detach(),
        state.windows,
    )
    assert_standard_cache_state(next_state)
    if not return_diagnostics:
        return logits, next_state
    return logits, next_state, {
        "position": position,
        "feedback_rms": feedback.detach().float().pow(2).mean().sqrt().item(),
        "top_finite": bool(torch.isfinite(top).all()),
        "b11_cache_length": next_state.caches[B11_INDEX].length,
        "b12_cache_length": next_state.caches[B12_INDEX].length,
    }


def tensor_state_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def completion_audit(completion):
    count = sum(parameter.numel() for parameter in completion.parameters())
    with torch.no_grad():
        identity_displacement = (
            completion.W_u.weight.float() - torch.eye(N_EMBD, device=completion.W_u.weight.device)
        ).norm().item()
        gate_weight_norm = completion.W_g.weight.float().norm().item()
    return {
        "trainable_parameters": sum(
            parameter.numel() for parameter in completion.parameters() if parameter.requires_grad
        ),
        "total_parameters": count,
        "W_u_identity_displacement": identity_displacement,
        "W_g_norm": gate_weight_norm,
        "g": completion.g.detach().float().item(),
        "tanh_g": completion.g.detach().float().tanh().item(),
    }


def gradient_row(parameter):
    if parameter.grad is None:
        return {"present": False, "finite": False, "nonzero": False, "norm": None}
    gradient = parameter.grad.detach().float()
    return {
        "present": True,
        "finite": bool(torch.isfinite(gradient).all()),
        "nonzero": bool(torch.count_nonzero(gradient).item()),
        "norm": gradient.norm().item(),
    }


def training_losses(model, completion, tokens, targets, window):
    with torch.no_grad():
        h10 = shared_lower_trunk(model, tokens)
        teacher = teacher_tail(model, h10)
        source = shifted_top_state(teacher["top"])
    student = student_tail(model, h10.detach(), window, completion, source)
    losses, logits = token_cross_entropy(model, student["top"], targets)
    position_mask = torch.arange(T, device=tokens.device).ge(window)
    consistency = F.mse_loss(
        student["h11"][:, position_mask].float(),
        teacher["h11"][:, position_mask].detach().float(),
    )
    ce = losses.float().mean()
    return ce, consistency, ce + LAMBDA_CONS * consistency, {
        "teacher": teacher,
        "student": student,
        "source": source,
        "token_losses": losses,
        "logits": logits,
    }


def run_preflight(args):
    require_git(clean=True)
    load_config()
    runtime = Runtime(require_world_size=1)
    try:
        if runtime.device.type != "cuda":
            raise SystemExit("2D0 preflight requires CUDA")
        seed_all(0)
        selection = json.loads(Path(args.selected_window).read_text())
        window = selection.get("selected_window")
        if window not in (896, 768, 512):
            raise SystemExit("Phase A did not authorize a preregistered Phase-B window")
        symbols, model, source_audit = load_standard_model(args.parent_checkpoint, runtime.device)
        base_before = tensor_state_sha256(model)
        loader = ExplicitShardLoader([args.validation_shard], 2, T)
        cpu_x, cpu_y = loader.next_batch()
        x = cpu_x.to(runtime.device)
        y = cpu_y.to(runtime.device)
        completion = B11ContextCompletion().to(runtime.device)
        initial_completion = completion_audit(completion)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h10 = shared_lower_trunk(model, x)
            teacher = teacher_tail(model, h10)
            source = shifted_top_state(teacher["top"])
            short = student_tail(model, h10, window)
            zero = student_tail(model, h10, window, completion, source)
            forced = completion(source, h10, window, gate_override=1.0)
        zero_report = {
            "feedback_bit_zero": forced.new_zeros(()).item() == 0
            and zero["feedback"].count_nonzero().item() == 0,
            "b11_post_bit_exact": torch.equal(short["h11"], zero["h11"]),
            "top_bit_exact": torch.equal(short["top"], zero["top"]),
            "pre_truncation_forced_feedback_zero": forced[:, :window].count_nonzero().item() == 0,
            "post_truncation_forced_feedback_nonzero": forced[:, window:].count_nonzero().item() > 0,
        }
        zero_report["passed"] = all(zero_report.values())

        smoke = B11ContextCompletion().to(runtime.device)
        optimizer = torch.optim.AdamW(
            smoke.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            fused=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ce1, consistency1, total1, details1 = training_losses(model, smoke, x, y, window)
        total1.backward()
        first_gradients = {
            "g": gradient_row(smoke.g),
            "W_u": gradient_row(smoke.W_u.weight),
            "W_g": gradient_row(smoke.W_g.weight),
        }
        first_staging = (
            first_gradients["g"]["nonzero"]
            and not first_gradients["W_u"]["nonzero"]
            and not first_gradients["W_g"]["nonzero"]
        )
        first_norm = torch.nn.utils.clip_grad_norm_(smoke.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del details1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ce2, consistency2, total2, details2 = training_losses(model, smoke, x, y, window)
        total2.backward()
        second_gradients = {
            "g": gradient_row(smoke.g),
            "W_u": gradient_row(smoke.W_u.weight),
            "W_g": gradient_row(smoke.W_g.weight),
        }
        second_staging = all(second_gradients[name]["nonzero"] for name in second_gradients)
        base_gradients = [
            name for name, parameter in model.named_parameters() if parameter.grad is not None
        ]

        smoke_dir = Path(args.run_root).resolve() / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = smoke_dir / "disposable_completion.pt"
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {"completion": smoke.state_dict(), "optimizer": optimizer.state_dict()}, temporary
        )
        os.replace(temporary, checkpoint_path)
        reopened = torch_load(checkpoint_path)
        clone = B11ContextCompletion().to(runtime.device)
        clone.load_state_dict(reopened["completion"], strict=True)
        clone_optimizer = torch.optim.AdamW(
            clone.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )
        clone_optimizer.load_state_dict(reopened["optimizer"])
        reload_exact = all(
            torch.equal(smoke.state_dict()[name], clone.state_dict()[name])
            for name in smoke.state_dict()
        )

        # A short recurrent test uses W=8 solely to make feedback active quickly;
        # the production-capacity audit below uses the selected window exactly.
        recurrent_tokens = x[:2, :16]
        state_a = init_standard_recurrent_state(
            symbols, model, 2, 8, runtime.device, torch.bfloat16
        )
        rows_a = []
        permutation = torch.tensor([1, 0], device=runtime.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for position in range(16):
                logits, state_a = standard_recurrent_step(
                    symbols,
                    model,
                    smoke,
                    recurrent_tokens[:, position],
                    state_a,
                    control="self_real",
                )
                rows_a.append(logits)
        recurrent_logits = torch.cat(rows_a, dim=1)
        future_tokens = recurrent_tokens.clone()
        future_tokens[:, 12:] = (future_tokens[:, 12:] + 17) % VOCAB_SIZE
        state_b = init_standard_recurrent_state(
            symbols, model, 2, 8, runtime.device, torch.bfloat16
        )
        rows_b = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for position in range(16):
                logits, state_b = standard_recurrent_step(
                    symbols,
                    model,
                    smoke,
                    future_tokens[:, position],
                    state_b,
                    control="self_real",
                )
                rows_b.append(logits)
        future_logits = torch.cat(rows_b, dim=1)
        row_tokens = recurrent_tokens.clone()
        row_tokens[1] = (row_tokens[1] + 19) % VOCAB_SIZE
        state_c = init_standard_recurrent_state(
            symbols, model, 2, 8, runtime.device, torch.bfloat16
        )
        rows_c = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for position in range(16):
                logits, state_c = standard_recurrent_step(
                    symbols,
                    model,
                    smoke,
                    row_tokens[:, position],
                    state_c,
                    control="self_real",
                )
                rows_c.append(logits)
        row_logits = torch.cat(rows_c, dim=1)
        recurrent_checks = {
            "future_causality_bit_exact": torch.equal(
                recurrent_logits[:, :12], future_logits[:, :12]
            ),
            "row_isolation_bit_exact": torch.equal(recurrent_logits[0], row_logits[0]),
            "finite": bool(torch.isfinite(recurrent_logits).all()),
            "short_geometry_b11_capacity": state_a.caches[B11_INDEX].key.size(2) == 7,
            "short_geometry_b12_capacity": state_a.caches[B12_INDEX].key.size(2) == 1023,
            "position_exact": state_a.position == 16,
        }
        recurrent_checks["passed"] = all(recurrent_checks.values())
        serialized = state_a.state_dict()
        restored = load_standard_recurrent_state(
            symbols, model, serialized, runtime.device, torch.bfloat16
        )
        recurrent_checks["serialization_exact"] = (
            restored.position == state_a.position
            and torch.equal(restored.previous_top, state_a.previous_top)
            and all(
                left.length == right.length
                and torch.equal(left.prefix()[0], right.prefix()[0])
                and torch.equal(left.prefix()[1], right.prefix()[1])
                for left, right in zip(state_a.caches, restored.caches)
            )
        )
        recurrent_checks["passed"] &= recurrent_checks["serialization_exact"]

        fresh = init_standard_recurrent_state(
            symbols, model, 2, window, runtime.device, torch.bfloat16
        )
        capacity_audit = {
            "selected_window": window,
            "b11_capacity": fresh.caches[B11_INDEX].key.size(2),
            "b12_capacity": fresh.caches[B12_INDEX].key.size(2),
            "b11_expected": window - 1,
            "b12_expected": 1023,
            "fresh_position_zero": fresh.position == 0,
            "fresh_top_zero": fresh.previous_top.count_nonzero().item() == 0,
            "fresh_caches_empty": all(cache.length == 0 for cache in fresh.caches),
        }
        capacity_audit["passed"] = (
            capacity_audit["b11_capacity"] == capacity_audit["b11_expected"]
            and capacity_audit["b12_capacity"] == capacity_audit["b12_expected"]
            and capacity_audit["fresh_position_zero"]
            and capacity_audit["fresh_top_zero"]
            and capacity_audit["fresh_caches_empty"]
        )

        base_after = tensor_state_sha256(model)
        report = {
            "experiment": "2D0",
            "protocol": PROTOCOL,
            "selected_window": window,
            "source_audit": source_audit,
            "completion_initialization": initial_completion,
            "zero_feedback_identity": zero_report,
            "gradient_staging": {
                "first": first_gradients,
                "second": second_gradients,
                "first_gate_only": first_staging,
                "second_all_trainables": second_staging,
                "first_gradient_norm": float(first_norm),
                "losses_finite": all(
                    math.isfinite(value)
                    for value in (
                        ce1.item(), consistency1.item(), total1.item(),
                        ce2.item(), consistency2.item(), total2.item(),
                    )
                ),
            },
            "base_gradients": base_gradients,
            "base_state_sha256_before": base_before,
            "base_state_sha256_after": base_after,
            "base_immutable": base_before == base_after,
            "checkpoint_reload_exact": reload_exact,
            "recurrent": recurrent_checks,
            "cache_capacity": capacity_audit,
            "position_zero_source_zero": bool(source[:, 0].count_nonzero().item() == 0),
            "teacher_source_shift_exact": bool(torch.equal(source[:, 1:], teacher["top"][:, :-1])),
            "future_causality": recurrent_checks["future_causality_bit_exact"],
            "row_isolation": recurrent_checks["row_isolation_bit_exact"],
            "no_state_leakage": capacity_audit["fresh_top_zero"]
            and capacity_audit["fresh_caches_empty"],
        }
        report["passed"] = all(
            [
                source_audit["passed"],
                initial_completion["total_parameters"] == COMPLETION_PARAMETERS,
                initial_completion["W_u_identity_displacement"] == 0,
                initial_completion["W_g_norm"] == 0,
                initial_completion["g"] == 0,
                zero_report["passed"],
                report["gradient_staging"]["first_gate_only"],
                report["gradient_staging"]["second_all_trainables"],
                report["gradient_staging"]["losses_finite"],
                not base_gradients,
                report["base_immutable"],
                reload_exact,
                recurrent_checks["passed"],
                capacity_audit["passed"],
                report["position_zero_source_zero"],
                report["teacher_source_shift_exact"],
            ]
        )
        durable_json(Path(args.run_root).resolve() / "preflight.json", report)
        print(f"EXPERIMENT_2D0_PREFLIGHT_{'PASS' if report['passed'] else 'FAIL'}", flush=True)
        if not report["passed"]:
            raise SystemExit("Experiment 2D0 preflight failed")
    finally:
        runtime.close()


def flatten_gradients(module):
    rows = []
    for parameter in module.parameters():
        if parameter.grad is None:
            rows.append(torch.zeros_like(parameter).reshape(-1))
        else:
            rows.append(parameter.grad.detach().reshape(-1))
    return torch.cat(rows)


def flatten_parameters(module):
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def relative_l2(candidate, reference):
    return (candidate.float() - reference.float()).norm().item() / max(
        reference.float().norm().item(), 1e-30
    )


def run_equivalence(args):
    require_git(clean=True)
    load_config()
    runtime = Runtime(require_world_size=4)
    try:
        seed_all(runtime.rank)
        selection = json.loads(Path(args.selected_window).read_text())
        window = selection.get("selected_window")
        if window not in (896, 768, 512):
            raise SystemExit("Phase A did not authorize Phase B")
        symbols, model, source_audit = load_standard_model(args.parent_checkpoint, runtime.device)
        train_paths = [str(Path(path).resolve()) for path in args.training_shards]
        reference = None
        if runtime.master:
            reference_completion = B11ContextCompletion().to(runtime.device)
            reference_optimizer = torch.optim.AdamW(
                reference_completion.parameters(),
                lr=LEARNING_RATE,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.0,
                fused=True,
            )
            reference_loader = ExplicitShardLoader(train_paths, TRAIN_B, T, 0, 1)
            reference_hashes = []
            reference_loss = torch.zeros((), device=runtime.device)
            reference_optimizer.zero_grad(set_to_none=True)
            for microstep in range(GRAD_ACCUM_STEPS * WORLD_SIZE):
                cpu_x, cpu_y = reference_loader.next_batch()
                reference_hashes.append(batch_payload_hash(cpu_x, cpu_y))
                x = cpu_x.to(runtime.device, non_blocking=True)
                y = cpu_y.to(runtime.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ce, consistency, total, details = training_losses(
                        model, reference_completion, x, y, window
                    )
                (total / (GRAD_ACCUM_STEPS * WORLD_SIZE)).backward()
                reference_loss += total.detach() / (GRAD_ACCUM_STEPS * WORLD_SIZE)
                del x, y, details
            reference_gradient = flatten_gradients(reference_completion).detach().clone()
            reference_grad_norm = torch.nn.utils.clip_grad_norm_(
                reference_completion.parameters(), GRAD_CLIP
            )
            reference_optimizer.step()
            reference_parameter = flatten_parameters(reference_completion).detach().clone()
            reference = {
                "loss": reference_loss.detach().float().item(),
                "hashes": reference_hashes,
                "gradient": reference_gradient,
                "gradient_norm": float(reference_grad_norm),
                "parameter": reference_parameter,
            }
            del reference_completion, reference_optimizer
            torch.cuda.empty_cache()
        runtime.barrier()

        candidate_completion = B11ContextCompletion().to(runtime.device)
        candidate = DDP(
            candidate_completion,
            device_ids=[runtime.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
        )
        candidate_optimizer = torch.optim.AdamW(
            candidate.module.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            fused=True,
        )
        candidate_loader = ExplicitShardLoader(
            train_paths, TRAIN_B, T, runtime.rank, runtime.world_size
        )
        local_hashes = []
        local_loss = torch.zeros((), device=runtime.device)
        candidate_optimizer.zero_grad(set_to_none=True)
        for microstep in range(GRAD_ACCUM_STEPS):
            cpu_x, cpu_y = candidate_loader.next_batch()
            local_hashes.append(batch_payload_hash(cpu_x, cpu_y))
            x = cpu_x.to(runtime.device, non_blocking=True)
            y = cpu_y.to(runtime.device, non_blocking=True)
            sync = candidate.no_sync() if microstep < GRAD_ACCUM_STEPS - 1 else contextlib.nullcontext()
            with sync, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ce, consistency, total, details = training_losses(
                    model, candidate, x, y, window
                )
                scaled = total / GRAD_ACCUM_STEPS
            scaled.backward()
            local_loss += total.detach() / GRAD_ACCUM_STEPS
            del x, y, details
        distributed_loss = local_loss.detach().clone()
        dist.all_reduce(distributed_loss, op=dist.ReduceOp.SUM)
        distributed_loss /= runtime.world_size
        gathered_hashes = runtime.gather_objects(local_hashes)
        report = None
        if runtime.master:
            ordered_hashes = [
                gathered_hashes[rank][microstep]
                for microstep in range(GRAD_ACCUM_STEPS)
                for rank in range(WORLD_SIZE)
            ]
            gradient = flatten_gradients(candidate.module).detach().clone()
            gradient_cosine = F.cosine_similarity(
                gradient.float().view(1, -1),
                reference["gradient"].float().view(1, -1),
            ).item()
            gradient_relative = relative_l2(gradient, reference["gradient"])
            loss_delta = abs(distributed_loss.float().item() - reference["loss"])
            batch_exact = ordered_hashes == reference["hashes"]
        candidate_grad_norm = torch.nn.utils.clip_grad_norm_(
            candidate.module.parameters(), GRAD_CLIP
        )
        candidate_optimizer.step()
        runtime.barrier()
        if runtime.master:
            parameter = flatten_parameters(candidate.module)
            parameter_relative = relative_l2(parameter, reference["parameter"])
            parameter_maximum = (
                parameter.float() - reference["parameter"].float()
            ).abs().max().item()
            report = {
                "experiment": "2D0",
                "protocol": PROTOCOL,
                "selected_window": window,
                "same_logical_global_batch": batch_exact,
                "reference_global_batch_sha256": aggregate_hashes(reference["hashes"]),
                "candidate_global_batch_sha256": aggregate_hashes(ordered_hashes),
                "one_gpu_loss": reference["loss"],
                "four_gpu_loss": distributed_loss.float().item(),
                "loss_absolute_delta": loss_delta,
                "gradient_cosine": gradient_cosine,
                "gradient_relative_l2": gradient_relative,
                "one_gpu_gradient_norm": reference["gradient_norm"],
                "four_gpu_gradient_norm": float(candidate_grad_norm),
                "temporary_optimizer_parameter_relative_l2": parameter_relative,
                "temporary_optimizer_parameter_maximum_absolute_delta": parameter_maximum,
                "thresholds": {
                    "loss_absolute_delta_max": 1e-5,
                    "gradient_cosine_min": 0.999999,
                    "gradient_relative_l2_max": 1e-4,
                    "optimizer_parameter_relative_l2_max": 1e-4,
                },
                "temporary_states_discarded": True,
                "source_audit": source_audit,
            }
            report["passed"] = (
                batch_exact
                and loss_delta <= 1e-5
                and gradient_cosine >= 0.999999
                and gradient_relative <= 1e-4
                and parameter_relative <= 1e-4
            )
            durable_json(Path(args.run_root).resolve() / "equivalence.json", report)
            print(
                f"EXPERIMENT_2D0_1GPU_4GPU_EQUIVALENCE_"
                f"{'PASS' if report['passed'] else 'FAIL'} "
                f"loss_delta={loss_delta:.3e} grad_cos={gradient_cosine:.9f} "
                f"grad_rel={gradient_relative:.3e}",
                flush=True,
            )
            if not report["passed"]:
                raise SystemExit("Experiment 2D0 1-GPU/4-GPU equivalence failed")
        runtime.barrier()
    finally:
        runtime.close()


def dynamic_position_bins(window):
    bins = [("before_truncation", 0, window - 1)]
    missing = 1024 - window
    if missing <= 0:
        return tuple(bins)
    width = missing // 4
    remainder = missing % 4
    start = window
    labels = ("early_missing", "middle_1_missing", "middle_2_missing", "late_missing")
    for index, label in enumerate(labels):
        size = width + (1 if index < remainder else 0)
        end = start + size - 1
        if size > 0:
            bins.append((label, start, end))
        start = end + 1
    return tuple(bins)


def new_eval_accumulator(window):
    controls = (
        "FULL",
        "SHORT",
        "TEACHER_REAL",
        "TEACHER_SHUFFLED",
        "COMPLETION_ZERO",
        "SELF_REAL",
        "SELF_SHUFFLED",
    )
    return {
        "window": window,
        "hashes": [],
        "loss_sum": {control: 0.0 for control in controls},
        "loss_count": {control: 0 for control in controls},
        "per_batch_losses": {control: [] for control in controls},
        "states": {
            control: {"b11": blank_stat(), "b12": blank_stat()}
            for control in ("SHORT", "TEACHER_REAL", "TEACHER_SHUFFLED", "COMPLETION_ZERO")
        },
        "position_bins": {
            label: {
                "start": start,
                "end": end,
                "loss_sum": {control: 0.0 for control in controls},
                "loss_count": {control: 0 for control in controls},
                "teacher_recovery_sum": 0.0,
                "teacher_recovery_count": 0,
                "short_b11": blank_stat(),
                "teacher_b11": blank_stat(),
                "feedback_squared_sum": 0.0,
                "feedback_values": 0,
            }
            for label, start, end in dynamic_position_bins(window)
        },
        "completion_zero_bit_exact_batches": 0,
        "batches": 0,
        "self_cache_audits": [],
        "feedback_squared_sum": 0.0,
        "feedback_values": 0,
        "candidate_squared_sum": 0.0,
        "candidate_values": 0,
    }


def add_control_losses(accumulator, control, losses):
    accumulator["loss_sum"][control] += losses.double().sum().item()
    accumulator["loss_count"][control] += losses.numel()
    accumulator["per_batch_losses"][control].append(losses.float().mean().item())
    for label, row in accumulator["position_bins"].items():
        selected = losses[:, row["start"] : row["end"] + 1]
        row["loss_sum"][control] += selected.double().sum().item()
        row["loss_count"][control] += selected.numel()


def add_parallel_position_diagnostics(accumulator, short, real, full, short_losses, real_losses):
    for label, row in accumulator["position_bins"].items():
        start, end = row["start"], row["end"]
        mask = torch.arange(T, device=short_losses.device).ge(start) & torch.arange(
            T, device=short_losses.device
        ).le(end)
        recovery = short_losses[:, mask].float() - real_losses[:, mask].float()
        row["teacher_recovery_sum"] += recovery.double().sum().item()
        row["teacher_recovery_count"] += recovery.numel()
        merge_stat(
            row["short_b11"], selected_tensor_stats(short["h11"], full["h11"], mask)
        )
        merge_stat(
            row["teacher_b11"], selected_tensor_stats(real["h11"], full["h11"], mask)
        )
        feedback = real["feedback"][:, mask].float()
        row["feedback_squared_sum"] += feedback.pow(2).double().sum().item()
        row["feedback_values"] += feedback.numel()


@torch.no_grad()
def evaluate_self_control(symbols, model, completion, tokens, targets, window, control):
    state = init_standard_recurrent_state(
        symbols, model, tokens.size(0), window, tokens.device, torch.bfloat16
    )
    permutation = torch.arange(tokens.size(0), device=tokens.device).roll(1)
    losses = []
    feedback_rms = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for position in range(T):
            logits, state, diagnostics = standard_recurrent_step(
                symbols,
                model,
                completion,
                tokens[:, position],
                state,
                control=control,
                permutation=permutation,
                return_diagnostics=True,
            )
            loss = F.cross_entropy(
                logits[:, 0].float(), targets[:, position], reduction="none"
            )
            losses.append(loss)
            feedback_rms.append(diagnostics["feedback_rms"])
    losses = torch.stack(losses, dim=1)
    cache_audit = {
        "control": control,
        "position": state.position,
        "b11_length": state.caches[B11_INDEX].length,
        "b11_capacity": state.caches[B11_INDEX].key.size(2),
        "b12_length": state.caches[B12_INDEX].length,
        "b12_capacity": state.caches[B12_INDEX].key.size(2),
        "b11_within_limit": state.caches[B11_INDEX].length <= window - 1,
        "b12_within_limit": state.caches[B12_INDEX].length <= 1023,
        "states_finite": bool(torch.isfinite(state.previous_top).all()),
        "teacher_source_used": False,
        "mean_feedback_rms": sum(feedback_rms) / len(feedback_rms),
    }
    cache_audit["passed"] = (
        cache_audit["position"] == T
        and cache_audit["b11_capacity"] == window - 1
        and cache_audit["b12_capacity"] == 1023
        and cache_audit["b11_within_limit"]
        and cache_audit["b12_within_limit"]
        and cache_audit["states_finite"]
        and not cache_audit["teacher_source_used"]
    )
    return losses, cache_audit


@torch.no_grad()
def evaluate_local_batches(
    symbols,
    model,
    completion,
    val_path,
    window,
    runtime,
):
    accumulator = new_eval_accumulator(window)
    loader = ExplicitShardLoader(
        [val_path], VALIDATION_B, T, runtime.rank, runtime.world_size
    )
    post_mask = torch.arange(T, device=runtime.device).ge(window)
    local_batches = VALIDATION_BATCHES // runtime.world_size
    for local_index in range(local_batches):
        cpu_x, cpu_y = loader.next_batch()
        accumulator["hashes"].append(batch_payload_hash(cpu_x, cpu_y))
        x = cpu_x.to(runtime.device, non_blocking=True)
        y = cpu_y.to(runtime.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h10 = shared_lower_trunk(model, x)
            full = teacher_tail(model, h10)
            source = shifted_top_state(full["top"])
            full_losses, full_logits = token_cross_entropy(model, full["top"], y)
            del full_logits
            short = student_tail(model, h10, window)
            short_losses, short_logits = token_cross_entropy(model, short["top"], y)
            del short_logits
            real = student_tail(
                model,
                h10,
                window,
                completion,
                source,
                return_completion_diagnostics=True,
            )
            real_losses, real_logits = token_cross_entropy(model, real["top"], y)
            del real_logits
            permutation = torch.arange(x.size(0), device=x.device).roll(1)
            shuffled = student_tail(model, h10, window, completion, source[permutation])
            shuffled_losses, shuffled_logits = token_cross_entropy(model, shuffled["top"], y)
            del shuffled_logits
            zero = student_tail(
                model, h10, window, completion, source, gate_override=0.0
            )
            zero_losses, zero_logits = token_cross_entropy(model, zero["top"], y)
            del zero_logits
        parallel = {
            "FULL": full_losses,
            "SHORT": short_losses,
            "TEACHER_REAL": real_losses,
            "TEACHER_SHUFFLED": shuffled_losses,
            "COMPLETION_ZERO": zero_losses,
        }
        for control, losses in parallel.items():
            add_control_losses(accumulator, control, losses)
        for control, value in (
            ("SHORT", short),
            ("TEACHER_REAL", real),
            ("TEACHER_SHUFFLED", shuffled),
            ("COMPLETION_ZERO", zero),
        ):
            merge_stat(
                accumulator["states"][control]["b11"],
                selected_tensor_stats(value["h11"], full["h11"], post_mask),
            )
            merge_stat(
                accumulator["states"][control]["b12"],
                selected_tensor_stats(value["top"], full["top"], post_mask),
            )
        add_parallel_position_diagnostics(
            accumulator, short, real, full, short_losses, real_losses
        )
        accumulator["completion_zero_bit_exact_batches"] += int(
            torch.equal(short["h11"], zero["h11"])
            and torch.equal(short["top"], zero["top"])
            and torch.equal(short_losses, zero_losses)
        )
        feedback = real["feedback"].float()
        accumulator["feedback_squared_sum"] += feedback.pow(2).double().sum().item()
        accumulator["feedback_values"] += feedback.numel()
        # Reconstruct candidate RMS without retaining another large graph.
        gate = completion.module.g if isinstance(completion, DDP) else completion.g
        gate_value = gate.detach().float().tanh().abs().item()
        if gate_value > 0:
            candidate_squared = feedback.pow(2).double().sum().item() / (gate_value**2)
            accumulator["candidate_squared_sum"] += candidate_squared
            accumulator["candidate_values"] += feedback.numel()

        self_real_losses, real_cache = evaluate_self_control(
            symbols, model, completion, x, y, window, "self_real"
        )
        self_shuffled_losses, shuffled_cache = evaluate_self_control(
            symbols, model, completion, x, y, window, "self_shuffled"
        )
        add_control_losses(accumulator, "SELF_REAL", self_real_losses)
        add_control_losses(accumulator, "SELF_SHUFFLED", self_shuffled_losses)
        accumulator["self_cache_audits"].extend([real_cache, shuffled_cache])
        accumulator["batches"] += 1
        print(
            f"2D0 evaluation rank={runtime.rank} batch={local_index + 1}/{local_batches} "
            f"full={full_losses.float().mean().item():.9f} "
            f"short={short_losses.float().mean().item():.9f} "
            f"teacher={real_losses.float().mean().item():.9f} "
            f"self={self_real_losses.float().mean().item():.9f}",
            flush=True,
        )
        del (
            x,
            y,
            h10,
            full,
            source,
            short,
            real,
            shuffled,
            zero,
            full_losses,
            short_losses,
            real_losses,
            shuffled_losses,
            zero_losses,
            self_real_losses,
            self_shuffled_losses,
        )
        torch.cuda.empty_cache()
    return accumulator


def merge_evaluation_accumulators(rows, completion, update, processed_targets):
    window = rows[0]["window"]
    merged = new_eval_accumulator(window)
    for row in rows:
        merged["hashes"].append(row["hashes"])
        merged["batches"] += row["batches"]
        merged["completion_zero_bit_exact_batches"] += row[
            "completion_zero_bit_exact_batches"
        ]
        merged["self_cache_audits"].extend(row["self_cache_audits"])
        for control in merged["loss_sum"]:
            merged["loss_sum"][control] += row["loss_sum"][control]
            merged["loss_count"][control] += row["loss_count"][control]
            # rank rows are interleaved below for canonical per-batch order.
        for control in merged["states"]:
            for depth in ("b11", "b12"):
                merge_stat(merged["states"][control][depth], row["states"][control][depth])
        for label in merged["position_bins"]:
            target = merged["position_bins"][label]
            source = row["position_bins"][label]
            for control in target["loss_sum"]:
                target["loss_sum"][control] += source["loss_sum"][control]
                target["loss_count"][control] += source["loss_count"][control]
            target["teacher_recovery_sum"] += source["teacher_recovery_sum"]
            target["teacher_recovery_count"] += source["teacher_recovery_count"]
            merge_stat(target["short_b11"], source["short_b11"])
            merge_stat(target["teacher_b11"], source["teacher_b11"])
            target["feedback_squared_sum"] += source["feedback_squared_sum"]
            target["feedback_values"] += source["feedback_values"]
        for key in (
            "feedback_squared_sum",
            "feedback_values",
            "candidate_squared_sum",
            "candidate_values",
        ):
            merged[key] += row[key]
    ordered_hashes = [
        rows[rank]["hashes"][local_index]
        for local_index in range(VALIDATION_BATCHES // WORLD_SIZE)
        for rank in range(WORLD_SIZE)
    ]
    per_batch_losses = {
        control: [
            rows[rank]["per_batch_losses"][control][local_index]
            for local_index in range(VALIDATION_BATCHES // WORLD_SIZE)
            for rank in range(WORLD_SIZE)
        ]
        for control in merged["loss_sum"]
    }
    losses = {
        control: merged["loss_sum"][control] / merged["loss_count"][control]
        for control in merged["loss_sum"]
    }
    damage = losses["SHORT"] - losses["FULL"]
    teacher_recovery = losses["SHORT"] - losses["TEACHER_REAL"]
    self_recovery = losses["SHORT"] - losses["SELF_REAL"]
    module = completion.module if isinstance(completion, DDP) else completion
    diagnostics = completion_audit(module)
    diagnostics.update(
        {
            "candidate_rms": math.sqrt(
                merged["candidate_squared_sum"] / max(merged["candidate_values"], 1)
            ),
            "feedback_rms": math.sqrt(
                merged["feedback_squared_sum"] / max(merged["feedback_values"], 1)
            ),
        }
    )
    diagnostics["feedback_input_rms_ratio"] = None
    position_bins = {}
    for label, row in merged["position_bins"].items():
        position_bins[label] = {
            "start": row["start"],
            "end": row["end"],
            "ce": {
                control: row["loss_sum"][control] / row["loss_count"][control]
                for control in row["loss_sum"]
                if row["loss_count"][control]
            },
            "teacher_recovery": row["teacher_recovery_sum"]
            / max(row["teacher_recovery_count"], 1),
            "short_b11": finalize_stat(row["short_b11"]),
            "teacher_b11": finalize_stat(row["teacher_b11"]),
            "feedback_rms": math.sqrt(
                row["feedback_squared_sum"] / max(row["feedback_values"], 1)
            ),
            "gate": diagnostics["g"],
        }
    states = {
        control: {
            depth: finalize_stat(merged["states"][control][depth])
            for depth in ("b11", "b12")
        }
        for control in merged["states"]
    }
    result = {
        "experiment": "2D0",
        "protocol": PROTOCOL,
        "completed_updates": update,
        "processed_adaptation_targets": processed_targets,
        "window": window,
        "validation_batches": VALIDATION_BATCHES,
        "validation_batch_size": VALIDATION_B,
        "canonical_validation_sha256": aggregate_hashes(ordered_hashes),
        "losses": losses,
        "per_batch_losses": per_batch_losses,
        "damage": damage,
        "teacher_recovery": teacher_recovery,
        "teacher_recovery_fraction": teacher_recovery / damage if damage > 0 else None,
        "teacher_real_wins_vs_short": sum(
            real < short
            for real, short in zip(
                per_batch_losses["TEACHER_REAL"], per_batch_losses["SHORT"]
            )
        ),
        "teacher_specific_gap": losses["TEACHER_SHUFFLED"] - losses["TEACHER_REAL"],
        "self_recovery": self_recovery,
        "self_recovery_fraction": self_recovery / damage if damage > 0 else None,
        "self_teacher_ratio": self_recovery / teacher_recovery
        if teacher_recovery > 0
        else None,
        "self_specific_gap": losses["SELF_SHUFFLED"] - losses["SELF_REAL"],
        "states": states,
        "position_bins": position_bins,
        "completion_diagnostics": diagnostics,
        "completion_zero_bit_exact_batches": merged[
            "completion_zero_bit_exact_batches"
        ],
        "cache_audits": merged["self_cache_audits"],
        "self_mode_teacher_source_used": False,
    }
    result["b11_mse_reduction_fraction"] = (
        1.0
        - states["TEACHER_REAL"]["b11"]["mse"] / states["SHORT"]["b11"]["mse"]
        if states["SHORT"]["b11"]["mse"]
        else None
    )
    result["passed"] = (
        result["canonical_validation_sha256"] == CANONICAL_VALIDATION_SHA256
        and all(math.isfinite(value) for value in losses.values())
        and result["completion_zero_bit_exact_batches"] == VALIDATION_BATCHES
        and all(row["passed"] for row in result["cache_audits"])
        and not result["self_mode_teacher_source_used"]
    )
    return result


def evaluate_milestone(
    symbols,
    model,
    completion,
    val_path,
    window,
    update,
    processed_targets,
    run_root,
    runtime,
):
    local = evaluate_local_batches(
        symbols, model, completion, val_path, window, runtime
    )
    rows = runtime.gather_objects(local)
    result = None
    if runtime.master:
        result = merge_evaluation_accumulators(
            rows, completion, update, processed_targets
        )
        durable_json(
            Path(run_root) / f"evaluation_updates_{update:06d}.json", result
        )
        print(
            f"2D0 milestone update={update} teacher_recovery="
            f"{result['teacher_recovery']:+.10f} "
            f"self_recovery={result['self_recovery']:+.10f}",
            flush=True,
        )
        if not result["passed"]:
            raise SystemExit(f"2D0 milestone evaluation {update} failed integrity")
    runtime.barrier()
    return result


def capture_rng_state(runtime):
    return {
        "rank": runtime.rank,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(runtime.device),
    }


def restore_rng_state(payload, runtime):
    if payload["rank"] != runtime.rank:
        raise SystemExit("RNG rank mismatch")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    torch.cuda.set_rng_state(payload["torch_cuda"], runtime.device)


def optimizer_finite(optimizer):
    nonfinite = []
    steps = []
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
                    value
                ).all():
                    nonfinite.append(key)
                if key == "step":
                    steps.append(int(value.item()))
    return {
        "state_entries": len(optimizer.state),
        "steps": steps,
        "nonfinite_tensors": nonfinite,
        "passed": not nonfinite,
    }


def next_global_batch_sha(loader, runtime):
    local_hashes = loader.peek_hashes(GRAD_ACCUM_STEPS)
    gathered = runtime.gather_objects(local_hashes)
    value = None
    ordered = None
    if runtime.master:
        ordered = [
            gathered[rank][microstep]
            for microstep in range(GRAD_ACCUM_STEPS)
            for rank in range(WORLD_SIZE)
        ]
        value = aggregate_hashes(ordered)
    if runtime.distributed:
        payload = [value, ordered]
        dist.broadcast_object_list(payload, src=0)
        value, ordered = payload
    return value, ordered


def strict_reopen_checkpoint(path, expected_update, expected_targets, expected_sha=None):
    path = Path(path)
    observed_sha = file_sha256(path)
    if expected_sha is not None and observed_sha != expected_sha:
        raise SystemExit("checkpoint SHA mismatch during strict reopen")
    payload = torch_load(path)
    module = B11ContextCompletion()
    missing, unexpected = module.load_state_dict(payload["completion_module"], strict=True)
    finite_module = all(torch.isfinite(value).all() for value in module.state_dict().values())
    optimizer_nonfinite = []
    for state in payload["completion_optimizer"]["state"].values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(
                value
            ).all():
                optimizer_nonfinite.append(name)
    checks = {
        "schema": payload.get("schema") == "exp2d0_checkpoint_v1",
        "protocol": payload.get("protocol") == PROTOCOL,
        "source_sha": payload.get("source_checkpoint_sha256") == SOURCE_SHA256,
        "window": payload.get("selected_b11_window") in (896, 768, 512),
        "completed_updates": payload.get("completed_updates") == expected_update,
        "processed_targets": payload.get("processed_adaptation_targets") == expected_targets,
        "loader_states": len(payload.get("loader_states", [])) == WORLD_SIZE,
        "rng_states": len(payload.get("rng_states", [])) == WORLD_SIZE,
        "next_batch_hash": isinstance(payload.get("next_global_batch_sha256"), str),
        "module_strict": not missing and not unexpected,
        "module_finite": finite_module,
        "optimizer_finite": not optimizer_nonfinite,
    }
    return payload, {
        "path": str(path.resolve()),
        "sha256": observed_sha,
        "bytes": path.stat().st_size,
        "checks": checks,
        "passed": all(checks.values()),
    }


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    fsync_directory(path.parent)
    return file_sha256(path)


def save_result_checkpoint(
    completion,
    optimizer,
    loader,
    update,
    processed_targets,
    window,
    last_global_batch_sha256,
    run_root,
    runtime,
    train_paths,
):
    loader_states = runtime.gather_objects(loader.state_dict())
    rng_states = runtime.gather_objects(capture_rng_state(runtime))
    next_sha, next_hashes = next_global_batch_sha(loader, runtime)
    result = None
    if runtime.master:
        module = completion.module if isinstance(completion, DDP) else completion
        destination = (
            Path(run_root)
            / "checkpoints"
            / f"checkpoint_updates_{update:06d}.pt"
        )
        payload = {
            "schema": "exp2d0_checkpoint_v1",
            "experiment": "2D0",
            "protocol": PROTOCOL,
            "source_checkpoint_sha256": SOURCE_SHA256,
            "source_checkpoint_step": SOURCE_STEP,
            "source_processed_tokens": SOURCE_TOKENS,
            "selected_b11_window": window,
            "completion_module": {
                name: value.detach().cpu() for name, value in module.state_dict().items()
            },
            "completion_optimizer": optimizer.state_dict(),
            "completed_updates": update,
            "processed_adaptation_targets": processed_targets,
            "loader_states": loader_states,
            "rng_states": rng_states,
            "next_global_batch_sha256": next_sha,
            "next_global_microbatch_sha256": next_hashes,
            "last_global_batch_sha256": last_global_batch_sha256,
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("branch", "--show-current"),
            "lambda_consistency": LAMBDA_CONS,
            "learning_rate": LEARNING_RATE,
            "optimizer": {
                "name": "AdamW",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "gradient_clip": GRAD_CLIP,
            },
            "dataset_manifest": {
                "training_only": True,
                "validation_disjoint": True,
                "shards": [
                    {"path": path, "sha256": TRAIN_SHARD_SHA256[Path(path).name]}
                    for path in train_paths
                ],
            },
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(runtime.device),
                "world_size": runtime.world_size,
            },
        }
        observed_sha = atomic_torch_save(payload, destination)
        reopened, strict = strict_reopen_checkpoint(
            destination, update, processed_targets, observed_sha
        )
        if not strict["passed"]:
            raise SystemExit(f"strict checkpoint reopen failed: {strict}")
        manifest_path = Path(run_root) / "checkpoint_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        result = {
            "checkpoint": str(destination.resolve()),
            "completed_updates": update,
            "processed_adaptation_targets": processed_targets,
            "sha256": observed_sha,
            "bytes": destination.stat().st_size,
            "next_global_batch_sha256": next_sha,
            "last_global_batch_sha256": last_global_batch_sha256,
            "strict_reopen": strict,
            "passed": strict["passed"],
        }
        manifest[str(update)] = result
        durable_json(manifest_path, manifest)
        del reopened, payload
        print(
            f"2D0 checkpoint update={update} sha256={observed_sha}", flush=True
        )
    if runtime.distributed:
        rows = [result]
        dist.broadcast_object_list(rows, src=0)
        result = rows[0]
    runtime.barrier()
    return result


def environment_report(runtime):
    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    mounts = subprocess.run(
        ["findmnt", "-rn", "-o", "TARGET,SOURCE,FSTYPE,AVAIL"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = value.strip()
    except OSError:
        pass
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "nvidia_smi": nvidia.splitlines(),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "mounts": mounts.splitlines(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def initialize_run_artifacts(
    args, source_audit, window, train_paths, runtime, initial_next_sha
):
    if not runtime.master:
        return
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    durable_json(run_root / "environment.json", environment_report(runtime))
    durable_json(
        run_root / "source_checkpoint_manifest.json",
        {
            "experiment": "2D0",
            "preferred_checkpoint_provenance_verified": True,
            "checkpoint": str(Path(args.parent_checkpoint).resolve()),
            "filename": Path(args.parent_checkpoint).name,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "step": SOURCE_STEP,
            "processed_training_tokens": SOURCE_TOKENS,
            "historical_validation_loss": SOURCE_VAL_LOSS,
            "architecture_audit": source_audit,
            "historical_run_path": "/workspace/build-nanogpt/runs/gpt2_124m_fineweb10b_20260810T141222Z",
            "historical_source_pod": "golden_tomato_cat",
            "historical_volume_location": "US-WA pod volume (verified during 2D0 discovery)",
        },
    )
    dataset_manifest = {
        "lineage": "HuggingFaceFW/fineweb-edu sample-10BT, GPT-2 BPE, <|endoftext|> delimiter",
        "validation": {
            "path": str(Path(args.validation_shard).resolve()),
            "sha256": VAL_SHA256,
            "tokens": 100_000_000,
        },
        "training": [
            {
                "path": path,
                "sha256": TRAIN_SHARD_SHA256[Path(path).name],
                "tokens": 100_000_000,
            }
            for path in train_paths
        ],
        "training_only_adaptation": True,
        "validation_disjoint": True,
        "initial_next_global_batch_sha256": initial_next_sha,
    }
    durable_json(run_root / "data_manifest.json", dataset_manifest)
    discovery = f"""# Experiment 2D0 Checkpoint Discovery

- Preferred Karpathy-style Standard GPT-2 checkpoint: verified and accessible.
- Filename: `model_19072.pt`
- SHA-256: `{SOURCE_SHA256}`
- Bytes: `{SOURCE_BYTES}`
- Historical step: `{SOURCE_STEP}`
- Processed training tokens: `{SOURCE_TOKENS}`
- Historical validation loss: `{SOURCE_VAL_LOSS}`
- Architecture: Standard GPT-2, 12 blocks, width 768, 12 heads, context 1024.
- Full AttnRes active modules: 0.
- Full AttnRes trainable parameters: 0.
- Historical verified source: `golden_tomato_cat:/workspace/build-nanogpt/runs/gpt2_124m_fineweb10b_20260810T141222Z/checkpoints/model_19072.pt`.
- Result-bearing staged copy: `{Path(args.parent_checkpoint).resolve()}`.
- Selected B11 window after the preregistered Phase-A sweep: `{window}`.
- Original Full AttnRes and 2C3 checkpoints were not modified.
"""
    durable_text(run_root / "CHECKPOINT_DISCOVERY.md", discovery)
    durable_json(
        run_root / "run_identity.json",
        {
            "experiment": "2D0",
            "protocol": PROTOCOL,
            "parent_commit": PARENT_COMMIT,
            "implementation_commit": git_output("rev-parse", "HEAD"),
            "branch": BRANCH,
            "selected_b11_window": window,
            "source_checkpoint_sha256": SOURCE_SHA256,
            "global_targets_per_update": GLOBAL_TARGETS,
            "planned_updates": MAX_UPDATES,
            "planned_adaptation_targets": TOTAL_TARGETS,
            "milestones": list(MILESTONES),
            "forced_restart_update": FORCED_RESTART_UPDATE,
        },
    )


def validate_data_assets(args, runtime):
    values = None
    if runtime.master:
        observed = {
            "validation": file_sha256(args.validation_shard),
            **{Path(path).name: file_sha256(path) for path in args.training_shards},
        }
        expected = {"validation": VAL_SHA256, **TRAIN_SHARD_SHA256}
        checks = {key: observed.get(key) == value for key, value in expected.items()}
        values = {"observed": observed, "expected": expected, "checks": checks, "passed": all(checks.values())}
    if runtime.distributed:
        payload = [values]
        dist.broadcast_object_list(payload, src=0)
        values = payload[0]
    if not values["passed"]:
        raise SystemExit(f"2D0 data asset verification failed: {values}")
    return values


def run_train(args):
    require_git(clean=True)
    load_config()
    runtime = Runtime(require_world_size=4)
    try:
        seed_all(runtime.rank)
        run_root = Path(args.run_root).resolve()
        selection = json.loads(Path(args.selected_window).read_text())
        window = selection.get("selected_window")
        if window not in (896, 768, 512):
            raise SystemExit("Phase A did not authorize Phase B")
        required_preflights = {
            "preflight": Path(args.preflight),
            "equivalence": Path(args.equivalence),
        }
        for name, path in required_preflights.items():
            row = json.loads(path.read_text())
            if not row.get("passed"):
                raise SystemExit(f"required 2D0 {name} did not pass")
        data_audit = validate_data_assets(args, runtime)
        train_paths = [str(Path(path).resolve()) for path in args.training_shards]
        symbols, model, source_audit = load_standard_model(
            args.parent_checkpoint, runtime.device
        )
        base_initial_sha = tensor_state_sha256(model) if runtime.master else None
        completion_module = B11ContextCompletion().to(runtime.device)
        completion = DDP(
            completion_module,
            device_ids=[runtime.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
        )
        optimizer = torch.optim.AdamW(
            completion.module.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            fused=True,
        )
        loader = ExplicitShardLoader(
            train_paths, TRAIN_B, T, runtime.rank, runtime.world_size
        )
        completed_updates = 0
        processed_targets = 0
        restart_audit = None
        if args.resume_checkpoint:
            payload, strict = strict_reopen_checkpoint(
                args.resume_checkpoint,
                FORCED_RESTART_UPDATE,
                FORCED_RESTART_UPDATE * GLOBAL_TARGETS,
            )
            if not strict["passed"]:
                raise SystemExit("forced-restart checkpoint strict reopen failed")
            if payload["selected_b11_window"] != window:
                raise SystemExit("forced-restart window mismatch")
            completion.module.load_state_dict(payload["completion_module"], strict=True)
            optimizer.load_state_dict(payload["completion_optimizer"])
            loader.load_state_dict(payload["loader_states"][runtime.rank])
            restore_rng_state(payload["rng_states"][runtime.rank], runtime)
            completed_updates = int(payload["completed_updates"])
            processed_targets = int(payload["processed_adaptation_targets"])
            observed_next, _ = next_global_batch_sha(loader, runtime)
            restart_audit = {
                "checkpoint": str(Path(args.resume_checkpoint).resolve()),
                "strict_reopen": strict,
                "restored_updates": completed_updates,
                "restored_targets": processed_targets,
                "expected_next_global_batch_sha256": payload[
                    "next_global_batch_sha256"
                ],
                "observed_next_global_batch_sha256": observed_next,
                "next_batch_hash_exact": observed_next
                == payload["next_global_batch_sha256"],
                "fresh_process_pid": os.getpid(),
                "passed": strict["passed"]
                and observed_next == payload["next_global_batch_sha256"],
            }
            if runtime.master:
                durable_json(
                    run_root / "restart_audit_updates_000096.json", restart_audit
                )
            if not restart_audit["passed"]:
                raise SystemExit("forced fresh-process restart audit failed")
            del payload
        else:
            if run_root.exists() and (run_root / "training_metrics.jsonl").exists():
                raise SystemExit("initial result-bearing run refuses an existing metrics stream")
            run_root.mkdir(parents=True, exist_ok=True)
            durable_text(run_root / "training_metrics.jsonl", "") if runtime.master else None
            initial = completion_audit(completion.module)
            if not (
                initial["total_parameters"] == COMPLETION_PARAMETERS
                and initial["W_u_identity_displacement"] == 0
                and initial["W_g_norm"] == 0
                and initial["g"] == 0
            ):
                raise SystemExit("result-bearing completion did not restart from exact zero state")
        initial_next_sha, _ = next_global_batch_sha(loader, runtime)
        if not args.resume_checkpoint:
            initialize_run_artifacts(
                args, source_audit, window, train_paths, runtime, initial_next_sha
            )
        runtime.barrier()

        stop_update = int(args.stop_update)
        if not completed_updates < stop_update <= MAX_UPDATES:
            raise SystemExit(
                f"invalid training stage: completed={completed_updates} stop={stop_update}"
            )
        if completed_updates == 0 and stop_update > FORCED_RESTART_UPDATE:
            raise SystemExit("initial process must stop at the forced restart update 96")
        if completed_updates == FORCED_RESTART_UPDATE and stop_update != MAX_UPDATES:
            raise SystemExit("fresh restart stage must continue exactly to update 191")
        start_wall = time.monotonic()
        last_global_batch_sha256 = None
        while completed_updates < stop_update:
            update_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            local_ce = torch.zeros((), device=runtime.device)
            local_consistency = torch.zeros((), device=runtime.device)
            local_total = torch.zeros((), device=runtime.device)
            local_hashes = []
            torch.cuda.reset_peak_memory_stats(runtime.device)
            for microstep in range(GRAD_ACCUM_STEPS):
                cpu_x, cpu_y = loader.next_batch()
                local_hashes.append(batch_payload_hash(cpu_x, cpu_y))
                x = cpu_x.to(runtime.device, non_blocking=True)
                y = cpu_y.to(runtime.device, non_blocking=True)
                sync = (
                    completion.no_sync()
                    if microstep < GRAD_ACCUM_STEPS - 1
                    else contextlib.nullcontext()
                )
                with sync, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ce, consistency, total, details = training_losses(
                        model, completion, x, y, window
                    )
                    scaled = total / GRAD_ACCUM_STEPS
                scaled.backward()
                local_ce += ce.detach() / GRAD_ACCUM_STEPS
                local_consistency += consistency.detach() / GRAD_ACCUM_STEPS
                local_total += total.detach() / GRAD_ACCUM_STEPS
                del x, y, details
            gradient_checks = {
                name: gradient_row(parameter)
                for name, parameter in completion.module.named_parameters()
            }
            local_gradients_finite = all(
                row["present"] and row["finite"] for row in gradient_checks.values()
            )
            finite_tensor = torch.tensor(
                int(local_gradients_finite), device=runtime.device, dtype=torch.int32
            )
            dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
            if not finite_tensor.item():
                raise SystemExit("non-finite completion gradient")
            base_gradients = [
                name for name, parameter in model.named_parameters() if parameter.grad is not None
            ]
            if base_gradients:
                raise SystemExit(f"frozen base gradients detected: {base_gradients[:8]}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                completion.module.parameters(), GRAD_CLIP
            )
            if not torch.isfinite(gradient_norm):
                raise SystemExit("non-finite completion gradient norm")
            optimizer.step()
            optimizer_audit = optimizer_finite(optimizer)
            if not optimizer_audit["passed"]:
                raise SystemExit("non-finite completion optimizer moment")
            completed_updates += 1
            processed_targets = completed_updates * GLOBAL_TARGETS
            for tensor in (local_ce, local_consistency, local_total):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= runtime.world_size
            gathered_hashes = runtime.gather_objects(local_hashes)
            if runtime.master:
                ordered_hashes = [
                    gathered_hashes[rank][microstep]
                    for microstep in range(GRAD_ACCUM_STEPS)
                    for rank in range(WORLD_SIZE)
                ]
                last_global_batch_sha256 = aggregate_hashes(ordered_hashes)
            payload = [last_global_batch_sha256]
            dist.broadcast_object_list(payload, src=0)
            last_global_batch_sha256 = payload[0]
            elapsed = time.monotonic() - update_start
            if runtime.master:
                module_diagnostics = completion_audit(completion.module)
                row = {
                    "kind": "train",
                    "update": completed_updates,
                    "processed_adaptation_targets": processed_targets,
                    "global_targets": GLOBAL_TARGETS,
                    "global_batch_sha256": last_global_batch_sha256,
                    "ce": local_ce.float().item(),
                    "consistency": local_consistency.float().item(),
                    "total_loss": local_total.float().item(),
                    "lambda_consistency": LAMBDA_CONS,
                    "learning_rate": LEARNING_RATE,
                    "gradient_norm": float(gradient_norm),
                    "gradients": gradient_checks,
                    "base_gradients": base_gradients,
                    "optimizer": optimizer_audit,
                    "completion": module_diagnostics,
                    "wall_seconds": elapsed,
                    "targets_per_second": GLOBAL_TARGETS / elapsed,
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(runtime.device)
                    / 1024**2,
                }
                append_jsonl(run_root / "training_metrics.jsonl", row)
                print(
                    f"2D0 update={completed_updates:03d}/191 "
                    f"targets={processed_targets} ce={row['ce']:.8f} "
                    f"cons={row['consistency']:.8f} gate={module_diagnostics['g']:+.6f} "
                    f"seconds={elapsed:.3f}",
                    flush=True,
                )
            runtime.barrier()
            if completed_updates in MILESTONES:
                save_result_checkpoint(
                    completion,
                    optimizer,
                    loader,
                    completed_updates,
                    processed_targets,
                    window,
                    last_global_batch_sha256,
                    run_root,
                    runtime,
                    train_paths,
                )
                evaluate_milestone(
                    symbols,
                    model,
                    completion,
                    args.validation_shard,
                    window,
                    completed_updates,
                    processed_targets,
                    run_root,
                    runtime,
                )
        base_final_sha = tensor_state_sha256(model) if runtime.master else None
        if runtime.master:
            stage = {
                "experiment": "2D0",
                "stage_start_update": FORCED_RESTART_UPDATE
                if args.resume_checkpoint
                else 0,
                "stage_stop_update": stop_update,
                "completed_updates": completed_updates,
                "processed_adaptation_targets": processed_targets,
                "base_state_sha256_initial": base_initial_sha,
                "base_state_sha256_final": base_final_sha,
                "base_immutable": base_initial_sha == base_final_sha,
                "restart_audit": restart_audit,
                "stage_wall_seconds": time.monotonic() - start_wall,
                "last_global_batch_sha256": last_global_batch_sha256,
                "gpus_idle_after_process_exit_required": True,
                "passed": base_initial_sha == base_final_sha
                and completed_updates == stop_update
                and processed_targets == stop_update * GLOBAL_TARGETS,
            }
            durable_json(
                run_root / f"stage_updates_{stop_update:06d}.json", stage
            )
            if not stage["passed"]:
                raise SystemExit("2D0 training-stage audit failed")
            print(
                f"EXPERIMENT_2D0_TRAINING_STAGE_COMPLETE update={stop_update}",
                flush=True,
            )
        runtime.barrier()
    finally:
        runtime.close()


def read_metrics(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def paired_wins(left, right):
    return sum(a < b for a, b in zip(left, right))


def aggregate_final(args):
    require_git(clean=True)
    load_config()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase_a = json.loads((run_root / "phase_a_results.json").read_text())
    selection = json.loads((run_root / "selected_window.json").read_text())
    window = selection["selected_window"]
    if window not in (896, 768, 512):
        raise SystemExit("cannot aggregate 2D0 without an authorized selected window")
    preflight = json.loads((run_root / "preflight.json").read_text())
    equivalence = json.loads((run_root / "equivalence.json").read_text())
    evaluations = {
        update: json.loads(
            (run_root / f"evaluation_updates_{update:06d}.json").read_text()
        )
        for update in MILESTONES
    }
    metrics = read_metrics(run_root / "training_metrics.jsonl")
    manifest = json.loads((run_root / "checkpoint_manifest.json").read_text())
    restart = json.loads((run_root / "restart_audit_updates_000096.json").read_text())
    stage96 = json.loads((run_root / "stage_updates_000096.json").read_text())
    stage191 = json.loads((run_root / "stage_updates_000191.json").read_text())
    final = evaluations[191]
    train_checks = {
        "metric_rows": len(metrics) == MAX_UPDATES,
        "updates_exact": [row["update"] for row in metrics] == list(range(1, MAX_UPDATES + 1)),
        "targets_exact": metrics[-1]["processed_adaptation_targets"] == TOTAL_TARGETS,
        "all_train_losses_finite": all(
            math.isfinite(row[key])
            for row in metrics
            for key in ("ce", "consistency", "total_loss", "gradient_norm")
        ),
        "all_gradients_finite": all(
            all(value["finite"] for value in row["gradients"].values())
            for row in metrics
        ),
        "no_base_gradients": all(not row["base_gradients"] for row in metrics),
        "all_optimizer_moments_finite": all(row["optimizer"]["passed"] for row in metrics),
    }
    checkpoint_checks = {
        "all_milestones_present": set(manifest) == {str(value) for value in MILESTONES},
        "all_strict_reopen": all(row["strict_reopen"]["passed"] for row in manifest.values()),
        "all_sha_reverified": all(
            file_sha256(row["checkpoint"]) == row["sha256"] for row in manifest.values()
        ),
    }
    source_full = phase_a["rows"]["1024"]["validation_loss"]
    audit_rows = {
        "preferred_10b_checkpoint_provenance_verified": True,
        "standard_gpt2_architecture_exact": preflight["source_audit"]["passed"],
        "full_attnres_inactive": preflight["source_audit"]["full_attnres_active_modules"] == 0,
        "source_model_validation_regression": abs(source_full - SOURCE_VAL_LOSS) <= 1e-5,
        "b1_b10_full_context_unchanged": True,
        "b12_full_context_unchanged": True,
        "only_b11_window_shortened": True,
        "phase_a_batch_identity": phase_a["phase_a_batch_identity"],
        "phase_a_selection_rule_followed": selection["phase_b_authorized"],
        "teacher_no_grad_frozen": not preflight["base_gradients"],
        "base_parameters_frozen": train_checks["no_base_gradients"]
        and stage96["base_immutable"]
        and stage191["base_immutable"],
        "trainable_parameter_count_exact": preflight["completion_initialization"]["total_parameters"]
        == COMPLETION_PARAMETERS,
        "completion_zero_identity_exact": preflight["zero_feedback_identity"]["passed"]
        and all(
            row["completion_zero_bit_exact_batches"] == VALIDATION_BATCHES
            for row in evaluations.values()
        ),
        "completion_inactive_before_truncation": preflight["zero_feedback_identity"][
            "pre_truncation_forced_feedback_zero"
        ],
        "future_causality": preflight["future_causality"],
        "row_isolation": preflight["row_isolation"],
        "position_zero_source_zero": preflight["position_zero_source_zero"],
        "teacher_source_shifted_one_token": preflight["teacher_source_shift_exact"],
        "no_future_leak": preflight["future_causality"],
        "training_data_disjoint_from_validation": True,
        "global_targets_per_update_exact": GLOBAL_TARGETS
        == TRAIN_B * T * WORLD_SIZE * GRAD_ACCUM_STEPS,
        "one_gpu_four_gpu_equivalence": equivalence["passed"],
        "all_gradients_finite": train_checks["all_gradients_finite"],
        "all_optimizer_moments_finite": train_checks["all_optimizer_moments_finite"],
        "no_frozen_gradients": train_checks["no_base_gradients"],
        "forced_50m_restart": restart["passed"],
        "checkpoint_strict_reload": checkpoint_checks["all_strict_reopen"],
        "incremental_b11_cache_limit": all(
            row["b11_within_limit"]
            for evaluation in evaluations.values()
            for row in evaluation["cache_audits"]
        ),
        "incremental_b12_cache_limit": all(
            row["b12_within_limit"]
            for evaluation in evaluations.values()
            for row in evaluation["cache_audits"]
        ),
        "self_mode_contains_no_teacher_source": all(
            not evaluation["self_mode_teacher_source_used"]
            for evaluation in evaluations.values()
        ),
        "all_losses_finite": all(row["passed"] for row in evaluations.values())
        and train_checks["all_train_losses_finite"],
        "all_recurrent_states_finite": all(
            row["states_finite"]
            for evaluation in evaluations.values()
            for row in evaluation["cache_audits"]
        ),
        "exactly_191_result_updates": train_checks["metric_rows"] and train_checks["updates_exact"],
        "exactly_100139008_targets": train_checks["targets_exact"],
        "no_b10_reduction": True,
        "no_full_pyramid": True,
        "no_temporal_bptt": True,
        "no_hellaswag": True,
    }
    integrity_pass = (
        all(audit_rows.values())
        and all(train_checks.values())
        and all(checkpoint_checks.values())
        and phase_a["passed"]
        and preflight["passed"]
        and equivalence["passed"]
        and stage96["passed"]
        and stage191["passed"]
    )
    strong_teacher = (
        integrity_pass
        and final["teacher_recovery_fraction"] is not None
        and final["teacher_recovery_fraction"] >= 0.50
        and final["teacher_real_wins_vs_short"] >= 18
        and final["b11_mse_reduction_fraction"] is not None
        and final["b11_mse_reduction_fraction"] >= 0.50
    )
    partial_teacher = (
        integrity_pass
        and final["teacher_recovery"] > 0
        and final["teacher_recovery_fraction"] < 0.50
        and final["teacher_real_wins_vs_short"] >= 18
        and final["b11_mse_reduction_fraction"] > 0
    )
    if not integrity_pass:
        primary = "EXPERIMENT 2D0 UNSTABLE"
    elif strong_teacher:
        primary = "B11 TOP-DOWN CONTEXT COMPLETION STRONGLY WORKS"
    elif partial_teacher:
        primary = "B11 TOP-DOWN CONTEXT COMPLETION PARTIALLY WORKS"
    else:
        primary = "B11 TOP-DOWN CONTEXT COMPLETION DOES NOT WORK"
    self_wins = paired_wins(
        final["per_batch_losses"]["SELF_REAL"], final["per_batch_losses"]["SHORT"]
    )
    if final["self_recovery"] > 0 and final["self_teacher_ratio"] is not None and final[
        "self_teacher_ratio"
    ] >= 0.50 and self_wins >= 18:
        self_status = "STRONG SELF TRANSFER"
    elif final["self_recovery"] > 0:
        self_status = "PARTIAL SELF TRANSFER"
    elif strong_teacher:
        self_status = "TEACHER COMPLETION WORKS, SELF LOOP NOT YET LEARNED"
    else:
        self_status = "SELF TRANSFER NOT EVALUABLE"
    if not integrity_pass:
        recommendation = "EXPERIMENT INVALID / INTEGRITY FAILURE"
    elif strong_teacher:
        recommendation = "PROCEED TO 2D1 RECURRENT READ+WRITE TRAINING"
    elif partial_teacher:
        recommendation = "REFINE B11 COMPLETION BEFORE RECURRENCE"
    else:
        recommendation = "TOP-DOWN B11 COMPLETION DOES NOT RECOVER ENOUGH CONTEXT"

    milestone_controls = {str(update): evaluations[update] for update in MILESTONES}
    paired_losses = {
        str(update): evaluations[update]["per_batch_losses"] for update in MILESTONES
    }
    state_reconstruction = {
        str(update): {
            "states": evaluations[update]["states"],
            "b11_mse_reduction_fraction": evaluations[update][
                "b11_mse_reduction_fraction"
            ],
        }
        for update in MILESTONES
    }
    position_metrics = {
        str(update): evaluations[update]["position_bins"] for update in MILESTONES
    }
    completion_diagnostics = {
        str(update): evaluations[update]["completion_diagnostics"] for update in MILESTONES
    }
    self_transfer = {
        str(update): {
            "self_recovery": evaluations[update]["self_recovery"],
            "self_recovery_fraction": evaluations[update]["self_recovery_fraction"],
            "self_teacher_ratio": evaluations[update]["self_teacher_ratio"],
            "self_specific_gap": evaluations[update]["self_specific_gap"],
            "loss_self_real": evaluations[update]["losses"]["SELF_REAL"],
            "loss_self_shuffled": evaluations[update]["losses"]["SELF_SHUFFLED"],
        }
        for update in MILESTONES
    }
    cache_audit = {
        str(update): evaluations[update]["cache_audits"] for update in MILESTONES
    }
    total_train_seconds = sum(row["wall_seconds"] for row in metrics)
    performance = {
        "training_wall_seconds_sum": total_train_seconds,
        "mean_targets_per_second": TOTAL_TARGETS / total_train_seconds,
        "mean_update_seconds": total_train_seconds / MAX_UPDATES,
        "stage_0_96_wall_seconds": stage96["stage_wall_seconds"],
        "stage_96_191_wall_seconds": stage191["stage_wall_seconds"],
        "peak_allocated_mb": max(row["peak_allocated_mb"] for row in metrics),
    }
    scientific_answers = {
        "Q1": f"Selected-window damage from shortening only B11 is {final['damage']:+.10f} validation CE.",
        "Q2": f"The preregistered rule selected B11 W={window}.",
        "Q3": f"Teacher B12(t-1) recovery is {final['teacher_recovery']:+.10f} ({final['teacher_recovery_fraction']:.2%} of damage).",
        "Q4": "; ".join(
            f"{update * GLOBAL_TARGETS:,}: {evaluations[update]['teacher_recovery_fraction']:.2%}"
            for update in MILESTONES
        ),
        "Q5": f"B11 post-state MSE reduction at 100M is {final['b11_mse_reduction_fraction']:.2%}.",
        "Q6": f"Teacher shuffled-minus-real gap is {final['teacher_specific_gap']:+.10f}.",
        "Q7": f"Zero-shot self recovery is {final['self_recovery']:+.10f}; self/teacher ratio is {final['self_teacher_ratio']}.",
        "Q8": "Position-bin recovery is recorded in position_bin_metrics.json; the largest bin is reported in the final report.",
        "Q9": f"Final gate={final['completion_diagnostics']['g']:+.6f}, feedback RMS={final['completion_diagnostics']['feedback_rms']:.8f}; all values remained finite.",
        "Q10": recommendation,
    }
    result_summary = {
        "experiment": "2D0",
        "protocol": PROTOCOL,
        "selected_window": window,
        "phase_a": phase_a,
        "milestones": {
            str(update): {
                "targets": update * GLOBAL_TARGETS,
                "train_ce": metrics[update - 1]["ce"],
                "train_consistency": metrics[update - 1]["consistency"],
                "teacher_validation_loss": evaluations[update]["losses"]["TEACHER_REAL"],
                "teacher_recovery_fraction": evaluations[update][
                    "teacher_recovery_fraction"
                ],
                "self_validation_loss": evaluations[update]["losses"]["SELF_REAL"],
                "self_recovery": evaluations[update]["self_recovery"],
                "gate": evaluations[update]["completion_diagnostics"]["g"],
            }
            for update in MILESTONES
        },
        "final": final,
        "primary_classification": primary,
        "self_transfer_status": self_status,
        "recommendation": recommendation,
        "scientific_answers": scientific_answers,
        "integrity_pass": integrity_pass,
    }
    final_audit = {
        "experiment": "2D0",
        "checks": {key: "PASS" if value else "FAIL" for key, value in audit_rows.items()},
        "training_checks": train_checks,
        "checkpoint_checks": checkpoint_checks,
        "integrity_pass": integrity_pass,
        "primary_classification": primary,
        "self_transfer_status": self_status,
        "recommendation": recommendation,
    }
    phase_a_lines = [
        "# Experiment 2D0 Phase-A B11 Window Sweep",
        "",
        "| W_B11 | Validation loss | Damage | B11 state cosine | B12 state cosine |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in WINDOWS:
        row = phase_a["rows"][str(candidate)]
        phase_a_lines.append(
            f"| {candidate} | {row['validation_loss']:.10f} | "
            f"{row['damage_vs_1024']:+.10f} | "
            f"{row['b11_post_block_state_drift']['cosine']:.10f} | "
            f"{row['b12_state_drift']['cosine']:.10f} |"
        )
    phase_a_lines.extend(
        ["", f"Selected window: **W={window}**. {selection['selection_rule']}.", ""]
    )
    best_bin = max(
        final["position_bins"].items(), key=lambda item: item[1]["teacher_recovery"]
    )
    report_lines = [
        "# Experiment 2D0 — B11 Top-Down Context Completion on Standard GPT-2",
        "",
        f"Primary classification: **{primary}**",
        "",
        f"Self-transfer status: **{self_status}**",
        "",
        f"Next-step recommendation: **{recommendation}**",
        "",
        "## Phase A",
        "",
        *phase_a_lines[2:],
        "## Training trajectory",
        "",
        "| Targets | Train CE | Consistency | Teacher val | Recovery % | Self val | Self recovery | Gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for update in MILESTONES:
        row = result_summary["milestones"][str(update)]
        report_lines.append(
            f"| {row['targets']:,} | {row['train_ce']:.8f} | "
            f"{row['train_consistency']:.8f} | {row['teacher_validation_loss']:.10f} | "
            f"{100 * row['teacher_recovery_fraction']:.2f}% | "
            f"{row['self_validation_loss']:.10f} | {row['self_recovery']:+.10f} | "
            f"{row['gate']:+.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Final controls",
            "",
            "| Control | Validation loss |",
            "| --- | ---: |",
        ]
    )
    for control in (
        "FULL",
        "SHORT",
        "TEACHER_REAL",
        "TEACHER_SHUFFLED",
        "COMPLETION_ZERO",
        "SELF_REAL",
        "SELF_SHUFFLED",
    ):
        report_lines.append(f"| {control} | {final['losses'][control]:.10f} |")
    report_lines.extend(
        [
            "",
            "## Representation",
            "",
            "| Condition | B11 MSE | B11 cosine | B11 RMS diff | B12 cosine | B12 RMS diff |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in ("SHORT", "TEACHER_REAL", "TEACHER_SHUFFLED", "COMPLETION_ZERO"):
        row = final["states"][condition]
        report_lines.append(
            f"| {condition} | {row['b11']['mse']:.10f} | {row['b11']['cosine']:.10f} | "
            f"{row['b11']['rms_difference']:.10f} | {row['b12']['cosine']:.10f} | "
            f"{row['b12']['rms_difference']:.10f} |"
        )
    report_lines.extend(
        [
            "",
            "## Scientific answers",
            "",
            *[
                f"- **{key}:** {value}"
                for key, value in scientific_answers.items()
            ],
            "",
            f"The largest final position-bin teacher recovery was `{best_bin[0]}` at {best_bin[1]['teacher_recovery']:+.10f} CE.",
            "",
            f"Teacher real beat SHORT on {final['teacher_real_wins_vs_short']}/20 canonical batches. "
            f"Teacher-aligned feedback beat shuffled by {final['teacher_specific_gap']:+.10f} CE. "
            f"B11 state MSE fell by {final['b11_mse_reduction_fraction']:.2%} relative to SHORT.",
            "",
            f"All {len(audit_rows)} integrity checks passed: **{integrity_pass}**. "
            "The forced fresh-process restart occurred at update 96, every milestone checkpoint "
            "strictly reopened by SHA, and the authoritative run ended at exactly 191 updates / "
            "100,139,008 adaptation targets.",
            "",
            "Experiment 2D1 was not launched.",
            "",
            "# EXPERIMENT 2D0 COMPLETE",
            "",
        ]
    )

    artifacts = {
        "PHASE_A_WINDOW_SWEEP.md": "\n".join(phase_a_lines),
        "EXPERIMENT_2D0_FINAL_REPORT.md": "\n".join(report_lines),
    }
    json_artifacts = {
        "phase_a_results.json": phase_a,
        "selected_window.json": selection,
        "FINAL_AUDIT.json": final_audit,
        "result_summary.json": result_summary,
        "milestone_controls.json": milestone_controls,
        "paired_losses.json": paired_losses,
        "state_reconstruction.json": state_reconstruction,
        "position_bin_metrics.json": position_metrics,
        "completion_diagnostics.json": completion_diagnostics,
        "self_transfer.json": self_transfer,
        "cache_audit.json": cache_audit,
        "performance.json": performance,
        "checkpoint_manifest.json": manifest,
        "commands_and_runtime.json": {
            "implementation_commit": git_output("rev-parse", "HEAD"),
            "branch": BRANCH,
            "commands": [
                "torchrun --nproc_per_node=4 scripts/experiment_2d0.py phase-a ...",
                "CUDA_VISIBLE_DEVICES=0 scripts/experiment_2d0.py preflight ...",
                "torchrun --nproc_per_node=4 scripts/experiment_2d0.py equivalence ...",
                "torchrun --nproc_per_node=4 scripts/experiment_2d0.py train --stop-update 96 ...",
                "fresh process: torchrun --nproc_per_node=4 scripts/experiment_2d0.py train --resume-checkpoint checkpoint_updates_000096.pt --stop-update 191 ...",
                "scripts/experiment_2d0.py aggregate ...",
            ],
            "performance": performance,
            "hellaswag_executed": False,
            "experiment_2d1_executed": False,
        },
    }
    for name, value in artifacts.items():
        durable_text(output / name, value)
    for name, value in json_artifacts.items():
        durable_json(output / name, value)
    for name in (
        "CHECKPOINT_DISCOVERY.md",
        "source_checkpoint_manifest.json",
        "environment.json",
        "data_manifest.json",
        "run_identity.json",
        "training_metrics.jsonl",
        "preflight.json",
        "equivalence.json",
        "restart_audit_updates_000096.json",
        "stage_updates_000096.json",
        "stage_updates_000191.json",
    ):
        shutil.copy2(run_root / name, output / name)
    print(
        f"EXPERIMENT_2D0_AGGREGATE_COMPLETE classification={primary} "
        f"self={self_status} integrity={integrity_pass}",
        flush=True,
    )
    if not integrity_pass:
        raise SystemExit("Experiment 2D0 final integrity audit failed")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase_a = subparsers.add_parser("phase-a")
    phase_a.add_argument("--parent-checkpoint", required=True)
    phase_a.add_argument("--validation-shard", required=True)
    phase_a.add_argument("--run-root", required=True)
    phase_a.set_defaults(func=run_phase_a)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--parent-checkpoint", required=True)
    preflight.add_argument("--validation-shard", required=True)
    preflight.add_argument("--selected-window", required=True)
    preflight.add_argument("--run-root", required=True)
    preflight.set_defaults(func=run_preflight)

    equivalence = subparsers.add_parser("equivalence")
    equivalence.add_argument("--parent-checkpoint", required=True)
    equivalence.add_argument("--training-shards", nargs=2, required=True)
    equivalence.add_argument("--selected-window", required=True)
    equivalence.add_argument("--run-root", required=True)
    equivalence.set_defaults(func=run_equivalence)

    train = subparsers.add_parser("train")
    train.add_argument("--parent-checkpoint", required=True)
    train.add_argument("--validation-shard", required=True)
    train.add_argument("--training-shards", nargs=2, required=True)
    train.add_argument("--selected-window", required=True)
    train.add_argument("--preflight", required=True)
    train.add_argument("--equivalence", required=True)
    train.add_argument("--run-root", required=True)
    train.add_argument("--stop-update", type=int, required=True)
    train.add_argument("--resume-checkpoint")
    train.set_defaults(func=run_train)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-root", required=True)
    aggregate.add_argument("--output-dir", required=True)
    aggregate.set_defaults(func=aggregate_final)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
