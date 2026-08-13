#!/usr/bin/env python3
"""Single-GPU diagnostics and guarded training for Experiment 2A0."""

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import smoke_test as support  # noqa: E402


PARENT_COMMIT = "abecd3e91e89e1259f7198d72d15664943ad48bf"
EXPECTED_PARENT_SHA256 = "6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91"
EXPECTED_PARENT_UPDATES = 954
EXPECTED_PARENT_TOKENS = 500_170_752
EXPECTED_NEXT_GLOBAL_BATCH_SHA256 = "8f1848a7f86750145743c77e58cb766a0bb5eddd1137aeb6ade62897df112000"
EXPECTED_DATASET_MANIFEST_SHA256 = "be14a17c21682a018aef68ce02847cced77e921374c01f806deccfba72870f54"
EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256 = "3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb"
CHECKPOINT_SCHEMA = "exp2a0_single_gpu_v1"
SEED = 1337
SOURCE_DEPTHS = (16, 17, 20, 24)
T = 1024
LEGACY_WORLD_SIZE = 4
LEGACY_B = 64
LEGACY_GRAD_ACCUM = 2
GLOBAL_BATCH_TOKENS = LEGACY_WORLD_SIZE * LEGACY_B * T * LEGACY_GRAD_ACCUM
VALIDATION_BATCHES = 20
VALIDATION_B = 64
MAX_LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 715
MAX_STEPS = 19073
SOURCE_FILES = (
    "train_gpt2.py",
    "scripts/experiment_2a0.py",
    "configs/exp2a0_smoke.json",
    "configs/exp2a0_5m.json",
)


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path, mmap=False):
    kwargs = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path, payload):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def nested_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def state_tensor_sha256(model, include_topdown):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        is_topdown = name.startswith("transformer.topdown_attnres.")
        if is_topdown != include_topdown:
            continue
        tensor = value.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def source_file_hashes():
    return {name: file_sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def validate_config(config, kind):
    if tuple(config.get("source_depths", ())) != SOURCE_DEPTHS:
        raise SystemExit(f"{kind} source-depth mismatch")
    if config.get("seed") != SEED or config.get("sequence_length") != T:
        raise SystemExit(f"{kind} seed/sequence-length mismatch")
    if config.get("optimizer_updates") != 10 or config.get("checkpoint_after_updates") != 5:
        raise SystemExit(f"{kind} update/checkpoint cadence mismatch")
    if config.get("mode") != "masked_l1_topdown_teacher":
        raise SystemExit(f"{kind} mode mismatch")
    if kind == "smoke":
        if config.get("protocol") != "exp2a0_teacher_topdown_smoke_v1":
            raise SystemExit("smoke protocol mismatch")
        expected_tokens = (
            config["optimizer_updates"]
            * config["micro_batch_sequences"]
            * config["sequence_length"]
            * config["gradient_accumulation"]
        )
    elif kind == "learn-5m":
        if config.get("protocol") != "exp2a0_teacher_topdown_5m_v1":
            raise SystemExit("5M protocol mismatch")
        expected_geometry = {
            "legacy_world_size": LEGACY_WORLD_SIZE,
            "legacy_micro_batch_sequences_per_rank": LEGACY_B,
            "legacy_gradient_accumulation": LEGACY_GRAD_ACCUM,
            "sequential_microbatches_per_update": LEGACY_WORLD_SIZE * LEGACY_GRAD_ACCUM,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "validation_batches": VALIDATION_BATCHES,
            "validation_batch_sequences": VALIDATION_B,
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected_geometry.items()
            if config.get(key) != value
        }
        if mismatches:
            raise SystemExit(f"5M geometry mismatch: {mismatches}")
        expected_tokens = config["optimizer_updates"] * GLOBAL_BATCH_TOKENS
    else:
        raise ValueError(kind)
    if config.get("processed_student_tokens") != expected_tokens:
        raise SystemExit(f"{kind} processed-token formula mismatch")
    return config


def require_cuda():
    support.assert_cuda_environment(require_a100_80gb=True)
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f"Experiment 2A0 requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    return torch.device("cuda", 0)


def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coefficient * (MAX_LR - MIN_LR)


def dataset_manifest_report(verify_shards=False):
    candidates = [
        REPO_ROOT / "experiment_artifacts" / "edu_fineweb10B_sha256_manifest.txt",
        REPO_ROOT.parent / "build-nanogpt" / "experiment_artifacts" / "edu_fineweb10B_sha256_manifest.txt",
        Path("/workspace/build-nanogpt/experiment_artifacts/edu_fineweb10B_sha256_manifest.txt"),
    ]
    manifest = next((path for path in candidates if path.is_file()), None)
    if manifest is None:
        raise SystemExit("canonical FineWeb SHA256 manifest is unavailable")
    digest = file_sha256(manifest)
    if digest != EXPECTED_DATASET_MANIFEST_SHA256:
        raise SystemExit(f"dataset manifest SHA256 mismatch: {digest}")
    report = {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": digest,
        "shards_verified": False,
    }
    if verify_shards:
        subprocess.run(
            ["sha256sum", "--check", str(manifest.resolve())],
            cwd=REPO_ROOT / "edu_fineweb10B",
            check=True,
            stdout=subprocess.DEVNULL,
        )
        report["shards_verified"] = True
    return report


def validate_parent_payload(checkpoint):
    required = {
        "model",
        "optimizer",
        "training_state",
        "dataloader_states",
        "rng_states",
        "metadata",
        "next_global_batch_sha256",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise SystemExit(f"parent checkpoint missing fields: {missing}")
    state = checkpoint["training_state"]
    if state.get("completed_updates") != EXPECTED_PARENT_UPDATES:
        raise SystemExit(f"parent update mismatch: {state}")
    if state.get("processed_tokens") != EXPECTED_PARENT_TOKENS:
        raise SystemExit(f"parent token mismatch: {state}")
    if len(checkpoint["dataloader_states"]) != LEGACY_WORLD_SIZE:
        raise SystemExit("parent checkpoint does not contain four loader states")
    if checkpoint["next_global_batch_sha256"] != EXPECTED_NEXT_GLOBAL_BATCH_SHA256:
        raise SystemExit("parent next-global-batch hash metadata mismatch")


def inspect_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    digest = file_sha256(checkpoint_path)
    if digest != EXPECTED_PARENT_SHA256:
        raise SystemExit(
            f"parent checkpoint SHA256 mismatch: expected {EXPECTED_PARENT_SHA256}, got {digest}"
        )
    checkpoint = torch_load(checkpoint_path, mmap=True)
    validate_parent_payload(checkpoint)
    state = checkpoint["model"]
    tensor_count = sum(isinstance(value, torch.Tensor) for value in state.values())
    nonfinite = [
        name
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
        and (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all()
    ]
    report = {
        "checkpoint": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": digest,
        "model_tensors": tensor_count,
        "model_nonfinite_tensors": nonfinite,
        "optimizer_state_entries": len(checkpoint["optimizer"]["state"]),
        "completed_updates": checkpoint["training_state"]["completed_updates"],
        "processed_tokens": checkpoint["training_state"]["processed_tokens"],
        "dataloader_rank_states": len(checkpoint["dataloader_states"]),
        "rng_rank_states": len(checkpoint["rng_states"]),
        "world_size": checkpoint["metadata"].get("world_size"),
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "passed": not nonfinite,
    }
    del checkpoint
    gc.collect()
    return report


def data_preflight(checkpoint_path, out_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    digest = file_sha256(checkpoint_path)
    if digest != EXPECTED_PARENT_SHA256:
        raise SystemExit(f"parent checkpoint SHA256 mismatch: {digest}")
    checkpoint = torch_load(checkpoint_path, mmap=True)
    validate_parent_payload(checkpoint)
    symbols = support.load_training_symbols()
    replay_loaders = make_replay_loaders(symbols, checkpoint["dataloader_states"])
    continued_hash = next_update_hash(replay_loaders, symbols, replay=True)
    if continued_hash != EXPECTED_NEXT_GLOBAL_BATCH_SHA256:
        raise SystemExit(f"continued global-batch hash mismatch: {continued_hash}")
    validation_loader = symbols["DataLoaderLite"](
        B=VALIDATION_B, T=T, process_rank=0, num_processes=1, split="val"
    )
    validation_hash = hashlib.sha256()
    for _ in range(VALIDATION_BATCHES):
        x, y = validation_loader.next_batch()
        validation_hash.update(bytes.fromhex(batch_payload_hash(x, y)))
    report = {
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_sha256": digest,
        "dataset": dataset_manifest_report(verify_shards=True),
        "continued_next_global_batch_sha256": continued_hash,
        "continued_next_global_batch_matches_parent": True,
        "validation_batches": VALIDATION_BATCHES,
        "validation_B": VALIDATION_B,
        "validation_T": T,
        "validation_global_batches_sha256": validation_hash.hexdigest(),
        "passed": True,
    }
    write_json(out_path, report)
    return report


def model_config(symbols, enable_topdown):
    return symbols["GPTConfig"](
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        residual_mode="full_attnres",
        enable_topdown_feedback=enable_topdown,
    )


def load_models(checkpoint_path, device, include_teacher=True):
    checkpoint_path = Path(checkpoint_path).resolve()
    digest = file_sha256(checkpoint_path)
    if digest != EXPECTED_PARENT_SHA256:
        raise SystemExit(
            f"parent checkpoint SHA256 mismatch: expected {EXPECTED_PARENT_SHA256}, got {digest}"
        )
    checkpoint = torch_load(checkpoint_path, mmap=True)
    validate_parent_payload(checkpoint)
    symbols = support.load_training_symbols()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    student = symbols["GPT"](model_config(symbols, enable_topdown=True))
    student.load_experiment1_full_attnres_state(checkpoint["model"])
    student.freeze_for_topdown_training()
    teacher = None
    if include_teacher:
        teacher = symbols["GPT"](model_config(symbols, enable_topdown=False))
        teacher.load_state_dict(checkpoint["model"], strict=True)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()

    parent_aux = {
        "dataloader_states": checkpoint["dataloader_states"],
        "rng_states": checkpoint["rng_states"],
        "metadata": checkpoint["metadata"],
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
    }
    del checkpoint
    gc.collect()

    student.to(device)
    if teacher is not None:
        teacher.to(device)
    trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
    if trainable != 1537:
        raise SystemExit(f"expected 1,537 trainable feedback parameters, got {trainable}")
    return symbols, teacher, student, parent_aux


def definition_prefix(source):
    marker = "# -----------------------------------------------------------------------------\n# simple launch:"
    if marker not in source:
        raise RuntimeError("could not find training launch marker")
    return source.split(marker)[0]


def load_reference_symbols():
    source = subprocess.check_output(
        ["git", "show", f"{PARENT_COMMIT}:train_gpt2.py"],
        cwd=REPO_ROOT,
        text=True,
    )
    namespace = {
        "__name__": "experiment_1_checkpoint_reference",
        "__file__": f"{PARENT_COMMIT}:train_gpt2.py",
        "master_process": True,
    }
    exec(compile(definition_prefix(source), namespace["__file__"], "exec"), namespace)
    namespace["master_process"] = True
    return namespace


@torch.no_grad()
def checkpoint_regression(checkpoint_path, device, B=1, sequence_length=1024):
    checkpoint_path = Path(checkpoint_path).resolve()
    digest = file_sha256(checkpoint_path)
    if digest != EXPECTED_PARENT_SHA256:
        raise SystemExit(
            f"parent checkpoint SHA256 mismatch: expected {EXPECTED_PARENT_SHA256}, got {digest}"
        )
    symbols = support.load_training_symbols()
    reference = load_reference_symbols()
    checkpoint = torch_load(checkpoint_path, mmap=True)
    validate_parent_payload(checkpoint)
    reference_model = reference["GPT"](
        reference["GPTConfig"](vocab_size=50304, residual_mode="full_attnres")
    )
    reference_model.load_state_dict(checkpoint["model"], strict=True)
    student = symbols["GPT"](model_config(symbols, enable_topdown=True))
    student.load_experiment1_full_attnres_state(checkpoint["model"])
    del checkpoint
    gc.collect()
    reference_model.to(device).eval()
    student.to(device).eval()

    loader = symbols["DataLoaderLite"](
        B=B,
        T=sequence_length,
        process_rank=0,
        num_processes=1,
        split="val",
    )
    idx, targets = loader.next_batch()
    idx = idx.to(device)
    targets = targets.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected_logits, expected_loss = reference_model(idx, targets)
        actual_logits, actual_loss = student(idx, targets, mode="full_context")
    logits_exact = torch.equal(actual_logits, expected_logits)
    loss_exact = torch.equal(actual_loss, expected_loss)
    maximum_difference = (
        actual_logits.float() - expected_logits.float()
    ).abs().max().item()
    report = {
        "parent_commit": PARENT_COMMIT,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "B": B,
        "T": sequence_length,
        "reference_parameter_count": sum(p.numel() for p in reference_model.parameters()),
        "student_parameter_count": sum(p.numel() for p in student.parameters()),
        "new_parameter_count": sum(
            p.numel() for p in student.transformer.topdown_attnres.parameters()
        ),
        "logits_bit_exact": logits_exact,
        "loss_bit_exact": loss_exact,
        "maximum_absolute_logit_difference": maximum_difference,
        "reference_loss": expected_loss.float().item(),
        "student_loss": actual_loss.float().item(),
        "passed": logits_exact and loss_exact,
    }
    del reference_model, student, expected_logits, actual_logits
    torch.cuda.empty_cache()
    return report


def loader_state(loader):
    return {
        "process_rank": loader.process_rank,
        "num_processes": loader.num_processes,
        "B": loader.B,
        "T": loader.T,
        "current_shard": loader.current_shard,
        "current_position": loader.current_position,
        "current_shard_path": loader.shards[loader.current_shard],
    }


def restore_loader_state(loader, state, symbols):
    expected = {
        "process_rank": loader.process_rank,
        "num_processes": loader.num_processes,
        "B": loader.B,
        "T": loader.T,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise SystemExit(f"loader {key} mismatch: checkpoint={state.get(key)} runtime={value}")
    shard = int(state["current_shard"])
    if not 0 <= shard < len(loader.shards):
        raise SystemExit(f"invalid loader shard: {shard}")
    if loader.shards[shard] != state["current_shard_path"]:
        raise SystemExit(
            f"loader shard identity mismatch: {loader.shards[shard]} != {state['current_shard_path']}"
        )
    loader.current_shard = shard
    loader.tokens = symbols["load_tokens"](loader.shards[shard])
    loader.current_position = int(state["current_position"])
    if loader.current_position < 0 or loader.current_position + loader.B * loader.T + 1 > len(loader.tokens):
        raise SystemExit(f"invalid loader position: {loader.current_position}")


class SharedReplayLoader:
    """Experiment 1B loader semantics with a shared in-process shard cache."""

    def __init__(self, symbols, process_rank, cache):
        self.B = LEGACY_B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = LEGACY_WORLD_SIZE
        self.symbols = symbols
        self.cache = cache
        names = sorted(name for name in os.listdir("edu_fineweb10B") if "train" in name)
        self.shards = [os.path.join("edu_fineweb10B", name) for name in names]
        if not self.shards:
            raise SystemExit("no FineWeb training shards found")
        self.current_shard = 0
        self.current_position = self.B * self.T * process_rank
        self.tokens = self._tokens(self.current_shard)

    def _tokens(self, shard):
        path = self.shards[shard]
        if path not in self.cache:
            self.cache.clear()
            self.cache[path] = self.symbols["load_tokens"](path)
        return self.cache[path]

    def next_batch(self):
        count = self.B * self.T
        buf = self.tokens[self.current_position:self.current_position + count + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += count * self.num_processes
        if self.current_position + (count * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._tokens(self.current_shard)
            self.current_position = count * self.process_rank
        return x, y


def restore_replay_loader(loader, state):
    expected = {
        "process_rank": loader.process_rank,
        "num_processes": loader.num_processes,
        "B": loader.B,
        "T": loader.T,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise SystemExit(f"replay loader {key} mismatch: {state.get(key)} != {value}")
    shard = int(state["current_shard"])
    if not 0 <= shard < len(loader.shards):
        raise SystemExit(f"invalid replay loader shard: {shard}")
    if loader.shards[shard] != state["current_shard_path"]:
        raise SystemExit("replay loader shard path mismatch")
    loader.current_shard = shard
    loader.tokens = loader._tokens(shard)
    loader.current_position = int(state["current_position"])
    minimum = loader.B * loader.T * loader.process_rank
    if (
        loader.current_position < minimum
        or loader.current_position + loader.B * loader.T + 1 > len(loader.tokens)
    ):
        raise SystemExit(f"invalid replay loader position: {loader.current_position}")


def make_replay_loaders(symbols, states):
    if len(states) != LEGACY_WORLD_SIZE:
        raise SystemExit(
            f"expected {LEGACY_WORLD_SIZE} replay loader states, got {len(states)}"
        )
    cache = {}
    loaders = [SharedReplayLoader(symbols, rank, cache) for rank in range(LEGACY_WORLD_SIZE)]
    for loader, state in zip(loaders, states):
        restore_replay_loader(loader, state)
    return loaders


def make_loaders_from_states(symbols, states, replay):
    if replay:
        return make_replay_loaders(symbols, states)
    if len(states) != 1:
        raise SystemExit(f"smoke checkpoint must contain one loader state, got {len(states)}")
    state = states[0]
    loader = symbols["DataLoaderLite"](
        B=state["B"],
        T=state["T"],
        process_rank=state["process_rank"],
        num_processes=state["num_processes"],
        split="train",
    )
    restore_loader_state(loader, state, symbols)
    return [loader]


def batch_payload_hash(x, y):
    digest = hashlib.sha256()
    digest.update(x.contiguous().numpy().tobytes())
    digest.update(y.contiguous().numpy().tobytes())
    return digest.hexdigest()


def aggregate_batch_hash(batches):
    digest = hashlib.sha256()
    for x, y in batches:
        digest.update(bytes.fromhex(batch_payload_hash(x, y)))
    return digest.hexdigest()


def snapshot_loaders(loaders):
    return [loader_state(loader) for loader in loaders]


def restore_loader_group(loaders, states, symbols, replay):
    if len(loaders) != len(states):
        raise SystemExit(f"loader-state count mismatch: {len(loaders)} != {len(states)}")
    for loader, state in zip(loaders, states):
        if replay:
            restore_replay_loader(loader, state)
        else:
            restore_loader_state(loader, state, symbols)


def next_update_hash(loaders, symbols, replay):
    states = snapshot_loaders(loaders)
    batches = []
    if replay:
        for _microstep in range(LEGACY_GRAD_ACCUM):
            for loader in loaders:
                batches.append(loader.next_batch())
    else:
        batches.append(loaders[0].next_batch())
    digest = aggregate_batch_hash(batches)
    restore_loader_group(loaders, states, symbols, replay)
    return digest


def update_batches(loaders, replay):
    if replay:
        for _microstep in range(LEGACY_GRAD_ACCUM):
            for loader in loaders:
                yield loader.next_batch()
    else:
        yield loaders[0].next_batch()


def capture_rng_state():
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(0),
    }


def restore_rng_state(state):
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], 0)


def feedback_optimizer(student):
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    kwargs = dict(lr=get_lr(EXPECTED_PARENT_UPDATES), betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    if fused_available:
        kwargs["fused"] = True
    return torch.optim.AdamW(parameters, **kwargs)


def cpu_feedback_optimizer(student):
    return torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=get_lr(EXPECTED_PARENT_UPDATES),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        fused=False,
    )


@torch.no_grad()
def teacher_memory(teacher, x, symbols):
    teacher.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sources = teacher.capture_residual_sources(x, SOURCE_DEPTHS)
    return symbols["shift_teacher_sources"](sources)


def gradient_report(student, teacher):
    router = student.transformer.topdown_attnres

    def row(parameter):
        if parameter.grad is None:
            return {"present": False, "finite": None, "nonzero": False, "l2_norm": 0.0}
        gradient = parameter.grad.detach().float()
        return {
            "present": True,
            "finite": bool(torch.isfinite(gradient).all()),
            "nonzero": bool(torch.count_nonzero(gradient).item()),
            "l2_norm": gradient.norm().item(),
        }

    base_with_grad = [
        name
        for name, parameter in student.named_parameters()
        if not name.startswith("transformer.topdown_attnres.") and parameter.grad is not None
    ]
    teacher_with_grad = [name for name, parameter in teacher.named_parameters() if parameter.grad is not None]
    return {
        "gate": row(router.gate),
        "query": row(router.query),
        "rmsnorm": row(router.norm.weight),
        "base_tensors_with_grad": base_with_grad,
        "teacher_tensors_with_grad": teacher_with_grad,
    }


def checkpoint_payload(student, optimizer, loaders, symbols, replay, training_state, parent_aux, metadata):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_state": dict(training_state),
        "dataloader_states": snapshot_loaders(loaders),
        "rng_state": capture_rng_state(),
        "metadata": dict(metadata),
        "parent_checkpoint_sha256": parent_aux["checkpoint_sha256"],
        "next_global_batch_sha256": next_update_hash(loaders, symbols, replay),
    }


def optimizer_state_report(state, completed_updates):
    groups = state.get("param_groups", [])
    if len(groups) != 1:
        raise SystemExit(f"expected one feedback optimizer group, got {len(groups)}")
    group = groups[0]
    expected_lr = get_lr(EXPECTED_PARENT_UPDATES + completed_updates - 1)
    expected = {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "lr": expected_lr,
    }
    mismatches = {
        key: (group.get(key), value)
        for key, value in expected.items()
        if group.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"feedback optimizer hyperparameter mismatch: {mismatches}")
    if completed_updates > 0 and len(state.get("state", {})) != 3:
        raise SystemExit(
            f"expected three feedback optimizer states, got {len(state.get('state', {}))}"
        )
    nonfinite = []
    for parameter_id, values in state.get("state", {}).items():
        for name, value in values.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                nonfinite.append(f"{parameter_id}:{name}")
    if nonfinite:
        raise SystemExit(f"non-finite feedback optimizer tensors: {nonfinite}")
    return {
        "state_entries": len(state.get("state", {})),
        "lr": group["lr"],
        "nonfinite_tensors": nonfinite,
    }


def verify_exp2a0_checkpoint(
    path,
    symbols,
    student,
    optimizer,
    expected_payload,
    replay,
):
    checkpoint = torch_load(path, mmap=True)
    required = {
        "schema",
        "model",
        "optimizer",
        "training_state",
        "dataloader_states",
        "rng_state",
        "metadata",
        "parent_checkpoint_sha256",
        "next_global_batch_sha256",
    }
    if set(checkpoint) != required:
        raise SystemExit(f"Experiment 2A0 checkpoint fields mismatch: {sorted(checkpoint)}")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise SystemExit("Experiment 2A0 checkpoint schema mismatch")
    if checkpoint["parent_checkpoint_sha256"] != EXPECTED_PARENT_SHA256:
        raise SystemExit("Experiment 2A0 checkpoint parent mismatch")
    if checkpoint["training_state"] != expected_payload["training_state"]:
        raise SystemExit("Experiment 2A0 training-state reload mismatch")
    if checkpoint["metadata"] != expected_payload["metadata"]:
        raise SystemExit("Experiment 2A0 metadata reload mismatch")
    if not nested_equal(checkpoint["model"], student.state_dict()):
        raise SystemExit("Experiment 2A0 live/saved model mismatch")
    if not nested_equal(checkpoint["optimizer"], optimizer.state_dict()):
        raise SystemExit("Experiment 2A0 live/saved optimizer mismatch")
    if not nested_equal(checkpoint["dataloader_states"], expected_payload["dataloader_states"]):
        raise SystemExit("Experiment 2A0 live/saved loader mismatch")
    if not nested_equal(checkpoint["rng_state"], expected_payload["rng_state"]):
        raise SystemExit("Experiment 2A0 live/saved RNG mismatch")
    fresh_loaders = make_loaders_from_states(
        symbols, checkpoint["dataloader_states"], replay
    )
    expected_next = next_update_hash(fresh_loaders, symbols, replay)
    if checkpoint["next_global_batch_sha256"] != expected_next:
        raise SystemExit("Experiment 2A0 next-batch reload mismatch")

    live_rng = capture_rng_state()
    try:
        with torch.random.fork_rng(devices=[]):
            clone = symbols["GPT"](model_config(symbols, enable_topdown=True))
        clone.freeze_for_topdown_training()
        clone.load_state_dict(checkpoint["model"], strict=True)
        clone_optimizer = cpu_feedback_optimizer(clone)
        clone_optimizer.load_state_dict(checkpoint["optimizer"])
        if not nested_equal(clone.state_dict(), checkpoint["model"]):
            raise SystemExit("Experiment 2A0 strict model reload mismatch")
        if not nested_equal(clone_optimizer.state_dict(), checkpoint["optimizer"]):
            raise SystemExit("Experiment 2A0 optimizer reload mismatch")
    finally:
        restore_rng_state(live_rng)
    completed_updates = checkpoint["training_state"]["completed_updates"]
    optimizer_report = optimizer_state_report(checkpoint["optimizer"], completed_updates)
    report = {
        "schema": checkpoint["schema"],
        "model_tensors": len(checkpoint["model"]),
        "optimizer_state_entries": len(checkpoint["optimizer"]["state"]),
        "completed_updates": completed_updates,
        "processed_student_tokens": checkpoint["training_state"]["processed_student_tokens"],
        "loader_states": len(checkpoint["dataloader_states"]),
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "model_strict_reload": True,
        "optimizer_reload": True,
        "optimizer": optimizer_report,
        "live_state_match": True,
        "serialized_loader_replay_match": True,
        "rng_reload_match": True,
        "rng_fields": sorted(checkpoint["rng_state"]),
        "passed": True,
    }
    del checkpoint, clone, clone_optimizer, fresh_loaders
    gc.collect()
    return report


def save_exp2a0_checkpoint(path, student, optimizer, loaders, symbols, replay, training_state, parent_aux, metadata):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise SystemExit(f"stale incomplete checkpoint requires inspection: {temporary}")
    payload = checkpoint_payload(
        student, optimizer, loaders, symbols, replay, training_state, parent_aux, metadata
    )
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    verification = verify_exp2a0_checkpoint(
        temporary, symbols, student, optimizer, payload, replay
    )
    digest = file_sha256(temporary)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    sidecar = {
        "checkpoint": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "verification": verification,
    }
    write_json(path.with_suffix(path.suffix + ".verification.json"), sidecar)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return sidecar


def verify_checkpoint_sidecars(path):
    path = Path(path).resolve()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    verification_path = path.with_suffix(path.suffix + ".verification.json")
    if not checksum_path.is_file() or not verification_path.is_file():
        raise SystemExit(f"checkpoint sidecars are missing for {path}")
    fields = checksum_path.read_text().strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise SystemExit(f"malformed checkpoint checksum sidecar: {checksum_path}")
    digest = file_sha256(path)
    if digest != fields[0]:
        raise SystemExit(f"checkpoint SHA256 sidecar mismatch: {digest} != {fields[0]}")
    verification = json.loads(verification_path.read_text())
    if verification.get("sha256") != digest or not verification.get("verification", {}).get("passed"):
        raise SystemExit("checkpoint verification sidecar mismatch")
    return {"checkpoint": str(path), "sha256": digest, "verification": verification_path.name}


@torch.no_grad()
def production_causality_test(teacher, symbols, device, position=512):
    loader = symbols["DataLoaderLite"](
        B=2, T=T, process_rank=0, num_processes=1, split="val"
    )
    first, _ = loader.next_batch()
    second = first.clone()
    second[:, position:] = (second[:, position:] + 1) % 50257
    first = first.to(device)
    second = second.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first_sources = teacher.capture_residual_sources(first, SOURCE_DEPTHS)
        second_sources = teacher.capture_residual_sources(second, SOURCE_DEPTHS)
    first_memory = symbols["shift_teacher_sources"](first_sources)
    second_memory = symbols["shift_teacher_sources"](second_sources)
    per_source_exact = {
        f"v{depth}": torch.equal(first_memory[index, :, position], second_memory[index, :, position])
        for index, depth in enumerate(SOURCE_DEPTHS)
    }
    position_zero_is_zero = (
        torch.count_nonzero(first_memory[:, :, 0]).item() == 0
        and torch.count_nonzero(second_memory[:, :, 0]).item() == 0
    )
    return {
        "B": 2,
        "T": T,
        "tested_position": position,
        "suffix_difference_starts_at": position,
        "per_source_memory_bit_exact": per_source_exact,
        "position_zero_memory_exactly_zero": position_zero_is_zero,
        "passed": all(per_source_exact.values()) and position_zero_is_zero,
    }


@torch.no_grad()
def frozen_diagnostics(checkpoint_path, device, out_path):
    symbols, teacher, student, parent_aux = load_models(checkpoint_path, device, include_teacher=True)
    student.eval()
    loader = symbols["DataLoaderLite"](
        B=VALIDATION_B, T=T, process_rank=0, num_processes=1, split="val"
    )
    totals = {"full_context": 0.0, "masked_l1_no_feedback": 0.0, "zero_gate_feedback": 0.0}
    batch_loss_equality = True
    validation_hash = hashlib.sha256()
    causality = production_causality_test(teacher, symbols, device)
    if not causality["passed"]:
        raise SystemExit(f"future-token leakage test failed: {causality}")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for batch_index in range(VALIDATION_BATCHES):
        x, y = loader.next_batch()
        validation_hash.update(bytes.fromhex(batch_payload_hash(x, y)))
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, full_loss = student(x, y, mode="full_context")
        del logits
        full_value = full_loss.detach().double().item()
        del full_loss
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, masked_loss = student(x, y, mode="masked_l1_no_feedback")
        del logits
        masked_value = masked_loss.detach().double().item()
        del masked_loss
        memory = teacher_memory(teacher, x, symbols)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, zero_loss = student(
                x,
                y,
                mode="masked_l1_topdown_teacher",
                feedback_sources=memory,
            )
        del logits
        zero_value = zero_loss.detach().double().item()
        del zero_loss, memory
        values = {
            "full_context": full_value,
            "masked_l1_no_feedback": masked_value,
            "zero_gate_feedback": zero_value,
        }
        for name, value in values.items():
            totals[name] += value
        batch_loss_equality &= values["masked_l1_no_feedback"] == values["zero_gate_feedback"]
        print(
            f"validation batch {batch_index + 1:02d}/{VALIDATION_BATCHES} "
            f"full={values['full_context']:.6f} masked={values['masked_l1_no_feedback']:.6f}",
            flush=True,
        )
    losses = {name: total / VALIDATION_BATCHES for name, total in totals.items()}
    report = {
        "experiment": "Experiment 2A0 frozen diagnostics",
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short", "--branch"),
        "parent_checkpoint": parent_aux["checkpoint"],
        "parent_checkpoint_sha256": parent_aux["checkpoint_sha256"],
        "validation_batches": VALIDATION_BATCHES,
        "validation_B": VALIDATION_B,
        "validation_T": T,
        "validation_tokens": VALIDATION_BATCHES * VALIDATION_B * T,
        "validation_global_batches_sha256": validation_hash.hexdigest(),
        "causality": causality,
        "losses": losses,
        "damage": losses["masked_l1_no_feedback"] - losses["full_context"],
        "zero_gate_equals_masked_each_batch": batch_loss_equality,
        "new_parameter_count": sum(
            parameter.numel()
            for parameter in student.transformer.topdown_attnres.parameters()
        ),
        "gate": student.transformer.topdown_attnres.gate.item(),
        "query_norm": student.transformer.topdown_attnres.query.float().norm().item(),
        "routing_weights": {f"v{depth}": 0.25 for depth in SOURCE_DEPTHS},
        "routing_entropy": math.log(len(SOURCE_DEPTHS)),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "passed": batch_loss_equality and causality["passed"],
    }
    write_json(out_path, report)
    return report


def training_metadata(kind, config, parent_aux):
    return {
        "experiment": "Experiment 2A0",
        "kind": kind,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short", "--branch"),
        "parent_commit": PARENT_COMMIT,
        "parent_checkpoint": parent_aux["checkpoint"],
        "parent_checkpoint_sha256": parent_aux["checkpoint_sha256"],
        "source_depths": list(SOURCE_DEPTHS),
        "destination": "Block 1 Attention input",
        "gate": "scalar tanh, initialized exactly zero",
        "teacher": "frozen eval/no_grad; raw residual sources shifted one token and detached",
        "student_base": "frozen",
        "optimizer": "AdamW betas=(0.9,0.95), eps=1e-8, weight_decay=0",
        "schedule_start_completed_update": EXPECTED_PARENT_UPDATES,
        "config": config,
        "source_file_sha256": source_file_hashes(),
        "dataset": dataset_manifest_report(verify_shards=False),
        "determinism": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def require_clean_training_tree():
    if git_output("status", "--porcelain"):
        raise SystemExit("optimizer runs require a clean committed Git worktree")
    if git_output("branch", "--show-current") != "experiment-2a0-topdown-l1":
        raise SystemExit("optimizer runs require branch experiment-2a0-topdown-l1")


def gradient_staging_report(rows):
    if len(rows) < 3:
        raise SystemExit("gradient staging requires at least three updates")
    first = rows[0]["gradients"]
    second = rows[1]["gradients"]
    first_exact = (
        first["gate"]["nonzero"]
        and not first["query"]["nonzero"]
        and not first["rmsnorm"]["nonzero"]
    )
    second_exact = (
        second["gate"]["nonzero"]
        and second["query"]["nonzero"]
        and not second["rmsnorm"]["nonzero"]
    )
    later_rmsnorm = any(row["gradients"]["rmsnorm"]["nonzero"] for row in rows[2:])
    report = {
        "update_1_gate_only": first_exact,
        "update_2_gate_and_query_only": second_exact,
        "rmsnorm_nonzero_from_update_3_or_later": later_rmsnorm,
    }
    report["passed"] = all(report.values())
    return report


def reconcile_metrics(path, completed_updates):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    retained = [row for row in rows if row.get("update", -1) < completed_updates]
    updates = [row["update"] for row in retained]
    expected = list(range(completed_updates))
    if updates != expected:
        raise SystemExit(f"metrics before resume checkpoint are not exact: {updates} != {expected}")
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in retained))
    return {
        "rows_before": len(rows),
        "rows_retained": len(retained),
        "rows_truncated": len(rows) - len(retained),
        "completed_updates": completed_updates,
    }


def train_updates(
    teacher,
    student,
    optimizer,
    loaders,
    symbols,
    replay,
    start_update,
    end_update,
    tokens_per_update,
    metrics_path,
):
    student.train()
    teacher.eval()
    rows = []
    topdown_nonzero_seen = {"gate": False, "query": False, "rmsnorm": False}
    microbatches = LEGACY_WORLD_SIZE * LEGACY_GRAD_ACCUM if replay else 1
    for update in range(start_update, end_update):
        optimizer.zero_grad(set_to_none=True)
        student.set_topdown_instrumentation(True)
        loss_total = 0.0
        forward_seconds = 0.0
        backward_seconds = 0.0
        routing_sums = torch.zeros(len(SOURCE_DEPTHS), dtype=torch.float64)
        entropy_sum = 0.0
        update_hash = hashlib.sha256()
        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        for x_cpu, y_cpu in update_batches(loaders, replay):
            update_hash.update(bytes.fromhex(batch_payload_hash(x_cpu, y_cpu)))
            x = x_cpu.to("cuda", non_blocking=True)
            y = y_cpu.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            memory = teacher_memory(teacher, x, symbols)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = student(
                    x,
                    y,
                    mode="masked_l1_topdown_teacher",
                    feedback_sources=memory,
                )
            del logits
            scaled_loss = loss / microbatches
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            scaled_loss.backward()
            torch.cuda.synchronize()
            backward_seconds += time.perf_counter() - backward_start
            loss_total += scaled_loss.detach().float().item()
            stats = student.get_topdown_stats()
            routing_sums += torch.tensor(stats["mean_weights"], dtype=torch.float64)
            entropy_sum += stats["mean_entropy"]
            del x, y, memory, loss, scaled_loss
        student.set_topdown_instrumentation(False)

        gradients = gradient_report(student, teacher)
        for name in topdown_nonzero_seen:
            topdown_nonzero_seen[name] |= gradients[name]["nonzero"]
        for name in ("gate", "query", "rmsnorm"):
            if not gradients[name]["present"] or not gradients[name]["finite"]:
                raise SystemExit(
                    f"missing/non-finite {name} gradient at update {update}: {gradients}"
                )
        if gradients["base_tensors_with_grad"] or gradients["teacher_tensors_with_grad"]:
            raise SystemExit(f"freeze boundary violated at update {update}: {gradients}")
        if not math.isfinite(loss_total):
            raise SystemExit(f"non-finite loss at update {update}")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters() if parameter.requires_grad], 1.0
        )
        lr = get_lr(EXPECTED_PARENT_UPDATES + update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        router = student.transformer.topdown_attnres
        row = {
            "kind": "train",
            "update": update,
            "completed_updates": update + 1,
            "processed_student_tokens": (update + 1) * tokens_per_update,
            "global_schedule_step": EXPECTED_PARENT_UPDATES + update,
            "lr": lr,
            "loss": loss_total,
            "grad_norm": float(grad_norm),
            "gradients": gradients,
            "gate": router.gate.detach().float().item(),
            "gate_coefficient": router.gate.detach().float().tanh().item(),
            "query_norm": router.query.detach().float().norm().item(),
            "rmsnorm_displacement": (
                router.norm.weight.detach().float() - 1
            ).norm().item(),
            "routing_weights": {
                f"v{depth}": value
                for depth, value in zip(SOURCE_DEPTHS, (routing_sums / microbatches).tolist())
            },
            "routing_entropy": entropy_sum / microbatches,
            "global_batch_sha256": update_hash.hexdigest(),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "wall_seconds": time.perf_counter() - wall_start,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        }
        append_jsonl(metrics_path, row)
        rows.append(row)
        print(
            f"update {update + 1:02d}/{end_update} loss={loss_total:.6f} "
            f"gate={row['gate']:.6g} qnorm={row['query_norm']:.6g} "
            f"peak={row['peak_allocated_mb']:.1f} MiB",
            flush=True,
        )
    return rows, topdown_nonzero_seen


def prepare_run_dir(run_dir, config_path, metadata):
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite run directory: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    config = json.loads(Path(config_path).read_text())
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metadata.json", metadata)
    (run_dir / "metrics.jsonl").write_text("")
    return run_dir, config


def run_smoke(args, device):
    if not args.allow_optimizer_steps:
        raise SystemExit("optimizer steps are locked; explicit --allow-optimizer-steps is required")
    require_clean_training_tree()
    config_path = REPO_ROOT / "configs" / "exp2a0_smoke.json"
    config = validate_config(json.loads(config_path.read_text()), "smoke")
    symbols, teacher, student, parent_aux = load_models(args.checkpoint, device, include_teacher=True)
    metadata = training_metadata("10-update architecture smoke", config, parent_aux)
    run_dir, _ = prepare_run_dir(args.run_dir, config_path, metadata)
    loaders = initial_training_loaders(symbols, config, False, parent_aux)
    optimizer = feedback_optimizer(student)
    frozen_before = state_tensor_sha256(student, include_topdown=False)
    initial = {
        "gate": student.transformer.topdown_attnres.gate.item(),
        "query_norm": student.transformer.topdown_attnres.query.float().norm().item(),
        "rmsnorm_displacement": (
            student.transformer.topdown_attnres.norm.weight.float() - 1
        ).norm().item(),
        "routing_weights": {f"v{depth}": 0.25 for depth in SOURCE_DEPTHS},
        "routing_entropy": math.log(4),
    }
    write_json(run_dir / "initial_routing.json", initial)
    first_rows, first_nonzero = train_updates(
        teacher,
        student,
        optimizer,
        loaders,
        symbols,
        replay=False,
        start_update=0,
        end_update=config["checkpoint_after_updates"],
        tokens_per_update=config["micro_batch_sequences"] * T,
        metrics_path=run_dir / "metrics.jsonl",
    )
    midpoint_state = {
        "completed_updates": config["checkpoint_after_updates"],
        "processed_student_tokens": (
            config["checkpoint_after_updates"] * config["micro_batch_sequences"] * T
        ),
    }
    if state_tensor_sha256(student, include_topdown=False) != frozen_before:
        raise SystemExit("frozen base model changed before smoke midpoint checkpoint")
    midpoint_checkpoint = save_exp2a0_checkpoint(
        run_dir / "checkpoints" / "checkpoint_updates_000005.pt",
        student,
        optimizer,
        loaders,
        symbols,
        False,
        midpoint_state,
        parent_aux,
        metadata,
    )
    student, optimizer, loaders, restart_state, restart_audit = force_checkpoint_restart(
        run_dir / "checkpoints" / "checkpoint_updates_000005.pt",
        student,
        optimizer,
        loaders,
        symbols,
        False,
        parent_aux,
        metadata,
        device,
    )
    if restart_state != midpoint_state:
        raise SystemExit("smoke restart training-state mismatch")
    second_rows, second_nonzero = train_updates(
        teacher,
        student,
        optimizer,
        loaders,
        symbols,
        replay=False,
        start_update=config["checkpoint_after_updates"],
        end_update=config["optimizer_updates"],
        tokens_per_update=config["micro_batch_sequences"] * T,
        metrics_path=run_dir / "metrics.jsonl",
    )
    rows = first_rows + second_rows
    nonzero_seen = {
        name: first_nonzero[name] or second_nonzero[name]
        for name in first_nonzero
    }
    staging = gradient_staging_report(rows)
    frozen_after = state_tensor_sha256(student, include_topdown=False)
    state = {
        "completed_updates": config["optimizer_updates"],
        "processed_student_tokens": config["processed_student_tokens"],
    }
    checkpoint = save_exp2a0_checkpoint(
        run_dir / "checkpoints" / "checkpoint_updates_000010.pt",
        student,
        optimizer,
        loaders,
        symbols,
        False,
        state,
        parent_aux,
        metadata,
    )
    passed = staging["passed"] and frozen_before == frozen_after and all(nonzero_seen.values()) and all(
        not row["gradients"]["base_tensors_with_grad"]
        and not row["gradients"]["teacher_tensors_with_grad"]
        and math.isfinite(row["loss"])
        for row in rows
    )
    summary = {
        "updates": len(rows),
        "processed_student_tokens": config["processed_student_tokens"],
        "trainable_parameters": 1537,
        "loss_trajectory": [row["loss"] for row in rows],
        "nonzero_gradient_seen": nonzero_seen,
        "gradient_staging": staging,
        "forced_restart": restart_audit,
        "frozen_model_sha256_before": frozen_before,
        "frozen_model_sha256_after": frozen_after,
        "frozen_model_bit_exact": frozen_before == frozen_after,
        "first_update_gradients": rows[0]["gradients"],
        "last_update_gradients": rows[-1]["gradients"],
        "gate_trajectory": [row["gate"] for row in rows],
        "query_norm_trajectory": [row["query_norm"] for row in rows],
        "peak_allocated_mb": max(row["peak_allocated_mb"] for row in rows),
        "forward_seconds": sum(row["forward_seconds"] for row in rows),
        "backward_seconds": sum(row["backward_seconds"] for row in rows),
        "checkpoint": checkpoint,
        "midpoint_checkpoint": midpoint_checkpoint,
        "passed": passed,
    }
    write_json(run_dir / "smoke_summary.json", summary)
    if not passed:
        raise SystemExit("Experiment 2A0 smoke failed")
    return summary


@torch.no_grad()
def evaluate_controls(student, teacher, symbols, device):
    loader = symbols["DataLoaderLite"](
        B=VALIDATION_B, T=T, process_rank=0, num_processes=1, split="val"
    )
    names = [
        "full_context",
        "masked_l1_no_feedback",
        "real_feedback",
        "shuffled_feedback",
        "zero_feedback",
    ]
    ablation_names = [f"mask_v{depth}" for depth in SOURCE_DEPTHS]
    totals = {name: 0.0 for name in names + ablation_names}
    routing_weights = torch.zeros(len(SOURCE_DEPTHS), dtype=torch.float64)
    routing_entropy = 0.0
    zero_equals_masked = True
    validation_hash = hashlib.sha256()
    permutation = symbols["fixed_derangement"](VALIDATION_B, device)
    student.eval()
    teacher.eval()
    for batch_index in range(VALIDATION_BATCHES):
        x, y = loader.next_batch()
        validation_hash.update(bytes.fromhex(batch_payload_hash(x, y)))
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        memory = teacher_memory(teacher, x, symbols)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = student(x, y, mode="full_context")
        del logits
        totals["full_context"] += loss.detach().double().item()
        del loss
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, masked_loss = student(x, y, mode="masked_l1_no_feedback")
        del logits
        totals["masked_l1_no_feedback"] += masked_loss.detach().double().item()

        student.set_topdown_instrumentation(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, real_loss = student(
                x, y, mode="masked_l1_topdown_teacher", feedback_sources=memory
            )
        del logits
        stats = student.get_topdown_stats()
        student.set_topdown_instrumentation(False)
        routing_weights += torch.tensor(stats["mean_weights"], dtype=torch.float64)
        routing_entropy += stats["mean_entropy"]
        totals["real_feedback"] += real_loss.detach().double().item()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, shuffled_loss = student(
                x,
                y,
                mode="masked_l1_shuffled_feedback",
                feedback_sources=memory,
                feedback_permutation=permutation,
            )
        del logits
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, zero_loss = student(
                x,
                y,
                mode="masked_l1_topdown_teacher",
                feedback_sources=memory,
                feedback_gate_override=0.0,
            )
        del logits
        totals["shuffled_feedback"] += shuffled_loss.detach().double().item()
        totals["zero_feedback"] += zero_loss.detach().double().item()
        zero_equals_masked &= torch.equal(zero_loss, masked_loss)

        for depth in SOURCE_DEPTHS:
            student.set_topdown_source_mask(depth)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, ablated_loss = student(
                    x, y, mode="masked_l1_topdown_teacher", feedback_sources=memory
                )
            del logits
            totals[f"mask_v{depth}"] += ablated_loss.detach().double().item()
            del ablated_loss
        student.set_topdown_source_mask(None)
        del x, y, memory, real_loss, shuffled_loss, zero_loss, masked_loss
        print(f"control validation batch {batch_index + 1:02d}/{VALIDATION_BATCHES}", flush=True)

    losses = {name: total / VALIDATION_BATCHES for name, total in totals.items()}
    validation_digest = validation_hash.hexdigest()
    if validation_digest != EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256:
        raise SystemExit(
            f"control validation prefix mismatch: {validation_digest}"
        )
    damage = losses["masked_l1_no_feedback"] - losses["full_context"]
    recovery = losses["masked_l1_no_feedback"] - losses["real_feedback"]
    recovery_fraction = recovery / damage if damage > 0 else None
    source_ablation = {
        f"v{depth}": {
            "validation_loss": losses[f"mask_v{depth}"],
            "delta_vs_real_feedback": losses[f"mask_v{depth}"] - losses["real_feedback"],
        }
        for depth in SOURCE_DEPTHS
    }
    confirmed = (
        damage > 0
        and recovery > 0
        and zero_equals_masked
        and losses["shuffled_feedback"] > losses["real_feedback"]
        and losses["zero_feedback"] > losses["real_feedback"]
    )
    router = student.transformer.topdown_attnres
    return {
        "validation_batches": VALIDATION_BATCHES,
        "validation_B": VALIDATION_B,
        "validation_T": T,
        "validation_global_batches_sha256": validation_digest,
        "losses": {name: losses[name] for name in names},
        "damage": damage,
        "recovery": recovery,
        "recovery_fraction": recovery_fraction,
        "zero_equals_masked_each_batch": zero_equals_masked,
        "routing": {
            "mean_weights": {
                f"v{depth}": value
                for depth, value in zip(
                    SOURCE_DEPTHS, (routing_weights / VALIDATION_BATCHES).tolist()
                )
            },
            "mean_tokenwise_entropy": routing_entropy / VALIDATION_BATCHES,
            "query_norm": router.query.detach().float().norm().item(),
            "gate": router.gate.detach().float().item(),
            "gate_coefficient": router.gate.detach().float().tanh().item(),
        },
        "source_ablation": source_ablation,
        "signal_confirmed": confirmed,
    }


def load_exp2_resume(
    path,
    student,
    optimizer,
    loaders,
    symbols,
    replay,
    parent_aux,
    expected_metadata,
):
    integrity = verify_checkpoint_sidecars(path)
    checkpoint = torch_load(path, mmap=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("resume checkpoint schema mismatch")
    if checkpoint.get("parent_checkpoint_sha256") != parent_aux["checkpoint_sha256"]:
        raise SystemExit("resume checkpoint parent mismatch")
    if checkpoint.get("metadata") != expected_metadata:
        raise SystemExit("resume checkpoint metadata/config/source mismatch")
    state = checkpoint.get("training_state", {})
    completed = state.get("completed_updates")
    config = expected_metadata["config"]
    tokens_per_update = (
        GLOBAL_BATCH_TOKENS
        if replay
        else config["micro_batch_sequences"] * config["sequence_length"]
    )
    if not isinstance(completed, int) or not 0 < completed <= config["optimizer_updates"]:
        raise SystemExit(f"invalid resume completed-updates state: {state}")
    if state.get("processed_student_tokens") != completed * tokens_per_update:
        raise SystemExit(f"invalid resume processed-token state: {state}")
    optimizer_state_report(checkpoint["optimizer"], completed)
    student.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_loader_group(loaders, checkpoint["dataloader_states"], symbols, replay)
    restore_rng_state(checkpoint["rng_state"])
    expected_next = next_update_hash(loaders, symbols, replay)
    if expected_next != checkpoint["next_global_batch_sha256"]:
        raise SystemExit("resume checkpoint next-batch hash mismatch")
    audit = {
        "integrity": integrity,
        "completed_updates": completed,
        "processed_student_tokens": state["processed_student_tokens"],
        "next_global_batch_sha256": expected_next,
        "model_strict_reload": nested_equal(student.state_dict(), checkpoint["model"]),
        "optimizer_exact_reload": nested_equal(optimizer.state_dict(), checkpoint["optimizer"]),
        "loader_exact_reload": nested_equal(snapshot_loaders(loaders), checkpoint["dataloader_states"]),
        "rng_exact_reload": nested_equal(capture_rng_state(), checkpoint["rng_state"]),
    }
    audit["passed"] = all(
        audit[key]
        for key in (
            "model_strict_reload",
            "optimizer_exact_reload",
            "loader_exact_reload",
            "rng_exact_reload",
        )
    )
    if not audit["passed"]:
        raise SystemExit(f"resume state audit failed: {audit}")
    return state, audit


def initial_training_loaders(symbols, config, replay, parent_aux):
    if replay:
        return make_replay_loaders(symbols, parent_aux["dataloader_states"])
    return [
        symbols["DataLoaderLite"](
            B=config["micro_batch_sequences"],
            T=config["sequence_length"],
            process_rank=0,
            num_processes=1,
            split="train",
        )
    ]


def force_checkpoint_restart(
    checkpoint_path,
    student,
    optimizer,
    loaders,
    symbols,
    replay,
    parent_aux,
    metadata,
    device,
):
    old_model = student.state_dict()
    old_optimizer = optimizer.state_dict()
    old_loaders = snapshot_loaders(loaders)
    old_rng = capture_rng_state()
    config = metadata["config"]

    with torch.random.fork_rng(devices=[]):
        restarted_student = symbols["GPT"](model_config(symbols, enable_topdown=True))
    restarted_student.freeze_for_topdown_training()
    restarted_student.to(device)
    restarted_optimizer = feedback_optimizer(restarted_student)
    restarted_loaders = initial_training_loaders(symbols, config, replay, parent_aux)
    state, load_audit = load_exp2_resume(
        checkpoint_path,
        restarted_student,
        restarted_optimizer,
        restarted_loaders,
        symbols,
        replay,
        parent_aux,
        metadata,
    )
    boundary_audit = {
        "model_exact_at_restart_boundary": nested_equal(
            restarted_student.state_dict(), old_model
        ),
        "optimizer_exact_at_restart_boundary": nested_equal(
            restarted_optimizer.state_dict(), old_optimizer
        ),
        "loader_exact_at_restart_boundary": nested_equal(
            snapshot_loaders(restarted_loaders), old_loaders
        ),
        "rng_exact_at_restart_boundary": nested_equal(capture_rng_state(), old_rng),
        "load_audit": load_audit,
    }
    boundary_audit["passed"] = (
        all(
            boundary_audit[key]
            for key in (
                "model_exact_at_restart_boundary",
                "optimizer_exact_at_restart_boundary",
                "loader_exact_at_restart_boundary",
                "rng_exact_at_restart_boundary",
            )
        )
        and load_audit["passed"]
    )
    if not boundary_audit["passed"]:
        raise SystemExit(f"forced checkpoint restart failed: {boundary_audit}")
    return restarted_student, restarted_optimizer, restarted_loaders, state, boundary_audit


def run_learning(args, device):
    if not args.allow_optimizer_steps:
        raise SystemExit("optimizer steps are locked; explicit --allow-optimizer-steps is required")
    require_clean_training_tree()
    config_path = REPO_ROOT / "configs" / "exp2a0_5m.json"
    config = validate_config(json.loads(config_path.read_text()), "learn-5m")
    symbols, teacher, student, parent_aux = load_models(args.checkpoint, device, include_teacher=True)
    optimizer = feedback_optimizer(student)
    loaders = initial_training_loaders(symbols, config, True, parent_aux)
    initial_next = next_update_hash(loaders, symbols, replay=True)
    if initial_next != EXPECTED_NEXT_GLOBAL_BATCH_SHA256:
        raise SystemExit(
            f"parent data continuation mismatch: expected {EXPECTED_NEXT_GLOBAL_BATCH_SHA256}, got {initial_next}"
        )
    metadata = training_metadata("5M-token single-GPU learning test", config, parent_aux)
    frozen_before = state_tensor_sha256(student, include_topdown=False)
    resume_audit = None
    if args.resume:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise SystemExit("resume run directory does not exist")
        stored_config = json.loads((run_dir / "config.json").read_text())
        stored_metadata = json.loads((run_dir / "metadata.json").read_text())
        if stored_config != config or stored_metadata != metadata:
            raise SystemExit("resume run-dir config/metadata does not match current protocol")
        dataset_verification = json.loads(
            (run_dir / "dataset_verification.json").read_text()
        )
        if (
            dataset_verification.get("manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256
            or not dataset_verification.get("shards_verified")
        ):
            raise SystemExit("resume dataset verification artifact is invalid")
        live_dataset_verification = dataset_manifest_report(verify_shards=True)
        if live_dataset_verification != dataset_verification:
            raise SystemExit("live dataset no longer matches the original resume verification")
        state, resume_audit = load_exp2_resume(
            args.resume,
            student,
            optimizer,
            loaders,
            symbols,
            True,
            parent_aux,
            metadata,
        )
        start_update = state["completed_updates"]
        metrics_reconciliation = reconcile_metrics(
            run_dir / "metrics.jsonl", start_update
        )
        write_json(run_dir / "resume_audit.json", {
            "checkpoint": str(Path(args.resume).resolve()),
            "state_audit": resume_audit,
            "metrics_reconciliation": metrics_reconciliation,
        })
    else:
        run_dir, _ = prepare_run_dir(args.run_dir, config_path, metadata)
        dataset = dataset_manifest_report(verify_shards=True)
        write_json(run_dir / "dataset_verification.json", dataset)
        write_json(
            run_dir / "data_continuation.json",
            {
                "parent_next_global_batch_sha256": parent_aux["next_global_batch_sha256"],
                "replayed_next_global_batch_sha256": initial_next,
                "match": True,
                "loader_states": snapshot_loaders(loaders),
            },
        )
        start_update = 0

    rows = []
    nonzero_seen = {"gate": False, "query": False, "rmsnorm": False}
    midpoint = config["checkpoint_after_updates"]
    if start_update < midpoint:
        first_rows, first_nonzero = train_updates(
            teacher,
            student,
            optimizer,
            loaders,
            symbols,
            replay=True,
            start_update=start_update,
            end_update=midpoint,
            tokens_per_update=GLOBAL_BATCH_TOKENS,
            metrics_path=run_dir / "metrics.jsonl",
        )
        rows.extend(first_rows)
        for name in nonzero_seen:
            nonzero_seen[name] |= first_nonzero[name]
        midpoint_state = {
            "completed_updates": midpoint,
            "processed_student_tokens": midpoint * GLOBAL_BATCH_TOKENS,
        }
        if state_tensor_sha256(student, include_topdown=False) != frozen_before:
            raise SystemExit("frozen base model changed before 5M midpoint checkpoint")
        midpoint_checkpoint_path = run_dir / "checkpoints" / "checkpoint_updates_000005.pt"
        save_exp2a0_checkpoint(
            midpoint_checkpoint_path,
            student,
            optimizer,
            loaders,
            symbols,
            True,
            midpoint_state,
            parent_aux,
            metadata,
        )
        student, optimizer, loaders, restart_state, forced_restart = force_checkpoint_restart(
            midpoint_checkpoint_path,
            student,
            optimizer,
            loaders,
            symbols,
            True,
            parent_aux,
            metadata,
            device,
        )
        if restart_state != midpoint_state:
            raise SystemExit("5M forced-restart training-state mismatch")
        write_json(run_dir / "forced_restart_audit.json", forced_restart)
        start_update = midpoint
    if start_update < config["optimizer_updates"]:
        second_rows, second_nonzero = train_updates(
            teacher,
            student,
            optimizer,
            loaders,
            symbols,
            replay=True,
            start_update=start_update,
            end_update=config["optimizer_updates"],
            tokens_per_update=GLOBAL_BATCH_TOKENS,
            metrics_path=run_dir / "metrics.jsonl",
        )
        rows.extend(second_rows)
        for name in nonzero_seen:
            nonzero_seen[name] |= second_nonzero[name]
    state = {
        "completed_updates": config["optimizer_updates"],
        "processed_student_tokens": config["processed_student_tokens"],
    }
    checkpoint_path = run_dir / "checkpoints" / "checkpoint_updates_000010.pt"
    frozen_after = state_tensor_sha256(student, include_topdown=False)
    if frozen_before != frozen_after:
        raise SystemExit("frozen base model changed before final checkpoint publication")
    if checkpoint_path.exists():
        if not args.resume or Path(args.resume).resolve() != checkpoint_path.resolve():
            raise SystemExit(f"final checkpoint already exists unexpectedly: {checkpoint_path}")
        checkpoint = verify_checkpoint_sidecars(checkpoint_path)
    else:
        checkpoint = save_exp2a0_checkpoint(
            checkpoint_path,
            student,
            optimizer,
            loaders,
            symbols,
            True,
            state,
            parent_aux,
            metadata,
        )
    controls = evaluate_controls(student, teacher, symbols, device)
    all_training_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if [row["update"] for row in all_training_rows] != list(range(config["optimizer_updates"])):
        raise SystemExit("final training metrics are incomplete or duplicated")
    staging = gradient_staging_report(all_training_rows)
    if not staging["passed"]:
        raise SystemExit(f"5M gradient staging failed: {staging}")
    summary = {
        "updates_this_invocation": len(rows),
        "completed_updates": config["optimizer_updates"],
        "processed_student_tokens": config["processed_student_tokens"],
        "trainable_parameters": 1537,
        "nonzero_gradient_seen_this_invocation": nonzero_seen,
        "gradient_staging": staging,
        "resume_audit": resume_audit,
        "frozen_model_sha256_before": frozen_before,
        "frozen_model_sha256_after": frozen_after,
        "frozen_model_bit_exact": True,
        "checkpoint": checkpoint,
        "controls": controls,
        "interpretation": (
            "HIGHER-STATE FEEDBACK SIGNAL CONFIRMED"
            if controls["signal_confirmed"]
            else "HIGHER-STATE FEEDBACK SIGNAL NOT YET CONFIRMED"
        ),
    }
    write_json(run_dir / "result_summary.json", summary)
    return summary


def evaluate_trained_checkpoint(args, device):
    require_clean_training_tree()
    run_dir = Path(args.run_dir)
    config = validate_config(
        json.loads((run_dir / "config.json").read_text()), "learn-5m"
    )
    symbols, teacher, student, parent_aux = load_models(
        args.checkpoint, device, include_teacher=True
    )
    parent_frozen_sha256 = state_tensor_sha256(student, include_topdown=False)
    dataset = dataset_manifest_report(verify_shards=True)
    expected_metadata = training_metadata(
        "5M-token single-GPU learning test", config, parent_aux
    )
    stored_metadata = json.loads((run_dir / "metadata.json").read_text())
    if stored_metadata != expected_metadata:
        raise SystemExit("trained-checkpoint evaluation metadata mismatch")
    optimizer = feedback_optimizer(student)
    loaders = initial_training_loaders(symbols, config, True, parent_aux)
    state, resume_audit = load_exp2_resume(
        args.trained_checkpoint,
        student,
        optimizer,
        loaders,
        symbols,
        True,
        parent_aux,
        expected_metadata,
    )
    if state["completed_updates"] != config["optimizer_updates"]:
        raise SystemExit("trained-checkpoint evaluation requires the final update-10 checkpoint")
    trained_frozen_sha256 = state_tensor_sha256(student, include_topdown=False)
    if trained_frozen_sha256 != parent_frozen_sha256:
        raise SystemExit("trained checkpoint changed the frozen Experiment 1 base")
    controls = evaluate_controls(student, teacher, symbols, device)
    report = {
        "trained_checkpoint": str(Path(args.trained_checkpoint).resolve()),
        "resume_audit": resume_audit,
        "dataset": dataset,
        "parent_frozen_model_sha256": parent_frozen_sha256,
        "trained_frozen_model_sha256": trained_frozen_sha256,
        "frozen_model_bit_exact": True,
        "controls": controls,
        "interpretation": (
            "HIGHER-STATE FEEDBACK SIGNAL CONFIRMED"
            if controls["signal_confirmed"]
            else "HIGHER-STATE FEEDBACK SIGNAL NOT YET CONFIRMED"
        ),
    }
    write_json(args.out, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-checkpoint")
    inspect_parser.add_argument("--checkpoint", required=True)
    inspect_parser.add_argument("--out", required=True)

    data_parser = subparsers.add_parser("data-preflight")
    data_parser.add_argument("--checkpoint", required=True)
    data_parser.add_argument("--out", required=True)

    regression_parser = subparsers.add_parser("regression")
    regression_parser.add_argument("--checkpoint", required=True)
    regression_parser.add_argument("--out", required=True)
    regression_parser.add_argument("--B", type=int, default=1)
    regression_parser.add_argument("--T", type=int, default=1024)

    frozen_parser = subparsers.add_parser("frozen-diagnostics")
    frozen_parser.add_argument("--checkpoint", required=True)
    frozen_parser.add_argument("--out", required=True)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--checkpoint", required=True)
    smoke_parser.add_argument("--run-dir", required=True)
    smoke_parser.add_argument("--allow-optimizer-steps", action="store_true")

    learn_parser = subparsers.add_parser("learn-5m")
    learn_parser.add_argument("--checkpoint", required=True)
    learn_parser.add_argument("--run-dir", required=True)
    learn_parser.add_argument("--resume")
    learn_parser.add_argument("--allow-optimizer-steps", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate-trained")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--trained-checkpoint", required=True)
    evaluate_parser.add_argument("--run-dir", required=True)
    evaluate_parser.add_argument("--out", required=True)

    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    if args.command == "inspect-checkpoint":
        report = inspect_checkpoint(args.checkpoint)
        write_json(args.out, report)
    elif args.command == "data-preflight":
        report = data_preflight(args.checkpoint, args.out)
    else:
        device = require_cuda()
        torch.set_float32_matmul_precision("high")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        if args.command == "regression":
            report = checkpoint_regression(args.checkpoint, device, args.B, args.T)
            write_json(args.out, report)
            if not report["passed"]:
                raise SystemExit("full-context checkpoint regression failed")
        elif args.command == "frozen-diagnostics":
            report = frozen_diagnostics(args.checkpoint, device, args.out)
            if not report["passed"]:
                raise SystemExit("frozen diagnostics failed")
        elif args.command == "smoke":
            report = run_smoke(args, device)
        elif args.command == "learn-5m":
            report = run_learning(args, device)
        elif args.command == "evaluate-trained":
            report = evaluate_trained_checkpoint(args, device)
        else:
            raise AssertionError(args.command)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
