#!/usr/bin/env python3
"""Exact-resume four-GPU DDP harness for Experiment 1B.

This is intentionally separate from the frozen Experiment 1A harness. It
requires the reviewed 4 x A100 batch geometry and refuses cross-world-size
resume.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import smoke_test as support
from experiment_train import routing_parameter_stats, verify_shared_initialization


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "a834ca88b7b6c4e81c2a71eef0edde29b2ee2ccb"
EXPECTED_INIT_SHA256 = "39de351efe080de4e2409355c572095f17dcbaea76154a2f55e375acfdafc3b6"
EXPECTED_DATASET_MANIFEST_SHA256 = "be14a17c21682a018aef68ce02847cced77e921374c01f806deccfba72870f54"
CHECKPOINT_SCHEMA = "exp1b_exact_resume_v1"
SEED = 1337
WORLD_SIZE = 4
B = 64
T = 1024
GRAD_ACCUM_STEPS = 2
GLOBAL_BATCH_TOKENS = B * T * WORLD_SIZE * GRAD_ACCUM_STEPS


class Runtime:
    def __init__(self):
        required = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise SystemExit(f"launch with torchrun; missing environment variables: {missing}")
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        if self.world_size != WORLD_SIZE:
            raise SystemExit(f"Experiment 1B requires world_size={WORLD_SIZE}, got {self.world_size}")
        if torch.cuda.device_count() != WORLD_SIZE:
            raise SystemExit(
                f"Experiment 1B requires exactly {WORLD_SIZE} visible GPUs, got {torch.cuda.device_count()}"
            )
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend="nccl", device_id=self.device)
        self.object_group = dist.new_group(backend="gloo")
        self.master = self.rank == 0

    def barrier(self):
        dist.barrier()

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group(self.object_group)
            dist.destroy_process_group()


def master_print(runtime, *args, **kwargs):
    if runtime.master:
        print(*args, **kwargs)


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value) + "\n")


def all_gather_objects(value, runtime):
    values = [None for _ in range(runtime.world_size)]
    dist.all_gather_object(values, value, group=runtime.object_group)
    return values


def broadcast_object(value, runtime):
    values = [value]
    dist.broadcast_object_list(values, src=0, group=runtime.object_group)
    return values[0]


def validate_config(config):
    if config.get("protocol") != "exp1b_ddp_v1":
        raise SystemExit("unreviewed Experiment 1B protocol")
    if config.get("lr_schedule") != "preserve_original_10b_recipe":
        raise SystemExit("unreviewed learning-rate protocol")
    if config["max_updates"] * GLOBAL_BATCH_TOKENS != config["actual_tokens"]:
        raise SystemExit("actual_tokens does not equal max_updates * 524,288")
    for key in ("eval_completed_updates", "hellaswag_completed_updates", "checkpoint_completed_updates"):
        values = config[key]
        if values != sorted(set(values)):
            raise SystemExit(f"{key} must be sorted and unique")
        if any(value < 0 or value > config["max_updates"] for value in values):
            raise SystemExit(f"{key} contains an out-of-range update")
    if config["validation_global_batches"] != 20:
        raise SystemExit("Experiment 1B requires exactly 20 global validation batches")
    if config["validation_global_batches"] % WORLD_SIZE:
        raise SystemExit("global validation batches must divide evenly across four ranks")
    if config["experiment"] == "exp1b_500m_seed1337":
        required = {
            "max_updates": 954,
            "actual_tokens": 500170752,
            "eval_completed_updates": [0, 191, 477, 715, 954],
            "hellaswag_completed_updates": [0, 191, 954],
            "checkpoint_completed_updates": [191, 477, 715, 954],
        }
        for key, expected in required.items():
            if config[key] != expected:
                raise SystemExit(f"500M config {key} must be {expected}, got {config[key]}")


def seed_rank(rank):
    seed = SEED + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state(runtime):
    return {
        "rank": runtime.rank,
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(runtime.device),
    }


def restore_rng_state(state, runtime):
    if state["rank"] != runtime.rank:
        raise SystemExit(f"RNG rank mismatch: checkpoint={state['rank']} runtime={runtime.rank}")
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], runtime.device)


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
        if state[key] != value:
            raise SystemExit(f"DataLoader {key} mismatch: checkpoint={state[key]} runtime={value}")
    shard = int(state["current_shard"])
    if shard < 0 or shard >= len(loader.shards):
        raise SystemExit(f"invalid DataLoader shard index: {shard}")
    if loader.shards[shard] != state["current_shard_path"]:
        raise SystemExit(
            f"DataLoader shard identity mismatch: {loader.shards[shard]} != {state['current_shard_path']}"
        )
    loader.current_shard = shard
    loader.tokens = symbols["load_tokens"](loader.shards[shard])
    loader.current_position = int(state["current_position"])
    if loader.current_position < 0 or loader.current_position + B * T + 1 > len(loader.tokens):
        raise SystemExit(f"invalid DataLoader position: {loader.current_position}")


def batch_payload_hash(x, y):
    payload = x.contiguous().numpy().tobytes() + y.contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def combine_global_batch_hashes(per_rank_hashes):
    digest = hashlib.sha256()
    for microstep in range(GRAD_ACCUM_STEPS):
        for rank_hashes in per_rank_hashes:
            digest.update(bytes.fromhex(rank_hashes[microstep]))
    return digest.hexdigest()


def probe_global_batches(loader, symbols, runtime, count):
    snapshot = loader_state(loader)
    local_batch_hashes = []
    local_batch_edges = []
    for _ in range(count):
        local_hashes = []
        local_edges = []
        for _microstep in range(GRAD_ACCUM_STEPS):
            x, y = loader.next_batch()
            local_hashes.append(batch_payload_hash(x, y))
            local_edges.append((int(x.view(-1)[0]), int(y.view(-1)[-1])))
        local_batch_hashes.append(local_hashes)
        local_batch_edges.append(local_edges)
    gathered_hashes = all_gather_objects(local_batch_hashes, runtime)
    gathered_edges = all_gather_objects(local_batch_edges, runtime)
    restore_loader_state(loader, snapshot, symbols)
    runtime.barrier()
    if not runtime.master:
        return None
    global_hashes = []
    first_tokens = []
    last_targets = []
    for batch_index in range(count):
        per_rank_hashes = [rank_hashes[batch_index] for rank_hashes in gathered_hashes]
        per_rank_edges = [rank_edges[batch_index] for rank_edges in gathered_edges]
        global_hashes.append(combine_global_batch_hashes(per_rank_hashes))
        first_tokens.append(per_rank_edges[0][0][0])
        last_targets.append(per_rank_edges[-1][-1][1])
    combined = hashlib.sha256()
    for digest in global_hashes:
        combined.update(bytes.fromhex(digest))
    return {
        "global_batches_hashed": count,
        "global_batch_hashes": global_hashes,
        "combined_sha256": combined.hexdigest(),
        "first_input_tokens": first_tokens,
        "last_target_tokens": last_targets,
    }


def finite_gradients(model):
    return all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def gradient_category_norms(model):
    squared = {
        "attnres_queries": 0.0,
        "attnres_rmsnorm": 0.0,
        "gpt2_attention": 0.0,
        "gpt2_mlp": 0.0,
    }
    tensors = {key: 0 for key in squared}
    nonzero = {key: 0 for key in squared}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if ".attnres." in name and name.endswith(".query"):
            category = "attnres_queries"
        elif ".attnres." in name and name.endswith(".norm.weight"):
            category = "attnres_rmsnorm"
        elif ".attn." in name:
            category = "gpt2_attention"
        elif ".mlp." in name:
            category = "gpt2_mlp"
        else:
            continue
        grad = parameter.grad.detach().float()
        squared[category] += grad.square().sum().item()
        tensors[category] += 1
        nonzero[category] += int(torch.count_nonzero(grad).item() > 0)
    return {
        key: {
            "l2_norm": value ** 0.5,
            "tensors_with_grad": tensors[key],
            "tensors_with_nonzero_grad": nonzero[key],
        }
        for key, value in squared.items()
    }


def memory_stats(runtime):
    local = torch.tensor([
        torch.cuda.memory_allocated(runtime.device) / 1024**2,
        torch.cuda.memory_reserved(runtime.device) / 1024**2,
        torch.cuda.max_memory_allocated(runtime.device) / 1024**2,
        torch.cuda.max_memory_reserved(runtime.device) / 1024**2,
    ], device=runtime.device, dtype=torch.float64)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    values = local.cpu().tolist()
    return {
        "allocated_mb_per_gpu_max": values[0],
        "reserved_mb_per_gpu_max": values[1],
        "peak_allocated_mb_per_gpu_max": values[2],
        "peak_reserved_mb_per_gpu_max": values[3],
    }


def timed_optimizer_update(model, raw_model, optimizer, loader, symbols, runtime, step, hash_batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(runtime.device)
    start = time.perf_counter()
    local_loss = torch.zeros((), device=runtime.device, dtype=torch.float64)
    local_hashes = []
    for microstep in range(GRAD_ACCUM_STEPS):
        x, y = loader.next_batch()
        if hash_batch:
            local_hashes.append(batch_payload_hash(x, y))
        x = x.to(runtime.device, non_blocking=True)
        y = y.to(runtime.device, non_blocking=True)
        sync_context = model.no_sync() if microstep < GRAD_ACCUM_STEPS - 1 else contextlib.nullcontext()
        with sync_context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            local_loss += loss.detach().double() / GRAD_ACCUM_STEPS
            (loss / GRAD_ACCUM_STEPS).backward()

    finite = torch.tensor(int(finite_gradients(raw_model)), device=runtime.device)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if finite.item() != 1:
        raise SystemExit("non-finite gradient detected")
    categories = gradient_category_norms(raw_model) if runtime.master else None
    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    lr = support.get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize(runtime.device)
    local_seconds = time.perf_counter() - start
    max_seconds = torch.tensor(local_seconds, device=runtime.device, dtype=torch.float64)
    dist.all_reduce(max_seconds, op=dist.ReduceOp.MAX)
    loss_value = local_loss.clone()
    dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
    loss_value /= runtime.world_size
    grad_value = grad_norm.detach().double()
    dist.all_reduce(grad_value, op=dist.ReduceOp.MAX)
    global_hash = None
    if hash_batch:
        gathered = all_gather_objects(local_hashes, runtime)
        if runtime.master:
            global_hash = combine_global_batch_hashes(gathered)
    memory = memory_stats(runtime)
    if not runtime.master:
        return None
    seconds = max_seconds.item()
    return {
        "kind": "train",
        "step": step,
        "completed_updates": step + 1,
        "tokens": (step + 1) * GLOBAL_BATCH_TOKENS,
        "train_loss": loss_value.item(),
        "lr": lr,
        "grad_norm": grad_value.item(),
        "gradient_categories": categories,
        "step_time_ms": seconds * 1000,
        "tokens_per_second": GLOBAL_BATCH_TOKENS / seconds,
        "global_batch_sha256": global_hash,
        **memory,
    }


@torch.no_grad()
def validation_loss_ddp(raw_model, val_loader, runtime, global_batches, collect_routing):
    if global_batches % runtime.world_size:
        raise SystemExit("validation global batch count must divide by world_size")
    raw_model.eval()
    val_loader.reset()
    local_steps = global_batches // runtime.world_size
    local_loss = torch.zeros((), device=runtime.device, dtype=torch.float64)
    routing_sums = None
    if collect_routing:
        raw_model.set_attnres_instrumentation(True)
    for _ in range(local_steps):
        x, y = val_loader.next_batch()
        x = x.to(runtime.device, non_blocking=True)
        y = y.to(runtime.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = raw_model(x, y)
        local_loss += loss.detach().double()
        if collect_routing:
            stats = raw_model.get_attnres_stats()
            if routing_sums is None:
                routing_sums = [
                    {
                        "destination": row["destination"],
                        "source_depths": row["source_depths"],
                        "weight_sum": torch.tensor(row["mean_weights"], device=runtime.device, dtype=torch.float64),
                        "entropy_sum": torch.tensor(row["mean_entropy"], device=runtime.device, dtype=torch.float64),
                    }
                    for row in stats
                ]
            else:
                for accumulator, row in zip(routing_sums, stats):
                    accumulator["weight_sum"] += torch.tensor(
                        row["mean_weights"], device=runtime.device, dtype=torch.float64
                    )
                    accumulator["entropy_sum"] += row["mean_entropy"]
    if collect_routing:
        raw_model.set_attnres_instrumentation(False)
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    loss_value = (local_loss / global_batches).item()
    routing = []
    if collect_routing:
        for accumulator in routing_sums:
            dist.all_reduce(accumulator["weight_sum"], op=dist.ReduceOp.SUM)
            dist.all_reduce(accumulator["entropy_sum"], op=dist.ReduceOp.SUM)
            routing.append({
                "destination": accumulator["destination"],
                "source_depths": accumulator["source_depths"],
                "mean_weights": (accumulator["weight_sum"] / global_batches).cpu().tolist(),
                "mean_entropy": (accumulator["entropy_sum"] / global_batches).item(),
            })
    val_loader.reset()
    runtime.barrier()
    return loss_value, routing


@torch.no_grad()
def evaluate_hellaswag_ddp(raw_model, symbols, runtime, expected_examples):
    raw_model.eval()
    local_total = 0
    local_correct = 0
    for index, example in enumerate(symbols["iterate_examples"]("val")):
        if index % runtime.world_size != runtime.rank:
            continue
        _, tokens, mask, label = symbols["render_example"](example)
        tokens = tokens.to(runtime.device)
        mask = mask.to(runtime.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = raw_model(tokens)
        prediction = symbols["get_most_likely_row"](tokens, mask, logits)
        local_total += 1
        local_correct += int(prediction == label)
    counts = torch.tensor([local_total, local_correct], device=runtime.device, dtype=torch.long)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    total, correct = counts.cpu().tolist()
    if total != expected_examples:
        raise SystemExit(f"incomplete HellaSwag evaluation: expected {expected_examples}, got {total}")
    return {"examples": total, "correct_norm": correct, "accuracy_norm": correct / total}


def optimizer_verification(optimizer_state):
    states = optimizer_state.get("state", {})
    moment_entries = 0
    nonzero_first = 0
    nonzero_second = 0
    finite = True
    for state in states.values():
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            continue
        moment_entries += 1
        first = state["exp_avg"]
        second = state["exp_avg_sq"]
        nonzero_first += int(torch.count_nonzero(first).item() > 0)
        nonzero_second += int(torch.count_nonzero(second).item() > 0)
        finite = finite and bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all())
    return {
        "optimizer_state_entries": len(states),
        "moment_entries": moment_entries,
        "nonzero_first_moment_tensors": nonzero_first,
        "nonzero_second_moment_tensors": nonzero_second,
        "all_moments_finite": finite,
        "valid": bool(states) and moment_entries == len(states) and nonzero_first > 0 and nonzero_second > 0 and finite,
    }


def checkpoint_metadata(args, config, dataset_report, environment_report, runtime):
    gpu_names = all_gather_objects(torch.cuda.get_device_name(runtime.device), runtime)
    return {
        "git_sha": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "baseline_init_sha256": EXPECTED_INIT_SHA256,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "dataset_report_sha256": file_sha256(args.dataset_report),
        "dataset_total_tokens": dataset_report["total_token_count"],
        "dataset_dtype": dataset_report["dtype"],
        "gpu_names": gpu_names,
        "world_size": WORLD_SIZE,
        "B_per_gpu": B,
        "T": T,
        "gradient_accumulation": GRAD_ACCUM_STEPS,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "precision": "BF16 autocast; FP32 parameters/residual accumulator",
        "optimizer": {
            "type": "AdamW",
            "fused": True,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
        "lr_schedule": {
            "max_lr": support.MAX_LR,
            "min_lr": support.MIN_LR,
            "warmup_updates": support.WARMUP_STEPS,
            "max_schedule_horizon": support.MAX_STEPS,
        },
        "seed": SEED,
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "environment_report_sha256": file_sha256(args.environment_report),
        "experiment": config["experiment"],
    }


def verify_checkpoint_file(path, expected_completed_updates):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema", "model", "model_config", "residual_mode", "optimizer", "training_state",
        "dataloader_states", "rng_states", "metadata", "next_global_batch_sha256",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise SystemExit(f"checkpoint missing required fields: {missing}")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise SystemExit(f"checkpoint schema mismatch: {checkpoint['schema']}")
    state = checkpoint["training_state"]
    if state["completed_updates"] != expected_completed_updates:
        raise SystemExit(
            f"checkpoint completed_updates mismatch: {state['completed_updates']} != {expected_completed_updates}"
        )
    if state["next_update"] != expected_completed_updates:
        raise SystemExit("checkpoint next_update is not the completed-update count")
    if state["processed_tokens"] != expected_completed_updates * GLOBAL_BATCH_TOKENS:
        raise SystemExit("checkpoint processed-token count is invalid")
    if len(checkpoint["dataloader_states"]) != WORLD_SIZE or len(checkpoint["rng_states"]) != WORLD_SIZE:
        raise SystemExit("checkpoint does not contain four per-rank data/RNG states")
    if checkpoint["metadata"].get("world_size") != WORLD_SIZE:
        raise SystemExit("checkpoint metadata world size is invalid")
    if [state.get("process_rank") for state in checkpoint["dataloader_states"]] != list(range(WORLD_SIZE)):
        raise SystemExit("checkpoint DataLoader states do not cover ranks 0..3 in order")
    if [state.get("rank") for state in checkpoint["rng_states"]] != list(range(WORLD_SIZE)):
        raise SystemExit("checkpoint RNG states do not cover ranks 0..3 in order")
    if state["scheduler_position"] != expected_completed_updates:
        raise SystemExit("checkpoint LR scheduler position is invalid")
    if state["last_lr"] != support.get_lr(expected_completed_updates - 1):
        raise SystemExit("checkpoint last LR is invalid")
    if state["next_lr"] != support.get_lr(expected_completed_updates):
        raise SystemExit("checkpoint next LR is invalid")
    optimizer_report = optimizer_verification(checkpoint["optimizer"])
    if not optimizer_report["valid"]:
        raise SystemExit(f"checkpoint optimizer verification failed: {optimizer_report}")
    report = {
        "path": str(Path(path).resolve()),
        "bytes": Path(path).stat().st_size,
        "sha256": file_sha256(path),
        "completed_updates": expected_completed_updates,
        "processed_tokens": state["processed_tokens"],
        "next_global_batch_sha256": checkpoint["next_global_batch_sha256"],
        "optimizer": optimizer_report,
        "dataloader_rank_states": len(checkpoint["dataloader_states"]),
        "rng_rank_states": len(checkpoint["rng_states"]),
    }
    del checkpoint
    return report


def save_checkpoint(
    path, raw_model, optimizer, train_loader, symbols, runtime, metadata, completed_updates,
    residual_mode, completed_evaluations, completed_hellaswag,
):
    runtime.barrier()
    data_states = all_gather_objects(loader_state(train_loader), runtime)
    rng_states = all_gather_objects(capture_rng_state(runtime), runtime)
    next_probe = probe_global_batches(train_loader, symbols, runtime, 1)
    next_hash = next_probe["global_batch_hashes"][0] if runtime.master else None
    if runtime.master:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "model": raw_model.state_dict(),
            "model_config": vars(raw_model.config),
            "residual_mode": residual_mode,
            "optimizer": optimizer.state_dict(),
            "training_state": {
                "completed_updates": completed_updates,
                "next_update": completed_updates,
                "last_zero_based_update": completed_updates - 1,
                "processed_tokens": completed_updates * GLOBAL_BATCH_TOKENS,
                "last_lr": support.get_lr(completed_updates - 1),
                "next_lr": support.get_lr(completed_updates),
                "scheduler_position": completed_updates,
                "completed_evaluations": sorted(completed_evaluations),
                "completed_hellaswag": sorted(completed_hellaswag),
            },
            "dataloader_states": data_states,
            "rng_states": rng_states,
            "metadata": metadata,
            "next_global_batch_sha256": next_hash,
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)
    runtime.barrier()
    report = verify_checkpoint_file(path, completed_updates) if runtime.master else None
    if runtime.master:
        write_json(Path(path).with_suffix(".verification.json"), report)
        Path(str(path) + ".sha256").write_text(f"{report['sha256']}  {Path(path).name}\n")
    runtime.barrier()
    return report


def validate_resume_checkpoint(checkpoint, metadata, residual_mode):
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("resume checkpoint schema mismatch")
    if checkpoint.get("residual_mode") != residual_mode:
        raise SystemExit("resume checkpoint residual mode mismatch")
    for key in (
        "git_sha", "baseline_init_sha256", "dataset_manifest_sha256", "world_size", "B_per_gpu",
        "T", "gradient_accumulation", "global_batch_tokens", "precision", "optimizer", "lr_schedule",
        "seed", "pytorch", "cuda",
        "determinism",
    ):
        if checkpoint["metadata"].get(key) != metadata.get(key):
            raise SystemExit(
                f"resume metadata mismatch for {key}: checkpoint={checkpoint['metadata'].get(key)!r} "
                f"runtime={metadata.get(key)!r}"
            )


def read_metrics(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--residual-mode", choices=("standard", "full_attnres"), required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-report", required=True)
    parser.add_argument("--environment-report", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--expected-data-order")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--stop-after-completed-updates", type=int)
    args = parser.parse_args()

    runtime = Runtime()
    try:
        config = json.loads(Path(args.config).read_text())
        validate_config(config)
        if GLOBAL_BATCH_TOKENS != 524288:
            raise SystemExit(f"global batch mismatch: {GLOBAL_BATCH_TOKENS}")
        if git_output("branch", "--show-current") != "experiment-1-full-attnres":
            raise SystemExit("Experiment 1B must run on experiment-1-full-attnres")
        if git_output("status", "--porcelain"):
            raise SystemExit("working tree must be clean before Experiment 1B")
        if git_output("rev-parse", "baseline-gpt2-124m-10b^{commit}") != BASELINE_COMMIT:
            raise SystemExit("frozen baseline tag mismatch")
        observed_init_sha256 = broadcast_object(
            file_sha256(args.init_checkpoint) if runtime.master else None, runtime
        )
        if observed_init_sha256 != EXPECTED_INIT_SHA256:
            raise SystemExit("canonical initialization SHA256 mismatch")
        observed_manifest_sha256 = broadcast_object(
            file_sha256(args.dataset_manifest) if runtime.master else None, runtime
        )
        if observed_manifest_sha256 != EXPECTED_DATASET_MANIFEST_SHA256:
            raise SystemExit("dataset manifest SHA256 mismatch")

        dataset_report = json.loads(Path(args.dataset_report).read_text())
        environment_report = json.loads(Path(args.environment_report).read_text())
        if dataset_report.get("failures") or environment_report.get("failures"):
            raise SystemExit("dataset or environment report contains failures")
        if dataset_report["number_of_shard_files"] != 100:
            raise SystemExit("dataset must contain exactly 100 shards")
        if dataset_report["total_token_count"] != 9953989344 or dataset_report["dtype"] != ["uint16"]:
            raise SystemExit("dataset token count or dtype mismatch")
        validation_tokens = sum(
            shard["tokens"] for shard in dataset_report["shards"]
            if "val" in shard["filename"]
        )
        training_tokens = sum(
            shard["tokens"] for shard in dataset_report["shards"]
            if "train" in shard["filename"]
        )
        if validation_tokens != 100000000 or training_tokens != 9853989344:
            raise SystemExit(
                "dataset split counts mismatch: "
                f"validation={validation_tokens}, training={training_tokens}"
            )
        torch_report = environment_report.get("torch", {})
        if (
            torch_report.get("gpu_count") != WORLD_SIZE
            or not torch_report.get("bf16_supported")
            or [gpu.get("name") for gpu in torch_report.get("gpus", [])]
            != ["NVIDIA A100-SXM4-80GB"] * WORLD_SIZE
        ):
            raise SystemExit("environment report is not the verified 4 x A100-SXM4-80GB environment")

        seed_rank(runtime.rank)
        torch.set_float32_matmul_precision("high")
        symbols = support.load_training_symbols()
        symbols["master_process"] = runtime.master
        model_config = symbols["GPTConfig"](vocab_size=50304, residual_mode=args.residual_mode)
        raw_model = symbols["GPT"](model_config)
        init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if args.residual_mode == "standard":
            raw_model.load_state_dict(init_checkpoint["model"], strict=True)
        else:
            raw_model.load_shared_baseline_state(init_checkpoint["model"])
        initialization = verify_shared_initialization(raw_model, init_checkpoint["model"])
        raw_model.to(runtime.device)
        optimizer = raw_model.configure_optimizers(
            weight_decay=0.1, learning_rate=support.MAX_LR, device_type="cuda"
        )
        model = DDP(
            raw_model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

        DataLoaderLite = symbols["DataLoaderLite"]
        train_loader = DataLoaderLite(B=B, T=T, process_rank=runtime.rank, num_processes=WORLD_SIZE, split="train")
        val_loader = DataLoaderLite(B=B, T=T, process_rank=runtime.rank, num_processes=WORLD_SIZE, split="val")
        metadata = checkpoint_metadata(args, config, dataset_report, environment_report, runtime)
        metadata.update({
            "residual_mode": args.residual_mode,
            "model_config": vars(model_config),
            "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
        })

        run_dir = Path(args.run_dir)
        checkpoint_dir = run_dir / "checkpoints"
        metrics_file = run_dir / "metrics.jsonl"
        attnres_file = run_dir / "attnres_stats.jsonl"
        start_update = 0
        completed_evaluations = set()
        completed_hellaswag = set()

        if args.resume_checkpoint:
            if not run_dir.is_dir():
                raise SystemExit("resume run directory does not exist")
            try:
                checkpoint = torch.load(
                    args.resume_checkpoint, map_location="cpu", weights_only=False, mmap=True
                )
            except (TypeError, RuntimeError):
                checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
            validate_resume_checkpoint(checkpoint, metadata, args.residual_mode)
            raw_model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_update = checkpoint["training_state"]["next_update"]
            if start_update <= 0:
                raise SystemExit("resume checkpoint must contain at least one completed update")
            restore_loader_state(train_loader, checkpoint["dataloader_states"][runtime.rank], symbols)
            completed_evaluations = set(checkpoint["training_state"]["completed_evaluations"])
            completed_hellaswag = set(checkpoint["training_state"]["completed_hellaswag"])
            expected_next_hash = checkpoint["next_global_batch_sha256"]
            observed_next = probe_global_batches(train_loader, symbols, runtime, 1)
            if runtime.master and observed_next["global_batch_hashes"][0] != expected_next_hash:
                raise SystemExit("resume next-global-batch hash mismatch")
            local_optimizer_audit = optimizer_verification(optimizer.state_dict())
            optimizer_audits = all_gather_objects(local_optimizer_audit, runtime)
            if not all(audit["valid"] for audit in optimizer_audits):
                raise SystemExit(f"restored optimizer moments are invalid: {optimizer_audits}")
            expected_loaded_lr = support.get_lr(start_update - 1)
            local_loaded_lrs = [group["lr"] for group in optimizer.param_groups]
            loaded_lrs = all_gather_objects(local_loaded_lrs, runtime)
            if any(any(lr != expected_loaded_lr for lr in rank_lrs) for rank_lrs in loaded_lrs):
                raise SystemExit(
                    f"restored optimizer LR mismatch: expected={expected_loaded_lr}, got={loaded_lrs}"
                )
            if runtime.master:
                write_json(run_dir / f"resume_audit_at_{start_update:04d}.json", {
                    "checkpoint": str(Path(args.resume_checkpoint).resolve()),
                    "completed_updates": start_update,
                    "processed_tokens": start_update * GLOBAL_BATCH_TOKENS,
                    "expected_loaded_lr": expected_loaded_lr,
                    "next_lr": support.get_lr(start_update),
                    "loaded_lrs_by_rank": loaded_lrs,
                    "optimizer_by_rank": optimizer_audits,
                    "expected_next_global_batch_sha256": expected_next_hash,
                    "observed_next_global_batch_sha256": observed_next["global_batch_hashes"][0],
                    "world_size": runtime.world_size,
                    "gradient_accumulation": GRAD_ACCUM_STEPS,
                    "global_batch_tokens": GLOBAL_BATCH_TOKENS,
                })
            restore_rng_state(checkpoint["rng_states"][runtime.rank], runtime)
            del checkpoint
            runtime.barrier()
        else:
            if run_dir.exists():
                raise SystemExit(f"refusing to overwrite existing run: {run_dir}")
            if runtime.master:
                run_dir.mkdir(parents=True)
                checkpoint_dir.mkdir()
                metrics_file.write_text("")
                if args.residual_mode == "full_attnres":
                    attnres_file.write_text("")
                shutil.copy2(args.dataset_report, run_dir / "dataset.json")
                shutil.copy2(args.environment_report, run_dir / "environment.json")
                write_json(run_dir / "initialization_verification.json", initialization)
                write_json(run_dir / "metadata.json", metadata)
            runtime.barrier()
            master_print(runtime, "probing matched global training-data order", flush=True)
            data_order = probe_global_batches(
                train_loader, symbols, runtime, config["data_probe_global_batches"]
            )
            rank_loader_states = all_gather_objects(loader_state(train_loader), runtime)
            if runtime.master:
                data_order["rank_loader_states"] = rank_loader_states
                if args.expected_data_order:
                    expected = json.loads(Path(args.expected_data_order).read_text())
                    if data_order["global_batch_hashes"] != expected["global_batch_hashes"]:
                        raise SystemExit("Standard/AttnRes global training-data hashes differ")
                write_json(run_dir / "data_order.json", data_order)
            runtime.barrier()
            master_print(runtime, "training-data order probe complete", flush=True)

        end_update = config["max_updates"]
        if args.stop_after_completed_updates is not None:
            end_update = args.stop_after_completed_updates
        if end_update <= start_update or end_update > config["max_updates"]:
            raise SystemExit(
                f"invalid stop point: start={start_update}, stop={end_update}, max={config['max_updates']}"
            )

        run_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(runtime.device)
        master_print(
            runtime,
            f"starting optimizer updates {start_update}..{end_update - 1} on world_size={WORLD_SIZE}",
            flush=True,
        )

        def evaluate_milestone(completed_updates):
            tokens = completed_updates * GLOBAL_BATCH_TOKENS
            collect_routing = args.residual_mode == "full_attnres"
            val_loss, routing = validation_loss_ddp(
                raw_model, val_loader, runtime, config["validation_global_batches"], collect_routing
            )
            routing_params = routing_parameter_stats(raw_model) if runtime.master else None
            if runtime.master and collect_routing:
                write_jsonl(attnres_file, {
                    "step": completed_updates - 1,
                    "completed_updates": completed_updates,
                    "tokens": tokens,
                    "destinations": routing,
                    "parameters": routing_params,
                })
            hella = None
            if completed_updates in config["hellaswag_completed_updates"]:
                hella = evaluate_hellaswag_ddp(
                    raw_model, symbols, runtime, config["expected_hellaswag_examples"]
                )
                completed_hellaswag.add(completed_updates)
            if runtime.master:
                write_jsonl(metrics_file, {
                    "kind": "val",
                    "step": completed_updates - 1,
                    "completed_updates": completed_updates,
                    "tokens": tokens,
                    "val_loss": val_loss,
                    "validation_global_batches": config["validation_global_batches"],
                    "wall_clock_seconds": time.perf_counter() - run_start,
                })
                if hella is not None:
                    write_jsonl(metrics_file, {
                        "kind": "hellaswag",
                        "step": completed_updates - 1,
                        "completed_updates": completed_updates,
                        "tokens": tokens,
                        "hellaswag_accuracy": hella["accuracy_norm"],
                        "hellaswag_correct": hella["correct_norm"],
                        "hellaswag_examples": hella["examples"],
                        "wall_clock_seconds": time.perf_counter() - run_start,
                    })
                master_print(
                    runtime,
                    f"evaluation updates={completed_updates} tokens={tokens:,} val={val_loss:.9f} "
                    + ("" if hella is None else f"HellaSwag={hella['accuracy_norm']:.6f}"),
                    flush=True,
                )
            completed_evaluations.add(completed_updates)
            runtime.barrier()

        if start_update == 0 and 0 in config["eval_completed_updates"]:
            evaluate_milestone(0)

        for step in range(start_update, end_update):
            row = timed_optimizer_update(
                model, raw_model, optimizer, train_loader, symbols, runtime, step,
                hash_batch=step < config["data_probe_global_batches"],
            )
            if runtime.master:
                row["wall_clock_seconds"] = time.perf_counter() - run_start
                write_jsonl(metrics_file, row)
                print(
                    f"step {step:4d} | updates {step + 1:4d} | tokens {row['tokens']:,} | "
                    f"loss {row['train_loss']:.6f} | {row['tokens_per_second']:.1f} tok/s",
                    flush=True,
                )
            completed = step + 1
            if completed in config["eval_completed_updates"] and completed not in completed_evaluations:
                evaluate_milestone(completed)
            if completed in config["checkpoint_completed_updates"]:
                checkpoint_path = checkpoint_dir / f"checkpoint_tokens_{completed * GLOBAL_BATCH_TOKENS:012d}.pt"
                report = save_checkpoint(
                    checkpoint_path, raw_model, optimizer, train_loader, symbols, runtime, metadata,
                    completed, args.residual_mode, completed_evaluations, completed_hellaswag,
                )
                if runtime.master:
                    print(
                        f"verified resumable checkpoint {checkpoint_path.name} "
                        f"sha256={report['sha256']}",
                        flush=True,
                    )

        runtime.barrier()
        if runtime.master:
            rows = read_metrics(metrics_file)
            train_rows = [row for row in rows if row["kind"] == "train"]
            val_rows = [row for row in rows if row["kind"] == "val"]
            hella_rows = [row for row in rows if row["kind"] == "hellaswag"]
            expected_steps = list(range(end_update))
            actual_steps = [row["step"] for row in train_rows]
            if actual_steps != expected_steps:
                raise SystemExit(
                    f"metrics do not contain a contiguous training trajectory: "
                    f"expected {expected_steps[:3]}...{expected_steps[-3:]}, got {actual_steps[:3]}...{actual_steps[-3:]}"
                )
            memory = {
                "peak_allocated_mb_per_gpu_max": max(
                    row["peak_allocated_mb_per_gpu_max"] for row in train_rows
                ),
                "peak_reserved_mb_per_gpu_max": max(
                    row["peak_reserved_mb_per_gpu_max"] for row in train_rows
                ),
            }
            checkpoint_reports = [
                json.loads(path.read_text())
                for path in sorted(checkpoint_dir.glob("*.verification.json"))
            ]
            summary = {
                "residual_mode": args.residual_mode,
                "completed_updates": end_update,
                "last_zero_based_update": end_update - 1,
                "processed_tokens": end_update * GLOBAL_BATCH_TOKENS,
                "final_train_loss": train_rows[-1]["train_loss"],
                "final_gradient_norm": train_rows[-1]["grad_norm"],
                "final_validation_loss": val_rows[-1]["val_loss"] if val_rows else None,
                "final_hellaswag": hella_rows[-1] if hella_rows else None,
                "session_wall_clock_seconds": time.perf_counter() - run_start,
                "training_update_seconds": sum(row["step_time_ms"] for row in train_rows) / 1000,
                "mean_seconds_per_update": statistics.mean(row["step_time_ms"] for row in train_rows) / 1000,
                "mean_tokens_per_second": statistics.mean(row["tokens_per_second"] for row in train_rows),
                "world_size": WORLD_SIZE,
                "B_per_gpu": B,
                "T": T,
                "gradient_accumulation": GRAD_ACCUM_STEPS,
                "global_batch_tokens": GLOBAL_BATCH_TOKENS,
                "training_data_probe_sha256": json.loads((run_dir / "data_order.json").read_text())["combined_sha256"],
                "checkpoint_verifications": checkpoint_reports,
                **memory,
            }
            write_json(run_dir / "run_summary.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        runtime.barrier()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
