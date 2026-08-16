#!/usr/bin/env python3
"""Four-GPU continuation of Experiment 2B2 memory-writer learning."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.distributed as dist
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2a0 as a0  # noqa: E402
import experiment_2b2 as b2  # noqa: E402


BRANCH = "experiment-2b2a-writers-scaling-4gpu"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2b2a_4gpu.json"
SOURCE_SHA256 = "a125c81acb9e4ec3395bd8b38dee8fade62012c642b102a1b6c4c0e0997f0637"
SOURCE_NEXT_SHA256 = "e3289bee6ed5a5b2fa1d2c05a615cd3f10f07c51b71aa091ee40380ebeedc21b"
SOURCE_RESULTS_COMMIT = "be74383a15c1834d3b56a0767586f1a991fc5dbc"
SOURCE_IMPLEMENTATION_COMMIT = "5305709f93cee736d33b93402324a7d3fed40235"
SOURCE_SCHEMA = b2.CHECKPOINT_SCHEMA
CHECKPOINT_SCHEMA = "exp2b2a_writers_4gpu_v1"
WORLD_SIZE = 4
MICROSTEPS_PER_RANK = 2
B = 64
T = 1024
GLOBAL_TARGETS = 524_288
RANK_TARGETS = 131_072
BACKWARD_CHUNK = 16
MAX_UPDATE = 48
MILESTONES = {20: "10m", 29: "15m", 48: "25m"}
SOURCE_REAL = 5.590033102035522
SOURCE_SHUFFLED = 5.625972080230713
FROZEN_2B1_REAL = 5.701308727264404
MASKED = 5.973674488067627
FULL_CONTEXT = 4.078654408454895
GRADIENT_NAMES = sorted(b2.TRAINABLE_NAMES)


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def file_sha256(path):
    return a0.file_sha256(Path(path))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise SystemExit(f"stale incomplete artifact: {temporary}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    checks = {
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_next_global_batch_sha256": SOURCE_NEXT_SHA256,
        "world_size": WORLD_SIZE,
        "microsteps_per_rank": MICROSTEPS_PER_RANK,
        "batch_sequences": B,
        "sequence_length": T,
        "rank_targets_per_update": RANK_TARGETS,
        "global_targets_per_update": GLOBAL_TARGETS,
        "backward_chunk_tokens": BACKWARD_CHUNK,
        "trainable_parameters": b2.WRITER_PARAMETER_COUNT,
        "maximum_writer_update": MAX_UPDATE,
    }
    for name, expected in checks.items():
        if config.get(name) != expected:
            raise SystemExit(f"config {name} mismatch: {config.get(name)} != {expected}")
    if config["hellaswag"] != "forbidden":
        raise SystemExit("HellaSwag must remain forbidden")
    return config


def require_git(clean=True):
    if git_output("branch", "--show-current") != BRANCH:
        raise SystemExit(f"requires branch {BRANCH}")
    if git_output("rev-parse", "experiment-2b2-writers-5m^{}") != SOURCE_RESULTS_COMMIT:
        raise SystemExit("frozen 2B2 tag target mismatch")
    if clean and git_output("status", "--porcelain"):
        raise SystemExit("result-bearing execution requires a clean worktree")


def init_distributed():
    if not all(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise SystemExit("this command must be launched with torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE):
        raise SystemExit(f"requires exactly {WORLD_SIZE} ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if torch.cuda.get_device_name(local_rank) != "NVIDIA A100-SXM4-80GB":
        raise SystemExit("each rank requires NVIDIA A100-SXM4-80GB")
    return rank, local_rank


def seed_rank(rank):
    seed = 2_026_082_000 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed


def capture_rank_rng(rank, local_rank):
    return {
        "rank": rank,
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(local_rank),
    }


def restore_rank_rng(state, local_rank):
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], local_rank)


def tensor_digest(rows):
    digest = hashlib.sha256()
    for name, tensor in rows:
        value = tensor.detach().float().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def writer_parameters(model):
    named = dict(model.named_parameters())
    if set(name for name, value in named.items() if value.requires_grad) != b2.TRAINABLE_NAMES:
        raise SystemExit("writer trainable-tensor contract mismatch")
    parameters = [(name, named[name]) for name in GRADIENT_NAMES]
    if sum(value.numel() for _, value in parameters) != b2.WRITER_PARAMETER_COUNT:
        raise SystemExit("writer parameter count mismatch")
    return parameters


def flatten_gradients(model):
    rows = writer_parameters(model)
    missing = [name for name, value in rows if value.grad is None]
    if missing:
        raise SystemExit(f"missing writer gradients: {missing}")
    flat = torch.cat([value.grad.detach().float().reshape(-1) for _, value in rows])
    if flat.numel() != b2.WRITER_PARAMETER_COUNT or not torch.isfinite(flat).all():
        raise SystemExit("invalid flattened writer gradient")
    return flat.contiguous()


def scatter_gradients(model, flat):
    offset = 0
    for _, parameter in writer_parameters(model):
        count = parameter.numel()
        parameter.grad = flat[offset : offset + count].view_as(parameter).to(parameter.dtype)
        offset += count
    if offset != b2.WRITER_PARAMETER_COUNT:
        raise SystemExit("gradient scatter length mismatch")


def flat_parameters(model):
    return torch.cat(
        [value.detach().float().reshape(-1) for _, value in writer_parameters(model)]
    )


def flat_optimizer_moment(model, optimizer, field):
    values = []
    for _, parameter in writer_parameters(model):
        state = optimizer.state.get(parameter)
        if state is None or field not in state:
            raise SystemExit(f"optimizer state {field} missing")
        values.append(state[field].detach().float().reshape(-1))
    return torch.cat(values)


def optimizer_hashes(model, optimizer):
    steps = []
    for _, parameter in writer_parameters(model):
        state = optimizer.state.get(parameter, {})
        steps.append(int(state["step"].item()))
    return {
        "writer": tensor_digest(
            (name, value) for name, value in writer_parameters(model)
        ),
        "adam_steps": steps,
        "adam_m1": tensor_digest(
            (name, optimizer.state[value]["exp_avg"])
            for name, value in writer_parameters(model)
        ),
        "adam_m2": tensor_digest(
            (name, optimizer.state[value]["exp_avg_sq"])
            for name, value in writer_parameters(model)
        ),
    }


def all_equal_across_ranks(payload, label):
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, payload)
    if any(value != gathered[0] for value in gathered[1:]):
        raise SystemExit(f"cross-rank {label} divergence: {gathered}")
    return gathered[0]


def load_source_checkpoint(path):
    path = Path(path).resolve()
    digest = file_sha256(path)
    if digest != SOURCE_SHA256:
        raise SystemExit(f"source checkpoint SHA mismatch: {digest}")
    checkpoint = a0.torch_load(path, mmap=True)
    if checkpoint.get("schema") != SOURCE_SCHEMA:
        raise SystemExit("source 2B2 checkpoint schema mismatch")
    state = checkpoint.get("training_state", {})
    if state.get("local_completed_updates") != 10:
        raise SystemExit(f"source writer update mismatch: {state}")
    if state.get("processed_2b2_tokens") != 5_242_880:
        raise SystemExit(f"source writer token mismatch: {state}")
    if checkpoint.get("next_global_batch_sha256") != SOURCE_NEXT_SHA256:
        raise SystemExit("source next-global-batch field mismatch")
    if len(checkpoint.get("dataloader_states", [])) != WORLD_SIZE:
        raise SystemExit("source must contain four replay-loader states")
    b2.optimizer_report(checkpoint["optimizer"], 10)
    if checkpoint.get("implementation_git_commit") != SOURCE_IMPLEMENTATION_COMMIT:
        raise SystemExit("source implementation lineage mismatch")
    if checkpoint.get("metadata", {}).get("git_commit") != SOURCE_IMPLEMENTATION_COMMIT:
        raise SystemExit("source metadata implementation lineage mismatch")
    return checkpoint, digest


def instantiate_runtime(checkpoint, device):
    symbols = a0.support.load_training_symbols()
    with torch.random.fork_rng(devices=[]):
        model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.freeze_for_memory_writer_training()
    b2.assert_runtime_contract(model)
    model.to(device)
    optimizer = b2.fresh_optimizer(model, device_type="cuda")
    optimizer.load_state_dict(checkpoint["optimizer"])
    completed = checkpoint["training_state"].get(
        "writer_updates", checkpoint["training_state"].get("local_completed_updates")
    )
    b2.optimizer_report(optimizer, completed)
    return symbols, model, optimizer, completed


def make_rank_loader(symbols, states, rank):
    loaders = a0.make_replay_loaders(symbols, states)
    return loaders[rank]


def clone_loader_hashes(loader, symbols):
    state = a0.loader_state(loader)
    hashes = []
    for _ in range(MICROSTEPS_PER_RANK):
        x, y = loader.next_batch()
        hashes.append(a0.batch_payload_hash(x, y))
    a0.restore_replay_loader(loader, state)
    if a0.loader_state(loader) != state:
        raise SystemExit("loader preview did not restore exactly")
    return hashes


def canonical_batch_hash(rank_hashes):
    if len(rank_hashes) != WORLD_SIZE:
        raise SystemExit("canonical hash requires four ranks")
    digest = hashlib.sha256()
    for microstep in range(MICROSTEPS_PER_RANK):
        for rank in range(WORLD_SIZE):
            digest.update(bytes.fromhex(rank_hashes[rank][microstep]))
    return digest.hexdigest()


def distributed_preview_hash(loader, symbols):
    local = clone_loader_hashes(loader, symbols)
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local)
    return canonical_batch_hash(gathered), gathered


def process_batches(model, batches, update_number):
    model.train()
    model.zero_grad(set_to_none=True)
    raw_loss_sum = 0.0
    target_seen = 0
    hashes = []
    routing_sum = torch.zeros(4, dtype=torch.float64)
    source_sum = torch.zeros(4, dtype=torch.float64)
    adapted_sum = torch.zeros(4, dtype=torch.float64)
    delta_sum = torch.zeros(4, dtype=torch.float64)
    ratio_max = 0.0
    entropy_sum = 0.0
    topdown_sum = 0.0
    feedback_sum = 0.0
    metric_count = 0
    states = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for x_cpu, y_cpu in batches:
        hashes.append(a0.batch_payload_hash(x_cpu, y_cpu))
        device = torch.device("cuda", torch.cuda.current_device())
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        state = model.init_recurrent_state(
            B, "masked_l1_topdown_self", device=x.device, dtype=torch.bfloat16
        )
        pending = None
        for position in range(T):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, state, diagnostics = model.forward_step(
                    x[:, position], state, use_memory_writers=True,
                    return_diagnostics=True,
                )
                token_loss = F.cross_entropy(
                    logits[:, 0], y[:, position], reduction="sum"
                )
                scaled = token_loss / GLOBAL_TARGETS
                pending = scaled if pending is None else pending + scaled
            raw_loss_sum += token_loss.detach().double().item()
            target_seen += B
            routing_sum += diagnostics["routing_weights"].detach().double().sum(
                dim=(1, 2)
            ).cpu()
            source_sum += diagnostics["source_rms"].detach().double().sum(dim=1).cpu()
            adapted_sum += diagnostics["adapted_rms"].detach().double().sum(dim=1).cpu()
            delta_rms = diagnostics["writer_delta"].detach().float().pow(2).mean(
                dim=(2, 3)
            ).sqrt()
            delta_sum += delta_rms.double().sum(dim=1).cpu()
            ratio = delta_rms / diagnostics["source_rms"].detach().float().clamp_min(1e-12)
            ratio_max = max(ratio_max, ratio.max().item())
            entropy_sum += diagnostics["routing_entropy"].detach().double().sum().item()
            topdown_sum += diagnostics["topdown_rms"].detach().double().sum().item()
            feedback_sum += diagnostics["feedback_rms"].detach().double().sum().item()
            metric_count += B
            if (position + 1) % BACKWARD_CHUNK == 0 or position + 1 == T:
                if not torch.isfinite(pending):
                    raise SystemExit(f"non-finite loss at update {update_number}")
                pending.backward()
                pending = None
        health = b2.state_health(state, T)
        if not health["passed"]:
            raise SystemExit(f"recurrent-state invariant failed: {health}")
        states.append(health)
        del x, y, state
    torch.cuda.synchronize()
    return {
        "raw_loss_sum": raw_loss_sum,
        "target_seen": target_seen,
        "batch_hashes": hashes,
        "routing_sum": routing_sum.tolist(),
        "source_sum": source_sum.tolist(),
        "adapted_sum": adapted_sum.tolist(),
        "delta_sum": delta_sum.tolist(),
        "maximum_residual_ratio": ratio_max,
        "entropy_sum": entropy_sum,
        "topdown_sum": topdown_sum,
        "feedback_sum": feedback_sum,
        "metric_count": metric_count,
        "state_health": states,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "local_wall_seconds": time.perf_counter() - started,
    }


def comparison(left, right):
    left = left.detach().double()
    right = right.detach().double()
    difference = right - left
    left_norm = left.norm().item()
    right_norm = right.norm().item()
    return {
        "left_norm": left_norm,
        "right_norm": right_norm,
        "relative_norm_difference": abs(right_norm - left_norm) / max(left_norm, 1e-30),
        "cosine_similarity": F.cosine_similarity(left, right, dim=0).item(),
        "relative_l2_difference": difference.norm().item() / max(left_norm, 1e-30),
        "maximum_absolute_difference": difference.abs().max().item(),
    }


def optimizer_temporary_vectors(model, optimizer):
    before = flat_parameters(model).detach().clone()
    pre_clip = torch.nn.utils.clip_grad_norm_(
        [value for _, value in writer_parameters(model)], 1.0
    )
    optimizer.step()
    torch.cuda.synchronize()
    after = flat_parameters(model).detach().clone()
    return {
        "pre_clip_norm": float(pre_clip),
        "update": (after - before).cpu(),
        "parameters": after.cpu(),
        "m1": flat_optimizer_moment(model, optimizer, "exp_avg").cpu(),
        "m2": flat_optimizer_moment(model, optimizer, "exp_avg_sq").cpu(),
    }


def source_audit(args):
    require_git(clean=True)
    load_config()
    checkpoint, digest = load_source_checkpoint(args.source_checkpoint)
    symbols = a0.support.load_training_symbols()
    loaders = a0.make_replay_loaders(symbols, checkpoint["dataloader_states"])
    next_hash = a0.next_update_hash(loaders, symbols, replay=True)
    if next_hash != SOURCE_NEXT_SHA256:
        raise SystemExit(f"source replay mismatch: {next_hash}")
    with torch.random.fork_rng(devices=[]):
        model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.freeze_for_memory_writer_training()
    report = {
        "stage": "2b2a_source_audit",
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": digest,
        "schema": checkpoint["schema"],
        "training_state": checkpoint["training_state"],
        "writer_parameters": sum(p.numel() for _, p in writer_parameters(model)),
        "optimizer": b2.optimizer_report(checkpoint["optimizer"], 10),
        "loader_states": checkpoint["dataloader_states"],
        "rng_fields": sorted(checkpoint["rng_state"]),
        "next_global_batch_sha256": next_hash,
        "source_2b1_checkpoint_sha256": checkpoint["source_2b1_checkpoint_sha256"],
        "writer_architecture": checkpoint["writer_architecture"],
        "implementation_git_commit": checkpoint["implementation_git_commit"],
        "result_commit": SOURCE_RESULTS_COMMIT,
        "passed": True,
    }
    write_json(Path(args.run_dir) / "SOURCE_CHECKPOINT_AUDIT.json", report)
    print(json.dumps(report, indent=2, default=str))


def migration_reference(args):
    require_git(clean=True)
    load_config()
    if torch.cuda.device_count() < 1:
        raise SystemExit("migration reference requires GPU 0")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    checkpoint, source_digest = load_source_checkpoint(args.source_checkpoint)
    symbols, model, optimizer, completed = instantiate_runtime(checkpoint, device)
    if completed != 10:
        raise SystemExit("migration reference did not restore update 10")
    loaders = a0.make_replay_loaders(symbols, copy.deepcopy(checkpoint["dataloader_states"]))
    expected = a0.next_update_hash(loaders, symbols, replay=True)
    if expected != SOURCE_NEXT_SHA256:
        raise SystemExit("migration reference next-batch mismatch")
    batches = list(a0.update_batches(loaders, replay=True))
    metrics = process_batches(model, batches, 11)
    actual = a0.aggregate_batch_hash(batches)
    if actual != expected or metrics["target_seen"] != GLOBAL_TARGETS:
        raise SystemExit("migration reference batch consumption mismatch")
    gradient = flatten_gradients(model).detach().clone()
    gradients_by_name = {}
    offset = 0
    for name, parameter in writer_parameters(model):
        count = parameter.numel()
        gradients_by_name[name] = gradient[offset : offset + count].cpu()
        offset += count
    temporary = optimizer_temporary_vectors(model, optimizer)
    artifact = {
        "source_checkpoint_sha256": source_digest,
        "source_next_global_batch_sha256": expected,
        "global_batch_sha256": actual,
        "global_loss": metrics["raw_loss_sum"] / GLOBAL_TARGETS,
        "flat_gradient": gradient.cpu(),
        "gradients_by_name": gradients_by_name,
        "temporary": temporary,
        "batch_hashes": metrics["batch_hashes"],
        "targets": metrics["target_seen"],
    }
    output = Path(args.run_dir) / "migration" / "one_gpu_reference.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite migration reference: {output}")
    torch.save(artifact, output)
    write_json(
        output.with_suffix(".json"),
        {
            "source_checkpoint_sha256": source_digest,
            "global_batch_sha256": actual,
            "global_loss": artifact["global_loss"],
            "gradient_norm": gradient.double().norm().item(),
            "targets": metrics["target_seen"],
            "artifact_sha256": file_sha256(output),
            "authoritative_checkpoint_unchanged": file_sha256(args.source_checkpoint) == SOURCE_SHA256,
            "passed": True,
        },
    )
    print(f"1GPU_REFERENCE_PASS loss={artifact['global_loss']:.10f} grad={gradient.norm().item():.10f}")


def migration_candidate(args):
    require_git(clean=True)
    load_config()
    rank, local_rank = init_distributed()
    try:
        seed = seed_rank(rank)
        device = torch.device("cuda", local_rank)
        checkpoint, source_digest = load_source_checkpoint(args.source_checkpoint)
        symbols, model, optimizer, completed = instantiate_runtime(checkpoint, device)
        if completed != 10:
            raise SystemExit("migration candidate did not restore update 10")
        loader = make_rank_loader(symbols, copy.deepcopy(checkpoint["dataloader_states"]), rank)
        expected, preview_hashes = distributed_preview_hash(loader, symbols)
        if expected != SOURCE_NEXT_SHA256:
            raise SystemExit(f"migration candidate next-batch mismatch: {expected}")
        batches = [loader.next_batch() for _ in range(MICROSTEPS_PER_RANK)]
        metrics = process_batches(model, batches, 11)
        flat = flatten_gradients(model)
        reduce_started = time.perf_counter()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        reduce_seconds = time.perf_counter() - reduce_started
        scatter_gradients(model, flat)
        local_rows = [metrics]
        gathered_metrics = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_metrics, metrics)
        actual_hash = canonical_batch_hash([row["batch_hashes"] for row in gathered_metrics])
        if actual_hash != expected:
            raise SystemExit("migration candidate consumed-batch hash mismatch")
        all_equal_across_ranks(
            tensor_digest([("gradient", flat)]), "synchronized gradient"
        )
        reference_path = Path(args.run_dir) / "migration" / "one_gpu_reference.pt"
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        if reference["global_batch_sha256"] != actual_hash:
            raise SystemExit("migration paths used different global batches")
        global_loss = sum(row["raw_loss_sum"] for row in gathered_metrics) / GLOBAL_TARGETS
        grad_metrics = comparison(reference["flat_gradient"], flat.cpu())
        per_writer = {}
        offset = 0
        for name, parameter in writer_parameters(model):
            count = parameter.numel()
            per_writer[name] = comparison(
                reference["flat_gradient"][offset : offset + count],
                flat.cpu()[offset : offset + count],
            )
            offset += count
        temporary = optimizer_temporary_vectors(model, optimizer)
        consistency = all_equal_across_ranks(
            optimizer_hashes(model, optimizer), "temporary optimizer step"
        )
        update_comparison = comparison(
            reference["temporary"]["update"], temporary["update"]
        )
        parameter_comparison = comparison(
            reference["temporary"]["parameters"], temporary["parameters"]
        )
        m1_comparison = comparison(reference["temporary"]["m1"], temporary["m1"])
        m2_comparison = comparison(reference["temporary"]["m2"], temporary["m2"])
        passed = (
            abs(global_loss - reference["global_loss"]) <= 1e-5
            and grad_metrics["cosine_similarity"] >= 0.999999
            and grad_metrics["relative_l2_difference"] <= 1e-4
            and grad_metrics["relative_norm_difference"] <= 1e-4
            and update_comparison["cosine_similarity"] >= 0.999999
            and update_comparison["relative_l2_difference"] <= 1e-4
            and source_digest == SOURCE_SHA256
            and file_sha256(args.source_checkpoint) == SOURCE_SHA256
        )
        if rank == 0:
            audit = {
                "experiment": "2B2A",
                "stage": "1GPU_to_4GPU_migration",
                "source_checkpoint_sha256": source_digest,
                "source_next_global_batch_sha256": SOURCE_NEXT_SHA256,
                "consumed_global_batch_sha256": actual_hash,
                "gpu_model": torch.cuda.get_device_name(local_rank),
                "world_size": WORLD_SIZE,
                "rank_to_loader_mapping": {str(value): value for value in range(WORLD_SIZE)},
                "microsteps_per_rank": MICROSTEPS_PER_RANK,
                "loss_scaling": "each token-loss sum / 524288; no division after SUM",
                "gradient_reduction": "one flattened FP32 NCCL all_reduce(SUM)",
                "gradient_elements": flat.numel(),
                "gradient_all_reduce_seconds": reduce_seconds,
                "rng_policy": {
                    "stochastic_training_operations": "none found",
                    "rank_seed_base": 2_026_082_000,
                    "rank_seeds": [2_026_082_000 + value for value in range(WORLD_SIZE)],
                },
                "one_gpu_loss": reference["global_loss"],
                "four_gpu_loss": global_loss,
                "absolute_loss_difference": abs(global_loss - reference["global_loss"]),
                "gradient_comparison": grad_metrics,
                "per_writer_gradient_comparison": per_writer,
                "temporary_optimizer_comparison": {
                    "parameter_update": update_comparison,
                    "parameters_after_step": parameter_comparison,
                    "m1": m1_comparison,
                    "m2": m2_comparison,
                    "cross_rank_state": consistency,
                },
                "acceptance": {
                    "loss_absolute_difference_max": 1e-5,
                    "gradient_cosine_min": 0.999999,
                    "gradient_relative_l2_max": 1e-4,
                    "gradient_norm_relative_difference_max": 1e-4,
                    "parameter_update_cosine_min": 0.999999,
                    "parameter_update_relative_l2_max": 1e-4,
                },
                "temporary_states_disposition": "discarded",
                "authoritative_checkpoint_unchanged": file_sha256(args.source_checkpoint) == SOURCE_SHA256,
                "status": "PASS" if passed else "FAIL",
                "passed": passed,
            }
            audit_path = Path(args.run_dir) / "FOUR_GPU_MIGRATION_AUDIT.json"
            write_json(audit_path, audit)
            markdown = (
                "# Four-GPU migration audit\n\n"
                f"Status: **{audit['status']}**\n\n"
                f"- Global batch: `{actual_hash}`\n"
                f"- 1-GPU loss: {reference['global_loss']:.10f}\n"
                f"- 4-GPU loss: {global_loss:.10f}\n"
                f"- Loss difference: {audit['absolute_loss_difference']:.3e}\n"
                f"- Gradient cosine: {grad_metrics['cosine_similarity']:.10f}\n"
                f"- Gradient relative L2: {grad_metrics['relative_l2_difference']:.3e}\n"
                f"- Parameter-update cosine: {update_comparison['cosine_similarity']:.10f}\n"
                f"- Parameter-update relative L2: {update_comparison['relative_l2_difference']:.3e}\n"
            )
            (Path(args.run_dir) / "FOUR_GPU_MIGRATION_AUDIT.md").write_text(markdown)
            print(f"FOUR_GPU_MIGRATION_{audit['status']} loss_diff={audit['absolute_loss_difference']:.3e} grad_cos={grad_metrics['cosine_similarity']:.10f}")
        verdict = torch.tensor([1 if passed else 0], device=device)
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
        if verdict.item() != 1:
            raise SystemExit("four-GPU migration equivalence failed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def load_training_runtime(args, rank, local_rank):
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_digest = file_sha256(checkpoint_path)
    checkpoint = a0.torch_load(checkpoint_path, mmap=True)
    device = torch.device("cuda", local_rank)
    if checkpoint.get("schema") == SOURCE_SCHEMA:
        if checkpoint_digest != SOURCE_SHA256:
            raise SystemExit("result training source is not the frozen 2B2 checkpoint")
        symbols, model, optimizer, completed = instantiate_runtime(checkpoint, device)
        loader_states = checkpoint["dataloader_states"]
        seed = seed_rank(rank)
        source_kind = "untouched_2b2_5m"
    elif checkpoint.get("schema") == CHECKPOINT_SCHEMA:
        symbols, model, optimizer, completed = instantiate_runtime(checkpoint, device)
        loader_states = checkpoint["dataloader_states"]
        restore_rank_rng(checkpoint["rank_rng_states"][rank], local_rank)
        seed = checkpoint["rank_seeds"][rank]
        source_kind = "fresh_4gpu_milestone_resume"
        expected_sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
        if not expected_sidecar.is_file() or expected_sidecar.read_text().split()[0] != checkpoint_digest:
            raise SystemExit("milestone checkpoint SHA sidecar mismatch")
    else:
        raise SystemExit("unsupported training checkpoint schema")
    loader = make_rank_loader(symbols, loader_states, rank)
    expected_next, _ = distributed_preview_hash(loader, symbols)
    if expected_next != checkpoint["next_global_batch_sha256"]:
        raise SystemExit("training resume next-global-batch mismatch")
    return checkpoint, checkpoint_digest, symbols, model, optimizer, loader, completed, seed, source_kind


def aggregate_update_metrics(rows, update_number, global_hash, pre_clip, post_clip,
                             reduction_seconds, total_wall, consistency):
    if sum(row["target_seen"] for row in rows) != GLOBAL_TARGETS:
        raise SystemExit("global target count mismatch")
    count = sum(row["metric_count"] for row in rows)
    def vector(field):
        return [sum(row[field][index] for row in rows) / count for index in range(4)]
    source = vector("source_sum")
    adapted = vector("adapted_sum")
    delta = vector("delta_sum")
    routing = vector("routing_sum")
    return {
        "kind": "train",
        "writer_update": update_number,
        "writer_training_tokens": update_number * GLOBAL_TARGETS,
        "global_training_loss": sum(row["raw_loss_sum"] for row in rows) / GLOBAL_TARGETS,
        "rank_local_losses": [row["raw_loss_sum"] / RANK_TARGETS for row in rows],
        "global_batch_sha256": global_hash,
        "targets": GLOBAL_TARGETS,
        "rank_targets": RANK_TARGETS,
        "microsteps_per_rank": MICROSTEPS_PER_RANK,
        "pre_clip_global_gradient_norm": float(pre_clip),
        "post_clip_global_gradient_norm": post_clip,
        "learning_rate": 1e-4,
        "routing_weights": {f"v{depth}": routing[index] for index, depth in enumerate(b2.SOURCE_DEPTHS)},
        "source_state_rms": {f"v{depth}": source[index] for index, depth in enumerate(b2.SOURCE_DEPTHS)},
        "adapted_memory_rms": {f"v{depth}": adapted[index] for index, depth in enumerate(b2.SOURCE_DEPTHS)},
        "writer_delta_rms": {f"v{depth}": delta[index] for index, depth in enumerate(b2.SOURCE_DEPTHS)},
        "mean_delta_to_source_ratio": {f"v{depth}": delta[index] / max(source[index], 1e-30) for index, depth in enumerate(b2.SOURCE_DEPTHS)},
        "maximum_writer_residual_to_source_rms_ratio": max(row["maximum_residual_ratio"] for row in rows),
        "routing_entropy": sum(row["entropy_sum"] for row in rows) / count,
        "topdown_rms": sum(row["topdown_sum"] for row in rows) / count,
        "feedback_rms": sum(row["feedback_sum"] for row in rows) / count,
        "rank_peak_allocated_mb": [row["peak_allocated_mb"] for row in rows],
        "rank_peak_reserved_mb": [row["peak_reserved_mb"] for row in rows],
        "rank_step_seconds": [row["local_wall_seconds"] for row in rows],
        "gradient_all_reduce_seconds": reduction_seconds,
        "global_update_wall_seconds": total_wall,
        "effective_tokens_per_second": GLOBAL_TARGETS / total_wall,
        "cross_rank_consistency": consistency,
        "teacher_training_forward_calls": 0,
        "hellaswag_run": False,
    }


def save_distributed_checkpoint(args, checkpoint_source, source_digest, symbols, model,
                                optimizer, loader, update_number, rank, local_rank,
                                rank_seed):
    loader_states = [None] * WORLD_SIZE
    rng_states = [None] * WORLD_SIZE
    rank_meta = [None] * WORLD_SIZE
    dist.all_gather_object(loader_states, a0.loader_state(loader))
    dist.all_gather_object(rng_states, capture_rank_rng(rank, local_rank))
    dist.all_gather_object(rank_meta, {
        "rank": rank, "gpu": local_rank, "loader_state": rank,
        "hostname": os.uname().nodename, "pid": os.getpid(),
    })
    next_hash, next_rank_hashes = distributed_preview_hash(loader, symbols)
    consistency = all_equal_across_ranks(
        optimizer_hashes(model, optimizer), "pre-checkpoint state"
    )
    output = Path(args.run_dir) / "checkpoints" / f"checkpoint_updates_{update_number:06d}.pt"
    sidecar = None
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise SystemExit(f"refusing to overwrite checkpoint: {output}")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_state": {
                "writer_updates": update_number,
                "writer_training_tokens": update_number * GLOBAL_TARGETS,
                "fineweb_lineage_completed_update": 497 + update_number,
                "kind": f"2b2a_{MILESTONES[update_number]}",
            },
            "dataloader_states": loader_states,
            "rank_rng_states": rng_states,
            "rank_seeds": [2_026_082_000 + value for value in range(WORLD_SIZE)],
            "rank_metadata": rank_meta,
            "next_rank_microstep_hashes": next_rank_hashes,
            "next_global_batch_sha256": next_hash,
            "world_size": WORLD_SIZE,
            "rank_to_loader_mapping": {str(value): value for value in range(WORLD_SIZE)},
            "gradient_synchronization": "one flattened FP32 NCCL all_reduce(SUM) per update",
            "loss_scaling": "token-loss sums / 524288; no post-SUM division",
            "temporal_credit": "exactly one token; historical Blocks 2-12 KV detached",
            "source_2b2_checkpoint_path": str(Path(args.source_checkpoint).resolve()),
            "source_2b2_checkpoint_sha256": SOURCE_SHA256,
            "source_2b2_results_commit": SOURCE_RESULTS_COMMIT,
            "source_2b2_implementation_commit": SOURCE_IMPLEMENTATION_COMMIT,
            "source_2b1_checkpoint_sha256": checkpoint_source["source_2b1_checkpoint_sha256"],
            "source_2b1_commit": checkpoint_source["source_2b1_commit"],
            "implementation_git_commit": git_output("rev-parse", "HEAD"),
            "frozen_base_sha256": b2.state_subset_sha256(model, "base"),
            "frozen_reader_sha256": b2.state_subset_sha256(model, "reader"),
            "writer_sha256": b2.state_subset_sha256(model, "writers"),
            "optimizer_consistency": consistency,
            "writer_architecture": checkpoint_source["writer_architecture"],
            "saved_by_pid": os.getpid(),
        }
        temporary = output.with_name(output.name + ".incomplete")
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        reopened = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
        with torch.random.fork_rng(devices=[]):
            clone = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
        clone.load_state_dict(reopened["model"], strict=True)
        clone.freeze_for_memory_writer_training()
        clone_optimizer = b2.fresh_optimizer(clone, device_type="cpu")
        clone_optimizer.load_state_dict(reopened["optimizer"])
        b2.optimizer_report(clone_optimizer, update_number)
        if reopened["next_global_batch_sha256"] != next_hash:
            raise SystemExit("checkpoint reopen next-batch mismatch")
        digest = file_sha256(temporary)
        os.replace(temporary, output)
        output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
        sidecar = {
            "checkpoint": str(output.resolve()),
            "sha256": digest,
            "bytes": output.stat().st_size,
            "model_strict_reload": True,
            "optimizer_strict_reload": True,
            "optimizer": b2.optimizer_report(reopened["optimizer"], update_number),
            "loader_states": len(reopened["dataloader_states"]),
            "rank_rng_states": len(reopened["rank_rng_states"]),
            "next_global_batch_sha256": next_hash,
            "cross_rank_consistency": consistency,
            "passed": True,
        }
        write_json(output.with_suffix(output.suffix + ".verification.json"), sidecar)
        print(f"CHECKPOINT_PASS update={update_number} sha256={digest} next={next_hash}", flush=True)
    dist.barrier()
    return sidecar


def train(args):
    require_git(clean=True)
    load_config()
    audit_path = Path(args.run_dir) / "FOUR_GPU_MIGRATION_AUDIT.json"
    if not audit_path.is_file() or not json.loads(audit_path.read_text()).get("passed"):
        raise SystemExit("a passing FOUR_GPU_MIGRATION_AUDIT.json is required")
    if args.target_update not in MILESTONES:
        raise SystemExit("target update must be 20, 29, or 48")
    rank, local_rank = init_distributed()
    try:
        stage_started = time.perf_counter()
        (checkpoint_source, source_digest, symbols, model, optimizer, loader,
         completed, rank_seed, source_kind) = load_training_runtime(args, rank, local_rank)
        if completed >= args.target_update or args.target_update > MAX_UPDATE:
            raise SystemExit(f"invalid training interval {completed}->{args.target_update}")
        if completed == 10 and source_kind != "untouched_2b2_5m":
            raise SystemExit("update 11 must freshly reload the untouched 2B2 source")
        if completed in (20, 29) and source_kind != "fresh_4gpu_milestone_resume":
            raise SystemExit("continuation must use a fresh milestone checkpoint resume")
        if completed not in (10, 20, 29):
            raise SystemExit(f"unauthorized stage starting update: {completed}")
        frozen_base = b2.state_subset_sha256(model, "base")
        frozen_reader = b2.state_subset_sha256(model, "reader")
        metrics_path = Path(args.run_dir) / "training_metrics.jsonl"
        for update_number in range(completed + 1, args.target_update + 1):
            update_started = time.perf_counter()
            expected_hash, expected_rank_hashes = distributed_preview_hash(loader, symbols)
            optimizer.zero_grad(set_to_none=True)
            batches = [loader.next_batch() for _ in range(MICROSTEPS_PER_RANK)]
            local = process_batches(model, batches, update_number)
            if local["batch_hashes"] != expected_rank_hashes[rank]:
                raise SystemExit("rank consumed batch differs from preview")
            flat = flatten_gradients(model)
            reduction_started = time.perf_counter()
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            reduction_seconds = time.perf_counter() - reduction_started
            scatter_gradients(model, flat)
            gradients = b2.gradient_report(model)
            if gradients["reader_tensors_with_grad"] or gradients["frozen_tensors_with_grad"]:
                raise SystemExit("gradient leaked outside writers")
            pre_clip = torch.nn.utils.clip_grad_norm_(
                [value for _, value in writer_parameters(model)], 1.0
            )
            if not torch.isfinite(pre_clip):
                raise SystemExit("non-finite global gradient norm")
            post_clip = b2.grad_global_norm([value for _, value in writer_parameters(model)])
            optimizer.step()
            torch.cuda.synchronize()
            b2.optimizer_report(optimizer, update_number)
            consistency = all_equal_across_ranks(
                optimizer_hashes(model, optimizer), f"update {update_number} state"
            )
            gathered = [None] * WORLD_SIZE
            dist.all_gather_object(gathered, local)
            actual_hash = canonical_batch_hash([row["batch_hashes"] for row in gathered])
            if actual_hash != expected_hash:
                raise SystemExit("global consumed batch differs from preview")
            total_wall = time.perf_counter() - update_started
            row = aggregate_update_metrics(
                gathered, update_number, actual_hash, pre_clip, post_clip,
                reduction_seconds, total_wall, consistency,
            )
            row["gradients"] = gradients
            row["source_kind"] = source_kind if update_number == completed + 1 else "same_stage"
            if any(value > 0.25 for value in row["mean_delta_to_source_ratio"].values()):
                raise SystemExit("writer residual hard-stop threshold exceeded")
            if rank == 0:
                append_jsonl(metrics_path, row)
                print(
                    f"RESULT_UPDATE_PASS update={update_number:02d} "
                    f"loss={row['global_training_loss']:.6f} "
                    f"grad={row['pre_clip_global_gradient_norm']:.6f} "
                    f"wall={total_wall:.1f}s tok/s={row['effective_tokens_per_second']:.0f}",
                    flush=True,
                )
        if b2.state_subset_sha256(model, "base") != frozen_base:
            raise SystemExit("frozen base changed during stage")
        if b2.state_subset_sha256(model, "reader") != frozen_reader:
            raise SystemExit("frozen reader changed during stage")
        sidecar = save_distributed_checkpoint(
            args, checkpoint_source, source_digest, symbols, model, optimizer,
            loader, args.target_update, rank, local_rank, rank_seed,
        )
        if rank == 0:
            summary = {
                "stage": MILESTONES[args.target_update],
                "start_update": completed,
                "end_update": args.target_update,
                "new_updates": args.target_update - completed,
                "writer_training_tokens": args.target_update * GLOBAL_TARGETS,
                "source_kind": source_kind,
                "source_checkpoint_sha256": source_digest,
                "checkpoint": sidecar,
                "frozen_base_unchanged": True,
                "frozen_reader_unchanged": True,
                "teacher_training_forward_calls": 0,
                "stage_wall_seconds": time.perf_counter() - stage_started,
                "fresh_process_restart_required_before_continuation": args.target_update in (20, 29),
                "passed": True,
            }
            write_json(Path(args.run_dir) / f"training_{MILESTONES[args.target_update]}.json", summary)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def milestone_evaluate(args):
    require_git(clean=True)
    load_config()
    if args.update not in MILESTONES:
        raise SystemExit("evaluation update must be 20, 29, or 48")
    rank, local_rank = init_distributed()
    try:
        device = torch.device("cuda", local_rank)
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint = a0.torch_load(checkpoint_path, mmap=True)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise SystemExit("milestone checkpoint schema mismatch")
        if checkpoint["training_state"]["writer_updates"] != args.update:
            raise SystemExit("milestone checkpoint update mismatch")
        symbols = a0.support.load_training_symbols()
        with torch.random.fork_rng(devices=[]):
            model = symbols["GPT"](b2.model_config(symbols, enable_writers=True))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.freeze_for_memory_writer_training()
        model.to(device)
        temporal = None
        loader = b2.validation_loader(symbols)
        first_x, first_y = loader.next_batch()
        if rank == 0:
            temporal = b2.production_temporal_gradient_test(
                model, first_x[:2, :4].to(device), first_y[:2, :4].to(device),
                expect_initial_staging=False,
            )
        dist.barrier()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        loader = b2.validation_loader(symbols)
        permutation = symbols["fixed_derangement"](B, device)
        local_rows = []
        all_payload_hashes = []
        for batch_index in range(a0.VALIDATION_BATCHES):
            x_cpu, y_cpu = loader.next_batch()
            payload_hash = a0.batch_payload_hash(x_cpu, y_cpu)
            all_payload_hashes.append(payload_hash)
            if batch_index % WORLD_SIZE != rank:
                continue
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            real = b2.stream_loss(
                model, x, y, mode="masked_l1_topdown_self", use_writers=True
            )
            shuffled = b2.stream_loss(
                model, x, y, mode="masked_l1_shuffled_self_feedback",
                use_writers=True, permutation=permutation,
            )
            local_rows.append({
                "batch_index": batch_index,
                "payload_sha256": payload_hash,
                "real": real,
                "shuffled": shuffled,
            })
            print(f"rank={rank} eval={batch_index:02d} real={real['loss']:.6f} shuffled={shuffled['loss']:.6f}", flush=True)
        digest = hashlib.sha256()
        for value in all_payload_hashes:
            digest.update(bytes.fromhex(value))
        if digest.hexdigest() != a0.EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256:
            raise SystemExit("canonical validation aggregate hash mismatch")
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local_rows)
        temporals = [None] * WORLD_SIZE
        dist.all_gather_object(temporals, temporal)
        if rank == 0:
            rows = sorted((row for group in gathered for row in group), key=lambda row: row["batch_index"])
            if [row["batch_index"] for row in rows] != list(range(20)):
                raise SystemExit("distributed canonical evaluation coverage mismatch")
            real_values = [row["real"]["loss"] for row in rows]
            shuffled_values = [row["shuffled"]["loss"] for row in rows]
            real_mean = statistics.fmean(real_values)
            shuffled_mean = statistics.fmean(shuffled_values)
            paired = b2.paired_statistics(real_values, shuffled_values)
            previous_real = SOURCE_REAL
            previous_gap = SOURCE_SHUFFLED - SOURCE_REAL
            if args.update != 20:
                previous_update = 20 if args.update == 29 else 29
                previous_path = Path(args.run_dir) / f"milestone_{MILESTONES[previous_update]}.json"
                previous = json.loads(previous_path.read_text())
                previous_real = previous["losses"]["real"]
                previous_gap = previous["scaling"]["specific_gap"]
            gap = shuffled_mean - real_mean
            scaling = {
                "writer_gain_vs_2b1": FROZEN_2B1_REAL - real_mean,
                "incremental_real_gain": previous_real - real_mean,
                "specific_gap": gap,
                "specific_gap_change": gap - previous_gap,
                "recovery_fraction": (MASKED - real_mean) / (MASKED - FULL_CONTEXT),
            }
            integrity = {
                "temporal_gradient_boundary": temporals[0],
                "all_losses_finite": all(math.isfinite(value) for value in real_values + shuffled_values),
                "writer_mean_residual_below_hard_stop": all(
                    row["real"]["writer_behavior"][f"v{depth}"]["delta_to_source_rms_ratio"] <= 0.25
                    for row in rows for depth in b2.SOURCE_DEPTHS
                ),
                "base_sha256_matches_checkpoint": b2.state_subset_sha256(model, "base") == checkpoint["frozen_base_sha256"],
                "reader_sha256_matches_checkpoint": b2.state_subset_sha256(model, "reader") == checkpoint["frozen_reader_sha256"],
                "hellaswag_run": False,
            }
            integrity["passed"] = (
                integrity["temporal_gradient_boundary"]["passed"]
                and integrity["all_losses_finite"]
                and integrity["writer_mean_residual_below_hard_stop"]
                and integrity["base_sha256_matches_checkpoint"]
                and integrity["reader_sha256_matches_checkpoint"]
            )
            if args.update == 20:
                gate = (
                    scaling["incremental_real_gain"] >= 0.010
                    and gap >= 0.0359389782 - 0.005
                    and paired["real_wins"] >= 18
                    and integrity["passed"]
                )
            elif args.update == 29:
                gate = (
                    scaling["incremental_real_gain"] >= 0.010
                    and gap >= previous_gap - 0.005
                    and paired["real_wins"] >= 18
                    and integrity["passed"]
                )
            else:
                gate = False
            report = {
                "experiment": "2B2A",
                "milestone": MILESTONES[args.update],
                "writer_update": args.update,
                "writer_training_tokens": args.update * GLOBAL_TARGETS,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "validation_global_batches_sha256": digest.hexdigest(),
                "rank_batch_mapping": {str(value): list(range(value, 20, 4)) for value in range(4)},
                "losses": {"real": real_mean, "shuffled": shuffled_mean},
                "paired": paired,
                "scaling": scaling,
                "writer_behavior": b2.average_writer_behavior([row["real"] for row in rows]),
                "integrity": integrity,
                "continuation_gate_passed": gate,
                "terminal": args.update == 48 or not gate,
                "batch_rows": rows,
                "passed": integrity["passed"],
            }
            write_json(Path(args.run_dir) / f"milestone_{MILESTONES[args.update]}.json", report)
            print(
                f"MILESTONE_{MILESTONES[args.update].upper()} real={real_mean:.10f} "
                f"shuffled={shuffled_mean:.10f} gap={gap:.10f} wins={paired['real_wins']}/20 "
                f"continue={'YES' if gate else 'NO'}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "source-audit", "migration-reference", "migration-candidate", "train", "milestone-evaluate"
    ))
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--target-update", type=int)
    parser.add_argument("--update", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "source-audit":
        source_audit(args)
    elif args.command == "migration-reference":
        migration_reference(args)
    elif args.command == "migration-candidate":
        migration_candidate(args)
    elif args.command == "train":
        if args.checkpoint is None or args.target_update is None:
            raise SystemExit("train requires --checkpoint and --target-update")
        train(args)
    elif args.command == "milestone-evaluate":
        if args.checkpoint is None or args.update is None:
            raise SystemExit("milestone-evaluate requires --checkpoint and --update")
        milestone_evaluate(args)


if __name__ == "__main__":
    main()
