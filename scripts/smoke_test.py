#!/usr/bin/env python3
"""Run production-shape standard or Full AttnRes smoke tests on one A100."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOTAL_BATCH_SIZE = 524288
B = 64
T = 1024
GRAD_ACCUM_STEPS = 8
MAX_LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 715
MAX_STEPS = 19073
BASELINE_COMMIT = "a834ca88b7b6c4e81c2a71eef0edde29b2ee2ccb"
EXPECTED_INIT_SHA256 = "39de351efe080de4e2409355c572095f17dcbaea76154a2f55e375acfdafc3b6"


def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def load_training_symbols():
    train_path = REPO_ROOT / "train_gpt2.py"
    source = train_path.read_text()
    marker = "# -----------------------------------------------------------------------------\n# simple launch:"
    if marker not in source:
        raise SystemExit("could not find training launch marker in train_gpt2.py")
    namespace = {
        "__name__": "nanogpt_smoke_symbols",
        "__file__": str(train_path),
        "master_process": True,
    }
    sys.path.insert(0, str(REPO_ROOT))
    exec(compile(source.split(marker)[0], str(train_path), "exec"), namespace)
    namespace["master_process"] = True
    return namespace


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def gpu_memory_mb():
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def assert_cuda_environment(require_a100_80gb):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the faithful B64/T1024 smoke test")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 is required for the faithful smoke test")
    if require_a100_80gb:
        props = torch.cuda.get_device_properties(0)
        if "A100" not in props.name or props.total_memory < 79 * 1024**3:
            raise SystemExit(f"A100 80GB is required, detected {props.name} with {props.total_memory / 1024**3:.2f} GiB")


def finite_gradients(model):
    return all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


def gradient_category_norms(model):
    squared = {
        "attnres_queries": 0.0,
        "attnres_rmsnorm": 0.0,
        "gpt2_attention": 0.0,
        "gpt2_mlp": 0.0,
    }
    tensors_with_grad = {key: 0 for key in squared}
    tensors_nonzero = {key: 0 for key in squared}
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
        tensors_with_grad[category] += 1
        tensors_nonzero[category] += int(torch.count_nonzero(grad).item() > 0)
    return {
        category: {
            "l2_norm": value ** 0.5,
            "tensors_with_grad": tensors_with_grad[category],
            "tensors_with_nonzero_grad": tensors_nonzero[category],
        }
        for category, value in squared.items()
    }


def write_jsonl(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row) + "\n")


def next_cuda_batch(loader, device):
    x, y = loader.next_batch()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def validation_loss(model, val_loader, device, steps):
    model.eval()
    val_loader.reset()
    loss_accum = 0.0
    for _ in range(steps):
        x, y = next_cuda_batch(val_loader, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss_accum += loss.detach().float().item() / steps
    return loss_accum


@torch.no_grad()
def hellaswag_smoke(model, symbols, device, examples):
    if examples <= 0:
        return {"examples": 0, "correct_norm": 0, "accuracy_norm": None}
    model.eval()
    total = 0
    correct = 0
    for example in symbols["iterate_examples"]("val"):
        _, tokens, mask, label = symbols["render_example"](example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(tokens)
        pred_norm = symbols["get_most_likely_row"](tokens, mask, logits)
        total += 1
        correct += int(pred_norm == label)
        if total >= examples:
            break
    return {
        "examples": total,
        "correct_norm": correct,
        "accuracy_norm": correct / total if total else None,
    }


def timed_optimizer_update(model, optimizer, train_loader, device, lr, metrics_file, step, wall_clock_start=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    forward_events = []
    backward_events = []
    wall_start = time.perf_counter()
    loss_accum = 0.0
    for _ in range(GRAD_ACCUM_STEPS):
        x, y = next_cuda_batch(train_loader, device)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        forward_end.record()
        loss = loss / GRAD_ACCUM_STEPS
        loss_accum += loss.detach().float().item()
        backward_start.record()
        loss.backward()
        backward_end.record()
        forward_events.append((forward_start, forward_end))
        backward_events.append((backward_start, backward_end))

    if not finite_gradients(model):
        raise SystemExit("non-finite gradient detected")
    gradient_categories = gradient_category_norms(model)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    optimizer_start.record()
    optimizer.step()
    optimizer_end.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    forward_ms = sum(start.elapsed_time(end) for start, end in forward_events)
    backward_ms = sum(start.elapsed_time(end) for start, end in backward_events)
    optimizer_ms = optimizer_start.elapsed_time(optimizer_end)
    memory = gpu_memory_mb()
    row = {
        "kind": "train",
        "step": step,
        "tokens": (step + 1) * TOTAL_BATCH_SIZE,
        "train_loss": loss_accum,
        "val_loss": None,
        "hellaswag_accuracy": None,
        "lr": lr,
        "grad_norm": float(grad_norm),
        "gradient_categories": gradient_categories,
        "step_time_ms": wall_seconds * 1000,
        "forward_time_ms_8_microsteps": forward_ms,
        "backward_time_ms_8_microsteps": backward_ms,
        "optimizer_step_time_ms": optimizer_ms,
        "tokens_per_second": TOTAL_BATCH_SIZE / wall_seconds,
        "wall_clock_seconds": time.perf_counter() - wall_clock_start if wall_clock_start is not None else None,
        "gpu_allocated_mb": memory["allocated_mb"],
        "gpu_reserved_mb": memory["reserved_mb"],
        "gpu_peak_mb": memory["peak_allocated_mb"],
        "gpu_peak_reserved_mb": memory["peak_reserved_mb"],
    }
    write_jsonl(metrics_file, row)
    return row


def phase_b_probe(model, optimizer, train_loader, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x, y = next_cuda_batch(train_loader, device)

    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    forward_start.record()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    forward_end.record()
    torch.cuda.synchronize()
    after_forward = gpu_memory_mb()

    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    backward_start.record()
    loss.backward()
    backward_end.record()
    torch.cuda.synchronize()
    if not finite_gradients(model):
        raise SystemExit("non-finite gradient detected after Phase B backward")
    after_backward = gpu_memory_mb()
    gradient_categories = gradient_category_norms(model)
    result = {
        "loss": float(loss.detach().float()),
        "forward_time_ms": forward_start.elapsed_time(forward_end),
        "backward_time_ms": backward_start.elapsed_time(backward_end),
        "after_forward_memory": after_forward,
        "after_backward_memory": after_backward,
        "gradient_categories": gradient_categories,
    }
    optimizer.zero_grad(set_to_none=True)
    train_loader.reset()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--residual-mode", choices=("standard", "full_attnres"), required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--optimizer-steps", type=int, default=12)
    parser.add_argument("--val-steps", type=int, default=2)
    parser.add_argument("--hellaswag-examples", type=int, default=32)
    parser.add_argument("--require-a100-80gb", action="store_true")
    args = parser.parse_args()
    if args.optimizer_steps < 1:
        raise SystemExit("--optimizer-steps must be at least one")

    subprocess.run([
        sys.executable,
        str(REPO_ROOT / "scripts" / "assert_training_config.py"),
        "--train-file", str(REPO_ROOT / "train_gpt2.py"),
        "--world-size", "1",
        "--require-world-size-one",
    ], check=True)

    assert_cuda_environment(args.require_a100_80gb)
    device = "cuda"
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"
    metrics_file.write_text("")

    symbols = load_training_symbols()
    model_config = symbols["GPTConfig"](vocab_size=50304, residual_mode=args.residual_mode)
    model = symbols["GPT"](model_config)
    init_path = Path(args.init_checkpoint)
    init_sha256 = file_sha256(init_path)
    if init_sha256 != EXPECTED_INIT_SHA256:
        raise SystemExit(
            f"initialization checkpoint SHA256 mismatch: expected {EXPECTED_INIT_SHA256}, got {init_sha256}"
        )
    init_checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    baseline_state = init_checkpoint["model"]
    if args.residual_mode == "standard":
        model.load_state_dict(baseline_state, strict=True)
    else:
        model.load_shared_baseline_state(baseline_state)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    before_forward_memory = gpu_memory_mb()
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=MAX_LR, device_type="cuda")

    dataset_path = run_dir / "dataset.json"
    environment_path = run_dir / "environment.json"
    dataset_report = json.loads(dataset_path.read_text()) if dataset_path.exists() else None
    metadata = {
        "run_kind": "production-shape-smoke",
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short", "--branch"),
        "baseline_parent_commit": BASELINE_COMMIT,
        "baseline_tag": "baseline-gpt2-124m-10b",
        "model_config": vars(model_config),
        "attnres_config": {
            "mode": args.residual_mode,
            "query_initialization": "zeros",
            "key_normalization": "RMSNorm",
            "rmsnorm_epsilon": model_config.attnres_rms_eps,
            "softmax_dimension": "residual_depth",
        },
        "seed": 1337,
        "dataset_verification": "dataset.json" if dataset_path.exists() else None,
        "dataset_verification_sha256": file_sha256(dataset_path) if dataset_path.exists() else None,
        "dataset_summary": None if dataset_report is None else {
            "number_of_shard_files": dataset_report["number_of_shard_files"],
            "total_token_count": dataset_report["total_token_count"],
            "dtype": dataset_report["dtype"],
            "validation_shards": dataset_report["validation_shards_upstream"],
            "training_shard_count": len(dataset_report["training_shards_upstream"]),
        },
        "environment_verification": "environment.json" if environment_path.exists() else None,
        "environment_verification_sha256": file_sha256(environment_path) if environment_path.exists() else None,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "B": B,
        "T": T,
        "gradient_accumulation": GRAD_ACCUM_STEPS,
        "global_batch_tokens": TOTAL_BATCH_SIZE,
        "precision": "BF16 autocast, FP32 parameters/residual accumulator",
        "optimizer": "fused AdamW, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1",
        "lr_schedule": "original 10B: 715-step warmup then cosine through 19073 updates",
        "training_token_budget": args.optimizer_steps * TOTAL_BATCH_SIZE,
        "planned_updates": args.optimizer_steps,
        "baseline_init_checkpoint": str(init_path.resolve()),
        "baseline_init_sha256": init_sha256,
        "parameter_count": total_params,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    DataLoaderLite = symbols["DataLoaderLite"]
    train_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="val")

    phase_b = phase_b_probe(model, optimizer, train_loader, device)

    initial_attnres_stats = []
    if args.residual_mode == "full_attnres":
        model.eval()
        model.set_attnres_instrumentation(True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            probe_x, _ = next_cuda_batch(val_loader, device)
            model(probe_x[:2, :32])
        initial_attnres_stats = model.get_attnres_stats()
        model.set_attnres_instrumentation(False)
        val_loader.reset()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_rows = []
    for step in range(args.optimizer_steps):
        row = timed_optimizer_update(
            model, optimizer, train_loader, device, get_lr(step), metrics_file, step
        )
        train_rows.append(row)
        print(
            f"step {step:3d} | loss {row['train_loss']:.6f} | norm {row['grad_norm']:.4f} | "
            f"{row['step_time_ms']:.2f} ms | {row['tokens_per_second']:.2f} tok/s | "
            f"peak {row['gpu_peak_mb']:.2f} MiB",
            flush=True,
        )

    post_attnres_stats = []
    if args.residual_mode == "full_attnres":
        model.set_attnres_instrumentation(True)
    val_loss = validation_loss(model, val_loader, device, args.val_steps)
    if args.residual_mode == "full_attnres":
        post_attnres_stats = model.get_attnres_stats()
        model.set_attnres_instrumentation(False)
    completed_step = train_rows[-1]["step"]
    completed_tokens = train_rows[-1]["tokens"]
    write_jsonl(metrics_file, {
        "kind": "val", "step": completed_step, "tokens": completed_tokens,
        "train_loss": None, "val_loss": val_loss, "hellaswag_accuracy": None,
        "lr": None, "grad_norm": None, "step_time_ms": None,
        "tokens_per_second": None, **{f"gpu_{key}": value for key, value in gpu_memory_mb().items()},
    })

    hella = hellaswag_smoke(model, symbols, device, args.hellaswag_examples)
    write_jsonl(metrics_file, {
        "kind": "hellaswag", "step": completed_step, "tokens": completed_tokens,
        "train_loss": None, "val_loss": None, "hellaswag_accuracy": hella["accuracy_norm"],
        "lr": None, "grad_norm": None, "step_time_ms": None,
        "tokens_per_second": None, **{f"gpu_{key}": value for key, value in gpu_memory_mb().items()},
    })

    checkpoint_path = checkpoints_dir / "smoke_model.pt"
    torch.save({
        "model": model.state_dict(),
        "config": vars(model.config),
        "step": completed_step,
        "val_loss": val_loss,
        "baseline_init_sha256": file_sha256(init_path),
    }, checkpoint_path)

    query_max_abs = None
    norm_max_deviation = None
    if args.residual_mode == "full_attnres":
        query_max_abs = max(router.query.detach().abs().max().item() for router in model.transformer.attnres)
        norm_max_deviation = max((router.norm.weight.detach() - 1).abs().max().item() for router in model.transformer.attnres)
        (run_dir / "attnres_initial_stats.json").write_text(json.dumps(initial_attnres_stats, indent=2) + "\n")
        (run_dir / "attnres_post_smoke_stats.json").write_text(json.dumps(post_attnres_stats, indent=2) + "\n")

    average_tps = sum(row["tokens_per_second"] for row in train_rows) / len(train_rows)
    average_step_seconds = sum(row["step_time_ms"] for row in train_rows) / len(train_rows) / 1000
    steady_rows = train_rows[1:] if len(train_rows) > 1 else train_rows
    summary = {
        "residual_mode": args.residual_mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "B": B,
        "T": T,
        "total_batch_size": TOTAL_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "precision": "BF16 autocast on CUDA with FP32 parameters/residual accumulator",
        "init_checkpoint": str(init_path),
        "init_checkpoint_sha256": init_sha256,
        "before_first_forward_memory": before_forward_memory,
        "phase_b": phase_b,
        "optimizer_steps": args.optimizer_steps,
        "processed_tokens": completed_tokens,
        "initial_train_loss": train_rows[0]["train_loss"],
        "final_train_loss": train_rows[-1]["train_loss"],
        "loss_delta": train_rows[-1]["train_loss"] - train_rows[0]["train_loss"],
        "validation_loss": val_loss,
        "hellaswag_smoke": hella,
        "peak_vram": gpu_memory_mb(),
        "average_tokens_per_second": average_tps,
        "steady_state_tokens_per_second": sum(row["tokens_per_second"] for row in steady_rows) / len(steady_rows),
        "average_seconds_per_optimizer_step": average_step_seconds,
        "steady_state_seconds_per_optimizer_step": sum(row["step_time_ms"] for row in steady_rows) / len(steady_rows) / 1000,
        "all_losses_finite": all(math.isfinite(row["train_loss"]) for row in train_rows),
        "all_gradient_norms_finite": all(math.isfinite(row["grad_norm"]) for row in train_rows),
        "initial_attnres_stats": initial_attnres_stats,
        "post_attnres_stats": post_attnres_stats,
        "query_max_abs_after_smoke": query_max_abs,
        "rmsnorm_max_abs_deviation_from_one": norm_max_deviation,
        "theoretical_bf16_residual_stack_bytes": 25 * B * T * 768 * 2,
        "theoretical_bf16_residual_stack_gib": 25 * B * T * 768 * 2 / 1024**3,
        "checkpoint_path": str(checkpoint_path),
        "metrics_file": str(metrics_file),
    }
    (run_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
