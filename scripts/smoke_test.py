#!/usr/bin/env python3
"""Run faithful, short smoke tests without launching the full 10B-token run."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TOTAL_BATCH_SIZE = 524288
B = 64
T = 1024
GRAD_ACCUM_STEPS = 8
MAX_LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 715
MAX_STEPS = 19073


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
    prefix = source.split(marker)[0]
    namespace = {
        "__name__": "nanogpt_smoke_symbols",
        "__file__": str(train_path),
        "master_process": True,
    }
    sys.path.insert(0, str(REPO_ROOT))
    exec(compile(prefix, str(train_path), "exec"), namespace)
    namespace["master_process"] = True
    return namespace


def gpu_memory_mb():
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
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
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def write_jsonl(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row) + "\n")


def next_cuda_batch(loader, device):
    x, y = loader.next_batch()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def validation_loss(model, val_loader, device, device_type, steps):
    model.eval()
    val_loader.reset()
    loss_accum = 0.0
    for _ in range(steps):
        x, y = next_cuda_batch(val_loader, device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss_accum += loss.detach().float().item() / steps
    return loss_accum


@torch.no_grad()
def hellaswag_smoke(model, symbols, device, device_type, examples):
    render_example = symbols["render_example"]
    iterate_examples = symbols["iterate_examples"]
    get_most_likely_row = symbols["get_most_likely_row"]
    model.eval()
    total = 0
    correct = 0
    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(tokens)
        pred_norm = get_most_likely_row(tokens, mask, logits)
        total += 1
        correct += int(pred_norm == label)
        if total >= examples:
            break
    return {
        "examples": total,
        "correct_norm": correct,
        "accuracy_norm": correct / total if total else None,
    }


def optimizer_update(model, optimizer, train_loader, device, device_type, lr, metrics_file, step):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    loss_accum = 0.0
    for _ in range(GRAD_ACCUM_STEPS):
        x, y = next_cuda_batch(train_loader, device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / GRAD_ACCUM_STEPS
        loss_accum += loss.detach().float().item()
        loss.backward()
    if not finite_gradients(model):
        raise SystemExit("non-finite gradient detected")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tokens_per_sec = TOTAL_BATCH_SIZE / dt
    row = {
        "kind": "train",
        "step": step,
        "tokens": (step + 1) * TOTAL_BATCH_SIZE,
        "train_loss": loss_accum,
        "val_loss": None,
        "hellaswag_accuracy": None,
        "lr": lr,
        "grad_norm": float(grad_norm),
        "step_time_ms": dt * 1000,
        "tokens_per_second": tokens_per_sec,
        "gpu_allocated_mb": gpu_memory_mb()["allocated_mb"],
        "gpu_reserved_mb": gpu_memory_mb()["reserved_mb"],
        "gpu_peak_mb": gpu_memory_mb()["peak_mb"],
    }
    write_jsonl(metrics_file, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=2, help="short training optimizer steps after the single-step checks")
    parser.add_argument("--val-steps", type=int, default=2)
    parser.add_argument("--hellaswag-examples", type=int, default=32)
    parser.add_argument("--require-a100-80gb", action="store_true")
    args = parser.parse_args()

    subprocess.run([
        sys.executable,
        str(REPO_ROOT / "scripts" / "assert_training_config.py"),
        "--train-file",
        str(REPO_ROOT / "train_gpt2.py"),
        "--world-size",
        "1",
        "--require-world-size-one",
    ], check=True)

    assert_cuda_environment(args.require_a100_80gb)
    device = "cuda"
    device_type = "cuda"
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()

    run_dir = Path(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"
    metrics_file.write_text("")

    symbols = load_training_symbols()
    GPT = symbols["GPT"]
    GPTConfig = symbols["GPTConfig"]
    DataLoaderLite = symbols["DataLoaderLite"]

    model = GPT(GPTConfig(vocab_size=50304))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    model.to(device)
    raw_model = model
    before_forward_memory = gpu_memory_mb()
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=MAX_LR, device_type=device_type)

    train_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="val")

    # Smoke C: one full B64/T1024 forward and backward pass.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x, y = next_cuda_batch(train_loader, device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        _, first_loss = model(x, y)
    after_first_forward_memory = gpu_memory_mb()
    first_loss.backward()
    if not finite_gradients(model):
        raise SystemExit("non-finite gradient detected after first backward")
    first_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    after_first_backward_memory = gpu_memory_mb()
    optimizer.zero_grad(set_to_none=True)

    # Smoke D: exactly one optimizer update from 8 B64/T1024 microsteps.
    probe = next(model.parameters()).detach().float().norm().item()
    accum_row = optimizer_update(model, optimizer, train_loader, device, device_type, get_lr(0), metrics_file, 0)
    probe_after = next(model.parameters()).detach().float().norm().item()
    if probe_after == probe:
        raise SystemExit("optimizer update did not change the probe parameter norm")

    # Smoke E: a few more faithful optimizer steps, then validation, HellaSwag, checkpoint, and plots.
    train_rows = [accum_row]
    for step in range(1, args.steps + 1):
        train_rows.append(optimizer_update(model, optimizer, train_loader, device, device_type, get_lr(step), metrics_file, step))
    completed_step = train_rows[-1]["step"]
    completed_tokens = train_rows[-1]["tokens"]

    val_loss = validation_loss(model, val_loader, device, device_type, args.val_steps)
    write_jsonl(metrics_file, {
        "kind": "val",
        "step": completed_step,
        "tokens": completed_tokens,
        "train_loss": None,
        "val_loss": val_loss,
        "hellaswag_accuracy": None,
        "lr": None,
        "grad_norm": None,
        "step_time_ms": None,
        "tokens_per_second": None,
        "gpu_allocated_mb": gpu_memory_mb()["allocated_mb"],
        "gpu_reserved_mb": gpu_memory_mb()["reserved_mb"],
        "gpu_peak_mb": gpu_memory_mb()["peak_mb"],
    })

    hella = hellaswag_smoke(model, symbols, device, device_type, args.hellaswag_examples)
    write_jsonl(metrics_file, {
        "kind": "hellaswag",
        "step": completed_step,
        "tokens": completed_tokens,
        "train_loss": None,
        "val_loss": None,
        "hellaswag_accuracy": hella["accuracy_norm"],
        "lr": None,
        "grad_norm": None,
        "step_time_ms": None,
        "tokens_per_second": None,
        "gpu_allocated_mb": gpu_memory_mb()["allocated_mb"],
        "gpu_reserved_mb": gpu_memory_mb()["reserved_mb"],
        "gpu_peak_mb": gpu_memory_mb()["peak_mb"],
    })

    checkpoint_path = checkpoints_dir / "smoke_model.pt"
    torch.save({
        "model": raw_model.state_dict(),
        "config": raw_model.config,
        "step": completed_step,
        "val_loss": val_loss,
    }, checkpoint_path)

    average_tps = sum(row["tokens_per_second"] for row in train_rows) / len(train_rows)
    average_step_seconds = sum(row["step_time_ms"] for row in train_rows) / len(train_rows) / 1000
    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "parameter_dtypes": dtypes,
        "B": B,
        "T": T,
        "total_batch_size": TOTAL_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "precision": "bf16 autocast on CUDA",
        "initial_forward_backward_loss": float(first_loss.detach().float().item()),
        "first_backward_grad_norm": float(first_grad_norm),
        "later_smoke_loss": train_rows[-1]["train_loss"],
        "loss_delta": train_rows[-1]["train_loss"] - train_rows[0]["train_loss"],
        "validation_loss": val_loss,
        "hellaswag_smoke": hella,
        "before_first_forward_memory": before_forward_memory,
        "after_first_forward_memory": after_first_forward_memory,
        "after_first_backward_memory": after_first_backward_memory,
        "peak_vram_mb": gpu_memory_mb()["peak_mb"],
        "average_tokens_per_second": average_tps,
        "average_seconds_per_optimizer_step": average_step_seconds,
        "estimated_full_run_seconds": MAX_STEPS * average_step_seconds,
        "estimated_full_run_hours": MAX_STEPS * average_step_seconds / 3600,
        "checkpoint_path": str(checkpoint_path),
        "metrics_file": str(metrics_file),
    }
    (run_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
