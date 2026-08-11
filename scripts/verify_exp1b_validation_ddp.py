#!/usr/bin/env python3
"""Prove that Experiment 1B's 4-rank validation covers the original 20 batches."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import smoke_test as support
from experiment_train_ddp import (
    B,
    T,
    WORLD_SIZE,
    Runtime,
    all_gather_objects,
    batch_payload_hash,
    file_sha256,
    validation_loss_ddp,
)


@torch.no_grad()
def serial_validation_loss(model, loader, device, batches):
    model.eval()
    loader.reset()
    total = torch.zeros((), device=device, dtype=torch.float64)
    losses = []
    for _ in range(batches):
        x, y = loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += loss.detach().double()
        losses.append(loss.detach().double().item())
    loader.reset()
    return (total / batches).item(), losses


def loader_hashes(loader, batches):
    loader.reset()
    hashes = []
    for _ in range(batches):
        x, y = loader.next_batch()
        hashes.append(batch_payload_hash(x, y))
    loader.reset()
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-mode", choices=("standard", "full_attnres"), default="standard")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runtime = Runtime()
    try:
        symbols = support.load_training_symbols()
        symbols["master_process"] = runtime.master
        model = symbols["GPT"](
            symbols["GPTConfig"](vocab_size=50304, residual_mode=args.residual_mode)
        )
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if args.residual_mode == "standard":
            model.load_state_dict(init["model"], strict=True)
        else:
            model.load_shared_baseline_state(init["model"])
        model.to(runtime.device)

        loader_type = symbols["DataLoaderLite"]
        serial_loss = None
        serial_batch_losses = None
        serial_hashes = None
        if runtime.master:
            serial_loader = loader_type(B=B, T=T, process_rank=0, num_processes=1, split="val")
            serial_hashes = loader_hashes(serial_loader, 20)
            serial_loss, serial_batch_losses = serial_validation_loss(
                model, serial_loader, runtime.device, 20
            )
        runtime.barrier()

        distributed_loader = loader_type(
            B=B, T=T, process_rank=runtime.rank, num_processes=WORLD_SIZE, split="val"
        )
        local_hashes = loader_hashes(distributed_loader, 5)
        hashes_by_rank = all_gather_objects(local_hashes, runtime)
        distributed_loss, _ = validation_loss_ddp(
            model, distributed_loader, runtime, global_batches=20, collect_routing=False
        )

        if runtime.master:
            interleaved_hashes = [
                hashes_by_rank[rank][local_index]
                for local_index in range(5)
                for rank in range(WORLD_SIZE)
            ]
            absolute_difference = abs(serial_loss - distributed_loss)
            report = {
                "residual_mode": args.residual_mode,
                "checkpoint": str(Path(args.init_checkpoint).resolve()),
                "checkpoint_sha256": file_sha256(args.init_checkpoint),
                "B": B,
                "T": T,
                "serial_world_size": 1,
                "serial_global_batches": 20,
                "distributed_world_size": WORLD_SIZE,
                "distributed_batches_per_rank": 5,
                "distributed_global_batches": 20,
                "serial_loss": serial_loss,
                "distributed_loss": distributed_loss,
                "absolute_loss_difference": absolute_difference,
                "loss_tolerance": 1e-8,
                "batch_hashes_match_in_global_order": interleaved_hashes == serial_hashes,
                "serial_batch_hashes_sha256": hashlib.sha256(
                    "".join(serial_hashes).encode()
                ).hexdigest(),
                "distributed_batch_hashes_sha256": hashlib.sha256(
                    "".join(interleaved_hashes).encode()
                ).hexdigest(),
                "serial_batch_losses": serial_batch_losses,
                "passed": interleaved_hashes == serial_hashes and absolute_difference <= 1e-8,
            }
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            if not report["passed"]:
                raise SystemExit("single-GPU and 4-GPU validation protocols are inconsistent")
        runtime.barrier()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
