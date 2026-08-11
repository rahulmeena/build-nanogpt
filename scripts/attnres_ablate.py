#!/usr/bin/env python3
"""Offline causal source ablation for a trained Full AttnRes checkpoint."""

import argparse
import json
from pathlib import Path

import torch

import smoke_test as support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--val-steps", type=int, default=20)
    parser.add_argument("--sources", nargs="*", type=int, default=list(range(25)))
    args = parser.parse_args()

    support.assert_cuda_environment(require_a100_80gb=False)
    symbols = support.load_training_symbols()
    model = symbols["GPT"](symbols["GPTConfig"](vocab_size=50304, residual_mode="full_attnres"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda().eval()
    loader = symbols["DataLoaderLite"](B=args.B, T=args.T, process_rank=0, num_processes=1, split="val")

    normal_loss = support.validation_loss(model, loader, "cuda", args.val_steps)
    results = []
    for source in args.sources:
        if source < 0 or source > 24:
            raise SystemExit(f"source depth out of range: {source}")
        model.set_attnres_source_mask(source)
        masked_loss = support.validation_loss(model, loader, "cuda", args.val_steps)
        results.append({
            "source_depth": source,
            "normal_validation_loss": normal_loss,
            "masked_validation_loss": masked_loss,
            "causal_contribution_delta": masked_loss - normal_loss,
            "source_zero_limitation": source == 0,
        })
    model.set_attnres_source_mask(None)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "B": args.B,
        "T": args.T,
        "validation_steps": args.val_steps,
        "normal_validation_loss": normal_loss,
        "source_zero_note": "v0 remains the sole input to the first attention sublayer; it is masked from all later routers.",
        "ablations": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
