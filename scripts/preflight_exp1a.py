#!/usr/bin/env python3
"""Abort-before-training controls for the Experiment 1A matched A/B run."""

import argparse
import gc
import json
import math
import subprocess
from pathlib import Path

import torch

import experiment_train as experiment
import smoke_test as support


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--microbatches", type=int, default=8)
    args = parser.parse_args()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tag_commit = subprocess.check_output(
        ["git", "rev-parse", "baseline-gpt2-124m-10b^{commit}"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if tag_commit != experiment.BASELINE_COMMIT:
        raise SystemExit(f"baseline tag mismatch: {tag_commit}")
    init_sha256 = support.file_sha256(args.init_checkpoint)
    if init_sha256 != experiment.EXPECTED_INIT_SHA256:
        raise SystemExit(f"canonical initialization SHA256 mismatch: {init_sha256}")

    symbols = support.load_training_symbols()
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    baseline_state = checkpoint["model"]
    model_reports = {}
    for mode in ("standard", "full_attnres"):
        model = symbols["GPT"](symbols["GPTConfig"](vocab_size=50304, residual_mode=mode))
        if mode == "standard":
            model.load_state_dict(baseline_state, strict=True)
        else:
            model.load_shared_baseline_state(baseline_state)
        verification = experiment.verify_shared_initialization(model, baseline_state)
        model_reports[mode] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "verification": verification,
        }
        del model
        gc.collect()

    DataLoaderLite = symbols["DataLoaderLite"]
    data_reports = []
    for _ in range(2):
        loader = DataLoaderLite(
            B=support.B,
            T=support.T,
            process_rank=0,
            num_processes=1,
            split="train",
        )
        data_reports.append(experiment.data_order_probe(loader, microbatches=args.microbatches))
        del loader
        gc.collect()
    if data_reports[0]["combined_sha256"] != data_reports[1]["combined_sha256"]:
        raise SystemExit("independent training DataLoaders produced different starting hashes")

    report = {
        "experiment": "Experiment 1A matched 100M-token A/B",
        "experiment_commit": commit,
        "baseline_commit": experiment.BASELINE_COMMIT,
        "baseline_tag": "baseline-gpt2-124m-10b",
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "init_checkpoint_sha256": init_sha256,
        "models": model_reports,
        "shared_tensor_mismatches": sum(
            row["verification"]["mismatches"] for row in model_reports.values()
        ),
        "maximum_shared_tensor_difference": max(
            row["verification"]["maximum_absolute_difference"] for row in model_reports.values()
        ),
        "independent_data_loader_hashes": [row["combined_sha256"] for row in data_reports],
        "data_order_match": data_reports[0]["combined_sha256"] == data_reports[1]["combined_sha256"],
        "data_order": data_reports[0],
        "initial_uniform_entropies": [math.log(source_count) for source_count in range(1, 26)],
        "B": support.B,
        "T": support.T,
        "gradient_accumulation": support.GRAD_ACCUM_STEPS,
        "global_batch_tokens": support.TOTAL_BATCH_SIZE,
        "optimizer_updates": 191,
        "processed_tokens": 191 * support.TOTAL_BATCH_SIZE,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
