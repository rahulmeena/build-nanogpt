#!/usr/bin/env python3
"""Summarize a completed reproduction run without modifying its artifacts."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


TOTAL_BATCH_SIZE = 524288


def read_metrics(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluation_rows(rows):
    train_by_tokens = {
        row["tokens"]: row for row in rows if row.get("kind") == "train"
    }
    evaluations = {}
    for row in rows:
        if row.get("kind") not in {"val", "hellaswag"}:
            continue
        key = (row["step"], row["tokens"])
        output = evaluations.setdefault(
            key,
            {
                "step": row["step"],
                "tokens_billion": row["tokens"] / 1e9,
                "train_loss": None,
                "val_loss": None,
                "hellaswag_accuracy": None,
                "lr": None,
                "grad_norm": None,
            },
        )
        output["val_loss"] = row.get("val_loss") or output["val_loss"]
        output["hellaswag_accuracy"] = (
            row.get("hellaswag_accuracy") or output["hellaswag_accuracy"]
        )

    for (_, tokens), output in evaluations.items():
        train = train_by_tokens.get(tokens)
        if train:
            for key in ("train_loss", "lr", "grad_norm"):
                output[key] = train.get(key)
    return [evaluations[key] for key in sorted(evaluations)]


def summarize(rows):
    train = [row for row in rows if row.get("kind") == "train"]
    validation = [row for row in rows if row.get("kind") == "val"]
    hellaswag = [row for row in rows if row.get("kind") == "hellaswag"]
    steady = [
        row
        for row in train
        if row["step"] % 250 != 0 and row["step"] != train[-1]["step"]
    ]
    total_step_seconds = sum(row["step_time_ms"] for row in train) / 1000
    best_validation = min(validation, key=lambda row: row["val_loss"])
    best_hellaswag = max(hellaswag, key=lambda row: row["hellaswag_accuracy"])
    peak_vram_mb = max(
        row["gpu_peak_mb"] for row in rows if row.get("gpu_peak_mb") is not None
    )
    return {
        "evaluation_points": len(validation),
        "final_train": train[-1],
        "final_validation": validation[-1],
        "final_hellaswag": hellaswag[-1],
        "best_validation": best_validation,
        "best_hellaswag": best_hellaswag,
        "optimizer_steps": len(train),
        "total_processed_tokens": train[-1]["tokens"],
        "timed_runtime_seconds": total_step_seconds,
        "aggregate_tokens_per_second": train[-1]["tokens"] / total_step_seconds,
        "steady_state_tokens_per_second": (
            len(steady) * TOTAL_BATCH_SIZE
            / (sum(row["step_time_ms"] for row in steady) / 1000)
        ),
        "steady_state_median_tokens_per_second": statistics.median(
            row["tokens_per_second"] for row in steady
        ),
        "peak_vram_mb": peak_vram_mb,
        "all_metrics_finite": all(
            math.isfinite(value)
            for row in rows
            for value in row.values()
            if isinstance(value, float)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    rows = read_metrics(args.metrics)
    csv_path = Path(args.csv)
    summary_path = Path(args.summary)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "step",
        "tokens_billion",
        "train_loss",
        "val_loss",
        "hellaswag_accuracy",
        "lr",
        "grad_norm",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(evaluation_rows(rows))

    summary_path.write_text(json.dumps(summarize(rows), indent=2) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
