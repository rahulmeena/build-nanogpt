#!/usr/bin/env python3
"""Generate the six required matched Experiment 1 plots."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def series(records, key, xkey="tokens"):
    selected = [(row.get(xkey), row.get(key)) for row in records if row.get(xkey) is not None and row.get(key) is not None]
    return [x for x, _ in selected], [y for _, y in selected]


def save_comparison(standard, full, key, ylabel, title, out_dir, stem, xkey="tokens", xlabel="processed tokens"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, records in (("standard GPT-2", standard), ("Full AttnRes", full)):
        xs, ys = series(records, key, xkey=xkey)
        ax.plot(xs, ys, marker="o" if key != "train_loss" else None, linewidth=1.5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=240)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def latest_stats(full_dir):
    jsonl = Path(full_dir) / "attnres_stats.jsonl"
    if jsonl.exists():
        records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        return records[-1]["destinations"]
    smoke = Path(full_dir) / "attnres_post_smoke_stats.json"
    if smoke.exists():
        return json.loads(smoke.read_text())
    raise SystemExit(f"no AttnRes statistics found in {full_dir}")


def save_attnres_plots(stats, out_dir):
    matrix = np.full((len(stats), 25), np.nan)
    entropies = []
    labels = []
    for row_index, row in enumerate(stats):
        weights = row["mean_weights"]
        matrix[row_index, :len(weights)] = weights
        entropies.append(row["mean_entropy"])
        labels.append(row["destination"])

    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("source residual depth")
    ax.set_ylabel("destination sublayer")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Full AttnRes mean depth-attention weights")
    fig.colorbar(image, ax=ax, label="mean attention weight")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_5_depth_attention_heatmap.png", dpi=240)
    fig.savefig(out_dir / "plot_5_depth_attention_heatmap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(len(entropies)), entropies, marker="o", linewidth=1.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("mean depth-attention entropy")
    ax.set_title("Full AttnRes entropy by destination")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_6_depth_attention_entropy.png", dpi=240)
    fig.savefig(out_dir / "plot_6_depth_attention_entropy.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("standard_run_dir")
    parser.add_argument("full_attnres_run_dir")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    standard = rows(args.standard_run_dir)
    full = rows(args.full_attnres_run_dir)
    save_comparison(standard, full, "val_loss", "validation cross-entropy", "Validation loss vs processed tokens", out_dir, "plot_1_val_loss_vs_tokens")
    save_comparison(standard, full, "train_loss", "training cross-entropy", "Training loss vs processed tokens", out_dir, "plot_2_train_loss_vs_tokens")
    save_comparison(standard, full, "hellaswag_accuracy", "HellaSwag normalized accuracy", "HellaSwag vs processed tokens", out_dir, "plot_3_hellaswag_vs_tokens")
    save_comparison(standard, full, "val_loss", "validation cross-entropy", "Validation loss vs wall-clock time", out_dir, "plot_4_val_loss_vs_wall_clock", xkey="wall_clock_seconds", xlabel="wall-clock seconds")
    save_attnres_plots(latest_stats(args.full_attnres_run_dir), out_dir)
    print(f"wrote Experiment 1 plots to {out_dir}")


if __name__ == "__main__":
    main()
