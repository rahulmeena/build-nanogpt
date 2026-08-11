#!/usr/bin/env python3
"""Create reproduction plots from saved JSONL metrics or upstream log.txt."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOTAL_BATCH_SIZE = 524288


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_upstream_log(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            step = int(parts[0])
            kind = parts[1]
            value = float(parts[2])
            row = {
                "kind": kind,
                "step": step,
                "tokens": step * TOTAL_BATCH_SIZE,
                "train_loss": None,
                "val_loss": None,
                "hellaswag_accuracy": None,
                "tokens_per_second": None,
            }
            if kind == "train":
                row["tokens"] = (step + 1) * TOTAL_BATCH_SIZE
                row["train_loss"] = value
            elif kind == "val":
                row["val_loss"] = value
            elif kind == "hella":
                row["kind"] = "hellaswag"
                row["hellaswag_accuracy"] = value
            rows.append(row)
    return rows


def resolve_input(path):
    path = Path(path)
    if path.is_dir():
        metrics = path / "metrics.jsonl"
        if metrics.exists():
            return metrics
        upstream_log = path / "log" / "log.txt"
        if upstream_log.exists():
            return upstream_log
        raise SystemExit(f"no metrics.jsonl or log/log.txt found in {path}")
    return path


def load_rows(path):
    path = resolve_input(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path), path
    return read_upstream_log(path), path


def billion_tokens(rows, key):
    xs, ys = [], []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        xs.append(row["tokens"] / 1e9)
        ys.append(value)
    return xs, ys


def smooth(values, window):
    if window <= 1 or len(values) <= 2:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def save_figure(fig, out_dir, stem):
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=240)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def save_loss_plot(rows, out_dir, smoothing_window):
    train_x, train_y = billion_tokens(rows, "train_loss")
    val_x, val_y = billion_tokens(rows, "val_loss")
    fig, ax = plt.subplots(figsize=(9, 5))
    if train_x:
        ax.plot(train_x, train_y, alpha=0.35, linewidth=0.8, label="train loss raw")
        ax.plot(train_x, smooth(train_y, smoothing_window), linewidth=1.6, label=f"train loss smoothed ({smoothing_window})")
    if val_x:
        ax.plot(val_x, val_y, linewidth=1.6, label="validation loss")
    ax.set_xlabel("training tokens processed (billions)")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir, "plot_a_train_val_loss")


def save_val_plot(rows, out_dir):
    val_x, val_y = billion_tokens(rows, "val_loss")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_x, val_y, linewidth=1.8)
    ax.set_xlabel("training tokens processed (billions)")
    ax.set_ylabel("validation cross-entropy loss")
    ax.set_title("Validation loss")
    ax.grid(True, alpha=0.25)
    save_figure(fig, out_dir, "plot_b_val_loss")


def save_hellaswag_plot(rows, out_dir):
    xs, ys = billion_tokens(rows, "hellaswag_accuracy")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, linewidth=1.8)
    ax.set_xlabel("training tokens processed (billions)")
    ax.set_ylabel("HellaSwag normalized accuracy")
    ax.set_title("HellaSwag accuracy")
    ax.grid(True, alpha=0.25)
    save_figure(fig, out_dir, "plot_c_hellaswag_accuracy")


def save_throughput_plot(rows, out_dir):
    xs, ys = [], []
    for row in rows:
        value = row.get("tokens_per_second")
        if value is not None:
            xs.append(row["tokens"] / 1e9)
            ys.append(value)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, linewidth=1.2)
    ax.set_xlabel("training tokens processed (billions)")
    ax.set_ylabel("tokens/sec")
    ax.set_title("Training throughput")
    ax.grid(True, alpha=0.25)
    save_figure(fig, out_dir, "plot_d_throughput")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_or_run_dir")
    parser.add_argument("--out-dir", help="defaults to <run>/plots or metrics parent/plots")
    parser.add_argument("--smoothing-window", type=int, default=50)
    args = parser.parse_args()

    rows, input_path = load_rows(args.metrics_or_run_dir)
    if not rows:
        raise SystemExit(f"no rows found in {input_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif input_path.name == "metrics.jsonl":
        out_dir = input_path.parent / "plots"
    elif input_path.name == "log.txt" and input_path.parent.name == "log":
        out_dir = input_path.parent.parent / "plots"
    else:
        out_dir = input_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_loss_plot(rows, out_dir, args.smoothing_window)
    save_val_plot(rows, out_dir)
    save_hellaswag_plot(rows, out_dir)
    save_throughput_plot(rows, out_dir)
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
