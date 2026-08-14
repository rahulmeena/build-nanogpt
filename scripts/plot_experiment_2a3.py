#!/usr/bin/env python3
"""Generate the ten frozen Experiment 2A3 plots plus CSV/JSON plot data."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def save(fig, out_dir, stem):
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summary["trajectory"]
    tokens_m = [row["tokens"] / 1e6 for row in rows]
    real = [row["real_feedback_loss"] for row in rows]
    shuffled = [row["shuffled_feedback_loss"] for row in rows]
    total = [100 * row["total_recovery_fraction"] for row in rows]
    specific = [100 * row["sequence_specific_recovery_fraction"] for row in rows]
    share = [100 * row["sequence_specific_share_of_total_recovery"] for row in rows]
    gap = [row["real_minus_shuffled"] for row in rows]
    routing = [row["routing"] for row in rows]

    plot_rows = []
    for row in rows:
        router = row["routing"]
        plot_rows.append({
            "completed_updates": row["completed_updates"],
            "tokens": row["tokens"],
            "real_feedback_loss": row["real_feedback_loss"],
            "shuffled_feedback_loss": row["shuffled_feedback_loss"],
            "real_minus_shuffled": row["real_minus_shuffled"],
            "total_recovery_fraction": row["total_recovery_fraction"],
            "sequence_specific_recovery_fraction": row["sequence_specific_recovery_fraction"],
            "sequence_specific_share_of_total_recovery": row["sequence_specific_share_of_total_recovery"],
            "gate": router["gate"],
            "gate_coefficient": router["gate_coefficient"],
            "query_norm": router["query_norm"],
            "rmsnorm_displacement": router["rmsnorm_displacement"],
            "routing_entropy": router["mean_tokenwise_entropy"],
            "normalized_routing_entropy": router["normalized_entropy"],
            **{f"weight_{source}": router["mean_weights"][source] for source in ("v16", "v17", "v20", "v24")},
        })
    (out_dir / "plot_data.json").write_text(json.dumps(plot_rows, indent=2, sort_keys=True) + "\n")
    with (out_dir / "plot_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)

    def line_plot(values, ylabel, stem):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(tokens_m, values, marker="o")
        ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel=ylabel)
        ax.grid(alpha=0.25)
        save(fig, out_dir, stem)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tokens_m, real, marker="o", label="Real aligned feedback")
    ax.plot(tokens_m, shuffled, marker="o", label="Shuffled feedback")
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Validation loss")
    ax.grid(alpha=0.25); ax.legend()
    save(fig, out_dir, "plot_1_real_shuffled_validation")
    line_plot(total, "Total recovery (% damage)", "plot_2_total_recovery_fraction")
    line_plot(specific, "Sequence-specific recovery (% damage)", "plot_3_specific_recovery_fraction")
    line_plot(share, "Sequence-specific share of total recovery (%)", "plot_4_specific_share")
    line_plot(gap, "Real minus shuffled validation loss", "plot_5_real_minus_shuffled_gap")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tokens_m, [r["gate"] for r in routing], marker="o", label="gate")
    ax.plot(tokens_m, [r["gate_coefficient"] for r in routing], marker="o", label="tanh(gate)")
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Gate value")
    ax.grid(alpha=0.25); ax.legend()
    save(fig, out_dir, "plot_6_gate")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tokens_m, [r["query_norm"] for r in routing], marker="o", label="query norm")
    ax.plot(tokens_m, [r["rmsnorm_displacement"] for r in routing], marker="o", label="RMSNorm displacement")
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Norm")
    ax.grid(alpha=0.25); ax.legend()
    save(fig, out_dir, "plot_7_query_rmsnorm")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tokens_m, [r["mean_tokenwise_entropy"] for r in routing], marker="o", label="entropy")
    ax.plot(tokens_m, [r["normalized_entropy"] for r in routing], marker="o", label="normalized entropy")
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Routing entropy")
    ax.grid(alpha=0.25); ax.legend()
    save(fig, out_dir, "plot_8_routing_entropy")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for source in ("v16", "v17", "v20", "v24"):
        ax.plot(tokens_m, [r["mean_weights"][source] for r in routing], marker="o", label=source)
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Mean routing weight")
    ax.grid(alpha=0.25); ax.legend(ncol=4)
    save(fig, out_dir, "plot_9_source_weights")

    hella = summary["hellaswag"]
    bars = [
        ("Standard\n500M historical", hella["historical_references"]["standard_gpt2_500m"]["accuracy"]),
        ("Full AttnRes\n500M historical", hella["historical_references"]["full_attnres_500m"]["accuracy"]),
        ("Full context\ncurrent", hella["accuracy"]["full_context"]),
        ("Masked/no\nfeedback", hella["accuracy"]["masked_l1_no_feedback"]),
        ("Real teacher\nfeedback", hella["accuracy"]["real_feedback"]),
        ("Gate zero", hella["accuracy"]["zero_feedback"]),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    values = [100 * value for _, value in bars]
    rectangles = ax.bar([label for label, _ in bars], values)
    ax.bar_label(rectangles, fmt="%.2f%%", padding=3)
    ax.set_ylabel("HellaSwag normalized accuracy (%)")
    ax.set_ylim(0, max(values) + 4); ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir, "plot_10_hellaswag")

    manifest = {
        "summary": str(summary_path.resolve()),
        "plot_data_csv": "plot_data.csv",
        "plot_data_json": "plot_data.json",
        "plots": sorted(path.name for path in out_dir.iterdir()),
    }
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
