#!/usr/bin/env python3
"""Generate the seven frozen Experiment 2A2 result plots."""

import argparse
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
    summary = json.loads(Path(args.summary).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summary["trajectory"]
    tokens_m = [row["tokens"] / 1e6 for row in rows]
    real = [row["real_feedback_loss"] for row in rows]
    shuffled = [row["shuffled_feedback_loss"] for row in rows]
    total = [100 * row["total_recovery_fraction"] for row in rows]
    specific = [100 * row["sequence_specific_recovery_fraction"] for row in rows]
    share = [100 * row["sequence_specific_share_of_total_recovery"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tokens_m, real, marker="o", label="Real aligned feedback")
    ax.plot(tokens_m, shuffled, marker="o", label="Shuffled feedback")
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, out_dir, "plot_1_real_shuffled_validation")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tokens_m, total, marker="o")
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Total recovery (% damage)")
    ax.grid(alpha=0.25)
    save(fig, out_dir, "plot_2_total_recovery_fraction")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tokens_m, specific, marker="o")
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Sequence-specific recovery (% damage)")
    ax.grid(alpha=0.25)
    save(fig, out_dir, "plot_3_specific_recovery_fraction")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tokens_m, share, marker="o")
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Sequence-specific share of total recovery (%)")
    ax.grid(alpha=0.25)
    save(fig, out_dir, "plot_4_specific_share")

    routing = [row["routing"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    axes[0].plot(tokens_m, [row["gate"] for row in routing], marker="o")
    axes[0].set_ylabel("Gate")
    axes[1].plot(tokens_m, [row["query_norm"] for row in routing], marker="o")
    axes[1].set_ylabel("Query norm")
    axes[2].plot(tokens_m, [row["mean_tokenwise_entropy"] for row in routing], marker="o")
    axes[2].set_ylabel("Routing entropy")
    axes[2].set_xlabel("Experiment-2 student tokens (millions)")
    for ax in axes:
        ax.grid(alpha=0.25)
    save(fig, out_dir, "plot_5_gate_query_entropy")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for source in ("v16", "v17", "v20", "v24"):
        ax.plot(
            tokens_m,
            [row["mean_weights"][source] for row in routing],
            marker="o",
            label=source,
        )
    ax.set(xlabel="Experiment-2 student tokens (millions)", ylabel="Mean routing weight")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4)
    save(fig, out_dir, "plot_6_source_weights")

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
    ax.set_ylim(0, max(values) + 4)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir, "plot_7_hellaswag")

    manifest = {
        "summary": str(Path(args.summary).resolve()),
        "plots": sorted(path.name for path in out_dir.iterdir()),
    }
    (out_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
