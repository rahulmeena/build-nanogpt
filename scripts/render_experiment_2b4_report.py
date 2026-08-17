#!/usr/bin/env python3
"""Render the final Experiment 2B4 Markdown report from audited JSON artifacts."""

import argparse
import json
from pathlib import Path


CONTROL_ORDER = (
    "zero",
    "real",
    "coherent_shuffle",
    "independent_source_shuffle",
    "batch_mean",
    "position_template",
    "global_template",
    "norm_random",
    "lag8",
    "lag32",
)


def number(value, digits=10):
    return "—" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "—" if value is None else f"{100.0 * value:.4f}%"


def part_a_answer_rows(part_a):
    controls = part_a["controls"]
    mechanism = part_a["mechanism_metrics"]
    coherent = controls["coherent_shuffle"]
    independent = controls["independent_source_shuffle"]
    position = controls["position_template"]
    global_template = controls["global_template"]
    random_control = controls["norm_random"]
    lag8 = controls["lag8"]["restricted"]
    return [
        (
            "A1 — Exact sequence identity",
            f"The measured coherent-shuffle penalty was {coherent['delta_vs_real']:.10f}; "
            f"real won {coherent['real_wins']}/20 paired batches. Exact row identity therefore "
            "contributed exactly this measured amount; the control does not support attributing "
            "the remaining recovery to row identity.",
        ),
        (
            "A2 — Coherence among sources",
            f"Breaking the common donor across v16/v17/v20/v24 changed loss by "
            f"{mechanism['cross_source_coherence_value']:+.10f} relative to coherent shuffling "
            f"(independent-source loss {independent['loss']:.10f}).",
        ),
        (
            "A3 — Absolute-position-conditioned generic state",
            f"The position template retained {percent(position['fraction_of_real_recovery_retained'])} "
            f"of real recovery at loss {position['loss']:.10f}.",
        ),
        (
            "A4 — Constant generic template",
            f"The global template retained {percent(global_template['fraction_of_real_recovery_retained'])} "
            f"of real recovery at loss {global_template['loss']:.10f}.",
        ),
        (
            "A5 — Memory-vector norms",
            f"Norm-matched random directions retained {percent(random_control['fraction_of_real_recovery_retained'])} "
            f"of real recovery at loss {random_control['loss']:.10f}. This directly measures what "
            "norms alone can supply without learned direction/content.",
        ),
        (
            "A6 — Temporal alignment versus row identity",
            f"Lag-8 increased restricted loss over matched real by {lag8['specific_delta']:.10f}, "
            f"versus the coherent cross-sequence identity gap of {coherent['delta_vs_real']:.10f}; "
            f"the difference was {mechanism['lag8_minus_cross_sequence_identity_gap']:+.10f}.",
        ),
    ]


def recommendations(summary):
    part_b = summary["part_b"]
    part_c = summary["part_c"]
    classification = summary["classification"]
    strong = part_b["strong_support"]
    b1 = (
        "Yes. The preregistered strong-support criterion passed, so a separately approved "
        "Blocks-1–2 masked training experiment is supported."
        if strong
        else "No. The preregistered diagnostic did not establish strong sequence-specific mask pressure."
    )
    b2 = (
        "Keep feedback only at Block 1 initially; Part B isolated that channel. Test a second "
        "destination only in a separate protocol."
        if strong
        else "Do not add a second destination here. If masking is revisited, first preserve the "
        "single Block-1 destination so the comparison remains identified."
    )
    if strong and part_c["status"] == "RUN":
        best = str(part_c["best_depth"])
        final_gap = part_c["depths"][best]["specific_gap"]
        writer_gap = part_b["depths"][best]["specific_gap"]
        if final_gap >= writer_gap:
            source = (
                "Use the final 2B3 checkpoint: conditional Part C retained at least the primary "
                "10M checkpoint's specificity at the selected depth."
            )
        else:
            source = (
                "Use the 10M high-specificity writer checkpoint. It is the frozen source that "
                "produced the stronger primary mask-depth specificity result."
            )
    elif strong:
        source = "Use the 10M high-specificity writer checkpoint that produced the primary result."
    else:
        source = (
            "Do not select a training source yet. If a follow-up diagnostic is approved, retain "
            "the 10M checkpoint as the high-specificity reference rather than silently advancing lineage."
        )
    b4 = (
        "Yes. Experiment 2B4 performed no training and provides no evidence authorizing a longer "
        "credit horizon; keep temporal credit at one token."
    )
    b5 = (
        "No. Keep coherent shuffled memory, and make the global-template, norm-matched-random, "
        "and same-sequence lag-8 controls mandatory when mechanism attribution is claimed."
    )
    return [
        ("B1 — Train Blocks 1–2 masked next?", b1),
        ("B2 — Feedback destination", b2),
        ("B3 — Source checkpoint", source),
        ("B4 — Temporal credit", b4),
        ("B5 — Future control set", b5),
        ("Observed classification", classification),
    ]


def render(results_dir):
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "result_summary.json").read_text())
    audit = json.loads((results_dir / "FINAL_AUDIT.json").read_text())
    if not summary.get("passed") or not audit.get("passed"):
        raise SystemExit("refusing to render a completion report from a failed audit")
    part_a = summary["part_a"]
    part_b = summary["part_b"]
    part_c = summary["part_c"]
    controls = part_a["controls"]
    lines = [
        "# Experiment 2B4 — Final Report",
        "",
        "## Outcome",
        "",
        f"The zero-optimizer diagnostic completed with classification: **{summary['classification']}**. "
        f"Part A measured a real recurrent loss of {part_a['real_loss']:.10f} against gate-zero "
        f"{part_a['zero_loss']:.10f}. Part B's specificity trajectory was classified as "
        f"{part_b['trajectory_classification']}.",
        "",
        "No optimizer, scheduler, GradScaler, backward pass, optimizer step, parameter update, "
        "training continuation, or HellaSwag evaluation occurred.",
        "",
        "## Part A — Memory content",
        "",
        "| Control | Loss | Δ vs real | Recovery | Real recovery retained | Real wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for control in CONTROL_ORDER:
        row = controls[control]
        lines.append(
            f"| {row['label']} | {number(row['loss'])} | {number(row['delta_vs_real'])} | "
            f"{number(row['recovery'])} | {percent(row['fraction_of_real_recovery_retained'])} | "
            f"{row['real_wins']}/20 |"
        )
    lines += [
        "",
        "### Paired statistics versus real",
        "",
        "| Control | Real wins | Control wins | Ties | Mean | Median | Sample std | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for control in CONTROL_ORDER:
        row = controls[control]
        paired = row["paired_statistics_vs_real"]
        lines.append(
            f"| {row['label']} | {paired['real_wins']} | {paired['shuffled_wins']} | "
            f"{paired['ties']} | {number(paired['mean'])} | {number(paired['median'])} | "
            f"{number(paired['sample_standard_deviation'])} | {number(paired['minimum'])} | "
            f"{number(paired['maximum'])} |"
        )
    lines += [
        "",
        "### Lag-restricted comparisons",
        "",
        "| Control | Target subset | Lag loss | Matched real loss | Specific delta | Real wins |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for control in ("lag8", "lag32"):
        row = controls[control]["restricted"]
        lines.append(
            f"| {controls[control]['label']} | {row['target_positions']} | {number(row['loss'])} | "
            f"{number(row['real_loss'])} | {number(row['specific_delta'])} | "
            f"{row['paired_statistics_vs_real']['real_wins']}/20 |"
        )
    lines += [
        "",
        "### Routing and state diagnostics",
        "",
        "| Control | Input RMS v16/v17/v20/v24 | Routing v16/v17/v20/v24 | Entropy | Top-down RMS | Feedback RMS |",
        "|---|---|---|---:|---:|---:|",
    ]
    for control in CONTROL_ORDER:
        metrics = controls[control]["routing_state_diagnostics"]
        input_rms = "/".join(number(metrics["input_memory_rms"][f"v{depth}"], 6) for depth in (16, 17, 20, 24))
        routing = "/".join(number(metrics["routing_weights"][f"v{depth}"], 6) for depth in (16, 17, 20, 24))
        lines.append(
            f"| {controls[control]['label']} | {input_rms} | {routing} | "
            f"{number(metrics['routing_entropy'], 6)} | {number(metrics['topdown_rms'], 6)} | "
            f"{number(metrics['feedback_rms'], 6)} |"
        )
    lines += ["", "## Part A — What the recurrent signal contains", ""]
    for heading, answer in part_a_answer_rows(part_a):
        lines += [f"### {heading}", "", answer, ""]

    lines += [
        "## Part B — Mask-depth sweep",
        "",
        "| Mask depth | Zero loss | Real loss | Shuffled loss | Specific gap | Recovery % | Specific share | Real wins |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in (1, 2, 3, 4):
        row = part_b["depths"][str(depth)]
        lines.append(
            f"| {depth} | {number(row['zero_loss'])} | {number(row['real_loss'])} | "
            f"{number(row['shuffled_loss'])} | {number(row['specific_gap'])} | "
            f"{percent(row['real_recovery_fraction'])} | {percent(row['specific_share_of_recovery'])} | "
            f"{row['real_wins']}/20 |"
        )
    lines += [
        "",
        "### Gap minus depth 1",
        "",
        "| Depth | Gap minus depth 1 |",
        "|---:|---:|",
    ]
    for depth in (2, 3, 4):
        lines.append(f"| {depth} | {number(part_b['gap_minus_depth1'][str(depth)])} |")
    lines += ["", "## Gap trajectory", ""]
    for depth in (1, 2, 3, 4):
        lines.append(f"- depth {depth}: {part_b['gap_trajectory'][str(depth)]:.10f}")
    lines += [
        "",
        f"Trajectory: **{part_b['trajectory_classification']}**. Strong-support depths: "
        f"{part_b['strong_support_depths'] or 'none'}.",
        "",
        "## Conditional Part C",
        "",
        f"**{part_c['status']}**",
        "",
    ]
    if part_c["status"] == "RUN":
        lines += [
            "| Mask depth | Zero loss | Real loss | Shuffled loss | Specific gap | Real wins |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for depth in (1, part_c["best_depth"]):
            row = part_c["depths"][str(depth)]
            lines.append(
                f"| {depth} | {number(row['zero_loss'])} | {number(row['real_loss'])} | "
                f"{number(row['shuffled_loss'])} | {number(row['specific_gap'])} | "
                f"{row['real_wins']}/20 |"
            )
        lines += [
            "",
            f"The deeper-mask specific-gap increase over final-2B3 depth 1 was "
            f"{part_c['gap_increase_over_depth1']:+.10f}.",
            "",
        ]
    else:
        lines += [part_c["reason"], ""]

    lines += ["## Integrity", ""]
    for key, value in audit["hard_audit_checklist"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        f"Final 2B3 checkpoint: `{summary['source_checkpoints']['final_2b3']['sha256']}`",
        "",
        f"2B2A 10M checkpoint: `{summary['source_checkpoints']['writer_10m']['sha256']}`",
        "",
        f"Canonical validation: `{summary['canonical_validation_sha256']}`",
        "",
        "## Classification",
        "",
        summary["classification"],
        "",
        "## Recommendations",
        "",
    ]
    for heading, answer in recommendations(summary):
        lines += [f"### {heading}", "", answer, ""]
    lines += [
        "No optimizer update may be launched automatically after this diagnostic.",
        "",
        "# EXPERIMENT 2B4 COMPLETE",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output = results_dir / "EXPERIMENT_2B4_FINAL_REPORT.md"
    output.write_text(render(results_dir))
    print(output)


if __name__ == "__main__":
    main()
