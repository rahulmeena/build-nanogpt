#!/usr/bin/env python3
"""Render the final Experiment 2B5 Markdown report from audited artifacts."""

import argparse
import json
from pathlib import Path


LABELS = (
    "C0_2B2_5M",
    "C1_2B2A_10M",
    "C2_2B2A_15M",
    "C3_2B3_FINAL",
)
SOURCES = ("v16", "v17", "v20", "v24")


def number(value, digits=10):
    return "—" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "—" if value is None else f"{100.0 * value:.4f}%"


def monotonic(values, direction):
    pairs = zip(values, values[1:])
    if direction == "up":
        return all(left <= right for left, right in pairs)
    return all(left >= right for left, right in pairs)


def render(results_dir, results_commit=None):
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "result_summary.json").read_text())
    audit = json.loads((results_dir / "FINAL_AUDIT.json").read_text())
    longitudinal = json.loads(
        (results_dir / "longitudinal_results.json").read_text()
    )["checkpoints"]
    alpha = json.loads((results_dir / "alpha_sweep.json").read_text())[
        "checkpoints"
    ]
    paired = json.loads((results_dir / "paired_losses.json").read_text())[
        "checkpoints"
    ]
    geometry = json.loads((results_dir / "memory_geometry.json").read_text())[
        "checkpoints"
    ]
    cosines = json.loads((results_dir / "generic_mean_cosines.json").read_text())[
        "sources"
    ]
    routing = json.loads((results_dir / "routing_diagnostics.json").read_text())[
        "checkpoints"
    ]
    controls = json.loads((results_dir / "decomposition_controls.json").read_text())[
        "checkpoints"
    ]
    performance = json.loads((results_dir / "performance.json").read_text())
    if not summary.get("passed") or not audit.get("passed"):
        raise SystemExit("refusing to render completion report from a failed audit")

    classification = summary["classification"]
    lines = [
        "# Experiment 2B5 — Final Report",
        "",
        "## Outcome",
        "",
        f"The four-GPU zero-optimizer decomposition completed with classification: **{classification}**. "
        f"The frozen rule was `{summary['classification_rule']}`.",
        "",
        "Each checkpoint ran on its own A100-SXM4-80GB. No optimizer, LR scheduler, "
        "GradScaler, backward pass, optimizer step, parameter update, additional training token, "
        "or HellaSwag evaluation occurred.",
        "",
        "## Central longitudinal result",
        "",
        "| Checkpoint | Tokens | Real | Shuffled | μ-only | Residual-only | Generic retention | Residual retention | Gap α=.25 | Gap α=.5 | Gap α=1 | Gap α=2 | Residual-only gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = longitudinal[label]
        lines.append(
            f"| {row['stage']} | {row['writer_lineage_tokens']:,} | "
            f"{number(row['real_loss'])} | {number(row['shuffled_loss'])} | "
            f"{number(row['mu_only_loss'])} | {number(row['residual_only_loss'])} | "
            f"{percent(row['generic_recovery_retention'])} | "
            f"{percent(row['residual_recovery_retention'])} | "
            f"{number(row['specific_gap_alpha_0.25'])} | "
            f"{number(row['specific_gap_alpha_0.5'])} | "
            f"{number(row['specific_gap_alpha_1'])} | "
            f"{number(row['specific_gap_alpha_2'])} | "
            f"{number(row['residual_only_specific_gap'])} |"
        )

    lines += ["", "## Alpha sweeps", ""]
    for label in LABELS:
        lines += [
            f"### {longitudinal[label]['stage']}",
            "",
            "| α | Real loss | Shuffled loss | Specific gap | Real wins | Shuffled wins | Ties |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in alpha[label]:
            lines.append(
                f"| {row['alpha']:.2f} | {number(row['real_loss'])} | "
                f"{number(row['shuffled_loss'])} | {number(row['specific_gap'])} | "
                f"{row['real_wins']} | {row['shuffled_wins']} | {row['ties']} |"
            )
        residual = paired[label]["residual_only"]
        lines += [
            "",
            f"Residual-only paired gap: {number(residual['mean'])}; real wins "
            f"{residual['real_wins']}/20, shuffled wins {residual['shuffled_wins']}/20.",
            "",
        ]

    lines += [
        "## Decomposition controls",
        "",
        "| Checkpoint | Control | Loss | Δ vs real | Recovery from zero | Recovery retained |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        for control in (
            "zero",
            "real",
            "mu_only",
            "residual_only",
            "residual_only_shuffled",
            "independent_source_residual_shuffle",
        ):
            row = controls[label][control]
            lines.append(
                f"| {longitudinal[label]['stage']} | {control.replace('_', ' ')} | "
                f"{number(row['loss'])} | {number(row['delta_vs_real'])} | "
                f"{number(row['recovery_from_zero'])} | "
                f"{percent(row['recovery_retained_vs_real'])} |"
            )

    lines += [
        "",
        "## Memory geometry",
        "",
        "| Checkpoint | Source | μ RMS | Memory RMS | Residual RMS | Residual/Memory | Mean cosine(memory, μ) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        for source in SOURCES:
            row = geometry[label][source]
            lines.append(
                f"| {longitudinal[label]['stage']} | {source} | "
                f"{number(row['mu_rms'], 6)} | {number(row['memory_rms'], 6)} | "
                f"{number(row['residual_rms'], 6)} | "
                f"{number(row['residual_to_memory_ratio'], 6)} | "
                f"{number(row['mean_cosine_memory_mu'], 6)} |"
            )

    lines += ["", "## Generic-direction cosine matrices", ""]
    short = {
        "C0_2B2_5M": "5M",
        "C1_2B2A_10M": "10M",
        "C2_2B2A_15M": "15M",
        "C3_2B3_FINAL": "Final",
    }
    for source in SOURCES:
        matrix = cosines[source]["cosine_matrix"]
        lines += [
            f"### {source}",
            "",
            "| | 5M | 10M | 15M | Final |",
            "|---|---:|---:|---:|---:|",
        ]
        for left in LABELS:
            values = " | ".join(number(matrix[left][right], 6) for right in LABELS)
            lines.append(f"| {short[left]} | {values} |")
        rms = cosines[source]["mu_rms"]
        lines += [
            "",
            "μ RMS trajectory: "
            + ", ".join(f"{short[label]}={rms[label]:.6f}" for label in LABELS)
            + ".",
            "",
        ]

    lines += [
        "## Reader routing under decomposition",
        "",
        "| Checkpoint | Control | Routing v16/v17/v20/v24 | Entropy | Input RMS v16/v17/v20/v24 | Top-down RMS | Feedback RMS |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for label in LABELS:
        for control in (
            "real",
            "mu_only",
            "residual_only",
            "alpha_0.5_real",
            "alpha_1_real",
            "alpha_2_real",
        ):
            row = routing[label][control]
            weights = "/".join(
                number(row["mean_routing_weight"][source], 6) for source in SOURCES
            )
            rms = "/".join(
                number(row["input_memory_rms"][source], 6) for source in SOURCES
            )
            lines.append(
                f"| {longitudinal[label]['stage']} | {control.replace('_', ' ')} | "
                f"{weights} | {number(row['routing_entropy'], 6)} | {rms} | "
                f"{number(row['topdown_rms'], 6)} | {number(row['feedback_rms'], 6)} |"
            )

    retention = [longitudinal[label]["generic_recovery_retention"] for label in LABELS]
    residual_retention = [
        longitudinal[label]["residual_recovery_retention"] for label in LABELS
    ]
    gaps1 = [longitudinal[label]["specific_gap_alpha_1"] for label in LABELS]
    gaps2 = [longitudinal[label]["specific_gap_alpha_2"] for label in LABELS]
    final = longitudinal["C3_2B3_FINAL"]
    final_alpha2_wins = paired["C3_2B3_FINAL"]["alpha_2"]["real_wins"]
    late_cosines = [
        cosines[source]["cosine_matrix"]["C2_2B2A_15M"]["C3_2B3_FINAL"]
        for source in SOURCES
    ]
    lines += [
        "",
        "## Scientific questions",
        "",
        "### Q1. Does generic recovery increase over training?",
        "",
        ("Yes, monotonically" if monotonic(retention, "up") else "Not monotonically")
        + ": generic-recovery retention was "
        + " → ".join(percent(value) for value in retention)
        + ".",
        "",
        "### Q2. Does residual-only utility decrease?",
        "",
        ("Yes, monotonically" if monotonic(residual_retention, "down") else "Not monotonically")
        + ": residual-recovery retention was "
        + " → ".join(percent(value) for value in residual_retention)
        + ", while the α=1 sequence gaps were "
        + " → ".join(number(value) for value in gaps1)
        + ".",
        "",
        "### Q3. At final 2B3, is sequence memory gone or underweighted?",
        "",
        f"The frozen classification is **{classification}**. Final α=1 and α=2 gaps were "
        f"{final['specific_gap_alpha_1']:.10f} and {final['specific_gap_alpha_2']:.10f}; "
        f"real α=2 won {final_alpha2_wins}/20 paired batches.",
        "",
        "### Q4. Does the generic corrective direction converge?",
        "",
        "The 15M→final per-source mean cosines were "
        + ", ".join(
            f"{source}={value:.6f}" for source, value in zip(SOURCES, late_cosines)
        )
        + ". These values, together with the full matrices above, measure directional convergence without equating geometry with utility.",
        "",
        "### Q5. Is the final generic correction already present early?",
        "",
        f"At 5M and 10M, μ-only retained {percent(retention[0])} and "
        f"{percent(retention[1])} of real recovery, versus {percent(retention[-1])} at final 2B3. "
        "This directly locates how early the calibration-derived correction became useful.",
        "",
        "## Architectural decisions",
        "",
        "### A. Split generic correction and sequence memory into two branches?",
        "",
        ("Yes." if final["generic_recovery_retention"] >= 0.50 else "Not yet.")
        + " The measured generic and centered-residual interventions should be independently controllable in the next approved architecture.",
        "",
        "### B. Freeze or static-initialize the generic branch?",
        "",
        ("Yes, initially." if final["generic_recovery_retention"] >= 0.90 else "Treat this as an open design choice.")
        + " A frozen calibration-derived branch would isolate whether optimization can preserve sequence residuals without generic drift.",
        "",
        "### C. Mean-center the sequence-memory branch?",
        "",
        ("Yes." if final["generic_recovery_retention"] >= 0.50 else "The evidence is insufficient.")
        + " Centering should be evaluated as a distinct branch operation, not interpreted as additive attribution.",
        "",
        "### D. Is next-token cross entropy alone sufficient for actual memory?",
        "",
        ("No: the α=1 correct-sequence gap declined materially across the observed optimization trajectory."
         if gaps1[-1] < max(gaps1[0], gaps1[1]) - 0.020
         else "The observed trajectory does not establish insufficiency conclusively."),
        "",
        "### E. Keep mask-depth experiments paused?",
        "",
        "Yes. Keep deeper-mask training paused until generic compensation and sequence-specific recurrence are separated.",
        "",
        "### F. Keep one-token temporal credit?",
        "",
        "Yes. This diagnostic performed no training and provides no controlled evidence authorizing a longer horizon.",
        "",
        "## Interpretation limits",
        "",
        "μ is a checkpoint-specific estimate from the frozen, disjoint calibration set. It is not a learned model parameter, "
        "not the true FineWeb expectation, and not necessarily the only generic component. μ-only, residual-only, and combined "
        "effects are nonlinear interventions; their recoveries must not be added or treated as Shapley attributions.",
        "",
        "## Integrity and performance",
        "",
    ]
    for key, value in audit["hard_audit_checklist"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        f"Total four-GPU wall time: {performance['total_four_gpu_wall_seconds']:.1f} seconds.",
        "",
        f"2B4 frozen tag: `{audit.get('git_workflow', {}).get('frozen_2b4_tag', 'experiment-2b4-memory-content-mask-depth-final')}`",
        "",
        f"2B4 parent commit: `{audit.get('git_workflow', {}).get('parent_commit', '692fd80ba9fb5e81731397dcd4bf149c3c705d41')}`",
        "",
        f"2B5 branch: `{audit.get('git_workflow', {}).get('branch', 'experiment-2b5-mean-residual-decomposition-4gpu')}`",
        "",
        f"Implementation commit: `{summary['implementation_git_commit']}`",
        "",
    ]
    if results_commit:
        lines += [f"Results commit: `{results_commit}`", ""]
    lines += [
        "Optimizer updates: `0`",
        "",
        "Additional training tokens: `0`",
        "",
        f"## Classification\n\n{classification}",
        "",
        "# EXPERIMENT 2B5 COMPLETE",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--results-commit")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output = results_dir / "EXPERIMENT_2B5_FINAL_REPORT.md"
    output.write_text(render(results_dir, args.results_commit))
    print(output)


if __name__ == "__main__":
    main()
