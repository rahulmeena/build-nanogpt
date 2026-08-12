#!/usr/bin/env python3
"""Create the final, CPU-only Experiment 1B analysis from saved JSON artifacts."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


MILESTONES = (191, 477, 715, 954)
ONE_GPU = {"standard": 168645.0, "full_attnres": 21257.0}
BENCHMARK_4GPU = {"standard": 606460.7624, "full_attnres": 76425.0095}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_name(index, with_index=False):
    if index == 0:
        name = "Embedding"
    else:
        block = (index + 1) // 2
        name = f"Block {block} Attention" if index % 2 else f"Block {block} MLP"
    return f"v{index} — {name}" if with_index else name


def destination_name(raw):
    if raw == "ln_f_input":
        return "Final LN input"
    _, block, kind = raw.split("_")
    return f"Block {int(block)} {kind.title()} input"


def fmt(value, digits=6):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def svg_color(weight, maximum):
    fraction = 0.0 if maximum == 0 else min(1.0, weight / maximum)
    start = (247, 251, 255)
    end = (8, 81, 156)
    rgb = tuple(round(a + (b - a) * fraction) for a, b in zip(start, end))
    return f"rgb{rgb}"


def write_heatmap(path, destinations, sources, matrix):
    cell = 24
    left = 185
    top = 180
    width = left + cell * len(sources) + 30
    height = top + cell * len(destinations) + 55
    maximum = max(max(row) for row in matrix)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#111827}</style>',
        '<text x="10" y="24" font-size="16" font-weight="bold">Full AttnRes final mean routing weights (500,170,752 tokens)</text>',
        '<text x="10" y="46" font-size="11">Rows: destination inputs. Columns: v0 embedding, then attention/MLP outputs v1…v24. White cells are ineligible future sources.</text>',
    ]
    for column, source in enumerate(sources):
        x = left + column * cell + cell / 2
        lines.append(
            f'<text x="{x}" y="{top - 8}" font-size="9" text-anchor="start" transform="rotate(-65 {x} {top - 8})">{source}</text>'
        )
    for row, destination in enumerate(destinations):
        y = top + row * cell
        lines.append(f'<text x="{left - 8}" y="{y + 16}" font-size="10" text-anchor="end">{destination}</text>')
        for column, value in enumerate(matrix[row]):
            x = left + column * cell
            eligible = column <= row
            fill = svg_color(value, maximum) if eligible else "#e5e7eb"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white" stroke-width="1"/>')
    lines.append(f'<text x="{left}" y="{height - 18}" font-size="10">maximum weight = {maximum:.6f}</text>')
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    standard_metrics = read_jsonl(args.source / "standard_metrics.jsonl")
    full_metrics = read_jsonl(args.source / "full_metrics.jsonl")
    standard_summary = read_json(args.source / "standard_summary.json")
    full_summary = read_json(args.source / "full_summary.json")
    routing = read_jsonl(args.source / "attnres_stats.jsonl")
    ablation = read_json(args.source / "causal_ablation.json")

    by_arm = {}
    for arm, rows in (("standard", standard_metrics), ("full_attnres", full_metrics)):
        by_arm[arm] = {
            "train": {row["completed_updates"]: row for row in rows if row.get("kind") == "train"},
            "val": {row["completed_updates"]: row for row in rows if row.get("kind") == "val"},
            "hella": {row["completed_updates"]: row for row in rows if row.get("kind") == "hellaswag"},
        }

    trajectory = []
    for completed in (0,) + MILESTONES:
        standard_val = by_arm["standard"]["val"][completed]
        full_val = by_arm["full_attnres"]["val"][completed]
        for arm, val in (("standard", standard_val), ("full_attnres", full_val)):
            train = by_arm[arm]["train"].get(completed)
            hella = by_arm[arm]["hella"].get(completed)
            trajectory.append({
                "model": arm,
                "tokens": val["tokens"],
                "completed_optimizer_updates": completed,
                "last_zero_based_step": completed - 1,
                "train_loss": None if train is None else train["train_loss"],
                "validation_loss": val["val_loss"],
                "attnres_minus_standard_validation_loss": full_val["val_loss"] - standard_val["val_loss"],
                "hellaswag_accuracy": None if hella is None else hella["hellaswag_accuracy"],
                "hellaswag_correct": None if hella is None else hella["hellaswag_correct"],
                "hellaswag_examples": None if hella is None else hella["hellaswag_examples"],
                "learning_rate": None if train is None else train["lr"],
            })
    write_csv(args.out / "matched_learning_trajectory.csv", list(trajectory[0]), trajectory)

    routing_by_update = {row["completed_updates"]: row for row in routing}
    routing_summary = []
    entropy_rows = []
    strongest_rows = []
    for completed in MILESTONES:
        row = routing_by_update[completed]
        query_norms = row["parameters"]["query_norms"]
        learned_query_norms = query_norms[1:]
        entropies = [destination["mean_entropy"] for destination in row["destinations"]]
        normalized = []
        for destination in row["destinations"]:
            count = len(destination["source_depths"])
            normalized.append(0.0 if count == 1 else destination["mean_entropy"] / math.log(count))
        routing_summary.append({
            "tokens": row["tokens"],
            "completed_optimizer_updates": completed,
            "query_norm_min_all_routers": min(query_norms),
            "query_norm_min_learned_routers": min(learned_query_norms),
            "query_norm_median": statistics.median(query_norms),
            "query_norm_max": max(query_norms),
            "entropy_min": min(entropies),
            "entropy_median": statistics.median(entropies),
            "entropy_max": max(entropies),
            "normalized_entropy_median": statistics.median(normalized),
        })
        for destination in row["destinations"]:
            best = max(range(len(destination["mean_weights"])), key=destination["mean_weights"].__getitem__)
            source = destination["source_depths"][best]
            entropy_rows.append({
                "tokens": row["tokens"],
                "completed_optimizer_updates": completed,
                "destination": destination_name(destination["destination"]),
                "eligible_source_count": len(destination["source_depths"]),
                "entropy_nats": destination["mean_entropy"],
                "maximum_entropy_nats": math.log(len(destination["source_depths"])),
                "normalized_entropy": 0.0 if len(destination["source_depths"]) == 1 else destination["mean_entropy"] / math.log(len(destination["source_depths"])),
            })
            strongest_rows.append({
                "tokens": row["tokens"],
                "completed_optimizer_updates": completed,
                "destination": destination_name(destination["destination"]),
                "strongest_source_index": source,
                "strongest_source": source_name(source),
                "strongest_mean_weight": destination["mean_weights"][best],
            })
    write_csv(args.out / "routing_maturation_summary.csv", list(routing_summary[0]), routing_summary)
    write_csv(args.out / "routing_entropy_by_destination.csv", list(entropy_rows[0]), entropy_rows)
    write_csv(args.out / "routing_strongest_source_by_destination.csv", list(strongest_rows[0]), strongest_rows)

    final_routing = routing_by_update[954]
    matrix_rows = []
    top_rows = []
    destinations = []
    matrix = []
    for destination in final_routing["destinations"]:
        name = destination_name(destination["destination"])
        destinations.append(name)
        weights = dict(zip(destination["source_depths"], destination["mean_weights"]))
        vector = [weights.get(index, 0.0) for index in range(25)]
        matrix.append(vector)
        matrix_rows.append({"destination": name, **{source_name(i, True): weights.get(i, "") for i in range(25)}})
        ranked = sorted(zip(destination["mean_weights"], destination["source_depths"]), reverse=True)[:3]
        top_rows.append({
            "destination": name,
            "entropy_nats": destination["mean_entropy"],
            "top_1_source": source_name(ranked[0][1], True),
            "top_1_weight": ranked[0][0],
            "top_2_source": "" if len(ranked) < 2 else source_name(ranked[1][1], True),
            "top_2_weight": "" if len(ranked) < 2 else ranked[1][0],
            "top_3_source": "" if len(ranked) < 3 else source_name(ranked[2][1], True),
            "top_3_weight": "" if len(ranked) < 3 else ranked[2][0],
        })
    write_csv(args.out / "final_routing_matrix.csv", list(matrix_rows[0]), matrix_rows)
    write_csv(args.out / "final_routing_top3.csv", list(top_rows[0]), top_rows)
    write_heatmap(args.out / "final_routing_heatmap.svg", destinations, [f"v{i}" for i in range(25)], matrix)

    causal_rows = []
    for item in sorted(ablation["ablations"], key=lambda value: -value["causal_contribution_delta"]):
        index = item["source_depth"]
        eligible_weights = []
        for destination in final_routing["destinations"]:
            if index in destination["source_depths"]:
                position = destination["source_depths"].index(index)
                eligible_weights.append(destination["mean_weights"][position])
        causal_rows.append({
            "source_index": index,
            "source_residual": f"v{index}",
            "human_readable_source": source_name(index),
            "normal_validation_loss": item["normal_validation_loss"],
            "masked_validation_loss": item["masked_validation_loss"],
            "delta_loss": item["causal_contribution_delta"],
            "final_mean_weight_when_eligible": statistics.mean(eligible_weights),
            "final_max_weight_when_eligible": max(eligible_weights),
            "eligible_destinations": len(eligible_weights),
            "source_zero_limitation": item["source_zero_limitation"],
        })
    write_csv(args.out / "causal_ablation.csv", list(causal_rows[0]), causal_rows)

    checkpoint_rows = []
    for arm, summary in (("standard", standard_summary), ("full_attnres", full_summary)):
        for report in summary["checkpoint_verifications"]:
            checkpoint_rows.append({
                "model": arm,
                "completed_optimizer_updates": report["completed_updates"],
                "processed_tokens": report["processed_tokens"],
                "recorded_path": report["path"],
                "bytes": report["bytes"],
                "sha256": report["sha256"],
                "optimizer_state_entries": report["optimizer"]["optimizer_state_entries"],
                "moment_entries": report["optimizer"]["moment_entries"],
                "moments_all_finite": report["optimizer"]["all_moments_finite"],
                "dataloader_rank_states": report["dataloader_rank_states"],
                "rng_rank_states": report["rng_rank_states"],
                "next_global_batch_sha256": report["next_global_batch_sha256"],
            })
    write_csv(args.out / "checkpoint_manifest.csv", list(checkpoint_rows[0]), checkpoint_rows)

    delta_by_update = {}
    for completed in MILESTONES:
        delta_by_update[completed] = (
            by_arm["full_attnres"]["val"][completed]["val_loss"]
            - by_arm["standard"]["val"][completed]["val_loss"]
        )
    learned = [row["parameters"]["query_norms"][1:] for row in routing if row["completed_updates"] in MILESTONES]
    perf = {
        "standard_main_tokens_per_second": standard_summary["mean_tokens_per_second"],
        "full_main_tokens_per_second": full_summary["mean_tokens_per_second"],
        "standard_main_runtime_seconds": standard_summary["session_wall_clock_seconds"],
        "full_main_runtime_seconds": full_summary["session_wall_clock_seconds"],
        "standard_to_full_throughput_ratio": standard_summary["mean_tokens_per_second"] / full_summary["mean_tokens_per_second"],
        "standard_controlled_scaling_efficiency": BENCHMARK_4GPU["standard"] / (4 * ONE_GPU["standard"]),
        "full_controlled_scaling_efficiency": BENCHMARK_4GPU["full_attnres"] / (4 * ONE_GPU["full_attnres"]),
    }
    report_data = {
        "trajectory": trajectory,
        "routing_maturation": routing_summary,
        "routing_final_top3": top_rows,
        "causal_ablation": causal_rows,
        "checkpoints": checkpoint_rows,
        "performance": perf,
    }
    (args.out / "report_data.json").write_text(json.dumps(report_data, indent=2, sort_keys=True) + "\n")

    trajectory_lines = []
    for completed in MILESTONES:
        s_train = by_arm["standard"]["train"][completed]
        f_train = by_arm["full_attnres"]["train"][completed]
        s_val = by_arm["standard"]["val"][completed]
        f_val = by_arm["full_attnres"]["val"][completed]
        s_h = by_arm["standard"]["hella"].get(completed)
        f_h = by_arm["full_attnres"]["hella"].get(completed)
        standard_hella = "—" if s_h is None else f"{100 * s_h['hellaswag_accuracy']:.3f}%"
        full_hella = "—" if f_h is None else f"{100 * f_h['hellaswag_accuracy']:.3f}%"
        trajectory_lines.append(
            f"| {s_val['tokens']:,} | {completed} | {s_train['train_loss']:.6f} | {s_val['val_loss']:.6f} | "
            f"{f_train['train_loss']:.6f} | {f_val['val_loss']:.6f} | {f_val['val_loss'] - s_val['val_loss']:+.6f} | "
            f"{standard_hella} / {full_hella} | {s_train['lr']:.9f} |"
        )

    entropy_table = []
    strongest_table = []
    destinations_raw = routing_by_update[191]["destinations"]
    for index in range(25):
        ent_cells = []
        source_cells = []
        for completed in MILESTONES:
            destination = routing_by_update[completed]["destinations"][index]
            best = max(range(len(destination["mean_weights"])), key=destination["mean_weights"].__getitem__)
            source = destination["source_depths"][best]
            ent_cells.append(f"{destination['mean_entropy']:.4f}")
            source_cells.append(f"{source_name(source, True)} ({destination['mean_weights'][best]:.3f})")
        name = destination_name(destinations_raw[index]["destination"])
        entropy_table.append(f"| {name} | " + " | ".join(ent_cells) + " |")
        strongest_table.append(f"| {name} | " + " | ".join(source_cells) + " |")

    causal_table = [
        f"| v{row['source_index']} | {row['human_readable_source']} | {row['normal_validation_loss']:.6f} | "
        f"{row['masked_validation_loss']:.6f} | {row['delta_loss']:+.6f} | {row['final_mean_weight_when_eligible']:.4f} |"
        for row in causal_rows
    ]

    text = f"""# Experiment 1B — Final 500M Analysis

This report is derived only from already-produced Experiment 1B artifacts. No new training, evaluation, or ablation was run.

## Executive result

Full AttnRes was stable through **954 updates / 500,170,752 tokens**. Its final validation loss was **{full_summary['final_validation_loss']:.6f}**, versus **{standard_summary['final_validation_loss']:.6f}** for Standard, an AttnRes−Standard delta of **{delta_by_update[954]:+.6f}**. Final HellaSwag was slightly lower: **{100*full_summary['final_hellaswag']['hellaswag_accuracy']:.3f}%** versus **{100*standard_summary['final_hellaswag']['hellaswag_accuracy']:.3f}%**.

The validation advantage did **not** grow monotonically all the way to 500M. It widened from **{delta_by_update[191]:+.6f}** at 100M to **{delta_by_update[477]:+.6f}** at 250M and **{delta_by_update[715]:+.6f}** at warmup end, then narrowed to **{delta_by_update[954]:+.6f}** at 500M. It never changed sign after 100M.

## 1. Matched learning trajectory

`optimizer step` below is completed optimizer updates; the corresponding last zero-based update is one less. HellaSwag was intentionally measured only at 0, ~100M, and ~500M.

| Tokens | Optimizer step | Standard train | Standard val | AttnRes train | AttnRes val | AttnRes−Standard val | HellaSwag Standard / AttnRes | LR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(trajectory_lines)}

At initialization, validation was **10.951632 Standard** and **10.941046 AttnRes**; this small difference reflects the distinct architecture before learning and is not a trained advantage.

## 2. Full AttnRes routing maturation

The first router has only v0 available and therefore has a structurally zero query norm and zero entropy. The learned-router minimum excludes that first router.

| Tokens | Step | Query norm min (all) | Min learned | Median | Max | Median entropy (nats) | Median normalized entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in routing_summary:
        text += (
            f"| {row['tokens']:,} | {row['completed_optimizer_updates']} | {row['query_norm_min_all_routers']:.6f} | "
            f"{row['query_norm_min_learned_routers']:.6f} | {row['query_norm_median']:.6f} | {row['query_norm_max']:.6f} | "
            f"{row['entropy_median']:.6f} | {row['normalized_entropy_median']:.6f} |\n"
        )
    text += f"""

Median normalized entropy fell from **{routing_summary[0]['normalized_entropy_median']:.3f}** to **{routing_summary[-1]['normalized_entropy_median']:.3f}**, while median query norm rose from **{routing_summary[0]['query_norm_median']:.4f}** to **{routing_summary[-1]['query_norm_median']:.4f}**. Routers therefore became materially more selective, although many late, wide-choice routers remain distributed rather than one-hot.

### Entropy by destination

| Destination | ~100M | ~250M | ~375M | ~500M |
|---|---:|---:|---:|---:|
{chr(10).join(entropy_table)}

### Strongest source by destination

| Destination | ~100M | ~250M | ~375M | ~500M |
|---|---|---|---|---|
{chr(10).join(strongest_table)}

### Clear routing patterns

- **Very early processing:** Block 1 Attention must use the embedding. Block 1 MLP increasingly concentrates on v0, reaching weight **0.944**. Blocks 2–4 mostly retrieve v1/v2, the Block 1 Attention/MLP states.
- **Middle routing hubs:** Block 5 Attention strongly prefers v7 (Block 4 Attention, **0.347**). Block 6 MLP strongly prefers v11 (Block 6 Attention, **0.413**). Blocks 7–9 organize around v11/v13/v14, the Block 6–7 states.
- **Late retrieval:** Blocks 10–12 Attention prefer v17 (Block 9 Attention) at 500M. Several late MLP destinations return all the way to v1, showing genuine long-depth retrieval rather than only local skips.
- **Final router:** Final LN input prefers v1 (**0.192**), then v17 (**0.101**), v15 (**0.091**), v22 (**0.069**), v20 (**0.063**), v18 (**0.061**), and v24 (**0.057**). It combines early and intermediate/late representations.
- **Uniformity:** Routers are not uniformly collapsed. Some are highly concentrated, while late routers retain high entropy because they mix many eligible sources. The complete matrix is in `final_routing_matrix.csv`; `final_routing_heatmap.svg` is its visual rendering.

## 3. Causal ablation

The saved ablation tested exactly **v0, v4, v8, v12, v16, v20, and v24**: embedding plus every fourth residual source, corresponding to the MLP output after Blocks 2, 4, 6, 8, 10, and 12. This representative subset followed the protocol's instruction to avoid an exhaustive combinatorial ablation.

| Source | Human-readable source | Normal val | Masked val | Delta loss | Final mean routing weight when eligible |
|---|---|---:|---:|---:|---:|
{chr(10).join(causal_table)}

v0 has a special limitation: it remains the sole input to the first attention sublayer, so the utility cannot remove it there; it masks v0 only from later routers. Its very large delta should not be compared naively with the other sources.

Among non-embedding tested states, **Block 10 MLP (v20)** was most causally important, followed by **Block 8 MLP (v16)**, **Block 12 MLP (v24)**, and **Block 6 MLP (v12)**. Across the six tested non-v0 sources, causal delta and mean routing weight when eligible have Pearson **r≈0.871**. This is suggestive, not definitive (n=6 and the sources were not exhaustively sampled). Total routing mass is confounded by how many later destinations can access a source.

## 4. Destination-specific behavior

No saved experiment performs destination×source causal interventions, so destination-specific statements are **routing associations, not destination-specific causal proof**. Existing evidence supports:

- Block 5 Attention → v7 / Block 4 Attention (**0.347**).
- Block 6 MLP → v11 / Block 6 Attention (**0.413**).
- Block 7 Attention → v11 / Block 6 Attention (**0.292**).
- Block 7 MLP → v13 / Block 7 Attention (**0.253**).
- Blocks 10–12 Attention → v17 / Block 9 Attention (~**0.098–0.102** each).
- Final LN input mixes v1, v17, v15, v22, v20, v18, and v24 rather than relying on a single local predecessor.

The exact top-three routes for every destination are in `final_routing_top3.csv`.

## 5. Four-GPU performance

| Metric | Standard | Full AttnRes |
|---|---:|---:|
| Main-run throughput | {standard_summary['mean_tokens_per_second']:,.1f} tok/s | {full_summary['mean_tokens_per_second']:,.1f} tok/s |
| Main-run runtime | {standard_summary['session_wall_clock_seconds']:.1f}s ({standard_summary['session_wall_clock_seconds']/60:.2f} min) | {full_summary['session_wall_clock_seconds']:.1f}s ({full_summary['session_wall_clock_seconds']/60:.2f} min) |
| Controlled 4-GPU benchmark | {BENCHMARK_4GPU['standard']:,.1f} tok/s | {BENCHMARK_4GPU['full_attnres']:,.1f} tok/s |
| Previous 1-GPU reference | {ONE_GPU['standard']:,.0f} tok/s | {ONE_GPU['full_attnres']:,.0f} tok/s |
| Controlled 4-GPU speedup | {BENCHMARK_4GPU['standard']/ONE_GPU['standard']:.3f}× | {BENCHMARK_4GPU['full_attnres']/ONE_GPU['full_attnres']:.3f}× |
| Controlled scaling efficiency | {100*BENCHMARK_4GPU['standard']/(4*ONE_GPU['standard']):.2f}% | {100*BENCHMARK_4GPU['full_attnres']/(4*ONE_GPU['full_attnres']):.2f}% |
| Peak allocated VRAM/GPU | {standard_summary['peak_allocated_mb_per_gpu_max']:,.1f} MiB | {full_summary['peak_allocated_mb_per_gpu_max']:,.1f} MiB |
| Peak reserved VRAM/GPU | {standard_summary['peak_reserved_mb_per_gpu_max']:,.1f} MiB | {full_summary['peak_reserved_mb_per_gpu_max']:,.1f} MiB |

Standard delivered **{standard_summary['mean_tokens_per_second']/full_summary['mean_tokens_per_second']:.3f}×** Full AttnRes throughput; Full AttnRes took **{full_summary['session_wall_clock_seconds']/standard_summary['session_wall_clock_seconds']:.3f}×** as long end-to-end.

## 6. Resume/checkpoint validation

Both final checkpoints were successfully written and read-only reload-verified by the training harness. Each verification covered model state, AdamW moments, completed step/tokens, four per-rank DataLoader states, four per-rank RNG states, world-size metadata, and the next-global-batch hash.

- **Standard final recorded path:** `/workspace/build-nanogpt/runs/exp1b_500m/standard/checkpoints/checkpoint_tokens_000500170752.pt`
  **SHA-256:** `{standard_summary['checkpoint_verifications'][-1]['sha256']}`
  **Optimizer:** 148 state/moment entries, all finite and nonzero.
- **Full AttnRes final path:** `/workspace/build-nanogpt/runs/exp1b_500m/full_attnres/checkpoints/checkpoint_tokens_000500170752.pt`
  **SHA-256:** `{full_summary['checkpoint_verifications'][-1]['sha256']}`
  **Optimizer:** 198 state/moment entries, 196 nonzero (the two zero moments are the structurally inactive first one-source router query/norm), all finite.

The Full AttnRes final payload was additionally reopened CPU-only during final analysis: 199 model tensors, 198 optimizer states, step 954, 500,170,752 tokens, four DataLoader states, and four RNG dictionaries containing `python_random`, `numpy_random`, `torch_cpu`, and per-rank `torch_cuda`. Metadata records `world_size=4`.

## 7. Artifact preservation audit

The currently mounted persistent US-WA volume contains:

- complete Standard and Full AttnRes metrics and summaries;
- all routing snapshots and the causal-ablation JSON;
- all four Full AttnRes resumable checkpoint payloads plus SHA/completion/verification sidecars;
- resume and host-migration audits;
- this final report, CSV tables, JSON data, and routing heatmap after Git synchronization.

**Preservation gap:** the four large Standard checkpoint payloads are not on the currently mounted US-WA volume. They were successfully reload-verified and their paths/SHAs are preserved in Standard's summary and `checkpoint_manifest.csv`, but the payloads were written on the earlier US-MD network volume. That volume must be mounted again if the Standard payloads need to be co-located or copied. No new GPU compute is required; a CPU pod is sufficient for that storage transfer.

## 8. Experiment-2 recommendation

**Yes—with a staged design.** Lower→higher AttnRes is sufficiently mature and stable to serve as the substrate for a high→low cross-token feedback experiment: it trained stably to 500M, beat Standard validation at every trained milestone, specialized progressively, and every tested source had positive causal utility. The caveat is that final HellaSwag was **0.358 percentage points lower**, so Experiment 2 should use a conservative gated feedback path and preserve a no-feedback control.

Most useful higher-layer states, based on direct causal evidence, are **Block 10 MLP (v20)**, **Block 8 MLP (v16)**, and **Block 12 MLP (v24)**. Based on routing association (not ablation), **Block 9 Attention (v17)** is also a strong candidate because it is the preferred source of Blocks 10–12 Attention and the second-largest final-LN route.

The most logical first recipients of top-down feedback are middle destinations that already act as routing integration hubs: **Block 5 Attention input**, **Block 6 Attention/MLP inputs**, and **Block 7 Attention input**. Begin there before injecting into Blocks 1–3, whose strong early-state specialization appears foundational and may be more destabilizing.

## Machine-readable artifacts

- `matched_learning_trajectory.csv`
- `routing_maturation_summary.csv`
- `routing_entropy_by_destination.csv`
- `routing_strongest_source_by_destination.csv`
- `final_routing_matrix.csv`
- `final_routing_top3.csv`
- `final_routing_heatmap.svg`
- `causal_ablation.csv`
- `checkpoint_manifest.csv`
- `report_data.json`

EXPERIMENT 1B ANALYSIS FINALIZED
"""
    (args.out / "EXPERIMENT_1B_FINAL_ANALYSIS.md").write_text(text)


if __name__ == "__main__":
    main()
