#!/usr/bin/env python3
"""CPU-only statistics, audit, and final report for Experiment 2D8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


ARMS = ("N", "O1", "O2")
CONDITIONS = {"N": "N_REAL", "O1": "O1_REAL", "O2": "O2_REAL"}
PARENT_SHA256 = "6e5023b127032dbb4d32a23bf1be052702d51177437b2795b99c52bcd83314c7"
N_SHA256 = "57e62a2094693205b520e2986047d46c28d042d4ec34d6e65b2135f474adec20"
O1_SHA256 = "c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6"
O2_SHA256 = "792b85f99164e8d1a096e2913e9d116f7f544e84d63c5a48c9a20e136cd9b69f"
CONTINUATION_MANIFEST_SHA256 = "f15a5de4b5428031adfe8877f01e6487dcdfc6749e337f552feb7c6f92e9cc4d"
CONTINUATION_LEDGER_SHA256 = "555ac4b4425fcd711edf2e923412ecfac1db49802653570fe56b02ae4139c1aa"
TERMINAL_CURSOR_SHA256 = "682abcbcc8db8886274ccbb604927683af38d010ecae31b1494987513ae982d3"
TERMINAL_NEXT_BATCH_SHA256 = "a021ce09f7a25b6617632e2a76da1acb0980ac1dda888df5f1c8eb65b3939fbe"
TERMINAL_NEXT_STREAM_SHA256 = "7918ea7e6f979b8e49fca89c60dd68ace44b768854d34bc96dd751bee07b2567"
FRESH_SEQUENCES = 4_096
FRESH_TARGETS = 4_194_304
OLD_SEQUENCES = 2_048
OLD_TARGETS = 2_097_152
STARTING_GLOBAL_UPDATE = 2_099
FIRST_GLOBAL_UPDATE = 2_100
FINAL_GLOBAL_UPDATE = 2_290
UPDATES = 191
TARGETS_PER_UPDATE = 524_288
NEW_TARGETS = 100_139_008
FINAL_TARGETS = 1_200_619_520
BOOTSTRAP_SEED = 2_026_090_2
BOOTSTRAP_RESAMPLES = 50_000
DELTA_CE = 0.0001
OLD_ESTIMATE = 0.00014959104405374863
OLD_CI = (0.000077991189, 0.000221717781)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def all_true(mapping):
    return bool(mapping) and all(value is True for value in mapping.values())


def percentile_row(distribution, estimate):
    return {
        "estimate": float(estimate),
        "lower_95": float(np.percentile(distribution, 2.5)),
        "upper_95": float(np.percentile(distribution, 97.5)),
    }


def fresh_bootstrap(differences):
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    distributions = {
        name: np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
        for name in differences
    }
    cursor = 0
    while cursor < BOOTSTRAP_RESAMPLES:
        count = min(125, BOOTSTRAP_RESAMPLES - cursor)
        indices = generator.integers(
            0, FRESH_SEQUENCES, size=(count, FRESH_SEQUENCES), dtype=np.int32
        )
        for name, values in differences.items():
            distributions[name][cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return {
        "method": "paired per-sequence percentile bootstrap",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "sampling_unit": "sequence",
        "paired_sequences": FRESH_SEQUENCES,
        "shared_resample_indices_all_contrasts": True,
        "contrasts": {
            name: percentile_row(distributions[name], values.mean())
            for name, values in differences.items()
        },
    }


def single_panel_bootstrap(values):
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    cursor = 0
    while cursor < BOOTSTRAP_RESAMPLES:
        count = min(250, BOOTSTRAP_RESAMPLES - cursor)
        indices = generator.integers(
            0, len(values), size=(count, len(values)), dtype=np.int32
        )
        distribution[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return percentile_row(distribution, values.mean())


def stratified_bootstrap(old, fresh):
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    old_distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    fresh_distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    pooled_distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    heterogeneity_distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    cursor = 0
    while cursor < BOOTSTRAP_RESAMPLES:
        count = min(125, BOOTSTRAP_RESAMPLES - cursor)
        old_indices = generator.integers(
            0, OLD_SEQUENCES, size=(count, OLD_SEQUENCES), dtype=np.int32
        )
        fresh_indices = generator.integers(
            0, FRESH_SEQUENCES, size=(count, FRESH_SEQUENCES), dtype=np.int32
        )
        old_means = old[old_indices].mean(axis=1)
        fresh_means = fresh[fresh_indices].mean(axis=1)
        span = slice(cursor, cursor + count)
        old_distribution[span] = old_means
        fresh_distribution[span] = fresh_means
        pooled_distribution[span] = (OLD_SEQUENCES * old_means + FRESH_SEQUENCES * fresh_means) / (OLD_SEQUENCES + FRESH_SEQUENCES)
        heterogeneity_distribution[span] = fresh_means - old_means
        cursor += count
    old_mean = float(old.mean())
    fresh_mean = float(fresh.mean())
    pooled_mean = (OLD_SEQUENCES * old_mean + FRESH_SEQUENCES * fresh_mean) / (OLD_SEQUENCES + FRESH_SEQUENCES)
    return {
        "method": "stratified paired per-sequence percentile bootstrap",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "old_sequences": OLD_SEQUENCES,
        "fresh_sequences": FRESH_SEQUENCES,
        "independent_within_panel_resampling": True,
        "fixed_sequence_count_weights": {"old": 1 / 3, "fresh": 2 / 3},
        "old": percentile_row(old_distribution, old_mean),
        "fresh": percentile_row(fresh_distribution, fresh_mean),
        "pooled": percentile_row(pooled_distribution, pooled_mean),
        "heterogeneity_fresh_minus_old": percentile_row(
            heterogeneity_distribution, fresh_mean - old_mean
        ),
    }


def classify(row, left, right):
    lower, upper = row["lower_95"], row["upper_95"]
    flags = {
        "right_statistically_superior": lower > 0,
        "right_superior_beyond_margin": lower > DELTA_CE,
        "left_statistically_superior": upper < 0,
        "left_superior_beyond_margin": upper < -DELTA_CE,
        "practical_equivalence_established": lower > -DELTA_CE and upper < DELTA_CE,
    }
    if flags["right_superior_beyond_margin"]:
        label = f"{right} superior to {left} beyond delta_CE"
    elif flags["left_superior_beyond_margin"]:
        label = f"{left} superior to {right} beyond delta_CE"
    elif flags["right_statistically_superior"]:
        label = f"{right} statistically superior to {left}; beyond-margin superiority not established"
    elif flags["left_statistically_superior"]:
        label = f"{left} statistically superior to {right}; beyond-margin superiority not established"
    elif flags["practical_equivalence_established"]:
        label = "practical equivalence established"
    else:
        label = "unresolved for superiority and practical equivalence"
    return {"label": label, "flags": flags}


def wins(left, right):
    difference = left - right
    return {
        "left_wins": int(np.sum(difference < 0)),
        "right_wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
    }


def interpretation(summary):
    rows = summary["fresh_bootstrap"]["contrasts"]
    classes = summary["classification"]
    ranking = summary["point_estimate_ranking"]
    n_o1 = classes["N_vs_O1"]
    o1_o2 = classes["O1_vs_O2"]
    n_o2 = classes["N_vs_O2"]

    if n_o1["flags"]["right_statistically_superior"]:
        q1 = "Yes. Fresh data replicate O1's statistical superiority over N" + (
            " beyond the 0.0001 CE margin." if n_o1["flags"]["right_superior_beyond_margin"]
            else ", although superiority beyond 0.0001 CE is not established."
        )
    elif rows["D_N_O1"]["estimate"] > 0:
        q1 = "The fresh point estimate again favors O1, but the replication is unresolved by the paired CI."
    else:
        q1 = "No. The fresh result does not replicate an O1 advantage over N."

    if o1_o2["flags"]["right_statistically_superior"]:
        q2 = "Yes. O2 is statistically better than O1" + (
            " beyond the 0.0001 CE margin." if o1_o2["flags"]["right_superior_beyond_margin"]
            else ", but not beyond the 0.0001 CE margin."
        )
    elif o1_o2["flags"]["left_statistically_superior"]:
        q2 = "No. O1 is statistically better than O2."
    elif o1_o2["flags"]["practical_equivalence_established"]:
        q2 = "No established improvement; O1 and O2 are practically equivalent at the ±0.0001 CE scale."
    else:
        q2 = "Not established; the O1–O2 paired result is unresolved."

    ordered = ranking == ["O2", "O1", "N"]
    both_steps = (
        n_o1["flags"]["right_statistically_superior"]
        and o1_o2["flags"]["right_statistically_superior"]
    )
    if ordered and both_steps:
        q3 = "Yes. The fresh ordering is O2 < O1 < N and both successive improvements are statistically supported."
    elif ordered:
        q3 = "The fresh point estimates order O2 < O1 < N, but a monotonic dose response is not established by both paired CIs."
    else:
        q3 = "No established monotonic overlap-width dose response; the fresh point-estimate ordering is " + " < ".join(ranking) + "."

    all_equivalent = all(
        value["flags"]["practical_equivalence_established"]
        for value in classes.values()
    )
    if o1_o2["flags"]["right_statistically_superior"] and not n_o2["flags"]["left_statistically_superior"]:
        preferred = "O2"
        preferred_reason = "O2 establishes superiority over O1 on the fresh matched panel without a material contradiction from N."
    elif (
        (o1_o2["flags"]["left_statistically_superior"] or o1_o2["flags"]["practical_equivalence_established"])
        and rows["D_N_O1"]["estimate"] > 0
    ):
        preferred = "O1"
        preferred_reason = "O1 remains favored over N and wider overlap does not establish an improvement over O1."
    elif all_equivalent:
        preferred = "N"
        preferred_reason = "All fresh pairwise results establish practical equivalence, so N is the simplest established geometry."
    else:
        preferred = "O1"
        preferred_reason = "Evidence is unresolved, so retain the strongest previously supported geometry provisionally."

    if o1_o2["flags"]["right_statistically_superior"]:
        q5 = "Yes, scientifically, because O2 establishes improvement over O1; no follow-up was launched automatically."
    elif o1_o2["flags"]["left_statistically_superior"] or o1_o2["flags"]["practical_equivalence_established"]:
        q5 = "No. Wider overlap is equivalent to or worse than O1, so further width expansion is not currently warranted."
    else:
        q5 = "Not currently established. O1 versus O2 is unresolved, so additional GPU spend should not be escalated automatically."
    return {
        "q1_fresh_replication": q1,
        "q2_o2_vs_o1": q2,
        "q3_dose_response": q3,
        "q4_preferred_geometry": preferred,
        "q4_reason": preferred_reason,
        "q5_further_overlap": q5,
    }


def report(summary):
    ce = summary["mean_ce"]
    ppl = summary["perplexity"]
    rows = summary["fresh_bootstrap"]["contrasts"]
    ratios = summary["perplexity_ratios"]
    classes = summary["classification"]
    win = summary["per_sequence_wins"]
    rank = summary["point_estimate_ranking"]
    pooled = summary["stratified_n_o1"]
    persistent = summary["persistent_state"]
    terminal = summary["terminal_stream"]
    interp = summary["scientific_interpretation"]
    checkpoint = summary["o2_checkpoint"]
    return f"""# EXPERIMENT 2D8 — OVERLAP WIDTH N/O1/O2 COMPLETE

## FRESH PANEL

Sequences: 4096

Fresh N CE: `{ce['N']:.12f}`
Fresh O1 CE: `{ce['O1']:.12f}`
Fresh O2 CE: `{ce['O2']:.12f}`

Fresh perplexities:
N: `{ppl['N']:.12f}`
O1: `{ppl['O1']:.12f}`
O2: `{ppl['O2']:.12f}`

## PAIRWISE FRESH RESULTS

N − O1: `{rows['D_N_O1']['estimate']:+.12f}`
95% CI: `[{rows['D_N_O1']['lower_95']:+.12f}, {rows['D_N_O1']['upper_95']:+.12f}]`
Perplexity ratio: `{ratios['D_N_O1']:.12f}`
Classification: {classes['N_vs_O1']['label']}

O1 − O2: `{rows['D_O1_O2']['estimate']:+.12f}`
95% CI: `[{rows['D_O1_O2']['lower_95']:+.12f}, {rows['D_O1_O2']['upper_95']:+.12f}]`
Perplexity ratio: `{ratios['D_O1_O2']:.12f}`
Classification: {classes['O1_vs_O2']['label']}

N − O2: `{rows['D_N_O2']['estimate']:+.12f}`
95% CI: `[{rows['D_N_O2']['lower_95']:+.12f}, {rows['D_N_O2']['upper_95']:+.12f}]`
Perplexity ratio: `{ratios['D_N_O2']:.12f}`
Classification: {classes['N_vs_O2']['label']}

delta_CE:
0.0001

Per-sequence wins:
N vs O1: N {win['N_vs_O1']['left_wins']}, O1 {win['N_vs_O1']['right_wins']}, ties {win['N_vs_O1']['ties']}
O1 vs O2: O1 {win['O1_vs_O2']['left_wins']}, O2 {win['O1_vs_O2']['right_wins']}, ties {win['O1_vs_O2']['ties']}
N vs O2: N {win['N_vs_O2']['left_wins']}, O2 {win['N_vs_O2']['right_wins']}, ties {win['N_vs_O2']['ties']}

Fresh point-estimate ranking:
1. {rank[0]}
2. {rank[1]}
3. {rank[2]}

## N-vs-O1 CONFIRMATION

Old reused sealed matched panel:
N − O1: `{pooled['old']['estimate']:+.12f}`
95% CI: `[{summary['old_reproduction']['lower_95']:+.12f}, {summary['old_reproduction']['upper_95']:+.12f}]`

Fresh panel:
N − O1: `{pooled['fresh']['estimate']:+.12f}`
95% CI: `[{rows['D_N_O1']['lower_95']:+.12f}, {rows['D_N_O1']['upper_95']:+.12f}]`

Stratified pooled N − O1: `{pooled['pooled']['estimate']:+.12f}`
95% CI: `[{pooled['pooled']['lower_95']:+.12f}, {pooled['pooled']['upper_95']:+.12f}]`
Classification: {summary['pooled_classification']['label']}

Heterogeneity:
D_fresh − D_old: `{pooled['heterogeneity_fresh_minus_old']['estimate']:+.12f}`
95% CI: `[{pooled['heterogeneity_fresh_minus_old']['lower_95']:+.12f}, {pooled['heterogeneity_fresh_minus_old']['upper_95']:+.12f}]`

## PERSISTENT STATE

N: `{persistent['N']:,}` bytes/sequence
O1: `{persistent['O1']:,}` bytes/sequence
O2: `{persistent['O2']:,}` bytes/sequence

O1 − N: `{persistent['O1_minus_N']:+,}` bytes/sequence
O2 − N: `{persistent['O2_minus_N']:+,}` bytes/sequence
O2 − O1: `{persistent['O2_minus_O1']:+,}` bytes/sequence

## O2 TRAINING / PROVENANCE

Starting global update: 2099
First new update:       2100
Final global update:    2290

Final cumulative targets:
1,200,619,520

O2 checkpoint:
`{checkpoint['path']}`
SHA-256: `{checkpoint['sha256']}`

Terminal loader cursor SHA:
`{terminal['final_loader_cursor_sha256']}`
Terminal next-global-batch SHA:
`{terminal['next_global_batch_sha256']}`
Terminal next-stream SHA:
`{terminal['next_stream_sha256']}`
{terminal['status']}

AUDIT:
{summary['audit_status']}

GPU STATUS:
{summary['gpu_status']}

## SCIENTIFIC INTERPRETATION

1. Did fresh data replicate the O1 advantage over N?
{interp['q1_fresh_replication']}

2. Is O2 better than O1?
{interp['q2_o2_vs_o1']}

3. Is there evidence for an overlap-width dose response?
{interp['q3_dose_response']}

4. Which geometry should become the preferred architecture?
{interp['q4_preferred_geometry']}. {interp['q4_reason']}

5. Is any further overlap-width experiment warranted?
{interp['q5_further_overlap']}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-artifacts", required=True)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--old-evaluation-dir", required=True)
    parser.add_argument("--old-bootstrap", required=True)
    parser.add_argument("--stop-verification", required=True)
    parser.add_argument("--o2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    gpu = Path(args.gpu_artifacts).resolve()
    output = Path(args.output_dir).resolve()
    panel_path = Path(args.panel_manifest).resolve()
    panel = read_json(panel_path)
    disjointness = read_json(panel_path.with_name("EVALUATION_PANEL_DISJOINTNESS_AUDIT.json"))
    preflight = read_json(gpu / "preflight/PREFLIGHT_AUDIT.json")
    training = read_json(gpu / "training/O2/TRAINING_COMPLETE_O2.json")
    training_log_path = gpu / "training/O2/TRAINING_LOG_O2.jsonl"
    training_log = [json.loads(line) for line in training_log_path.read_text().splitlines()]
    evaluations = {
        arm: read_json(gpu / f"evaluation/{arm}_REAL.json") for arm in ARMS
    }
    values = {
        arm: np.asarray(evaluations[arm]["per_sequence_ce"], dtype=np.float64)
        for arm in ARMS
    }
    differences = {
        "D_N_O1": values["N"] - values["O1"],
        "D_O1_O2": values["O1"] - values["O2"],
        "D_N_O2": values["N"] - values["O2"],
    }
    fresh = fresh_bootstrap(differences)
    old_dir = Path(args.old_evaluation_dir).resolve()
    old_n = np.asarray(read_json(old_dir / "N_REAL.json")["per_sequence_ce"], dtype=np.float64)
    old_o1 = np.asarray(read_json(old_dir / "O_REAL.json")["per_sequence_ce"], dtype=np.float64)
    old_difference = old_n - old_o1
    old_reproduction = single_panel_bootstrap(old_difference)
    old_saved = read_json(args.old_bootstrap)["contrasts"]["D_NO"]
    stratified = stratified_bootstrap(old_difference, differences["D_N_O1"])

    classes = {
        "N_vs_O1": classify(fresh["contrasts"]["D_N_O1"], "N", "O1"),
        "O1_vs_O2": classify(fresh["contrasts"]["D_O1_O2"], "O1", "O2"),
        "N_vs_O2": classify(fresh["contrasts"]["D_N_O2"], "N", "O2"),
    }
    pooled_classification = classify(stratified["pooled"], "N", "O1")
    mean_ce = {arm: float(values[arm].mean()) for arm in ARMS}
    ranking = sorted(ARMS, key=mean_ce.get)
    persistent = {
        arm: int(evaluations[arm]["persistent_state"]["physical_bytes_per_sequence"])
        for arm in ARMS
    }
    persistent.update(
        O1_minus_N=persistent["O1"] - persistent["N"],
        O2_minus_N=persistent["O2"] - persistent["N"],
        O2_minus_O1=persistent["O2"] - persistent["O1"],
    )
    checkpoint_path = Path(args.o2_checkpoint).resolve()
    checkpoint_hash = sha256(checkpoint_path)
    stop = read_json(args.stop_verification)
    stop_confirmed = (
        stop.get("status") == "STOPPED"
        and stop.get("gpu_stopped_confirmed") is True
        and stop.get("network_volume_retained") is True
    )
    final_checkpoint = training["final_checkpoint"]
    cursor_hash = canonical_sha(training["final_loader_cursor"])
    terminal = {
        "final_loader_cursor_sha256": cursor_hash,
        "next_global_batch_sha256": training["next_global_batch_sha256"],
        "next_stream_sha256": training["next_stream_sha256"],
    }
    terminal["status"] = "MATCH" if (
        cursor_hash == TERMINAL_CURSOR_SHA256
        and terminal["next_global_batch_sha256"] == TERMINAL_NEXT_BATCH_SHA256
        and terminal["next_stream_sha256"] == TERMINAL_NEXT_STREAM_SHA256
    ) else "MISMATCH"

    o2 = preflight["o2"]
    geometry = o2["geometry"]
    manifest = o2["architecture_manifest"]
    existing = preflight["existing_models"]
    evaluation_files = sorted(path.name for path in (gpu / "evaluation").glob("*.json"))
    checks = {
        "parent_sha_exact": preflight["source"]["sha256"] == PARENT_SHA256,
        "parent_counters_exact": preflight["source"]["global_update"] == STARTING_GLOBAL_UPDATE and preflight["source"]["cumulative_targets"] == 1_100_480_512,
        "o2_independent_from_sealed_2d6": o2["construction"]["parent_arm"] and o2["construction"]["parent_experiment"] and o2["construction"]["parent_global_update"] and o2["construction"]["parent_cumulative_targets"],
        "n_sha_exact_strict": existing["N"]["sha256"] == N_SHA256 and all_true(existing["N"]["checks"]),
        "o1_sha_exact_strict": existing["O1"]["sha256"] == O1_SHA256 and all_true(existing["O1"]["checks"]),
        "n_o1_not_retrained": preflight["checks"]["n_o1_not_retrained"] and training.get("existing_models_retrained") == [],
        "b6_w1024_b7_recurrence_absent": manifest["local_windows"]["B6"] == 1024 and manifest["b7_to_b6_computational_link"] is False and geometry["checks"]["b6_runtime_w1024"] and geometry["checks"]["b7_to_b6_absent"],
        "separate_softmax_preserved": all(block.get("separate_local_recurrent_softmax") is True for block in manifest["blocks"] if block["block"] in (1, 3, 5)),
        "existing_gates_preserved_trainable": o2["construction"]["gate_inventory"] and o2["construction"]["gate_values_inherited"] and o2["construction"]["active_gates_trainable"],
        "no_new_gates_or_parameters": manifest["new_parameters"] == 0,
        "o2_minima_exact": manifest["recurrent_minimum_lags"] == {"B1": 1, "B3": 30, "B5": 62},
        "o2_runtime_boundaries_exact": geometry["passed"] and all_true(geometry["checks"]),
        "no_source_shift_or_future_access": geometry["checks"]["no_future_or_source_shift"] and all(row["source_identity"] == "j=t-lag" and row["no_future_or_current_recurrent"] for row in geometry["boundaries"].values()),
        "continuation_hashes_exact": preflight["continuation"]["manifest_sha256"] == CONTINUATION_MANIFEST_SHA256 and preflight["continuation"]["ledger_sha256"] == CONTINUATION_LEDGER_SHA256,
        "preflight_pass": preflight["authorized"] and all_true(preflight["checks"]),
        "first_batch_exact": preflight["checks"]["first_batch_exact"] and all(o2["first_logical_batch"].values()),
        "training_exact_191_updates": len(training_log) == UPDATES and training_log[0]["global_update"] == FIRST_GLOBAL_UPDATE and training_log[-1]["global_update"] == FINAL_GLOBAL_UPDATE,
        "training_targets_exact": all(row["target_count"] == TARGETS_PER_UPDATE for row in training_log) and training_log[-1]["new_targets"] == NEW_TARGETS,
        "training_continuation_exact": training["checks"]["batches"] and training["checks"]["updates"] and training["checks"]["targets"],
        "inherited_state_no_reset": all(o2["construction"][name] for name in ("optimizer_state_restored", "scheduler_state_restored", "rng_state_restored", "loader_state_restored")),
        "ce_only_no_freezing": all(row["ce_only"] and all(group["finite"] and group["nonzero"] for group in row["active_gradient_groups"].values()) for row in training_log),
        "final_counters_exact": final_checkpoint["global_update"] == FINAL_GLOBAL_UPDATE and final_checkpoint["cumulative_targets"] == FINAL_TARGETS,
        "terminal_stream_exact": terminal["status"] == "MATCH",
        "o2_checkpoint_sealed_exported": checkpoint_hash == O2_SHA256 == final_checkpoint["sha256"] and checkpoint_path.stat().st_size == final_checkpoint["bytes"] and final_checkpoint["strict_reopen"]["passed"] and all_true(final_checkpoint["strict_reopen"]["checks"]),
        "fresh_panel_disjoint_frozen": disjointness["passed"] and all_true(disjointness["checks"]) and panel["sealed_after_o2_checkpoint_before_any_scoring"] and not panel["checkpoint_losses_inspected_during_selection"],
        "fresh_panel_exact_size": panel["sequence_count"] == FRESH_SEQUENCES and panel["targets_per_condition"] == FRESH_TARGETS,
        "same_ordered_fresh_panel": all(evaluations[arm]["panel_sha256"] == panel["panel_sha256"] and evaluations[arm]["batch_identities"] == panel["batch_identities"] and evaluations[arm]["batch_indices_in_evaluation_order"] == panel["batch_indices_in_evaluation_order"] for arm in ARMS),
        "three_4096_loss_arrays": all(len(values[arm]) == FRESH_SEQUENCES and evaluations[arm]["targets"] == FRESH_TARGETS and evaluations[arm]["passed"] for arm in ARMS),
        "exactly_three_allowed_evaluations": evaluation_files == ["N_REAL.json", "O1_REAL.json", "O2_REAL.json"] and {evaluations[arm]["condition"] for arm in ARMS} == set(CONDITIONS.values()),
        "fresh_shared_bootstrap_50000": fresh["resamples"] == BOOTSTRAP_RESAMPLES and fresh["shared_resample_indices_all_contrasts"],
        "old_arrays_reproduce_2d7": len(old_difference) == OLD_SEQUENCES and abs(old_reproduction["estimate"] - OLD_ESTIMATE) < 5e-13 and abs(old_reproduction["lower_95"] - OLD_CI[0]) < 5e-12 and abs(old_reproduction["upper_95"] - OLD_CI[1]) < 5e-12 and all(abs(old_reproduction[key] - old_saved[key]) < 5e-15 for key in ("estimate", "lower_95", "upper_95")),
        "stratified_bootstrap_50000": stratified["resamples"] == BOOTSTRAP_RESAMPLES and stratified["independent_within_panel_resampling"],
        "heterogeneity_reported": "heterogeneity_fresh_minus_old" in stratified,
        "persistent_state_accounted": all(persistent[arm] > 0 for arm in ARMS),
        "gpu_stopped_volume_retained": stop_confirmed,
    }
    audit_passed = all(checks.values())
    summary = {
        "experiment": "2D8",
        "mean_ce": mean_ce,
        "perplexity": {arm: math.exp(mean_ce[arm]) for arm in ARMS},
        "fresh_bootstrap": fresh,
        "perplexity_ratios": {name: math.exp(row["estimate"]) for name, row in fresh["contrasts"].items()},
        "classification": classes,
        "per_sequence_wins": {
            "N_vs_O1": wins(values["N"], values["O1"]),
            "O1_vs_O2": wins(values["O1"], values["O2"]),
            "N_vs_O2": wins(values["N"], values["O2"]),
        },
        "point_estimate_ranking": ranking,
        "delta_ce": DELTA_CE,
        "old_reproduction": old_reproduction,
        "stratified_n_o1": stratified,
        "pooled_classification": pooled_classification,
        "heterogeneity_established": stratified["heterogeneity_fresh_minus_old"]["lower_95"] > 0 or stratified["heterogeneity_fresh_minus_old"]["upper_95"] < 0,
        "persistent_state": persistent,
        "o2_checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash, "bytes": checkpoint_path.stat().st_size},
        "terminal_stream": terminal,
        "audit_status": "PASS" if audit_passed else "FAIL",
        "gpu_status": "STOPPED" if stop_confirmed else stop.get("status", "NOT STOPPED"),
        "checks": checks,
    }
    summary["scientific_interpretation"] = interpretation(summary)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "FRESH_PAIRED_BOOTSTRAP.json", fresh)
    write_json(output / "STRATIFIED_POOLED_N_O1.json", {**stratified, "classification": pooled_classification})
    write_json(output / "PANEL_HETEROGENEITY.json", stratified["heterogeneity_fresh_minus_old"])
    write_json(output / "PERSISTENT_STATE_SUMMARY.json", persistent)
    write_json(output / "SCIENTIFIC_RESULT_SUMMARY.json", summary)
    write_json(output / "FINAL_AUDIT.json", {
        "schema": "experiment_2d8_final_audit_v1",
        "experiment": "2D8",
        "checks": checks,
        "audit_status": summary["audit_status"],
        "passed": audit_passed,
        "evidence": {
            "parent": preflight["source"],
            "existing_models": existing,
            "o2_checkpoint": summary["o2_checkpoint"],
            "terminal_stream": terminal,
            "panel_sha256": panel["panel_sha256"],
            "evaluation_conditions": {arm: evaluations[arm]["condition"] for arm in ARMS},
            "old_reproduction": old_reproduction,
            "gpu_stop": stop,
        },
    })
    write_text(output / "EXPERIMENT_2D8_FINAL_REPORT.md", report(summary))
    print(json.dumps({
        "audit": summary["audit_status"],
        "mean_ce": mean_ce,
        "fresh_contrasts": fresh["contrasts"],
        "classification": {key: value["label"] for key, value in classes.items()},
        "pooled": stratified["pooled"],
        "heterogeneity": stratified["heterogeneity_fresh_minus_old"],
        "ranking": ranking,
        "preferred": summary["scientific_interpretation"]["q4_preferred_geometry"],
    }, indent=2, sort_keys=True))
    if not audit_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
