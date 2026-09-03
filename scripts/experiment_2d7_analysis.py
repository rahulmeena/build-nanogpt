#!/usr/bin/env python3
"""CPU-only paired analysis and final audit for Experiment 2D7."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


ARMS = ("N", "O", "G")
CONDITIONS = {"N": "BASELINE_REAL", "O": "OVERLAP1_REAL", "G": "GAP1_REAL"}
PARENT_SHA256 = "6e5023b127032dbb4d32a23bf1be052702d51177437b2795b99c52bcd83314c7"
PANEL_SEQUENCES = 2_048
PANEL_TARGETS = 2_097_152
STARTING_GLOBAL_UPDATE = 2_099
FIRST_GLOBAL_UPDATE = 2_100
UPDATES_PER_ARM = 191
TARGETS_PER_UPDATE = 524_288
NEW_TARGETS_PER_ARM = 100_139_008
PARENT_TARGETS = 1_100_480_512
FINAL_GLOBAL_UPDATE = 2_290
FINAL_TARGETS = 1_200_619_520
BOOTSTRAP_SEED = 2_026_090_2
BOOTSTRAP_RESAMPLES = 50_000
DELTA_CE = 0.0001


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


def paired_bootstrap(arrays):
    length = len(next(iter(arrays.values())))
    if length != PANEL_SEQUENCES or any(len(values) != length for values in arrays.values()):
        raise SystemExit("paired bootstrap requires three aligned 2,048-sequence arrays")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    distributions = {
        name: np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64) for name in arrays
    }
    cursor = 0
    while cursor < BOOTSTRAP_RESAMPLES:
        count = min(250, BOOTSTRAP_RESAMPLES - cursor)
        indices = generator.integers(
            0, length, size=(count, length), dtype=np.int32
        )
        for name, values in arrays.items():
            distributions[name][cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return {
        "method": "paired per-sequence percentile bootstrap",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "sampling_unit": "sequence",
        "paired_sequences": length,
        "shared_resample_indices_all_contrasts": True,
        "contrasts": {
            name: {
                "estimate": float(values.mean()),
                "lower_95": float(np.percentile(distributions[name], 2.5)),
                "upper_95": float(np.percentile(distributions[name], 97.5)),
            }
            for name, values in arrays.items()
        },
    }


def classify(row, left, right):
    lower, upper = row["lower_95"], row["upper_95"]
    if lower > DELTA_CE:
        return f"{right} superior to {left} by more than delta_CE"
    if upper < -DELTA_CE:
        return f"{left} superior to {right} by more than delta_CE"
    if lower > 0:
        return f"{right} statistically superior to {left}, but not beyond delta_CE"
    if upper < 0:
        return f"{left} statistically superior to {right}, but not beyond delta_CE"
    if lower > -DELTA_CE and upper < DELTA_CE:
        return "practical equivalence established"
    return "unresolved for superiority and practical equivalence"


def wins(left, right):
    difference = left - right
    return {
        "left_wins": int(np.sum(difference < 0)),
        "right_wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
    }


def report(summary):
    ce = summary["mean_ce"]
    ppl = summary["perplexity"]
    contrast = summary["bootstrap"]["contrasts"]
    ratio = summary["perplexity_ratios"]
    win = summary["per_sequence_wins"]
    classes = summary["classification"]
    persistent = summary["persistent_state"]
    terminal = summary["terminal_stream"]
    checkpoints = summary["final_checkpoints"]
    ranking = summary["point_estimate_ranking"]
    return f"""# EXPERIMENT 2D7 — BOUNDARY ALIGNMENT N/O/G COMPLETE

Baseline CE: `{ce['N']:.12f}` (perplexity `{ppl['N']:.12f}`)
Overlap-1 CE: `{ce['O']:.12f}` (perplexity `{ppl['O']:.12f}`)
Gap-1 CE: `{ce['G']:.12f}` (perplexity `{ppl['G']:.12f}`)

Baseline − Overlap: `{contrast['D_NO']['estimate']:+.12f}`
95% CI: `[{contrast['D_NO']['lower_95']:+.12f}, {contrast['D_NO']['upper_95']:+.12f}]`

Baseline − Gap: `{contrast['D_NG']['estimate']:+.12f}`
95% CI: `[{contrast['D_NG']['lower_95']:+.12f}, {contrast['D_NG']['upper_95']:+.12f}]`

Overlap − Gap: `{contrast['D_OG']['estimate']:+.12f}`
95% CI: `[{contrast['D_OG']['lower_95']:+.12f}, {contrast['D_OG']['upper_95']:+.12f}]`

Perplexity ratios:
Baseline/Overlap: `{ratio['D_NO']:.12f}`
Baseline/Gap: `{ratio['D_NG']:.12f}`
Overlap/Gap: `{ratio['D_OG']:.12f}`

Per-sequence wins:
Baseline vs Overlap: Baseline `{win['N_vs_O']['left_wins']}`, Overlap `{win['N_vs_O']['right_wins']}`, ties `{win['N_vs_O']['ties']}`
Baseline vs Gap: Baseline `{win['N_vs_G']['left_wins']}`, Gap `{win['N_vs_G']['right_wins']}`, ties `{win['N_vs_G']['ties']}`
Overlap vs Gap: Overlap `{win['O_vs_G']['left_wins']}`, Gap `{win['O_vs_G']['right_wins']}`, ties `{win['O_vs_G']['ties']}`

Point-estimate ranking:
1. {ranking[0]}
2. {ranking[1]}
3. {ranking[2]}

delta_CE:
`0.0001`

Practical/statistical classification:
Baseline vs Overlap: {classes['N_vs_O']}
Baseline vs Gap: {classes['N_vs_G']}
Overlap vs Gap: {classes['O_vs_G']}

Persistent state:
Baseline: `{persistent['N']:,}` bytes/sequence
Overlap-1: `{persistent['O']:,}` bytes/sequence
Gap-1: `{persistent['G']:,}` bytes/sequence
Overlap − Baseline: `{persistent['O_minus_N']:+,}` bytes/sequence
Gap − Baseline: `{persistent['G_minus_N']:+,}` bytes/sequence

Training counters:
Starting global update: 2099
First new update:       2100
Final global update:    2290
Final cumulative targets: 1,200,619,520

Terminal stream equality:
Final loader cursor SHA: `{terminal['final_loader_cursor_sha256']}`
Next-global-batch SHA: `{terminal['next_global_batch_sha256']}`
Next-stream SHA: `{terminal['next_stream_sha256']}`
{terminal['status']}

Final checkpoints:
N: `{checkpoints['N']['path']}`
SHA-256: `{checkpoints['N']['sha256']}`

O: `{checkpoints['O']['path']}`
SHA-256: `{checkpoints['O']['sha256']}`

G: `{checkpoints['G']['path']}`
SHA-256: `{checkpoints['G']['sha256']}`

AUDIT:
{summary['audit_status']}

GPU STATUS:
{summary['gpu_status']}

## SCIENTIFIC INTERPRETATION

{summary['scientific_interpretation']}

## RECOMMENDATION

{summary['recommendation']}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-artifacts", required=True)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--stop-verification", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    gpu = Path(args.gpu_artifacts).resolve()
    panel_path = Path(args.panel_manifest).resolve()
    panel = read_json(panel_path)
    panel_disjointness = read_json(
        panel_path.with_name("EVALUATION_PANEL_DISJOINTNESS_AUDIT.json")
    )
    stop = read_json(args.stop_verification)
    preflight = read_json(gpu / "preflight/PREFLIGHT_AUDIT.json")
    continuation = read_json(gpu / "preflight/CONTINUATION_MANIFEST.json")
    ledger_path = gpu / "preflight/CONTINUATION_LEDGER.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    training = {
        arm: read_json(gpu / f"training/{arm}/TRAINING_COMPLETE_{arm}.json")
        for arm in ARMS
    }
    training_logs = {
        arm: [
            json.loads(line)
            for line in (gpu / f"training/{arm}/TRAINING_LOG_{arm}.jsonl")
            .read_text()
            .splitlines()
        ]
        for arm in ARMS
    }
    evaluations = {
        arm: read_json(gpu / f"evaluation/{arm}_REAL.json") for arm in ARMS
    }
    values = {
        arm: np.asarray(evaluations[arm]["per_sequence_ce"], dtype=np.float64)
        for arm in ARMS
    }
    arrays = {
        "D_NO": values["N"] - values["O"],
        "D_NG": values["N"] - values["G"],
        "D_OG": values["O"] - values["G"],
    }
    bootstrap = paired_bootstrap(arrays)
    mean_ce = {arm: float(values[arm].mean()) for arm in ARMS}
    ranking = [CONDITIONS[arm] for arm in sorted(ARMS, key=mean_ce.get)]
    classes = {
        "N_vs_O": classify(bootstrap["contrasts"]["D_NO"], "Baseline", "Overlap-1"),
        "N_vs_G": classify(bootstrap["contrasts"]["D_NG"], "Baseline", "Gap-1"),
        "O_vs_G": classify(bootstrap["contrasts"]["D_OG"], "Overlap-1", "Gap-1"),
    }
    persistent = {
        arm: int(evaluations[arm]["persistent_state"]["physical_bytes_per_sequence"])
        for arm in ARMS
    }
    persistent.update(
        O_minus_N=persistent["O"] - persistent["N"],
        G_minus_N=persistent["G"] - persistent["N"],
    )
    cursor_hashes = {
        arm: canonical_sha(training[arm]["final_loader_cursor"]) for arm in ARMS
    }
    batches = {arm: training[arm]["next_global_batch_sha256"] for arm in ARMS}
    streams = {arm: training[arm]["next_stream_sha256"] for arm in ARMS}
    terminal_match = (
        len(set(cursor_hashes.values())) == 1
        and len(set(batches.values())) == 1
        and len(set(streams.values())) == 1
    )
    checkpoints = {}
    for arm in ARMS:
        path = (
            Path(args.checkpoint_root).resolve()
            / arm
            / "scientific_cumulative_001200619520.pt"
        )
        checkpoints[arm] = {
            "path": str(path),
            "sha256": sha256(path),
            "recorded_sha256": training[arm]["final_checkpoint"]["sha256"],
            "bytes": path.stat().st_size,
        }
    stop_confirmed = (
        stop.get("status") == "STOPPED"
        and stop.get("gpu_stopped_confirmed") is True
        and stop.get("pod", {}).get("desiredStatus") == "EXITED"
        and stop.get("pod", {}).get("runtimeStatus") == "stopped"
        and stop.get("network_volume_retained") is True
    )
    constructions = {arm: preflight["arms"][arm]["construction"] for arm in ARMS}
    geometries = {arm: preflight["arms"][arm]["geometry"] for arm in ARMS}
    manifests = {arm: preflight["arms"][arm]["architecture_manifest"] for arm in ARMS}
    expected_minima = {
        "N": {"B1": 2, "B3": 32, "B5": 64},
        "O": {"B1": 1, "B3": 31, "B5": 63},
        "G": {"B1": 3, "B3": 33, "B5": 65},
    }
    expected_windows = {
        "B1": 2, "B2": 1024, "B3": 32, "B4": 1024,
        "B5": 64, "B6": 1024, "B7": 1024, "B8": 1024,
        "B9": 1024, "B10": 1024, "B11": 1024, "B12": 1024,
    }
    ledger_exact = (
        len(ledger) == UPDATES_PER_ARM
        and [row["local_update"] for row in ledger] == list(range(1, UPDATES_PER_ARM + 1))
        and [row["global_update"] for row in ledger]
        == list(range(FIRST_GLOBAL_UPDATE, FINAL_GLOBAL_UPDATE + 1))
        and all(row["target_count"] == TARGETS_PER_UPDATE for row in ledger)
        and continuation["ledger_sha256"] == sha256(ledger_path)
    )
    strict_reopens = {
        arm: training[arm]["final_checkpoint"]["strict_reopen"] for arm in ARMS
    }
    checks = {
        # Parent / lineage
        "parent_sha_exact": preflight["provenance"]["parent_sha256"] == PARENT_SHA256
        and continuation["parent"]["sha256"] == PARENT_SHA256,
        "parent_lineage_sealed_2d6_only": continuation["parent"]["path"].endswith(
            "/exp2d6_b6_native_100m/checkpoints/scientific_cumulative_001100480512.pt"
        ),
        "parent_cumulative_targets": preflight["provenance"]["parent_cumulative_targets"] == PARENT_TARGETS,
        "starting_global_update": preflight["provenance"]["parent_global_update"] == STARTING_GLOBAL_UPDATE
        and continuation["starting_global_update"] == STARTING_GLOBAL_UPDATE,
        "first_new_update": continuation["first_global_update"] == FIRST_GLOBAL_UPDATE,
        "parent_continuation_state_strict": all_true(continuation["parent"]["checks"]),
        "independent_same_parent_loads": all(
            constructions[arm]["parent_arm"]
            and constructions[arm]["parent_experiment"]
            and constructions[arm]["parent_global_update"]
            and constructions[arm]["parent_cumulative_targets"]
            and strict_reopens[arm]["checks"]["parent"]
            for arm in ARMS
        ),
        # Model / parameter state
        "parameter_count_state_compatibility": preflight["checks"]["parameter_counts_identical"]
        and len(set(preflight["parameter_counts"].values())) == 1
        and all(constructions[arm]["state_dict_keys_exact"] for arm in ARMS),
        "existing_recurrent_gates_preserved_inherited_trainable": all(
            constructions[arm]["gate_inventory"]
            and constructions[arm]["gate_values_inherited"]
            and constructions[arm]["active_gates_trainable"]
            and manifests[arm]["active_recurrent_gates"] == ["g_rec", "g_rec_b3", "g_rec_b5"]
            for arm in ARMS
        ) and preflight["checks"]["gate_values_identical"],
        "b6_w1024_all_arms": all(
            manifests[arm]["local_windows"]["B6"] == 1024
            and geometries[arm]["checks"]["b6_runtime_capacity_w1024"]
            for arm in ARMS
        ),
        "b7_to_b6_absent_all_arms": all(
            manifests[arm]["b7_recurrent_ring"] is False
            and manifests[arm]["b7_to_b6_computational_link"] is False
            and geometries[arm]["checks"]["b7_ring_absent"]
            and geometries[arm]["checks"]["b7_to_b6_runtime_absent"]
            for arm in ARMS
        ),
        "local_windows_unchanged": all(
            manifests[arm]["local_windows"] == expected_windows for arm in ARMS
        ),
        # Boundary geometry
        "recurrent_minima_exact": all(
            manifests[arm]["recurrent_minimum_lags"] == expected_minima[arm]
            for arm in ARMS
        ),
        "boundary_runtime_audit": all(
            geometries[arm]["passed"]
            and geometries[arm]["checks"]["boundary_masks"]
            and geometries[arm]["checks"]["maximum_recurrent_lag"]
            for arm in ARMS
        ),
        "no_future_or_source_shift": all(
            geometries[arm]["checks"]["runtime_source_identity"]
            and all(
                boundary["no_future"] and boundary["source_identity"] == "j=t-lag"
                for boundary in geometries[arm]["boundaries"].values()
            )
            for arm in ARMS
        ),
        "separate_local_recurrent_softmax": all(
            all(
                block.get("separate_local_recurrent_softmax") is True
                for block in manifests[arm]["blocks"]
                if block["block"] in (1, 3, 5)
            )
            for arm in ARMS
        ),
        "no_new_parameters_or_gates": all(manifests[arm]["new_parameters"] == 0 for arm in ARMS),
        "config_diff_only_allowed_geometry": preflight["config_diff_audit"]["common_architecture_exact"]
        and preflight["config_diff_audit"]["only_allowed_geometry_minima"],
        "disposable_smokes_clean": all(
            preflight["arms"][arm]["disposable_smoke"]["passed"]
            and preflight["arms"][arm]["disposable_smoke"]["official_optimizer_updates"] == 0
            for arm in ARMS
        ),
        # Matched training and semantics
        "continuation_ledger_exact_191": continuation["passed"] and ledger_exact,
        "same_first_matched_batch": preflight["checks"]["first_batch_matches_manifest"]
        and all(training_logs[arm][0]["batch_sha256"] == continuation["first_global_batch_sha256"] for arm in ARMS),
        "same_ordered_191_batches": all(training[arm]["checks"]["batches"] for arm in ARMS)
        and all(
            [row["batch_sha256"] for row in training_logs[arm]]
            == [row["logical_global_batch_sha256"] for row in ledger]
            and [row["stream_sha256"] for row in training_logs[arm]]
            == [row["logical_global_stream_sha256"] for row in ledger]
            for arm in ARMS
        ),
        "optimizer_updates_per_arm": continuation["updates"] == UPDATES_PER_ARM
        and all(training[arm]["checks"]["updates"] for arm in ARMS),
        "targets_per_update": continuation["targets_per_update"] == TARGETS_PER_UPDATE,
        "new_targets_per_arm": continuation["new_targets"] == NEW_TARGETS_PER_ARM
        and all(training[arm]["checks"]["targets"] for arm in ARMS),
        "final_counters": continuation["final_global_update"] == FINAL_GLOBAL_UPDATE
        and continuation["final_cumulative_targets"] == FINAL_TARGETS
        and all(
            training[arm]["final_checkpoint"]["global_update"] == FINAL_GLOBAL_UPDATE
            and training[arm]["final_checkpoint"]["cumulative_targets"] == FINAL_TARGETS
            for arm in ARMS
        ),
        "terminal_stream_equality": terminal_match,
        "optimizer_scheduler_rng_loader_inherited": all(
            constructions[arm]["optimizer_state_restored"]
            and constructions[arm]["scheduler_state_restored"]
            and constructions[arm]["rng_state_restored"]
            and constructions[arm]["loader_state_restored"]
            for arm in ARMS
        ),
        "ce_only_no_parameter_freezing": all(
            len(training_logs[arm]) == UPDATES_PER_ARM
            and all(row["ce_only"] for row in training_logs[arm])
            and all(
                all(
                    group["finite"] and group["nonzero"] and group["gradient_tensors"] > 0
                    for group in row["active_gradient_groups"].values()
                )
                for row in training_logs[arm]
            )
            for arm in ARMS
        ),
        "no_extra_training": all(
            len(training_logs[arm]) == UPDATES_PER_ARM
            and training_logs[arm][0]["global_update"] == FIRST_GLOBAL_UPDATE
            and training_logs[arm][-1]["global_update"] == FINAL_GLOBAL_UPDATE
            and training_logs[arm][-1]["new_targets"] == NEW_TARGETS_PER_ARM
            for arm in ARMS
        ),
        "training_summaries_pass": all(
            training[arm].get("passed") is True
            and all_true(training[arm]["checks"])
            and strict_reopens[arm]["passed"]
            and all_true(strict_reopens[arm]["checks"])
            for arm in ARMS
        ),
        # Checkpoints / evaluation / scope
        "three_checkpoints_sealed_and_exported": all(
            checkpoints[arm]["sha256"] == checkpoints[arm]["recorded_sha256"]
            and checkpoints[arm]["bytes"] == training[arm]["final_checkpoint"]["bytes"]
            for arm in ARMS
        ),
        "fresh_disjoint_panel": panel_disjointness["passed"]
        and all_true(panel_disjointness["checks"])
        and panel.get("sealed_before_checkpoint_loading") is True
        and panel.get("checkpoint_losses_inspected_during_selection") is False,
        "same_ordered_2048_sequence_panel": panel["sequence_count"] == PANEL_SEQUENCES
        and all(
            evaluations[arm]["panel_sha256"] == panel["panel_sha256"]
            and evaluations[arm]["batch_identities"] == panel["batch_identities"]
            and len(values[arm]) == PANEL_SEQUENCES
            for arm in ARMS
        ),
        "targets_per_condition": panel["targets_per_condition"] == PANEL_TARGETS
        and all(evaluations[arm]["targets"] == PANEL_TARGETS for arm in ARMS),
        "three_real_evaluations_pass": all(evaluations[arm].get("passed") is True for arm in ARMS),
        "exactly_three_allowed_conditions": len(evaluations) == 3
        and {evaluations[arm]["condition"] for arm in ARMS} == set(CONDITIONS.values()),
        "bootstrap_shared_50000": bootstrap["resamples"] == BOOTSTRAP_RESAMPLES
        and bootstrap["paired_sequences"] == PANEL_SEQUENCES
        and bootstrap["shared_resample_indices_all_contrasts"] is True,
        "persistent_state_accounted": all(persistent[arm] > 0 for arm in ARMS),
        "gpu_stopped_and_volume_retained": stop_confirmed,
    }
    meaningful_o = (
        bootstrap["contrasts"]["D_NO"]["lower_95"] > DELTA_CE
        and bootstrap["contrasts"]["D_OG"]["upper_95"] < -DELTA_CE
    )
    meaningful_g = (
        bootstrap["contrasts"]["D_NG"]["lower_95"] > DELTA_CE
        and bootstrap["contrasts"]["D_OG"]["lower_95"] > DELTA_CE
    )
    all_equivalent = all(
        row["lower_95"] > -DELTA_CE and row["upper_95"] < DELTA_CE
        for row in bootstrap["contrasts"].values()
    )
    if meaningful_o:
        recommendation = "Recommend OVERLAP-1 because it established a greater-than-0.0001 CE advantage over both alternatives."
    elif meaningful_g:
        recommendation = "Recommend GAP-1 because it established a greater-than-0.0001 CE advantage over both alternatives."
    elif all_equivalent:
        recommendation = "Recommend BASELINE/N because all three geometries are practically equivalent and N is the established implementation."
    else:
        recommendation = "Recommend BASELINE/N provisionally because no alternative established sufficient superiority."
    if all_equivalent:
        interpretation = (
            "N, O, and G are practically equivalent at the preregistered ±0.0001 CE scale. "
            f"The numerical point-estimate winner ({ranking[0]}) is not an established meaningful winner."
        )
    elif classes["N_vs_O"].startswith("Overlap-1 statistically") \
            and classes["N_vs_G"].startswith("Baseline superior") \
            and classes["O_vs_G"].startswith("Overlap-1 superior"):
        interpretation = (
            "The numerical ordering is OVERLAP1_REAL < BASELINE_REAL < GAP1_REAL. "
            "Overlap-1 is statistically better than Baseline, but its confidence interval does not "
            "establish an advantage beyond 0.0001 CE; both Overlap-1 and Baseline establish "
            "greater-than-margin superiority to Gap-1. Boundary-token continuity matters, while "
            "dual-depth duplication has no established practically meaningful gain over Baseline."
        )
    else:
        interpretation = (
            f"The numerical ordering is {' < '.join(ranking)}. "
            "The pairwise classifications above distinguish established effects from unresolved numerical differences."
        )
    audit_passed = all(checks.values())
    summary = {
        "experiment": "2D7",
        "mean_ce": mean_ce,
        "perplexity": {arm: math.exp(mean_ce[arm]) for arm in ARMS},
        "bootstrap": bootstrap,
        "perplexity_ratios": {
            name: math.exp(row["estimate"])
            for name, row in bootstrap["contrasts"].items()
        },
        "per_sequence_wins": {
            "N_vs_O": wins(values["N"], values["O"]),
            "N_vs_G": wins(values["N"], values["G"]),
            "O_vs_G": wins(values["O"], values["G"]),
        },
        "point_estimate_ranking": ranking,
        "delta_ce": DELTA_CE,
        "classification": classes,
        "persistent_state": persistent,
        "terminal_stream": {
            "final_loader_cursor_sha256": cursor_hashes["N"],
            "next_global_batch_sha256": batches["N"],
            "next_stream_sha256": streams["N"],
            "per_arm_cursor_sha256": cursor_hashes,
            "per_arm_next_batch_sha256": batches,
            "per_arm_next_stream_sha256": streams,
            "per_arm_final_loader_cursor": {
                arm: training[arm]["final_loader_cursor"] for arm in ARMS
            },
            "status": "MATCH" if terminal_match else "MISMATCH",
        },
        "final_checkpoints": checkpoints,
        "checks": checks,
        "audit_status": "PASS" if audit_passed else "FAIL",
        "gpu_status": "STOPPED" if stop_confirmed else "NOT STOPPED",
        "scientific_interpretation": interpretation,
        "recommendation": recommendation,
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "PAIRED_BOOTSTRAP.json", bootstrap)
    write_json(output / "PERSISTENT_STATE_SUMMARY.json", persistent)
    write_json(output / "SCIENTIFIC_RESULT_SUMMARY.json", summary)
    audit_groups = {
        "parent_lineage": [
            "parent_sha_exact", "parent_lineage_sealed_2d6_only",
            "parent_cumulative_targets", "starting_global_update", "first_new_update",
            "parent_continuation_state_strict", "independent_same_parent_loads",
        ],
        "model_parameter_state": [
            "parameter_count_state_compatibility",
            "existing_recurrent_gates_preserved_inherited_trainable",
            "b6_w1024_all_arms", "b7_to_b6_absent_all_arms", "local_windows_unchanged",
        ],
        "boundary_geometry": [
            "recurrent_minima_exact", "boundary_runtime_audit", "no_future_or_source_shift",
            "separate_local_recurrent_softmax", "no_new_parameters_or_gates",
            "config_diff_only_allowed_geometry", "disposable_smokes_clean",
        ],
        "matched_training": [
            "continuation_ledger_exact_191", "same_first_matched_batch",
            "same_ordered_191_batches", "optimizer_updates_per_arm", "targets_per_update",
            "new_targets_per_arm", "final_counters", "terminal_stream_equality",
            "optimizer_scheduler_rng_loader_inherited", "ce_only_no_parameter_freezing",
            "no_extra_training", "training_summaries_pass",
        ],
        "checkpoints_evaluation_scope": [
            "three_checkpoints_sealed_and_exported", "fresh_disjoint_panel",
            "same_ordered_2048_sequence_panel", "targets_per_condition",
            "three_real_evaluations_pass", "exactly_three_allowed_conditions",
            "bootstrap_shared_50000", "persistent_state_accounted",
            "gpu_stopped_and_volume_retained",
        ],
    }
    write_json(output / "FINAL_AUDIT.json", {
        "schema": "experiment_2d7_final_audit_v1",
        "experiment": "2D7",
        "checks": checks,
        "groups": {
            group: {
                "passed": all(checks[name] for name in names),
                "checks": names,
            }
            for group, names in audit_groups.items()
        },
        "evidence": {
            "parent": preflight["provenance"],
            "continuation": {
                key: continuation[key]
                for key in (
                    "updates", "first_global_update", "final_global_update",
                    "targets_per_update", "new_targets", "final_cumulative_targets",
                    "ledger_sha256", "first_global_batch_sha256", "first_stream_sha256",
                    "next_global_batch_sha256", "next_stream_sha256",
                )
            },
            "terminal_stream": summary["terminal_stream"],
            "checkpoint_sha256": {
                arm: checkpoints[arm]["sha256"] for arm in ARMS
            },
            "evaluation_panel_sha256": panel["panel_sha256"],
            "evaluation_conditions": {
                arm: evaluations[arm]["condition"] for arm in ARMS
            },
            "gpu_stop": stop,
        },
        "audit_status": summary["audit_status"],
        "passed": audit_passed,
    })
    write_text(output / "EXPERIMENT_2D7_FINAL_REPORT.md", report(summary))
    print(json.dumps({
        "audit": summary["audit_status"],
        "mean_ce": mean_ce,
        "contrasts": bootstrap["contrasts"],
        "classification": classes,
        "ranking": ranking,
        "recommendation": recommendation,
    }, indent=2, sort_keys=True))
    if not audit_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
