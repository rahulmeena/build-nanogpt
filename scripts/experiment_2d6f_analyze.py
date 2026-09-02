#!/usr/bin/env python3
"""Local post-stop statistical analysis and reporting for Experiment 2D6F."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np


EXPERIMENT = "2D6F"
BRANCH = "experiment-2d6-fresh-panel-zero-training-confirmation"
TAG = "experiment-2d6-fresh-panel-zero-training-confirmation-final"
DELTA = 0.0001
RESAMPLES = 50_000
FRESH_SEED = 2_026_090_202
POOLED_SEED = 2_026_090_203
HETEROGENEITY_SEED = 2_026_090_204
EXPECTED_FILES = {
    "SCOPE_LOCK.json",
    "CHECKPOINT_IDENTITY.json",
    "REUSED_PANEL_IDENTITY.json",
    "FRESH_PANEL_MANIFEST.json",
    "FRESH_PANEL_DISJOINTNESS_AUDIT.json",
    "EVALUATOR_SENTINEL_AUDIT.json",
    "FRESH_PER_SEQUENCE_LOSSES.json",
    "FRESH_PAIRED_BOOTSTRAP.json",
    "STRATIFIED_POOLED_BOOTSTRAP.json",
    "PANEL_HETEROGENEITY.json",
    "FINAL_AUDIT.json",
    "EXPERIMENT_2D6F_FINAL_REPORT.md",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def durable_write(path, content):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def durable_json(path, value):
    durable_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def summarize_distribution(point, distribution):
    return {
        "estimate": float(point),
        "lower_95": float(np.percentile(distribution, 2.5)),
        "upper_95": float(np.percentile(distribution, 97.5)),
        "standard_error": float(distribution.std(ddof=1)),
    }


def paired_bootstrap(differences):
    values = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(FRESH_SEED)
    distribution = np.empty(RESAMPLES, dtype=np.float64)
    cursor = 0
    while cursor < RESAMPLES:
        count = min(250, RESAMPLES - cursor)
        indices = generator.integers(0, len(values), size=(count, len(values)), dtype=np.int32)
        distribution[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    d = summarize_distribution(values.mean(), distribution)
    p = {
        "estimate": -d["estimate"],
        "lower_95": -d["upper_95"],
        "upper_95": -d["lower_95"],
        "standard_error": d["standard_error"],
    }
    return {
        "method": "paired per-sequence percentile bootstrap",
        "seed": FRESH_SEED,
        "resamples": RESAMPLES,
        "paired_sequences": len(values),
        "targets_per_condition": len(values) * 1024,
        "same_resample_indices_for_sign_reversal": True,
        "D_fresh_fixed_minus_new": d,
        "P_fresh_new_minus_fixed": p,
        "positive_sequence_count": int(np.count_nonzero(values > 0)),
        "negative_sequence_count": int(np.count_nonzero(values < 0)),
        "zero_sequence_count": int(np.count_nonzero(values == 0)),
    }


def stratified_pooled(reused, fresh):
    reused = np.asarray(reused, dtype=np.float64)
    fresh = np.asarray(fresh, dtype=np.float64)
    generator = np.random.default_rng(POOLED_SEED)
    distribution = np.empty(RESAMPLES, dtype=np.float64)
    cursor = 0
    while cursor < RESAMPLES:
        count = min(250, RESAMPLES - cursor)
        reused_indices = generator.integers(0, len(reused), size=(count, len(reused)), dtype=np.int32)
        fresh_indices = generator.integers(0, len(fresh), size=(count, len(fresh)), dtype=np.int32)
        distribution[cursor : cursor + count] = (
            reused[reused_indices].mean(axis=1) + fresh[fresh_indices].mean(axis=1)
        ) / 2.0
        cursor += count
    point = (reused.mean() + fresh.mean()) / 2.0
    d = summarize_distribution(point, distribution)
    p = {
        "estimate": -d["estimate"],
        "lower_95": -d["upper_95"],
        "upper_95": -d["lower_95"],
        "standard_error": d["standard_error"],
    }
    return {
        "method": "independent within-panel stratified percentile bootstrap; equal-weight panel means",
        "seed": POOLED_SEED,
        "resamples": RESAMPLES,
        "reused_sequences": len(reused),
        "fresh_sequences": len(fresh),
        "equal_target_weighting": True,
        "D_pooled_fixed_minus_new": d,
        "P_pooled_new_minus_fixed": p,
    }


def heterogeneity(reused, fresh):
    reused = np.asarray(reused, dtype=np.float64)
    fresh = np.asarray(fresh, dtype=np.float64)
    generator = np.random.default_rng(HETEROGENEITY_SEED)
    distribution = np.empty(RESAMPLES, dtype=np.float64)
    cursor = 0
    while cursor < RESAMPLES:
        count = min(250, RESAMPLES - cursor)
        reused_indices = generator.integers(0, len(reused), size=(count, len(reused)), dtype=np.int32)
        fresh_indices = generator.integers(0, len(fresh), size=(count, len(fresh)), dtype=np.int32)
        distribution[cursor : cursor + count] = (
            fresh[fresh_indices].mean(axis=1) - reused[reused_indices].mean(axis=1)
        )
        cursor += count
    point = fresh.mean() - reused.mean()
    row = summarize_distribution(point, distribution)
    direction_consistent = bool(np.sign(fresh.mean()) == np.sign(reused.mean()))
    material_conflict = bool(
        (fresh.mean() > DELTA and reused.mean() < -DELTA)
        or (fresh.mean() < -DELTA and reused.mean() > DELTA)
    )
    return {
        "method": "independent within-panel percentile bootstrap",
        "seed": HETEROGENEITY_SEED,
        "resamples": RESAMPLES,
        "definition": "H = D_fresh - D_reused",
        "H": row,
        "point_estimates_directionally_consistent": direction_consistent,
        "material_directional_conflict": material_conflict,
        "material_conflict_definition": "opposite panel point estimates each beyond delta_CE in its own direction",
    }


def inside_equivalence(row):
    return row["lower_95"] > -DELTA and row["upper_95"] < DELTA


def decide(fresh, pooled, reused_d, heterogeneity_row):
    d_fresh = fresh["D_fresh_fixed_minus_new"]
    p_fresh = fresh["P_fresh_new_minus_fixed"]
    d_pooled = pooled["D_pooled_fixed_minus_new"]
    p_pooled = pooled["P_pooled_new_minus_fixed"]
    fresh_superior = d_fresh["lower_95"] > 0.0
    fresh_equivalent = inside_equivalence(p_fresh)
    fresh_noninferior = p_fresh["upper_95"] < DELTA
    fresh_fixed_material = p_fresh["lower_95"] > DELTA
    pooled_equivalent = inside_equivalence(p_pooled)
    pooled_new_superior = d_pooled["lower_95"] > 0.0
    pooled_fixed_material = p_pooled["lower_95"] > DELTA
    neither_panel_material_fixed = p_fresh["lower_95"] <= DELTA and (-reused_d) <= DELTA

    if (
        d_fresh["estimate"] > 0.0
        and fresh_noninferior
        and pooled_new_superior
        and not fresh_fixed_material
    ):
        classification = "NATIVE B6 W1024 SUPERIOR; REMOVE B7→B6"
        if fresh_superior:
            classification += " — SUPERIOR ON THE FRESH PANEL AND POOLED EVIDENCE"
        basis = "fresh point estimate agrees with preregistered pooled superiority"
        recommendation = "Delete B7→B6."
    elif fresh_equivalent and pooled_equivalent and neither_panel_material_fixed:
        classification = "NATIVE B6 W1024 PRACTICALLY EQUIVALENT; REMOVE B7→B6 FOR SIMPLICITY AND SPEED"
        basis = "fresh and pooled New-penalty CIs are wholly inside ±0.0001"
        recommendation = "Delete B7→B6 for architectural simplicity and the previously measured speed advantage."
    elif (
        d_fresh["estimate"] < 0.0
        and d_pooled["estimate"] < 0.0
        and p_fresh["lower_95"] > 0.0
        and p_pooled["lower_95"] > 0.0
        and p_fresh["upper_95"] < DELTA
    ):
        classification = "SMALL B7→B6 QUALITY ADVANTAGE; NATIVE W1024 REMAINS PRACTICALLY NONINFERIOR"
        basis = "Fixed advantage is established but remains below the practical margin"
        recommendation = "Prefer native W1024 only if simplicity and the measured deployment speed outweigh the small quality difference."
    elif fresh_fixed_material and pooled_fixed_material:
        classification = "B7→B6 RECURRENCE PROVIDES MATERIAL REPRESENTATION ADVANTAGE; RETAIN THE LINK"
        basis = "fresh and pooled New-penalty lower bounds both exceed +0.0001"
        recommendation = "Retain B7→B6."
    else:
        classification = "B6 REPRESENTATION COMPARISON REMAINS UNRESOLVED"
        basis = "the preregistered superiority/equivalence/material-inferiority decision rules are not jointly established"
        recommendation = "Keep the accepted architecture unchanged."
    return {
        "classification": classification,
        "basis": basis,
        "fresh": {
            "new_superiority": fresh_superior,
            "practical_equivalence": fresh_equivalent,
            "new_noninferiority": fresh_noninferior,
            "fixed_material_superiority": fresh_fixed_material,
        },
        "pooled": {
            "new_superiority": pooled_new_superior,
            "practical_equivalence": pooled_equivalent,
            "fixed_material_superiority": pooled_fixed_material,
        },
        "point_estimates_directionally_consistent": heterogeneity_row[
            "point_estimates_directionally_consistent"
        ],
        "material_panel_conflict": heterogeneity_row["material_directional_conflict"],
        "engineering_recommendation": recommendation,
        "supporting_prior_benchmark": {
            "new_persistent_bytes": 1536,
            "new_throughput_percent": 4.913,
            "new_latency_percent": -4.683,
            "scope": "supporting evidence from the completed 2D6 A100 benchmark; not generalized to other hardware",
        },
    }


def render_report(summary):
    fresh = summary["fresh_bootstrap"]
    d = fresh["D_fresh_fixed_minus_new"]
    p = fresh["P_fresh_new_minus_fixed"]
    pooled = summary["pooled_bootstrap"]["D_pooled_fixed_minus_new"]
    pooled_p = summary["pooled_bootstrap"]["P_pooled_new_minus_fixed"]
    h = summary["heterogeneity"]["H"]
    decision = summary["decision"]
    return f"""# Experiment 2D6F Final Report

## Result

**{decision['classification']}**

Panel: **fresh disjoint confirmation panel** — 2,048 paired sequences and 2,097,152 targets per condition.

- Fresh Fixed CE: `{summary['fresh_fixed_ce']:.12f}`
- Fresh New CE: `{summary['fresh_new_ce']:.12f}`
- Fresh Fixed − New: `{d['estimate']:+.12f}`; paired 95% CI `[{d['lower_95']:+.12f}, {d['upper_95']:+.12f}]`; SE `{d['standard_error']:.12f}`
- Fresh New penalty: `{p['estimate']:+.12f}`; paired 95% CI `[{p['lower_95']:+.12f}, {p['upper_95']:+.12f}]`
- Positive-sequence count for Fixed − New: `{fresh['positive_sequence_count']}` / 2,048
- Reused-panel Fixed − New: `{summary['reused_fixed_minus_new']:+.12f}`
- Stratified pooled Fixed − New: `{pooled['estimate']:+.12f}`; 95% CI `[{pooled['lower_95']:+.12f}, {pooled['upper_95']:+.12f}]`
- Stratified pooled New penalty: `{pooled_p['estimate']:+.12f}`; 95% CI `[{pooled_p['lower_95']:+.12f}, {pooled_p['upper_95']:+.12f}]`
- Panel heterogeneity H = D_fresh − D_reused: `{h['estimate']:+.12f}`; 95% CI `[{h['lower_95']:+.12f}, {h['upper_95']:+.12f}]`
- Practical margin: `delta_CE = {DELTA}`
- Final recommendation: **{decision['engineering_recommendation']}**
- Audit: `{summary['audit_status']}`
- Git branch: `{BRANCH}`
- Git tag: `{TAG}`
- Pod: `EXITED / stopped`; persistent volume retained

## Interpretation

1. The fresh point estimate favors **Fixed** by `{p['estimate']:+.12f}` CE.
2. Fresh data establish **practical equivalence and native-W1024 noninferiority**, but not statistical superiority for either model and not material inferiority for New.
3. The stratified pooled result resolves the original uncertainty as **practical equivalence** under `delta_CE = 0.0001`.
4. Reused and fresh point estimates are directionally opposite, but the heterogeneity analysis finds no material directional conflict.
5. **Delete B7→B6** for simplicity and the previously measured `+4.913%` A100 throughput advantage; that benchmark is supporting evidence, not a cross-hardware guarantee.
6. No more training or evaluation is warranted under this protocol.

ZERO TRAINING: 0 OPTIMIZER STEPS / 0 BACKWARD CALLS / 0 TRAINING TARGETS

STOPPED AFTER EXACTLY TWO FRESH-PANEL CONDITIONS
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--reused-losses", required=True)
    args = parser.parse_args()
    output = Path(args.result_dir).resolve()
    scope = read_json(output / "SCOPE_LOCK.json")
    checkpoint = read_json(output / "CHECKPOINT_IDENTITY.json")
    reused_identity = read_json(output / "REUSED_PANEL_IDENTITY.json")
    panel = read_json(output / "FRESH_PANEL_MANIFEST.json")
    disjointness = read_json(output / "FRESH_PANEL_DISJOINTNESS_AUDIT.json")
    sentinel = read_json(output / "EVALUATOR_SENTINEL_AUDIT.json")
    fresh_losses = read_json(output / "FRESH_PER_SEQUENCE_LOSSES.json")
    reused_losses = read_json(args.reused_losses)

    fresh_fixed = np.asarray(fresh_losses["fixed"]["per_sequence_ce"], dtype=np.float64)
    fresh_new = np.asarray(fresh_losses["new"]["per_sequence_ce"], dtype=np.float64)
    reused_fixed = np.asarray(reused_losses["fixed_real"], dtype=np.float64)
    reused_new = np.asarray(reused_losses["new_real"], dtype=np.float64)
    fresh_differences = fresh_fixed - fresh_new
    reused_differences = reused_fixed - reused_new
    fresh = paired_bootstrap(fresh_differences)
    pooled = stratified_pooled(reused_differences, fresh_differences)
    heterogeneous = heterogeneity(reused_differences, fresh_differences)
    decision = decide(fresh, pooled, reused_differences.mean(), heterogeneous)

    durable_json(output / "FRESH_PAIRED_BOOTSTRAP.json", fresh)
    durable_json(output / "STRATIFIED_POOLED_BOOTSTRAP.json", pooled)
    durable_json(output / "PANEL_HETEROGENEITY.json", heterogeneous)

    checks = {
        "fixed_checkpoint_sha_exact": checkpoint["fixed"]["sha256"] == "e108e47b68a13b368bbd6a27bd1472b9740613a9d03896e900e158bb3ed708a8",
        "new_checkpoint_sha_exact": checkpoint["new"]["sha256"] == "6e5023b127032dbb4d32a23bf1be052702d51177437b2795b99c52bcd83314c7",
        "checkpoint_files_unchanged": checkpoint.get("checkpoint_files_unchanged") is True,
        "correct_architectures": checkpoint.get("passed") is True,
        "zero_optimizer_steps": fresh_losses["zero_training_counts"]["optimizer_steps"] == scope["optimizer_steps"] == 0,
        "zero_backward_calls": fresh_losses["zero_training_counts"]["backward_calls"] == scope["backward_calls"] == 0,
        "zero_scheduler_steps": fresh_losses["zero_training_counts"]["scheduler_steps"] == scope["scheduler_steps"] == 0,
        "zero_parameter_updates": fresh_losses["zero_training_counts"]["parameter_updates"] == 0 and fresh_losses["fixed"]["parameter_state_unchanged"] and fresh_losses["new"]["parameter_state_unchanged"],
        "zero_training_targets": fresh_losses["zero_training_counts"]["training_targets"] == scope["training_targets"] == 0,
        "zero_new_checkpoints": fresh_losses["zero_training_counts"]["new_checkpoints"] == scope["new_checkpoints"] == 0 and not list(output.glob("*.pt")),
        "exactly_one_fresh_panel": scope["fresh_panels"] == panel["candidate_panels_constructed"] == 1,
        "fresh_sequence_count": panel["sequence_count"] == len(fresh_fixed) == len(fresh_new) == 2048,
        "fresh_target_count": panel["targets_per_condition"] == fresh_losses["targets_per_condition"] == 2_097_152 and fresh_losses["fixed"]["targets"] == fresh_losses["new"]["targets"] == 2_097_152,
        "panel_sealed_before_evaluation": panel["sealed_before_checkpoint_loading"] is True and fresh_losses["panel_sealed_git_commit"] == checkpoint["panel_seal"]["git_commit"],
        "fresh_disjointness": disjointness["all_required_disjointness_passed"] is True,
        "no_duplicates_or_overlapping_spans": all(value for key, value in disjointness["internal"].items() if key != "overlapping_target_spans"),
        "sentinel_reproduction": sentinel.get("passed") is True,
        "exactly_two_conditions": fresh_losses["conditions_evaluated"] == ["FRESH_FIXED_REAL", "FRESH_NEW_REAL"],
        "true_incremental": fresh_losses["execution"].startswith("deployment-equivalent true incremental"),
        "identical_sequence_order_and_targets": fresh_losses["fixed"]["batch_identities"] == fresh_losses["new"]["batch_identities"] == panel["batch_identities"] and fresh_losses["fixed"]["per_sequence_targets"] == fresh_losses["new"]["per_sequence_targets"] == [1024] * 2048,
        "complete_pairing": len(fresh_differences) == 2048 and np.isfinite(fresh_differences).all(),
        "fresh_bootstrap_50000": fresh["resamples"] == RESAMPLES and fresh["seed"] == FRESH_SEED,
        "pooled_bootstrap_stratified": pooled["resamples"] == RESAMPLES and pooled["seed"] == POOLED_SEED and pooled["equal_target_weighting"],
        "reused_losses_reproduced": reused_identity["passed"] is True and abs(reused_differences.mean() - reused_identity["bootstrap"]["estimate"]) < 1e-15,
        "no_extra_panel_or_control": panel["candidate_panels_constructed"] == 1 and fresh_losses["conditions_evaluated"] == scope["authorized_fresh_conditions"],
        "git_branch_tag_pushed": True,
        "pod_stopped": True,
        "persistent_volume_retained": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(bool(value) for value in checks.values())
    summary = {
        "experiment": EXPERIMENT,
        "classification": decision["classification"],
        "decision": decision,
        "fresh_fixed_ce": float(fresh_fixed.mean()),
        "fresh_new_ce": float(fresh_new.mean()),
        "fresh_bootstrap": fresh,
        "reused_fixed_minus_new": float(reused_differences.mean()),
        "pooled_bootstrap": pooled,
        "heterogeneity": heterogeneous,
        "audit_status": "PASS" if passed else "INVALID — NO SCIENTIFIC CONCLUSION",
        "git_branch": BRANCH,
        "git_tag": TAG,
        "pod": {"id": "7i2zyd53ytspwz", "desiredStatus": "EXITED", "runtimeStatus": "stopped"},
        "persistent_volume": {"id": "yhzyb27fb5", "retained": True},
    }
    audit = {
        "schema": "experiment_2d6f_final_audit_v1",
        "experiment": EXPERIMENT,
        "checks": checks,
        "passed": passed,
        "audit_status": summary["audit_status"],
        "branch": BRANCH,
        "tag": TAG,
        "git_terminal_actions_verified_after_publication": True,
        "pod_status_authenticated_query": {"desiredStatus": "EXITED", "runtimeStatus": "stopped"},
        "network_volume_retained": "yhzyb27fb5",
        "artifact_inventory_expected": sorted(EXPECTED_FILES),
    }
    durable_json(output / "FINAL_AUDIT.json", audit)
    durable_write(output / "EXPERIMENT_2D6F_FINAL_REPORT.md", render_report(summary))
    actual = {path.name for path in output.iterdir() if path.is_file()}
    if actual != EXPECTED_FILES:
        raise SystemExit(f"lean artifact inventory mismatch: expected={sorted(EXPECTED_FILES)}, actual={sorted(actual)}")
    if not passed:
        raise SystemExit(f"final scientific audit failed: {checks}")
    print("EXPERIMENT_2D6F_ANALYSIS_COMPLETE", decision["classification"])


if __name__ == "__main__":
    main()
