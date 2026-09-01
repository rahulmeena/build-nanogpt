#!/usr/bin/env python3
"""CPU-only local postflight analysis for Experiment 2D6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


EXPERIMENT = "2D6"
BRANCH = "experiment-2d6-b6-w1024-no-b7-recurrence-matched-100m"
FINAL_TAG = BRANCH + "-final"
FIXED_EXPECTED_CE = 3.044323022936
BOOTSTRAP_SEED = 2_026_090_2
BOOTSTRAP_RESAMPLES = 50_000
DELTA_CE = 0.0001


def read_json(path):
    return json.loads(Path(path).read_text())


def durable_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def paired_bootstrap(arrays):
    length = len(next(iter(arrays.values())))
    if length != 2048 or any(len(values) != length for values in arrays.values()):
        raise SystemExit("paired bootstrap requires 2,048 aligned sequences")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    distributions = {
        name: np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64) for name in arrays
    }
    batch = 250
    cursor = 0
    while cursor < BOOTSTRAP_RESAMPLES:
        count = min(batch, BOOTSTRAP_RESAMPLES - cursor)
        indices = generator.integers(0, length, size=(count, length), dtype=np.int32)
        for name, values in arrays.items():
            distributions[name][cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    contrasts = {}
    for name, values in arrays.items():
        distribution = distributions[name]
        contrasts[name] = {
            "estimate": float(np.mean(values)),
            "lower_95": float(np.percentile(distribution, 2.5)),
            "upper_95": float(np.percentile(distribution, 97.5)),
        }
    return {
        "method": "paired per-sequence percentile bootstrap",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "paired_sequences": length,
        "identical_resample_indices_all_contrasts": True,
        "contrasts": contrasts,
    }


def interval_overlaps(row, boundary):
    return row["lower_95"] <= boundary <= row["upper_95"]


def classify(primary_penalty):
    borderline = any(
        interval_overlaps(primary_penalty, boundary)
        for boundary in (-DELTA_CE, 0.0, DELTA_CE)
    )
    if borderline:
        label = "B6 REPRESENTATION COMPARISON UNRESOLVED"
        reason = "reused-panel CI overlaps zero or a binding practical boundary"
    elif primary_penalty["upper_95"] < 0.0:
        label = "B6 NATIVE W1024 SUPERIOR; B7→B6 RECURRENCE NOT JUSTIFIED"
        reason = "complete Fixed-minus-New CI is above zero"
    elif primary_penalty["lower_95"] > DELTA_CE:
        label = "B7→B6 RECURRENCE PROVIDES MATERIAL REPRESENTATION ADVANTAGE"
        reason = "complete New-minus-Fixed CI is above +0.0001"
    elif primary_penalty["lower_95"] > 0.0 and primary_penalty["upper_95"] < DELTA_CE:
        label = "SMALL RECURRENT QUALITY ADVANTAGE; NATIVE W1024 REMAINS PRACTICALLY NONINFERIOR"
        reason = "small Fixed advantage is wholly inside the +0.0001 margin"
    elif primary_penalty["lower_95"] > -DELTA_CE and primary_penalty["upper_95"] < DELTA_CE:
        label = "B6 NATIVE W1024 PRACTICALLY EQUIVALENT; PREFER SIMPLER PATH IF SPEED IS NOT WORSE"
        reason = "complete two-sided CI lies inside ±0.0001"
    elif primary_penalty["upper_95"] < DELTA_CE:
        label = "B6 NATIVE W1024 PRACTICALLY NONINFERIOR"
        reason = "upper penalty bound is below +0.0001"
    else:
        label = "B6 REPRESENTATION COMPARISON UNRESOLVED"
        reason = "binding superiority/noninferiority/equivalence rules are not established"
    return {
        "classification": label,
        "reason": reason,
        "delta_ce": DELTA_CE,
        "borderline_reused_panel_rule_applied": borderline,
        "fresh_panel_confirmation_needed": borderline,
    }


def render_report(summary):
    primary = summary["bootstrap"]["contrasts"]["new_minus_fixed_penalty"]
    inverse = summary["bootstrap"]["contrasts"]["fixed_minus_new"]
    off = summary["bootstrap"]["contrasts"]["fixed_off_minus_fixed_real"]
    memory = summary["memory_speed"]["new_minus_fixed"]
    classification = summary["classification"]
    fresh = "Yes" if classification["fresh_panel_confirmation_needed"] else "No"
    return f"""# Experiment 2D6 Final Report

## Result

**{classification['classification']}**

Evaluation set: **reused sealed matched panel** (2,048 paired sequences; 2,097,152 targets per condition).

- Fixed REAL CE: `{summary['fixed_real_ce']:.12f}`
- New REAL CE: `{summary['new_real_ce']:.12f}`
- Fixed − New: `{inverse['estimate']:+.12f}`; paired 95% CI `[{inverse['lower_95']:+.12f}, {inverse['upper_95']:+.12f}]`
- New penalty (New − Fixed): `{primary['estimate']:+.12f}`; paired 95% CI `[{primary['lower_95']:+.12f}, {primary['upper_95']:+.12f}]`
- Binding practical margin: `delta_CE = {DELTA_CE}`
- Zero-training geometry shock (New geometry − Original): `{summary['zero_training_shock']['parent_new_minus_original_ce']:+.12f}`
- Fixed B6-OFF effect (OFF − REAL): `{off['estimate']:+.12f}`; paired 95% CI `[{off['lower_95']:+.12f}, {off['upper_95']:+.12f}]`
- Persistent state, New − Fixed: `{memory['physical_unique_bytes']:+,d}` bytes physical (`{memory['logical_persistent_bytes']:+,d}` logical)
- Median latency, New − Fixed: `{memory['latency_seconds']:+.6f}` s (`{memory['latency_percent']:+.3f}%`)
- Median throughput, New − Fixed: `{memory['tokens_per_second']:+.3f}` token/s (`{memory['throughput_percent']:+.3f}%`)
- Final checkpoint SHA-256: `{summary['final_checkpoint']['sha256']}`
- Audit: `{summary['audit_status']}`
- Git branch: `{BRANCH}`
- Git tag: `{FINAL_TAG}`
- Pod status: `STOPPED`; volume retained

## Answers

1. Native B6 W1024 is classified as: **{classification['classification']}**.
2. B7→B6 inside mature W512 changes CE by `{off['estimate']:+.12f}` when removed (OFF − REAL); positive means recurrence helps.
3. The approximately equal-memory native architecture is preferable only if the classification and measured speed support it; the measured physical memory delta is `{memory['physical_unique_bytes']:+,d}` bytes.
4. Exact tradeoff: quality `{primary['estimate']:+.12f}` CE penalty, memory `{memory['physical_unique_bytes']:+,d}` bytes, throughput `{memory['throughput_percent']:+.3f}%`.
5. Fresh-panel confirmation needed: **{fresh}**.
6. No further training is warranted.

The Fixed stored losses were reused only after sentinel reproduction. This was not a fresh confirmation set.

STOPPED AFTER ONE NEW ARM AT EXACTLY 191 UPDATES / 100,139,008 TARGETS
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--local-checkpoint", required=True)
    args = parser.parse_args()
    output = Path(args.result_dir).resolve()
    fixed_stored = read_json(output / "FIXED_LARGE.json")
    sentinel = read_json(output / "FIXED_REAL_SENTINEL.json")
    new = read_json(output / "NEW_REAL.json")
    fixed_off = read_json(output / "FIXED_B6_RECURRENCE_OFF.json")
    shock = read_json(output / "ZERO_TRAINING_SHOCK.json")
    memory = read_json(output / "MEMORY_SPEED_AUDIT.json")
    final_checkpoint = read_json(output / "FINAL_CHECKPOINT_PROVENANCE.json")
    training = read_json(output / "TRAINING_COMPLETE.json")
    stop = read_json(output / "STOP_VERIFICATION.json")
    backup = read_json(output / "LOCAL_BACKUP_AUDIT.json")

    fixed_row = fixed_stored["conditions"]["all_real"]
    fixed_values = np.asarray(fixed_row["per_sequence_ce"], dtype=np.float64)
    new_values = np.asarray(new["per_sequence_ce"], dtype=np.float64)
    off_values = np.asarray(fixed_off["per_sequence_ce"], dtype=np.float64)
    sentinel_expected = fixed_values[: len(sentinel["per_sequence_ce"])]
    sentinel_actual = np.asarray(sentinel["per_sequence_ce"], dtype=np.float64)
    sentinel_max_abs = float(np.max(np.abs(sentinel_expected - sentinel_actual)))
    arrays = {
        "fixed_minus_new": fixed_values - new_values,
        "new_minus_fixed_penalty": new_values - fixed_values,
        "fixed_off_minus_fixed_real": off_values - fixed_values,
        "fixed_off_minus_new": off_values - new_values,
    }
    bootstrap = paired_bootstrap(arrays)
    classification = classify(bootstrap["contrasts"]["new_minus_fixed_penalty"])
    checkpoint_sha = sha256(args.local_checkpoint)
    fixed_ce = float(fixed_values.mean())
    new_ce = float(new_values.mean())
    off_ce = float(off_values.mean())
    checks = {
        "training_complete": training.get("passed") is True,
        "new_complete": new.get("passed") is True and new.get("paired_sequences") == 2048,
        "fixed_off_complete": fixed_off.get("passed") is True and fixed_off.get("paired_sequences") == 2048,
        "stored_fixed_complete": fixed_stored.get("passed") is True and len(fixed_values) == 2048,
        "fixed_ce_anchor": abs(fixed_ce - FIXED_EXPECTED_CE) < 5e-13,
        "fixed_sentinel_reproduced": sentinel_max_abs <= 1e-12,
        "panel_exact": fixed_stored["panel_sha256"] == new["panel_sha256"] == fixed_off["panel_sha256"],
        "pairing_exact": len(fixed_values) == len(new_values) == len(off_values) == 2048,
        "bootstrap": bootstrap["resamples"] == BOOTSTRAP_RESAMPLES,
        "memory_speed": memory.get("passed") is True,
        "zero_training_shock": shock.get("passed") is True,
        "final_checkpoint_reopen": final_checkpoint.get("passed") is True,
        "local_remote_checkpoint_sha": checkpoint_sha == final_checkpoint["checkpoint_sha256"] == backup["remote_sha256"],
        "pod_stopped": stop.get("pod", {}).get("desiredStatus") == "EXITED" and stop.get("pod", {}).get("runtimeStatus") == "stopped",
        "volume_retained": stop.get("network_volume_retained") is True,
    }
    passed = all(checks.values())
    if not passed:
        classification = {
            **classification,
            "classification": "INVALID — NO SCIENTIFIC CONCLUSION",
            "reason": "critical scientific or operational audit failed",
        }
    losses = {
        "schema": "experiment_2d6_large_paired_losses_v1",
        "evaluation_set_label": "reused sealed matched panel",
        "panel_sha256": new["panel_sha256"],
        "paired_sequences": 2048,
        "targets_per_condition": 2_097_152,
        "fixed_real": fixed_values.tolist(),
        "new_real": new_values.tolist(),
        "fixed_b6_recurrence_off": off_values.tolist(),
    }
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "fixed_real_ce": fixed_ce,
        "new_real_ce": new_ce,
        "fixed_b6_off_ce": off_ce,
        "bootstrap": bootstrap,
        "zero_training_shock": shock,
        "memory_speed": memory,
        "fixed_sentinel_max_abs_ce": sentinel_max_abs,
        "final_checkpoint": {"path": str(Path(args.local_checkpoint).resolve()), "sha256": checkpoint_sha},
        "audit_status": "PASS" if passed else "INVALID — NO SCIENTIFIC CONCLUSION",
        "git_branch": BRANCH,
        "git_tag": FINAL_TAG,
        "pod_status": "STOPPED",
    }
    durable_json(output / "LARGE_PAIRED_LOSSES.json", losses)
    durable_json(output / "PAIRED_BOOTSTRAP.json", bootstrap)
    durable_json(output / "SCIENTIFIC_RESULT_SUMMARY.json", summary)
    durable_json(output / "FINAL_AUDIT.json", {
        "schema": "experiment_2d6_final_audit_v1",
        "experiment": EXPERIMENT,
        "checks": checks,
        "audit_status": summary["audit_status"],
        "passed": passed,
    })
    durable_write(output / "EXPERIMENT_2D6_FINAL_REPORT.md", render_report(summary))
    if not passed:
        raise SystemExit(f"final 2D6 audit failed: {checks}")
    print("EXPERIMENT_2D6_ANALYSIS_COMPLETE", classification["classification"])


if __name__ == "__main__":
    main()
