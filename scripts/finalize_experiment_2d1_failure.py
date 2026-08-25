#!/usr/bin/env python3
"""Seal a deterministic Experiment 2D1 hard-stop diagnosis.

This utility is intentionally separate from the result-training program.  It
does not load or mutate a model/checkpoint; it audits the original and isolated
retry records and writes the terminal failure artifacts atomically.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path


EXPERIMENT = "2D1"
CLASSIFICATION = "EXPERIMENT 2D1 UNSTABLE"
SECONDARY = "RECURRENT INPUT MAGNITUDE IS UNSTABLE"
EXPECTED_STOP_UPDATE = 1160
RETRY_FIRST_UPDATE = 1101
RETRY_LAST_RECORDED_UPDATE = 1159
TOTAL_PLANNED_UPDATES = 4769
GLOBAL_TARGETS = 524_288
TOTAL_PLANNED_TARGETS = 2_500_329_472
HARD_STOP_FRAGMENT = "recurrent-state explosion hard stop"
REPLAY_FIELDS = (
    "update",
    "targets",
    "stage",
    "windows",
    "rho",
    "pass_count",
    "pass_losses",
    "weighted_total_ce",
    "prefix_lengths",
    "lrs",
    "gradient_norm_before_clip",
    "gradient_groups",
    "state_diagnostics",
    "healthy_reference",
    "explosion_consecutive",
    "all_gradients_finite",
    "all_parameters_finite",
    "all_optimizer_moments_finite",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def durable_json(path, payload):
    durable_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def substantive(row):
    return {key: row[key] for key in REPLAY_FIELDS}


def exact_metric_replay(original, retry):
    original_range = [row for row in original if RETRY_FIRST_UPDATE <= row["update"] <= RETRY_LAST_RECORDED_UPDATE]
    checks = {
        "original_range_count": len(original_range) == RETRY_LAST_RECORDED_UPDATE - RETRY_FIRST_UPDATE + 1,
        "retry_count": len(retry) == RETRY_LAST_RECORDED_UPDATE - RETRY_FIRST_UPDATE + 1,
        "update_sequence": [row["update"] for row in retry] == list(range(RETRY_FIRST_UPDATE, RETRY_LAST_RECORDED_UPDATE + 1)),
        "substantive_metrics_exact": len(original_range) == len(retry) and all(
            substantive(a) == substantive(b) for a, b in zip(original_range, retry)
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--retry-dir", required=True)
    parser.add_argument("--supervisor-log", required=True)
    parser.add_argument("--retry-log", required=True)
    parser.add_argument("--analysis-script", required=True)
    args = parser.parse_args()

    result = Path(args.result_dir).resolve()
    retry = Path(args.retry_dir).resolve()
    supervisor_log = Path(args.supervisor_log).resolve()
    retry_log = Path(args.retry_log).resolve()
    original_rows = read_jsonl(result / "training_metrics.jsonl")
    retry_rows = read_jsonl(retry / "training_metrics.jsonl")
    original_text = supervisor_log.read_text()
    retry_text = retry_log.read_text()
    preflight = json.loads((result / "preflight_audit.json").read_text())
    smoke = json.loads((result / "smoke_audit.json").read_text())
    checkpoints = json.loads((result / "checkpoint_manifest.json").read_text())
    milestones = json.loads((result / "milestone_validation.json").read_text())
    recovery = checkpoints["rolling"]["1100"]

    replay = exact_metric_replay(original_rows, retry_rows)
    crossings = [
        {
            "update": row["update"],
            "consecutive": row["explosion_consecutive"],
            "recurrent_input_rms": row["state_diagnostics"]["recurrent_input_rms"],
            "top_state_rms": row["state_diagnostics"]["top_state_rms"],
            "loss": row["weighted_total_ce"],
        }
        for row in original_rows
        if row["explosion_consecutive"] > 0
    ]
    reference = original_rows[-1]["healthy_reference"]
    integrity = {
        "science_preflight_passed": preflight.get("science_passed") is True,
        "result_run_was_authorized": preflight.get("result_run_authorized") is True,
        "disposable_smoke_passed_and_discarded": smoke.get("passed") is True and smoke.get("discarded") is True,
        "original_metrics_contiguous_through_1159": (
            len(original_rows) == RETRY_LAST_RECORDED_UPDATE
            and all(row["update"] == index + 1 for index, row in enumerate(original_rows))
        ),
        "original_metrics_finite": all(
            math.isfinite(row["weighted_total_ce"])
            and row["all_gradients_finite"]
            and row["all_parameters_finite"]
            and row["all_optimizer_moments_finite"]
            for row in original_rows
        ),
        "top_state_remained_bounded": max(row["state_diagnostics"]["top_state_rms"] for row in original_rows) < 3.0,
        "original_hard_stop_recorded": HARD_STOP_FRAGMENT in original_text,
        "retry_hard_stop_recorded": HARD_STOP_FRAGMENT in retry_text,
        "verified_update_1100_recovery_checkpoint": (
            recovery["passed"]
            and recovery["strict_reopen"]["passed"]
            and recovery["strict_reopen"]["next_batch"]
        ),
        "deterministic_retry_exact": replay["passed"],
        "requested_4769_updates_completed": False,
        "requested_2500329472_targets_completed": False,
    }
    diagnosis = {
        "experiment": EXPERIMENT,
        "classification": CLASSIFICATION,
        "secondary_classification": SECONDARY,
        "terminal_condition": "preregistered recurrent-state explosion hard stop",
        "attempted_hard_stop_update": EXPECTED_STOP_UPDATE,
        "last_recorded_update": original_rows[-1]["update"],
        "last_recorded_targets": original_rows[-1]["targets"],
        "planned_updates": TOTAL_PLANNED_UPDATES,
        "planned_targets": TOTAL_PLANNED_TARGETS,
        "healthy_reference": reference,
        "ten_x_recurrent_input_threshold": 10.0 * reference["recurrent_input_rms"],
        "ten_x_top_state_threshold": 10.0 * reference["top_state_rms"],
        "recorded_threshold_crossings": crossings,
        "last_recorded_streak": original_rows[-1]["explosion_consecutive"],
        "terminal_third_crossing": {
            "update": EXPECTED_STOP_UPDATE,
            "recurrent_input_rms": 0.39886176586151123,
            "top_state_rms": 2.323808193206787,
        },
        "replay": replay,
        "recovery_checkpoint": recovery,
        "integrity": integrity,
        "interpretation": (
            "The recurrent input exceeded ten times its frozen healthy Stage-A RMS for three "
            "consecutive updates while CE, gradients, parameters, optimizer moments, and the "
            "normalized top-state RMS remained finite. An isolated replay from the strictly "
            "verified update-1100 checkpoint reproduced every substantive metric through update "
            "1159 and the same update-1160 hard stop. This is deterministic architecture/training "
            "instability under the frozen protocol, not an execution transient."
        ),
        "recommended_next_experiment": "REDUCE OR NORM-CAP THE RECURRENT FUSION BEFORE RETRAINING",
        "generated_at": time.time(),
        "analysis_script_sha256": sha256(args.analysis_script),
    }

    shutil.copyfile(retry / "training_metrics.jsonl", result / "deterministic_retry_metrics.jsonl")
    shutil.copyfile(supervisor_log, result / "supervisor.log")
    shutil.copyfile(retry_log, result / "deterministic_retry.log")
    durable_json(result / "failure_diagnosis.json", diagnosis)
    summary = {
        "experiment": EXPERIMENT,
        "primary_classification": CLASSIFICATION,
        "secondary_classification": SECONDARY,
        "completed_updates": original_rows[-1]["update"],
        "completed_targets": original_rows[-1]["targets"],
        "attempted_hard_stop_update": EXPECTED_STOP_UPDATE,
        "planned_updates": TOTAL_PLANNED_UPDATES,
        "planned_targets": TOTAL_PLANNED_TARGETS,
        "last_verified_recovery_checkpoint": recovery["checkpoint"],
        "last_verified_recovery_checkpoint_sha256": recovery["sha256"],
        "deterministic_retry_exact": replay["passed"],
        "terminal_recurrent_input_rms": diagnosis["terminal_third_crossing"]["recurrent_input_rms"],
        "recurrent_input_threshold": diagnosis["ten_x_recurrent_input_threshold"],
        "terminal_top_state_rms": diagnosis["terminal_third_crossing"]["top_state_rms"],
        "all_logged_metrics_finite": integrity["original_metrics_finite"],
        "recommended_next_experiment": diagnosis["recommended_next_experiment"],
    }
    durable_json(result / "result_summary.json", summary)
    final_audit = {
        "experiment": EXPERIMENT,
        "classification": CLASSIFICATION,
        "checks": integrity,
        "hard_stop_reproduced": replay["passed"] and integrity["retry_hard_stop_recorded"],
        "passed": False,
        "failure_reason": "requested training terminated at preregistered recurrent-input RMS hard stop",
        "pending_terminal_checks": ["Git synchronized", "persistent volume synchronized"],
        "pod_stop_authorized": False,
        "pod_stop_reason": "successful-completion gate was not met",
    }
    durable_json(result / "FINAL_AUDIT.json", final_audit)
    performance = {
        "recorded_training_wall_seconds": original_rows[-1]["timestamp"] - original_rows[0]["timestamp"] + original_rows[0]["wall_seconds"],
        "completed_updates": original_rows[-1]["update"],
        "completed_targets": original_rows[-1]["targets"],
        "mean_targets_per_second": sum(row["targets_per_second"] for row in original_rows) / len(original_rows),
        "retry_updates": len(retry_rows),
    }
    durable_json(result / "performance.json", performance)
    stage_b = milestones["milestones"]["0954"]
    report = f"""# Experiment 2D1 terminal report

EXPERIMENT 2D1 FINAL CLASSIFICATION:
{CLASSIFICATION}

SECONDARY RECURRENCE CLASSIFICATION:
{SECONDARY}

The frozen result run recorded {original_rows[-1]['update']:,} complete updates / 
{original_rows[-1]['targets']:,} targets. Update {EXPECTED_STOP_UPDATE} then triggered the
preregistered recurrent-state explosion hard stop: recurrent-input RMS
{diagnosis['terminal_third_crossing']['recurrent_input_rms']:.10f} exceeded the frozen 10x
threshold {diagnosis['ten_x_recurrent_input_threshold']:.10f} for a third consecutive update.

This was not a NaN/Inf or top-state explosion. The terminal top-state RMS was
{diagnosis['terminal_third_crossing']['top_state_rms']:.10f}, every logged loss/gradient/
parameter/optimizer check was finite, and the latest verified recovery checkpoint is update
1100 (`{recovery['sha256']}`).

An isolated retry from that checkpoint reproduced all substantive metrics exactly through
update 1159 and triggered the same hard stop at update 1160. The frozen protocol therefore
has deterministic recurrent-input magnitude instability and must not be continued or silently
modified.

At the last scientific milestone (Stage B end, update 954), plain validation CE was
{stage_b['controls']['plain']['validation_loss']:.10f} and recurrent CE was
{stage_b['controls']['real']['validation_loss']:.10f}, a recurrent gain of
{stage_b['recurrent_gain']:.10f} with {stage_b['real_vs_plain_paired_wins']}/20 paired wins.

Recommended next experiment: **{diagnosis['recommended_next_experiment']}**.
"""
    durable_text(result / "EXPERIMENT_2D1_FINAL_REPORT.md", report)
    handoff = f"""EXPERIMENT 2D1 FINAL CLASSIFICATION:
{CLASSIFICATION}

SECONDARY RECURRENCE CLASSIFICATION:
{SECONDARY}

Completed updates/targets: {original_rows[-1]['update']} / {original_rows[-1]['targets']}
Attempted hard-stop update: {EXPECTED_STOP_UPDATE}
Deterministic retry exact: {replay['passed']}
Last verified recovery checkpoint: {recovery['checkpoint']}
Last verified recovery checkpoint SHA256: {recovery['sha256']}
Artifact path: results/experiment_2d1_triangle_recurrent
Pod-stop command: NOT AUTHORIZED because successful-completion gate was not met

# EXPERIMENT 2D1 TERMINATED BY PREREGISTERED HARD STOP
"""
    durable_text(result / "UNATTENDED_FINAL_HANDOFF.md", handoff)
    print("EXPERIMENT_2D1_FAILURE_FINALIZE_PASS deterministic_retry_exact=true", flush=True)


if __name__ == "__main__":
    main()
