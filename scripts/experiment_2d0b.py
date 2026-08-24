#!/usr/bin/env python3
"""Experiment 2D0B: resolve the B11 sensitivity curve between W128 and W1."""

import argparse
import json
import math
import platform
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0 as d0  # noqa: E402
import experiment_2d0a as d0a  # noqa: E402


EXPERIMENT = "2D0B"
PROTOCOL = "exp2d0b_b11_micro_window_sweep_v1"
BRANCH = "experiment-2d0b-b11-micro-window-sweep"
FROZEN_TAG = "experiment-2d0a-b11-extreme-window-sweep-final"
FROZEN_COMMIT = "f939e29792ab45e5463eda94bbb2296efa59a019"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2d0b_b11_micro_window_sweep.json"
PREVIOUS_RESULTS = (
    REPO_ROOT
    / "results"
    / "experiment_2d0a_b11_extreme_window_sweep"
    / "result_summary.json"
)
PREVIOUS_EXTREME = (
    REPO_ROOT
    / "results"
    / "experiment_2d0a_b11_extreme_window_sweep"
    / "extreme_window_results.json"
)
OUTPUT_NAME = "experiment_2d0b_b11_micro_window_sweep"
NEW_WINDOWS = (64, 32, 16, 8, 4, 2)
GPU_BY_WINDOW = {64: 0, 32: 1, 16: 2, 8: 3, 4: 0, 2: 1}
WAVES = ((64, 32, 16, 8), (4, 2))
COMBINED_WINDOWS = (1024, 896, 768, 512, 384, 256, 128, 64, 32, 16, 8, 4, 2, 1)
SENTINEL_TOLERANCE = 1e-8


def configure_reused_evaluator():
    """Point the frozen 2D0A evaluator at the 2D0B preregistration."""
    d0a.EXPERIMENT = EXPERIMENT
    d0a.PROTOCOL = PROTOCOL
    d0a.BRANCH = BRANCH
    d0a.FROZEN_TAG = FROZEN_TAG
    d0a.FROZEN_COMMIT = FROZEN_COMMIT
    d0a.CONFIG_PATH = CONFIG_PATH
    d0a.NEW_WINDOWS = NEW_WINDOWS
    d0a.GPU_BY_WINDOW = GPU_BY_WINDOW


configure_reused_evaluator()


def load_config():
    config = d0a.load_config()
    if tuple(tuple(row) for row in config["evaluation"]["waves"]) != WAVES:
        raise SystemExit("2D0B wave geometry mismatch")
    return config


def require_git(clean=True):
    d0a.require_git(clean=clean)


def run_sentinel(args):
    load_config()
    d0a.run_sentinel(args)


def run_worker(args):
    load_config()
    d0a.run_worker(args)


def smallest_window(damages, threshold):
    candidates = [window for window in sorted(COMBINED_WINDOWS) if damages[window] <= threshold]
    return candidates[0] if candidates else None


def run_assemble(args):
    require_git(clean=True)
    load_config()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sentinel = json.loads((run_root / "regression_sentinel.json").read_text())
    previous = json.loads(PREVIOUS_RESULTS.read_text())
    workers = {
        window: json.loads((run_root / "workers" / f"window_{window}.json").read_text())
        for window in NEW_WINDOWS
    }
    archived_full = previous["combined_rows"]["1024"]["validation_loss"]
    archived_full_batches = json.loads(
        (
            REPO_ROOT
            / "results"
            / "experiment_2d0_standard_b11_context_completion"
            / "phase_a_results.json"
        ).read_text()
    )["rows"]["1024"]["per_batch_losses"]
    full_control_deltas = {
        str(window): max(
            abs(left - right)
            for left, right in zip(row["per_batch_full_losses"], archived_full_batches)
        )
        for window, row in workers.items()
    }
    full_control_pass = all(value <= SENTINEL_TOLERANCE for value in full_control_deltas.values())

    combined_rows = dict(previous["combined_rows"])
    for window, row in workers.items():
        combined_rows[str(window)] = {
            "window": window,
            "validation_loss": row["validation_loss"],
            "damage_vs_1024": row["validation_loss"] - archived_full,
            "historical_kv_retained": window - 1,
            "kv_fraction": window / d0.T,
            "b11_cosine": row["b11_state_drift"]["all_positions"]["cosine"],
            "b11_rms": row["b11_state_drift"]["all_positions"]["rms_difference"],
            "b12_cosine": row["b12_state_drift"]["all_positions"]["cosine"],
            "b12_rms": row["b12_state_drift"]["all_positions"]["rms_difference"],
            "diagnostic_provenance": "Experiment 2D0B B12 post-block H12 state",
        }
    damages = {
        window: combined_rows[str(window)]["validation_loss"] - archived_full
        for window in COMBINED_WINDOWS
    }
    monotonicity_deviations = []
    for left, right in zip(COMBINED_WINDOWS, COMBINED_WINDOWS[1:]):
        if damages[right] < damages[left]:
            monotonicity_deviations.append(
                {
                    "from_window": left,
                    "to_window": right,
                    "from_damage": damages[left],
                    "to_damage": damages[right],
                    "decrease": damages[left] - damages[right],
                }
            )
    adjacent_increments = []
    micro_path = (128, 64, 32, 16, 8, 4, 2, 1)
    for left, right in zip(micro_path, micro_path[1:]):
        adjacent_increments.append(
            {
                "from_window": left,
                "to_window": right,
                "additional_damage": damages[right] - damages[left],
                "damage_at_to_window": damages[right],
            }
        )
    thresholds = {
        str(value): {
            "smallest_window": smallest_window(damages, value),
            "kv_fraction": None
            if smallest_window(damages, value) is None
            else smallest_window(damages, value) / d0.T,
        }
        for value in (0.01, 0.02, 0.05, 0.10, 0.20, 0.25)
    }
    integrity = {
        "sentinel": sentinel["passed"],
        "workers": all(row["passed"] for row in workers.values()),
        "canonical_validation": all(
            row["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256
            for row in workers.values()
        ),
        "full_control_per_batch_regression": full_control_pass,
        "source_exact": all(row["source_audit"]["passed"] for row in workers.values()),
        "model_immutable": all(row["model_tensors_unchanged"] for row in workers.values()),
        "finite": all(row["all_losses_and_activations_finite"] for row in workers.values()),
        "h10_identical": all(row["incoming_b11_h10"]["passed"] for row in workers.values()),
        "independent_processes": all(
            not row["distributed_initialized"] and row["visible_cuda_device_count"] == 1
            for row in workers.values()
        ),
        "zero_training": all(
            row["optimizer_objects"] == 0
            and row["backward_calls"] == 0
            and row["parameter_updates"] == 0
            and row["training_targets"] == 0
            for row in workers.values()
        ),
    }
    integrity["passed"] = all(integrity.values())
    performance_rows = {str(window): row["performance"] for window, row in workers.items()}
    wave_performance = []
    for index, wave in enumerate(WAVES, start=1):
        rows = [workers[window]["performance"] for window in wave]
        wave_performance.append(
            {
                "wave": index,
                "windows": list(wave),
                "elapsed_wall_seconds": max(row["wall_end_unix"] for row in rows)
                - min(row["wall_start_unix"] for row in rows),
            }
        )
    summary = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "new_windows": list(NEW_WINDOWS),
        "combined_windows": list(COMBINED_WINDOWS),
        "combined_rows": {str(window): combined_rows[str(window)] for window in COMBINED_WINDOWS},
        "damages": {str(window): damages[window] for window in COMBINED_WINDOWS},
        "monotonic_nondecreasing": not monotonicity_deviations,
        "monotonicity_deviations": monotonicity_deviations,
        "micro_window_adjacent_increments": adjacent_increments,
        "quality_thresholds": thresholds,
        "integrity_pre_audit": integrity,
        "training_performed": False,
    }
    artifacts = {
        "source_manifest.json": {
            "checkpoint": workers[64]["source_audit"]["checkpoint"],
            "checkpoint_sha256": d0.SOURCE_SHA256,
            "checkpoint_bytes": d0.SOURCE_BYTES,
            "historical_training_tokens": d0.SOURCE_TOKENS,
            "validation_shard": workers[64]["validation_shard"],
            "validation_shard_sha256": d0.VAL_SHA256,
            "canonical_validation_sha256": d0.CANONICAL_VALIDATION_SHA256,
            "frozen_2d0a_tag": FROZEN_TAG,
            "frozen_2d0a_commit": FROZEN_COMMIT,
            "architecture": workers[64]["source_audit"],
        },
        "micro_window_results.json": {
            "experiment": EXPERIMENT,
            "protocol": PROTOCOL,
            "rows": {str(window): row for window, row in workers.items()},
            "passed": integrity["passed"],
        },
        "combined_sensitivity_curve.json": summary["combined_rows"],
        "position_bin_loss.json": {
            str(window): row["position_bins"] for window, row in workers.items()
        },
        "b11_state_drift.json": {
            str(window): row["b11_state_drift"] for window, row in workers.items()
        },
        "b12_state_drift.json": {
            str(window): row["b12_state_drift"] for window, row in workers.items()
        },
        "logit_drift.json": {
            str(window): row["logit_drift"] for window, row in workers.items()
        },
        "per_position_loss.json": {
            str(window): row["per_position_loss"] for window, row in workers.items()
        },
        "performance.json": {
            "workers": performance_rows,
            "waves": wave_performance,
            "sum_of_wave_elapsed_wall_seconds": sum(
                row["elapsed_wall_seconds"] for row in wave_performance
            ),
        },
        "commands_and_runtime.json": {
            "branch": BRANCH,
            "implementation_commit": d0a.git_output("rev-parse", "HEAD"),
            "python": sys.version,
            "platform": platform.platform(),
            "commands": [
                "CUDA_VISIBLE_DEVICES=0 python scripts/experiment_2d0b.py sentinel ...",
                "wave 1: four independent workers for W64/W32/W16/W8",
                "wave 2: two independent workers for W4/W2",
                "python scripts/experiment_2d0b.py assemble ...",
                "python scripts/experiment_2d0b.py finalize ...",
            ],
            "ddp": False,
            "nccl": False,
            "training": False,
        },
        "result_summary.json": summary,
    }
    for name, payload in artifacts.items():
        d0.durable_json(output / name, payload)
    print(
        f"EXPERIMENT_2D0B_RESULTS_ASSEMBLED integrity={integrity['passed']} "
        f"monotonic={not monotonicity_deviations}",
        flush=True,
    )
    if not integrity["passed"]:
        raise SystemExit("2D0B assembled result integrity failed")


def fmt(value, digits=10):
    return f"{value:.{digits}f}"


def run_finalize(args):
    require_git(clean=True)
    load_config()
    output = Path(args.output_dir).resolve()
    summary = json.loads((output / "result_summary.json").read_text())
    micro = json.loads((output / "micro_window_results.json").read_text())
    performance = json.loads((output / "performance.json").read_text())
    workers = {int(key): value for key, value in micro["rows"].items()}
    required = (
        "source_manifest.json",
        "micro_window_results.json",
        "combined_sensitivity_curve.json",
        "position_bin_loss.json",
        "b11_state_drift.json",
        "b12_state_drift.json",
        "logit_drift.json",
        "per_position_loss.json",
        "performance.json",
        "commands_and_runtime.json",
        "result_summary.json",
    )
    checks = {
        "source checkpoint exact": all(
            row["source_audit"]["sha256"] == d0.SOURCE_SHA256 for row in workers.values()
        ),
        "Standard GPT-2; Full AttnRes absent": all(
            row["source_audit"]["checks"]["standard_mode"]
            and row["source_audit"]["full_attnres_active_modules"] == 0
            for row in workers.values()
        ),
        "canonical validation exact": all(
            row["canonical_validation_sha256"] == d0.CANONICAL_VALIDATION_SHA256
            for row in workers.values()
        ),
        "W1024/W512 regression sentinels": summary["integrity_pre_audit"]["sentinel"],
        "windows exactly 64/32/16/8/4/2": set(workers) == set(NEW_WINDOWS),
        "B1-B10 and B12 remain W1024": all(
            row["b1_to_b10_window"] == 1024 and row["b12_window"] == 1024
            for row in workers.values()
        ),
        "only B11 modified": all(row["b11_window"] in NEW_WINDOWS for row in workers.values()),
        "same precision and denominator": all(
            row["evaluation_precision"] == "torch.autocast(cuda,bfloat16)"
            and row["loss_denominator"] == 1_310_720
            for row in workers.values()
        ),
        "losses and activations finite": all(
            math.isfinite(row["validation_loss"])
            and row["all_losses_and_activations_finite"]
            for row in workers.values()
        ),
        "incoming h10 identical": all(row["incoming_b11_h10"]["passed"] for row in workers.values()),
        "model tensors unchanged": all(row["model_tensors_unchanged"] for row in workers.values()),
        "zero optimizer/backward/update/training": summary["integrity_pre_audit"]["zero_training"],
        "no recurrence or completion": all(
            not row["recurrence_active"] and not row["completion_module_active"]
            for row in workers.values()
        ),
        "independent processes; no DDP/NCCL": summary["integrity_pre_audit"]["independent_processes"],
        "all required artifacts present": all((output / name).is_file() for name in required),
    }
    passed = all(checks.values())
    d0.durable_json(
        output / "FINAL_AUDIT.json",
        {
            "experiment": EXPERIMENT,
            "checks": {
                key: {"status": "PASS" if value else "FAIL", "passed": value}
                for key, value in checks.items()
            },
            "passed": passed,
        },
    )

    rows = summary["combined_rows"]
    lines = [
        "# Experiment 2D0B — B11 Micro-Window Sensitivity Sweep",
        "",
        "## Outcome",
        "",
        "The six requested windows were evaluated with the exact frozen 2D0A path. This was evaluation-only: no optimizer, backward pass, update, recurrence, or completion module was used.",
        "",
        "## New micro-window points",
        "",
        "| Window | Validation loss | Damage vs W1024 | KV fraction | B11 cosine | B11 RMS | B12 cosine | B12 RMS | Logit argmax agreement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in NEW_WINDOWS:
        row = rows[str(window)]
        raw = workers[window]
        lines.append(
            f"| {window} | {fmt(row['validation_loss'])} | {row['damage_vs_1024']:+.10f} | "
            f"{row['kv_fraction']:.6f} | {fmt(row['b11_cosine'])} | {fmt(row['b11_rms'])} | "
            f"{fmt(row['b12_cosine'])} | {fmt(row['b12_rms'])} | "
            f"{fmt(raw['logit_drift']['argmax_agreement'])} |"
        )
    lines.extend(
        [
            "",
            "## Complete measured curve",
            "",
            "| Window | Validation loss | Damage vs W1024 | B11 KV fraction |",
            "|---:|---:|---:|---:|",
        ]
    )
    for window in COMBINED_WINDOWS:
        row = rows[str(window)]
        lines.append(
            f"| {window} | {fmt(row['validation_loss'])} | {row['damage_vs_1024']:+.10f} | {row['kv_fraction']:.6f} |"
        )
    lines.extend(["", "## Adjacent micro-window increments", ""])
    lines.append("| Transition | Additional damage | Damage at shorter window |")
    lines.append("|---|---:|---:|")
    for row in summary["micro_window_adjacent_increments"]:
        lines.append(
            f"| W{row['from_window']}→W{row['to_window']} | "
            f"{row['additional_damage']:+.10f} | {row['damage_at_to_window']:+.10f} |"
        )
    lines.extend(["", "## Quality thresholds", ""])
    lines.append("| Allowed damage | Smallest measured window | KV fraction |")
    lines.append("|---:|---:|---:|")
    for threshold, row in summary["quality_thresholds"].items():
        lines.append(
            f"| {float(threshold):.3f} | {row['smallest_window']} | {row['kv_fraction']:.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if summary["monotonic_nondecreasing"]:
        lines.append("The complete 1024→1 curve is monotonic at every measured point.")
    else:
        lines.append(
            "The curve has these exact monotonicity deviations: `"
            + json.dumps(summary["monotonicity_deviations"], sort_keys=True)
            + "`."
        )
    lines.append(
        "These results isolate B11's direct history while B1-B10 and B12 remain full-context. They do not establish the optimal B11 window when other layers are shortened jointly."
    )
    lines.append(
        "The B12 RMS comparison shows whether the final block absorbs or preserves the B11 state disturbance; the exact per-window values are in the table and machine-readable drift artifacts."
    )
    lines.extend(["", "## Integrity audit", ""])
    for name, value in checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — {name}")
    lines.extend(["", "## Performance", ""])
    lines.append("| Window/GPU | Wall s | Targets/s | Peak allocated MB | Peak reserved MB |")
    lines.append("|---|---:|---:|---:|---:|")
    for window in NEW_WINDOWS:
        row = performance["workers"][str(window)]
        lines.append(
            f"| W{window}/GPU{GPU_BY_WINDOW[window]} | {row['wall_seconds']:.3f} | "
            f"{row['targets_per_second']:.1f} | {row['peak_allocated_vram_mb']:.1f} | "
            f"{row['peak_reserved_vram_mb']:.1f} |"
        )
    for row in performance["waves"]:
        lines.append(
            f"\nWave {row['wave']} {row['windows']} elapsed wall time: {row['elapsed_wall_seconds']:.3f} seconds."
        )
    lines.extend(["", "# EXPERIMENT 2D0B COMPLETE"])
    d0.durable_text(output / "EXPERIMENT_2D0B_FINAL_REPORT.md", "\n".join(lines) + "\n")
    print(f"EXPERIMENT_2D0B_FINAL_AUDIT_{'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        raise SystemExit("2D0B final audit failed")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sentinel = subparsers.add_parser("sentinel")
    sentinel.add_argument("--parent-checkpoint", required=True)
    sentinel.add_argument("--validation-shard", required=True)
    sentinel.add_argument("--run-root", required=True)
    sentinel.set_defaults(func=run_sentinel)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--parent-checkpoint", required=True)
    worker.add_argument("--validation-shard", required=True)
    worker.add_argument("--run-root", required=True)
    worker.add_argument("--window", type=int, required=True)
    worker.add_argument("--physical-gpu", type=int, required=True)
    worker.set_defaults(func=run_worker)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--run-root", required=True)
    assemble.add_argument("--output-dir", required=True)
    assemble.set_defaults(func=run_assemble)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-dir", required=True)
    finalize.set_defaults(func=run_finalize)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
