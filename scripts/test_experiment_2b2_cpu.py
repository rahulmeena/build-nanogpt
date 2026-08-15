#!/usr/bin/env python3
"""CPU/FP32 hard-prerequisite diagnostics for Experiment 2B2."""

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_2b2 as experiment  # noqa: E402


def main():
    symbols = experiment.a0.support.load_training_symbols()
    report = {
        "mathematical_regression": experiment.cpu_fp32_writer_regression(symbols),
        "one_step_temporal_gradient": experiment.cpu_one_step_temporal_gradient_test(
            symbols
        ),
    }
    report["passed"] = all(row["passed"] for row in report.values())
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("Experiment 2B2 CPU/FP32 prerequisite failed")


if __name__ == "__main__":
    main()
