#!/usr/bin/env python3
"""CPU contracts for Experiment 2D0A."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0a as d0a  # noqa: E402


def test_frozen_configuration():
    config = d0a.load_config()
    assert tuple(config["evaluation"]["windows"]) == d0a.NEW_WINDOWS
    assert config["evaluation"]["loss_denominator"] == 1_310_720
    assert config["training"] == {
        "optimizer_objects": 0,
        "backward_calls": 0,
        "parameter_updates": 0,
        "training_targets": 0,
        "recurrence_active": False,
        "completion_module_active": False,
    }


def test_window_one_is_self_only():
    mask = d0a.d0.sliding_mask(8, 1, torch.device("cpu"))
    assert torch.equal(mask, torch.eye(8, dtype=torch.bool))
    assert mask.tril(-1).count_nonzero().item() == 0


def test_position_bins_cover_exact_requested_positions():
    positions = []
    for _, start, end in d0a.POSITION_BINS:
        positions.extend(range(start, end + 1))
    assert positions == list(range(1, 1024))


def test_pareto_selects_smallest_numeric_window():
    damages = {
        1024: 0.0,
        896: 0.0002,
        768: 0.0008,
        512: 0.002,
        384: 0.004,
        256: 0.008,
        128: 0.02,
        1: 0.04,
    }
    rows = d0a.smallest_windows(damages)
    assert rows[0]["smallest_b11_window"] == 768
    assert next(row for row in rows if row["allowed_damage"] == 0.005)[
        "smallest_b11_window"
    ] == 384
    assert next(row for row in rows if row["allowed_damage"] == 0.01)[
        "smallest_b11_window"
    ] == 256
    assert rows[-1]["smallest_b11_window"] == 1


def test_classification_rules():
    base = {window: 0.0 for window in d0a.ALL_WINDOWS}
    assert d0a.classify({**base, 1: 0.009}, []) == "B11 EXPLICIT HISTORY HIGHLY REDUNDANT"
    assert (
        d0a.classify({**base, 1: 0.02, 128: 0.009}, [])
        == "B11 EXPLICIT HISTORY MODERATELY REDUNDANT"
    )
    assert (
        d0a.classify({**base, 1: 0.05, 128: 0.02, 256: 0.02}, [])
        == "B11 REQUIRES SUBSTANTIAL EXPLICIT HISTORY"
    )
    deviations = [{"decrease": 0.001}]
    assert d0a.classify(base, deviations) == "B11 WINDOW RESULT IS MIXED"


def test_drift_identity():
    value = torch.randn(2, 5, 7)
    accumulator = d0a.blank_drift()
    d0a.add_drift(accumulator, value, value, 1, 4)
    row = d0a.finish_drift(accumulator)
    assert row["count"] == 8
    assert row["rms_difference"] == 0.0
    assert abs(row["cosine"] - 1.0) < 1e-6
    assert abs(row["norm_ratio"] - 1.0) < 1e-6


if __name__ == "__main__":
    test_frozen_configuration()
    test_window_one_is_self_only()
    test_position_bins_cover_exact_requested_positions()
    test_pareto_selects_smallest_numeric_window()
    test_classification_rules()
    test_drift_identity()
    print("EXPERIMENT_2D0A_CPU_CONTRACTS_PASS")
