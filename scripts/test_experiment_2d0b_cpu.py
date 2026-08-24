#!/usr/bin/env python3
"""CPU contracts for Experiment 2D0B."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0b as d0b  # noqa: E402


def test_reused_evaluator_is_bound_to_2d0b():
    assert d0b.d0a.EXPERIMENT == "2D0B"
    assert d0b.d0a.PROTOCOL == d0b.PROTOCOL
    assert d0b.d0a.BRANCH == d0b.BRANCH
    assert d0b.d0a.FROZEN_COMMIT == d0b.FROZEN_COMMIT
    assert d0b.d0a.NEW_WINDOWS == d0b.NEW_WINDOWS
    assert d0b.d0a.GPU_BY_WINDOW == d0b.GPU_BY_WINDOW


def test_frozen_configuration_and_waves():
    config = d0b.load_config()
    assert tuple(config["evaluation"]["windows"]) == (64, 32, 16, 8, 4, 2)
    assert tuple(tuple(row) for row in config["evaluation"]["waves"]) == (
        (64, 32, 16, 8),
        (4, 2),
    )
    assert config["evaluation"]["loss_denominator"] == 1_310_720
    assert config["training"]["optimizer_objects"] == 0
    assert config["training"]["training_targets"] == 0


def test_micro_window_masks_are_exact():
    for window in d0b.NEW_WINDOWS:
        mask = d0b.d0.sliding_mask(128, window, torch.device("cpu"))
        for query in range(128):
            expected = torch.zeros(128, dtype=torch.bool)
            expected[max(0, query - window + 1) : query + 1] = True
            assert torch.equal(mask[query], expected)


def test_combined_curve_order_and_threshold_helper():
    assert d0b.COMBINED_WINDOWS == (
        1024,
        896,
        768,
        512,
        384,
        256,
        128,
        64,
        32,
        16,
        8,
        4,
        2,
        1,
    )
    damages = {window: 1 / window for window in d0b.COMBINED_WINDOWS}
    assert d0b.smallest_window(damages, 0.01) == 128
    assert d0b.smallest_window(damages, 0.1) == 16


if __name__ == "__main__":
    test_reused_evaluator_is_bound_to_2d0b()
    test_frozen_configuration_and_waves()
    test_micro_window_masks_are_exact()
    test_combined_curve_order_and_threshold_helper()
    print("EXPERIMENT_2D0B_CPU_CONTRACTS_PASS")
