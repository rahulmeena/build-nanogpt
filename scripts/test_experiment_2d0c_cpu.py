#!/usr/bin/env python3
"""CPU contracts for Experiment 2D0C."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0c as d0c  # noqa: E402


def tiny_model():
    symbols = d0c.d0.support.load_training_symbols()
    config = symbols["GPTConfig"](
        block_size=16,
        vocab_size=64,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="standard",
    )
    torch.manual_seed(17)
    return symbols["GPT"](config).eval()


def test_frozen_grid_and_assignment():
    config = d0c.load_config()
    assert tuple(config["evaluation"]["windows"]) == d0c.WINDOWS
    assert tuple(config["evaluation"]["layers"]) == d0c.LAYERS
    assigned = [layer for gpu in range(4) for layer in d0c.GPU_LAYERS[gpu]]
    assert sorted(assigned) == list(d0c.LAYERS)
    assert len(assigned) * len(d0c.WINDOWS) == 120


def test_human_to_zero_based_mapping():
    for human_layer in d0c.LAYERS:
        assert human_layer - 1 == list(d0c.LAYERS).index(human_layer)
    assert 11 - 1 == d0c.d0.B11_INDEX
    assert 12 - 1 == d0c.d0.B12_INDEX


def test_configurable_full_window_is_semantically_exact():
    model = tiny_model()
    tokens = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        baseline = d0c.forward_top(model, tokens)
        for block_index in (0, 5, 10, 11):
            selected = d0c.forward_top(model, tokens, block_index, window=16)
            assert torch.equal(selected, baseline)


def test_exactly_one_shortened_layer_geometry():
    for layer in d0c.LAYERS:
        for window in d0c.WINDOWS:
            values = [1024] * 12
            values[layer - 1] = window
            assert sum(value != 1024 for value in values) == 1
            assert values[layer - 1] == window


def test_causal_window_semantics():
    for window in d0c.WINDOWS:
        mask = d0c.d0.sliding_mask(32, window, torch.device("cpu"))
        effective = min(window, 32)
        for query in range(32):
            expected = torch.zeros(32, dtype=torch.bool)
            expected[max(0, query - effective + 1) : query + 1] = True
            assert torch.equal(mask[query], expected)


def test_position_bins_cover_requested_targets():
    positions = []
    for _, start, end in d0c.POSITION_BINS:
        positions.extend(range(start, end + 1))
    assert positions == list(range(1, 1024))


def test_smallest_window_and_shape_accounting():
    damages = {window: 1 / window for window in d0c.MATRIX_WINDOWS}
    assert d0c.smallest_window(damages, 0.01) == 128
    widths = {layer: 2 ** layer for layer in d0c.LAYERS}
    row = d0c.shape_row(widths)
    assert row["adjacent_increases"] == 11
    assert row["adjacent_decreases"] == 0
    assert row["adjacent_ties"] == 0
    assert abs(row["spearman_depth_vs_selected_window"] - 1.0) < 1e-12


if __name__ == "__main__":
    test_frozen_grid_and_assignment()
    test_human_to_zero_based_mapping()
    test_configurable_full_window_is_semantically_exact()
    test_exactly_one_shortened_layer_geometry()
    test_causal_window_semantics()
    test_position_bins_cover_requested_targets()
    test_smallest_window_and_shape_accounting()
    print("EXPERIMENT_2D0C_CPU_CONTRACTS_PASS")
