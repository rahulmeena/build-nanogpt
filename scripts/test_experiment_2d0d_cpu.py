#!/usr/bin/env python3
"""CPU contracts for Experiment 2D0D."""

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0d as d0d  # noqa: E402


def tiny_model():
    symbols = d0d.d0.support.load_training_symbols()
    config = symbols["GPTConfig"](
        block_size=16,
        vocab_size=64,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="standard",
    )
    torch.manual_seed(29)
    return symbols["GPT"](config).eval()


def test_geometry_manifest_and_budget():
    config = d0d.load_config()
    assert config["geometries"] == {key: list(value) for key, value in d0d.GEOMETRIES.items()}
    assert config["gpu_assignment"] == {str(key): value for key, value in d0d.GPU_ASSIGNMENT.items()}
    assert len(config["geometries"]) == 4
    for schedule in config["geometries"].values():
        assert len(schedule) == 12
        assert all(isinstance(value, int) and 1 <= value <= 1024 for value in schedule)
        assert sum(schedule) == 5312
        assert sum(value - 1 for value in schedule) == 5300


def test_full_schedule_is_native_forward_exact():
    model = tiny_model()
    tokens = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        native = d0d.d0c.forward_top(model, tokens)
        scheduled = d0d.forward_top_schedule(model, tokens, [16] * 12)
    assert torch.equal(native, scheduled)


def test_single_layer_schedule_matches_2d0c():
    model = tiny_model()
    tokens = torch.randint(0, 64, (2, 16))
    for block_index, window in ((0, 8), (10, 4), (11, 1)):
        schedule = [16] * 12
        schedule[block_index] = window
        with torch.no_grad():
            expected = d0d.d0c.forward_top(model, tokens, block_index, window)
            observed = d0d.forward_top_schedule(model, tokens, schedule)
        assert torch.equal(expected, observed)


def test_sliding_window_semantics():
    for window in {value for schedule in d0d.GEOMETRIES.values() for value in schedule}:
        mask = d0d.d0.sliding_mask(1024, window, torch.device("cpu"))
        for query in (0, 1, 127, 511, 1023):
            expected = torch.zeros(1024, dtype=torch.bool)
            expected[max(0, query - window + 1) : query + 1] = True
            assert torch.equal(mask[query], expected)


def test_position_bins_and_pairing():
    positions = []
    for _, start, end in d0d.POSITION_BINS:
        positions.extend(range(start, end + 1))
    assert positions == list(range(1, 1024))
    row = d0d.paired_stats("A", "B", [1.0, 2.0, 4.0], [2.0, 2.0, 3.0])
    assert row["a_wins"] == 1
    assert row["b_wins"] == 1
    assert row["ties"] == 1
    assert abs(row["mean_a_minus_b"]) < 1e-12


def test_empirical_terms_are_exact_cells():
    matrix = json.loads(d0d.MARGINAL_MATRIX_PATH.read_text())
    terms = d0d.empirical_marginal_terms(matrix, d0d.GEOMETRIES["EMPIRICAL"])
    assert len(terms) == 12
    assert terms[0]["damage"] == 0.0
    assert terms[2]["window"] == 128
    assert terms[2]["damage"] == matrix["B3"]["128"]


if __name__ == "__main__":
    test_geometry_manifest_and_budget()
    test_full_schedule_is_native_forward_exact()
    test_single_layer_schedule_matches_2d0c()
    test_sliding_window_semantics()
    test_position_bins_and_pairing()
    test_empirical_terms_are_exact_cells()
    print("EXPERIMENT_2D0D_CPU_CONTRACTS_PASS")
