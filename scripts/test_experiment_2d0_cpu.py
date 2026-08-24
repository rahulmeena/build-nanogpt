#!/usr/bin/env python3
"""CPU contracts for Experiment 2D0 Standard-GPT B11 completion."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2d0 as d0  # noqa: E402


def test_completion_initialization_mask_and_gradient_staging():
    torch.manual_seed(7)
    completion = d0.B11ContextCompletion(n_embd=8)
    source = torch.randn(2, 12, 8)
    destination = torch.randn(2, 12, 8)
    feedback = completion(source, destination, window=8)
    assert torch.count_nonzero(feedback).item() == 0
    forced = completion(source, destination, window=8, gate_override=1.0)
    assert torch.count_nonzero(forced[:, :8]).item() == 0
    assert torch.count_nonzero(forced[:, 8:]).item() > 0
    loss = (destination + feedback).square().mean()
    loss.backward()
    assert completion.g.grad is not None
    assert torch.count_nonzero(completion.g.grad).item() > 0
    assert completion.W_u.weight.grad is not None
    assert torch.count_nonzero(completion.W_u.weight.grad).item() == 0
    assert completion.W_g.weight.grad is not None
    assert torch.count_nonzero(completion.W_g.weight.grad).item() == 0


def test_shift_is_one_token_without_wraparound():
    value = torch.arange(2 * 5 * 3).view(2, 5, 3).float()
    shifted = d0.shifted_top_state(value)
    assert torch.count_nonzero(shifted[:, 0]).item() == 0
    assert torch.equal(shifted[:, 1:], value[:, :-1])
    assert not shifted.requires_grad


def test_sliding_mask_includes_current_and_w_minus_one_history():
    mask = d0.sliding_mask(7, 3, torch.device("cpu"))
    expected = torch.zeros(7, 7, dtype=torch.bool)
    for query in range(7):
        expected[query, max(0, query - 2) : query + 1] = True
    assert torch.equal(mask, expected)


def test_full_window_block_matches_standard_block():
    symbols = d0.support.load_training_symbols()
    config = symbols["GPTConfig"](
        block_size=16,
        vocab_size=64,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="standard",
    )
    torch.manual_seed(11)
    block = symbols["Block"](config).eval()
    value = torch.randn(2, 16, 8)
    with torch.no_grad():
        expected = block(value)
        actual, _ = d0.run_block(block, value, window=16)
    assert torch.equal(actual, expected)


def test_window_selection_is_exactly_preregistered():
    def rows(d896, d768, d512):
        full = 3.0
        return {
            1024: {"validation_loss": full},
            896: {"validation_loss": full + d896},
            768: {"validation_loss": full + d768},
            512: {"validation_loss": full + d512},
        }

    assert d0.select_window(rows(0.005, 0.02, 0.08))["selected_window"] == 768
    assert d0.select_window(rows(0.005, 0.005, 0.03))["selected_window"] == 512
    assert d0.select_window(rows(0.03, 0.20, 0.30))["selected_window"] == 896
    assert d0.select_window(rows(0.005, 0.005, 0.006))["selected_window"] is None


def test_selected_window_position_bins_partition_context():
    bins = d0.dynamic_position_bins(768)
    assert bins == (
        ("before_truncation", 0, 767),
        ("early_missing", 768, 831),
        ("middle_1_missing", 832, 895),
        ("middle_2_missing", 896, 959),
        ("late_missing", 960, 1023),
    )


if __name__ == "__main__":
    test_completion_initialization_mask_and_gradient_staging()
    test_shift_is_one_token_without_wraparound()
    test_sliding_mask_includes_current_and_w_minus_one_history()
    test_full_window_block_matches_standard_block()
    test_window_selection_is_exactly_preregistered()
    test_selected_window_position_bins_partition_context()
    print("EXPERIMENT_2D0_CPU_CONTRACTS_PASS")
