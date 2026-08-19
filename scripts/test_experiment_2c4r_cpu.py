#!/usr/bin/env python3
"""CPU contracts for Experiment 2C4R path-consistent rolling-KV windows."""

import copy
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from smoke_test import load_training_symbols  # noqa: E402


def make_model(block_size=16):
    torch.manual_seed(20260202)
    symbols = load_training_symbols()
    network = symbols["GPT"](
        symbols["GPTConfig"](
            block_size=block_size,
            vocab_size=64,
            n_layer=12,
            n_head=2,
            n_embd=8,
            residual_mode="full_attnres",
            enable_topdown_feedback=True,
            topdown_feedback_destinations=(0, 1, 2, 3),
        )
    ).eval()
    return symbols, network


def nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(nested_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def recurrent_logits(network, tokens, windows, diagnostics=False):
    state = network.init_recurrent_state(
        tokens.size(0),
        "masked_l1_no_feedback",
        device="cpu",
        dtype=torch.float32,
        mask_depth=4,
        attention_windows=windows,
    )
    rows = []
    last_diagnostics = None
    with torch.no_grad():
        for position in range(tokens.size(1)):
            result = network.forward_step(
                tokens[:, position], state, return_diagnostics=diagnostics
            )
            if diagnostics:
                logits, state, last_diagnostics = result
            else:
                logits, state = result
            rows.append(logits)
    return torch.cat(rows, dim=1), state, last_diagnostics


def test_physical_capacities_eviction_and_absolute_positions():
    _, network = make_model()
    windows = (1, 2, 4, 8) + (16,) * 8
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    state = network.init_recurrent_state(
        1,
        "masked_l1_no_feedback",
        device="cpu",
        dtype=torch.float32,
        mask_depth=4,
        attention_windows=windows,
    )
    assert state.kv_caches[0] is None
    assert [cache.key.size(2) for cache in state.kv_caches[1:4]] == [1, 3, 7]
    position_trace = []
    with torch.no_grad():
        for position in range(tokens.size(1)):
            _, state = network.forward_step(tokens[:, position], state)
            position_trace.append(state.position)
            for block, window in enumerate(windows):
                cache = state.kv_caches[block]
                if window == 1:
                    assert cache is None
                else:
                    assert cache.key.size(2) == window - 1
                    assert cache.value.size(2) == window - 1
                    assert cache.length == min(position + 1, window - 1)
    assert position_trace == list(range(1, 10))
    assert state.position == 9
    assert [state.kv_caches[index].length for index in (1, 2, 3)] == [1, 3, 7]


def test_serialization_resume_and_fresh_reset():
    symbols, network = make_model()
    windows = (1, 2, 4, 8) + (16,) * 8
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    state = network.init_recurrent_state(
        2,
        "masked_l1_no_feedback",
        device="cpu",
        dtype=torch.float32,
        mask_depth=4,
        attention_windows=windows,
    )
    with torch.no_grad():
        for position in range(5):
            _, state = network.forward_step(tokens[:, position], state)
        payload = state.state_dict()
        restored = network.load_recurrent_state(payload, device="cpu", dtype=torch.float32)
        logits_a, next_a = network.forward_step(tokens[:, 5], state)
        logits_b, next_b = network.forward_step(tokens[:, 5], restored)
    assert payload["schema"] == "full_attnres_recurrent_state_v4_rolling_windows"
    assert payload["attention_windows"] == list(windows)
    assert torch.equal(logits_a, logits_b)
    assert nested_equal(next_a.state_dict(), next_b.state_dict())
    fresh = network.init_recurrent_state(
        2,
        "masked_l1_no_feedback",
        device="cpu",
        dtype=torch.float32,
        mask_depth=4,
        attention_windows=windows,
    )
    assert fresh.position == 0
    assert fresh.feedback_memory.count_nonzero().item() == 0
    assert all(cache is None or cache.length == 0 for cache in fresh.kv_caches)


def test_full_window_equivalence_and_diagnostics():
    _, network = make_model()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    old = network.init_recurrent_state(
        2, "full_context", device="cpu", dtype=torch.float32, mask_depth=0
    )
    rolling = network.init_recurrent_state(
        2,
        "full_context",
        device="cpu",
        dtype=torch.float32,
        mask_depth=0,
        attention_windows=(16,) * 12,
    )
    old_rows = []
    rolling_rows = []
    with torch.no_grad():
        for position in range(tokens.size(1)):
            old_logits, old = network.forward_step(tokens[:, position], old)
            rolling_logits, rolling, diagnostics = network.forward_step(
                tokens[:, position], rolling, return_diagnostics=True
            )
            old_rows.append(old_logits)
            rolling_rows.append(rolling_logits)
    assert torch.equal(torch.cat(old_rows, 1), torch.cat(rolling_rows, 1))
    assert diagnostics["source_memory"].shape == (4, 2, 1, 8)
    assert diagnostics["receiver_states"].shape == (4, 2, 1, 8)
    full = network.capture_full_context_diagnostics(tokens)
    assert full["sources"].shape == (4, 2, 6, 8)
    assert full["receivers"].shape == (4, 2, 6, 8)


def test_future_causality_row_isolation_and_parameter_immutability():
    _, network = make_model()
    before = copy.deepcopy(network.state_dict())
    windows = (1, 4, 8, 16) + (16,) * 8
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    future = tokens.clone()
    future[:, 4:] = (future[:, 4:] + 13) % network.config.vocab_size
    row = tokens.clone()
    row[1] = (row[1] + 17) % network.config.vocab_size
    reference, state, _ = recurrent_logits(network, tokens, windows)
    future_logits, _, _ = recurrent_logits(network, future, windows)
    row_logits, _, _ = recurrent_logits(network, row, windows)
    assert torch.equal(reference[:, :4], future_logits[:, :4])
    assert torch.equal(reference[0], row_logits[0])
    assert torch.isfinite(reference).all()
    assert torch.isfinite(state.feedback_memory).all()
    assert set(before) == set(network.state_dict())
    assert all(torch.equal(before[name], value) for name, value in network.state_dict().items())


if __name__ == "__main__":
    test_physical_capacities_eviction_and_absolute_positions()
    test_serialization_resume_and_fresh_reset()
    test_full_window_equivalence_and_diagnostics()
    test_future_causality_row_isolation_and_parameter_immutability()
    print("EXPERIMENT_2C4R_CPU_CONTRACTS_PASS")
