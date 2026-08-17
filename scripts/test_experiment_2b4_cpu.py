#!/usr/bin/env python3
"""CPU contract tests for Experiment 2B4 diagnostic semantics."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2b4 as candidate  # noqa: E402


def test_config():
    config = candidate.load_config()
    assert config["training"] == "forbidden"
    assert config["hellaswag"] == "forbidden"
    assert config["mask_depths"] == [1, 2, 3, 4]
    assert config["part_a_controls"] == list(candidate.PART_A_CONTROLS)


def test_source_permutations():
    permutations = candidate.independent_source_permutations(64, torch.device("cpu"))
    expected = torch.arange(64)
    assert len(permutations) == 4
    assert len({tuple(value.tolist()) for value in permutations}) == 4
    assert all(torch.equal(torch.sort(value).values, expected) for value in permutations)
    assert all(not torch.any(value == expected) for value in permutations)


def fake_state(memory):
    symbols = candidate.a0.support.load_training_symbols()
    return symbols["RecurrentState"](
        position=1,
        mode="masked_l1_topdown_self",
        kv_caches=(),
        feedback_memory=memory,
        mask_depth=1,
    )


def test_leave_one_out_mean():
    memory = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 1, 3)
    supplied = candidate.supplied_memory(
        "batch_mean", fake_state(memory), 1, 0
    )
    for row in range(5):
        expected = torch.cat((memory[:, :row], memory[:, row + 1 :]), dim=1).mean(1)
        assert torch.equal(supplied[:, row], expected)


def test_norm_random_is_deterministic_and_rms_matched():
    memory = torch.linspace(-2, 2, 4 * 5 * 16).reshape(4, 5, 1, 16)
    first = candidate.norm_random_memory(memory, batch_index=3, position=7)
    second = candidate.norm_random_memory(memory, batch_index=3, position=7)
    assert torch.equal(first, second)
    expected = memory.float().pow(2).mean(-1).sqrt()
    actual = first.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_generalized_mask_state():
    symbols = candidate.a0.support.load_training_symbols()
    config = symbols["GPTConfig"](
        block_size=8,
        vocab_size=32,
        n_layer=12,
        n_head=2,
        n_embd=16,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
        enable_memory_writers=True,
        memory_writer_rank=8,
        memory_writer_init_seed=7,
    )
    model = symbols["GPT"](config).eval()
    state = model.init_recurrent_state(
        2, "masked_l1_topdown_self", mask_depth=4
    )
    assert state.mask_depth == 4
    assert all(cache is None for cache in state.kv_caches[:4])
    assert all(cache.length == 0 for cache in state.kv_caches[4:])
    tokens = torch.randint(0, 32, (2, 3))
    with torch.no_grad():
        for position in range(3):
            _, state = model.forward_step(
                tokens[:, position], state, use_memory_writers=True
            )
    payload = state.state_dict()
    assert payload["schema"] == "full_attnres_recurrent_state_v2"
    assert payload["mask_depth"] == 4
    restored = model.load_recurrent_state(payload)
    assert restored.mask_depth == 4
    assert all(cache is None for cache in restored.kv_caches[:4])
    assert all(cache.length == 3 for cache in restored.kv_caches[4:])


def test_state_override_preserves_cache_identity():
    symbols = candidate.a0.support.load_training_symbols()
    state = symbols["RecurrentState"](
        3,
        "masked_l1_topdown_self",
        (None,),
        torch.zeros(4, 5, 1, 16),
        1,
    )
    overridden = replace(state, feedback_memory=torch.ones_like(state.feedback_memory))
    assert overridden.position == state.position
    assert overridden.kv_caches is state.kv_caches
    assert torch.count_nonzero(overridden.feedback_memory).item() == 4 * 5 * 16


def main():
    test_config()
    test_source_permutations()
    test_leave_one_out_mean()
    test_norm_random_is_deterministic_and_rms_matched()
    test_generalized_mask_state()
    test_state_override_preserves_cache_identity()
    json.loads(candidate.CONFIG_PATH.read_text())
    print("Experiment 2B4 CPU contract tests: PASS")


if __name__ == "__main__":
    main()
