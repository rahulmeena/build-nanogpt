#!/usr/bin/env python3
"""CPU contract tests for Experiment 2C0 separated branch semantics."""

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2c0 as candidate  # noqa: E402


def small_runtime():
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
    reader = symbols["TopDownAttnRes"](16, candidate.SOURCE_DEPTHS, eps=1e-5).eval()
    return symbols, model, reader


def test_config_contract():
    config = candidate.load_config()
    assert config["trainable_parameters"] == 1537
    assert config["world_size"] * config["microsteps_per_rank"] * config["batch_sequences"] * config["sequence_length"] == 524_288
    assert config["writers"] == "forbidden in active recurrence"
    assert config["hellaswag"] == "forbidden"


def test_direct_feedback_bypasses_old_reader():
    _, model, _ = small_runtime()
    tokens = torch.randint(0, 32, (2, 1))
    zero_state = model.init_recurrent_state(2, "masked_l1_no_feedback", dtype=torch.float32)
    one_state = model.init_recurrent_state(2, "masked_l1_no_feedback", dtype=torch.float32)
    with torch.no_grad():
        logits_zero, next_zero = model.forward_step(
            tokens[:, 0], zero_state, block1_feedback=torch.zeros(2, 1, 16)
        )
        logits_one, next_one = model.forward_step(
            tokens[:, 0], one_state, block1_feedback=torch.ones(2, 1, 16)
        )
    assert not torch.equal(logits_zero, logits_one)
    assert next_zero.kv_caches[0] is None and next_one.kv_caches[0] is None
    assert all(cache.length == 1 for cache in next_zero.kv_caches[1:])
    assert all(writer.W_up.weight.grad is None for writer in model.transformer.memory_writers.values())


def test_direct_feedback_contract_rejects_legacy_mode_and_shape():
    _, model, _ = small_runtime()
    token = torch.randint(0, 32, (2,))
    legacy = model.init_recurrent_state(2, "masked_l1_topdown_self", dtype=torch.float32)
    try:
        model.forward_step(token, legacy, block1_feedback=torch.zeros(2, 1, 16))
    except ValueError as error:
        assert "requires masked_l1_no_feedback" in str(error)
    else:
        raise AssertionError("direct feedback accepted a legacy reader mode")
    direct = model.init_recurrent_state(2, "masked_l1_no_feedback", dtype=torch.float32)
    try:
        model.forward_step(token, direct, block1_feedback=torch.zeros(2, 16))
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("direct feedback accepted an invalid shape")


def test_centering_shuffle_batchmean_and_zero_contract():
    symbols, _, reader = small_runtime()
    memory = torch.arange(4 * 4 * 16, dtype=torch.float32).reshape(4, 4, 1, 16)
    means = torch.zeros(4, 16)
    state = symbols["RecurrentState"](
        position=1,
        mode="masked_l1_no_feedback",
        kv_caches=(),
        feedback_memory=memory,
        mask_depth=1,
    )
    generic = torch.zeros(16)
    permutation = torch.arange(4).roll(1)
    _, shuffled = candidate.direct_feedback(
        reader, state, means, generic, "shuffle", permutation=permutation
    )
    assert torch.equal(shuffled["centered_sources"], memory[:, permutation])
    _, batchmean = candidate.direct_feedback(
        reader, state, means, generic, "batchmean"
    )
    expected = (memory.sum(dim=1, keepdim=True) - memory) / 3.0
    assert torch.equal(batchmean["centered_sources"], expected)
    zero_state = symbols["RecurrentState"](
        position=1,
        mode="masked_l1_no_feedback",
        kv_caches=(),
        feedback_memory=torch.zeros_like(memory),
        mask_depth=1,
    )
    _, zero = candidate.direct_feedback(
        reader, zero_state, means, generic, "real"
    )
    assert zero["sequence_topdown"].count_nonzero().item() == 0
    assert zero["sequence_feedback"].count_nonzero().item() == 0


def test_classification_rules():
    assert candidate.classification_from_metrics(.011, .021, 18, .011, True) == "SEPARATED SEQUENCE BRANCH LEARNS SEQUENCE MEMORY"
    assert candidate.classification_from_metrics(.011, .009, 20, .011, True) == "SEPARATED SEQUENCE BRANCH IMPROVES GENERIC COMPENSATION ONLY"
    assert candidate.classification_from_metrics(.005, .030, 20, .020, True) == "SEPARATED SEQUENCE BRANCH IS NEUTRAL"
    assert candidate.classification_from_metrics(-.011, .030, 20, .020, True) == "SEPARATED SEQUENCE BRANCH DEGRADES"
    assert candidate.classification_from_metrics(.020, .030, 20, .020, False) == "SEPARATED ARCHITECTURE UNSTABLE"


def main():
    test_config_contract()
    test_direct_feedback_bypasses_old_reader()
    test_direct_feedback_contract_rejects_legacy_mode_and_shape()
    test_centering_shuffle_batchmean_and_zero_contract()
    test_classification_rules()
    json.loads(candidate.CONFIG_PATH.read_text())
    print("Experiment 2C0 CPU contract tests: PASS")


if __name__ == "__main__":
    main()
