#!/usr/bin/env python3
"""CPU contract tests for Experiment 2C2 cumulative matched feedback."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2c2 as candidate  # noqa: E402


def make_model(depth=4):
    symbols = candidate.a0.support.load_training_symbols()
    config = symbols["GPTConfig"](
        block_size=16,
        vocab_size=64,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
        topdown_feedback_destinations=tuple(range(depth)),
    )
    model = symbols["GPT"](config)
    model.freeze_for_topdown_training()
    return symbols, model


def test_initialization_and_counts():
    for depth in range(1, 5):
        _, model = make_model(depth)
        assert set(model.transformer.topdown_attnres_by_destination) == {
            str(block) for block in range(depth)
        }
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == depth * 17
        for reader in model.transformer.topdown_attnres_by_destination.values():
            assert reader.query.count_nonzero().item() == 0
            assert torch.equal(reader.norm.weight, torch.ones_like(reader.norm.weight))
            assert reader.gate.item() == 0


def test_cumulative_mask_and_reader_mapping():
    symbols, model = make_model(4)
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    memory = torch.randn(4, 2, 8, model.config.n_embd)
    masked, _ = model(x, mode="masked_cumulative_no_feedback")
    zero, _ = model(
        x,
        mode="masked_cumulative_topdown_teacher",
        feedback_sources=memory,
        feedback_gate_override=0.0,
    )
    assert torch.equal(masked, zero)
    for reader in model.transformer.topdown_attnres_by_destination.values():
        reader.gate.data.fill_(0.25)
    for active in range(4):
        calls = {block: 0 for block in range(4)}
        handles = []
        for block in range(4):
            def hook(_module, _inputs, block=block):
                calls[block] += 1
            handles.append(
                model.transformer.topdown_attnres_by_destination[str(block)]
                .register_forward_pre_hook(hook)
            )
        logits, _ = model(
            x,
            mode="masked_cumulative_topdown_teacher",
            feedback_sources=memory,
            feedback_active_destination_blocks=(active,),
        )
        for handle in handles:
            handle.remove()
        assert calls == {block: int(block == active) for block in range(4)}
        assert not torch.equal(logits, masked)
    permutation = symbols["fixed_derangement"](2)
    shuffled, _ = model(
        x,
        mode="masked_cumulative_shuffled_feedback",
        feedback_sources=memory,
        feedback_permutation=permutation,
    )
    explicit, _ = model(
        x,
        mode="masked_cumulative_topdown_teacher",
        feedback_sources=memory[:, permutation],
    )
    assert torch.equal(shuffled, explicit)


def test_incremental_caches_direct_feedback_and_resume():
    _, model = make_model(4)
    state = model.init_recurrent_state(
        2,
        "masked_l1_no_feedback",
        device="cpu",
        dtype=torch.float32,
        mask_depth=4,
    )
    assert all(state.kv_caches[block] is None for block in range(4))
    assert all(state.kv_caches[block] is not None for block in range(4, 12))
    token = torch.randint(0, model.config.vocab_size, (2,))
    memory = torch.randn(4, 2, 1, model.config.n_embd)
    feedback = {}
    for block in range(4):
        reader = model.transformer.topdown_attnres_by_destination[str(block)]
        reader.gate.data.fill_(0.25)
        feedback[block] = reader.gate.tanh() * reader(list(memory.unbind(0)))
    logits, next_state = model.forward_step(
        token, state, attention_feedback_by_block=feedback
    )
    assert torch.isfinite(logits).all()
    assert all(next_state.kv_caches[block] is None for block in range(4))
    assert all(next_state.kv_caches[block].length == 1 for block in range(4, 12))
    payload = next_state.state_dict()
    assert payload["schema"] == "full_attnres_recurrent_state_v2"
    restored = model.load_recurrent_state(payload, device="cpu", dtype=torch.float32)
    next_token = torch.randint(0, model.config.vocab_size, (2,))
    bank_a = next_state.feedback_memory.detach()
    bank_b = restored.feedback_memory.detach()
    direct_a = {
        block: model.transformer.topdown_attnres_by_destination[str(block)].gate.tanh()
        * model.transformer.topdown_attnres_by_destination[str(block)](list(bank_a.unbind(0)))
        for block in range(4)
    }
    direct_b = {
        block: model.transformer.topdown_attnres_by_destination[str(block)].gate.tanh()
        * model.transformer.topdown_attnres_by_destination[str(block)](list(bank_b.unbind(0)))
        for block in range(4)
    }
    logits_a, state_a = model.forward_step(
        next_token, next_state, attention_feedback_by_block=direct_a
    )
    logits_b, state_b = model.forward_step(
        next_token, restored, attention_feedback_by_block=direct_b
    )
    assert torch.equal(logits_a, logits_b)
    assert candidate.a0.nested_equal(state_a.state_dict(), state_b.state_dict())


def test_reject_invalid_configurations():
    symbols = candidate.a0.support.load_training_symbols()
    try:
        symbols["GPT"](
            symbols["GPTConfig"](
                block_size=8,
                vocab_size=32,
                n_layer=4,
                n_head=2,
                n_embd=8,
                residual_mode="full_attnres",
                enable_topdown_feedback=True,
                topdown_feedback_destinations=(0, 2),
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-consecutive cumulative destinations accepted")


if __name__ == "__main__":
    test_initialization_and_counts()
    test_cumulative_mask_and_reader_mapping()
    test_incremental_caches_direct_feedback_and_resume()
    test_reject_invalid_configurations()
    print("EXPERIMENT_2C2_CPU_TESTS_PASS")
