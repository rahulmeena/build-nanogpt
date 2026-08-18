#!/usr/bin/env python3
"""CPU contract tests for Experiment 2C1 destination-depth semantics."""

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import experiment_2c1 as candidate  # noqa: E402


def small_model():
    symbols = candidate.a0.support.load_training_symbols()
    config = symbols["GPTConfig"](
        block_size=16,
        vocab_size=64,
        n_layer=12,
        n_head=2,
        n_embd=16,
        residual_mode="full_attnres",
        enable_topdown_feedback=True,
    )
    return symbols["GPT"](config).eval()


def test_config_contract():
    config = candidate.load_config()
    assert config["destinations"] == candidate.DESTINATIONS
    assert config["gpu_mapping"] == candidate.GPU_MAPPING
    assert config["ddp"] is False
    assert config["global_targets_per_update"] == 524_288
    assert config["milestone_updates"] == [10, 20, 29, 48]
    assert config["forced_process_restart_after_update"] == 20
    assert config["writers"] == "forbidden"
    assert config["hellaswag"] == "forbidden"


def test_all_destination_batch_contracts():
    torch.manual_seed(7)
    model = small_model()
    model.transformer.topdown_attnres.gate.data.fill_(0.2)
    tokens = torch.randint(0, 64, (3, 12))
    memory = torch.randn(4, 3, 12, 16)
    permutation = torch.tensor([1, 2, 0])
    for destination, block_number in candidate.DESTINATIONS.items():
        block = block_number - 1
        real, _ = model(
            tokens,
            mode="masked_destination_topdown_teacher",
            feedback_sources=memory,
            feedback_destination_block=block,
        )
        shuffled, _ = model(
            tokens,
            mode="masked_destination_shuffled_feedback",
            feedback_sources=memory,
            feedback_permutation=permutation,
            feedback_destination_block=block,
        )
        masked, _ = model(
            tokens,
            mode="masked_destination_no_feedback",
            feedback_destination_block=block,
        )
        zero, _ = model(
            tokens,
            mode="masked_destination_topdown_teacher",
            feedback_sources=memory,
            feedback_gate_override=0.0,
            feedback_destination_block=block,
        )
        assert torch.equal(masked, zero)
        assert not torch.equal(real, shuffled)
        assert torch.isfinite(real).all()


def test_future_causality_and_row_isolation():
    torch.manual_seed(11)
    model = small_model()
    model.transformer.topdown_attnres.gate.data.fill_(0.3)
    tokens = torch.randint(0, 64, (2, 12))
    memory = torch.randn(4, 2, 12, 16)
    future_tokens = tokens.clone()
    future_tokens[:, 7:] = (future_tokens[:, 7:] + 5) % 64
    future_memory = memory.clone()
    future_memory[:, :, 7:] += 3
    row_tokens = tokens.clone()
    row_tokens[1] = (row_tokens[1] + 9) % 64
    row_memory = memory.clone()
    row_memory[:, 1] += 4
    for block in (0, 4, 8, 11):
        original, _ = model(
            tokens,
            mode="masked_destination_topdown_teacher",
            feedback_sources=memory,
            feedback_destination_block=block,
        )
        future, _ = model(
            future_tokens,
            mode="masked_destination_topdown_teacher",
            feedback_sources=future_memory,
            feedback_destination_block=block,
        )
        changed_row, _ = model(
            row_tokens,
            mode="masked_destination_topdown_teacher",
            feedback_sources=row_memory,
            feedback_destination_block=block,
        )
        assert torch.equal(original[:, :7], future[:, :7])
        assert torch.equal(original[0], changed_row[0])


def test_d1_legacy_equivalence():
    torch.manual_seed(17)
    model = small_model()
    model.transformer.topdown_attnres.gate.data.fill_(0.4)
    tokens = torch.randint(0, 64, (2, 10))
    memory = torch.randn(4, 2, 10, 16)
    legacy, _ = model(
        tokens,
        mode="masked_l1_topdown_teacher",
        feedback_sources=memory,
    )
    generalized, _ = model(
        tokens,
        mode="masked_destination_topdown_teacher",
        feedback_sources=memory,
        feedback_destination_block=0,
    )
    assert torch.equal(legacy, generalized)


def test_single_block_recurrent_state_and_feedback():
    torch.manual_seed(23)
    model = small_model()
    token = torch.randint(0, 64, (2,))
    for block in (0, 4, 8, 11):
        zero_state = model.init_recurrent_state(
            2,
            "masked_single_no_feedback",
            dtype=torch.float32,
            mask_depth=0,
            masked_block_index=block,
        )
        one_state = model.init_recurrent_state(
            2,
            "masked_single_no_feedback",
            dtype=torch.float32,
            mask_depth=0,
            masked_block_index=block,
        )
        assert all(
            (cache is None) == (index == block)
            for index, cache in enumerate(zero_state.kv_caches)
        )
        zero_logits, zero_next = model.forward_step(
            token,
            zero_state,
            attention_feedback=torch.zeros(2, 1, 16),
        )
        one_logits, one_next = model.forward_step(
            token,
            one_state,
            attention_feedback=torch.ones(2, 1, 16),
        )
        assert not torch.equal(zero_logits, one_logits)
        assert zero_next.kv_caches[block] is None
        assert all(
            cache.length == 1
            for index, cache in enumerate(zero_next.kv_caches)
            if index != block
        )
        payload = zero_next.state_dict()
        assert payload["schema"] == "full_attnres_recurrent_state_v3"
        assert payload["masked_block_index"] == block
        restored = model.load_recurrent_state(payload, dtype=torch.float32)
        assert restored.masked_block_index == block
        assert restored.kv_caches[block] is None
        assert torch.equal(restored.feedback_memory, zero_next.feedback_memory)


def test_instrumentation_contract():
    model = small_model()
    reader = model.transformer.topdown_attnres
    reader.instrumentation_enabled = True
    values = [torch.randn(2, 3, 16) for _ in range(4)]
    output = reader(values)
    assert torch.isfinite(output).all()
    assert set(("topdown_rms", "feedback_rms")) <= set(reader.last_stats)
    assert reader.last_stats["topdown_rms"] > 0
    assert reader.last_stats["feedback_rms"] == 0


def test_scalar_reader_hash_serialization():
    model = small_model()
    initial = candidate.reader_state_sha(model)
    assert initial == candidate.reader_state_sha(model)
    scalar_hash = candidate.tensor_sha256(
        "scalar_gate", model.transformer.topdown_attnres.gate
    )
    assert len(scalar_hash) == 64
    model.transformer.topdown_attnres.gate.data.fill_(0.25)
    assert candidate.reader_state_sha(model) != initial


def main():
    test_config_contract()
    test_all_destination_batch_contracts()
    test_future_causality_and_row_isolation()
    test_d1_legacy_equivalence()
    test_single_block_recurrent_state_and_feedback()
    test_instrumentation_contract()
    test_scalar_reader_hash_serialization()
    json.loads(candidate.CONFIG_PATH.read_text())
    print("Experiment 2C1 CPU contract tests: PASS")


if __name__ == "__main__":
    main()
