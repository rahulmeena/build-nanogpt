import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as support  # noqa: E402
import experiment_2d2e_core as source_core  # noqa: E402
import experiment_2d2i_core as core  # noqa: E402

SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)

TEST_LENGTH = 136


def base_model(seed=41):
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(
            block_size=TEST_LENGTH,
            vocab_size=32,
            n_layer=12,
            n_head=2,
            n_embd=8,
            residual_mode="standard",
        )
    )


def tiny_model(seed=41):
    return core.RecurrentKVGPT(base_model(seed))


def test_parameter_inventory_is_2d2e_plus_exactly_one_scalar():
    source = source_core.RecurrentKVGPT(base_model())
    target = core.RecurrentKVGPT(base_model())
    old = [(name, tuple(value.shape)) for name, value in source.named_parameters()]
    new = [(name, tuple(value.shape)) for name, value in target.named_parameters()]
    assert [row for row in new if row[0] != "g_rec_b4"] == old
    assert dict(new)["g_rec_b4"] == ()
    assert sum(value.numel() for value in target.parameters()) == 1 + sum(
        value.numel() for value in source.parameters()
    )


def test_four_complementary_banks_have_exact_nonoverlapping_geometry():
    model = tiny_model().eval()
    states = torch.randn(2, TEST_LENGTH, model.config.n_embd)
    banks = (
        (model.build_recurrent_bank(states), model.local_mask(TEST_LENGTH, states.device), 2),
        (model.build_recurrent_bank_b2(states), model.b2_local_mask(TEST_LENGTH, states.device), 32),
        (model.build_recurrent_bank_b3(states), model.b3_local_mask(TEST_LENGTH, states.device), 64),
        (model.build_recurrent_bank_b4(states), model.b4_local_mask(TEST_LENGTH, states.device), 128),
    )
    for bank, local, minimum_lag in banks:
        assert bank.values.data_ptr() == states.data_ptr()
        for t in range(TEST_LENGTH):
            recurrent = set(torch.where(bank.valid_mask[t])[0].tolist())
            ordinary = set(torch.where(local[t])[0].tolist())
            expected_recurrent = set(
                range(max(0, t - 1023), max(0, t - minimum_lag + 1))
            )
            expected_ordinary = set(range(max(0, t - minimum_lag + 1), t + 1))
            assert recurrent == expected_recurrent
            assert ordinary == expected_ordinary
            assert recurrent.isdisjoint(ordinary)
            assert recurrent | ordinary == set(range(max(0, t - 1023), t + 1))

    full = model.b4_recurrent_mask(1024, 1024, states.device)
    assert int(full[127].sum()) == 0
    assert torch.where(full[128])[0].tolist() == [0]
    assert torch.where(full[129])[0].tolist() == [0, 1]
    assert torch.where(full[1023])[0].tolist() == list(range(896))
    assert int(full.sum(-1).max()) == core.B4_MAX_RECURRENT_ENTRIES == 896


def test_b4_projection_reuses_destination_ln_and_kv_slices():
    model = tiny_model().eval()
    states = torch.randn(2, TEST_LENGTH, model.config.n_embd)
    key, value = model.project_recurrent_kv_b4(states)
    block = model.base.transformer.h[3]
    _, expected_key, expected_value = block.attn.c_attn(block.ln_1(states)).split(8, -1)
    expected_key = expected_key.view(2, TEST_LENGTH, 2, 4).transpose(1, 2)
    expected_value = expected_value.view(2, TEST_LENGTH, 2, 4).transpose(1, 2)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


def test_b4_gate_zero_identity_and_separate_softmax():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    first = model.forward_pass(tokens)
    without = model.forward_pass(
        tokens,
        b1_recurrent_source=first["h12"],
        b2_recurrent_source=first["h11"],
        b3_recurrent_source=first["h10"],
        b4_gate_override=0.0,
    )["logits"]
    active = model.forward_pass(
        tokens,
        b1_recurrent_source=first["h12"],
        b2_recurrent_source=first["h11"],
        b3_recurrent_source=first["h10"],
        b4_recurrent_source=first["h9"],
        b4_gate_override=0.0,
        return_diagnostics=True,
    )
    assert torch.equal(without, active["logits"])
    diagnostics = active["diagnostics"]["b4"]
    recurrent = diagnostics["recurrent_attention_weights"]
    mask = diagnostics["recurrent_valid_mask"]
    assert recurrent.masked_select(~mask.view(1, 1, TEST_LENGTH, TEST_LENGTH)).count_nonzero() == 0
    assert recurrent[:, :, :128].count_nonzero() == 0
    torch.testing.assert_close(
        recurrent[:, :, 128:].sum(-1),
        torch.ones_like(recurrent[:, :, 128:, 0]),
        rtol=0,
        atol=5e-7,
    )
    local = diagnostics["local_attention_weights"]
    local_mask = diagnostics["local_valid_mask"]
    assert local.masked_select(~local_mask.view(1, 1, TEST_LENGTH, TEST_LENGTH)).count_nonzero() == 0
    torch.testing.assert_close(local.sum(-1), torch.ones_like(local[..., 0]), rtol=0, atol=5e-7)


def test_b4_full_counterfactual_matches_frozen_2d2e_source():
    source = source_core.RecurrentKVGPT(base_model()).eval()
    target = tiny_model().eval()
    missing, unexpected = target.load_state_dict(source.state_dict(), strict=False)
    assert missing == ["g_rec_b4"] and unexpected == []
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    expected = source.forward_pass(tokens)["logits"]
    actual = target.forward_pass(tokens, b4_full_counterfactual=True)["logits"]
    assert torch.equal(expected, actual)


def test_incremental_state_has_w2_w32_w64_w128_caches_and_four_rings():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    result = model.incremental_logits(tokens, control="all_real", return_diagnostics=True)
    assert result["max_cache_lengths"] == (1, 31, 63, 127) + (TEST_LENGTH - 1,) * 8
    assert result["max_h9_ring_length"] == TEST_LENGTH
    assert result["max_h10_ring_length"] == TEST_LENGTH
    assert result["max_h11_ring_length"] == result["max_h12_ring_length"] == TEST_LENGTH
    audit = result["cache_audit"]
    assert audit["passed"] and audit["physical_storage_exact"]
    assert audit["b1_historical_kv"] == 1
    assert audit["b2_historical_kv"] == 31
    assert audit["b3_historical_kv"] == 63
    assert audit["b4_historical_kv"] == 127
    expected_positions = tuple(range(TEST_LENGTH))
    assert audit["h9_ring_positions"] == expected_positions
    assert audit["h10_ring_positions"] == expected_positions
    assert audit["h11_ring_positions"] == expected_positions
    assert audit["h12_ring_positions"] == expected_positions
    final = result["diagnostics"][-1]
    assert final["b1"]["recurrent_positions"].tolist() == [list(range(TEST_LENGTH - 2))]
    assert final["b2"]["recurrent_positions"].tolist() == [list(range(TEST_LENGTH - 32))]
    assert final["b3"]["recurrent_positions"].tolist() == [list(range(TEST_LENGTH - 64))]
    assert final["b4"]["recurrent_positions"].tolist() == [list(range(TEST_LENGTH - 128))]

    full = model.incremental_logits(tokens, control="b4_full_counterfactual")
    assert full["max_cache_lengths"] == (1, 31, 63, TEST_LENGTH - 1) + (TEST_LENGTH - 1,) * 8
    assert full["cache_audit"]["passed"]


def test_gate_zero_incremental_matches_one_pass_parallel():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    parallel = model.forward_pass(tokens)["logits"]
    incremental = model.incremental_logits(
        tokens,
        control="all_real",
        b1_gate_override=0.0,
        b2_gate_override=0.0,
        b3_gate_override=0.0,
        b4_gate_override=0.0,
    )["logits"]
    torch.testing.assert_close(incremental, parallel, rtol=2e-5, atol=2e-6)
