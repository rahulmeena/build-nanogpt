import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as support  # noqa: E402
import experiment_2d2d_core as source_core  # noqa: E402
import experiment_2d2e_core as core  # noqa: E402

SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)

TEST_LENGTH = 72


def base_model(seed=29):
    torch.manual_seed(seed)
    return GPT(GPTConfig(block_size=TEST_LENGTH, vocab_size=32, n_layer=12,
                         n_head=2, n_embd=8, residual_mode="standard"))


def tiny_model(seed=29):
    return core.RecurrentKVGPT(base_model(seed))


def test_parameter_inventory_is_2d2d_plus_exactly_one_scalar():
    source = source_core.RecurrentKVGPT(base_model())
    target = core.RecurrentKVGPT(base_model())
    old = [(name, tuple(value.shape)) for name, value in source.named_parameters()]
    new = [(name, tuple(value.shape)) for name, value in target.named_parameters()]
    assert [row for row in new if row[0] != "g_rec_b3"] == old
    assert dict(new)["g_rec_b3"] == ()
    assert sum(value.numel() for value in target.parameters()) == 1 + sum(
        value.numel() for value in source.parameters())


def test_three_complementary_banks_have_exact_nonoverlapping_geometry():
    model = tiny_model().eval()
    states = torch.randn(2, TEST_LENGTH, model.config.n_embd)
    banks = (
        (model.build_recurrent_bank(states), model.local_mask(TEST_LENGTH, states.device), 2),
        (model.build_recurrent_bank_b2(states), model.b2_local_mask(TEST_LENGTH, states.device), 32),
        (model.build_recurrent_bank_b3(states), model.b3_local_mask(TEST_LENGTH, states.device), 64),
    )
    for bank, local, minimum_lag in banks:
        assert bank.values.data_ptr() == states.data_ptr()
        for t in range(TEST_LENGTH):
            recurrent = set(torch.where(bank.valid_mask[t])[0].tolist())
            ordinary = set(torch.where(local[t])[0].tolist())
            expected_recurrent = set(range(max(0, t - 1023), max(0, t - minimum_lag + 1)))
            expected_ordinary = set(range(max(0, t - minimum_lag + 1), t + 1))
            assert recurrent == expected_recurrent
            assert ordinary == expected_ordinary
            assert recurrent.isdisjoint(ordinary)
            assert recurrent | ordinary == set(range(max(0, t - 1023), t + 1))

    full_b3 = model.b3_recurrent_mask(1024, 1024, states.device)
    assert int(full_b3[63].sum()) == 0
    assert torch.where(full_b3[64])[0].tolist() == [0]
    assert torch.where(full_b3[65])[0].tolist() == [0, 1]
    assert torch.where(full_b3[1023])[0].tolist() == list(range(960))
    assert int(full_b3.sum(-1).max()) == core.B3_MAX_RECURRENT_ENTRIES == 960


def test_recurrent_projection_reuses_each_destination_ln_and_kv_slices():
    model = tiny_model().eval()
    states = torch.randn(2, TEST_LENGTH, model.config.n_embd)
    projectors = (model.project_recurrent_kv, model.project_recurrent_kv_b2,
                  model.project_recurrent_kv_b3)
    for block_index, projector in enumerate(projectors):
        key, value = projector(states)
        block = model.base.transformer.h[block_index]
        _, expected_key, expected_value = block.attn.c_attn(block.ln_1(states)).split(8, -1)
        expected_key = expected_key.view(2, TEST_LENGTH, 2, 4).transpose(1, 2)
        expected_value = expected_value.view(2, TEST_LENGTH, 2, 4).transpose(1, 2)
        torch.testing.assert_close(key, expected_key)
        torch.testing.assert_close(value, expected_value)


def test_b3_gate_zero_identity_and_all_links_use_separate_softmaxes():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    first = model.forward_pass(tokens)
    without = model.forward_pass(
        tokens, b1_recurrent_source=first["h12"], b2_recurrent_source=first["h11"],
        b3_gate_override=0.0,
    )["logits"]
    active = model.forward_pass(
        tokens, b1_recurrent_source=first["h12"], b2_recurrent_source=first["h11"],
        b3_recurrent_source=first["h10"], b3_gate_override=0.0,
        return_diagnostics=True,
    )
    assert torch.equal(without, active["logits"])
    for link, first_valid in (("b1", 2), ("b2", 32), ("b3", 64)):
        diagnostics = active["diagnostics"][link]
        recurrent = diagnostics["recurrent_attention_weights"]
        mask = diagnostics["recurrent_valid_mask"]
        assert recurrent.masked_select(~mask.view(1, 1, TEST_LENGTH, TEST_LENGTH)).count_nonzero() == 0
        assert recurrent[:, :, :first_valid].count_nonzero() == 0
        torch.testing.assert_close(recurrent[:, :, first_valid:].sum(-1),
                                   torch.ones_like(recurrent[:, :, first_valid:, 0]),
                                   rtol=0, atol=5e-7)
        local = diagnostics["local_attention_weights"]
        local_mask = diagnostics["local_valid_mask"]
        assert local.masked_select(~local_mask.view(1, 1, TEST_LENGTH, TEST_LENGTH)).count_nonzero() == 0
        torch.testing.assert_close(local.sum(-1), torch.ones_like(local[..., 0]), rtol=0, atol=5e-7)


def test_all_three_writer_paths_are_attached_after_gates_open():
    model = tiny_model().train()
    with torch.no_grad():
        model.g_rec.fill_(0.25); model.g_rec_b2.fill_(0.20); model.g_rec_b3.fill_(0.15)
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    first = model.forward_pass(tokens)
    sources = (("h12", "b1", 2), ("h11", "b2", 32), ("h10", "b3", 64))
    for source_name, active_link, lag in sources:
        kwargs = {
            "b1_recurrent_source": first["h12"].detach(),
            "b2_recurrent_source": first["h11"].detach(),
            "b3_recurrent_source": first["h10"].detach(),
            "b1_gate_override": 0.0,
            "b2_gate_override": 0.0,
            "b3_gate_override": 0.0,
        }
        source = first[source_name]
        kwargs[f"{active_link}_recurrent_source"] = source
        kwargs[f"{active_link}_gate_override"] = 0.2
        second = model.forward_pass(tokens, **kwargs)
        gradient = torch.autograd.grad(second["logits"][:, -1].square().sum(), source,
                                       retain_graph=True)[0]
        assert torch.isfinite(gradient).all()
        assert gradient[:, : TEST_LENGTH - lag].count_nonzero() > 0
        assert gradient[:, TEST_LENGTH - lag :].count_nonzero() == 0


def test_future_causality_and_row_isolation():
    model = tiny_model().eval()
    with torch.no_grad():
        model.g_rec.fill_(0.3); model.g_rec_b2.fill_(0.2); model.g_rec_b3.fill_(0.1)
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    reference = model.forward_multi_pass(tokens, num_passes=2)["logits"]
    future = tokens.clone(); future[0, -1] = (future[0, -1] + 1) % 32
    changed = model.forward_multi_pass(future, num_passes=2)["logits"]
    assert torch.equal(reference[0, :-1], changed[0, :-1])
    row = tokens.clone(); row[1] = (row[1] + 3) % 32
    isolated = model.forward_multi_pass(row, num_passes=2)["logits"]
    assert torch.equal(reference[0], isolated[0])


def test_incremental_state_has_w2_w32_w64_caches_and_three_raw_rings():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    result = model.incremental_logits(tokens, control="all_real", return_diagnostics=True)
    assert result["max_cache_lengths"] == (1, 31, 63) + (TEST_LENGTH - 1,) * 9
    assert result["max_h10_ring_length"] == TEST_LENGTH
    assert result["max_h11_ring_length"] == result["max_h12_ring_length"] == TEST_LENGTH
    audit = result["cache_audit"]
    assert audit["passed"] and audit["physical_storage_exact"]
    assert audit["b1_historical_kv"] == 1
    assert audit["b2_historical_kv"] == 31
    assert audit["b3_historical_kv"] == 63
    expected_positions = tuple(range(TEST_LENGTH))
    assert audit["h10_ring_positions"] == expected_positions
    assert audit["h11_ring_positions"] == expected_positions
    assert audit["h12_ring_positions"] == expected_positions
    final = result["diagnostics"][-1]
    assert final["b1"]["recurrent_positions"].tolist() == [list(range(TEST_LENGTH - 2))]
    assert final["b2"]["recurrent_positions"].tolist() == [list(range(TEST_LENGTH - 32))]
    assert final["b3"]["recurrent_positions"].tolist() == [list(range(TEST_LENGTH - 64))]

    full = model.incremental_logits(tokens, control="b3_full_counterfactual")
    assert full["max_cache_lengths"] == (1, 31, TEST_LENGTH - 1) + (TEST_LENGTH - 1,) * 9
    assert full["cache_audit"]["passed"]


def test_gate_zero_incremental_matches_one_pass_parallel():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    parallel = model.forward_pass(tokens)["logits"]
    incremental = model.incremental_logits(
        tokens, control="all_real", b1_gate_override=0.0,
        b2_gate_override=0.0, b3_gate_override=0.0,
    )["logits"]
    torch.testing.assert_close(incremental, parallel, rtol=2e-5, atol=2e-6)
