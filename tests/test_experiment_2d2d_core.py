import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as support  # noqa: E402
import experiment_2d2b_core as source_core  # noqa: E402
import experiment_2d2d_core as core  # noqa: E402

SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)

TEST_LENGTH = 40


def base_model(seed=29):
    torch.manual_seed(seed)
    return GPT(GPTConfig(block_size=TEST_LENGTH, vocab_size=32, n_layer=12, n_head=2,
                         n_embd=8, residual_mode="standard"))


def tiny_model(seed=29):
    return core.RecurrentKVGPT(base_model(seed))


def test_parameter_inventory_is_2d2b_plus_exactly_one_scalar():
    source = source_core.RecurrentKVGPT(base_model())
    target = core.RecurrentKVGPT(base_model())
    old = [(name, tuple(value.shape)) for name, value in source.named_parameters()]
    new = [(name, tuple(value.shape)) for name, value in target.named_parameters()]
    assert [row for row in new if row[0] != "g_rec_b2"] == old
    assert dict(new)["g_rec_b2"] == ()
    assert sum(value.numel() for value in target.parameters()) == 1 + sum(
        value.numel() for value in source.parameters())


def test_b1_full_and_b2_older_banks_use_exact_nonoverlapping_geometry():
    model = tiny_model().eval()
    source = torch.randn(2, TEST_LENGTH, model.config.n_embd)
    b1_bank = model.build_recurrent_bank(source)
    b2_bank = model.build_recurrent_bank_b2(source)
    assert b1_bank.values.data_ptr() == source.data_ptr()
    assert b2_bank.values.data_ptr() == source.data_ptr()
    assert torch.equal(
        b1_bank.valid_mask.sum(-1),
        torch.tensor([0, 0] + list(range(1, TEST_LENGTH - 1))),
    )
    assert torch.equal(
        b2_bank.valid_mask.sum(-1),
        torch.tensor([0] * 32 + list(range(1, TEST_LENGTH - 31))),
    )
    b1_local = model.local_mask(TEST_LENGTH, source.device)
    b2_local = model.b2_local_mask(TEST_LENGTH, source.device)
    for t in range(TEST_LENGTH):
        b1_expected = set(range(max(0, t - 1023), max(0, t - 1)))
        b2_expected = set(range(max(0, t - 1023), max(0, t - 31)))
        b2_local_expected = set(range(max(0, t - 31), t + 1))
        assert set(torch.where(b1_bank.valid_mask[t])[0].tolist()) == b1_expected
        assert set(torch.where(b2_bank.valid_mask[t])[0].tolist()) == b2_expected
        assert set(torch.where(b2_local[t])[0].tolist()) == b2_local_expected
        assert b2_expected.isdisjoint(b2_local_expected)
        assert b2_expected | b2_local_expected == set(
            range(max(0, t - 1023), t + 1)
        )
        assert not bool((b1_bank.valid_mask[t] & b1_local[t]).any())

    full_b2 = model.b2_recurrent_mask(1024, 1024, source.device)
    assert int(full_b2[31].sum()) == 0
    assert torch.where(full_b2[32])[0].tolist() == [0]
    assert torch.where(full_b2[100])[0].tolist() == list(range(69))
    assert torch.where(full_b2[1023])[0].tolist() == list(range(992))
    assert int(full_b2.sum(-1).max()) == core.B2_MAX_RECURRENT_ENTRIES == 992


def test_recurrent_projection_reuses_each_destination_ln_and_kv_slices():
    model = tiny_model().eval()
    states = torch.randn(2, TEST_LENGTH, model.config.n_embd)
    for block_index, projector in ((0, model.project_recurrent_kv),
                                   (1, model.project_recurrent_kv_b2)):
        key, value = projector(states)
        block = model.base.transformer.h[block_index]
        _, expected_key, expected_value = block.attn.c_attn(block.ln_1(states)).split(8, -1)
        expected_key = expected_key.view(2, TEST_LENGTH, 2, 4).transpose(1, 2)
        expected_value = expected_value.view(2, TEST_LENGTH, 2, 4).transpose(1, 2)
        torch.testing.assert_close(key, expected_key)
        torch.testing.assert_close(value, expected_value)


def test_b2_gate_zero_identity_and_separate_softmaxes():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    first = model.forward_pass(tokens)
    without = model.forward_pass(tokens, b1_recurrent_source=first["h12"],
                                 b2_gate_override=0.0)["logits"]
    active = model.forward_pass(tokens, b1_recurrent_source=first["h12"],
                                b2_recurrent_source=first["h11"],
                                b2_gate_override=0.0, return_diagnostics=True)
    assert torch.equal(without, active["logits"])
    for link, first_valid in (("b1", 2), ("b2", 32)):
        diagnostics = active["diagnostics"][link]
        weights = diagnostics["recurrent_attention_weights"]
        mask = diagnostics["recurrent_valid_mask"]
        assert weights.masked_select(
            ~mask.view(1, 1, TEST_LENGTH, TEST_LENGTH)
        ).count_nonzero() == 0
        assert weights[:, :, :first_valid].count_nonzero() == 0
        torch.testing.assert_close(weights[:, :, first_valid:].sum(-1),
                                   torch.ones_like(weights[:, :, first_valid:, 0]),
                                   rtol=0, atol=5e-7)

    b2 = active["diagnostics"]["b2"]
    local_weights = b2["local_attention_weights"]
    local_mask = b2["local_valid_mask"]
    assert local_weights is not None
    assert local_weights.masked_select(
        ~local_mask.view(1, 1, TEST_LENGTH, TEST_LENGTH)
    ).count_nonzero() == 0
    torch.testing.assert_close(
        local_weights.sum(-1), torch.ones_like(local_weights[..., 0]),
        rtol=0, atol=2e-7,
    )


def test_both_writer_paths_are_attached_after_gates_open():
    model = tiny_model().train()
    with torch.no_grad():
        model.g_rec.fill_(0.25)
        model.g_rec_b2.fill_(0.20)
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))

    first_b1 = model.forward_pass(tokens)
    second_b1 = model.forward_pass(
        tokens,
        b1_recurrent_source=first_b1["h12"],
        b2_recurrent_source=first_b1["h11"].detach(),
        b2_gate_override=0.0,
    )
    g12 = torch.autograd.grad(
        second_b1["logits"][:, -1].square().sum(), first_b1["h12"]
    )[0]
    assert torch.isfinite(g12).all()
    assert g12[:, : TEST_LENGTH - 2].count_nonzero() > 0
    assert g12[:, TEST_LENGTH - 2 :].count_nonzero() == 0

    first_b2 = model.forward_pass(tokens)
    second_b2 = model.forward_pass(
        tokens,
        b1_recurrent_source=first_b2["h12"].detach(),
        b2_recurrent_source=first_b2["h11"],
        b1_gate_override=0.0,
    )
    g11 = torch.autograd.grad(
        second_b2["logits"][:, -1].square().sum(), first_b2["h11"]
    )[0]
    assert torch.isfinite(g11).all()
    assert g11[:, : TEST_LENGTH - 32].count_nonzero() > 0
    assert g11[:, TEST_LENGTH - 32 :].count_nonzero() == 0


def test_future_causality_and_row_isolation():
    model = tiny_model().eval()
    with torch.no_grad():
        model.g_rec.fill_(0.3)
        model.g_rec_b2.fill_(0.2)
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    reference = model.forward_multi_pass(tokens, num_passes=2)["logits"]
    future = tokens.clone(); future[0, -1] = (future[0, -1] + 1) % 32
    changed = model.forward_multi_pass(future, num_passes=2)["logits"]
    assert torch.equal(reference[0, :-1], changed[0, :-1])
    row = tokens.clone(); row[1] = (row[1] + 3) % 32
    isolated = model.forward_multi_pass(row, num_passes=2)["logits"]
    assert torch.equal(reference[0], isolated[0])


def test_incremental_state_has_w2_w32_caches_and_two_raw_rings():
    model = tiny_model().eval()
    with torch.no_grad():
        model.g_rec.fill_(0.2); model.g_rec_b2.fill_(0.1)
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    result = model.incremental_logits(tokens, control="both_real", return_diagnostics=True)
    assert result["max_cache_lengths"] == (1, 31) + (TEST_LENGTH - 1,) * 10
    assert result["max_h11_ring_length"] == result["max_h12_ring_length"] == TEST_LENGTH
    audit = result["cache_audit"]
    assert audit["passed"]
    assert audit["b1_historical_kv"] == 1
    assert audit["b2_historical_kv"] == audit["b2_historical_kv_limit"] == 31
    assert audit["h11_ring_positions"] == audit["h12_ring_positions"] == tuple(
        range(TEST_LENGTH)
    )
    for t, diagnostics in enumerate(result["diagnostics"]):
        b1_expected = [list(range(max(0, t - 1023), max(0, t - 1)))]
        b2_expected = [list(range(max(0, t - 1023), max(0, t - 31)))]
        assert diagnostics["b1"]["recurrent_positions"].tolist() == b1_expected
        assert diagnostics["b2"]["recurrent_positions"].tolist() == b2_expected

    full = model.incremental_logits(tokens, control="b2_full_counterfactual")
    assert full["max_cache_lengths"] == (1, TEST_LENGTH - 1) + (
        TEST_LENGTH - 1,
    ) * 10
    assert full["cache_audit"]["passed"]


def test_gate_zero_incremental_matches_one_pass_parallel():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    parallel = model.forward_pass(tokens)["logits"]
    incremental = model.incremental_logits(tokens, control="both_real",
                                           b1_gate_override=0.0,
                                           b2_gate_override=0.0)["logits"]
    torch.testing.assert_close(incremental, parallel, rtol=2e-5, atol=2e-6)
