import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as support  # noqa: E402
import experiment_2d2b_core as source_core  # noqa: E402
import experiment_2d2c_core as core  # noqa: E402

SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)


def base_model(seed=29):
    torch.manual_seed(seed)
    return GPT(GPTConfig(block_size=8, vocab_size=32, n_layer=12, n_head=2,
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


def test_both_full_banks_use_exact_nonoverlapping_lag_geometry():
    model = tiny_model().eval()
    source = torch.randn(2, 8, model.config.n_embd)
    bank = model.build_recurrent_bank(source)
    assert bank.values.data_ptr() == source.data_ptr()
    assert torch.equal(bank.valid_mask.sum(-1), torch.tensor([0, 0, 1, 2, 3, 4, 5, 6]))
    for t in range(8):
        expected = set(range(max(0, t - 1023), max(0, t - 1)))
        actual = set(torch.where(bank.valid_mask[t])[0].tolist())
        assert actual == expected
        assert actual.isdisjoint(set(range(max(0, t - 1), t + 1)))
    assert torch.equal(model.local_mask(8, source.device), model.b2_local_mask(8, source.device))


def test_recurrent_projection_reuses_each_destination_ln_and_kv_slices():
    model = tiny_model().eval()
    states = torch.randn(2, 8, model.config.n_embd)
    for block_index, projector in ((0, model.project_recurrent_kv),
                                   (1, model.project_recurrent_kv_b2)):
        key, value = projector(states)
        block = model.base.transformer.h[block_index]
        _, expected_key, expected_value = block.attn.c_attn(block.ln_1(states)).split(8, -1)
        expected_key = expected_key.view(2, 8, 2, 4).transpose(1, 2)
        expected_value = expected_value.view(2, 8, 2, 4).transpose(1, 2)
        torch.testing.assert_close(key, expected_key)
        torch.testing.assert_close(value, expected_value)


def test_b2_gate_zero_identity_and_separate_softmaxes():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, 8))
    first = model.forward_pass(tokens)
    without = model.forward_pass(tokens, b1_recurrent_source=first["h12"],
                                 b2_gate_override=0.0)["logits"]
    active = model.forward_pass(tokens, b1_recurrent_source=first["h12"],
                                b2_recurrent_source=first["h11"],
                                b2_gate_override=0.0, return_diagnostics=True)
    assert torch.equal(without, active["logits"])
    for link in ("b1", "b2"):
        diagnostics = active["diagnostics"][link]
        weights = diagnostics["recurrent_attention_weights"]
        mask = diagnostics["recurrent_valid_mask"]
        assert weights.masked_select(~mask.view(1, 1, 8, 8)).count_nonzero() == 0
        torch.testing.assert_close(weights[:, :, 2:].sum(-1),
                                   torch.ones_like(weights[:, :, 2:, 0]),
                                   rtol=0, atol=2e-7)


def test_both_writer_paths_are_attached_after_gates_open():
    model = tiny_model().train()
    with torch.no_grad():
        model.g_rec.fill_(0.25)
        model.g_rec_b2.fill_(0.20)
    tokens = torch.randint(0, 32, (2, 8))
    first = model.forward_pass(tokens)
    second = model.forward_pass(tokens, b1_recurrent_source=first["h12"],
                                b2_recurrent_source=first["h11"])
    g12, g11 = torch.autograd.grad(second["logits"][:, 7].square().sum(),
                                   (first["h12"], first["h11"]))
    for gradient in (g12, g11):
        assert torch.isfinite(gradient).all()
        assert gradient[:, :6].count_nonzero() > 0
        assert gradient[:, 6:].count_nonzero() == 0


def test_future_causality_and_row_isolation():
    model = tiny_model().eval()
    with torch.no_grad():
        model.g_rec.fill_(0.3)
        model.g_rec_b2.fill_(0.2)
    tokens = torch.randint(0, 32, (2, 8))
    reference = model.forward_multi_pass(tokens, num_passes=2)["logits"]
    future = tokens.clone(); future[0, 7] = (future[0, 7] + 1) % 32
    changed = model.forward_multi_pass(future, num_passes=2)["logits"]
    assert torch.equal(reference[0, :7], changed[0, :7])
    row = tokens.clone(); row[1] = (row[1] + 3) % 32
    isolated = model.forward_multi_pass(row, num_passes=2)["logits"]
    assert torch.equal(reference[0], isolated[0])


def test_incremental_state_has_two_w2_caches_and_two_raw_rings():
    model = tiny_model().eval()
    with torch.no_grad():
        model.g_rec.fill_(0.2); model.g_rec_b2.fill_(0.1)
    tokens = torch.randint(0, 32, (2, 8))
    result = model.incremental_logits(tokens, control="both_real", return_diagnostics=True)
    assert result["max_cache_lengths"] == (1, 1) + (7,) * 10
    assert result["max_h11_ring_length"] == result["max_h12_ring_length"] == 8
    audit = result["cache_audit"]
    assert audit["passed"]
    assert audit["b1_historical_kv"] == audit["b2_historical_kv"] == 1
    assert audit["h11_ring_positions"] == audit["h12_ring_positions"] == tuple(range(8))
    for t, diagnostics in enumerate(result["diagnostics"]):
        for link in ("b1", "b2"):
            expected = [[]] if t < 2 else [list(range(t - 1))]
            assert diagnostics[link]["recurrent_positions"].tolist() == expected


def test_gate_zero_incremental_matches_one_pass_parallel():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, 8))
    parallel = model.forward_pass(tokens)["logits"]
    incremental = model.incremental_logits(tokens, control="both_real",
                                           b1_gate_override=0.0,
                                           b2_gate_override=0.0)["logits"]
    torch.testing.assert_close(incremental, parallel, rtol=2e-5, atol=2e-6)
