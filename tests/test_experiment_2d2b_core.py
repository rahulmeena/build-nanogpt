import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as support  # noqa: E402
import experiment_2d2a_core as old_core  # noqa: E402
import experiment_2d2b_core as core  # noqa: E402


SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)


def tiny_model(seed=29):
    torch.manual_seed(seed)
    base = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=12,
            n_head=2,
            n_embd=8,
            residual_mode="standard",
        )
    )
    return core.RecurrentKVGPT(base)


def test_parameter_inventory_is_identical_to_2d2a():
    model = tiny_model()
    old = old_core.RecurrentKVGPT(
        GPT(
            GPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=12,
                n_head=2,
                n_embd=8,
                residual_mode="standard",
            )
        )
    )
    new_inventory = [(name, tuple(value.shape)) for name, value in model.named_parameters()]
    old_inventory = [(name, tuple(value.shape)) for name, value in old.named_parameters()]
    assert new_inventory == old_inventory
    assert dict(new_inventory)["g_rec"] == ()
    assert sum(value.numel() for value in model.parameters()) == sum(
        value.numel() for value in old.parameters()
    )


def test_full_bank_is_shared_source_and_exact_nonoverlapping_mask():
    model = tiny_model().eval()
    source = torch.randn(2, 8, model.config.n_embd)
    bank = model.build_recurrent_bank(source)
    assert tuple(bank.values.shape) == (2, 8, model.config.n_embd)
    assert bank.values.data_ptr() == source.data_ptr()
    assert tuple(bank.valid_mask.shape) == (8, 8)
    for t in range(8):
        expected = set(range(max(0, t - 1023), max(0, t - 1)))
        actual = set(torch.where(bank.valid_mask[t])[0].tolist())
        assert actual == expected
        local = set(range(max(0, t - 1), t + 1))
        assert actual.isdisjoint(local)
    assert torch.equal(
        bank.valid_mask.sum(dim=-1), torch.tensor([0, 0, 1, 2, 3, 4, 5, 6])
    )


def test_bank_modes_have_exact_lag_sets():
    full = core.RecurrentKVGPT.recurrent_mask(40, 40, torch.device("cpu"), "full")
    two = core.RecurrentKVGPT.recurrent_mask(
        40, 40, torch.device("cpu"), "two_slot"
    )
    recent = core.RecurrentKVGPT.recurrent_mask(
        40, 40, torch.device("cpu"), "recent_only"
    )
    old = core.RecurrentKVGPT.recurrent_mask(
        40, 40, torch.device("cpu"), "old_only"
    )
    assert set(torch.where(two[39])[0].tolist()) == {36, 37}
    assert set(torch.where(recent[39])[0].tolist()) == set(range(8, 38))
    assert set(torch.where(old[39])[0].tolist()) == set(range(0, 8))
    assert torch.equal(full, recent | old)
    assert not (recent & old).any()


def test_projection_reuses_exact_b1_ln_and_fused_kv_rows():
    model = tiny_model().eval()
    source = torch.randn(2, 8, model.config.n_embd)
    key, value = model.project_recurrent_kv(source)
    block1 = model.base.transformer.h[0]
    _, expected_key, expected_value = block1.attn.c_attn(block1.ln_1(source)).split(
        model.config.n_embd, dim=-1
    )
    expected_key = expected_key.view(2, 8, 2, 4).transpose(1, 2)
    expected_value = expected_value.view(2, 8, 2, 4).transpose(1, 2)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


def test_full_bank_probabilities_and_old_writer_gradients():
    model = tiny_model().train()
    with torch.no_grad():
        model.g_rec.fill_(0.25)
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    first = model.forward_pass(tokens)
    second = model.forward_pass(
        tokens, recurrent_source=first["h12"], return_diagnostics=True
    )
    weights = second["diagnostics"]["recurrent_attention_weights"]
    assert tuple(weights.shape) == (2, 2, 8, 8)
    mask = second["diagnostics"]["recurrent_valid_mask"]
    assert weights.masked_select(~mask.view(1, 1, 8, 8)).count_nonzero() == 0
    torch.testing.assert_close(
        weights[:, :, 2:].sum(dim=-1),
        torch.ones_like(weights[:, :, 2:, 0]),
        rtol=0,
        atol=2e-7,
    )
    gradient = torch.autograd.grad(
        second["logits"][:, 7].square().sum(), first["h12"]
    )[0]
    assert torch.isfinite(gradient).all()
    assert gradient[:, 0].norm() > 0
    assert gradient[:, 3].norm() > 0
    assert gradient[:, 5].norm() > 0
    assert gradient[:, 6:].count_nonzero() == 0


def test_future_perturbation_and_row_isolation():
    model = tiny_model().eval()
    with torch.no_grad():
        model.g_rec.fill_(0.3)
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        reference = model.forward_multi_pass(tokens, num_passes=2)["logits"]
        future = tokens.clone()
        future[0, 7] = (future[0, 7] + 1) % model.config.vocab_size
        changed = model.forward_multi_pass(future, num_passes=2)["logits"]
        assert torch.equal(reference[0, :7], changed[0, :7])
        row = tokens.clone()
        row[1] = (row[1] + 3) % model.config.vocab_size
        isolated = model.forward_multi_pass(row, num_passes=2)["logits"]
        assert torch.equal(reference[0], isolated[0])


def test_incremental_full_bank_excludes_newest_b12_and_bounds_state():
    model = tiny_model().eval()
    with torch.no_grad():
        model.g_rec.fill_(0.2)
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        result = model.incremental_logits(
            tokens, control="real", return_diagnostics=True
        )
    assert result["max_cache_lengths"] == (1,) + (7,) * 11
    assert result["max_h12_ring_length"] == 8
    assert result["cache_audit"]["passed"]
    assert result["cache_audit"]["b1_historical_kv"] == 1
    assert result["cache_audit"]["h12_ring_positions"] == tuple(range(8))
    diagnostics = result["diagnostics"]
    assert diagnostics[0]["recurrent_positions"].numel() == 0
    assert diagnostics[1]["recurrent_positions"].numel() == 0
    for t in range(2, 8):
        assert diagnostics[t]["recurrent_positions"].tolist() == [list(range(t - 1))]
        assert (t - 1) not in diagnostics[t]["recurrent_positions"]


def test_gate_zero_incremental_matches_parallel_with_kernel_tolerance():
    model = tiny_model().eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        parallel = model.forward_pass(tokens)["logits"]
        plain = model.incremental_logits(tokens, control="plain")["logits"]
        full = model.incremental_logits(tokens, control="real")["logits"]
    torch.testing.assert_close(plain, parallel, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(full, parallel, rtol=2e-5, atol=2e-6)
