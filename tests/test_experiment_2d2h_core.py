import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import experiment_2d2b_core as source_core  # noqa: E402
import experiment_2d2h_core as core  # noqa: E402
import smoke_test as support  # noqa: E402

SYMBOLS = support.load_training_symbols()
GPT, GPTConfig = SYMBOLS["GPT"], SYMBOLS["GPTConfig"]
torch.set_num_threads(1)
LENGTH = 40


def base_model(seed=29):
    torch.manual_seed(seed)
    return GPT(GPTConfig(block_size=LENGTH, vocab_size=32, n_layer=12,
                         n_head=2, n_embd=8, residual_mode="standard"))


def tiny_model(seed=29):
    return core.RecurrentKVGPT(base_model(seed))


def test_parameter_swap_physically_removes_b1_gate():
    source, target = source_core.RecurrentKVGPT(base_model()), tiny_model()
    source_base = {n: tuple(p.shape) for n, p in source.named_parameters() if n != "g_rec"}
    target_base = {n: tuple(p.shape) for n, p in target.named_parameters() if n != "g_rec_b2"}
    assert source_base == target_base
    assert "g_rec" not in dict(target.named_parameters())
    assert dict(target.named_parameters())["g_rec_b2"].shape == torch.Size([])
    assert sum(p.numel() for p in target.parameters()) == sum(p.numel() for p in source.parameters())


def test_b1_is_physical_w2_only_and_rejects_removed_api():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, LENGTH))
    result = model.forward_pass(tokens, return_diagnostics=True)
    b1 = result["diagnostics"]["b1"]
    assert result["diagnostics"]["b1_recurrent_path_present"] is False
    assert b1["recurrent_path_present"] is False
    assert b1["gate_raw"] is None and b1["recurrent_attention_weights"] is None
    with pytest.raises(TypeError):
        model.forward_pass(tokens, b1_recurrent_source=result["h12"])


def test_b2_w32_and_older_bank_partition_context_exactly():
    model = tiny_model().eval()
    source = torch.randn(2, LENGTH, 8)
    bank = model.build_recurrent_bank_b2(source)
    local = model.b2_local_mask(LENGTH, source.device)
    assert bank.values.data_ptr() == source.data_ptr()
    for t in range(LENGTH):
        old = set(torch.where(bank.valid_mask[t])[0].tolist())
        recent = set(torch.where(local[t])[0].tolist())
        assert old == set(range(max(0, t - 1023), max(0, t - 31)))
        assert recent == set(range(max(0, t - 31), t + 1))
        assert old.isdisjoint(recent)
        assert old | recent == set(range(max(0, t - 1023), t + 1))


def test_gate_zero_identity_and_writer_gradient():
    model = tiny_model().train()
    tokens = torch.randint(0, 32, (2, LENGTH))
    first = model.forward_pass(tokens)
    plain = model.forward_pass(tokens)["logits"]
    zero = model.forward_pass(tokens, b2_recurrent_source=first["h11"],
                              b2_gate_override=0.0)["logits"]
    assert torch.equal(plain, zero)
    with torch.no_grad():
        model.g_rec_b2.fill_(0.2)
    first = model.forward_pass(tokens)
    second = model.forward_pass(tokens, b2_recurrent_source=first["h11"])
    gradient = torch.autograd.grad(second["logits"][:, -1].square().sum(), first["h11"])[0]
    assert torch.isfinite(gradient).all()
    assert gradient[:, : LENGTH - 32].count_nonzero() > 0
    assert gradient[:, LENGTH - 32 :].count_nonzero() == 0


def test_incremental_cache_has_no_b12_ring_and_exact_bounds():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, LENGTH))
    result = model.incremental_logits(tokens, control="real", return_diagnostics=True)
    state, audit = result["state"], result["cache_audit"]
    assert not hasattr(state, "h12_ring")
    assert result["max_cache_lengths"] == (1, 31) + (LENGTH - 1,) * 10
    assert result["max_h11_ring_length"] == LENGTH
    assert result["max_h12_ring_length"] == 0
    assert audit["passed"] and audit["h12_recurrent_ring_present"] is False
    assert audit["b1_historical_kv"] <= 1 and audit["b2_historical_kv"] <= 31
    assert audit["h11_ring_length"] <= 1023


def test_incremental_gate_zero_matches_one_pass_parallel():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, LENGTH))
    parallel = model.forward_pass(tokens)["logits"]
    incremental = model.incremental_logits(tokens, control="b2_recurrence_off",
                                           b2_gate_override=0.0)["logits"]
    torch.testing.assert_close(incremental, parallel, rtol=2e-5, atol=2e-6)
