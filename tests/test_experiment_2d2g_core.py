import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d2b_core as stage_a_core  # noqa: E402
import experiment_2d2g as driver  # noqa: E402
import experiment_2d2g_core as core  # noqa: E402
import smoke_test as support  # noqa: E402

SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)
TEST_LENGTH = 72


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


def model(seed=41):
    return core.StageBRecurrentKVGPT(base_model(seed))


def test_stage_a_is_exact_frozen_2d2b_kernel():
    assert core.StageARecurrentKVGPT is stage_a_core.FullB12ToB1RecurrentKVGPT


def test_stage_b_inventory_is_stage_a_plus_one_scalar_and_no_b2_gate():
    stage_a = core.StageARecurrentKVGPT(base_model())
    stage_b = model()
    a = [(name, tuple(value.shape)) for name, value in stage_a.named_parameters()]
    b = [(name, tuple(value.shape)) for name, value in stage_b.named_parameters()]
    assert [row for row in b if row[0] != "g_rec_b3"] == a
    assert dict(b)["g_rec_b3"] == ()
    assert "g_rec_b2" not in dict(b)
    assert sum(p.numel() for p in stage_b.parameters()) == sum(
        p.numel() for p in stage_a.parameters()
    ) + 1


def test_b3_geometry_is_exact_and_nonoverlapping():
    current = model().eval()
    states = torch.randn(2, TEST_LENGTH, current.config.n_embd)
    bank = current.build_recurrent_bank_b3(states)
    local = current.b3_local_mask(TEST_LENGTH, states.device)
    assert bank.values.data_ptr() == states.data_ptr()
    for t in range(TEST_LENGTH):
        recurrent = set(torch.where(bank.valid_mask[t])[0].tolist())
        ordinary = set(torch.where(local[t])[0].tolist())
        expected_recurrent = set(range(max(0, t - 1023), max(0, t - 63)))
        expected_local = set(range(max(0, t - 63), t + 1))
        assert recurrent == expected_recurrent
        assert ordinary == expected_local
        assert recurrent.isdisjoint(ordinary)
        assert recurrent | ordinary == set(range(max(0, t - 1023), t + 1))


def test_b3_projection_reuses_destination_ln_and_kv():
    current = model().eval()
    states = torch.randn(2, TEST_LENGTH, current.config.n_embd)
    key, value = current.project_recurrent_kv_b3(states)
    block = current.base.transformer.h[2]
    _, expected_key, expected_value = block.attn.c_attn(block.ln_1(states)).split(8, -1)
    expected_key = expected_key.view(2, TEST_LENGTH, 2, 4).transpose(1, 2)
    expected_value = expected_value.view(2, TEST_LENGTH, 2, 4).transpose(1, 2)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


def test_b3_gate_zero_identity_and_full_b2_is_ordinary():
    current = model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    first = current.forward_pass(tokens)
    without = current.forward_pass(
        tokens,
        b1_recurrent_source=first["h12"],
        b3_gate_override=0.0,
    )["logits"]
    active = current.forward_pass(
        tokens,
        b1_recurrent_source=first["h12"],
        b3_recurrent_source=first["h10"],
        b3_gate_override=0.0,
    )["logits"]
    assert torch.equal(without, active)
    assert not hasattr(current, "g_rec_b2")
    assert not hasattr(current, "build_recurrent_bank_b2")


def test_writer_path_attached_and_causal():
    current = model().train()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    first = current.forward_pass(tokens)
    second = current.forward_pass(
        tokens,
        b1_recurrent_source=first["h12"].detach(),
        b3_recurrent_source=first["h10"],
        b1_gate_override=0.0,
        b3_gate_override=0.2,
    )
    gradient = torch.autograd.grad(second["logits"][:, -1].square().sum(), first["h10"])[0]
    assert gradient[:, : TEST_LENGTH - 64].count_nonzero() > 0
    assert gradient[:, TEST_LENGTH - 64 :].count_nonzero() == 0


def test_runtime_causality_and_post_open_writer_audits():
    current = model().train()
    with torch.no_grad():
        current.g_rec_b3.fill_(0.02)
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    targets = torch.randint(0, 32, (2, TEST_LENGTH))
    causality = driver.b3_causality_audit(current, TEST_LENGTH, tokens.device)
    writer = driver.b3_writer_gradient_audit(current, tokens, targets)
    assert causality["passed"]
    assert writer["checks"]["gate_open"]
    assert writer["checks"]["gradient_finite"]
    assert writer["checks"]["eligible_writer_gradient_nonzero"]
    assert writer["checks"]["ineligible_last_64_exact_zero"]


def test_no_future_leakage_or_row_cross_talk():
    current = model().eval()
    with torch.no_grad():
        current.g_rec.fill_(0.2)
        current.g_rec_b3.fill_(0.15)
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    reference = current.forward_multi_pass(tokens, num_passes=2)["logits"]
    future = tokens.clone()
    future[0, -1] = (future[0, -1] + 1) % 32
    changed = current.forward_multi_pass(future, num_passes=2)["logits"]
    assert torch.equal(reference[0, :-1], changed[0, :-1])
    row = tokens.clone()
    row[1] = (row[1] + 3) % 32
    isolated = current.forward_multi_pass(row, num_passes=2)["logits"]
    assert torch.equal(reference[0], isolated[0])


def test_incremental_state_has_no_b11_ring_and_exact_cache_geometry():
    current = model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    result = current.incremental_logits(tokens, control="real", return_diagnostics=True)
    assert result["max_cache_lengths"] == (
        1,
        TEST_LENGTH - 1,
        63,
    ) + (TEST_LENGTH - 1,) * 9
    assert result["max_h10_ring_length"] == TEST_LENGTH
    assert result["max_h12_ring_length"] == TEST_LENGTH
    state = result["state"]
    assert not hasattr(state, "h11_ring")
    audit = result["cache_audit"]
    assert audit["passed"] and audit["physical_storage_exact"]
    assert audit["has_b11_ring"] is False


def test_gate_zero_incremental_matches_one_pass_parallel():
    current = model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    parallel = current.forward_pass(tokens)["logits"]
    incremental = current.incremental_logits(
        tokens, control="real", b1_gate_override=0.0, b3_gate_override=0.0
    )["logits"]
    torch.testing.assert_close(incremental, parallel, rtol=2e-5, atol=2e-6)


def test_b3_full_counterfactual_uses_full_cache():
    current = model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    result = current.incremental_logits(tokens, control="b3_full_counterfactual")
    assert result["max_cache_lengths"] == (1,) + (TEST_LENGTH - 1,) * 11
    assert result["cache_audit"]["passed"]
