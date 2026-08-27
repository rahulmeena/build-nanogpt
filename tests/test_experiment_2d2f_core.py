import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import smoke_test as support  # noqa: E402
import experiment_2d2d_core as source_core  # noqa: E402
import experiment_2d2f_core as core  # noqa: E402

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


def test_parameter_inventory_replaces_b2_gate_with_fresh_b3_gate():
    source = source_core.RecurrentKVGPT(base_model())
    target = tiny_model()
    source_rows = dict(source.named_parameters())
    target_rows = dict(target.named_parameters())
    assert "g_rec_b2" in source_rows
    assert "g_rec_b2" not in target_rows
    assert target_rows["g_rec_b3"].shape == torch.Size([])
    assert set(target_rows) - {"g_rec_b3"} == set(source_rows) - {"g_rec_b2"}
    assert sum(p.numel() for p in target.parameters()) == sum(
        p.numel() for p in source.parameters()
    )


def test_only_b1_and_b3_recurrent_banks_exist_and_partition_time():
    model = tiny_model().eval()
    states = torch.randn(2, TEST_LENGTH, model.config.n_embd)
    assert not hasattr(model, "build_recurrent_bank_b2")
    assert not hasattr(model, "project_recurrent_kv_b2")
    banks = (
        (model.build_recurrent_bank(states), model.local_mask(TEST_LENGTH, states.device), 2),
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
    b2_local = model.b2_local_mask(TEST_LENGTH, states.device)
    for t in range(TEST_LENGTH):
        assert torch.where(b2_local[t])[0].tolist() == list(range(max(0, t - 31), t + 1))


def test_b3_gate_zero_identity_and_b2_has_no_recurrent_diagnostics():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    first = model.forward_pass(tokens)
    without = model.forward_pass(
        tokens, b1_recurrent_source=first["h12"], b3_gate_override=0.0
    )["logits"]
    active = model.forward_pass(
        tokens, b1_recurrent_source=first["h12"],
        b3_recurrent_source=first["h10"], b3_gate_override=0.0,
        return_diagnostics=True,
    )
    assert torch.equal(without, active["logits"])
    assert active["diagnostics"]["b2"]["recurrent_path_present"] is False
    assert "recurrent_attention_weights" not in active["diagnostics"]["b2"]
    for link, first_valid in (("b1", 2), ("b3", 64)):
        diagnostics = active["diagnostics"][link]
        recurrent = diagnostics["recurrent_attention_weights"]
        mask = diagnostics["recurrent_valid_mask"]
        assert recurrent.masked_select(~mask.view(1, 1, TEST_LENGTH, TEST_LENGTH)).count_nonzero() == 0
        assert recurrent[:, :, :first_valid].count_nonzero() == 0


def test_b1_and_b3_writer_paths_are_attached():
    model = tiny_model().train()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    first = model.forward_pass(tokens)
    for source_name, argument, gate_argument, lag in (
        ("h12", "b1_recurrent_source", "b1_gate_override", 2),
        ("h10", "b3_recurrent_source", "b3_gate_override", 64),
    ):
        source = first[source_name]
        kwargs = {
            "b1_recurrent_source": first["h12"].detach(),
            "b3_recurrent_source": first["h10"].detach(),
            "b1_gate_override": 0.0,
            "b3_gate_override": 0.0,
            argument: source,
            gate_argument: 0.2,
        }
        second = model.forward_pass(tokens, **kwargs)
        gradient = torch.autograd.grad(
            second["logits"][:, -1].square().sum(), source, retain_graph=True
        )[0]
        assert torch.isfinite(gradient).all()
        assert gradient[:, : TEST_LENGTH - lag].count_nonzero() > 0
        assert gradient[:, TEST_LENGTH - lag :].count_nonzero() == 0


def test_incremental_cache_has_no_b11_ring():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    result = model.incremental_logits(tokens, control="all_real", return_diagnostics=True)
    assert result["max_cache_lengths"] == (1, 31, 63) + (TEST_LENGTH - 1,) * 9
    assert result["max_h10_ring_length"] == TEST_LENGTH
    assert result["max_h12_ring_length"] == TEST_LENGTH
    assert result["b11_recurrent_ring_present"] is False
    assert not hasattr(result["state"], "h11_ring")
    audit = result["cache_audit"]
    assert audit["passed"] and audit["physical_storage_exact"]
    assert audit["b11_recurrent_ring_present"] is False
    assert audit["b1_historical_kv"] == 1
    assert audit["b2_historical_kv"] == 31
    assert audit["b3_historical_kv"] == 63
    assert result["diagnostics"][-1]["b2"]["recurrent_path_present"] is False


def test_gate_zero_incremental_matches_one_pass_parallel():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    parallel = model.forward_pass(tokens)["logits"]
    incremental = model.incremental_logits(
        tokens, control="all_real", b1_gate_override=0.0, b3_gate_override=0.0
    )["logits"]
    torch.testing.assert_close(incremental, parallel, rtol=2e-5, atol=2e-6)
