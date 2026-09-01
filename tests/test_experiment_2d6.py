import ast
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d3a_core as fixed_core  # noqa: E402
import experiment_2d6_core as core  # noqa: E402
import smoke_test as support  # noqa: E402


SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)
# Exercise every retained recurrent link; B8->B5 first becomes eligible at 64.
TEST_LENGTH = 70


def base_model(seed=206):
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


def new_model(seed=206):
    return core.B6NativeNoB7RecurrenceGPT(base_model(seed))


def fixed_model(seed=206):
    return fixed_core.AlternatingIntegrationRecurrentPyramidGPT(base_model(seed))


def inventory(model):
    return [
        (name, tuple(parameter.shape), parameter.numel())
        for name, parameter in model.named_parameters()
    ]


def test_sources_config_and_tests_parse():
    for path in (
        ROOT / "scripts" / "experiment_2d6.py",
        ROOT / "scripts" / "experiment_2d6_core.py",
        Path(__file__),
    ):
        ast.parse(path.read_text())
    config = json.loads(
        (ROOT / "configs" / "exp2d6_b6_w1024_no_b7_recurrence_matched_100m.json").read_text()
    )
    assert config["training"]["local_updates"] == 191
    assert config["training"]["entry_point_refuses_update_192"] is True
    assert config["architecture"]["local_windows"]["B6"] == 1024
    assert config["architecture"]["b7_to_b6_link"] is False
    assert config["architecture"]["b7_ring"] is False
    assert config["statistics"]["delta_ce"] == 0.0001
    assert config["statistics"]["evaluation_set_label"] == "reused sealed matched panel"


def test_parameter_and_state_dict_compatibility_is_exact():
    candidate = new_model()
    accepted = fixed_model()
    assert inventory(candidate) == inventory(accepted)
    assert list(candidate.state_dict()) == list(accepted.state_dict())
    assert all(
        torch.equal(value, accepted.state_dict()[name])
        for name, value in candidate.state_dict().items()
    )
    assert "g_rec_b6" in dict(candidate.named_parameters())


def test_parallel_execution_is_exact_fixed_b6_full_native_counterfactual():
    candidate = new_model().eval()
    accepted = fixed_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    with torch.no_grad():
        new = candidate.forward_multi_pass(tokens, num_passes=2)["logits"]
        reference = accepted.forward_multi_pass(
            tokens, num_passes=2, full_counterfactual_blocks=(5,)
        )["logits"]
    assert torch.equal(new, reference)
    assert candidate._b6_recurrent_branch_calls == 0
    assert all(candidate._active_special_branch_calls[index] > 0 for index in (0, 2, 4))


def test_incremental_state_has_b6_native_cache_and_no_h7_ring():
    candidate = new_model().eval()
    accepted = fixed_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    with torch.no_grad():
        new = candidate.incremental_logits(tokens)
        reference = accepted.incremental_logits(tokens, control="b6_full_native")
    torch.testing.assert_close(new["logits"], reference["logits"], rtol=2e-5, atol=2e-6)
    assert not hasattr(new["state"], "h7_ring")
    assert new["cache_audit"]["b7_ring_present"] is False
    assert new["max_cache_lengths"][5] == TEST_LENGTH - 1
    assert new["cache_audit"]["ring_lengths"] == {
        "h8": TEST_LENGTH,
        "h10": TEST_LENGTH,
        "h12": TEST_LENGTH,
    }
    assert new["cache_audit"]["passed"]


def test_dormant_b6_gate_has_no_gradient_but_active_gates_do():
    candidate = new_model().train()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    targets = torch.randint(0, 32, (2, TEST_LENGTH))
    result = candidate.forward_multi_pass(tokens, targets=targets, num_passes=2)
    result["loss"].backward()
    assert candidate.g_rec_b6.grad is None
    for parameter in (candidate.g_rec, candidate.g_rec_b3, candidate.g_rec_b5):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)
        assert torch.count_nonzero(parameter.grad)


def test_b6_recurrent_inputs_and_controls_fail_closed():
    candidate = new_model()
    tokens = torch.randint(0, 32, (1, TEST_LENGTH))
    source = torch.randn(1, TEST_LENGTH, 8)
    with pytest.raises(ValueError):
        candidate.forward_pass(tokens, b6_recurrent_source=source)
    state = candidate.init_incremental_state(1)
    with pytest.raises(ValueError):
        candidate.incremental_step(tokens[:, 0], state, control="b6_off")


def test_manifest_is_exact_and_fingerprinted():
    assert core.ARCHITECTURE_MANIFEST["active_writers"] == {
        "B1": "B12",
        "B3": "B10",
        "B5": "B8",
    }
    assert core.ARCHITECTURE_MANIFEST["b7_to_b6_computational_link"] is False
    assert core.ARCHITECTURE_MANIFEST["b7_recurrent_ring"] is False
    assert core.ARCHITECTURE_MANIFEST["inactive_compatibility_state"]["parameter"] == "g_rec_b6"
    assert len(core.ARCHITECTURE_FINGERPRINT) == 64
    int(core.ARCHITECTURE_FINGERPRINT, 16)
