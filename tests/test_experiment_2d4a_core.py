import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d3a_core as fixed_core  # noqa: E402
import experiment_2d4a_core as routed_core  # noqa: E402
import smoke_test as support  # noqa: E402

SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]


torch.set_num_threads(1)
TEST_LENGTH = 72


def base_model(seed=83):
    torch.manual_seed(seed)
    return GPT(GPTConfig(
        block_size=TEST_LENGTH,
        vocab_size=64,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="standard",
    ))


def sibling_models(seed=83):
    fixed = fixed_core.AlternatingIntegrationRecurrentPyramidGPT(base_model(seed))
    routed = routed_core.RoutedRecurrentPyramidGPT(fixed.base)
    routed.g_rec = fixed.g_rec
    routed.g_rec_b3 = fixed.g_rec_b3
    routed.g_rec_b5 = fixed.g_rec_b5
    routed.g_rec_b6 = fixed.g_rec_b6
    return fixed, routed


def test_candidate_order_and_parameter_inventory_are_exact():
    assert routed_core.CANDIDATE_BLOCKS == {
        0: tuple(range(1, 12)),
        2: tuple(range(3, 12)),
        4: tuple(range(5, 12)),
        5: tuple(range(6, 12)),
    }
    assert routed_core.BASELINE_BLOCKS == {0: 11, 2: 9, 4: 7, 5: 6}
    fixed, routed = sibling_models()
    fixed_names = set(dict(fixed.named_parameters()))
    routed_names = set(dict(routed.named_parameters()))
    added = sorted(routed_names - fixed_names)
    assert added == sorted(routed_core.expected_router_parameter_names())
    assert len(added) == 12
    assert sum(dict(routed.named_parameters())[name].numel() for name in added) == 68


def test_zero_route_is_exact_fixed_identity_parallel():
    fixed, routed = sibling_models()
    fixed.eval()
    routed.eval()
    tokens = torch.randint(0, 64, (2, TEST_LENGTH))
    with torch.no_grad():
        fixed_first = fixed.forward_pass(tokens)
        routed_first = routed.forward_pass(tokens)
        fixed_multi = fixed.forward_multi_pass(tokens, num_passes=2)
        routed_multi = routed.forward_multi_pass(tokens, num_passes=2)
    assert torch.equal(fixed_first["logits"], routed_first["logits"])
    assert torch.equal(routed_first["m_b1"], routed_first["h12"])
    assert torch.equal(routed_first["m_b3"], routed_first["h10"])
    assert torch.equal(routed_first["m_b5"], routed_first["h8"])
    assert torch.equal(routed_first["m_b6"], routed_first["h7"])
    assert torch.equal(fixed_multi["logits"], routed_multi["logits"])


def test_zero_route_is_exact_fixed_identity_incremental():
    fixed, routed = sibling_models()
    fixed.eval()
    routed.eval()
    tokens = torch.randint(0, 64, (2, TEST_LENGTH))
    with torch.no_grad():
        fixed_result = fixed.incremental_logits(tokens)
        routed_result = routed.incremental_logits(tokens)
    assert torch.equal(fixed_result["logits"], routed_result["logits"])
    def state_bytes(audit):
        cache = sum(
            0 if row["key"] is None else row["key"]["expected_bytes"] + row["value"]["expected_bytes"]
            for row in audit["cache_storage"]
        )
        rings = sum(row["expected_bytes"] for row in audit["ring_storage"].values())
        return cache + rings
    assert state_bytes(fixed_result["cache_audit"]) == state_bytes(routed_result["cache_audit"])
    assert fixed_result["cache_audit"]["ring_lengths"] == routed_result["cache_audit"]["ring_lengths"]


def test_effective_coefficients_sum_to_one():
    _, routed = sibling_models()
    routed.eval()
    tokens = torch.randint(0, 64, (2, TEST_LENGTH))
    with torch.no_grad():
        result = routed.forward_pass(tokens, return_route_diagnostics=True)
    for row in result["route_diagnostics"].values():
        total = row["effective_coefficients"].sum(dim=0)
        torch.testing.assert_close(total, torch.ones_like(total))


def test_staged_router_gradients_follow_zero_effect_parameterization():
    for count in (11, 9, 7, 6):
        router = routed_core.RecurrentSourceDepthRouter(
            8, tuple(range(count)), count - 1
        ).train()
        values = [torch.randn(2, 5, 8, requires_grad=True) for _ in range(count)]
        router(values).square().mean().backward()
        assert router.gate.grad is not None and torch.isfinite(router.gate.grad)
        assert router.gate.grad.abs().item() > 0
        assert router.query.grad is not None and router.query.grad.count_nonzero().item() == 0
        assert router.norm.weight.grad is not None and router.norm.weight.grad.count_nonzero().item() == 0
        with torch.no_grad():
            router.gate.add_(-0.1 * router.gate.grad)
        router.zero_grad(set_to_none=True)
        router(values).square().mean().backward()
        assert router.query.grad.count_nonzero().item() > 0
        assert router.norm.weight.grad.count_nonzero().item() == 0
        with torch.no_grad():
            router.query.add_(-0.1 * router.query.grad)
        router.zero_grad(set_to_none=True)
        router(values).square().mean().backward()
        assert router.norm.weight.grad.count_nonzero().item() > 0
        assert all(torch.isfinite(parameter.grad).all() for parameter in router.parameters())


def test_suffix_and_row_mutation_do_not_change_routed_prefix():
    _, routed = sibling_models()
    routed.eval()
    tokens = torch.randint(0, 64, (2, TEST_LENGTH))
    suffix = tokens.clone()
    suffix[:, 48:] = torch.randint(0, 64, suffix[:, 48:].shape)
    row = tokens.clone()
    row[1] = torch.randint(0, 64, row[1].shape)
    with torch.no_grad():
        left = routed.forward_multi_pass(tokens, num_passes=2)["logits"]
        changed_suffix = routed.forward_multi_pass(suffix, num_passes=2)["logits"]
        changed_row = routed.forward_multi_pass(row, num_passes=2)["logits"]
    assert torch.equal(left[:, :48], changed_suffix[:, :48])
    assert torch.equal(left[0], changed_row[0])
