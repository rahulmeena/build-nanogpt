import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as support  # noqa: E402
import experiment_2d1 as d1  # noqa: E402


SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]


def tiny_model():
    torch.manual_seed(7)
    base = GPT(GPTConfig(
        block_size=16,
        vocab_size=64,
        n_layer=12,
        n_head=4,
        n_embd=16,
        residual_mode="standard",
    ))
    model = d1.TriangleRecurrentGPT(base)
    model.fusion.initialize(0.05, 1.0)
    return model


def test_frozen_schedule_endpoints():
    assert d1.stage_for_update(1)["rho"] == 0.0
    assert d1.stage_for_update(477)["rho"] == 0.5
    assert d1.stage_for_update(478)["stage"] == "B"
    assert d1.stage_for_update(955)["rho"] == 0.75
    assert d1.stage_for_update(1909)["rho"] == 1.0
    assert d1.stage_for_update(4769)["windows"] == d1.TARGET_WINDOWS
    assert d1.learning_rate_fraction(100) == 1.0
    assert d1.learning_rate_fraction(4292) == 1.0
    assert abs(d1.learning_rate_fraction(4769) - 0.1) < 1e-15


def test_rho_zero_is_exact_identity_and_shift_has_no_wraparound():
    model = tiny_model().eval()
    tokens = torch.randint(0, 64, (2, 8))
    windows = (8,) * 12
    with torch.no_grad():
        plain = model.forward_top(tokens, windows)
        dormant = model.forward_top(tokens, windows, previous_top=plain, rho=0.0, prefix_length=0)
    assert torch.equal(plain, dormant)
    shifted = torch.zeros_like(plain)
    shifted[:, 1:] = plain[:, :-1]
    assert shifted[:, 0].count_nonzero() == 0
    assert torch.equal(shifted[:, 1:], plain[:, :-1])


def test_sliding_masks_are_causal_and_exact_width():
    model = tiny_model()
    for window in (1, 2, 4, 8):
        mask = model.sliding_mask(8, window, torch.device("cpu"))
        expected = torch.minimum(torch.arange(8) + 1, torch.tensor(window))
        assert torch.equal(mask.sum(dim=1), expected)
        assert not torch.triu(mask, diagonal=1).any()


def test_temporal_gradients_cross_one_and_two_transitions_without_future_edge():
    model = tiny_model().train()
    tokens = torch.randint(0, 64, (1, 6))
    windows = (6,) * 12
    first = model.forward_top(tokens, windows)
    second = model.forward_top(tokens, windows, previous_top=first, rho=1.0, prefix_length=0)
    loss2 = model.logits_from_top(second)[:, 2].float().sum()
    gradient2 = torch.autograd.grad(loss2, first, retain_graph=True)[0]
    assert gradient2[:, 1].norm() > 0
    assert gradient2[:, 2:].count_nonzero() == 0
    third = model.forward_top(tokens, windows, previous_top=second, rho=1.0, prefix_length=0)
    loss3 = model.logits_from_top(third)[:, 3].float().sum()
    gradient3 = torch.autograd.grad(loss3, first)[0]
    assert gradient3[:, 1].norm() > 0
    assert gradient3[:, 2:].count_nonzero() == 0


def test_incremental_cache_lengths_never_exceed_window_minus_one():
    model = tiny_model().eval()
    tokens = torch.randint(0, 64, (2, 8))
    windows = (2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8)
    state = d1.init_incremental_state(model, 2, windows, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        for position in range(tokens.size(1)):
            _, state = d1.incremental_step(model, tokens[:, position], state, rho=1.0)
            assert d1.incremental_cache_lengths(state) == [
                min(position + 1, window - 1) for window in windows
            ]
