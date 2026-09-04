"""Small deterministic O1 equivalence, dynamic causality, and state tests."""
import copy
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import smoke_test as support
import experiment_2d7_core as old
import experiment_2d9_core as core

SYMBOLS = support.load_training_symbols()
DEVICE = os.environ.get("EXP2D9_TEST_DEVICE", "cpu")
torch.set_num_threads(1)


def foundation(seed=209):
    torch.manual_seed(seed)
    return SYMBOLS["GPT"](SYMBOLS["GPTConfig"](
        block_size=70, vocab_size=32, n_layer=12, n_head=2,
        n_embd=16, residual_mode="standard")).to(DEVICE)


def model(arm):
    m = core.DynamicGatingGPT(foundation(), arm).to(DEVICE)
    with torch.no_grad():
        for b, value in zip((0, 2, 4), (.03, -.02, .015)):
            m.gate_parameter(b).fill_(value)
    return m


def inputs():
    x = torch.arange(140, device=DEVICE).remainder(32).reshape(2, 70)
    return x, (x + 1).remainder(32)


def incremental(m, x):
    dtype = torch.bfloat16 if torch.is_autocast_enabled(DEVICE) else torch.float32
    state = m.init_incremental_state(x.size(0), dtype=dtype)
    logits = []
    for i in range(x.size(1)):
        current, state = m.incremental_step(x[:, i], state)
        logits.append(current)
    return torch.cat(logits, dim=1), state


def close(a, b):
    torch.testing.assert_close(a.float(), b.float(), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("mode", ["parallel", "incremental"])
def test_zero_exact_o1_and_dynamic(bf16, mode):
    s, d = model("S").eval(), model("D").eval()
    reference = old.BoundaryAlignmentGPT(foundation(), "O").eval()
    reference.load_state_dict(s.state_dict(), strict=True)
    d.load_state_dict(s.state_dict(), strict=False)
    x, y = inputs()
    captures = []
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=bf16):
        for m in (reference, s, d):
            projected = []
            handles = [m.base.transformer.h[b].attn.c_proj.register_forward_pre_hook(
                lambda module, args: projected.append(args[0].detach().clone())) for b in (0, 2, 4)]
            if mode == "parallel":
                result = m.forward_multi_pass(x, targets=y, num_passes=2)
                logits, loss = result["logits"], result["loss"]
            else:
                logits, _ = incremental(m, x)
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            for h in handles:
                h.remove()
            captures.append((logits, loss, projected))
    for candidate in captures[1:]:
        close(candidate[0], captures[0][0])
        assert abs(float(candidate[1]) - float(captures[0][1])) <= 1e-6
        for a, b in zip(candidate[2], captures[0][2]):
            close(a, b)
    h = torch.arange(2 * 70 * 16, device=DEVICE).reshape(2, 70, 16).float() / 100
    for b in (0, 2, 4):
        gate = d.intrinsic_gate(h, b)
        assert torch.equal(gate.coefficient, d.gate_parameter(b).tanh().expand(2, 70, 1))
        ref = torch.empty((2, 2, 70, 8), device=DEVICE, dtype=torch.bfloat16 if bf16 else torch.float32)
        assert torch.equal(d._gate_coefficient(b, ref, gate),
                           s._gate_coefficient(b, ref, None).expand(2, 1, 70, 1))


@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("mode", ["parallel", "incremental"])
def test_future_suffix_and_row_isolation(nonzero, mode):
    d = model("D").eval()
    if nonzero:
        with torch.no_grad():
            for name in core.W_NAMES.values():
                getattr(d, name).copy_(torch.linspace(-.05, .07, 16, device=DEVICE))
    x, _ = inputs()
    suffix = x.clone(); suffix[:, 66:] = (suffix[:, 66:] + 7) % 32
    row = x.clone(); row[1] = (row[1] + 9) % 32
    def run(tokens):
        if mode == "parallel":
            return d.forward_multi_pass(tokens, num_passes=3)["logits"]
        return incremental(d, tokens)[0]
    with torch.no_grad():
        baseline, altered_suffix, altered_row = run(x), run(suffix), run(row)
    close(baseline[:, :66], altered_suffix[:, :66])
    close(baseline[0], altered_row[0])


def test_attached_gate_formula_and_first_backward():
    d = model("D").train()
    h = torch.randn(2, 70, 16, device=DEVICE, requires_grad=True)
    with torch.no_grad():
        d.w_B1.copy_(torch.linspace(-.05, .07, 16, device=DEVICE))
    gate = d.intrinsic_gate(h, 0)
    expected = torch.tanh(d.g_rec.float() + (h.float() * torch.rsqrt(h.float().square().mean(-1, keepdim=True) + 1e-5) * d.w_B1).sum(-1, keepdim=True))
    close(gate.coefficient, expected)
    gate.coefficient.sum().backward()
    assert h.grad is not None and h.grad.isfinite().all() and h.grad.abs().sum() > 0
    d = model("D").train()
    x, y = inputs()
    d.forward_multi_pass(x, targets=y, num_passes=2, activation_checkpointing=True)["loss"].backward()
    for name in core.W_NAMES.values():
        g = getattr(d, name).grad
        assert g is not None and g.isfinite().all() and g.abs().sum() > 0
    assert d.g_rec_b6.grad is None
    assert d.base.transformer.wte.weight is d.base.lm_head.weight


def test_staticized_roundtrip_and_geometry():
    d = model("D").eval()
    with torch.no_grad():
        for name in core.W_NAMES.values():
            getattr(d, name).fill_(.005)
    before = copy.deepcopy(d.state_dict())
    x, _ = inputs()
    with torch.no_grad():
        real, _ = incremental(d, x)
        d.set_gate_mode("staticized")
        staticized, state = incremental(d, x)
        s = model("S").eval()
        s.load_state_dict({n: v for n, v in d.state_dict().items() if n not in core.W_NAMES.values()}, strict=True)
        expected, _ = incremental(s, x)
        close(staticized, expected)
        d.set_gate_mode("real")
        again, _ = incremental(d, x)
        close(real, again)
    assert all(torch.equal(v, before[n]) for n, v in d.state_dict().items())
    assert not hasattr(state, "h7_ring") and d._b6_recurrent_branch_calls == 0
    assert d._last_b6_local_capacity == 69
    q, k = torch.arange(1100)[:, None], torch.arange(1100)[None, :]
    for b, minimum in ((0, 1), (2, 31), (4, 63)):
        assert torch.equal(d.recurrent_mask(b, 1100, 1100, "cpu"), ((q-k) >= minimum) & ((q-k) <= 1023))
        assert torch.equal(d.local_mask(b, 70, "cpu"), ((q[:70]-k[:, :70]) >= 0) & ((q[:70]-k[:, :70]) < {0: 2, 2: 32, 4: 64}[b]))
