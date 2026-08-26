import sys
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test as support  # noqa: E402
import experiment_2d2a_core as core  # noqa: E402


SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)


def tiny_model(seed=17):
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


def assert_exact_physical_storage(tensor):
    assert tensor.storage_offset() == 0
    assert tensor.is_contiguous()
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def b1_w2_oracle(base, tokens, targets=None):
    """Independent generalized-window oracle for the gate-zero identity test."""
    batch, length = tokens.shape
    channels = base.config.n_embd
    heads = base.config.n_head
    head_size = channels // heads
    positions = torch.arange(length, device=tokens.device)
    residual = base.transformer.wte(tokens) + base.transformer.wpe(positions)
    block1 = base.transformer.h[0]
    normalized = block1.ln_1(residual)
    query, key, value = block1.attn.c_attn(normalized).split(channels, dim=-1)
    query = query.view(batch, length, heads, head_size).transpose(1, 2)
    key = key.view(batch, length, heads, head_size).transpose(1, 2)
    value = value.view(batch, length, heads, head_size).transpose(1, 2)
    receiver = torch.arange(length).view(length, 1)
    source = torch.arange(length).view(1, length)
    mask = (source <= receiver) & (source >= receiver - 1)
    attention = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, is_causal=False
    )
    attention = attention.transpose(1, 2).contiguous().view(batch, length, channels)
    residual = residual + block1.attn.c_proj(attention)
    residual = residual + block1.mlp(block1.ln_2(residual))
    for block in base.transformer.h[1:]:
        residual = block(residual)
    h12 = residual
    top = base.transformer.ln_f(h12)
    logits = base.lm_head(top)
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return {"h12": h12, "top": top, "logits": logits, "loss": loss}


def test_wrapper_adds_exactly_one_scalar_parameter_and_preserves_tying():
    model = tiny_model()
    parent_parameters = tuple(model.base.parameters())
    parent_ids = {id(parameter) for parameter in parent_parameters}
    additions = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) not in parent_ids
    ]
    assert additions == [("g_rec", model.g_rec)]
    assert model.g_rec.shape == torch.Size([])
    assert model.g_rec.item() == 0.0
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        sum(parameter.numel() for parameter in parent_parameters) + 1
    )
    assert model.base.transformer.wte.weight is model.base.lm_head.weight
    assert not any(
        fragment in name.lower()
        for name, _ in model.named_parameters()
        for fragment in ("rec_k", "rec_v", "adapter", "router")
    )


def test_gate_zero_is_exact_b1_w2_oracle_identity_and_returns_raw_h12():
    model = tiny_model().eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    targets = torch.randint(0, model.config.vocab_size, (2, 7))
    source = torch.randn(2, 7, model.config.n_embd)
    with torch.no_grad():
        oracle = b1_w2_oracle(model.base, tokens, targets)
        plain = model.forward_pass(tokens, targets)
        dormant = model.forward_pass(tokens, targets, recurrent_source=source)
    for key in ("h12", "top", "logits", "loss"):
        assert torch.equal(plain[key], oracle[key])
        assert torch.equal(dormant[key], oracle[key])
    assert dormant["raw_h12"] is dormant["h12"]
    assert torch.equal(dormant["top"], model.base.transformer.ln_f(dormant["h12"]))
    mask = model.local_mask(7, torch.device("cpu"))
    assert torch.equal(mask.sum(dim=-1), torch.tensor([1, 2, 2, 2, 2, 2, 2]))
    assert not torch.triu(mask, diagonal=1).any()


def test_recurrent_bank_positions_boundaries_and_probabilities_are_exact():
    model = tiny_model().eval()
    source = torch.arange(2 * 6 * model.config.n_embd, dtype=torch.float32).view(
        2, 6, model.config.n_embd
    )
    bank = model.build_recurrent_bank(source)
    expected_positions = torch.tensor(
        [[-3, -2], [-2, -1], [-1, 0], [0, 1], [1, 2], [2, 3]]
    )
    expected_valid = expected_positions.ge(0)
    assert torch.equal(bank.positions, expected_positions)
    assert torch.equal(bank.valid_mask, expected_valid)
    assert bank.values[:, :2].count_nonzero() == 0
    assert bank.values[:, 2, 0].count_nonzero() == 0
    assert torch.equal(bank.values[:, 2, 1], source[:, 0])
    assert torch.equal(bank.values[:, 3, 0], source[:, 0])
    assert torch.equal(bank.values[:, 3, 1], source[:, 1])

    tokens = torch.randint(0, model.config.vocab_size, (2, 6))
    with torch.no_grad():
        result = model.forward_pass(
            tokens, recurrent_source=source, return_diagnostics=True
        )
    diagnostics = result["diagnostics"]
    weights = diagnostics["recurrent_attention_weights"]
    assert tuple(weights.shape) == (2, model.config.n_head, 6, 2)
    assert weights[:, :, :2].count_nonzero() == 0
    assert weights[:, :, 2, 0].count_nonzero() == 0
    assert torch.equal(weights[:, :, 2, 1], torch.ones_like(weights[:, :, 2, 1]))
    torch.testing.assert_close(
        weights[:, :, 3:].sum(dim=-1),
        torch.ones_like(weights[:, :, 3:, 0]),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(diagnostics["recurrent_output_rms"])


def test_recurrent_projection_is_b1_ln_and_fused_kv_slices_and_cproj_runs_once():
    model = tiny_model().eval()
    source = torch.randn(2, 6, model.config.n_embd)
    bank = model.build_recurrent_bank(source)
    projected_key, projected_value = model.project_recurrent_kv(bank.values)
    block1 = model.base.transformer.h[0]
    normalized = block1.ln_1(bank.values)
    _, expected_key, expected_value = block1.attn.c_attn(normalized).split(
        model.config.n_embd, dim=-1
    )
    batch, length, slots, channels = expected_key.shape
    head_size = channels // model.config.n_head
    expected_key = expected_key.view(
        batch, length, slots, model.config.n_head, head_size
    ).permute(0, 3, 1, 2, 4)
    expected_value = expected_value.view(
        batch, length, slots, model.config.n_head, head_size
    ).permute(0, 3, 1, 2, 4)
    # A sliced GEMM and a wider fused GEMM can choose different CPU kernels;
    # they must nevertheless implement the same learned rows.
    torch.testing.assert_close(projected_key, expected_key, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(projected_value, expected_value, rtol=1e-6, atol=1e-7)

    calls = []
    handle = block1.attn.c_proj.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(1)
    )
    try:
        with torch.no_grad():
            model.forward_pass(
                torch.randint(0, model.config.vocab_size, (2, 6)),
                recurrent_source=source,
            )
    finally:
        handle.remove()
    assert calls == [1]


def test_initial_gate_gradient_and_open_gate_temporal_writer_gradient():
    model = tiny_model().train()
    tokens = torch.randint(0, model.config.vocab_size, (2, 6))
    targets = torch.randint(0, model.config.vocab_size, (2, 6))
    first = model.forward_pass(tokens)
    second = model.forward_pass(tokens, targets, recurrent_source=first["h12"])
    gate_gradient, zero_writer_gradient = torch.autograd.grad(
        second["loss"], (model.g_rec, first["h12"])
    )
    assert torch.isfinite(gate_gradient)
    assert gate_gradient.abs() > 0
    assert torch.isfinite(zero_writer_gradient).all()
    assert zero_writer_gradient.count_nonzero() == 0

    with torch.no_grad():
        model.g_rec.fill_(0.25)
    first = model.forward_pass(tokens)
    second = model.forward_pass(tokens, recurrent_source=first["h12"])
    # A receiver at t=3 may use only h12[0] and h12[1].
    temporal_gradient = torch.autograd.grad(
        second["logits"][:, 3].square().sum(), first["h12"]
    )[0]
    assert torch.isfinite(temporal_gradient).all()
    assert temporal_gradient[:, :2].norm() > 0
    assert temporal_gradient[:, 2:].count_nonzero() == 0


def test_multi_pass_is_attached_uses_exact_ce_weights_and_checkpointing():
    model = tiny_model().train()
    with torch.no_grad():
        model.g_rec.fill_(0.2)
    tokens = torch.randint(0, model.config.vocab_size, (1, 5))
    targets = torch.randint(0, model.config.vocab_size, (1, 5))
    result = model.forward_multi_pass(tokens, targets, num_passes=3)
    assert result["pass_weights"] == (0.20, 0.40, 0.40)
    expected = sum(
        weight * loss
        for weight, loss in zip(result["pass_weights"], result["pass_losses"])
    )
    assert torch.equal(result["loss"], expected)
    temporal = torch.autograd.grad(
        result["passes"][1]["loss"], result["passes"][0]["h12"], retain_graph=True
    )[0]
    assert temporal.norm() > 0

    checkpointed = model.forward_multi_pass(
        tokens,
        targets,
        num_passes=2,
        activation_checkpointing=True,
    )
    direct = model.forward_multi_pass(tokens, targets, num_passes=2)
    torch.testing.assert_close(checkpointed["logits"], direct["logits"], rtol=0, atol=0)
    checkpointed["loss"].backward()
    assert model.g_rec.grad is not None and torch.isfinite(model.g_rec.grad)


def test_multi_pass_future_causality_and_batch_row_isolation():
    model = tiny_model().eval()
    with torch.no_grad():
        model.g_rec.fill_(0.3)
    tokens = torch.randint(0, model.config.vocab_size, (2, 6))
    with torch.no_grad():
        reference = model.forward_multi_pass(tokens, num_passes=2)["logits"]

        changed_future = tokens.clone()
        changed_future[0, 5] = (changed_future[0, 5] + 1) % model.config.vocab_size
        future_result = model.forward_multi_pass(changed_future, num_passes=2)["logits"]
        assert torch.equal(reference[0, :5], future_result[0, :5])

        changed_row = tokens.clone()
        changed_row[1] = (changed_row[1] + 3) % model.config.vocab_size
        row_result = model.forward_multi_pass(changed_row, num_passes=2)["logits"]
        assert torch.equal(reference[0], row_result[0])


def test_incremental_gate_zero_kernel_equivalence_exact_slots_and_cache_bounds():
    model = tiny_model().eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    permutation = torch.tensor([1, 0])
    with torch.no_grad():
        parallel = model.forward_pass(tokens)["logits"]
        plain = model.incremental_logits(tokens, control="plain", return_diagnostics=True)
        real = model.incremental_logits(tokens, control="real", return_diagnostics=True)
        shuffled = model.incremental_logits(
            tokens,
            control="shuffled",
            recurrent_permutation=permutation,
            return_diagnostics=True,
        )
    for result in (plain, real, shuffled):
        torch.testing.assert_close(result["logits"], parallel, rtol=2e-5, atol=2e-6)
        assert result["max_cache_lengths"] == (1,) + (7,) * 11
        assert result["max_h12_ring_length"] == 3
        assert result["cache_audit"]["passed"]
        assert result["cache_audit"]["b1_historical_kv"] == 1
        assert max(result["cache_audit"]["b2_b12_historical_kv"]) <= 1023
        assert result["cache_audit"]["h12_ring_length"] == 3
        assert result["cache_audit"]["h12_ring_positions"] == (5, 6, 7)
        assert result["cache_audit"]["physical_storage_exact"]

        # The final token forces eviction in B1, every upper-layer cache, and
        # the H12 ring.  Audit backing allocations, not only logical views.
        state = result["state"]
        head_size = model.config.n_embd // model.config.n_head
        for block_index, cache in enumerate(state.caches):
            expected_length = 1 if block_index == 0 else 7
            expected_shape = (
                tokens.size(0), model.config.n_head, expected_length, head_size
            )
            assert tuple(cache.key.shape) == expected_shape
            assert tuple(cache.value.shape) == expected_shape
            assert_exact_physical_storage(cache.key)
            assert_exact_physical_storage(cache.value)
        assert tuple(state.h12_ring.shape) == (
            tokens.size(0), core.RECURRENT_RING_CAPACITY, model.config.n_embd
        )
        assert_exact_physical_storage(state.h12_ring)

    real_diagnostics = real["diagnostics"]
    assert torch.equal(
        real_diagnostics[0]["recurrent_valid_mask"], torch.tensor([[False, False]])
    )
    assert torch.equal(
        real_diagnostics[1]["recurrent_valid_mask"], torch.tensor([[False, False]])
    )
    assert torch.equal(
        real_diagnostics[2]["recurrent_valid_mask"], torch.tensor([[False, True]])
    )
    assert torch.equal(
        real_diagnostics[3]["recurrent_valid_mask"], torch.tensor([[True, True]])
    )
    assert real_diagnostics[2]["recurrent_attention_weights"][:, :, :, 0].count_nonzero() == 0
    assert torch.equal(
        real_diagnostics[2]["recurrent_attention_weights"][:, :, :, 1],
        torch.ones_like(
            real_diagnostics[2]["recurrent_attention_weights"][:, :, :, 1]
        ),
    )
