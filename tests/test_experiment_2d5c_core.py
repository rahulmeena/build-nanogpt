import ast
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d3a_core as fixed_core  # noqa: E402
import experiment_2d5c_core as core  # noqa: E402
import smoke_test as support  # noqa: E402


SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)

TEST_LENGTH = 12
SPECIAL_BLOCKS = (0, 2, 4, 5)


def base_model(seed=205, length=TEST_LENGTH):
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(
            block_size=length,
            vocab_size=32,
            n_layer=12,
            n_head=2,
            n_embd=8,
            residual_mode="standard",
        )
    )


def c_model(seed=205):
    return core.FixedWriterB3B5W2RepresentationPressureGPT(base_model(seed))


def fixed_control_model(seed=205):
    return core.FixedControlEvaluationGPT(base_model(seed))


def accepted_fixed_model(seed=205):
    return fixed_core.AlternatingIntegrationRecurrentPyramidGPT(base_model(seed))


def inventory(model):
    return [(name, tuple(value.shape), value.numel()) for name, value in model.named_parameters()]


def test_source_and_test_files_parse():
    for path in (
        ROOT / "scripts" / "experiment_2d5c_core.py",
        Path(__file__),
    ):
        ast.parse(path.read_text())


def test_constants_are_exact_immutable_fingerprint_inputs():
    assert core.EXPERIMENT == "2D5C"
    assert core.EXPECTED_PARAMETER_COUNT == 124_475_908
    assert core.INCREMENTAL_CONTROLS == (
        "all_real",
        "b3_off",
        "b3_shuffled",
        "b5_off",
        "b5_shuffled",
        "b3_b5_off",
        "b3_b5_shuffled",
    )
    assert dict(core.FIXED_WRITER_SOURCES) == {0: 11, 2: 9, 4: 7, 5: 6}
    assert dict(core.C_LOCAL_WINDOWS) == {0: 2, 2: 2, 4: 2, 5: 512}
    assert dict(core.FIXED_CONTROL_LOCAL_WINDOWS) == {0: 2, 2: 32, 4: 64, 5: 512}
    with pytest.raises(TypeError):
        core.C_LOCAL_WINDOWS[2] = 32
    with pytest.raises(TypeError):
        core.FIXED_WRITER_SOURCES[2] = 8
    assert core.ARCHITECTURE_FINGERPRINT_INPUTS == (
        ("experiment", "2D5C"),
        ("model_weight_lineage", "accepted-2d3a-fixed-writer"),
        (
            "checkpoint_sha256",
            "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b",
        ),
        ("geometry", "b3-w2-b5-w2-representation-pressure"),
        ("block_geometry", core.C_BLOCK_GEOMETRY),
        ("controls", core.INCREMENTAL_CONTROLS),
        ("recurrent_ring_capacity", 1023),
        ("expected_parameter_count", 124_475_908),
    )
    assert dict(core.FIXED_WRITERS) == {
        "B1": "B12",
        "B3": "B10",
        "B5": "B8",
        "B6": "B7",
    }
    with pytest.raises(TypeError):
        core.FIXED_WRITERS["B3"] = "B9"
    assert core.FixedWriterW2PressureGPT is core.FixedWriterB3B5W2RepresentationPressureGPT


@pytest.mark.parametrize(
    "constructor", (c_model, fixed_control_model)
)
def test_state_dict_parameter_names_shapes_and_numel_are_exact_fixed(constructor):
    candidate = constructor()
    accepted = accepted_fixed_model()
    assert inventory(candidate) == inventory(accepted)
    assert list(candidate.state_dict()) == list(accepted.state_dict())
    assert sum(value.numel() for value in candidate.parameters()) == sum(
        value.numel() for value in accepted.parameters()
    )
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, accepted.state_dict()[name]), name


def test_public_driver_contract_has_exact_family_fingerprints_and_control_sets():
    c = c_model()
    fixed = fixed_control_model()
    assert c.architecture_fingerprint() == core.ARCHITECTURE_FINGERPRINT_C
    assert fixed.architecture_fingerprint() == core.ARCHITECTURE_FINGERPRINT_FIXED
    assert c.control_sets("b3_b5_shuffled") == (set(), {2, 4})
    assert fixed.control_sets("b3_b5_off") == ({2, 4}, set())
    with pytest.raises(ValueError):
        c.control_sets("b1_off")


def test_driver_style_construction_preserves_every_parameter_object_identity():
    fixed = fixed_core.AlternatingIntegrationRecurrentPyramidGPT(base_model())
    candidate = core.FixedWriterW2PressureGPT(fixed.base)
    candidate.g_rec = fixed.g_rec
    candidate.g_rec_b3 = fixed.g_rec_b3
    candidate.g_rec_b5 = fixed.g_rec_b5
    candidate.g_rec_b6 = fixed.g_rec_b6
    fixed_named = dict(fixed.named_parameters())
    candidate_named = dict(candidate.named_parameters())
    assert fixed_named.keys() == candidate_named.keys()
    assert all(fixed_named[name] is candidate_named[name] for name in fixed_named)


def _assert_geometry(model, windows):
    length = 1024
    query = torch.arange(length).view(length, 1)
    source = torch.arange(length).view(1, length)
    causal = source <= query
    for block_index in SPECIAL_BLOCKS:
        local = model.local_mask(block_index, length, torch.device("cpu"))
        recurrent = model.recurrent_mask(
            block_index, length, length, torch.device("cpu")
        )
        assert not (local & recurrent).any(), block_index
        assert torch.equal(local | recurrent, causal), block_index
        lag = query - source
        assert torch.equal(local, (lag >= 0) & (lag < windows[block_index]))
        assert torch.equal(
            recurrent,
            (lag >= windows[block_index]) & (lag <= core.RECURRENT_MAX_LAG),
        )


def test_c_geometry_has_exact_full_coverage_without_overlap():
    current = c_model().eval()
    _assert_geometry(current, {0: 2, 2: 2, 4: 2, 5: 512})
    assert torch.equal(
        current.recurrent_mask(2, 5, 5, torch.device("cpu")),
        core.FixedWriterB3B5W2RepresentationPressureGPT.recurrent_mask(
            2, 5, 5, torch.device("cpu")
        ),
    )
    assert torch.where(current.local_mask(2, 5, torch.device("cpu"))[4])[0].tolist() == [3, 4]
    assert torch.where(current.recurrent_mask(2, 5, 5, torch.device("cpu"))[4])[0].tolist() == [0, 1, 2]
    assert torch.where(current.local_mask(4, 5, torch.device("cpu"))[4])[0].tolist() == [3, 4]
    assert torch.where(current.recurrent_mask(4, 5, 5, torch.device("cpu"))[4])[0].tolist() == [0, 1, 2]


def test_fixed_control_preserves_original_b3_b5_geometry_exactly():
    current = fixed_control_model().eval()
    _assert_geometry(current, {0: 2, 2: 32, 4: 64, 5: 512})
    b3 = current.recurrent_mask(2, 65, 65, torch.device("cpu"))
    b5 = current.recurrent_mask(4, 65, 65, torch.device("cpu"))
    assert torch.where(b3[31])[0].numel() == 0
    assert torch.where(b3[32])[0].tolist() == [0]
    assert torch.where(b5[63])[0].numel() == 0
    assert torch.where(b5[64])[0].tolist() == [0]


@pytest.mark.parametrize(
    "constructor,expected_windows",
    (
        (c_model, {0: 2, 2: 2, 4: 2, 5: 512}),
        (fixed_control_model, {0: 2, 2: 32, 4: 64, 5: 512}),
    ),
)
def test_parallel_and_incremental_geometry_paths_agree_at_zero_gates(
    constructor, expected_windows
):
    current = constructor().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    with torch.no_grad():
        parallel = current.forward_pass(tokens)["logits"]
        incremental = current.incremental_logits(tokens, control="all_real")
    torch.testing.assert_close(
        incremental["logits"], parallel, rtol=2e-5, atol=2e-6
    )
    expected = tuple(
        min(TEST_LENGTH, expected_windows[index] - 1)
        if index in expected_windows
        else TEST_LENGTH - 1
        for index in range(12)
    )
    assert incremental["max_cache_lengths"] == expected
    assert incremental["cache_audit"]["cache_limits"] == [
        expected_windows[index] - 1
        if index in expected_windows
        else TEST_LENGTH - 1
        for index in range(12)
    ]
    assert incremental["cache_audit"]["passed"]
    assert incremental["cache_audit"]["physical_storage_exact"]


def test_fixed_control_is_exact_accepted_fixed_execution_identity():
    accepted = accepted_fixed_model().eval()
    evaluator = fixed_control_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    with torch.no_grad():
        accepted_parallel = accepted.forward_multi_pass(tokens, num_passes=2)["logits"]
        evaluator_parallel = evaluator.forward_multi_pass(tokens, num_passes=2)["logits"]
        accepted_incremental = accepted.incremental_logits(tokens, control="all_real")["logits"]
        evaluator_incremental = evaluator.incremental_logits(tokens, control="all_real")["logits"]
    assert torch.equal(accepted_parallel, evaluator_parallel)
    assert torch.equal(accepted_incremental, evaluator_incremental)


def test_c_cache_is_w2_at_b3_b5_and_physical_audit_counts_unique_bytes():
    current = c_model().eval()
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    with torch.no_grad():
        result = current.incremental_logits(
            tokens, control="all_real", return_diagnostics=True
        )
    audit = result["cache_audit"]
    assert result["max_cache_lengths"] == (
        1,
        TEST_LENGTH - 1,
        1,
        TEST_LENGTH - 1,
        1,
        TEST_LENGTH,
        TEST_LENGTH - 1,
        TEST_LENGTH - 1,
        TEST_LENGTH - 1,
        TEST_LENGTH - 1,
        TEST_LENGTH - 1,
        TEST_LENGTH - 1,
    )
    assert audit["b1_historical_local_kv"] == 1
    assert audit["b3_historical_local_kv"] == 1
    assert audit["b5_historical_local_kv"] == 1
    assert audit["b6_historical_local_kv"] == TEST_LENGTH
    assert audit["logical_payload_bytes"] == audit["actual_unique_storage_bytes"]
    assert audit["storage_alias_free"]
    assert audit["physical_storage_exact"] and audit["passed"]
    for position, row in enumerate(result["diagnostics"]):
        for name in ("b3", "b5"):
            recurrent_positions = row["links"][name]["recurrent_positions"]
            assert recurrent_positions.tolist() == [list(range(max(0, position - 1023), max(0, position - 1)))]


def test_incremental_diagnostics_expose_exact_positions_weights_and_branch_outputs():
    current = c_model().eval()
    with torch.no_grad():
        current.g_rec_b3.fill_(0.2)
        current.g_rec_b5.fill_(0.2)
    tokens = torch.randint(0, 32, (2, 4))
    plain_state = current.init_incremental_state(2)
    diagnostic_state = current.init_incremental_state(2)
    plain_logits = []
    diagnostic_logits = []
    rows = []
    with torch.no_grad():
        for position in range(tokens.size(1)):
            logits, plain_state = current.incremental_step(
                tokens[:, position], plain_state
            )
            plain_logits.append(logits)
            logits, diagnostic_state, row = current.incremental_step(
                tokens[:, position],
                diagnostic_state,
                return_diagnostics=True,
                diagnostic_attention_weights=True,
            )
            diagnostic_logits.append(logits)
            rows.append(row)
    assert torch.equal(torch.cat(plain_logits, 1), torch.cat(diagnostic_logits, 1))
    assert not diagnostic_state.h8_ring.requires_grad
    row = rows[2]
    for name, source_block in (("b3", 10), ("b5", 8)):
        link = row["links"][name]
        assert link["source_block"] == source_block
        assert link["query_position"] == 2
        assert link["query_positions"].tolist() == [[2]]
        assert link["local_positions"].tolist() == [[1, 2]]
        assert link["recurrent_positions"].tolist() == [[0]]
        assert link["local_attention_weights"].shape == (2, 2, 1, 2)
        assert link["recurrent_attention_weights"].shape == (2, 2, 1, 1)
        torch.testing.assert_close(
            link["local_attention_weights"].sum(-1),
            torch.ones((2, 2, 1)),
        )
        torch.testing.assert_close(
            link["recurrent_attention_weights"].sum(-1),
            torch.ones((2, 2, 1)),
        )
        assert link["local_pre_c_proj"].shape == (2, 2, 1, 4)
        assert link["recurrent_pre_c_proj"].shape == (2, 2, 1, 4)
        assert link["gated_recurrent_pre_c_proj"].shape == (2, 2, 1, 4)
        assert link["local_post_c_proj"].shape == (2, 1, 8)
        assert link["gated_recurrent_post_c_proj"].shape == (2, 1, 8)
        for tensor_name in (
            "query",
            "recurrent_source_reads",
            "recurrent_key_reads",
            "recurrent_value_reads",
            "local_pre_c_proj",
            "recurrent_pre_c_proj",
            "gated_recurrent_pre_c_proj",
        ):
            assert not link[tensor_name].requires_grad
        assert not link["diagnostic_retain_grad"]
        assert link["retained_tensor_names"] == ()
    assert row["writer_block_states"] is None


def test_diagnostic_retain_grad_reaches_actual_b10_b8_writers_and_sdpa_reads():
    current = c_model().train()
    with torch.no_grad():
        current.g_rec.fill_(0.25)
        current.g_rec_b3.fill_(0.30)
        current.g_rec_b5.fill_(0.35)
        current.g_rec_b6.fill_(0.20)
    before = inventory(current)
    before_state = {
        name: value.detach().clone() for name, value in current.state_dict().items()
    }
    tokens = torch.randint(0, 32, (2, 4))
    targets = torch.randint(0, 32, (2,))
    state = current.init_incremental_state(2)
    rows = []
    for position in range(tokens.size(1)):
        logits, state, row = current.incremental_step(
            tokens[:, position],
            state,
            return_diagnostics=True,
            diagnostic_attention_weights=True,
            return_block_states=True,
            diagnostic_retain_grad=True,
        )
        rows.append(row)
    torch.nn.functional.cross_entropy(logits[:, 0], targets).backward()
    assert state.h8_ring.requires_grad and state.h10_ring.requires_grad
    final = rows[-1]
    assert final["diagnostic_retain_grad"]
    for name in ("b3", "b5"):
        link = final["links"][name]
        assert set(
            (
                "query",
                "recurrent_source_reads",
                "recurrent_key_reads",
                "recurrent_value_reads",
                "local_pre_c_proj",
                "recurrent_pre_c_proj",
                "gated_recurrent_pre_c_proj",
            )
        ).issubset(link["retained_tensor_names"])
        assert link["recurrent_positions"].tolist() == [[0, 1]]
        for tensor_name in (
            "recurrent_source_reads",
            "recurrent_key_reads",
            "recurrent_value_reads",
            "recurrent_pre_c_proj",
            "gated_recurrent_pre_c_proj",
        ):
            gradient = link[tensor_name].grad
            assert gradient is not None, (name, tensor_name)
            assert torch.isfinite(gradient).all(), (name, tensor_name)
            assert torch.count_nonzero(gradient), (name, tensor_name)
        assert link["local_attention_weights"].grad is None
        assert link["recurrent_attention_weights"].grad is None
    for writer_name in ("b8", "b10"):
        writer = rows[0]["writer_block_states"][writer_name]
        assert writer.grad is not None, writer_name
        assert torch.isfinite(writer.grad).all(), writer_name
        assert torch.count_nonzero(writer.grad), writer_name
    assert inventory(current) == before
    for name, value in current.state_dict().items():
        assert torch.equal(value, before_state[name]), name


def test_diagnostic_retain_grad_is_explicit_and_cannot_start_midstream():
    current = c_model().eval()
    state = current.init_incremental_state(2)
    token = torch.tensor([1, 2])
    with pytest.raises(ValueError):
        current.incremental_step(
            token, state, diagnostic_retain_grad=True
        )
    with torch.no_grad(), pytest.raises(ValueError):
        current.incremental_step(
            token,
            state,
            return_diagnostics=True,
            diagnostic_retain_grad=True,
        )
    with torch.no_grad():
        _, detached_state = current.incremental_step(token, state)
    with pytest.raises(ValueError):
        current.incremental_step(
            token,
            detached_state,
            return_diagnostics=True,
            diagnostic_retain_grad=True,
        )


def test_recurrent_ring_capacity_rollover_and_short_lag_selection_are_exact():
    current = c_model().eval()
    channels = current.config.n_embd
    ring = torch.arange(1023, dtype=torch.float32).view(1, 1023, 1).expand(
        2, 1023, channels
    ).clone()
    positions = tuple(range(1023))
    value = torch.full((2, 1, channels), 1023.0)
    updated, updated_positions = current._append_ring(ring, positions, value, 1023)
    assert tuple(updated.shape) == (2, core.RECURRENT_RING_CAPACITY, channels)
    assert updated_positions == tuple(range(1, 1024))
    assert updated.untyped_storage().nbytes() == updated.numel() * updated.element_size()
    bank = current._incremental_bank_from_ring(
        updated, updated_positions, 1024, minimum_lag=2, mode="full"
    )
    assert bank.positions.tolist() == [list(range(1, 1023))]
    assert bank.values[0, :, 0].tolist() == list(range(1, 1023))
    assert 1023 not in bank.positions

    second_value = torch.full((2, 1, channels), 1024.0)
    second, second_positions = current._append_ring(
        updated, updated_positions, second_value, 1024
    )
    assert second_positions == tuple(range(2, 1025))
    bank = current._incremental_bank_from_ring(
        second, second_positions, 1025, minimum_lag=2, mode="full"
    )
    selected = bank.positions[0].tolist()
    assert selected == list(range(2, 1024))
    assert bank.values[0, :, 0].tolist() == list(range(2, 1024))
    # The two preregistered endpoints exercise the actual logical-to-physical
    # index formula after rollover, not just a mask constructed in isolation.
    assert second_positions.index(2) == 0
    assert selected.index(2) == 0  # lag 1023
    assert second_positions.index(1023) == 1021
    assert selected.index(1023) == 1021  # lag 2
    assert second_positions.index(1024) == 1022
    assert 1024 not in selected  # lag 1 is not a recurrent read

    excluded_changed = second.clone()
    excluded_changed[:, -1].fill_(-9999.0)
    excluded_bank = current._incremental_bank_from_ring(
        excluded_changed, second_positions, 1025, minimum_lag=2, mode="full"
    )
    assert torch.equal(bank.values, excluded_bank.values)
    with pytest.raises(ValueError, match="current or future"):
        current._incremental_bank_from_ring(
            second,
            tuple(range(3, 1026)),
            1025,
            minimum_lag=2,
            mode="full",
        )
    with pytest.raises(ValueError, match="current or future"):
        current._incremental_bank_from_ring(
            second,
            tuple(range(4, 1027)),
            1025,
            minimum_lag=2,
            mode="full",
        )


def test_recurrent_selector_causality_is_exercised_on_live_post_rollover_state():
    length = 1024
    current = core.FixedWriterB3B5W2RepresentationPressureGPT(
        base_model(length=length)
    ).eval()
    with torch.no_grad():
        current.g_rec.fill_(0.25)
        current.g_rec_b3.fill_(0.20)
        current.g_rec_b5.fill_(0.15)
        current.g_rec_b6.fill_(0.10)
    tokens = (torch.arange(length) * 7 + 3).remainder(32).view(1, -1)
    state = current.init_incremental_state(1)
    with torch.no_grad():
        for position in range(length):
            _, state = current.incremental_step(tokens[:, position], state)
    assert state.position == 1024
    assert state.h10_positions == tuple(range(1, 1024))
    assert state.h8_positions == tuple(range(1, 1024))
    for ring_name in ("h10", "h8"):
        ring = getattr(state, f"{ring_name}_ring")
        positions = getattr(state, f"{ring_name}_positions")
        bank = current._incremental_bank_from_ring(
            ring, positions, 1024, minimum_lag=2, mode="full"
        )
        assert bank.positions[0, 0].item() == 1  # lag 1023
        assert bank.positions[0, -1].item() == 1022  # lag 2
        assert 1023 not in bank.positions  # lag 1
        excluded_changed = ring.clone()
        excluded_changed[:, -1].fill_(float("nan"))
        changed_bank = current._incremental_bank_from_ring(
            excluded_changed, positions, 1024, minimum_lag=2, mode="full"
        )
        assert torch.equal(bank.values, changed_bank.values)
        assert torch.isfinite(changed_bank.values).all()


def test_chunk_reload_compares_every_kv_cache_and_all_four_writer_rings(tmp_path):
    length = 20
    current = core.FixedWriterB3B5W2RepresentationPressureGPT(
        base_model(length=length)
    ).eval()
    tokens = torch.randint(0, 32, (2, length))
    uninterrupted = current.init_incremental_state(2)
    chunked = current.init_incremental_state(2)
    uninterrupted_logits = []
    chunked_logits = []
    state_path = tmp_path / "incremental_state.pt"
    with torch.no_grad():
        for position in range(length):
            logits, uninterrupted = current.incremental_step(
                tokens[:, position], uninterrupted
            )
            uninterrupted_logits.append(logits)
        for position in range(length):
            logits, chunked = current.incremental_step(tokens[:, position], chunked)
            chunked_logits.append(logits)
            if position == 10:
                torch.save(chunked, state_path)
                chunked = torch.load(state_path, weights_only=False)
    assert torch.equal(
        torch.cat(uninterrupted_logits, 1), torch.cat(chunked_logits, 1)
    )
    assert uninterrupted.position == chunked.position
    assert len(uninterrupted.caches) == len(chunked.caches) == 12
    for left, right in zip(uninterrupted.caches, chunked.caches):
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.equal(left.key, right.key)
            assert torch.equal(left.value, right.value)
    for name in ("h7", "h8", "h10", "h12"):
        assert getattr(uninterrupted, f"{name}_positions") == getattr(
            chunked, f"{name}_positions"
        )
        assert torch.equal(
            getattr(uninterrupted, f"{name}_ring"),
            getattr(chunked, f"{name}_ring"),
        )


class _ControlSpyMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_trace = []

    def _incremental_special_block(self, *args, **kwargs):
        self.control_trace.append(
            {
                "block": args[1],
                "bank": args[3],
                "permutation": args[4],
                "gate_override": args[5],
                "local_capacity": args[6],
            }
        )
        return super()._incremental_special_block(*args, **kwargs)


class _SpyC(_ControlSpyMixin, core.FixedWriterB3B5W2RepresentationPressureGPT):
    pass


class _SpyFixed(_ControlSpyMixin, core.FixedControlEvaluationGPT):
    pass


@pytest.mark.parametrize(
    "spy_class,windows",
    (
        (_SpyC, {0: 2, 2: 2, 4: 2, 5: 512}),
        (_SpyFixed, {0: 2, 2: 32, 4: 64, 5: 512}),
    ),
)
@pytest.mark.parametrize("control", core.INCREMENTAL_CONTROLS)
def test_controls_are_exactly_isolated_to_b3_b5(spy_class, windows, control):
    current = spy_class(base_model()).eval()
    permutation = torch.tensor([1, 0])
    off, shuffled = current._control_sets(control)
    state = current.init_incremental_state(2)
    recurrent_permutation = permutation if shuffled else None
    with torch.no_grad():
        current.incremental_step(
            torch.tensor([1, 2]),
            state,
            control=control,
            recurrent_permutation=recurrent_permutation,
        )
    assert [row["block"] for row in current.control_trace] == list(SPECIAL_BLOCKS)
    for row in current.control_trace:
        block = row["block"]
        assert row["local_capacity"] == windows[block] - 1
        if block in off:
            assert row["bank"] is None
            assert row["gate_override"] == 0.0
        else:
            assert row["bank"] is not None
            assert row["gate_override"] is None
        if block in shuffled:
            assert torch.equal(row["permutation"], permutation)
        else:
            assert row["permutation"] is None
    assert not ({0, 5} & off)
    assert not ({0, 5} & shuffled)


@pytest.mark.parametrize("constructor", (c_model, fixed_control_model))
def test_nonregistered_controls_and_permutation_leaks_are_rejected(constructor):
    current = constructor().eval()
    state = current.init_incremental_state(2)
    token = torch.tensor([1, 2])
    for forbidden in ("b1_off", "b6_off", "b1_shuffled", "b6_shuffled"):
        with pytest.raises(ValueError):
            current.incremental_step(token, state, control=forbidden)
    with pytest.raises(ValueError):
        current.incremental_step(
            token,
            state,
            control="all_real",
            recurrent_permutation=torch.tensor([1, 0]),
        )
    with pytest.raises(ValueError):
        current.incremental_step(token, state, control="b3_shuffled")
    with pytest.raises(ValueError):
        current.init_incremental_state(2, b6_full_native=True)
    with pytest.raises(ValueError):
        current.forward_pass(
            torch.randint(0, 32, (2, TEST_LENGTH)),
            full_counterfactual_blocks=(2,),
        )


def test_combined_shuffle_changes_only_b3_b5_and_is_repeatable_without_state_leak():
    length = 66
    current = core.FixedControlEvaluationGPT(base_model(length=length)).eval()
    with torch.no_grad():
        current.g_rec.fill_(0.2)
        current.g_rec_b3.fill_(0.2)
        current.g_rec_b5.fill_(0.2)
        current.g_rec_b6.fill_(0.2)
    tokens = torch.stack((torch.arange(length) % 32, (torch.arange(length) + 7) % 32))
    permutation = torch.tensor([1, 0])
    with torch.no_grad():
        before = current.incremental_logits(tokens, control="all_real")["logits"]
        shuffled = current.incremental_logits(
            tokens,
            control="b3_b5_shuffled",
            recurrent_permutation=permutation,
        )["logits"]
        after = current.incremental_logits(tokens, control="all_real")["logits"]
    assert not torch.equal(before, shuffled)
    assert torch.equal(before, after)


def test_every_registered_control_has_asserted_actual_intervention_outputs():
    current = c_model().eval()
    with torch.no_grad():
        current.g_rec.fill_(0.30)
        current.g_rec_b3.fill_(0.35)
        current.g_rec_b5.fill_(0.40)
        current.g_rec_b6.fill_(0.25)
    tokens = torch.stack(
        (torch.arange(8).remainder(32), (torch.arange(8) + 13).remainder(32))
    )
    permutation = torch.tensor([1, 0])
    with torch.no_grad():
        baseline = current.incremental_logits(
            tokens, control="all_real", return_diagnostics=True
        )
        repeated = current.incremental_logits(tokens, control="all_real")["logits"]
    assert torch.equal(baseline["logits"], repeated)
    reference = baseline["diagnostics"][2]
    for control in core.INCREMENTAL_CONTROLS[1:]:
        off, shuffled = current.control_sets(control)
        with torch.no_grad():
            result = current.incremental_logits(
                tokens,
                control=control,
                recurrent_permutation=permutation if shuffled else None,
                return_diagnostics=True,
            )
        assert not torch.equal(result["logits"], baseline["logits"]), control
        probe = result["diagnostics"][2]
        assert probe["control"] == control
        for block_index in (0, 2, 4, 5):
            name = f"b{block_index + 1}"
            observed = probe["links"][name]
            expected = reference["links"][name]
            if block_index in off:
                assert observed["recurrent_positions"] is None
                assert observed["recurrent_source_reads"] is None
                assert float(observed["gate_coefficient"]) == 0.0
            elif block_index in shuffled:
                assert torch.equal(
                    observed["recurrent_positions"], expected["recurrent_positions"]
                )
                assert torch.equal(
                    observed["recurrent_source_reads"],
                    expected["recurrent_source_reads"].index_select(0, permutation),
                )
                assert torch.equal(
                    observed["gate_coefficient"], expected["gate_coefficient"]
                )
            else:
                assert torch.equal(
                    observed["recurrent_positions"], expected["recurrent_positions"]
                )
                assert torch.equal(
                    observed["recurrent_source_reads"],
                    expected["recurrent_source_reads"],
                )
                assert torch.equal(
                    observed["gate_coefficient"], expected["gate_coefficient"]
                )


@pytest.mark.parametrize("constructor", (c_model, fixed_control_model))
def test_parallel_and_incremental_execution_are_causal_and_row_isolated(constructor):
    current = constructor().eval()
    with torch.no_grad():
        current.g_rec.fill_(0.25)
        current.g_rec_b3.fill_(0.20)
        current.g_rec_b5.fill_(0.15)
        current.g_rec_b6.fill_(0.10)
    tokens = torch.randint(0, 32, (2, TEST_LENGTH))
    future = tokens.clone()
    future[:, 8:] = torch.randint(0, 32, future[:, 8:].shape)
    other_row = tokens.clone()
    other_row[1] = (other_row[1] + 5) % 32
    with torch.no_grad():
        parallel = current.forward_multi_pass(tokens, num_passes=2)["logits"]
        parallel_future = current.forward_multi_pass(future, num_passes=2)["logits"]
        parallel_row = current.forward_multi_pass(other_row, num_passes=2)["logits"]
        incremental = current.incremental_logits(tokens, control="all_real")["logits"]
        incremental_future = current.incremental_logits(future, control="all_real")["logits"]
        incremental_row = current.incremental_logits(other_row, control="all_real")["logits"]
    assert torch.equal(parallel[:, :8], parallel_future[:, :8])
    assert torch.equal(parallel[0], parallel_row[0])
    assert torch.equal(incremental[:, :8], incremental_future[:, :8])
    assert torch.equal(incremental[0], incremental_row[0])


@pytest.mark.parametrize("constructor", (c_model, fixed_control_model))
def test_no_router_projection_parameter_or_module_was_added(constructor):
    candidate = constructor()
    accepted = accepted_fixed_model()
    assert inventory(candidate) == inventory(accepted)
    candidate_children = [
        (name, type(module)) for name, module in candidate.named_modules() if name
    ]
    accepted_children = [
        (name, type(module)) for name, module in accepted.named_modules() if name
    ]
    assert candidate_children == accepted_children
    forbidden = ("router", "source_depth", "teacher", "distill", "auxiliary")
    for name, module in candidate.named_modules():
        description = f"{name} {type(module).__name__}".lower()
        assert not any(word in description for word in forbidden)
    assert "nn.Linear" not in inspect.getsource(core._B3B5ControlledFixedWriterGPT)
    configuration = json.loads(
        (
            ROOT
            / "configs"
            / "exp2d5c_fixed_writer_b3_b5_w2_matched_100m.json"
        ).read_text()
    )["architecture_c"]
    assert configuration["parameters"] == core.EXPECTED_PARAMETER_COUNT
    assert configuration["new_parameters"] == 0
    serialized_architecture = json.dumps(configuration, sort_keys=True).lower()
    assert not any(word in serialized_architecture for word in forbidden)
    projection_modules = [
        name
        for name, module in candidate.named_modules()
        if name and (
            "proj" in name.lower()
            or type(module).__name__.lower() in {"linear", "conv1d"}
        )
    ]
    allowed_suffixes = (
        "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", "lm_head"
    )
    assert projection_modules
    assert all(name.endswith(allowed_suffixes) for name in projection_modules)
