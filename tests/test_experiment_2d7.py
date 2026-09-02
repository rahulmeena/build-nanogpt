import ast
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d7 as experiment  # noqa: E402
import experiment_2d7_core as core  # noqa: E402
import smoke_test as support  # noqa: E402


SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)


def base_model(seed=207):
    torch.manual_seed(seed)
    return GPT(GPTConfig(
        block_size=70,
        vocab_size=32,
        n_layer=12,
        n_head=2,
        n_embd=8,
        residual_mode="standard",
    ))


def model(arm):
    return core.BoundaryAlignmentGPT(base_model(), arm)


def inventory(value):
    return [(name, tuple(parameter.shape), parameter.numel()) for name, parameter in value.named_parameters()]


def test_sources_and_config_parse():
    for path in (ROOT / "scripts/experiment_2d7.py", ROOT / "scripts/experiment_2d7_core.py", Path(__file__)):
        ast.parse(path.read_text())
    config = json.loads((ROOT / "configs/exp2d7_trained_boundary_alignment_nog.json").read_text())
    assert config["parent"]["sha256"] == experiment.PARENT_SHA256
    assert config["training"]["optimizer_updates_per_arm"] == 191
    assert config["training"]["final_global_update"] == 2290
    assert config["evaluation"]["conditions"] == list(core.GEOMETRY_NAMES.values())


def test_geometry_is_only_parameter_free_difference():
    models = {arm: model(arm) for arm in core.GEOMETRIES}
    assert inventory(models["N"]) == inventory(models["O"]) == inventory(models["G"])
    assert list(models["N"].state_dict()) == list(models["O"].state_dict()) == list(models["G"].state_dict())
    assert experiment.normalized_common_manifest("N") == experiment.normalized_common_manifest("O") == experiment.normalized_common_manifest("G")
    assert core.GEOMETRIES == {
        "N": {0: 2, 2: 32, 4: 64},
        "O": {0: 1, 2: 31, 4: 63},
        "G": {0: 3, 2: 33, 4: 65},
    }


@pytest.mark.parametrize("arm", ("N", "O", "G"))
def test_parallel_boundary_masks_are_exact(arm):
    candidate = model(arm)
    length = 70
    query = torch.arange(length).view(length, 1)
    source = torch.arange(length).view(1, length)
    lag = query - source
    for block, minimum in core.GEOMETRIES[arm].items():
        recurrent = candidate.recurrent_mask(block, length, length, "cpu")
        local = candidate.local_mask(block, length, "cpu")
        assert torch.equal(recurrent, (lag >= minimum) & (lag <= 1023))
        assert torch.equal(local, (lag >= 0) & (lag < core.LOCAL_WINDOWS[block]))
        assert not bool((recurrent & (lag <= 0)).any())


@pytest.mark.parametrize("arm", ("N", "O", "G"))
def test_incremental_positions_preserve_j_equals_t_minus_lag(arm):
    candidate = model(arm).eval()
    tokens = torch.arange(70).remainder(32).view(1, -1)
    state = candidate.init_incremental_state(1)
    final = None
    with torch.no_grad():
        for position in range(tokens.size(1)):
            _, state, final = candidate.incremental_step(tokens[:, position], state, return_diagnostics=True, diagnostic_attention_weights=False)
    for block, minimum in core.GEOMETRIES[arm].items():
        observed = final["links"][f"b{block + 1}"]["recurrent_positions"][0].tolist()
        assert observed == list(range(0, 70 - minimum))
    assert candidate._last_b6_local_capacity == 69  # test model context is 70
    assert candidate._b6_recurrent_branch_calls == 0
    assert not hasattr(state, "h7_ring")


def test_overlap_and_gap_boundary_neighborhoods():
    expected = {
        "N": {0: ((True, False), (False, True), (False, True)), 2: ((True, False), (False, True), (False, True)), 4: ((True, False), (False, True), (False, True))},
        "O": {0: ((True, True), (False, True), (False, True)), 2: ((True, True), (False, True), (False, True)), 4: ((True, True), (False, True), (False, True))},
        "G": {0: ((True, False), (False, False), (False, True)), 2: ((True, False), (False, False), (False, True)), 4: ((True, False), (False, False), (False, True))},
    }
    for arm in expected:
        candidate = model(arm)
        for block, rows in expected[arm].items():
            length = 70
            query = torch.arange(length).view(length, 1)
            source = torch.arange(length).view(1, length)
            lag = query - source
            local = candidate.local_mask(block, length, "cpu")
            recurrent = candidate.recurrent_mask(block, length, length, "cpu")
            boundary = core.LOCAL_WINDOWS[block]
            observed = tuple((bool(((lag == value) & local).any()), bool(((lag == value) & recurrent).any())) for value in (boundary - 1, boundary, boundary + 1))
            assert observed == rows


def test_scope_constants_are_exact():
    assert experiment.LOCAL_UPDATES * experiment.TARGETS_PER_UPDATE == experiment.LOCAL_TARGETS
    assert experiment.PARENT_TARGETS + experiment.LOCAL_TARGETS == experiment.FINAL_TARGETS
    assert experiment.PARENT_GLOBAL_UPDATE + experiment.LOCAL_UPDATES == experiment.FINAL_GLOBAL_UPDATE
    assert experiment.PANEL_SEQUENCES * 1024 == experiment.PANEL_TARGETS
