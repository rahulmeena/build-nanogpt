import ast
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d8 as experiment  # noqa: E402
import experiment_2d8_core as core  # noqa: E402
import smoke_test as support  # noqa: E402


SYMBOLS = support.load_training_symbols()
GPT = SYMBOLS["GPT"]
GPTConfig = SYMBOLS["GPTConfig"]
torch.set_num_threads(1)


def base_model(seed=208):
    torch.manual_seed(seed)
    return GPT(GPTConfig(
        block_size=70, vocab_size=32, n_layer=12, n_head=2, n_embd=8,
        residual_mode="standard",
    ))


def model(arm):
    return core.OverlapWidthGPT(base_model(), arm)


def inventory(value):
    return [
        (name, tuple(parameter.shape), parameter.numel())
        for name, parameter in value.named_parameters()
    ]


def test_sources_and_config_parse():
    for path in (
        ROOT / "scripts/experiment_2d8.py",
        ROOT / "scripts/experiment_2d8_core.py",
        Path(__file__),
    ):
        ast.parse(path.read_text())
    config = json.loads(
        (ROOT / "configs/exp2d8_trained_overlap_width_n_o1_o2.json").read_text()
    )
    assert config["parent"]["sha256"] == experiment.PARENT_SHA256
    assert config["training"]["new_arms"] == ["O2"]
    assert config["evaluation"]["sequences"] == 4096


def test_geometry_is_the_only_parameter_free_difference():
    models = {arm: model(arm) for arm in core.GEOMETRIES}
    assert inventory(models["N"]) == inventory(models["O1"]) == inventory(models["O2"])
    assert list(models["N"].state_dict()) == list(models["O1"].state_dict()) == list(models["O2"].state_dict())
    assert experiment.normalized_manifest("N") == experiment.normalized_manifest("O1") == experiment.normalized_manifest("O2")
    assert core.GEOMETRIES == {
        "N": {0: 2, 2: 32, 4: 64},
        "O1": {0: 1, 2: 31, 4: 63},
        "O2": {0: 1, 2: 30, 4: 62},
    }


@pytest.mark.parametrize("arm", ("N", "O1", "O2"))
def test_parallel_masks_preserve_the_full_horizon(arm):
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


def test_o2_boundary_neighborhoods():
    candidate = model("O2")
    expected = {
        0: {0: (True, False), 1: (True, True), 2: (False, True)},
        2: {29: (True, False), 30: (True, True), 31: (True, True), 32: (False, True)},
        4: {61: (True, False), 62: (True, True), 63: (True, True), 64: (False, True)},
    }
    length = 70
    query = torch.arange(length).view(length, 1)
    source = torch.arange(length).view(1, length)
    lag = query - source
    for block, rows in expected.items():
        local = candidate.local_mask(block, length, "cpu")
        recurrent = candidate.recurrent_mask(block, length, length, "cpu")
        for value, wanted in rows.items():
            observed = (
                bool(((lag == value) & local).any()),
                bool(((lag == value) & recurrent).any()),
            )
            assert observed == wanted


def test_incremental_source_identity_and_no_b7_recurrence():
    candidate = model("O2").eval()
    tokens = torch.arange(70).remainder(32).view(1, -1)
    state = candidate.init_incremental_state(1)
    diagnostic = None
    with torch.no_grad():
        for position in range(tokens.size(1)):
            _, state, diagnostic = candidate.incremental_step(
                tokens[:, position], state, return_diagnostics=True,
                diagnostic_attention_weights=False,
            )
    for block, minimum in core.GEOMETRIES["O2"].items():
        observed = diagnostic["links"][f"b{block + 1}"]["recurrent_positions"][0].tolist()
        assert observed == list(range(0, 70 - minimum))
    assert candidate._last_b6_local_capacity == 69
    assert candidate._b6_recurrent_branch_calls == 0
    assert not hasattr(state, "h7_ring")


def test_scope_constants_are_exact():
    assert experiment.LOCAL_UPDATES * experiment.TARGETS_PER_UPDATE == experiment.LOCAL_TARGETS
    assert experiment.PARENT_TARGETS + experiment.LOCAL_TARGETS == experiment.FINAL_TARGETS
    assert experiment.PARENT_GLOBAL_UPDATE + experiment.LOCAL_UPDATES == experiment.FINAL_GLOBAL_UPDATE
    assert experiment.PANEL_SEQUENCES * 1024 == experiment.PANEL_TARGETS
