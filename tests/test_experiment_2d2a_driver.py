import json
import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_2d2a as exp  # noqa: E402


def _incremental(gain, gap, wins=65, losses=63):
    return {
        "recurrent_gain": gain,
        "sequence_gap": gap,
        "real_vs_plain_sequences": {
            "count": wins + losses,
            "wins": wins,
            "losses": losses,
        },
        "real_vs_shuffled_sequences": {
            "count": wins + losses,
            "wins": wins,
            "losses": losses,
        },
    }


def test_frozen_protocol_arithmetic_and_cadence():
    assert exp.SOURCE_PARAMETERS == 124_475_904
    assert exp.TOTAL_PARAMETERS == 124_475_905
    assert exp.GLOBAL_TARGETS == 524_288
    assert exp.MAX_UPDATES == 96
    assert exp.TOTAL_TARGETS == 50_331_648
    assert [u for u in range(1, 97) if exp.pass_count(u) == 3] == [32, 64, 96]
    assert exp.pass_weights(31) == (0.25, 0.75)
    assert exp.pass_weights(32) == (0.20, 0.40, 0.40)


def test_warmup_reaches_peak_on_update_ten_then_stays_constant():
    assert exp.learning_rate_fraction(1) == 0.1
    assert exp.learning_rate_fraction(9) == 0.9
    assert exp.learning_rate_fraction(10) == 1.0
    assert exp.learning_rate_fraction(96) == 1.0


def test_preregistered_config_matches_driver_constants():
    config = json.loads(exp.CONFIG_PATH.read_text())
    assert config["architecture"]["new_parameters"] == ["g_rec"]
    assert config["architecture"]["recurrent_positions"] == ["t-3", "t-2"]
    assert config["training"]["total_targets"] == exp.TOTAL_TARGETS
    assert config["training"]["milestones"] == list(exp.MILESTONES)
    assert config["training"]["scientific_checkpoint_updates"] == list(
        exp.SCIENTIFIC_CHECKPOINTS
    )


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.ones(3, 2))
        self.bias = nn.Parameter(torch.ones(3))
        self.g_rec = nn.Parameter(torch.zeros(()))


def test_optimizer_preserves_two_logical_lr_classes_and_decay_semantics():
    model = _TinyModel()
    optimizer, report = exp.configure_optimizer(model, device_type="cpu")
    assert report["logical_parameter_groups"] == 2
    assert report["physical_parameter_groups"] == 3
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["base_decay"]["weight_decay"] == 0.1
    assert groups["base_nodecay"]["weight_decay"] == 0.0
    assert groups["gate"]["weight_decay"] == 0.0
    lrs = exp.set_optimizer_lrs(optimizer, 10)
    assert lrs == {
        "base_decay": exp.BASE_LR,
        "base_nodecay": exp.BASE_LR,
        "gate": exp.GATE_LR,
    }


def test_classification_thresholds_and_recommendation_priority():
    parallel = {"recurrent_gain": 0.0}
    strong = _incremental(0.02, 0.003, wins=116, losses=12)
    assert exp.classify_result(strong, parallel) == (
        "TOKEN-INDEXED RECURRENT K/V LEARNS CLEAR POSITIVE UTILITY",
        "SEQUENCE-SPECIFIC RECURRENT K/V",
    )
    neutral = _incremental(-0.001, 0.002, wins=70, losses=58)
    assert exp.classify_result(neutral, parallel)[0] == (
        "TOKEN-INDEXED RECURRENT K/V APPROACHES NEUTRALITY"
    )
    harmful = _incremental(-0.02, -0.001, wins=20, losses=108)
    assert exp.classify_result(harmful, parallel)[0] == (
        "TOKEN-INDEXED RECURRENT K/V REMAINS HARMFUL"
    )
    assert exp.choose_recommendation(
        "TOKEN-INDEXED RECURRENT K/V APPROACHES NEUTRALITY",
        "NO SEQUENCE-SPECIFIC RECURRENT K/V",
        0.001,
        -0.001,
    ) == "TRAIN FOR SELF-RECURRENT K/V DISTRIBUTION COMPATIBILITY"

