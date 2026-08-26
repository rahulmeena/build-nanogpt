import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_2d2b as exp  # noqa: E402


def _incremental(full, plain, shuffled, two_slot, wins=100, count=128):
    losses = {
        "full_real": full,
        "plain": plain,
        "full_shuffled": shuffled,
        "two_slot_real": two_slot,
    }
    result = {
        "controls": {
            name: {"validation_loss": value} for name, value in losses.items()
        },
        "true_full_gain": plain - full,
        "true_sequence_gap": shuffled - full,
        "true_bank_width_gain": two_slot - full,
    }
    for name in (
        "full_vs_plain_sequences",
        "full_vs_shuffled_sequences",
        "full_vs_two_slot_sequences",
    ):
        result[name] = {
            "wins": wins,
            "losses": count - wins,
            "ties": 0,
            "count": count,
        }
    return result


def test_protocol_arithmetic_and_cadence():
    assert exp.TOTAL_PARAMETERS == 124_475_905
    assert exp.GLOBAL_TARGETS == 524_288
    assert exp.MAX_UPDATES == 191
    assert exp.ADDITIONAL_TARGETS == 100_139_008
    assert exp.CUMULATIVE_TARGETS == 150_470_656
    assert [update for update in range(1, 192) if exp.pass_count(update) == 3] == [
        32,
        64,
        96,
        128,
        160,
    ]
    assert exp.pass_weights(31) == (0.25, 0.75)
    assert exp.pass_weights(32) == (0.20, 0.40, 0.40)


def test_preregistered_config_matches_driver():
    config = json.loads(exp.CONFIG_PATH.read_text())
    assert config["frozen_2d2a_commit"] == exp.FROZEN_COMMIT
    assert config["source_checkpoint_sha256"] == exp.SOURCE_SHA256
    assert config["architecture"]["maximum_recurrent_entries"] == 1022
    assert config["architecture"]["new_parameter_count_vs_2d2a"] == 0
    assert config["training"]["additional_targets"] == exp.ADDITIONAL_TARGETS
    assert config["training"]["cumulative_2d2_targets"] == exp.CUMULATIVE_TARGETS
    assert config["training"]["local_milestones"] == list(exp.MILESTONES)


def test_memory_accounting_exact_bf16_geometry():
    report = exp.memory_accounting()
    expected_recurrent_b1 = 1023 * 768 * 2
    expected_b1_local = 1 * 768 * 2 * 2
    expected_upper_b1 = 11 * 1023 * 768 * 2 * 2
    assert report["B1"]["b12_recurrent_raw_state_bytes"] == expected_recurrent_b1
    assert report["B1"]["b1_local_kv_bytes"] == expected_b1_local
    assert report["B1"]["b2_b12_ordinary_kv_bytes"] == expected_upper_b1
    assert report["B64"]["total_experimental_inference_state_bytes"] == 64 * report[
        "B1"
    ]["total_experimental_inference_state_bytes"]


def test_classification_rules():
    positive = _incremental(3.0, 3.004, 3.003, 3.002, wins=90)
    assert exp.classify_result(positive) == (
        "FULL-BANK RECURRENT K/V SCALES POSITIVE UTILITY"
    )
    strong = _incremental(3.0, 3.02, 3.003, 3.01, wins=100)
    assert exp.classify_result(strong) == (
        "FULL-BANK RECURRENT K/V STRONGLY SCALES UTILITY"
    )
    no_width = _incremental(3.0, 3.002, 3.003, 2.999, wins=90)
    assert exp.classify_result(no_width).startswith("RECURRENT K/V REMAINS USEFUL")
    harmful = _incremental(3.01, 3.0, 3.02, 3.0, wins=20)
    assert exp.classify_result(harmful).startswith(
        "FULL-BANK MEMORY IS SEQUENCE-SPECIFIC"
    )


def test_next_experiment_rule_priority():
    attention = {
        "mass_partitions": {
            "lags_32_127": 0.2,
            "lags_128_511": 0.3,
            "lags_512_1023": 0.1,
        }
    }
    parallel = {"full_bank_gain": 0.01, "bank_width_gain": 0.006}
    incremental = {"true_full_gain": 0.008, "true_bank_width_gain": 0.004}
    assert exp.choose_recommendation(
        "FULL-BANK RECURRENT K/V SCALES POSITIVE UTILITY",
        parallel,
        incremental,
        attention,
    ) == "EXTEND FULL-BANK RECURRENT K/V TO MIRRORED HIGH→LOW LAYER PAIRS"
    failed_true = {"true_full_gain": -0.001, "true_bank_width_gain": -0.001}
    assert exp.choose_recommendation(
        "FULL-BANK RECURRENT K/V DOES NOT ESTABLISH POSITIVE UTILITY",
        parallel,
        failed_true,
        attention,
    ) == "TRAIN FULL-BANK SELF-RECURRENT DISTRIBUTION COMPATIBILITY"
