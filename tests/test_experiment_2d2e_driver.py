import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_2d2e as exp  # noqa: E402


def incremental(real, off, shuffled, wins=140, count=256):
    return {
        "true_b3_recurrent_gain": off - real,
        "true_b3_sequence_gap": shuffled - real,
        "all_real_vs_b3_off_sequences": {"wins": wins, "losses": count - wins, "ties": 0, "count": count},
        "all_real_vs_b3_shuffled_sequences": {"wins": wins, "losses": count - wins, "ties": 0, "count": count},
    }


def test_protocol_arithmetic_and_cadence():
    assert exp.TOTAL_PARAMETERS == 124_475_907
    assert exp.SOURCE_PARAMETERS == 124_475_906
    assert exp.GLOBAL_TARGETS == 524_288
    assert exp.MAX_UPDATES == 191
    assert exp.ADDITIONAL_TARGETS == 100_139_008
    assert exp.CUMULATIVE_TARGETS == 350_748_672
    assert exp.INCREMENTAL_BATCHES * exp.VALIDATION_B * exp.T == 262_144
    assert [u for u in range(1, 192) if exp.pass_count(u) == 3] == [32, 64, 96, 128, 160]
    assert exp.pass_weights(31) == (0.25, 0.75)
    assert exp.pass_weights(32) == (0.20, 0.40, 0.40)


def test_preregistered_config_matches_driver():
    config = json.loads(exp.CONFIG_PATH.read_text())
    assert config["frozen_2d2d_commit"] == exp.FROZEN_COMMIT
    assert config["source_checkpoint_sha256"] == exp.SOURCE_SHA256
    assert config["architecture"]["b1"]["maximum_recurrent_entries"] == 1022
    assert config["architecture"]["b2"]["local_window"] == 32
    assert config["architecture"]["b2"]["maximum_recurrent_entries"] == 992
    assert config["architecture"]["b3"]["local_window"] == 64
    assert config["architecture"]["b3"]["recurrent_min_lag"] == 64
    assert config["architecture"]["b3"]["maximum_recurrent_entries"] == 960
    assert config["architecture"]["new_parameters_vs_2d2d"] == ["g_rec_b3"]
    assert config["training"]["additional_targets"] == exp.ADDITIONAL_TARGETS
    assert config["training"]["cumulative_2d2_targets"] == exp.CUMULATIVE_TARGETS


def test_memory_accounting_exact_bf16_geometry():
    report = exp.memory_accounting()
    b1 = 1 * 768 * 2 * 2
    b2 = 31 * 768 * 2 * 2
    b3 = 63 * 768 * 2 * 2
    rings = 3 * 1023 * 768 * 2
    upper = 9 * 1023 * 768 * 2 * 2
    expected = b1 + b2 + b3 + rings + upper
    assert report["B1"]["total_experimental_inference_state_bytes"] == expected
    assert report["B1"]["b1_local_kv_bytes"] == b1
    assert report["B1"]["b2_local_kv_bytes"] == b2
    assert report["B1"]["b3_local_kv_bytes"] == b3
    assert report["B1"]["three_recurrent_raw_state_rings_bytes"] == rings
    assert report["B1"]["b4_b12_ordinary_kv_bytes"] == upper
    assert report["B1"]["saving_bytes_vs_standard_gpt2"] > 0
    assert report["B64"]["total_experimental_inference_state_bytes"] == 64 * expected


def test_preregistered_classification_thresholds():
    assert exp.classify_result(incremental(3.0, 3.0004, 3.0003)) == (
        "B10→B3 W64 RECURRENT LINK ESTABLISHES POSITIVE UTILITY")
    assert exp.classify_result(incremental(3.0, 3.002, 3.0015, wins=180)) == (
        "B10→B3 W64 RECURRENT LINK STRONGLY ESTABLISHES UTILITY")
    assert exp.classify_result(incremental(3.01, 3.0, 3.02, wins=120)) == (
        "B10→B3 W64 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY")
    assert exp.classify_result(incremental(3.0, 3.00001, 2.99999, wins=128)) == (
        "B10→B3 W64 RECURRENT LINK REMAINS NEAR ZERO")


def test_exact_next_experiment_rules():
    assert exp.choose_recommendation("B10→B3 W64 RECURRENT LINK ESTABLISHES POSITIVE UTILITY", 0.03) == (
        "RUN MATCHED NO-B11→B2 CONTROL BEFORE ADDING B9→B4")
    assert exp.choose_recommendation("B10→B3 W64 RECURRENT LINK REMAINS NEAR ZERO", 0.03) == (
        "RUN MATCHED B2-W32 TRAINING WITHOUT B11→B2 RECURRENCE")
    assert exp.choose_recommendation("B10→B3 W64 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY", 0.03) == (
        "IMPROVE DEEPER-LAYER RECURRENT READOUT BEFORE ADDING B9→B4")
