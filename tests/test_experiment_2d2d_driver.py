import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_2d2d as exp  # noqa: E402


def incremental(real, off, shuffled, wins=90, count=128):
    result = {
        "controls": {
            "both_real": {"validation_loss": real},
            "b2_recurrence_off": {"validation_loss": off},
            "b2_shuffled": {"validation_loss": shuffled},
        },
        "true_b2_recurrent_gain": off - real,
        "true_b2_sequence_gap": shuffled - real,
        "tanh_g_rec_b2": 0.02,
    }
    for name in ("both_real_vs_b2_off_sequences",
                 "both_real_vs_b2_shuffled_sequences"):
        result[name] = {"wins": wins, "losses": count - wins,
                        "ties": 0, "count": count}
    return result


def test_protocol_arithmetic_and_cadence():
    assert exp.TOTAL_PARAMETERS == 124_475_906
    assert exp.SOURCE_PARAMETERS == 124_475_905
    assert exp.GLOBAL_TARGETS == 524_288
    assert exp.MAX_UPDATES == 191
    assert exp.ADDITIONAL_TARGETS == 100_139_008
    assert exp.CUMULATIVE_TARGETS == 250_609_664
    assert [u for u in range(1, 192) if exp.pass_count(u) == 3] == [32, 64, 96, 128, 160]
    assert exp.pass_weights(31) == (0.25, 0.75)
    assert exp.pass_weights(32) == (0.20, 0.40, 0.40)


def test_preregistered_config_matches_driver():
    config = json.loads(exp.CONFIG_PATH.read_text())
    assert config["frozen_2d2b_commit"] == exp.FROZEN_COMMIT
    assert config["source_checkpoint_sha256"] == exp.SOURCE_SHA256
    assert config["architecture"]["b1"]["maximum_recurrent_entries"] == 1022
    assert config["architecture"]["b2"]["local_window"] == 32
    assert config["architecture"]["b2"]["recurrent_min_lag"] == 32
    assert config["architecture"]["b2"]["maximum_recurrent_entries"] == 992
    assert config["architecture"]["new_parameter_count_vs_2d2b"] == 1
    assert config["training"]["additional_targets"] == exp.ADDITIONAL_TARGETS
    assert config["training"]["cumulative_2d2_targets"] == exp.CUMULATIVE_TARGETS


def test_memory_accounting_exact_bf16_geometry():
    report = exp.memory_accounting()
    local = 1 * 768 * 2 * 2
    ring = 1023 * 768 * 2
    upper = 10 * 1023 * 768 * 2 * 2
    b2_local = 31 * 768 * 2 * 2
    expected = local + b2_local + 2 * ring + upper
    assert report["B1"]["b1_local_kv_bytes"] == local
    assert report["B1"]["b2_local_kv_bytes"] == b2_local
    assert report["B1"]["b11_recurrent_raw_state_bytes"] == ring
    assert report["B1"]["b12_recurrent_raw_state_bytes"] == ring
    assert report["B1"]["b3_b12_ordinary_kv_bytes"] == upper
    assert report["B1"]["total_experimental_inference_state_bytes"] == expected
    assert report["B1"]["total_experimental_inference_state_bytes"] == 34_667_520
    assert report["B1"]["delta_bytes_vs_final_2d2b"] == -1_476_096
    assert report["B1"]["delta_bytes_vs_final_2d2c"] == 92_160
    assert report["B1"]["delta_bytes_vs_standard_gpt2"] == -3_044_352
    assert report["B1"]["delta_bytes_vs_final_2d2b"] < 0
    assert report["B1"]["delta_bytes_vs_standard_gpt2"] < 0
    assert report["B64"]["total_experimental_inference_state_bytes"] == 64 * expected


def test_classification_rules():
    assert exp.classify_result(incremental(3.0, 3.0004, 3.0003)) == (
        "B2 W32 ESTABLISHES POSITIVE SECOND-LINK RECURRENT UTILITY")
    assert exp.classify_result(incremental(3.0, 3.002, 3.0015, wins=80)) == (
        "W32 STRONGLY RESCUES B11→B2 RECURRENT UTILITY")
    assert exp.classify_result(incremental(3.01, 3.0, 3.02, wins=60)) == (
        "B2 W32 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY")
    near_zero = incremental(3.0, 3.00001, 2.99999, wins=64)
    assert exp.classify_result(near_zero) == (
        "B2 W32 SECOND RECURRENT LINK REMAINS NEAR ZERO")


def test_exactly_one_next_experiment_rule():
    assert exp.choose_recommendation(
        "B2 W32 ESTABLISHES POSITIVE SECOND-LINK RECURRENT UTILITY", 0.2
    ) == "ADD B10→B3 WITH A GRADED LOCAL WINDOW W64"
    assert exp.choose_recommendation(
        "B2 W32 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY",
        0.2,
    ) == "IMPROVE B2 RECURRENT READOUT BEFORE ADDING B10→B3"
    assert exp.choose_recommendation(
        "B2 W32 SECOND RECURRENT LINK REMAINS NEAR ZERO", 0.2
    ) == "RUN MATCHED B2-W32 TRAINING WITHOUT B11→B2 RECURRENCE"
