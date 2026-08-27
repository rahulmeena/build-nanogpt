import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_2d2h as exp  # noqa: E402


def incremental(real, off, shuffled, wins, count=256):
    return {
        "true_b2_recurrent_gain": off - real,
        "true_b2_sequence_gap": shuffled - real,
        "real_vs_b2_off_sequences": {"wins": wins, "count": count},
        "real_vs_b2_shuffled_sequences": {"wins": wins, "count": count},
    }


def test_protocol_arithmetic_and_cadence():
    assert exp.TOTAL_PARAMETERS == exp.SOURCE_PARAMETERS == 124_475_905
    assert exp.MAX_UPDATES == 191
    assert exp.ADDITIONAL_TARGETS == 100_139_008
    assert exp.CUMULATIVE_TARGETS == 250_609_664
    assert exp.INCREMENTAL_BATCHES * exp.VALIDATION_B * exp.T == 262_144
    assert [u for u in range(1, 192) if exp.pass_count(u) == 3] == [32, 64, 96, 128, 160]
    assert exp.MILESTONES == (0, 20, 48, 96, 143, 191)
    assert exp.FORCED_RESTART_UPDATE == 96


def test_preregistered_config_matches_removal_and_addition():
    config = json.loads(exp.CONFIG_PATH.read_text())
    assert config["source_checkpoint_sha256"] == exp.SOURCE_SHA256
    assert config["architecture"]["b1"]["recurrent_path_present"] is False
    assert config["architecture"]["b1"]["b12_raw_ring_present"] is False
    assert config["architecture"]["b2"]["local_window"] == 32
    assert config["architecture"]["b2"]["maximum_recurrent_entries"] == 992
    assert config["architecture"]["total_parameters"] == 124_475_905
    assert config["incremental_cache_bounds"]["b12_raw_ring"] == 0


def test_memory_accounting_has_no_b12_ring():
    report = exp.memory_accounting()
    expected = (1 * 768 * 2 * 2 + 31 * 768 * 2 * 2
                + 1023 * 768 * 2 + 10 * 1023 * 768 * 2 * 2)
    assert report["B1"]["b12_recurrent_raw_state_bytes"] == 0
    assert report["B1"]["total_experimental_inference_state_bytes"] == expected
    assert report["B64"]["total_experimental_inference_state_bytes"] == 64 * expected


def test_classification_uses_256_sequence_thresholds():
    assert exp.classify_result(incremental(3.0, 3.0004, 3.0003, 140)) == (
        "B2 W32 ESTABLISHES POSITIVE SECOND-LINK RECURRENT UTILITY")
    assert exp.classify_result(incremental(3.0, 3.002, 3.0015, 170)) == (
        "W32 STRONGLY RESCUES B11→B2 RECURRENT UTILITY")
    assert exp.classify_result(incremental(3.01, 3.0, 3.02, 120)) == (
        "B2 W32 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY")
    assert exp.classify_result(incremental(3.01, 3.0, 2.99, 120)) == (
        "B2 W32 SECOND RECURRENT LINK IS HARMFUL")


def test_ephemeral_and_persistent_checkpoint_cli_are_distinct():
    parser = exp.build_parser()
    train = parser._subparsers._group_actions[0].choices["train"]
    options = {option for action in train._actions for option in action.option_strings}
    assert "--ephemeral-checkpoint-root" in options
    assert "--checkpoint-persist-lock" in options
