import json
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_2d2f as exp  # noqa: E402


def incremental(real, off, shuffled, wins=140, count=256):
    return {
        "true_b3_recurrent_gain": off - real,
        "true_b3_sequence_gap": shuffled - real,
        "all_real_vs_b3_off_sequences": {"wins": wins, "losses": count - wins, "ties": 0, "count": count},
        "all_real_vs_b3_shuffled_sequences": {"wins": wins, "losses": count - wins, "ties": 0, "count": count},
    }


def test_protocol_arithmetic_and_cadence():
    assert exp.EXPERIMENT == "2D2F"
    assert exp.TOTAL_PARAMETERS == exp.SOURCE_PARAMETERS == 124_475_906
    assert exp.GLOBAL_TARGETS == 524_288
    assert exp.MAX_UPDATES == 191
    assert exp.ADDITIONAL_TARGETS == 100_139_008
    assert exp.CUMULATIVE_TARGETS == 350_748_672
    assert exp.INCREMENTAL_BATCHES * exp.VALIDATION_B * exp.T == 262_144
    assert [u for u in range(1, 192) if exp.pass_count(u) == 3] == [32, 64, 96, 128, 160]


def test_config_and_architecture_manifest_physically_remove_b2_recurrence():
    config = json.loads(exp.CONFIG_PATH.read_text())
    assert config["frozen_2d2d_commit"] == exp.FROZEN_COMMIT
    assert config["source_checkpoint_sha256"] == exp.SOURCE_SHA256
    assert config["architecture"]["b2"]["local_window"] == 32
    assert config["architecture"]["b2"]["recurrent"] is False
    assert config["architecture"]["b3"]["maximum_recurrent_entries"] == 960
    manifest = exp.architecture_manifest()
    assert "B11_to_B2" not in manifest["links"]
    assert manifest["incremental_b11_raw_residual_capacity"] == 0
    assert manifest["forbidden_modules_absent"]["g_rec_b2"]


def test_memory_accounting_has_exactly_two_raw_rings():
    report = exp.memory_accounting()
    b1 = 1 * 768 * 2 * 2
    b2 = 31 * 768 * 2 * 2
    b3 = 63 * 768 * 2 * 2
    rings = 2 * 1023 * 768 * 2
    upper = 9 * 1023 * 768 * 2 * 2
    expected = b1 + b2 + b3 + rings + upper
    assert report["B1"]["total_experimental_inference_state_bytes"] == expected
    assert report["B1"]["two_recurrent_raw_state_rings_bytes"] == rings


def test_preregistered_classification_thresholds():
    assert exp.classify_result(incremental(3.0, 3.0004, 3.0003)) == (
        "B10→B3 W64 RECURRENT LINK ESTABLISHES POSITIVE UTILITY")
    assert exp.classify_result(incremental(3.0, 3.002, 3.0015, wins=180)) == (
        "B10→B3 W64 RECURRENT LINK STRONGLY ESTABLISHES UTILITY")
    assert exp.classify_result(incremental(3.0, 3.0001, 3.0002, wins=120)) == (
        "B10→B3 W64 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY")
    assert exp.classify_result(incremental(3.0, 3.00001, 2.99999, wins=128)) == (
        "B10→B3 W64 RECURRENT LINK REMAINS NEAR ZERO")
    assert exp.classify_result(incremental(3.01, 3.0, 3.02, wins=100)) == (
        "B10→B3 W64 RECURRENT LINK IS HARMFUL")


def test_frozen_2d2e_matched_stream_is_read_from_final_commit():
    original = exp.subprocess.check_output
    exp.subprocess.check_output = lambda *args, **kwargs: json.dumps(
        {"scientific_global_stream_sha256": exp.SOURCE_NEXT_STREAM_SHA256}
    ).encode()
    try:
        reference = exp.frozen_2d2e_batch_manifest()
    finally:
        exp.subprocess.check_output = original
    assert reference["commit"] == exp.FROZEN_2D2E_FINAL_COMMIT
    assert reference["path"] == exp.FROZEN_2D2E_BATCH_MANIFEST
    assert reference["payload"]["scientific_global_stream_sha256"] == (
        exp.SOURCE_NEXT_STREAM_SHA256
    )


def test_smoke_and_final_persistence_are_protocol_complete():
    smoke_source = inspect.getsource(exp.run_smoke)
    assert "save_checkpoint(" in smoke_source
    assert "load_checkpoint_runtime(" in smoke_source
    assert '"writer_gradient_after_gate_opens"' in smoke_source
    persist_source = inspect.getsource(exp.save_run_checkpoint)
    assert "checkpoint_persist_lock()" in persist_source
    assert "persistent_copy_sha_verified" in persist_source
    assert "local_stage" in persist_source


def test_master_protocol_minimum_artifact_names_are_required():
    required = set(exp.REQUIRED_ARTIFACTS)
    assert {
        "FINAL_REPORT.md",
        "FINAL_AUDIT.json",
        "result_summary.json",
        "attention_diagnostics.json",
        "temporal_gradient_diagnostics.json",
        "incremental_validation.json",
        "incremental_cache_audit.json",
        "stability_8pass.json",
        "UNATTENDED_FINAL_HANDOFF.md",
    } <= required


def test_matched_2d2e_trajectory_comparison_is_signed_2d2f_minus_2d2e():
    current = {
        "b3_recurrent_gain": 0.003,
        "b3_sequence_gap": 0.002,
        "tanh_g_rec_b3": 0.04,
    }
    frozen = {
        "b3_recurrent_gain": 0.001,
        "b3_sequence_gap": -0.001,
        "tanh_g_rec_b3": 0.03,
    }
    row = exp.trajectory_comparison_to_2d2e(current, frozen)
    assert abs(row["gain_2d2f_minus_2d2e"] - 0.002) < 1e-15
    assert abs(row["sequence_gap_2d2f_minus_2d2e"] - 0.003) < 1e-15
    assert abs(row["tanh_g_rec_b3_2d2f_minus_2d2e"] - 0.01) < 1e-15
