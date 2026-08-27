import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_2d2i as exp  # noqa: E402


def incremental(real, off, shuffled, wins=140, count=256):
    return {
        "true_b4_recurrent_gain": off - real,
        "true_b4_sequence_gap": shuffled - real,
        "all_real_vs_b4_off_sequences": {
            "wins": wins, "losses": count - wins, "ties": 0, "count": count,
        },
        "all_real_vs_b4_shuffled_sequences": {
            "wins": wins, "losses": count - wins, "ties": 0, "count": count,
        },
    }


def test_protocol_arithmetic_and_cadence():
    assert exp.SOURCE_PARAMETERS == 124_475_907
    assert exp.TOTAL_PARAMETERS == 124_475_908
    assert exp.SOURCE_TARGETS == 350_748_672
    assert exp.GLOBAL_TARGETS == 524_288
    assert exp.MAX_UPDATES == 191
    assert exp.ADDITIONAL_TARGETS == 100_139_008
    assert exp.CUMULATIVE_TARGETS == 450_887_680
    assert exp.INCREMENTAL_BATCHES * exp.VALIDATION_B * exp.T == 262_144
    assert [u for u in range(1, 192) if exp.pass_count(u) == 3] == [32, 64, 96, 128, 160]
    assert exp.pass_weights(31) == (0.25, 0.75)
    assert exp.pass_weights(32) == (0.20, 0.40, 0.40)


def test_preregistered_config_matches_driver():
    config = json.loads(exp.CONFIG_PATH.read_text())
    assert config["frozen_2d2e_commit"] == exp.FROZEN_COMMIT
    assert config["source_checkpoint_sha256"] == exp.SOURCE_SHA256
    assert config["source_2d2e"]["raw_g_rec_b3"] == exp.SOURCE_B3_GATE_RAW
    assert config["architecture"]["b1"]["local_window"] == 2
    assert config["architecture"]["b2"]["local_window"] == 32
    assert config["architecture"]["b3"]["local_window"] == 64
    assert config["architecture"]["b4"]["local_window"] == 128
    assert config["architecture"]["b4"]["recurrent_min_lag"] == 128
    assert config["architecture"]["b4"]["maximum_recurrent_entries"] == 896
    assert config["architecture"]["new_parameters_vs_2d2e"] == ["g_rec_b4"]
    assert config["training"]["additional_targets"] == exp.ADDITIONAL_TARGETS
    assert config["training"]["cumulative_2d2_targets"] == exp.CUMULATIVE_TARGETS


def test_memory_accounting_exact_bf16_geometry():
    report = exp.memory_accounting()
    b1 = 1 * 768 * 2 * 2
    b2 = 31 * 768 * 2 * 2
    b3 = 63 * 768 * 2 * 2
    b4 = 127 * 768 * 2 * 2
    rings = 4 * 1023 * 768 * 2
    upper = 8 * 1023 * 768 * 2 * 2
    expected = b1 + b2 + b3 + b4 + rings + upper
    assert report["B1"]["total_experimental_inference_state_bytes"] == expected
    assert report["B1"]["b4_local_kv_bytes"] == b4
    assert report["B1"]["four_recurrent_raw_state_rings_bytes"] == rings
    assert report["B1"]["b5_b12_ordinary_kv_bytes"] == upper
    assert report["B1"]["saving_bytes_vs_standard_gpt2"] > 0
    assert report["B64"]["total_experimental_inference_state_bytes"] == 64 * expected


def test_preregistered_classification_thresholds():
    assert exp.classify_result(incremental(3.0, 3.0004, 3.0003)) == (
        "B9→B4 W128 RECURRENT LINK ESTABLISHES POSITIVE UTILITY")
    assert exp.classify_result(incremental(3.0, 3.002, 3.0015, wins=180)) == (
        "B9→B4 W128 RECURRENT LINK STRONGLY ESTABLISHES UTILITY")
    assert exp.classify_result(incremental(3.0, 3.0001, 3.02, wins=120)) == (
        "B9→B4 W128 RECURRENCE IS SEQUENCE-SPECIFIC BUT DOES NOT ESTABLISH UTILITY")
    assert exp.classify_result(incremental(3.0, 3.00001, 2.99999, wins=128)) == (
        "B9→B4 W128 RECURRENT LINK REMAINS NEAR ZERO")
    assert exp.classify_result(incremental(3.01, 3.0, 3.0, wins=100)) == (
        "B9→B4 W128 RECURRENT LINK IS HARMFUL")


def test_required_protocol_artifacts_and_controls_are_registered():
    required = {
        "FINAL_REPORT.md", "FINAL_AUDIT.json", "result_summary.json",
        "source_manifest.json", "architecture_manifest.json", "parameter_manifest.json",
        "training_metrics.jsonl", "milestone_validation.json", "paired_controls.json",
        "gate_diagnostics.json", "attention_diagnostics.json",
        "temporal_gradient_diagnostics.json", "incremental_validation.json",
        "incremental_cache_audit.json", "memory_accounting.json", "stability_8pass.json",
        "performance.json", "checkpoint_manifest.json", "commands_and_runtime.json",
        "storage_cleanup_manifest.json", "HEARTBEAT.json", "UNATTENDED_FINAL_HANDOFF.md",
    }
    assert required <= set(exp.REQUIRED_ARTIFACTS)
    assert set(exp.INCREMENTAL_CONTROLS) == {
        "all_real", "b4_off", "b4_shuffled", "b4_full_counterfactual", "all_shuffled"
    }


def test_checkpoint_and_storage_policy():
    assert exp.SCIENTIFIC_CHECKPOINTS == (96, 191)
    assert exp.FORCED_RESTART_UPDATE == 96
    assert exp.RECOVERY_CHECKPOINTS == ()
    assert exp.OUTPUT_NAME == "experiment_2d2i_b4_w128_b9_recurrent"
    assert exp.PERSISTENT_VOLUME_IDENTITY == "yhzyb27fb5"
    assert exp.VOLUME_CAPACITY_DECIMAL_BYTES == 100_000_000_000
    assert exp.PERSISTENT_SAFETY_MARGIN_BYTES == 8_000_000_000
    assert exp.MASTER_STORAGE_PREFLIGHT.name == "storage_preflight.json"
    assert str(exp.PERSISTENT_FINAL_CHECKPOINT) == (
        "/workspace/exp2d2i_run/checkpoints/scientific_update_0191.pt"
    )
    source = Path(exp.__file__).read_text()
    audit = source[source.index("def workspace_mount_audit"):source.index(
        "def authenticated_stop_audit"
    )]
    assert "logical_quota_free_decimal_bytes" in audit
    assert "persistent_projection_with_margin_fits" in audit
    assert "shutil.disk_usage" not in audit
    assert "df -" not in audit
