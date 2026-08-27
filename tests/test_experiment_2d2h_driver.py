import json
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_master_four_gpu_stop_audit_targets_only_current_pod(tmp_path):
    manifest = {
        "schema": "parallel_2d2_runpod_stop_capability_v1",
        "authenticated_list_probe": True,
        "stop_credential_available": True,
        "secret_recorded": False,
        "authenticated_pod_identity_response": {
            "id": "7i2zyd53ytspwz",
            "name": "empirical_tan_panda",
            "gpuCount": 4,
            "runtimeStatus": "running",
            "networkVolumeId": exp.PERSISTENT_VOLUME_IDENTITY,
        },
        "pod_id": "7i2zyd53ytspwz",
        "pod_name": "empirical_tan_panda",
        "gpu_count": 4,
        "volume_id": exp.PERSISTENT_VOLUME_IDENTITY,
        "exact_stop_target": "7i2zyd53ytspwz",
        "pod_delete_forbidden": True,
        "pod_delete_authorized": False,
        "persistent_volume_delete_authorized": False,
        "passed": True,
    }
    path = tmp_path / "AUTO_STOP_PREFLIGHT.json"
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(
        stop_audit_path=str(path),
        pod_id="7i2zyd53ytspwz",
        pod_name="empirical_tan_panda",
        stop_authenticated=True,
    )
    audit = exp.authenticated_stop_audit(args)
    assert audit["driver_passed"]
    assert all(audit["driver_checks"].values())


def test_active_training_progress_does_not_read_removed_b1_gate():
    source = inspect.getsource(exp.run_train)
    assert "tanh_g_rec_b1" not in source
    assert "b1=REMOVED" in source
    assert "require_git(clean=False)" in source
    assert "training has non-result worktree changes" in source


def test_master_artifact_names_and_local_first_persistence_are_preregistered():
    assert {
        "FINAL_REPORT.md",
        "attention_diagnostics.json",
        "temporal_gradient_diagnostics.json",
    }.issubset(exp.REQUIRED_ARTIFACTS)
    source = inspect.getsource(exp.save_run_checkpoint)
    assert source.index("local_verification = save_checkpoint") < source.index(
        "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)"
    )
    assert 'verification["persisted_under_global_lock"] = True' in source


def test_matched_2d2d_cursor_audit_requires_exact_hashes(tmp_path):
    expected_batch = "a" * 64
    expected_stream = "b" * 64
    audit_path = tmp_path / "matched_2d2d_data_audit.json"
    audit_path.write_text(json.dumps({
        "same_first_batch": True,
        "same_target_stream": True,
        "frozen_2d2d_cursor_reference": {
            "passed": True,
            "cursor_hashes": {
                "96": {
                    "kind": "scientific",
                    "next_batch_sha256": expected_batch,
                    "next_stream_sha256": expected_stream,
                }
            },
        },
        "cursor_comparisons": {},
    }))
    exp.record_matched_2d2d_cursor(
        tmp_path,
        96,
        verification={
            "next_global_batch_sha256": expected_batch,
            "next_global_batch_stream_sha256": expected_stream,
        },
    )
    observed = json.loads(audit_path.read_text())
    assert observed["passed_so_far"]
    assert observed["cursor_comparisons"]["96"]["exact"]
