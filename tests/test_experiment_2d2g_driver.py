import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_2d2g as exp  # noqa: E402


def incremental(real, off, shuffled, off_wins=140, shuffled_wins=140):
    return {
        "true_b3_recurrent_gain": off - real,
        "true_b3_sequence_gap": shuffled - real,
        "real_vs_off_sequences": {"wins": off_wins},
        "real_vs_shuffled_sequences": {"wins": shuffled_wins},
    }


def test_protocol_arithmetic_and_stage_counts():
    assert exp.STAGE_A_PARAMETERS == 124_475_905
    assert exp.STAGE_B_PARAMETERS == 124_475_906
    assert exp.GLOBAL_TARGETS == 524_288
    assert exp.UPDATES_PER_STAGE == 191
    assert exp.TARGETS_PER_STAGE == 100_139_008
    assert exp.INCREMENTAL_BATCHES * exp.VALIDATION_B * exp.T == 262_144
    assert [u for u in range(1, 192) if exp.pass_count(u) == 3] == [32, 64, 96, 128, 160]


def test_preregistered_config_matches_driver():
    config = json.loads(exp.CONFIG_PATH.read_text())
    assert config["source_2d2b"]["sha256"] == exp.SOURCE_SHA256
    assert config["stage_a"]["architecture"]["parameters"] == exp.STAGE_A_PARAMETERS
    assert config["stage_b"]["architecture"]["parameters"] == exp.STAGE_B_PARAMETERS
    assert config["stage_b"]["architecture"]["b2_local_window"] == 1024
    assert config["stage_b"]["architecture"]["b11_to_b2_recurrence"] is False
    assert config["stage_b"]["architecture"]["b11_raw_ring_present"] is False
    assert config["stage_b"]["architecture"]["b3_max_recurrent_entries"] == 960


def test_mandatory_restart_boundary_is_enforced():
    exp.require_segment_boundary(0, 96)
    exp.require_segment_boundary(96, 191, resumed_pid=-1)
    with pytest.raises(SystemExit):
        exp.require_segment_boundary(0, 191)
    with pytest.raises(SystemExit):
        exp.require_segment_boundary(20, 96)


def test_exact_matched_data_cursors_are_preregistered():
    assert exp.expected_cursor("a", 96) == (
        exp.STAGE_A_UPDATE96_BATCH,
        exp.STAGE_A_UPDATE96_STREAM,
    )
    assert exp.expected_cursor("a", 191) == (
        exp.STAGE_A_FINAL_BATCH,
        exp.STAGE_A_FINAL_STREAM,
    )
    assert exp.expected_cursor("b", 96) == (
        exp.STAGE_B_UPDATE96_BATCH,
        exp.STAGE_B_UPDATE96_STREAM,
    )
    assert exp.expected_cursor("b", 191) == (
        exp.STAGE_B_FINAL_BATCH,
        exp.STAGE_B_FINAL_STREAM,
    )


def test_milestone_key_audit_requires_exact_preregistered_set():
    exact = {str(update): {} for update in exp.MILESTONES}
    assert exp.milestone_key_audit(exact)["passed"]
    with_duplicate = {**exact, "191_final": {}}
    audit = exp.milestone_key_audit(with_duplicate)
    assert not audit["passed"]
    assert audit["unexpected_keys"] == ["191_final"]


def test_classification_thresholds():
    assert exp.classify_result(incremental(3.0, 3.0004, 3.0003)) == "POSITIVE UTILITY ESTABLISHED"
    assert exp.classify_result(
        incremental(3.0, 3.002, 3.0015, 180, 180)
    ) == "STRONG POSITIVE"
    assert exp.classify_result(
        incremental(3.0, 3.0002, 3.0001, 128, 140)
    ) == "SEQUENCE-SPECIFIC BUT NOT ESTABLISHED"
    assert exp.classify_result(
        incremental(3.0, 2.9998, 3.0001, 100, 140)
    ) == "SEQUENCE-SPECIFIC BUT NOT ESTABLISHED"
    assert exp.classify_result(
        incremental(3.0, 2.9998, 2.9999, 100, 100)
    ) == "HARMFUL"
    assert exp.classify_result(
        incremental(3.0, 3.002, 3.0015, 180, 180), integrity=False
    ) == "INVALID"


def test_memory_accounting_has_no_b11_ring():
    report = exp.memory_accounting()
    assert report["B1"]["b11_ring_bytes"] == 0
    assert report["B64"]["b11_ring_bytes"] == 0
    assert report["B64"]["total_inference_state_bytes"] == 64 * report["B1"]["total_inference_state_bytes"]


def test_final_persistence_requires_ephemeral_to_workspace_and_exact_shared_lock():
    valid = exp.validate_final_persistence_paths(
        "/tmp/parallel_2d2_ephemeral/2d2g/stage_b_scientific_update_0191.pt",
        "/workspace/exp2d2g_run/checkpoints",
        "/workspace/parallel_2d2_master/locks/checkpoint_persist.lock",
    )
    assert valid["passed"]
    assert not exp.validate_final_persistence_paths(
        "/workspace/unsafe_local.pt",
        "/workspace/exp2d2g_run/checkpoints",
        "/workspace/parallel_2d2_master/locks/checkpoint_persist.lock",
    )["passed"]
    assert not exp.validate_final_persistence_paths(
        "/tmp/parallel_2d2_ephemeral/2d2g/local.pt",
        "/workspace/exp2d2g_run/checkpoints",
        "/tmp/private.lock",
    )["passed"]


def test_required_artifact_inventory_is_strict(tmp_path):
    for name in exp.REQUIRED_TRAINING_ARTIFACTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    assert exp.required_artifact_inventory(tmp_path)["passed"]
    (tmp_path / "FINAL_REPORT.md").write_text("")
    assert not exp.required_artifact_inventory(tmp_path)["passed"]


def test_smoke_cli_requires_explicit_ephemeral_checkpoint_directory():
    parser = exp.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "smoke-b",
                "--output-dir", "/workspace/results",
                "--pod-id", "pod",
                "--pod-name", "name",
                "--stage-a-checkpoint", "/tmp/stage-a.pt",
                "--data-root", "/workspace/data",
            ]
        )
    args = parser.parse_args(
        [
            "smoke-b",
            "--output-dir", "/workspace/results",
            "--pod-id", "pod",
            "--pod-name", "name",
            "--stage-a-checkpoint", "/tmp/stage-a.pt",
            "--data-root", "/workspace/data",
            "--checkpoint-dir", "/tmp/parallel_2d2_ephemeral/2d2g/smoke",
        ]
    )
    assert args.checkpoint_dir.endswith("/2d2g/smoke")
