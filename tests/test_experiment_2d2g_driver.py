from contextlib import nullcontext
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_2d2g as exp  # noqa: E402


class MixedDeviceEvaluationModel(torch.nn.Module):
    """Mimic the exact-continuation wrapper/device split used by Stage B."""

    def __init__(self):
        super().__init__()
        self.g_rec = torch.nn.Parameter(torch.zeros(()))
        self.g_rec_b3 = torch.nn.Parameter(torch.zeros(()))
        self.base = torch.nn.Module()
        self.base.transformer = torch.nn.Module()
        self.base.transformer.wte = torch.nn.Embedding(16, 4, device="meta")
        self.observed = []

    @property
    def recurrent_scale_b1(self):
        return self.g_rec.tanh()

    @property
    def recurrent_scale_b3(self):
        return self.g_rec_b3.tanh()

    def forward_multi_pass(
        self,
        tokens,
        targets=None,
        b3_recurrent_permutation=None,
        **kwargs,
    ):
        expected = self.base.transformer.wte.weight.device
        assert tokens.device == expected
        assert targets is not None and targets.device == expected
        if b3_recurrent_permutation is not None:
            assert b3_recurrent_permutation.device == expected
        self.observed.append(
            (tokens.device, targets.device, b3_recurrent_permutation is not None)
        )
        return {"loss": torch.tensor(1.0)}


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


def test_validation_batch_uses_embedding_device_not_first_wrapper_parameter():
    model = MixedDeviceEvaluationModel()
    x = torch.zeros((2, 3), dtype=torch.long)
    y = torch.ones((2, 3), dtype=torch.long)

    moved_x, moved_y = exp.validation_batch_to_model_device(model, x, y)

    assert next(model.parameters()).device.type == "cpu"
    assert moved_x.device.type == "meta"
    assert moved_y.device.type == "meta"
    assert moved_x.shape == x.shape and moved_y.shape == y.shape
    assert moved_x.dtype == x.dtype and moved_y.dtype == y.dtype
    assert x.device.type == "cpu" and y.device.type == "cpu"


def test_checkpoint_gate_exact_is_device_safe_without_weakening_equality():
    cpu_gate = torch.tensor(0.125, dtype=torch.float32)
    assert exp.checkpoint_gate_exact(cpu_gate, cpu_gate.clone())
    assert not exp.checkpoint_gate_exact(cpu_gate, torch.tensor(0.25))
    assert not exp.checkpoint_gate_exact(cpu_gate, cpu_gate.to(torch.float64))
    assert not exp.checkpoint_gate_exact(cpu_gate, cpu_gate.reshape(1))
    if torch.cuda.is_available():
        assert exp.checkpoint_gate_exact(cpu_gate, cpu_gate.cuda())


def test_smoke_parallel_evaluation_uses_gpt_input_device(monkeypatch):
    class CpuValidationLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def next_batch(self):
            tokens = torch.arange(exp.VALIDATION_B * exp.T, dtype=torch.long).view(
                exp.VALIDATION_B, exp.T
            )
            return tokens.remainder(16), tokens.add(1).remainder(16)

    model = MixedDeviceEvaluationModel()
    assert next(model.parameters()).device.type == "cpu"
    assert exp.model_execution_device(model).type == "meta"
    monkeypatch.setattr(exp.d1, "ExplicitShardLoader", CpuValidationLoader)
    monkeypatch.setattr(exp.torch, "autocast", lambda **_kwargs: nullcontext())

    result = exp.evaluate_parallel(model, "unused-validation.npy", batches=1)

    assert len(model.observed) == len(exp.INCREMENTAL_CONTROLS)
    assert all(
        tokens.type == "meta" and targets.type == "meta"
        for tokens, targets, _ in model.observed
    )
    assert sum(shuffled for _, _, shuffled in model.observed) == 1
    assert all(
        row["validation_targets"] == exp.VALIDATION_B * exp.T
        for row in result["controls"].values()
    )
    assert result["b3_gain"] == 0.0
    assert result["b3_sequence_gap"] == 0.0


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
