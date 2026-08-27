from contextlib import nullcontext
import copy
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


def build_recovery_provenance(tmp_path, monkeypatch):
    ephemeral = tmp_path / "ephemeral"
    checkpoint_dir = ephemeral / "2d2g" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    output = tmp_path / "output"
    output.mkdir()
    provenance = tmp_path / "sealed-provenance"
    provenance.mkdir()
    monkeypatch.setattr(exp, "EPHEMERAL_ROOT", ephemeral)

    manifest = {"stage_a": {}, "stage_b": {}}
    sealed_checkpoints = {}
    for update in (96, 191):
        checkpoint = checkpoint_dir / f"stage_a_scientific_update_{update:04d}.pt"
        checkpoint.write_bytes((f"checkpoint-{update}-" * 7).encode())
        sha = exp.file_sha256(checkpoint)
        checkpoint.with_suffix(checkpoint.suffix + ".sha256").write_text(
            f"{sha}  {checkpoint.name}\n"
        )
        verification = {"passed": True, "strict_model_load": True, "updates": True}
        exp.durable_json(
            checkpoint.with_suffix(checkpoint.suffix + ".verification.json"),
            verification,
        )
        batch, stream = exp.expected_cursor("a", update)
        manifest["stage_a"][str(update)] = {
            "checkpoint": str(checkpoint.resolve()),
            "sha256": sha,
            "bytes": checkpoint.stat().st_size,
            "next_global_batch_sha256": batch,
            "next_global_batch_stream_sha256": stream,
            "strict_reopen": verification,
        }
        sealed_checkpoints[update] = {
            "filename": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": sha,
        }

    stage_a_match = {
        "reference": "2D2D",
        "source_batch": exp.SOURCE_NEXT_BATCH,
        "source_stream": exp.SOURCE_NEXT_STREAM,
        "passed": True,
    }
    for update in (96, 191):
        batch, stream = exp.expected_cursor("a", update)
        stage_a_match[f"update_{update}"] = {
            "observed_next_global_batch_sha256": batch,
            "observed_next_global_batch_stream_sha256": stream,
            "expected": [batch, stream],
            "exact": True,
        }

    runtime = {
        "commands": [
            {
                "command": "scripts/experiment_2d2g.py preflight --output-dir old",
                "kind": "preflight",
            },
            {
                "command": "scripts/experiment_2d2g.py train-a --end-update 96",
                "stage": "a",
                "start_update": 1,
                "end_update": 96,
                "pid": 111,
                "wall_seconds": 100.0,
            },
            {
                "command": (
                    "scripts/experiment_2d2g.py train-a --resume stage-a-96 "
                    "--end-update 191"
                ),
                "stage": "a",
                "start_update": 97,
                "end_update": 191,
                "pid": 222,
                "wall_seconds": 101.0,
            },
        ]
    }
    restart_required = {
        "checkpoint": copy.deepcopy(manifest["stage_a"]["96"]),
        "saved_process_id": 111,
        "status": "MANDATORY_FRESH_PROCESS_REQUIRED",
    }
    restart_batch, restart_stream = exp.expected_cursor("a", 96)
    forced_restart = {
        "checkpoint_process_id": 111,
        "resumed_process_id": 222,
        "fresh_process": True,
        "passed": True,
        "next_global_batch_sha256": restart_batch,
        "expected_next_global_batch_sha256": restart_batch,
        "next_global_batch_stream_sha256": restart_stream,
        "expected_next_global_batch_stream_sha256": restart_stream,
    }
    final_checkpoint = manifest["stage_a"]["191"]["checkpoint"]
    heartbeat = {
        "experiment": exp.EXPERIMENT,
        "stage": "a",
        "phase": "training",
        "local_update": exp.UPDATES_PER_STAGE,
        "targets": exp.TARGETS_PER_STAGE,
        "last_loss": 3.0,
        "g_rec_b1": 0.1,
        "g_rec_b3": None,
        "pid": 222,
        "checkpoint": final_checkpoint,
        "timestamp": 1.0,
    }
    json_payloads = {
        "checkpoint_manifest.json": manifest,
        "commands_and_runtime.json": runtime,
        "stage_a_data_match.json": stage_a_match,
        "stage_a_restart_required_update_96.json": restart_required,
        "stage_a_forced_restart_update_96.json": forced_restart,
        "HEARTBEAT.json": heartbeat,
    }
    for name, payload in json_payloads.items():
        exp.durable_json(provenance / name, payload)

    metric_lines = []
    for update in range(1, exp.UPDATES_PER_STAGE + 1):
        metric_lines.append(
            json.dumps(
                {
                    "stage": "a",
                    "local_update": update,
                    "processed_stage_targets": update * exp.GLOBAL_TARGETS,
                    "wall_seconds": 1.0,
                    "pass_losses": [3.0],
                    "tanh_g_rec_b1": 0.1,
                },
                sort_keys=True,
            )
        )
    (provenance / "stage_a_training_metrics.jsonl").write_text(
        "\n".join(metric_lines) + "\n"
    )
    for name in exp.RECOVERY_STAGE_A_SUPPORT_FILES:
        (output / name).write_bytes((provenance / name).read_bytes())

    expected_hashes = {
        name: exp.file_sha256(provenance / name)
        for name in exp.RECOVERY_PROVENANCE_SHA256
    }
    monkeypatch.setattr(exp, "RECOVERY_PROVENANCE_SHA256", expected_hashes)
    monkeypatch.setattr(exp, "RECOVERY_STAGE_A_CHECKPOINTS", sealed_checkpoints)
    return output, provenance, manifest, runtime, stage_a_match


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


def test_recovery_provenance_cli_is_double_locked(monkeypatch, tmp_path):
    monkeypatch.delenv("MASTER_RECOVERY_MODE", raising=False)
    assert exp.recovery_provenance_dir(None) is None
    with pytest.raises(SystemExit, match="legal only"):
        exp.recovery_provenance_dir(str(tmp_path))
    monkeypatch.setenv("MASTER_RECOVERY_MODE", "1")
    with pytest.raises(SystemExit, match="requires"):
        exp.recovery_provenance_dir(None)
    assert exp.recovery_provenance_dir(str(tmp_path)) == tmp_path.resolve()


def test_exact_recovery_provenance_restores_and_appends_without_reset(
    monkeypatch, tmp_path
):
    output, provenance, manifest, original_runtime, stage_a_match = (
        build_recovery_provenance(tmp_path, monkeypatch)
    )
    exp.durable_json(output / "checkpoint_manifest.json", {"stage_a": {}, "stage_b": {}})
    exp.durable_json(
        output / "commands_and_runtime.json",
        {"commands": [{"command": "failed preflight", "kind": "preflight"}]},
    )
    exp.durable_json(
        output / "stage_a_data_match.json",
        {"reference": "2D2D", "passed": True},
    )
    cleanup = {
        "scientific_source_removed": False,
        "cleanup_actions": [{"path": "/tmp/orphan.pt", "removed": True}],
    }
    exp.durable_json(output / "storage_cleanup_manifest.json", cleanup)

    recovery = exp.require_exact_stage_a_recovery_provenance(output, provenance)
    first = exp.publish_preflight_bookkeeping(
        output,
        "scripts/experiment_2d2g.py preflight --recovery-provenance-dir sealed",
        {"reference": "unused"},
        recovery,
    )

    assert exp.read_json(output / "checkpoint_manifest.json") == manifest
    assert exp.read_json(output / "stage_a_data_match.json") == stage_a_match
    commands = exp.read_json(output / "commands_and_runtime.json")["commands"]
    assert commands[:3] == original_runtime["commands"]
    assert [row["kind"] for row in commands[3:]] == ["recovery_preflight"]
    assert exp.read_json(output / "storage_cleanup_manifest.json") == cleanup
    assert first["passed"]
    assert first["source"]["runtime_wall_seconds_reconstructed"] is False
    assert first["restored"]["original_runtime_rows_preserved_exactly"]

    second = exp.publish_preflight_bookkeeping(
        output,
        "scripts/experiment_2d2g.py preflight --recovery-provenance-dir sealed",
        {"reference": "unused"},
        recovery,
    )
    commands = exp.read_json(output / "commands_and_runtime.json")["commands"]
    assert commands[:3] == original_runtime["commands"]
    assert [row["kind"] for row in commands[3:]] == [
        "recovery_preflight",
        "recovery_preflight",
    ]
    assert len(second["attempts"]) == 2


def test_recovery_provenance_rejects_checkpoint_or_support_mutation(
    monkeypatch, tmp_path
):
    output, provenance, manifest, _, _ = build_recovery_provenance(
        tmp_path, monkeypatch
    )
    checkpoint = Path(manifest["stage_a"]["191"]["checkpoint"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"mutation")
    with pytest.raises(SystemExit, match="manifest/checkpoint mismatch"):
        exp.require_exact_stage_a_recovery_provenance(output, provenance)

    output, provenance, _, _, _ = build_recovery_provenance(
        tmp_path / "support-case", monkeypatch
    )
    (output / "HEARTBEAT.json").write_text("{}\n")
    with pytest.raises(SystemExit, match="hash mismatch"):
        exp.require_exact_stage_a_recovery_provenance(output, provenance)


def test_existing_recovery_bookkeeping_refuses_post_preflight_rows(
    monkeypatch, tmp_path
):
    output, provenance, _, _, _ = build_recovery_provenance(tmp_path, monkeypatch)
    recovery = exp.require_exact_stage_a_recovery_provenance(output, provenance)
    exp.publish_preflight_bookkeeping(
        output, "recovery preflight one", {"reference": "unused"}, recovery
    )
    runtime = exp.read_json(output / "commands_and_runtime.json")
    runtime["commands"].append({"kind": "disposable_stage_b_smoke"})
    exp.durable_json(output / "commands_and_runtime.json", runtime)
    with pytest.raises(SystemExit, match="post-preflight work"):
        exp.publish_preflight_bookkeeping(
            output, "recovery preflight two", {"reference": "unused"}, recovery
        )


def test_finalize_recovery_integrity_cannot_fail_open_after_tampering(
    monkeypatch, tmp_path
):
    output, provenance, _, _, _ = build_recovery_provenance(tmp_path, monkeypatch)
    recovery = exp.require_exact_stage_a_recovery_provenance(output, provenance)
    preflight = {"recovery_stage_a_provenance": recovery["audit"]}
    exp.durable_json(output / "preflight_audit.json", preflight)
    exp.publish_preflight_bookkeeping(
        output, "recovery preflight", {"reference": "unused"}, recovery
    )
    monkeypatch.setenv("MASTER_RECOVERY_MODE", "1")
    manifest = exp.read_json(output / "checkpoint_manifest.json")
    match = exp.read_json(output / "stage_a_data_match.json")
    runtime = exp.read_json(output / "commands_and_runtime.json")

    baseline = exp.recovery_finalize_integrity(
        output, preflight, runtime, manifest, match
    )
    assert baseline["required"] and baseline["passed"]

    lost_row_runtime = {"commands": runtime["commands"][:3]}
    lost_row = exp.recovery_finalize_integrity(
        output, preflight, lost_row_runtime, manifest, match
    )
    assert lost_row["required"] and not lost_row["passed"]
    assert not lost_row["checks"]["recovery_rows_canonical"]

    audit_path = output / "recovery_stage_a_provenance_audit.json"
    exact_audit = audit_path.read_bytes()
    tampered_audit = exp.read_json(audit_path)
    tampered_audit["source"]["identity"]["source_sha256"][
        "checkpoint_manifest.json"
    ] = "0" * 64
    exp.durable_json(audit_path, tampered_audit)
    changed_source = exp.recovery_finalize_integrity(
        output, preflight, runtime, manifest, match
    )
    assert not changed_source["passed"]
    assert not changed_source["checks"]["source_identity_exact"]
    audit_path.write_bytes(exact_audit)

    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["stage_a"].pop("96")
    manifest_result = exp.recovery_finalize_integrity(
        output, preflight, runtime, changed_manifest, match
    )
    assert not manifest_result["passed"]
    assert not manifest_result["checks"]["current_stage_a_manifest_exact"]

    audit_path.unlink()
    missing_audit = exp.recovery_finalize_integrity(
        output, preflight, runtime, manifest, match
    )
    assert missing_audit["required"] and not missing_audit["passed"]
    assert not missing_audit["checks"]["audit_present_regular"]
