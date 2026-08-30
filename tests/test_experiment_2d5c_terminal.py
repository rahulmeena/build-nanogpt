"""Stdlib-only terminal sealing tests for Experiment 2D5C."""

from __future__ import annotations

import ast
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FINALIZER_PATH = SCRIPTS / "experiment_2d5c_finalizer.py"
DRIVER_PATH = SCRIPTS / "experiment_2d5c.py"
sys.path.insert(0, str(SCRIPTS))
import experiment_2d5c_finalizer as finalizer  # noqa: E402


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    rows = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(rows) != 1:
        raise AssertionError(f"expected one function {name}, got {len(rows)}")
    return rows[0]


def write_json(path: Path, value, mode: int | None = None) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if mode is not None:
        path.chmod(mode)


def contrast(point: float) -> dict:
    return {
        "point_estimate": point,
        "lower_95": point - 0.001,
        "upper_95": point + 0.001,
        "positive_per_sequence_differences": 1_500,
        "paired_standard_error": 0.0001,
    }


class TerminalFixture:
    def __init__(self, root: Path):
        self.root = root
        self.provisional = root / "SCIENTIFIC_AUDIT_PRETAG.json"
        self.summary = root / "SCIENTIFIC_RESULT_SUMMARY.json"
        self.representation = root / "REPRESENTATION_PRESSURE_DIAGNOSTICS.json"
        self.git = root / "GIT_VERIFICATION.json"
        self.authorization = root / "RUNPOD_AUTHORIZATION.json"
        self.trigger = root / "RUNPOD_TRIGGER.json"
        self.stop = root / "RUNPOD_STOP_VERIFICATION.json"
        self.final_audit = root / "FINAL_AUDIT.json"
        self.report = root / "EXPERIMENT_2D5C_FINAL_REPORT.md"
        checks = {name: True for name in finalizer.REQUIRED_FINAL_AUDIT_CHECKS}
        checks["git_branch_commit_tag_pushed_verified"] = False
        provisional = {
            "experiment": finalizer.EXPERIMENT,
            "phase": "scientific-results-pretag",
            "checks": checks,
            "critical_scientific_checks_passed": True,
            "passed": False,
            "pending_operational_checks": ["git_branch_commit_tag_pushed_verified"],
        }
        rows = {
            name: contrast((index + 1) / 10_000)
            for index, name in enumerate(finalizer.REQUIRED_REPORT_CONTRASTS)
        }
        longitudinal = {
            str(update): {
                "all_real_ce": 3.0 + update / 100_000,
                "b3_recurrent_gain": 0.01,
                "b3_sequence_gap": 0.02,
                "b5_recurrent_gain": 0.03,
                "b5_sequence_gap": 0.04,
                "combined_recurrent_gain": 0.05,
                "combined_sequence_gap": 0.06,
            }
            for update in (0, 48, 96, 144, 191)
        }
        implementation_commit = "1" * 40
        summary = {
            "experiment": finalizer.EXPERIMENT,
            "classification": "W2/W2 REPRESENTATION-PRESSURE RESULT UNRESOLVED",
            "recommendation": {"recommendation": "NEITHER", "reason": "Evidence remains mixed"},
            "fixed_all_real_ce": 3.01,
            "c_all_real_ce": 3.0,
            "bootstrap": {"contrasts": rows},
            "longitudinal": {"c": longitudinal},
            "recovery": {
                "initial_shock": {"point_estimate": 0.02},
                "recovery_fraction": {"point_estimate": 0.5},
            },
            "bf16_persistent_state": {
                "logical": {"fixed_bytes": 1000, "c_bytes": 700, "reduction_bytes": 300},
                "allocated_unique_storage": {
                    "fixed_bytes": 1100, "c_bytes": 800, "reduction_bytes": 300
                },
            },
            "final_checkpoint": {"path": "/workspace/final.pt", "sha256": "a" * 64},
            "audit": provisional,
            "git_branch": finalizer.BRANCH,
            "git_commit": implementation_commit,
            "git_tag": finalizer.FINAL_TAG,
        }
        representation = {"experiment": finalizer.EXPERIMENT, "passed": True}
        scientific_commit = "2" * 40
        git = {
            "schema": finalizer.GIT_SCHEMA,
            "experiment": finalizer.EXPERIMENT,
            "passed": True,
            "branch": finalizer.BRANCH,
            "implementation_commit": implementation_commit,
            "scientific_results_commit": scientific_commit,
            "origin_branch_commit": scientific_commit,
            "final_tag": finalizer.FINAL_TAG,
            "local_tag_commit": scientific_commit,
            "origin_tag_commit": scientific_commit,
            "branch_push_verified": True,
            "tag_push_verified": True,
            "worktree_clean": True,
        }
        nonce = "3" * 64
        authorization = {
            "schema": finalizer.GUARD_AUTHORIZATION_SCHEMA,
            "experiment": finalizer.EXPERIMENT,
            "action": finalizer.ACTION,
            "pod_id": finalizer.POD_ID,
            "pod_name": finalizer.POD_NAME,
            "gpu_count": finalizer.GPU_COUNT,
            "network_volume_id": finalizer.VOLUME_ID,
            "network_volume_name": finalizer.VOLUME_NAME,
            "network_volume_size_gb": finalizer.VOLUME_SIZE_GB,
            "network_volume_datacenter": finalizer.VOLUME_DATACENTER,
            "volume_mount_path": finalizer.VOLUME_MOUNT_PATH,
            "pod_created_at": "2026-08-30T00:00:00Z",
            "pod_running_last_status_change": "2026-08-30T00:00:01Z",
            "identity_sha256": "5" * 64,
            "authorization_nonce": nonce,
            "exact_stop_command": finalizer.EXACT_STOP_COMMAND,
            "credential_source": {
                "kind": "macOS Keychain generic password",
                "service": "runpod-codex-pod-stopper",
                "account": "rahul",
            },
            "issued_at_utc": "2026-08-30T00:00:02Z",
            "expires_at_utc": "2026-08-31T00:00:02Z",
        }
        write_json(self.provisional, provisional)
        write_json(self.summary, summary)
        write_json(self.representation, representation)
        write_json(self.git, git)
        write_json(self.authorization, authorization, 0o600)
        authorization_sha = finalizer.file_identity(self.authorization)["sha256"]
        trigger = {
            "schema": finalizer.GUARD_TRIGGER_SCHEMA,
            "experiment": finalizer.EXPERIMENT,
            "action": finalizer.ACTION,
            "pod_id": finalizer.POD_ID,
            "pod_name": finalizer.POD_NAME,
            "gpu_count": finalizer.GPU_COUNT,
            "network_volume_id": finalizer.VOLUME_ID,
            "authorization_sha256": authorization_sha,
            "authorization_nonce": nonce,
            "terminal_outcome": "success",
            "exit_code": 0,
            "source": "supervised_child",
            "created_at_utc": "2026-08-30T12:00:00Z",
        }
        write_json(self.trigger, trigger, 0o600)
        trigger_sha = finalizer.file_identity(self.trigger)["sha256"]
        stop = {
            "schema": finalizer.GUARD_REPORT_SCHEMA,
            "mode": "watchdog_supervise",
            "child_exit_code": 0,
            "passed": True,
            "stop_invoked": True,
            "status": "stopped_and_volume_retained_verified",
            "pod": {
                "id": finalizer.POD_ID,
                "name": finalizer.POD_NAME,
                "desiredStatus": "EXITED",
                "runtimeStatus": "stopped",
                "gpuCount": finalizer.GPU_COUNT,
                "networkVolumeId": finalizer.VOLUME_ID,
                "volumeMountPath": finalizer.VOLUME_MOUNT_PATH,
            },
            "network_volume": {
                "id": finalizer.VOLUME_ID,
                "name": finalizer.VOLUME_NAME,
                "size": finalizer.VOLUME_SIZE_GB,
                "dataCenterId": finalizer.VOLUME_DATACENTER,
            },
            "authorization_sha256": authorization_sha,
            "trigger_sha256": trigger_sha,
            "terminal_outcome": "success",
            "secret_recorded": False,
        }
        write_json(self.stop, stop)

    def run_postflight(self):
        return finalizer.run_postflight(
            provisional_audit_path=self.provisional,
            summary_path=self.summary,
            representation_path=self.representation,
            git_verification_path=self.git,
            stop_verification_path=self.stop,
            guard_authorization_path=self.authorization,
            guard_trigger_path=self.trigger,
            output_path=self.final_audit,
        )


class Experiment2D5CTerminalTests(unittest.TestCase):
    def test_finalizer_is_stdlib_only_and_has_no_infrastructure_execution(self):
        source = FINALIZER_PATH.read_text()
        tree = ast.parse(source)
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertFalse(imports.intersection({"torch", "numpy", "subprocess", "requests"}))
        self.assertNotIn("stop_exact_pod(", source)
        self.assertNotIn("urlopen(", source)
        self.assertEqual(
            set(finalizer.build_parser()._subparsers._group_actions[0].choices),
            {"postflight-audit", "render-report"},
        )

    def test_driver_midpoint_and_final_seal_bind_byte_scheduler_process_and_artifacts(self):
        source = DRIVER_PATH.read_text()
        tree = ast.parse(source)
        implementation = ast.get_source_segment(
            source, function_node(tree, "implementation_file_sha256")
        )
        preexit = ast.get_source_segment(source, function_node(tree, "preexit_restart_record"))
        restart = ast.get_source_segment(source, function_node(tree, "midpoint_restart_audit"))
        train = ast.get_source_segment(source, function_node(tree, "run_train"))
        seal = ast.get_source_segment(source, function_node(tree, "run_seal_final"))
        for required in (
            "scripts/experiment_2d5c_build_continuation_calibration.py",
            "scripts/experiment_2d5c_continuation_probe.py",
            "scripts/experiment_2d5c_finalizer.py",
            "scripts/experiment_2d5c_runpod_guard.py",
            "scripts/experiment_2d5c_workflow.py",
        ):
            self.assertIn(required, implementation)
        for required in (
            '"checkpoint_file_sha256"',
            '"checkpoint_file_bytes"',
            '"scheduler_state"',
            '"scheduler_sha256"',
        ):
            self.assertIn(required, preexit)
        for required in (
            '"checkpoint_byte_sha256"',
            '"checkpoint_byte_count"',
            '"scheduler_exact_preexit"',
            '"scheduler_exact_source"',
            '"saved_process_exact"',
        ):
            self.assertIn(required, restart)
        for required in (
            '"process_id": os.getpid()',
            '"two_exact_training_processes"',
            '"optimizer_terminal_digest_exact"',
            '"scheduler_evidence"',
            '"artifact_identity"',
        ):
            self.assertIn(required, train)
        for required in (
            '"training_complete"',
            '"training_log"',
            '"training_replay_actual"',
            '"replay_ledger"',
            '"midpoint_restart_audit"',
            '"strict_reopen_new_process"',
            '"optimizer_state_exact"',
            '"process_evidence_exact"',
            '"scheduler_digest_exact"',
            '"milestone_manifest_exact"',
        ):
            self.assertIn(required, seal)

    def test_evaluation_protocol_matrix_is_parent_only_all_real_and_requires_parallel_milestones(self):
        source = DRIVER_PATH.read_text()
        tree = ast.parse(source)
        function = copy.deepcopy(function_node(tree, "evaluation_protocol_checks"))
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        environment = {"LOCAL_UPDATES": 191, "MILESTONES": (48, 96, 144, 191)}
        exec(compile(module, str(DRIVER_PATH), "exec"), environment)
        matrix = environment["evaluation_protocol_checks"]
        self.assertTrue(matrix("Parent", "core", 0, True, None, None, None)["passed"])
        self.assertFalse(matrix("Parent", "core", 0, False, None, None, None)["passed"])
        self.assertFalse(matrix("C0", "core", 0, True, "parallel.json", None, None)["passed"])
        self.assertFalse(matrix("C0", "core", 0, False, None, None, None)["passed"])
        self.assertTrue(matrix("C0", "core", 0, False, "parallel.json", None, None)["passed"])
        self.assertFalse(matrix("C", "core", 96, False, None, None, "milestones.json")["passed"])
        self.assertTrue(matrix("C", "core", 96, False, "parallel.json", None, "milestones.json")["passed"])
        self.assertTrue(matrix("C", "core", 48, False, None, None, "milestones.json")["passed"])
        self.assertFalse(matrix("C", "large", 191, False, None, None, "milestones.json")["passed"])
        self.assertTrue(matrix("C", "large", 191, False, None, "seal.json", "milestones.json")["passed"])
        self.assertTrue(matrix("Fixed", "large", 191, False, None, None, None)["passed"])

    def test_passed_postflight_binds_science_git_guard_and_exact_stopped_state(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TerminalFixture(Path(directory))
            result = fixture.run_postflight()
            self.assertTrue(result["passed"])
            self.assertTrue(result["pod_stopped"])
            self.assertTrue(result["persistent_volume_retained"])
            self.assertEqual(result["runpod_status"], "STOPPED")
            self.assertTrue(all(result["git_checks"].values()))
            self.assertTrue(all(result["runpod_stop_checks"].values()))
            self.assertEqual(
                result["input_artifact_identity"]["summary"],
                finalizer.file_identity(fixture.summary),
            )

    def test_postflight_refuses_git_or_guard_identity_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TerminalFixture(Path(directory))
            git = finalizer.read_json(fixture.git)
            git["origin_tag_commit"] = "4" * 40
            write_json(fixture.git, git)
            with self.assertRaises(finalizer.FinalizerError):
                fixture.run_postflight()
            self.assertFalse(fixture.final_audit.exists())
        with tempfile.TemporaryDirectory() as directory:
            fixture = TerminalFixture(Path(directory))
            stop = finalizer.read_json(fixture.stop)
            stop["pod"]["desiredStatus"] = "RUNNING"
            write_json(fixture.stop, stop)
            with self.assertRaises(finalizer.FinalizerError):
                fixture.run_postflight()
            self.assertFalse(fixture.final_audit.exists())

    def test_report_requires_exact_bound_postflight_and_has_real_pressure_lift_table(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TerminalFixture(Path(directory))
            fixture.run_postflight()
            text = finalizer.render_report(
                summary_path=fixture.summary,
                representation_path=fixture.representation,
                postflight_audit_path=fixture.final_audit,
                output_path=fixture.report,
            )
            self.assertIn("## Fixed-versus-C pressure-lift table", text)
            self.assertIn(
                "| Link | Intervention | Fixed effect | C effect | Fixed-to-C lift | Paired 95% CI of lift |",
                text,
            )
            self.assertIn("| B3 | OFF |", text)
            self.assertIn("| B5 | SHUFFLED |", text)
            self.assertIn("| Combined | OFF |", text)
            self.assertTrue(text.rstrip().endswith(finalizer.FINAL_PHRASE))

            final_audit = finalizer.read_json(fixture.final_audit)
            final_audit["git_verification"]["origin_tag_commit"] = "6" * 40
            write_json(fixture.final_audit, final_audit)
            with self.assertRaises(finalizer.FinalizerError):
                finalizer.render_report(
                    summary_path=fixture.summary,
                    representation_path=fixture.representation,
                    postflight_audit_path=fixture.final_audit,
                    output_path=fixture.report,
                )
            fixture.run_postflight()

            summary = finalizer.read_json(fixture.summary)
            summary["fixed_all_real_ce"] += 0.01
            write_json(fixture.summary, summary)
            with self.assertRaises(finalizer.FinalizerError):
                finalizer.render_report(
                    summary_path=fixture.summary,
                    representation_path=fixture.representation,
                    postflight_audit_path=fixture.final_audit,
                    output_path=fixture.report,
                )


if __name__ == "__main__":
    unittest.main()
