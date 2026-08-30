"""Focused fail-closed tests for adjudicated 2D5C terminal completion."""

from __future__ import annotations

import argparse
import ast
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import experiment_2d5c_posttrain_complete as posttrain  # noqa: E402


class Experiment2D5CPosttrainCompletionTests(unittest.TestCase):
    def test_wrapper_is_stdlib_only_and_imports_existing_finalizer(self):
        source = (SCRIPTS / "experiment_2d5c_posttrain_complete.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & {"torch", "numpy", "requests", "runpod"})
        self.assertIn("from experiment_2d5c_complete import (", source)
        self.assertIn("finalize_and_commit", source)
        self.assertIn("experiment_2d5c_posttrain_workflow.py", source)
        self.assertNotIn("runpodctl", source.lower())
        self.assertNotIn("pod delete", source.lower())
        self.assertNotIn("volume delete", source.lower())

    def test_parser_requires_exact_training_freeze_and_fresh_path_arguments(self):
        parser = posttrain.build_parser()
        required = [
            "--authorization-artifact", "/tmp/auth.json",
            "--trigger-file", "/tmp/trigger.json",
            "--stop-report", "/tmp/report.json",
            "--runtime-log", "/tmp/runtime.jsonl",
            "--terminal-git-verification", "/tmp/git.json",
            "--ssh-host", "example.invalid",
            "--ssh-port", "15280",
        ]
        with self.assertRaises(SystemExit):
            parser.parse_args(required)
        with self.assertRaises(SystemExit):
            parser.parse_args([
                *required,
                "--training-freeze-commit",
                "0" * 40,
            ])
        prefixed = list(required)
        prefixed[prefixed.index("--ssh-host") + 1] = "root@example.invalid"
        with self.assertRaises(SystemExit):
            parser.parse_args([
                *prefixed,
                "--training-freeze-commit",
                posttrain.TRAINING_FREEZE_COMMIT,
            ])
        parsed = parser.parse_args([
            *required,
            "--training-freeze-commit",
            posttrain.TRAINING_FREEZE_COMMIT,
        ])
        self.assertEqual(parsed.training_freeze_commit, posttrain.TRAINING_FREEZE_COMMIT)

    def test_success_command_forwards_exact_posttraining_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization = root / "authorization.json"
            authorization.write_text("private\n", encoding="utf-8")
            authorization.chmod(0o600)
            trigger = root / "trigger.json"
            report = root / "report.json"
            runtime = root / "runtime.jsonl"
            terminal = root / "terminal.json"
            args = argparse.Namespace(
                authorization_artifact=authorization,
                trigger_file=trigger,
                stop_report=report,
                runtime_log=runtime,
                terminal_git_verification=terminal,
                ssh_host="example.invalid",
                ssh_port=15280,
                training_freeze_commit=posttrain.TRAINING_FREEZE_COMMIT,
                watch_timeout_seconds=600.0,
                stop_timeout_seconds=300.0,
            )
            observed = {}

            def fake_run(command, **kwargs):
                observed["command"] = command
                trigger.write_text("trigger\n", encoding="utf-8")
                trigger.chmod(0o600)
                runtime.write_text('{"event":"success"}\n', encoding="utf-8")
                payload = {
                    "schema": posttrain.GUARD_REPORT_SCHEMA,
                    "mode": "watchdog_supervise",
                    "passed": True,
                    "terminal_outcome": "success",
                    "child_exit_code": 0,
                    "status": "stopped_and_volume_retained_verified",
                    "authorization_sha256": posttrain.sha256(authorization),
                    "trigger_sha256": posttrain.sha256(trigger),
                    "pod": {
                        "id": posttrain.POD_ID,
                        "name": posttrain.POD_NAME,
                        "desiredStatus": "EXITED",
                        "runtimeStatus": "stopped",
                    },
                    "network_volume": {
                        "id": posttrain.NETWORK_VOLUME_ID,
                        "name": posttrain.NETWORK_VOLUME_NAME,
                    },
                    "secret_recorded": False,
                }
                report.write_text(json.dumps(payload), encoding="utf-8")
                report.chmod(0o600)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(posttrain, "validate_git_lineage"), mock.patch.object(
                posttrain.subprocess, "run", side_effect=fake_run
            ):
                result = posttrain.run_guarded_posttrain_workflow(args)

            self.assertTrue(result["passed"])
            command = observed["command"]
            self.assertIn("experiment_2d5c_runpod_guard.py", " ".join(command))
            self.assertIn("experiment_2d5c_posttrain_workflow.py", " ".join(command))
            freeze_index = command.index("--training-freeze-commit")
            self.assertEqual(command[freeze_index + 1], posttrain.TRAINING_FREEZE_COMMIT)
            runtime_index = command.index("--runtime-log")
            self.assertEqual(command[runtime_index + 1], str(runtime))

    def test_existing_runtime_path_is_rejected_before_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization = root / "authorization.json"
            authorization.write_text("private\n", encoding="utf-8")
            authorization.chmod(0o600)
            runtime = root / "runtime.jsonl"
            runtime.write_text("stale\n", encoding="utf-8")
            args = argparse.Namespace(
                authorization_artifact=authorization,
                trigger_file=root / "trigger.json",
                stop_report=root / "report.json",
                runtime_log=runtime,
                terminal_git_verification=root / "terminal.json",
            )
            with self.assertRaisesRegex(
                posttrain.CompletionError,
                "refusing to reuse existing post-training runtime log",
            ):
                posttrain.validate_artifact_paths(args)

    def test_child_failure_cannot_enter_terminal_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization = root / "authorization.json"
            authorization.write_text("private\n", encoding="utf-8")
            authorization.chmod(0o600)
            argv = [
                "--authorization-artifact", str(authorization),
                "--trigger-file", str(root / "trigger.json"),
                "--stop-report", str(root / "report.json"),
                "--runtime-log", str(root / "runtime.jsonl"),
                "--terminal-git-verification", str(root / "terminal.json"),
                "--ssh-host", "example.invalid",
                "--ssh-port", "15280",
                "--training-freeze-commit", posttrain.TRAINING_FREEZE_COMMIT,
            ]
            with mock.patch.object(
                posttrain,
                "run_guarded_posttrain_workflow",
                side_effect=posttrain.CompletionError("child failed and pod retained"),
            ), mock.patch.object(posttrain, "finalize_and_commit") as finalize:
                self.assertEqual(posttrain.main(argv), 1)
            finalize.assert_not_called()

    def test_finalizer_is_ordered_after_verified_guard_success(self):
        source = (SCRIPTS / "experiment_2d5c_posttrain_complete.py").read_text(
            encoding="utf-8"
        )
        main = source[source.index("def main"):]
        self.assertLess(
            main.index("run_guarded_posttrain_workflow(args)"),
            main.index("finalize_and_commit(args)"),
        )
        guarded = source[
            source.index("def run_guarded_posttrain_workflow"):
            source.index("def build_parser")
        ]
        self.assertIn('stop.get("child_exit_code") == 0', guarded)
        self.assertIn('stop.get("terminal_outcome") == "success"', guarded)
        self.assertIn('stop.get("pod", {}).get("runtimeStatus") == "stopped"', guarded)


if __name__ == "__main__":
    unittest.main()
