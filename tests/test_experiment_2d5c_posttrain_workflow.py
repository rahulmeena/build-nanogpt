"""Focused static/unit tests for the append-only 2D5C post-training workflow."""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import experiment_2d5c_posttrain_workflow as posttrain  # noqa: E402


class Experiment2D5CPosttrainWorkflowTests(unittest.TestCase):
    def test_outbound_rsync_disables_only_owner_and_group_preservation(self):
        self.assertIs(
            posttrain.PostTrainingWorkflow.rsync_from,
            posttrain.frozen.Workflow.rsync_from,
        )
        self.assertIsNot(
            posttrain.PostTrainingWorkflow.rsync_to,
            posttrain.frozen.Workflow.rsync_to,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_json = root / "LOCAL_BACKUP_AUDIT.json"
            local_json.write_text("{}\n", encoding="utf-8")
            workflow = posttrain.PostTrainingWorkflow(
                root / "runtime.jsonl", "example.invalid", 15280
            )
            remote_json = (
                Path("/workspace/exp2d5c_w2w2_100m/results")
                / "LOCAL_BACKUP_AUDIT.json"
            )
            with mock.patch.object(workflow, "run") as run:
                workflow.rsync_to(local_json, remote_json, "outbound audit")
            command = run.call_args.args[0]
            self.assertEqual(command[0:2], ["rsync", "-a"])
            self.assertGreater(command.index("--no-owner"), command.index("-a"))
            self.assertGreater(command.index("--no-group"), command.index("-a"))
            self.assertIn("--partial", command)
            self.assertNotIn("--no-perms", command)
            self.assertNotIn("--no-times", command)
            self.assertEqual(command[-2], str(local_json))
            self.assertEqual(
                command[-1], f"{workflow.remote}:{remote_json}"
            )

            local_checkpoint = root / "checkpoint.pt"
            with mock.patch.object(workflow, "run") as run:
                workflow.rsync_from(
                    posttrain.frozen.checkpoint(191),
                    local_checkpoint,
                    "checkpoint backup",
                )
            self.assertEqual(
                run.call_args.args[0],
                [
                    "rsync", "-a", "--partial", "--human-readable",
                    "--progress", "-e", workflow.rsync_shell(),
                    f"{workflow.remote}:{posttrain.frozen.checkpoint(191)}",
                    str(local_checkpoint),
                ],
            )

    def test_main_constructs_posttraining_transport(self):
        observed = {}

        def fake_workflow(workflow, authorization, training_freeze_commit):
            observed["workflow"] = workflow
            observed["authorization"] = authorization
            observed["training_freeze_commit"] = training_freeze_commit
            return {"scientific_commit": "a" * 40}

        argv = [
            "--authorization-artifact", "/tmp/auth.json",
            "--ssh-host", "example.invalid",
            "--ssh-port", "15280",
            "--runtime-log", "/tmp/runtime.jsonl",
            "--training-freeze-commit", posttrain.TRAINING_FREEZE_COMMIT,
        ]
        with mock.patch.object(
            posttrain, "run_posttraining_workflow", side_effect=fake_workflow
        ), mock.patch.object(posttrain.frozen, "append_runtime"):
            self.assertEqual(posttrain.main(argv), 0)
        self.assertIsInstance(
            observed["workflow"], posttrain.PostTrainingWorkflow
        )
        self.assertEqual(
            observed["training_freeze_commit"],
            posttrain.TRAINING_FREEZE_COMMIT,
        )

    def test_parser_requires_exact_training_freeze_commit(self):
        parser = posttrain.build_parser()
        common = [
            "--authorization-artifact", "/tmp/auth.json",
            "--runtime-log", "/tmp/runtime.jsonl",
            "--ssh-host", "example.invalid",
            "--ssh-port", "15280",
        ]
        with self.assertRaises(SystemExit):
            parser.parse_args(common)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [*common, "--training-freeze-commit", "0" * 40]
            )
        prefixed = list(common)
        prefixed[prefixed.index("--ssh-host") + 1] = "root@example.invalid"
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    *prefixed,
                    "--training-freeze-commit",
                    posttrain.TRAINING_FREEZE_COMMIT,
                ]
            )
        parsed = parser.parse_args(
            [
                *common,
                "--training-freeze-commit",
                posttrain.TRAINING_FREEZE_COMMIT,
            ]
        )
        self.assertEqual(
            parsed.training_freeze_commit, posttrain.TRAINING_FREEZE_COMMIT
        )

    def test_no_preflight_or_training_driver_stage_exists(self):
        source = (
            SCRIPTS / "experiment_2d5c_posttrain_workflow.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr == "driver"
            ):
                continue
            string_arguments = [
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ]
            if "preflight" in string_arguments or "train" in string_arguments:
                forbidden.append((node.lineno, string_arguments))
        self.assertEqual(forbidden, [])
        self.assertNotIn("run_scientific_workflow(", source)

    def test_exact_posttraining_stage_matrix(self):
        evaluations = posttrain.evaluation_stages()
        diagnostics = posttrain.representation_stages()
        self.assertEqual(len(evaluations), 9)
        self.assertEqual(len(diagnostics), 7)
        self.assertEqual(
            [name for name, _ in evaluations],
            [
                "Parent core",
                "C0 core and secondary parallel",
                "C48 core",
                "C96 core and secondary parallel",
                "C144 core",
                "C191 core and secondary parallel",
                "Fixed100M core",
                "C191 final large",
                "Fixed100M final large",
            ],
        )
        self.assertEqual(
            [name for name, _ in diagnostics],
            ["Parent", "C0", "C48", "C96", "C144", "C191", "Fixed100M"],
        )
        for _name, arguments in [*evaluations, *diagnostics]:
            self.assertNotIn("train", arguments)
            self.assertNotIn("preflight", arguments)

    def test_analysis_uses_only_adjudicated_training_and_seal(self):
        arguments = posttrain.analysis_arguments()
        training_index = arguments.index("--training-complete")
        seal_index = arguments.index("--final-checkpoint-seal")
        self.assertEqual(
            arguments[training_index + 1],
            str(posttrain.ADJUDICATED_TRAINING_COMPLETE),
        )
        self.assertEqual(arguments[seal_index + 1], str(posttrain.FINAL_SEAL))
        self.assertNotEqual(
            str(posttrain.ORIGINAL_TRAINING_COMPLETE),
            arguments[training_index + 1],
        )

    def test_adjudication_state_is_fail_closed(self):
        valid_training = {
            "training_passed": True,
            "training_false": [],
            "training_optimizer": [2099, 2577],
            "training_embedded_passed": True,
            "training_original_sha": (
                posttrain.ORIGINAL_FAILED_TRAINING_COMPLETE_SHA256
            ),
            "training_sha256": "a" * 64,
            "training_tool_sha": posttrain.sha256(posttrain.LOCAL_ADJUDICATOR),
        }
        posttrain.require_valid_adjudicated_training(valid_training)
        for mutation in (
            {"training_passed": False},
            {"training_false": ["optimizer_terminal_step_exact"]},
            {"training_optimizer": [2099]},
        ):
            row = {**valid_training, **mutation}
            with self.assertRaises(posttrain.PostTrainingWorkflowError):
                posttrain.require_valid_adjudicated_training(row)

        valid_legacy = {
            "legacy_sealed": False,
            "legacy_false": ["optimizer_step_exact"],
            "legacy_sha": posttrain.FINAL_CHECKPOINT_SHA256,
            "legacy_training_sha": "a" * 64,
            "training_sha256": "a" * 64,
            "legacy_file_sha256": "b" * 64,
        }
        posttrain.require_valid_legacy_seal(valid_legacy)
        with self.assertRaises(posttrain.PostTrainingWorkflowError):
            posttrain.require_valid_legacy_seal(
                {**valid_legacy, "legacy_false": []}
            )

        valid_final = {
            "final_sealed": True,
            "final_false": [],
            "final_sha": posttrain.FINAL_CHECKPOINT_SHA256,
            "final_update": 191,
            "final_embedded_passed": True,
            "final_training_sha": "a" * 64,
            "training_sha256": "a" * 64,
            "final_legacy_sha": "b" * 64,
            "legacy_file_sha256": "b" * 64,
            "final_original_sha": (
                posttrain.ORIGINAL_FAILED_TRAINING_COMPLETE_SHA256
            ),
            "final_tool_sha": posttrain.sha256(posttrain.LOCAL_ADJUDICATOR),
        }
        posttrain.require_valid_final_seal(valid_final)
        with self.assertRaises(posttrain.PostTrainingWorkflowError):
            posttrain.require_valid_final_seal(
                {**valid_final, "final_update": 192}
            )

    def test_frozen_and_adjudication_artifacts_are_distinct(self):
        self.assertNotEqual(
            posttrain.ORIGINAL_TRAINING_COMPLETE,
            posttrain.ADJUDICATED_TRAINING_COMPLETE,
        )
        self.assertNotEqual(posttrain.LEGACY_FAILED_FINAL_SEAL, posttrain.FINAL_SEAL)
        self.assertEqual(
            posttrain.ORIGINAL_FAILED_TRAINING_COMPLETE_SHA256,
            "8fef8253f596d668462a8e4c313a762105c63a36bc23f7db3ee25fc9db04579c",
        )
        self.assertEqual(
            posttrain.FINAL_CHECKPOINT_SHA256,
            "f3ffbcfb687892a4bac0496f37bf93d1a2ad3b9934481b252f1f58e3671562fe",
        )


if __name__ == "__main__":
    unittest.main()
