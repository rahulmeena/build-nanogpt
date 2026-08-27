#!/usr/bin/env python3
"""CPU-only contract tests for the exact multi-lane recovery preflight."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import parallel_2d2_recovery_preflight as recovery


class RecoveryPreflightTests(unittest.TestCase):
    def test_registered_lane_recovery_contracts_are_exact(self) -> None:
        self.assertEqual(set(recovery.LANES), {"GPU0", "GPU1", "GPU2"})
        gpu1 = recovery.LANES["GPU1"]
        self.assertEqual(
            gpu1["base_checkpoint"],
            "/tmp/parallel_2d2_ephemeral/2d2g/checkpoints/"
            "stage_a_scientific_update_0191.pt",
        )
        self.assertIsNone(gpu1["base_sha256"])
        self.assertEqual(
            gpu1["allowed_untracked_result_roots"],
            {"results/experiment_2d2g_b2_full_b3_w64"},
        )
        self.assertIn("after exact 2D2G-A update 191", gpu1["recovery_reason"])
        self.assertIn("artifact publication failed", recovery.LANES["GPU0"]["recovery_reason"])
        self.assertIn("before scientific update 1", recovery.LANES["GPU2"]["recovery_reason"])

    def test_command_plan_is_independent_and_matches_bash_percent_q(self) -> None:
        root = Path("/workspace/parallel_2d2_master")
        run_root = root / "runs/00000000-0000-4000-8000-000000000001"
        plan = recovery.recovery_command_plan(
            root, run_root, ["GPU0", "GPU1", "GPU2"], ["GPU2"]
        )
        rows = plan["recovered_lanes"]
        self.assertEqual(
            rows["GPU2"]["recovery_evidence_schema"],
            "legacy_v1_without_recovery_reason",
        )
        self.assertEqual(
            rows["GPU0"]["recovery_evidence_schema"],
            "v2_with_recovery_reason",
        )
        self.assertEqual(len(rows["GPU0"]["expected_resumed_command_records"]), 6)
        self.assertEqual(len(rows["GPU1"]["expected_resumed_command_records"]), 5)
        self.assertEqual(len(rows["GPU2"]["expected_resumed_command_records"]), 5)

        gpu1 = rows["GPU1"]["expected_resumed_command_records"]
        self.assertIn(" smoke-b ", gpu1[0])
        self.assertIn("stage_a_scientific_update_0191.pt", gpu1[0])
        self.assertIn("/tmp/parallel_2d2_ephemeral/2d2g/smoke", gpu1[0])
        self.assertIn(" train-b ", gpu1[1])
        self.assertIn("--end-update 96", gpu1[1])
        self.assertIn("--resume", gpu1[2])
        self.assertIn(" persist-final ", gpu1[3])
        self.assertIn(" finalize ", gpu1[4])
        self.assertNotIn("train-a", "\n".join(gpu1))
        self.assertNotIn(" preflight ", "\n".join(gpu1))

        for lane in ("GPU0", "GPU1", "GPU2"):
            specs = recovery.expected_recovery_argv(root, run_root, lane)
            expected = rows[lane]["expected_resumed_command_records"]
            self.assertEqual(len(specs), len(expected))
            for argv, rendered in zip(specs, expected):
                script = (
                    'result=""; for word in "$@"; do printf -v quoted "%q" "$word"; '
                    'result+="${result:+ }${quoted}"; done; printf "%s" "$result"'
                )
                observed = subprocess.check_output(
                    ["bash", "-c", script, "--", *argv], text=True
                )
                self.assertEqual(rendered, observed)

    def test_tracked_cleanliness_and_exact_untracked_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=worktree,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Recovery Test"],
                cwd=worktree,
                check=True,
            )
            tracked = worktree / "tracked.txt"
            tracked.write_text("sealed\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=worktree, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "fixture"], cwd=worktree, check=True
            )
            root = "results/experiment_exact"
            artifact = worktree / root / "result_summary.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}\n")

            accepted = recovery.audit_worktree_artifacts(worktree, {root})
            self.assertTrue(accepted["passed"])
            self.assertEqual(accepted["untracked_files"], [f"{root}/result_summary.json"])

            outside = worktree / "results/experiment_exact_extra/forged.json"
            outside.parent.mkdir(parents=True)
            outside.write_text("{}\n")
            rejected = recovery.audit_worktree_artifacts(worktree, {root})
            self.assertFalse(rejected["checks"]["untracked_only_in_exact_result_roots"])
            self.assertIn("results/experiment_exact_extra/forged.json", rejected["disallowed_untracked_files"])

            outside.unlink()
            tracked.write_text("modified\n")
            dirty = recovery.audit_worktree_artifacts(worktree, {root})
            self.assertFalse(dirty["checks"]["tracked_worktree_clean"])
            self.assertTrue(dirty["checks"]["untracked_only_in_exact_result_roots"])

    def test_checkpoint_requires_exact_colocated_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage_a_scientific_update_0191.pt"
            checkpoint.write_bytes(b"exact stage A 191")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoint.with_suffix(".pt.sha256").write_text(
                f"{digest}  {checkpoint.name}\n"
            )
            checkpoint.with_suffix(".pt.verification.json").write_text(
                json.dumps({"passed": True}) + "\n"
            )
            self.assertTrue(recovery.audit_checkpoint_sidecars(checkpoint, None)["passed"])
            checkpoint.with_suffix(".pt.sha256").write_text(f"{digest}  wrong.pt\n")
            self.assertFalse(recovery.audit_checkpoint_sidecars(checkpoint, None)["passed"])

    def test_lane1_shell_is_stage_b_only_and_marks_complete(self) -> None:
        path = Path(__file__).with_name("parallel_2d2_lane1_stage_b_recovery.sh")
        text = path.read_text()
        self.assertIn("source \"$MASTER_ROOT/worktrees/master/scripts/parallel_2d2_lane_common.sh\"", text)
        self.assertIn("export MASTER_RECOVERY_MODE=1", text)
        self.assertEqual(text.count("log_command "), 5)
        self.assertNotIn("train-a", text)
        self.assertNotIn("2D2G_PREFLIGHT", text)
        self.assertLess(text.index("smoke-b"), text.index("--end-update 96"))
        self.assertLess(text.index("--end-update 96"), text.index("--end-update 191"))
        self.assertLess(text.index("persist-final"), text.index(" finalize "))
        self.assertTrue(text.rstrip().endswith("lane_mark_science_complete"))

    def test_recovery_plan_is_preserved_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RECOVERY_COMMAND_PLAN.json"
            plan = {"schema_version": 1, "run_id": "exact", "recovered_lanes": {}}
            recovery.preserve_exact_json(path, plan)
            original = path.read_bytes()
            recovery.preserve_exact_json(path, plan)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(RuntimeError, "refusing to replace"):
                recovery.preserve_exact_json(path, {**plan, "run_id": "changed"})
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
