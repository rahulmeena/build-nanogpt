"""Fail-closed static tests for guarded terminal 2D5C completion."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPLETE = ROOT / "scripts" / "experiment_2d5c_complete.py"
WORKFLOW = ROOT / "scripts" / "experiment_2d5c_workflow.py"
GUARD = ROOT / "scripts" / "experiment_2d5c_runpod_guard.py"
DRIVER = ROOT / "scripts" / "experiment_2d5c.py"


class Experiment2D5CCompletionTests(unittest.TestCase):
    def test_completion_is_stdlib_and_has_no_direct_infrastructure_mutation(self):
        source = COMPLETE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & {"torch", "numpy", "requests", "runpod"})
        lowered = source.lower()
        self.assertNotIn("runpodctl", lowered)
        self.assertNotIn("pod delete", lowered)
        self.assertNotIn("volume delete", lowered)
        self.assertIn("experiment_2d5c_runpod_guard.py", source)

    def test_guard_writes_a_clean_report_apart_from_child_stdout(self):
        guard = GUARD.read_text(encoding="utf-8")
        self.assertIn('"--report-artifact"', guard)
        self.assertIn("def _record_report", guard)
        self.assertIn("write_private_json_exclusive(path, payload)", guard)
        completion = COMPLETE.read_text(encoding="utf-8")
        self.assertIn('"--report-artifact", str(args.stop_report)', completion)
        self.assertIn("guard returned without a dedicated stop report", completion)

    def test_terminal_sequence_is_guard_then_finalizer_then_commit_and_push(self):
        source = COMPLETE.read_text(encoding="utf-8")
        self.assertLess(source.index("run_guarded_workflow(args)"), source.index("finalize_and_commit(args)"))
        function = source[source.index("def finalize_and_commit"):]
        self.assertLess(function.index('"postflight-audit"'), function.index('"render-report"'))
        self.assertLess(function.index('"render-report"'), function.index('"git", "commit"'))
        self.assertIn("Record terminal Experiment 2D5C postflight", function)
        self.assertIn('git("status", "--porcelain")', function)
        self.assertIn("worktree_clean_after_terminal_commit", function)

    def test_private_guard_inputs_are_not_added_to_git(self):
        source = COMPLETE.read_text(encoding="utf-8")
        self.assertIn('"private_guard_artifacts_retained_outside_git": True', source)
        self.assertEqual(source.count('git("add", "--"'), 1)
        add_line = next(line for line in source.splitlines() if 'git("add", "--"' in line)
        self.assertIn("LOCAL_RESULTS", add_line)
        self.assertNotIn("authorization", add_line)
        self.assertNotIn("trigger", add_line)

    def test_failure_path_preserves_report_checkpoint_and_artifact_roots(self):
        source = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("def preserve_failure_best_effort", source)
        self.assertIn("FAILURE_REPORT_", source)
        self.assertIn("latest_checkpoint", source)
        for label in ('("pretrain", PRETRAIN)', '("preflight", PREFLIGHT)', '("results", RESULTS)'):
            self.assertIn(label, source)
        self.assertIn("preserve_failure_best_effort(workflow, error)", source)

    def test_live_origin_tag_and_local_inventory_are_checked_before_preflight(self):
        source = WORKFLOW.read_text(encoding="utf-8")
        entry = source[source.index("def verify_entry"):source.index("def backup_checkpoints")]
        self.assertIn('"ls-remote", "origin", f"refs/heads/{BRANCH}"', entry)
        self.assertIn('"ls-remote", "origin", f"refs/tags/{FINAL_TAG}*"', entry)
        self.assertIn("local_inventory != remote_inventory", entry)
        self.assertIn("apparent_used, allocated_used, filesystem_available", entry)
        self.assertIn("free = min(quota_available, filesystem_available)", entry)

    def test_complete_script_is_frozen_with_implementation(self):
        source = DRIVER.read_text(encoding="utf-8")
        function = source[source.index("def implementation_file_sha256"):source.index("def require_branch")]
        self.assertIn('"scripts/experiment_2d5c_complete.py"', function)


if __name__ == "__main__":
    unittest.main()
