"""Static fail-closed tests for the stdlib-only official 2D5C workflow."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "experiment_2d5c_workflow.py"
SOURCE = PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(PATH))


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_2d5c_workflow", PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Experiment2D5CWorkflowTests(unittest.TestCase):
    def test_module_is_stdlib_only_and_has_no_runpod_mutation(self):
        imports = set()
        for node in ast.walk(TREE):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & {"torch", "numpy", "requests", "runpod"})
        lowered = SOURCE.lower()
        self.assertNotIn("pod delete", lowered)
        self.assertNotIn("pod remove", lowered)
        self.assertNotIn("volume delete", lowered)
        self.assertNotIn("runpodctl", lowered)

    def test_scope_and_lineage_constants_are_exact(self):
        module = load_module()
        self.assertEqual(module.EXPERIMENT, "2D5C")
        self.assertEqual(
            module.BRANCH,
            "experiment-2d5c-fixed-writer-b3-b5-w2-matched-100m",
        )
        self.assertEqual(
            module.MILESTONE_NAMES,
            {
                48: "scientific_cumulative_001025507328.pt",
                96: "scientific_cumulative_001050673152.pt",
                144: "scientific_cumulative_001075838976.pt",
                191: "scientific_cumulative_001100480512.pt",
            },
        )
        self.assertEqual(module.SOURCE_SHA256[:16], "de80d0886a42e414")
        self.assertEqual(module.CONTROL_SHA256[:16], "e108e47b68a13b36")

    def test_ephemeral_ssh_endpoint_is_runtime_bound(self):
        module = load_module()
        workflow = module.Workflow(
            ROOT / "runtime-test.jsonl", "192.0.2.10", 42_222
        )
        self.assertEqual(
            workflow.ssh_prefix()[-3:], ["-p", "42222", "root@192.0.2.10"]
        )
        self.assertIn("-p 42222", workflow.rsync_shell())
        self.assertNotIn("10302", SOURCE)
        parser_source = ast.get_source_segment(
            SOURCE,
            next(
                node for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name == "build_parser"
            ),
        )
        self.assertIn('"--ssh-host"', parser_source)
        self.assertIn('"--ssh-port"', parser_source)

    def test_evaluation_surface_cannot_reduce_required_c_or_fixed_controls(self):
        module = load_module()
        c = module.evaluate_args(
            "C", "C191_LARGE", 191,
            checkpoint_path=module.checkpoint(191), panel="large",
            final_seal=True,
        )
        fixed = module.evaluate_args(
            "Fixed", "FIXED_LARGE", 191,
            checkpoint_path=module.CONTROL, panel="large",
        )
        self.assertNotIn("--all-real-only", c)
        self.assertNotIn("--all-real-only", fixed)
        self.assertIn("--final-checkpoint-seal", c)
        self.assertNotIn("--parallel-output", c)
        self.assertNotIn("--parallel-output", fixed)

    def test_only_parent_helper_request_can_be_all_real_only(self):
        module = load_module()
        parent = module.evaluate_args(
            "Parent", "PARENT_CORE", 0, all_real_only=True
        )
        self.assertIn("--all-real-only", parent)
        workflow_source = ast.get_source_segment(
            SOURCE,
            next(
                node for node in TREE.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "run_scientific_workflow"
            ),
        )
        self.assertEqual(workflow_source.count("all_real_only=True"), 1)

    def test_training_commands_are_exact_two_c_only_segments(self):
        function = next(
            node for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_scientific_workflow"
        )
        source = ast.get_source_segment(SOURCE, function)
        self.assertEqual(source.count('"train", "--arm", "C"'), 2)
        self.assertIn('"--end-local-update", "96"', source)
        self.assertIn('"--end-local-update", "191"', source)
        self.assertNotIn('"--arm", "A"', source)
        self.assertNotIn('"--arm", "B"', source)
        self.assertNotIn('"--arm", "Fixed"', source)

    def test_secondary_parallel_evidence_is_exact_c0_c96_c191(self):
        function = next(
            node for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_scientific_workflow"
        )
        source = ast.get_source_segment(SOURCE, function)
        self.assertEqual(source.count("parallel=True"), 3)
        for label in ("C0_CORE", "C96_CORE", "C191_CORE"):
            self.assertIn(label, source)


if __name__ == "__main__":
    unittest.main()
