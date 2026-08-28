import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/experiment_2d3a_1b.py"


class Experiment2D3A1BStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DRIVER.read_text()
        cls.tree = ast.parse(cls.text)

    def test_exact_budget_and_lineage_are_literals(self):
        self.assertIn("SOURCE_UPDATE = 954", self.text)
        self.assertIn("SOURCE_TARGETS = 500_170_752", self.text)
        self.assertIn("FINAL_UPDATE = 1908", self.text)
        self.assertIn("FINAL_TARGETS = 1_000_341_504", self.text)
        self.assertIn("RESTART_UPDATE = 1431", self.text)
        self.assertIn('BRANCH = "experiment-2d3a-alternating-integration-pyramid-1b"', self.text)
        self.assertIn("a81a7984468cb89bbf6b6e633e6fa3670068041c", self.text)

    def test_exact_milestones_and_recovery_cadence(self):
        for literal in (
            '1192: "625m"', '1431: "750m"', '1669: "875m"', '1908: "1b"',
            "624_951_296", "750_256_128", "875_036_672",
            "RECOVERY_UPDATES = (1050, 1146, 1288, 1384, 1527, 1623, 1765, 1861)",
        ):
            self.assertIn(literal, self.text)

    def test_training_semantics_are_inherited(self):
        function_names = {
            node.name for node in self.tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("train_update", function_names)
        self.assertIn("base.train_update(model, optimizer, loader, accumulation, update, device)", self.text)
        self.assertIn('output / "training_metrics_500m_to_1b.jsonl"', self.text)
        self.assertIn("if update in (955, 1432)", self.text)

    def test_frozen_evaluation_geometry(self):
        self.assertIn("model, val_path, names, 4, 16", self.text)
        self.assertIn("model, val_path, list(FRESH_CONTROLS), 20, 32", self.text)
        self.assertIn("MATCHED_BOOTSTRAP_SEED = 20_260_829", self.text)
        self.assertIn("FRESH_BOOTSTRAP_SEED = 20_260_829", self.text)
        self.assertIn("resamples=50_000", self.text)
        self.assertIn('"b3_b6_off", "new_links_off"', self.text)

    def test_factorial_controls_remain_diagnostic_only(self):
        core = (ROOT / "scripts/experiment_2d3a_core.py").read_text()
        start = core.index("INCREMENTAL_CONTROLS = (")
        end = core.index("FACTORIAL_CONTROLS =", start)
        canonical = core[start:end]
        for control in ("b3_b5_off", "b3_b6_off", "b5_b6_off"):
            self.assertNotIn(control, canonical)

    def test_terminal_artifacts_and_no_follow_on_training(self):
        for name in (
            "ONE_BILLION_BASELINE_MANIFEST.json", "m1000_matched_large.json",
            "m1000_fresh_final_confirmation.json", "stability_16pass_terminal.json",
            "EXPERIMENT_2D3A_1B_FINAL_REPORT.md",
        ):
            self.assertIn(name, self.text)
        self.assertIn("NO TRAINING BEYOND 1,000,341,504 TARGETS WAS RUN.", self.text)
        self.assertIn("# EXPERIMENT 2D3A 1B COMPLETE", self.text)

    def test_all_registered_classifications_and_recommendations_are_literal(self):
        for label in (
            "MATURE MULTI-LINK POSITIVE RECURRENT PYRAMID",
            "MATURE SYNERGISTIC RECURRENT PYRAMID",
            "PARTIAL RECURRENT PYRAMID", "B5-DOMINANT RECURRENT PYRAMID",
            "RECURRENT PYRAMID NEAR ZERO", "RECURRENT PYRAMID HARMFUL",
            "FREEZE 2D3A-1B AS THE RECURRENT BASELINE",
            "PRESERVE THE FULL RECURRENT CIRCUIT",
            "RUN A MATCHED CLEANED-PYRAMID ADAPTATION EXPERIMENT",
            "STOP THIS RECURRENT PYRAMID LINE",
        ):
            self.assertIn(label, self.text)


if __name__ == "__main__":
    unittest.main()
