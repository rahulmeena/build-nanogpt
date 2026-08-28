import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Experiment2D3A500MStaticTests(unittest.TestCase):
    def test_driver_parses_and_exact_budget_is_literal(self):
        path = ROOT / "scripts/experiment_2d3a_500m.py"
        text = path.read_text()
        ast.parse(text)
        self.assertIn("SOURCE_UPDATE = 477", text)
        self.assertIn("FINAL_UPDATE = 954", text)
        self.assertIn("FINAL_TARGETS = 500_170_752", text)
        self.assertIn("RESTART_UPDATE = 715", text)
        self.assertIn('BRANCH = "experiment-2d3a-alternating-integration-pyramid-500m"', text)

    def test_frozen_source_hashes_are_literal(self):
        text = (ROOT / "scripts/experiment_2d3a_500m.py").read_text()
        self.assertIn("e60de74aad3c295e8b3dae18ad42c5004e4c55faf47f5da0997a658467875194", text)
        self.assertIn("94c5ca6b84e6af3bc1cf66c44974f07f1972c6ec86af2a8cf36587d79b382291", text)
        self.assertIn("4bf738960f4324bf4016851e5d19780ef8b5108ebe1b49cfefe88252bf608c4d", text)

    def test_factorial_controls_are_diagnostic_only(self):
        text = (ROOT / "scripts/experiment_2d3a_core.py").read_text()
        self.assertIn('FACTORIAL_CONTROLS = ("b3_b5_off", "b3_b6_off", "b5_b6_off")', text)
        start = text.index("INCREMENTAL_CONTROLS = (")
        end = text.index("FACTORIAL_CONTROLS =", start)
        canonical = text[start:end]
        self.assertNotIn("b3_b5_off", canonical)
        self.assertNotIn("b3_b6_off", canonical)
        self.assertNotIn("b5_b6_off", canonical)


if __name__ == "__main__":
    unittest.main()
