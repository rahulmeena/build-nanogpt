import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Experiment2D4AStaticTests(unittest.TestCase):
    def test_config_contract(self):
        config = json.loads(
            (ROOT / "configs/exp2d4a_matched_source_depth_routing.json").read_text()
        )
        self.assertEqual(
            config["source"]["checkpoint_sha256"],
            "de80d0886a42e4142fa8b30d27eae4302a298bb207961b593f9401e908faaf7b",
        )
        self.assertEqual(config["arms"]["fixed"]["parameters"], 124475908)
        self.assertEqual(config["arms"]["routed"]["parameters"], 124482056)
        self.assertEqual(config["arms"]["routed"]["new_parameters"], 6148)
        self.assertEqual(config["training"]["updates_per_arm"], 191)
        self.assertEqual(config["training"]["targets_per_arm"], 100139008)

    def test_python_files_parse(self):
        for name in (
            "scripts/experiment_2d3a_core.py",
            "scripts/experiment_2d4a_core.py",
            "scripts/experiment_2d4a.py",
        ):
            ast.parse((ROOT / name).read_text())

    def test_core_geometry_is_literal(self):
        text = (ROOT / "scripts/experiment_2d4a_core.py").read_text()
        self.assertIn("0: tuple(range(1, 12))", text)
        self.assertIn("2: tuple(range(3, 12))", text)
        self.assertIn("4: tuple(range(5, 12))", text)
        self.assertIn("5: tuple(range(6, 12))", text)
        self.assertIn("BASELINE_BLOCKS = {0: 11, 2: 9, 4: 7, 5: 6}", text)
        self.assertIn("ROUTER_PARAMETER_COUNT = 4 * (768 + 768 + 1)", text)


if __name__ == "__main__":
    unittest.main()
