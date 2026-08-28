import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Experiment2D3AStaticTests(unittest.TestCase):
    def test_config_contract(self):
        cfg = json.loads((ROOT / "configs/exp2d3a_alternating_integration_pyramid_100m.json").read_text())
        self.assertEqual(cfg["source"]["checkpoint_sha256"], "cb5dd5904779617959b5619982a9dfe69f0c4d705679652f4f99a8285879b5e8")
        self.assertEqual(cfg["architecture"]["parameters"], 124475908)
        self.assertEqual(cfg["architecture"]["new_parameters"], ["g_rec_b3", "g_rec_b5", "g_rec_b6"])
        self.assertEqual(cfg["training"]["updates"], 191)
        self.assertEqual(cfg["training"]["targets"], 100139008)

    def test_python_files_parse(self):
        for name in ("scripts/experiment_2d3a.py", "scripts/experiment_2d3a_core.py"):
            ast.parse((ROOT / name).read_text())

    def test_core_geometry_is_literal(self):
        text = (ROOT / "scripts/experiment_2d3a_core.py").read_text()
        self.assertIn("SPECIAL_BLOCKS = (0, 2, 4, 5)", text)
        self.assertIn("SOURCE_BLOCKS = (11, 9, 7, 6)", text)
        self.assertIn("LOCAL_WINDOWS = {0: 2, 2: 32, 4: 64, 5: 512}", text)


if __name__ == "__main__":
    unittest.main()
