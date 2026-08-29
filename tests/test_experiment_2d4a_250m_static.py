import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Experiment2D4A250MStaticTests(unittest.TestCase):
    def test_continuation_contract(self):
        config = json.loads(
            (ROOT / "configs/exp2d4a_matched_source_depth_routing_250m.json").read_text()
        )
        continuation = config["continuation"]
        self.assertEqual(continuation["start_local_update"], 191)
        self.assertEqual(continuation["final_local_update"], 477)
        self.assertEqual(continuation["additional_updates_per_arm"], 286)
        self.assertEqual(continuation["final_targets_per_arm"], 250085376)
        self.assertEqual(config["milestones"], [239, 286, 334, 382, 429, 477])
        self.assertEqual(config["mandatory_restart_local_update"], 334)

    def test_continuation_driver_parses(self):
        tree = ast.parse((ROOT / "scripts/experiment_2d4a_250m.py").read_text())
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {
                "START_LOCAL_UPDATE", "LOCAL_UPDATES", "LOCAL_TARGETS",
                "CONTINUATION_UPDATES", "CONTINUATION_TARGETS",
                "RESTART_LOCAL_UPDATE", "MILESTONES", "LARGE_START_BATCH",
            }
        }
        self.assertEqual(assignments["START_LOCAL_UPDATE"], 191)
        self.assertEqual(assignments["LOCAL_UPDATES"], 477)
        self.assertEqual(assignments["LOCAL_TARGETS"], 250085376)
        self.assertEqual(assignments["CONTINUATION_UPDATES"], 286)
        self.assertEqual(assignments["CONTINUATION_TARGETS"], 149946368)
        self.assertEqual(assignments["RESTART_LOCAL_UPDATE"], 334)
        self.assertEqual(assignments["MILESTONES"], (239, 286, 334, 382, 429, 477))
        self.assertEqual(assignments["LARGE_START_BATCH"], 84)

    def test_only_authorized_training_segments_exist(self):
        text = (ROOT / "scripts/experiment_2d4a_250m.py").read_text()
        self.assertIn("((191, 334), (334, 477))", text)
        self.assertIn("if local_update in MILESTONES", text)
        self.assertIn("first 2D4A-250M continuation batch/stream mismatch", text)

    def test_checkpoint_writes_stage_off_fuse_and_never_overwrite(self):
        text = (ROOT / "scripts/experiment_2d4a_250m.py").read_text()
        self.assertIn("EXP2D4A_LOCAL_CHECKPOINT_STAGING", text)
        self.assertIn('persistent_stage.open("xb")', text)
        self.assertIn("checkpoint staging/persistent hash mismatch", text)
        self.assertIn('"rotating_recovery.pt"', text)
        self.assertIn("network_volume_free_bytes >= 10 * 1024**3", text)
        self.assertIn("local_storage.free >= 3 * 1024**3", text)


if __name__ == "__main__":
    unittest.main()
