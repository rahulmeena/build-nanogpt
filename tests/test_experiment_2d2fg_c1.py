import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/experiment_2d2fg_c1.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("experiment_2d2fg_c1", MODULE_PATH)
exp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp)


class FrozenHeadToHeadTests(unittest.TestCase):
    def test_preregistered_geometry(self):
        self.assertEqual(exp.BATCH, 64)
        self.assertEqual(exp.T, 1024)
        self.assertEqual(exp.BATCHES, 16)
        self.assertEqual(exp.SKIP_BATCHES, 20)
        self.assertEqual(exp.C1_START_TOKEN_OFFSET, 262_145)
        self.assertEqual(exp.NEW_START_TOKEN_OFFSET, 1_310_722)
        self.assertEqual(exp.SEQUENCES, 1024)
        self.assertEqual(exp.TARGETS, 1_048_576)

    def test_memory_accounting(self):
        row = exp.theoretical_memory()
        self.assertTrue(row["passed"])
        self.assertEqual(row["f_saving_bytes"], 3_047_424)
        self.assertLess(row["2D2F"]["total_inference_state_bytes_B1"], row["2D2G"]["total_inference_state_bytes_B1"])

    def test_scalar_tensor_hashing(self):
        scalar = torch.tensor(0.125, dtype=torch.float32)
        self.assertEqual(exp.tensor_bytes(scalar), scalar.numpy().tobytes())
        self.assertEqual(exp.state_dict_sha256({"gate": scalar}), exp.state_dict_sha256({"gate": scalar.clone()}))

    def test_incremental_step_adapter(self):
        logits = object()
        state = object()
        self.assertEqual(exp.unpack_incremental_step((logits, state)), (logits, state))
        self.assertEqual(exp.unpack_incremental_step((logits, state, {"diagnostic": 1})), (logits, state))
        with self.assertRaises(RuntimeError):
            exp.unpack_incremental_step((logits,))

    def test_live_cache_audit_accepts_tuple_lengths(self):
        audit = {
            "cache_lengths": (1, 31, 63) + (1023,) * 9,
            "h10_ring_length": 1023,
            "h12_ring_length": 1023,
            "b11_recurrent_ring_present": False,
            "physical_storage_exact": True,
            "passed": True,
        }
        self.assertTrue(exp.validate_cache_audit("F", audit))

    def test_absolute_classification_boundaries(self):
        self.assertEqual(
            exp.absolute_classification(0.001, {"lower": 0.0001}),
            "2D2G ABSOLUTE CE ADVANTAGE STRONGLY CONFIRMED",
        )
        self.assertEqual(
            exp.absolute_classification(0.001, {"lower": -0.0001}),
            "2D2G ABSOLUTE CE ADVANTAGE DIRECTIONALLY CONFIRMED",
        )
        self.assertEqual(
            exp.absolute_classification(0.0, {"lower": -0.0001}),
            "ABSOLUTE CE ADVANTAGE NOT CONFIRMED",
        )

    def test_recurrence_classification_boundaries(self):
        self.assertEqual(
            exp.recurrence_classification(0.1, 0.1, {"lower": 0.01}, {"lower": 0.02}),
            "STRONGLY CONFIRMED",
        )
        self.assertEqual(
            exp.recurrence_classification(0.1, 0.1, {"lower": -0.01}, {"lower": 0.02}),
            "DIRECTIONALLY CONFIRMED",
        )
        self.assertEqual(
            exp.recurrence_classification(-0.1, 0.1, {"lower": -0.2}, {"lower": 0.02}),
            "NOT CONFIRMED",
        )

    def test_paired_stats_orientation(self):
        row = exp.paired_stats([-2.0, 0.0, 1.0, 3.0])
        self.assertEqual(row["negative"], 1)
        self.assertEqual(row["positive"], 2)
        self.assertEqual(row["ties"], 1)
        self.assertEqual(row["median"], 0.5)

    def test_bootstrap_is_reproducible(self):
        values = np.asarray([-1.0, 0.5, 2.0, 3.0])
        left = exp.bootstrap(values, exp.BOOTSTRAP_SEED)
        right = exp.bootstrap(values, exp.BOOTSTRAP_SEED)
        self.assertEqual(left, right)
        self.assertEqual(left["resamples"], 20_000)

    def test_report_terminal_format(self):
        summary = {
            "absolute_quality_classification": "CLASS",
            "canonical_architecture": "2D2F",
            "2D2F": {"real_ce": 3.0, "b3_gain": 0.1, "b3_sequence_gap": 0.2, "gain_paired": {}, "gap_paired": {}, "recurrence_confirmation": "STRONGLY CONFIRMED"},
            "2D2G": {"real_ce": 2.9, "b3_gain": 0.05, "b3_sequence_gap": 0.1, "gain_paired": {}, "gap_paired": {}, "recurrence_confirmation": "STRONGLY CONFIRMED"},
            "f_minus_g_ce": 0.1,
            "paired_f_vs_g": {"paired_stats": {}, "f_wins": 1, "g_wins": 2, "ties": 0},
            "bootstrap": {name: {"lower": 0.01, "upper": 0.2} for name in ("f_minus_g", "f_gain", "f_gap", "g_gain", "g_gap", "recurrent_gain_difference", "sequence_gap_difference")},
            "delta_recurrent_gain": 0.05,
            "delta_sequence_gap": 0.1,
            "position_bins": {"1-31": {"f_ce": 3.0, "g_ce": 2.9, "f_minus_g": 0.1}},
            "memory": exp.theoretical_memory(),
            "ce_cost_per_mib_saved": 0.01,
            "questions": {f"Q{index}": index for index in range(1, 26)},
            "recommended_next_training_experiment": "2D2J: one experiment.",
            "mutation_counters": {"optimizer_steps": 0},
            "checkpoints": {"F": {"sha256": exp.F_SHA256}, "G": {"sha256": exp.G_SHA256}},
            "validation_subset_sha256": "0" * 64,
            "disjointness_passed": True,
            "artifact_path": "/tmp/result",
            "pod": {"id": "pod", "status": "RUNNING", "persistent_volume_id": "volume"},
        }
        audit = {"checks": {"all_cache_audits_passed": True, "both_archived_prefix_regressions_passed": True}}
        text = exp.report_text(summary, audit)
        self.assertTrue(text.startswith("EXPERIMENT 2D2FG-C1 COMPLETE\n"))
        self.assertTrue(text.rstrip().endswith("# EXPERIMENT 2D2FG-C1 COMPLETE"))
        self.assertEqual(text.count("## Exactly one recommended next training experiment"), 1)

    def test_no_training_api_calls_exist(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("backward", called_attributes)
        self.assertNotIn("step", called_attributes)
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("lr_scheduler", source)


if __name__ == "__main__":
    unittest.main()
