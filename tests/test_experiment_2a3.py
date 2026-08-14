#!/usr/bin/env python3
"""Fail-closed unit tests for the frozen Experiment 2A3 protocol."""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_2a3 as x


def isolation_record():
    return {
        "student_base": True,
        "student_topdown": True,
        "teacher": True,
        "optimizer": True,
        "loaders": True,
        "rng": True,
        "student_mode": True,
        "teacher_mode": True,
        "router_instrumentation": True,
        "router_source_mask": True,
        "passed": True,
    }


class ProtocolTests(unittest.TestCase):
    def test_cumulative_arithmetic(self):
        self.assertEqual(x.START_UPDATE, 191)
        self.assertEqual(x.TARGET_UPDATE, 477)
        self.assertEqual(x.MILESTONES, (286, 381, 477))
        self.assertEqual(x.TARGET_UPDATE - x.START_UPDATE, 286)
        self.assertEqual(191 * x.a0.GLOBAL_BATCH_TOKENS, 100_139_008)
        self.assertEqual(286 * x.a0.GLOBAL_BATCH_TOKENS, 149_946_368)
        self.assertEqual(381 * x.a0.GLOBAL_BATCH_TOKENS, 199_753_728)
        self.assertEqual(477 * x.a0.GLOBAL_BATCH_TOKENS, 250_085_376)
        self.assertEqual((477 - 191) * x.a0.GLOBAL_BATCH_TOKENS, 149_946_368)
        self.assertEqual(x.a0.EXPECTED_PARENT_UPDATES + 191, 1145)
        self.assertEqual(x.a0.EXPECTED_PARENT_UPDATES + 476, 1430)

    def test_source_and_replay_constants(self):
        self.assertEqual(
            x.EXPECTED_CONTINUATION_SHA256,
            "6c206a89422470061d7997764fbd9a5708be3d9043f8fab930dd4b800bd5cb95",
        )
        self.assertEqual(
            x.EXPECTED_CONTINUATION_NEXT_SHA256,
            "9f39510b105f068966ef6c052edc015d695827c422da37495fa7c244b965af0b",
        )
        self.assertEqual(
            x.EXPECTED_REPLAY_SEQUENCE_SHA256,
            "a8cfc9d2898191bd792df43339af944b38d0580d5a7804c2a3169c4ce19d57b8",
        )
        self.assertEqual(
            x.EXPECTED_MILESTONE_NEXT_SHA256[286],
            "94f21a6b52b3e14bddfd0221076172d2c04a9067dac6ca6e2e9ecfdaaed99ded",
        )
        self.assertEqual(
            x.EXPECTED_MILESTONE_NEXT_SHA256[381],
            "73dc271a2f06e5f841a8207a3d0243d09ad16b28106b39351381f76fc08d8af2",
        )
        self.assertEqual(
            x.EXPECTED_MILESTONE_NEXT_SHA256[477],
            "95081c5f68b7d05d6e39b68043f2714657c21ca05cc317549063ba9a4f9f6986",
        )

    def test_config_exact(self):
        config = json.loads(x.CONFIG_PATH.read_text())
        self.assertIs(x.validate_config(config), config)
        for key in (
            "start_completed_updates",
            "optimizer_updates",
            "lr_schedule",
            "hellaswag_examples",
            "hellaswag_controls",
        ):
            changed = copy.deepcopy(config)
            changed[key] = None
            with self.assertRaises(SystemExit):
                x.validate_config(changed)
        extra = copy.deepcopy(config)
        extra["unregistered"] = True
        with self.assertRaises(SystemExit):
            x.validate_config(extra)

    def test_architecture_source_files_unchanged_from_2a2(self):
        metadata = json.loads(x.source_metadata_path().read_text())
        for name, expected in metadata["source_file_sha256"].items():
            self.assertEqual(x.a0.file_sha256(ROOT / name), expected)

    def test_source_evaluation_is_exact_committed_100m_result(self):
        artifact = x.load_source_evaluation()
        self.assertEqual(artifact["completed_updates"], 191)
        self.assertEqual(artifact["checkpoint_sha256"], x.EXPECTED_CONTINUATION_SHA256)
        self.assertEqual(artifact["losses"]["real_feedback"], x.PINNED_START_REAL)
        self.assertEqual(
            artifact["losses"]["shuffled_feedback"], x.PINNED_START_SHUFFLED
        )
        self.assertTrue(artifact["full_control_matrix"])


class StatisticsTests(unittest.TestCase):
    def test_paired_statistics_orientation(self):
        real = [2.0, 3.0, 4.0]
        shuffled = [2.5, 2.0, 5.5]
        report = x.paired_statistics(real, shuffled)
        self.assertEqual(report["differences"], [0.5, -1.0, 1.5])
        self.assertAlmostEqual(report["mean"], 1.0 / 3.0)
        self.assertEqual(report["positive_count"], 2)
        self.assertEqual(report["negative_count"], 1)
        self.assertEqual(report["tie_count"], 0)

    def test_classification_frozen_cases(self):
        def rows(values):
            return [
                {"completed_updates": update, "sequence_specific_recovery": value}
                for update, value in zip((191, 286, 381, 477), values)
            ]

        self.assertEqual(
            x.classify_trajectory(rows((0.10, 0.12, 0.15, 0.20)))["label"],
            "MEMORY SIGNAL STILL ACCELERATING",
        )
        self.assertEqual(
            x.classify_trajectory(rows((0.10, 0.14, 0.16, 0.164)))["label"],
            "MEMORY SIGNAL SATURATING",
        )
        self.assertEqual(
            x.classify_trajectory(rows((0.10, 0.12, 0.14, 0.16)))["label"],
            "MEMORY SIGNAL STRENGTHENING",
        )
        self.assertEqual(
            x.classify_trajectory(rows((0.10, 0.12, 0.14, 0.13)))["label"],
            "MEMORY SIGNAL REVERSING",
        )
        with self.assertRaises(ValueError):
            x.classify_trajectory(list(reversed(rows((0.10, 0.12, 0.15, 0.20)))))


class HellaSwagTests(unittest.TestCase):
    def test_local_hellaswag_renderer_preserves_candidate_isolation(self):
        example = {
            "ctx": "A person walks",
            "endings": ["home.", "away.", "inside.", "outside."],
            "label": 2,
        }
        tokens, mask, label = x.render_hellaswag_example(example)
        self.assertEqual(tokens.shape[0], 4)
        self.assertEqual(mask.shape, tokens.shape)
        self.assertEqual(label, 2)
        self.assertTrue((mask.sum(dim=1) > 0).all())
        prefix_lengths = (mask == 0).sum(dim=1)
        self.assertTrue(torch.equal(prefix_lengths, prefix_lengths[0].expand_as(prefix_lengths)))

    def test_candidate_score_matches_completion_average(self):
        tokens = torch.tensor([[1, 2, 3], [1, 4, 5]])
        mask = torch.tensor([[0, 1, 1], [0, 0, 1]])
        logits = torch.zeros(2, 3, 8)
        logits[0, 0, 2] = 5
        logits[0, 1, 3] = 5
        logits[1, 1, 5] = 4
        scores = x.hellaswag_candidate_scores(tokens, mask, logits)
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertLess(scores[0], math.log(8))
        self.assertLess(scores[1], math.log(8))

    def test_hellaswag_artifact_validator_fails_closed(self):
        labels = [0] * 10042
        predictions = {
            "full_context": [0] * 2532 + [1] * (10042 - 2532),
            "masked_l1_no_feedback": [0] * 2000 + [1] * 8042,
            "real_feedback": [0] * 2100 + [1] * 7942,
            "zero_feedback": [0] * 2000 + [1] * 8042,
        }
        correct = {
            mode: sum(int(value == 0) for value in values)
            for mode, values in predictions.items()
        }
        candidate = {
            "candidate_zero_teacher_memory_bit_exact_when_other_candidates_change": True,
            "candidate_zero_logits_bit_exact_when_other_candidates_change": True,
            "example_reset_teacher_memory_bit_exact": True,
            "example_reset_logits_bit_exact": True,
            "position_zero_memory_exactly_zero": True,
            "upstream_example_scores_finite": True,
            "upstream_example_zero_equals_masked": True,
            "shuffled_feedback_evaluated": False,
            "shuffled_feedback_skip_reason": "pinned reason",
            "passed": True,
        }
        artifact = {
            "passed": True,
            "completed_updates": 477,
            "processed_student_tokens": 250_085_376,
            "checkpoint_sha256": "checkpoint",
            "metadata_sha256": "metadata",
            "source_file_sha256": {"runner": "hash"},
            "examples": 10042,
            "dataset": {"sha256": x.EXPECTED_HELLASWAG_VAL_SHA256},
            "labels": labels,
            "predictions": predictions,
            "correct": correct,
            "accuracy": {mode: value / 10042 for mode, value in correct.items()},
            "zero_equals_masked_each_example": True,
            "shuffled_feedback": {"evaluated": False, "reason": "skip"},
            "candidate_isolation": candidate,
            "state_isolation": isolation_record(),
        }
        self.assertIs(
            x.validate_hellaswag_artifact(
                artifact, "checkpoint", "metadata", {"runner": "hash"}
            ),
            artifact,
        )
        mutations = []
        bad = copy.deepcopy(artifact)
        bad["correct"]["real_feedback"] += 1
        mutations.append(bad)
        bad = copy.deepcopy(artifact)
        bad["candidate_isolation"].pop("example_reset_logits_bit_exact")
        mutations.append(bad)
        bad = copy.deepcopy(artifact)
        bad["state_isolation"]["rng"] = False
        mutations.append(bad)
        bad = copy.deepcopy(artifact)
        bad["predictions"]["zero_feedback"][0] = 3
        mutations.append(bad)
        for changed in mutations:
            with self.assertRaises(SystemExit):
                x.validate_hellaswag_artifact(
                    changed, "checkpoint", "metadata", {"runner": "hash"}
                )


class MetricsTests(unittest.TestCase):
    def test_training_rows_bind_schedule_tokens_and_hashes(self):
        expected_hashes = {191: "h191", 192: "h192"}

        def row(update):
            completed = update + 1
            return {
                "update": update,
                "completed_updates": completed,
                "processed_student_tokens": completed * x.a0.GLOBAL_BATCH_TOKENS,
                "global_schedule_step": x.a0.EXPECTED_PARENT_UPDATES + update,
                "lr": x.a0.get_lr(x.a0.EXPECTED_PARENT_UPDATES + update),
                "global_batch_sha256": expected_hashes[update],
                "optimizer": {"steps": [completed] * 3, "nonfinite_tensors": []},
                "loss": 5.0,
                "grad_norm": 1.0,
                "gate": 0.1,
                "gate_coefficient": 0.1,
                "query_norm": 1.0,
                "rmsnorm_displacement": 1.0,
                "routing_entropy": 1.0,
                "forward_seconds": 1.0,
                "backward_seconds": 1.0,
                "wall_seconds": 2.0,
                "peak_allocated_mb": 1.0,
                "peak_reserved_mb": 2.0,
                "routing_weights": {"v16": 0.25, "v17": 0.25, "v20": 0.25, "v24": 0.25},
                "teacher_eval_no_grad": True,
                "trainable_parameters_finite": True,
                "gradients": {
                    "base_tensors_with_grad": [],
                    "teacher_tensors_with_grad": [],
                    "gate": {"present": True, "finite": True, "nonzero": True},
                    "query": {"present": True, "finite": True, "nonzero": True},
                    "rmsnorm": {"present": True, "finite": True, "nonzero": True},
                },
            }

        rows = [row(191), row(192)]
        report = x.validate_training_rows(rows, 193, expected_hashes)
        self.assertTrue(report["passed"])
        changed = copy.deepcopy(rows)
        changed[1]["global_schedule_step"] += 1
        with self.assertRaises(SystemExit):
            x.validate_training_rows(changed, 193, expected_hashes)

    def test_reconcile_starts_at_191(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                json.dumps({"update": 191})
                + "\n"
                + json.dumps({"update": 192})
                + "\n"
                + json.dumps({"update": 193})
                + "\n"
            )
            report = x.reconcile_metrics(path, 193)
            self.assertEqual(report["rows_retained"], 2)
            self.assertEqual(
                [json.loads(line)["update"] for line in path.read_text().splitlines()],
                [191, 192],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
