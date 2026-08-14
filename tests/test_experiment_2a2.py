#!/usr/bin/env python3
"""Fail-closed unit tests for the frozen Experiment 2A2 protocol."""

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
import experiment_2a2 as x


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
        self.assertEqual(x.START_UPDATE, 48)
        self.assertEqual(x.TARGET_UPDATE, 191)
        self.assertEqual(x.MILESTONES, (96, 191))
        self.assertEqual(x.TARGET_UPDATE - x.START_UPDATE, 143)
        self.assertEqual(48 * x.a0.GLOBAL_BATCH_TOKENS, 25_165_824)
        self.assertEqual(96 * x.a0.GLOBAL_BATCH_TOKENS, 50_331_648)
        self.assertEqual(191 * x.a0.GLOBAL_BATCH_TOKENS, 100_139_008)
        self.assertEqual((191 - 48) * x.a0.GLOBAL_BATCH_TOKENS, 74_973_184)
        self.assertEqual(x.a0.EXPECTED_PARENT_UPDATES + 48, 1002)
        self.assertEqual(x.a0.EXPECTED_PARENT_UPDATES + 190, 1144)

    def test_source_and_replay_constants(self):
        self.assertEqual(
            x.EXPECTED_CONTINUATION_SHA256,
            "d821b48a796b12bb489f5bc9bc1791c475c09a50de7d5b47c4a36cf766643ec2",
        )
        self.assertEqual(
            x.EXPECTED_CONTINUATION_NEXT_SHA256,
            "1c3290f72d60d356d636e57017bbc5f2cb2ec470af7860d9d95d5c95116d24a5",
        )
        self.assertEqual(
            x.EXPECTED_REPLAY_SEQUENCE_SHA256,
            "3ea6e3a4833f14f57df109bfc9fca01798b5f1aaeb9b8d6b1fbc6035dd92d604",
        )
        self.assertEqual(
            x.EXPECTED_MILESTONE_NEXT_SHA256[96],
            "8b5f60d4333ba75235e63cb6397dedc31b8105f629e044ae87ca096bf6a57864",
        )
        self.assertEqual(
            x.EXPECTED_MILESTONE_NEXT_SHA256[191],
            "9f39510b105f068966ef6c052edc015d695827c422da37495fa7c244b965af0b",
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

    def test_architecture_source_files_unchanged_from_2a1(self):
        metadata = json.loads(x.source_metadata_path().read_text())
        for name, expected in metadata["source_file_sha256"].items():
            self.assertEqual(x.a0.file_sha256(ROOT / name), expected)


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
                for update, value in zip((48, 96, 191), values)
            ]

        self.assertEqual(
            x.classify_trajectory(rows((0.04, 0.06, 0.09)))["label"],
            "MEMORY SIGNAL ACCELERATING",
        )
        self.assertEqual(
            x.classify_trajectory(rows((0.04, 0.06, 0.064)))["label"],
            "MEMORY SIGNAL SATURATING",
        )
        self.assertEqual(
            x.classify_trajectory(rows((0.04, 0.03, 0.035)))["label"],
            "MEMORY SIGNAL STRENGTHENING",
        )
        self.assertEqual(
            x.classify_trajectory(rows((0.04, 0.02, 0.0)))["label"],
            "MEMORY SIGNAL DISAPPEARING",
        )
        with self.assertRaises(ValueError):
            x.classify_trajectory(list(reversed(rows((0.04, 0.06, 0.09)))))


class HellaSwagTests(unittest.TestCase):
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
            "completed_updates": 191,
            "processed_student_tokens": 100_139_008,
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
        expected_hashes = {48: "h48", 49: "h49"}

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

        rows = [row(48), row(49)]
        report = x.validate_training_rows(rows, 50, expected_hashes)
        self.assertTrue(report["passed"])
        changed = copy.deepcopy(rows)
        changed[1]["global_schedule_step"] += 1
        with self.assertRaises(SystemExit):
            x.validate_training_rows(changed, 50, expected_hashes)

    def test_reconcile_starts_at_48(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                json.dumps({"update": 48})
                + "\n"
                + json.dumps({"update": 49})
                + "\n"
                + json.dumps({"update": 50})
                + "\n"
            )
            report = x.reconcile_metrics(path, 50)
            self.assertEqual(report["rows_retained"], 2)
            self.assertEqual(
                [json.loads(line)["update"] for line in path.read_text().splitlines()],
                [48, 49],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
