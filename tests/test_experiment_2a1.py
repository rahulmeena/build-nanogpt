#!/usr/bin/env python3
"""Bounded, optimizer-free protocol tests for Experiment 2A1."""

import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "experiment_2a1.py"
CONFIG_PATH = REPO_ROOT / "configs" / "exp2a1_25m.json"

EXPECTED_CONTINUATION_SHA256 = (
    "cf68b9765072e2403c16e935ba02e92f826d48600953f904e11f2bd4d266638e"
)
EXPECTED_CONTINUATION_NEXT_SHA256 = (
    "01cc1b4fe5b9e3e40047f7a686fa683be3bfaaa0f9bbd2fde1752541bd8e0a6a"
)
EXPECTED_MILESTONE_NEXT_SHA256 = {
    20: "921abc217182d1f7596f26ac421e0ba317b0c9b8b510a3baa03bf26c604d4471",
    29: "51c1a47728a9293c62481fdd1e5b4f8fe92a5eb5a98494e3a886de29dfa86674",
    48: "1c3290f72d60d356d636e57017bbc5f2cb2ec470af7860d9d95d5c95116d24a5",
}
EXPECTED_TRAINING_HASHES = (
    "01cc1b4fe5b9e3e40047f7a686fa683be3bfaaa0f9bbd2fde1752541bd8e0a6a",
    "f32f9713f981648be7c34336f55519ae1a0d4d2306c279076657703e42cd6790",
    "ea58474bc65e45012ab2540d7331870558f2e1a5f477f28a784ba30501d4daf0",
    "faa3f14cef81dcc4b353efb98241d6622b84d0989bfda7b1f7359af024e2e7db",
    "12874aaf7a4087a3ba7ecd3f82f844f6a28ffe603eb9b7123220cd8712c26c29",
    "67ae9ec62b31966043379f8580e3eb1b84e84a046601ae02e293170693032fcc",
    "278fe745a0dca1ff7e54c7b012929e79c868e2e739edd105e74798809e1cca8f",
    "01f83f664658d13ed1df4fe096e87b1e51189279fb7d6aa6bb56c1c4ddd1e9c2",
    "c5e282add7e9874417aaa04909fd3b4b90be0c3f16c688217b864b71848bfb8e",
    "2cd187df0ee2984705c0f47c2b67cbf9e63846db8ccc0e34a525e61e15604bba",
    "921abc217182d1f7596f26ac421e0ba317b0c9b8b510a3baa03bf26c604d4471",
    "22026663f31f700dcffb403771ac4776e43336944a4f0950a4ad5529686dacd3",
    "749bfa57d092690ef2af2b16658ded5c2bde5a6fbd615f7a7c518fde8d4f12b6",
    "12e2be89f1cc84220b3b91932fe48599fc6e84818c20ca4094d2b62efa38bc97",
    "aa7a995a92ef7f7a4600022f86014676c18d1447ac11224ac066aa2a02404618",
    "07c74bb1d7ab73ef3d74754d5d8d73fed9c60a1336c4bfe78d06bd0c3d161cdf",
    "a9101a740c653502d2db2b3c137c31427846da1fce74968c1b14660db32db050",
    "44439d1af97b4b238799fc7967ebd80336a9749be19907349614a95f0a4b0738",
    "6209a0303cc1a5a6fc1100ac3664f0ecdec86d1a1eac8bfb532b25164fa3348c",
    "51c1a47728a9293c62481fdd1e5b4f8fe92a5eb5a98494e3a886de29dfa86674",
    "be078fdfaaf003572c9b4c8fc1dbe8545f72b1eb13cfa18da60383b80edfde99",
    "d638662befe9b603611a8045b9b7b082a73d3893570919661399c7e26158798f",
    "758684ea3ce8043918171b0bb938d42b6851f4574ceb77559625f238bdf2bc1b",
    "8d11d06c89b1a79987a0f5ea23124d26707aeecd5525d994a623287d80c8a90c",
    "27361e0a8571fff0f2b2230905fc2c1074731390674e6ae72df511c6bfcc597d",
    "132e5153970301048918a328b75411481c6784490abcf9ed2299502533f76c2e",
    "7870ff7767bde30988c20d6d65f1610aa8dcd88c7c1d1a9d0d70c20cf844d842",
    "414bd2dc3eb6b5cc12638a7f018e7e4b96d1a14b3e84fea47f2572d9eed14eb9",
    "d2e36519c3bda897292ee0301a8f9b007e6dfc87e30c73bc390dde729c1d9b74",
    "42ab6edb525e1589155dea9ba1417041cf793a68945ec7cb84531f8331ef786a",
    "9ee1f17b9a18bb9a2529efec47b8c496aefe3fd1b0e0d1a4840540217a4ce492",
    "3870edef4b558633e9d6ebe174ef60230b699013119962def468bc9f5cb409cf",
    "1e1597d784eaba1c4e0d9a1b0d3b297e94646bd487e874da748f4feb2d44a149",
    "3642f263729e9325babec3781ff90df9d92c711c148da83d53c726206efb5305",
    "ecc590a4a24e74b822dc5e6ecf54fa193bdad11e8e4c83c6db292dac4ed06efe",
    "d2cfb10e5c4ef5b0550124910e425ea923ab97b3a9a400bfdee6d287d4303d44",
    "ea92439a4ee5186e2107e8d11eeb541a4222b5434b5c5cb5dba36f8ff4c6f12c",
    "5c5acc9efdb3af6c97e46f75e471f4866e29c3bb7cf4657fa67f468d17c9dc97",
)
EXPECTED_SOURCE_FILE_HASHES = {
    "train_gpt2.py": "1" * 64,
    "scripts/experiment_2a0.py": "2" * 64,
    "scripts/experiment_2a1.py": "3" * 64,
    "configs/exp2a0_5m.json": "4" * 64,
    "configs/exp2a1_25m.json": "5" * 64,
}


def load_runner():
    if not RUNNER_PATH.is_file():
        raise RuntimeError(f"Experiment 2A1 runner is missing: {RUNNER_PATH}")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("experiment_2a1_under_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def result_field(result, *names):
    for name in names:
        if name in result:
            return result[name]
    raise AssertionError(f"paired-statistics result omitted all aliases {names}: {result}")


def assert_rejected(testcase, config):
    with testcase.assertRaises((SystemExit, ValueError, AssertionError)):
        RUNNER.validate_config(config)


def reconcile(path, completed_updates):
    function = getattr(RUNNER, "reconcile_metrics", None)
    if function is not None:
        return function(path, completed_updates=completed_updates, start_update=10)
    function = getattr(RUNNER, "reconcile_metrics_rows", None)
    if function is None:
        raise AssertionError("runner provides neither reconcile_metrics nor reconcile_metrics_rows")
    return function(path, completed_updates=completed_updates, start_update=10)


def cached_artifact(full=False, completed_updates=20):
    payload_hashes = [
        __import__("hashlib").sha256(f"validation-batch-{index}".encode()).hexdigest()
        for index in range(20)
    ]
    validation_hash = __import__("hashlib").sha256(
        b"".join(bytes.fromhex(value) for value in payload_hashes)
    ).hexdigest()
    def exact_mean_vector(value):
        values = [value] * 19
        values.append(value * 20 - sum(values))
        if sum(values) / 20 != value:
            raise AssertionError(f"failed to construct exact mean for {value}")
        return values

    if completed_updates == 10:
        real = exact_mean_vector(RUNNER.PINNED_UPDATE10_REAL)
        shuffled = exact_mean_vector(RUNNER.PINNED_UPDATE10_SHUFFLED)
    else:
        real = [5.90 + index / 10_000 for index in range(20)]
        shuffled = [value + 0.01 for value in real]
    vectors = {"real_feedback": real, "shuffled_feedback": shuffled}
    losses = {
        "full_context_reference": RUNNER.PINNED_FULL,
        "masked_l1_no_feedback_reference": RUNNER.PINNED_MASKED,
        "real_feedback": sum(real) / 20,
        "shuffled_feedback": sum(shuffled) / 20,
    }
    full_value = RUNNER.PINNED_FULL
    masked_value = RUNNER.PINNED_MASKED
    damage = masked_value - full_value
    recovery = masked_value - losses["real_feedback"]
    sequence_specific = losses["shuffled_feedback"] - losses["real_feedback"]
    paired = RUNNER.paired_statistics(real, shuffled)
    artifact = {
        "completed_updates": completed_updates,
        "processed_student_tokens": completed_updates * 524_288,
        "checkpoint_sha256": "checkpoint-sha",
        "metadata_sha256": "metadata-sha",
        "full_control_matrix": full,
        "validation_global_batches_sha256": validation_hash,
        "batch_payload_sha256": payload_hashes,
        "batch_losses": vectors,
        "losses": losses,
        "damage": damage,
        "total_recovery": recovery,
        "total_recovery_fraction": recovery / damage,
        "sequence_specific_recovery": sequence_specific,
        "sequence_specific_recovery_fraction": sequence_specific / damage,
        "paired_shuffled_minus_real": paired,
        "requested_real_minus_shuffled": {
            "mean": -paired["mean"],
            "median": -paired["median"],
            "real_feedback_beats_shuffled_batches": paired["positive_count"],
            "shuffled_feedback_beats_real_batches": paired["negative_count"],
            "ties": paired["tie_count"],
        },
        "source_file_sha256": copy.deepcopy(EXPECTED_SOURCE_FILE_HASHES),
        "state_isolation": {
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
        },
        "routing": {
            "gate": 0.01,
            "gate_coefficient": math.tanh(0.01),
            "query_norm": 0.2,
            "rmsnorm_displacement": 0.1,
            "mean_tokenwise_entropy": 1.2,
            "normalized_entropy": 1.2 / math.log(4),
            "effective_source_count": math.exp(1.2),
            "mean_weights": {"v16": 0.3, "v17": 0.25, "v20": 0.2, "v24": 0.25},
        },
    }
    if full:
        masked = exact_mean_vector(RUNNER.PINNED_MASKED)
        full_context = exact_mean_vector(RUNNER.PINNED_FULL)
        vectors.update(
            {
                "full_context": full_context,
                "masked_l1_no_feedback": masked,
                "zero_feedback": list(masked),
            }
        )
        losses.update(
            {
                "full_context": sum(full_context) / 20,
                "masked_l1_no_feedback": sum(masked) / 20,
                "zero_feedback": sum(masked) / 20,
            }
        )
        artifact["zero_equals_masked_each_batch"] = True
        artifact["source_ablation"] = {}
        for depth in (16, 17, 20, 24):
            name = f"mask_v{depth}"
            values = [value + depth / 100_000 for value in real]
            vectors[name] = values
            losses[name] = sum(values) / 20
            artifact["source_ablation"][f"v{depth}"] = {
                "validation_loss": losses[name],
                "delta_vs_real_feedback": losses[name] - losses["real_feedback"],
                "paired_ablation_minus_real": RUNNER.paired_statistics(real, values),
            }
    return artifact, validation_hash


def training_rows(completed_updates=12):
    rows = []
    for update in range(10, completed_updates):
        rows.append(
            {
                "kind": "train",
                "update": update,
                "completed_updates": update + 1,
                "processed_student_tokens": (update + 1) * 524_288,
                "global_schedule_step": 954 + update,
                "lr": RUNNER.a0.get_lr(954 + update),
                "loss": 5.9,
                "grad_norm": 0.1,
                "gate": 0.01,
                "gate_coefficient": math.tanh(0.01),
                "query_norm": 0.2,
                "rmsnorm_displacement": 0.1,
                "routing_entropy": 1.0,
                "forward_seconds": 1.0,
                "backward_seconds": 1.5,
                "wall_seconds": 3.0,
                "peak_allocated_mb": 10_000.0,
                "peak_reserved_mb": 12_000.0,
                "routing_weights": {
                    "v16": 0.25,
                    "v17": 0.25,
                    "v20": 0.25,
                    "v24": 0.25,
                },
                "optimizer": {
                    "steps": [update + 1] * 3,
                    "nonfinite_tensors": [],
                },
                "gradients": {
                    name: {"present": True, "finite": True, "nonzero": True}
                    for name in ("gate", "query", "rmsnorm")
                }
                | {
                    "base_tensors_with_grad": [],
                    "teacher_tensors_with_grad": [],
                },
                "teacher_eval_no_grad": True,
                "trainable_parameters_finite": True,
                "global_batch_sha256": EXPECTED_TRAINING_HASHES[update - 10],
            }
        )
    return rows


def restart_record(completed_updates=20):
    return {
        "student_base": True,
        "student_topdown": True,
        "teacher": True,
        "optimizer": True,
        "loaders": True,
        "rng": True,
        "load_audit": {
            "integrity": {
                "checkpoint": "/run/checkpoints/checkpoint_updates_000020.pt",
                "sha256": "a" * 64,
                "verification": "checkpoint_updates_000020.pt.verification.json",
            },
            "completed_updates": completed_updates,
            "processed_student_tokens": completed_updates * 524_288,
            "next_global_batch_sha256": EXPECTED_MILESTONE_NEXT_SHA256[
                completed_updates
            ],
            "model_strict_reload": True,
            "optimizer_exact_reload": True,
            "loader_exact_reload": True,
            "rng_exact_reload": True,
            "passed": True,
        },
        "passed": True,
    }


def causality_record():
    isolation = {
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
    return {
        "teacher_memory": {
            "B": 2,
            "T": 1024,
            "tested_position": 512,
            "suffix_difference_starts_at": 512,
            "per_source_memory_bit_exact": {
                "v16": True,
                "v17": True,
                "v20": True,
                "v24": True,
            },
            "position_zero_memory_exactly_zero": True,
            "passed": True,
        },
        "end_to_end": {
            "B": 2,
            "T": 1024,
            "changed_suffix_starts_at": 512,
            "compared_logit_positions": [0, 511],
            "prefix_logits_bit_exact": True,
            "maximum_absolute_difference": 0.0,
            "passed": True,
        },
        "state_isolation": isolation,
        "passed": True,
    }


def source_restore_audit():
    return {
        "integrity": {"sha256": EXPECTED_CONTINUATION_SHA256},
        "model_exact_reload": True,
        "optimizer_exact_reload": True,
        "loader_exact_reload": True,
        "rng_exact_reload": True,
        "next_global_batch_sha256": EXPECTED_CONTINUATION_NEXT_SHA256,
        "optimizer": {
            "steps": [10, 10, 10],
            "state_entries": 3,
            "nonfinite_tensors": [],
        },
        "passed": True,
    }


class ConfigAndConstantsTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text())

    def test_cumulative_config_and_milestones(self):
        validated = RUNNER.validate_config(copy.deepcopy(self.config))
        self.assertEqual(validated["start_completed_updates"], 10)
        self.assertEqual(validated["start_processed_student_tokens"], 5_242_880)
        self.assertEqual(validated["optimizer_updates"], 48)
        self.assertEqual(validated["additional_optimizer_updates"], 38)
        self.assertEqual(validated["global_batch_tokens"], 524_288)
        self.assertEqual(validated["processed_student_tokens"], 25_165_824)
        self.assertEqual(validated["evaluation_updates"], [20, 29, 48])
        self.assertEqual(validated["checkpoint_updates"], [20, 29, 48])
        self.assertEqual(RUNNER.START_UPDATE, 10)
        self.assertEqual(RUNNER.TARGET_UPDATE, 48)
        self.assertEqual(tuple(RUNNER.MILESTONES), (20, 29, 48))

    def test_cumulative_arithmetic_mutations_are_rejected(self):
        mutations = {
            "start_completed_updates": 0,
            "start_processed_student_tokens": 0,
            "optimizer_updates": 47,
            "additional_optimizer_updates": 48,
            "global_batch_tokens": 524_287,
            "processed_student_tokens": 25_165_823,
            "evaluation_updates": [20, 30, 48],
            "checkpoint_updates": [20, 29],
            "source_depths": [16, 17, 20, 23],
            "sequence_length": 512,
            "sequential_microbatches_per_update": 4,
            "optimizer": "fresh AdamW",
            "lr_schedule": "restart at schedule step 0",
            "data_start": "fresh FineWeb stream",
            "intermediate_controls": ["real_feedback"],
            "final_controls": ["real_feedback", "shuffled_feedback"],
            "final_source_ablations": [16, 17, 20],
        }
        for field, bad_value in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(self.config)
                changed[field] = bad_value
                assert_rejected(self, changed)

    def test_config_rejects_extra_or_missing_fields(self):
        extra = copy.deepcopy(self.config)
        extra["unreviewed_option"] = True
        missing = copy.deepcopy(self.config)
        del missing["protocol"]
        for label, changed in (("extra", extra), ("missing", missing)):
            with self.subTest(case=label):
                assert_rejected(self, changed)

    def test_immutable_hash_constants(self):
        self.assertEqual(RUNNER.EXPECTED_CONTINUATION_SHA256, EXPECTED_CONTINUATION_SHA256)
        self.assertEqual(
            RUNNER.EXPECTED_CONTINUATION_NEXT_SHA256,
            EXPECTED_CONTINUATION_NEXT_SHA256,
        )
        if hasattr(RUNNER, "EXPECTED_MILESTONE_NEXT_SHA256"):
            self.assertEqual(RUNNER.EXPECTED_MILESTONE_NEXT_SHA256, EXPECTED_MILESTONE_NEXT_SHA256)
        hash_table = getattr(RUNNER, "EXPECTED_TRAINING_HASHES", None)
        if hash_table is not None:
            if isinstance(hash_table, dict):
                actual = tuple(hash_table[index] for index in range(10, 48))
            else:
                actual = tuple(hash_table)
            self.assertEqual(actual, EXPECTED_TRAINING_HASHES)


class PairedStatisticsTests(unittest.TestCase):
    def test_paired_statistics_use_shuffled_minus_real_and_sample_variance(self):
        real = [1.0, 2.0, 5.0, 4.0]
        shuffled = [2.0, 1.0, 5.0, 7.0]
        result = RUNNER.paired_statistics(real, shuffled)
        differences = [1.0, -1.0, 0.0, 3.0]
        mean = sum(differences) / 4
        sample_std = math.sqrt(sum((value - mean) ** 2 for value in differences) / 3)
        standard_error = sample_std / 2
        self.assertEqual(result["n"], 4)
        self.assertEqual(
            result_field(result, "differences", "shuffled_minus_real"), differences
        )
        self.assertAlmostEqual(result_field(result, "mean", "mean_difference"), mean)
        self.assertAlmostEqual(result_field(result, "median", "median_difference"), 0.5)
        self.assertEqual(result_field(result, "minimum", "min"), -1.0)
        self.assertEqual(result_field(result, "maximum", "max"), 3.0)
        self.assertEqual(result_field(result, "positive_count", "positive"), 2)
        self.assertEqual(result_field(result, "negative_count", "negative"), 1)
        self.assertEqual(result_field(result, "tie_count", "ties"), 1)
        self.assertAlmostEqual(
            result_field(result, "sample_standard_deviation", "sample_std"), sample_std
        )
        self.assertAlmostEqual(result["standard_error"], standard_error)

    def test_twenty_batch_interval_and_degenerate_case(self):
        real = [float(index) for index in range(20)]
        shuffled = [value + 0.25 for value in real]
        result = RUNNER.paired_statistics(real, shuffled)
        self.assertEqual(result["n"], 20)
        self.assertEqual(result_field(result, "positive_count", "positive"), 20)
        self.assertEqual(result_field(result, "negative_count", "negative"), 0)
        self.assertEqual(result_field(result, "tie_count", "ties"), 0)
        self.assertEqual(
            result_field(result, "sample_standard_deviation", "sample_std"), 0.0
        )
        self.assertEqual(result["standard_error"], 0.0)
        self.assertEqual(result["ci95_lower"], 0.25)
        self.assertEqual(result["ci95_upper"], 0.25)

    def test_paired_statistics_reject_invalid_vectors(self):
        invalid = [
            ([], []),
            ([1.0], [1.0, 2.0]),
            ([1.0, math.nan], [1.0, 2.0]),
            ([1.0, 2.0], [1.0, math.inf]),
        ]
        for real, shuffled in invalid:
            with self.subTest(real=real, shuffled=shuffled):
                with self.assertRaises((SystemExit, ValueError, AssertionError)):
                    RUNNER.paired_statistics(real, shuffled)


class MetricsReconciliationTests(unittest.TestCase):
    def write_rows(self, path, updates):
        path.write_text("".join(json.dumps({"update": update}) + "\n" for update in updates))

    def test_reconcile_starts_at_ten_and_truncates_only_trailing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            self.write_rows(path, range(10, 23))
            result = reconcile(path, completed_updates=20)
            retained = [json.loads(line)["update"] for line in path.read_text().splitlines()]
            self.assertEqual(retained, list(range(10, 20)))
            self.assertEqual(result["rows_before"], 13)
            self.assertEqual(result["rows_retained"], 10)
            self.assertEqual(result["rows_truncated"], 3)

    def test_reconcile_rejects_preten_missing_duplicate_or_out_of_order_rows(self):
        cases = (
            list(range(0, 20)),
            [10, 11, 13, 14],
            [10, 11, 11, 12],
            [10, 12, 11, 13],
        )
        for updates in cases:
            with self.subTest(updates=updates):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "metrics.jsonl"
                    self.write_rows(path, updates)
                    with self.assertRaises((SystemExit, ValueError, AssertionError)):
                        reconcile(path, completed_updates=14)

    def test_reconcile_exact_milestone_history_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            self.write_rows(path, range(10, 29))
            before = path.read_bytes()
            result = reconcile(path, completed_updates=29)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(result["rows_truncated"], 0)


class ModelHashTests(unittest.TestCase):
    def test_topdown_hash_supports_scalar_gate(self):
        torch = RUNNER.torch

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.topdown_attnres = torch.nn.Module()
                self.transformer.topdown_attnres.gate = torch.nn.Parameter(
                    torch.zeros(())
                )

        model = Toy()
        initial = RUNNER.model_state_sha256(model, include_topdown=True)
        self.assertEqual(initial, RUNNER.model_state_sha256(model, True))
        with torch.no_grad():
            model.transformer.topdown_attnres.gate.fill_(0.25)
        self.assertNotEqual(initial, RUNNER.model_state_sha256(model, True))


class TrainingRowValidationTests(unittest.TestCase):
    def test_training_rows_bind_cumulative_schedule_and_hashes(self):
        rows = training_rows()
        result = RUNNER.validate_training_rows(copy.deepcopy(rows), 12)
        self.assertTrue(result["passed"])
        self.assertEqual(result["updates"], [10, 11])

    def test_training_rows_reject_protocol_or_finiteness_mutations(self):
        mutations = {
            "batch hash": ("global_batch_sha256", "0" * 64),
            "schedule": ("global_schedule_step", 0),
            "tokens": ("processed_student_tokens", 0),
            "optimizer": ("optimizer", {"steps": [0, 0, 0]}),
            "loss": ("loss", math.nan),
            "routing simplex": (
                "routing_weights",
                {"v16": 1.0, "v17": 1.0, "v20": 0.0, "v24": 0.0},
            ),
            "teacher contract": ("teacher_eval_no_grad", False),
        }
        for label, (field, value) in mutations.items():
            with self.subTest(case=label):
                rows = training_rows()
                rows[0][field] = value
                with self.assertRaises((ValueError, SystemExit, AssertionError)):
                    RUNNER.validate_training_rows(rows, 12)


class RecoveryRecordValidationTests(unittest.TestCase):
    def test_restart_record_accepts_complete_proof(self):
        self.assertTrue(RUNNER.validate_restart_record(restart_record()))

    def test_restart_record_rejects_top_level_and_load_audit_tampering(self):
        mutations = []
        for field in ("student_base", "student_topdown", "teacher", "optimizer", "loaders", "rng"):
            changed = restart_record()
            changed[field] = False
            mutations.append((f"false {field}", changed))
        missing = restart_record()
        del missing["optimizer"]
        mutations.append(("missing top-level proof", missing))
        extra = restart_record()
        extra["unreviewed"] = True
        mutations.append(("extra top-level field", extra))
        load_failed = restart_record()
        load_failed["load_audit"]["passed"] = False
        mutations.append(("failed load audit", load_failed))
        nested_false = restart_record()
        nested_false["load_audit"]["model_strict_reload"] = False
        mutations.append(("false nested reload proof", nested_false))
        nested_missing = restart_record()
        del nested_missing["load_audit"]["rng_exact_reload"]
        mutations.append(("missing nested reload proof", nested_missing))
        wrong_completed = restart_record()
        wrong_completed["load_audit"]["completed_updates"] = 48
        mutations.append(("wrong completed update", wrong_completed))
        wrong_tokens = restart_record()
        wrong_tokens["load_audit"]["processed_student_tokens"] += 1
        mutations.append(("wrong cumulative tokens", wrong_tokens))
        wrong_next = restart_record()
        wrong_next["load_audit"]["next_global_batch_sha256"] = "0" * 64
        mutations.append(("wrong next-batch hash", wrong_next))
        for label, changed in mutations:
            with self.subTest(case=label):
                self.assertFalse(RUNNER.validate_restart_record(changed))

    def test_causality_record_accepts_complete_proof(self):
        self.assertTrue(RUNNER.validate_causality_record(causality_record()))

    def test_causality_record_rejects_nested_or_isolation_tampering(self):
        mutations = []
        extra = causality_record()
        extra["unreviewed"] = True
        mutations.append(("extra top-level field", extra))
        memory_false = causality_record()
        memory_false["teacher_memory"]["per_source_memory_bit_exact"]["v16"] = False
        mutations.append(("false source causality", memory_false))
        memory_missing = causality_record()
        del memory_missing["teacher_memory"]["per_source_memory_bit_exact"]["v16"]
        mutations.append(("missing source causality", memory_missing))
        zero_false = causality_record()
        zero_false["teacher_memory"]["position_zero_memory_exactly_zero"] = False
        mutations.append(("nonzero position zero", zero_false))
        prefix_false = causality_record()
        prefix_false["end_to_end"]["prefix_logits_bit_exact"] = False
        mutations.append(("future leakage", prefix_false))
        nonzero_difference = causality_record()
        nonzero_difference["end_to_end"]["maximum_absolute_difference"] = 0.1
        mutations.append(("nonzero prefix difference", nonzero_difference))
        isolation_false = causality_record()
        isolation_false["state_isolation"]["rng"] = False
        mutations.append(("changed RNG", isolation_false))
        isolation_missing = causality_record()
        del isolation_missing["state_isolation"]["rng"]
        mutations.append(("missing isolation proof", isolation_missing))
        for label, changed in mutations:
            with self.subTest(case=label):
                self.assertFalse(RUNNER.validate_causality_record(changed))

    def test_source_restore_audit_rejects_lineage_or_reload_tampering(self):
        self.assertTrue(RUNNER.validate_source_restore_audit(source_restore_audit()))
        mutations = []
        for field in (
            "model_exact_reload",
            "optimizer_exact_reload",
            "loader_exact_reload",
            "rng_exact_reload",
        ):
            changed = source_restore_audit()
            changed[field] = False
            mutations.append((f"false {field}", changed))
        wrong_sha = source_restore_audit()
        wrong_sha["integrity"]["sha256"] = "0" * 64
        mutations.append(("wrong source SHA", wrong_sha))
        wrong_next = source_restore_audit()
        wrong_next["next_global_batch_sha256"] = "0" * 64
        mutations.append(("wrong next-batch hash", wrong_next))
        wrong_steps = source_restore_audit()
        wrong_steps["optimizer"]["steps"] = [9, 10, 10]
        mutations.append(("wrong optimizer steps", wrong_steps))
        missing = source_restore_audit()
        del missing["rng_exact_reload"]
        mutations.append(("missing reload proof", missing))
        for label, changed in mutations:
            with self.subTest(case=label):
                self.assertFalse(RUNNER.validate_source_restore_audit(changed))


class ClassificationAndArtifactTests(unittest.TestCase):
    def test_classification_labels_and_nested_rows(self):
        cases = (
            ([0.01, 0.0101, 0.0099, 0.0100], "MEMORY SIGNAL STABLE"),
            ([0.001, 0.002, 0.003, 0.004], "MEMORY SIGNAL STRENGTHENING"),
            ([0.001, 0.003, 0.004, 0.0035], "MEMORY SIGNAL SATURATING"),
            ([0.004, 0.003, 0.002, 0.001], "MEMORY SIGNAL DISAPPEARING"),
        )
        for values, expected in cases:
            with self.subTest(expected=expected):
                rows = [
                    {
                        "completed_updates": completed,
                        "evaluation": {"sequence_specific_recovery": value},
                    }
                    for completed, value in zip((10, 20, 29, 48), values)
                ]
                result = RUNNER.classify_trajectory(rows)
                self.assertEqual(result["label"], expected)
                self.assertEqual(result["values"], values)
                self.assertIn("rule", result)

    def test_classification_rejects_wrong_length_missing_or_nonfinite(self):
        invalid = (
            [{"sequence_specific_recovery": 1.0}] * 3,
            [{"sequence_specific_recovery": 1.0}] * 3 + [{}],
            [{"sequence_specific_recovery": 1.0}] * 3
            + [{"sequence_specific_recovery": math.nan}],
        )
        for rows in invalid:
            with self.subTest(rows=rows):
                with self.assertRaises((ValueError, SystemExit, AssertionError)):
                    RUNNER.classify_trajectory(rows)

    def test_classification_rejects_wrong_milestone_order(self):
        rows = [
            {"completed_updates": update, "sequence_specific_recovery": value}
            for update, value in zip((20, 10, 29, 48), (0.01, 0.02, 0.03, 0.04))
        ]
        with self.assertRaises((ValueError, SystemExit, AssertionError)):
            RUNNER.classify_trajectory(rows)

    def validate_artifact(self, artifact, validation_hash, full=False, completed=20):
        with mock.patch.object(
            RUNNER.a0, "EXPECTED_VALIDATION_GLOBAL_BATCH_SHA256", validation_hash
        ):
            return RUNNER.validate_evaluation_artifact(
                artifact,
                completed,
                "checkpoint-sha",
                "metadata-sha",
                full,
                EXPECTED_SOURCE_FILE_HASHES,
            )

    def test_milestone_artifact_binding_and_paired_lengths(self):
        artifact, validation_hash = cached_artifact()
        validated = self.validate_artifact(copy.deepcopy(artifact), validation_hash)
        self.assertEqual(validated, artifact)
        for mutation in ("completed_updates", "checkpoint_sha256", "metadata_sha256"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(artifact)
                changed[mutation] = "wrong"
                with self.assertRaises((ValueError, SystemExit, AssertionError)):
                    self.validate_artifact(changed, validation_hash)
        short = copy.deepcopy(artifact)
        short["batch_losses"]["shuffled_feedback"].pop()
        with self.assertRaises((ValueError, SystemExit, AssertionError)):
            self.validate_artifact(short, validation_hash)

    def test_cached_artifact_rejects_nonfinite_pair_or_failed_isolation(self):
        artifact, validation_hash = cached_artifact()
        mutations = []
        nonfinite_vector = copy.deepcopy(artifact)
        nonfinite_vector["batch_losses"]["real_feedback"][0] = math.nan
        mutations.append(("nonfinite vector", nonfinite_vector))
        wrong_pair = copy.deepcopy(artifact)
        wrong_pair["paired_shuffled_minus_real"]["mean"] += 1
        mutations.append(("wrong paired mean", wrong_pair))
        wrong_count = copy.deepcopy(artifact)
        wrong_count["paired_shuffled_minus_real"]["positive_count"] = 0
        mutations.append(("wrong paired count", wrong_count))
        for field in (
            "minimum",
            "maximum",
            "sample_standard_deviation",
            "standard_error",
            "ci95_lower",
            "ci95_upper",
        ):
            wrong_arithmetic = copy.deepcopy(artifact)
            wrong_arithmetic["paired_shuffled_minus_real"][field] += 1
            mutations.append((f"wrong paired {field}", wrong_arithmetic))
        failed_isolation = copy.deepcopy(artifact)
        failed_isolation["state_isolation"]["passed"] = False
        mutations.append(("failed state isolation", failed_isolation))
        incomplete_isolation = copy.deepcopy(artifact)
        del incomplete_isolation["state_isolation"]["rng"]
        mutations.append(("incomplete state isolation", incomplete_isolation))
        for label, changed in mutations:
            with self.subTest(case=label):
                with self.assertRaises((ValueError, SystemExit, AssertionError)):
                    self.validate_artifact(changed, validation_hash)

    def test_cached_final_artifact_rejects_missing_controls_or_ablations(self):
        artifact, validation_hash = cached_artifact(full=True, completed_updates=48)
        self.validate_artifact(artifact, validation_hash, full=True, completed=48)
        missing_vector = copy.deepcopy(artifact)
        del missing_vector["batch_losses"]["mask_v16"]
        missing_ablation = copy.deepcopy(artifact)
        del missing_ablation["source_ablation"]["v16"]
        zero_mismatch = copy.deepcopy(artifact)
        zero_mismatch["batch_losses"]["zero_feedback"][0] += 0.1
        zero_mismatch["zero_equals_masked_each_batch"] = False
        for label, changed in (
            ("missing final vector", missing_vector),
            ("missing source ablation", missing_ablation),
            ("zero/masked mismatch", zero_mismatch),
        ):
            with self.subTest(case=label):
                with self.assertRaises((ValueError, SystemExit, AssertionError)):
                    self.validate_artifact(
                        changed, validation_hash, full=True, completed=48
                    )

    def test_cached_final_artifact_rejects_wrong_ablation_statistics(self):
        artifact, validation_hash = cached_artifact(full=True, completed_updates=48)
        for field in (
            "minimum",
            "maximum",
            "sample_standard_deviation",
            "standard_error",
            "ci95_lower",
            "ci95_upper",
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(artifact)
                changed["source_ablation"]["v16"][
                    "paired_ablation_minus_real"
                ][field] += 1
                with self.assertRaises((ValueError, SystemExit, AssertionError)):
                    self.validate_artifact(
                        changed, validation_hash, full=True, completed=48
                    )

    def test_cached_intermediate_artifact_rejects_changed_pinned_references(self):
        artifact, validation_hash = cached_artifact()
        changed = copy.deepcopy(artifact)
        changed["losses"]["full_context_reference"] += 0.1
        full_value = changed["losses"]["full_context_reference"]
        masked_value = changed["losses"]["masked_l1_no_feedback_reference"]
        real_value = changed["losses"]["real_feedback"]
        shuffled_value = changed["losses"]["shuffled_feedback"]
        damage = masked_value - full_value
        changed.update(
            {
                "damage": damage,
                "total_recovery": masked_value - real_value,
                "total_recovery_fraction": (masked_value - real_value) / damage,
                "sequence_specific_recovery": shuffled_value - real_value,
                "sequence_specific_recovery_fraction": (
                    shuffled_value - real_value
                )
                / damage,
            }
        )
        with self.assertRaises((ValueError, SystemExit, AssertionError)):
            self.validate_artifact(changed, validation_hash)

    def test_cached_artifact_rejects_wrong_source_hashes_and_nonfinite_summary(self):
        artifact, validation_hash = cached_artifact()
        wrong_source = copy.deepcopy(artifact)
        wrong_source["source_file_sha256"]["scripts/experiment_2a1.py"] = "0" * 64
        nonfinite_loss = copy.deepcopy(artifact)
        nonfinite_loss["losses"]["real_feedback"] = math.nan
        nonfinite_routing = copy.deepcopy(artifact)
        nonfinite_routing["routing"]["query_norm"] = math.inf
        for label, changed in (
            ("wrong source hashes", wrong_source),
            ("nonfinite summary loss", nonfinite_loss),
            ("nonfinite routing", nonfinite_routing),
        ):
            with self.subTest(case=label):
                with self.assertRaises((ValueError, SystemExit, AssertionError)):
                    self.validate_artifact(changed, validation_hash)

    def test_cached_update10_artifact_revalidates_pinned_means(self):
        artifact, validation_hash = cached_artifact(completed_updates=10)
        self.validate_artifact(artifact, validation_hash, completed=10)
        changed = copy.deepcopy(artifact)
        changed["batch_losses"]["real_feedback"] = [6.0] * 20
        changed["paired_shuffled_minus_real"] = RUNNER.paired_statistics(
            changed["batch_losses"]["real_feedback"],
            changed["batch_losses"]["shuffled_feedback"],
        )
        with self.assertRaises((ValueError, SystemExit, AssertionError)):
            self.validate_artifact(changed, validation_hash, completed=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
