"""Focused CPU-only tests for the append-only 2D5C adjudicator."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "experiment_2d5c_posttrain_adjudicator.py"
SPEC = importlib.util.spec_from_file_location("experiment_2d5c_posttrain_adjudicator", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError("could not load post-training adjudicator")
ADJUDICATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADJUDICATOR)


def optimizer_fixture():
    groups = []
    cursor = 0
    for name, count in ADJUDICATOR.EXPECTED_GROUP_COUNTS.items():
        parameters = list(range(cursor, cursor + count))
        cursor += count
        groups.append({"name": name, "params": parameters, "lr": 1e-4})
    low_step_ids = {
        group["params"][0]
        for group in groups if group["name"] in {"b3_gate", "b5_gate", "b6_gate"}
    }
    source = {
        "param_groups": copy.deepcopy(groups),
        "state": {
            parameter: {
                "step": 1_908 if parameter in low_step_ids else 2_386,
                "exp_avg": f"source-avg-{parameter}",
                "exp_avg_sq": f"source-square-{parameter}",
            }
            for parameter in range(cursor)
        },
    }
    final = {
        "param_groups": copy.deepcopy(groups),
        "state": {
            parameter: {
                "step": row["step"] + 191,
                "exp_avg": f"final-avg-{parameter}",
                "exp_avg_sq": f"final-square-{parameter}",
            }
            for parameter, row in source["state"].items()
        },
    }
    return source, final


def training_rows():
    rows = []
    for local_update in range(1, 192):
        rows.append({
            "local_update": local_update,
            "global_update": 1_908 + local_update,
            "new_targets": local_update * 524_288,
            "cumulative_targets": 1_000_341_504 + local_update * 524_288,
            "optimizer_steps": 1,
            "optimizer_step_success": True,
            "optimizer_step_increment_exact": True,
            "optimizer_steps_before_summary": [
                1_908 + local_update - 1,
                2_386 + local_update - 1,
            ],
            "optimizer_steps_after_summary": [
                1_908 + local_update,
                2_386 + local_update,
            ],
            "scheduler_steps": 0,
            "process_id": 100 if local_update <= 96 else 200,
        })
    return rows


class OptimizerProgressionTests(unittest.TestCase):
    def test_exact_two_population_lineage_and_every_slot_plus_191_pass(self):
        source, final = optimizer_fixture()
        audit = ADJUDICATOR.optimizer_progression_audit(source, final)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["source_step_counts"], {"1908": 3, "2386": 149})
        self.assertEqual(audit["final_step_counts"], {"2099": 3, "2577": 149})
        self.assertEqual(audit["progression_mismatches"], [])
        self.assertEqual(audit["state_entry_count"], 152)
        self.assertTrue(all(row["delta_counts"] == {"191": row["parameter_state_entries"]}
                            for row in audit["per_group"]))

    def test_collapsing_the_inherited_population_to_global_update_fails(self):
        source, final = optimizer_fixture()
        for state in final["state"].values():
            state["step"] = 2_099
        audit = ADJUDICATOR.optimizer_progression_audit(source, final)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["checks"]["final_step_populations_exact"])
        self.assertFalse(audit["checks"]["every_state_advanced_exactly_191"])

    def test_missing_state_or_changed_group_topology_fails_closed(self):
        source, final = optimizer_fixture()
        del final["state"][0]
        final["param_groups"][0]["params"] = final["param_groups"][0]["params"][1:]
        audit = ADJUDICATOR.optimizer_progression_audit(source, final)
        self.assertFalse(audit["checks"]["state_keys_exact"])
        self.assertFalse(audit["checks"]["group_topology_exact"])
        self.assertFalse(audit["checks"]["final_group_state_coverage_exact"])
        self.assertFalse(audit["passed"])

    def test_single_wrong_delta_fails_even_if_state_keys_are_preserved(self):
        source, final = optimizer_fixture()
        final["state"][0]["step"] -= 1
        audit = ADJUDICATOR.optimizer_progression_audit(source, final)
        self.assertTrue(audit["checks"]["state_keys_exact"])
        self.assertFalse(audit["checks"]["every_state_advanced_exactly_191"])
        self.assertEqual(len(audit["progression_mismatches"]), 1)
        self.assertFalse(audit["passed"])


class TrainingRowChainTests(unittest.TestCase):
    def test_all_191_rows_form_the_exact_expected_chain(self):
        audit = ADJUDICATOR.training_row_chain_audit(training_rows())
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["source_step_summary"], [1_908, 2_386])
        self.assertEqual(audit["final_step_summary"], [2_099, 2_577])
        self.assertEqual(audit["rows"], 191)

    def test_missing_row_broken_adjacency_and_false_increment_all_fail(self):
        missing = training_rows()[:-1]
        self.assertFalse(ADJUDICATOR.training_row_chain_audit(missing)["passed"])

        broken = training_rows()
        broken[100]["optimizer_steps_before_summary"] = [0, 0]
        audit = ADJUDICATOR.training_row_chain_audit(broken)
        self.assertFalse(audit["checks"]["every_row_summary_exact"])
        self.assertFalse(audit["checks"]["row_summary_chain_exact"])
        self.assertFalse(audit["passed"])

        false_increment = training_rows()
        false_increment[50]["optimizer_step_increment_exact"] = False
        audit = ADJUDICATOR.training_row_chain_audit(false_increment)
        self.assertFalse(audit["checks"]["full_state_increment_flag_each"])
        self.assertFalse(audit["passed"])

    def test_missing_fresh_process_boundary_fails(self):
        rows = training_rows()
        for row in rows:
            row["process_id"] = 100
        audit = ADJUDICATOR.training_row_chain_audit(rows)
        self.assertFalse(audit["checks"]["mandatory_process_boundary_exact"])
        self.assertFalse(audit["passed"])


class AppendOnlyCorrectionTests(unittest.TestCase):
    def test_training_correction_changes_only_known_false_check_and_adds_provenance(self):
        original = {
            "schema": "experiment_2d5c_training_complete_v1",
            "experiment": "2D5C",
            "checks": {
                "updates_exact": True,
                "optimizer_terminal_step_exact": False,
            },
            "passed": False,
            "optimizer_evidence": {"step_summary": [2_099, 2_577]},
        }
        before = copy.deepcopy(original)
        adjudication = {"schema": "test", "passed": True}
        result = ADJUDICATOR.correct_original_training_payload(
            original, adjudication
        )
        self.assertEqual(original, before)
        self.assertEqual(result["schema"], original["schema"])
        self.assertNotIn("optimizer_terminal_step_exact", result["checks"])
        self.assertTrue(result["checks"]["optimizer_step_lineage_exact"])
        self.assertFalse(
            result["legacy_checks"]["optimizer_terminal_step_exact"]
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["posttrain_adjudication"], adjudication)

    def test_training_correction_rejects_any_additional_false_check(self):
        original = {
            "schema": "experiment_2d5c_training_complete_v1",
            "experiment": "2D5C",
            "checks": {
                "optimizer_terminal_step_exact": False,
                "another_failure": False,
            },
            "passed": False,
        }
        with self.assertRaises(ADJUDICATOR.AdjudicationError):
            ADJUDICATOR.correct_original_training_payload(
                original, {"passed": True}
            )

    def test_seal_correction_requires_the_one_known_legacy_failure(self):
        legacy = {
            "schema": "experiment_2d5c_final_checkpoint_provenance_v1",
            "experiment": "2D5C",
            "checks": {"strict_reopen": True, "optimizer_step_exact": False},
            "sealed": False,
        }
        before = copy.deepcopy(legacy)
        result = ADJUDICATOR.correct_legacy_seal_payload(
            legacy, {"passed": True}
        )
        self.assertEqual(legacy, before)
        self.assertNotIn("optimizer_step_exact", result["checks"])
        self.assertTrue(result["checks"]["optimizer_step_lineage_exact"])
        self.assertFalse(result["legacy_checks"]["optimizer_step_exact"])
        self.assertTrue(result["sealed"])

        wrong = copy.deepcopy(legacy)
        wrong["checks"]["strict_reopen"] = False
        with self.assertRaises(ADJUDICATOR.AdjudicationError):
            ADJUDICATOR.correct_legacy_seal_payload(
                wrong, {"passed": True}
            )

    def test_output_creation_is_exclusive_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adjudicated.json"
            ADJUDICATOR.durable_json_exclusive(path, {"passed": True})
            self.assertEqual(json.loads(path.read_text()), {"passed": True})
            before = path.read_bytes()
            with self.assertRaises(ADJUDICATOR.AdjudicationError):
                ADJUDICATOR.durable_json_exclusive(path, {"passed": False})
            self.assertEqual(path.read_bytes(), before)

    def test_cli_exposes_only_the_two_adjudication_commands(self):
        parser = ADJUDICATOR.build_parser()
        subparsers = [
            action for action in parser._actions
            if action.__class__.__name__ == "_SubParsersAction"
        ]
        self.assertEqual(len(subparsers), 1)
        self.assertEqual(
            set(subparsers[0].choices),
            {"adjudicate-training", "adjudicate-seal"},
        )


if __name__ == "__main__":
    unittest.main()
