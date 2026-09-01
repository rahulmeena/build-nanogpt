"""Fail-closed tests for the append-only 2D5C analysis adjudicator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import experiment_2d5c_analysis_adjudicator as adjudicator  # noqa: E402


def checked_row(*false_names: str) -> dict:
    checks = {"identity": True, "condition_keys_exact": True}
    for name in false_names:
        checks[name] = False
    return {"checks": checks, "passed": not false_names}


def official_failure_shape() -> dict:
    groups = {
        "core_evaluations": {
            name: checked_row("condition_keys_exact")
            for name in ("c0", "c48", "c96", "c144", "c191", "fixed100m")
        },
        "large_evaluations": {
            name: checked_row("condition_keys_exact")
            for name in ("c191", "fixed100m")
        },
        "secondary_parallel_evaluations": {
            name: checked_row("condition_keys_exact")
            for name in ("c0", "c96", "c191")
        },
        "representation_diagnostics": {
            name: checked_row()
            for name in ("parent", "c0", "c48", "c96", "c144", "c191", "fixed100m")
        },
    }
    groups["core_evaluations"]["parent"] = checked_row()
    for name in (
        "artifact_sets", "cross_artifact_binding", "frozen_manifests",
        "milestone_manifest", "pretrain_freeze",
    ):
        groups[name] = checked_row()
    return {"experiment": "2D5C", "groups": groups, "passed": False}


class Experiment2D5CAnalysisAdjudicatorTests(unittest.TestCase):
    def test_official_failure_must_contain_only_the_eleven_order_checks(self):
        result = adjudicator.validate_official_failed_audit(official_failure_shape())
        self.assertTrue(result["passed"])
        self.assertEqual(result["correction_count"], 11)

        wrong = official_failure_shape()
        wrong["groups"]["core_evaluations"]["c96"]["checks"]["identity"] = False
        with self.assertRaises(adjudicator.AnalysisAdjudicationError):
            adjudicator.validate_official_failed_audit(wrong)

        wrong = official_failure_shape()
        wrong["groups"]["representation_diagnostics"]["c191"]["passed"] = False
        with self.assertRaises(adjudicator.AnalysisAdjudicationError):
            adjudicator.validate_official_failed_audit(wrong)

    def test_correction_requires_ordered_controls_and_exact_key_set(self):
        corrections = []

        def frozen_check(_evaluation, _spec, *_args, **_kwargs):
            return checked_row("condition_keys_exact")

        check = adjudicator.corrected_identity_checker(frozen_check, corrections)
        controls = ["all_real", "b3_off", "b3_b5_off"]
        spec = {"controls": tuple(controls)}
        evaluation = {
            "family": "C",
            "controls_requested": controls,
            # This is the order produced by sorted-key JSON serialization.
            "conditions": {
                "all_real": {}, "b3_b5_off": {}, "b3_off": {},
            },
            "evaluation_identity": {
                "local_update": 191,
                "checkpoint_sha256": "a" * 64,
            },
        }
        result = check(evaluation, spec)
        self.assertTrue(result["passed"])
        self.assertEqual(len(corrections), 1)

        wrong = dict(evaluation)
        wrong["controls_requested"] = sorted(controls)
        with self.assertRaises(adjudicator.AnalysisAdjudicationError):
            check(wrong, spec)

        wrong = dict(evaluation)
        wrong["conditions"] = {**evaluation["conditions"], "unexpected": {}}
        with self.assertRaises(adjudicator.AnalysisAdjudicationError):
            check(wrong, spec)

    def test_non_order_failure_is_never_corrected(self):
        def frozen_check(_evaluation, _spec, *_args, **_kwargs):
            row = checked_row("condition_keys_exact")
            row["checks"]["checkpoint_sha256_exact"] = False
            return row

        check = adjudicator.corrected_identity_checker(frozen_check, [])
        evaluation = {
            "controls_requested": ["all_real"],
            "conditions": {"all_real": {}},
        }
        with self.assertRaises(adjudicator.AnalysisAdjudicationError):
            check(evaluation, {"controls": ("all_real",)})

    def test_wrapper_allows_only_analyze(self):
        parser = adjudicator.build_parser()
        common = [
            "--frozen-driver", "/tmp/driver.py",
            "--failed-audit", "/tmp/failed.json",
            "--preserved-failed-audit", "/tmp/legacy.json",
            "--adjudication-output", "/tmp/adjudication.json",
            "--",
        ]
        parsed = parser.parse_args([*common, "analyze", "--output-dir", "/tmp/out"])
        self.assertEqual(parsed.driver_arguments[0], "--")
        parsed = parser.parse_args([*common, "train"])
        self.assertEqual(parsed.driver_arguments[-1], "train")


if __name__ == "__main__":
    unittest.main()
